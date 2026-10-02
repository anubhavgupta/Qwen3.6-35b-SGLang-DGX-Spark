#!/usr/bin/env bash
set -euo pipefail

# MTP wrapper. Serves the start.sh target (default
# unsloth/Qwen3.6-35B-A3B-NVFP4, override with MODEL_ID) with speculative
# decoding from the checkpoint's own MTP head (mtp.* weights, 1 layer, left
# unquantized BF16 by the Unsloth export). No second checkpoint: SGLang
# loads the same repo as the draft and remaps
# Qwen3_5MoeForConditionalGeneration -> Qwen3_5ForCausalLMMTP (MoE-aware),
# sharing the target's embeddings and lm_head.
#
# Flags: --speculative-algorithm EAGLE (NEXTN is an alias), chain drafting
# steps/topk/draft = 3/1/4 (the model card's SGLang MTP recipe; topk=1
# requires DRAFT = STEPS + 1). Override with MTP_STEPS / MTP_DRAFT; on GB10
# the draft-token count is the main tune knob.
#
# start.sh's pool math is told about the draft:
#   SPEC_DRAFT_TOKENS        = MTP_DRAFT (D intermediate GDN states / request)
#   DRAFT_KV_BYTES_PER_TOKEN = 1 full-attention layer x 2 (K,V) x 2 KV heads
#                              x 256 dim x 1 B (fp8) = 1 KB per token
#   DRAFT_WEIGHTS_GIB        = 0 (mtp.* bytes are already in WEIGHTS_GIB;
#                              the target load skips them)
# YaRN stays allowed: the draft reads the same checkpoint config, so the
# text_config rope override applies to it as intended (SPEC_ALLOW_YARN=1).
# If spec decode errors at boot with flashinfer, try
#   MTP_EXTRA="--attention-backend triton" ./start-mtp.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MTP_STEPS="${MTP_STEPS:-3}"
MTP_DRAFT="${MTP_DRAFT:-4}"
if ! [[ "${MTP_STEPS}" =~ ^[1-9][0-9]*$ && "${MTP_DRAFT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MTP_STEPS and MTP_DRAFT must be positive integers (got steps=${MTP_STEPS}, draft=${MTP_DRAFT})"; exit 1
fi
if (( MTP_DRAFT != MTP_STEPS + 1 )); then
  echo "MTP_DRAFT must equal MTP_STEPS + 1 for topk=1 chain drafting (got steps=${MTP_STEPS}, draft=${MTP_DRAFT})"; exit 1
fi

EXTRA_ARGS="--speculative-algorithm EAGLE \
--speculative-num-steps ${MTP_STEPS} \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens ${MTP_DRAFT} ${MTP_EXTRA:-}"
export EXTRA_ARGS

export SPEC_LABEL="MTP (EAGLE steps=${MTP_STEPS} topk=1 draft=${MTP_DRAFT})"
export SPEC_DRAFT_TOKENS="${MTP_DRAFT}"
export DRAFT_KV_BYTES_PER_TOKEN=$(( 1 * 2 * 2 * 256 * 1 ))
export DRAFT_WEIGHTS_GIB="${DRAFT_WEIGHTS_GIB:-0}"
export SPEC_ALLOW_YARN=1

echo "MTP mode: EXTRA_ARGS=${EXTRA_ARGS}"
echo "Delegating to ${SCRIPT_DIR}/start.sh"
exec "${SCRIPT_DIR}/start.sh"
