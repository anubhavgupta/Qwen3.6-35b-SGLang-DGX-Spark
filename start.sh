#!/usr/bin/env bash
set -euo pipefail

# Qwen3.6-35B-A3B (Unsloth NVFP4) on SGLang (DGX Spark / GB10, aarch64)
#
# Checkpoint: https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4
#
# Notes:
#   - Hybrid GDN MoE vision-language model (Qwen3_5MoeForConditionalGeneration):
#     40 layers = 30 linear-attention (GDN) + 10 full-attention, 256
#     experts / 8 active (~3B active params). The vision tower is live.
#   - Unsloth NVFP4 export (compressed-tensors, W4A4 NVFP4 experts and
#     linears; vision tower left unquantized). Override the checkpoint
#     with MODEL_ID=<hf repo> (e.g. unsloth/Qwen3.6-35B-A3B-NVFP4-Fast).
#   - DGX Spark specifics: 128GB unified memory; 8192-token prefill
#     chunks, --mem-fraction-static ${MEM_FRACTION_STATIC} (default 0.5) and --disable-prefill-cuda-graph.
#   - --attention-backend flashinfer is required on SM120/SM121
#     (trtllm_mha is SM100-only).
#   - KV cache is explicitly FP8 (--kv-cache-dtype fp8_e4m3). The
#     checkpoint declares an FP8 kv_cache_scheme, but this image finds no
#     KV scales in it ("no scaling factors provided ... 1.0" at boot), so
#     scales default to 1.0. Only the 10 full-attention layers hold KV
#     (2 KV heads x 256 dim): ~10 KB/token -> a 1M-token sequence needs
#     ~10GB of KV.
#   - Speculative decoding is OFF (plain autoregressive decode). The
#     checkpoint does carry an MTP head; to experiment with it, use
#       EXTRA_ARGS="--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4"
#   - Thinking & tool calling (model card):
#       * Thinking mode is ON by default (chat template default
#         enable_thinking=true). --reasoning-parser qwen3 surfaces the
#         <think> block as reasoning_content instead of inline text.
#         Disable per request via chat_template_kwargs
#         {"enable_thinking": false} (then prefer temperature=0.7,
#         top_p=0.8, presence_penalty=1.5 per card).
#       * --sampling-defaults model (SGLang default, pinned):
#         defaults come from the checkpoint's generation_config.json
#         (temperature=1.0, top_p=0.95, top_k=20). The card also
#         recommends presence_penalty=1.5 for general thinking tasks;
#         send it per request.
#       * Tool calling: --tool-call-parser qwen3_coder decodes the
#         template's <tool_call><function=...>/<parameter=...>
#         payload into structured tool_calls. SGLang needs no
#         vLLM-style --enable-auto-tool-choice; just send `tools`.
#   - GDN state pool sizing (sglang compute-mamba-ratio skill):
#       state/slot  = 30 GDN layers x (SSM 32x128x128 bf16 + conv 3x8192 bf16)
#                   ~= 31.4 MiB
#       KV/token    = 10 attn layers x 2 x 2 KV heads x 256 x 1 B (fp8) = 10 KB
#       token_equiv = 31.4 MiB / 10 KB ~= 3216
#       r* = S x token_equiv / L  (D=0, no spec; dcp=1)
#          = 1.57 @ L=8K, 0.39 @ 32K, 0.098 @ 128K, 0.049 @ 262K
#     start.sh applies the skill at launch (see the "GDN state pool"
#     block below), so changing MAX_CONCURRENT_REQUESTS / CONTEXT_LENGTH /
#     MAMBA_AVG_CONTEXT_LEN / MEM_FRACTION_STATIC re-derives the pool:
#     it pins --max-mamba-cache-size = concurrency x S when the
#     concurrency cap binds (or r* < 0.15), else passes
#     --mamba-full-memory-ratio r* (memory binds). MAMBA_POOL_MODE=
#     pin|ratio forces either. SGLang ignores the ratio when the pin is set.
#     S=4 for extra_buffer_lazy + overlap scheduler (base 3 + lazy 1);
#     S=3 with MAMBA_SKIP_DECODE_LOCK=1 (sets
#     SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK, freeing one resident slot per
#     request). --mamba-ssm-dtype bfloat16 halves the per-slot state vs
#     the checkpoint's fp32 default. --max-running-requests pins the
#     scheduler cap to match. After boot, check the "Mamba Cache is
#     allocated" / "KV Cache is allocated" / max_running_requests lines
#     in .sglang.log to confirm the byte constants above.
#   - Context: YARN=0|1 in .env, plus CONTEXT_LENGTH (range 1024..1000000;
#     the card validates up to 1,010,000). YaRN (rope scaling) is applied
#     when CONTEXT_LENGTH exceeds 262144 and either YARN=1 or the length
#     is exactly 1000000. Factor = round(CONTEXT_LENGTH/262144) ->
#     524288 => 2.0, 1000000 => 4.0. The rope override lives under
#     text_config in the JSON; SGLang also needs
#     SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 or it stays at 262K.

# Load optional .env overrides (MODEL_ID, YARN, CONTEXT_LENGTH, MAX_CONCURRENT_REQUESTS).
# Shell env vars already set win; .env fills the gaps; defaults apply last.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  # `|| [[ -n "${key}" ]]` so a final line without a trailing newline is not dropped.
  while IFS='=' read -r key value || [[ -n "${key}" ]]; do
    # Tolerate CRLF files and surrounding whitespace, and skip indented comments —
    # otherwise `${!key}` below would be an indirect expansion on an invalid name.
    key="${key%$'\r'}"; value="${value%$'\r'}"
    key="${key#"${key%%[![:space:]]*}"}"; key="${key%"${key##*[![:space:]]}"}"
    [[ -z "${key}" || "${key}" == \#* ]] && continue
    if [[ -z "${!key:-}" ]]; then
      export "${key}=${value}"
    fi
  done < "${SCRIPT_DIR}/.env"
fi

MODEL_ID="${MODEL_ID:-unsloth/Qwen3.6-35B-A3B-NVFP4}"

# Context length: any value from native up to the model's validated 1M.
# YaRN controlled explicitly by YARN (0|1), plus auto-on at exactly 1M.
YARN="${YARN:-0}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-2}"

CHUNKED_PREFILL="${CHUNKED_PREFILL:-8192}"
# 1 = set SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK in the container (frees one
# GDN state slot per running request; S 4 -> 3). 0 = stock locking.
MAMBA_SKIP_DECODE_LOCK="${MAMBA_SKIP_DECODE_LOCK:-0}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.5}"
# GDN state-pool sizing (sglang compute-mamba-ratio skill), see header:
#   auto  = pin when the concurrency cap binds or r* < 0.15, else ratio r*
#   pin   = always --max-mamba-cache-size = concurrency x S
#   ratio = always --mamba-full-memory-ratio r*
MAMBA_POOL_MODE="${MAMBA_POOL_MODE:-auto}"
# L in the skill formula: average context (input + output) per request.
# Defaults to CONTEXT_LENGTH (worst case: every request uses the full window).
MAMBA_AVG_CONTEXT_LEN="${MAMBA_AVG_CONTEXT_LEN:-}"
# Per-GPU byte constants for Qwen3.6-35B-A3B with --mamba-ssm-dtype
# bfloat16 and fp8 KV. Derived from the config and confirmed by a boot log
# (2026-10-01, 8-slot pin): "Mamba Cache is allocated ... conv_state 0.01GB,
# ssm_state 0.26GB" = 9 slots (8 + padding) x 31.4 MiB; "KV Cache ...
# #tokens: 524288, K 2.50 GB, V 2.50 GB" = 10240 B/token. Re-measure from
# those two lines if you change model or dtypes.
MAMBA_STATE_BYTES_PER_SLOT="${MAMBA_STATE_BYTES_PER_SLOT:-32931840}"
KV_BYTES_PER_TOKEN="${KV_BYTES_PER_TOKEN:-10240}"
# Weight footprint, used only to estimate the post-weight budget. Measured:
# "Load weight end ... mem usage=27.36 GB" (24.7 GiB of safetensors plus
# load-time overhead).
WEIGHTS_GIB="${WEIGHTS_GIB:-27.4}"
# KV pool cap in tokens (--max-total-tokens). Empty = auto: in pin mode
# MAX_CONCURRENT_REQUESTS x CONTEXT_LENGTH (e.g. 2 x 262144 = 524288), so
# every running request can reach the full window and no more KV is
# allocated; in ratio mode (memory binds) no cap. 0 = never cap.
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-}"
# GDN radix-cache strategy: extra_buffer_lazy (S=4) or extra_buffer (S=5),
# both with the overlap scheduler; MAMBA_SKIP_DECODE_LOCK=1 subtracts 1.
MAMBA_RADIX_STRATEGY="${MAMBA_RADIX_STRATEGY:-extra_buffer_lazy}"
# Speculative-decoding inputs for the pool math, set by wrappers such as
# start-dflash.sh (which also passes the spec flags via EXTRA_ARGS).
# Defaults = no spec decode.
#   SPEC_LABEL                 shown in the startup banner
#   SPEC_DRAFT_TOKENS          D: verify width; each running request holds D
#                              intermediate GDN states (separate buffer)
#   DRAFT_KV_BYTES_PER_TOKEN   draft-model KV per target token (0 if none)
#   DRAFT_WEIGHTS_GIB          draft checkpoint size
#   SPEC_ALLOW_YARN            1 = the draft shares the target config (MTP), so
#                              the YaRN override is safe; otherwise YaRN is
#                              refused while a draft is configured
SPEC_LABEL="${SPEC_LABEL:-off}"
SPEC_DRAFT_TOKENS="${SPEC_DRAFT_TOKENS:-0}"
DRAFT_KV_BYTES_PER_TOKEN="${DRAFT_KV_BYTES_PER_TOKEN:-0}"
DRAFT_WEIGHTS_GIB="${DRAFT_WEIGHTS_GIB:-0}"
if ! [[ "${SPEC_DRAFT_TOKENS}" =~ ^[0-9]+$ ]]; then
  echo "SPEC_DRAFT_TOKENS must be a non-negative integer, got '${SPEC_DRAFT_TOKENS}'"; exit 1
fi
# Pin the container to GB10's 10 Cortex-X5 cores (5-9, 15-19; the A725
# efficiency cores are 0-4, 10-14). Keeps scheduler/tokenizer Python off
# the 2.8GHz little cores. Empty = no pinning.
CPUSET="${CPUSET:-5-9,15-19}"
# Free-form extra SGLang server flags, appended LAST so argparse's
# last-wins rule lets them override anything above. Experiments go here:
#   EXTRA_ARGS="--enable-fused-qk-norm-rope" ./start.sh
#   EXTRA_ARGS="--moe-runner-backend flashinfer_cutlass" ./start.sh
# 1 = enable prefill CUDA graphs (default 0 = recipe's --disable-prefill-
# cuda-graph; SM121 boot test before trusting).
PREFILL_CUDA_GRAPH="${PREFILL_CUDA_GRAPH:-0}"
read -ra EXTRA_ARGS_ARR <<< "${EXTRA_ARGS:-}"
# Extra container env (NAME=value pairs). Used for DSpark compact/SPS:
#   DOCKER_ENV='SGLANG_RAGGED_VERIFY_MODE=compact' ./start-dspark.sh
DOCKER_ENV_ARGS=()
if [[ -n "${DOCKER_ENV:-}" ]]; then
  read -ra _docker_env_pairs <<< "${DOCKER_ENV}"
  for _pair in "${_docker_env_pairs[@]}"; do
    DOCKER_ENV_ARGS+=(-e "${_pair}")
  done
fi
if (( CONTEXT_LENGTH < 1024 || CONTEXT_LENGTH > 1000000 )); then
  echo "CONTEXT_LENGTH '${CONTEXT_LENGTH}' unsupported (use 1024..1000000)"; exit 1
fi
case "${YARN}" in
  0|1) : ;;
  *) echo "YARN must be 0 or 1, got '${YARN}'"; exit 1 ;;
esac
if (( CHUNKED_PREFILL < 256 )); then
  echo "CHUNKED_PREFILL '${CHUNKED_PREFILL}' unsupported (use >= 256)"; exit 1
fi
case "${MAMBA_SKIP_DECODE_LOCK}" in
  0|1) : ;;
  *) echo "MAMBA_SKIP_DECODE_LOCK must be 0 or 1, got '${MAMBA_SKIP_DECODE_LOCK}'"; exit 1 ;;
esac
NEED_YARN=0
if (( CONTEXT_LENGTH > 262144 )); then
  if [[ "${YARN}" == "1" ]] || [[ "${CONTEXT_LENGTH}" == "1000000" ]]; then
    NEED_YARN=1
  else
    echo "warning: CONTEXT_LENGTH=${CONTEXT_LENGTH} needs YaRN but YARN=0; build will stay at 262K"
  fi
fi
if (( NEED_YARN )); then
  if (( SPEC_DRAFT_TOKENS > 0 )) && [[ "${SPEC_ALLOW_YARN:-0}" != "1" ]]; then
    echo "CONTEXT_LENGTH=${CONTEXT_LENGTH} needs YaRN, but the --json-model-override-args rope override"
    echo "is also applied to the speculative draft's config (crashes the draft at boot)."
    echo "Use CONTEXT_LENGTH=262144 with a draft model, or start.sh without speculative decoding."
    exit 1
  fi
  # Model card's YaRN recipe: rope_parameters override under text_config.
  # Factor derived from CONTEXT_LENGTH (round(len/262144)); the card
  # validates 2.0 for 524288 and 4.0 for 1M.
  YARN_FACTOR="$(awk -v n="${CONTEXT_LENGTH}" 'BEGIN{printf "%.0f", n/262144}')"
  (( YARN_FACTOR < 1 )) && YARN_FACTOR=1
  YARN_OVERRIDE=$(printf '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": %s, "original_max_position_embeddings": 262144}}}' "${YARN_FACTOR}")
  CONTEXT_ARGS=(--json-model-override-args "${YARN_OVERRIDE}" --context-length "${CONTEXT_LENGTH}")
  ALLOW_LONGER_ARGS=(-e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1)
  YARN_SUFFIX=" (YaRN factor ${YARN_FACTOR})"
else
  CONTEXT_ARGS=(--context-length "${CONTEXT_LENGTH}")
  ALLOW_LONGER_ARGS=()
  YARN_SUFFIX=""
fi


# GDN state pool (compute-mamba-ratio skill). S = state slots per running
# request: 4 for extra_buffer_lazy, 5 for extra_buffer (overlap scheduler
# on), minus 1 with MAMBA_SKIP_DECODE_LOCK=1. D = SPEC_DRAFT_TOKENS (0 = no
# spec decode), dcp_size = 1. KV bytes per token include the draft model's
# KV when a wrapper declares one (its pool aliases the target's tokens).
#   token_equiv = state_bytes_per_slot / (kv_bytes + draft_kv_bytes)
#   r*          = (S + D) x token_equiv / L
# The post-weight budget ("rest") is estimated as MemTotal x mem-fraction
# minus target + draft weights (SGLang also reserves activations/graphs, so
# it is an upper bound). The concurrency memory can carry at balance is
#   mem_conc = rest / ((S + D) x state_bytes + L x (kv + draft_kv bytes))
# If MAX_CONCURRENT_REQUESTS <= mem_conc the user cap binds: pin the state
# pool to exactly cap x S and give everything else to KV (a ratio would
# over-reserve state slots that the cap never uses). If r* < 0.15 the skill
# also says pin (tiny ratios are fragile). Otherwise memory binds and
# --mamba-full-memory-ratio r* balances the two pools so neither runs out
# first. Note: SGLang ignores the ratio whenever --max-mamba-cache-size is set.
# The pin stays cap x S even with spec decode: SGLang reserves the
# D intermediate states per request in a separate buffer.
case "${MAMBA_RADIX_STRATEGY}" in
  extra_buffer_lazy) MAMBA_BASE_SLOTS=4 ;;
  extra_buffer)      MAMBA_BASE_SLOTS=5 ;;
  *) echo "MAMBA_RADIX_STRATEGY must be extra_buffer_lazy|extra_buffer, got '${MAMBA_RADIX_STRATEGY}'"; exit 1 ;;
esac
MAMBA_SLOTS_PER_REQ=$(( MAMBA_BASE_SLOTS - MAMBA_SKIP_DECODE_LOCK ))
MAMBA_L="${MAMBA_AVG_CONTEXT_LEN:-${CONTEXT_LENGTH}}"
MEM_TOTAL_KB="$(awk '/^MemTotal:/{print $2}' /proc/meminfo)"
read -r MAMBA_TOKEN_EQUIV MAMBA_RATIO MAMBA_REST_GIB MAMBA_MEM_CONC < <(awk \
  -v S="${MAMBA_SLOTS_PER_REQ}" -v D="${SPEC_DRAFT_TOKENS}" -v L="${MAMBA_L}" \
  -v st="${MAMBA_STATE_BYTES_PER_SLOT}" -v kv="${KV_BYTES_PER_TOKEN}" -v dkv="${DRAFT_KV_BYTES_PER_TOKEN}" \
  -v memkb="${MEM_TOTAL_KB}" -v frac="${MEM_FRACTION_STATIC}" \
  -v w="${WEIGHTS_GIB}" -v dw="${DRAFT_WEIGHTS_GIB}" \
  'BEGIN{
     te = st / (kv + dkv); r = (S + D) * te / L
     rest = memkb / 1048576 * frac - w - dw
     per_req = ((S + D) * st + L * (kv + dkv)) / 1073741824
     conc = (rest > 0) ? int(rest / per_req) : 0
     printf "%.0f %.4f %.2f %d\n", te, r, rest, conc
   }')
if awk -v r="${MAMBA_REST_GIB}" 'BEGIN{exit !(r <= 0)}'; then
  echo "MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC} leaves no memory after ~${WEIGHTS_GIB}+${DRAFT_WEIGHTS_GIB} GiB of weights"; exit 1
fi
case "${MAMBA_POOL_MODE}" in
  pin|ratio) : ;;
  auto)
    if (( MAX_CONCURRENT_REQUESTS <= MAMBA_MEM_CONC )) \
       || awk -v r="${MAMBA_RATIO}" 'BEGIN{exit !(r < 0.15)}'; then
      MAMBA_POOL_MODE=pin
    else
      MAMBA_POOL_MODE=ratio
    fi ;;
  *) echo "MAMBA_POOL_MODE must be auto|pin|ratio, got '${MAMBA_POOL_MODE}'"; exit 1 ;;
esac
if [[ "${MAMBA_POOL_MODE}" == "pin" ]]; then
  MAMBA_CACHE_SIZE=$(( MAX_CONCURRENT_REQUESTS * MAMBA_SLOTS_PER_REQ ))
  MAMBA_POOL_ARGS=(--max-mamba-cache-size "${MAMBA_CACHE_SIZE}")
  MAMBA_POOL_DESC="pinned ${MAMBA_CACHE_SIZE} slots (${MAX_CONCURRENT_REQUESTS} x S=${MAMBA_SLOTS_PER_REQ})"
else
  MAMBA_POOL_ARGS=(--mamba-full-memory-ratio "${MAMBA_RATIO}")
  MAMBA_POOL_DESC="ratio ${MAMBA_RATIO} (memory binds: ~${MAMBA_MEM_CONC} concurrent at L=${MAMBA_L})"
fi
if (( MAX_CONCURRENT_REQUESTS > MAMBA_MEM_CONC )); then
  echo "warning: ~${MAMBA_REST_GIB} GiB post-weight budget fits only ~${MAMBA_MEM_CONC} requests of ${MAMBA_L} tokens;"
  echo "         SGLang will clamp/retract above that (lower MAX_CONCURRENT_REQUESTS or MAMBA_AVG_CONTEXT_LEN, or raise MEM_FRACTION_STATIC)"
fi
if [[ -z "${MAX_TOTAL_TOKENS}" ]]; then
  if [[ "${MAMBA_POOL_MODE}" == "pin" ]]; then
    # + D per request: the speculative verify block is allocated on top of
    # a request's committed tokens.
    MAX_TOTAL_TOKENS=$(( MAX_CONCURRENT_REQUESTS * (CONTEXT_LENGTH + SPEC_DRAFT_TOKENS) ))
  else
    MAX_TOTAL_TOKENS=0
  fi
fi
if ! [[ "${MAX_TOTAL_TOKENS}" =~ ^[0-9]+$ ]]; then
  echo "MAX_TOTAL_TOKENS must be a non-negative integer, got '${MAX_TOTAL_TOKENS}'"; exit 1
fi
KV_POOL_ARGS=()
KV_POOL_DESC="uncapped (all remaining budget)"
if (( MAX_TOTAL_TOKENS > 0 )); then
  KV_POOL_ARGS=(--max-total-tokens "${MAX_TOTAL_TOKENS}")
  KV_POOL_DESC="$(awk -v t="${MAX_TOTAL_TOKENS}" -v kv="${KV_BYTES_PER_TOKEN}" -v dkv="${DRAFT_KV_BYTES_PER_TOKEN}" -v rest="${MAMBA_REST_GIB}" 'BEGIN{
    g = t * (kv + dkv) / 1073741824
    printf "%d tokens (~%.1f GiB%s)", t, g, (dkv > 0 ? " incl. draft KV" : "")
    if (g > rest) printf "; exceeds the ~%.1f GiB budget, SGLang will use its profiled size", rest
  }')"
fi

SERVED_MODEL_NAME="qwen3.6-35b-a3b-sglang"
# Image: official lmsysorg/sglang nightly (main 708f51e44, 2026-09-09),
# pinned by multi-arch index digest; docker pulls it on first run (same
# image start-dflash.sh uses). Do NOT use the older lmsysorg/sglang:qwen38-27b
# for this checkpoint: the Unsloth export quantizes lm_head to FP8
# (per-channel scales), and that image drops lm_head.weight_scale
# ("Parameter lm_head.weight_scale not found in params_dict"), producing
# garbage logits (output collapses into one repeated word). The nightly
# carries the quantized-lm_head support (sglang #35496). IMAGE=<ref> overrides.
IMAGE="${IMAGE:-lmsysorg/sglang@sha256:00205b89f74691f76a0ffbd6846376d9323971930a5d59bf63a65dadc7d67927}"
CONTAINER_NAME="qwen3.6-35b-a3b-sglang"
HOST="0.0.0.0"
PORT="8888"
PID_FILE=".sglang.pid"
LOG_FILE=".sglang.log"
WORK_DIR="$(pwd)"
HF_HOME="${WORK_DIR}/.cache/huggingface"
TRITON_CACHE_DIR="${WORK_DIR}/.cache/triton"
READY_URL="http://127.0.0.1:${PORT}/v1/models"

command -v docker >/dev/null 2>&1 || {
  echo "docker is not on PATH"
  exit 1
}

command -v curl >/dev/null 2>&1 || {
  echo "curl is not on PATH"
  exit 1
}

mkdir -p "${HF_HOME}" "${TRITON_CACHE_DIR}"

# Pick up HF_TOKEN from ~/.bashrc (defined without `export` there) so the
# container gets authenticated Hub access (higher rate limits, faster downloads).
if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.bashrc" ]]; then
  # Accept `export HF_TOKEN=…` (the idiomatic form: a bare assignment in .bashrc is
  # not exported to child processes) as well as a plain `HF_TOKEN=…`.
  HF_TOKEN="$(sed -n 's/^[[:space:]]*\(export[[:space:]]\+\)\?HF_TOKEN=["'"'"']\?\([A-Za-z0-9_-]\+\).*/\2/p' "${HOME}/.bashrc" | head -1)"
fi
export HF_TOKEN

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container ${CONTAINER_NAME} is already running"
    echo "Log: ${LOG_FILE}"
    exit 0
  fi
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

echo "Starting SGLang container for ${MODEL_ID}"
echo "Context: ${CONTEXT_LENGTH} tokens${YARN_SUFFIX:-}"
echo "Max concurrent requests: ${MAX_CONCURRENT_REQUESTS}"
echo "Mamba pool: ${MAMBA_POOL_DESC}; r*=${MAMBA_RATIO} (token_equiv ${MAMBA_TOKEN_EQUIV}, L=${MAMBA_L}), est. post-weight budget ${MAMBA_REST_GIB} GiB"
echo "KV pool: ${KV_POOL_DESC}"
echo "Spec decode: ${SPEC_LABEL}"
echo "Image: ${IMAGE}"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Listening on ${HOST}:${PORT}"
echo "Writing progress to ${LOG_FILE}"

cat >"${LOG_FILE}" <<EOF
[$(date -Is)] launching SGLang container
EOF

PIN_ARGS=()
[[ -n "${CPUSET}" ]] && PIN_ARGS=(--cpuset-cpus "${CPUSET}")
PREFILL_GRAPH_ARGS=(--disable-prefill-cuda-graph)
[[ "${PREFILL_CUDA_GRAPH}" == "1" ]] && PREFILL_GRAPH_ARGS=()

docker run -d \
  --name "${CONTAINER_NAME}" \
  --network host \
  --ipc host \
  --privileged \
  --gpus all \
  --shm-size 32g \
  "${PIN_ARGS[@]}" \
  -e HF_HOME=/root/.cache/huggingface \
  -e TRITON_CACHE_DIR=/root/.triton \
  -e SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK="${MAMBA_SKIP_DECODE_LOCK}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  "${DOCKER_ENV_ARGS[@]}" \
  "${ALLOW_LONGER_ARGS[@]}" \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  -v "${TRITON_CACHE_DIR}:/root/.triton" \
  "${IMAGE}" \
  python3 -m sglang.launch_server \
  --model-path "${MODEL_ID}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --sleep-on-idle \
  --attention-backend flashinfer \
  --chunked-prefill-size "${CHUNKED_PREFILL}" \
  "${PREFILL_GRAPH_ARGS[@]}" \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-ssm-dtype bfloat16 \
  --mamba-radix-cache-strategy "${MAMBA_RADIX_STRATEGY}" \
  "${MAMBA_POOL_ARGS[@]}" \
  "${KV_POOL_ARGS[@]}" \
  --max-running-requests "${MAX_CONCURRENT_REQUESTS}" \
  "${CONTEXT_ARGS[@]}" \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --sampling-defaults model \
  --enable-metrics \
  --enable-cache-report \
  --host "${HOST}" \
  --port "${PORT}" \
  "${EXTRA_ARGS_ARR[@]}" \
  >/dev/null

container_id="$(docker inspect -f '{{.Id}}' "${CONTAINER_NAME}")"
echo "${container_id}" > "${PID_FILE}"
echo "Spawned container ${CONTAINER_NAME} (${container_id})"

log_follow_pid=""
cleanup() {
  if [[ -n "${log_follow_pid}" ]] && kill -0 "${log_follow_pid}" 2>/dev/null; then
    kill "${log_follow_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Stream the SGLang startup log to the terminal AND record it in .sglang.log.
# $! tracks the pipeline group; killing it on exit SIGPIPEs docker logs.
# The terminal copy is filtered to drop the per-layer "Enabled fused
# SiLU+mul+FP4-quant for dense MLP down_proj input." notices (they fire once
# per MLP layer); .sglang.log keeps the complete, unfiltered stream.
docker logs -f "${CONTAINER_NAME}" 2>&1 | tee -a "${LOG_FILE}" | grep --line-buffered -v "Enabled fused SiLU+mul+FP4-quant for dense MLP down_proj input" &
log_follow_pid=$!

echo "Waiting for HTTP readiness at ${READY_URL}"
heartbeat=0
until curl -fsS "${READY_URL}" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "SGLang container exited before becoming ready"
    tail -n 200 "${LOG_FILE}" || true
    exit 1
  fi
  # The log itself is streaming above; only a light heartbeat every ~30s.
  if (( heartbeat % 6 == 0 )); then
    echo "  still starting..."
  fi
  heartbeat=$((heartbeat + 1))
  sleep 5
done

echo "SGLang is ready"
echo "OpenAI base URL: http://${HOST}:${PORT}/v1"
echo "Anthropic-compatible: http://${HOST}:${PORT}/v1/messages (no /v1 suffix in ANTHROPIC_BASE_URL)"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Thinking: ON by default (disable per request: chat_template_kwargs {\"enable_thinking\": false})"

echo "SGLang is ready and responding; shell is now free."
