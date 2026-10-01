#!/usr/bin/env bash
set -euo pipefail

# DFlash2 wrapper. Serves the start.sh target (default
# unsloth/Qwen3.6-35B-A3B-NVFP4, override with MODEL_ID) with the DFlash2
# block-diffusion draft incoai/Qwen3.6-35B-A3B-DFlash2 (trained against
# Qwen/Qwen3.6-35B-A3B; 6 sliding-window layers, block size 8, ~1 GB BF16),
# by injecting the spec flags via EXTRA_ARGS (appended last, argparse
# last-wins) and telling start.sh's pool math about the draft
# (SPEC_DRAFT_TOKENS, DRAFT_KV_BYTES_PER_TOKEN, DRAFT_WEIGHTS_GIB).
# The draft is pinned to DRAFT_MODEL@DRAFT_REVISION.
# Image: an official multi-arch lmsysorg/sglang nightly from main (pinned
# by its index digest, pulled from Docker Hub on first run). It carries
# DFlash2 (DFlash2DraftModel), Qwen3_5MoeForConditionalGeneration DFLASH
# aux-hidden capture, and extra_buffer_lazy support for DFLASH verify
# (#34763), so the target keeps start.sh's extra_buffer_lazy strategy
# (set DF_MAMBA_STRATEGY=extra_buffer to fall back to the strategy the
# 27B setup was validated with). Override with IMAGE=<ref>.
# Memory: --mem-fraction-static comes from MEM_FRACTION_STATIC (start.sh
# default 0.5). Never go above 0.90 on GB10: 0.95 hard-rebooted the box
# once at draft-graph capture, and the cookbook pins 0.80 because 0.85
# trips DGX OS earlyoom.
# Draft KV: without a draft window the draft pool aliases the target's
# token slots, costing 6 layers x 8 KV heads x 128 x 2 x 1 B (fp8, follows
# --kv-cache-dtype) = 12 KB per token on top of the target's 10 KB.
# DF_DRAFT_WINDOW=<n> (>= 8; the draft's own sliding window is 2048)
# enables SGLang's compact draft KV cache, bounding draft KV per request.
# DF_DRAFT_ATTN=<backend> overrides the draft attention backend (default:
# the target's flashinfer). The model card uses fa4 (measured on GB300);
# fa4 forces a BF16 draft KV (24 KB/token) and is untested on SM121.
# YaRN: the draft inherits --json-model-override-args, so start.sh refuses
# CONTEXT_LENGTH > 262144 while a draft is configured.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure the draft (and only the draft) is cached where the container reads it.
HF_CACHE="${SCRIPT_DIR}/.cache/huggingface/hub"
mkdir -p "${HF_CACHE}"

DRAFT_MODEL="${DRAFT_MODEL:-incoai/Qwen3.6-35B-A3B-DFlash2}"
DRAFT_REVISION="${DRAFT_REVISION:-51ef7b6923ad6c14cb1bb41c37a9041446496ab6}"

snapshot_present() {
  local base="${HF_CACHE}/models--${1//\//--}"
  local rev=""
  [[ -n "${DRAFT_REVISION}" ]] && rev="/snapshots/${DRAFT_REVISION}" || rev="/snapshots"
  [[ -n "$(find -L "${base}${rev}" -maxdepth 2 -type f -print -quit 2>/dev/null)" ]]
}

if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.bashrc" ]]; then
  HF_TOKEN="$(sed -n 's/^[[:space:]]*\(export[[:space:]]\+\)\?HF_TOKEN=["'"'"']\?\([A-Za-z0-9_-]\+\).*/\2/p' "${HOME}/.bashrc" | head -1)"
fi
export HF_TOKEN

# Official image, pinned by the multi-arch index digest (docker resolves the
# linux/arm64 child). Upstream build: main commit 708f51e44 (2026-09-09),
# nightly-cu134-20260909-708f51e — the first arm64 line carrying sglang
# #35255 (zombie-request fix; dev-qwen38-27b-dflash2 predates it, and
# v0.5.19 was tagged before it). To bump: `docker buildx imagetools
# inspect lmsysorg/sglang:<tag>` prints the index digest;
# update IMAGE_DIGEST, then re-validate on the box before trusting numbers.
IMAGE_REPO="lmsysorg/sglang"
IMAGE_TAG="nightly-cu134-20260909-708f51e"
IMAGE_DIGEST="sha256:00205b89f74691f76a0ffbd6846376d9323971930a5d59bf63a65dadc7d67927"
IMAGE_UPSTREAM_COMMIT="708f51e44"
IMAGE="${IMAGE:-${IMAGE_REPO}@${IMAGE_DIGEST}}"
LEGACY_IMAGE="lmsysorg/sglang:qwen38-27b-dflash2"   # the retired self-built image

ensure_image() {
  local pinned=0
  [[ "${IMAGE}" == "${IMAGE_REPO}@${IMAGE_DIGEST}" ]] && pinned=1
  if docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "Using ${IMAGE}"
    return
  fi
  echo "${IMAGE} not present locally — pulling from Docker Hub (~14 GB compressed for arm64) ..."
  docker pull "${IMAGE}" \
    || { echo "pull failed for ${IMAGE} — check network / Docker Hub rate limit (docker login helps); a tag that only ever existed locally must be rebuilt or loaded first (the retired builder is at commit 751e29e), or set IMAGE= to an image you have"; exit 1; }
  docker image inspect "${IMAGE}" >/dev/null 2>&1 \
    || { echo "pull reported success but ${IMAGE} is still missing"; exit 1; }
  (( pinned )) || return 0
  # A digest pull shows TAG=<none> in `docker images`; alias it so it reads as
  # what it is and does not look like a prune candidate. Never re-point a tag
  # that already exists (e.g. a manual `docker pull` of the rolling dev tag):
  # the container runs by digest either way.
  if docker image inspect "${IMAGE_REPO}:${IMAGE_TAG}" >/dev/null 2>&1; then
    docker image inspect --format '{{join .RepoDigests ","}}' "${IMAGE_REPO}:${IMAGE_TAG}" | grep -q "${IMAGE_DIGEST}" \
      || echo "note: local tag ${IMAGE_REPO}:${IMAGE_TAG} points at a different build; leaving it alone (this run uses the pinned digest)"
  else
    docker tag "${IMAGE}" "${IMAGE_REPO}:${IMAGE_TAG}" || true
  fi
  # Only now, with the new image safely on disk, mention the retired one.
  if docker image inspect "${LEGACY_IMAGE}" >/dev/null 2>&1; then
    echo "note: the self-built ${LEGACY_IMAGE} is no longer the default (IMAGE=${LEGACY_IMAGE} keeps using it; docker image rm ${LEGACY_IMAGE} frees the space)"
  fi
}
ensure_image
if [[ "${IMAGE}" == "${IMAGE_REPO}@${IMAGE_DIGEST}" ]]; then
  echo "DFlash image: ${IMAGE_REPO}:${IMAGE_TAG} @ ${IMAGE_DIGEST:0:19}… (upstream ${IMAGE_UPSTREAM_COMMIT})"
else
  echo "DFlash image: ${IMAGE} (IMAGE override)"
fi
export IMAGE

ensure_cached() {
  local repo="$1"
  local label="${repo}${DRAFT_REVISION:+ @ ${DRAFT_REVISION}}"
  if snapshot_present "${repo}"; then
    echo "draft already cached (${label})"
  else
    echo "draft not cached — pulling ${label} ..."
    docker run --rm --network host \
      -e HF_HOME=/root/.cache/huggingface \
      -e HF_TOKEN="${HF_TOKEN:-}" \
      -v "${SCRIPT_DIR}/.cache/huggingface:/root/.cache/huggingface" \
      "${IMAGE}" \
      python3 -c "from huggingface_hub import snapshot_download; snapshot_download('${repo}'${DRAFT_REVISION:+, revision='${DRAFT_REVISION}'})" \
      || { echo "pull failed for ${label}"; exit 1; }
    snapshot_present "${repo}" || { echo "pull failed for ${label}"; exit 1; }
  fi
}
ensure_cached "${DRAFT_MODEL}"

DF_BLOCK_SIZE="${DF_BLOCK_SIZE:-8}"
DF_DRAFT_WINDOW="${DF_DRAFT_WINDOW:-}"
DF_DRAFT_ATTN="${DF_DRAFT_ATTN:-}"
if ! [[ "${DF_BLOCK_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "DF_BLOCK_SIZE must be a positive integer, got '${DF_BLOCK_SIZE}'"; exit 1
fi

EXTRA_ARGS="--speculative-algorithm DFLASH \
--speculative-draft-model-path ${DRAFT_MODEL}${DRAFT_REVISION:+ --speculative-draft-model-revision ${DRAFT_REVISION}} \
--speculative-num-draft-tokens ${DF_BLOCK_SIZE}"

# Draft KV bytes per target token: 6 layers x 2 (K,V) x 8 heads x 128 dim
# x dtype bytes (fp8 = 1, follows --kv-cache-dtype; fa4 forces bf16 = 2).
DRAFT_KV_DTYPE_BYTES=1
if [[ -n "${DF_DRAFT_ATTN}" ]]; then
  EXTRA_ARGS+=" --speculative-draft-attention-backend ${DF_DRAFT_ATTN}"
  [[ "${DF_DRAFT_ATTN}" == "fa4" ]] && DRAFT_KV_DTYPE_BYTES=2
fi
DRAFT_KV_BYTES_PER_TOKEN=$(( 6 * 2 * 8 * 128 * DRAFT_KV_DTYPE_BYTES ))
if [[ -n "${DF_DRAFT_WINDOW}" ]]; then
  if ! [[ "${DF_DRAFT_WINDOW}" =~ ^[0-9]+$ ]] || (( DF_DRAFT_WINDOW < DF_BLOCK_SIZE )); then
    echo "DF_DRAFT_WINDOW must be an integer >= DF_BLOCK_SIZE (${DF_BLOCK_SIZE}), got '${DF_DRAFT_WINDOW}'"; exit 1
  fi
  EXTRA_ARGS+=" --speculative-draft-window-size ${DF_DRAFT_WINDOW}"
  # Compact cache: draft KV is bounded per request, not per target token.
  DRAFT_KV_BYTES_PER_TOKEN=0
fi
EXTRA_ARGS+=" ${DF_EXTRA:-}"
export EXTRA_ARGS

export SPEC_LABEL="DFLASH ${DRAFT_MODEL} (block ${DF_BLOCK_SIZE}${DF_DRAFT_WINDOW:+, draft window ${DF_DRAFT_WINDOW}}${DF_DRAFT_ATTN:+, draft attn ${DF_DRAFT_ATTN}})"
export SPEC_DRAFT_TOKENS="${DF_BLOCK_SIZE}"
export DRAFT_KV_BYTES_PER_TOKEN
export DRAFT_WEIGHTS_GIB="${DRAFT_WEIGHTS_GIB:-1.0}"
export MAMBA_RADIX_STRATEGY="${DF_MAMBA_STRATEGY:-${MAMBA_RADIX_STRATEGY:-extra_buffer_lazy}}"

echo "DFlash mode: EXTRA_ARGS=${EXTRA_ARGS}"
echo "Delegating to ${SCRIPT_DIR}/start.sh"
exec "${SCRIPT_DIR}/start.sh"
