# Qwen3.6-35B-A3B on SGLang for DGX Spark

> ⚡ Decode (output-token) throughput: **162.7 tok/s single-stream** with DFlash2 speculative decoding (1.8× plain decoding). Peak total decode throughput by context per stream: **489 tok/s @ 262K** (28 streams) · **1,059 tok/s @ 32K** (189 streams) · **1,213 tok/s @ 16K** (263 streams) · **1,390 tok/s @ 8K** (367 streams) · **1,434 tok/s @ 4K** (457 streams), all on one GB10.

Ready-to-run scripts to serve **[Qwen3.6-35B-A3B](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4)** (Unsloth NVFP4) with **[SGLang](https://docs.sglang.io)** in Docker on an NVIDIA DGX Spark (GB10, 128 GB unified memory). Three decode modes are available:

| Script | Decode mode | Best for |
|---|---|---|
| `./start-dflash.sh` | **DFlash2** speculative decoding ([`incoai/Qwen3.6-35B-A3B-DFlash2`](https://huggingface.co/incoai/Qwen3.6-35B-A3B-DFlash2), block 8) | **Recommended.** Fastest per stream: 1.3-1.8× at 1-13 concurrent streams |
| `./start-mtp.sh` | MTP speculative decoding (the checkpoint's built-in head, 3 steps) | Context above 262K (YaRN), which DFlash doesn't support |
| `./start.sh` | Plain decoding (no speculation) | Many concurrent streams (dozens to hundreds) |

All three serve an OpenAI-compatible API on port **8888** with the model name **`qwen3.6-35b-a3b-sglang`**. They share one container name, so only one can run at a time.

## Requirements

- DGX Spark / GB10 (aarch64, SM121) with Docker and the NVIDIA container runtime
- About 30 GB of disk for the weights (target 24.7 GB plus the DFlash draft at about 1 GB), downloaded into `./.cache/huggingface` on first start
- Optional: `export HF_TOKEN=...` in `~/.bashrc` for faster Hub downloads. The first boot without a token took about 18 minutes; cached boots take about 2-3 minutes.

## Quick start

```bash
cp .env.sample .env          # optional; defaults work without it
./start-dflash.sh            # or ./start.sh / ./start-mtp.sh
./stop.sh                    # stops whichever is running
```

The start script streams the server log (also saved to `.sglang.log`) and returns once the API responds. Test it with:

```bash
curl http://127.0.0.1:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.6-35b-a3b-sglang",
  "messages": [{"role": "user", "content": "Hello"}]
}'
```

- **Thinking** is on by default; reasoning comes back in `reasoning_content`. Disable it per request with `"chat_template_kwargs": {"enable_thinking": false}`.
- **Tool calling** works out of the box (`qwen3_coder` parser); just send `tools`.
- **Default sampling** is temperature 0.6, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0 (the model card's "precise coding" profile). It applies only when a request omits a value; values your client sends always win. Change it with `SAMPLING_*` in `.env`.

## Configuration (`.env` or shell env)

| Variable | Default | Meaning |
|---|---|---|
| `MAX_CONCURRENT_REQUESTS` | `2` | Concurrent streams. The script sizes the GDN state pool and the KV cache cap from this. |
| `CONTEXT_LENGTH` | `262144` | Max context per stream (`1024..1000000`). Above 262144 needs `YARN=1` (auto at exactly 1M). Not available with DFlash. |
| `MEM_FRACTION_STATIC` | `0.5` | Ceiling on the memory SGLang may reserve. **Keep ≤ 0.80** (see [Memory safety](#memory-safety)). |
| `MODEL_ID` | `unsloth/Qwen3.6-35B-A3B-NVFP4` | Target checkpoint |
| `MAX_TOTAL_TOKENS` | auto | KV cap. Default `N × (CONTEXT_LENGTH + draft tokens)`; `0` = use everything the fraction allows. |
| `MAMBA_POOL_MODE` | `auto` | GDN state pool sizing: `pin` (N × 4 slots) or `ratio` (computed `--mamba-full-memory-ratio`) |
| `SAMPLING_TEMPERATURE` / `_TOP_P` / `_TOP_K` / `_MIN_P` / `_REPETITION_PENALTY` | `0.6` / `0.95` / `20` / `0.0` / `1.0` | Server default sampling. Applied by mounting a patched `generation_config.json` into the container; the host cache isn't changed. |
| `DF_BLOCK_SIZE` | `8` | DFlash draft tokens per step (8 measured best) |
| `MTP_STEPS` / `MTP_DRAFT` | `3` / `4` | MTP chain length (3 measured best) |
| `EXTRA_ARGS` | — | Extra SGLang flags, appended last |
| `DOCKER_ENV` | — | Extra container env, e.g. `SGLANG_FLASHINFER_WORKSPACE_SIZE=1073741824` (needed for spec modes above ~150 streams) |

On every start, the script prints the derived state pool and KV cache sizes, and warns if `MAX_CONCURRENT_REQUESTS × CONTEXT_LENGTH` won't fit in the memory budget.

## Image

All scripts use the official SGLang nightly, pinned by digest: `lmsysorg/sglang@sha256:00205b89…` (= `nightly-cu134-20260909-708f51e`). It's pulled automatically on first run.

> ⚠️ Do **not** use the older `lmsysorg/sglang:qwen38-27b` image with this checkpoint. It drops the FP8 `lm_head` scales, and every reply collapses into one repeated word ("hello hello hello…").

## Benchmark results (GB10)

Output tok/s, wall time including prefill, default sampling. Full data and method are in [`numbers.md`](numbers.md); the exact best launch command is in [`best-config.txt`](best-config.txt).

### Single and low concurrency (0.5 fraction, 262K context)

| Mode | c=1 mean | reasoning | chat | code | essay | c=2 | c=8 | c=10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no spec | 66.7 | 66.4 | 67.0 | 66.6 | 66.6 | 94.0 | 204.3 | 229.2 |
| **DFlash block 8** | **120.1** | **154.8** | **85.9** | **162.7** 🏆 | **77.1** | **149.8** | **280.0** | **301.0** |
| MTP 3 steps | 86.3 | 91.5 | 74.2 | 104.0 | 75.6 | 131.8 | 254.0 | 288.6 |

- **Bold** marks the best value in each column; 🏆 marks the single-stream peak (**162.7 tok/s**, DFlash on code).
- **DFlash is 1.8× faster at c=1** and 1.3-1.4× at c=8-10. Gains are biggest on reasoning and code (~2.4×) and smallest on free-form prose (~1.2×).
- **DFlash block sizes** 4/6/10/12/16 and fa4 draft attention were all tested; block 8 with flashinfer is best (see `numbers.md`).
- **More MTP steps** (6, 8) are slower than 3.

### High concurrency (0.95 fraction, max safe streams)

| Context per stream | No spec: concurrency | No spec: total ctx | No spec: total tok/s | MTP: concurrency | MTP: total ctx | MTP: total tok/s | DFlash: concurrency | DFlash: total ctx | DFlash: total tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **262K** | **31** 🏆 | 8.13M | 394 | 28 | 7.34M | **489** 🏆 | 13 | 3.41M | 351 |
| **32K** | **189** 🏆 | 6.19M | **1,059** 🏆 | 126 | 4.13M | 766 | 77\* | 2.52M | — |
| **16K** | **263** 🏆 | 4.31M | **1,213** 🏆 | 174 | 2.85M | 607 | 113\* | 1.85M | — |
| **8K** | **367** 🏆 | 3.01M | **1,390** 🏆 | 219 | 1.79M | 547 | 150\* | 1.23M | — |
| **4K** | **457** 🏆 | 1.87M | **1,434** 🏆 | 276\* | 1.13M | — | 160\* | 0.66M | — |

- **concurrency** = concurrent streams that all fit at their full context; **total ctx** = concurrency × context per stream (KV tokens in use); **total tok/s** = combined output throughput across all streams.
- **Bold** + 🏆 = best in each row (per context size), marked separately for highest **concurrency** and highest **total tok/s**.
- \* = the boot-measured maximum; boots, but the throughput benchmark wasn't run. Expect the safe value to be about 90% of it.

- **Above ~10-13 streams, plain decoding wins** on combined throughput: speculation pays for its draft steps across the whole batch.
- **Max safe concurrency ≈ 90% of what boots.** At the exact boot maximum, load pushed free RAM under 3 GB within seconds.

### How many full-length (262K) streams fit

| `MEM_FRACTION_STATIC` | DFlash | MTP | no spec |
|---:|---:|---:|---:|
| 0.50 | ~4.7 | ~9.6 | ~11 |
| 0.80 | ~10 | — | ~25 |
| 0.95 | 13 | 28 | 31 |

DFlash fits the fewest because its draft keeps its own KV (22 KB/token vs 10 KB). Shorter contexts fit proportionally more.

## Memory safety

GB10's GPU and OS share one memory pool, and **GPU allocations can't be swapped**. If SGLang plus its runtime buffers exhaust RAM, the box doesn't fail cleanly: during testing it hard-froze twice (no display; it needed a power-cycle) at 0.95 with spec modes at their maximum concurrency.

- **Keep `MEM_FRACTION_STATIC` ≤ 0.80**, and set `MAX_CONCURRENT_REQUESTS` to about 90% of what fits.
- **The 0.5 default** with 2 streams uses about 40 GB and leaves about 70 GB free.
- **Before high-concurrency experiments**, run a watchdog that does `docker kill` when `MemAvailable` drops below ~3 GB. That turned every would-be freeze into a clean container kill.

## Files

| File | Purpose |
|---|---|
| `start.sh` | Main launcher (no spec); holds all the pool-sizing logic |
| `start-dflash.sh` / `start-mtp.sh` | Thin wrappers that add the speculative-decoding flags, then call `start.sh` |
| `stop.sh` | Stops the container (idempotent) |
| `.env.sample` | All settings, documented |
| `best-config.txt` | Best known launch config with the exact server command |
| `numbers.md` | All measurements: memory constants, sweeps, capacity, crash notes |

## Links

- [Unsloth Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4) · [DFlash2 draft](https://huggingface.co/incoai/Qwen3.6-35B-A3B-DFlash2) · [SGLang docs](https://docs.sglang.io)

## Credits

- **[MiaAI-Lab / Qwen3.8-27B-SGLang-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark):** the base recipe this repo is built on (launch scripts, GDN state-pool sizing, DFlash/MTP wrappers, GB10 tuning).
- **[Travis's Ornith-1.5 recipe on Spark Arena](https://spark-arena.com/travis):** a reference DGX Spark SGLang config for a Qwen3.6-35B-A3B finetune with DFlash, used for comparison (MoE backend, draft tokens, page-cache handling).
- **[SGLang](https://github.com/sgl-project/sglang):** the serving engine, including DFlash2, MTP/EAGLE, the hybrid GDN radix cache and the quantized `lm_head` fix this setup depends on.
- **Model owners:**
  - [Qwen team](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) for Qwen3.6-35B-A3B
  - [Unsloth](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4) for the NVFP4 quantization
  - [Inco AI](https://huggingface.co/incoai/Qwen3.6-35B-A3B-DFlash2) / [z-lab](https://github.com/z-lab/dflash) for the DFlash2 draft model
