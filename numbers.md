# Numbers: Qwen3.6-35B-A3B NVFP4 on SGLang, DGX Spark (GB10)

All measurements were taken on this box (GB10, SM121, 128 GB unified memory;
SGLang sees 124544 MiB ≈ 121.6 GiB) on 2026-10-01/02. The best launch config
is in [`best-config.txt`](best-config.txt).

## Setup under test

| Item | Value |
|---|---|
| Target | `unsloth/Qwen3.6-35B-A3B-NVFP4` (compressed-tensors, mixed precision) |
| Draft (DFlash) | `incoai/Qwen3.6-35B-A3B-DFlash2` @ `51ef7b6`, BF16, ~1 GB |
| Image | `lmsysorg/sglang@sha256:00205b89…` (= `nightly-cu134-20260909-708f51e`) |
| Fixed flags | flashinfer attention, `--kv-cache-dtype fp8_e4m3`, `--mamba-ssm-dtype bfloat16`, `extra_buffer_lazy`, chunked prefill 8192, CPU pinned to `5-9,15-19` |
| Default memory | `--mem-fraction-static 0.5`, `MAX_CONCURRENT_REQUESTS=2`, `CONTEXT_LENGTH=262144` |

### Checkpoint facts that matter

- 40 layers: 30 GDN (linear attention) + 10 full attention (2 KV heads × 256). There are 256 experts, 8 of them active.
- Quantization groups:
  - **FP8 W8A8 per-channel:** attention q/k/v/o, GDN in_proj_qkv/in_proj_z/out_proj, **lm_head**, and every expert plus the shared expert in **layers 32-39**.
  - **NVFP4 W4A4:** all other experts.
  - **Unquantized:** the vision tower, the MTP head (`mtp.*`) and the GDN gates.
- The checkpoint declares an FP8 KV scheme but ships no KV scales, so SGLang uses a scale of 1.0.

## Bug found: output collapses into one repeated word

| Image | Result |
|---|---|
| `lmsysorg/sglang:qwen38-27b` | "Hello" → `…greeting the user hello hello hello…` (686× identical), same at temp 1.0 and greedy |
| nightly `708f51e` | Normal output ("Hello! How can I help you today?"), correct reasoning |

- **Cause:** the old image drops the FP8 `lm_head` per-channel scales (`Parameter lm_head.weight_scale not found in params_dict`), so the logits come out wrong.
- **Fix:** the nightly image includes the quantized-lm_head support (sglang #35496).
- **Also found:** `--moe-runner-backend flashinfer_cutlass`, used by a community recipe, crashes this checkpoint at boot (`'CompressedTensorsW8A8Fp8MoE' object has no attribute 'runner'`) because layers 32-39 are FP8 MoE. Keep `auto`.

## Per-unit memory constants (derived, then confirmed by boot logs)

| Constant | Derived | Measured (boot log) |
|---|---|---|
| GDN state per slot (bf16 SSM + conv) | 30 × (32·128·128·2 + 3·8192·2) B = 31.4 MiB | 8-slot pin → `ssm 0.26 + conv 0.01 GB` = 9 slots (8 + padding) × 31.4 MiB ✓ |
| Target KV per token (fp8) | 10 × 2 × 2 × 256 × 1 B = 10,240 B | 524,288 tokens → K+V 5.00 GB = 10,240 B ✓ |
| DFlash draft KV per token (fp8) | 6 × 2 × 8 × 128 × 1 B = 12,288 B | 524,304 tokens → K+V 6.00 GB = 12,288 B ✓ |
| MTP draft KV per token | 1 × 2 × 2 × 256 × 1 B = 1,024 B | not measured |
| DFlash intermediate GDN states | (N+1) × 8 × 31.4 MiB | N=2 → 0.70 GB ✓; N=10 → 2.58 GB; N=13 → 3.28 GB |
| Target weights | 24.7 GiB safetensors | 24.5-24.6 GB (cached boots); 26.2-27.4 GB on some boots |
| Draft weights | ~1.0 GB | 1.18-1.22 GB |

### compute-mamba-ratio skill values

- token_equiv = state / KV per token:
  - **3216** without a draft
  - **1462** with DFlash (10 + 12 KB per token)
  - **2924** with MTP
- r\* = (S + D) × token_equiv / L, with S = 4 and D = draft tokens:

| L | no spec | DFlash (D=8) | MTP (D=4) |
|---|---|---|---|
| 8K | 1.57 | — | — |
| 32K | 0.39 | — | — |
| 262K | 0.049 | 0.067 | 0.089 |

At 262K r\* is below 0.15, so `start.sh` pins `--max-mamba-cache-size = N × 4` and caps `--max-total-tokens = N × (CONTEXT_LENGTH + D)`. SGLang ignores the ratio when the pin is set.

## Boot timings

| Event | Time |
|---|---|
| First boot, weights downloaded without `HF_TOKEN` | 1102 s weight load (≈ 18 min) |
| Cached target weight load | 29-31 s (one boot: 112 s) |
| DFlash draft load | ~6 s |
| Decode CUDA graph capture (bs 1-2) | ~13 s |

## Speed: concurrency 1 and 2

**Harness:**
- Each config was booted fresh, then warmed up with one request.
- **c=1:** 4 prompts × 2 runs, `max_tokens` 1024, default sampling (temp 1.0, top_p 0.95, top_k 20), non-streaming. Speed is output tok/s over wall time, prefill included.
- **c=2:** the code and essay prompts sent together, 2 rounds; the table shows combined tok/s.
- **Accept len:** generated tokens ÷ spec verify calls, including the bonus token.

**Prompts:**
- **reason:** bat-and-ball puzzle (thinking on)
- **chat:** "Explain how a refrigerator works" (thinking on)
- **code:** Python LRU cache (thinking off)
- **essay:** 500-word printing-press essay (thinking off)

| Config | c=1 mean | reason | chat | code | essay | c=2 agg | accept len |
|---|---:|---:|---:|---:|---:|---:|---:|
| no spec | 66.7 | 66.4 | 67.0 | 66.6 | 66.6 | 94.0 | — |
| DFlash block 4 | 76.3 | 89.4 | 66.3 | 89.0 | 60.6 | 114.9 | 2.95 |
| DFlash block 6 | 103.8 | 137.1 | 81.9 | 130.2 | 66.2 | 114.1 | 3.61 |
| **DFlash block 8** | **120.1** | 154.8 | 85.9 | 162.7 | 77.1 | **149.8** | 3.85 |
| DFlash block 10 | 113.3 | 140.9 | 79.8 | 163.7 | 68.8 | 139.2 | 4.07 |
| DFlash block 12 | 120.5 | 172.5 | 80.8 | 160.3 | 68.3 | 126.8 | 4.20 |
| DFlash block 16 | 114.8 | 158.1 | 77.7 | 162.9 | 60.4 | 123.0 | 4.27 |
| DFlash block 8 + fa4 draft attn | 120.8 | 156.3 | 89.6 | 161.5 | 75.8 | 145.5 | 3.84 |
| MTP 3 steps (draft 4) | 86.3 | 91.5 | 74.2 | 104.0 | 75.6 | 131.8 | 2.93 |
| MTP 6 steps (draft 7) | 85.1 | 104.3 | 66.7 | 109.7 | 59.6 | 116.1 | 3.67 |
| MTP 8 steps (draft 9) | 75.4 | 94.2 | 57.0 | 101.9 | 48.7 | 95.6 | 3.86 |

- DFlash block 6, MTP 6 and MTP 8 each had one "reason" answer-check miss at temp 1.0. Spec verification is lossless, so this is a phrasing miss or a thinking trace that hit `max_tokens`, not a decode fault.
- Differences under about 5% at c=1 are within run-to-run noise.

## Speed: concurrency 10 (0.5 fraction)

10 concurrent requests (the 4 prompts round-robin), 2 rounds.

| Config | Combined tok/s | Per-request mean | Slowest request | Accept len |
|---|---:|---:|---:|---:|
| no spec | 223.9 | 26.4 | 25.8 | — |
| **DFlash block 8** | **308.1** | **44.5** | 22.5 | 3.87 |

## KV capacity by `--mem-fraction-static` (boot logs, KV cap off)

Full context = 262,144 tokens. These numbers come from boots only; no load test was run.

| Fraction | Mode | N pinned | `max_total_num_tokens` | Full contexts | KV size (target + draft) | Free after boot |
|---:|---|---:|---:|---:|---|---:|
| 0.50 | DFlash block 8 | 10 | 1,225,770 | **4.7** | 11.7 + 14.0 GB | ~52 GB |
| 0.90 | DFlash block 8 | 13 | 3,357,023 | **12.8** | 32.0 + 38.4 GB | ~7 GB |
| 0.90 | no spec | 28 | 7,544,800 | **28.8** | 72.0 GB | ~11 GB |
| 0.93 | DFlash block 8 | 13 | 3,506,760 | **13.4** | 33.4 + 40.1 GB | ~8 GB (min 6.5 GB during boot) |
| 0.93 | no spec | 30 | 7,881,624 | **30.1** | 75.2 GB | ~10.5 GB (min 10.4 GB during boot) |
| 0.95 | DFlash block 8 | 13 | 3,605,460 | **13.75** | 34.4 + 41.3 GB | ~5.4 GB (min 5.5 GB; **swap rose to 4.9 GB**) |
| 0.95 | no spec | 31 | 8,209,240 | **31.3** | 78.3 GB | ~4.5 GB (min 4.2 GB; swap 1.0 GB) |

- **Estimate accuracy:** before booting, I estimated 13.3 contexts for DFlash at 0.9 (4% high) and about 28 for no spec at 0.9 (on target).
- **Interpolation:** about 2.4 full DFlash contexts per +0.1 of fraction, so 0.8 ≈ 10.4 and 0.7 ≈ 8. No spec fits about 2.25× as many contexts as DFlash at the same fraction.
- **0.9 → 0.93:** +0.03 added about 150K DFlash tokens (+0.6 contexts) and about 337K no-spec tokens (+1.3 contexts). Free memory after boot did *not* shrink, because SGLang took the extra from page cache: weight load slowed from ~30 s to ~107 s as cached weight files were evicted. Both 0.93 boots finished without earlyoom; the lowest free memory seen was 6.5 GB, during DFlash's draft-graph capture.
- **0.93 → 0.95:** +0.4 DFlash contexts, +1.2 no-spec contexts. Both booted, but the margin is gone: DFlash pushed 4.9 GB of CPU-side memory into swap during boot, and no spec left only 4.2-4.5 GB free. GPU memory can't be swapped, so under long-context load any further runtime allocation (prefill working buffers) has nowhere to go. Capacity gain over 0.8 is about +3.3 (DFlash) and +6 (no spec) full contexts.
- **Safety:** at 0.9 only 6-7 GB (DFlash) or 10-11 GB (no spec) is left for the OS and runtime buffers. That's fine for booting, but risky under long-context load. The old 27B notes say 0.95 hard-rebooted the box, and the cookbook warns that 0.85 trips earlyoom (earlyoom isn't running on this box; see below). **Keep ≤ 0.80.**

### Capacity summary: how many full 262K requests can run at once

| Fraction | DFlash block 8 | No spec | Free after boot (DFlash / no spec) | Verdict |
|---:|---:|---:|---|---|
| 0.50 | 4 (4.7) | ~11 | ~52 GB / — | default; plenty of headroom |
| 0.70 | ~8 (est.) | ~20 (est.) | — | |
| 0.80 | ~10 (est.) | ~25 (est.) | — | **recommended ceiling** |
| 0.90 | 12 (12.8) | 28 (28.8) | ~7 / ~11 GB | boots; risky under load |
| 0.93 | 13 (13.4) | 30 (30.1) | ~8 / ~10.5 GB | boots; risky under load |
| 0.95 | 13 (13.75) | 31 (31.3) | ~5.4 / ~4.5 GB (DFlash swapped 4.9 GB) | boots; no margin left |

Whole numbers are full contexts that fit at once; measured values are in brackets, and "est." rows are interpolated. Shorter conversations fit proportionally more (e.g. ~4× as many at 64K). Each boot was capacity-only (KV cap off, no load test).

### What happens past 100% memory on this box

- **Setup:** 16 GB swap file (`/swap.img`), swappiness 60. **earlyoom and systemd-oomd are both inactive**, so only the kernel's built-in OOM killer is present.
- **GPU allocations can't be swapped.** Weights, KV, state pools and CUDA graphs are pinned driver memory, which is almost everything `--mem-fraction-static` reserves. Swap only absorbs CPU-side memory (Python, tokenizer, other apps).
- **Order of events as memory runs out:**
  1. The page cache is dropped (seen at 0.93/0.95 as slower weight loads, ~30 s → ~107 s).
  2. CPU-side memory is swapped out (seen at 0.95: DFlash pushed 4.9 GB to swap), and the whole box slows down.
  3. A CUDA allocation fails. At boot this is a clean out-of-memory exit; at runtime the request fails or the server crashes.
  4. The kernel OOM killer kills the largest process, or the driver/kernel stalls and the box hangs or reboots (the old 0.95 hard reboot).

## Concurrency 8 at 262K context (0.5 fraction)

`CONTEXT_LENGTH=262144`, `MAX_CONCURRENT_REQUESTS=8`, `MEM_FRACTION_STATIC=0.5`. 8 concurrent requests (the 4 prompts round-robin), 2 rounds, `max_tokens` 1024.

| Mode | Combined tok/s | Per-request mean | Slowest request | Accept len | KV tokens | Full 262K contexts that fit | Free GPU after boot |
|---|---:|---:|---:|---:|---:|---:|---:|
| no spec | 204.3 | 29.4 | 28.1 | — | 2,097,152 (cap) | 8 | 64.4 GB |
| **DFlash block 8** | **280.0** | **50.3** | 32.0 | 3.94 | 1,230,813 | **4.7** | 52.9 GB |
| MTP 3 steps | 254.0 | 40.0 | 30.5 | 2.89 | 2,097,184 (cap) | 8 | 57.6 GB |

- **DFlash is fastest:** 1.37× no spec combined, 1.7× per request. Even its slowest request (32.0) beats no spec's per-request mean.
- **MTP sits in between:** 1.24× combined.
- **Capacity:** at 0.5, DFlash fits only ~4.7 full 262K contexts. Eight full-length conversations at once need ~0.72 for DFlash; no spec and MTP fit all 8 at 0.5.

## Concurrency 10 at 262K context (0.5 fraction)

Same setup as concurrency 8, with `MAX_CONCURRENT_REQUESTS=10`.

| Mode | Combined tok/s | Per-request mean | Slowest request | Accept len | KV tokens | Full 262K contexts that fit | Free GPU after boot |
|---|---:|---:|---:|---:|---:|---:|---:|
| no spec | 229.2 | 26.8 | 25.8 | — | 2,621,440 (cap) | 10 | 58.5 GB |
| **DFlash block 8** | **301.0** | **43.7** | 27.1 | 3.87 | 1,190,689 | **4.5** | 52.2 GB |
| MTP 3 steps | 288.6 | 35.3 | 26.8 | 2.89 | 2,513,770 | 9.6 | 52.5 GB |

- **DFlash is still fastest:** 1.31× no spec combined, 1.6× per request. Its lead shrinks as concurrency grows (1.37× at c=8).
- **MTP closes in:** 1.26× combined, nearly tied with DFlash on aggregate.
- **Slowest requests (prose):** about equal across all three at c=10.
- **Capacity at 0.5:** DFlash fits ~4.5 full 262K contexts and MTP ~9.6; no spec fits all 10.

## Max concurrency at 0.95 by context per stream (boot-measured)

`MEM_FRACTION_STATIC=0.95`, `CONTEXT_LENGTH=C`, `MAMBA_POOL_MODE=pin`, KV cap off. MTP and DFlash need `SGLANG_FLASHINFER_WORKSPACE_SIZE=1 GiB` at these batch sizes; the default 384 MB overflows during CUDA graph capture. **Max N** is the largest `MAX_CONCURRENT_REQUESTS` whose boot gave KV tokens ≥ N × (C + draft tokens), i.e. every stream can reach its full C at once. This is boot-only; no load was applied.

| Context per stream | **no spec** | **MTP 3 steps** | **DFlash block 8** |
|---:|---:|---:|---:|
| 32K (32,768) | **189** | **140** | **77** |
| 16K (16,384) | **293** | **194** | **113** |
| 8K (8,192) | **408** | **244** | **150** |
| 4K (4,096) | **508** | **276** | ✗ machine froze at N=179 (see below) |
| 262K (from earlier) | 31 | 28 | 13 |

Lowest free RAM during each winning boot:

| Context | no spec | MTP | DFlash |
|---:|---:|---:|---:|
| 32K | 4.5 GB | 0.9 GB | 0.5 GB |
| 16K | 5.3 GB | 1.5 GB | 0.4 GB |
| 8K | 5.6 GB | 1.8 GB | **0.2 GB** |
| 4K | 5.1 GB | 2.3 GB | — |

- **Why spec modes fit fewer:** each request adds draft intermediate GDN states (MTP +4, DFlash +8 slots × 31.4 MiB) on top of the 4 state slots, and DFlash also adds 12 KB/token of draft KV. At short contexts the per-request state dominates, so the gap widens: no spec fits 1.8× MTP and 3.4×+ DFlash at 8K.
- **Margins:** no spec keeps ~4.5-5.6 GB free at its max. Spec modes at their max leave 0.2-2.3 GB, which is too little to survive real load (see the crashes below).

### Machine freeze #2: DFlash, 179 × 4K, 0.95 (2026-10-02 11:27)

- **Sequence:**
  1. The 150 × 8K boot passed with only 201 MiB free.
  2. The next boot (179 × 4K) drove free RAM to **2-48 MiB**, and the box sat there for **~50 minutes**. The memory log shows ~40 MiB available, sampled every 2 s, from 11:28 until it stopped at 12:19.
  3. The kernel logged repeated `NVRM: Out of memory [NV_ERR_NO_MEMORY]` from 11:29 on.
  4. No OOM kill freed anything. The display went dark and the machine needed a manual power-cycle (new boot at 12:19).
- **Cause:** the GPU driver's own allocations couldn't be satisfied, and pinned GPU memory can't be swapped. This is the same failure as the first crash (MTP 256 × 4K at 0.95).
- **Conclusion:** max N at 0.95 for spec modes is a **paper limit**. At or near it, booting alone can hang the machine.

## Threshold throughput at 0.95 (max safe concurrency per context)

**Method:** boot each mode at the boot-measured max N (previous section) with `MEM_FRACTION_STATIC=0.95`, `CONTEXT_LENGTH=C`, pinned state pool, and the KV cap **on** (N × (C + draft tokens)). MTP and DFlash use a 1 GiB FlashInfer workspace. Then load with N concurrent requests (the 4 prompts round-robin, 2 rounds, `max_tokens` 1024).

**Safety:**
- A **watchdog** `docker kill`s the container if `MemAvailable` drops below 3 GB.
- The benchmark only starts if the boot leaves at least 3.5 GB free.
- If either fails, N is cut by 10% and the boot retried, up to 3 attempts.

"Max safe N" below is the largest N that booted and finished the benchmark without tripping the watchdog.

| Mode | Context | Boot max N | **Max safe N** | Combined tok/s | Per-request mean | Slowest request | Accept len | Min free RAM under load |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no spec | 32K | 189 | **189** | **1059.3** | 6.4 | 6.2 | — | 3.8 GB |
| no spec | 16K | 293 | **263** | **1212.9** | 5.3 | 4.9 | — | 11.3 GB |
| no spec | 8K | 408 | **367** | **1389.9** | 4.3 | 3.1 | — | 10.8 GB |
| no spec | 4K | 508 | **457** | **1434.2** | 3.6 | 2.4 | — | 10.8 GB |
| MTP 3 steps | 32K | 140 | **126** | **766.0** | 7.2 | 5.2 | 2.9 | 4.9 GB |
| MTP 3 steps | 16K | 194 | **174** | **607.1** | 3.9 | 2.8 | 2.9 | 5.0 GB |
| MTP 3 steps | 8K | 244 | **219** | **547.4** | 2.8 | 2.0 | 2.9 | 5.8 GB |
| MTP 3 steps | 4K | 276 | not run (stopped) | — | — | — | — | — |
| DFlash block 8 | 32K | 77 | not run (stopped) | — | — | — | — | — |
| DFlash block 8 | 16K | 113 | not run (stopped) | — | — | — | — | — |
| DFlash block 8 | 8K | 150 | not run (stopped) | — | — | — | — | — |
| DFlash block 8 | 4K | — (froze at 179) | 160 boots (3.96 GB free after boot); benchmark not run | — | — | — | — | — |

**What happened at the boot max N:**
- **no spec 16K/8K/4K (293/408/508):** booted with ~5.1-5.6 GB free, but the watchdog fired **within ~7 s of load starting** (free RAM fell to 2.6-3.0 GB). Runtime buffers pushed the box under. The 10% back-off then left 14-15 GB free after boot and 10.8-11.3 GB under load.
- **no spec 32K (189):** the only boot-max point that survived load, with 3.8 GB minimum free.
- **MTP 32K/16K/8K (140/194/244):** tripped the watchdog **during boot**. With the KV cap on, boot memory differs a little from the uncapped search. At 90% of boot max they ran with ~5-6 GB minimum free.
- **The watchdog prevented every freeze:** all 6 trips ended in a clean `docker kill`, against 2 hard freezes without it.

**Takeaways at 0.95:**
- **Max safe concurrency ≈ 90% of the boot-measured maximum.** Plan around the safe N, not the boot max.
- **At high concurrency, no spec beats MTP on combined throughput** (1059-1434 vs 547-766 tok/s): every MTP draft step multiplies across the whole batch. MTP only wins per request at 32K (7.2 vs 6.4), and with 126 vs 189 streams.
- **Shorter context → more streams → more combined tok/s, but slower streams:**
  - no spec: 32K → 4K goes 1059 → 1434 tok/s combined, but per-request 6.4 → 3.6 tok/s.
  - MTP gets *slower* in combined terms as N grows (766 → 547), because draft overhead grows with batch size.
- **For many concurrent streams, use no spec.** DFlash/MTP pay off at low concurrency (≤ ~10-13 streams, 1.3-1.8×; see the concurrency 1/2/8/10 sections).

## 0.95: throughput at max concurrency, and 256 instances

`MEM_FRACTION_STATIC=0.95`, KV cap off. Throughput uses the same 4 prompts round-robin across N concurrent requests, 2 rounds, `max_tokens` 1024. Free memory is `free -m` "available", sampled every 2 s during boot and benchmark.

### 1) Max concurrency at 262K context

N = full 262K contexts that fit (KV tokens ÷ (262144 + draft tokens)); the server was then booted with `MAX_CONCURRENT_REQUESTS=N`.

| Mode | KV tokens | N | Combined tok/s | Per-request mean | Slowest request | Accept len | Free GPU after boot | Min free RAM | Max swap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no spec | 8,194,393 | **31** | 393.8 | 14.8 | 14.2 | — | 7.4 GB | 5.0 GB | 4.5 GB |
| DFlash block 8 | 3,607,636 | **13** | 351.3 | 39.3 | 23.7 | 3.96 | 5.4 GB | 3.6 GB | 4.8 GB |
| MTP 3 steps | 7,519,133 | **28** | **488.8** | 21.3 | 15.7 | 2.89 | 0.9 GB | **0.4 GB** | 6.9 GB |

- **Fastest combined:** MTP (28 requests), at the cost of almost no free memory.
- **Fastest per request:** DFlash, but it fits only 13 long requests.
- **Most long requests:** no spec (31).

### 2) 256 instances × 4K context

| Mode | Result | Combined tok/s | Per-request mean |
|---|---|---:|---:|
| no spec | ✅ ran | **1214.5** | 5.4 (min 5.1) |
| DFlash block 8 | ❌ boot failed: state pool alone exceeds the budget ("Loaded weights leave no GPU memory for the KV cache") | — | — |
| DFlash block 3 | ❌ boot failed during CUDA graph capture: FlashInfer workspace overflow (`batch_prefill_tmp_v`, 585 MB needed, 384 MB default workspace) | — | — |
| MTP 3 steps | ❌ boot failed during CUDA graph capture: FlashInfer workspace overflow (default 384 MB) | — | — |
| MTP 3 steps, 1 GB workspace (`SGLANG_FLASHINFER_WORKSPACE_SIZE`) | booted (only 3.1 GB GPU free), then **the machine went down** as the 256-request benchmark started: free RAM fell to 8 MiB with ~8 GB in swap; the container exited 137 and the box rebooted | — | — |

MTP's state at 256 requests: 30.0 GB state + 30.1 GB draft intermediate states (1,024 slots each). This is the 0.95 failure mode described below: GPU memory can't swap, so the runtime buffers had nowhere to go.

### 3) Max context per instance at 256 instances (KV tokens ÷ 256)

| Mode | KV tokens at N=256 | **Max context each** |
|---|---:|---:|
| no spec | 5,350,151 | **~20,899** (estimate was ~20.8K) |
| MTP 3 steps (1 GB workspace) | 1,638,752 | **~6,401**, but crashed the box under load |
| DFlash block 8 | — | **0** (can't boot at 256) |
| DFlash block 3 | — | not determined (boot failed; the 1 GB-workspace retry was not run) |

**Conclusion:** at 256 instances only no spec is viable. It runs 256 × 4K at about 1,215 tok/s combined, and allows up to ~20.9K context each.

## Takeaways

1. **The image matters most.** Use the nightly `708f51e` digest; the older `qwen38-27b` image gives garbage output with this checkpoint.
2. **DFlash block 8 is the best config.**
   - c=1: 1.8× no-spec
   - c=2: 1.6×
   - c=10: 1.38× combined, 1.7× per request
3. **Gains depend on the workload.** Reasoning and code get about 2.3-2.4×; open-ended chat and essays only about 1.15-1.3×. At c=10 the slowest prose requests end up about 13% slower than no-spec.
4. **Block size:**
   - Below 8 loses everywhere; block 4 is slower than no-spec on essays.
   - Above 8 gets slightly more accepted per step (up to 4.27), but the bigger check step costs more, and c=2 drops to 123-139.
5. **fa4 draft attention** works on SM121 but is within noise of flashinfer, and it doubles the draft's KV to 24 KB/token. Not used.
6. **MTP** peaks at 3 steps (1.3× at c=1) and falls behind DFlash on reasoning and code. More steps make it slower. MTP still matters for YaRN contexts above 262K, which DFlash refuses (the YaRN override breaks the draft's config).
7. **DFlash's draft KV is the capacity cost:** 22 KB/token vs 10 KB. At equal memory, DFlash fits about 2.25× fewer full contexts.
8. **At the default 0.5 with N=2,** the KV cap (524,304 tokens, about 11 GiB including draft KV) uses far less than the fraction allows, leaving about 69 GB free. The fraction only acts as a ceiling.
9. **Untuned kernels.** Untuned Triton MoE configs are used for the FP8 experts (`E=256,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8`). Tuning them may add speed.
