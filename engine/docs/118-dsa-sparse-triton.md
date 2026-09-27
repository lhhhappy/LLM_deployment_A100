# 118 — DSA sparse attention through a Triton kernel on A100 (candidate)

Default off. Two mutually exclusive modes use the existing numerical kernel in
`srt/layers/attention/dsa/sparse_attention_triton.py`:

- `SGLANG_AX_DSA_SPARSE_TRITON=1`: all TileLang sparse-attention calls; still refuses DCP.
- `SGLANG_AX_DSA_SPARSE_TRITON_PREFILL=1`: only ordinary `ForwardMode.EXTEND` calls with full KV and no LSE.
  DCP is allowed in this mode because 116 has already gathered and remapped KV before the call. Local extend,
  target verify, draft-extend-v2, mixed mode and decode retain their existing kernels. Ordinary draft-model
  EXTEND, if used, has the same full-KV contract and is included.

The prefill-only integration is a **candidate, not GPU-validated as of 2026-09-27**. The historical operator
measurements below are not a new DCP or TP8 result. See the [integration report](../../notes/reports/prefill-sm80-0927.md).

## Prefill-only integration

`forward_extend` supplies `is_prefill=True` only at its ordinary full-KV call site. The earlier partial-attention
branch returns before reaching it. `_forward_tilelang` also requires `return_lse=False`; neither a request length
nor the absence of an LSE alone is used to infer the phase. This keeps DCP local-extend output/merge and all
speculative partial calls intact. The backend warms the same local-head variants as ordinary prefill, not the
DCP-gathered head count used by partial attention. Both switches unset preserves the old tensor operations.

The mechanism token is `118=on:prefill` only after a suitable prefill backend passes the guards and warmup.
The first actual use per backend additionally logs `[ax] 118 route=full_kv_prefill tokens=... heads=... width=...
dcp=...`. It is an entry receipt, not a replay/batch counter. Unsupported dtype, RoPE tail, HiSparse and
deterministic inference keep their startup guards. Both switches together fail startup.

No numerical kernel, persistent tensor, DCP collective, index mapping or scheduler policy changes in this mode.
At 8192 query rows it removes the 8192 × 2112 × 4-byte padding copy (66 MiB); the attention output is unchanged
in size. Short ordinary extends can use the existing split buffers, up to about 5.1 MiB at the measured A100/H8
occupancy. Actual process peak, graph pool and KV capacity still require measurement.

Local dispatch contracts: `python3 -B -m unittest discover -s scripts/tests -p test_dsa_triton_prefill.py`.
The GPU suite now includes prefill-only startup/dispatch and fresh-process warmup coverage. The new
[`bench_prefill_sparse.py`](../../scripts/analysis/bench_prefill_sparse.py) uses contiguous full KV, 8k–262k
contexts, 333–16384 query rows and a four-request batch; it checks an fp32 reference and records paired eager
time, peak temporary allocation and late Triton loads. It does not execute DCP collectives.

## Why
GLM-5.3-Flash runs absorbed-MLA sparse attention in its 11 DSA layers, plus one in the MTP draft layer. On A100 the
TileLang kernel `sparse_attention_fwd_kernel_v1` does this work (`tail_dim = 0`, because `qk_rope_head_dim = 0`).
At TP8 each rank holds 8 of the 64 heads, a 512-wide latent, and 2051 index columns (2048 top-k plus 3 kpool tail
tokens, padded to 2112).

- **Pod profile (measured).** One cold 49,143-token prefill in 8192-token chunks was profiled on TP8 (rank 0). The
  TileLang kernel (`main_kernel`) took 718.7 ms of about 3.93 s of kernel time, or 18%. That is 6 chunks × 12
  calls × about 10 ms ([L081p](../../evidence/L081p-tp8_prefill_profile_8k_16k/tp8_8k/ledger-rank0-rank1.json);
  the ledger's `dsa_attn` class misses this kernel name).
- **Dev box, phase 1 (measured).** One call on 8192 rows takes 9.96 ms. No other TileLang launch configuration
  went below 9.7 ms: block_I 16/32/64, 1–4 stages, 128/256 threads, with or without the O stage.
- **Why TileLang is slower (inferred from the generated code).** It runs one 256-thread block per SM
  (`__launch_bounds__(256, 1)`). Its allocations fill most of an SM's shared memory: two 64-row KV stages of
  64 KiB each, plus Q, O and S. The compiled Triton kernel uses 49 KiB of shared memory and 128 registers per
  128-thread program, so three programs fit on each SM. That puts more independent row gathers in flight.

## What it does
- **Kernel.** One program handles one query row, one block of up to 16 heads, and one split.
  - Heads are zero-padded to 16, the MMA M tile.
  - KV rows are gathered 32 per tile (4 warps) with masked `cp.async` into one shared-memory buffer. This was
    read from the compiled TTGIR. With `num_stages=3`, Triton prefetches the index tiles up to two tiles ahead
    and issues the next KV gather after the current tile's two MMAs. It then waits for all copies at the top of
    each iteration, so a program hides little of its own gather; the other programs resident on the SM hide it.
  - Measured at 8192 rows with uniform random indices (eager): `num_stages` 1 gives 180 registers, 80 KiB and
    15.1 ms; 2 gives 128 registers, 49 KiB and 9.87 ms; 3 gives 128 registers, 49 KiB and 9.69 ms.
  - Tiles after a row's last valid index are skipped.
  - The index table is read at its real width (2051), so the -1 padding copy of `_forward_tilelang` is gone.
- **Split for small batches.** A small grid would leave SMs idle, so each row's tiles are split over the largest
  power of two (≤16) that still fits the grid in one wave. A second kernel merges the partial softmax states.
  - The warmup measures the wave from the compiled kernel's registers, shared memory and warps against the
    device's per-SM limits: 3 programs × 108 SMs = 324 on A100, limited by shared memory.
  - The split count depends only on the row count and the device, so it is fixed for a captured CUDA graph.
  - It is not batch-invariant, unlike TileLang. A row's summation order, and so its last bits, depends on how
    many rows share the call. Deterministic-inference mode is therefore refused.
  - The partial buffers are M × splits × H × 512 fp32. Splitting only happens while M × splits ≤ 324, so this is
    at most 5.1 MiB for 8 heads. It is allocated per call from the caching allocator (the graph pool under
    capture). There is no persistent buffer. At prefill the path also drops the served -1-padded copy of the
    index table (8192 × 2112 × 4 B = 66 MiB per call, computed).
- **Dispatch.** The switch sits at the top of `DeepseekSparseAttnBackend._forward_tilelang`, the only served caller
  of `tilelang_sparse_fwd`. That covers:
  - prefill extend;
  - target verify and draft extend (`forward_extend` with the decode backend);
  - decode and draft decode.

  With the switch off, the base code runs unchanged; the only addition is the flag test. The LSE output is kept
  for the drop-in contract, but serving never requests it because DCP is refused.
- **Startup (`_ax118_init`).** It acts only when the switch is on and a DSA backend is `tilelang`.
  - It refuses to start with a `ValueError` that names the reason for each of these cases:
    - not CUDA;
    - below sm80;
    - a non-bf16 KV cache;
    - `qk_rope_head_dim != 0`;
    - DCP in all-phase mode (partial-attention heads/LSE merge are not validated with 118);
    - HiSparse (not validated);
    - deterministic inference.
  - It then compiles and loads every kernel specialization serving can launch: 5 split counts and 4 combine
    kernels. Nothing is then compiled or loaded during graph capture or while serving (compare
    `utils/triton_load_watch.py`).
  - Pool size, lengths and the index row stride are runtime integers (`do_not_specialize`).
  - Direct calls without a warmup warm up on first use; inside graph capture that fails an assertion.
- **Report.** The `[ax] mechanisms:` line shows one of:
  - `118=on`;
  - `118=on:prefill`;
  - `118=off:SGLANG_AX_DSA_SPARSE_TRITON_unset`;
  - `118=off:no_tilelang_dsa_backend` (switch on, but no DSA backend in the process uses TileLang);
  - `118=off:no_tilelang_dsa_prefill` (prefill-only switch on, but no TileLang prefill backend engaged);
  - `118=off:no_dsa_backend` (the process never imported the DSA backend; the report never imports it).

## Semantics compared with TileLang
- **Output and LSE.** The output is bf16 [1, M, H, 512]. The LSE is fp32 [1, M, H] in the same log2 units. The
  softmax weights are rounded to bf16 before the PV product, as in TileLang.
- **Fully masked rows.** A row with no valid index returns 0 (TileLang: 0/0 = NaN); the LSE is -inf in both. No
  served row is affected, for three reasons:
  - Every real query row contains its own token (tail or complete group), so it always has a valid index.
  - A fully masked row is a padding row from `_pad_topk_indices`, whose output is discarded.
  - Nothing in the served path relies on the NaN. The only `nan_to_num` calls are on the DCP paths, which 118
    refuses.
- **Out-of-range indices.** An index ≥ pool size is masked like -1. The function docstring says so, and the
  kernel does not check for it. TileLang zero-fills that KV row and still counts it with logit 0: its generated
  gather is `cp_async_gs_conditional` on `0 <= idx < seq_len_kv`, and its mask is `idx >= 0`. Pool locations are
  always below the pool size, so this only closes an out-of-bounds path.
- **NaN inputs.** They propagate instead of being hidden: a NaN in q gives NaN in that row and head (tested).

## Evidence (dev box, measured)
Setup: A100-SXM4-80GB, GPU 0, torch 2.13, Triton 3.7.1. One card, random bf16 q and KV. Index rows are laid out
like the kpool transform: 64-token pages scattered over a 1M-row pool, and past 512 groups, 512 random groups in
random order. Receipts are in [evidence/dsa118](../../evidence/dsa118/).

[tests/gpu/test_dsa_sparse_118.py](../../tests/gpu/test_dsa_sparse_118.py): 11 tests (14 numeric cases), all pass.
- **Numerics.** Checked against fp32 on every row with a valid index.
  - The bound is |error| ≤ 2^-8 × (|ref| + the row's max |KV|). The served kernel must meet the same bound.
  - The worst ratio to the bound is 0.37 for Triton and 0.37 for served (the first-chunk case).
  - The mean absolute error of Triton is at or below the served kernel's in all 10 comparable cases (1024
    prefill rows: 8.165e-5 against 8.171e-5).
  - Triton and served differ by at most 0.0039 per element in the first-chunk case, 0.0078 in the peaky case
    (outputs up to ±3), and 0.002 elsewhere.
  - LSE: both kernels pass |error| ≤ 1e-4 + 1e-6 × |ref|. The largest error is 2.9e-6 for both, except the
    peaky case, where it is 4.9e-4 for both at LSE ≈ 2e3 (two fp32 ulps).
- **Numeric cases.**
  - Prefill: 1024 rows at 48k context; a 1000-row first chunk; rows crossing 512 groups; M = 333.
  - Decode shapes: verify 32×4, decode 32×1, M = 1.
  - Peaky logits: q × 30, and for head 0 a key aligned with q at each row's last valid index. That head's
    maximum therefore lands in the last non-empty split (8 rows, 16 splits; some splits are empty).
  - Masking: -1 holes, including whole -1 tiles of 32 and 64 columns; fully masked rows (Triton gives 0 and
    LSE -inf, served gives NaN and LSE -inf).
  - Pool boundary: index P−1 is read; indices P and P+7 give output bitwise equal to -1.
  - Width 2048 (kpool 1); 64 heads with and without split.
- **Dispatch.** With the switch off, `_forward_tilelang` equals the served call bitwise, NaN rows included. With the
  switch on, it equals the 118 kernel bitwise. The LSE has shape [M, H] on both paths.
- **CUDA graph.** Captured at M = 1, 32, 128 (16, 8 and 2 splits) and 512 (no split). Each graph was replayed 3
  times with new q, KV and indices. Every replay equals eager bitwise and stays within the fp32 bound.
- **Warmup.** A fresh process warms up through the backend's own `_ax118_init`. It uses the GLM TP8 config
  (64/8 heads, top-k 2048) with kpool 4 and with kpool 1.
  - That loads 14 kernels: 5 per index width, plus 4 combine kernels that do not depend on the width.
  - After that, no further kernel is loaded. The calls use the indexer's output width across 21 row counts,
    packed and column-sliced tables, and varying pool size.
- **Startup.** With the switch on, the backend engages whenever either impl is TileLang, and never when both
  are other impls or the switch is off. Each of the 7 unsupported cases raises a `ValueError` naming its
  reason.
- **Negative controls.** Reading -1 as row 0 lands 294× over the bound; reading the neighbouring row, 7.1×.
- **Registers and memory.** 128 registers, 0 spills and 50,432 B of shared memory in every specialization.
- **Startup cost.** The warmup takes 7.6 s per process with an empty Triton cache (9 kernels) and 0.6 s from the
  disk cache.

Timing. Per-call GPU time inside a CUDA graph, with L2 flushed before each call; median of 5 interleaved rounds.
Served means `_forward_tilelang` with the switch off, including its padding copy.

| shape | rows M | splits | served ms | 118 ms | speedup |
|---|---:|---:|---:|---:|---:|
| prefill, 8192 rows at 49k context | 8192 | 1 | 10.59 | 6.59 | 1.61× |
| prefill, first 8192-token chunk | 8192 | 1 | 9.94 | 5.15 | 1.93× |
| prefill, 333 rows at 30k context | 333 | 1 | 0.542 | 0.396 | 1.37× |
| target verify 32 × 4 | 128 | 2 | 0.297 | 0.183 | 1.62× |
| target verify 8 × 4 | 32 | 8 | 0.141 | 0.066 | 2.15× |
| target verify 1 × 4 | 4 | 16 | 0.133 | 0.025 | 5.34× |
| decode 32 | 32 | 8 | 0.142 | 0.068 | 2.10× |
| decode 8 | 8 | 16 | 0.133 | 0.029 | 4.55× |
| decode 1 | 1 | 16 | 0.131 | 0.020 | 6.57× |

- **Against the phase-1 prototype.** Outputs are bitwise equal at all 9 shapes, and time is within 1% (6.592
  against 6.590 ms at prefill). A runtime head-block offset first cost 7%; it is now compiled in only for more
  than 16 heads.
- **Split rule.** Every split count was swept for 1–500 rows. The one-wave rule picked the fastest split for 18 of
  25 row counts and was within 4% elsewhere, with two exceptions:
  - 1–2 rows: 7.6–8.5% slower; 32 splits would be faster by about 0.002 ms.
  - 333 rows: 13% slower; above one wave the rule does not split.

  The measured switch from 2 splits to 1 falls between 162 and 170 rows, which is 324 = 3 × 108 slots.

## Not validated
- **No 8-card run.** Nothing has been measured with real weights or real top-k locality, and TP8 timing inside a
  full prefill or decode step is unmeasured. The MTP draft layer and the breakable prefill graph (170) are
  untested. DCP partial attention in all-phase mode and HiSparse are refused; the kernel's 64-head path is checked
  only against fp32. The new full-KV prefill-only DCP integration has local dispatch tests, but no new GPU result.
- **Single-buffered gathers.** Each program still waits for its own gather. True KV double buffering (2 × 32 KiB
  per program) is a possible next step and has not been tried.
- **Timing locality.** Timing uses randomly selected groups, which have poor locality. Phase 1's correlated
  pattern gave 5.86 ms against 9.87 ms at prefill.
- **Savings are estimates.**
  - Prefill: about 12 calls × 4 ms ≈ 48 ms per 8192-token chunk, or about 290 of the 718.7 ms measured for the
    49k-token prefill.
  - Target verify at 32 × 4: 11 layers × 0.11 ms ≈ 1.3 ms per step.
  - Both come from single-card kernel times; whether they turn into TTFT, TPOT or N is for the pod to show.

## What the TP8 pod run must check
- **Mechanism line.** Add `118=on` to `G_EXPECT` and set `G_ENV=SGLANG_AX_DSA_SPARSE_TRITON=1`. The engine must
  start (no `ValueError`) and print `118=on`.
- **Triton version.** Every measurement here used Triton 3.7.1. Record the image's version; the kernel relies on
  `do_not_specialize`, and on the specialization rules that the warmup test checks.
- **Capability smoke.** It must pass: this is the first real-weight check of the outputs.
- **Server log.** No `triton_load_watch` late-compile or late-load warnings.
- **KV pool.** `max_total_num_tokens` must be unchanged against the same commit with the switch off.
- **Profile.** A TP8 prefill profile like L081p must show the sparse-attention kernel time drop, and the time of a
  whole 8192-token chunk.
- **Timed replay.** At N30, compare cold-prefill TTFT, TPOT and chunk throughput against official A.
