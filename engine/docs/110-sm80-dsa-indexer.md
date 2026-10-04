# 110 — DSA indexer on A100 (sm80)

One patch for the whole sm80 indexer path (T57 folded the former 110 + 112 + 113 into this file; the source tree is
byte-identical, evidence/T57/equivalence.log).

## Why
The base computes the DSA indexer logits with DeepGEMM FP8 kernels, which exist only for sm90+. On A100 the base cannot start
(F56–F59). The indexer must also stay cheap: an early torch fallback took about 60% of prefill GPU time (F63).

## What it does
- **Dispatch** (`dsa_indexer.py`, `dsa_indexer_kpool.py`, `kpool_fp8_index.py`, `dsa_backend.py`, `sm80_deep_gemm.py`):
  on sm80 the two logits entry points (paged decode, ragged prefill) go to `sm80_indexer_kernels.py`. The packed fp8 KV
  (per 64-token page: 64×128 fp8 bytes, then 64 fp32 scales) is read as uint8 and decoded in software (`ax_soft_fp8.py`).
- **Decode kernel** (Triton): one program per (query row, 64-token page); reads context length on the GPU; bf16 MMA with fp32
  accumulation; fuses relu × head weight, the head reduction and the key scale. Graph-safe: no host reads, grid depends only
  on shapes, every output rewritten on each replay.
- **Prefill kernel** (Triton): all heads in one MMA tile; with `clean_logits=True` it skips tiles outside every query's
  [ks, ke) range. For the model shape (32 heads, ≥32 queries, ≥1024 keys) a faster path first decodes q and k to bf16 scratch
  once, then runs query-major MMA (2 queries × 128 keys, 4 key tiles per program, 32-query-tile groups sharing L2).
  Scratch for an 8192 × 190k call: 110 MiB, allocated per call.
- **Semantics kept from the reference implementation**: every head's dot product is rounded to bf16 before relu/weights; a
  negative block-table entry reads page 0; `clean=True` writes `-inf` outside the range; `clean=False` computes full width.
- Kernels take runtime lengths and strides, so new prompt lengths do not trigger recompiles (50 random shapes, 0 JIT, F68).

## Switches
`SGLANG_AX_SM80_INDEXER` (default 1; 0 restores the DeepGEMM path, which does not run on sm80).
2026-09-26 review repair: with an explicit 0, `_install_on_module` leaves the original module untouched,
including absent attributes. Previously a missing base entry point could still call our kernel through the
`_orig is None` fallback. Switches are startup configuration, not a runtime hot-swap API.

`SGLANG_AX_SM80_INDEXER_CHUNK_BYTES` is a legacy parsed value, **not an effective logits memory limit** in the
current 112/113 kernels: they allocate the complete fp32 output. Do not use it to budget peak memory.
See the [S1/S2 review](../../notes/reports/review-s1s2-patches-0926.md) for the memory formula and optimization path.

## Evidence
- A100 kernel tests (dev box): 88 numeric cases against the torch oracle, fp8 exhaustive decode, CUDA-graph replay bit-identical
  to eager (evidence/T43); prefill fast path 222 numeric cases, 30 graph cases (evidence/T44); no-recompile checks (evidence/T47).
- Speed (single-card kernels): decode 4.5–4.75× the torch reference; prefill 6.2–6.7× (131–186 effective TFLOPS, F65).
- 8 cards: starts and serves (F59); kernel tests re-run on the pod pass; indexer is 4.3% of cold-prefill GPU time (F76).
