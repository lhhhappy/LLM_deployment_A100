# 117 — FP8 MoE experts through Humming on A100 (sm80)

Candidate, default off. The initial numerical measurements below use one A100, a TP=1 layer with the TP8 per-rank shape.
Numbered 117 because 115 is decode context parallel (commits 0cb0eab6, 8a44a24d) and 116 is used by its code markers.

## Why
A100 has no FP8 tensor cores, so GLM-5.3-Flash's block-FP8 experts run weight-only: FP8 e4m3 weights are decoded to BF16
inside the GEMM. 111 does this with the Marlin MoE kernel, which uses at most 64 rows per expert block and reaches about
80 TFLOPS at prefill sizes ([R25](../../research/claude/R25_moe_sm80_path.md) §2–3). Humming (humming-kernels 0.1.12, in the
serving image and already wired into the base) picks 96/128-row blocks. Kernel-level dev-box numbers are in
[evidence/moe-kernels-devbox-20260925](../../evidence/moe-kernels-devbox-20260925/README.md); this mechanism makes that path
servable.

## What it does
- **Switch** `SGLANG_AX_SM80_FP8_MOE_HUMMING=1` (default 0), read when each `Fp8MoEMethod` is built, like 111's switch.
  Eligibility is 111's (CUDA sm80–88, block-quantized FP8 experts, not FP4/MXFP8). When on it takes precedence over 111
  for MoE layers; dense FP8 layers keep their Marlin path. Off: `ax_sm80_marlin` is the same boolean as before and
  `Fp8Config.get_quant_method` returns `Fp8MoEMethod` as before.
- **`quantization/fp8.py`**: `Fp8MoEMethod.__init__` computes the shared eligibility once and sets `ax_sm80_humming`;
  `Fp8Config.get_quant_method` returns `Fp8HummingMoEMethod(fp8_method)` for such layers, next to the existing FP4
  Humming/Marlin wrappers.
- **`quantization/fp8_humming_moe.py`** (new, `Fp8HummingMoEMethod`):
  - `create_weights` delegates to `Fp8MoEMethod`, so parameters, loader attributes and `FusedMoE.weight_loader` (including
    the fused shared expert in slot 288) are exactly today's.
  - `create_moe_runner` refuses what is not tested, with one error naming each reason: `--moe-runner-backend` other than
    auto/humming, an all-to-all backend other than none, expert parallelism, expert biases, non-SwiGLU or GPT-OSS-style
    activations, `apply_router_weight_on_input`, `no_combine`, and `SGLANG_HUMMING_INPUT_QUANT_CONFIG` (activation
    quantization). It then builds one `HummingRunnerCore` per layer. The base `("none", "humming")` fused func builds a new
    core on every call; measured host cost 0.68 ms vs 0.52 ms per call at M=128.
  - `process_weights_after_loading` converts with the base `prepare_humming_moe_layer` (FP8 weights repacked; 128×128 fp32
    block scales become per-column group-128 BF16 scales, as Marlin does). It then builds the tuning table and JIT-compiles
    every kernel the table names with one single-token forward, so no compilation is left for the first request. Humming's
    heuristics initialise and shut down NVML twice per candidate batch size. Holding one NVML handle open around the table
    cuts it from 71 s to 0.7 s with an identical table (`nvml_probe`).
  - `apply` feeds the standard dispatch output to the core (indexed GEMM unless `SGLANG_HUMMING_MOE_GEMM_TYPE` says
    otherwise) and returns a `StandardCombineInput`.
- **Scheduler**: the switch is in the `[ax] mechanisms` "requested:" list (token-wise `G_EXPECT` matching is unaffected).

## Semantics compared with 111 (172 off)
- Activation: both clamp gate to ≤ 10 and up to [−10, 10], round SiLU to BF16, then multiply by up (Marlin:
  `swiglu_limit_func`; Humming: `act_and_mul_triton` with `swiglu_limit`). 172 only changes the Marlin path, so it has no
  effect while 117 is on.
- `routed_scaling_factor` is applied once, in the final top-k sum (Marlin `moe_sum_reduce`, Humming `moe_fused_mul_sum`).
  Marlin multiplies the top-k weights in the down-GEMM epilogue, Humming in the sum; the math is the same, the bits differ.
- The shared expert is fused by the model as expert 288 (E=289, top-k 9); nothing here is shared-expert specific.
- Both round the fp32 block scales to BF16.
- Expert id −1: Marlin writes zero rows for them; Humming under TP reads unwritten workspace rows. SGLang's TP forward never
  produces −1 ids (`num_token_non_padded` reaches top-k only on the DeepEP and two-batch-overlap paths); EP is refused.
- Neither path is bitwise repeatable: `moe_align_block_size` orders tokens within an expert with atomics, and Humming's
  stream-K adds partial sums in BF16 in global memory (plain read-modify-write for 2–3 slices, BF16 atomics beyond),
  where Marlin reduces in fp32. Humming therefore varies more run to run (1237 of 2048 rows differ between two identical
  calls vs 81 for Marlin; relative L2 ≤ 4.0e-3 vs ≤ 1.3e-4); with the alignment held fixed it was bitwise repeatable at
  M = 2048 only (not where more than 3 slices meet). Accuracy against fp32 stays within 1.25× Marlin's.
- The runner sorts tokens with the w13 table's block height and the w2 kernel reads the expert-block ids with its own
  table's height; they agree in 0.1.12 on sm80, and the layer refuses to load if they ever differ (`_check_block_heights`).
- `process_weights_after_loading` converts in place and is not idempotent (as for 111): a second call or a weight reload
  is unsupported.

## Evidence (dev box, measured)
One A100-SXM4-80GB (GPU 1), torch 2.13, humming-kernels 0.1.12 PyPI sdist on `PYTHONPATH`. Test:
[tests/gpu/test_fp8_moe_humming_117.py](../../tests/gpu/test_fp8_moe_humming_117.py) builds real `FusedMoE` layers through
`Fp8Config`, loads a random block-FP8 checkpoint with `FusedMoE.weight_loader`, calls `process_weights_after_loading` and
`FusedMoE.forward`; TP8 per-rank shape (288+1 experts, top-k 8+1, hidden 4096, intermediate 256), clamp 10, scale 2.5,
uniform random routing. Logs: [evidence/moe-humming-117-devbox-20260925](../../evidence/moe-humming-117-devbox-20260925/)
(`test_117_run.log`: 57 checks, 0 failed; `test_117_run_coldcache.log`: the first run on an empty Humming cache).

- **Accuracy**, relative L2 against an fp32 MoE on the dequantized checkpoint, M = 1, 32, 128, 2048, 8192 (and 128/2048
  with 4× inputs to drive the clamp): Marlin 5.74–6.19e-3, Humming 5.89–6.28e-3, Humming vs Marlin
  4.1–6.0e-3. Negative controls against Humming's output: shared expert dropped 0.49–0.81, routed scaling omitted 1.50,
  clamp removed 0.76, a layer whose block scales come from the neighbouring expert 0.17–0.20.
- **CUDA graphs**: capture and replay with fresh inputs at M = 1, 8, 32, 64, 128 on both paths; replay error equals eager
  error. Replay time (ms): 0.053→0.043, 0.159→0.145, 0.403→0.375, 0.541→0.493, 0.649→0.590 (Marlin→Humming).
- **Layer-path time**, eager, host overhead included (wall ms per call, median of ABAB):

  | M | Marlin | Humming | speedup |
  |---:|---:|---:|---:|
  | 128 | 0.689 | 0.619 | 1.11 |
  | 2048 | 1.526 | 1.315 | 1.16 |
  | 8192 | 5.820 | 3.496 | 1.66 |
  | 16384 | 11.414 | 6.944 | 1.64 |

  Host enqueue time per call: Marlin 0.48–0.51 ms, Humming 0.57–0.59 ms.
- **Memory per layer**: weights 923,320,320 B (Marlin) vs 923,324,416 B (+4 KiB stream-K locks). Load-time transient
  (one layer's repack, freed afterwards) 1159 vs 1192 MiB. Per-call transient at M = 128/2048/8192/16384:
  16/189/756/1513 vs 10/160/640/1281 MiB.
- **Startup**: first Humming layer's post-load step 10.0 s on an empty `HUMMING_CACHE_DIR` (launcher build and kernel JIT),
  1.9 s with the cache filled; every later layer 0.02–0.03 s. The load-time warm-up then runs forwards at 1, 129, 144,
  1025 and 1040 tokens, which compile the Triton top-k sum (`moe_fused_mul_sum`) for every BLOCK_M size range above the
  decode graph sizes, with and without the divisible-by-16 specialization, so the first long prefill compiles nothing
  (measured before this warm-up: first forward at M=4096 0.29 s cold / 0.06 s warm).
- **Switch off**: same class, flags and Marlin runner as `fp8.py` at 37e90023; bitwise-equal weights after loading;
  bitwise-equal outputs at M = 1…8192 with the alignment held fixed.

## Not covered on the dev box; the TP8 pod run must check
- The image: `import humming` works with the installed 0.1.12 wheel (NVRTC, launcher, `nvidia-ml-py`); `HUMMING_CACHE_DIR`
  and `HUMMING_TMP_DIR` point to a verified mount (Pod root-disk quota); startup time delta and the rank-0 log line
  "Humming MoE tuning table and kernel JIT took …".
- The mechanism line shows the effective state, `117=on:<n>_layers` (43 expected: 42 MoE layers + the MTP layer), and
  `G_EXPECT` includes `117=on`; KV token count and state slots at startup equal 111's; no compilation after the first
  long prefills.
- Real weights and TP8 communication: per-layer comparison with 111 (L054 numtrace), the 12-question smoke, and MTP
  acceptance (the draft layer is FP8 MoE and also takes this path).
- Decode CUDA-graph capture for every batch size, including MTP verify shapes, and no Humming compilation after startup.
- Prefill chunk cost at 8k/16k with real routing skew (the dev box used uniform routing), then TPOT and N at the
  standard test point.

## Known limits
- The NVML hold works around humming 0.1.12 re-querying constant device properties; the real fix belongs in Humming.
- The base per-call-core fused func is left as is for other Humming users; only this method avoids it.
- Only the indexed GEMM type goes through the layer tests; `SGLANG_HUMMING_MOE_GEMM_TYPE=grouped` is base code, measured
  only at kernel level (slower, [evidence](../../evidence/moe-kernels-devbox-20260925/README.md)).

## 2026-09-26 review: bounded shape metadata cache

117 now instantiates `_AxFp8HummingRunnerCore`. It caches the shape/dtype dictionaries shared by
`_workspace_shapes` and `prepare_buffers`, keyed by both input shapes, GEMM mode, expert count, layer geometry
and activation/output dtypes. Consecutive equal keys bypass the dtype hash via a last-entry check.
The cache keeps at most 64 entries per layer, bypasses symbolic shapes, and retains
no tensors or GPU workspaces. Actual workspace allocation and numerical kernels are unchanged. All consumers
were checked for mutation of the returned dictionaries. The 117-off runner stays the base runner.

CPU and single-GPU validation, including paired timings and their limits, are recorded in the
[S1/S2 review](../../notes/reports/review-s1s2-patches-0926.md). This small host optimization must not be credited
with the much larger Marlin-to-Humming gains measured above.
