# MoE kernel alternatives at the TP8 per-rank shape, dev box (2026-09-25)

One A100-SXM4-80GB each on the GPU dev box, source 37e90023 (/sjtu/linhang/arena/runs/ep8/src), random fp8
block-quantized weights, uniform routing, TP8 rank-0 slice: E=289 (288 routed + shared fused as id 288, top-k 9),
hidden 4096, intermediate slice 256, clamp 10, routed scale 2.5. Provisional until TP8 with real weights.
Scripts: `scripts/pod/verify/bench_moe_alt.py` (Marlin, C6, Humming), `scripts/pod/verify/bench_moe_bf16.py`
(BF16 ceiling, Triton default/tuned, per-call dequant). Raw logs here: `alt0_*` (GPU 0), `alt1_*` (GPU 1).

ms per layer call (TFLOPS):

| M | Marlin (served) | Marlin, empty buffer (C6) | Humming indexed | Triton BF16 tuned* | dequant + Triton* |
|---|---|---|---|---|---|
| 128 | 0.678 | 0.665 | 0.614 | – | – |
| 2048 | 1.505 (77) | 1.425 | 1.317 (88) | 1.51 | 3.26 |
| 8192 | 5.82 (80) | 5.53 | 3.53 (131) | 3.36 (138) | 5.03 |
| 16384 | 11.40 (81) | 10.80 | 6.94 (134) | 6.47 (143) | 8.14 |

\* BF16 weights; a persistent BF16 copy would need 78 GB per rank (not feasible); per-call dequant costs 1.77 ms.
Decode sizes under CUDA graphs: Humming 0.35 / 0.49 / 0.59 ms vs Marlin 0.38–0.40 / 0.54 / 0.68 at M=32/64/128.
Accuracy vs an fp32 reference of the dequantized weights (64 tokens): Marlin 5.7–6.1e-3, Humming indexed
5.7–6.2e-3, Triton BF16 4.7e-3; negative controls (swapped or shifted block scales, shifted ids) 0.17–0.61.
Dense cuBLAS pair with the same FLOPs: 222–232 TFLOPS. Marlin's ≤64-row blocks, not the TP8 shape, limit it.
C6 (`torch.empty` for the non-EP intermediate buffer) is bitwise identical to Marlin with the alignment output
held fixed (Marlin itself varies run to run through atomic ordering in moe_align_block_size) and saves ~5%.
Humming: first call 64–77 s (repeated NVML queries in its tuning heuristics plus ~10 s JIT for 37 kernels); CUDA
graph capture works; the base's serving glue builds a runner per call (host overhead in eager prefill).
