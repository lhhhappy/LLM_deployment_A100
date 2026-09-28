# MoE FP8 scale-fold research receipt, 2026-09-28

Chain has not been measured. This isolated A100-SXM4-80GB experiment produced a small full-MoE improvement, so the candidate is **closed without production adoption or another shape sweep**. The comparison is current 117 up+down tuning, reduce off, versus the same configuration with a private FP8 exponent-fold header. It is not a comparison against the original untuned baseline.

| CUDA graph, 9 paired rounds | Native median ms | Fold median ms | Median paired reduction |
| --- | ---: | ---: | ---: |
| W13, M8192 | 1.643984 | 1.558350 | 4.690% |
| W2, M8192 | 1.164268 | 1.092284 | 5.728% |
| Full MoE, M8192 | 3.318620 | 3.229828 | 2.846% |
| W13, M16384 | 3.254070 | 3.119138 | 3.726% |
| W2, M16384 | 2.300184 | 2.222146 | 3.044% |
| Full MoE, M16384 | 6.563152 | 6.421220 | 1.967% |

Each graph measurement uses 16 replays; arm order is randomized each round. All nine full-MoE pairs favor fold at both sizes. The paired percentage is the median of per-round percentages, not a ratio of the two displayed medians. This is one development-GPU run, without a frequency trace or a separate repeat run. The absolute times must not be compared across earlier experiments. Serial eager results are retained in raw output as secondary evidence.

## Correctness and isolation

- CPU exact dyadic BF16 RN-even: all 254 finite FP8 patterns × 17,280 sign-bit-zero BF16 scale patterns from +0 through 255, including subnormals; 4,389,120 pairs, zero differences.
- Actual SM80 BF16 `__hmul2` instructions with an intervening inline-assembly barrier: the same 4,389,120 pairs, two BF16 lanes per pair, zero differences. Scale 256 produces differences for all 254 finite FP8 patterns and is excluded.
- The research dispatch accepts zero, minimum subnormal, minimum normal, one and 255; it rejects 256, negative, Inf and NaN. A scale-256 mutation in the actual packed W13 and W2 scale tensors selects the native kernel and passes raw-bit comparison. Original scale hashes match again after restoration and at completion.
- Domain limit found by independent review: the scale guard's numerical `>= 0` comparison also accepts BF16 negative zero, which was not enumerated. FP8 NaN encodings `0x7f/0xff` were excluded from the proof, while the research guard checks only scales. The positive scales and finite weights of this fixture do not exercise these corners. This is not a complete production guard or a full-domain proof; future adoption would need tests/proof or explicit rejection of those bit patterns.
- Sixteen raw BF16-bit records pass: W13; W2 with identical activation; full eager output; fresh-input graph output; graph against fresh eager output, at both M8192/M16384, plus the two real scale-256 fallbacks. The layers use the existing 117 synthetic-checkpoint fixture, with the actual Humming conversion and execution path. These are not production-checkpoint or TP8 tests.
- Native/fold are co-resident, with identical parsed configuration fields, weights, alignment, loop ordering and output buffers. The only changed header is `humming/arith/mainloop_arith.cuh`. Scale preparation occurs at `j == 0` after each scale-register load; the registers are reloaded before reuse, so the factor does not compound across K iterations.
- Each variant has a private content-addressed header directory and a stamped compile source. Raw JSON whitespace separates the dispatch-cache keys while parsed values stay equal. Only Humming constructor memoization is evicted before each preparation; registered native kernel IDs and dispatch tables remain live. Unexpected compilation is forbidden after preparation. Final records confirm unchanged native config cache and scales.
- All four stage cubins have distinct paths, IDs and hashes. `cuobjdump` was unavailable, so no SASS instruction-count claim is made.

| Stage/variant | Cubin cache directory | SHA-256 |
| --- | --- | --- |
| W13 native | `e6309cbf448d9629` | `5f5d411f93ff25bf1cad2ef71ce639c6fa26b739500a56da609b6a22128be953` |
| W2 native | `e93030395c73cffb` | `fd2fcbbec59053af5377c02f0e791b108b139182e02355b3ed1c9c93729bf784` |
| W13 fold | `3e853c726194680e` | `6259bd9664233cfc8fdfb313f9e31b260d9f5ec19895adfa9db2eec6fc8f1f51` |
| W2 fold | `a7d1374de0ece5ca` | `52972b0a621a50eec675d6aadeccbec41c64dfcdb5c4611f1018eb926da233bf` |

The GPU intrinsic cubin is `7d80fa72a7dd5611`, SHA-256 `137c5054edc40b48b53160e1de2389fd8162a571d2da25eb98eef83b677c6dee`.

## Raw artifacts

- Successful run: [moe-scale-fold-b.jsonl](results/moe-scale-fold-b.jsonl), [exit](results/moe-scale-fold-b.exit) = 0, [stderr](results/moe-scale-fold-b.stderr) empty; 10:52:53Z–10:53:32Z.
- The failed `a` run is preserved in full. Its intrinsic test passed, then the research harness called the nonexistent `run_activation` instead of `apply_activation`; no timing from that run is used. Both exact probe versions are in `results/`.
- [Manifest](results/moe-scale-fold-0928.manifest.json): all 156 listed files were verified after extraction. The archive has 157 files including the manifest, and contains both header trees, five cubin builds with sources/signatures/commands, probe/helper sources and both run receipts.
- [receipts.tar.gz](receipts.tar.gz): 331,161 bytes; SHA-256 `21354aaca4f31ecaa1785d6597d244e78fe6799cf96180b273e3c8edd5ef8d09`. The remote archive remains `/sjtu/linhang/arena/codex/mhc-moe-sm80-0928/results/moe-scale-fold-0928-receipts.tar.gz`.
- Final probe SHA-256: `9a3af57c62830620d6aa2cff6c91903c25f6ba37aadf6b281dcee48ebf1c7faf`; helper SHA-256: `9b584f218d28cc57fd9cd776dde2b878a348176d643473bb8f5aa22693d95422`.

Reproduction command, under the already configured remote environment and after assigning a free GPU:

```bash
bash /sjtu/linhang/arena/codex/mhc-moe-sm80-0928/scripts/analysis/run_kernel_probe.sh moe-scale-fold-b 1 scripts/analysis/bench_moe_scale_fold_sm80.py --rows 8192 16384 --rounds 9 --inner 16
```

GPU1 was released after the run. No production file was changed by this experiment. The [independent read-only review](../../notes/reports/prefill-kernels-0928-review/scale-fold-review.md) recomputed the archive manifest, inspected source/command isolation and raw bits, and confirmed the limited local speed result and numerical-domain boundaries above. The broader offline disposition is recorded in [R36](../../research/codex/R36_prefill_kernel_options_0928.md).
