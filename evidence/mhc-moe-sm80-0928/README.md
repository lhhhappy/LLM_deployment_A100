# SM80 MoE / mHC developer evidence, 2026-09-28

Status: completed single-A100 diagnostics. **No TP8, chain, or official score.**
Interpretation and final candidate: [report](../../notes/reports/moe-mhc-sm80-0928.md).
Independent source/raw review: [review](../../notes/reports/moe-mhc-sm80-0928-review.md).

## Evidence map

| Run prefix | Status and scope |
| --- | --- |
| `moe-reduce-production-a` | PASS: production reduction numeric/dispatch/graph/streams/warmup |
| `moe-reduce-production-timing-a` | PASS: isolated production reduction, cold/eager and graph |
| `moe-down-production-a` | FAIL retained: shape tuple/list mismatch silently disabled tuning |
| `moe-down-production-b` | PASS after fix: actual W2 dispatch, bits, fallbacks, no serving compilation |
| `moe-up-down-production-c` | PASS: final off/down/up/both production integration |
| `moe-four-arm-a` | INVALID for performance: down no-op and native graph repeat assertion failure |
| `moe-four-arm-b` | PASS: active W2, uniform routing, two seeds and two shapes |
| `moe-four-arm-c` | PASS: skew routing, 8191/8192/16384/16385; outside-range no-op controls |
| `moe-seven-arm-d` | PASS: final seven arms, uniform, 8192/16384, input scale 1 |
| `moe-seven-arm-e` | PASS: final seven arms, skew, 8192/16384, input scale 4 |
| `moe-sweep/stream/vector-*` | Closed isolated reduction exploration; only bounded selected kernel adopted |
| `moe-layer-a/b` | FAIL retained: incorrect full-8k bit-identity expectation |
| `moe-layer-c` | Closed diagnostic: same-intermediate bits, whole-layer timing and slower internal tiling |
| `moe-down-tune-a/b`, `moe-up-tune-a` | Closed GEMM configuration sweeps; not full-model timing |
| `mhc-*` | Closed negative fusion/prenorm explorations; no production integration |
| `moe-epilogue-*` except sanity-b | Historical negative prototypes, preceding memory-order/error-word fixes |
| `moe-epilogue-sanity-b` | PASS at M256 with hardened research code, still slower; not deployable ABI |

PASS here means the named diagnostic completed its assertions. It is not
`VALID PASS` under the service scoring contract. Failed outputs are retained
and are never averaged into successful timing claims.

## Identity and integrity

Each closed probe has raw `.jsonl`, `.stderr`, `.exit`, `.started` and
`.finished` receipts where archived. Production upload/source receipts bind
the measured implementation to its SHA256. Four-/seven-arm runs additionally
have source snapshots, `.manifest.json` (per-file SHA256 and bytes), a
`.summary.json` derived from raw records, and launch metadata. The summary
does not replace the raw. `closed-probes-{a,b,c,d}.sha256.json` record archive
integrity for the earlier probe batches.

The final D/E production source matches engine commit `eaf41653`; SHA256 values are:

```text
moe_fused_mul_sum.py    82766f32d74390c4195a41f0b218a33e60568bd8e3c277bd5c6cee39ff6e84ab
fp8_humming_moe.py     155e916fafd86a96bebb18e12c79add42a2493231c7e7cf341726d574c83333e
fp8_humming_tuning.py  ef1d15fb6c9dd34f96a2d379071f2d48ed622c23100f7ba65283068571a4b008
```

Humming is the Pod-matched 0.1.12 sdist, not the developer vLLM environment's
0.1.16. Early negative epilogue runs do not all have complete historical
source snapshots: current probe scripts include later hardening. Do not
attribute their old large-shape timings to the corrected code. Only the
private copied include directory was modified, never the shared package.

## Reproduction

The isolated developer root is
`/sjtu/linhang/arena/codex/mhc-moe-sm80-0928`. Its engine starts from 20a58da9
with the three candidate production files applied. The runner fixes Humming
0.1.12 and places all caches under this root. Run only after coordinating an
idle GPU; the commands below do not allocate, stop or modify a Pod.

```sh
KERNEL_ROOT=/sjtu/linhang/arena/codex/mhc-moe-sm80-0928
KERNEL_GPU=0 bash "$KERNEL_ROOT/scripts/analysis/run_kernel_devbox.sh" "$KERNEL_ROOT" \
  tests/gpu/test_moe_reduce_117_followup.py
KERNEL_GPU=0 bash "$KERNEL_ROOT/scripts/analysis/run_kernel_devbox.sh" "$KERNEL_ROOT" \
  tests/gpu/test_moe_down_tune_117_followup.py
KERNEL_GPU=0 bash "$KERNEL_ROOT/scripts/analysis/run_kernel_devbox.sh" "$KERNEL_ROOT" \
  scripts/analysis/bench_moe_candidate_sm80.py --include-up --rows 8192 16384 \
  --routes uniform --seeds 4 --rounds 9 --inner 16
KERNEL_GPU=1 bash "$KERNEL_ROOT/scripts/analysis/run_kernel_devbox.sh" "$KERNEL_ROOT" \
  scripts/analysis/bench_moe_candidate_sm80.py --include-up --rows 8192 16384 \
  --routes skew --seeds 5 --x-scale 4 --rounds 9 --inner 16
```

The integration test intentionally toggles cached flags in its own process.
Serving workers read flags once at startup. The benchmark uses production
table helpers and an actual FusedMoE runner; the separate integration test
verifies startup gating and forwarded runtime configurations.

CPU-only table regression from the repository root:

```sh
python3 tests/test_fp8_humming_tuning_117.py
```
