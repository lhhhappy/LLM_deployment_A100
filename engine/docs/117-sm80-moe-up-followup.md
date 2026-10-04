# 117 follow-up: avoid stream-K partial sums in large SM80 W13

Default-off execution candidate; TP8 and chain gains are not established.
Set `SGLANG_AX_SM80_MOE_UP_TUNE=1` before workers start, alongside 117.
It is independent of the down-projection and weighted-reduction switches.

The guard and inclusive 8192–16384-token range match the
[down-projection follow-up](117-sm80-moe-down-followup.md), checking W13's
converted BF16 input/output, FP8 weights and BF16 K128/N0 scale metadata.
The native W13 table must exactly match M128/N256/K64, warp64×64×64,
four stages, one CTA per SM, 108 SMs, one write split, FP32 accumulation and
`use_stream_k=True`. The override changes only `use_stream_k` to `False`.
Other shapes, configurations and ranges retain the native table. W1/W2's
common M-block indexing remains unchanged.

In Humming 0.1.12, stream-K partitions a dot product between CTAs. Its
`epilogue/smem_writer.cuh` converts each partial accumulator to the output
dtype, and `epilogue/gmem_writer.cuh` combines BF16 partial outputs with
BF16 addition/atomics. Disabling stream-K keeps the dot product in FP32
through K and rounds its final result to BF16. Thus this is **not a bitwise
identity transformation**; it removes intermediate rounding and changes the
summation order. It retains all weights, expert contributions, activation
clamping and routing semantics. Only the separate W2/reduction changes are
checked by raw output-bit identity.

A developer A100 sweep measured this one-field change at the actual per-rank
shape, with synthetic checkpoint weights and routing:

| Tokens | Native W13 | W13 without stream-K | Native graph | Candidate graph |
| ---: | ---: | ---: | ---: | ---: |
| 8192 | 1.781656 ms | 1.585804 ms | 1.832406 ms | 1.687278 ms |
| 16384 | 3.553244 ms | 3.208624 ms | 3.596714 ms | 3.324000 ms |

This is about 8% less time for the W13 graph, not for a full layer or
prefill. An independent sampled check uses FP64 accumulation over the
original checkpoint dequantized with its FP32 scales. Candidate errors must
stay within the reference bound; baseline self-repeat variation does not
replace this gate. Full-layer clamp/SiLU/MoE and fresh-input graph checks
use a separate FP32 reference before adoption.

The [table regression](../../tests/test_fp8_humming_tuning_117.py) checks
native tuple versus JSON list shapes, exact boundaries, unknown-family
fallback, immutable input, changing only stream-K, idempotence, and
composition of the disjoint W13/W2 overrides in either order.
The existing first-token layer warmup compiles all table variants before
serving. No weight, activation workspace or retained tensor is added.

[Sweep source](../../scripts/analysis/bench_moe_up_tune_sm80.py).
The full-layer experiment separately measures baseline, reduction, down,
reduction+down, up, up+down and all three changes; it must not credit the
sum of isolated kernel gains as a measured layer gain.

## Completed full-layer checks

The final D/E seven-arm comparison passed at 8192/16384 rows, using uniform
routing and skewed routing with input amplitude multiplied by four. It checks
finite full outputs, 16 sampled rows against the independent FP32 checkpoint
reference, and fresh inputs/routes on two graph replays per shape and arm.
The maximum graph/eager reference relative L2 was 0.0059831; every candidate
also met `error <= 1.05 * baseline_error + 1e-4`. The fixed reference gate was
0.02. Baseline repeat variation only calibrates the graph/eager comparison.
The clamp is present in both paths; the probe does not count clipped values.

Up+down reduced complete MoE graph time by paired medians of 8.16%/6.12%
(uniform 8k/16k), and 8.12%/6.15% (skewed, amplitude x4). Peak transient
allocation was unchanged. GPU clocks were observed but not locked; these
are within-run comparisons on synthetic weights, not whole-prefill or chain
claims. Reduce's additional full-layer increment remained inconclusive.

Separate production wiring checks build off/down/up/both layers, assert
actual W13 stream-K selection, preserve converted weights, check fallback
boundaries, and forbid Humming compilation after the first warmup token.
W13 repeat checks were stable in all four paths including baseline; do not
claim a demonstrated stability improvement.

[Integration receipt](../../evidence/mhc-moe-sm80-0928/moe-up-down-production-c.jsonl),
[D raw](../../evidence/mhc-moe-sm80-0928/moe-seven-arm-d.jsonl),
[E raw](../../evidence/mhc-moe-sm80-0928/moe-seven-arm-e.jsonl),
[independent review](../../notes/reports/moe-mhc-sm80-0928-review.md).
TP8 with real weights, same-ID chain and the online capability gate remain
outside this developer validation. Use up+down with reduce off for the main
service candidate; all flags are read before worker startup.
