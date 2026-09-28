# 117 follow-up: SM80 MoE down-projection tuning

Default-off execution candidate. There is no TP8 or chain result yet.
This changes the existing Humming GEMM's tiling and persistent CTA count;
it does not change expert selection, weights, precision or model semantics.

## Scope and fallback

Set `SGLANG_AX_SM80_MOE_DOWN_TUNE=1` before worker startup, alongside 117's
`SGLANG_AX_SM80_FP8_MOE_HUMMING=1`. The new flag is read once at import.
The override is confined to 117's runner subclass and is independent of the
[reduction follow-up](117-sm80-moe-reduce-followup.md).

The exact target is indexed MoE on SM80, BF16 input/output, 289 local
experts, top-k 9, hidden 4096 and intermediate 256. Humming's converted W2
must use FP8 e4m3 weights, BF16 scales, K scale groups of 128, and N group
size zero: 117 expands each original checkpoint scale over its 128 N rows.
Capability is checked on the weight tensor's explicit device at table setup.

Only 8192 through 16384 tokens, inclusive, are eligible. Humming's table
counts routed rows (`tokens * 9`) and uses `(lower, upper]` intervals. The
override splits those intervals at the exact bounds, without changing
coverage or mutating the native table. It only recognizes this exact native
configuration:

```text
block_shape=[128,256,64], warp_shape=[64,64,64], use_stream_k=false,
use_f16_accum=false, num_sms=108, num_stages=4,
num_ctas_per_sm=1, num_write_splits=1
```

It changes `block_shape` to `[128,128,64]`, stages to 3 and CTAs per SM to 2.
The M block height remains 128, preserving W1/W2's shared expert-block
index, and K remains 64, preserving the tested reduction order. Unknown
native configurations, extra future tuning options and all other sizes keep
the native table. The table and its serialized JSON are updated together.

Default-off returns the original table. The original 117 startup compiles
all configurations in that table, so the additional specialization is
compiled at startup too. No new model parameter, GPU workspace or persistent
tensor is allocated by the override.

## Development evidence

Humming 0.1.12, one A100-SXM4-80GB, synthetic TP8 per-rank weights and routing.
The second sweep compared 24 requested configurations while retaining M128,
K64 and FP32 accumulation. The selected configuration used the same output
values as native in the probe, and passed an independent sampled projection
reference based on the original FP8 checkpoint and FP32 scales.

| Rows | Native W2 + reduce | Tuned W2 + reduce | Native graph | Tuned graph |
| ---: | ---: | ---: | ---: | ---: |
| 8192 | 1.613776 ms | 1.447924 ms | 1.593530 ms | 1.478561 ms |
| 16384 | 3.145708 ms | 2.918040 ms | 3.118844 ms | 2.976471 ms |

Graph measurements exclude Python enqueue gaps, and show 7.2% and 4.6%
reduction in this stage's time. These are not whole-layer, prefill or chain
gains. `num_ctas_per_sm` affects both launch bounds and the persistent launch
grid; it does not prove the actual number of simultaneously resident CTAs.

Source: [sweep](../../scripts/analysis/bench_moe_down_tune_sm80.py).
Raw data: [first sweep](../../evidence/mhc-moe-sm80-0928/moe-down-tune-a.jsonl),
[second sweep](../../evidence/mhc-moe-sm80-0928/moe-down-tune-b.jsonl).
The [CPU regression](../../tests/test_fp8_humming_tuning_117.py) verifies exact
interval coverage, W1/W2 M-block agreement, unchanged fallback ranges,
immutable input, live tuple/serialized list equivalence, unknown-configuration fallback and JSON/table agreement.

The complete MoE comparison first used baseline, reduce-only, down-only and
both arms, then added up-only, up+down and all three. Its independent FP32
reference and fresh-input graph checks remain separate from Humming's
inherited ordering/stream-K variation. Final adoption requires TP8 and the
same-ID chain/TPOT comparison.

The [production integration regression](../../tests/gpu/test_moe_down_tune_117_followup.py)
completed on the developer A100. It builds separate default-off and enabled
layers, compares all five converted weight tensors by their bits, and intercepts
actual W2 calls at 4096/8191/8192/8193/16384/16385 tokens. All six same-activation
comparisons are bitwise identical; all three target sizes really select the
new configuration, and the other sizes retain native. Startup after the first
one-token warmup and subsequent prefills cannot call Humming's compiler.
Thirteen early fallback cases plus unsupported capability and unknown native
configuration checks pass, and the baseline cache stays unchanged.

The first integration run correctly failed: native Python shape tuples did
not match a list-valued fixture, silently disabling the override. Matching now
normalizes only those two shape fields, retaining the strict other values and
key set; the regression insists on actual activation of the fast path.
[Failed receipt](../../evidence/mhc-moe-sm80-0928/moe-down-production-a.jsonl),
[successful rerun](../../evidence/mhc-moe-sm80-0928/moe-down-production-b.jsonl),
[verified source hashes](../../evidence/mhc-moe-sm80-0928/moe-down-production-b-upload.json).

The final combined integration run also passed with separate off/down/up/both
layers: 48 effective configurations, 40 forwarded dispatch configurations,
18 same-activation W2 bit comparisons and 15 converted-tensor bit comparisons.
All four cores forbade Humming compilation after the first one-token warmup.
The two metadata gates are independent, the baseline table remains unchanged,
and unsupported boundaries/capability/configurations retain native behavior.
[Final receipt](../../evidence/mhc-moe-sm80-0928/moe-up-down-production-c.jsonl).

Whole-layer graph results for down-only were positive in the B/C uniform and
skewed cases, approximately 2.9–4.3% paired time reduction. The recommended
up+down candidate subsequently measured 8.1% at 8192 and 6.1% at 16384 in both
D/E cases. Each arm's transient allocated peak was identical: 640.427 and
1280.711 MiB. These are synthetic single-rank MoE results, excluding TP
communication and the rest of the model; see the
[final report and raw links](../../notes/reports/moe-mhc-sm80-0928.md).
