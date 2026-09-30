# 175: SM80 Humming small-M down projection

The optional `SGLANG_AX_SM80_MOE_DECODE_DOWN_TUNE=1` changes one measured
Humming 0.1.12 W2 schedule. It defaults off. The existing 117 guard requires
indexed MoE, SM80, hidden size 4096, per-rank intermediate size 256, 289
experts, top-k nine, BF16 activations/output and the original block-FP8 scale
contract. Unknown configurations fall back unchanged.

Only the exact native interval `64 < routed_rows <= 4160` and complete native
configuration match. The K tile changes 128 to 64 and CTAs per SM one to two.
M16, N256, four pipeline stages, FP32 accumulation, stream-K off and output
layout stay fixed. This interval includes batches 8 through 462 for top-k
nine, including short prefill; the flag does not restrict it to decode.
The cached source tables are never mutated; the new table and JSON string
are published together. Existing 8k–16k prefill schedules are unaffected.

Compiled cubin and CUDA driver resource queries show native W2 uses 256
threads, 81 registers/thread and 152576 bytes dynamic shared memory, allowing
one resident block per SM. The candidate uses 128 threads, 155 registers/thread
and 78848 bytes shared memory, allowing two. Both have zero local-memory spill
bytes. Maximum resident warps remains eight: this increases independent block
capacity, not warp occupancy. These are static resource limits, not measured
achieved occupancy; see `kernel-resource-review.json` in the evidence below.

The development probe uses an A100-SXM4-80GB, Humming 0.1.12, real TP8
geometry confirmed against the Pod model configuration, and synthetic FP8
weights. It includes FP32 router, sigmoid/top-k/shared expert routing,
alignment, both GEMMs, clamped activation, weighted reduction and CUDA graph.
Interleaved normal-route full-layer medians in microseconds were:

| Batch | Native | Candidate |
| --- | ---: | ---: |
| 24 | 300.7 | 287.7 |
| 28 | 338.3 | 323.1 |
| 32 | 364.8 | 349.7 |
| 40 | 394.1 | 378.0 |
| 48 | 439.4 | 420.6 |

The measured full-layer reduction is about 4–4.5%, with positive results for
skewed and clamp-stressing inputs. B8/12/16/64/128/256/462 also had positive
full-layer results; B1/4 and B464 keep the original schedules. Comparisons
use independent FP32 references and changed-input graph replays. Changing
the K tile can change floating-point accumulation order: output is not
claimed bitwise identical. The weak up-projection candidate was rejected.

Probe: `scripts/analysis/bench_moe_decode_0930.py`; evidence:
`evidence/execution-0930/moe/{full-candidates,bounds}.json`. These are single
GPU execution measurements, not proof of TP8 service, chain, quality-gate or
ranking improvement. Before selection, require real-weight TP8 smoke,
confirmed engagement, memory accounting and same-N mixed-load comparison.
