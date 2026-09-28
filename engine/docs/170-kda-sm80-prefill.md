# 170 follow-up — A100 KDA prefill execution

This follow-up removes redundant preparation work and optionally increases
recurrence parallelism inside the existing Triton KDA prefill path. It does
not require breakable CUDA graphs. All three switches default to **off**;
decode, target verification, scheduling, model weights, and precision are
unchanged.

## Switches and supported paths

| Environment variable | Work removed / changed | Guard and fallback |
| --- | --- | --- |
| `SGLANG_AX_KDA_PREFILL_CPU_LENGTH=1` | Reuse the host logical token count once per live batch instead of reading the last CUDA prefix-sum scalar in every KDA layer | EXTEND/MIXED, valid host lengths and prefix-sum shape, no TBO parent range, attention CP size 1 and no CP metadata. Otherwise read the original device scalar. Physical graph padding is never treated as logical tokens. |
| `SGLANG_AX_KDA_PREFILL_PREPARE=1` | One Triton launch replaces three strided Q/K/V copies and two Q/K L2-normalization launches | Selected Triton prefill backend; CUDA SM80; BF16 `[1,T,8,128]`, `T` exactly 8192 or 16384, token/head/feature strides `(3072,128,1)`, same device and 16-byte-aligned pointers. Other shapes/layouts use the original preparation. |
| `SGLANG_AX_KDA_PREFILL_STATE_BV16=1` | Run the existing recurrence body with V tile 16 rather than 32 | Explicit KDA caller; warmed SM80 device; BF16 K/W/U, FP32 gate and persistent state; H=8, K=V=128; 8k only N=1, 16k N<=3; contiguous inputs, native int32 query prefix sum and int64 chunk offsets, int32/int64 state indices. Snapshots must use int64 offsets/slots. Unsupported signatures retain the original single-config Autotuner. |

The state option respects nondefault `SGLANG_GDN_CHUNK_H_{BV,NUM_WARPS,NUM_STAGES}`:
it only arms when the imported native configuration is `(32,4,2)` in both
recurrence modules. It never changes a global GDN tile or adds state-mutating
autotune trials. The kernel still processes chunks in the original order.

## Numerical and memory contracts

Preparation reads the original **BF16-rounded convolution output**. Q/K use
the native FP32 sum of squares, `+1e-6`, and division by `sqrt`; V is a direct
BF16 bit copy, including NaN payloads. The three outputs have separate storage:
the attention output overwrites V and must not retain already-dead Q/K buffers.
A single shared three-plane allocation was rejected for that lifetime reason.

Recurrence changes only the V tile and launch grid. FP32 persistent state,
envelope-strided state-slot pitch, invalid-slot masks, intermediate states,
and mechanism 140 snapshots keep their original arithmetic and addressing.
The optional tile uses the existing JIT body with one explicit `(BV=16,
warps=4, stages=2)` configuration. State-slot validity and sequence bounds
remain the existing caller contract; no GPU scalar read is added by dispatch.

Both GPU options are armed during `KDAAttnBackend` initialization, after backend
selection and before graph capture. Unselected Triton fallbacks are not warmed.
Preparation executes a tiny T=16 launch; recurrence executes twelve tiny T=64
signatures (two index dtypes, two NT buckets, normal/snapshot-without-export/
snapshot-with-export). Warmup uses private tensors and retains no tensor cache.
Its CUDA-module/allocator cost is measured separately from live tensors.

Startup logs identify each armed GPU option with `[ax] KDA prefill prepare:`
and `[ax] KDA prefill state:`. A requested flag alone is not proof a particular
shape used the optimized route. The CPU-count option is independent of these
GPU options and independent of the old `--cuda-graph-backend-prefill` flag.

## Validation and limits

The focused tests are:

```bash
python3 tests/test_kda_prefill_cpu_length.py
python tests/gpu/test_kda_prepare_sm80.py
python tests/gpu/test_kda_state_sm80.py
python scripts/analysis/bench_kda_prepare_sm80.py --production --rows 8192 16384 --rounds 9 --inner 8
```

GPU commands require this engine on `PYTHONPATH`, the recorded dependency
environment, and an A100. The isolated developer runner is
`scripts/analysis/run_kernel_devbox.sh`; it operates only under the designated
developer workspace and does not deploy a service.

Tests compare actual wrappers and complete KDA-core outputs, convolution/SSM
state envelopes, snapshots, changed inputs/slots, and graph replay. The core
probe compares native, preparation only, recurrence only, and both on a shared
CPU-length baseline. It reports eager event span, host enqueue time, captured
core time, output storage size, and transient allocated peak separately.

The experiment history, final measurements, source identities and review are
in [the KDA report](../../research/codex/R37_kda_prefill_sm80_0928.md) and
[raw evidence](../../evidence/prefill-kernels-0928/). Synthetic per-rank KDA
measurements exclude model projections, output normalization, TP collectives,
DSA/MoE/mHC, and service queueing. They establish no N@SLO or chain improvement.
Deployment acceptance still requires TP8 real-model smoke and a matched
same-request service arm, with chain, TPOT and pool sizes checked together.
