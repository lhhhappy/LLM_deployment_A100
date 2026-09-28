# 117 follow-up — large-prefill weighted reduction on SM80

Candidate, default off. This follow-up changes only the execution of the
weighted expert-output reduction. It has no TP8 or same-ID chain result yet.

## Dispatch and arithmetic

`SGLANG_AX_SM80_MOE_REDUCE=1` enables the branch. The flag is read once when
`moe_fused_mul_sum.py` is imported; set it before worker startup. Leaving it
unset or setting `0` preserves the native heuristic and kernel launch, without
checking the new branch's shape or GPU capability on each call.

The candidate accepts real contiguous CUDA BF16 input `[M, 9, 4096]` with
`M >= 4096`, contiguous FP32 routing weights, and contiguous BF16 output on
the same device. The device must have capability exactly `(8, 0)`; the check
uses the input's explicit device and caches only that device's result.
Routed scale is `None`, `1`, or `2.5`. Expert mapping and expert parallelism
keep the native path. Other shapes, dtypes, output layouts and scales also
keep the native path and its existing validation. Fake inputs still return
shape-correct output without any kernel launch or capability query.

One CTA owns one token and 1024 columns, with four warps and one stage.
Adjacent CTAs traverse the four column tiles of each token before advancing
to the next token. Input reads use `.cg`; output stores use the default
policy. There is no persistent loop, atomic operation, shared accumulator,
extra workspace or cross-CTA synchronization. Row offsets are 64-bit.

The kernel visits the nine expert slots in the native order, multiplies
each FP32 route weight by the routed scale before accumulation, uses the
same serial FP32 FMA accumulation, and rounds once to BF16 at output.
It retains all expert contributions, including the fused shared expert.
The kernel follows the existing Triton launch contract: the caller selects
the CUDA device/current stream. It adds no per-call device context switch.

## Startup and memory

117's layer-load warmup calls `warmup_sm80_moe_reduce` after the existing
warmup forwards. The helper returns immediately when the switch is off.
When eligible it compiles `M=4096` and `M=4097`, covering Triton's scalar
divisible-by-16 and other-integer specializations; `M` is not `constexpr`.

Each warmup launches only four CTAs with one row of buffers. Because the
kernel has no persistent loop, those CTAs touch row zero only even though
the representative `M` is large. This uses 81,956 bytes of tensor storage
(allocator rounding aside), avoiding a full 4096-token MoE workspace.
Warmup is cached per device and scale, keeps no tensors, and uses an explicit
CUDA device context. Serving allocates exactly the same output as before
when the caller does not supply one, and no new storage when it does.

## Evidence and limits

The original conservative configuration in
[`bench_moe_stream_sm80.py`](../../scripts/analysis/bench_moe_stream_sm80.py)
is `[BM=1, BK=1024, warps=4, load=.cg, store=default, waves=0, order=0]`.
Five paired rounds on one A100-SXM4-80GB, with L2-cold inputs, measured:

| M | Native reduce | Candidate reduce | Native graph | Candidate graph |
|---:|---:|---:|---:|---:|
| 8192 | 0.398816 ms | 0.390016 ms | 0.385620 ms | 0.377876 ms |
| 16384 | 0.778016 ms | 0.762336 ms | 0.765304 ms | 0.749984 ms |

Those are prototype reduction timings, not production-dispatch timings or
service throughput. All compared synthetic outputs were exact. Raw evidence:
[`moe-stream-a.jsonl`](../../evidence/mhc-moe-sm80-0928/moe-stream-a.jsonl).
The larger context and negative mHC results remain in the
[development report](../../notes/reports/moe-mhc-sm80-0928.md).

The focused regression is
[`test_moe_reduce_117_followup.py`](../../tests/gpu/test_moe_reduce_117_followup.py).
It checks raw BF16 bits against the unchanged native reducer at 4095, 4096,
4097, 8192, 8193, 16384 and 16385 tokens, scale 1/2.5, zero/negative weights,
cancellation, shared-expert contribution and reused outputs. It also covers
default-off dispatch, unsupported metadata, EP/map masking with poisoned
excluded rows, fake tensors, explicit-device capability caching, fresh-input
CUDA graph replay, concurrent independent streams and actual Humming down
outputs from the existing 117 synthetic-checkpoint layer fixture.

Run in the developer environment with one A100 visible and the candidate
engine on `PYTHONPATH`:

```sh
python tests/gpu/test_moe_reduce_117_followup.py
```

The test writes JSON records and stops on a failed assertion; it needs no
pytest. `--only dispatch` exercises metadata/launch selection without GPU
launches. Tests toggle the cached flag inside their isolated process.

On 2026-09-28, the production regression completed on GPU 1 (A100-SXM4-80GB,
torch 2.13.0+cu130, Triton 3.7.1): 61 bitwise identities, 19 fallback cases,
and all five test groups passed. The warmup test forbids any subsequent
Triton compilation and successfully runs real 4096/4097/8192/8193/16384/16385
buffers with `None`, integer `1`, float `1.0` and `2.5` scales. Both warmup
scale cases together peaked at 164,864 bytes of allocated GPU storage.
Receipts: [regression](../../evidence/mhc-moe-sm80-0928/moe-reduce-production-a.jsonl),
[exit code](../../evidence/mhc-moe-sm80-0928/moe-reduce-production-a.exit),
[uploaded source hashes](../../evidence/mhc-moe-sm80-0928/moe-reduce-production-a-upload.json).

Paired production-dispatch timing also completed, five randomized cold rounds
and fifteen graph rounds. Each graph contains eight reductions:

| M | Native API event | Candidate API event | Native graph | Candidate graph |
|---:|---:|---:|---:|---:|
| 4096 | 0.219296 ms | 0.206720 ms | 0.198852 ms | 0.194600 ms |
| 4097 | 0.218112 ms | 0.204736 ms | 0.195876 ms | 0.191964 ms |
| 8192 | 0.407520 ms | 0.390464 ms | 0.385512 ms | 0.377908 ms |
| 16384 | 0.791584 ms | 0.763264 ms | 0.765556 ms | 0.750916 ms |

API event times can include host enqueue gaps; graph times show the device
gain remains about 2%. Median host enqueue at 16k was 39.66 to 21.90 us:
the native heuristic queries capability per call, while the new guard caches
the explicit-device result. This is wrapper cost, not additional device
throughput. These are one developer run with synthetic inputs, with no
service or chain claim. [All rounds](../../evidence/mhc-moe-sm80-0928/moe-reduce-production-timing-a.jsonl)
and [exact timing source](../../evidence/mhc-moe-sm80-0928/moe-reduce-production-timing-a-source.json)
are retained.

The Humming GEMMs retain their existing stream-K/alignment nondeterminism;
the direct down-output comparison does not use that as an error tolerance.
Adoption still requires the TP8 and same-ID chain/TPOT checks owned by the
service experiment, with chain regressions rejecting the candidate.

The final seven-arm D/E full-MoE comparison did not establish a consistent
extra layer gain from this reducer. Its isolated graph gain remains about
2%, while full-layer reduce-only or adding reduce to up+down can be within
the observed noise. Keep it available as an independently switchable option;
the main service candidate enables up+down and leaves reduce off. See the
[final report](../../notes/reports/moe-mhc-sm80-0928.md) for exact arm results
and limits. This does not remove the large intermediate down-output tensor.
