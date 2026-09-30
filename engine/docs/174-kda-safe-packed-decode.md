# 174: SM80 safe-gate KDA packed decode

Optional switch: `SGLANG_AX_SM80_KDA_PACKED_DECODE=1`; default off.

GLM's finite negative `lower_bound` prevented its decode from using the existing
packed recurrence. This route uses the existing packed Triton recurrence after
the unchanged convolution update and before the unchanged state tracking. It
combines Q/K/V extraction, normalization, safe gate, beta and state update.

Scope: exact SM80, Triton decode backend, no speculative algorithm or ReplaySSM,
one token per request, BF16 projections, FP32 state/gate parameters, int32 slots,
K=V=128 and matching Q/K heads. Unsupported inputs retain generic decode.
State pools retain their slot pitch and dense inner `[HV,V,K]` layout. Split
projection views can have gapped token rows; their inner elements must be dense.
No host tensor read, state copy or new persistent buffer is introduced.

The optional safe-gate route disables the packed wrapper's CUDA row-streaming
path. Its default remains enabled for existing callers. A100 measurements show
that the existing eight-warp CUDA kernel regresses B8/12/16; the four-warp variant
was slower too. The CUDA coverage check also rejects misaligned float4 state
offsets and slot pitches, falling back to scalar-load Triton.

Developer evidence (A100-SXM4-80GB, synthetic TP8 per-card H=HV=8): packed Triton
output and FP32 states match generic decode bitwise in the tested zero/nonzero
initial states, gapped projections, GQA H4/HV8, envelope-strided pools, 2048-step
normal/weak/extreme gate recurrence, and live-index CUDA-graph replays. Negative
slots produce zero and do not update any state. Generic negative-slot outputs
are not a reference; only active rows are compared. Exact updated Python
wrappers were also tested with CUDA dispatch instrumented to fail if reached.

Interleaved graph timings, generic → packed Triton, microseconds:
B8 9.66→8.55; B24 15.14→13.29; B32 17.82→17.96;
B36 19.02→19.33; B48 26.89→23.48. This is a local recurrence result,
not a complete decode-step, TPOT or chain improvement. B32/36 regress slightly.

Independent review: GPT-6.1 Sol. Evidence is in
`evidence/execution-0930/kda/` in the experiment workspace. TP8 real-weight smoke,
actual runtime engagement, full service metrics and capability remain pending.
