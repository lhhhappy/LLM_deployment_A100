# 176: SM80 DSA decode query reuse across four pages

`SGLANG_AX_SM80_INDEXER_DECODE_LOOP=1` opts into a paged logits kernel that
reuses the decoded query and weights across four consecutive 64-key pages.
It defaults off. Only SM80, 32 index heads and the existing D128/page64
contract are covered. The existing paged wrapper handles strides, output
shape, packed cache and context-length broadcasting as before.

The launch decision uses shape metadata, never device tensor contents:
use the loop when rows >=16 and pooled logits width >=8192, or width >=32768.
Smaller grids retain the native kernel because grouping pages can reduce
parallelism. Other head counts/devices also retain the native kernel.
No workspace, persistent state, quantization or top-k implementation is added.

Each page retains the original BF16 dot-product rounding per head, FP32
weighted reduction and scale application, with FP fusion disabled. Negative
physical page IDs still clamp to zero. Every logits position is overwritten,
including skipped pages on shrinking or empty CUDA graph replay. Prefill
logits and the 118 prefill-only attention path are untouched.

On A100-SXM4-80GB with synthetic inputs and real H32/KPool4 geometry, the
complete logits + original KPool selector stage at B32/raw250k was
1003.9→717.6 microseconds for uniform lengths and 445.5→333.8 for mixed
lengths. Fixed pooled width262144 with changing live lengths also improved
on the measured B8/24/32/40/48 cases. Exact final-source PAGED qualification
had bitwise logits and equal selected sets in 35/35 candidate comparisons.
Native top-k uses unordered atomic append: output order changed on all 150
offset-mapping baseline self-comparisons. PAGED baseline self selected sets
were equal in 172/175 comparisons, with three failures on a tied mixed
case; the selector itself is unchanged. This is not a claim of deterministic
top-k output, bitwise identical whole-model output or checkpoint state.

Qualification includes noncontiguous query/weights/context/page-table views,
gapped cache rows, multi-query and shared-context broadcasting, negative
pages, partial pages, zero outputs and changed-input graph replay through
shrinking, empty and growing contexts. A wider 128-key variant had FP32
last-bit differences and was not selected. Small-grid regressions motivate
the shape fallback above; the candidate is not universally faster.

Evidence: `evidence/execution-0930/dsa/{qualification-results,wrapper-results,final-wrapper-results,paged-stage-results}.json`
and corresponding launcher/source hashes. These are development measurements.
The first live route logs `SM80 DSA decode page loop engaged`; require all
eight ranks, real-weight smoke, memory accounting and same-N mixed-load
comparison before choosing a service candidate.
