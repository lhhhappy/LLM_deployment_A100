# SM80 prefill execution, 2026-09-28

Final decision and limits: [KDA R37](../../research/codex/R37_kda_prefill_sm80_0928.md).
Other rejected/local-only candidates: [R36](../../research/codex/R36_prefill_kernel_options_0928.md).

The final KDA evidence is `kda-prepare-f.*` (complete core, actual production
arms), `kda-prepare-production-test-f.*` (preparation contract),
`kda-state-production-b/` (recurrence signatures and state),
`kda-state-memory-a/` (native-warmed memory comparison), and
`kda-state-dispatch-g.*` (final 8k multi-request fallback).
`kda-production-f.tar` freezes the F source; `kda-production-g.tar` freezes
the narrowed instance gate and its focused test. The final branch uses G's
gate with unchanged F kernels. G does not claim a repeat of F's timing.

Earlier prepare a/c/e compare prototypes and the transition to independent
output storage; b is a retained compilation failure. `kda-production-d.tar`
and the first production-test receipt cover the earlier shared-storage
preparation, superseded by e/f. `kda-state-production-a/` is a retained test
memory-baseline failure, corrected in b without changing production kernels.

`kda-evidence-index.json` covers the earlier CPU-length/local-graph/BV16/BV8
research jobs only; its “no production recurrence change” refers to those
research jobs. Later production evidence above supersedes it for deployment.
All raw rounds, stderr, exits, source capsules and reviews remain available.
Remote uploads were per-file: an environment SHA is authoritative, not a
claim that the whole remote engine matched the local git HEAD.

`manifest-sha256.json` records all files in this directory except itself.
No TP8 real-model, whole-prefill or chain result is stored here.
