# engine/ — engine sources (git-managed)

The user has authorized parallel SGLang and vLLM work. Codex maintains `engine/sglang/`;
Claude Code owns the new `engine/vllm/` route, beginning with a working baseline and interface/evaluation parity.
The vLLM base is official vLLM `main` a811738a6 (tag `vllm-base-a811738a6`); base, rules and mechanisms are in
[engine/docs/vllm/](docs/vllm/README.md). vLLM uses separate tags, `engine vllm NNN:` commits and `engine/docs/vllm/` notes. Existing tree/export/image/Pod
tools below are SGLang-specific and must not be used as if they already support vLLM.
The remainder of this file describes the existing SGLang implementation.

`engine/sglang/` is the organizer's base (`build/base_exact/sglang`, byte-exact L3) plus our mechanisms. Each mechanism is
one or more commits whose subject starts `engine NNN:`. Its design notes live in `engine/docs/NNN-*.md`. Replaced the
numbered-patch stack on 2026-09-24 (user decision). A run or submission is identified by a git commit, and
`git diff <a> <b> -- engine/sglang` shows exactly what differs.

## Tags
- `engine-base`: the organizer base, unchanged.
- `official-A-0923a`: official A (image 0923a, attempts 45979/45980). Equal file-by-file (4690 files) to the 13-patch
  tree `000 101 106 110 111 114 120 121 130 140 150 160 170`, which T57 proved equal to the built image.

## Working rules
- Change the engine only here. One commit or a small series per mechanism. A fix goes into the mechanism it belongs to
  as a follow-up `engine NNN:` commit, not a new number.
- Every mechanism is switchable (flag/env) and its default-off path is the base code. No silent bypass: an unsupported
  combination works with tests or refuses to start.
- Optional 120 admission diagnostics use `SGLANG_AX_ADMISSION_TRACE=1` (default off), reported as
  `120_trace=on|off`. They count observed admission decisions on TP0 and do not change scheduling;
  CPU regressions are complete, diagnostic overhead and real failure attribution require TP8 sampling.
- The separate 120 KV fallback uses `SGLANG_AX_SCHED_KV_SCAN=K` (default `0`, max `16`),
  reported as `120_scan=off|K`. It scans a bounded tail after a rejected KV candidate,
  retaining native budgets and single-partial ownership. CPU candidate only; TP8 confirmation pending.
- The scheduler logs one `[ax] mechanisms:` line at startup (`101 120 122 123 140 180` as `on` / `off:<reason>`, plus
  speculative algorithm as resolved by the base, so `--speculative-algorithm NEXTN` shows as `spec=EAGLE`, DCP size and the requested model-side switches). Pod jobs declare `G_EXPECT` and refuse to
  measure on a mismatch.
- Trees for tests and tools: `python3 scripts/engine/tree.py <ref>` (`official-A-0923a`, `HEAD`, `mech:NNN` = latest
  commit of NNN, `before:NNN` = before its first commit).
- Pod: `scripts/engine/export.sh <ref>` writes `build/engine/<commit>.diff` (base → commit). The pod applies it to its base
  in `lib.sh prepare_src <commit>`. Job files set `G_COMMIT`, `G_ARGS`, `G_ENV`, `G_EXPECT`.

## Mechanisms (HEAD = official A + candidates, all candidates default off)
| NNN | Mechanism | In official A | Switch (default) | 8-card status |
|---|---|---|---|---|
| 000 | `/flush_cache` JSON + every worker; server receive timestamp | yes | — | all runs |
| 101 | KDA state at the last `<|user|>`/`<|observation|>` boundary | yes (but 160 clears it under MTP) | `SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS` | yes |
| 106 | defer a chunked continuation when KV is exhausted | yes | `SGLANG_AX_DEFER_CHUNK_ON_NO_KV` (1) | yes |
| 110 | DSA indexer on A100 (fp8 soft-decode, bf16 MMA) | yes | `SGLANG_AX_SM80_INDEXER` (1) | yes |
| 111 | FP8 MoE via Marlin W8A16 on sm80 | yes | `SGLANG_AX_SM80_FP8_MOE_MARLIN` (1) | yes |
| 114 | indexer prefill rows split over TP ranks | yes | `SGLANG_AX_INDEXER_ROW_SHARD` (1) | yes |
| 115 | decode context parallel on A100 | no | `--dcp-size N` (off) | TP8-validated at `--dcp-size 2` across many N30/N34/N38 windows (e.g. 130ed/130ee5/130eez/130eezy) and 2 official submissions (46757 N22, 46758 N26); capability-checked (AIME 28/30, GPQA 178/197 on the DCP2 engine, roughly matching the non-DCP engine on the same public question set). Doubles the local KV pool (1.40M→2.38M tokens at N30) and removes the queueing the local `v3` dataset's short synthetic gaps created — that queueing turned out to be a data artifact, not present online (46757/46758 show no chain improvement over the non-DCP S1/S2 pair), so DCP alone is **not adopted for chain**. `SGLANG_AX_DCP_LOCAL_EXTEND` (local re-extend of a continuation) was tried once together with DCP (S4/46758) without a clean single-variable online comparison; not separately confirmed. `--dcp-size 4` removes the N38 pool ceiling too but had one unexplained capability-smoke miss (11/12), not used for scoring. The historical row-alignment bug this row used to describe is fixed; the fix (commits `7d6099ac`/`791453ca`/`ebfcaaa9` etc.) lives on a review branch not yet merged into this tree. Current numbers: [knowledge.md](../notes/knowledge.md), [program-n30-v3.md](../notes/program-n30-v3.md) |
| 117 | FP8 MoE experts through Humming on A100 (replaces 111/Marlin execution for MoE layers only; still FP8 weights, BF16 activations) | no | `SGLANG_AX_SM80_FP8_MOE_HUMMING` (0) | TP8-measured pure per-token cost cut, twice: 084 TPOT mean 32.90→30.33 ms (−7.8%), 110 TPOT mean 50.63→45.56 ms (−10%, TPOT>0.10 23→4); does not move chain_start miss counts (it is not a scheduling change). In official submissions 46676/46677/46757/46758 and the 0927a candidate |
| 118 | DSA sparse attention through a Triton kernel on A100, prefill-only, DCP-compatible | no | `SGLANG_AX_DSA_SPARSE_TRITON_PREFILL` (0) | TP8-measured −8% per prefill chunk (8k: 588→541 ms; 16k: 1085→989 ms). N34 opening probe: chain unchanged (11=11 — a pure cost cut, not a scheduling fix), fast/overall/TPOT improve. 15 operator-level numeric tests are within tolerance but not bit-exact; no full-model logits equivalence yet. **Not in the upload candidate**: needs an AIME/GPQA capability check first |
| 119 | gate attention-TP input scatter to extends of at least `SGLANG_AX_SCATTER_MIN_TOKENS` tokens | no | `SGLANG_AX_SCATTER_MIN_TOKENS` | Not separately measured in the ledger; see commits `928b9e18`/`afd9b7d2` |
| 120 | protect chain: cold-chunk cap while others wait, short hits share the batch, decode turn | yes | `SGLANG_AX_SCHED_PROTECT` (1), `_COLD_CAP`, `_SHORT_TOKENS` | yes |
| 121 | also cap continuations while decoding | yes | — | only inside official A |
| 122 | TPOT-paced prefill budget (Sarathi-style) | no | `SGLANG_AX_PACE_TPOT` (off) | Fixed version: official 46174 host32 N18 PASS; 46173 off N14 PASS. Full local N30 067/068 both FAIL; host64+122 recovery experiment 071: fast improved, chain/overall did not, TPOT held. Retried 2026-09-26 as a substitute for aggressive 125 (130e, same request IDs): chain 14→21, turn 0→6 — loses exactly the chain/turn margin the current strategy needs, so **not carried into S1 or later**; superseded by 125/126/131 for prefill-budget decisions |
| 123 | SRPT admission with aging | no | `SGLANG_AX_SRPT_AGING` (off) | old 037c→037d improved chain, both failed; superseded 2026-09-26 by 124 (single-request shortest-remaining-first is one tier inside 124's deadline order) and 128p (family-aware ranking on top of that); no further standalone validation planned |
| 124 | deadline-tiered admission with chunk-level parking (server-visible cold/warm budget tiers, starvation bound, park a continuation for a waiter that still fits its budget) | no | `SGLANG_AX_DEADLINE_TIERS` (0), plus `SGLANG_AX_DEADLINE_WARM_S` / `_MAX_WAIT_S` / `_MAX_WAIT_WARM_S` / `_WARM_MULTI_S` / `_FREEZE_CLASS` | Central mechanism of every S1–S4 submission and the 0927a candidate; TP8-validated across dozens of windows since 071. Follow-ups fixed the parking rule to park for any feasible waiter and resume a rejected turn immediately (`bab21ddf`/`efe837c8`) and separated the warm/cold starvation bounds (`84dcca0e`); **`_MAX_WAIT_WARM_S` still silently inherits `_MAX_WAIT_S` unless set explicitly** (Codex, 2026-09-28) — the 0927a candidate sets both (600 cold / 120 warm). A related flush-state bug (short-hit reserve state surviving a `/flush_cache`) was found and fixed under mechanisms 126/128 (`5c30b9ed`), not 124 itself. Numbering note: an unrelated, never-migrated "124 short-hit reserve" idea from the pre-2026-09-24 patch scheme also used this number; see the note below the table |
| 125 | opening-mode backlog relief: widen the cold-chunk cap and decode interval while the cold backlog is large | no | `SGLANG_AX_BACKLOG_RELIEF` (0), `_HIGH_S` / `_LOW_S` / `_COLD_CAP` / `_INTERVAL` / `_MAX_SLOW` | In every S1+ submission (aggressive settings, "125x"). `_MAX_SLOW` 80→250 was tried alone as 46677/S2 without a prior local test (N26, chain p95 39.1 s, no clear gain over 46676/S1's 37.0 s) — a reminder to test single changes locally before spending an upload on them |
| 126 | size the cold chunk by waiting short-hits' demand (seat reservation, shared logic with 122) | no | `SGLANG_AX_SCHED_COLD_CAP_MAX` (0) | Tested twice on top of the then-current admission mechanism (130c on "A′": chain 14→13, no gate worse; 130e9 on "A": chain 13→15, one pathological new case) — no clear, reproducible chain benefit, so **not carried forward** into the S1+/chain-max stack. Not re-tested against the current 124/128p baseline |
| 127 | keep the prompt-end KDA checkpoint over a deep branch point in a request's last prefill chunk | no | see commit `6c9b6648` | Not separately measured in the ledger; current 8-card status not recorded |
| 128 | family/prefix-sharing admission — two independent sub-mechanisms sharing the number, both default off | no | see the two rows below | see the two rows below |
| 128p | track shared-prefix producers and reserve ready sibling admission ("prefix producer") | no | `SGLANG_AX_PREFIX_PRODUCER` (0), needs 120/124 | TP8-measured net chain fix at N34 opening (16→11 misses; family waits fell from 40–60 s to 4–19 s), reproduced across several pairings (130ez6zzz, 130ezn5, …). In the chain-max-16k stack and the 0927a upload candidate |
| 128g | reject a native LPM in-batch hold when it has zero possible reusable gain | no | `SGLANG_AX_LPM_REUSE_GUARD` (0) | TP8-measured: no measurable chain net gain on top of S6 (ezn2 vs ezn2a: chain 8=8, one case fixed / one new), overall and fast slightly worse. **Not adopted.** The plain config alternative — raising the base `IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD` (see below the table) — was adopted instead |
| 130 | tokenizer off the HTTP loop | yes (off by env) | `SGLANG_AX_ASYNC_TOKENIZE` | — |
| 131 | chain-risk decode interval: suppress decode rounds around a cold chain start's chunk while it is still within reach of its budget (steady state; the opening is already prefill-only under 125) | no | `SGLANG_AX_CHAIN_RISK_INTERVAL` (unset), needs 124 | TP8-measured: cuts 20–25% off a 100k+ token chain start's post-admission execution time by suppressing decode interleaving near its deadline. Rank-inconsistency (P1) found and fixed by commit `b3f8c3c0` — cadence is now rank0-synchronized and the cost model charges the batch's actual planned chunk size (e.g. 16k) instead of a fixed 8k assumption. In the 0927a upload candidate |
| 132 | chain-first admission order inside 124: rank rescuable cold requests ahead of rescuable warm ones, same rule applied inside 128p's ranking | no | `SGLANG_AX_DEADLINE_CHAIN_FIRST` (0) | TP8-measured as part of the "chain-max-16k" stack (16k chunks, 2048-token short-hit reserve, 124 cost coefficient 1.05): chain misses 8→6 (v5g-tail, N26, 40 min), harness-computed chain p95 41.3→24.8 s, at a cost of more fast/TPOT misses. Rank-consistency fixed together with 131 by commit `20a58da9`. In the 0927a upload candidate |
| 140 | fp32 KDA states at role boundary and prompt end | yes (off: 160 under MTP) | `SGLANG_AX_KDA_DUAL_SNAPSHOT` (0) | yes (without MTP) |
| 150 | representative-shape warmup | yes (skipped under MTP) | `--warmups ax_shapes` | — |
| 160 | MTP/NEXTN on sm80; clears 101/140 env when on | yes | `--speculative-algorithm NEXTN` | yes |
| 170 | breakable prefill CUDA graph | yes (not enabled) | `--cuda-graph-backend-prefill breakable` | v2 TP8 recheck pending |
| 171 | KDA BF16 projection fusion | no | `SGLANG_AX_KDA_FUSE_PROJ` (0) | 056 dev N22: mixed result, smaller KV/state pools; no confirmed net gain |
| 172 | Marlin MoE clamped-SwiGLU fusion | no | `SGLANG_AX_MOE_FUSE_SWIGLU` (0) | 057 dev N22: TPOT mean −1.6%, mixed TTFT; one run |
| 180 | HiCache host tier for GLM DSA; keeps 120/122 on with the host tier | no | `--enable-hierarchical-cache --hicache-size N` (off) | 067–069 completed TP8 full N30; 069 host64 passes 10/11 gates. Official host32 candidates pass capability, N14/N18. Dedicated GPU restore numerical validation remains separate |

124 (short-hit reserve) was not migrated during the 2026-09-24 renumbering: it conflicted with 122/123, and 122 already
had its own short-hit reserve. Its patch remains in git history (the `patches/` directory was removed after the
migration). **The number was reused from 2026-09-26 onward** for an unrelated mechanism, "deadline-tiered admission"
(the 124 row above): the two share nothing but the number. What the old, dropped 124 tried to do — reserve room for
waiting short hits beside a cold chunk — was later built properly as 126.

Mechanisms 117–119 and 124–132 above were developed and TP8-validated on review/topic branches (`codex/…`, `claude/…`)
that are not yet merged into this tree's `engine/sglang/`; their per-mechanism `engine/docs/NNN-*.md` files are
therefore not present here (`engine/docs/115-dcp-sm80.md` is the exception — the base 115 mechanism is on this tree
and its doc carries a current-status pointer; the newer `115-dcp-local-extend.md` sub-doc is on the same
unmerged branches as 117–132). This table's status cells summarize the shared ledger
([knowledge.md](../notes/knowledge.md), [program-n30-v3.md](../notes/program-n30-v3.md),
[experiments.md](../notes/experiments.md)); check [queue.md](../notes/queue.md) for the actual commit a given pod job
runs, since it is frequently a specific unmerged commit hash rather than anything reachable from this branch's `HEAD`.

Two native (organizer) SGLang parameters are currently tuned away from their defaults rather than gated by a numbered
mechanism, so they are not in the table above:
- `--max-mamba-cache-size` (pins the unified KDA/mamba slot pool instead of letting it auto-size from the memory
  fraction). Pinning to 400 slots after dropping MTP raises the KV token pool from 1.26M to 1.81M (+44%), because the
  unpinned pool otherwise grows to whatever the freed MTP speculative buffers leave it (779 slots, of which only
  ~110 are ever used at runtime). Measured once (ezn8): it also let an already-doomed giant chain head admit earlier
  and displace two rescuable smaller heads, costing 2 opening chain misses at N26 (6→8). Chain is judged first, so
  **this is not in the upload candidate**; revisit once a "let a doomed head yield" mechanism exists (see
  `question.md` §7 Q1, the infra-expert brief).
- `IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD` (`schedule_policy.py`, default 32 tokens): the base SGLang
  threshold for deprioritizing a request in-batch when it shares a short prefix with one already running. Raised to
  4096 in the 0927a upload candidate after tracing two rescuable 35k/56k cold heads that a same-pack giant — sharing
  only a 413-token system prefix, too short to ever land on a reusable KDA checkpoint boundary — pushed to the back
  of the queue. This is a plain config change on the native path, independent of mechanism 128g (128g's opt-in guard
  on the same native path was tested and not adopted; see above).

## Default-off audit (2026-09-24, by reading `git diff official-A-0923a HEAD`)
- 122, 123, 171, 172: identical when off.
- 115: source review confirms the default TP8 path retains staged output (`stage_output=True`); generated IR equivalence
  has not been independently checked.
- 180: identical while every extend starts on a 64-token boundary. The one exception: a continuing chunk shortened to
  1–63 tokens by KV headroom. The base then saved a KDA checkpoint under a page-rounded key, where the state did not
  match the key. 180 saves none. This prevents publishing a state under the wrong prefix, but can lose cache hits.
  Record KV/mamba pool sizes and `skip_unaligned_prefix` events on 8 cards. Repeated free-generation equality is not a gate.

## Couplings (read before combining)
- 140 needs 101's role IDs env. 160 turns off 140 and clears 101's role IDs when MTP is on (untested whether 140 and MTP
  truly conflict).
- 120's single decode turn is overridden by an explicit `--prefill-decode-interval K`; 122 replaces both when on.
- `--chunked-prefill-size 16384` lowers the automatic `mem_fraction_static` to 0.646; set it explicitly.
- On A100: DSA backends must be tilelang; `SGLANG_OPT_USE_TOPK_V2=0` is required (task.md).
