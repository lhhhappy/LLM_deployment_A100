# engine/ — our SGLang source (git-managed)

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
- The scheduler logs one `[ax] mechanisms:` line at startup (`101 118 120 122 123 140 180` as `on` / `off:<reason>`, plus
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
| 115 | decode context parallel on A100 | no | `--dcp-size N` (off) | TP8 33/40 row mismatch unfixed |
| 117 | FP8 MoE via Humming W8A16 on sm80 (takes precedence over 111) | no | `SGLANG_AX_SM80_FP8_MOE_HUMMING` (0) | dev box only |
| 118 | DSA sparse attention through a Triton kernel instead of TileLang (prefill, verify, draft, decode) | no | `SGLANG_AX_DSA_SPARSE_TRITON` (off) | not run on 8 cards |
| 128 | family leader ranking inside 124: waiting cold requests sharing an uncached prefix (256-token block hashes) form a family; the leader ranks by work per rider, riders wait for it | no | `SGLANG_AX_DEADLINE_FAMILY` (0), needs 124 | CPU only; N34 opening probe pending |
| 120 | protect chain: cold-chunk cap while others wait, short hits share the batch, decode turn | yes | `SGLANG_AX_SCHED_PROTECT` (1), `_COLD_CAP`, `_SHORT_TOKENS` | yes |
| 121 | also cap continuations while decoding | yes | — | only inside official A |
| 122 | TPOT-paced prefill budget (Sarathi-style) | no | `SGLANG_AX_PACE_TPOT` (off) | 048 dev N22: fast passes, overall/chain fail; 061s full N30 timed replay in progress |
| 123 | SRPT admission with aging | no | `SGLANG_AX_SRPT_AGING` (off) | no single-change run |
| 130 | tokenizer off the HTTP loop | yes (off by env) | `SGLANG_AX_ASYNC_TOKENIZE` | — |
| 140 | fp32 KDA states at role boundary and prompt end | yes (off: 160 under MTP) | `SGLANG_AX_KDA_DUAL_SNAPSHOT` (0) | yes (without MTP) |
| 150 | representative-shape warmup | yes (skipped under MTP) | `--warmups ax_shapes` | — |
| 160 | MTP/NEXTN on sm80; clears 101/140 env when on | yes | `--speculative-algorithm NEXTN` | yes |
| 170 | breakable prefill CUDA graph | yes (not enabled) | `--cuda-graph-backend-prefill breakable` | v2 TP8 recheck pending |
| 171 | KDA BF16 projection fusion | no | `SGLANG_AX_KDA_FUSE_PROJ` (0) | 056 dev N22: mixed result, smaller KV/state pools; no confirmed net gain |
| 172 | Marlin MoE clamped-SwiGLU fusion | no | `SGLANG_AX_MOE_FUSE_SWIGLU` (0) | 057 dev N22: TPOT mean −1.6%, mixed TTFT; one run |
| 180 | HiCache host tier for GLM DSA; keeps 120/122 on with the host tier | no | `--enable-hierarchical-cache --hicache-size N` (off) | 059/060 used the old version (120 silently off); 063s full N30 with 122 queued; GPU restore correctness still unverified |

124 (short-hit reserve) was not migrated: it conflicts with 122/123, and 122 has its own short-hit reserve. Its patch remains in
git history (the `patches/` directory was removed after the migration).

## Default-off audit (2026-09-24, by reading `git diff official-A-0923a HEAD`)
- 118, 122, 123, 171, 172: identical when off.
- 117: off keeps `Fp8MoEMethod` with the same 111 flag; dev box shows bitwise-equal weights and outputs against
  37e90023's `fp8.py` (routing alignment held fixed), see `engine/docs/117-sm80-fp8-moe-humming.md`.
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
