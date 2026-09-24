# patches/ — our changes to the organizer's SGLang (one version per mechanism)

Every patch applies with `patch -p3 --fuzz=0` onto `build/base_exact/sglang` (the byte-exact L3 base), in
**numeric order**. Each mechanism has exactly one patch and one `.md`; a change is made in place, and old versions live only
in git history (user policy 2026-09-24). To try a change, add a small single-purpose patch on top of the baseline; fold it
in if it wins, delete it if it loses.

Build or compare trees on the CPU: `python3 scripts/patch_stack.py apply OUT <patches...>` / `same TREE_A TREE_B`.

## Baseline S0 (the verified best config; = run 026, 035)
`000 101 106 110 111 120 140`, launched with
`--chunked-prefill-size 16384 --mem-fraction-static 0.75 --enable-attn-tp-input-scattered --cuda-graph-max-bs-decode 64
--max-mamba-cache-size 200 --schedule-policy lpm --dsa-prefill-backend tilelang --dsa-decode-backend tilelang` and env
`SGLANG_OPT_USE_TOPK_V2=0 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=154827,154829
SGLANG_AX_KDA_DUAL_SNAPSHOT=1 SGLANG_AX_SCHED_COLD_CAP=8192 SGLANG_AX_SCHED_SHORT_TOKENS=8192`.
Dev N22 (035, scored with the harness scorer): 10 of 11 gates pass; tpot_p95 0.296 fails; see `evidence/L035/`.

S1 = S0 + `114-indexer-row-shard.patch`, with the same launch configuration; run 036 tested this single change at dev N22. Its 11-gate verdict still fails only `tpot_p95` (0.253); see `evidence/L036/`.

## Patches
| Patch | Mechanism | In S0 | Switches (default) | Verified on 8 cards |
|---|---|---|---|---|
| 000-interface-compliance | `/flush_cache` returns JSON and needs every worker; server receive timestamp | yes | — | yes (all runs) |
| 101-role-boundary-split | KDA state at the last `<|user|>`/`<|observation|>` boundary; at most one partial prefill per round | yes | `SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS` (unset = off) | yes |
| 106-defer-chunk-on-no-kv | skip a chunked continuation for one round when KV is exhausted (was a fatal "Prefill out of memory") | yes | `SGLANG_AX_DEFER_CHUNK_ON_NO_KV` (1) | yes (025b: KV hit 1.00 three times, no crash) |
| 110-sm80-dsa-indexer | DSA indexer on A100: fp8 KV soft-decoded, bf16 MMA; fused decode and prefill kernels | yes | `SGLANG_AX_SM80_INDEXER` (1) | yes |
| 111-sm80-fp8-moe-marlin | FP8 MoE through Marlin W8A16 on sm80 | yes | `SGLANG_AX_SM80_FP8_MOE_MARLIN` (true) | yes |
| 114-indexer-row-shard | indexer prefill rows split over TP ranks (removes 8x duplicate work) | candidate | `SGLANG_AX_INDEXER_ROW_SHARD` (1), `_MIN_ROWS` (1024) | probe only (026g/h); N22 run 036 |
| 115-dcp-sm80 | decode context parallel on A100: 64-head sparse kernel + DSA address fix | candidate | `--dcp-size N` (off) | no (dev box TP2+DCP2 only) |
| 120-sched-protect-chain | decode turn after each prefill; continuation capped while others wait; short hits share the batch | yes | `SGLANG_AX_SCHED_PROTECT` (1), `_COLD_CAP` (2048), `_SHORT_TOKENS` (4096) | yes |
| 121-sched-cap-while-decoding | also cap continuations while any request is decoding | candidate | — (patch applied = on) | only inside multi-change runs 027/028 |
| 122-tpot-paced-prefill | prefill token budget from each decoder's measured TPOT pace (Sarathi-style), full chunks only, short-hit reserve replaces COLD_CAP; replaces interval/120 decode turn when on | candidate | `SGLANG_AX_PACE_TPOT` (off) | no; CPU tests + simulation only |
| 123-srpt-admission | admit waiting requests by remaining prefill work (with aging) instead of prefix length | candidate | `SGLANG_AX_SRPT_AGING` (off) | pending (S1+122+123 at N22) |
| 130-async-tokenize | tokenizer off the HTTP event loop; routing key plumbed | candidate | `SGLANG_AX_ASYNC_TOKENIZE` (1) | no single-change run |
| 140-kda-dual-snapshot | fp32 KDA states at the role boundary and prompt end from one prefill | yes | `SGLANG_AX_KDA_DUAL_SNAPSHOT` (0) | yes |
| 150-startup-warmup | representative-shape warmup at startup | candidate | `--warmups ax_shapes` (off) | no single-change run |
| 160-nextn-sm80 | MTP/NEXTN speculative decoding on sm80 | candidate | `--speculative-algorithm NEXTN ...` (off) | only inside multi-change runs 028/034 |
| 170-glm-bcg-prefill | breakable prefill CUDA graph for GLM | candidate | `--cuda-graph-backend-prefill breakable` (off) | v2 fix proven on TP2 only; TP8 recheck pending |
| 171-kda-bf16-proj-fusion | fuse already unquantized KDA projections within the FP8 model | candidate | `SGLANG_AX_KDA_FUSE_PROJ` (0) | no; single-GPU operator/loader screening, see [171](171-kda-bf16-proj-fusion.md) |
| 172-moe-clamped-swiglu | fuse BF16 clamped SwiGLU in Marlin MoE, preserving intermediate rounding | candidate | `SGLANG_AX_MOE_FUSE_SWIGLU` (0) | no; single-GPU numeric and full MoE cost screening, see [172](172-moe-clamped-swiglu.md) |

Every candidate applies alone on top of S0 (checked with `patch_stack.py`, evidence/T57), so each can be tested as a single change.

## Couplings (read before combining)
- 140 needs 101's role IDs env. 160 turns off 140 and clears 101's role IDs when MTP is on (code guard in 160; that 140 and MTP
  truly conflict is **untested**).
- 120's single decode turn after a prefill is overridden by an explicit `--prefill-decode-interval K`.
- `--chunked-prefill-size 16384` lowers the automatic `mem_fraction_static` to 0.646 (F81); S0 therefore sets 0.75 explicitly.
- `--max-mamba-cache-size 200` trades KDA state slots for KV (1.32M KV tokens in S0). 140's extra role slot is skipped when no
  slot is free, and tail states are evicted first; R20 (T56) found 61 follow-ups whose tail state was gone.
- On A100: DSA backends must be tilelang (fa3 is Hopper-only, F57); env `SGLANG_OPT_USE_TOPK_V2=0` is required (task.md:424).

## Submitted images
- 0923a (attempts 45979/45980, commit c405467): `000 101 106 110 111 114 120 121 130 140 150 160 170` in this stack,
  byte-identical to what was built (evidence/T57/equivalence.log).
