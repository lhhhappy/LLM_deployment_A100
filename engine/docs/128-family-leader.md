# 128 — family leader ranking inside 124 (candidate, default off)

Replaces the withdrawn first design (branch `claude/128-family`, 3aad9c4f), which grouped requests by LPM's
in-batch holdbacks. Those exist only for requests whose real cache match is at most 32 tokens; in the opening the
shared app prefix (about 8k) is cached within 1–2 s, so later chain starts were never grouped
(`evidence/engine-128-review-20260926.md`).

## Why
Run 109 (v3 N26 opening, 124 + aggressive 125): of the 9 remaining chain-start misses, 4 were siblings of one
family — prompts of 35–37k tokens sharing 32k, each with about 3k of its own work — that waited 30–44 s for
their leader (36k, cold) to compute the shared prefix. The leader ranked by its own 36k behind the smaller lone
heads and finished at 29.7 s; the siblings then rode along as short hits (batches `seq=2/3 cached=32768/65792`
at 30–33 s) but had already missed 30 s (`notes/reports/109-chain-turn-rootcause-0926.md`). In the organizer's
311 chain heads 291 belong to 24 prefix families and the source-side cache ratio of a head is 0.96 at the median,
so this structure is the real workload, not an artifact of the synthetic set.

## What it does (`ax_deadline.family_plan`, `Scheduler._ax_family_plan`)
Off unless `SGLANG_AX_DEADLINE_FAMILY=1`; requires 124 (`SGLANG_AX_DEADLINE_TIERS=1`), refuses otherwise.
On request-plane rank 0, inside 124's plan (so the result travels with the broadcast order):
1. Every waiting cold request (124's `is_cold`) gets a hash per 256-token block of its prompt, computed once.
2. Partition by radix namespace (`extra_key`, `cache_salt`): different namespaces cannot be a family. On a
   membership change, sort the block-hash sequences in each namespace, compute their adjacent LCPs, then fill
   pair LCPs with interval minima. This needs only n−1 Python prefix walks per namespace. Existing pairs remain
   cached; a changed namespace or reused RID invalidates affected pairs. Two requests are linked when the prefix
   minus the longer cache match of the two is
   at least `link_min` (4096): the part one of them computes and the other reuses.
3. Families are the connected components. The leader is the member with the least remaining work. A rider is
   another member whose remaining after the leader's prefix is at most `rider_max` (6144: it then finishes as
   a short hit beside the next cold chunk). A member with a longer tail neither holds nor counts.
4. The leader ranks in 124's order by `remaining / (1 + riders)` (its rescuable/hopeless judgement keeps the
   real remaining); the riders are held (tier 3, last in the order, like LPM's holdbacks) while the leader waits.
   The starvation priority releases family-held riders; LPM holdbacks remain separate. Held is a priority, not a
   ban: the admission loop does not skip them, but a rider is reached only after every
   earlier waiter was admitted or refused, and with the leader admitted as the round's partial a multi-round rider
   is refused anyway (one partial per batch). Once the leader runs, its chunks enter the tree, the riders' matches
   grow and they become ordinary short hits.
5. Bounded CPU: with more than `max_candidates` (64) waiting cold requests the pairwise scan is skipped for the
   round and 124's per-request order applies. Pair metadata is reset by `/flush_cache`; prompt hashes live on the
   request. Hashing uses bounded token tuples rather than copying the full prompt into an int64 array. Both first
   and cached rounds are measured by `scripts/analysis/benchmark_s1s2_cpu.py`; see the
   [S1/S2 review](../../notes/reports/review-s1s2-patches-0926.md) for paired results and source hashes.
Logs `[ax-128] families=N leader=…:riders=k:others=m:work=…` when the set changes (at most every 2 s); the
mechanism line shows `128=on`.

## Switches
`SGLANG_AX_DEADLINE_FAMILY` (0/1); `SGLANG_AX_FAMILY_BLOCK` (256), `SGLANG_AX_FAMILY_LINK_MIN` (4096),
`SGLANG_AX_FAMILY_RIDER_MAX` (6144), `SGLANG_AX_FAMILY_MAX_CANDIDATES` (64).

## Evidence
CPU (`tests/test_ax_deadline.py` `Family`, `tests/test_ax_admission_scheduler.py` `FamilyOrder`): block hashes,
leader and rider selection, the link needs an uncached shared prefix, a big-tailed member neither holds nor
dilutes, the override only ranks; on the real scheduler the family leader is admitted before a 14k and a 19k lone
head while its riders stay held, with 128 off the 14k head goes first, 128 without 124 refuses.
Estimate (queueing model of 109's opening, 10.3k tok/s): the family's 4 misses become passes, chain 9 → 5.
Additional regressions compare every pair against a naive LCP reference, separate namespaces, reuse RIDs and
flush state. Local CPU timings do not establish TP8 or end-to-end SLO improvements.
