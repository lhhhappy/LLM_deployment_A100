# 124 — deadline-tiered admission and chunk-level parking (candidate, on top of 120)

## Why
Run 071 (N30) failed only chain_start (30 over, 29 allowed); 24 of the 30 arrived in the first 5.2 minutes.
In the opening wave a single long-prefill lane is busy, and LPM lets large cold requests and later arrivals with
more cache hits take it: 11 opening chain starts that needed ≤36.7k tokens (≤5.2 s of execution) waited 30–160 s
(`research/claude/R30_071_online_bottlenecks.md`). 073/074 (0925a, opening 10 minutes) showed prefill throughput
flat at about 9.2k tok/s from N22 to N26 while first-minute chain misses rose 13→18; an offline ordering model
(an estimate, not a proven bound) put the same arrivals at 8 and 11 misses with shortest-remaining-first and
demotion of requests that can no longer make it.

## What it does
Off unless `SGLANG_AX_DEADLINE_TIERS=1`. Requires 120's protection; exclusive with 123 (it replaces 123's order).
Unsupported combinations refuse at startup (`Scheduler._ax_admission_cfgs`, called by the mechanism report).

**Order** (`ax_deadline.tier_order`, applied after `calc_priority` matched prefixes):
1. requests waiting longer than `max_wait_s` (120 s), or `max_wait_warm_s` for warm requests, oldest first;
2. requests that can still meet their TTFT budget, shortest remaining prefill first;
3. requests that cannot, shortest remaining first;
4. LPM's in-batch prefix-sharing holdbacks, last as under LPM.

Only server-visible quantities are used, never the harness's frozen labels or phases:
- budget: 30 s if the matched prefix is under half the prompt (`is_cold`), else 3 s for ≤4096 uncached tokens and
  5 s above — a heuristic stand-in for the four TTFT buckets, which the server cannot see;
- service estimate `load × (chunks × fixed + tokens × per_token)` with the round's chunk (the cold cap under 120);
  defaults α 0.13 s per chunk, β 68 µs/token, load 1.27 were measured on 071 (8192-token chunks, 122 on) and need
  their own values for other chunk sizes or decode cadences (all `SGLANG_AX_DEADLINE_*` knobs);
- waited time from the scheduler's receive stamp plus `arrival_offset_s` (0.4 s, 071's receive-to-dispatch).
The estimate ignores the queue ahead, so "can still make it" is optimistic and "cannot" conservative. Tiers are
recomputed every round, so a request misjudged as hopeless is promoted as soon as it is not.

**Parking** (`ax_deadline.should_park`): rank 0 scans at most 64 ranked waiters for a complete, feasible prefill.
`PrefillAdder.ax_complete_waiter_budget` checks resident KV including host reload, output reserve, page overhead,
shared Mamba charge and the candidate's chunk limit. Device short hits can use the full round; cold requests
remain bounded by the cold cap. A free request slot is required. The selected waiter is broadcast first in the
order, so an infeasible head cannot hide it behind a native `NO_TOKEN` break. Unless the waiter is starved, it must
still have nonnegative slack and the continuation must have more than `park_min_remaining` (64k) tokens left or be
hopeless. A starved waiter can also trigger parking after its deadline. What the continuation has left is its sequence
minus the prefix computed so far (`prefix_indices`, as 122 and `add_chunked_req` count it); its match from when it
waited is not updated per chunk, so it only sets the budget, which stays the one it was admitted with. A waiter that
would need truncation cannot run beside a continuation (protection refuses a second partial), so it never triggers
parking. At most `park_max_rounds` (8) rounds and `park_max_s` (2 s) in a row. The parked continuation keeps its KV
and state; it computes nothing, so `extend_range` still ends at the cached prefix, the next round skips the stash, and
120's batch accounting counts only continuations that run. It is not marked as the adder's continuation, so cold
waiters take the normal complete-fit admission path. The preview does not lock, allocate or load cache state.
The native adder remains authoritative: if no waiter is admitted after re-match/COW/locking/delay, resume the
continuation in the same pass with live budgets; count a park only when a waiter actually runs. These wait limits
are admission priorities, not elapsed-time guarantees when resources are unavailable or parking limits apply.

**TP consistency**: arrival stamps and clocks are rank-local and a 1 ms difference can flip an order
(`evidence/opening-q123-20260925/`). The whole plan (order as RIDs, park flag) is computed on request-plane rank 0
(`Scheduler._ax_admission_plan`) and broadcast by `Scheduler._ax_rank0_decide` (the helper introduced with 123's
fix); every rank applies it and refuses if its queue's RIDs differ. It is entered only past the prefill
early-returns, whose conditions depend on replicated state. Parked-round counts live on the request and change only
from the broadcast decision.

Logs: `[ax] 124 <config>` at start (with 125 when on), `[ax-124/125] parks=… relief_rounds=…` every 30 s on rank 0;
the mechanism line shows `124=on`.

## Switches
`SGLANG_AX_DEADLINE_TIERS` (0/1); budgets `SGLANG_AX_DEADLINE_{COLD_S,WARM_S,FAST_S,FAST_TOKENS,COLD_HIT_RATIO}`;
cost `SGLANG_AX_DEADLINE_{FIXED_S,PER_TOKEN_S,LOAD,ARRIVAL_OFFSET_S}`; `SGLANG_AX_DEADLINE_MAX_WAIT_S`;
`SGLANG_AX_DEADLINE_MAX_WAIT_WARM_S` (defaults to `MAX_WAIT_S`);
parking `SGLANG_AX_PARK_{MIN_REMAINING,MAX_ROUNDS,MAX_S}` (`MAX_ROUNDS=0` turns parking off).

## Evidence
CPU, real scheduler code (`get_next_batch_to_run`, `PrefillAdder`, 120/121/123 paths) with the project's CPU
fakes for pools, cache tree and forwards (`tests/test_sched_protect_chain.py` harness):
- `tests/test_ax_deadline.py`: estimates, budgets, tier order (starvation bound, stability, cache hits), parking
  conditions (fit, KV room, hopeless waiter, short continuation, round and wall-clock bounds), config refusals;
- `tests/test_ax_admission_scheduler.py`: a small cold start admitted before a giant; two simulated ranks with
  different local stamps apply rank 0's order; a parked continuation lets a fitting waiter run, keeps
  `inflight_middle_chunks` at 0 and resumes; a continuation with 3392 of 200k tokens left runs instead of parking; a
  waiter above the cold cap does not trigger parking; parking bounded by rounds; randomized arrivals never form two
  partials or exceed the round budget; refusals; with 124 off the decision sequences equal the base `83197755` step
  by step (5 seeds × 80 rounds) and no broadcast is entered.
The 2026-09-26 repair regressions are in `tests/test_s1s2_review_optimizations.py`: actual scheduler/adder methods
with fake pools, including native rejection cleanup, same-pass fallback and two simulated ranks. See the
[S1/S2 review](../../notes/reports/review-s1s2-patches-0926.md) for counterexamples and validation boundaries.
The new repair has not established TP8 performance; the estimate's constants remain from 071.

## Size-tiered warm budget (follow-up, 2026-09-26)

`SGLANG_AX_DEADLINE_WARM_MULTI_S` (default = `SGLANG_AX_DEADLINE_WARM_S`) and `SGLANG_AX_DEADLINE_WARM_MULTI_TOKENS`
(default 8192): a warm request whose remaining prefill exceeds the token threshold, so it needs more than one
chunked round and holds the single partial-prefill lane for several rounds, is judged against the multi-round
budget; one-round warm requests keep the warm budget. Why: with one 15 s warm budget (run 130b versus 112) the
20k-70k-token warm turn starts became "rescuable" and outranked cold chain heads, and the one-round warm hits
queued behind them (chain 26->30, overall 448->512, fast 506->554), while the requests the 15 s budget actually
saved were one-round turn starts (turn 24->16). Setting WARM_S=15 with WARM_MULTI_S=5 keeps the rescue for the
one-round class and leaves the multi-round class where the 5 s budget put it. Unset, the order is unchanged.

