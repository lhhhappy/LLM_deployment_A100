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
1. requests waiting longer than `max_wait_s` (120 s), oldest first — starvation bound;
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

**Parking** (`ax_deadline.should_park`): the running continuation yields one round when the head of the order can
still make it, fits entirely in the round (the cold cap, and the KV room) and the continuation has more than
`park_min_remaining` (64k) tokens left or cannot make it any more. A waiter that would need truncation cannot run
beside a continuation (protection refuses a second partial), so it never triggers parking. At most
`park_max_rounds` (8) rounds and `park_max_s` (2 s) in a row. The parked continuation keeps its KV and state; it
computes nothing, so `extend_range` still ends at the cached prefix, the next round skips the stash, and 120's
batch accounting counts only continuations that run. It is not marked as the adder's continuation, so cold waiters
take the normal complete-fit admission path.
Note: under 120 a new cold request is cut to the cold cap even when the round has room, so parking can only rescue
waiters up to the cap (4096 in 0925a); ordering is the main effect for larger chain starts.

**TP consistency**: arrival stamps and clocks are rank-local and a 1 ms difference can flip an order
(`evidence/opening-q123-20260925/`). The whole plan (order as RIDs, park flag) is computed on request-plane rank 0
(`Scheduler._ax_admission_plan`) and broadcast by `Scheduler._ax_rank0_decide` (the helper introduced with 123's
fix); every rank applies it and refuses if its queue's RIDs differ. It is entered only past the prefill
early-returns, whose conditions depend on replicated state. Parked-round counts live on the request and change only
from the broadcast decision.

Logs: `[ax] 124 <config>` at start, `[ax-124/125] parks=… relief_rounds=…` every 30 s on rank 0; the mechanism line
shows `124=on`.

## Switches
`SGLANG_AX_DEADLINE_TIERS` (0/1); budgets `SGLANG_AX_DEADLINE_{COLD_S,WARM_S,FAST_S,FAST_TOKENS,COLD_HIT_RATIO}`;
cost `SGLANG_AX_DEADLINE_{FIXED_S,PER_TOKEN_S,LOAD,ARRIVAL_OFFSET_S}`; `SGLANG_AX_DEADLINE_MAX_WAIT_S`;
parking `SGLANG_AX_PARK_{MIN_REMAINING,MAX_ROUNDS,MAX_S}` (`MAX_ROUNDS=0` turns parking off).

## Evidence
CPU, real scheduler code (`get_next_batch_to_run`, `PrefillAdder`, 120/121/123 paths) with the project's CPU
fakes for pools, cache tree and forwards (`tests/test_sched_protect_chain.py` harness):
- `tests/test_ax_deadline.py`: estimates, budgets, tier order (starvation bound, stability, cache hits), parking
  conditions (fit, KV room, hopeless waiter, short continuation, round and wall-clock bounds), config refusals;
- `tests/test_ax_admission_scheduler.py`: a small cold start admitted before a giant; two simulated ranks with
  different local stamps apply rank 0's order; a parked continuation lets a fitting waiter run, keeps
  `inflight_middle_chunks` at 0 and resumes; a waiter above the cold cap does not trigger parking; parking bounded by
  rounds; randomized arrivals never form two partials or exceed the round budget; refusals; with 124/125 off the
  decision sequences equal the base `83197755` step by step (5 seeds × 80 rounds) and no broadcast is entered.
Not validated: real multi-process broadcast cost, TP8, and any performance effect; the estimate's constants are
from 071.
