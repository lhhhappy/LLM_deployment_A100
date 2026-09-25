# 126 — cold chunk sized by the waiting short hits' demand (candidate, on top of 120)

## Why
Opening probes at N26 (paired with 074, same 350 request IDs, `evidence/opening-balanced-20260925/`): raising
120's fixed cold cap from 4096 to 6144 (078) cut chain-start misses 21→16 (5 fixed, 0 new). Its cost fell almost
entirely on fast requests that need 2049–4096 new tokens (misses 5→23), while those needing at most 2048 went
9→1: with a fixed 6144 of the 8192-token round only 2048 is left, so a 2–4k hit cannot join a batch beside a
cold continuation and waits the whole continuation. A fixed 8192 (075) leaves no room at all (fast 23→61).

## What it does
Off unless `SGLANG_AX_SCHED_COLD_CAP_MAX` > 0. Requires 120's protection; refuses with 122 (which sizes the same
cap from its TPOT pacing and reservation) and a maximum below `SGLANG_AX_SCHED_COLD_CAP`.
Each prefill round (`Scheduler._ax_demand_limits`), where 120 would cap a cold chunk, the cap becomes
`clamp(budget − reserve, COLD_CAP, COLD_CAP_MAX)` on the checkpoint grid. `reserve`
(`Scheduler._ax_short_hit_reserve`, shared with 122) is the page-rounded new tokens of the waiting complete short
hits (device prefix hit, no host load-back, at most `SHORT_TOKENS` new: exactly the requests 120 lets share a
batch with a partial), in queue order, counting only hits that fit beside 120's cap: a hit that cannot fit beside
the floor can never join and would only idle part of the round. A hit reserved beside a continuation and still
waiting the next round was refused for another reason (request slots, KV, mamba slots); it is not reserved again
until that continuation ends (122's rule). With no hit waiting the cold chunk takes the maximum, also while
requests decode (121's cap is then the maximum, which is 078's TPOT cost); the cold request never gets less than
120's cap. When 120 would not cap (engine otherwise idle) nothing changes.
The inputs are the queue after `calc_priority`'s prefix match, identical on every TP rank, so no broadcast is
needed. The mechanism line shows `126=on`; startup logs `[ax] 126 on: …`.

## Switches
`SGLANG_AX_SCHED_COLD_CAP_MAX` (0 = off). 120's `SGLANG_AX_SCHED_COLD_CAP` is the floor and
`SGLANG_AX_SCHED_SHORT_TOKENS` defines a short hit, as before.

## Evidence
CPU, real `get_next_batch_to_run` / `PrefillAdder` with the fakes of `tests/test_sched_protect_chain.py`
(`tests/test_demand_cap.py`): with a 3000-token hit waiting, a fixed 6144 cap runs the continuation alone while
126 runs the continuation (5120 at the fakes' 256 grid) and the hit in one batch; with no hit waiting the
continuation takes the maximum; of two 3000-token hits only the one that fits beside the floor is reserved
(5120 + one hit); a 5000-token hit that can never join leaves the maximum; a hit refused for a request slot is
reserved once, then the continuation runs at the maximum; a new cold request's first chunk leaves room for a
waiting hit; refusals; with 126 off the decision sequences equal the base 37e90023 step by step over random
arrivals; 122's suite is unchanged by the shared reserve helper.
Not validated: TP8 and any performance effect (TPOT while the cold chunk is at the maximum is 078's cost).
