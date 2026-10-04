# 131 — chain-risk decode interval (candidate, on top of 124)

## Why
Same-ID measurements across runs 130ez1/ez7/ez8/ez9/eze2/ezf/ezh/ezi (v3, N34, steady state): the 251,916-token
chain start `scimaster:canon:YM1Suj4E5fA4Bx-ZGQj4c:llm:1` (waited 0.4 s in every run, same cache state) took
28.0–29.5 s after batch entry with MTP and `--prefill-decode-interval 2`, 24.8 s with interval 1, 23.4–23.7 s
without MTP, 21.6 s in the run whose relief guard never latched. These runs vary multiple settings; they motivate
testing interleaved decode rounds but do not isolate their contribution. The chain_start gate counts requests over 30 s, so a giant that would finish at
28–33 s is decided by those rounds. In the opening the lane is already prefill-only (125 relief, interval 0);
this mechanism covers the steady state, where 125 rarely engages.

## What it does
Off unless `SGLANG_AX_CHAIN_RISK_INTERVAL` is set. Requires 124 (`SGLANG_AX_DEADLINE_TIERS=1`) for the cold
budget and the cost model; refuses to start otherwise. Every time an extend batch is armed
(`Scheduler._arm_prefill_decode_interval`), request-plane rank 0 checks each request using
`ax_deadline.chain_risk_interval`, then broadcasts the final interval, including when no request is at risk.
Non-root ranks do not read their local clock or classify the requests for this decision:

- cold class (124's `deadline_cold`), and at least `SGLANG_AX_CHAIN_RISK_MIN_REMAINING` (32768) tokens of prefill
  left after the current chunk;
- optimistic projection = waited so far + 124's arrival offset + the unexecuted current batch's cost + 124's
  service estimate of the remaining tokens alone. The current batch pays one fixed cost plus all its new tokens
  (including riders); those tokens have already been subtracted from each request's remaining work;
- `SGLANG_AX_CHAIN_RISK_CHUNK=0` (default) uses the batch's final planned cold cap after 125/126/READY reservation,
  bounded by the input/chunk budgets. A positive value explicitly overrides this model chunk. The estimate uses
  the plan, not a role-aligned tiny last fragment. It estimates future rounds with today's cap; it cannot predict
  future arrivals or reservation changes. 124's earlier ranking still uses its planning-time cap;
- at risk when the projection lies within `SGLANG_AX_CHAIN_RISK_MARGIN_S` (8 s) of the budget on either side.

For an at-risk request the decode rounds armed after its chunk become `SGLANG_AX_CHAIN_RISK_INTERVAL` (0 = none)
instead of the configured interval (or 125's relaxed interval, whichever is lower). Comfortable requests keep the
configured interval; requests estimated past budget + margin do too, to avoid spending decode time on a rescue
the model considers unlikely. This is a heuristic, not proof a request cannot pass. Re-evaluated at every chunk, so the
protection ends as soon as the request is comfortable or lost.

Mechanism report: `131=on 131_sync=rank0 131_chunk=auto` (or the explicit positive chunk).
When off: `131=off:SGLANG_AX_CHAIN_RISK_INTERVAL_unset 131_sync=off 131_chunk=off`.
Log: `[ax-131] risk rounds=N` on rank 0, every 30 s while the risk predicate matches;
125 may already have selected an even lower interval. The OFF path adds no collective.

## Cost
While engaged, the running decoders get fewer or no tokens for the remaining chunks of the at-risk request (at
most ~20 s for a 250k head). Their per-request mean TPOT rises; the tpot_p95 gate (0.10 s) tolerates this for
long outputs but short-output requests fully inside such a window can cross it. Judge every run on TPOT>0.10
counts next to the chain-start TTFT distribution. Interval 1 is the conservative setting, 0 the aggressive one.

## Verification
CPU: `tests/test_ax_deadline.py::ChainRisk` and `tests/test_chain_risk_scheduler.py` (actual scheduler methods,
rank-skew boundaries, OFF path, current-batch time, auto/explicit cap, 125 minimum, final 126 cap).
`scripts/analysis/verify_chain_risk_gloo.py` tests the production arming/transport through eight real CPU Gloo
processes. This verifies control flow and broadcast, not GPU model execution or TP8 latency.
TP8: v4 dataset, N26, 40-minute windows, S5b configuration; arms 131 off / interval 1 / interval 0; judge the
steady-state chain-start TTFT bins (15–20 s, 20–30 s, >30 s), giants' time after batch entry, TPOT>0.10 counts and
the other gates on the same request IDs.
