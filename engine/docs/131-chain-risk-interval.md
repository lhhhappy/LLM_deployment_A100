# 131 — chain-risk decode interval (candidate, on top of 124)

## Why
Same-ID measurements across runs 130ez1/ez7/ez8/ez9/eze2/ezf/ezh/ezi (v3, N34, steady state): the 251,916-token
chain start `scimaster:canon:YM1Suj4E5fA4Bx-ZGQj4c:llm:1` (waited 0.4 s in every run, same cache state) took
28.0–29.5 s after batch entry with MTP and `--prefill-decode-interval 2`, 24.8 s with interval 1, 23.4–23.7 s
without MTP, 21.6 s in the run whose relief guard never latched. The difference is only the decode rounds
interleaved between its chunks. The chain_start gate counts requests over 30 s, so a giant that would finish at
28–33 s is decided by those rounds. In the opening the lane is already prefill-only (125 relief, interval 0);
this mechanism covers the steady state, where 125 rarely engages.

## What it does
Off unless `SGLANG_AX_CHAIN_RISK_INTERVAL` is set. Requires 124 (`SGLANG_AX_DEADLINE_TIERS=1`) for the cold
budget and the cost model; refuses to start otherwise. Every time an extend batch is armed
(`Scheduler._arm_prefill_decode_interval`), each request in it is checked by `ax_deadline.chain_risk_interval`:

- cold class (124's `deadline_cold`), and at least `SGLANG_AX_CHAIN_RISK_MIN_REMAINING` (32768) tokens of prefill
  left after the current chunk;
- optimistic projection = waited so far + 124's arrival offset + 124's service estimate of the remaining tokens
  alone (`SGLANG_AX_CHAIN_RISK_CHUNK`, 8192, as the chunk);
- at risk when the projection lies within `SGLANG_AX_CHAIN_RISK_MARGIN_S` (8 s) of the budget on either side.

For an at-risk request the decode rounds armed after its chunk become `SGLANG_AX_CHAIN_RISK_INTERVAL` (0 = none)
instead of the configured interval (or 125's relaxed interval, whichever is lower). Comfortable requests keep the
configured interval; hopeless ones (projection past budget + margin) too, because interrupting a request that
will miss anyway costs nothing at the gate but would cost the decoders TPOT. Re-evaluated at every chunk, so the
protection ends as soon as the request is comfortable or lost.

Mechanism report token: `131=on` / `131=off:SGLANG_AX_CHAIN_RISK_INTERVAL_unset`. Log: `[ax-131] risk rounds=N`
every 30 s while engaged.

## Cost
While engaged, the running decoders get fewer or no tokens for the remaining chunks of the at-risk request (at
most ~20 s for a 250k head). Their per-request mean TPOT rises; the tpot_p95 gate (0.10 s) tolerates this for
long outputs but short-output requests fully inside such a window can cross it. Judge every run on TPOT>0.10
counts next to the chain-start TTFT distribution. Interval 1 is the conservative setting, 0 the aggressive one.

## Verification
CPU: `tests/test_ax_deadline.py::ChainRisk` (at-risk, comfortable, hopeless, warm, config bounds).
TP8: v4 dataset, N26, 40-minute windows, S5b configuration; arms 131 off / interval 1 / interval 0; judge the
steady-state chain-start TTFT bins (15–20 s, 20–30 s, >30 s), giants' time after batch entry, TPOT>0.10 counts and
the other gates on the same request IDs.
