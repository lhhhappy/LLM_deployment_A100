# 133 — decode rounds from exact TPOT deadlines (candidate, default off)

## Why
The decode gate is `tpot_p95 <= 0.10 s/token` with no statistical allowance, judged per request as
`(last SSE - first SSE) / (n - 1)`, and the server knows `n` exactly (`max_new_tokens` + `ignore_eos`).
Today the prefill lane yields to decode on a fixed cadence (`--prefill-decode-interval 2`, 125's
relief to 0 in the opening, 131's risk interval 1), which protects nobody in particular: short
outputs (public p10 94 tokens) are pushed over 0.10 by one 16k prefill batch (1.1 s) while long
outputs (p90 554) could absorb 39 s of prefill. Without MTP the pod runs 4% of requests over 0.10
at N26 (ezn3/ezn5) — next to the 5% line — while chain starts wait for decode rounds they do not need.
The experts' advice (notes/reports/expert-replies-0928.md): allocate decode by each decoder's
deadline and spend the 5% allowance deliberately for chain starts.

## What it does
`SGLANG_AX_DECODE_BUDGET=1`. After every extend batch the scheduler (rank 0, broadcast through the
128p/131 rank-0 decision channel) computes for each decoding request
`slack = first_output_time + gate*margin*(n-1) - now - remaining_tokens*step_s` and asks whether the
next planned prefill batch (124's cost model for the planned chunk, or `prefill_s`) fits into the
smallest slack of the protected decoders (slack >= 0, not yet given up):
- fits: `interval 0`, prefill may run at once;
- does not fit: arm decode rounds until the tightest decoder finishes (capped at `max_rounds`);
- a rescuable cold request (124) waits with less slack than those rounds cost, and the given-up
  count stays under `spend_fraction * 5%` of the requests that have started decoding: prefill runs
  now and the decoders it pushes over the line are marked spent (never protected again).
Decoders whose slack is already negative are not protected. The first output time is stamped in
`process_batch_result` on every rank (rank-local clock; only rank 0's values decide).

Knobs: `SGLANG_AX_DECODE_BUDGET_GATE_S` 0.10, `_MARGIN` 0.9, `_STEP_S` 0.035, `_MAX_ROUNDS` 64,
`_SPEND` 0.6, `_PREFILL_S` 1.1 (fallback when 124 is off). Exclusive with 122. Report token `133`.

## Cost and risk
CPU only. Risk: the step estimate and the stamp lag under overlap (one step) make deadlines slightly
optimistic; `margin` 0.9 covers 10 ms/token. Chain starts get the lane whenever decoders can absorb
it, so fast/overall may move; judged chain first.

## Verification
CPU: tests/test_ax_decode_budget.py (planner), the existing scheduler suites unchanged.
TP8: rot150/N30 40-minute window on the upload configuration (eznb) plus this switch, same IDs:
chain (opening/steady, by head kind) first; TPOT>0.10 count must stay inside the allowance; spent
count and `[ax-133]` lines in server.log.
