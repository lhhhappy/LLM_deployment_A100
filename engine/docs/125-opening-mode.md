# 125 — opening mode: spend TPOT on cold prefill while the cold backlog is large (candidate, on top of 120)

## Why
Chain starts are cold (run 071 at N30: the 33 chain starts of the first minute needed 1.88M prompt tokens, 9.3% cached)
and the chain gate is lost when they pile up: 25 of 071's 30 chain misses arrived in the first 10 minutes, while the
rest of the run had almost no request above 0.10 s/token (0 of 5601). Opening probes at N26 (paired with 074, same
request IDs) show that giving cold prefill more of the machine buys chain starts at a TPOT cost:
cold cap 8192 (075) chain 21→17, requests above 0.10 s/token 1→55; interval 1 (077) 21→18, 1→61; cold cap 6144
(078) 21→16, 1→33. Applied all the time, that cost is paid even when there is no cold backlog. The tpot_p95 gate
allows about 5% of a level above 0.10 s/token (about 89 of 1788 if an official level has 1788 requests; that count
is inferred, not confirmed).

## What it does
Off unless `SGLANG_AX_BACKLOG_RELIEF=1`. Requires 120's protection; refuses with 122 (which replaces both the cold
cap and the fixed interval). It must change something: `SGLANG_AX_BACKLOG_COLD_CAP` (0 keeps 120's cap) and/or
`SGLANG_AX_BACKLOG_INTERVAL` below `--prefill-decode-interval` (default: the configured interval); otherwise it
refuses at startup.

- **Trigger** (`ax_deadline.BacklogState.decide`): cold backlog = remaining prefill of waiting requests whose match is
  under half the prompt, plus what the running continuation has left (its sequence minus the prefix computed so far,
  as in 124). Converted to seconds at the recent prefill rate (`BacklogState.note_batch`: EWMA over prefill batches of
  a batch's tokens per wall second from its scheduling to the next prefill's, i.e. its execution and the decode
  rounds after it). It is sampled on every prefill, whatever `--prefill-decode-interval` is. An idle stretch between
  two prefills counts in the sample (8192 tokens then 60 s idle is one ~136 tok/s sample, about 20% off the EWMA),
  so after idle periods relief can start early; a flush opens a fresh interval. Enter above `high_s` (30 s), leave
  below `low_s` (10 s). It follows the load, not the clock, so it also engages when cold chain starts pile up later.
- **Effects while relieved**: cold chunks capped at `COLD_CAP` instead of 120's cap (8192 = the whole default
  round, which leaves no room for short hits beside a continuation; 6144 leaves 2048), and `INTERVAL` decode rounds
  armed after each prefill.
- **Guard**: relief stops and stays off until the next `/flush_cache` once `max_slow` (40) requests have run above
  `gate` (0.09 s/token). Requests still decoding count by their TPOT so far, checked every prefill round; one
  under the gate now can still end above it, and a request finishing inside decode-only rounds is checked only at
  the last prefill round it was running, so the gate is set below the harness's 0.10. An optional ratio guard
  (`MAX_SLOW_RATIO`, off at 1.0) exists for levels of unknown size; it latches like the count, so a ratio that falls
  back under the limit as more requests are seen does not re-enable relief. The platform calls `/flush_cache` before
  every level, which resets the guard, the relief state and the open rate interval (the learned rate is kept).
- **TP consistency**: the decision uses rank-local clocks, so it is computed on request-plane rank 0 inside 124's
  plan (`Scheduler._ax_admission_plan`) and broadcast (`_ax_rank0_decide`, from 123's fix). The cap uses the
  previous round's broadcast decision, because this round's plan needs the cap as its round budget; the interval
  uses this round's decision.

Logs: `[ax] 124 … | 125 <config>` at start, `[ax-125] relief on|off: cold backlog … tokens at … tok/s, slow …/…`
on each change, `[ax-124/125] parks=… relief_rounds=…` every 30 s on rank 0; the mechanism line shows `125=on`.

## Switches
`SGLANG_AX_BACKLOG_RELIEF` (0/1); `SGLANG_AX_BACKLOG_{COLD_CAP,INTERVAL}`; trigger `SGLANG_AX_BACKLOG_{HIGH_S,LOW_S,
RATE_WEIGHT}`; guard `SGLANG_AX_BACKLOG_{MAX_SLOW,GATE,MAX_SLOW_RATIO,MIN_SEEN}`.

## Evidence
CPU, real scheduler code with the project's CPU fakes (`tests/test_sched_protect_chain.py` harness):
- `tests/test_ax_deadline.py`: hysteresis, the guard (each request counted once, stays counted, ratio latched until
  the flush), the rate (smoothing, no sample across a flush), config refusals (no effect, interval above the
  configured one);
- `tests/test_ax_admission_scheduler.py`: a relieved prefill arms the relaxed interval, a small backlog keeps the
  configured one; the backlog counts what a continuation has left (3392 of 200k); the rate charges an interval to the
  prefill that ran in it (8192 tokens then 32, 0.8 s apart: 10240 tok/s, not 40), and is sampled with interval 0
  (cold-cap-only relief); the opening cap applies from the
  round after relief is decided (4096, then 8192); 125 with 122 refuses; 125 without an effect refuses; with 124/125
  off the decision sequences equal the base step by step.
Not validated: TP8, the rate estimate under real load, whether the guard's count tracks the harness's TPOT, and any
performance effect.
