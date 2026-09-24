# 120 — scheduler: protect decoding and short requests from long cold prefills

## Why
In the base scheduler a long cold prefill runs chunk after chunk while nothing else happens: prefill has priority over decode,
and a continuing chunked request takes the whole chunk budget (`scheduler.py:3476`, `schedule_policy.py:810`). Short
cache-hit requests in the middle of a chain queue behind it (fast_intra) and running streams stop producing tokens.

## What it does (`srt/managers/scheduler.py`, `srt/managers/schedule_policy.py`)
1. **Decode turn after a prefill**: if the previous round was a prefill and live decoders exist, the next round is a decode
   round. An explicit `--prefill-decode-interval K` (base feature: K decode rounds after each extend) replaces this turn.
2. **Continuation budget**: a continuing chunked request still enters first with the base KV/state accounting. While other
   requests are waiting, its chunk is capped at `COLD_CAP` and the rest of the batch budget goes, in LPM order, to short
   requests that are full prefix hits with at most `SHORT_TOKENS` new tokens. When nothing is waiting, it takes the full budget.
3. **First admission of a cold or long request**: chunk capped at `COLD_CAP` only when other requests are waiting.
4. **Queue heads that don't fit** while a partial exists (long, cold, or too big) are skipped with the base rejection cleanup,
   and the scan continues; LPM order is unchanged. At most one partial prefill per batch.
Model computation, token counts, timestamps, flush and outputs are untouched.

## Switches (read once at engine start)
`SGLANG_AX_SCHED_PROTECT` (1; 0 = base behaviour), `SGLANG_AX_SCHED_COLD_CAP` (2048; S0 uses 8192),
`SGLANG_AX_SCHED_SHORT_TOKENS` (4096; S0 uses 8192). Supported with plain TP, LPM, page 64, chunked prefill, no mixed chunk,
no HiCache, no DP attention; other modes bypass the protection.

## Evidence
- CPU: `tests/test_sched_protect_chain.py` runs the real scheduler and `PrefillAdder` code on fakes (27 tests pass; build the trees
  with `scripts/patch_stack.py` as its header says).
- 8 cards: N6 intra queue p95 6.41 → 0.36 s with the always-cap variant (013). With this behaviour: N10 TTFT gates pass but
  tpot_p95 0.13 (025b); S0 at N18 (026) passes the TTFT gates, tpot_p95 0.219; S0 at N22 (035) passes 10 of 11 gates, tpot_p95
  0.296 (F96). Remaining problem: during heavy prefill a stream gets one decode step per 16k chunk (~1.3 s).
- The variant that also caps while requests decode is patch 121.
