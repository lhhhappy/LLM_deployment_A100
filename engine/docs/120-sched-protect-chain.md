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
no DP attention. The current 180 changes the cache-tier restriction: the host tier preserves protection, while L3 storage
bypasses it. Short-hit sharing excludes requests still waiting for host restore; other unsupported modes bypass protection.

## Evidence
- CPU: `tests/test_sched_protect_chain.py` runs the real scheduler and `PrefillAdder` code on fakes (trees come from git via
  `scripts/engine/tree.py`; 32 tests pass, including the HiCache tier and mechanism-report tests).
- 8 cards: N6 intra queue p95 6.41 → 0.36 s with the always-cap variant (013). With this behaviour: N10 TTFT gates pass but
  tpot_p95 0.13 (025b); S0 at N18 (026) passes the TTFT gates, tpot_p95 0.219; S0 at N22 (035) passes 10 of 11 gates, tpot_p95
  0.296 (F96). Remaining problem: during heavy prefill a stream gets one decode step per 16k chunk (~1.3 s).
- The variant that also caps while requests decode is patch 121.

## Opt-in admission diagnostics (2026-09-24)

`SGLANG_AX_ADMISSION_TRACE=1` enables CPU-only observations on TP0; default `0`.
The startup mechanism line reports `120_trace=on|off`; a diagnostic job should
include the expected token in `G_EXPECT`. This is a follow-up to 120, not a new
scheduling policy. No job or frozen 069/071 configuration is changed by this code.

It distinguishes the observed branches for `partial_host_restore`,
`partial_no_device_prefix`, `partial_long_tail`, `partial_token_budget`,
`kv_budget`, `kv_budget_after_lock`, decode cadence, request slots, and latched
`batch_is_full`. Unvisited queue tails are explicitly labeled `unscanned_after_*`;
their own resource feasibility has not been tested. Other adder outcomes retain
generic labels. KDA-specific pool shortage and H2D completion timing are not yet
separately instrumented.

There is one JSON log per newly admitted request, plus at most one waiting
snapshot per 30 seconds (first 32 queue entries, with truncation explicitly
marked). A lifetime 8 MiB byte budget per TP0 process includes 128 bytes of prefix
allowance per line; on exhaustion it emits one `budget_exhausted` record and stops
observing/logging. No per-step logs, CUDA synchronization, collectives, or GPU buffers are
added. Counters live on requests and are cleared at admission, so retractions get
new episodes and aborts do not leak collector-owned references. Diagnostic CPU
overhead and logging cost still need measurement; leave this off for official
submissions and for clean performance comparisons.

Counter values are numbers of observed decisions, **not seconds or causal delay
shares**. `observed_wait_s` begins at the first diagnostic observation, not request
receipt. Cache fields describe the latest available prefix match, not necessarily
residency at arrival. `admit` means selected for a batch, not H2D or GPU execution
completed. Timestamp/token reporting used for scoring is untouched.

CPU validation runs real scheduler/adder ASTs: trace off/on preserve frozen
759a6eb decisions and budgets, with both fixed cadence and 122; tests distinguish
host rejection, KV rejection and unvisited queue tails, enforce TP0-only logging,
bounded snapshots, and retraction-state reset. The log parser checks cumulative
snapshot accounting and retains unknowns instead of replacing missing data by
zero. Byte-budget tests ensure repeated admissions cannot grow this diagnostic
stream indefinitely and parsing explicitly marks observations cut short by the cap.

```sh
python3 -B -m unittest discover -s tests -p 'test_admission_trace*.py'
python3 -B scripts/analysis/admission_trace.py server.log --raw raw.jsonl --json admission.json --csv admission.csv
```

The source changes add no new GPU/host cache allocation. CPU per-waiting-request
state consists of one timestamp, a small fixed-vocabulary counter dictionary and
one last-reason field. Production performance or N@SLO improvement is not claimed.
