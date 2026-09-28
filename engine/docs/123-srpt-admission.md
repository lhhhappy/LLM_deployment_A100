# 123 — shortest remaining prefill first, with aging (candidate, on top of 120)

## Why
In 037 (N22) 87 chain_start requests exceeded 30 s; 60 of them needed under 64k tokens and waited a median 54 s in the
queue while one large cold request was being prefilled chunk after chunk (only one chunked request can be active). Small
heads that needed 1.4k–10k tokens waited 49–75 s through 49–72 prefill batches of 8192+ tokens (scripts/analysis/badcase.py,
evidence/L037). Under LPM, cold requests are ordered by shared-prefix length and then arrival, not by the work they need.

## What it does (`srt/managers/schedule_policy.py`)
With `SGLANG_AX_SRPT_AGING` set, the LPM sort becomes: remaining prefill tokens (prompt + already generated output −
matched prefix) minus `aging × seconds waited`, ascending. At equal age, smaller remaining work ranks first;
a large request gains `aging` tokens of priority per second (2000 tokens/s: a 100k-token request waiting 60 s ranks
ahead of a fresh 3k one). Requests held back by LPM's in-batch prefix sharing stay last. Equal scores retain queue order.

This changes admission order only. It does not preempt an active chunk, override KV/request-slot limits, or make a
host hit or a short cold request eligible beside an active partial. Aging therefore does not guarantee bounded waiting.
The matched prefix includes device and host hits; the score excludes host restoration time, context-dependent kernel
cost and KDA pressure. It estimates remaining token work, not remaining execution seconds. The base FCFS fallback
for queues above 128 requests remains in effect.

## Switches
`SGLANG_AX_SRPT_AGING` (tokens per second; unset or 0 = plain LPM).

## Evidence
CPU: `tests/test_srpt_admission.py` now uses the working engine rather than the historical 123 commit. All 16 tests pass:
real policy dispatch, host/device ordering, aging, stable ties, retracted output, one-partial protection, slot/KV rejection,
and resource invariants with 122 on/off. Cache matching results, pools, clocks and forwards are fakes; this does not
validate DMA, GPU state restoration, MTP numerics, TP8 agreement or performance.

8 cards: historical 037c → 037d (old S1 + 122, then only 123 aging=2000) reduced chain failures 33 → 23, with CP
allowance 22. TPOT p95 was .1424 → .1466, so both full N22 runs failed. This is a useful lead, not evidence for the
current host64/MTP baseline.
See [experiment records](../../notes/experiments.md) and [R26](../../research/codex/R26_n30_slo_levers.md).

**2026-09-28 update: superseded, not "pending".** 123's single-request shortest-remaining-first idea was folded into
124 (deadline-tiered admission, `SGLANG_AX_DEADLINE_TIERS`) as one tier of its order, then extended with family-aware
ranking on top of that (128p, `SGLANG_AX_PREFIX_PRODUCER`) once same-pack cold requests turned out to need
group-level rather than per-request ordering (job 104, 2026-09-26: 124 alone reached 13/27 chain misses in the first
minute against a shortest-remaining-first estimate of 7, because sibling requests share a prefix the single-request
estimate cannot see). No further standalone 123 validation is
planned; see the "124" and "128p" rows in [engine/README.md](../README.md) and [program-n30-v3.md](../../notes/program-n30-v3.md).
