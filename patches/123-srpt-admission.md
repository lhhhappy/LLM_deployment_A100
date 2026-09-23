# 123 — shortest remaining prefill first, with aging (candidate, on top of 120)

## Why
In 037 (N22) 87 chain_start requests exceeded 30 s; 60 of them needed under 64k tokens and waited a median 54 s in the
queue while one large cold request was being prefilled chunk after chunk (only one chunked request can be active). Small
heads that needed 1.4k–10k tokens waited 49–75 s through 49–72 prefill batches of 8192+ tokens (scripts/analysis/badcase.py,
evidence/L037). Under LPM, cold requests are ordered by shared-prefix length and then arrival, not by the work they need.

## What it does (`srt/managers/schedule_policy.py`)
With `SGLANG_AX_SRPT_AGING` set, the LPM sort becomes: remaining prefill tokens (prompt − matched prefix) minus
`aging × seconds waited`, ascending. Cheap cache hits still go first (little remaining work), small cold requests are admitted
before large ones, and a large request gains `aging` tokens of priority per second so it is not starved (2000 tokens/s: a
100k-token request waiting 60 s ranks ahead of a fresh 3k one). Requests held back by LPM's in-batch prefix sharing stay last.
It changes only the admission order: an already active chunked request is not preempted.

## Switches
`SGLANG_AX_SRPT_AGING` (tokens per second; unset or 0 = plain LPM).

## Evidence
CPU: `tests/test_srpt_admission.py` (5 tests on the real sort and scheduler code with fakes); all 37 scheduler tests pass.
8 cards: pending (S1 + 122 + 123 at N22, compared with S1 + 122).
