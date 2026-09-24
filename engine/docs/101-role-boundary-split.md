# 101 — KDA state at the last role boundary

One patch (T57 folded the former 101 and its crash fix 105 into this file; source tree byte-identical, evidence/T57).

## Why
Agent prompts end with a user turn (often a `<system-reminder>`) that the next request replaces. The token LCP with the next
request therefore ends at the last `<|user|>` (154827) or `<|observation|>` (154829) token, not at the prompt end (F3). The KDA
linear-attention layers can only resume from a saved state, so without a state at that boundary the next request recomputes
everything after the previous saved point.

## What it does (`srt/managers/schedule_policy.py`)
- At admission, if the prompt's new part contains a role-boundary token, the extend is split at the last boundary so that the
  chunk ends there and the scheduler caches the KDA state at that position; the final chunk is split the same way.
- At most one partial prefill per round while the split is on (a second truncation with a continuing chunked request made two
  partials and crashed the scheduler, F62).
- With 140 enabled (`SGLANG_AX_KDA_DUAL_SNAPSHOT=1`), 140 exports the boundary state inside the same prefill instead of splitting;
  101's boundary detection is still used.

## Switches
`SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=154827,154829` turns it on (unset = base behaviour);
`SGLANG_ARENA_ROLE_BOUNDARY_SCAN_WINDOW` (default 32768) limits how far back the boundary is searched. 160 clears the role IDs
when MTP is on.

## Evidence
Stand-in model (dev box): reminder-heavy fast requests, missed tokens p95 8135 → 2439, no regressions (E2b). 8 cards: every run
since 013 includes it; no scheduler crash after the one-partial guard (013/016).
