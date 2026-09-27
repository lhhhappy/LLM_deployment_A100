# 132 — chain-first admission order (candidate, a switch inside 124)

## Why
Online the chain_start gate (30 s) is the only tight gate (46676/46677/46758 at N26: chain p95 37-41 s) while fast,
overall and turn keep 2-5x margin. 124 orders rescuable requests by remaining work only, so a stream of short warm
hits keeps jumping a cold chain start that could still make 30 s; each jump costs the head 0.2-0.6 s and, under
contention, the 30 s line. The user's rule: between a chain start and a warm request, protect the chain start.

## What it does
`SGLANG_AX_DEADLINE_CHAIN_FIRST=1` (needs 124). In `ax_deadline.tier_order` and in 128p's ranking, tier 1
(rescuable) is split: cold requests (124's `deadline_cold`) first, then warm ones, each group shortest remaining
work first. Starved requests stay first, hopeless ones stay behind, held-back ones stay last. Off = 124's order.

Startup rejects `SGLANG_AX_DEADLINE_CHAIN_FIRST=1` without 124, rather than silently ignoring it. The effective
mechanism receipt includes `132=on` or `132=off:SGLANG_AX_DEADLINE_CHAIN_FIRST_unset`; candidate jobs require
`132=on` in `G_EXPECT`. This reports the resolved deadline config, not just a requested environment variable.

## Cost
Warm requests wait behind cold heads: fast/overall/turn misses rise locally; online they have margin. Judge every
run on all gates together.

## Verification
CPU: tests/test_ax_deadline.py::ChainFirst. TP8: v5g N26 windows against the S1 + no-MTP arm (same request IDs):
opening misses, steady chain TTFT bins, giants after entry, fast/overall/turn counts, TPOT.
