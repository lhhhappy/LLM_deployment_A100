# 122 — adaptive decode rounds after each prefill batch (candidate, on top of 120)

## Why
At dev N22 the only failing gate of S0/S1 is tpot_p95 (035: 0.296, 036: 0.253). When requests are decoding and the queue
is empty, 120 lets a new long prefill run 16k chunks (~1.2 s each) with one decode round between chunks, so those streams
get about one token per chunk. In 035, 243 of 858 chunks of 8192+ tokens ran in exactly that state (scripts/analysis/badcase.py;
Codex R21). A fixed `--prefill-decode-interval K` spends the same K rounds after a 16k chunk and after a 500-token short hit.

## What it does (`srt/managers/scheduler.py`)
When `SGLANG_AX_TPOT_TARGET` is set, every prefill batch that runs while some request is decoding is followed by
`ceil((fixed + per_token × new_tokens) / (target − decode_step))` decode rounds (at most `max_rounds`); with no request
decoding nothing is owed, so a lone cold prefill is never delayed. It replaces 120's single decode turn; an explicit
`--prefill-decode-interval` is ignored while it is on. DP-attention runs fall back to the fixed interval.

## Switches
`SGLANG_AX_TPOT_TARGET` (s/token; unset or 0 = off), `SGLANG_AX_PREFILL_FIXED_S` (0.11), `SGLANG_AX_PREFILL_PER_TOKEN_S`
(0.000065), `SGLANG_AX_DECODE_STEP_S` (0.022), `SGLANG_AX_DECODE_ROUNDS_MAX` (64). Defaults come from 8-card TP8
measurements (F87: ~107 ms fixed + ~60 µs/token; F88 with 114: 63–65 µs/token at long context). Example: target 0.085 →
11 rounds after an 8192-token chunk, 19 after 16384, 3 after a 512-token hit.

## Evidence
CPU: `tests/test_adaptive_decode_rounds.py` (5 tests on the real scheduler code with fakes) plus the 27 tests of
`tests/test_sched_protect_chain.py` still pass. 8 cards: pending (S1 + 122 at N22).
