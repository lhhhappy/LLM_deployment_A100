# 121 — scheduler: also cap cold chunks while requests are decoding (candidate, on top of 120)

## What it does (`srt/managers/schedule_policy.py`, 2 conditions)
120 caps a cold continuation, and the first chunk of a cold request, only while other requests wait. 121 also applies the cap
while any request is decoding, so a long uncapped chunk cannot stall running streams; chunks are uncapped only when the engine
is otherwise idle. Applying the patch turns it on; there is no switch.

## Status
- Motivation: 120 alone gave tpot_p95 0.13 at N10 (025b).
- Never tested as a single change. It was part of the multi-change runs 027 (cap 2048 + interval 3), 028/028b/028c/034 (with MTP,
  114, other chunk sizes) and of image 0923a (attempts 45979/45980). Those runs cannot separate its effect.
- Known cost: small chunks pay a ~107 ms fixed overhead each (F87), so capping during decode lowers prefill throughput.
