# Official submission results

Official submissions: 45734 (A) and 45735 (B) on 2026-09-22, see below. Daemon queue results are appended below; full status history and all four TTFT p95 values are in `data/submissions.json`.

| Item | Attempt | Status | AIME26 | GPQA | Gate | N@SLO | TPOT mean | TPOT p95 | Completed UTC |
|---|---|---|---|---|---|---|---|---|---|

## 2026-09-22 official submissions (manual, via scripts/submit_official.sh; CLI 0.1.39 trace bug workaround)
| Arm | Attempt | Job | Image | Config | Status |
|---|---|---|---|---|---|
| A (basic) | 45734 | 24008 | lh-img:0922e@sha256:eb0f2fe4… | D0 + lpm | queued 11:16 UTC |
| B (aggressive) | 45735 | 24009 | lh-img:0922f@sha256:f7725115… | D0 + D1(base-rebased, env on) + lpm | queued 11:19 UTC |
Results: check with `playground` attempt status for 45734 / 45735 (~18 h). Daily quota 2/2 used for 2026-09-22.

## 2026-09-23 resubmission (image format fix, F55)
| Arm | Attempt | Job | Image | Config | Status |
|---|---|---|---|---|---|
| A | 45766 | 24040 | lh-img:0922e | 000 + lpm | queued |
| B | 45767 | 24041 | lh-img:0922f | 000 + 101 (D1 on) + lpm | queued |
45734/45735 failed at deploy (image name format), not scored, not charged.
