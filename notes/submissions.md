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

## 2026-09-23 提交（用户批准：A、B 用同一个镜像，打完直接提交，不先跑冒烟）
- 镜像 lh-img:0923a（15 个补丁：000,101,105,106,110,111,112,113,114,140,120,130,150,160,170v2；清单在 build/image/0923a.patches.txt）
- A：028 配置（MTP+114+v3 cap4096 interval2，eager prefill，chunk 自动）；8 卡 N18：tpot 0.0528/p95 0.082，TTFT 三门 FAIL ⇒ 预期 N10–14。
- B：026 式大块（不限块、无 interval）+ MTP+114，chunk 8192（B2，034 同时在 8 卡验证）。未测，搏一把。
- 生成器 scripts/make_submission_0923.py；提交 `DAY=0923 bash scripts/submit_official.sh A|B`。
- 镜像 digest：sha256:f791ac3e12bbcf5aca14fd2b566ec41d148bcbcf3cd8f76c034dbc6050e05d97（build 164197）
| Arm | Attempt | Image | Config | Status |
|---|---|---|---|---|
| A | 45979 | lh-img:0923a（build 164197，sha256:f791ac3e…） | MTP+114+v3（cap4096，interval2，chunk 自动） | submitted 15:18 UTC |
| B | 45980 | 同上 | MTP+114，chunk 8192，cap16384（不限块），无 interval | submitted 15:18 UTC |
- 已核对上传包 outputs/submission.json（A sha256 058ecb08…、B 6cd0e104…）。check_submission.py 解析不了新底包用注解字段定义的参数（全部报成 flag 不存在），这次用 SKIP_FLAG_CHECK=1 跳过；所有参数都在 pod 上实际跑过。
