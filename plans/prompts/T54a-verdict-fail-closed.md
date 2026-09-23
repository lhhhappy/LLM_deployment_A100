# T54a — 8 卡单档判定改为"失败即失败"，统一评分器（Claude 派发，subagent Opus）

先读 `plans/prompts/_context-0924.md`（共享背景、赛题口径、方法规则、红线）和 `research/codex/R19_progress_and_cache_review.md` §2 的 S1/S3/S8。

## 要读的代码
`scripts/pod/jobs/dev_ladder_template.sh`、`scripts/pod/lib.sh`、`scripts/pod/verify/analyze_run.py`、`scripts/pod/verify/make_kit.sh`、`scripts/pod/qpush`、`scripts/pod/podq`、`scripts/score_formal.py`（docstring 与 API）、`scripts/ladder_search.py`（FlushGuard / verified_flush）、`scripts/test_ladder_search.py`（已有反例）、`s1-dev/run_dev.py`（flush、S1_FLUSH_URL、summary.json、退出码）。

## 交付
1. 新 `scripts/pod/verify/run_level.py`：跑**一档** dev。
   - 用与模板相同的参数调用 `run_dev.py`，`S1_FLUSH_URL` 指向严格 flush 守卫（import 复用 `scripts/ladder_search.py` 的 FlushGuard/verified_flush；import 不可行就说明原因并包一层，不要改原文件）。flush 必须 HTTP 2xx 且 JSON `success: true`，否则终止本档。
   - 记录本档起止 epoch（以及 pod 本地时间字符串）、run_dev 退出码、argv、每次 flush 的结果，写入 `DIR/level_manifest.json`。
   - 通过 run_dev 的 `summary.json` 找本次测量的 raw/run 文件，**不能取"最新文件"**。校验：req_id 集合与本次 cohort 的测量集合完全相等、无重复、`idx_in_chain` 与 cohort 一致、按 harness 分类统计错误。
   - 评分只用 `scripts/score_formal.py`（harness evaluate + 正式估算 + TPOT 门），不写私有门逻辑。
   - 写 `DIR/verdict.json`：`status`（VALID/INVALID）、`reasons`、`harness_all_pass`、`formal_est_all_pass`、`tpot_mean`、`tpot_p95`、`gates`（每门 n/p95/over/allowed_over/两种判定）、`n_rows`、`n_expected`、`errors`、`window`。
   - 退出码：0 有效且正式估算全过（含 tpot_p95）；1 有效但未过；2 测量无效；3 引擎挂了。任何缺失、空、解析失败都算 INVALID，绝不判过。空门桶如果按 harness 定义本不该空，也是 INVALID。
2. `dev_ladder_template.sh` 改为调用 run_level.py。
   - 新模式 `LEVELS="22 18"`：按顺序跑完所列各档，不因某档没过而停；遇到 INVALID 或引擎挂了才停。
   - 保留 `LADDER_UP`（首败即停）。
   - 每档输出一行 `LEVEL N=.. status=.. formal_est=.. harness=.. tpot_mean=.. tpot_p95=.. <各门>`。
   - 保留现有 PRECHECK 和冒烟门。另打印所用每个补丁的 sha256，以及实际生效的参数和环境变量。
3. `analyze_run.py` 不再自行给出判定：要么改为调用 score_formal，空或不完整输入时非零退出；要么降级为纯诊断（排队/执行拆分），并在文件头注明它不是判定。
   - 验收：在隔离副本里重跑 `evidence/T53/reproduce_review.py`，空 raw 不得再出现 ALL_PASS=true。
   - 隔离副本的做法：把脚本和 raw 拷到临时目录，`s1-dev` 和 `scripts` 用软链。不要覆盖 Codex 的 evidence 文件。
4. `lib.sh`：引擎日志用持久路径（记录真实 server.log 路径，或软链进每个 RUN_DIR）；复用引擎时打印仍在写的那份日志的路径；不再依赖静态副本 `engine_current.log`。
5. job 不再复制模板正文：
   - 把模板推进 pod（例如 `$AX/bin/scripts/pod/jobs/`，改 qpush 或 podq init）；job 文件只写变量加一行 `source`。
   - 新生成器 `scripts/pod/mkjob.py`：从 `scripts/pod/jobs/specs/*.json` 读基线规格，用 `--set` 改项，写出 job 文件，并打印、记录与基线的差异。改动超过一项（补丁集、参数、环境变量各算一项）就拒绝，除非给 `--multi "理由"`。
   - 附 S0 规格：026 精确栈，含 `drafts/120-sched-protect-chain-v2.patch`。
6. `make_kit.sh` 把 run_level.py 依赖的文件带进 kit（score_formal.py、守卫所在模块、harness 路径约定）。
   - 用 `scripts/gssh "cd /sjtu/linhang/arena/repo && scripts/pod/pexec_codex 'python3 --version'"` 查 pod 的 Python 版本。score_formal 需要 ≥3.10；如需先同步脚本到 GPU 机，只同步到 `/sjtu/linhang/arena/repo`，不要推进 pod。
7. CPU 测试 `scripts/pod/verify/test_run_level.py`（标准库 unittest；需要时用假的 run_dev 或假引擎）。以下每种都必须 INVALID、exit 2：
   - 空 raw；
   - 721/722 行；
   - 重复 ID；
   - 多 token 输出缺 tpot；
   - run_dev 非零退出；
   - flush 返回 HTTP 200 但 `success:false`（runner 被终止）。

   Golden：`evidence/T53/026_N18_raw.jsonl` 必须 VALID，且 fast_intra n=328、over=10，tpot_p95≈0.219，正式估算 FAIL（exit 1），各门数值与 harness 报告一致。run json 需要时可用 `scripts/pod/pread ls /tmp/ax/runs/026-ladder_best140/N18` 查看并 `pread cat` 取回。日志存 `evidence/T54/`。
8. 在 `scripts/pod/README.md` 加一节"Level verdicts (T54)"。只更新 `notes/dispatch.md` 里 T54 那一行的状态，其余记录由 Claude 维护。

## 约束
- 不往 pod 推送，不起引擎，不碰 pod 上正在跑的 035（它用旧模板）。
- 不改 `score_formal.py`、`ladder_search.py`、`s1-dev`；发现其中有问题写进报告。
- 改动保持最小、可读，风格与周围代码一致。

## 最后回报（≤300 字）
改了哪些文件、测试结果计数、score_formal/ladder_search 里看起来不对的地方，以及 Claude 推送新 kit 要执行的确切命令。
