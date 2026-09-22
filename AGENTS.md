# AGENTS.md — 入口（Codex 会自动读取；Claude 见 CLAUDE.md）

本仓库：Agentic Science Challenge 推理服务部署赛（GLM-5.3-Flash @ 8×A100，比 N@SLO → tpot_mean）。
这里只做导航。正式规则在 `rule.md`，当前状态在 `README.md`。

## 每轮开始先读
- `rule.md`：协作规则、目录约定、**红线**
- `README.md`：当前状态与阻塞项
- `notes/dispatch.md`：你被派的任务（T 编号）

## 动手前按需读
- `plans/active/*.md`：正在执行的计划（目标、范围、验证方式、进度）
- `tests/TIERS.md`：**三级验证体系**（L1 2 卡自测 → L2 提测（Trisol 8 卡）→ L3 正式提交）：每一级能确认什么、晋级条件
- `tests/TEST_PLAN.md`：**测试用例总表**（ID、环境 CPU/L2/T8、通过标准、状态）。改了代码就重跑相关用例，并回填状态；实验结果引用用例 ID
- **`research/README.md` 与 `research/claude/base/00-summary-mainline.md`**：底包源码地图与当前主线（底包 = `build/base_exact/`，不是 v0.5.20）
- `research/shared/pipeline.md`：评测、打镜像、提交的流程；`tests/L2.md`：L2 队列
- `notes/findings.md`（事实）、`notes/decisions.md`（决策）、`notes/experiments.md`（实验）、`notes/tech-debt.md`（技术债）、`notes/quality.md`（质量评分）

## 必须遵守（摘要，全文见 rule.md §4）
- 不关 thinking、不压输出、不截历史、不删 tools；`meta_info` 时间戳与 token 计数必须如实；`/flush_cache` 必须真清。
- 不碰其他选手的镜像或服务；不探测评测平台。
- **技术路线保密（rule.md §4.1）**：Trisol 上的镜像名、tag、服务名、描述、command、env 都是 arena 队友可见的，一律中性（例如 `lh-img:0922a`、`lh-t1`，描述留空；command 用 `/opt/ax/serve <profile>`）。
- 审批：L2 自测与每天 2 次正式提交由用户授权 Claude 批准（决策 23/25/27）；Codex 不自行提交、不起 8 卡、不打镜像。GPU 开发机只在 `/sjtu/linhang/arena/` 下工作。
- 写补丁照 `build/base_exact/`（L3 实际代码），不要照 `src/sglang/`（v0.5.20，不是底包）。
- `s1-dev/`、`src/sglang/`、`build/base_exact/`、`llm-challenge-arena-v1/` 只读。
- 任务状态写进 `notes/dispatch.md`（accepted → in-progress → done / blocked）。**落盘即通知**。
- 不要改 `board.md` 里的「活跃 Codex 实例」表（Claude 维护）。
- 改动让文档过期，就在同一轮里把文档一起更新。
- 往 findings / dispatch / decisions 追加新编号时，**先运行 `python3 scripts/next_id.py F|T|D` 取号**，写完立刻运行 `python3 scripts/check_records.py`（它会报重复编号）。
- **原始证据（日志、JSON、模拟输出、测试 log）放 `evidence/<任务号>/`，不要放进 `notes/`**；`notes/` 只放账本。改完账本后运行 `python3 scripts/index_notes.py` 刷新索引。
- 收尾前运行 `python3 scripts/check_records.py`。
