# T55 — 全仓库清理审查：只保留正确、现行的文档（Claude 派发，4 个 subagent 分区审查）

用户 09-24 的要求（原话）："过时的…清理干净，我很怕误导了后面的迭代和智能，能不能只维护正确的文档"，"我觉得那就直接删除吧 反正以后不再需要读到"。
**政策**：工作区只留正确、现行的内容。过时或错误的内容**直接删除**（git 历史保留，不建归档目录）；仍然有用的部分并入现行文档后，再删原处。

先读 `plans/prompts/_context-0924.md`（已核实事实与赛题口径就是判断标准）。判断只依据：task.md、F93/R19、原始数据、源码。拿不准的标 UNSURE，并写出怎样才能确定。

## 输出
只写你自己那份 `evidence/T55/<你的分区>.md`，不要改任何其它文件。内容：
1. 按文件列一行：`路径 | 结论 KEEP / REWRITE / DELETE | 理由（引用 task.md:行 / F 编号 / 源码行 / raw）`。
2. 每个 REWRITE 文件，列出必须删改的具体段落（行号，引用原句 ≤20 字）和改成什么（一句话；若改成"删除该段"也写明）。
3. DELETE 前检查引用：被哪些仍保留的文件引用（`grep -rn`）；删了会不会断掉在用的工具、job、补丁或提交追溯。
4. 顶部写"最会误导后续智能体的 10 条"。

**不能删**：`llm-challenge-arena-v1/`、`s1-dev/`、`src/sglang/`、`build/base_exact/`、`refs/`（只读输入）；已提交或待出分镜像 0923a 的补丁（`build/image/0923a.patches.txt`）；S0 基线要用的 `patches/drafts/120-sched-protect-chain-v2.patch`；当前在用的工具。

## 分区
- **B1-notes**：`notes/` 全部。findings F1–F93 逐条、decisions、experiments、ledger-8card、submissions、tech-debt、quality、README；dispatch 只看状态与事实不符的行，例如 T52b 仍写 in-progress、T50 重复。
- **B2-docs-research**：`README.md`、`HANDOFF.md`、`board.md`、`AGENTS.md`、`CLAUDE.md`、`rule.md`、`research/README.md`、`research/claude/*`（R1–R14、base/00–04）、`research/shared/*`、`research/archive/`、`plans/`、`docs/`。`research/codex/*` 也要判 KEEP/REWRITE/DELETE（Codex 的文件由 Claude 通知 Codex 后处理）。
- **B3-patches-tests-scripts**：
  - `patches/`：每个 `.md` 与对应 `.patch` 的现行行为是否一致，README、RELEASE、drafts/、v0520/；
  - `tests/`：TIERS、TEST_PLAN、L2.md、queue*；
  - `scripts/`：已退役或已无用的工具、重复实现评分口径的脚本、与补丁不同步的生成器、复制旧模板的 `pod/jobs/*.sh`、archive/；
  - 另外：`submission/`、`build/`（文档和清单）、`data/`、`evidence/`（孤立证据）、`logs/`（只看体积和杂乱）。
- **B4-rules**：拿 `llm-challenge-arena-v1/`（task.md 为准）逐条核对全仓库对赛规的转述（排名、11 道门、统计余量、TPOT 门、爬坡、单档时长、能力门、flush/诚实要求、开发集用途），以及 `scripts/score_formal.py`、`scripts/pod/verify/*`、`scripts/check_submission.py` 的实现是否与 task.md 一致。列出每处不一致：文件:行 ↔ task.md:行。

## 最后回报（≤200 字）
各类计数，加最重要的 5 条。
