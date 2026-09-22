# R6 — harness-template-cn 借鉴评估（Claude，2026-09-22）

来源：`refs/harness-template-cn`（commit 337fbc7，2026-05-26；只作参考，不混入现有结构）。
结论：**有成型的迭代闭环**：导航入口 → 执行计划 → 验证 → 记录（history / 决策）→ 技术债 → 质量评分 → 清理归档。整体不是空脚手架，值得借鉴其中 5 点；其余与我们已有机制重复，或与本赛无关。

## 模板的核心做法（V，读自原文件）
1. **入口极短，只做导航**（`AGENTS.md`）。正式知识放在 `docs/`，按"每轮先读 / 改完前读 / 按需选读"分层。`CLAUDE.md` 只有一句"必须先阅读 AGENTS.md"。
2. **执行计划**（`docs/PLANS_GUIDE.md`、`exec-plans/templates/execution-plan.md`）：适用于跨多轮、高风险、多 Agent 共同推进的任务。字段：目标、范围（包含/不包含）、背景、风险与缓解、里程碑、**验证方式（命令、手工、观测）**、进度勾选、决策记录。放在 `active/`，完成后移到 `completed/`，过期的及时关闭，"保证 active 目录可信"。
3. **history**（`docs/HISTORY_GUIDE.md`）：每个完成的代码变更写一份，内容是用户诉求、改动、设计动机和受影响文件；调研任务不写；要脱敏。
4. **技术债追踪**（`tech-debt-tracker.md`）：暂不阻塞、但值得留档的问题。
5. **质量评分**（`QUALITY_SCORE.md`）：按区域给 A/B/C/D 评分，写明原因和下一步，让人随时知道最薄弱的环节。
6. **核心理念**（`core-beliefs.md`）：
   - 人定方向，Agent 执行；
   - 仓库里的知识比私有上下文更重要；
   - **Agent 反复失败时修脚手架和环境，而不是加 prompt 压力**；
   - **能变成机械检查的约束，就不要只停留在口头规范**；
   - 持续整理，控制熵增。
7. 脚本化：`make new-plan` / `make new-history` 从模板生成文件。

## 与我们现状对照
| 模板机制 | 我们已有的 | 缺口 | 是否采纳 |
|---|---|---|---|
| 短导航入口 | `README.md`（阅读顺序）+ `rule.md` | **没有 `AGENTS.md`**。Codex 在仓库根目录会自动读取 `AGENTS.md`，新开的 worker 不必靠 prompt 灌规则 | **采纳**：新建 `AGENTS.md`（导航 + 红线 + 落盘规则），`CLAUDE.md` 指向它 |
| 执行计划 | `research/shared/directions.md`（方向）、`patches/*.md`（设计）、`pipeline.md` | 缺**按任务的 plan**，尤其是占用稀缺 8 卡的会话，需要写明验证方式、回滚、进度 | **采纳**：`plans/active/`、`plans/completed/`、模板；先建"首次 8 卡会话"和"D1 端到端"两份 |
| 决策记录 | `notes/decisions.md` | 无 | 已有 |
| history | `notes/dispatch.md`（任务）+ `board.md` 已完成栏 + `patches/*.md` | 代码变更的"动机与受影响文件"散在各处 | **部分采纳**：不另建 histories 目录；代码变更的动机和文件写进对应 plan 的决策记录和 `patches/*.md`，避免重复 |
| 技术债 | 散落在各文档的"未决"和"待办" | 没有统一清单 | **采纳**：`notes/tech-debt.md` |
| 质量评分 | 无 | 看不出最薄弱的环节 | **采纳**：`notes/quality.md`（按组件：D0、D1、D2、工具、提交包、环境、证据） |
| 机械检查 | `check_submission.py`、各类单测 | 记录的一致性全靠自觉（例如 dispatch 状态值是否合法、实例表有没有被覆盖、有没有泄露 token） | **采纳**：`scripts/check_records.py` |
| 清理归档 | 看板"已完成"越来越长 | 缺归档节奏 | **采纳**：看板只保留最近完成的项，更早的移到 `notes/board-archive.md`；plan 完成后移到 `completed/` |
| 发布记录、前端、供应链、CI/CD | — | 与本赛无关 | 不采纳 |
