# rule.md — Claude × Codex 协作规则与交接（arena / GLM-5.3-Flash 部署赛）

> 两个 agent（Claude Code、Codex）在同一个仓库里一起开发。**动手前先读本文件**，按这里的约定放文件、记结论、交接。
> 本文件是唯一的协作约定来源；约定有变就改这里，并在文末「变更记录」追加一行。

---

## 0. 一句话目标

固定模型 GLM-5.3-Flash、8× A100-80GB、固定负载。提交 `submission.json`（image / command / env / model_name）。
排名：**N@SLO**（2/6/10/14/18/22… 能过全部硬门的最大档）→ **tpot_mean**。2026-09-22：3 人 N=22（LewyM tpot 0.0273 领先）。目标：N=26，或 N=22 且 tpot_mean < 0.0273。
完整题面：`llm-challenge-arena-v1/task.md`（以它为准）。

## 1. 目录约定（本地容器 `/workspace/Agentic_science_challenge`）

| 路径 | 用途 | 谁写 |
|---|---|---|
| `AGENTS.md` / `CLAUDE.md` | 极简导航入口 + 红线摘要（Codex 自动读取） | Claude 维护 |
| `README.md` | 项目索引与当前状态（入口） | Claude 维护 |
| `plans/active/`、`plans/completed/` | 执行计划（目标、范围、风险与回滚、里程碑、验证方式、进度、决策），模板在 `plans/templates/plan.md` | 负责人维护，完成后归档 |
| `notes/tech-debt.md`、`notes/quality.md` | 技术债与未决项；按组件的质量评分 | 两者，Claude 维护 |
| `rule.md` | 本文件：协作规则 + 交接 | 两者（改动记变更） |
| `board.md` | 任务看板 | 两者 |
| `evidence/<任务号>/` | 原始证据：日志、JSON、模拟输出（从 notes/ 移出，见 evidence/MOVES.txt） | 产出者 |
| `notes/README.md` | notes 自动索引（`scripts/index_notes.py`） | 自动 |
| `notes/dispatch.md` | 派发任务日志（每个任务的派发、接收、进展、产出；落盘即通知） | 两者 |
| `notes/decisions.md` | 决策记录（定了什么、为什么；最新在上） | 两者，Claude 维护 |
| `llm-challenge-arena-v1/` | 赛题原件（task.md 等） | **只读** |
| `s1-dev/` | 公开开发集 + harness（`run_dev.py`） | **只读**（不要改 harness，改了结果不可比） |
| `build/base_exact/` | **底包 SGLang 逐字节副本（= L3 实际代码，F54）**；补丁一律照它写 | 只读 |
| `src/sglang/` | SGLang v0.5.20 源码（**不是底包**，仅旧 L1 替身参考） | 只读 |
| `research/claude/` | Claude 的调研报告 | Claude |
| `research/codex/` | Codex 的调研报告 / 审阅意见 | Codex |
| `research/shared/` | 双方对齐后的**结论** | 两者，需注明作者 |
| `research/README.md`、`research/claude/base/` | 调研索引；底包源码地图与当前主线 | Claude 维护 |
| `notes/findings.md` | 已验证/已量化的事实日志（F1, F2…编号，只追加） | 两者 |
| `notes/experiments.md` | 实验计划与结果台账（E1, E2…编号） | 两者 |
| `scripts/` | 分析脚本、工具脚本（小而独立，文件头写用途） | 两者 |
| `patches/` | 对 SGLang 的补丁（`NNN-short-name.patch` + 同名 `.md` 说明） | 两者 |
| `submission/` | 候选 `submission.json`（`candidate-XX.json`）+ 最终版 | 两者；**提交前需双方确认** |
| `data/` | 爬到的排行榜/提交数据（`lb.json`, `all_att.json`） | 两者 |
| `env.sh` | 本地环境变量（`source env.sh`） | 两者 |

命名：报告 `R<n>_<topic>.md`；补丁 `NNN-<name>.patch`；实验 `E<n>`；事实 `F<n>`。编号不复用。

## 2. GPU 开发机（2× A100-80GB，NVLink）

- 连接：`ssh GPU`（配置见 `scripts/ssh_config.GPU`，经本地代理 127.0.0.1:7890 用 `scripts/ssh_httpconnect.py` 隧道；密钥 `~/.ssh/id_ed25519_gpu`）。
- **只在 `/sjtu/linhang/arena/` 下工作**（`/sjtu` 是共享盘，`/sjtu/linhang` 下有用户其他项目，别动）。根盘只剩 ~3 GB，不要往 `/`、`/root` 装东西。
- 子目录：`env/`（conda 环境，如 `env/ana`）、`s1-dev/`（开发集副本）、`code/`、`models/`、`runs/`、`cache/`。
- 登录后先 `source /sjtu/linhang/arena/env.sh`（pip 走 `mirrors.ivolces.com` 且不走代理——机器上配置的 7890 代理没在跑；HF 走 hf-mirror；所有缓存落 /sjtu）。
- 可用：GitHub、pypi 镜像、hf-mirror（`zai-org/GLM-5.3-Flash` 有，62 个分片 ≈328 GB——**2 卡放不下完整模型**，只能裁层/小模型/算子级验证）。`registry.dp.tech` 需认证（401）。
- 用 GPU 前 `nvidia-smi` 看一眼有没有别人的进程；长任务用 `tmux`/`nohup`，日志写 `runs/`。

## 3. 账号与 CLI（本地容器）

- `source env.sh` 后再用 `playground`（Node 需要 `NODE_USE_ENV_PROXY=1` 才走代理）。已登录（token 在 `~/.config/playground/credentials.env`，**不要打印、不要写进任何文件/日志**）。
- `bohr` 已用企业账号（dpt）登录；Trisol 已加入 arena team（GPU 机上的守护进程使用）。
- Codex：`codex exec -m gpt-6-astra --skip-git-repo-check -s read-only "<prompt>"` 可用。

## 4. 硬性红线（违反 = 成绩作废）

1. 不关 thinking、不压输出预算、不截历史、不删 tools。
2. `meta_info` 的时间戳、`prompt_tokens`/`cached_tokens`/`completion_tokens` 必须如实。
3. `/flush_cache` 必须真清（含 HiCache 主机层）；清不干净宁可报错。
4. 不探测/干扰评测平台，不还原隐藏题；密钥不进提交物/日志/报告。
5. `command` 是 argv 不是 shell；A100 上 SGLang 必须 `SGLANG_OPT_USE_TOPK_V2=0`（放 `env`）。
6. 审批：L2 自测（Trisol 8 卡）与每天 2 次正式提交由用户授权 Claude 批准，每次正式提交向用户汇报（决策 23/25/27）；其他 agent 不自行提交、不起 8 卡、不打镜像。

## 4.1 技术路线保密（对外可见的元数据一律中性）
arena 队友（即竞争对手）能看到：Trisol 镜像目录里的**镜像名和 tag**，以及推理服务的**名字、描述、启动命令、环境变量**。所以：
- 镜像名和 tag 一律中性，例如 `lh-img:0922a`，不带 spf、d1、kda、snapshot、edf 之类的词；
- 服务名中性（如 `lh-t1`），**描述留空**；
- Trisol 自测时，服务的 command 只写 `/opt/ax/serve <profile>`；具体 flag 和环境变量放在镜像内部的启动脚本里（由 `scripts/build_image.sh` 生成，profile 为 b0–b3）；
- 正式提交的 submission.json 只有作者和主办方能看到，可以写明确的 command；但镜像名同样要中性；
- 不在任何公开渠道（群聊、共享文档、镜像描述）写我们的方向名。

## 4.2 模型分配（按任务难度，别一律用最贵的）
| 任务类型 | Codex worker | Claude subagent |
|---|---|---|
| 核心补丁的设计与审阅、疑难调试、需要仔细推理的性能分析 | `gpt-6-astra`，effort xhigh | Opus（主会话自己做） |
| 写工具和脚本、测试夹具、数据整理、批量改文档、汇总 | `gpt-5.6-luna`（或 `terra`），effort medium | Sonnet |
| 简单查找、格式化、监控类小事 | `luna` low | Haiku |
启动时用 `CODEX_MODEL=… CODEX_EFFORT=… scripts/codex_worker.sh`；看板实例表写明所用模型。

## 5. 协作流程

0. **派发落盘**：Claude 派的每个任务都记在 `notes/dispatch.md`（T 编号）。Codex 收到后立即改状态，有进展就更新，阻塞马上写明。Claude 监视这个文件，所以落盘就等于通知。
1. **认领**：开始一项工作前，在 `board.md`「进行中」加一行：`[agent] 任务 — 产出路径`。完成后移到「已完成」。
2. **结论分级**：每条结论标注 `VERIFIED`（读源码/实测/数据算出）或 `INFERRED`（推测），并给出处（URL、`文件:行号`、脚本名）。
3. **决策入账**：凡是改变方向、撤回结论、选定方案的，写进 `notes/decisions.md`。
4. **事实入账**：可复用的已验证事实写进 `notes/findings.md`（新编号，只追加；推翻旧结论时写新条目并注明 "supersedes F<n>"）。
4. **交叉审阅**：一方的关键方向/补丁，另一方审阅后在 `research/shared/` 或补丁 `.md` 里留意见（同意/反对 + 理由）。
5. **不覆盖对方文件**：只追加或在自己目录写；要改对方的东西先在 `board.md` 留言。
6. **收尾前**运行 `python3 scripts/check_records.py`（检查 dispatch 状态、实例表、plan 必填节、密钥泄露、编号重复）。
7. 测试用例统一登记在 `tests/TEST_PLAN.md`（ID + 通过标准 + 状态）；实验结果要引用用例 ID。
8. 跨多轮或占用 8 卡的任务先写 `plans/active/` 计划；完成后移到 `plans/completed/`。
8. 大文件（模型、镜像、运行输出）只放 GPU 机 `/sjtu/linhang/arena/`，本地只留小结果摘要。

## 6. 当前状态

见 `README.md`「当前状态」与 `research/claude/base/00-summary-mainline.md`。旧的交接内容（2026-09-22 上午，基于 v0.5.20 假设）已归档到 `research/archive/rule-handoff-2026-09-22.md`。

## 7. 任务看板

已拆到独立文件 **`board.md`**（认领、进行中、已完成都在那里更新）。

## 变更记录
- 2026-09-22 Claude：创建本文件；排行榜数据移到 `data/`；新增 `research/shared/`。
- 2026-09-22 Codex：已阅读并接受协作规则；认领 F3/R2 审阅。遵循用户最新要求，仅调研、不跑实验；此前自有目录中的脚本与补丁为未部署草案，未修改 `src/sglang/`、赛题或 harness。
- 2026-09-22 Claude：R1 完成；写 `research/shared/directions.md`；撤回「vLLM main 无 Glm5Next」（R1/R2 均称 main 已有，PR #53906）。
- 2026-09-22 Codex：调研报告与交叉审阅完成；只追加 shared 审阅 / F7–F11，未覆盖 Claude 报告。认可暂不启用 HiCache / mixed-chunk；F3 收益需标 oracle，底包来源仍待核对。当前继续遵守仅调研、不跑实验。
- 2026-09-22 Codex：收到 Claude 定向审阅请求后整理 R2；HiCache 暂缓、D1 有条件可行、D6 前置检查同意但不认定私有补丁；追加 F12 并更新看板，未运行实验。
- 2026-09-22 Claude：回应 Codex 审阅（接受 F7–F12 修正；F13 非 oracle 复核 D1 仍成立）；发布优先级与分工 v2（directions §G）。
- 2026-09-22 Codex：确认 §G v2 分工，认领 R3/R4；DP 显存账单列 R5。F13 按已产出的模拟结果审阅，不重跑；全部工作仍限于调研。
- 2026-09-22 Claude：完成 D0（接口合规）与 D1（角色边界快照，方案A=边界切 chunk）设计稿，交 Codex 审阅。
- 2026-09-22 Codex：完成 R3/R4/R5 与 D0/D1 追加审阅，确认 §G 并以 §H 交接；新增 F14–F18。认可 F13 修复 oracle，但 role-over-branch 策略尚无该模拟支撑；path cap=2 不保证 b/end。全程仅调研，无镜像拉取、GPU、模拟重跑、补丁实现或 harness 改动。
- 2026-09-22 Codex：针对 D0/D1 再次交接补充定向源码二审：F19 admission break 条件、F20 lazy tracking/donation 与失败策略；时间戳仍需区分 typed handler 与 body 解析前入口。只追加审阅，未运行测试、模拟或服务。
- 2026-09-22 Claude：派发 R6（中文社区/GitHub 先例）给 Codex；完成 R3 英文先例：TRT-LLM `additional_snapshot_offsets_from_end`、vLLM #45238、Kimi K3 单 forward 导出 KDA 中间状态（方案 B 先例）。
- 2026-09-22 Claude：按用户要求把任务看板拆到 `board.md`。
- 2026-09-22 Claude：新增 `README.md`（入口索引）与 `notes/decisions.md`（决策记录）、`notes/experiments.md`；Claude 负责统筹与记录维护（用户指定）。
- 2026-09-22 Claude：新增 `notes/dispatch.md`（派发任务日志，落盘即通知）；回填 T1–T9。
- 2026-09-22 Claude：借鉴 harness-template-cn（R6、决策 #15），新增 AGENTS.md、CLAUDE.md、plans/、tech-debt、quality、check_records.py；F25 编号冲突，Claude 的那条改为 F33。
- 2026-09-22 Claude：新增 `tests/TEST_PLAN.md`（A 接口 / B D1 / C 性能 / D 能力 / E 提交包 / F 工具）；AGENTS.md 与 check_records 纳入。
- 2026-09-22 Claude：新增 §4.1 技术路线保密（用户要求）；已清空会话 A 的服务描述。
- 2026-09-22 Claude：新增 §4.2 模型分配（用户要求：用 luna/haiku 处理简单任务）。
- 2026-09-22 Claude：整理 notes/：原始证据移到 evidence/（已完成的 T12/T15/T16/T18/T19/T20/T22/E1），引用路径全部改写；新增 notes/README.md 自动索引。进行中的 t25_slo、e2_d1 等任务结束后再移。
