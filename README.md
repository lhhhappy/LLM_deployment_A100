# Agentic Science Challenge — 推理服务部署赛（llm-challenge-arena-v1）

GLM-5.3-Flash 部署在 8×A100-80GB 上，比 **N@SLO**（能过全部硬门的最大并发档），同档再比 **tpot_mean**。
**分工**：用户决策；Claude 统筹（派活、维护记录、自测与提交审批由用户授权给 Claude）；Codex 协同开发与交叉审阅。

## 从哪里开始读（按顺序）
0. **`HANDOFF.md` — 最新交接（新会话先读）**；`AGENTS.md` — 导航与红线；`CLAUDE.md` — Claude 的入口
1. **`research/README.md` → `research/claude/base/00-summary-mainline.md`** — 底包源码地图、当前主线、L1 复盘（每次会话与每次压缩后先读）
2. `rule.md` — 协作规则、目录约定、红线
3. `board.md` — 看板；`notes/dispatch.md` — 派发日志（T 编号）
4. `notes/decisions.md`（决策，最新在上）、`notes/findings.md`（事实，顶部有"当前事实基线"）
5. `tests/TIERS.md`（L1/L2/L3 三级验证）、`tests/L2.md`（L2 队列用法）、`plans/active/`

## 目录索引
| 路径 | 内容 |
|---|---|
| `llm-challenge-arena-v1/task.md` | 赛题原文（只读，以它为准） |
| `build/base_exact/` | **底包 SGLang 逐字节副本** = 公开提交 fe236ea6c3 + 两处多模态修复；4686 文件指纹与镜像全对（F53/F54）。写补丁一律照它 |
| `build/l3_0922e/`、`build/l3_0922f/` | 正式提交 A/B 实际运行的代码（base_exact + 补丁） |
| `patches/` | 现行 000、101（`RELEASE`）；`drafts/` 草稿；`v0520/` 已退役的 v0.5.20 线 |
| `research/` | 调研（索引见 `research/README.md`）；已退役内容在 `archive/` |
| `scripts/` | 工具：`l2.py`（L2 队列）、`submit_official.sh`（正式提交）、`build_image.sh`（打镜像）等；`archive/` 为模拟器与旧线 |
| `tests/` | 三级验证、L2 队列 `tests/queue/`（守护进程在 GPU 机自动跑） |
| `submission/` | 正式提交 JSON 与候选 profile |
| `data/` | 公开成绩（`all_att_2026-09-22b.json`，523 条） |
| `s1-dev/` | 公开开发集与 harness（只读） |
| `src/sglang/` | SGLang v0.5.20（只读；**不是底包**，仅 L1 旧替身参考） |
| `refs/` | 参考代码：底包公开提交 `sglang-fe236ea6c3`、vLLM #56960、SGLang #31170 |

GPU 开发机（2×A100）：`ssh GPU`，只在 `/sjtu/linhang/arena/` 下工作；仓库镜像在 `/sjtu/linhang/arena/repo`。

## 当前状态（2026-09-22 晚）
- **正式提交**：45734（A = 底包 + 000）、45735（B = A + 101 D1），排队评分中（约 18 小时）。每天 2 次。
- **L2**：会话 `lh-arena-sess-a` 排队等卡；守护进程在 GPU 机 tmux `arena-daemons:l2` 自动运行队列。
- **排行榜**：3 人 N=22（LewyM tpot 0.0273 领先，Jinbo hu 各门都在限内）。夺第一需 N=26，或 N=22 且 tpot_mean < 0.0273。
- **主线（决策 30）**：M0 证明 A100 上 DSA 后端能跑 → M1 调度保护链中间请求 → M2 KDA 双点 fp32 快照 → M3 分词与路由键 → M4 MTP；调参放最后。
- **最大风险**：A100 默认 DSA 预填充后端在 index_kpool=4 下可能对 >2048 token 报错（源码成立，运行时未证），A/B 都未显式指定后端。
