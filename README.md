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

## 当前状态（2026-09-23，8 卡实测后）
- **8 卡服务** `lh-arena-sess-b`（2102486579267252224）在线；旧守护进程已退役（scripts/archive/retired/），任何在用脚本都不含停/删服务。实验全部走 pod 队列 `scripts/pod/podq`。
- **能跑 + 能力**：b113（000+101+105+110–113，tilelang DSA）在比赛镜像上启动、探测、能力冒烟 12/12；开发机 kernel 结论在真机 11/11 复现（F73/F74）。
- **N6 基线（F76）**：fast_intra / overall_intra FAIL（p95 7.5s，主因**排队** p95 6.4s）；turn_start、chain_start PASS；TPOT 0.0304；KV 实际峰值 92%（日志显示的 50% 未计可淘汰缓存；R18）→ 容量已是约束。
- **正在测**：120（调度）N6 A/B → 114/CP/DCP(+115) 探针 → 140 A/B → N10/N14。代码级分析：T49（Codex W23：缓存丢失根因+容量；Fable：排队+预填充结构）。
- **排行榜**：无人过 N=26；3 人 N=22（LewyM tpot 0.0273）。正式提交暂停（用户指示先跑通测评）。
- **方向与问题清单**：`research/claude/R8_next_directions.md`；逐条事实 `notes/findings.md`（F56–F76）；决策 `notes/decisions.md`。
