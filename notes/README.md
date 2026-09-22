# notes/ 索引（自动生成：`python3 scripts/index_notes.py`，请勿手改）

`notes/` 只放**账本**；原始证据（日志、JSON、模拟输出）放在 `evidence/<任务号>/`。

| 文件 | 用途 |
|---|---|
| decisions.md | 决策（最新在上） |
| findings.md | 已验证事实 F# |
| experiments.md | 实验与工具台账 |
| dispatch.md | 派发任务 T# |
| submissions.md | 正式提交台账 |
| quality.md / tech-debt.md | 质量评分 / 技术债 |

## Findings

- **F1** Model is a hybrid, not plain MLA (from glm_tok/config.json)
- **F2** Workload shape (dev set; hidden-split chain metadata is in chains.jsonl)
- **F3** The "reminder" divergence (KEY)
- **F4** Scheduler: chunked long prefill starves short requests (R2)
- **F6** R1 confirmations (research/claude/R1_model_and_engines.md)
- **F8** Codex: exact baseline pool accounting (refines F1 / R1 / R2 estimates)
- **F9** Codex: native flush needs JSON and all-worker success
- **F10** Codex: dev PASS is not formal PASS
- **F11** Codex cross-check of Claude F6 / upstream correctness reports
- **F12** Codex: D1 checkpoint lifecycle / scheduler constraints (source only)
- **F16** Codex: static hybrid pool split and DP request-flooring
- **F18** Codex: Mamba per-path cap is soft, not role/end protection
- **F20** Codex: lazy supports native prefill donation, with distinct allocation-failure semantics
- **F21** Claude: organizer base images in the Trisol catalog (read-only lookup, 2026-09-22)
- **F22** Codex: merged role-boundary checkpoint prior art, not a local performance result
- **F23** Codex: public dev phases reuse rid; successful process exit is not a flush guarantee
- **F26** Codex: vLLM internal-checkpoint precedent is merged, with dtype and hardware caveats
- **F33** Claude: arena Trisol queue policy (renumbered from a duplicate F25) (read-only, 2026-09-22)
- **F27** Codex W1: AgentX decode interval is local, but the +141% headline is fixed-length evidence
- **F28** Codex W1: AgentX DP affinity proposals are not equivalent to shipped internal routing
- **F29** Codex W1: AgentX architecture filters and the hybrid DSA offload blocker
- **F30** Codex W1: Transferable frontend work exists; external/mocker speedups are not local engine gains
- **F31** Claude: formal-set composition is recorded in the dev cohort file (read-only)
- **F32** Claude: chat template vs SGLang parsers (source read, 2026-09-22)
- **F35** Claude: public formal results contradict the uncalibrated simulator's "TPOT binds first" (data/all_att.json)
- **F36** Codex main: E1 stock KDA+MLA stand-in works; F13 misses resident FULL-KV lifetime
- **F37** Claude (from Codex E1): real stock stand-in confirms the D1 premise, and stock is slightly WORSE than modelled
- **F39** Claude: technical routes visible in the public Trisol image catalog (names/tags only; contents NOT inspected)
- **F40** Claude: first N@SLO=22 on the leaderboard, and its gate profile points to SLO-aware prioritisation (data/all_att.json refreshed, 494 attempts)
- **F42** E2实测D1 v1.1局部增命中，但reminder-heavy尾部目标未过（T13 / Codex main）
- **F43** Claude (verifying Codex main E2, notes/e2_d1/comparison.json): D1 v1.1 has no cache regression and helps (numerics not yet safe, see F45), but misses
- **F44** Claude: official submission quota — **user-confirmed 2 per day** (older platform changelogs said 3; user confirmed on 2026-09-22 that the doc/limit is
- **F45** E2 raw-logits验收失败；“缓存无退步”不等于数值安全（Codex main，T13）
- **F46** W12 / T25: chain-start统计条数余量≠秒数放宽；固定R10模型里EDF未胜SPF
- **F47** T29：v1.1的flush/池回收与N4单partial取得L2实测；统计只覆盖新准入
- **F48** T33：Trisol CLI 真实返回格式与守护进程的模拟不一致（已修）
- **F49** T34：AgentX快照保留三阶段与本地prefill节拍核对（仅调研）
- **F50** T35：GLM内部快照新PR、ReplaySSM接线与DP亲和路由；容量统计不等于容量增加
- **F51** T33：底包源码（diag04，407 文件，sha256 025ee541…）里两条路线的现状
- **F52** T36：实际底包的单点快照不等价101；S1 routing族键不等于session
- **F53** 底包 SGLang = GitHub 公开提交 fe236ea6c3 + 一处与我们无关的多模态修复；中途状态 h 是 bf16
- **F54** L3 实际运行的 SGLang 代码 = build/base_exact + 我们的补丁（证据链已闭合）
- **F55** 平台拒收 `name:tag@sha256:digest` 形式的 image；45734/45735 因此部署失败（不计额度）
- **F56** 原版底包在真实 8×A100 上启动即崩（DeepGEMM 不支持 sm80）；A/B 提交不可能运行
- **F57** A100 上 DSA 注意力后端必须用 tilelang，不能用 fa3（supersedes F56 中的 fa3 参数）
- **F58** A100 上 FP8 MoE 专家必须走 Marlin W8A16（补丁 111）
- **F59** B+110+111（tilelang）在真实 8×A100 启动并通过功能探测；首次形状触发服务期 Triton 编译
- **F60** T42 / 130：完整分词线程池逐 token 等价，真实长输入显著降低 HTTP loop 阻塞（CPU）
- **F61** T41/120 调度保护 CPU 验证与 101 既有双 partial 反例（2026-09-22，Codex W15）

## Decisions（最新在上）

- **#32** T41/120 默认启用普通 TP 调度保护：续算先保留对齐的 2048 token 上限预算，剩余预算按原 LPM 接完整短命中；prefill 后有存活 decoder 才交替一次；不启用 mixed、不重排 LPM。环境变量关可逐决策回退
- **#31** T42/130 用每个 TokenizerManager 的单线程池执行原始完整分词；所有 regular text（含小输入）串行移出 loop；暂不加入可选前缀缓存。body routing_key 优先，再 Routing-Key 头，再 Session-ID 头
- **#30** 新主线（research/claude/base/00-summary-mainline.md）：M0 先证 A100 上 DSA 后端能跑（L1 装 base_exact + 带 DSA 替身）；M1 调度保护链中间请求（冷启动分块独占 GPU 是源码里的主瓶颈）；M2 KDA 双点 fp32 快
- **#29** 底包 SGLang 以 `build/base_exact/`（= 公开提交 fe236ea6c3 + 两处多模态修复，4686 文件指纹全对）为唯一依据；v0.5.20 线（001/002/003/004、SPF/HRRN、模拟器结论）退役，只保留为 L1 替身参考；补丁一律照底包源码写
- **#28** 撤下 35 项纯调参 L2 任务（归档 tests/queue_archive/param_sweep_0922）；L2 只排"针对已测瓶颈的机制补丁"：先用 01–05 诊断 B 卡在哪道门、TTFT 花在哪、缓存命中率，再做 102（对话边界记状态、不拆分预填充，参考 vLLM#56960）、D
- **#27** L2 改为"底包 + 与提交镜像相同的补丁（000/1xx）+ 提交原文命令/环境"，统一入口 `scripts/l2.py`，入队即由 Claude 批准、自动排队；旧 5 项（v0.5.20 补丁、SPF）退役到 tests/queue_archive
- **#26** 定义三级验证体系：L1 2 卡自测、L2 提测（Trisol 8 卡）、L3 正式提交；逐级晋级，每级只回答它能回答的问题（tests/TIERS.md）
- **#25** 正式提交的批准权也交给 Claude：只提交自测证明有效的配置，每天 2 次（稳定线 + 进取线）
- **#24** T29 mixed-only诊断提高context到262144、KV池到524288；权重/补丁/page64/chunk8192/extra_buffer不变，旧E2矩阵不重跑、不混性能比较
- **#23** 自测队列项的批准权交给 Claude（由 Claude 放 APPROVED）；自测与提交两个守护进程迁到 GPU 开发机（/sjtu/linhang/arena/repo，tmux 常驻），以免合上笔记本后停掉
- **#22** 〔已被 29 取代〕T25/003采用独立调度session字段、首见30s/续轮actual-miss 3或5s的保守代理；EDF/least-slack/weight2均默认关闭，交付草稿而不替代SPF默认候选
- **#21** 每天 2 次正式提交，两次都要用上；每天准备两个镜像：A = 稳定线（只含已通过审阅和验证的补丁），B = 进取线（加上当天最新的补丁）。自动提交守护进程每天上限 2
- **#20** I6（SLO 感知 / EDF 调度）升为 P0，与 D2 合并推进：优先保护链中间请求，允许链首在正式统计余量内排后
- **#19** E2采用三组（原stock / D0+D1关闭 / D0+D1开启），统一context131072，先无trace回放再独立导出raw logits
- **#18** 〔已被 29 取代〕D2（调度，移植 #40024 SPF）升为与 D1 并列的最高优先级，立即开始写代码；D1 继续做同引擎 A/B，在拿到实测之前不宣称 D1 能提高临界档
- **#17** T8早期Codex草案全部原样归档到`research/codex/archive/`，不晋升为当前工具；E1继续使用已有`scripts/`
- **#16** E1/E2替身固定为`e1-kimi-linear-4l-qfull`，q_lora_rank=null；保留未修改stock源码与Python UnifiedTreeCore默认cache路径；F13/F24仅作理想预测，不作精确oracle
- **#15** 借鉴 harness-template-cn：采纳 `AGENTS.md`（Codex 自动读取的入口）+ `CLAUDE.md`、`plans/active` 执行计划（含验证方式与回滚）、`notes/tech-debt.md`、`notes/quality.md`、`scripts/check
- **#14** 启动命令加 `--tool-call-parser glm47`，保留 `--reasoning-parser glm45`
- **#12** 按任务启动完全权限的独立 Codex 实例（`scripts/codex_worker.sh`，每个任务一个新上下文）；主 Codex 会话负责审阅和讨论；Claude 也可以派自己的 subagent
- **#13** 借鉴 AgentX/ClawPerf 的经验，新增 I1–I5（directions §I）
- **#11** 〔已被 29 取代〕D1 方案 A 维持为首选，并参考 llama.cpp #22929 的实现
- **#10** 分工：Trisol 8 卡做真实验证（底包内容、完整模型、开发集梯子、能力抽检）；2 卡 GPU 机只做小而快的事（替身模型功能测试、补丁开发与单测、数据分析、Dockerfile 预演）
- **#9** 能力分抽检加入自测流程（task.md 没有明确要求这一项）
- **#8** 批准本地 2 卡验证（替身模型 E1/E2）；Trisol 8 卡和提交仍然不行
- **#7** 本地验证用随机权重小 Kimi-Linear 做替身，不用裁层 GLM-5.3
- **#6** 〔已被 29 取代〕D1 先做方案 A（在边界切 chunk），方案 B（同一 forward 导出中间状态）排在后面；若 A 验证有收益，B 优先级上调
- **#5** 暂不启用 HiCache 和 `--enable-mixed-chunk`
- **#4** 〔已被 29 取代〕D2 先移植 #40024（SPF），#39717 以后再说
- **#3** 撤回"vLLM main 不支持 Glm5Next"的说法
- **#2** 撤回"MLA 在 TP8 下复制、改 DP-attention 就是最大杠杆"的初始判断，改为以 D1 为头号方向
- **#1** 以 SGLang 为主路线

## Experiments / Tools sections

- E1 — Codex 接收与范围（2026-09-22 UTC）
- E1 环境搭建进展 1（Codex，非服务结果）
- E1 环境搭建进展 2（Codex，2026-09-22 06:34 UTC）
- E1 环境搭建进展 3（Codex，2026-09-22 06:58 UTC）
- E1 启动排障（Codex，2026-09-22 07:11 UTC）
- E1 替身配置修订（Codex，2026-09-22）
- E1 最终结果与 E2 交接（Codex，2026-09-22 UTC）
- E2 — D1 v1.1 开关 A/B 与数值检查（Codex main，2026-09-22）
- E2 中间交接（08:23 UTC；数值检查仍在进行）
- E2 最终结果（Codex main，2026-09-22 08:33 UTC）
- Tools
- T23 W10 — Session A 整批 runner（CPU/dry-run，未执行会话）
- T19 — IF/D1/CAP 测试缺口工具（Codex W6，2026-09-22 UTC）
- 提交包工具（Claude，2026-09-22；仅本地，未构建/未提交）
- T12 — 临界 N 搜索与 formal-ish 离线评分（Codex W2，2026-09-22 UTC）
- harness 自带测试（Claude，2026-09-22；只用 CPU）
- T16 — 真实前缀用例 / 加权诊断 / 8 卡规划 / 容器自检（Codex W4，2026-09-22 UTC）
- T20 — D2 SPF port and fixed R10 simulator cross-check（Codex W7，2026-09-22）
- T22 — Unattended submission queue and daemon (Codex W9, 2026-09-22 UTC)
- Tools — T27 keep-alive 提测 daemon（Codex W13；CPU/mock）
- Tools — T25 / W12 I6 statistical analysis + CPU model + draft 003（2026-09-22）
- T24 — 自动提测队列与回收守护进程（Codex W11；CPU-only）
- T28 — E1/E2 测试 ID 与证据回填（Codex main，2026-09-22）
- T29 — E2 v1.1 遗留用例补测（Codex main，2026-09-22）
- E2b — 004最终chunk切分：v1.1 / 001+002控制 / v1.2（T26，Codex main，2026-09-22）
- 环境、控制与命令
- 结果与TEST_PLAN映射
- 审阅、失败保留与清理

## Plans

- plans/active/102-role-track.md
- plans/completed/2026-09-22-d1-role-boundary.md
- plans/completed/2026-09-22-evaluation-strategy.md
- plans/completed/2026-09-22-first-8gpu-session.md
- plans/prompts/T41-M1-scheduler.md
- plans/prompts/T42-M3-tokenize.md

## Reports

- research/README.md
- research/archive/R5_patch_review.md
- research/claude/R1_model_and_engines.md
- research/claude/R2_serving_techniques.md
- research/claude/R3_prior_art_en.md
- research/claude/R4_similar_competitions.md
- research/claude/R6_harness_template_review.md
- research/claude/R7_top_players_analysis.md
- research/codex/R13_agentx_mlperf_reading.md
- research/codex/R14_recent_pr_watchlist.md
- research/codex/R15_base_source_exploration.md
- research/codex/R5_dp_memory_accounting.md
- research/codex/R6_prior_art_cn_github.md
- research/codex/R7_kda_internal_checkpoints.md
- research/codex/R8_agentx_checklist.md
- research/codex/README.md
- research/codex/archive/R10_sim_calibration.md
- research/codex/archive/R11_overall_summary.md
- research/codex/archive/R12_slo_aware_scheduling.md
- research/codex/archive/R1_directions_and_review.md
- research/codex/archive/R2_review_shared_directions.md
- research/codex/archive/R3_base_images.md
- research/codex/archive/R4_prefill_scheduling.md
- research/codex/archive/R9_offline_simulator.md
- research/codex/archive/README.md

