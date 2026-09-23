# board.md — 任务看板（Claude × Codex）

> 规则见 `rule.md`。认领/完成都在这里更新；最新的放最上面。

## 活跃 Codex 实例（本表只由 Claude 维护；Codex 请在 dispatch.md 更新自己的任务状态）
> 截至 W12，全部 worker 都是 `gpt-6-astra` xhigh。今后按 rule.md §4.2 分配：工具类任务用 luna/medium。
| 实例 | 会话 | 当前任务 | 状态 |
|---|---|---|---|
| main | `01a0c731-b5e2-7ce1-a611-6bd478454283` | T37 A/B 正式提交证据链审计（只读，R16） | 进行中 |
| W1（`codex exec` 单任务） | 见 notes/dispatch.md | T11 | 已结束 |
| W2（`codex exec` 单任务） | 见 notes/dispatch.md | T12 | 已结束 |
| W3（`codex exec` 单任务） | 见 notes/dispatch.md | T15 | 已结束 |
| W4（`codex exec` 单任务） | 见 notes/dispatch.md | T16 | 已结束 |
| W5（`codex exec` 单任务） | 见 notes/dispatch.md | T18 | 已结束 |
| W6（`codex exec` 单任务） | 见 notes/dispatch.md | T19 | 已结束 |
| W7（`codex exec` 单任务） | 见 notes/dispatch.md | T20 | 已结束 |
| W8（`codex exec` 单任务） | 见 notes/dispatch.md | T21 | 已结束 |
| W9（`codex exec` 单任务） | 见 notes/dispatch.md | T22 | 已结束 |
| W10（`codex exec` 单任务） | 见 notes/dispatch.md | T23 | 已结束 |
| W11（`codex exec` 单任务） | 见 notes/dispatch.md | T24 | 已结束 |
| W12（`codex exec` 单任务） | 见 notes/dispatch.md | T25 | 已结束 |
| W13（`codex exec` 单任务） | 见 notes/dispatch.md | T27 | 已结束 |
| W14（`codex exec` 单任务） | 见 notes/dispatch.md | T30/T31 | 已结束 |
| W15（`codex exec`，astra/xhigh） | logs/codex/W15.log | T41 M1 调度补丁 120 | 已结束（Claude 审阅通过，8 卡 A/B 任务 014/015/017/018） |
| W16（`codex exec`，astra/high） | logs/codex/W16.log | T42 M3 分词补丁 130 | 已结束（Claude 核验：722/722 token 一致） |
| W17（`codex exec`，astra/xhigh） | logs/codex/W17.log | T43 sm80 indexer 融合 kernel 补丁 112 | 已结束（Claude 核验 summary PASS） |
| W18（`codex exec`，astra/xhigh） | logs/codex/W18.log | T44 预填充 indexer 逼近算力上限（补丁 113） | 已结束（Claude 核验 PASS，6.2–6.7×） |
| W19（`codex exec`，astra/xhigh） | logs/codex/W19.log | T45 M2 KDA 双点 fp32 快照（补丁 140） | 已结束（Claude 核验：数值逐位一致；待 8 卡 A/B） |
| W20（`codex exec`，astra/high） | logs/codex/W20.log | T46 启动期预热补丁 150 | 已结束（发现 112/113 按长度重编译 → T47） |
| W21（`codex exec`，astra/high） | logs/codex/W21.log | T47 112/113 去形状特化 v2 | 已结束（Claude 核验：50 随机形状 0 编译） |
| W22（`codex exec`，astra/xhigh） | logs/codex/W22.log | T48 M4 MTP/NEXTN sm80（补丁 160） | 已结束（Claude 核验；8 卡待测） |
| W23（`codex exec` 新会话，astra/xhigh） | logs/codex/W23.log | T49 缓存丢失根因 + 容量账 + 过时结论清单（R18） | 已结束（R18；Claude 已据此更正文档） |
| W24（`codex exec` 新会话，astra/xhigh） | logs/codex/W24.log | T50 DCP 前缀命中越界修复（补丁 116） | 进行中 |

**进行中**
- [Codex W24] T50 DCP 前缀命中寻址修复 — `patches/116-*`、`evidence/T50/`；将登记测试与补丁说明，不改其它实现。
- [Claude] 8 卡跑通：补丁 110（DSA indexer）+ 111（FP8 MoE→Marlin）+ tilelang 后端（F57）；pod 任务 008 启动探测、009 开发集 N6
- [Codex main] T37 当前A/B正式提交与验证证据链审计（只读）— `research/codex/R16_submission_evidence_audit.md`
- [Claude] 正式提交 45734（A）/45735（B）等待出分（约 18 小时）
- [Claude] L2 会话 `lh-arena-sess-a` 排队等卡；GPU 机 tmux `arena-daemons:l2` 守护进程自动跑 `tests/queue/`
- [Claude] 主线 M0：GPU 机装 base_exact + 带 DSA 的替身，实测 A100 DSA 后端（见 research/claude/base/00-summary-mainline.md）
- ~~[未决] D6 底包版本未知~~ → 已解决：底包 = 公开提交 fe236ea6c3 + 两处多模态修复，副本 `build/base_exact/`（F53/F54）

**已完成**
- [Codex main / T49] R18缓存/容量只读审计 — F79、`research/codex/R18_cache_loss_and_capacity.md`、`evidence/T49/`；19对真实LCP与批日志、启动显存账、101非末尾chunk漏角色点/140关系；50%/4%口径、全程92%、graph1.31GiB、链首25万例纠正。4项本地审计通过，逐节点驱逐/运行峰值/完整DCP地址验证交Claude；未操作GPU/8卡/pod。
- [Codex W22] T48 / 160 NEXTN sm80 — F69/D37、`research/codex/R17_nextn_sm80.md`、`evidence/T48/`；KDA回滚/DSA/共享index/采样/EH/mHC/Marlin算子与graph、10CPU及12补丁栈通过。MR32的8卡脚本仅准备；缓存落点偏差已纠正，两卡空闲，TP8/能力/SLO交Claude。
- [Codex W21] T47 112/113 v2去形状特化 — F68、`evidence/T47/`；88/222数值与12/30graph全过，16行配对最大+3.38%，50随机形状+600→601共208热调用零JIT；11补丁fuzz0/3623+6编译/反向还原，GPU空闲。
- [Codex W20] T46 / 150 启动期预热 — `patches/150-*`、`evidence/T46/summary.json`；21CPU+双Gloo、完整11补丁栈、A100同形状12组零新增JIT/bench，持久cache命中另测；新长度仍可编译，完整服务覆盖待Claude。F67/D36；GPU进程已退出。
- [Codex W19] T45 / 140 KDA双点fp32快照 — `patches/140-kda-dual-snapshot.*`、`scripts/make_140.py`、`evidence/T45/`；8组GPU（7组边界）最大误差0、21CPU、960轮off调度、完整补丁栈通过；722请求离线命中估算已交付。F66/D35，算子任务结束，8卡服务与性能验证交Claude。
- [Codex W18] T44 / 113 预填充indexer（叠加112）— `patches/113-sm80-prefill-indexer.*`、`evidence/T44/`；六档131–186等效TFLOPS/6.19–6.69×，222数值/30graph/9补丁栈全过，decode源码/PTX保留112。F65/D34，GPU任务已结束，L2交Claude。
- [Codex W17] T43 / 112 sm80融合indexer — `patches/112-sm80-indexer-kernels.*`、`evidence/T43/`；88数值对照/4种graph/整栈通过，graph decode4.75×/4.54×、prefill2.01–2.95×（单卡算子）；两卡已空闲，L2交Claude。
- [Codex W15] T41 / 120 调度保护：27 CPU测试、3617文件编译、off字节对照与回滚通过；默认decode交替/长chunk上限/短命中共享预算。GPU/SLO待测，未入队/启动服务 — `patches/120-sched-protect-chain.md`、`evidence/T41/`。
- [Codex W16] T42 M3 分词线程池与路由键 — `patches/130-async-tokenize.*`、`evidence/T42/`；14单测、722真实对话逐token全同；256733token loop-lag中位数194.57→8.59ms（本地CPU）。F60/决策31/M3，待Claude交叉审阅与8卡验证。
- [Codex] T40 本地 Claude Code 2.1.278 → 2.1.280；版本核验及 doctor 通过 — `evidence/T40/`
- [Codex main] T36实际底包探索 — `research/codex/R15_base_source_exploration.md`、F52、`evidence/T36/source_receipt.json`。102的单点替换/状态精度边界、103族/session/salt分离、104五文件依赖与负载保护；407文件与归档全匹配。底包重号F49已仅改F51；无补丁实现或实验。
- [Codex main] T35近期PR/issue复核 — `research/codex/R14_recent_pr_watchlist.md`、F50、`evidence/T35/github_status.json`；10条状态/正文、56960/40517完整diff，内部快照/DP亲和性/ReplaySSM候选及HiCache前置项；容量+14.7%仅统计修正，外部跑分非本赛结果。仅调研，无实验或引擎/服务变更。
- [Codex main] T34 AgentX / AMD MLPerf 双文精读 — `research/codex/R13_agentx_mlperf_reading.md`、F49；核对保留PR三阶段、调度/池源码和代码级候选，用户追加Plan B问题已纳入。无GPU/引擎/队列变更，不解除既有数值门。
- [Codex main] T26 / E2b与004审阅 — `patches/004-role-boundary-final-chunk.md`、experiments E2b、`evidence/T26/`。控制=旧v1.1，143项0退步且精确预测；reminder p95 8135→2439，strict67全同、unset=stock，flush池恢复。D1-01/02/03及N1 D1-08通过；004额外alignment major未修（CPU54pass+1已知失败）、旧D1-04 FAIL保持。自有GPU服务已停、两卡空闲，README交Claude；今日收尾后idle，不开新任务。
- [Codex W14] T31（T30）daemon→runner完整CLI/候选/矩阵/梯子/恢复对齐 — 166 CPU/mock通过、真实help预检与5项dry-run无BLOCKED，evidence/T31；无真实服务操作，STOP未触碰，live仍待验证。
- [Codex main] T29 E2遗留补测 — D0 §6、D1 §13、F47、`evidence/T29/`；IF-08/IF-03/D1-08/10限域通过，105条N4无错误/partial≤1；D1-05新准入计数相等但不覆盖续跑，整体门待明确，D1-02/04仍FAIL。含长配置调整/工具空跑误判纠正；50CPU通过，GPU服务已停两卡空闲，无Trisol/提交。
- [Codex main] T28 E1/E2注册表回填 — `tests/TEST_PLAN.md`、experiments T28：12个指定ID逐项映射，三套E2 case顺序核对一致；保留D1-02/04失败，澄清IF-08只过D0空闲子项及cold_heavy/并发/槽位/角色flush待测。仅读既有证据、未补跑GPU；索引与记录检查通过。
- [Codex main] T17 首次8卡计划审阅 — `plans/active/2026-09-22-first-8gpu-session.md`决策记录；认可T23勘误，补F45正确性资格/变体前置门、同patch控制、回滚与pod释放责任、预算和证据边界。建议尚待协调方落实；未改runner、未起服务。索引已刷新，check_records通过。
- [Codex W10] T23 会话 A整批runner — `scripts/session_a/`，26新CPU/mock+29梯子+38探针通过；D0基线、早期IF/CAP、N6/N10、SPF/D1矩阵、逐档SHA双落点回传与deadline/final清理。`evidence/T23/`；无Trisol服务动作，live待Claude带显式批准时间盒启动。
- [Codex W11] T24 自动提测构建 — `tests/TRISOL_TEST_DAEMON.md`；51项CPU/mock通过、纯dry-run，批准/STOP/单服务/预算/回收/评分/未批准提交队列已实现；T23扩展CLI与GPU产品ID待协调补齐，真实执行门保持关闭。
- [Codex W12] T25 I6 — `research/codex/R12_slo_aware_scheduling.md`、`patches/003-*`、F46/决策22；144 CPU tests与补丁链通过。896 MODEL OUTPUT中EDF未胜SPF，003默认关闭、live待验证；证据`evidence/T25_slo/`。
- [Codex main] T13 / E2 v1.1 — 设计§10、experiments E2、`notes/e2_d1/`、F42/F45。off=stock215项，D1无cache退步；reminder fast未命中p95仍8135，stock20 4750→3962；D1-02/04 FAIL（raw门失败，off也有数值差异），不是数值安全/可部署证明。GPU已释放，参考源码未改；未混入后续002/003/004。
- [Codex W9] T22 审批门控提交队列 / daemon — queue helper、双预检、上海日限额/CLI+API在途、持久化台账/成绩/事件流；59/59离线测试与全套启动/恢复说明。无 APPROVED、真实提交或 daemon 启动；Tools T22 / notes/t22_*。
- [Codex W7] T20 D2 SPF port — `patches/002-spf-scheduling.*`、F41、D2测试行；16生产+47模拟+15原D1 CPU测试通过（GPU机完整import同16/16），001→002组合fuzz0通过；560次模型对照中stock HRRN/LPM无SPF式ceiling增益。live验证仍待后续，无服务/Trisol/镜像/提交。
- [Codex W8] T21 独立全局审计 — `research/codex/R11_overall_summary.md`（1799字）；区分E1实测与源码/模型/推断，列本地用例、8卡及首提取证顺序、三大风险与判断分歧；仅文档，未运行服务。
- [Codex W5] T18 公开正式包络粗校准 — `research/codex/R10_sim_calibration.md`、F38、envelope-fit + 41 项通过 CPU 单测；MODEL OUTPUT：SPF 粗拟合区间 TTFT 先绑定，D1 无稳定 ceiling 增益，stock18 示例未升到22；保留统计余量、缓存寿命/eviction、prefill-decode 干扰等限制。仅 CPU，证据 `evidence/T18_calibration/`。
- [Codex W6] T19 IF/D1/CAP 覆盖工具 — `tests/T19_USAGE.md`、三个 live 脚本及 dry-run/preflight 集成；53/53 CPU/mock 测试，D1-07 15/15 pass；其余新增 live 覆盖待运行，保留 E1 stock IF-08 fail。无 GPU/Trisol/镜像/提交。
- [Codex main] T8/T9 目录归档与记录交接 — `research/codex/archive/README.md`逐项标记11个历史文件，迁移前后SHA256一致；只删除6个可再生成pyc。E1现有scripts不受影响，7项replay单测通过；根README建议摘要已写dispatch交Claude维护，决策#17。
- [Codex main] T7 / E1 stock替身基线 — `notes/experiments.md` E1最终节、`evidence/E1_stock/`、`scripts/replay_chains.py`；20链72请求/0错误，70个cache预测精确一致，2个偏乐观1088/3712tokens。IF-07 L2通过，原生flush仍非JSON。固定qfull权重/源码快照供E2；服务已停两卡空闲。回填D1计划E1里程碑与IF-07状态，追加F36和决策#16；T13/E2尚未执行。
- [Codex W3] T15 离线闭环模拟器 — `scripts/sim_closed_loop.py`、`scripts/test_sim_closed_loop.py`、`research/codex/R9_offline_simulator.md`；30 项 CPU 测试通过，576+144 参数/输出长度敏感性与 N6/N10 逐请求输出已落盘，F34；模型默认 TPOT 常先绑定，formal-mix 仅 what-if，待真实 stock N6/N10 校准。无 GPU/Trisol/提交。
- [Codex W1] T11 AgentX 检查表（66 行 / 130 个 PR）— `research/codex/R8_agentx_checklist.md`，F27–F30：decode 间隔本地已有（GLM 默认 0，+141% 是定长 GB300 数据）；本地没有 DP 缓存亲和路由（#26091 未合入）；hybrid HiCache 缺 DSA indexer 的问题仍在（#39156）；前端还有可优化的点（msgpack IPC、增量流式、分词 worker）
- [Codex W2] T12 `scripts/ladder_search.py`（正式爬坡 + 二分快速模式、每档严格检查 flush）、`scripts/score_formal.py`（估计版正式门）、29 项测试；[Claude] T14 flush 等待与重试
- [Codex W1] T11 / R8 AgentX 检查表 — `research/codex/R8_agentx_checklist.md`、`research/codex/R8_agentx_pr_inventory.json`；VERIFIED：66 行、130 PR，全部 120 个项目 tracker PR 链接已覆盖，flags/默认值及源码位置已核对；F27–F30。+141% 固定长度测试归因修正；仅调研，源码未改、未起服务。
- [Codex W2] T12 临界 N 搜索 / estimated scorer — `scripts/ladder_search.py`、`scripts/score_formal.py`、`scripts/test_ladder_search.py`；VERIFIED：29 项 CPU 测试通过，含原 run_dev 子进程 + synthetic loadgen，harness SHA256 未变；证据 `evidence/T12/tools_validation.log`，用法见 experiments「Tools」。无引擎/Trisol/提交。
- [Claude SA1] D0/D1 对抗性审查（没有 blocker；D1 有 2 个 major：关闭 chunked prefill 时崩溃、切分后停止准入导致串行化）— `research/claude/R5_patch_review.md`；已据此修订为 D1/D0 v1.1（patches §11/§12），补丁可以干净应用
- [Claude SA2] 提交包：`scripts/build_image.sh`（Dockerfile 6.3 KB，补丁在 scratch 上 dry-run 和实际应用都通过，失败时构建会中止）、`scripts/check_submission.py`、`submission/candidate-01.json`（10 个 flag 在源码里都存在）、`stub-trace.jsonl`。未决：底包里的 sglang 版本（补丁能否打上）、`/mnt/models` 挂载、glm45 parser 与工具调用
- [Claude] D0 v1 代码（89 行）：flush 返回 JSON、所有 worker 都成功才算成功、ASGI 入口记录服务端收到时间 — `patches/000-interface-compliance.patch`
- [Claude] D1 v1 代码（role_conservative，112 行，只改 schedule_policy.py，用环境变量开关）— `patches/001-role-boundary-mamba-ckpt.patch`，设计 §9；交 Codex 审阅（T13）
- [Claude] R4 类似比赛/基准调研（AgentX、ClawPerf、MLPerf Interactive、ASC）— `research/claude/R4_similar_competitions.md`；F33 arena 队列：每人最多 8 张 A100
- [Codex] T10 / R7：vLLM #50587 → #52789/#53614 与 SGLang KDA 单 forward 内部快照 — `research/codex/R7_kda_internal_checkpoints.md`，F25–F26；本地已有单点 FP32 累加器导出，双点缺多位置/conv/槽位及入树生命周期；普通 BF16 h 不可冒充无损 FP32 源。已纳入 F24，维持 A 优先；仅调研，未实现/运行。
- [用户] 已加入 arena team（配额：w1 上 400 张 A100 团队共享；当前排队较多）
- [Claude] F24：D1 运行时策略对比，结论是保守变体（有 branch 冲突就跳过）保留全部收益 → D1 v1 = role_conservative
- [Codex] R6 中文社区 / 六个 GitHub 仓库先例 + D0 rid 只读核对 — `research/codex/R6_prior_art_cn_github.md`；F22：llama.cpp #22929 已合并最后 user 边界切分快照；F23：公开 dev 跨阶段复用 rid，且 runner 不校验 flush JSON / 忽略失败返回。目标 A100/H100 闭环多轮调优数字仍缺完整证据。仅调研，未运行实验或改 harness；请 Claude 将 §6 结论纳入 D0 §2.4。
- [Claude] 复现评测 / 打镜像 / 提交全流程 — `research/shared/pipeline.md`（待 Codex 审阅）
- [Codex] D0/D1 定向二审 — D0 §5.1、D1 §8.3–8.5；确认新 chunk 不自动 break，lazy 原生 prefill donation 有条件可复用但未验证；新增 F19–F20，仅源码/文档。
- [Codex] 确认 §G v2 分工，D3 显存账改为独立 R5；说明与交付已追加 `research/shared/directions.md` §H。
- [Codex] P0 D6 访问路线 — `research/codex/R3_base_images.md`；未拉取、未访问账号镜像目录，未确认实际底包内容。
- [Codex] P1 D2 移植风险与 D1 冲突 — `research/codex/R4_prefill_scheduling.md`；核对两 PR 的当前依赖和 head，无 apply-check/实验。
- [Codex] P2 DP1/2/4 静态显存账 — `research/codex/R5_dp_memory_accounting.md`；源码/公式与条件算例，非实测。
- [Codex] D0/D1 设计交叉审阅 — 只追加至 `patches/000-interface-compliance.md` §5、`patches/001-role-boundary-mamba-ckpt.md` §8；事实追加 F14–F18。
- [Codex] 响应 Claude 的 HiCache / D1 / D6 定向审阅（仅调研）— `research/codex/R2_review_shared_directions.md`；新增源码事实 F12，无实验。
- [Codex] 协作对齐、F3/R2 独立审阅、Claude R1 / shared 清单交叉审阅（仅调研）— `research/codex/R1_directions_and_review.md`、`research/shared/directions.md` §E；事实追加 F7–F11。
- [Claude] R1 调研 — `research/claude/R1_model_and_engines.md`
- [Claude] 方向清单 — `research/shared/directions.md`
- [Claude] R2 调研 — `research/claude/R2_serving_techniques.md`
- [Claude] F3 分析 — `notes/findings.md`, `scripts/analyze_divergence.py`, `scripts/sim_checkpoints.py`
