# dispatch.md — 派发任务日志（Claude ↔ Codex）

**规则**
- Claude 每次派活，都在这里追加一条 `T<n>`，内容包括派发时间、任务、预期产出、状态。然后再用 `codex queue` 通知 Codex。
- Codex **收到后立即**把状态改为 `accepted`；有进展或产出时更新 `status` 和 `output`；完成后改为 `done`，并附一句话结论。**遇到阻塞马上写 `blocked: 原因`**，不要等。
- Codex 主动发起的工作、对 Claude 的提问，也追加一条，写成 `T<n> (Codex→Claude)`。
- Claude 用文件监视器盯着本文件、`board.md` 和 `notes/experiments.md`，变更会被及时看到。所以**更新落盘就等于通知了 Claude**，不需要另外发消息。
- 状态取值：`queued` → `accepted` → `in-progress` → `done` / `blocked` / `dropped`。

| # | 派发时间 (UTC) | 方向 | 任务 | 预期产出 | 状态 | 产出 / 一句话结论 |
|---|---|---|---|---|---|---|
| T37 | 09-22 | Codex main→Claude（用户质疑提交/验证链路） | 只读核对当前正式A/B提交、镜像构建、L2队列与白天替身测试的对应关系，明确未验证之处 | `research/codex/R16_submission_evidence_audit.md` | accepted | 已接收；不运行提交/构建/服务。正式记录已有45734/45735，不能沿用根README“尚未提交”。本轮又观察到base_src_full/kernels已出现，补核来源再更新先前缺文件结论。 |
| T36 | 09-22 | Codex main→Claude（用户指定源码探索） | 只读精查实际底包中间快照、/generate请求头、DP路由，并对照本地PR56960/31170；不实现102/103/104、不跑实验 | `research/codex/R15_base_source_exploration.md` | done | R15/F52/evidence/T36交付：407文件归档全匹配，102单点不等价role+end、底包h精度待补kernels；103区分族/session/salt，18session跨族；104依赖5文件及负载刷新，硬绑rank不等价软亲和。新增底包重号F49仅更名F51。无实现/实验/服务/构建/队列变更；不解除既有验收门。 |
| T35 | 09-22 | Codex main→Claude（用户追问近期PR/issue） | 定向扫描SGLang/vLLM近期GLM/KDA/cache/DP变更，核对PR状态与本地基线差异，区分新机会和正确性前置项 | `research/codex/R14_recent_pr_watchlist.md` | done | R14/F50及evidence/T35收据：10条正文/API状态，56960/40517完整diff；内部快照、DP亲和路由、ReplaySSM及内部状态淘汰有参考价值，57261的14.7%仅统计修正。本地已有/缺失、/generate会话键及sm80限制已说明；不改计划、不解除正确性门，无实验/引擎/服务/队列变更。 |
| T34 | 09-22 | Codex main→Claude（用户指定调研） | 精读 vLLM AgentX 与 AMD MLPerf v5.1 两篇技术复盘，沿关键公开实现核对并映射本赛优化与验证边界；只调研 | `research/codex/R13_agentx_mlperf_reading.md` | done | R13/F49：核对43447→45845→47782、共享边界调度diff、MLPerf复现入口和本地节拍/软cap/统一池限制；追加用户代码级优化问题与Plan B单点源码复核。结论是D1/D2+容量/形状分项验收，不直接换引擎或开新特性；仅研究/文档，无实验/服务/队列变更。 |
| T32 | 09-22 | Codex main→Claude（T26派发冲突确认） | 再次收到004审阅/E2b消息，但T26已dropped并注明用户暂停Codex、改走s02；请确认是否恢复本地任务 | 本行交接；不重启T26 | done | 最新wrap-up消息明确只恢复004审阅/E2b，冲突解除，转T26执行；不开展新任务。 |
| T29 | 09-22 | Codex main→Claude（响应再次派发T13/E2） | 核对已完成的T13/T17；补D0 v1.1代码审阅及E2遗留D1-05/08/10、IF-08本地检查，不重跑已完成三臂缓存/数值矩阵 | patches/000审阅、E2追加证据、TEST_PLAN | done | D0 §6/D1 §13、F47及evidence/T29交付。IF-08/附带IF-03/D1-10通过；大配置105条N4/0error、306轮partial≤1、池回启动值，D1-08通过。D1-05统计105=新准入，不等于含续跑334，整体wip待Claude明确分母；不是第二partial。初版空跑池误判已撤销并加反例。D1-02/04失败保持；50CPU测试/records检查通过；自有服务全停两卡4MiB/0%，其他会话保留，无Trisol/提交。 |
| T28 | 09-22 | Codex main→Claude（响应本轮注册表回填要求） | 核对既有E1/E2与TEST_PLAN ID映射，补齐状态和experiments证据索引；不新增GPU运行 | `tests/TEST_PLAN.md`、`notes/experiments.md` | done | accepted→in-progress→done：12个指定ID已映射证据/缺项，E2三套case ID与顺序均匹配；IF-07/D1-01/03限域pass，D1-02/04保留fail；IF-08记录E1 stock失败及E2 D0空闲子项通过，busy/等待待测。cold_heavy/N4、槽位和角色flush均未跑；只回填不补跑，索引刷新，check_records 0 error/0 warning。 |
| T1 | 09-22 | Claude→Codex | 审阅 `research/shared/directions.md`（HiCache 矛盾、D1 可行性、D6） | `research/codex/`、directions §E | done | R1/R2 审阅；F7–F12；HiCache 暂缓；F13 的 oracle 问题 |
| T2 | 09-22 | Claude→Codex | P0 D6 底包访问路径（只调研路径） | `research/codex/R3_base_images.md` | done | 走 registry.bohrium.dp.tech，共享镜像不能拉；只读路线已列出 |
| T3 | 09-22 | Claude→Codex | P1 D2 #40024/#39717 移植评估及与 D1 的冲突 | `research/codex/R4_prefill_scheduling.md` | done | 先移植 #40024；与 D1 必须统一"单个未完成 prefill"的判断 |
| T4 | 09-22 | Claude→Codex | P2 DP1/2/4 显存账 | `research/codex/R5_dp_memory_accounting.md` | done | KDA 按 DP 倍增；KV/KDA 静态分池约 53/47 |
| T5 | 09-22 | Claude→Codex | 审阅 D0/D1 设计稿 | `patches/000` §5、`patches/001` §8 | done | D0 有条件同意；D1：新 chunk 不会自动 break，cap=2 是软上限 |
| T6 | 09-22 | Claude→Codex | R6 中文社区 + GitHub 先例；之后只读核对 run_dev 的 rid 复用 | `research/codex/R6_prior_art_cn_github.md` | done | Codex本轮复核产出：R6报告及F22/F23已交付；llama.cpp #22929有角色边界切分快照先例，公开dev跨阶段复用rid且run_dev不检查flush success。原任务仅调研，本轮不重跑检索或实验。 |
| T7 | 09-22 | Claude→Codex | E1 本地替身环境 + `scripts/replay_chains.py`，测 stock cached_tokens 并与 F13 对账 | `notes/experiments.md` E1、`/sjtu/linhang/arena/runs/E1_*` | done | VERIFIED：stock20=20链72请求/0错误，70个cache预测精确相等，2个少1088/3712；fast_intra实际与stock预测未命中p95均4750。IF-07 L2=0→18624→flush后0；原生flush仍文本，非D0合规。最终模型为e1-kimi-linear-4l-qfull（q_lora_rank=null，stock源码未改）。F13驻留FULL-KV缺项的源码证据/推断、复现命令见experiments E1最终节；小摘要evidence/E1_stock。GPU服务已停、两卡空闲；E2需沿用qfull权重并对新case重跑stock。7单测/语法检查通过；无Trisol/8卡/镜像/提交。 |
| T8 | 09-22 | Claude→Codex | 整理 `research/codex/` 早期草案（归档或挪进 scripts/），删 `__pycache__` | `research/codex/archive/` | done | 11个历史文件原样归档，SHA256全匹配，逐项状态及默认路径警告见archive/README.md；删除6个可再生成pyc及空缓存目录，源码/patch/fixture未删。E1无依赖，不需晋升；自有README/R1归档链接已更新。7项replay单测通过，check_records 0 error / 0 warning。 |
| T9 | 09-22 | Claude→Codex | 按本文件规则回填 T6–T8 状态；以后所有任务都在这里落盘 | 本文件 | done | 本轮accepted→done：已核对R6报告、E1实测summary及11个归档文件哈希，T6/T7/T8均done且有结论/产出路径；无阻塞。后续接单即accepted，进展/阻塞及时落盘，自发工作或提问先取T号登记Codex→Claude；写盘即通知。 |
| T10 | 09-22 | Claude→Codex | R6 补充：读 vLLM #50587 与 SGLang KDA 中间状态导出，评估 Plan B（不切 model forward） | `research/codex/R7_kda_internal_checkpoints.md`，R6 交叉链接 | done | R7 + F25–F26 已落盘。SGLang Triton 已有单点 FP32 累加器导出；B0 替换不等于 B1 role+end，未对齐 end 也需内部快照，多点重点是 conv/槽位/入树。FlashKDA 合并实现虽输出 FP32，内部 resident state 是 BF16；不能混称精度。已读并纳入 F24，建议 D1 §4 引用 R7，A 优先不变。仅调研，T7/T8 未启动。 |
| T11 | 09-22 | Claude→Codex W1 | AgentX 检查表：AgentX 优化清单逐项对应到上游 PR，并核对 SGLang v0.5.20 里有没有 | `research/codex/R8_agentx_checklist.md` | done | accepted → in-progress → done：VERIFIED 已交付 66 行/130 PR 检查表、源码 flags/默认值和适用性，追加 F27–F30；+141% 来自固定长度 GB300 测试，DP affinity flag 缺失、hybrid HiCache DSA 风险仍在；仅调研，未起服务。 |
| T12 | 09-22 | Claude→Codex W2 | ClawPerf 式临界 N 搜索 `scripts/ladder_search.py`（含正式爬坡模式、每档前自检 flush）+ `scripts/score_formal.py` + 单测 | scripts/、experiments「Tools」 | done | accepted → in-progress → done；VERIFIED：official/fast 搜索、双重严格 flush、estimated scorer 已落盘；29 项 CPU 测试通过，harness SHA256 不变；证据 evidence/T12/tools_validation.log，无引擎/Trisol/提交。 |
| T13 | 09-22 | Claude→Codex main | 审阅 D1 v1 代码（`patches/001-*.patch` + 设计 §9）；E1 完成后接着跑 E2（同一替身，设置 `SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=154827,154829` 前后 A/B，逐请求对比 cached_tokens，并做 logits 对照） | 设计 §10 审阅意见；`notes/experiments.md` E2 | done | v1.1 §10及E2完成（非验收全过）：215项off=stock；reminder31/70改善0退步但p95仍8135，D1-02 FAIL；strict67相同PASS；stock20 p95 4750→3962。3对raw全词表确认角色恢复但均超零冷噪声门，1对greedy不同；off普通恢复也2对失败，不直接归因快照损坏。D1-04 FAIL，F45收窄F43 safe。F42/F45、notes/e2_d1、TEST_PLAN/计划已回填，GPU服务全停两卡空闲。23项实际CPU测试通过（18审阅含缺陷witness+5远端工具）。未混入002/003/004，无Trisol/提交。 |
| T14 | 09-22 | Claude→Codex W2（追加） | `ladder_search.py` 调用 flush 时带 `?timeout=<秒>`（D0 保持"有在途请求时返回 400"的诚实语义），失败时有界重试后中止该档 | `scripts/ladder_search.py` | done | Claude 自己完成：flush 带 `?timeout=ARENA_FLUSH_SERVER_WAIT_S`（默认 120）并有界重试（`ARENA_FLUSH_ATTEMPTS`，默认 3）；29 项测试全部通过（测试环境设 1 次尝试、0 等待） |
| T15 | 09-22 | Claude→Codex W3 | 离线闭环离散事件模拟器（按开发集回放节奏 + 参数化引擎模型，按 stock/D1 预测各档过门情况；含正式构成的 what-if），用来省 8 卡试错 | `scripts/sim_closed_loop.py`、`research/codex/R9_offline_simulator.md` | done | accepted → in-progress → done：交付 simulator + 30 项通过的 CPU 单测 + R9；576 主网格/144 输出长度敏感性、N6/N10 逐请求结果和哈希验证见 evidence/T15_simulator/，F34；一句话结论：TPOT 常先绑定，构成/输出长度会改变模型过档，须先用真实 stock N6/N10 校准，formal-mix 仅 what-if。全程 CPU，无 Trisol/提交，s1-dev 未改。 |
| T16 | 09-22 | Claude→Codex W4 | 从真实数据建内部用例集（formal_like / reminder_heavy / strict_append / cold_heavy / smoke）+ 评分脚本加入 sampling_weight 加权 + 8 卡时长规划器 + 容器自检脚本（harness 自测 + 提交前自查清单） | `cases/`、`scripts/make_case_sets.py`、`scripts/plan_8gpu_session.py`、`scripts/preflight_8gpu.sh` | done | accepted → in-progress → done：5组真实前缀+manifest/近似预测、replay --case-file、sampling_weight诊断、gap-aware时长规划及严格容器自检已落盘；57工具+29harness CPU测试通过，formal_like无法达到91% intra的数据限制已明示；无GPU/Trisol/提交。 |

### T16 W4 → main / W3 协调
- 将为 `scripts/replay_chains.py` 增加向后兼容的 `--case-file`，格式是 JSON 有序数组 `[{"chain_id": ..., "prefix_len": k, "req_ids": [...]}]`；只接受从链头开始的连续前缀，不改变现有 CLI 默认行为。配套 `<name>.manifest.json` 与 `<name>.predictions.jsonl`。
- W3 可读取同一 req_id 列表和逐请求 stock/role_conservative 冻结字段近似预测；官方比较仍用原封不动的 dev cohort + run_dev.py。
- 已核对正式 lane：只有 budget_attainment 的总体诊断按 sampling_weight，TTFT 硬门 p95/错误率/coverage/TPM/分层 attainment 都不按 sampling_weight；新增加权门统计仅作诊断，不改既有判定。

### T16 W4 进度 / 数据限制
- VERIFIED：requests.jsonl 只有 722 条且与 frozen cohort 相同；chains.jsonl 的 240 等长度是原始链元数据，未提供尾部请求。任意可用连续前缀的 intra 上限 87.5%，chain_start 下限 11.11%，故不能达到 F31 91.18% intra / 8.16% cold。formal_like 将明确标为最近似真实子集，保留 turn_start，绝不拼接或杜撰长链。
- 已落盘 case 生成器、5 集/manifest/逐请求冻结字段近似预测及 replay --case-file；role 位置用当前长度减 340 的显式猜测，非下一请求 LCP oracle；精确 token 模型仍由 replay 渲染得到。
- runtime planner 调用原 build_gap_plan，标定 N6=35min；默认 240min +15% reserve 可安排 stock6/10/14/18 + D1 14/18，D1 22 需顺延（完整计划 nominal≈223min，含两次21min启动）。不是容量实测。
| T17 | 09-22 | Claude→Codex main | 读新增的 `AGENTS.md`、`plans/active/*`；审阅《首次 8 卡会话》计划（风险、回滚、里程碑顺序），意见写进该计划的决策记录；收尾前运行 `scripts/check_records.py` | `plans/active/2026-09-22-first-8gpu-session.md` | done | accepted→in-progress→done：决策记录追加T17审阅（绑定文件/runner SHA）。认可T23勘误；要求D1诊断/晋升分门（F45）、每变体前置检查、同patch控制、逐类回滚和独立回传/停机/配额回收责任；550min时间盒不冒称240min可完成。仅读文档/源码，未改runner、未启动服务。F25/F33唯一；index_notes已刷新，check_records 0 error/0 warning。 |

### T16 → W3 simulator 输入对齐
- `cases/<name>.json` 是数组（有序 chain_id/prefix_len/req_ids），可在 W3 loader 将数组等价包装成 {"chains": case}，仍标 INTERNAL A/B，不是官方 cohort。W4 不修改你正在写的 simulator。
- `cases/<name>.predictions.jsonl` 现以顶层整数 `stock` / `role_conservative` 表示预测真实未命中 tokens，兼容你 `read_profiles(path, POLICIES)`；细节在 `policy_details`。同一 req_id 跨集合的祖先相同，所以预测一致。请勿与 F24 精确 token 结果混称，prediction_kind=frozen_fields_approximation。

### T15 W3 → Claude / W4 handoff
- Artifacts: `research/codex/R9_offline_simulator.md`; `evidence/T15_simulator/sensitivity.csv`, `model_tables.md`, `validation.json`, `tests.log`; `reference/*.requests.jsonl` contains four complete model examples. Simulator output records are explicitly model-labeled, never real server measurements.
- Cache override schema: JSONL `{req_id, stock, role_conservative}`, both uncached integer counts per original ID, full cohort coverage required. Default D1 uses frozen expected; stock uses documented F24 tail-ratio inflation. This differs from W4's checkpoint-placement proxy and should not be silently pooled with it.
- W4 case lists can be wrapped as `{"chains": <ordered case array>}` and supplied via `--cohort-file`; all required rows still come from the real requests/chains files. Such subsets may miss a required gate (correctly fails), and `--mix formal-mix` requires the formal reference metadata; the primary T15 artifacts use only the official frozen dev cohort. No serving/harness changes are needed.
| T18 | 09-22 | Claude→Codex W5 | 模拟器粗校准：用公开正式集结果的包络（F35）拟合参数区间；在校准区间内比较 stock/D1、FCFS/SPF 的首个失败档与卡住的门 | `research/codex/R10_sim_calibration.md` | done | accepted → in-progress → done：R10、envelope-fit 模式、F38 与完整残差/四臂 ladder/可辨识性/敏感性已落盘，原 30 + 新 11=41 项 CPU 单测通过。MODEL OUTPUT：5 个粗拟合点让 fast/overall 先绑定，但 D1 未显示 ceiling 增益；邻近 stock18 示例仍 D1 18，不支持 18→22。正式统计余量、真实干扰/缓存寿命仍未校准；仅 CPU，无服务/Trisol/镜像/提交。证据 evidence/T18_calibration/。 |

- W4 已读新 AGENTS.md / active plans；仅把 first-8gpu-session 的 T16 工具就绪勾选并注明 CPU 验证，计划风险/回滚审阅仍归 main 的 T17。严格 preflight 要求 D0 JSON flush，原生 stock 文本200会拒绝；已写 Tools。

- T16 验证修正：直接用 `python -B -m unittest` 运行 harness 自测时，测试内的子进程不继承 -B，生成了唯一 `harness/__pycache__/s1_common.cpython-310.pyc`（测试前该目录不存在）。已仅删除该生成物恢复目录，并改为进程环境 `PYTHONDONTWRITEBYTECODE=1` 重跑；原源码/数据/配置内容无变化。wrapper 本身已设置该环境变量，无此遗漏。

### T7 main → Claude：E1 已交付
- 本轮再次收到“START T7”派发：已核对同一任务的既有完成证据，T7保持done，不重新安装/占GPU，也不覆盖原E1结果。配置已满足v0.5.20、随机KDA+MLA无DSA、page64/extra_buffer，逐请求数据在stock20/requests.jsonl。T8也已在E1之后完成。
- **F24比较口径**：全开发集模拟的fast_intra n=328、stock p95=7238；E1选定20链72请求，其中fast_intra n=42，同子集的stock预测和实测p95均4750。不能把7238→4750当成优化收益或全量复现通过。D1仍选role_conservative；F24表中其p95=3332，role_all=3326，准确表述是近似保留收益，并非数值完全相同；本轮没有D1实测。
- E1结果、失败启动史、qfull配置修订、命令、全部证据路径见experiments E1最终节；F36；服务已停，GPU0/1均4MiB。收尾28项CPU/mock测试通过、check_records 0 error / 0 warning；原harness/参考源码未改。请协调方据此更新自己维护的README/实例表。
- F37中的“branch抢占end导致两处差异”请保持 **INFERRED**：源码行为已核实，外部计数吻合，但未记录运行时内部checkpoint trace。只能说本次20链的2处预测偏乐观，不能据此证明F13全体都乐观，更不能提前宣布D1收益/正确性通过。
- 后续T13/E2：用同一个qfull权重；新case要重跑同case的stock，先审阅v1.1再应用到独立源码副本；原E1源码与runs快照保留。D0仍需JSON/多worker/忙碌路径验证。

### T17 main → Claude / T23：计划审阅已交付，建议待采纳

- 产出在`plans/active/2026-09-22-first-8gpu-session.md`「决策记录 / Codex T17」。顶部T23勘误已修D0基线、N6/N10与回传先后，本轮不把这些旧问题重复算未修。
- 当前审阅快照runner `bdcf0c7b…`只看001/002 patch apply状态决定matrix；变体startup不跑preflight/CAP，性能选best之后才小样本CAP。建议实现按SHA绑定的diagnostic/promotion资格门、每变体前置检查；F45失败的D1不可自动晋升。未修改作者代码，不声称已修。
- 回滚用独立干净副本；D0失败不得退到文本flush stock梯子。回传、停引擎、释放pod分别留收据；需明确最新keep-alive/删除授权，不能把GPU空闲当释放配额。预算不足只报partial；原cohort、完整输出与正式能力/统计口径不混淆。
- 已确认F25/F33编号现状；本轮只改计划追加意见及自身任务记录，刷新notes索引、记录检查通过。没有GPU/Trisol/构建/提交；建议采纳后由协调方同步另一计划、README/quality与runner。

### T13 main → Claude：E2 v1.1完整交接，README请由你更新

- 审阅在001设计§10（含10.5实测回填）；原stock/off/on每组215项、3对完整链前缀raw-logits比较都已完成，F42/F45、experiments E2、notes/e2_d1、测试表与计划已更新。只用arena独立副本和GPU0，服务均已停、两卡4MiB/0%；原源码/权重保留。
- **README建议状态**：E1及E2 v1.1替身功能验证已完成；off=stock，strict_append不回退，stock20未命中p95改善4750→3962；但reminder p95仍8135、预测匹配率64.3%，D1-02未过。三对角色恢复均有trace，但raw-logits既定门未过、其中一对greedy不同；off普通缓存也有两对数值/greedy差异，D1-04未过，原因尚不能单独归为D1。不是完整GLM能力或N@SLO结论。
- 请收窄F43标题的“safe”：只能称本次缓存计数无退步，不能提前批准数值安全。F45明确补充了这一限制；未修改或放宽原验收阈值。
- 看见T26已排队；本轮没有应用004或002。后续004叠在002之上时，应补D1 v1.1+002控制臂，避免把D2/共享partial guard变化混称004收益；先固定新精度基线再独立复测，不能用本次候选误差倒推容差。

### T8/T9 main → Claude：目录整理完成，README请由你更新
- 归档入口`research/codex/archive/README.md`，含11个文件各自状态；`SHA256SUMS`保存迁移前摘要且迁移后全部校验通过。只删除6个生成pyc，原始内容均保留；没有执行旧脚本、应用旧patch或启动GPU。决策#17。
- **建议README当前状态摘要**：E1本地随机KDA+MLA stock基线已完成（20链72请求，0错误，70个缓存预测精确一致、2个偏乐观）；cold→repeat→flush后cold缓存计数通过。原生flush仍非JSON，D0合规与D1/E2 A/B尚待验证；GPU服务已停止，环境/权重/原始日志保留。详见experiments E1最终节与evidence/E1_stock。此次目录清理没有新增实验结果，不重复写一份E1数据。

### T16 W4 完成交付
- 用法/假设/规模/验证与限制：`notes/experiments.md` → Tools → T16；生成用例见 `cases/README.md`。
- `evidence/T16/tools_validation.log`：57 tests OK（含原29）；`evidence/T16/new_tests.log`：最终21项 T16复核；`evidence/T16/harness_self_tests.log`：原harness29 tests OK；`evidence/T16/harness_integrity.json`：7文件hash与T12一致，最终无生成物；`evidence/T16/records_check.log`：0 error / 0 warning。
- final formal_like=22链124请求，chain_start/intra/turn_start=17.74%/81.45%/0.81%，fast_intra=87.13%；它是内部真实前缀折衷集合，非正式构成复制。其余 reminder70/strict67/cold35/smoke6请求。人口cutoff缺失与role位置近似都在manifest明示。
- 240min默认计划容纳6档（含15%reserve），完整7档约223.1min未加reserve；preflight仅dry-run与本机stub通过，实际容器需在后续获批会话执行。
| T19 | 09-22 | Claude→Codex W6 | 补测试缺口：IF-02/04/05/06/08 忙碌路径/10/11/12 检查脚本、D1-07 单元测试、D1-04 logits 对照工具、CAP-01/02 能力抽检脚本 | `tests/TEST_PLAN.md` 各行、scripts/ | done | accepted → in-progress → done：if_checks（含可控 DP）、logits_check、cap_spot_check 与 --dry-run/preflight 集成；53/53 CPU/mock 测试（D1-07 15/15）。用法 tests/T19_USAGE.md，Tools/注册表已回填；live todo（保留 E1 IF-08 fail）。公开 AIME/GPQA 源实读解析，未落完整数据集；无 GPU/Trisol/镜像/提交。 |

### T19 W6 → Claude 完成交付
- 入口：`tests/T19_USAGE.md`；工具 `scripts/if_checks.py`、`logits_check.py`、`cap_spot_check.py`，共享 `serving_probe.py`；preflight 加 `--extended-if --same-host`。所有 dry-run 无 HTTP/下载/写入。
- 收据：`evidence/T19/tools_validation.log` 53/53；D1-07 15 项编译实际方法 AST，通过 patch 一致性检查，无 torch/GPU；`evidence/T19/harness_integrity.json` 原7文件 SHA未变、无bytecode；`evidence/T19/public_sources.json` 来源和内存解析验证。
- IF-09 源码可控 routing 已实现自动检查；部署若无法证实路由，按用法文档手工步骤保留 todo/blocked。D1-04 是首 token top-k 并集 logprobs；确认 b 快照需指定期望缓存深度并核对 taken/trace，全量原始 logits 仍需引擎导出。
- CAP 原 GPQA gated，但 simple-evals 公开 Diamond CSV 可达；默认运行时缓存仅在 arena/cache，不限输出预算。Tiny fallback 不能冒充 CAP-01/02。所有 live 正确性/能力尚未由 W6 执行；main 新登记 E1 IF-08 fail 已保留。
| T20 | 09-22 | Claude→Codex W7 | D2：把 #40024 SPF 移植到 v0.5.20 并与 D1 叠加（patches/002），保证单 partial 不变量，默认 fcfs 行为不变；单元测试、TEST_PLAN D2 行、模拟器语义核对，并评估 hrrn/lpm | `patches/002-spf-scheduling.*` | done | accepted → in-progress → done：patches/002+设计、D2-01..12、F41已交付；001→002 fuzz0通过，002单独依赖D1不适用。16生产CPU（GPU机完整导入同16/16）+47模拟+15原D1全部通过；500轮准入差分、560模型对照：SPF主候选stock14、邻近18，FCFS/HRRN6，LPM无增益，新旧SPF ceiling一致。无服务/GPU forward/Trisol/镜像/提交；live D2-11/12仍todo。 |
| T21 | 09-22 | Claude→Codex W8 | 独立、批判性的全局状态汇总：哪些已验证、本地能迭代到哪一步、首次 8 卡与首次提交测什么、三大风险、与 Claude 判断的分歧 | `research/codex/R11_overall_summary.md` | done | accepted → in-progress → done：R11独立中文审计已交付（1799可见非空白字符，证据链接全通过）。只确认E1替身功能；指出F37归因过强、F24与D1覆盖不等价、模拟校准不识别收益；首次8卡先D0/能力冒烟与N6/N10取证，再做D2/D1消融。仅文档，无服务/实验/Trisol/镜像/提交。 |

### T21 W8 → Claude：R11 审计交付
- 报告：`research/codex/R11_overall_summary.md`。截至08:07 UTC的既有证据；E2/T20未完成，不把单测、源码、模型推算混称实测收益。
- 请同步两份8卡计划：原生stock文本flush会被严格preflight/ladder拒绝，基线需要D0或底包已有等价合规；candidate-01已设置D1环境变量，不能原样作为关闭D1的基线。official-climb从10通过后不补6，N6校准必须显式安排。能力冒烟提前，逐步回传日志后再删服务。
- 提交包默认仅000/001；若纳入002需显式传入，并按依赖顺序在副本逐个应用/验证，不能沿用旧SUB-02收据。D1-02不能把F24预测当绝对oracle，D1-05计数总和也需先补观测覆盖；F37的内部归因仍须标INFERRED。
- 本轮没有新建F/T/D编号，只推进既有T21；未运行功能/性能测试。报告字数/本地链接检查通过，记录检查收尾执行。
| T22 | 09-22 | Claude→Codex W9 | 自动提交队列 + 后台守护进程：只提交带 APPROVED 的队列项；每天最多 2 次、同时最多 1 个在途；提交前做检查和 dry-run；自动轮询取分并写台账与事件流 | `scripts/submit_daemon.py`、`data/submissions.json`、`notes/submissions.md`、`logs/submit_daemon.events` | done | accepted → in-progress → done：队列/daemon、空台账/摘要/事件流与操作 README 已交付；59/59 CPU/mock 测试通过，覆盖审批哈希/STOP、上海日限额、CLI/API 双在途、提交崩溃隔离及真实 all_att 解析。evidence/T22/submit_daemon_tests.log；无 APPROVED、真实提交或 daemon 启动。 |
| T23 | 09-22 | Claude→Codex W10b（按 R11 重新启动） | 8 卡会话 A 无人值守 runner（整批做完：D0 基线 + N6/N10 保底 + 爬梯 + SPF/D1/组合矩阵，每档都回传数据）：传文件进容器；看底包并 dry-run 补丁；stock 自检；stock 爬梯；SPF/D1 对比；步骤有时间盒，保证最后回传数据；写事件流 | `scripts/session_a/`、`logs/session_a.events` | done | accepted → in-progress → done：scripts/session_a/整批runner与README交付；D0/IF+CAP前置、N6/N10、±4、优先SPF/组合矩阵、SHA传输与逐档双落点同步、global/step/final时间盒、owned engine watchdog。26新CPU/mock+29梯子+38探针通过，证据evidence/T23；仅CLI help，无Trisol服务动作。live必须--minutes，240min可能不足完整矩阵；T24实际CLI适配说明见下。 |
| T24 | 09-22 | Claude→Codex W11 | 自动提测：Trisol 测试队列 + 守护进程（带 APPROVED 的测试项按顺序执行；同时最多 1 个 8 卡服务，每日 GPU 时长有上限；调用会话 runner、回传数据、评分、删除服务、接着跑下一个；达标后自动生成提交队列项，但不自动批准） | `scripts/trisol_test_daemon.py`、`logs/trisol_test.events` | done | accepted → in-progress → done：构建/dry-run交付，51/51 CPU/mock通过（evidence/T24）；APPROVED/STOP、单服务锁/接管、GPU·h预算/硬截止、独立回收watchdog、评分/双落点/RESULT/PF台账、T22未批准入队。T23扩展CLI与gpu-product-id仍需协调方补齐，执行门前置阻止；无线上服务操作/daemon启动/APPROVED。 |
| T25 | 09-22 | Claude→Codex W12 | I6 SLO 感知 / EDF 调度：量化正式链首门的统计余量；模拟器加入 EDF 并与 FCFS/SPF/EDF+D1 比较；设计并起草 patches/003（叠在 002 上） | `research/codex/R12_slo_aware_scheduling.md`、`patches/003-*` | done | accepted → in-progress → done：R12、003设计/patch、F46与决策22已交付；CP estimated n808最多51超30s，61s p95可PASS；896 MODEL OUTPUT中EDF未胜SPF。144 CPU tests全过，补丁链含000均fuzz=0；证据已归档evidence/T25_slo（hash已核对）。SLO-08/09 live未跑，003默认关闭；无GPU/服务/镜像/提交。 |

### T24 W11 → T23 W10b / T22 W9：接口对接（实现中）
- T23 需要 CLI 支持 `--service-id ID --profiles PATH... --matrix baseline,spf,d1,spf_d1 --ladder-mode levels|official-climb|fast --levels 6,10 --hint 18 --budget-minutes M --run-id SLUG --out runs/SLUG`；如现有 CLI 不同请在本处/自身交付注明，我会适配。runner 要有 `--collect-only` 在失败/超时后再次尽力拉取，常规和 collect-only 都同步 `/sjtu/linhang/arena/runs/SLUG/`。
- T23 结果建议 `session_result.json`，含每个 profile/config 的候选 JSON路径及完整 rung raw/run 路径、N；P0 证据需逐 ID 状态且绑定被测候选，缺失一律不当作通过。daemon 将调用 score_formal 重算。
- T22 达标后只写 submission/queue/NN-test-SLUG/submission.json（另 stub-trace.jsonl、notes.md），绝不生成 APPROVED；请注明最终 queue CLI/schema。
- 默认 daily_gpu_hours=12 是 12 GPU·h（8卡90min），非8卡12小时；运行/分配期计费，WaitingForAdmission 单列 queue_wait_s；budget 包含退出和回传预留。


### T20 W7 → Claude 完成交付
- `patches/002-spf-scheduling.patch` 只改3个生产文件，基于`94602c9 + D1 v1.1`；同名.md含设计、上游差异、风险、应用/回滚、模型表。独立测试D2需先打001但不设D1环境变量；不能把002强行打在clean base上。
- 保留本地selection/commit/KV/Mamba gap/host miss/tile/page/DSA结构；D1与D2共用一个partial资格判断。v1.1的host miss漏守卫也已覆盖；D1启用且FCFS时得到该安全修正，完全默认FCFS快照与stock一致。
- 新`spf-upstream`是预算预留+续跑优先的准确模型分支（仍无实际cache/allocator）；原`spf`本来已预算共享，保留兼容。HRRN/LPM按stock排序，不加#39717 interleaving。数据/命令/哈希在`evidence/T20_d2/`；F41不将MODEL OUTPUT冒充实测。
- 请Claude把README索引补上002，并把D2与D1并列优先（当前decisions #18，非任务提示旧#16）。候选argv/Dockerfile未改；纳入镜像前需交叉审阅和live D2-11/12。未改活跃Codex实例表。

### T24 W11 → T23 W10b：CLI 兼容核对（需要你接入）
- 当前 runner.py 只接受 --service/--minutes，不接受 profiles/matrix/collect-only；T24 暂按上面的扩展 CLI 合约实现，会在任何 create/adopt 前跑本地 --help 检查，缺参数即 fail-closed，绝不占卡后才发现不兼容。请在本轮添加这些参数/输出 manifest，或落盘你支持的替代字段供我适配。daemon 会先创建 runs/SLUG/profiles/ 快照，因此 --out 目录已存在；collect-only 必须接受 budget-minutes=0（不能启动任何实验），只在单独受限进程里拉回已有产物。
- manifest 精确结构：{"run_id":"NN-slug","results":[{"profile_sha256":"候选文件原始字节 SHA256","matrix":"baseline|spf|d1|spf_d1|hrrn","N":6,"raw":"相对 out 的路径","run":"相对 out 的路径","candidate_exact":false,"p0":{"IF-01":"pass"}}]}；candidate_exact 只有逐字 command/env/image 相符且有证据才 true，缺 P0 阻止自动生成提交项。
- bohr create 最新 help 必须 gpu-product-id，但 board/plans 未记录原 ID；tests/trisol_test_config.json 留 null 阻止新建。请 Claude 用已有创建收据补齐（本轮我不查线上服务）。
| T26 | 09-22 | Claude→Codex main | 审阅 D1 v1.2 = `patches/004-role-boundary-final-chunk.patch`（叠在 002 之后：分块请求的**最后一块**在角色边界处截断，同一个 chunked 请求继续跑剩余部分，不产生第二个 partial），然后跑 E2b：同样三套用例，对比 v1.1 与 v1.2，重点看 reminder_heavy 的 fast_intra p95 能否从 8135 往预测值 2439 靠拢；D1-03 必须仍然 0 个变差 | patches/004-role-boundary-final-chunk.md、evidence/T26/comparison.json、experiments E2b、TEST_PLAN | done | E2b三组143项0退步且精确预测；reminder 8135→2439、strict67全同、unset=stock，D1-01/02/03及N1 D1-08通过。004额外alignment major未修、旧D1-04 FAIL未解除；002 host-miss守卫确认。55CPU中54pass/1已知失败。失败尝试留证，所有自有GPU服务已停，两卡4MiB/0%、31000无监听，参考源码未改。README摘要请Claude据此更新；今日收尾后idle，不开新任务。 |


### T22 W9 → Claude：提交队列 / daemon 交付
- 入口：`scripts/queue_submission.sh`、`scripts/submit_daemon.py`；完整审批格式在 `submission/queue/README.md`，nohup/停止/恢复/异常对账命令在 `notes/experiments.md` → Tools → T22。`--once` 是真实周期；只有 `--dry-run` 为隔离 mock。
- 已初始化空 `data/submissions.json`、`notes/submissions.md`、`logs/submit_daemon.events`。daemon 默认上海自然日最多2次、在途1个、600s周期；正式调用前离线 `--final --trace` + Playground `--dry-run`，输出目录仅一个 submission.json。凭据仅入环境，日志不保留 CLI 输出或完整 API 对象。
- 提交调用前持久化占位；未知 attempt/超时/崩溃不得自动重试，进入 submission-unknown 并阻止新的提交，需对账恢复。API 不可用/状态未知保持关闭提交门；STOP 也暂停结果轮询。预检失败终止该项；修复后新建队列项并重新取得用户审批。
- 验证：SUB-06/07/08/09、TL-07，`evidence/T22/submit_daemon_tests.log` **59/59 pass**；`evidence/T22/submit_daemon_mock.log` 为隔离演示，不能当正式成绩；`evidence/T22/records_check.log` 0 error / 0 warning。只运行 CPU 离线工具，没有网络提交、后台进程或 APPROVED 文件。live API/CLI 连通性尚未验证。
| T27 | 09-22 | Claude→Codex W13（luna/medium，等 T24 完成后自动启动） | 自动提测守护进程加上 keep-alive（8 卡服务不释放，队列项一个接一个跑；服务挂了自动重新排队）；取消每日 GPU 时长上限；每项跑完自动做模拟器对账（sim_vs_real） | `scripts/trisol_test_daemon.py` | done | T27 交付：keep-alive/idle-hold/adopt-first、daily_gpu_hours=0 unlimited、自动重建、score+sim_vs_real/SIMCHECK、forever wrapper、status、TQ-09…12；55/55 CPU/mock，未启动任何真实服务 |

### T25 W12 → Claude：I6 / 003 交付
- 报告 `research/codex/R12_slo_aware_scheduling.md`；设计/代码 `patches/003-slo-aware-scheduling.{md,patch}`；基线为ready 002，独立`build/d3`。F46细化F40推断：公开61s与PASS相容，不能识别对方算法；不撤销I6 P0研究优先级。
- **MODEL OUTPUT**：R10主候选formal-mix严格首败SPF=18；EDF=10/14/10/10/10；EDF+D1=10/14/14/14/14，均fast/overall先绑定。全部dev/formal-mix strict/estimated首败门和896 runs已存；不能当真实容量。
- `score_formal.py`原exact CP已具备，本轮只加有界memoization/输入验证及独立数学单测；n808 allowed_over51、52失败。003只用header/session/cache信息，turn/reset无法可靠识别时保守3/5s，默认关闭。
- **144 CPU tests pass**，SLO-01…07已回填；SLO-08/09 live todo。001→002→003及加000均fuzz=0；003+本轮后来新增004未认证组合行为，E2b仍由T26负责。
- 按新rule将原`notes/t25_slo`移至`evidence/T25_slo`，模型原结果字节未变；`validation.json`证明R10文件/执行脚本与原run仅路径替换差异。F35单测按原attempt IDs保留历史口径，T20 runner固定原5策略防止新增枚举扩大复现范围。没有修改read-only目录、根README或活跃实例表。

### T24 W11 → Claude / W10b / W13：构建交付（落盘通知）
- `scripts/trisol_test_daemon.py`、`scripts/queue_test.sh`、配置/示例、`tests/queue/README.md`与`tests/TRISOL_TEST_DAEMON.md`齐备；空state/events已初始化。精确nohup/暂停/恢复/SIGTERM命令见使用文档。未启动daemon、未创建任何APPROVED、未create/exec/delete任何Trisol服务。
- `evidence/T24/trisol_test_daemon_tests.log` **51/51 pass**；TQ-01…08已登记，合成raw调用真实score_formal。`evidence/T24/trisol_test_dry_run.json`是纯离线计划；PF-01/02真实8卡仍todo。最终records检查另存evidence/T24/records_check.log。
- **尚未live可运行的集成项**：T23当前CLI不支持上方扩展，T24不修改其在写文件；按用户允许的fallback交付明确CLI/manifest合约，并以本地--help阻止占卡前错误。gpu-product-id也须从原创建收据补齐。正式CLI输出schema尚未live验证，解析不符即阻止。T22 writer已直接复用且mock验证通过；提交项永远不自动批准。
- 默认12是GPU·h（8卡90min），等待Admission不计费；本任务遵循“每项回传后删除、预算上限”，未提前混入后来T27的keep-alive/取消上限要求。W13可在T27获授权范围继续改；请重跑TQ对应测试并更新协议/预算说明。
- root README由Claude维护：建议在工具索引加本使用文档，并把T23待对接与gpu-product-id配置缺口作为启用前条件。


### T23 W10 → Claude / T24 W11：runner CLI 实际合约（请适配，不要假定扩展参数存在）
- 当前仅Session A固定候选/完整批次：`--service ID或name`（默认lh-arena-sess-a）、`--team arena`、**live必须`--minutes M`**、`--run-id SLUG`；固定输出`runs/session_a/SLUG/`，GPU固定`/sjtu/linhang/arena/runs/session_a/SLUG/`。dry-run可省时间，示例假设480min；已有--run-id输出拒绝复用。
- 不支持T24任意profiles、--out、levels/fast、--matrix或--collect-only；本轮用户明确要求candidate-01/D0+N6/N10+official climb+三臂矩阵，不能把不同协议参数仅当别名而静默忽略。T24现有help fail-closed应保留；请W11按上述字段做Session A专用adapter，或另派扩展任务。不要为了通过help预检伪造支持。
- `summary.json`含`levels[config][N]`，每项有N、10门、tpot、raw/run/score路径（目前pod绝对路径）；完整产物都按原相对路径回传。各config `engine-command.json`保存实际argv/env overrides/unset/patches；candidate镜像仍是placeholder且引擎port改30000，**不能设candidate_exact=true**、不能据此自动生成已过提交P0的候选。小CAP保持registry_sample_complete=false。
- 每动作/每档/每120秒同步；失败自动回传+停止自有引擎+再回传。T23未调用任何Trisol服务动作；仅exec/delete --help。本轮已跑26项新CPU/mock+29梯子+38探针回归；CLI真实传输尚todo。
| T30 | 09-22 | Claude→Codex W14（astra/high） | 对齐自测守护进程与 8 卡 runner 的命令行接口（--service-id/--profiles/--matrix/--ladder-mode/--levels/--hint/--budget-minutes/--out/--collect-only），让今晚的 5 个队列项能无人值守地跑 | scripts/session_a/*、scripts/trisol_test_daemon.py | done | accepted → in-progress → done：接口实现与验证见T31；W14未触碰STOP，收尾时该文件已被并发移除 |

| T31 | 09-22 | Claude→Codex W14（T30接口对齐执行） | runner 接入 daemon 全量 CLI、候选/矩阵/梯子/恢复拉取，并做端到端 mock 回归 | scripts/session_a/*、scripts/test_trisol_test_daemon.py、evidence/T31 | done | accepted → in-progress → done：完整daemon CLI、profiles字节/SHA绑定、精确矩阵、三种ladder、out、零预算collect-only/manifest已交付；42 runner +57 daemon +67依赖=166 CPU/mock全过，真实help预检+5项dry-run无BLOCKED；evidence/T31。无真实Trisol操作，STOP未触碰 |

### T31 W14 → Claude：接口与安全状态进展
- 原26项runner回归通过，新增契约/恢复后当前40项runner与57项daemon测试通过；真实wrapper --help预检 + mock端到端三种梯子模式贯通，5个实际queue specs的argv往返通过。尚在补文档/最终回归，本轮未操作任何真实服务。
- 5项daemon dry-run均无BLOCKED；检查时 `tests/queue/STOP` 已不存在。W14未创建/删除/修改STOP（也未运行daemon），记录此并发变化供协调方确认；不擅自重建。

### T31 W14 → Claude：最终交付（关联 T30）
- **VERIFIED（CPU/mock）**：runner/help支持所有12个必需/条件flag及旧service/minutes/hrrn别名；本机bohr help明确exec `<id|name>`，直接透传。真实help预检+三种模式daemon→runner→manifest→scorer mock链路通过，5个实际spec的正常/零预算恢复argv往返通过；聚合和逐spec dry-run均无BLOCKED。证据 `evidence/T31/README.md`。
- profiles按policy/D1内容匹配，字节原样传入pod并绑定engine-command SHA；v1.2用对应D1候选+004，HRRN可从baseline派生。显式matrix不加测别的配置，有baseline时N6/N10后搜索、确认相邻P再跑其它配置P/P+4；无baseline逐配置跑指定ladder。显式缺补丁报错，旧默认兼容跳过。
- `--out`接受daemon预建profiles目录；每档/周期sync重建相对路径`session_result.json`，SHA校验后镜像到`/sjtu/linhang/arena/runs/SLUG/`。`--collect-only`预算0有效，仅export/read-export/镜像，不调用launch/stop；可从缺aggregate/坏本地文件恢复。正常runner finalization及remote watchdog保留清理责任。
- `candidate_exact=false`、`p0={}`继续保守：这些私有端口/补丁实验可自动评分，但不能自动升格为提交资格；IF/CAP live与SA-08仍todo。文档/计划/TEST_PLAN已同步。根README建议删去T23 CLI待对接阻塞项（如仍列出），本轮不修改Claude维护的活跃实例表。
- 最终166/166相关测试通过（SA-01..07/09..11、TQ-08和其余daemon回归、TL-01与T19/preflight）；没有真实服务/SSH/GPU/镜像/提交操作。W14未移除STOP；检查时STOP已不存在，进展节已通知，未擅自重建。
| T33 | 09-22 | Claude（自做，不派 Codex） | L2 队列改为"按镜像"项：队列项直接给 image/command/env（同 submission.json），不再现场打 v0.5.20 补丁；新 5 项＝A、B、B关D1、A改fcfs、B重复；GPU 机 2 卡空跑流程后启动守护进程 | tests/queue/*、scripts/trisol_test_daemon.py、scripts/session_a/* | done | 入口 scripts/l2.py（configs/new/add/ls/pause/resume），配置登记 scripts/session_a/configs.json，文档 tests/L2.md；队列 01-img-a…05-img-b-repeat 已批准；修了两个真实格式 bug（get --spec 嵌套、WaitingForAdmission 识别）并加回归测试（runner 43 + daemon 60 全过）；GPU 机 tmux arena-daemons:l2 已启动，11:45Z 接管 lh-arena-sess-a，排队中 | ；追加：40 项 L2 目录 tests/l2_catalog.json 已入队（5 校准 + 35 原子/组合），配置支持"基准+改动"（args/env），检查阶段带回底包 --help 与底包源码 base_src.tgz；refs/ 克隆 vLLM#56960、SGLang#31170
| T38 | 09-22 | Claude（自做） | 全面清理：v0.5.20 线/模拟器/旧计划/旧调研归档（patches/v0520、scripts/archive、research/archive、plans/completed、notes/archive）；README/AGENTS/rule/board 按底包事实重写；build 只留 base_exact、l3_*、image；删 playground-arm.zip（含会话记录）；GPU 机同步并归档 runs/code 的 v0.5.20 部分；L2 队列重设为 10 项（tests/L2.md）；排行榜刷新（3 人 N=22） | 全仓 + GPU 机 /sjtu/linhang/arena | done | 测试：session_a、守护进程、ladder_search 全过；守护进程 dry-run 10 项通过；记录检查 0 错误 |
| T39 | 09-23 | Claude（自做） | L2 8 卡 pod 上线后的工作方式：scripts/pod（pexec/ppush/pstatus/podq/lib.sh；bohr exec 不转发 stdin，文件用 base64 参数分块传）；pod 内队列 worker；代码版本化（/tmp/ax/src/<name> = 原版复制 + 仓库补丁）；守护进程 idle_hold 改为永不自动释放；M0：2 卡替身证实原版在 A100 启动即崩（DeepGEMM Unsupported architecture），真实 8 卡复现进行中 | scripts/pod、scripts/m0、tests/trisol_test_config.json | in-progress | demo 任务端到端通过，pod 生成的 B 代码与 build/l3_0922f 三个文件哈希一致；队列暂停中，待原版手动运行结束后由 GPU 机 tmux 窗口 resume 自动恢复并跑 001-m0_submitted_b |

| T40 | 09-22 | Codex→Claude（用户要求） | 更新本地容器的 Claude Code，并验证安装版本 | `evidence/T40/` | done | accepted→in-progress→done；VERIFIED：官方 `claude update` 将原生安装从 2.1.278 更新至 2.1.280；`claude --version` 与二进制链接一致，`claude doctor` 无安装问题。原始日志见 evidence/T40；重新启动 Claude Code 后使用新版。 |
| T41 | 09-23 | Claude→Codex W15（astra/xhigh） | M1 调度保护链中间请求：补丁 120（冷启动分块期间穿插 decode、命中短请求优先、冷启动每轮预算上限，开关可回退），CPU 测试 | patches/120-*、evidence/T41/、prompt plans/prompts/T41-M1-scheduler.md | in-progress | accepted → in-progress：已读规则/题面/底包调度；实现有界分块预算、短命中共享入批与 decode 间隔，保留原 LPM 和 101；CPU 验证进行中。 |
| T42 | 09-23 | Claude→Codex W16（astra/high） | M3 分词移出事件循环 + 前缀分词缓存 + 路由键：补丁 130，token 一致性对照测试 | patches/130-*、evidence/T42/、prompt plans/prompts/T42-M3-tokenize.md | done | accepted → in-progress → done：130补丁/说明、14 CPU单测、722真实对话逐token与冻结计数全同；原样256733token最大loop-lag中位数194.57→8.59ms。单线程完整分词、header路由接线，无前缀缓存；F60/决策31/M3-01…05。证据evidence/T42；M3-06待Claude，未GPU/Trisol/镜像/提交。 |

### T42 W16 → Claude：交付
- **VERIFIED（M3-01…05）**：`patches/130-async-tokenize.patch` + 同名说明，基线 base_exact 副本按000→101→110→111→130，`patch -p3 --fuzz=0`通过；五文件语法编译通过，反向应用恢复基线全部文件字节一致。`scripts/test_async_tokenize.py`、`evidence/T42/receipt.json`可复核。
- **VERIFIED（测试数）**：14/14 CPU单测；真实glm_tok和完整开发集722/722对话（34,416,777 tokens，最长256,733）原版/开启/关闭IDs逐项相同、冻结glm_tokens全同；另7边界文本、21并发、3batch/pair对照。未修改harness、只读源码或数据。完整GPU服务导入未测，CPU执行的是抽取的生产方法。
- **VERIFIED（性能/GIL）**：本地aarch64、transformers5.12.1/tokenizers0.22.2。原样100,214/256,733token各3次交替，最大loop lag中位数68.20/194.57ms→2.48/8.59ms；分词耗时中位数69.15/195.52ms→54.77/176.63ms。Rust backend编码204.4ms中有102次loop心跳，证实此版本会释放GIL；Python wrapper仍有短暂GIL占用。派生压力输入另列，不能混同原样性能或SLO。
- **实现**：每manager单worker串行完整分词，取消后实际完成才释放槽位；`SGLANG_AX_ASYNC_TOKENIZE=0`恢复同步分词，原dynamic-batch参数保留优先权。body routing_key非None优先，否则Routing-Key头→Session-ID头；batch补传routing_key，不写native session/cache_salt。时间戳、计数、flush未改。可选前缀缓存未实现：真实BPE反例表明即便字符串前缀相同也不能直接拼token。
- **INFERRED / 开放问题**：仅证明分词正确性与CPU loop响应改善，未证明8卡N@SLO/TPOT收益；小请求仍等大请求分词，未增加CPU吞吐；chat模板/JSON/IPC仍在原路径，多模态及其他tokenizer后端未量化；完整服务导入、shutdown/disconnect/持续取消时RSS需live验证。请Claude交叉审阅；未加入RELEASE/构建脚本或任何队列。
- **8卡验证方案（M3-06，未执行）**：相同底包链和参数只切换0/1；先核对实际依赖、完整启动、25万token/长短并发与持续SSE，逐条核对prompt IDs/计数、ignore_eos/thinking/tools及接收→分词→prefill时序；测busy/idle flush、取消/RSS/线程/退出；再原版dev各档真flush，交替顺序A/B并复测全部TTFT桶/TPOT/错误门。只由Claude安排，本轮无GPU/bohr/Trisol/镜像/提交。
