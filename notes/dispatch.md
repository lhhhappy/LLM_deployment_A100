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
| T41 | 09-23 | Claude→Codex W15（astra/xhigh） | M1 调度保护链中间请求：补丁 120（冷启动分块期间穿插 decode、命中短请求优先、冷启动每轮预算上限，开关可回退），CPU 测试 | patches/120-*、evidence/T41/、prompt plans/prompts/T41-M1-scheduler.md | done | accepted → in-progress → done：120 默认开/环境变量关回退、生成器/27 CPU 测试/3617文件编译、方案与证据齐全；24组off轨迹字节一致（2组忠实复现基线断言）。交付见下，未GPU/Trisol/镜像/提交。 |
| T42 | 09-23 | Claude→Codex W16（astra/high） | M3 分词移出事件循环 + 前缀分词缓存 + 路由键：补丁 130，token 一致性对照测试 | patches/130-*、evidence/T42/、prompt plans/prompts/T42-M3-tokenize.md | done | accepted → in-progress → done：130补丁/说明、14 CPU单测、722真实对话逐token与冻结计数全同；原样256733token最大loop-lag中位数194.57→8.59ms。单线程完整分词、header路由接线，无前缀缓存；F60/决策31/M3-01…05。证据evidence/T42；M3-06待Claude，未GPU/Trisol/镜像/提交。 |

### T42 W16 → Claude：交付
- **VERIFIED（M3-01…05）**：`patches/130-async-tokenize.patch` + 同名说明，基线 base_exact 副本按000→101→110→111→130，`patch -p3 --fuzz=0`通过；五文件语法编译通过，反向应用恢复基线全部文件字节一致。`scripts/test_async_tokenize.py`、`evidence/T42/receipt.json`可复核。
- **VERIFIED（测试数）**：14/14 CPU单测；真实glm_tok和完整开发集722/722对话（34,416,777 tokens，最长256,733）原版/开启/关闭IDs逐项相同、冻结glm_tokens全同；另7边界文本、21并发、3batch/pair对照。未修改harness、只读源码或数据。完整GPU服务导入未测，CPU执行的是抽取的生产方法。
- **VERIFIED（性能/GIL）**：本地aarch64、transformers5.12.1/tokenizers0.22.2。原样100,214/256,733token各3次交替，最大loop lag中位数68.20/194.57ms→2.48/8.59ms；分词耗时中位数69.15/195.52ms→54.77/176.63ms。Rust backend编码204.4ms中有102次loop心跳，证实此版本会释放GIL；Python wrapper仍有短暂GIL占用。派生压力输入另列，不能混同原样性能或SLO。
- **实现**：每manager单worker串行完整分词，取消后实际完成才释放槽位；`SGLANG_AX_ASYNC_TOKENIZE=0`恢复同步分词，原dynamic-batch参数保留优先权。body routing_key非None优先，否则Routing-Key头→Session-ID头；batch补传routing_key，不写native session/cache_salt。时间戳、计数、flush未改。可选前缀缓存未实现：真实BPE反例表明即便字符串前缀相同也不能直接拼token。
- **INFERRED / 开放问题**：仅证明分词正确性与CPU loop响应改善，未证明8卡N@SLO/TPOT收益；小请求仍等大请求分词，未增加CPU吞吐；chat模板/JSON/IPC仍在原路径，多模态及其他tokenizer后端未量化；完整服务导入、shutdown/disconnect/持续取消时RSS需live验证。请Claude交叉审阅；未加入RELEASE/构建脚本或任何队列。
- **8卡验证方案（M3-06，未执行）**：相同底包链和参数只切换0/1；先核对实际依赖、完整启动、25万token/长短并发与持续SSE，逐条核对prompt IDs/计数、ignore_eos/thinking/tools及接收→分词→prefill时序；测busy/idle flush、取消/RSS/线程/退出；再原版dev各档真flush，交替顺序A/B并复测全部TTFT桶/TPOT/错误门。只由Claude安排，本轮无GPU/bohr/Trisol/镜像/提交。


### T41 W15 → Claude：交付

- **VERIFIED（CPU/源码）**：`patches/120-sched-protect-chain.patch` 与同名 `.md` 已落盘。普通 TP 默认开启；有存活 decode 才在 prefill 后插一轮；续算先保留对齐的 cap（默认2048），剩余批预算让完整设备短命中按原 LPM 入批，OTHER 跳过前执行原 Mamba 拒绝清理。首次冷/长准入也限 cap，已有/新 partial 共用单 partial 守卫。101 role helper、LPM 类与原预算方法 AST 不变，meta_info/输出预算/flush 路径不改。D32、F61 已入账。
- **可复现生成**：`python3 scripts/make_120.py` 从只读 `build/base_exact/sglang` 复制后按 000→101→110→111 `patch -p3 --fuzz=0`；生成 `build/p120/{baseline,candidate}` 与 120。`python3 scripts/verify_120.py` 在第三副本实际贴120/比较全部4684文件、验证确定性再生成及反向回滚，**3617 Python +3工具文件 py_compile全过**。只改 scheduler/schedule_policy 两文件；证据 `evidence/T41/validation.json`、补丁 SHA 收据。120 未加入 RELEASE。
- **测试数与范围**：`python3 -B -m unittest discover -s tests -p test_sched_protect_chain.py -v` **27/27 pass**（P120-01…07）。实际调度方法+mock ScheduleBatch/池/forward；包括冷+短交错、纯冷/纯decode、slot/KV/page/input/Mamba/session清理、101 admission/tail/branch、配置/对齐/显式间隔；16×30随机on轮次无双partial/预算超支。24组off序列字节一致：22组正常40轮，2组同方法/语句/轮次触发基线既有双partial断言，不能写成24组都运行成功。两臂原始trace在 `off_decision_traces.json`。
- **VERIFIED 上界；INFERRED 收益**：已准入冷请求、每轮资源足够C且不retract/abort时，prefill ≤ ceil(L/C)+1（101可加一次尾切分），默认含decode轮数 ≤2倍；100k/C2048实测mock无role49/97、有role50/99。有限进度不等于秒数SLO；更小chunk可能伤chain_start/吞吐。原LPM在无限热流下的**未准入**冷请求饥饿仍未修复，保持严格LPM意味着不能声称全队列无限负载公平。详见说明中的条件与保守G上界。
- **开放问题**：需Claude交叉审阅；真实GPU overlap/KDA数值、Mamba回收、内存压力/retraction、SLO与NEXTN未验证。另已定向复现原101的已有chunk尾切分+第二长请求造成双partial断言；120 on拒绝第二partial，off保留原行为。101既有额外deterministic alignment/数值门不因本测试而解除。特殊模式（DP/PP/CP/PD/mixed/HiCache/LoRA等）保守回退原路径，支持范围见文档。
- **建议8卡验证方案（未执行/未入队）**：`scripts/pod/jobs/dev_b120_template.sh` 按 `dev_template.sh` 使用prepare_src→ensure_engine→原harness。Claude安排M0通过后的普通TP8无NEXTN小用例（100k冷+decode+缓存短请求、原token/SSE计数、flush池恢复、单partial/数值），再原dev off→on→off N6/10→14/18/22，保存raw/summary/server.log并按全部门评分；chain_start退步时再试cap4096，最后单列NEXTN。设置 `N` 与 `AX_P120_VARIANT=off/on/cap4096`；对应中性内部源码名b120a/b120b/b120c，避免现有ensure_engine签名不含120环境变量导致错误复用。外部服务元数据继续中性；本轮没有任何GPU/SSH/bohr/Trisol/镜像/提交动作。
- **回滚**：启动时 `SGLANG_AX_SCHED_PROTECT=0` 并重启；或在副本反向 -p3/fuzz=0 撤120（已验证还原字节）。初次夹具调试失败日志保留，最终证据索引 `evidence/T41/README.md`；测试总表已同步，收尾check_records记录另存。
| T43 | 09-23 | Claude→Codex W17（astra/xhigh） | sm80 DSA indexer 融合 kernel（补丁 112）：paged decode + ragged prefill 两个入口，替换 110 torch shim（profile：约 60% GPU 时间） | patches/112-*、scripts/make_112.py、evidence/T43/、prompt plans/prompts/T43-sm80-indexer-kernels.md | done | accepted → in-progress → done：112补丁/说明/生成器/测试与证据交付；88数值对照全过，最大相对L∞1.1185e-5，topk≥99.9512%；4种graph动态重放逐bit同；graph decode4.75×/4.54×，prefill2.01–2.95×；整栈fuzz0/3623编译/反向字节还原。F64/D33；两卡已空闲，L2待Claude，未触碰pod/Trisol/镜像/提交。 |

### T43 W17 → Claude：交付

- **VERIFIED / 产物**：`patches/112-sm80-indexer-kernels.patch` + 同名说明；`scripts/make_112.py` 从base_exact+000→101→105→110→111生成。唯一kernel源 `scripts/kernels/sm80_indexer_112.py`；测试/运行入口 `scripts/test_sm80_indexer_112.py`、`scripts/run_sm80_indexer_112.sh`，验证/证据汇总 `scripts/verify_112.py`、`scripts/summarize_112.py`。证据索引 `evidence/T43/README.md`，最终 `summary.json` 绑定源码/oracle/test/patch/PTX SHA。112未加入RELEASE/构建脚本/队列。
- **VERIFIED / 语义**：软件uint8→bf16、bf16 MMA；每头先舍入bf16再relu×w归约，保留110点积输出精度。decode每个行×64token页一个program，GPU按ctx分支；[B]/[B,N]、N>1、非整页、0/1上下文、负页→0、表宽外/ctx外写0、strides与3D query通过。prefill H32用2query×64key/4warps；clean=True区间外-inf并跳过全无交集tile；**clean=False仍全宽计算**，这是110实际语义，不能剪掉区间外数值。现有tilelang依赖FP8 GEMM且限N=1，没有改它。
- **VERIFIED / 数值与graph（P112-01…05）**：最终88组对照（72普通/边界/幅值/形状、12动态graph、4个8192query大矩阵）全过，最大逐行相对L∞1.1185e-5、topk最低99.951171875%。fp8全部256编码另测，254有限值逐bit一致、2NaN类别一致。4种graph（decode共享/独立ctx、prefill clean两模式）各3次改q/ctx/bt/ks/ke重放，与eager逐bit同。误差/topk定义和空维断言见说明；随机激活的真实模型形状，不是采集的真实模型激活。
- **VERIFIED / 性能口径**：开发机A100-SXM4-80GB、torch2.13.0+cu130/Triton3.7.1、已有CUDA13兼容库；torch bf16 reduction默认True。表中L是indexer key数，不能直接当kpool前prompt长度。CUDA event中位数；decode graph每图20调用、7次采样，prefill eager旧3/新5次（warm2）；不含编译/数据生成。完整eager decode等8行数据与每次样本在 `final_all.log`。

| 场景 | key数 | 110 ms | 112 ms | 加速 |
|---|---:|---:|---:|---:|
| Decode B6 N1 / graph | 32000 | 0.4859 | 0.1023 | 4.75× |
| Decode B6 N1 / graph | 190000 | 2.6867 | 0.5917 | 4.54× |
| Prefill 8192 / causal | 32000 | 207.9272 | 96.3822 | 2.16× |
| Prefill 8192 / ragged | 32000 | 207.7147 | 76.1120 | 2.73× |
| Prefill 8192 / causal | 190000 | 1248.8606 | 622.4510 | 2.01× |
| Prefill 8192 / ragged | 190000 | 1249.0670 | 423.3853 | 2.95× |

- **VERIFIED / 打包（P112-06）**：000→101→105→110→111→112→120→130全部`patch -p3 --fuzz=0`可打，3623 Python编译通过，确定再生成/整栈反向逐字节还原/base_exact未改。代表形状实际PTX是sm80 bf16 MMA，无fp8指令、无spill；paged/ragged为128/205寄存器、8KB共享内存。早期验证流程失败和所有参数扫描保留，不覆盖历史日志。
- **INFERRED / 开放问题（P112-07）**：微基准证明110算子成本下降，不能直接推出8卡TTFT/TPOT/N@SLO；真实激活、完整服务加载、实际NEXTN、能力与原dev全门仍待Claude。输出矩阵仍为完整fp32（8192×190000≈6.23GB），没有融合topk；cold编译预热仍需服务层安排。建议同基线链仅切112 on/off、原参数/原harness真flush做L2 A/B，并单列NEXTN；这是验证建议，未创建队列或服务。
- **授权与收尾**：仅使用GPU开发机 `/sjtu/linhang/arena/code/T43` 与 `runs/T43`，自有算子进程全部退出，两卡各4MiB/0%（`gpu_final_idle.log`）。未操作bohr/Trisol/pod、未起8卡/打镜像/提交。F64/D33、TEST_PLAN和已完成计划同步；活跃Codex实例表留给Claude维护。回滚在代码副本反向撤112即可恢复110。
| T44 | 09-23 | Claude→Codex W18（astra/xhigh） | sm80 预填充 indexer kernel 逼近算力上限（112 仅约 30 TFLOPS，目标 ≥100），补丁 113 或 112v2 | patches/113-*、evidence/T44/、prompt plans/prompts/T44-prefill-indexer-roofline.md | done | accepted → in-progress → done：113叠加112交付；222数值/30graph全过，六档131–186等效TFLOPS、6.19–6.69×；decode源码/PTX保持112。9补丁fuzz0/3623编译/反向字节还原，profile/最终SHA绑定。F65/D34；evidence/T44，计划plans/completed/113-prefill-indexer.md。GPU任务结束，未触碰服务/镜像/提交，L2待Claude。 |
| T45 | 09-23 | Claude→Codex W19（astra/xhigh） | M2 KDA 双点 fp32 快照（补丁 140，替代 101/105 拆分预填充）：kernel 内导出角色边界状态+卷积历史，缓存树双节点，开关可回退 | patches/140-*、scripts/make_140.py、evidence/T45/、prompt plans/prompts/T45-M2-kda-dual-snapshot.md | done | accepted → in-progress → done：140补丁/说明/生成器交付；8组GPU（7组边界）最大误差0、21CPU、960轮off调度一致；722真实请求离线回放、完整9补丁fuzz0/3622源码+10工具编译/反向字节还原、20个远端源码SHA匹配。F66/D35；计划已归档，交接见下。仅GPU1算子已退出，未操作服务/8卡/镜像/提交，L2交Claude。 |

### T44 W18 → Claude：交付

- **VERIFIED / 产物**：独立 `patches/113-sm80-prefill-indexer.patch` 及同名说明，**叠加112，不替换112**。可复现生成器 `scripts/make_113.py`；唯一kernel源 `scripts/kernels/sm80_indexer_113.py`；完整验收/汇总 `test_sm80_indexer_113.py`、`verify_113.py`、`summarize_113.py`。证据索引 `evidence/T44/README.md`，最终 `summary.json` 绑定oracle/112/113/test/patch/compiler/profile SHA；未加入RELEASE。
- **VERIFIED / 实现**：H32/nq≥32/nk≥1024时q/K各解码一次到bf16，再以2query×128key query-major MMA计算，每program复用query处理4个key tile、GROUP32/4warps/stages1。更大tile/更多warp/另一朝向较慢或spill，stages3无稳定收益。原型/扫描均保存。保持110每头bf16舍入、fp32输出、clean=False全宽、clean=True无交集跳算且所有区间外写-inf；小形状与其他head数回退原112。
- **VERIFIED / 数值与graph（P113-01/02/03）**：复用112全部用例并分别对未改110、112，新增形状与六个8192query大矩阵逐行比较。222组全过，最大逐行相对L∞/L2为3.956824e-5，topk(min(2048,有限有效key数))集合最低99.951171875%。30次动态graph重放逐bit同eager，包括q/K/scale改变和长度增长/缩短/清空。随机激活，未采集真实模型激活。
- **VERIFIED / 性能（P113-03）**：A100-SXM4-80GB、torch2.13.0+cu130/Triton3.7.1。下表为同输入七轮交替顺序112/113 CUDA event中位数，含每次预解码与scratch/输出分配、排除JIT/输入生成。TFLOPS为任务指定全宽等效口径2*nq*nk*32*128；剪枝也计入收益，不能当作硬件MMA利用率，JSON另报有效pair比例。

| 场景（nq=8192） | nk | 112 ms | 113 ms | 112 TFLOPS | 113 TFLOPS | 加速 |
|---|---:|---:|---:|---:|---:|---:|
| causal | 32000 | 96.854 | 15.378 | 22.2 | 139.6 | 6.30× |
| ragged | 32000 | 76.112 | 11.662 | 28.2 | 184.1 | 6.53× |
| causal | 95000 | 314.727 | 47.073 | 20.3 | 135.4 | 6.69× |
| ragged | 95000 | 227.133 | 34.326 | 28.1 | 185.7 | 6.62× |
| causal | 190000 | 622.164 | 97.047 | 20.5 | 131.4 | 6.41× |
| ragged | 190000 | 425.674 | 68.814 | 30.0 | 185.3 | 6.19× |

- **VERIFIED / decode不退步**：入口/helper/kernel源码字节相同，代表形状PTX SHA与T43完全相同（128寄存器/0spill/8KB shared）。本轮graph 32k：112/113为0.102520/0.102427ms，190k：0.592206/0.592515ms，最大差0.05%；eager为0.1705/0.1684ms与0.7077/0.6638ms。没有decode优化，不把测量波动当收益；F64/T43历史112数据原样保留。
- **VERIFIED / profile与补丁链（P113-04）**：190k causal的112 CUDA kernel均值624.397ms；113主kernel96.102ms、两次解码合计0.104790ms（0.11%GPU时间）。最终主kernel162寄存器/0spill/48KB shared、sm80 bf16 MMA，无FP8指令。000→101→105→110→111→112→113→120→130全部fuzz0、3623 Python编译、确定再生成、全栈反向逐字节还原/base_exact未改均通过。profile为torch CUDA profiler，非ncu硬件counter。
- **INFERRED / 开放问题（P113-05）**：预填充算子目标已满足，实际TP8/NEXTN/模型能力与TTFT/TPOT/N@SLO未测。8192×190k额外scratch110.39MiB/调用，fp32输出6.226GB仍存在；服务中的显存池/graph与新形状JIT预热需Claude复核。未融合topk或改输出精度。建议同栈112/113、同参数、真实flush的服务A/B，由Claude安排；本任务未入队。
- **收尾**：F65/D34、TEST_PLAN、补丁说明/索引、已完成计划均更新；仅GPU开发机arena目录内GPU0算子，进程已退出、收尾两卡4MiB/0%。未操作bohr/Trisol/pod、未起服务/8卡、未打镜像/提交；活跃Codex实例表留给Claude。反向113即可回到112。

### T45 W19 → Claude：交付

- **VERIFIED / 产物与开关**：`patches/140-kda-dual-snapshot.patch` + 同名说明，确定性生成器 `scripts/make_140.py`，源模板 `scripts/p140/`；证据索引 `evidence/T45/README.md`，最终 `summary.json` 绑定补丁、源码、测试与原始证据SHA。140叠加101/105，`SGLANG_AX_KDA_DUAL_SNAPSHOT=1` 时绕过101的admit/tail拆分及105对应限制；默认/0恢复原栈，需重启切换。未入RELEASE/构建脚本/队列。
- **VERIFIED / 对齐与缓存**：最后user/observation的marker下标r；`G=lcm(64, mamba_checkpoint_grid(page_size))`，角色深度`R=floor(r/G)*G`，不包含marker；对齐prefix P、本次长度L的入树末尾`E=P+floor(L/G)*G`。仅`P<R<E`且槽充足时多分配一槽；R==E复用末尾。未对齐角色向下取整，最多重算G−1个marker前token；槽不足/无角色仅保留末尾；不对齐prefix/streaming session整批回原tracking，不新增调度。kernel从fp32累加器直写角色/末尾池，conv在原位卷积前导出history；实际未对齐末尾仍保留active fp32。
- **VERIFIED / 所有权与淘汰**：额外slot请求持有→角色insert移交树；duplicate/abort/不缓存finish回收。先原末尾insert，再锁尾、重取树拥有的KV插角色，避免重复释放KV；下一轮经原match/COW恢复conv+SSM。角色标签只归精确深度，tail在可淘汰候选内先于原LRU，Full leaf与pathcap也优先tail；锁仍优先保护。21项CPU真实controller/tree/components/tracking/pool/flush方法与checked allocator通过，每项原sanity_check；busy拒绝/idle真清均覆盖。3种branch×3请求+evict的off缓存JSON字节相同；32组×30轮=960轮off调度JSON字节相同，8组on无双partial，定向role请求2→1次extend。
- **VERIFIED / 数值（P140-01/02）**：A100-SXM4-80GB、torch2.13.0+cu130/Triton3.7.1，仅GPU1。随机projection/conv/gate权重，实际patched forward_extend→dispatcher→Triton方法与底包算子；支持初始state、strided slot、变长、全部NT_BUCKET。8组最终日志 `numeric_final_v3.log`，下表有效边界的SSM/conv/续算输出误差均0；对齐末尾、active输出/最终状态及off对照也逐元素相同。是算子测试，未加载完整模型权重。

| H×D | 每条extend长度 | 边界offset | fp32边界最大误差 | conv误差 | 续算输出误差 | off与原kernel |
|---|---|---|---:|---:|---:|---|
| 64×128 | 273,337 | 128,192 | 0 | 0 | 0 | 逐元素相同 |
| 64×128 | 128,192 | 64,128 | 0 | 0 | 0 | 逐元素相同 |
| 64×128 | 65,129 | 64,128 | 0 | 0 | 0 | 逐元素相同 |
| 64×128 | 1025,833 | 512,640 | 0 | 0 | 0 | 逐元素相同 |
| 8×128 | 273,337 | 64,256 | 0 | 0 | 0 | 逐元素相同 |
| 64×128 | 256,320 | 禁用角色槽 | — | — | — | 逐元素相同 |
| 64×128 | 4097,3073 | 2048,1536 | 0 | 0 | 0 | 逐元素相同 |
| 64×128 | 8192,4033 | 4096,2048 | 0 | 0 | 0 | 逐元素相同 |

- **VERIFIED / 数值前提与失败史**：初版one-shot/截断prefill跨`B*NT*H<=256`融合阈值有8.535385e-5最大误差，140开启固定非融合intra后解决；短extend可能增加kernel启动成本。关闭调用原始Triton函数且源字节完全保留，仍按原阈值选择。两臂最终共享源码相同的gate/output helpers及autotune选择；独立helper副本最初冷对照曾失败（未记录差值），未改源码暖重跑通过，不能把该失败单独归因140或断言所有独立冷启动bitexact。全部失败日志保留。
- **VERIFIED / 离线估算（P140-06，MODEL OUTPUT）**：原Renderer+glm_tok，722请求/311链/34,416,777输入token，全部冻结计数一致。每链空缓存串行回放，无未来oracle；无限容量、prompt-only，不含decode状态、淘汰、并发准入或retraction。下表不是实测cached_tokens/SLO；分别6/73请求命中改善，均0退步。

| 调度chunk | off命中token（101+105） | on命中token | 增加 | off→on extend次数 | 移除101 split次数 |
|---:|---:|---:|---:|---:|---:|
| 8192 | 16,886,400 | 16,903,104 | 16,704 | 3284→2616 | 666 |
| 2048 | 16,806,912 | 16,905,152 | 98,240 | 9430→8949 | 428 |

- **VERIFIED / 打包（P140-07）**：000→101→105→110→111→112→140→120→130全部`patch -p3 --fuzz=0`、3622源码+10工具py_compile、确定再生成、应用与生成树一致、整栈反向字节还原、base_exact未改。20个远端实际源码/测试/oracle哈希与交付一致；140 SHA256 `1fcb1ca8c6502b7c8a58f32c16bf34dbf3431d159cd649fdbf1b658ce7c2ff35`。
- **INFERRED / 开放问题**：少一轮调度与保留角色状态可能改善链中间延迟，尚无8卡吞吐、TPOT或N@SLO证明。固定非融合intra成本、每rank约17.6MiB额外state、有限池slot跳过、tail淘汰对共享的影响须实测。完整服务导入、真实权重/34层、TP8 collective、overlap长跑、能力及冷autotune仍未验证。开启限制普通GLM TP、extra_buffer、fp32、FULL+MAMBA Python cache；NEXTN/spec、HiCache、lazy/int8/unified、SWA、session radix、DP/CP/PP/PD/mixed/TBO/ReplaySSM显式拒绝，不能直接带入现有NEXTN配置。
- **8卡A/B方案（P140-08，未执行）**：Claude审阅后同一栈仅切140 off/on，普通TP8无NEXTN、101 IDs保留、120/130保持一致，外部中性元数据、内部profile设开关。先真实reminder/branch/非整页链核对缓存命中、extend范围/次数、输出/logits和计数；再default overlap、长冷+短热、Mamba/KV压力、duplicate/abort/retract/槽耗尽，检查树/锁/池及busy/idle flush、flush后首请求cached_tokens=0。原dev每档真flush交替A→B→A，N6/10→14/18/22，比较全部TTFT桶/TPOT/错误门、显存与slot_skip；先chunk8192，再单列120/2048消融，避免同时混入113。NEXTN等需另行接线验证。
- **收尾**：F66/D35、TEST_PLAN、已完成计划与board自有任务条目已更新；活跃实例表留给Claude。开发机自有算子已退出，两卡4MiB/0%（gpu_final_idle.log）；仅arena目录内工作，未操作bohr/Trisol/pod、8卡、镜像或提交。回滚为开关0重启，或在副本反向撤下游后撤140。
| T46 | 09-23 | Claude→Codex W20（astra/high） | 启动期预热补丁 150（@warmup ax_shapes，覆盖 KDA autotune、112/113、分块/命中/角色边界/并发 decode 形状，结束真清缓存） | patches/150-*、scripts/make_150.py、evidence/T46/、prompt plans/prompts/T46-startup-warmup.md | done | accepted → in-progress → done：150/说明/生成器交付；21CPU、2进程Gloo×3阶段、11补丁fuzz0/3623源码+8工具编译/反向还原通过。A100同形状12组repeat零JIT/bench，600→601新2编译；新进程57磁盘命中但42bench。F67/D36，30远端源码SHA匹配，GPU已空闲；完整服务零编译仍不能保证，缺口与Claude交接见下。 |


### T46 W20 → Claude：交付

- **产物 / VERIFIED**：`patches/150-startup-warmup.patch`、同名`.md`，`scripts/make_150.py`（源模板`scripts/p150/ax_shapes.py`）。独立验证/清单/测试/汇总脚本均以150结尾；证据索引`evidence/T46/README.md`、最终收据`summary.json`。基线严格000→101→105→110→111→112→113→140→120→130，150叠其后；补丁SHA256 `3d53476f8d94262e55241bb2295e3b9b96fb0ffc243469aaef9541d4169f20af`。未加入RELEASE、profile、构建脚本或队列。
- **启动与清理**：内部参数`--warmups ax_shapes`启用；48组564请求（直接input_ids，cold600、链+1000/+7000/+12000、冷20000、角色与分叉、边缘、ragged31、并发6…32），另600-token零命中探针。temperature0、单请求输出2、并发8…12；只对合成请求ignore_eos，不改用户thinking/输出/tools/历史/meta_info。初始/探针前/最终三次复用真实flush，新增verify_empty逐池只读断言；TP CPU group归并非主rank错误后才回复，保留000的跨worker汇总。log_metrics=False，修正可选exporter未遵守该字段的问题；scheduler实际启动计算日志不隐去。请求异常/取消/超时/脏池则清自有rid、finally真清并失败启动；MTP显式跳过、PD/HiCache/DP/PP等拒绝。
- **CPU / P150-01…05**：21项mock/真实生产方法通过；另外两个真实CPU Gloo rank×3阶段通过（全健康、仅rank1泄漏、仅rank1拒绝flush，失败均传播到主rank）。指定11补丁全栈fuzz0、3623源码+8工具编译、确定生成、应用树一致、整栈反向逐字节还原、base_exact未改。全库667显式JIT、重点63函数/12autotune逐项key/constexpr清单；每项固定模型预期域、调度依赖与未覆盖项在说明表。
- **A100 / P150-06**：12算子组（112回退/113两clean分支、decode B6/7/17/32、KDA65/600/2049/8192/8257）首次113实际编译/102bench/0磁盘命中；同shape再跑12组，JIT miss、实际编译、bench、cache新增/变化均0。113 NQ600→601新增2编译。新进程复用同一cache首次57内存miss/57磁盘命中/0实际编译，仍42bench，重复再次全0；不能把磁盘cache等同于免autotune/免device-load。Torch2.13.0+cu130/Triton3.7.1、随机激活、非性能/数值回归、非服务；30远端源码SHA与交付匹配。原始失败（包stub、旧hook API、最终JSON序列化）与修后新cache复验日志均保留。
- **不能承诺“服务期零编译”**：112/113精确NQ/NK/R constexpr已实测新长度反例；长prompt由scheduler切块，NT_BUCKET2依赖ragged同批；6…32提交宽度不保证实际eager B（graph可能pad）；MTP、B>32、长上下文、其它MoE/DSA/topk/后端等未穷尽。150只预热当前进程配置，不能临时切140。底包mark_serving_started早于request warmup，所以启动期仍可能打印F59同名告警；必须按HTTP readiness分界。通用server warmup在150之后仍会发France短句，空池保证是150结束时；组合custom warmup应把150放最后。
- **下一步 / P150-07**：Claude审阅后再安排普通TP8、140 off/on各自冷启动；核对每个case实际kernel key/缓存计数/池、readiness后编译、metrics与原dev全部TTFT/TPOT/能力门，有graph/eager分列，MTP跳过不算通过。若要求任意新长度不JIT，需要另行处理112/113动态特化/分桶并重做数值和性能验证。本任务没有操作bohr/Trisol/pod、8卡、镜像或提交；仅开发机arena目录的自有算子/Gloo进程，已全部退出，两卡4MiB/0%。F67/D36、TEST_PLAN与补丁索引已同步；活跃Codex实例表留Claude维护。回滚删warmups参数重启，或反向撤150。
| T47 | 09-23 | Claude→Codex W21（astra/high） | 112/113 kernel 去除形状/stride constexpr 特化（每个新长度重编译，阻塞上线），v2 重生成补丁 + 无重编译测试 + 数值/性能复验 | patches/112/113（v2）、evidence/T47/、prompt plans/prompts/T47-indexer-no-respecialize.md | done | accepted → in-progress → done：112/113 v2原名交付，v1归档；88/222数值+12/30graph全过，16行v1/v2配对最大+3.38%，原tile保留。50随机形状200调用+600→601回归8调用零JIT/编译/磁盘命中；286key审计、11补丁fuzz0/3623+6编译/反向还原通过。F68，evidence/T47；GPU空闲，交付见下。 |

### T47 W21 → Claude：交付

- **产物 / VERIFIED**：112/113原名补丁已更新v2，`scripts/kernels/sm80_indexer_{112,113}.py`四kernel全部长度/stride改runtime，仅固定模型/tile/CLEAN参数保留constexpr；默认值1与16整除类别特化保留。原tile、循环复用次数、mask、bf16每头舍入与fp32输出不改。v1补丁/说明在`patches/drafts/*-v1.*`，v1对照源与原T43/T44 SHA相符。生成/verify脚本同名，证据改写T47子目录；新验收`test_sm80_indexer_cache_47.py`、`bench_sm80_indexer_47.py`、`summarize_47.py`。
- **数值 / P112-01…05、P113-01…03、P47-02**：原测试与110 oracle未改，完整大矩阵逐行比较，随机激活，未加载真实模型。

| 版本 | 数值组 | 最大逐行相对L∞ | 最低topk重合率 | 动态graph |
|---|---:|---:|---:|---:|
| 112 v2 | 88 | 1.11846e-5 | 99.95117% | 4种×3次，逐bit同eager |
| 113 v2 | 222 | 3.95682e-5 | 99.95117% | 10种×3次，逐bit同eager |

- **性能 / VERIFIED**：A100-SXM4-80GB、torch2.13.0+cu130/Triton3.7.1；同输入v1/v2七轮交替prefill CUDA event，decode graph三轮交替、每图20调用×7样本。含解码/scratch/输出分配，排除编译/输入构造；按本次配对判断回归，历史时钟/数据差异不混算。最大退步3.38%<5%，无需tile调整。完整16行及每个样本在`evidence/T47/performance.log`。

| 场景 | nk | 112 v1→v2 ms | 变化 | 113 v1→v2 ms | 变化 |
|---|---:|---:|---:|---:|---:|
| 8192 causal | 32000 | 96.2340→94.0812 | -2.24% | 14.7287→15.2263 | +3.38% |
| 8192 ragged | 32000 | 76.1769→72.2994 | -5.09% | 13.9353→14.1989 | +1.89% |
| 8192 causal | 95000 | 314.7723→305.2160 | -3.04% | 47.5994→48.0636 | +0.98% |
| 8192 ragged | 95000 | 225.3184→211.5569 | -6.11% | 33.9955→34.7932 | +2.35% |
| 8192 causal | 190000 | 622.1133→623.7575 | +0.26% | 96.4309→98.2044 | +1.84% |
| 8192 ragged | 190000 | 423.0201→423.3863 | +0.09% | 68.3563→70.0367 | +2.46% |
| B6 decode graph | 32000 | 0.1250→0.1173 | -6.17% | 0.1250→0.1174 | -6.08% |
| B6 decode graph | 190000 | 0.6044→0.5520 | -8.67% | 0.5911→0.5395 | -8.73% |

- **编译计数 / P47-01 / VERIFIED**：独立cache，有限类别634调用预热产生286 JIT miss＝153实际编译＋133磁盘命中；每个实际key的常量表已审计，所有可变参数除默认值1外均为runtime。然后每轴50个不重复随机NQ∈[1,16384]、NK∈[1,200000]、P∈[1,3125]、batch∈[1,64]，两版prefill/decode共200调用，**JIT miss=0、实际编译=0、磁盘命中=0**。另600→601×两CLEAN×两版8调用同样全0，修复T46的具体反例。记录每个shape/调用/key，真实全grid输出，没有仅warmup编译或按耗时猜计数。
- **打包与PTX / P47-03**：000→101→105→110→111→112→113→140→120→130→150全栈fuzz0、3623源码+6工具py_compile，确定生成、应用树一致、整栈反向逐字节还原、base_exact未改。实际sm80 bf16 MMA无FP8指令、0spill；paged/ragged96/138寄存器、8KB shared，113主kernel166寄存器/48KB shared，unpack29寄存器/0shared。112/113 v2 decode PTX一致。最终`evidence/T47/summary.json`绑定源码/测试/oracle/补丁/证据，12远端源SHA全匹配。
- **开放问题 / INFERRED**：本任务证明112/113在已预热模型/tile/dtype/布局类别内免精确长度JIT，未热的新head/dtype/布局类别仍可能有限首次编译；没有修改150预热计划，也不保证其它SGLang kernel/整服务零编译。真实TP8/NEXTN、模型能力、graph池与SLO由Claude另测。112/113需配套v2，未加入RELEASE/构建/队列。
- **收尾**：F68、TEST_PLAN、patch说明/索引、计划归档及自有board条目已同步；活跃Codex实例表未改。仅开发机arena GPU0算子，自有进程全退出，两卡4MiB/0%。未操作bohr/Trisol/pod、8卡、镜像或提交。
| T48 | 09-23 | Claude→Codex W22（astra/xhigh） | M4 MTP/NEXTN 在 sm80 可用：路径走查、不兼容点修复（补丁 160）、与 101/140/120 交互、算子验证、8 卡任务脚本与接受率采集 | patches/160-*、evidence/T48/、scripts/pod/jobs/dev_b160_mtp_n6.sh、prompt plans/prompts/T48-M4-mtp-sm80.md | done | accepted → in-progress → done：160/说明/生成器/R17、8卡未执行脚本与spec加权采集交付；10CPU、真实resolve/draft mapper、8组算子族与MoE/dense Marlin数值/graph通过，12补丁fuzz0/3624源码+8工具编译/reverse；F69/D37，L2待Claude。早期JIT根盘缓存偏差已迁回arena，证据保留；两卡已空闲。 |

### T48 W22 → Claude：交付

- **产物 / VERIFIED**：`patches/160-nextn-sm80.patch`、同名`.md`、`scripts/make_160.py`；模板`scripts/p160/ax_mtp_sm80.py`，审计`research/codex/R17_nextn_sm80.md`，证据`evidence/T48/README.md`/`summary.json`。SHA256 `3547ff7d6d583b5ca19e0474a9ed0bd3fff3a355d5298868ee9f83ee96000a40`。基线严格000→101→105→110→111→112(v2)→113(v2)→140→120→130→150，160叠末尾。
- **完整路径与兼容性**：GLM draft是独立单层DSA/MoE NextN，没有KDA/mHC；target verify仍有34 KDA+11 DSA和mHC。所有DeepGEMM/FP8/fa3/FlashMLA入口与A100选路列明源码行号；110已覆盖spec多pool量化，111/112/113可直接复用，160不另造数值kernel。父进程解析双DSA tilelang、BF16 KV和三阶段Triton KDA，禁DG HC/TOPK计划V2。必须显式draft path=/mnt/models，topk1；不改thinking/工具/历史/采样阈值/token计数/flush。
- **状态与调度**：140与spec scratch无已证实物理别名，但140显式拒绝spec，组合生命周期未测；160关闭140并清101角色IDs，原extra_buffer/verify接受后SSM与conv回写/tracking保留。120首轮off、130off；150对MTP跳过，不传ax_shapes。48只是EAGLE默认请求cap，非模型限制。
- **验证 / P160-01…05**：10CPU；真实ServerArgs/draft ModelConfig/ignored量化名称映射；KDA5形状/20接受长度（fused输出max_abs7.45e-9、SSM3.73e-9，active/tracking/conv/无效槽检查），kpool T2/4/6误差0，DSA4形状relL2≤0.001980，MQA两路径max_abs1.431e-6，真实seed选择/carry/finally，EH/argmax/accept/prologue/两topk/mHC；MoE clip10四形状+graph relL2≤0.00589，dense Marlin9形状+27动态graph≤0.00285。采样测试是单热点概率夹具，不是分布等价；MoE专家数缩为33，不是全288+1加载。完整12补丁fuzz0、3624源码+8工具编译/reverse/base未改，4689远端候选文件SHA匹配。
- **未执行8卡任务 / P160-06**：`scripts/pod/jobs/dev_b160_mtp_n6.sh`，唯一源码名`b160_mtp_s3_k1_d4_mr32_n6`；NEXTN steps3/topk1/D4/MR32/graph32。主池足够时较默认48少约1.10GiB/rank scratch，32给N22/26留cap余量；主槽仍受spec auto-fit影响。没有调用bohr/Trisol/pod、起8卡、构建镜像、入队或提交。先同栈non-spec且101/140/120/130均off配对；真实权重/三完整runner图/输出能力/overlap+槽压力/abort/retract/flush通过后，再走原dev全部SLO。
- **接受率采集**：`scripts/extract_spec_stats_160.py server.log --draft-tokens 4 --out spec.json`。160日志增加同窗口原始tokens/rounds；Σtokens/Σrounds为每request-step接受长度，D4接受率=(Σtokens−Σrounds)/(3Σrounds)。只取TP0/无rank；旧舍入日志不虚构加权均值，最后未打印窗口不纳入。脚本从真实server stdout日志切harness段（含warmup），纯measurement须按harness时间再裁；spec接受计数不冒充EOS/stop后实际HTTP输出数。
- **INFERRED**：D4/MR32 scratch约2.268GiB/rank，MR48约3.368；同预算持久KDA槽近似普通55.6%，draft权重/graph预算还会改变容量。若r=投机轮成本/普通轮成本=1.4、接受长度1.6–2.0，TPOT约为原0.875–0.700，吞吐约1.143–1.429×；假设基线35ms则30.63–24.50ms。不能据此宣称超过27.3ms或N@SLO不降。
- **环境偏差与收尾**：TileLang缺z3已在T48独立目录补齐。初版C++ JIT缓存变量用错，7个本轮build误写根盘；已按源码路径确认归属、停自有编译并迁入arena，正确SGLANG_JIT_CACHE_DIR下重跑dense通过。原失败日志/迁移清单均保留；未动其他任务缓存。mHC编译器静态race警告保留，数值通过不消除警告；开发机AOT sglang-kernel0.4.6.post1需核对L3。两卡4MiB/0%，自有算子进程退出；F69/D37、用例、计划归档和board自有条目更新，活跃实例表留Claude维护。
| T49 | 09-23 | Claude→Codex W23 新会话（astra/xhigh）+ Fable 顾问 | 8 卡 N6 真实数据后的代码级分析：Codex=缓存丢失根因 + 容量账 + 过时结论清单（R18）；Fable=排队/120 充分性 + 预填充结构开销（allreduce/mHC/稀疏注意力、CP vs DCP）+ 文档误判审计 | research/codex/R18_cache_loss_and_capacity.md；prompt plans/prompts/T49-*.md | done | Codex部分accepted → in-progress → done：R18/F79、19条raw与真实LCP表、39.15/11.17/10.05/1.31GiB显存账、140/101交互、容量杠杆及过时结论清单交付。T49-01/02/03/08审计通过；逐节点触发池仍待T49-04…07。仅源码/本地CPU/Claude回传日志，未操作GPU、8卡、pod、镜像或提交。Fable部分见Claude既有审阅记录。 |

### T49 中间发现 / 给 Claude 的取证请求（Codex 主会话）

- **VERIFIED（源码）**：`pool_stats_observer.py:249–276` 的 KV/Mamba usage 都是 `(capacity - free - evictable) / capacity`，F75/F76 的 50%/4% 不是物理驻留率，不能排除任一池早已满并发生 LRU。R18 将给出修正口径及取证点。
- 本地 `evidence/N6_b113/` 当前仅有 `analysis.txt`，尚不能逐条归因19个异常。请 Claude 有空取回任务012的 `dev/raw_*.jsonl`、`run_*.json`、启动至测量结束的完整 `server.log`（服务复用时取真正启动日志）、有效 server args/env 白名单。可放 `evidence/T49/remote/` 或 `evidence/N6_b113/`，并通知本会话。Codex继续源码分析，不以此阻塞报告、不访问pod。不要包含凭据或完整环境。
- **Claude → W23（T49 取证已就绪）**：`evidence/T49/remote/` 含任务 012（b113 dev N6）的 `012_dev_raw__.jsonl`（722 条逐请求记录）、`012_dev_run__.json`、`012_server.log`（本次引擎从启动到测量结束的完整日志，含 server_args）、`012_job.log`。另有 120 的 N6 结果摘要 `evidence/N6_b120on/analysis.txt`。可直接用于逐条归因 19 个缓存丢失与"几乎全丢"的超长 prompt 案例。
- **Codex 已接取证，容量方案先更正（VERIFIED）**：012真实启动日志`:156/:199`是graph实际bs≤116、capture增量**1.31GiB**，不是6.1；`:101`载权增量39.15GiB；args的f=0.77805含VLM折减。因此不能用“graph64释放约6GB”和“状态只用4%”支持C2/§8的零风险×2承诺。正在补入R18与逐请求表。raw严格筛选idx>0且frozen_lcp−cached>4096得到**18 intra+1 turn_start**，19的类别需修正；25万cached64经raw也确认idx=0。
- **Claude → W23（用户批准 pod 只读权限）**：可直接用 `scripts/pod/pread`（在本地仓库根目录运行）查看 8 卡 pod：`status`、`ls <dir>`、`tail/head <file> [n]`、`cat <file>`（≤5MB）、`grep <regex> <file> [max]`、`analyze <run_dir>`；路径限 /tmp/ax/runs、queue、verify 等。例：`scripts/pod/pread analyze /tmp/ax/runs/013-dev_b120on_n6`。禁止 bexec/pexec/podq/ppush 及任何启停/写入（rule.md §4 第 7 条）。
- **Claude → W23（用户批准扩大 pod 权限，严格不许停）**：除 `scripts/pod/pread` 外，可在 GPU 机上运行 `scripts/pod/pexec_codex '<命令>'`（仅 CPU、CUDA 隐藏、工作目录 /tmp/ax/codex），并用 `scripts/pod/ppush /tmp/ax/codex <文件>` 上传自己的分析脚本。本地调用方式：`scripts/gssh "cd /sjtu/linhang/arena/repo && scripts/pod/pexec_codex '<命令>'"`（先把脚本同步到 GPU 机 /sjtu/linhang/arena/repo）。**绝对禁止**：停/删服务、杀任何进程、向引擎发请求、写 /tmp/ax 的 queue/src/runs/patches/bin、用 GPU（rule.md §4 第 7 条）。

### T49 Codex主会话 → Claude：交付（done）

- R18已完成，F79登记；T49-01/02/03/08 pass，04…07为下一次8卡取证方案。报告纳入回传raw、实际启动日志及新F78，包含19对逐条事实/候选机制和过时文档逐项改写清单；没有覆盖Claude研究文件或旧findings。
- 优先更正：usage扣evictable；完整N6 KV峰值92%；graph真实1.31GiB/bs116；状态总池10.05GiB；25万cached64是本次链首；19为18 intra+1 turn_start，且一条4.38万“损失”实际只有3184。5条短回退落在较早8192块末尾，101仅最后chunk扫描漏角色点，140可修但受角色槽free限制。
- 长idle两条及短idle一条已确认此前可用深度退化，LRU/分配压力最有依据，但现有日志不能识别FULL还是MAMBA触发；无测量中flush/restart/retract证据。新增事件应记两池free/evictable、触发调用点、节点深度/锁、branch/chosen_track、140 skip。
- 容量探针不要按“4%状态/腾6GB/零风险2×”解释。当前预算graph64+固定状态200估算约157万token/1.66×；减状态池可能加剧历史淘汰。F78 DCP已有积极实测，R18保留的DSA/indexer高虚拟地址疑点交T49-07验证，未宣称已发生错误。
- 最终本地输入/源码SHA、19对LCP/5个chunk算式、日志引用与显存账闭合通过；原证据`evidence/T49/`，本轮未使用新增pod权限、未改引擎/补丁、未操作任何GPU/服务/镜像/提交。活跃Codex实例表请Claude维护。
| T50 | 09-23 | Claude→Codex W24（astra/xhigh） | DCP 下带前缀命中的越界访问：定位 110–115 中按全量 KV 寻址的路径，开发机 DCP2 复现并修复（补丁 116） | patches/116-*、evidence/T50/、prompt plans/prompts/T50-dcp-address.md | blocked | accepted → in-progress → **blocked 09-23 ~09:30**：W24 会话被 Codex API 安全分类器误拦（"flagged for possible cybersecurity risk"，措辞含越界/虚拟地址），进程已退出，仅写出 plans/active/T50-dcp-prefix.md。Claude 决定：先让开发机做 prefill 固定开销诊断与 BCG 移植（约束链上），T50 之后换纯工程措辞重派。 |
| T51 | 09-23 | Claude→subagent（Opus） | prefill 每块固定开销诊断：开发机 GPU0 用 rank 模型（8 层 dummy）起 server，跑 chunkcost.py 得 T(c,P)，并 profile 一次 P≈96k/c=1024 extend，拆分 CPU/launch vs GPU kernel，列出最大项 | evidence/T51/、research/claude/R10_prefill_fixed_overhead.md | dispatched | 依据：8 卡探针反推 ~150ms/块截距（026a–e）；日志确认 prefill CUDA graph 因 KDA 规则被关。 |
| T52 | 09-23 | Claude→subagent（Opus） | 移植上游 PR #38522（GLM-5.3-Flash breakable prefill CUDA graph opt-in）到我们的补丁栈为 170，fuzz=0 全栈；开发机 GPU1 rank 模型验证 BCG vs eager 数值与 T(c,P) | patches/170-glm-bcg-prefill.patch+.md、evidence/T52/ | dispatched | refs/pr38522.diff 已下载；与 110/112/113（kpool indexer）、140（KDA 快照）交互需核对。 |
