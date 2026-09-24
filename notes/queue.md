# 实验队列

本页记录决策，实时状态以 `scripts/pod/pread status` 为准。按用户最新安排，执行层主 Codex 统一管理 8 卡队列、集成、独立复核和提交准备；Claude 负责编排，数据 Codex 负责生成。会话、消息入口及汇报位置见 [coordination.md](coordination.md)。

## 当前安排（2026-09-24）

| Job | 对照 / 问题 | 状态与判据 |
|---|---|---|
| 044r / 045r | 正式 A@dev N14 / B@dev N10 校准 | 完成；各722条，主会话重新调用评分器复算一致，均有效FAIL；见 experiments.md 与 evidence/coordination-20260924/calibration-recheck.json |
| 046r | S1@N22 混合 profile | 已自然结束进入done；两段TP0/TP4共4份trace已复算；仅诊断，不作判档/性能对照 |
| 047-official_a_n22 | 正式 A 原样、原开发集 N22，不开 profiler | 完成且独立复算一致：722条VALID FAIL；TPOT .0620/.0870通过，fast33/23、overall55/27、chain73/22失败；13补丁/参数/env一致，冒烟12/12、KV池1,036,288。完整证据evidence/L047-official_a_n22/N22/ |
| 048-official_a_122_n22 | 同047，仅加新版122、τ=.085 | 完成且独立复算一致：722条VALID FAIL；fast33→10/23转通过，overall55→28/27、chain73→62/22仍失败；TPOT .0617/.0840通过。保留122为有希望候选，尚无正式N收益结论；不插队继续扫参数 |
| 049-official_a_171_num | 正式A与A+171各两次真实TP8输出/logprob初筛 | 已退出：原版两次重复13/16未过一致性门，REFERENCE_UNSTABLE；171候选未加载，暂无TP8结论 |
| 050-official_a_172_num | 正式A与A+172各两次真实TP8输出/logprob初筛 | 已退出：原版两次重复15/16未过一致性门，REFERENCE_UNSTABLE；172候选及路径trace均未运行 |

| 051-official_a_baseline_diag | 正式A固定输入、逐次flush后首token/长续写重复 | 已完成16/16；逐次flush的单token cold37仍首token分叉，cold256同token logprob漂移.263；052准备逐层定位，不加载算子候选 |
| 052-official_a_numtrace | 正式A目标prefill逐层张量指纹 | 请求12/12完成，但trace为空，诊断INVALID：多模态入口先embed后传input_ids=None，被helper跳过；主会话已修 |
| 052r-official_a_numtrace | 修复输入ID来源后的同配置逐层定位 | 入口CPU回归通过，模型启动；已完成一条cold37，下一次flush撞上最后启动预热，HTTP400退出；8rank记录保留 |
| 052s-official_a_numtrace | 复用052r引擎继续逐层定位 | 因发现已有8rank记录，保护退出；主会话核明来自052r已完成的一条cold37 |
| 053-official_a_numtrace_resume | 归档首条记录后复用同引擎完成干净重复 | 已完成；8rank完整45层，cold37/256各3次；首次差异在MoE层3（部分cold37比较层4），此前阶段一致；1.18MB压缩原始记录sha256一致，本地独立复算一致 |
| 054-official_a_moe_numtrace | 正式A首MoE内部路径：全权重、路由、排序、GEMM与归约 | 已完成；完整权重/输入/路由相同，排序首先变化，cold256随后8rank均GEMM1及本地输出变化；误差量化与排序覆盖检查已完成；按用户要求停止重复一致性排查，不作性能或算子故障结论 |

| 055-official_a_combo_cost | 正式配置与171+172组合短测速 | 按用户最新顺序要求主动停止，队列记failed为stopjob终止，不是候选错误；不作完整组合结论 |
| 056-official_a_171_n22 | 现有正式配置只加171，完整原开发集N22 | 完成并独立复算：722条VALID FAIL，fast35/23、overall57/27、turn1/3、chain64/22，TPOT .0628/.0869通过；无引擎错误，非一致性门退出。34层实际融合；KV 1,024,960、状态槽318，少于047的1,036,288/321。单次收益混合，尚无净收益结论 |
| 057-official_a_172_n22 | 现有正式配置只加172，完整原开发集N22 | 完成并独立复算：722条VALID FAIL、0请求错误；fast33/23、overall53/27、turn2/3、chain62/22，TPOT .0610/.0850通过。KV 1,036,288、状态槽321与047一致。均值略降但fast/overall TTFT p95上升，未证明稳定净收益 |
| 058-official_a_longchain_lite_n14 | 正式 A 原样，64条完整链／1123请求的 lite 集，固定 N14 | 已入队running，启动引擎中。Pod侧8个数据artifact校验通过，13补丁逐项SHA256全部OK（含121），源码补丁签名`5a359e41c183750a`与047一致；参数/env与正式A一致，保留MTP。只跑lite，预热后严格flush，原harness完整评分。证据入口 evidence/L058-official_a_longchain_lite_n14/ |

049/050因主会话自行设置的重复生成一致性门提前退出，不构成171/172实现错误的证据。用户已明确取消该门：删除两份旧数值门job入口、移除通用template的NUMREF阻断，停止后续逐字一致性排查。比赛要求的接口、清缓存、完整性和评分门继续保留。056/057均已完成；057结束后pread确认无running/pending job，服务未停止或释放，常驻watcher继续监控。

新任务使用修正的 checked runner、flush证据和评分kit；047打印工具及原数据哈希，每个job打印patch哈希。049使用通用numcheck，flush耗尽、并发请求失败、输出截断都会失败；其logprob阈值仅粗筛，原始差异必须复核。

047/048同请求对照工具已独立验收10项CPU回归：完整cohort/负载一致、原始精度判门、排除warmup与边界秒。T56 LCP仅核元数据/物理上界，来源未验证，仍只作诊断。[对照记录](../evidence/L048-official_a_122_n22/N22/compare_vs_047.txt)、[工具复核](reports/review-compare-sol.md)。

## 新数据与算子线

- **长链数据**：新版311链/5601请求已生成并完成CPU验收，独立真实GLM检查VALID、原harness自检PASS，均5601/5601、0错误。唯一成品为`data/s1-dev-longchain/`，见[data入口](../data/README.md)与[数据状态](reports/codex-data.md)。新增139次追问/118次重建；链长中位8、最大240，保持长尾。旧副本、失败产物和cache已清除。生成新增算量仍明显低于源摘要，gap分解为估计，仍是诊断集，不替代dev、不预测正式N。尚无GPU成绩，未做N槽闭环驻留模拟；本会话未改8卡实际队列。
- **171 KDA投影融合**：开发机加载/状态/MTP/graph与成本证据已交付；056首次完整TP8性能回放已独立复核，未证明净收益，暂不加入部署组合。fast执行段p50 .57→.55s，但首执行前等待p95 4.32→9.78s；不能用执行段代替孤立kernel测速。缓存池差异在graph捕获前已出现，加载后空闲显存不同，具体分配来源及对TTFT的影响未隔离。见[056对照](../evidence/L056-official_a_171_n22/N22/compare_vs_047.txt)。
- **172 MoE clamped SwiGLU**：单卡随机权重完整Marlin路径墙钟减少3.1–4.9%；BF16激活与graph检查通过。057完整TP8回放已独立复核，TPOT均值少约1.6%、chain超时73→62，但fast超时不变、fast/overall p95变差，不能认定稳定整体提升；保留候选，暂不合入或自动叠加171。缓存池未缩小；62条chain超时中55条在首执行前已超过30秒。下一步算子投入须对准整段prefill成本与等待积压，不能只优化均值。见[057对照](../evidence/L057-official_a_172_n22/N22/compare_vs_047.txt)、research/codex/R23。
- **170 prefill graph**：现有v2 TP2证据仍不能代替TP8；待171独立验证后决定下一项，不与171/122一起改。

## 暂缓的候选

| 方向 | 当前决定 |
|---|---|
| 124短命中留位 | Claude已准备和CPU测试，暂不入队；其样本同时有真实缓存缺口，先看047/048后再决定块预算还是复用修复值得下一档 |
| DCP 33/40补齐故障 | 独立故障支线，尚未修复；不是执行层或高N回放的前置 |
| 400状态槽 / KV容量改动 | 旧040有命中改善，兑现为SLO收益仍待同基线对照；不与本轮122同时改变 |
| 替代引擎 / EP / 大块BF16专家副本 | 暂不占用8卡；优先完成现有候选正确性和新数据验收 |

## 实验选择规则

1. 直接研究N22/N26，不以本地N14或开发集PASS为前提。有效FAIL逐门分析；INVALID、数值错误先排查。
2. 原开发集用于同配置比较和回归，不能预测正式N；新数据是来源可追溯的合成集，仍有gap、重建、工具语义等偏差。
3. 每次只改一个机制；算子和编排各自有效后再测组合，局部收益不能相加。差异接近重跑波动时补对照。
4. 所有硬门完整报告；profile扰动窗口不判档；12题只作能力冒烟。
5. 8卡只做必要确认与完整服务对照；开发机做算子迭代。不得停止/删除/释放Trisol服务；只通过既有队列切换实验引擎。正式发布须满足质量与证据要求。

历史037b–042 job快照见 evidence/jobs-0924/，不是待重推任务。最初缺121的044/045已作废，校准只认044r/045r。旧122固定decode轮数与本轮TPOT进度预算是不同版本，不能套用旧公式解释新实验。

## 用户最新要求：直接分别测171、172

先056测171，再057测172；不再以重复生成token/logprob相同为前置，不再排固定排序诊断，暂不测组合。清缓存必须2xx且JSON success为true；preflight/warmup之后、正式测量之前再清，失败终止该档。代码缓存与CUDA graph保留。以实际TTFT、TPOT、TPM和全部赛题硬门判断效果，生成文本允许变化；局部算子速度不能当成正式并发晋档。

## 用户后续安排：正式 A 的长链 lite N14

全量311链的任务尚未入队，按用户调整先058跑`data/s1-dev-longchain-lite/`。该子集为64条完整链／1123请求，manifest SHA256 `b8f8b668189d9a6ded593ff88462cda37285d8e7a68f7654cea9ab1624f9b999`，cohort顺序ID `f5ef90c6218b260f`；本次已核artifact哈希与完整roster，不重复整套CPU渲染自检。回放root/set/cohort及评分requests都指向lite，使用新输出目录；不追加原开发集、N18或其他候选。正式A等价13补丁为`000 101 106 110 111 114 120 121 130 140 150 160 170`，含MTP；Pod启动前再次逐项核对冻结补丁哈希。原harness的preflight/warmup/flush继续保留，无重复输出一致性门。指标采集增加KV/Mamba池的free/evictable/used以及实际暴露的淘汰计数，并保存标签；缺失指标不视为零。
