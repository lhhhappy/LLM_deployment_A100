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
| 054-official_a_moe_numtrace | 正式A首MoE内部路径：全权重、路由、排序、GEMM与归约 | 已完成；完整权重/输入/路由相同，排序首先变化，cold256随后8rank均GEMM1及本地输出变化；需固定排序反事实与误差量化，不作性能结论 |

049/050 的正式A基线含首token分叉，cold请求缓存均0、prompt长度相同；不能归因171/172或直接放宽数值门。051已复现原版首token漂移；052准备在同配置加同步张量收据定位最早差异，时序会受诊断影响。队列监控漏报由主会话负责；持久只读watcher已启动，每60秒轮询，真实通知已收到，运行状态见coordination.md。

新任务使用修正的 checked runner、flush证据和评分kit；047打印工具及原数据哈希，每个job打印patch哈希。049使用通用numcheck，flush耗尽、并发请求失败、输出截断都会失败；其logprob阈值仅粗筛，原始差异必须复核。

047/048同请求对照工具已独立验收10项CPU回归：完整cohort/负载一致、原始精度判门、排除warmup与边界秒。T56 LCP仅核元数据/物理上界，来源未验证，仍只作诊断。[对照记录](../evidence/L048-official_a_122_n22/N22/compare_vs_047.txt)、[工具复核](reports/review-compare-sol.md)。

## 新数据与算子线

- **长链数据**：96链/1,718请求诊断候选已冻结并独立验收；36文件hash全过、35项大小匹配、1,718条全量渲染收据与成品零失配，原harness自检通过。cohort=`cd106a80519548d4`；[冻结入口](../evidence/longchain-audit/frozen-candidate/README.md)、[独立验收](../evidence/longchain-audit/independent-freeze-acceptance.json)。后续工作量、长gap/reset和语义关联缺口仍成立，不作代表集、不入队。下一版方案已确认：s1-dev跨session借query/片段适配续接，Phoenix千session观测供行为结构，按[统一设计](../scripts/analysis/longchain.md)实现并验收；旧dev保留独立回归，算子队列不等新数据。见 [交接入口](codex-handoff-长链数据设计与文档索引.md)与[独立复核](reports/review-data-sol.md)。
- **171 KDA投影融合**：开发机加载/状态/MTP/graph与成本证据已交付，真实TP8输出筛选为049。新增每层加载路径日志与旧GPU证据的哈希区别见 patch说明；还缺模型层级状态/质量确认及整档收益。
- **172 MoE clamped SwiGLU**：单卡随机权重完整Marlin路径墙钟减少3.1–4.9%；BF16激活与graph检查通过。独立审查发现并修复探针两臂写同一实现的问题；补测M256原版自身也有数值波动，固定路由后原版重复/候选均逐位一致。单卡门槛通过，真实TP8筛选已安排050；未放宽容差，没有整模型或并发档收益结论。见patch说明与research/codex/R23。
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

## 用户最新优先级：先确认并发收益值得投入

054的定位收据只回答重复输出差异，不是加速成果。固定排序只做有界正确性/成本对照，若仅消除浮点顺序波动且增加耗时，不作为性能优化部署。171目前只有单卡投影成本，172完整单卡MoE快3–5%，均不能承诺正式N22/N26；禁止把局部百分比当服务吞吐百分比。

下一阶段先测正式A真实权重、TP8、无诊断同步下整段prefill/decode的同条件成本、峰值显存和KV容量，选择对服务贡献最大的方向；以整段成本明显下降（主攻筛选目标约10%，不是比赛硬门）或瓶颈TTFT明确改善筛选主攻候选。再做同负载N22完整A/B与重跑估计噪声，有稳定余量再测N26。缓存复用/调度、MoE矩阵乘主体、prefill graph按正式A的时间占比与等待归因排序，不能相加预期收益。172作为低成本小改保留，不单独承担晋档目标。
