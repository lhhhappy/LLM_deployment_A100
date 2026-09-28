# 实验记录

只记能指导下一轮的实验：相对基线的变化、完整数据判定、结论和原始证据。**开发集的通过档位不是正式 N@SLO 预测**。一条档位记录须保留完整 cohort、raw、run/report 和服务日志；`LEVEL` 为 `INVALID` 时不得当成绩比较。历史细节在 git 与 `evidence/`，不重复长篇过程日志。

## 证据索引与归档约定

这里是运行证据的唯一目录索引；实时状态仍以[队列](queue.md)为准。`DRAINED` 表示指定派发窗口的请求已排空，只能作同数据、同 ID 的机制对照；完整档是否有效以原 harness 和完整性收据为准。新运行在队列登记编号与脚本，结束后在此补状态、配置差异、数据哈希、原始证据和结论。原始 raw、服务日志、派发/flush 收据与判分不可因清理文档而删除；Pod 副本须在开发机归档并核 SHA256 后才能清理。

| 运行 | 状态与比较问题 | 证据入口 |
| --- | --- | --- |
| 094 公开 dev 原样开场 | 派发并排空 722/722；原 harness 四类 TTFT 均 FAIL，chain p95 69.69 秒；附加窗口 verdict 因 cohort 文件缺失为 INVALID，不能替代原 harness | [任务目录](../evidence/L094-dev_opening_release0_n22/) |
| 096 K9+117 校准 | 启动阶段按用户要求停止，未形成比较结果 | [任务目录](../evidence/L096-calib_k9v2_117_n22/) |
| 097 v3 初次部署 | 缺失数据检查脚本而落回默认数据；发现后停止，整次作废。098–102 尚未启动即撤队 | [任务目录](../evidence/L097-v3_open_base_n26/) |
| 103 v3/N26 底座 | 开场 600 秒、排空；后续对照基线 | [分析索引](../evidence/L103-v3_open_base_n26/opening/analysis.json) |
| 104 124 | 对 103 同 ID 443 条：chain 超时 19→17，fast 56→26、overall 40→21、turn 1→1；排空 | [分析索引](../evidence/L104-v3_open_124_n26/opening/analysis.json) |
| 105 124+125 | 对 104 同 ID：chain 超时 17→13；排空 | [分析索引](../evidence/L105-v3_open_124_125_n26/opening/analysis.json) |
| 106 124+冷块 8192 | 对 104 同 ID：chain 超时 17→15；排空 | [分析索引](../evidence/L106-v3_open_124_cap8192_n26/opening/analysis.json) |
| 107 124+117 | 启动机制校验失败，未测量；脚本已修正，见 110 | [任务目录](../evidence/L107-v3_open_124_117_n26/) |
| 108 全集 N26 | 用户要求中途停止，不能判分 | [任务目录](../evidence/L108-v3_46364_n26_full/) |
| 109 124+激进 125 | 对 105 同 ID 440 条：chain 超时 13→9（修 4、新 0），turn 2→3，TPOT 均值 55.45→61.84 ms；排空 453/453、错误 0 | [分析索引](../evidence/L109-v3_open_124_125x_n26/opening/analysis.json) |
| 110 124+117 重跑 | Pod done；已取回 500 条唯一 raw、错误 0。对 104 同 ID 452 条：chain 超时 17→18（修 1、新 2），fast 26→33、overall 21→31、turn 1→0；TPOT 均值 50.63→45.56 ms，>0.10 秒 23→4。短测只作机制诊断，不判完整 N26；定时排空收据与本地归档仍待闭合 | [原始 raw](../evidence/L110-v3_open_124_117_n26/N26/raw_s1-dev-longchain-v3_N26_1790410811.jsonl)、[传输 SHA256](../evidence/L110-v3_open_124_117_n26/N26/fetch_status.json) |

上述链接指向索引文件或任务目录，完整 raw 位于各运行的 `window/raw.jsonl`，服务日志和校验收据在同一证据目录。历史编号无需重排；失效、取消的编号要保留原因，避免误用其结果。

## 当前开发集对照

### 110–112、130 系列：候选 A/A′ 与今天的提交依据（2026-09-26，fable；110–112 由 claude 分析）

数据 v3（311 链/5601 条），N30；探针 = 600 秒派发后排空，60 分钟 = 3600 秒派发后排空；同 ID 配对，超时条数按原 harness 的桶。全部行在 [看板](kanban.md)，推断账本在 [program-n30-v3.md](program-n30-v3.md)。引擎 4f9d1f0b（stack2 + 126/128 后续）除非注明。

| 运行 | 对照（同 ID 条数） | chain | turn | overall | fast | TPOT>0.10 | 结论 |
|---|---|---|---|---|---|---|---|
| 110 A′ 前身：124+117，N26 开场 | 104（452） | 17→18 | — | — | — | 23→4 | 117 只降 TPOT |
| 111 底座（103 配置）60 分钟 | — | 53/243 | 12/106 | 742/2566 | 979/2353 | 48 | 四门全挂，等显存（KV 池 99.3%） |
| 112 候选 A = 124+激进 125+117，60 分钟 | 111（2915） | 53→25 | 12→24 | 742→430 | 979→492 | 48→35 | chain/fast/overall 大降；turn 翻倍：124 的 5 s warm 预算把命中过半且新增 >4096 的请求判“来不及”，6 条等到 120 s 饥饿上限 |
| 113 N34 底座 | — | — | — | — | — | — | 启动后 stopjob 腾卡，未测量；改名 136 |
| 130 A 开场探针（引擎 4f9d1f0b） | — | 13/70 | 1/23 | 35/415 | 36/381 | 37 | 130a/130e9 的对照 |
| 130a A′ = A + WARM_S=15 开场 | 130（508） | 13→14 | 1→0 | 35→47 | 36→51 | 37→34 | turn 修好，fast/overall 付代价 |
| 130b A′ 60 分钟 | 112（3063） | 26→30 | 24→16 | 448→512 | 506→554 | 35→37 | S1 依据；冒烟 12/12 |
| 130c A′ + 126 座位 开场 | 130a（516） | 14→13 | 0→0 | 47→44 | 51→43 | 34→33 | 座位在 A′ 上四门略降 |
| 130d A′ + 128 家族领头 开场 | 130a（515） | 14→14 | 0→0 | 47→39 | 51→48 | 34→41 | 家族识别出来了但等待不变，需决策日志 |
| 130e A′ 把 125x 换 122 开场 | 130a（512） | 14→21 | 0→6 | 47→38 | 51→39 | 34→1 | 122 保 TPOT 拖慢冷链首，不用 |
| 130e5 A″ = A + 饥饿上限 10 s 60 分钟（引擎 84dcca0e） | 112（3094） | 27→27 | 24→19 | 449→866 | 508→888 | 35→36 | 不看剩余工作量的提升让大 warm 请求占通道，warm 单轮请求多等 5–19 s；否定 |
| 130e9 A + 126 座位 开场 | 130（507） | 13→15 | 1→2 | 35→33 | 36→39 | 37→39 | 座位在 A 上无收益；一条 1601 token 冷链首等 123 s（疑 126 拒绝状态持续，待查） |
| 130ea A′ + 124 停车修正 开场（引擎 efe837c8） | 130a（510） | 14→14 | 0→2 | 47→34 | 51→40 | 34→39 | 有效停车 41 次对旧 195 次决策：空转轮存在；修正进入 46676/46677 |

提交：46676（S1 = A′，镜像 0926b/引擎 f546934e）、46677（S2 = S1 + MAX_SLOW=250），见 [submissions.md](submissions.md)。

### 093：118 Triton DSA，N30 派发 60 分钟（2026-09-26，Codex）

对081只开启118 DSA sparse kernel，沿用759a6eb其余配置、同一冻结长链、N30、cold6144、host64。启动机制核对`118=on`、能力抽检12/12；原harness派发3600秒后排空3389条，`timed_verdict=DRAINED`、请求错误0。共同3389个ID的结果：

| 指标 | 081 → 093（118） | 同ID的084（117） |
|---|---:|---:|
| chain超30秒 | 23→21（修复4、新增2） | 22 |
| turn超15秒 | 7→5 | 4 |
| fast超3秒 | 238→241 | 229 |
| overall超5秒 | 190→186 | 174 |
| TPOT均值 / p95 | 33.14/61.24→31.87/58.83ms | 30.58/55.07ms |

118对chain与turn有小幅正收益，fast多3条坏例；117在同一3389个ID上对turn、fast、overall和TPOT更好，chain比118多1条超时。两者均只在各自的081基线上单独开启，不能把收益相加或据此断言叠加效果。093实际未命中量比081少132,608 token（0.9%）；这是运行结果，不等于kernel节省的工作。60分钟诊断窗口已排空但不是5601条全量N30，也不能预测正式N26的失败门。

[排空收据与原harness诊断](../evidence/L093-cap6144_118_n30_60m/window/snapshot.json)、[同ID逐请求对照](../evidence/L093-cap6144_118_n30_60m/window/analysis/20260926T050632Z-280971f1/analysis.json)、[时间窗](../evidence/L093-cap6144_118_n30_60m/window/window_gates.txt)。

### 089：K9/N22 完整校准，负载未对齐正式（2026-09-26，Codex）

46364/081 的同一引擎与配置（759a6eb、cold cap6144）在每链前9请求的冻结子集完整运行N22。原 harness 核验1865/1865条、零请求错误、真flush，判 `VALID FAIL`；只失败fast和overall。K9的1865条和104,970,368 prompt tokens接近正式一档的推算量，但四桶结果不同：

| 指标 | K9本地N22 | 正式46364/N22 |
|---|---:|---:|
| fast p95 | 5.73s，超时96/允许75 | 1.86s |
| overall p95 | 6.62s，超时88/允许83 | 3.15s |
| turn p95 | 7.83s，超时3/允许10 | 10.66s |
| chain p95 | 23.23s，超时16/允许24 | 41.97s |
| TPOT均值 / p95 | 28.77 / 73.38ms | 22.61 / 43.79ms |

正式数据不公开四桶超时计数，因此表中只比较两边都有的p95。K9把本地紧门推向fast/overall，却低估正式chain/turn尾部；请求数和总token相近不足以作为校准。后续需校对链首与turn的新增工作分布、缓存历史和链中小请求比例，再用新数据比较机制。监控通知的旧消息发送失败曾令自动采样停在408条；改用独立通知进程后，采样补齐并与原harness全量判分一致。[完整判分](../evidence/L089-calib_k9v2_cap6144_n22/window/snapshot.json)、[窗口统计](../evidence/L089-calib_k9v2_cap6144_n22/window/window_gates.json)。

### 088：K9/N22 首次校准未进入测量（2026-09-26，Codex）

用46364/081同一引擎和配置，在每链前9请求的子集（311链、1865请求、104,970,368 prompt tokens）运行N22。请求与cohort哈希和预期相符，但Pod上`/tmp/ax`指向RAM工作区；数据生成脚本计算正文相对链接时混用了真实源路径与字面目标路径，导致16条预热请求全部`HARNESS_DATA:body_missing`。`run_dev`返回2，缺测量`summary.json`，故**INVALID，不能比较性能**。生成器已统一路径解析，在模拟符号链接路径下验证正文分片可见、样本可读；089以新数据根和run名重试，配置不变。

### 086：117 Humming，N34 派发窗口主动中止（2026-09-26，Codex）

相对082只开启117；用户改为优先校准本地与正式负载后，按单job停机流程在完成1038条时结束，未完成60分钟窗口或全量。共同已完成1038条相对082：fast超时251→222、overall195→175、turn5→5、chain31→27；TPOT均值42.31→38.39ms、p95 98.13→84.37ms，请求错误0。这些是开放窗口证据，**不能判N34通过或失败**。窗口另有未完成请求，不能由已完成请求推断完整负载尾部。[同ID窗口分析](../evidence/L086-cap6144_117_n34_60m/window/analysis/20260926T013108Z-6b69e3e0/analysis.json)。

### 084：117 Humming FP8 MoE，N30 派发 60 分钟（2026-09-25，Codex）

对081仅把sm80块FP8 MoE切换到117 Humming路径（引擎`26002a14`），其他引擎开关、host64、冷块6144、N30与冻结数据不变。原harness派发3600秒并排空3502条，零请求错误。共同3502个ID对081：fast超时238→230、overall190→175、turn7→4、chain23→22（修复5、新增4）；TPOT均值32.90→30.33ms（−7.8%）、p95 60.39→54.40ms，TPOT>0.10秒的请求36→12。以各运行首条派发为起点，前3600秒完成请求3250→3482条（+232、+7.14%）；这是混合服务完成量，不是MoE kernel净提速。实际未命中量少211,456 token（1.4%），也是运行结果，不能单独归因于MoE算子。

117在真实TP8混合负载中减少了整体等待与解码耗时，是目前执行层最有希望的候选；chain净改善仅1条，尚不能证明N34晋档。此项是60分钟短窗，不能判完整N30或正式成绩。086的N34筛选按用户要求主动中止，部分结果见本页；先用089校准本地负载与正式N22的差异。

[同ID对照](../evidence/L084-cap6144_117_n30_60m/window/analysis/20260925T233432Z-52cc39de/analysis.json)、[60分钟完成量与raw SHA](../evidence/L084-cap6144_117_n30_60m/window/completion_60m_comparison.json)、[短窗统计](../evidence/L084-cap6144_117_n30_60m/window/window_gates.json)、[机制说明](../research/codex/R32_117_humming_effect.md)。

### 083：127 深分叉检查点，N30 派发 60 分钟（2026-09-25，Codex）

对完整 081 的配置仅换 127 提示词末尾检查点机制（6c9b6648）；同一冻结链、N30、冷块 6144、host64、122/123off、interval2、MTP。原 harness 派发 3600 秒后排空 3281 条，零请求错误；这是筛选窗口，不是完整 N30 判分。共同 3281 个已完成 ID 对 081：chain 超时 23→21（修复 4、新增 2），fast 236→258，overall 188→202，turn 7→4；TPOT 均值 33.36→33.31ms、p95 61.70→62.66ms，实际未命中量少 193,280 token（1.3%）。

缓存机制有真实效果，但远小于原先从静态分叉点估计的 9% 整场潜力，且这次快请求等待变差；127 暂不晋级为 N34 主方案，也不据此否定后续链轮次的检查点价值。082 的新证据指向 N34 设备 KV 驻留与 host 恢复层级迁移，详见 [R31](../research/codex/R31_n34_waiting_bottleneck.md)。117 的后续执行层结果见本页 084。

[同 ID 对照](../evidence/L083-cap6144_127_n30_60m/window/analysis/20260925T222438Z-31ee0954/analysis.json)、[原窗口统计](../evidence/L083-cap6144_127_n30_60m/window/window_gates.txt)、[完整短窗 raw](../evidence/L083-cap6144_127_n30_60m/window/raw.jsonl)。

### 085：attn-TP输入scatter，N30派发60分钟（2026-09-25，Codex）

对081仅在759a6eb/host64/cold6144/122/123off/interval2/MTP上加`--enable-attn-tp-input-scattered`；能力抽检打开。服务日志TP0–TP7均打印`attn_tp_input_scattered is enabled`，能力12/12。原harness派发3600秒后自然排空3349条、零请求错误；`timed_window.json`为DRAINED、无未完成请求。相对完整081仅比较共同的3349条ID，不能将短窗判作完整N30门。

| 共同ID指标 | 081 → 085 |
|---|---:|
| fast超3秒 | 238→257 |
| overall超5秒 | 190→195 |
| turn超15秒 | 7→7 |
| chain超30秒 | 23→21（修复4、新增2） |
| TPOT均值 / p95 | 33.23/61.54→32.31/58.26ms |
| 实际未命中工作 | +191,744 tokens（+1.3%） |

scatter在这一窗口略减chain超时并改善TPOT，却新增19条fast坏例、5条overall坏例；不是无代价的整档提速。较前2287条开放窗口，fast代价在后半段继续扩大。离线B峰值约1459k tokens、排空残差−128；混合时段decode墙钟占比估计57.9%（覆盖3454秒），不是GPU实测。样本量、能力冒烟和输出合同足以支持继续研究，不足以证明正式能力与完整N30门；重算差异亦不能直接归因为scatter改变缓存机制，需结合到达与排队账分析。087计划在117基础上检验只对大块scatter是否消除fast代价，但该假设尚未由085的分桶证实；任务尚未启动，按校准优先级暂缓。

[同ID最终对照](../evidence/L085-cap6144_scatter_n30_60m/window/analysis/20260925T211458Z-97aba312/analysis.json)、[排空诊断](../evidence/L085-cap6144_scatter_n30_60m/opening/analysis.json)、[派发收据](../evidence/L085-cap6144_scatter_n30_60m/opening/source/timed_window.json)、[最终raw快照](../evidence/L085-cap6144_scatter_n30_60m/window/raw.jsonl)。Pod原run的`job.log`显示能力12/12；归档服务日志含TP0–TP7生效行及排空后SIGTERM（剩余请求0）。

### 082：cold6144，全量长链N34（2026-09-25，Codex）

对081只把回放并发N30改为N34；固定759a6eb、host64、122/123off、interval2、mem.87、MTP、rep16-v1和同一冻结311链/5601请求。原harness完整回放与真flush收据有效，5601条恰好完成一次、零请求错误。本地原harness加题面统计余量判定 **VALID FAIL**：fast、overall、chain三门失败，turn与TPOT等其余八门通过；不是正式N34判定。

| 指标 | 081 N30 → 082 N34 | 082超时 / 允许 |
|---|---:|---:|
| fast TTFT p95 | 3.1305→7.5985s | 559 / 263 |
| overall TTFT p95 | 3.8667→8.1013s | 409 / 276 |
| turn TTFT p95 | 13.1832→17.2926s | 9 / 13 |
| chain TTFT p95 | 30.2534→62.5416s | 39 / 29 |
| TPOT均值 / p95 | 28.458/52.058→30.764/56.623ms | p95门通过 |

fast超时比081多317条，overall多216条，chain多16条；运行中共同5200条快照已显示相同方向。082完整raw里超时请求的首入批前等待中位数：fast约6.0秒、overall约8.6秒、chain约65.1秒；入批至首token分别约0.6/0.7/5.0秒。完整同ID 5601条中，实际未命中工作较081多161,024 tokens（约0.8%），不足以单独解释大幅TTFT退化；这仍不是逐批因果分解。TPOT超0.10秒47/5601，集中在开场时段；不能把其均值变化当稳定噪声外收益或损失。N34已不是081所示的“仅chain差一门”形态，需要同时看排队、fast余量和prefill产能。

[完整本地判分](../evidence/L082-cap6144_full_n34/N34/level_verdict.json)、[原harness统计](../evidence/L082-cap6144_full_n34/N34/score_formal.json)、[坏例表](../evidence/L082-cap6144_full_n34/N34/badcases.csv)、[传输收据](../evidence/L082-cap6144_full_n34/N34/fetch_status.json)。完整raw、run与服务日志已取回，本地复算与Pod判分一致。

### 081q：126按需留位，N26开场（2026-09-25，Codex）

相对078仅改用engine 126（8f62c5f2）并开启按需留位：冷块下限4096、上限6144，给等待中可完整执行的短命中留位。固定host64、122/123off、interval2、MTP和rep16-v1。派发600秒后全部排空381条、零请求错误；原harness诊断收据为DRAINED，raw SHA与派发台账一致。首分钟到达chain 15/28、整窗16/67超过30秒。对078共同379条：chain 16→16（修复0/新增0），fast 33→15，overall 34→17，turn 0→0；TPOT超过0.10秒为33→37。对074共同370条：chain 21→16（修复5/新增0），fast 23→15，overall 30→17，turn 0→0；TPOT超过0.10秒为1→37。

全窗TPOT均值/p95/最大45.89/107.54/125.39ms，超过0.10秒37/381（9.71%）。离线B峰值约1296.7k tokens、排空残差0；混合时段decode墙钟占比估计39.9%（覆盖561秒，非GPU实测）。本次留位保住了078失去的fast/overall请求，且共同ID的chain收益未退；TPOT超线条数略增。短测仅诊断，不能据此判完整N26或保证N30收益。日志中的Traceback发生在排空后的SIGTERM，退出时剩余请求0，未影响已完成请求。

[自动分析](../evidence/L081q-opening_126_demand6144_n26/opening/analysis.json)、[对074](../evidence/L081q-opening_126_demand6144_n26/opening/comparison.json)、[对078](../evidence/L081q-opening_126_demand6144_n26/opening/comparisons/078-opening_q1b_cold6144_n26/comparison.json)、[排空收据](../evidence/L081q-opening_126_demand6144_n26/opening/source/timed_verdict.json)。

### 081p：TP8冷prefill 8k/16k块的执行时间账（2026-09-25，Codex）

固定759a6eb、host64、122/123off、MTP，原harness预热后真flush，同一个49,143-token冻结请求单独执行、输出1 token；两次均cached=0、错误0。rank0/1 profiler相符。8k与16k的客户端E2E分别4.260/4.279秒；EXTEND标记分别12步（约8192 tokens/步）与6步（约16381 tokens/步），标记数约为物理块数的两倍，疑含target/draft两段，不能当物理块数。两档EXTEND GPU标记总时长约4.12/4.14秒，连续EXTEND之间无kernel间隙总量约21/8毫秒。增大块长没有显示单请求净收益；profile有扰动，不代表混合负载吞吐。

rank0的8k/16k EXTEND kernel时长分类约为MoE 33/33%、通信12/11%、dense GEMM 10/9%、KDA 9/9%、mHC/norm 12/11%、未归类26/26%；这不是关键路径占比。未归类含约0.72秒的泛名`main_kernel`，不能据分类中的DSA实名0%断言注意力免费。可辨认的Marlin MoE总时长两档均约1.17秒，NCCL all-reduce约0.40秒。16k的nvidia-smi采样峰值较各自预热后基线多约1818MiB/卡，8k仅4MiB；rank0绝对峰值78416MiB对76538MiB。单样本只支持优先追每token算子成本与未归类kernel，尚不支持把16k加入服务配置。

[8k账本](../evidence/L081p-tp8_prefill_profile_8k_16k/tp8_8k/ledger-rank0-rank1.json)、[16k账本](../evidence/L081p-tp8_prefill_profile_8k_16k/tp8_16k/ledger-rank0-rank1.json)、[六文件SHA清单](../evidence/L081p-tp8_prefill_profile_8k_16k/transfer-manifest.json)。完整trace保留在GPU开发机归档与Pod原run。

### 081：cold6144，全量长链N30（2026-09-25，Codex）

对完整069仅把 `SGLANG_AX_SCHED_COLD_CAP` 从4096改为6144；固定759a6eb、host64、122/123off、interval2、mem.87、MTP、rep16-v1和同一冻结311链/5601请求。原harness完整回放与真flush收据均有效，5601条恰好完成一次、零请求错误。原harness加题面统计余量判定 **VALID PASS，11道硬门全部通过**；这是本地开发集结果，不能换算正式N@SLO。

| 指标 | 069 → 081 | 081超时 / 允许 |
|---|---:|---:|
| fast TTFT p95 | 2.7039→3.1305s | 242 / 263 |
| overall TTFT p95 | 3.6268→3.8667s | 193 / 276 |
| turn TTFT p95 | 13.6406→13.1832s | 7 / 13 |
| chain TTFT p95 | 40.3355→30.2534s | 23 / 29 |
| TPOT均值 / p95 | 28.824/55.902→28.458/52.058ms | p95门通过 |

chain超时比069少8条，fast多20条；完整共同5601条的四桶计数与窗口后段的同ID结果一致。chain p95数值略高于30秒，但题面统计余量允许29条超时，实测23条，因此本地整档通过。**turn p95虽然略降，同159个请求的超15秒坏例却是修复4条、新增4条；20条TTFT增加超过3秒。**正式46251→46364的turn p95同时从6.58升至10.66秒（+62%），本地汇总p95不能代表正式尾部。单配置各一次，不能把小幅TPOT变化当成稳定提速；6144仍消耗部分fast余量。[turn逐请求审计](../research/codex/R33_turn_start_regression.md)。

[完整判分与本地复核](../evidence/L081-cap6144_full_n30/N30/level_verdict.json)；[传输收据](../evidence/L081-cap6144_full_n30/N30/fetch_status.json)记录1,604,070字节归档的SHA256 `bba6cd9f5e4537938ef9ed2eb4a77ae99f538c22a845dd2d96a92adfbd90285b`。完整raw、run、服务日志取回后，本地原harness加题面余量复算与Pod判分一致。

### 079：123 aging300 + cold6144，N26开场（2026-09-25，Codex）

对076仅COLD_CAP 4096→6144，固定37e90023/interval2/host64/122off/rep16。600秒派发后排空375条、零请求错误。首分钟chain13/27、整窗15/68超30s。对074共同366条：fast23→38、overall30→44、turn0→1、chain21→15（修复7/新增1），TPOT>0.10为1→35；对076共同366条：fast12→38、overall17→44、turn1→1、chain19→15（修复4/新增0），TPOT>0.10为2→35。

全窗TPOT均值/p95/max45.67/108.91/132.67ms，超线35/375（9.33%）；离线B峰值1297.9k、终点残差0。decode墙钟估计36.9%（覆盖559秒，非GPU实测）。相对076改善chain，但fast/overall和TPOT代价明显；组合未保留单独排序时的短请求表现。079与078的直接比较等统一共同ID表，不能拿不同配对集合相减。仅开场诊断，不判完整档。

[对074](../evidence/L079-opening_q2_srpt300_cold6144_n26/opening/comparison.json)、[对076](../evidence/L079-opening_q2_srpt300_cold6144_n26/opening/comparisons/076-opening_q2_srpt300_n26/comparison.json)、[自动分析](../evidence/L079-opening_q2_srpt300_cold6144_n26/opening/analysis.json)。完整分析及raw已自动同步本地、21文件SHA验证。14:55:18服务告警来自排空后的正常换引擎，见[核查](../evidence/L079-opening_q2_srpt300_cold6144_n26/opening/shutdown-alert-review.json)。

### 078：cold6144，N26开场（2026-09-25，Codex）

对074仅COLD_CAP 4096→6144，仍759a6eb/123off/interval2。600秒派发后排空382条、零错误；首分钟chain14/27、整窗16/68超30s。对074共同371条：fast23→32、overall30→33、turn0→0、chain21→16（修复5/新增0），TPOT>0.10为1→33。

全窗TPOT均值/p95/max45.07/107.54/124.20ms，超线33/382（8.64%）；离线B峰值1299.1k、终点残差0；decode墙钟估计38.7%（覆盖556秒，非GPU实测）。有链首收益，也消耗fast/TPOT余量；只作开场诊断。18个完整分析与raw文件已同步本地并核SHA。[对074](../evidence/L078-opening_q1b_cold6144_n26/opening/comparison.json)、[分析](../evidence/L078-opening_q1b_cold6144_n26/opening/analysis.json)。

### 077：prefill-decode-interval 2→1，N26开场（2026-09-25，Codex）

相对074只改固定间隔，仍759a6eb、cold4096、123/122off。600秒派发并排空353条、零请求错误；首60秒chain16/27、整窗18/64超30s。同074共同352条：fast23→19、overall30→20、turn0→1、chain21→18（修复4/新增1）；TPOT>0.10为1/352→61/352。

全窗TPOT均值/p95/max为52.38/146.14/170.94ms，超0.10为61/353（17.28%）；离线B峰值1304.6k、终点残差0。decode墙钟占比估计36.0%（覆盖548秒），不是GPU实测。chain净少3条但TPOT代价较大，本轮不将interval1叠进079；不把短窗判成完整档。排空后正常换引擎产生SystemExit0/CancelledError，退出剩余请求0。

[自动对照](../evidence/L077-opening_q3_interval1_n26/opening/comparison.json)、[分析](../evidence/L077-opening_q3_interval1_n26/opening/analysis.json)、[告警复核](../evidence/L077-opening_q3_interval1_n26/opening/shutdown-alert-review.json)。

### 076：123 aging300，N26开场（2026-09-25，Codex）

相对074打开123 aging300，源码37e90023还包含必要的TP排序一致性修复：rank0产生RID顺序后广播，各rank按本地Req映射。其余host64/122off/cold4096/interval2/rep16不变。600秒派发并排空374条、零请求错误；首60秒chain15/28、整窗19/65超30s。同074共同368条：fast23→12、overall30→17、turn0→1、chain21→19（修复5/新增3）；TPOT>0.10为1/368→2/368。

全窗TPOT均值/p95/max43.69/87.76/104.07ms，超0.10为2/374（0.53%）；离线B峰值1284.7k、终点残差0。decode墙钟占比估计38.4%（覆盖570秒），不是GPU实测。短测中排序的整体取舍优于8192冷块或interval1；chain收益仍小，不能据此承诺正式更高N。未隔离测量广播本身的墙钟开销。14:02:48告警已确认为排空后正常换引擎，退出剩余请求0。

[自动对照](../evidence/L076-opening_q2_srpt300_n26/opening/comparison.json)、[分析](../evidence/L076-opening_q2_srpt300_n26/opening/analysis.json)、[告警复核](../evidence/L076-opening_q2_srpt300_n26/opening/shutdown-alert-review.json)。

### 075：冷块上限4096→8192，N26开场（2026-09-25，Codex）

相对074仅COLD_CAP=8192，仍759a6eb/host64/122off/rep16/N26，600秒派发并排空369条、零错误。首60秒chain15/27、整窗17/67超30s。共同360条：chain21→17（修复4/新增0）、fast23→63、overall30→56、turn0→2；TPOT>0.10为1/360→55/360。全窗TPOT均值/p95/max为45.86/121.98/145.71ms，超0.10为55/369（14.91%）。

离线B峰值约1301k、终点残差0；decode墙钟估计38.2%（覆盖540秒），非GPU实测。结果支持更大冷块能改善一些chain，但明显挤压其他请求，不能直接作为整体晋级配置；不把短测判成完整N26失败，也不废弃机制。076单测排序、077单测interval1继续。13:45:38告警发生在排空后的正常引擎更换路径，SystemExit0/CancelledError，退出时剩余请求0。

[自动对照](../evidence/L075-opening_q1_cold8192_n26/opening/comparison.json)、[分析](../evidence/L075-opening_q1_cold8192_n26/opening/analysis.json)、[告警复核](../evidence/L075-opening_q1_cold8192_n26/opening/shutdown-alert-review.json)。

### 073/074：0925a原引擎配置，N22/N26开场10分钟诊断（2026-09-25，Codex）

用户要求先停止072r并短测。两档固定759a6eb、host64/122off、mem.87、MTP和rep16；同一冻结长链负载，600秒关闭新派发后全部自然完成。派发台账、原harness统计、真flush和输出合同均核验；**DRAINED诊断，不是全量VALID/PASS**。同配置下只改变并发；实际到达随闭环完成改变。

| 指标 | 073 / N22 | 074 / N26 |
|---|---:|---:|
| 完成请求 / 请求错误 | 350 / 0 | 372 / 0 |
| 首60秒到达chain超时 / 样本 | 13 / 23 | 18 / 27 |
| 首600秒到达chain超时 / 样本 | 15 / 60 | 21 / 65 |
| TPOT均值 / p95 (ms) | 37.46 / 83.62 | 43.09 / 86.94 |
| TPOT >0.10 条数 / 实际分母 | 2 / 350 | 1 / 372 |
| 离线B峰值 / 排空残差（tokens） | 1,166,460 / 0 | 1,307,654 / −576 |
| 首分钟日志prefill新token / 墙钟秒 | 9,174 | 9,279 |
| 混合时段decode墙钟占比（模型估计） | 38.5%，覆盖547秒 | 39.0%，覆盖574秒 |

共同60条chain超时15→21，修复0、新增6。超时主要集中首分钟；前3分钟两档日志prefill吞吐均约8.5–9.3k token/s，增加并发没有带来对应吞吐提升，支持继续研究开场工作量、服务顺序和prefill单位成本。短测未验证排序或节拍干预的反事实收益，正式失败档原因仍未知。

decode占时来自同轮纯decode窗口及069同引擎拟合的估计，含CPU开销，不是GPU忙时实测。无decoder的连续prefill参考只有N22的7段/2.55秒、N26的3段/1.07秒，约10.19k/9.82k token/s，样本不足以判断稳定孤立产能。后段KV日志最高已接近/达到1.00，不能把“开场不缺显存”推广到全部10分钟。

自动证据：[N22](../evidence/L073-official_0925a_opening_n22/opening/analysis.json)、[N26](../evidence/L074-official_0925a_opening_n26/opening/analysis.json)、[同ID比较](../evidence/L074-official_0925a_opening_n26/opening/comparison.json)。raw、派发ledger及计分收据在GPU同run的window/opening和Pod原run；本地已取回自动分析及来源文件并核验SHA。两档结束；用户随后批准075–077独立探针。13:37更新：原opening_report的p95插值改为原harness floor(q*n)，与074 timed_score逐项一致。B曲线是实际到达账减页对齐batch工作、加末页padding修正；残差保留，不能当精确GPU待算量或清空时刻。

### 069：host容量32→64GB/rank，全量长链N30（2026-09-24，Codex，独立复核通过）

相对068只扩HiCache host预算，122仍off；源码759a6eb、mem0.87、MTP、冻结311链/5601请求N30一致。
运行工具2837b3c，29个运行文件与068逐字节相同。KV/indexer与KDA host一起扩，8卡预算256→512GB；
实测device KV=1,397,760、KDA=418不变，host FULL=2,903,808 token、KDA host=23.73GB/rank。
启动180秒，rep16-v1预热112.832秒，同计划/原预算，20:06:39 UTC真flush。
完整5601条每条恰好一次、原prompt/输出预算/gap一致、零错误、TTFT全部服务端打点；测量6455.19秒（107.59分钟）。
原harness及题面补充门复算 **VALID FAIL：10/11通过，仅chain失败**，CP/Wilson/Wald整档结论一致。

| 门 | 样本 | 068→069 p95秒 | 超标 / CP允许 | 069结果 |
|---|---:|---:|---:|---|
| fast_intra | 4765 | 8.3800→2.7039 | 222 / 263 | PASS |
| overall_intra | 5010 | 9.6509→3.6268 | 195 / 276 | PASS |
| turn_start | 159 | 20.8841→13.6406 | 7 / 13 | PASS |
| chain_start | 432 | 100.3816→40.3355 | 31 / 29 | FAIL |
| TPOT p95 | 5601 | .072501→.055902 | 2条>.10 | PASS |

TPOT均值.034262→.028824；固定[10,70)分钟稳态窗3444条，有效，TPM(all) 2,733,104.87→3,666,963.70，decode 29,532.55→41,081.13。
全量未命中prompt 30,809,642→19,131,690 token；测量内部prefill批9707→7293、new tokens约30.98M→19.31M，单序列且pending批5168→2545。
后一计数是日志形态，不证明每一批都由某个拒绝分支产生。host FULL占用中位99.903%，89.11%样本≥98%；仍不含KDA host，不能凭此认定具体淘汰。

同ID四桶修复/新增：fast367/128、overall327/112、turn12/6、chain24/5；唯一TTFT坏例589→311。
31个chain坏例为30个cohort链首+1个非链首context_reset，均在前38.377分钟发送；20条queue_time≥80%TTFT，2条exec→first本身>30秒。
其中lc302:0017两轮cached45,824相同却TTFT .369→82.872秒；最终cached不能揭示等待时状态层级，未证实120/180 host拒绝分支是该案例原因。
82.135分钟起剩链<30，低延迟尾段不当满载产能；前40分钟TPOT p95约.07311，也应跟踪前段解码压力。
一次完整对照支持保留host64继续研究，未测重跑噪声，不预测正式N30。

070保持069全部条件，只重新开启修复122（τ=.085），检验较低重算量下的预算/节奏取舍；不承诺能修两条就稳定过门。
CPU真实调度器47项通过（当前759的HiCache/122，历史机制基准保留），独立参与者复核原始数据后支持设计，无阻断理由。
全量比较311个固定坏例与新增坏例、四门点估计/统计余量、TPOT、前段时间账，不删保护、不改预热集。

取证误漏`--data-root`曾用旧dev集产生INVALID（extra4879）；保留原报错，正确冻结集重新判分，原raw/run未改。
后续显式使用`fetch_level.sh <run> 30 --data-root data/s1-dev-longchain`。
证据：[最终审计](../evidence/L069-official_b_pace_off_host64_full_n30_shortwarm/N30/final-audit.json)、[完整判分](../evidence/L069-official_b_pace_off_host64_full_n30_shortwarm/N30/level_verdict.json)、[5601条对照](../evidence/L069-official_b_pace_off_host64_full_n30_shortwarm/N30/compare_vs_068.csv)、[摘要](../evidence/L069-official_b_pace_off_host64_full_n30_shortwarm/N30/compare_vs_068.txt)、[311条坏例](../evidence/L069-official_b_pace_off_host64_full_n30_shortwarm/N30/ttft-cases.csv)、[独立复核](../evidence/L069-official_b_pace_off_host64_full_n30_shortwarm/N30/independent-review.json)。

### 068：关闭122，全量长链N30（2026-09-24，Codex，独立复核通过）

相对067唯一引擎变化为PACE .085→0，同源码759a6eb、mem0.87、新版180、MTP与32GB/rank HiCache。
关闭122同时恢复固定decode interval=2与冷块4096，不能把差异归给单一预算常数。
工具e77933d增加已有HiCache指标、收紧预热校验；采样仍10秒，观测开销没有单独量化。
同rep16计划预热114.722秒、真flush，完整311链/5601条每条恰好一次，原预算/gap一致，0错误。
约125分钟完成，原harness复算 **VALID FAIL**：三道TTFT失败，其余8门按当前CP通过。

| 门 | 样本 | p95秒 | 超标 / CP允许 | 结果 |
|---|---:|---:|---:|---|
| fast_intra | 4765 | 8.3800 | 461 / 263 | FAIL |
| overall_intra | 5010 | 9.6509 | 410 / 276 | FAIL |
| turn_start | 159 | 20.8841 | 13 / 13 | PASS（方法敏感） |
| chain_start | 432 | 100.3816 | 50 / 29 | FAIL |
| TPOT | 5601 | .072501 | 4条超过.10 | PASS |

TPOT均值.034262；固定[10,70)分钟TPM 2,733,104.87、decode 29,532.55，窗口有效。
turn_start按CP/Wald通过、Wilson允许12故失败；整档所有方法均FAIL，不当作稳健过门或官方预测。

全部5601条同ID比较：fast修复282/新增251，overall251/260，turn7/6，chain19/14。
唯一TTFT坏例616→589，但fast/overall/chain的p95和TPOT均回退，关闭122没有整体优势证据。
测量内部prefill批7178→9707、new tokens约32.81M→30.98M、partial-only2847→5168；更少工作量没有兑现为整体延迟收益。
缓存总量361.26M→363.08M，不能由这一个对照独立区分缓存、到达时序和调度的作用。

068测量窗744个样本：FULL host占用中位99.903%，97.45%样本≥98%。这只是FULL KV，不含KDA host；
`evicted_tokens_total`计device KV淘汰，不能当作host淘汰；write_through下dropped=0也不排除host churn。
lc139:0002两轮cached=0，TTFT288.65→265.63秒；前驱结束到执行326.57→301.78秒，不能只用45.82秒发送gap推测状态寿命。
真实LCP证明有可复用文本，不证明有效混合检查点存在、被淘汰或恢复成功。589条唯一坏例已冻结，下一轮逐ID跟踪。

069以068为基线，只扩host预算32→64GB/rank，122继续off；检验总host容量干预，不宣称已经定位缓存淘汰bug。
KV/indexer与KDA host份额一起增加，8卡总预算256→512GB，GPU池预算不变；CPU源码尺寸函数与cgroup余量筛查通过。
实际容量、启动、短预热与flush必须再次核对；容量无收益再补定向保存/淘汰/恢复trace。独立参与者复核全量原始数据后支持此设计，无阻断问题。

证据：[完整判定](../evidence/L068-official_b_pace_off_full_n30_shortwarm/N30/level_verdict.json)、[5601条对照](../evidence/L068-official_b_pace_off_full_n30_shortwarm/N30/compare_vs_067.csv)、[对照摘要](../evidence/L068-official_b_pace_off_full_n30_shortwarm/N30/compare_vs_067.txt)、[589条坏例](../evidence/L068-official_b_pace_off_full_n30_shortwarm/N30/ttft-cases.csv)、[哈希](../evidence/L068-official_b_pace_off_full_n30_shortwarm/N30/case-list-manifest.json)、[指标口径](../evidence/L068-official_b_pace_off_full_n30_shortwarm/N30/cache-runtime-summary.json)、[069预算筛查](../evidence/L069-official_b_pace_off_host64_full_n30_shortwarm/budget-probe.json)。

### 067：组合全量长链N30（2026-09-24，Codex）

正式A + mem0.87 + 新版180 + 修复122，MTP保留；引擎759a6eb、运行工具296caaa。
rep16-v1预热101.696秒后真flush；完整311链/5601请求每条恰好一次，原输出预算全部匹配、0错误，TTFT全部服务端打点。
本地独立原harness复算为**VALID FAIL**：四类TTFT失败，其余7门通过。测量约123分钟，无人工截断。

| 门 | 样本 | p95秒 | 超标 / CP允许 | 结果 |
|---|---:|---:|---:|---|
| fast_intra | 4765 | 7.6457 | 492 / 263 | FAIL |
| overall_intra | 5010 | 8.5156 | 401 / 276 | FAIL |
| turn_start | 159 | 25.0138 | 14 / 13 | FAIL |
| chain_start | 432 | 93.7016 | 55 / 29 | FAIL |
| TPOT | 5601 | .067469 | 0条超过.10 | PASS |

TPOT均值.033498秒/token。原固定稳态窗[10,70)分钟2585条，有效：逻辑TPM 2,787,756.35、decode TPM 29,805.45。
这些是本地冻结合成集结果，不与正式榜单直接排名；CP是本地估计口径，Wilson/Wald的整档结论也均FAIL。

冻结616个唯一TTFT坏例（桶重叠，不能把超标数相加）。fast坏例455/492在前60分钟；97.88分钟后剩余链<30，尾段不当满N30。
坏例recv→exec中位依次约5.4/8.0/21.1/63.4秒；该间隔不能直接归因调度。8条慢fast已验证prompt LCP大于cached，尚不能证明有效混合状态曾存在或被淘汰。
下一项068只切122 off；比较整个节奏/块预算机制，不预设净收益，不同时改成本模型或180。

[完整判定与收据](../evidence/L067-official_b_full_n30_shortwarm/N30/level_verdict.json)、[唯一坏例CSV](../evidence/L067-official_b_full_n30_shortwarm/N30/ttft-cases.csv)、[数据与raw哈希](../evidence/L067-official_b_full_n30_shortwarm/N30/case-list-manifest.json)、[过程与归因边界](reports/sglang-shortwarm-mainline-0924.md)。

| 实验 | 只改什么 / 环境 | 完整档结果 | 结论与证据 |
|---|---|---|---|
| 026 N18 | S0：底包 + `000 101 106 110 111 120 140`；完整 `dev-combined-v1` | 722 条；四道 TTFT 门通过，`tpot_mean=0.083`、`tpot_p95=0.219`，整档 FAIL | 解码 p95 是硬门；真实链内缓存缺口上限约 8%，不能用旧的冻结 token 差值归因。原始 [raw](../evidence/T53/026_N18_raw.jsonl)、[独立审阅](../research/codex/R19_progress_and_cache_review.md)、[真实 LCP](../research/codex/R20_true_lcp_attribution.md) |
| 028 N18 | 相对 S0 同时改变 MTP、114、140 开关、调度 cap/interval、池容量等 | `tpot_mean≈0.053`、`tpot_p95≈0.082` 通过解码门；fast/overall/chain TTFT FAIL | 多变量结果，只说明该组合不能过整档，不能把解码收益单独归给 MTP。[旧 8 卡账本审查](../evidence/T55/B1-notes.md)列出限制；需原始记录复核 |
| 034 N18 | 正式提交 B 的同配置开发集测试；相对 S0 多变量 | 已记录 FAIL，`tpot_p95≈0.210`；本地缺完整 raw | 不能预测正式 B 成绩，也不能用于单因素归因。补齐 raw 后再重判；见 [审查清单](../evidence/T55/B1-notes.md) |
| 035 N22 | 重测 S0，完整 722 请求 | 11 门过 10，只有 `tpot_p95=0.296193` FAIL；`tpot_mean=0.103956` | 以完整原始记录、原 harness + 题面规则复算；205 条请求超过 0.10 秒线。[原始记录与分数](../evidence/L035/)、[独立完整性复核](../evidence/T58/)、[205 条归因](../research/codex/R21_N22_tpot_failures.md) |
| 036 N22 | S0 + 114 indexer row sharding，其他条件沿用 035 | 722 条、0 错、VALID；fast 3.05→2.05s、overall 5.10→3.73s，`tpot_mean` 0.104→0.094，`tpot_p95` 0.296→0.253；仅 TPOT p95 FAIL | 单变量方向有效，仍需解决重 prefill 时的 decode 停顿。192 条请求超过 0.10 秒线；chain_start 17/22 是本档最窄的 TTFT 条数余量，不外推正式各门难度。本地 [原始 raw、run、服务日志与复算](../evidence/L036/) 已保存；`score_formal.py` 独立重跑与记录一致 |
| 037 N22 | S0 + `--prefill-decode-interval 16`，对照 035 | 722 条、0 错、VALID；`tpot_mean=0.05995`、`tpot_p95=0.07754`，0 条超过 0.10 秒线；overall 28/27、turn 7/3、chain 87/22，整档 FAIL | 16 轮解码让 TPOT 过门，却严重拖慢 prefill 和排队。fast p95=3.33s 虽超目标值，19/23 在统计余量内，仍 PASS；不可按点估计误判。[L037 原始记录与判分](../evidence/L037/)已用 `score_formal.py` 独立复算，结果一致 |

### 09-24 00:40–00:47 UTC：Codex只读复核补充

037b–042这批队列已结束，037c/d均未通过N22，没有自动进入N26。下列完整档均有722条、0请求错误，并用原harness与题面规则CPU重算；041没有完整档。各门明细、配置类型和源码限制见 [codex-分析](codex-分析-2026-09-24.md)，[复算JSON](../evidence/queue-review-20260924/rescored_levels.json)。

| 实验 | 相对基线只改什么 / 环境 | 完整档结果 | 结论与证据 |
|---|---|---|---|
| 037b N22 | S1+固定16轮decode，对照036 | TPOT mean/p95 .0562/.0695，0超标；overall29/27、turn5/3、chain78/22失败 | 固定间隔仍过度挤压prefill；[L037b](../evidence/L037b/) |
| 037c N22 | S1+122，target=.17 | TPOT .0783/.1424，187超标；chain33/22，整档FAIL | 相对固定16轮减轻TTFT回退，但未同时守住chain与TPOT；公式按固定成本估计，不是实测反馈；[L037c](../evidence/L037c/) |
| 037d N22 | 037c+123，aging=2000 | TPOT .0810/.1466，203超标；chain23/22（CP；Wald允许23），TPOT使整档在各方法下仍FAIL | chain超标33→23，小于64k未命中的超标chain21→8；仍不能抢占当前partial，长请求可等到110秒；[L037d](../evidence/L037d/) |
| 038 N22 | S1关闭140 | TPOT .0979/.2586，202超标；四TTFT过，整档FAIL | 同链LCP对齐缺口1,304,448，大于两次S1的863,936–916,480；关闭无改善证据，先保留140；[L038](../evidence/L038/) |
| 039 N22 | 038+MTP及配套graph/backend/槽位配置 | TPOT .0772/.2039，186超标；overall29/27、chain30/22，整档FAIL | 组合改善decode但新增TTFT失败；KV965,504，不能单独归因MTP；[L039](../evidence/L039/) |
| 040 N22 | S1状态槽200→400 | TPOT .0861/.2265，182超标；四TTFT过，整档FAIL | 同链对齐缺口450,496，约减半；全prefill只少5%–6%，KV容量少22%；p95相对042仅改善.0058，需复验；[L040](../evidence/L040/) |
| 041 | S1+DCP8 | 冒烟阶段引擎异常，未进入完整N22 | `dcp/comm.py:293`目标[33,1,512]、输入[40,1,512]；疑似TP8/scatter补齐契约不一致，未修复，不能评价性能；[服务日志](../evidence/queue-review-20260924/041-server-tail.log) |
| 042 N22 | S1原样重复036 | TPOT .0915/.2323，175超标；fast15/23、overall18/27、turn0/3、chain17/22；仅TPOT失败 | 同配置fast7→15、p95 .2532→.2323；一次重跑证明有波动，尚非稳定噪声估计；[L042](../evidence/L042/) |

真实同链LCP复核使用T56既有渲染结果，并核对同一req_id、前驱与prompt长度；只代表本轮前驱prompt的缺口，不是全部缓存潜力或已证明可恢复的时间。[缓存账本](../evidence/queue-review-20260924/cache_lcp_comparison.json)。旧为当时交接快照，最新结论以本表与codex分析为准。

## 正式A/B原样校准：044r/045r（2026-09-24，Claude）

镜像0923a同一13个补丁（含121），开发集完整722条、0错、VALID（pod `level_verdict`）。同配置同档对照官方通过档：

| 运行 | 失败门（开发集） | tpot mean/p95 | TTFT p95 fast / overall / turn / chain | 官方同档（已通过） |
|---|---|---|---|---|
| 044r A@N14 | overall 39/27、chain 32/22 | .0426/.0783 | 3.98 / 9.46 / 13.32 / 64.46 | .0174/.0364；1.52 / 3.87 / 6.39 / 30.23 |
| 045r B@N10 | fast 60/23、overall 43/27、tpot_p95 .118 | .0345/.1182 | 7.10 / 7.74 / 4.40 / 23.81 | .0121/.0206；3.46 / 3.72 / 4.91 / 10.89 |

结论（实测两点，不外推）：两套正式通过档在开发集同档都FAIL；四道TTFT p95开发集约为官方的2.0–2.6倍（turn n=20除外），TPOT p95为2.2倍（A）与5.7倍（B）。A、B之间各门的相对顺序在开发集与官方一致（A fast好、chain差；B相反）。开发集可比较我们自己的配置，但“开发集某档全过”不是正式候选的必要条件。官方通过档里TPOT p95离0.10门有约3–5倍余量，紧的是TTFT：A为chain、B为fast。原始记录在pod `/tmp/ax/runs/044r-cal_offA_n14/`、`045r-cal_offB_n10/`。

## Codex CPU复核：阻塞窗口脚本（2026-09-24）

- 原 `blocking.py` 把执行窗口与模型成本残差标成因果拆分，并用 `min()` 隐藏估计越界；已改为实测阶段、候选窗口重合、成本情景三类数据，schema=2。
- 8个CPU回归用例通过；042/037d各722条完整raw重算，超标条数不变。生成窗口重合中位数86.9%/98.9%；成本估计越界218/123条，直接保留而非截断。[042](../evidence/L042/blocking.json)、[037d](../evidence/L037d/blocking.json)。
- 新增真实cohort/时间/token/LCP校验、显式时钟偏移、逐请求候选ID与request-seconds；没有新增GPU实验或正式提交。[完整说明](codex-分析-阻塞归因与执行路线.md)。

## 过去尝试留下的教训

- 早期 v0.5.20 替身与模拟器无法代表正式底包；现在按只读 `build/base_exact/` 写补丁，以 TP8 真实权重/运行验证性能与正确性。底包和当前 S0 的等价性见 [evidence/T57/equivalence.log](../evidence/T57/equivalence.log)。
- `000/110/111` 等解决了接口与 A100 算子启动问题；功能能启动不等于能过能力和 SLO。补丁的开关和覆盖范围见 [engine/README.md](../engine/README.md)。
- 025 系列 DCP 遇到高位地址错误；历史 116 的修复现已并入 115，只在 TP2 开发机修复并复现，TP8 待验。旧「DCP 获得约 7.8 倍 KV」说法是错误写入路径上的计算。[evidence/T50](../evidence/T50/)
- 170 的 BCG+scatter 旧版 TP8 输出错误；v2 在 TP2 通过，真实 TP8 和与 140 的组合仍待验。[evidence/T52b](../evidence/T52b/)
- 旧 `analyze_run.py` 可把空 raw 判为通过，`numcheck` 有截断/分叉漏判。当前用完整性先行的 `level_verdict.py` 与 harness 评分器，数值验证要能检出已知反例。[R19 §2](../research/codex/R19_progress_and_cache_review.md)、[evidence/T54](../evidence/T54/)
- 12 题能力冒烟证明服务可回答这些题；正式能力门必须看平台完整 AIME26/GPQA Diamond 结果。正式提交只记录官方返回，不用开发集 N 预测它。[task.md](../llm-challenge-arena-v1/task.md)

## Codex CPU复核：profile账本与正式校准（2026-09-24）

- 问题：分析器是否正确计算一次CUDA graph forward？既有TP1 trace中只有一个decode，旧`prof_ledger.py`却报44步、平均0.6ms、outside=-2.9%。原因是把不同stream的同一External id重复计数。
- 改进：按逻辑forward合并标记；拒绝无法按时间归因的跨forward重叠和多GPU混合trace。mHC名称优先于泛GEMM；不按TileLang框架名猜DSA，空档不直接归因CPU。
- 验证：4个CPU用例通过，4份已有trace均还原为1个extend+1个decode；示例decode14.7ms、outside1.1%。[修前/修后及回归](../evidence/cost-audit-20260924/)。没有运行GPU实验、修改pod或提交。
- 同轮只读查回正式A=N14、B=N10及能力通过事实。下一步先补同配置同档校准，正式A保留MTP作部署基线；不再要求无MTP S1的dev N22全过后才能研究正式候选。开发集11门照实报告。[分析§10–12](codex-分析-2026-09-24.md#11-昨天正式ab带来的优先级修正)

## Codex CPU复核：评测工具与旧结论更新（2026-09-24）

- 修复清缓存失败仍测量、多档取证混文件、吞 runner/verdict 退出码及引擎复用缺日志；原 runner/harness 只读，CP 主评分不改，替代区间只输出敏感性。13个CPU回归用例通过，本地脚本语法检查通过。
- 035–042共10个完整档逐项比较修改前后主评分报告，全部一致，仍均FAIL；037d只有chain对区间方法敏感，TPOT仍FAIL。此次回归只验证分数不变，不伪造旧归档缺失的run_dev.log。[评分回归](../evidence/eval-tools-audit-20260924/score-regression.json)
- Fable评估中的冻结缓存归因、TPM推产能、驻留覆盖及正式门难度等旧判断已直接更正文；旧044/045同配置校准作废，使用044r/045r的完整重跑。没有同步pod、入队、打镜像或提交。[审计修复](fable-审计-2026-09-24.md)、[评估正文](fable-评估-2026-09-24.md)

## 047：正式A原样 dev N22（2026-09-24，Codex）

问题：保持正式A的13补丁、参数与环境变量，只把原开发集回放并发设为22，给122/执行层候选建立当前对照；无profiler。启动KV池1,036,288 token，12题冒烟12/12。完整回放722条、0请求错误、VALID；主会话取回raw/run/清缓存receipt与服务日志后，使用原harness及题面统计余量独立重判，与pod一致。

| 门 | p95 | 超标 / 允许 | 判定 |
|---|---:|---:|---|
| fast TTFT | 5.2276s | 33 / 23 | FAIL |
| overall TTFT | 10.8031s | 55 / 27 | FAIL |
| turn TTFT | 39.3831s | 2 / 3 | PASS（统计余量） |
| chain TTFT | 93.0873s | 73 / 22 | FAIL |
| TPOT | mean .061976、p95 .087032 s/token | 8/722条 > .10 | PASS |

其余硬门通过，整档有效FAIL；所查三种统计区间对本轮结论无敏感门。测量raw时间跨度1171.7s，TPM因未覆盖其规定窗口为null，不填外推值。此结果仍只代表原开发集，不预测正式N22。首token是本轮过门的重点；首执行前等待不能直接归因调度，需同请求、缓存与batch证据。048只加新版122，重点考察能否改善TTFT而守住TPOT；171/172另做真实TP8数值筛选。

[完整归档与复算](../evidence/L047-official_a_n22/N22/)、[11门判定](../evidence/L047-official_a_n22/N22/level_verdict.json)、[逐请求坏例](../evidence/L047-official_a_n22/N22/badcases.csv)、[下载SHA与文件清单](../evidence/L047-official_a_n22/N22/fetch_status.json)。服务端与本地复算一致；失败原因归因仍待与048配对。

## 048：正式A + 122，dev N22（2026-09-24，Codex）

对照047，仅增加冻结122补丁及`SGLANG_AX_PACE_TPOT=.085`；13→14补丁，原patch哈希、启动参数及KV容量1,036,288均一致。完整722条、0请求错误、VALID；主会话取回完整数据并用原harness和题面余量独立重判，与pod一致。12题冒烟12/12不是正式能力门。

| 指标 | 047基准 → 048候选 | 048判定 |
|---|---|---|
| fast TTFT p95 / 超标数 | 5.2276→2.0784s；33→10/23 | PASS |
| overall TTFT p95 / 超标数 | 10.8031→7.7787s；55→28/27 | FAIL |
| turn TTFT p95 / 超标数 | 39.3831→30.5393s；2→2/3 | PASS（统计余量） |
| chain TTFT p95 / 超标数 | 93.0873→76.6303s；73→62/22 | FAIL |
| TPOT mean / p95 | .061976/.087032→.061659/.084014s | PASS |
| 单请求TPOT>.10 | 8→2 / 722 | 诊断计数 |

整档有效FAIL（仅overall/chain）；三种统计区间判断一致。raw时间跨度1171.7→1126.3s，TPM因窗口不足为null。单次A/B支持保留122作为候选，不能声称稳定噪声外的精确收益或正式并发晋档。122内部含进度预算、短命中预留和完整块选择，本轮只识别整个补丁的效果。

同请求fast改善25条、新增2条；overall改善32、新增5；chain改善16、新增5。描述性分类中fast改善有21条以首执行前时间变化为主；这不是调度因果证明。测量窗排除预热和不完整边界秒后，prefill批2301→1585、partial单跑1815→1107，累计新token9.92→10.04M；块变大、批数变少，与减少固定前向开销相容，尚未拆开其他机制。已打印的pace统计guard均0，不作未打印尾段保证。

原比较脚本的预热污染、完整性缺口和舍入跨门已修，10项CPU回归与独立复核通过。T56 LCP只核元数据与物理上界，来源未验证，不用于缓存因果归因。下一轮顺序仍是049/050单独算子筛选，再决定完整N22候选；124不插队。

[完整归档](../evidence/L048-official_a_122_n22/N22/)、[同请求对照](../evidence/L048-official_a_122_n22/N22/compare_vs_047.txt)、[逐请求CSV](../evidence/L048-official_a_122_n22/N22/compare_vs_047.csv)、[单变量配置核验](../evidence/L048-official_a_122_n22/N22/comparison-verification.json)、[工具独立复核](reports/review-compare-sol.md)。

## 049 / 050：正式 A 原版重复不稳定，算子候选未执行（2026-09-24，Codex）

两项job均在ref1/ref2阶段触发`REFERENCE_UNSTABLE`退出：049为13/16未过，050为15/16未过。171/172候选均未加载、050融合路径trace未录制；没有TP8算子收益或正确性结论。

主会话取回四份原始输出复算：049有11个用例token序列分叉，其中cold_37、cold_4096、text_short、text_mid首token已分叉；050有13个序列分叉，其中cold_37首token分叉。所有配对prompt_tokens和cached_tokens相同，cold用例缓存均0。temperature=0在底包转top_k=1并走greedy argmax，因此不能归因采样随机种子；首token发生在target prefill，仍须隔离计算、状态与执行路径。分叉后的logprob属于不同历史，最大差值不能作为局部算子误差；复核文件另列共同前缀上的差值。

下一步051固定输入、每次flush、单token三次重复并录top10，再做独立48token重复，保持正式A与MTP配置不变。只作诊断，不放宽现有门。049/050入队后未持续跟踪导致延迟发现，由主会话承担；持久只读watcher已在05:58 UTC启动；真实通知主会话已收到，9项CPU回归及首次远程快照通过。

[049原始与独立复核](../evidence/L049-official_a_171_num/reference-review.json)、[050原始与独立复核](../evidence/L050-official_a_172_num/reference-review.json)。

## 051：原版每次flush后的单token重复诊断（2026-09-24，Codex）

复用049/050的正式A原版引擎，未加载171/172；12个单token请求（cold37/256/1024/text_short各三次）及4个48token请求（cold37/text_short各两次），16/16完整且缓存均0。首token top10及完整请求响应已保存。

- cold37单token首token为[7,5691,5691]；第一次top1两个token的logprob完全并列，后两次有差异。cold37长续写两次首token[62,25]。
- cold256单token三次均22995，但同history、同token logprob最大差.2631，前两名间隔2.00→1.40625→2.09375；并非只有argmax并列。
- cold1024单token三次均198，logprob差.0554；text_short单token三次576，top1/top2并列，logprob差.0144。
- text_short两次48token首token均715，第6位置分叉，分叉前同token logprob最大差.1360。尚不能用少量顺序样本断言输出长度改变首token。

结论：目标prefill边界已存在可测重复漂移，不能仅归因后续MTP verify或logprob拼接；仍未隔离哪个模块造成。下一项052按同输入逐层记录输入输出，定位首个差异，诊断同步可能改变执行时序，不能据此测速度。原数值门未放宽、候选未推进为通过。

[汇总](../evidence/L051-official_a_baseline_diag/summary.json)、[完整响应](../evidence/L051-official_a_baseline_diag/responses.jsonl)。051 new→done通知已由常驻watcher主动送达，主会话随即取回并分析。

## 052：逐层探针未命中，诊断INVALID（2026-09-24，Codex）

12个单token请求完成，但8rank均未生成tensor trace，仅有ARMED。主会话沿真实调用链查明：`general_mm_embed_routine`对纯文本也先算embedding，再以`input_ids=None`调用`Glm5NextModel`；helper的input_ids判空导致全跳过。此前只验证了插桩锚点/张量边界，缺少外层真实调用契约验证；主会话验收遗漏负责，052无逐层定位结论。

已亲自修helper：内部参数为None时读取`forward_batch.input_ids`并核对embedding行数；052r在真实pod torch上先回归该入口，跑后强制核8rank、每个冷37/256三次、层覆盖及stage覆盖。记录为空必失败，不再因请求完成就标trace完成。后续171/172及关联修复由主会话亲自实现，subagent停止代码改动。

[无效诊断收据](../evidence/L052-official_a_numtrace/trace-status.json)、[请求摘要](../evidence/L052-official_a_numtrace/summary.json)。


## 053：正式 A 重复差异首先出现在 MoE（主会话独立复核）

052r 已完成一条 cold37 后，第二次 flush 与启动预热重叠而 HTTP400；052s 发现已有记录后保护退出，均不能算算子故障。053 把已核明的8rank首条记录归档，复用同一引擎，flush显式等空闲后完成12个单token请求。cold37/256各3次、8rank、45层全部阶段完整；每次相同输入与参数采样收据一致。

32项跨重复比较中，cold37首/第二次在layer3/mlp_output首次不同、首/第三次在layer4/mlp_output首次不同；cold256两组均在layer3/mlp_output首次不同。所有rank一致，之前阶段逐位一致。该阶段包含MoE本地计算和TP归约，尚不能指认具体kernel、111适配错误或正常浮点误差。完整原始8rank记录以压缩包分块取回并核sha256，本地独立复算与pod一致；直接pread大文件会在传输层截断，未采信截断记录。

证据：[独立复算](../evidence/L053-official_a_numtrace_resume/independent-comparison.json)、[传输校验](../evidence/L053-official_a_numtrace_resume/trace-transfer.json)。054由主会话亲自加首MoE内部收据：完整本地专家权重、路由、有效排序区、GEMM1/激活/GEMM2/本地归约；同步诊断不测性能。首次完整权重快照会临时多占一个权重张量的显存，不进入正式部署。


## 054：首个 MoE 内部收据与收益边界

12请求诊断完成，8rank的cold37/256各三次收据完整。首MoE全量w1/w2与两组scale、输入、router_logits、topk_ids/weights均在重复间相同；32组比较中30组先在有效sorted_ids区不同，其余2组本地阶段全部相同、到TP归约后的mlp_output才变化（其他rank本地已变化）。cold256的16组均随后出现GEMM1、激活、GEMM2及local_output差异。证据：[pod汇总](../evidence/L054-official_a_moe_numtrace/comparison.json)。尚缺固定排序反事实、误差量化与原始dump独立复核，不能由此认定权重损坏、路由错选或171/172错误。

用户要求先确认对并发的回报。排序稳定化列为正确性诊断，不能称加速或承诺N22/N26。171只证实局部投影省时；172完整单卡MoE省3–5%，没有整模型收益。后续按整段prefill/decode成本及瓶颈TTFT筛选主要投入，避免为逐位一致增加未经评估的生产开销。

## 056：正式 A + 171，dev N22（2026-09-24，Codex 独立复核）

按用户最新要求取消重复生成一致性门，直接原harness preflight→warmup→严格flush→完整测量。055组合短测速主动停止，不作候选失败；056只加171，不含122或172。13个基准补丁哈希保持一致，启动配置除自动分配random_seed外一致；047做过额外本地能力冒烟，056跳过，两者均在原harness预热后清缓存。34个KDA层日志均为TP8/BF16/fused=True。

完整722条、唯一请求722、请求错误0，prompt计数/输出长度/服务端TTFT全部722/722。flush收据为HTTP200、JSON success=true，时间与服务日志吻合。压缩证据254,344字节且SHA256核对通过，本地主会话调用原harness及题面补充门，与pod一致：VALID FAIL。队列failed表示TTFT性能门失败，没有引擎崩溃或重复一致性阻断。

| 指标 | 047基准 → 056候选 | 056判定 |
|---|---|---|
| fast TTFT p95 / 超标 | 5.2276→10.4732s；33→35/23 | FAIL |
| overall TTFT p95 / 超标 | 10.8031→12.8434s；55→57/27 | FAIL |
| turn TTFT p95 / 超标 | 39.3831→19.5837s；2→1/3 | PASS（统计余量） |
| chain TTFT p95 / 超标 | 93.0873→81.1985s；73→64/22 | FAIL |
| TPOT mean / p95 | .061976/.087032→.062802/.086897s | PASS |
| 单请求TPOT>.10 | 8→3/722 | 诊断计数 |

raw跨度1171.7→1157.2s，单次少约1.2%，没有同配置重跑噪声估计，不当作稳定提速。正式TPM窗口仍不足，保持null。fast同请求改善11、新增13；overall改善18、新增20；chain改善16、新增7。fast执行段p50 .57→.55s、p95 .72→.71s，首执行前等待p95 4.32→9.78s；执行段仍受batch/chunk影响，不是kernel实测，等待也不能直接归因为调度。有效测量窗prefill批2301→2295，累计新token均约9.92M；总命中24.51→24.50M，75条请求缓存变化≥1024token。

显存实际变化：KV 1,036,288→1,024,960（−1.09%），Mamba状态槽321→318。该差异在CUDA graph捕获前的分池阶段已存在；目标权重加载阶段记录用量39.15→39.18GB、draft阶段.85→1.02GB。这些是进程阶段显存差，不等于持久参数字节差；现日志不能区分存储分配粒度、加载暂存/工作区或其他分配，更不能证明池缩小造成了本次TTFT变化。单层参数相等的开发机证据不能替代服务显存账。

结论：171局部省时尚未兑现为明确整档净收益，暂不加入部署组合；057按既定顺序独立测172。开发集结果不预测正式N22/N26。证据：[完整复算](../evidence/L056-official_a_171_n22/N22/level_verdict.json)、[同请求对照](../evidence/L056-official_a_171_n22/N22/compare_vs_047.txt)、[逐请求CSV](../evidence/L056-official_a_171_n22/N22/compare_vs_047.csv)、[配置/接口/显存复核](../evidence/L056-official_a_171_n22/N22/comparison-verification.json)。

## 057：正式 A + 172，dev N22（2026-09-24，Codex 独立复核）

问题：单卡MoE激活融合的收益是否兑现到完整TP8服务？正式A只增加172及融合开关，不加171或122；13个基准补丁哈希一致，有效启动参数除自动random_seed外一致。047有额外本地能力冒烟，057跳过；两者均用原harness preflight→warmup→flush→完整测量，不设重复输出一致性门。

常驻watcher通知failed后，主会话即时读取status/job.log并取回完整证据：722条唯一请求、0请求错误，prompt冻结计数/输出预算/服务端TTFT全部722/722。flush HTTP200、JSON success=true，服务日志时间吻合。252,267字节压缩归档SHA256=`ded0c692f658938912644b3f8dc918f3a3ec16bac01cca7318603d6b8412110c`；本地主会话调用原harness及补充门独立判分，与pod完全一致。VALID FAIL是三道TTFT门失败，无引擎崩溃或一致性门阻断。

| 指标 | 047基准 → 057候选 | 057判定 |
|---|---|---|
| fast TTFT p95 / 超标 | 5.2276→7.8674s；33→33/23 | FAIL |
| overall TTFT p95 / 超标 | 10.8031→12.1475s；55→53/27 | FAIL |
| turn TTFT p95 / 超标 | 39.3831→36.8287s；2→2/3 | PASS（统计余量） |
| chain TTFT p95 / 超标 | 93.0873→81.2091s；73→62/22 | FAIL |
| TPOT mean / p95 | .061976/.087032→.060998/.085004s | PASS |
| 单请求TPOT>.10 | 8→0/722 | 诊断计数 |

其余硬门通过，三种统计区间对整档结论一致。KV 1,036,288、Mamba状态槽321保持基准容量，未出现171那轮的池缩小。缓存总量24,514,624→24,512,960，71条请求缓存变化≥1024token；不能用总量相近排除逐请求缓存影响。测量窗内prefill批2301→2286、partial单跑1815→1786，累计新token均约9.92M；排除窗口外日志及不完整边界秒。

同请求fast改善10/新增10，overall改善17/新增15，chain改善15/新增4。fast执行段p50 .57→.55s，等待p95 4.32→6.32s；chain超时62条中55条首执行前已经等超过30秒，超时子集等待中位53.04s、执行段中位3.95s。以上按原始精度重新计算；执行段包含batch/chunk效应、等待不是纯调度计时，尚未隔离因果。T56 LCP只作未核源诊断。

raw跨度1171.7→1147.6s（少约2.1%），TPOT均值少约1.6%；各配置只有一次，没有同配置N22重跑噪声估计，fast/overall p95还变差，因此不能认定稳定整档净收益。正式TPM窗口不足，保持null。补丁哈希、融合flag、基线路径已核，本轮没有kernel trace或新的正式能力评测，不将其写成kernel级测速或质量通过。

决策：172保留候选、暂不合入或自动叠加171；下一步须对准整段prefill成本及长请求等待积压。057结束后pread确认running/pending均为空，服务未停止或释放。证据：[完整判分](../evidence/L057-official_a_172_n22/N22/level_verdict.json)、[同请求对照](../evidence/L057-official_a_172_n22/N22/compare_vs_047.txt)、[逐请求CSV](../evidence/L057-official_a_172_n22/N22/compare_vs_047.csv)、[配置/接口/容量复核](../evidence/L057-official_a_172_n22/N22/comparison-verification.json)。

## 058：正式 A 原样，longchain-lite N14（2026-09-24，Codex 独立复核）

64条完整链、1123请求，13补丁逐项SHA256核对（含121），MTP保留，参数/env原样。原harness预热31链/905请求后，09:37:28 UTC flush成功；测量09:37:36–10:24:41，共47.08分钟。1123条唯一请求，0错误，prompt冻结计数/输出预算/服务端TTFT均1123/1123；取回归档SHA256=`2886a70ae7b9132431a6b02725419765ff0efea645f7e248092542b3102bf840`，本地原harness及补充门复算与pod一致。

| 指标 | 044r 原dev N14 | 058 lite N14 | 正式A线上N14 |
|---|---|---|---|
| fast TTFT p95 / 超时 / 允许 | 3.9801s；22/23 | .9439s；8/59 | 1.5191s，PASS |
| overall | 9.4598s；39/27 | 1.2348s；13/62 | 3.8729s，PASS |
| turn | 13.3230s；0/3 | 9.3101s；0/4 | 6.3931s，PASS |
| chain | 64.4553s；32/22 | 54.5926s；12/8 | 30.2343s，PASS |
| TPOT mean / p95 | .042577/.078341 | .015852/.044248 | .017435/.036381 |

058是VALID FAIL，仅chain门失败，非引擎崩溃或一致性门阻断。12条chain坏例全部是选中链的首请求、全部来源original，接收时间集中在开场约40秒；11条首执行前已等超过30秒，接收→执行常规中位39.19秒，执行→首token中位4.33秒。首40秒15条chain请求共新算750,344 token，12失败；之后71条全部通过。开场约5秒时尚有1,002,240 KV token槽与301 Mamba槽空闲，可观测淘汰计数在首约95秒没有增加。证据支持初波冷prefill积压，不能把这12条归因缓存容量淘汰；两段计时也不等于纯调度/纯GPU。

负载变化明显：按token加权cache hit为71.50%→93.35%；实际新算prompt 9.808M→5.234M，输出.216M→.799M，chain门占比43.49%→7.66%。lite前20分钟762请求TPOT均值/p95为.020318/.060085；后27分钟大幅退载，全程时间平均存活链7.43、在途请求4.55，测量窗GPU利用率约76.6%（原dev94.2%）。全程均值接近线上不代表持续压力一致：lite的fast/overall更容易、turn/chain更难，线上未返回逐请求数据。两本地运行均缺完整固定稳态TPM窗口，保持null；线上TPM(all)=1,757,558.42、decode=20,546.92。

主会话完成原始复算，GPT-6 Sol medium独立复核12条坏例，结论一致。用户已取消同配置自动重跑；058无第二轮GPU回放。下一项优先补全量311链持续负载证据，当前尚未入队。[完整三方分析](../evidence/L058-official_a_longchain_lite_n14/three-way-review.md)、[结构化指标与复现脚本](../evidence/L058-official_a_longchain_lite_n14/three-way-metrics.json)、[完整评分](../evidence/L058-official_a_longchain_lite_n14/N14/level_verdict.json)。

## 059：旧180，lite N14；060按用户要求中断

059原13补丁保留，另加旧180（3b63d9c8）及每卡32GB主机池、write_through。1123/1123唯一请求、0错误，严格flush有效，主会话用本集requests和原harness独立复算，与pod一致。

| 指标 | 058 A | 059 旧180 |
|---|---|---|
| fast 超标 / 允许 | 8/59 | 26/59 |
| overall | 13/62 | 23/62 |
| turn | 0/4 | 0/4 |
| chain | 12/8 | 11/8 |
| TPOT mean / p95 (s/token) | .015852/.044248 | .014760/.027501 |

两轮同1123请求、cohort/workload和冻结元数据已逐项核对；059是VALID FAIL，仅chain门失败，非服务崩溃。TPOT改善与短命中TTFT变差同时发生，不能宣称全面提升。

代码后续发现旧180开启HiCache时触发120原有排除条件，冷块保护/短命中共享失效；121依赖同一保护参数；若叠122，其总开关也关闭。故059不代表保留正式A调度保护的HiCache对照。120专用decode让轮本就仅在interval=0运行，正式A=2，不能将其一并说成新丢失的机制。

用户要求中断060旧版N30，已用stopjob执行，保留734条不完整记录作诊断，不报整档成绩。新180（8312cbf7）只改总开关到排除L3 storage，独立CPU真实方法复核protect/pace启用；3项tier和12项host开启的pace测试通过。新版本062 lite N30已入队，与060z原样A比较，GPU结果待测。

证据：[059完整判分](../evidence/L059-official_a_180_hicache_lite_n14/N14/level_verdict.json)、[059原始记录目录](../evidence/L059-official_a_180_hicache_lite_n14/N14/)、[总开关复核](../evidence/hicache180-scheduler-review/README.md)。

## 130ez 系列：DCP2 引擎上的单变量探索与 S5 候选（2026-09-27，fable）

数据 v3（311 链 / 5601 请求），N34 除注明外；判分口径同 kanban（同 ID 计数，四门超时条数与 TPOT>0.10 条数；窗口运行为 TIMED_DIAGNOSTIC，不是整档 verdict）。参照 130ez1 = 46757 配置（S1 设置 + dcp 2、running 48）60 分钟。原始记录在 evidence/L130ez*/。

| 运行 | 单变量 | 对照 | chain | turn | overall | fast | TPOT>0.10 | 结论 |
|---|---|---|---|---|---|---|---|---|
| 130ez5 / ez6zzz / ez6zzzz（开场 600 s） | 本地续算 ON / 128p ON / 两者 ON | 各自 OFF 臂（同 ~500 条） | 16→11 / 16→11 / 16→11 | 0 | ±2 | ±3 | ±3 | 修的是同一批家族兄弟，不叠加；A/A 噪声 chain 0 条 |
| 130ez7 | prefill-decode-interval 2→1 | ez1（1576） | 16=16 | 2→1 | 42→34 | 46→32 | 75→93 | chain 不动；前 1800 s 吞吐相同 |
| 130ez8 | BACKLOG_HIGH_S 15→8 | ez1（1588） | 16=16 | 2→3 | 43→52 | 47→52 | 75→78 | 否决 |
| 130ez9 | MAX_SLOW 80→250 | ez1（1564） | 16=16 | 2=2 | 42→38 | 46→55 | 74→78 | 中性 |
| 130ezd（14 分钟停） | K3 分级 warm 预算 | ez1 同刻 | 16=16 | 0→1 | 20→27 | 28→22 | 71→67 | 无效 |
| 130ezd5（诊断） | 负载下开场剖析 | 130ezc 单请求 | — | — | — | — | — | 8k 块目标模型 584 ms 对单请求 592 ms（71–72 µs/token）；开场 decode 0%；小块固定成本 150–200 ms；prof_ledger 拆 extend_draft 后成立 |
| 130eze2 | 去 MTP（spec=-） | ez1（1369） | 16→15 | 2→1 | 38→20 | 42→23 | 74→96 | 池 2.22M→3.13M；稳态 TPOT 均值 42→54 ms |
| 130ezf | S5a：合并引擎 791453ca（128p + 本地续算） | ez1（1588） | 16→11 | 2→1 | 43→57 | 47→55 | 76→73 | chain −5 成立；新增 overall 是暖链内请求被压后 5–24 s |
| 130ezg（N38 开场） | S5a 在 N38 | — | 14/37 | 0 | 15 | 18 | 53/520 | N38 整轮估 18–20 对余量 24–28 |
| 130ezh | S5b：S5a 去 MTP | ezf（1362） | 11=11 | 1→2 | 45→19 | 39→20 | 70→104 | 对 ez1：16→11、38→19、42→20；S5b 四门全面优于 46757 配置，代价 TPOT |
| 130ezk（v3g，真实间隔） | S5b 配置在 v3g 上 | ezh（v3，1163） | 11=11 | 2→1 | 17→14 | 19→11 | 99→81 | 真实间隔下稳态在跑 17–19 条（v3 上 28–33）、KV 22%；稳态 86 个 chain 门样本无一超 30 s，两条 252k 大头入批后 21–24 s（10.7–11.9k tok/s，交错少）；说明本地稳态在线上般的并发下也不会超时，线上稳态超时需要更慢的通道或更重的冷活 |
| 130ezm7（N34 开场 600 s） | 118 引擎 3caadef4，开关关（S5b 配置，OFF 孪生） | — | 11/69 | 0/23 | 7/323 | 6/354 | 79/446 | 排空 446 条；作为 ezm8 的同引擎基线 |
| 130ezm8（N34 开场 600 s） | 118 prefill-only Triton 稀疏注意力 ON（SGLANG_AX_DSA_SPARSE_TRITON_PREFILL=1） | ezm7（446） | 11=11（修1/新1） | 0=0 | 7→4 | 6→4 | 79→72 | 排空 464 条（+18）；前 60 s 内第 32 个冷头 TTFT 76.1→70.4 s（−7.5%）；8 rank 收据 route=full_kv_prefill；冒烟 12/12；KV 1564736 不变；无晚编译；开场 chain 条数不变：漏的仍是 t=0 排队的 ≥42k 冷头，产能上限；新增的 1 条是 19.3k 冷头被原生 LPM in-batch 前缀去优先（held_by 轮换、held_depth 58）压在队尾到 seq64 才放行，31.1 s 入批（OFF 23.6 s）；Codex 复核修正，见 ledger 18:05 |
| 130ezm9 / 130ezma（单请求剖析） | 同引擎 OFF / ON，49k 提示，带 MTP | ezc（592 ms） | — | — | — | — | — | 实测 GPU 每 8k 目标块 OFF 588 ms（72 µs/tok，草稿 26.8 ms）→ ON 541 ms（66 µs/tok，草稿 22.6 ms）：−8.1%；未分类内核 50%→33%，dsa_attn 0→13%（tilelang 主核约 116 ms 换成 Triton 70 ms）；16k 块 OFF 1085 ms（66 µs/tok），ON 待读 |
| 130ezmb（旧 v5g，N26 40 分钟） | S1 锚点 = 46676 配置（f546934e，MTP，无 DCP，running 32） | 线上 46676（chain p95 37.0） | 10/135 | 5 | 43 | 46 | 41/1593 | 排空 1593；harness p95 chain 43.8 / turn 21.3 / overall 3.5 / fast 2.6 s；chain 漏全在开场（25 个开场链首漏 10），稳态 125 个样本 0 漏；248k 头入批后 25.2 s；host 层满、每 110 s 回载 74 GiB。旧 v5g 的强间隔复现不出线上稳态 chain 漏 |
| 130ezn1（v5g-tail，N26 40 分钟） | S1 锚点 = 46676 配置（f546934e，MTP，无 DCP） | ezmb（旧 v5g，同 1591 条） | 10=10 | 5=5 | 43→57 | 46→50 | 41→51 | 排空 1854；harness p95 chain 41.7 / turn 24.8 / overall 3.9 / fast 2.8 s；chain 漏全在开场，稳态 125 个样本 0 漏（10–30 s 档 11→16）；KV 峰值 97%、在跑 14（旧 v5g 43%、9）。v5g-tail 与旧 v5g 在 S1 上差别只在 fast/TPOT 尾部 |
| 130ezn2a（v5g-tail，N26 40 分钟） | S6 + 128g（Codex 873031fd：原生 LPM 零收益 hold 释放，实际网格 256） | ezn1（S1，同 1551 条） | 10→8（修3/新1） | 5=5 | 57→78 | 46→81 | 52→74 | 排空 1551；harness p95 chain 38.3 / turn 20.8 / overall 5.4 / fast 4.6 s；开场 25 头漏 10→7（修好 34.4k、1.7k[S1 里等 48.9 s]、79.0k），稳态新坏 1 条 80k reset（命中 78.6k，显存墙下等 37.8 s）；128g 保留 32/释放 31；KV 95–100%、在跑 16–18、回载 157 GiB/111 s。S6 与 128g 的份额待 ezn2 |
| 130ezn2（v5g-tail，N26 40 分钟） | 纯 S6：S1 设置去 MTP + 128p + 冻结分类 + 131（450e8580） | ezn1（S1，同 1547 条） | 10→8（修2/新0） | 5=5 | 55→75 | 47→75 | 52→73 | 排空 1547；harness p95 chain 41.3 / turn 24.0 / overall 5.7 / fast 5.0 s；chain −2 全在开场；249k 头入批后 25.2→19.7 s。对 ezn2a（S6+128g）同 1542 条：chain 8=8（修1/新1）、overall 74→81、fast 75→78、TPOT 73→74：**128g 无可测 chain 净收益** |
| 130ezn3（v5g-tail，N26 40 分钟） | chain-max 16k：S6 + 132 链首优先 + 稳态冷块 16384 + 短命中预留 2048 + 124 系数 1.05（c1fa4877） | ezn2（S6，同 1547 条） | 8→6（修2/新0） | 5=5 | 75→100 | 75→70 | 73→65 | 排空 1572；**harness p95 chain 24.8 s**、turn 19.7、overall 5.0、fast 4.8 s；对 S1 chain 10→6；稳态 10–30 s 档 11→6、249k 头入批后 19.7→17.6 s；fast 漏 100 条里 86% 是等待（中位 4.6 s）——132 与预留 2048 的代价 |
| 130ezn4（v5g-tail，N26 40 分钟） | chain-max 32k（冷块/块 32768，--max-prefill-tokens 32768） | ezn3（16k，同 1566 条） | 6→8（修0/新2） | 5→3 | 101→106 | 70→65 | 65→67 | 排空 1572；harness p95 chain 38.6 s；大头入批后只再省 0.3–0.7 s，开场多漏 2 条：32k 不比 16k 好，不采用 |
| 130ezn5（v5g-tail，N26 40 分钟） | chain-max 16k + 118（合并引擎 cfd25d0f，SGLANG_AX_DSA_SPARSE_TRITON_PREFILL=1） | ezn3（同 1572 条） | 6=6 | 5→6 | 101→79 | 70→49 | 67→58 | 排空 1613；harness p95 chain 23.3 / turn 18.6 / overall 4.1 / fast 3.7 s；大头入批后 −1.4 s；118 是纯成本下降：fast/overall/TPOT 明显回落，chain 条数不动。候选前需能力复核 |
| 130ezn8（v5g-tail，N26 40 分钟） | chain-max 16k + --max-mamba-cache-size 400（KV 池 1.26M→1.81M） | ezn3（同 1569 条） | **6→8（修0/新2，开场）** | 5=5 | 101→40 | 70→21 | 66→75 | 排空 1581；harness p95 chain 40.2 / fast 1.9 / overall 2.7 s；新坏的 2 条是 222k 巨头提前入批（24.5 s 对 29.2）压后的 35k/56k 可救头；稳态 0 漏。chain 第一：不进候选 |
