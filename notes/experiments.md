# 实验记录

只记能指导下一轮的实验：相对基线的变化、完整数据判定、结论和原始证据。**开发集的通过档位不是正式 N@SLO 预测**。一条档位记录须保留完整 cohort、raw、run/report 和服务日志；`LEVEL` 为 `INVALID` 时不得当成绩比较。历史细节在 git 与 `evidence/`，不重复长篇过程日志。

## 当前开发集对照

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

## DCP-LOCAL-20260926：eager prefill 的本地 KV 路径（Codex）

`codex/dcp-prefill-local-kv`；基线为本分支同一引擎关闭新开关，候选保持 W2、MTP、HiCache，仅改变 prefill 执行路线。GPU0/1 两卡开发诊断，不是 TP8 或 N@SLO 成绩。H8 形状、缩小模型的同进程交错完整前向，短尾/2K/8K 本次中位耗时下降约 2.9%/8.4%/7.1%；完整样本、显存与代码图见[报告](reports/dcp-chain-20260926.md)、[机制说明](../engine/docs/115-dcp-local-extend.md)。早期约 19% 的单次短尾结果受启动间漂移影响，不用作稳定收益。

CPU 13 项合同/既有回归通过；MTP、HiCache 真实回载、接受长度 1–4、实际接受跨 128-token 页、两个 rank 的敏感 attention 数值与图重放均有证据。原生 8K 端到端对照有一次可复现的 top-k 选键集合变化，不能记成全部数值 PASS；固定同一组选键后两次隔离对照、两个 rank 均通过。大块保持独立实验开关，真实 TP8 质量与性能待验；另一参与者复核请求随分支交付，确认前不写复核通过。

本轮还独立复算 130ed 已排空的 3123 个派发 ID，并给出全部坏例与同 ID 时间分解，状态为 DRAINED DIAGNOSTIC，未声称完整 cohort 通过；该运行未使用本轮新补丁。归因与原始记录入口同上，不把 W2 对旧 A 的差异视为 DCP 单变量结论。
