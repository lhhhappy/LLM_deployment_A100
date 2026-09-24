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

真实同链LCP复核使用T56既有渲染结果，并核对同一req_id、前驱与prompt长度；只代表本轮前驱prompt的缺口，不是全部缓存潜力或已证明可恢复的时间。[缓存账本](../evidence/queue-review-20260924/cache_lcp_comparison.json)。旧[09-24运行记录](runs-0924.md)为当时交接快照，最新结论以本表与codex分析为准。

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
- `000/110/111` 等解决了接口与 A100 算子启动问题；功能能启动不等于能过能力和 SLO。补丁的开关和覆盖范围见 [patches/README.md](../patches/README.md)。
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
