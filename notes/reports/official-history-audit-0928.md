# 正式提交历史复核（2026-09-28，独立只读）

依据是已归档的 Playground 官方原始返回和冻结提交配置；未查询或提交新 attempt。赛规以 [`task.md`](../../llm-challenge-arena-v1/task.md) 为准：先排 `n_at_slo`，同档才排 `tpot_mean`；TTFT 四桶目标分别为 fast 3、overall 5、turn 15、chain 30 秒，但 TTFT 判门按超标率的单侧置信下界，不直接按 p95 点估计。TPOT p95 硬门为 100 ms。

| Attempt / 配置变化（相对上一可比候选） | 最高 PASS N | chain / fast / overall / turn p95（秒） | TPOT mean / p95（毫秒） | 能力 AIME / GPQA |
|---|---:|---|---|---|
| 45979 A：0923a，MTP+114+v3，cap4096/interval2 | 14 | 30.23 / 1.52 / 3.87 / 6.39 | 17.44 / 36.38 | 43/44，153/156 |
| 45980 B：同镜像，MTP+114、chunk8192、无 interval；与 A 不止一个变量 | 10 | 10.89 / 3.46 / 3.72 / 4.91 | 12.10 / 20.59 | 42/44，153/156 |
| 46173：A 参数+mem0.87+新版180，0924c，host32，122 off | 14 | 27.21 / 1.19 / 2.11 / 6.73 | 16.54 / 31.84 | 44/44，150/156 |
| 46174：46173+修复122/pace .085，0924d，host32 | 18 | 34.54 / 1.48 / 2.35 / 6.43 | 20.52 / 39.68 | 44/44，153/156 |
| 46251：0925a/759a6eb，host64、mem0.87、新版180、122 off、MTP | 22 | 46.05 / 1.30 / 2.64 / 6.58 | 23.01 / 49.51 | 44/44，152/156 |
| 46364：与 46251 同镜像/命令，仅 cold cap4096→6144 | 22 | 41.97 / 1.86 / 3.15 / 10.66 | 22.61 / 43.79 | 44/44，152/156 |
| 46676 S1：0926b，117+124+激进125 等组合；曾有部署失败 eval，最终另一次 eval 已评分 | 26 | 37.05 / 1.94 / 2.69 / 5.43 | 25.24 / 47.69 | 44/44，148/156 |
| 46677 S2：S1 仅 `BACKLOG_MAX_SLOW=80→250` | 26 | 39.13 / 1.82 / 2.76 / 4.88 | 25.10 / 47.54 | 43/44，153/156 |
| 46757 S3：S1+DCP2、pool418、running/graph48 等组合 | 22 | 29.14 / 1.45 / 2.16 / 4.24 | 22.21 / 40.87 | 42/44，154/156 |
| 46758 S4：S3+本地续算路径/启用开关 | 26 | 41.11 / 1.42 / 2.36 / 9.43 | 25.66 / 47.75 | 44/44，152/156 |
| 47043 chain-max：0927a/20a58da9；相对 S1 去 MTP、16k cold chunk、running/graph48、131/132 与多项 env | **30** | **25.61 / 3.45 / 4.04 / 4.93** | **37.68 / 59.58** | **43/44，151/156** |

所有列出的正式 stress 均为官方 `evaluation_status=PASS` 且能力门通过；45979/45980 的归档是压缩版只读官方状态，余者为完整官方 JSON，46364 的最终外层 `outcome/score` 另由最终文本核实。外层 `outcome=partial` 不等于压测 FAIL。47043 为 `execStatus=completed`、`scoreIsFinal=true`；**47266/0928c 当时仅 queued，不能把 N30 或任何数值归给它**。

## 能成立的历史结论与已过时判断

- 正式容量轨迹按最高通过档看是 A N14、B N10、0924c N14、0924d N18、0925a/冷块6144 N22、S1/S2 N26、S3 N22、S4 N26、chain-max N30。每次的数字只说明该提交在正式负载上达到的最高 PASS 档；能力题分的几题波动不能直接归因配置。
- 46251→46364 是少有的同 N、同镜像/命令单 env 变化：本次 chain p95 46.05→41.97 秒，但 fast 1.30→1.86、overall 2.64→3.15、turn 6.58→10.66，最高档仍 N22。该单次正式结果支持“cold cap 有四桶取舍”，不支持“提高了正式 N”。46676→46677 也是同 N 单 env 变化；250 护栏没有提高 N，chain 37.05→39.13 秒；没有同配置重复，不能将约 2 秒差称为稳定退化。
- 46173→46174 达到 N14→N18；46251 即使 122 off 仍达到 N22，因此旧的“122 必需才能上 N22”结论过时。两提交最高 PASS 的 N 不同，TPOT 16.54→20.52 ms 或 chain 27.21→34.54 秒不能直接归因修复122；46174 的镜像/源码也变了。46757→46758 的 N22→N26 是单次提交观察，不能把不同 N 的 TPOT 22.21→25.66 ms 或 chain 29.14→41.11 秒解释成本地续算本身的速度变化。
- 47043 的 N30 比先前 N26 更高，且本次最高 PASS 档 chain p95 为 25.61 秒，低于 30 秒点目标；这是整个 chain-max 多变量包的正式结果，不能把 N 改善分给 131、无 MTP、16k chunk 中任一单项，也不能用不同 N 的 TPOT 均值 25→37.68 ms 证明单一机制退化。
- “线上只有 chain 门紧，fast 有 1.5–3 倍余量”的旧概括不适用于 47043 的最高 PASS 档：fast p95=3.445 秒，超过 3 秒点目标；overall=4.044/5，chain=25.608/30，turn=4.934/15，TPOT p95=59.58/100 ms。**从已返回点估计看，fast 是当前最紧的可见 TTFT 信号**，但官方 PASS 表明统计余量允许这次超标。旧结果 45980 fast=3.46、46174 chain=34.54、46251 chain=46.05、46676 chain=37.05 也都 PASS，进一步说明 p95 超过点目标不等于硬门失败。
- 官方返回只含最高 PASS 档的聚合 stress，没有更高失败档的逐桶超标数、允许条数、置信下界或逐请求记录。故不能断言 45979 的 N18、46757 的 N26、47043 的下一档 N34 各自败于哪道门；“N34 因 fast/chain 挂掉”只能是待验证假设。`slo_attainment` 也是总请求诊断比例，不能替代逐桶硬门重算。

## 原始来源

- 45979/45980：[45979 压缩官方状态](../../evidence/cost-audit-20260924/official-45979.json)、[45980](../../evidence/cost-audit-20260924/official-45980.json)；冻结 A/B 配置见 [`submission/official-0923-A.json`](../../submission/official-0923-A.json)、[`official-0923-B.json`](../../submission/official-0923-B.json)。
- 46173/46174：[46173](../../evidence/official/attempt-46173-20260924.json)、[46174](../../evidence/official/attempt-46174-20260924.json)；冻结配置见 [`submission_slotA_mem087_hicache_0924c.json`](../../evidence/image-0924d/submission_slotA_mem087_hicache_0924c.json) 与 [`submission_slotB_mem087_hicache_122fix_0924d.json`](../../evidence/image-0924d/submission_slotB_mem087_hicache_122fix_0924d.json)。
- 46251/46364：[46251 最终 JSON](../../evidence/official/attempt-46251-final-20260925.json)、[46364 stress 状态 JSON](../../evidence/official/attempt-46364-status-20260926.json)、[46364 最终外层状态](../../evidence/official/attempt-46364-final-20260926.txt)；[46364 单变量提交收据](../../evidence/submission-0925-cap6144/README.md)。
- 46676/46677：[46676 最终 JSON](../../evidence/official/attempt-46676-20260927.json)、[46677](../../evidence/official/attempt-46677-20260927.json)；[S1 冻结收据](../../evidence/submission-0926-s1/README.md)、[S2](../../evidence/submission-0926-s2/README.md)。
- 46757/46758：[46757](../../evidence/official/attempt-46757-20260927.json)、[46758](../../evidence/official/attempt-46758-20260927.json)；[S3 冻结收据](../../evidence/submission-0926-s3/README.md)、[S4](../../evidence/submission-0926-s4/README.md)。
- 47043：[正式最终 JSON](../../evidence/official/attempt-47043-final-20260928.json)、[冻结提交审计](../../build/worktrees/fix-chainmax-0927/evidence/submission-0927-chainmax-final/config-audit.md)。47266 仅排队的记录见 [`notes/submissions.md`](../../notes/submissions.md) 与 [`evidence/submission-0928-kda/README.md`](../../evidence/submission-0928-kda/README.md)。

## 本地 N34 短窗复核（非正式成绩）

只读复算 [`eznq N34` 的原始 raw 与 timed verdict](../../evidence/L130eznq-tail_rot150_n34_chainmax16k_fix_mamba400_118_scatter_moetune_kda_40m/N34/timed_verdict.json)：40 分钟派发、约 41.9 分钟排空，`status=DRAINED`、`full_cohort_complete=false`。run/dispatched/raw 均 1960 条、ID 唯一且集合一致、0 错误，raw SHA 与 verdict 一致。原始 TPOT 排序法 p95=**103.3216 ms**，超过 100 ms 硬门，102/1960 条 >100 ms；前 5 分钟（按客户端派发时间）248 条中有 51 条慢 TPOT，即全部 102 条的一半，10 分钟后 1465 条 p95=79.1316 ms。此窗的四个 TTFT **估计条数门均通过**（fast 55/1453≤87、overall 21/1618≤96、turn 4/74≤7、chain 6/268≤19）；turn p95 点估计 15.122 s 超过 15 s 与条数门通过并不矛盾。它是本地 DRAINED 诊断，不能称正式 N34 FAIL 或据短窗外推完整 cohort。

与 [`ezno N30`](../../evidence/L130ezno-tail_rot150_n30_chainmax16k_fix_mamba400_118_scatter_moetune_kda_40m/N30) 同 workload/cohort/pacing 的交集为 1850 个 ID（N30 共 1869，N34 共 1960）；由 [`n34-paired.csv`](../../evidence/review-0928-official-local/n34-paired.csv) 独立按标签重数：共同 ID 的 TPOT p95 **94.7438→104.8068 ms**，chain 超 30 秒修复 0、新坏 0、持续 6；fast 超 3 秒修复 24、新坏 49、持续 4。49 条新坏 fast 的同 ID 中位 `recv→exec` **0.1706→3.4722 s**，`exec→first` **0.3730→0.4062 s**，与 [`n34-paired-summary.json`](../../evidence/review-0928-official-local/n34-paired-summary.json) 一致。`recv→exec` 含入批/排队/准备等，并非纯调度计时；`exec→first` 也非纯 GPU 时间。这组分解支持“主要增量出现在执行前区间”的限定判断，不证明单个子机制。

按首派发至排空的 metrics 采样，N30 有 254 点且 `token_usage≥0.93` 占 **3.15%**，N34 有 250 点占 **8.0%**；这只是两窗口的相关描述，不能由此归因 fast/TPOT 退化或推断正式负载。N34 raw/flush/run 与派生配对材料的口径未发现明显不一致；本段绝不与上面的正式 47043 N30 结果混作同一 cohort。
