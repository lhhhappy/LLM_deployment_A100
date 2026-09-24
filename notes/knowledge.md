# 当前已核实的事实与使用边界

赛规以 [task.md](../llm-challenge-arena-v1/task.md) 为准。本页只放会改变下一轮实验选择的事实；每个性能结论须能追到完整原始记录。补丁开关、依赖与 S0/S1 见 [patches/README.md](../patches/README.md)。

## 赛题与测量

- 正式排名（主办方确认，2026-09-24）逐级比较：`n_at_slo` 越大越好（一档都过不了排最后）→ TPOT（`tpot_mean`）越小越好 → TPM 越大越好 → 先提交者靠前。能力 AIME26/GPQA Diamond 都须严格高于 90 分。正式爬坡从 N=10 开始，过了 +4、没过 −4；题面规定的 11 门均须通过。四道 TTFT 门按题面统计余量判，`tpot_p95 ≤0.10 s/token` 无余量。[task.md](../llm-challenge-arena-v1/task.md)
- 开发集是链前缀抽样，722 请求、311 条在本轮出现的链；链首占比远高于正式集。它只适合我们自己配置之间的 A/B 与回归，不推出正式 N@SLO。报告原开发集时不能使用截链、去间隔等改变负载的参数。另建独立、冻结的合成长链集用于机制研究，须注明来源与偏差，不冒充原开发集或正式成绩。见 [R19](../research/codex/R19_progress_and_cache_review.md)、[长链方案](codex-方案-长链负载与N22-N26验证.md)。
- 判分先检查本轮 cohort 中每个请求恰好出现一次、runner 成功和指标完整，再调用 harness 的 `s1_score.evaluate`（`scripts/score_formal.py`）并核对题面的 TPOT p95 门。空 raw、缺请求、缺指标或 runner 失败是 **INVALID**，不是 PASS。原 `run_dev.py` 忽略 flush 失败返回值。现新增 checked runner 在失败时阻断测量；level_verdict 核对本次 receipt、调用时间窗及真实服务日志，缺证据标 INVALID。CP 主评分不变，Wilson/Wald 仅诊断方法敏感性；本地修复待负责人同步 pod。[修复与历史影响](fable-审计-2026-09-24.md)。[R19 §2](../research/codex/R19_progress_and_cache_review.md)、[level_verdict.py](../scripts/pod/verify/level_verdict.py)
- `meta_info` 的时间戳、prompt/cached/completion token 计数必须如实；thinking、输出预算、历史和 tools 保持原样；`/flush_cache` 必须清掉前缀 KV。性能改善不能以牺牲这些条件取得。见 task.md「质量前提」「约束」。

## 已证实的性能状态

- **09-24 00:47 UTC最新复核**：037b–042已结束，完整dev N22仍无全过。037d的123准入排序将chain33→23（CP允许22，Wald允许23，方法敏感不改变整档失败），但TPOT p95 .1466失败；122实际为固定耗时公式，target=.17不等于.10的门限，K≈8预测p95≈.09已被037c实测否定。400槽将同链对齐缺口约864k–916k→450k、全prefill少5%–6%，但不能据此判短请求收益次要；042→040的intra真实缺口687,552→450,496（少34.5%），其SLO收益仍须复验；关闭140则缺口约1.304M。DCP8在冒烟阶段因[33,1,512]/[40,1,512]不匹配失败，未获性能结论。S1重跑fast7→15，单次小差异不能当稳定收益。[codex-分析](codex-分析-2026-09-24.md)、[完整实验](experiments.md)

- S0 当前 7 个补丁与 026/035 源码树逐文件一致；不要沿用旧文档「现在的 120 与 S0 不同」的说法。[等价性记录](../evidence/T57/equivalence.log)
- S0 在 N18 的 026 与 N22 的 035 均通过四道 TTFT 门，但分别以 `tpot_p95=0.219`、`0.296` 失败；035 共 722 条完整请求、0 错，11 门过 10 门。[026 复核](../evidence/T53/)、[035 复核](../evidence/T58/)、[035 分数](../evidence/L035/score_formal.json)
- S0+114 的 036 是单变量比较：N22 fast_intra 3.05→2.05s，overall_intra 5.10→3.73s，`tpot_mean` 0.104→0.094，`tpot_p95` 0.296→0.253；TPOT p95 仍未过门。722 条 raw 在本地，原 harness 评分器复算一致。下一步要在重 prefill 期间给 decode 更多连续进展。[本地证据](../evidence/L036/)
- S0+`--prefill-decode-interval 16` 的 037 在 N22 让所有请求的 TPOT ≤0.10 秒，但 overall、turn、chain 三道 TTFT 门失败；固定 16 轮给 decode 的时间过多。fast_intra 点估计 3.33s 仍按统计余量通过（19/23），不能简单按阈值误判。722 条 raw 与重评分见 [本地证据](../evidence/L037/)。
- 026 N18 的 10 条 fast 超时要按 harness 的 `in_ttft_gate` 分桶。先前「43 条 fast 超时」用了错误分桶。真实 token LCP 复算：可恢复的同链缓存缺口上限 850,432 token，仅占实际未命中 prefill 约 8.0%；先前「同链损失 5.3M、修好即可过 N22」把本轮链首算进去了。[R20](../research/codex/R20_true_lcp_attribution.md)、[复算证据](../evidence/T56/)
- 035 N22 中 205 条请求超过 TPOT p95 的 0.10 秒线；高命中请求也能慢，解码停顿与重 prefill 时段重合。事件重合本身不等于测出了精确阻塞份额。[R21](../research/codex/R21_N22_tpot_failures.md)
- 114 的 190k 冷预填充单次 19.29→15.44s；013b 加 DCP 的 15.58s 不能把收益归给 DCP。DCP 曾因 norope latent 写入未分片在高位地址出错；历史116的修复现已并入115、TP2通过。后续041实际TP8在extend KV汇总时出现33/40行不匹配，尚未修复，不能称TP8可用。[T50证据](../evidence/T50/)、[041异常](../evidence/queue-review-20260924/041-server-tail.log)
- 170 v2 的 BCG+scatter 在 TP2 数值检查通过；旧版 TP8 曾错，v2 尚未在 TP8 复验。BCG 对小块可能降低固定开销，同时每 token 斜率变化，需完整 A/B。[T52b 证据](../evidence/T52b/)、[R19](../research/codex/R19_progress_and_cache_review.md)

## 不再作为结论使用

- 不把 `uncached_expected` 当真实可复用量；需要本轮相邻 prompt 的真实 token LCP，并区分链首与后续请求。
- 不把在飞 prompt token 求和当物理 KV 驻留，不把 `1−p10/mean` 当已测停顿占比，也不凭榜单或镜像名推断对手方案。[R19 §4](../research/codex/R19_progress_and_cache_review.md)
- 不把 `12/12` 能力冒烟当 AIME/GPQA 门通过；不把单次冷探针或半程回放当完整档成绩；不从开发集预测正式档位。
- 早期 v0.5.20 替身、模拟调度、旧 daemon 队列和纯参数扫的结论已被真实底包及 8 卡实验取代。需要复盘时查 git 历史和原始证据，不用旧结论指导新实验。

最新可核对的本地公开榜单文件是 `data/all_att_2026-09-23.json`（内容截至 09-22 23:56 UTC，最高 N22）；「CalvinCao N26、tpot_mean 0.0551」来自后来转述，尚无本地榜单快照支持，刷新后再作为目标事实使用。

## 开发集与正式压测的关系（2026-09-24，已核对口径）
- 来源：`s1-dev/` 是主办方公开开发集与 harness（task.md「公开开发集」：同一套 harness 与评分口径），本仓库不做版本管理，只读。我们用原 `run_dev.py` 回放（preflight→warmup→flush→测量），节拍为默认 chain-total-gap-scaled-v1、cap 3600 s（0 条链被压缩），未使用 `--max-chains/--no-gap/--include-all`。`--tok-dir` 用 `/mnt/models`，但 722 条的 prompt_tokens 与数据集 glm_tokens 全部一致；TTFT 全部来自服务端打点；输出长度全部等于 max_output_i（ignore_eos）。判定用 harness 的 s1_score 加 task.md 的统计余量与 tpot_p95 门（harness 自带 summary 只按点估计判，另行报告）。
- 结构差异（主办方设计，不是脚本错误）：开发集链前缀抽样，311 链/722 请求、平均 2.3 请求/链，chain_start 314 条（43%）；loadgen 每个槽取一整条链顺序回放，链短则槽不断开新链，冷链首持续涌入；042 实际 prefill 中链首占 76%。正式用整链集，单档约 4 小时；开发集 N22 一档约 20 分钟，harness 的稳态 TPM 窗口 [10,70) min 不成立，本地 TPM 为空。
- 已撤回（Codex 复核）：旧「开发集 p95 高 3–6 倍」混用了配置和档位；「tpm_all 相当」混用了全程平均与稳态窗口。新的044r/045r提供同配置同档比较，但不能倒推旧论证正确，也不能推出所有N的换算系数。
- 校准更新：044r正式A原样（13补丁含121）在dev N14为overall/chain失败、TPOT=.0426/.0783；045r B@N10为fast/overall/TPOT失败、TPOT=.0345/.1182。两套正式同档已通过；pod结果见[实验记录校准节](experiments.md#正式ab原样校准044r045r2026-09-24claude)，本轮完整raw待本地独立复核。TPOT p95差距分别约2.2倍和5.7倍，不能统一除以二。开发集通过不作为研究N22/N26的前置；原044/045缺121已作废。[设计](codex-方案-长链负载与N22-N26验证.md)
- 正式提交内容对其他选手不可见（`scoringDetails`："部署赛提交内容仅作者与主办方可见"）。

## 产能与profile复核（Codex）

- 122的1.175s/16k是成本公式，不是所有上下文下的直测；由全程未命中token/墙钟及此公式得出的prefill占时、MFU只作条件估算，不按N线性外推。当前S1已启用mHC token scatter及reduce-scatter路径，不能照旧profile重复计入尚未实现的收益。TP4仅专家71.12GiB/rank，保留当前KDA池的4+4 PD方案不满足显存账。[分析§10](codex-分析-2026-09-24.md#10-对prefill效率是根的逐项复核与可改代码)
- 通用`prof_ledger.py`曾把一个decode的多stream标记算成44步，产生负outside时间；现按External id合并，4个CPU用例和4份历史trace回归通过。kernel名字分类只是启发式，no-kernel gap不能直接当host开销。[分析§12](codex-分析-2026-09-24.md#12-新复现的profile账本bug及修复)
- `blocking.py` 已更新为窗口诊断：排队位置不能直接归因调度，prefill 生命周期重合不等于 GPU 独占。042/037d 的旧成本估计分别在 218/123 条请求超过整个执行窗口；新版保留有符号残差、缺失 LCP 标未知，按真实 cohort 验证。累计重合秒数是 request-seconds；86.9%/98.9% 是 TPOT 超标请求的生成窗口重合中位数，不能称真实停顿占比。[工具复核与用法](codex-分析-阻塞归因与执行路线.md)
- 执行层研究不限于 SGLang：题面 vLLM sm80 backport 已有主办方接口验证，是首个替代引擎对照候选；不是已证明分数更高。KDA BF16 投影融合、170 v2、MoE 结构/大块路径按数值和真实成本筛选。[更新后的 R9](../research/claude/R9_upstream_since_base.md)

## 正式最终评测的规模（主办方确认，2026-09-24）
- 回放 341 条会话链、5150 个请求（平均每链约 15.1 个请求；开发集为 311 链 / 722 请求，每链约 2.3 个），按档搜索最大 N，整轮约 8–10 小时。由此估算，链首约占 6.6%，开发集为 43%。
- 排名顺序见上文「赛题与测量」。task.md 与 `challenge.json` 摘要里的旧排名写法（TPM 不排名；或 TPM(decode) → chain_start p95）以主办方这次确认为准。

## Fable 评估复核后的决策边界（Codex，2026-09-24）

- 逻辑 TPM 含缓存输入；数值相近不能证明实际 prefill 工作或产能相同。参考集 intra 请求数占 91% 也不等于计算量占 91%，不能据此排除执行效率瓶颈。
- 开发集已能观测驻留/复用问题，400 槽的单变量结果就是证据；更长历史与竞争分配覆盖不足需另做机制探针。仅拉长无竞争的 gap 不能检验 LRU 驱逐。相同开发负载上的候选比较仍有效，迁移到正式排名须校准。
- 超时伴随真实 LCP 缺口不等于超时由该缺口单独造成；冻结差值尤其不能当缓存损失。040 的 12 条 fast 超时中仅 10 条有 >64-token 真实缺口。启动阶段 19.79GB 空闲也不是已证实可回收的稳态显存。
- 旧“正式 chain 一定最松”“领先者故意牺牲最大5%链首”“A@N18几乎不可能挂TPOT”均无直接证据，不指导实验。Fable 评估正文已替换为[复核结论](fable-评估-2026-09-24.md)，逐请求复算见 [证据](../evidence/eval-tools-audit-20260924/fable-review.json)。

## 公开榜单（用户转贴，2026-09-24 约 07:20）
- 我们（正式 A，45979）：N14，tpot_mean 0.017435（全榜最小），tpm_all 1,757,558（接近全榜最大），tpm_decode 20,547，排第 17。
- 前列：N26 两名 tpot 0.040 / 0.055（第三名 0.043）、tpm 1.46–1.61M；N22 四名 tpot 0.027–0.052；N18 九名 tpot 0.024–0.048。
- 可以确认的比较：按用户转贴，我们 A 在已通过的 N14 上 TPOT 均值低、逻辑 TPM 高，但有效 N 落后；跨 N、跨配置的 TPOT 不能用来证明我们在 N22/N26 仍有相同余量，也不是纯 decode kernel 速度的直接比较。以上榜单全貌本轮未独立拉取。
- 待验证假设：约 1.76M/min 是转贴中观察到的高值，尚未证明是吞吐上限或固定请求速率。逻辑 TPM 包含缓存输入；较低 TPM 与较高 TPOT 同时出现，不能单凭相关性判定没跟上正式回放，更不能排除缓存、prefill 成本或测量窗口内请求构成的差异。
- 决策：优先提高通过全部硬门的 N。TTFT、缓存驻留和 prefill 成本是重点研究方向，但正式更高失败档的主因尚未确认。允许 TPOT 均值变慢来换取更高有效 N，前提是该档 tpot_p95≤0.10 s/token 且其他硬门全部通过；不能要求均值保持 N14 的 0.017435，也不能用均值代替 p95 门。相同有效 N 下再按已确认的同档排名规则优化。
