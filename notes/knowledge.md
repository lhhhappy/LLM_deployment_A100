# 当前已核实的事实与使用边界

赛规以 [task.md](../llm-challenge-arena-v1/task.md) 为准。本页只放会改变下一轮实验选择的事实；每个性能结论须能追到完整原始记录。补丁开关、依赖与 S0/S1 见 [patches/README.md](../patches/README.md)。

## 赛题与测量

- 正式排名依次比较 `n_at_slo` 与 `tpot_mean`；TPM 不排名。能力 AIME26/GPQA Diamond 都须严格高于 90 分。正式爬坡从 N=10 开始，过了 +4、没过 −4；题面规定的 11 门均须通过。四道 TTFT 门按题面统计余量判，`tpot_p95 ≤0.10 s/token` 无余量。[task.md](../llm-challenge-arena-v1/task.md)
- 开发集是链前缀抽样，722 请求、311 条在本轮出现的链；链首占比远高于正式集。它只适合我们自己配置之间的 A/B 与回归，不推出正式 N@SLO。不能使用截链、去间隔等改变负载的参数。见 [R19](../research/codex/R19_progress_and_cache_review.md) 与 task.md 开发集约束。
- 判分先检查本轮 cohort 中每个请求恰好出现一次、runner 成功和指标完整，再调用 harness 的 `s1_score.evaluate`（`scripts/score_formal.py`）并核对题面的 TPOT p95 门。空 raw、缺请求、缺指标或 runner 失败是 **INVALID**，不是 PASS。`run_dev.py` 对 flush 的公开检查只看 HTTP 成功，不能据此证明缓存真清；另核对真实清除行为。[R19 §2](../research/codex/R19_progress_and_cache_review.md)、[level_verdict.py](../scripts/pod/verify/level_verdict.py)
- `meta_info` 的时间戳、prompt/cached/completion token 计数必须如实；thinking、输出预算、历史和 tools 保持原样；`/flush_cache` 必须清掉前缀 KV。性能改善不能以牺牲这些条件取得。见 task.md「质量前提」「约束」。

## 已证实的性能状态

- **09-24 00:47 UTC最新复核**：037b–042已结束，完整dev N22仍无全过。037d的123准入排序将chain33→23，但TPOT p95 .1466失败；122实际为固定耗时公式，target=.17不等于.10的门限，K≈8预测p95≈.09已被037c实测否定。400槽将同链对齐缺口约864k–916k→450k、全prefill只少5%–6%；关闭140则缺口约1.304M。DCP8在冒烟阶段因[33,1,512]/[40,1,512]不匹配失败，未获性能结论。S1重跑fast7→15，单次小差异不能当稳定收益。[codex-分析](codex-分析-2026-09-24.md)、[完整实验](experiments.md)

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
- 已撤回（Codex 复核）：「开发集 p95 高 3–6 倍」没有同配置同档校准，不成立；「tpm_all 相当」混用了全程平均与稳态窗口，不成立。可说的只是：正式通过档 TPOT 余量大，开发集同类配置在更高档位 TPOT 失败。
- 校准：044 用正式 A 原配置跑开发集 N14、045 用正式 B 原配置跑 N10，逐门对照官方通过档，得到开发集对这两种配置的偏差；在此之前，开发集只用于 A/B 相对比较。
- 正式提交内容对其他选手不可见（`scoringDetails`："部署赛提交内容仅作者与主办方可见"）。

## 产能与profile复核（Codex）

- 122的1.175s/16k是成本公式，不是所有上下文下的直测；由全程未命中token/墙钟及此公式得出的prefill占时、MFU只作条件估算，不按N线性外推。当前S1已启用mHC token scatter及reduce-scatter路径，不能照旧profile重复计入尚未实现的收益。TP4仅专家71.12GiB/rank，保留当前KDA池的4+4 PD方案不满足显存账。[分析§10](codex-分析-2026-09-24.md#10-对prefill效率是根的逐项复核与可改代码)
- 通用`prof_ledger.py`曾把一个decode的多stream标记算成44步，产生负outside时间；现按External id合并，4个CPU用例和4份历史trace回归通过。kernel名字分类只是启发式，no-kernel gap不能直接当host开销。[分析§12](codex-分析-2026-09-24.md#12-新复现的profile账本bug及修复)
