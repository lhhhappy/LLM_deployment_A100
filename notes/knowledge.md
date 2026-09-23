# 当前已核实的事实与使用边界

赛规以 [task.md](../llm-challenge-arena-v1/task.md) 为准。本页只放会改变下一轮实验选择的事实；每个性能结论须能追到完整原始记录。补丁开关、依赖与 S0/S1 见 [patches/README.md](../patches/README.md)。

## 赛题与测量

- 正式排名依次比较 `n_at_slo` 与 `tpot_mean`；TPM 不排名。能力 AIME26/GPQA Diamond 都须严格高于 90 分。正式爬坡从 N=10 开始，过了 +4、没过 −4；题面规定的 11 门均须通过。四道 TTFT 门按题面统计余量判，`tpot_p95 ≤0.10 s/token` 无余量。[task.md](../llm-challenge-arena-v1/task.md)
- 开发集是链前缀抽样，722 请求、311 条在本轮出现的链；链首占比远高于正式集。它只适合我们自己配置之间的 A/B 与回归，不推出正式 N@SLO。不能使用截链、去间隔等改变负载的参数。见 [R19](../research/codex/R19_progress_and_cache_review.md) 与 task.md 开发集约束。
- 判分先检查本轮 cohort 中每个请求恰好出现一次、runner 成功和指标完整，再调用 harness 的 `s1_score.evaluate`（`scripts/score_formal.py`）并核对题面的 TPOT p95 门。空 raw、缺请求、缺指标或 runner 失败是 **INVALID**，不是 PASS。`run_dev.py` 对 flush 的公开检查只看 HTTP 成功，不能据此证明缓存真清；另核对真实清除行为。[R19 §2](../research/codex/R19_progress_and_cache_review.md)、[level_verdict.py](../scripts/pod/verify/level_verdict.py)
- `meta_info` 的时间戳、prompt/cached/completion token 计数必须如实；thinking、输出预算、历史和 tools 保持原样；`/flush_cache` 必须清掉前缀 KV。性能改善不能以牺牲这些条件取得。见 task.md「质量前提」「约束」。

## 已证实的性能状态

- S0 当前 7 个补丁与 026/035 源码树逐文件一致；不要沿用旧文档「现在的 120 与 S0 不同」的说法。[等价性记录](../evidence/T57/equivalence.log)
- S0 在 N18 的 026 与 N22 的 035 均通过四道 TTFT 门，但分别以 `tpot_p95=0.219`、`0.296` 失败；035 共 722 条完整请求、0 错，11 门过 10 门。[026 复核](../evidence/T53/)、[035 复核](../evidence/T58/)、[035 分数](../evidence/L035/score_formal.json)
- S0+114 的 036 是单变量比较：N22 fast_intra 3.05→2.05s，overall_intra 5.10→3.73s，`tpot_mean` 0.104→0.094，`tpot_p95` 0.296→0.253；TPOT p95 仍未过门。722 条 raw 在本地，原 harness 评分器复算一致。下一步要在重 prefill 期间给 decode 更多连续进展。[本地证据](../evidence/L036/)
- S0+`--prefill-decode-interval 16` 的 037 在 N22 让所有请求的 TPOT ≤0.10 秒，但 overall、turn、chain 三道 TTFT 门失败；固定 16 轮给 decode 的时间过多。fast_intra 点估计 3.33s 仍按统计余量通过（19/23），不能简单按阈值误判。722 条 raw 与重评分见 [本地证据](../evidence/L037/)。
- 026 N18 的 10 条 fast 超时要按 harness 的 `in_ttft_gate` 分桶。先前「43 条 fast 超时」用了错误分桶。真实 token LCP 复算：可恢复的同链缓存缺口上限 850,432 token，仅占实际未命中 prefill 约 8.0%；先前「同链损失 5.3M、修好即可过 N22」把本轮链首算进去了。[R20](../research/codex/R20_true_lcp_attribution.md)、[复算证据](../evidence/T56/)
- 035 N22 中 205 条请求超过 TPOT p95 的 0.10 秒线；高命中请求也能慢，解码停顿与重 prefill 时段重合。事件重合本身不等于测出了精确阻塞份额。[R21](../research/codex/R21_N22_tpot_failures.md)
- 114 的 190k 冷预填充单次 19.29→15.44s；013b 加 DCP 的 15.58s 不能把收益归给 DCP。DCP 曾因 norope latent 写入未分片在高位地址出错；历史 116 的修复现已并入 115，在开发机 TP2 通过，真实 TP8 仍待验证。[T50 证据](../evidence/T50/)、[账本审查](../evidence/T55/B1-notes.md)
- 170 v2 的 BCG+scatter 在 TP2 数值检查通过；旧版 TP8 曾错，v2 尚未在 TP8 复验。BCG 对小块可能降低固定开销，同时每 token 斜率变化，需完整 A/B。[T52b 证据](../evidence/T52b/)、[R19](../research/codex/R19_progress_and_cache_review.md)

## 不再作为结论使用

- 不把 `uncached_expected` 当真实可复用量；需要本轮相邻 prompt 的真实 token LCP，并区分链首与后续请求。
- 不把在飞 prompt token 求和当物理 KV 驻留，不把 `1−p10/mean` 当已测停顿占比，也不凭榜单或镜像名推断对手方案。[R19 §4](../research/codex/R19_progress_and_cache_review.md)
- 不把 `12/12` 能力冒烟当 AIME/GPQA 门通过；不把单次冷探针或半程回放当完整档成绩；不从开发集预测正式档位。
- 早期 v0.5.20 替身、模拟调度、旧 daemon 队列和纯参数扫的结论已被真实底包及 8 卡实验取代。需要复盘时查 git 历史和原始证据，不用旧结论指导新实验。

最新可核对的本地公开榜单文件是 `data/all_att_2026-09-23.json`（内容截至 09-22 23:56 UTC，最高 N22）；「CalvinCao N26、tpot_mean 0.0551」来自后来转述，尚无本地榜单快照支持，刷新后再作为目标事实使用。
