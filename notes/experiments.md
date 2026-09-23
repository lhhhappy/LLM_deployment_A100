# 实验记录

只记能指导下一轮的实验：相对基线的变化、完整数据判定、结论和原始证据。**开发集的通过档位不是正式 N@SLO 预测**。一条档位记录须保留完整 cohort、raw、run/report 和服务日志；`LEVEL` 为 `INVALID` 时不得当成绩比较。历史细节在 git 与 `evidence/`，不重复长篇过程日志。

## 当前开发集对照

| 实验 | 只改什么 / 环境 | 完整档结果 | 结论与证据 |
|---|---|---|---|
| 026 N18 | S0：底包 + `000 101 106 110 111 120 140`；完整 `dev-combined-v1` | 722 条；四道 TTFT 门通过，`tpot_mean=0.083`、`tpot_p95=0.219`，整档 FAIL | 解码 p95 是硬门；真实链内缓存缺口上限约 8%，不能用旧的冻结 token 差值归因。原始 [raw](../evidence/T53/026_N18_raw.jsonl)、[独立审阅](../research/codex/R19_progress_and_cache_review.md)、[真实 LCP](../research/codex/R20_true_lcp_attribution.md) |
| 028 N18 | 相对 S0 同时改变 MTP、114、140 开关、调度 cap/interval、池容量等 | `tpot_mean≈0.053`、`tpot_p95≈0.082` 通过解码门；fast/overall/chain TTFT FAIL | 多变量结果，只说明该组合不能过整档，不能把解码收益单独归给 MTP。[旧 8 卡账本审查](../evidence/T55/B1-notes.md)列出限制；需原始记录复核 |
| 034 N18 | 正式提交 B 的同配置开发集测试；相对 S0 多变量 | 已记录 FAIL，`tpot_p95≈0.210`；本地缺完整 raw | 不能预测正式 B 成绩，也不能用于单因素归因。补齐 raw 后再重判；见 [审查清单](../evidence/T55/B1-notes.md) |
| 035 N22 | 重测 S0，完整 722 请求 | 11 门过 10，只有 `tpot_p95=0.296193` FAIL；`tpot_mean=0.103956` | 以完整原始记录、原 harness + 题面规则复算；205 条请求超过 0.10 秒线。[原始记录与分数](../evidence/L035/)、[独立完整性复核](../evidence/T58/)、[205 条归因](../research/codex/R21_N22_tpot_failures.md) |
| 036 N22 | S0 + 114 indexer row sharding，其他条件沿用 035 | 722 条、0 错、VALID；fast 3.05→2.05s、overall 5.10→3.73s，`tpot_mean` 0.104→0.094，`tpot_p95` 0.296→0.253；仅 TPOT p95 FAIL | 单变量方向有效，仍需解决重 prefill 时的 decode 停顿。192 条请求超过 0.10 秒线；chain_start 17/22 是最窄的 TTFT 余量。本地 [原始 raw、run、服务日志与复算](../evidence/L036/) 已保存；`score_formal.py` 独立重跑与记录一致 |
| 037 N22 | S0 + `--prefill-decode-interval 16`，对照 035 | 722 条、0 错、VALID；`tpot_mean=0.05995`、`tpot_p95=0.07754`，0 条超过 0.10 秒线；overall 28/27、turn 7/3、chain 87/22，整档 FAIL | 16 轮解码让 TPOT 过门，却严重拖慢 prefill 和排队。fast p95=3.33s 虽超目标值，19/23 在统计余量内，仍 PASS；不可按点估计误判。[L037 原始记录与判分](../evidence/L037/)已用 `score_formal.py` 独立复算，结果一致 |

各档的超标数和允许数见 [09-24 运行记录](runs-0924.md)。037b/037c/038/039/040/041 仍在 [任务队列](queue.md)。039 的 MTP 启动同时改变 graph/KV/backend/槽位，是组合实验，不能做单机制归因。

## 过去尝试留下的教训

- 早期 v0.5.20 替身与模拟器无法代表正式底包；现在按只读 `build/base_exact/` 写补丁，以 TP8 真实权重/运行验证性能与正确性。底包和当前 S0 的等价性见 [evidence/T57/equivalence.log](../evidence/T57/equivalence.log)。
- `000/110/111` 等解决了接口与 A100 算子启动问题；功能能启动不等于能过能力和 SLO。补丁的开关和覆盖范围见 [patches/README.md](../patches/README.md)。
- 025 系列 DCP 遇到高位地址错误；历史 116 的修复现已并入 115，只在 TP2 开发机修复并复现，TP8 待验。旧「DCP 获得约 7.8 倍 KV」说法是错误写入路径上的计算。[evidence/T50](../evidence/T50/)
- 170 的 BCG+scatter 旧版 TP8 输出错误；v2 在 TP2 通过，真实 TP8 和与 140 的组合仍待验。[evidence/T52b](../evidence/T52b/)
- 旧 `analyze_run.py` 可把空 raw 判为通过，`numcheck` 有截断/分叉漏判。当前用完整性先行的 `level_verdict.py` 与 harness 评分器，数值验证要能检出已知反例。[R19 §2](../research/codex/R19_progress_and_cache_review.md)、[evidence/T54](../evidence/T54/)
- 12 题能力冒烟证明服务可回答这些题；正式能力门必须看平台完整 AIME26/GPQA Diamond 结果。正式提交只记录官方返回，不用开发集 N 预测它。[task.md](../llm-challenge-arena-v1/task.md)
