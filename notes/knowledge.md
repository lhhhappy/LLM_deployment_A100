# 当前已核实的事实与边界

赛规只以 [task.md](../llm-challenge-arena-v1/task.md) 为准；本页保留影响当前决策的事实。历史结果与失效结论在 [experiments.md](experiments.md) 和 Git 历史。

- 正式排名按主办方确认：最高通过并发 `n_at_slo` → 更低的 `tpot_mean` → 更高的TPM → 更早提交。每档须过11道门；四道TTFT按题面统计余量，`tpot_p95 ≤ 0.10 s/token`无余量。正式负载是341链、5150请求。正式失败档不返回逐请求或门明细。
- 正式46251（源码759a6eb，host64，122关闭，cold cap4096）通过N22，TPOT均值23.011ms、p95 49.513ms。正式46364只改cold cap为6144，已提交，尚无可核成绩。[正式记录](submissions.md)
- 冻结合成长链本地集是311链、5601请求，正文有合成部分；与正式负载不同。069完整N30只差chain首轮：31条超30秒、允许29条，其他10门通过；071打开122后是30/29。不能把本地档位换算成正式档位。[完整实验](experiments.md)
- 071的30条chain超时中，19条在开场30秒内到达，24条在前5分钟到达。此前“18条缓存没接上”的解释已撤回：它使用了冻结标签中的预期命中，前驱在本地回放中往往不存在。开场冷计算和服务顺序是主要待解问题，不能把排队直接等同排序错误。[归因复核](../evidence/L071-official_b_host64_full_n30_shortwarm/final-analysis/opening-correction-audit.json)
- 开场N26同ID短测：把cold cap4096→6144，chain超时21→16，同时fast23→32、overall30→33、TPOT超过0.10秒的请求1→33。这是局部取舍，不是全量或正式成绩。[共同ID表](../evidence/opening-balanced-20260925/common-ids-074-080.txt)
- 续算时内存只剩6 token可造成检查点永久错位，z9的下一轮少命中70,400 token已由日志精确解释。lc295命中仅8704，低于此前对齐检查点，不能全归因于同一个6-token缺陷。120修复已在83197755，但081/082固定759a6eb，不含它。[修复审阅](../evidence/L072-chunk_alignment_host64_n30/review-v2.json)
- MoE/EP8尚无本底包完整TP8收益实测。111的EP `expert_map`/`global_num_experts`契约须核对；本地单卡TP8形状的算子基准不能直接承诺并发收益。[R25](../research/claude/R25_moe_sm80_path.md)

本地判分要求5601个请求各恰好一次、runner与flush收据有效、指标完整，再用原harness与题面门；缺数据是INVALID。运行中raw只有已完成请求，不能判整档。缓存损失需以本轮实际合法前驱和完整状态为证；冻结`uncached_expected`不是真实可复用量。具体评估口径见 [evaluation.md](evaluation.md)。
