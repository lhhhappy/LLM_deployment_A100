# 当前已核实的事实与边界

赛规只以 [task.md](../llm-challenge-arena-v1/task.md) 为准；本页保留影响当前决策的事实。历史结果与失效结论在 [experiments.md](experiments.md) 和 Git 历史。

- **排名规则待核对：** `task.md` 的计分章节写明只按 `n_at_slo`、`tpot_mean` 排名，并明确 TPM 不排名；仓库入口指令（`AGENTS.md`）则记录主办方确认 TPM 为第三顺位。由主代理核实，不在此擅自改写正式规则。共同事实：每档有11道门，四道TTFT按题面统计余量，`tpot_p95 ≤ 0.10 s/token`无余量；正式负载341链、5150请求，正式失败档不返回逐请求或门明细。
- 正式46251（源码759a6eb，host64，122关闭，cold cap4096）通过N22，TPOT均值23.011ms、p95 49.513ms。正式46364只改cold cap为6144；09-26核实执行完成、压测最高通过档仍N22，TPOT均值22.607ms、p95 43.789ms，09-26 08:06 UTC核实终态`outcome=partial`、score 98.7179。同档chain p95 46.05→41.97秒，但**turn p95 6.58→10.66秒（+62%）**，fast/overall也变差。turn仍过15秒目标，不能因此忽略余量损失；平台不返回逐请求正式raw，不能定位单一原因。[正式记录](submissions.md)、[turn审计](../research/codex/R33_turn_start_regression.md)
- 冻结合成长链本地全量是311链、5601请求，正文有合成部分；与正式负载不同。069完整N30只差chain首轮：31条超30秒、允许29条，其他10门通过；071打开122后是30/29。K=9只取每链前9个请求，得到311链/1865请求/约1.05亿prompt token，**只是待校准的候选负载**，请求数和token总量接近正式不代表时序、缓存与四桶已对齐。[完整实验](experiments.md)、[队列](queue.md)
- 071的30条chain超时中，19条在开场30秒内到达，24条在前5分钟到达。此前“18条缓存没接上”的解释已撤回：它使用了冻结标签中的预期命中，前驱在本地回放中往往不存在。开场冷计算和服务顺序是主要待解问题，不能把排队直接等同排序错误。[归因复核](../evidence/L071-official_b_host64_full_n30_shortwarm/final-analysis/opening-correction-audit.json)
- 开场N26同ID短测：把cold cap4096→6144，chain超时21→16，同时fast23→32、overall30→33、TPOT超过0.10秒的请求1→33。这是局部取舍，不是全量或正式成绩。[共同ID表](../evidence/opening-balanced-20260925/common-ids-074-080.txt)
- cold cap6144的本地完整081/N30在5601条上VALID PASS；turn桶同ID 159条的p95 13.64→13.18秒、超15秒7→7，却有4条修复和4条新增。仅加并发的082/N34在fast、overall、chain三门FAIL。N34新增fast坏例主要多等在首次执行前，实际新计算量只多0.8%，设备命中转向host、full-KV驻留压力增大；逐请求资源资格尚未查清。[turn审计](../research/codex/R33_turn_start_regression.md)、[N34归因](../research/codex/R31_n34_waiting_bottleneck.md)
- 117 是在已通过本地N30的081上单独把A100块FP8 MoE专家从Marlin改走Humming，权重仍是FP8、激活仍是BF16，不改调度与缓存。084本地N30派发60分钟、同ID 3502条筛选中，TPOT均值32.90→30.33ms（−7.8%）、fast/overall/turn/chain超时238→230、190→175、7→4、23→22，零请求错误；首派发起60分钟内完成3250→3482条（+7.14%）。117未正式提交；本项不是完整N30或N34判分。086的N34筛选在1038条后为负载校准主动停止，仅保留局部证据。[设计与证据](../research/codex/R32_117_humming_effect.md)
- 续算时内存只剩6 token可造成检查点永久错位，z9的下一轮少命中70,400 token已由日志精确解释。lc295命中仅8704，低于此前对齐检查点，不能全归因于同一个6-token缺陷。120修复已在83197755，但081/082固定759a6eb，不含它。[修复审阅](../evidence/L072-chunk_alignment_host64_n30/review-v2.json)
- EP8开发机单层测量变慢13–24%，尚无完整TP8服务收益；不能与已在TP8服务短测见到收益的117混为一谈。111的EP `expert_map`/`global_num_experts`契约仍须核对；单层算子增益也不能直接当作整档晋档。[R25](../research/claude/R25_moe_sm80_path.md)、[117](../research/codex/R32_117_humming_effect.md)

本地有效性、同 ID 对照与判分合同见 [evaluation.md](evaluation.md)。本地冻结cohort与正式集不同，不得换算为正式档位。

## chain_start 门的负载结构：链首大小来自主办方原始数据，不是合成（2026-09-27 核实）

来源：`s1-dev/data/dev-combined-v1/requests.jsonl`（主办方公开开发集，311 链 / 722 条真实正文）与 `cache/s1-dev-longchain-v3/requests.jsonl`（本地 v3），按 dispatch_offset_ms 取每条链的第一条请求逐条比对。

- **v3 的 311 条链首与开发集逐条相同**（glm_tokens 差异 0 条；v3 只合成链首之后的请求，`original_requests: 722`）。本地开场的 chain 超时不是我们造数据造成的。
- 开发集链首 glm_tokens：min 8.4k、p25 18.7k、中位 35.7k、p75 60.8k、p90 102k、max 257k；≥64k 的 69 条（22%）、≥100k 的 33 条（11%）、≥200k 的 9 条。
- 只有 104 条链首是真正的会话冷首轮（phase=session_start、edge_type=chain-head，14–52k）；其余 207 条是**会话中段切出的链**（phase 为 intra 115、turn_start 72、context_reset 20），其中 148 条是 system/tools 变更边（`system-tools-changed`，与前一条 LCP=0，前一段即使在缓存里也复用不了）。一个会话可切成几十条链（d178f942 有 50 条、5f5b5fc16f0e43f5b26c6 有 33 条）。
- 主办方 harness（`s1-dev/harness/s1_loadgen.py` 第 250 行）把每条链的第一条请求（idx_in_chain==0）和 context_reset 都归入 chain_start（≤30 s）门，与正式压测同一套（task.md 第 322 行）。所以一条 100k–257k 的冷链首要在 30 s 内出首字，这是赛题本身的要求。
- 正式集（341 链 / 5150 请求）是另一种切法，逐条构成未知；开发集链首的 split 字段为 hidden 178 / dev 125 / validation 8 只是公开元数据里的来源标签，不能据此断言正式集包含这些链首。线上 46677 在 N26 的 chain p95 39.1 s 而 turn/overall/fast 都很宽，与“线上链首同样又大又冷、开场堆积”一致，与“线上链首都很小”不一致。
- 本地 N34 参照（130ez1）开场 16 条 chain 超时的构成（Codex 09-27 复核纠正，按 raw 的首次入批时间 t_exec_start_s 重算）：这 16 条从首次入批到首字只用 1.0–13.6 s（157k 那条 13.6 s，新算速率 8–12.5k tok/s），入批前的等待占 TTFT 的 91%。也就是说 65k–158k 的冷链首**单独算并不超过 30 s**；它们超时是因为排在更小的链首后面，轮到时已过 30 s。只有 ≥250k 的段首（稳态 2 条）入批后本身就要 33 s。8 条 ≥64k 大冷首 + 6 条有 16–33k 缓存的家族兄弟（128p/本地续算各修 5–7 条）+ 2 条中等。
- 开场是一条串行的 prefill 通道：130ez1 服务日志（Prefill batch 行）实测开场 0–100 s 机器新算速率 12.8k tok/s，每 10 s 17–18 个块、几乎全是 8192 的块（每块 0.57 s），GPU 利用率 97–99%（nvidia-smi）。**单请求也是这个速度**（Codex 09-27 纠正，逐 span 核实）：130ezc 的 49k 冷提示服务端 3.74 s 算完 = 13.1k tok/s；trace 里每个 8k 块是一对 span——目标模型 570–650 ms + MTP 草稿模型 17–28 ms，prof_ledger 原来把两者平均成"309 ms/块"，于是得出错误的 26.5k tok/s 和"负载下慢 1.85 倍"。负载下开场剖析 130ezd5（N34 开场第 8–28 s）：目标 span 567–597 ms/8k 块，与单请求相同；窗口内 98.7% 时间在 prefill、decode 0%（开场护栏只放 prefill）。**结论：负载没有让 prefill 变慢，开场速率 ≈ 13–14k tok/s 就是这台机器现在的 prefill 产能**（33 GFLOP/token 计约 56 TFLOPS/卡，A100 峰值的 18%）。8k 块内核时间分布（目标模型）：DSA 注意力/indexer（tilelang main_kernel）22%、MoE humming 25%、all-reduce 12%、mHC 三个 tilelang 核 14%、dense GEMM 10%、Marlin 5%、KDA 4%。16k 块 34 µs/token 对 8k 块 37，只快 8%。提高开场产能只能靠这些内核本身。
- 推断（按 12.8k tok/s 的实测开场机器速率与开发集链首分布估算，未实测；不是严格下界）：N38 开场 38 个链首按最短先做，30 s 内约算完 17–18 个，仅开场就超时 20–21 条，接近余量；余量按 chain 门的桶样本数算（开发集 311 链对应 314 个 chain 门请求，v3 是 432 个，因为非链首的 context_reset 也计入），不是按链数。把开场速率提到接近单请求的 26k tok/s、或家族共享前缀减少工作量，都能改变这个数；128p 已把同一开场从 16 降到 11，说明固定速率的模拟不是硬极限。
- 正式集的开场组成未知：主办方公开 cohort 与 v3 cohort 的前 34 条链只有 5 条重合（Codex 核对），正式 cohort 又是另一次冻结抽样；v3 前 34 条里有一条初始 gap 62 s，实际首 30 s 到达 33 个链首。split=hidden 的 178 条只是来源标签，不能证明正式集成员关系。所以本地开场只用于机制比较，不承诺正式档位。
