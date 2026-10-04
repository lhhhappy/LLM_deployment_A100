# 当前已核实的事实与边界

赛规只以 [task.md](../llm-challenge-arena-v1/task.md) 为准；本页保留影响当前决策的事实。历史结果与失效结论在 [experiments.md](experiments.md) 和 Git 历史。

- **最新冻结发布（10-03 核实）：** 09-30 CAP / 47798 已完成，能力门通过、正式 N42 PASS；同日 TPOT / 47800 能力门未通过、没有压测。当前引擎是 `ca5d646c`，配置与最终成绩以 [提交记录](submissions.md) 为准，源码阅读见 [地图](architecture.md)。下文日期标记的是各次测量的条件，不能把旧配置的成本份额直接套到最新配置。
- **排名规则已核对（09-28）：** `task.md` 与现行 `AGENTS.md` 均只按 `n_at_slo`、`tpot_mean` 排名，TPM 只回报、不排名；撤回此前“入口仍记 TPM 第三顺位”的过时描述。每档 11 道门，四道 TTFT 按题面统计余量，`tpot_p95 ≤ 0.10 s/token` 无余量；正式负载 341 链、5150 请求，失败档不返回逐请求或门明细。
- **09-28 原始数据复核：** 47043 正式 N30 PASS，旧的“只有 chain 紧”失效；今日 N30 三臂共同 **1687** 个 ID 上 chain 12→7→6、TPOT p95 98.387→120.941→96.730 ms。全候选 N34 的 40 分钟排空诊断 chain 仍 6、TPOT p95 103.322 ms；新增 fast 坏例主要多等在首次执行前，不能直接归因某个核或 KV。正式 47266 的最终 N42 结果已收录提交记录。[综合复盘](reports/0928-official-local-synthesis.md)、[正式状态](submissions.md)。
- 正式46251（源码759a6eb，host64，122关闭，cold cap4096）通过N22，TPOT均值23.011ms、p95 49.513ms。正式46364只改cold cap为6144；09-26核实执行完成、压测最高通过档仍N22，TPOT均值22.607ms、p95 43.789ms，09-26 08:06 UTC核实终态`outcome=partial`、score 98.7179。同档chain p95 46.05→41.97秒，但**turn p95 6.58→10.66秒（+62%）**，fast/overall也变差。turn仍过15秒目标，不能因此忽略余量损失；平台不返回逐请求正式raw，不能定位单一原因。[正式记录](submissions.md)、[turn审计](../research/codex/R33_turn_start_regression.md)
- 冻结合成长链本地全量是311链、5601请求，正文有合成部分；与正式负载不同。069完整N30只差chain首轮：31条超30秒、允许29条，其他10门通过；071打开122后是30/29。K=9只取每链前9个请求，得到311链/1865请求/约1.05亿prompt token，**只是待校准的候选负载**，请求数和token总量接近正式不代表时序、缓存与四桶已对齐。[完整实验](experiments.md)、[队列](queue.md)
- 071的30条chain超时中，19条在开场30秒内到达，24条在前5分钟到达。此前“18条缓存没接上”的解释已撤回：它使用了冻结标签中的预期命中，前驱在本地回放中往往不存在。开场冷计算和服务顺序是主要待解问题，不能把排队直接等同排序错误。归因复核（本地归档：`../evidence/L071-official_b_host64_full_n30_shortwarm/final-analysis/opening-correction-audit.json`）
- 开场N26同ID短测：把cold cap4096→6144，chain超时21→16，同时fast23→32、overall30→33、TPOT超过0.10秒的请求1→33。这是局部取舍，不是全量或正式成绩。[共同ID表](../evidence/opening-balanced-20260925/common-ids-074-080.txt)
- cold cap6144的本地完整081/N30在5601条上VALID PASS；turn桶同ID 159条的p95 13.64→13.18秒、超15秒7→7，却有4条修复和4条新增。仅加并发的082/N34在fast、overall、chain三门FAIL。N34新增fast坏例主要多等在首次执行前，实际新计算量只多0.8%，设备命中转向host、full-KV驻留压力增大；逐请求资源资格尚未查清。[turn审计](../research/codex/R33_turn_start_regression.md)、[N34归因](../research/codex/R31_n34_waiting_bottleneck.md)
- 117 是在已通过本地N30的081上单独把A100块FP8 MoE专家从Marlin改走Humming，权重仍是FP8、激活仍是BF16，不改调度与缓存。084本地N30派发60分钟、同ID 3502条筛选中，TPOT均值32.90→30.33ms（−7.8%）、fast/overall/turn/chain超时238→230、190→175、7→4、23→22，零请求错误；首派发起60分钟内完成3250→3482条（+7.14%）。117 后续已随 S1（46676）及 47043 组合正式提交，不能把组合升档归给 117 单项；本项不是完整 N30 或 N34 判分。086的N34筛选在1038条后为负载校准主动停止，仅保留局部证据。[设计与证据](../research/codex/R32_117_humming_effect.md)
- 续算时内存只剩6 token可造成检查点永久错位，z9的下一轮少命中70,400 token已由日志精确解释。lc295命中仅8704，低于此前对齐检查点，不能全归因于同一个6-token缺陷。120修复已在83197755，但081/082固定759a6eb，不含它。[修复审阅](../evidence/L072-chunk_alignment_host64_n30/review-v2.json)
- EP8开发机单层测量变慢13–24%，尚无完整TP8服务收益；不能与已在TP8服务短测见到收益的117混为一谈。111的EP `expert_map`/`global_num_experts`契约仍须核对；单层算子增益也不能直接当作整档晋档。[R25](../research/claude/R25_moe_sm80_path.md)、[117](../research/codex/R32_117_humming_effect.md)

本地有效性、同 ID 对照与判分合同见 [evaluation.md](evaluation.md)。本地冻结cohort与正式集不同，不得换算为正式档位。

## chain_start 门的负载结构：链首大小来自主办方原始数据，不是合成（2026-09-27 核实）

来源：`s1-dev/data/dev-combined-v1/requests.jsonl`（主办方公开开发集，311 链 / 722 条真实正文）与 `cache/s1-dev-longchain-v3/requests.jsonl`（本地 v3），按 dispatch_offset_ms 取每条链的第一条请求逐条比对。

- **v3 的 311 条链首与开发集逐条相同**（glm_tokens 差异 0 条；v3 只合成链首之后的请求，`original_requests: 722`）。本地开场的 chain 超时不是我们造数据造成的。
- 开发集链首 glm_tokens：min 8.4k、p25 18.7k、中位 35.7k、p75 60.8k、p90 102k、max 257k；≥64k 的 69 条（22%）、≥100k 的 33 条（11%）、≥200k 的 9 条。
- 104 条链首是主办方标注的会话首轮（phase=session_start、edge_type=chain-head，14–52k；是标注的起点，不代表源业务里实测全冷）；其余 207 条是**会话中段切出的链**（phase 为 intra 115、turn_start 72、context_reset 20），其中 148 条是 system/tools 变更边（`system-tools-changed`，与前一条 LCP=0，前一段即使在缓存里也复用不了）。一个会话可切成几十条链（d178f942 有 50 条、5f5b5fc16f0e43f5b26c6 有 33 条）。
- 主办方 harness（`s1-dev/harness/s1_loadgen.py` 第 250 行）把每条链的第一条请求（idx_in_chain==0）和 context_reset 都归入 chain_start（≤30 s）门，与正式压测同一套（task.md 第 322 行）。所以一条 100k–257k 的冷链首要在 30 s 内出首字，这是赛题本身的要求。
- 正式集（341 链 / 5150 请求）是另一种切法，逐条构成未知；开发集链首的 split 字段为 hidden 178 / dev 125 / validation 8 只是公开元数据里的来源标签，不能据此断言正式集包含这些链首。线上 46677 在 N26 的 chain p95 39.1 s 而 turn/overall/fast 都很宽，与“线上链首同样又大又冷、开场堆积”一致，与“线上链首都很小”不一致。
- 本地 N34 参照（130ez1）开场 16 条 chain 超时的构成（Codex 09-27 复核纠正，按 raw 的首次入批时间 t_exec_start_s 重算）：这 16 条从首次入批到首字只用 1.0–13.6 s（157k 那条 13.6 s，新算速率 8–12.5k tok/s），入批前的等待占 TTFT 的 91%。也就是说 65k–158k 的冷链首**单独算并不超过 30 s**；它们超时是因为排在更小的链首后面，轮到时已过 30 s。只有 ≥250k 的段首（稳态 2 条）入批后本身就要 33 s。8 条 ≥64k 大冷首 + 6 条有 16–33k 缓存的家族兄弟（128p/本地续算各修 5–7 条）+ 2 条中等。
- 开场是一条串行的 prefill 通道：130ez1 服务日志（Prefill batch 行）实测开场 0–100 s 机器新算速率 12.8k tok/s，每 10 s 17–18 个块、几乎全是 8192 的块（每块 0.57 s），GPU 利用率 97–99%（nvidia-smi）。**单请求也是这个速度**（Codex 09-27 纠正，逐 span 核实）：130ezc 的 49k 冷提示服务端 3.74 s 算完 = 13.1k tok/s；trace 里每个 8k 块是一对 span——目标模型 570–650 ms + MTP 草稿模型 17–28 ms，prof_ledger 原来把两者平均成"309 ms/块"，于是得出错误的 26.5k tok/s 和"负载下慢 1.85 倍"。负载下开场剖析 130ezd5（N34 开场第 8–28 s）：目标 span 567–597 ms/8k 块，与单请求相同；窗口内 98.7% 时间在 prefill、decode 0%（开场护栏只放 prefill）。**结论：负载没有让 prefill 变慢，开场速率 ≈ 13–14k tok/s 是当时这组含 MTP 配置的 prefill 产能**（33 GFLOP/token 计约 56 TFLOPS/卡，A100 峰值的 18%）。8k 块内核时间分布（目标模型）：DSA 注意力/indexer（tilelang main_kernel）22%、MoE humming 25%、all-reduce 12%、mHC 三个 tilelang 核 14%、dense GEMM 10%、Marlin 5%、KDA 4%。16k 块 34 µs/token 对 8k 块 37，只快 8%。提高开场产能只能靠这些内核本身。
- **attention-TP 输入分片（`--enable-attn-tp-input-scattered`）在 47043 及 09-28 早期钉池基线里关闭，今天后续组合已开启并随 47266 上传**（2026-09-28 实测：130ezn9 S1、130eznc 钉池候选、130ezng 能力复核三次运行的服务日志 `server_args` 行均为 `enable_attn_tp_input_scattered: False`；截至 47043 的历史上传包没有这个开关，0928c/47266 已启用）。今日重测之前，它在 0924 的本地运行 L036–L042 和 0925 的 085 里开过，Codex 09-24 分析里“S1 已开 scatter、mHC 只算本地 token”的描述是 042 时代的，对当时未分片的 S1/chain-max/47043 不成立。未分片基线的后果：TP8 每卡对 mHC、norm、dense GEMM、MoE 都处理**整块 token**（8k/16k 行），不是 1k/2k 行；只有 DSA indexer 的查询行被 114（`SGLANG_AX_INDEXER_ROW_SHARD=1`）分到各卡（每卡 nq = 行数/8，小于 1024 行不切）。未分片基线的算子计时与正确性以每卡 8k/16k 行为主形状；新候选开分片后须按真实分片与补齐形状测量，不能直接沿用旧份额。085（旧引擎 + MTP，N30 60 分钟，同 3349 条 ID）实测开 scatter：chain 超时 23→21、TPOT 均值 33.2→32.3 ms、fast +19、overall +5——按 chain 第一的判定是小幅正向，随后已在钉池候选上作单变量重测（130eznk）。**130eznk 实测（09-28，钉池候选 + 分片，N30 rot150，同 1239 条 ID 对 eznc）**：纯 prefill 阶段快 11%（服务日志 Prefill batch 行，开场前 60 s 15.6k 对 14.0k tok/s，前 100 s 14.8k 对 13.3k，200 s 后无差别）；chain 超标 7=7，同一批 7 个开场链首首字 42–80 s → 39.5–74 s（−6…−8%，其中 35–62 s 是入批前排队）；fast 修 25/新 25、overall 修 13/新 10、turn 修 2/新 0；此前所写 TPOT p95 121→126 ms 混用了分母；严格共同 1239 条为 158.419→126.146 ms，且仅旧 OPEN 快照（[复核](reports/local-today-audit-0928.md)）。分片只在 extend 且非 verify 的前向启用（`communicator.py::use_input_scattered`），decode 不走。结论：该窗口 chain 没有新坏，开场提速；单项没有救回 chain，后续 118+scatter 组合救回 1 条。不能据单次快照称无风险或预测正式档位。
- 推断（按 12.8k tok/s 的实测开场机器速率与开发集链首分布估算，未实测；不是严格下界）：N38 开场 38 个链首按最短先做，30 s 内约算完 17–18 个，仅开场就超时 20–21 条，接近余量；余量按 chain 门的桶样本数算（开发集 311 链对应 314 个 chain 门请求，v3 是 432 个，因为非链首的 context_reset 也计入），不是按链数。提高开场速率、或家族共享前缀减少工作量，都能改变这个数；“单请求 26k tok/s”已由上面的 target/MTP span 复核撤回，不能继续作为可达到的产能；128p 已把同一开场从 16 降到 11，说明固定速率的模拟不是硬极限。
- 正式集的开场组成未知：主办方公开 cohort 与 v3 cohort 的前 34 条链只有 5 条重合（Codex 核对），正式 cohort 又是另一次冻结抽样；v3 前 34 条里有一条初始 gap 62 s，实际首 30 s 到达 33 个链首。split=hidden 的 178 条只是来源标签，不能证明正式集成员关系。所以本地开场只用于机制比较，不承诺正式档位。

## 数据集 v4：s1-dev-longchain-v4（2026-09-27，fable）

- 父集 v3（同 311 链、同种子 20260924、同事件计划、同链首正文与 cohort 顺序）。唯一规则变化（生成器 `--rewrite-topup`，提交 d9778b61）：每条链计划内的历史改写把"分叉点"提前，使该链合成的新算 token 逼近来源整链的 `sum_uncached_expected`；追加块大小与提示总量不变。
- 结果（manifest）：合成新算 17.83M / 目标 18.94M（94%；v3 为 12.15M，64%），链级中位比 0.95（v3 0.76）；提示总量偏差中位 0.3%、>10% 的 29 条（v3 28 条，来源与原始请求本身的差）。仍不足的 48 条链（比 <0.5）多为计划里没有改写的链（追加式），无法在不动提示总量的前提下补。改写 316/310，顶补 249 次。
- 结构校验 5601/5601、0 错误（未做整体重渲染校验；构建本身逐条渲染过）。哈希：cohort_sha256 见 cohort.json（与 v3 相同的 cohort 结构），requests.jsonl sha256 04b3d49aca95532cae814a9964f05d90146e340c589c85af11b3921b937f3620。
- 用途：本地稳态更接近来源的冷重算负载（中途大改写挤链首）；用户批准的 N26 整集校准（46758 配置）在此集上做；候选比较也在此集上做。仍不能预测正式档位。
- 本地位置：`cache/s1-dev-longchain-v4`（元数据）；GPU 机 `/sjtu/linhang/arena/repo/cache/s1-dev-longchain-v4`（含 bodies）；pod `/tmp/ax/data/s1-dev-longchain-v4`。

## 数据集 v5：s1-dev-longchain-v5（2026-09-27，fable；只改输出预算）

- 父集 v4。唯一变化：合成请求（4879 条中 4835 条改动）的输出预算 `max_output_i` 按公开 722 条真实输出的分布（取 1.6 次幂加重尾巴）重新抽样，再按链缩放使每条链的合成输出总量与 v4 相同（= 来源整链总量减原始请求），单条上限 16384（1200 s 请求超时下 60 ms/token 仍能算完），最小 2，整数和精确保持；原始 722 条不动，正文、cohort、链元数据不动。工具 `scripts/longchain/rebudget.py`（seed 20270101）。
- v4 预算 P95 2,270；v5 全集 P50/P95/P99/max = 254/3,186/9,690/16,384，均值 734 不变。
- 文件：requests.jsonl（sha256 22917b5926f85a1c8ee38899be836c39579d50b090c87507ef77b5256e022855）；chains/cohort/provenance 与 v4 相同（cohort_sha256 ff1dccae1087a798，内部 set 标签仍为 s1-dev-longchain-v4，bodies 复用 v4 的文件，pod 上以符号链接指向 v4 的 bodies）。位置：本容器 `cache/s1-dev-longchain-v5/`（元数据）、GPU 机 `/sjtu/linhang/arena/repo/cache/s1-dev-longchain-v5/`、pod `/tmp/ax/data/s1-dev-longchain-v5/`。

## 数据集 v5g：s1-dev-longchain-v5g（2026-09-27，fable）

- = v5（长输出尾）+ v3g 的间隔规则（`scripts/longchain/regap.py`：每条链的累计等待按主办方 chains.jsonl 的真实链时长回填，单个 gap ≤310 s、链 ≤3600 s，与原 harness 规则一致）。间隔 p50/p90/p95/max = 6.9/49.1/98.0/310 s，累计 34.1 h（v4/v5 为 2.6/12.1/21.4/301 s、8.5 h；用户核对的真实相邻间隔 p95 约 92 s、累计约为 v4 的 3.5 倍）。正文、cohort（ff1dccae1087a798）、输出预算与 v5 相同；requests.jsonl sha256 cfb58cd24adac24a…；set 标签 s1-dev-longchain-v5g，bodies 指向 v4 的文件。
- 含义：真实间隔让同时活跃的上下文减少（v3g 实测在跑 17–19 条、KV 22%），本地不再出现线上没有的 KV 墙。但它把中位等待拉到 7.3 s、>60 s 的等待 440 个（源库 raw 只有 340 个），压力超过源数据，2026-09-27 起只作强压力参照；稳态机制的默认数据集改为下面的 v5g-tail。DCP 在本地"特别好"正是因为 v3 的短间隔造出了 KV 墙，线上没有这堵墙。

## 数据集 v5g-tail：s1-dev-longchain-v5g-tail-review-0927（2026-09-27，Codex c9c14153，fable 独立复核通过）
- 位置：Pod `/dev/shm/arena-runtime/ax/codex/longchain-repair-0927/data/s1-dev-longchain-v5g-tail-review-0927`（同目录还有 v5-review 对照与 v5g-review 强压力参照），开发机 `/sjtu/linhang/arena/codex/longchain-repair-0927/data/`；收据 `publication.json`；requests sha 6170fd82…、cohort ff1dccae1087a798（与 v4/v5/v5g 相同）。
- 规则：从 v5 出发，只把 239 条"合成、非 cohort 链首、原等待 >12.551 s（v5 链中 P90）、旧 v5g 提案 >max(60 s,原值)"的等待改成旧 v5g 的提案（≤310 s）；其余字段逐字段不变，正文链接 v4 分片。链中等待 P50/P90 2.653/12.551 s 不变，P95/P99 33.2/245.5 s，>60 s 250 个（源库 raw 340 个为上界），总和 59,455 s（v5 29,875、旧 v5g 122,279）；harness 每链 3600 s 封顶后 v5/tail 无链被压缩。
- 边界：有限假设的敏感性臂，不是复原源逐请求时间，也不是全量 token 重渲染的 VALID（结构 checker 0 错误）。旧 v5g 把中位拉到 7.3 s、>60 s 440 个已超源，不再作主数据。

## 118 prefill-only Triton 稀疏注意力：TP8 实测（2026-09-27）
- 单请求 49k 剖析（带 MTP）：8k 目标块 GPU 588→541 ms（−8.1%，72→66 µs/token），16k 块 1085→989 ms（−8.8%），草稿块 26.8→22.6 / 50.1→42.0 ms；未分类内核 50%→33%、dsa_attn 0→13%。N34 开场探针同 446 条：chain 11=11、fast 7→4、overall 6→4、TPOT>0.10 79→72；稳态 ≥100k 冷头入批后 −4%…−18%。
- 数值：Codex 算子级 15 项测试误差在界内但非逐位相等，未做全模型 logits 等价；12/12 冒烟通过；进候选引擎前需能力复核（AIME/GPQA）。路由只作用于普通 full-KV EXTEND（有无 DCP 都生效）。

## 原生 LPM in-batch 前缀去优先会把同 pack 冷链首压到队尾（2026-09-27，Codex 日志核实，机制待 CPU 复现）
- 118 探针 ON 臂里 19.3k 冷头从 seq3 到 seq63 一直 held=true（held_by 轮换、held_depth 58，state=ORDINARY），seq64 放行时 due_in 2.9 s 且 50.6k 头已成 chunked_req，31.1 s 才入批（OFF 臂 seq45 放行、23.6 s 入批）。128p 的 8 s max_hold 只约束它自己的依赖。推断：同 pack 冷头共享几十 token 系统前缀（>32 token 阈值）触发去优先，而共享前缀短于 KDA 可复用 checkpoint 网格 256、永远进不了缓存，hold 无收益。Codex 在 codex/lpm-reuse-guard-0927 做可复用粒度门（默认关）。

## 主办方的链是会话段，链首常是冷巨头（2026-09-28 核实；推翻 09-27 23:15 的"错配"说法）
- 用 `chains.jsonl` 的 `first_dispatch_offset_ms` 逐链对照 `requests.jsonl`：311 条链里 258 条的公开首行就是主办方链的第一条请求；这 258 个链首相位 session_start 104 / intra 82 / turn_start 72，`edge_type` chain-head 104 / system-tools-changed 148 / compact-rebuild 6，prompt p50 35k、p90 88k、p95 122k、max 257k，252/258 全未命中。207 条完整链根本没有 session_start。主办方是在系统提示/工具变化处把会话切成链，task.md 的"prompt 常达十万 token 级、结构性零缓存命中"是字面成立的。
- 公开集是链前缀抽样；53 条链的首行未公开，我们的生成器补的头偏重（cohort 里 ≥100k 的链首 31 条，公开可核的 258 个里 17 条）。除此之外本地 chain 桶的人群与主办方一致，09-27 的 chain 结果有效。
- 链中边：append-only 441/722（可命中）、system-tools-changed 148（链内整段重算 40k–257k）、unexplained-break 23、compact-rebuild 6；context_reset prompt 68k–230k、缓存在时未命中约 2k。

## 数据集 v5g-tail-rot150（2026-09-28，fable；只改 cohort 顺序）
- 位置：Pod `/tmp/ax/codex/data/s1-dev-longchain-v5g-tail-rot150`（实路径 /dev/shm/arena-runtime/ax/codex/data/…；requests/chains/bodies 软链到 v5g-tail-review-0927），cohort sha `b78593bdea138f58`，requests sha 6170fd82… 不变；cohort.json 的 `derivation` 记父集与规则。
- 规则：父集 cohort 的 311 条链整体旋转 150 位（chains[150:]+chains[:150]），请求、正文、链内顺序全部不动。用途：父集顺序下 40 分钟窗口只启动约 121 条链，50 个巨型链首里 37 个从未跑到；旋转后这些链首在窗口内先出现，稳态碰撞可以被量到。用户 09-28 的要求："更改一下排序，40min 内优先看这个"。
- 边界：开场那批链首换成了另一批，只能与同数据的锚点比；链深处的巨型 intra 仍要更长窗口才到。

