# 最后两次提交：独立代码与原始数据审查

2026-09-30 06:35 UTC，Astra 独立审查。范围：只读代码、原始 raw、已有资格材料、profile 和公开上游资料；只写本报告。未改 Pod、队列、harness、引擎、提交包或镜像。代码锚点为 `ca5d646c252688177480c7e67cec0a901e4b7069`；另行审查但不混入 HEAD 的 186 草稿只有初始化日志七行及说明 JSON。

**07 的 chain 结果支持保留当前执行底座，但五项小修没有在 05 上兑现新的平均 TPOT 收益。最后两臂建议是 N42 的 backlog relief interval 0→1，以及 N46 的常态 cold cap 16k→12k、backlog cap 仍 16k。优先研究首 token 之后被 prefill 打断的等待；不再把普通 PDI4 或纯算子微修当成追平 13.67% 的主要筹码。**

这与早期“relief1 新增两条 chain 即否决”的取舍不同：用户当前明确允许看相对收益及噪声，不能继续套用旧硬否决规则。它也不是宣称 relief1 已成功；第一臂就是检查其开场收益能否兑现、chain 代价是否系统存在。若代价明确且大，仍回到已测的 07 底座。

## 1. 我重新计算出的当前结果

下表来自各自原始 JSONL，而非抄 brief。p95 使用原 harness 的 `sorted[int(.95*n)]`；mean 是逐请求非加权均值。全部只是 1h 派发后排空的 DRAINED 窗口，不能写成完整 cohort 通过。

| 运行 | 完成/错误 | chain 超 30s / 分母 | chain p95 | TPOT mean / p95 |
|---|---:|---:|---:|---:|
| A1 | 3133 / 0 | 14 / 368 | 23.629s | 61.514 / 103.602ms |
| A2 | 3132 / 0 | 17 / 367 | 28.943s | 62.028 / 105.646ms |
| 05 | 3257 / 0 | 15 / 374 | 26.450s | 59.804 / 103.289ms |
| 07 | 3268 / 0 | 13 / 373 | 25.134s | 59.883 / 103.656ms |

| 07 对照 | 共同请求 | 共同请求 TPOT mean，基线→07 | chain 修好 / 新坏 |
|---|---:|---:|---:|
| A1 | 3133 | 61.514→60.335ms | 2 / 1 |
| A2 | 3132 | 62.028→60.314ms | 4 / 0 |
| 05 | 3241 | 59.907→59.940ms | 2 / 0 |

07 与 05 的均值差只有 +0.033ms，不能归因为五项小修退化，也不能宣传它们新增了几个百分点。07 对两次原配置的均值均改善，而且共同 ID 覆盖完整，配对解释比只比较 3268 与 3133 个不同分母更扎实。07 的 turn 超标 10 条，对 05 的 12 条也没有恶化迹象；这个小样本差异仍不应被写成稳定收益。

原始证据：[A1](../chain-night-0929/verdicts/a1-baseline/raw_s1-dev-longchain-v5g-tail-rot150_N42_1790702333.jsonl)、[A2](../chain-night-0929/verdicts/a2-baseline-repeat/raw_s1-dev-longchain-v5g-tail-rot150_N42_1790719811.jsonl)、[05](pod/moe-dsa-n42/raw_s1-dev-longchain-v5g-tail-rot150_N42_1790736693.jsonl)、[07](pod/smallfix-n42/raw_s1-dev-longchain-v5g-tail-rot150_N42_1790745153.jsonl)。独立计算的 raw SHA256：

```text
A1 798c5f17789e7b39b64371791ea0868883bfdc8068b1a091989486bfbff68255
A2 fa0173b362915fe1aca62d482a57559d25b2bf700ce611cdda2a2a360a32c485
05 ddc9add201a049119be23cbaac15c30e51c5ba925f90bfc7de6bd07333a89fd3
07 24c7362dfa2e2703b6e094d741f60c8ff6b0da9b1edd6ac14c5fa5c022aecde1
```

## 2. 真正值得动手的时间账：首 token 后的 prefill 等待

按每次 raw 中最早的 `client_dispatch_at_s` 对齐，把派发时间位于前 120s 的请求归为开场。这里是完整成绩的解释性分解，**评分仍保留所有请求，不删开场**。

| 运行 | 开场请求数 | 开场请求 TPOT mean | 对全窗口 TPOT mean 的贡献 | TPOT 总和占比 |
|---|---:|---:|---:|---:|
| A1 | 65 | 467.667ms | 9.703ms | 15.77% |
| A2 | 65 | 468.007ms | 9.713ms | 15.66% |
| 05 | 67 | 455.502ms | 9.370ms | 15.67% |
| 07 | 67 | 453.422ms | 9.296ms | 15.52% |

A1 最大五条 TPOT 都是开场短输出链首，输出分别 29、35、46、54、71 token，TPOT 分别约 3348、2643、2124、1727、1407ms。仅这五条就贡献整个窗口均值 3.590ms。05 同样五条分别约 3350、2639、2121、1723、1404ms，几乎原样保留。第一条请求首 token 前仅约 2.41s，首至末 token 却约 94s。这不是优化一个 20ms decode kernel 就能解释或解决的停顿。

源码中已有完整机制：`Scheduler.get_next_batch_to_run` 在新 prefill 存在时优先执行 prefill；`_arm_prefill_decode_interval` 的 ordinary2 被 relief0 覆盖，risk1 再取 min 也救不回 0。125 的观测只计 `produced > 1`；已经收到首 token、随后一直等 decode 的请求不会计入 slow。05 测量开场的日志在 02:51:35→02:53:04 之间 relief on，slow 一直 0/0，与这些短输出的长 post-first 区间相符。之后 80 个累计 slow RID 把 relief 永久关闭，直到 flush。

**因果边界：** raw 只有首末 token 时刻，不能证明这 94 秒里完全没有 decode，也不能独立把每一毫秒归给 125；日志与实际代码给出了强机制假设。第一臂应直接检验该假设，而不是继续拿纯 decode profile 减混合 TPOT。

可兑现预算也要保守：A1 开场全部 TPOT 贡献是 9.703ms，这只是“即使把这些请求 TPOT 降到零”的极松上界，真实只可能省一部分，还会改变后续到达。它说明存在值得研究的毫秒级空间，不证明正式能省 5.914ms。旧 relief1 N34 的观察是 52.953→51.319ms，约 3.08%；这比旧 PDI4 的 0.81% 更有力，却仍不是对当前 N42 或正式成绩的预测。

代码：[scheduler.py](../../build/worktrees/final-execution-0930/engine/sglang/srt/managers/scheduler.py) 的 1665–1682、1962–1995、4220–4277；[ax_deadline.py](../../build/worktrees/final-execution-0930/engine/sglang/srt/managers/ax_deadline.py) 的 `BacklogState`；[原 relief 日志](pod/moe-dsa-n42/relief-transitions.txt)。旧臂数值与范围见[参数复核](final4h-parameter-review.json)。

## 3. A1/A2 为什么不同，如何少受它干扰

我对 3108 个共同请求对齐了派发时刻、输出计数、缓存计数和服务端三个阶段。共同 ID 的 TPOT mean 为 61.659→62.142ms，差 0.483ms；输出 token 数全部相同，236 条的 `cached_tokens` 改变。两次冻结请求体相同，harness 也不会把上一条生成答案重写进下一条 prompt。

| 以 A1 派发时间分段 | 共同请求数 | 两次相对派发时刻差的绝对值 p95 | 缓存计数改变 |
|---|---:|---:|---:|
| 0–120s | 65 | 0.056s | 0 |
| 120–600s | 462 | 10.587s | 26 |
| 600–1800s | 1068 | 67.271s | 77 |
| 1800–3600s | 1513 | 53.185s | 133 |

366 个共同 chain 请求修好 2 条、新坏 5 条，七个转换全部发生在稳态。这组观测不支持“某次开场 GPU 整体慢了很多”作为主要解释，更符合闭环完成时间扰动逐步改变到达、排队和缓存竞争。最初的扰动仍可能来自 CPU、通信、native TopK 导致的路由差异等；当前 raw 不能唯一定位它。

几个可复核例子说明等待发生在哪里：

| 请求末尾 | A1 / A2 TTFT | A1 / A2 admit→首次 forward | 缓存与执行段 |
|---|---:|---:|---|
| `64JxJ2SsaBv32vna0Eq9U:llm:1` | 3.69 / 140.22s | 1.104 / 137.639s | cached 同为 18176；forward→first 约 2.48 / 2.47s |
| `lc_20260924_083:llm:0008` | 6.69 / 38.21s | 6.020 / 37.522s | cached 同为 150272；forward→first 约 0.40 / 0.41s |
| `lc_20260924_211:llm:0008` | 56.34 / 1.75s | 56.033 / 1.429s | cached 同为 67584；forward→first 约 0.18 / 0.18s |

`ReqTimeStats.set_forward_entry_time` 只在首次进入 forward 时写入，所以这里不是误把最后一个 chunk 当首次执行。等待的位置已定位，具体是哪个其他请求/host load/队列决定造成的，仍需要对应服务日志；不能仅据 raw 宣称 GPU 在空转，也不能宣称所有差异都是缓存 miss。

现在可缓解的做法是减少长 prefill 对已开始输出请求的极端干扰，并缩短常态冷块的不可抢占时间；这正是两臂的机制目的。比较时保留共同 ID、开场/稳态、链首类型、到达偏移与缓存是否变化，检验收益是否由少数尾部或完成分母造成。固定预热、flush、namespace 和构建身份，核实 125 `rate` 在 flush 后保留这一现有行为。**不要为追求重复性改 harness 到达方式、顺序或统计；也不建议现在引入新的全局确定性排序或重写 cache 淘汰。** 两次基线只能展示已有波动，不能给出严格统计置信界或为任何新增坏例免责。

## 4. harness 对优化目标的具体约束

`N` 是占用逻辑会话槽的数量，包含 replay gap。`drive` 在上一个请求完成、gap 结束后才派发后继；因此服务更快会反过来改变后面 GPU batch、prefix 竞争和窗口完成集合。不同配置下，N42 不等于每步 batch42，也不等于同一时间轴上的相同混合负载。

TPOT 由发压侧首末有效 SSE 时刻及输出计数计算，再逐请求平均。一个 1s stall 对 29-token 请求增加约 35.7ms/token，对 1000-token 请求只增加约 1ms/token；提高总体 TPM 不能保证降低这个均值。优化首 token 前排队主要改善 TTFT；优化首 token 后等待才直接作用于该请求 TPOT，但前者也会通过闭环间接改变后续负载。不能延迟或批量藏 token 来改变计时，也不能改输出预算。

fast 桶只读冻结 `uncached_expected`。一个被标签为 fast 的请求因实际 cache miss 需要更多 prefill，仍必须留在 fast 桶；缓存优化既可减少真实工作，也能减轻这些冻结 fast 请求的尾部。规则与代码：[task.md](../../llm-challenge-arena-v1/task.md)、[s1_loadgen.py](../../s1-dev/harness/s1_loadgen.py) 的 `call_engine`、`drive`，[s1_common.py](../../s1-dev/harness/s1_common.py) 的 `q`、`phase_gate`、`in_ttft_gate`。

正式 47266 已在 N42 PASS，mean43.266ms、p9571.276ms；chain 点 p95 为45.075s仍 PASS，说明不能把30s点门当平台最终判定。平台没给 N46 失败档的详细 raw，不能断言正式只卡 chain、fast 或 TPOT。原本地 N42 自己已有 fast/overall/TPOT 门失败，因此本轮以同条件相对效果决策是合理的；数值正确性、数据完整性、真实时间与 token 计数不因此放宽。[正式终态](../official/attempt-47266-final-20260929.json)

## 5. 五项小修与 175/176 的代码判断

| 改动 | 审查结论 | 不能夸大的地方 |
|---|---|---|
| 175 MoE W2 / 176 DSA 页复用 | 当前均值改善在 N42/N46 都有相对证据，保留；chain 不应写成稳定改善 | 两项只能按已测组合归因；06 的 TPOT p95 及 chain 代价必须保留 |
| 178 shared prefix mask | 按当前 metadata 生命周期共享；新 forward 重置 source，identity 相同才取缓存；capture 回原比较，合理 | 省的是 prefill 层间重复准备，不是每个 decode token 都省260us |
| 179 Mamba track gather | 修改 track 位置前快照，GPU gather 产出批次自有 tensor；pinned mask/length 保留于 batch/copy；静态 pool、无 spec、B≥8 的限制合理 | 实际 prefill 大量 B1，所以 B32 prepare 收益不能按所有 forward 乘 |
| 181 query view | BF16 完整连续对齐 query、无另接 RoPE 时只读复用；守卫不满足回旧 concat；正确性边界清楚 | 每层微秒量级，不能当毫秒级主因 |
| 182 alloc index | 两次原高级索引写共享一个新 pinned→GPU index；返回 CPU list 不变；静态非 lazy、PP1、非 spec/capture 限制合理 | 分配/prepare 路径的小项；不是 CUDA graph 内普遍收益 |
| 184 HiCache index alias | 只有 Tensor 对象 identity、成员数与顺序完全一致才共享；normalized transfer 构造新对象；未改变 tree-owned 原对象与事件契约 | clone 相等、子集、缺成员仍 fallback；不是所有 transfer 都省两次复制 |

没有看到应立即撤销这五项的具体正确性缺陷；07 的真实权重/服务结果也支持把它们保留为一个整体。179/182/184 的重叠、异步与缓存生命周期风险已受 guard 限制，不能据静态阅读升级成所有配置均安全。174、177、183、185继续保持当前未采用状态；不要把默认关闭的实验路径顺手加入最终包。

审查源码集中在[当前 worktree 引擎](../../build/worktrees/final-execution-0930/engine/sglang/)、[178–184 说明](../../build/worktrees/final-execution-0930/engine/docs/)。

## 6. DSA metadata fusion：可进 TP8，但不应阻塞两臂

结论是“已有材料足够批准真实 TP8 资格试验”，不是“已足够直接提交”。当前 guard 检查 CUDA/非 HIP、page64、page 与 pool 对齐、topk 与 pool 对齐；实测 envelope 是 pool4/topk2048/BF16、TOPK_V2=0、无 MTP。该环境应在提交包中固定。这个开关也覆盖 verify/draft 路径，当前未用 MTP不代表这些路径都取得本次资格。

融合 kernel 的 `bs`/grid 是静态的，live loop 边界从 device seq_len 读取，跨 replay 的缩长、增长和换 req 能更新有效 prefix。尾部保留旧值是有意的 contract，正确性依赖所有 consumer 按 live length 读取；page/pool 边界与完整 selected padding 已做 poison 检验。空行/无效行及 actual rounded-scale cache producer 的状态检查补上了原来最重要的缺口。现有 persistent buffer、req mapping、pool 与 graph 地址不变，没有另开 KV 池或缩历史。

native TopK 的 atomic 顺序本来就不确定。完整 selected 集合与 padding 一致、共同 selected 输入的 output/LSE 逐位相同，说明 metadata 路径未改数学集合；独立 native replay 的 BF16 output/LSE 可出现微小差异。当前 self/cross 样本分布支持其进入服务试验，但有限 baseline-self 最大误差不是全输入误差上界，也不是模型级输出逐位保证。[主资格](metadata-fusion/receipt.json)、[补充资格](metadata-fusion/supplement-receipt.json)

我直接打开 rank0 原 trace 重新计数：5次完整 forward 中，宽 gather grid `[32,1025,1]` 恰好5次，134218240字节 DtoD恰好5次；`_paged` 与 `main_kernel` 各55次。宽 gather/copy全5步平均分别153.85/158.38us。**元数据一完整 forward只做一次，不能乘11层或8rank。** warm完整步约18.74ms，当前B32宏观可兑现预算仍按0–0.35ms估计；B48单stage测到0.52ms也不能外推本次完整模型时延。[原 trace](pod/decode-profile/traces/130ezo03-130ezo03-TP-0.trace.json.gz)、[调用计数](pod/decode-profile/metadata-callcount-independent.json)、[关键路径](pod/decode-profile/criticalpath-audit.json)

容量臂可选附加融合，但只在以下检查能用现有工具于10–15min内完成时；否则保持 off，不拖延 TPOT 臂：

1. 固定186日志差分/commit、完整 env 与 source hash，8rank都确认 pool4/page64/topk2048；MTP关闭、TOPK_V2=0、BF16KV、KV容量1810112保持预期。
2. 既有12题真实权重冒烟通过；这只检查服务/答案基本行为，不替代正式双90。
3. 一个短混合长短上下文的并发组，真正触发高 batch graph 与 padding，随后请求完成缩批、换 RID、flush后再运行。核零 illegal access、NaN、计数错误、错误响应、池泄漏；确认 `/flush_cache` 成功且缓存真清。必须从 batch/route 日志确认覆盖，不能把“发48个HTTP请求”当“已经跑过B48 graph”。
4. 前置结束后重新按既有预热/flush协议开始测量，保存完整 raw、日志及收据。若触发不了所需高 batch，资格仍有限，不强行把剩余时间花在新平台上。

若容量臂加入 fusion，结果只支持 **cold12+fusion组合**；不能单独归因 cold12，也不能把未经单独测量的融合收益追加到 TPOT 包。

## 7. 外部稀疏注意力能借什么

本地 active BF16 decode已经是 TileLang tensor-core路线，具有 pipelined load/计算；不是未用 SM80 MMA 的朴素实现。TP8 profile 中 `main_kernel` 的 grid 为 `[32,1,1]`，说明 B32 时每层只有32个CTA，这给“按 selected KV 拆分以增加并行”留下机制上可信的研究方向，但 profile本身不能证明其收益。

AppMana 的 Ampere FlashMLA确有 split-KV、selected-row dequant 与 online-softmax combine，可借其任务划分；其 sparse cache API 是 FP8/INT8存储，公开性能形状/硬件也与当前 BF16、H8、TP8不同。直接移植须处理 cache layout、scratch、invalid slot、padding、输出/LSE、归约顺序和 graph 生命周期；本地 BF16无需 dequant，照抄预pass反而可能增加读写。[AppMana实现说明](https://github.com/AppMana/forks-flash-mla-int)

本次查到 FlashInfer 对应 `gen_batch_decode_mla_module` 的 SM80 tensor-core选择还带有 head/dtype 条件，不能从“支持Ampere”推到“本模型H8 BF16现成可用”。DeepSeek官方 FlashMLA在09-30的最新 main又变更架构与KV格式，当前文档要求SM100/103，不能按旧名字直接替换。[FlashInfer dispatch源码](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/jit/attention/modules.py)、[FlashMLA当前说明](https://github.com/deepseek-ai/FlashMLA)

更小的现有重复工作是 native KPool topk产生2051列，然后 `_forward_tilelang` 在11层分别 new_full61列再 cat到2112。可让 producer直接写physical padding，省掉小分配/复制；但 C++当前由 out_cols推导semantic topk，不能只改Python输出宽度，否则可能把2048预算静默改坏。需要分开“逻辑2048+尾部”和“物理2112”，且保留每个-1。它值得下一轮做，当前只预期几十us级，尚无完整graph收益证明，不值得抢最后两臂。[producer](../../build/worktrees/final-execution-0930/engine/sglang/kernels/ops/moe/kpool_topk_transform.py)、[consumer](../../build/worktrees/final-execution-0930/engine/sglang/srt/layers/attention/dsa_backend.py)

另一位执行层审查者随后对同一原 trace 重计，这组 padding fill+cat 在 rank0 五步中共55对，稳态四步平均42.20us/完整模型步；与这里的预算判断一致，不能把每层约几微秒说成全模型省几毫秒。[独立稀疏执行审查](sparse-attention-final-review.md)

因此，本轮最可靠的外部工程启发是 **只处理live/selected数据、避免每层重复准备**，当前fusion与176已经落实其中一部分；更大的split-KV值得做，但不应把“支持SM80”的仓库当三小时可直接兑现的确定方案。

## 8. 两项实验、两个提交包、改变判断的条件

| 选择 | 精确变化 | 主要证据目标 | 主要失败模式 |
|---|---|---|---|
| T：N42 TPOT臂 | 07底座；ordinary PDI2、risk1、cold16k都保留；仅 `BACKLOG_INTERVAL=1` | 开场五条/65条共同请求的 post-first时长显著缩短；全共同ID均值改善；chain代价明示 | 一轮轮增加decode推迟长冷链首；闭环提早到达造成后段拥堵；额外decode不兑现有效token |
| C：N46容量臂 | 07底座；relief0/risk1/PDI2不变；仅常态 `SCHED_COLD_CAP=12288`，全局chunk/backlog cap仍16384 | 更短常态不可抢占块能否降低fast/overall及TPOT尾部，同时保住chain；对06/02共同ID | 多付prefill固定开销、冷长请求完成变慢、更多调度轮次/缓存竞争；容量收益不成立 |

T为什么优于普通PDI4：PDI4不改变开场relief0，也不改变risk1；旧N34只有0.81%均值线索，且47607已提交旧包。T不是重复原条件：当前底座/N42不同，且新数据已定位其实际作用区间，用户也调整了取舍。T仍有旧chain新增两条的负面证据，必须如实保留。

C为什么不是把所有块切小：源码先取常态cap，再在relief时用上轮broadcast的backlog cap覆盖；保留16k开场产能，主要测试steady段的延迟/固定成本折中。它不是已证明N46可过；原06仍有高TPOT尾部与chain代价。如果第一臂已经表明开场瓶颈决定全部收益，而C没有改善steady等待，容量包就不值得采用。

**采用时看具体因果表现，不设“本地必须全PASS”门，也不把两次基线波动当免责条款。** T若开场stall没有明显缩短、共同ID均值改善不超过已有约0.5ms量级差异，则不值得为它承担chain代价。若chain同一批长冷请求对A1/A2/05/07都系统变差，不能用“只是新增一两条”掩盖机制退化。若改善来自窗口新增易请求而共同ID无改善，同样不采用。相反，开场stall与共同均值明确改善、chain代价小且落在已观察不稳定请求上，可作为TPOT包由用户决定；继续保留一个relief0的提交方向来分散风险。

优先的正式包T是通过上述证据的07+relief1；若失败，退为07，不能在最后一刻增加未测PDI4/fusion来补故事。正式包C是通过证据的07+cold12/backlog16；若实际加入且通过fusion，只提交已测组合。如果C失败，不强凑“容量包”名字，应回退到已测的较优候选；两次提交不要求一定凑出两种相反风险。

不建议为这次比赛再写新的pacing/deadline控制器。最小新代码设想是补125首token-only观察盲区/限制最长decode空隙，但只补`produced==1`仍受MAX_SLOW80钳制，改成即时反馈又需要rank0一致性、冷链保护、flush重置与过载稳定性资格。已有relief1可直接检验核心方向，代码风险更小。

墙钟预算按07实测派发+排空62.8min估算：两臂60min派发加启动/冒烟大约136–150min。当前不能再把15min附加资格当免费。建议把最终数据闭合与候选冻结压在08:40–08:45，给09:10截止留至少25min构包/检查。第一臂若启动较晚，第二臂使用45–50min派发再排空，仍按DRAINED共同ID诊断解释；不能为了凑整小时留下OPEN数据或错过提交。镜像工作可在固定代码后准备，但本报告没有执行任何构建或上传。

**最终判断：没有现有证据能承诺追平37.352ms或通过N46。最值得花最后两臂验证的，是已经由代码和原始坏例定位的prefill干扰；执行小修作为底座保留，fusion作为可放弃的附加项，外部kernel移植留给有时间完成完整资格的下一轮。**

## 9. 最终两项 job 的发布前只读核验

Root随后准备了[round6计划](pod/round6-publication/plan.json)。实际选择为T派发60min、C派发45min，**两者fusion都维持off**；这符合“优先现有可兑现路径、给构包留时间”的建议，本报告第6节的fusion资格描述不构成对本轮启用它的建议。

我按shell赋值解析对比round5原07/08，确认09唯一env变化是relief interval0→1，10唯一env变化是常态cold16384→12288，10另将派发3600→2700秒；两者G_ARGS、G_EXPECT、commit及删除上述行后的其余全文完全相同。PDI2、risk1、MAX_SLOW80、backlog16k、短请求2048、175/176与五小修均保留，174/177/fusion均off。两个脚本`bash -n`均返回0。

```text
130ezo09-n42_exec_relief1_1h.sh
SHA256 68d7cf0a5f5cae339147113721f872ff0e547c68a01811df90f890c54b0ccb99
130ezo10-n46_exec_cold12k_45m.sh
SHA256 e4afda51c61b2d83830d75f9c96b27893baebc37b178e1d34ffd659b48897210
```

这里只确认准备中的任务内容与建议一致；尚不声称已部署、实际机制已触发或实验已通过。部署与运行期核验由执行者继续保存。短45min与旧1hN46的比较必须明确时长差，并使用共同ID；本地绝对门失败本身不推翻候选，但不能略去真实chain、TPOT尾部或错误代价。

## 10. 08:39 UTC：09已闭合，开场收益与真实代价都得到验证

09的[完整窗口收据](pod/relief1-n42/timed_window.json)是DRAINED：3273派发、3273完成、0错误，排空于派发结束后146.64s。原始[raw](pod/relief1-n42/raw_s1-dev-longchain-v5g-tail-rot150_N42_1790750356.jsonl) SHA256为`e84ea78c6992179305d64cd2f8f7f24c532d246e995d749c94afb37d178dd52b`。总体mean57.0865ms/p9599.4424ms；chain16/374、p9528.5535s，turn5/119。这个p95优于07且低于100ms，只说明当前DRAINED诊断窗口，不能升级为完整N42已PASS。

同07的3259个共同请求mean59.9422→57.1510ms，省2.7912ms，约4.66%。这不是新增完成分母造成的。按07派发时刻固定的前120s共67个请求，mean453.422→335.574ms，对全体共同请求的均值改善贡献2.4228ms。

| 原07最慢的五条开场请求 | 输出数（两臂相同） | 首末token间隔，07→09 | TTFT，07→09 |
|---|---:|---:|---:|
| `zlBy3aBiTqMNBYqPsCGRf:llm:0` | 29 | 92.99→13.06s | 2.96→1.90s |
| `rz5iTPs02_Up8QlQwD-3I:llm:0` | 35 | 89.37→16.04s | 7.06→6.55s |
| `PwNBTuO_uXv_YBEeSwric:llm:0` | 46 | 91.53→22.04s | 5.33→1.30s |
| `B2LeYpwICaaES203Rc73M:llm:0` | 54 | 94.94→29.50s | 2.02→11.49s |
| `FWn2Kweqx4Q0VpssNR5W4:llm:0` | 71 | 94.37→36.61s | 5.06→0.74s |

仅这五条贡献净均值改善2.6436ms，约占全共同请求净收益95%；其余请求有改善也有抵消。这验证了此前定位的开场post-first问题，收益高度集中也意味着不能把当前4.66%同比套到正式隐藏负载。

**chain代价有明确方向，不应全部归为噪声。** 固定07开场51条chain：mean21.780→24.149s，超标11→12，11条增加超过5s、0条改善超过5s。`member-v5-subagent-46ba67487c3c0d98b3ae9b75:llm:8`在A1/A2/05/07分别28.77/28.94/28.93/28.13s，09为44.66s；这是在四次高度一致的开场基线上出现的约16s恶化。插入decode减少已经输出请求的等待，同时推迟部分尚未输出的冷请求，正是预期的服务取舍。

稳态322条共同chain则mean2.901→2.962s，超标2→4，增加>5s共14条、改善>5s共18条，未呈现同样的单向整体变慢。09对07总共新坏5/修好2；其中三个新坏ID是A1/A2已经翻转过的`lc_004`、`lc_083`、`e6pt...:101`。另一个稳态新坏`member-v5-subagent-cde2c24f6c06a661bc209ae3:llm:7`从四次基线2.2–5.2s变33.14s，等待主要在admit→forward，不能用已有重复噪声自动免责。

09对A1共同3133请求mean61.514→57.445ms，chain新4/修2；对A2共同3132 mean62.028→57.457ms，chain新3/修4；对05共同3242 mean59.907→57.174ms，chain新5/修4。因此其TPOT改善跨基线成立，chain总体仍有风险，开场代价尤其可信。

**提交建议更新：09可作为值得押的一包TPOT候选，但必须向用户同时展示上述chain代价；另一包保留relief0方向。10还未取得闭合原始数据时不能认定容量包成立；若10没有可靠相对改善，第二包回退07，而不把未经测量的09+cold12或fusion临时组合进去。正式提交由用户选择，本审查未执行任何上传。**

## 11. 08:48 UTC：10已闭合，最终优先推荐07与09

10的[raw](pod/cold12-n46/raw_s1-dev-longchain-v5g-tail-rot150_N46_1790754762.jsonl) SHA256为`955385b9fb46761b6dd1be581f4d5d683a154d9c5cdd1121fc0da20ebcba3aab`；[窗口收据](pod/cold12-n46/timed_window.json)确认45min派发、2541派发/完成、0错误、无outstanding、rc0、DRAINED，末请求在截止后268.87s排空。2541个候选ID全部出现在02和06，比较无需舍弃任何候选请求。

不能拿10总体67.006ms直接对06一小时总体62.666ms声称变慢：前45min的开场高TPOT占比更高。正确的共同ID比较如下。

| 2541共同请求指标 | 02原配置 | 06（175/176） | 10（175/176+五修+cold12） |
|---|---:|---:|---:|
| TPOT mean | 69.400ms | 67.767ms | 67.006ms |
| TPOT p95 | 116.618ms | 119.770ms | 114.926ms |
| chain超标 / 324 | 20 | 22 | 21 |
| turn超标 / 97 | 10 | 8 | 11 |
| overall超标 / 2120 | 244 | 243 | 247 |
| fast超标 / 1890 | 287 | 271 | 271 |

10对06共同chain修好5/新坏4；对02修好5/新坏6。对06均值省0.761ms，约1.12%，p95省4.844ms，属于真实观测到的组合相对改善，**不能因其绝对p95仍超过100ms就机械否决**。但逐请求TPOT差值的中位数为+0.080ms，并非普遍变快；turn及overall有所退化，fast超标不变。双方均在600s之后派发的1980个共同请求，mean53.869→52.736ms，chain超标6→6，也没有明确的容量门改善。这个分组仅排除开场作机制解释，正式/总体评分仍包含所有请求。

chain坏例中，`member-v5-subagent-3c7fedeb4c66e4f7a7ce64ca:llm:2`从02/06的1.43/1.65s变73.02s；`lc_20260924_083:llm:0011`从20.49/23.15s变95.56s；`member-v5-subagent-cf6c4e2ad57ddac6b56febef:llm:2`从1.39/1.92s变55.54s。它们主要是admit→forward等待变化，同时另有请求被修好，所以不能把单个极端值写成整个调度更慢，也不能忽略这些代价。

由于原08取消，没有N46“五修但cold16”的单独对照，10对06只支持“五修+cold12”的组合效应。07在N42五修的TPOT未分辨出宏观增量，并不等于能精确把10所有差异归给cold12。45min共同ID比较有效地排除了直接分母误读，仍不能补出未运行的后15min或完整cohort。

**最终优先推荐两个已测配置：BASE07（relief0、常态cold16k、backlog16k）与TPOT09（relief1，其余同07）。** 两者都保留175/176与五小修，fusion保持off，不增加未测组合。07沿用47266的调度取舍且已有当前N42结果；09有明显、可解释的均值收益，同时公开说明开场chain代价。10是有小幅相对收益的备选，并非INVALID；但当前证据不足以将它提升为“更可能过N46”的首选，不能仅为了凑容量包而替换07。

若用户仍选择容量风险，CAP10可按其已经冻结的原配置提交，说明其N46容量尚未得到完整资格、不要承诺超过07。这里是风险与收益的选择建议，不是严格统计胜负判定；只有一次N46候选且负载闭环，没有足够重复给出可靠置信界。正式提交继续由用户决定，本审查只读取文件并更新报告。
