# 外部执行实践对 GLM-5.3-Flash/A100 prefill 的可迁移性

2026-09-27，Codex 子任务；只读调研，没有改引擎、使用 GPU 或提交。本页的「本地实测」引用已有原始运行/研究；「外部实测」只代表来源所列硬件与版本；「适配推断」均待本地验证。读码对象是作者 worktree `b0ddb2ca` 的 `engine/sglang`；用户给出的合并测试引擎是 `791453ca`，不能把本 worktree 文件存在当作其开关已开启。

## 决策边界

- 正式排序先 `n_at_slo` 再 `tpot_mean`，TPOT p95 ≤100 ms。线上 S2 N26 均值25.1/p95 47.5 ms 是当前余量，不等于能自由花掉 52.5 ms。开场冷链、长提示、分块后首字及准入竞争是优化对象；本地与正式分布不同。[本地榜单与单请求测量](leaderboard-prefill-audit-0927.md)、[开场核验](chain-opening-source-audit-0927.md)。
- 单 49,143-token 冷提示实测首次入批→prefill 完成 3.738 s，实际约 13.1k **输入** token/s；原 26.5k 把 target/draft 两个同名 EXTEND span 当输入块，已撤回。8k target 约570–650 ms、draft 约17–28 ms；当日 20s 窗口 98.7% prefill，不能凭它推断长上下文的混跑损失。
- 已有 Humming 117；[R32](/workspace/Agentic_science_challenge/research/codex/R32_117_humming_effect.md) 的 A100 TP8 每卡 8k 单层 5.820→3.496 ms 是**本地单层实测**。EP8 在开发机单层慢 13–24%，不再把 EP8 flag 当近线候选。[R25](/workspace/Agentic_science_challenge/research/claude/R25_moe_sm80_path.md)。171 KDA 投影融合已实现，056/057 整档仅 1–2% 墙钟变化、门不变；170 breakable prefill graph 已有 v2，TP8 数值/收益未闭合。[171 说明](../../engine/docs/171-kda-bf16-proj-fusion.md)、[170 说明](../../engine/docs/170-glm-bcg-prefill.md)。
- 45 层（34 KDA/11 DSA）+ MTP，FP8 权重在 A100 上由 BF16 运算路径承接；A100 没有原生 FP8 MMA。每卡 target 约39.15 GiB、draft 0.85 GiB（L109/R34 实测）。简单 4+4 双完整副本会让 target 约78.3 GiB/卡，几乎无 KV、KDA 状态与激活空间，不能据此规划 PD。现行 DCP2 是 MLA KV 物理分片、indexer 复制，不是 prefill CP 算力翻倍，也不能默认与 MTP 正确组合。[R24](/workspace/Agentic_science_challenge/research/claude/R24_glm53flash_official.md)、[R34](/workspace/Agentic_science_challenge/research/claude/R34_kv_capacity_options.md)。

## 建议候选与优先级

### 1. P1：真正的长冷 prefill 线索——kpool indexer 计分/TopK 的 SM80 专项

**算快，兼顾临时显存。** 已有 Fable 粗分类中 DSA/indexer 约22%，但该分类不是无重叠关键路径份额；先对 8k 块、P=0/64k/128k/250k 与 batch=1/多链测 `_fp8_mqa_logits`、TopK、page-table transform 的 CUDA-event 时间和峰值。当前模型用 `index_kpool=4` 的 [`dsa_indexer_kpool.py`](../../engine/sglang/srt/layers/attention/dsa/dsa_indexer_kpool.py)，不能套非 kpool 的 `dsa_indexer.py`。现有 110/114 已对适用形状分行，别重复计旧收益。

外部可借的**具体 SM80 代码设计**是 SGLang [#35429，2026-08-19](https://github.com/sgl-project/sglang/pull/35429)：Triton paged-MQA 在 A100 以软件解 E4M3FN、BF16 `tl.dot`、FP32 累加处理每查询行/每 KV 页；外部实测的是 T≤32 的普通 eager **decode**，例如 H64/T32/P128k，Torch 255.6 ms、Triton 6.25 ms。这只是相对 Torch fallback，绝不是相对我们现行 DeepGEMM 的 41×。PR 明说没有 ragged prefill、MTP target verify、graph、CP 或完整模型验证，所以不能直接切 backend。可把其 FP8 解码、页寻址和软件 BF16 矩阵核作基础，给 kpool 的 ragged 大块另写/改 kernel，并与当前 DeepGEMM 计分、TopK 逐行对比；TopK 必须保持相同 tie、尾部强选、`-1`、池化页映射与输出行身份。

内存侧可以移植已合并 SGLang [#40854，merge `63d320c` 于 2026-09-26](https://github.com/sgl-project/sglang/pull/40854) 的**按查询行限制完整 FP32 logits**。外部 4×GB300/TP4、1M 输入实验把 allocator OOM retries 104→0、峰值283,210→271,382 MiB，TTFT 132.3→134.2 s：它是容量/稳定性补丁，**没有证明提速**。本地 8k×250k 的 FP32 logits 算术约1.91 GiB/层（kpool÷4 与 FP32×4 抵消），多请求合并 K 可能更大。先测是否触发分块、allocator 重试或显存峰，再决定是否移植；每层动态查 free memory 的上游实测约46µs/次，可与已排队 GPU 工作重叠，不可直接把调用次数乘成关键路径。

最小区分性实验：固定真实权重/TP8/DCP2/MTP 开关，单纯替换 kpool kernel 对 P=0/64k/128k/250k、c=4k/8k/16k 的同输入，分别记录该算子 CUDA 时间、整个 target forward、临时峰值、TopK 索引一致/边界、最终 logits 与 KDA 状态；再同 ID 开场短测。**待验证假设**：长 P 的每块 target 下降，且 chain 修复数高于新增数；若只省 logits 内存却不降 target 时间，则把它归为容量修复，不报 prefill 加速。

### 2. P1/P2：prefill context parallel 的受限试验，不与 DCP 混称

**算快的一部分，通信可能抵消。** 上游 [CP v2 roadmap，2026-06-04 起](https://github.com/sgl-project/sglang/issues/27252) 将 `--enable-prefill-cp --cp-strategy interleave --attn-cp-size N` 用于 DSA 长序列；[GLM-5.2 cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/GLM/GLM-5.2.mdx) 是相关模型公开配方，但没有 GLM-5.3-Flash/A100 的本题吞吐数字。现行源码 [`glm5_next.py`](../../engine/sglang/srt/models/glm5_next.py) 在入口 `cp_plain_split`；11 个 DSA 层可按 query 行分担 indexer/attention，`dsa_indexer_kpool.py` 也有 CP gather/plan 分支。然而 34 个 KDA 层的 `forward_qkvbfg` 先 `cp_plain_all_gather` 完整 hidden，再投影与递推，输出 `cp_plain_reduce_scatter` 或 split：KDA 的时序状态要按原 token 顺序推进，简单分块跨 rank 不能独立算而只拼结果。MoE/mHC 还经 CP communicator 重排。故 CP2 不是整模型 2×，并可能增加每层 all-gather/reduce-scatter 与临时激活。

这里必须分清 **prefill CP**（将同一次 forward 的 token/query 分给 rank，可能减少 DSA 工作）与 **DCP2**（decode/MLA KV 按 rank 存，主要扩物理池）。`--attn-cp-size 2` 与 DCP2 对同一 TP8 topology 的组映射、head shard、indexer KV 写入、`move_kv_cache`/MTP draft 及 HiCache 恢复要逐项检查；现行 `server_args.py`/`parallel_state.py` 存在两套独立大小，不能认为参数组合合法即数值正确。CP 若改模型 head 分片/权重装载，应记录 rank 映射；若加缓冲，先量其峰值并换算 KV/状态槽损失。

最小试验：保持 DCP2/MTP 的真实服务配置先做 CP1 与 CP2 各一次 TP8 正确性（多块 49k、≥128k、缓存续接与 `flush` 后冷算、MTP target/draft logits 与 KDA 状态）；测每层 DSA/KDA/MoE/collective 与整个 target forward，尤其 8k×P128k/250k。若 KDA/通信抵掉 DSA 节省或池缩小造成新 chain 失败，止步。**待验证假设**只限长上下文 DSA 项降时；没有外部证据支持本模型 A100 整体净收益。

### 3. P2：170 breakable prefill CUDA graph 配合小块调度

**小块固定成本变便宜，使 4k/8k 的 TPOT 交换可行。** 上游 SGLang [#38522，2026-09-10 合并](https://github.com/sgl-project/sglang/pull/38522) 针对 GLM-5.3-Flash KDA/DSA 提供 opt-in graph；外部 4×B300 agent 请求 output tok/min/GPU 在 C16/C32 增约20%，不能搬到 A100/当前服务。我们已有 170 v2，TP1 替身的小块 eager 有显著 host 间隙，但 R10 的约150 ms 固定截距是旧曲线推断；大块当前负载 target 约0.57–0.65 s 更可能 GPU 主导。graph 首先用于让小块更便宜、减少 decode 停顿，**不应承诺 16k prefill 更快**。

最小试验：真实 TP8、MTP/140/128p/DCP2 组合上做 eager170 对 BCG170，同 ID、同 c=2k/4k/8k 的逐块 target+draft、首字、decode TPOT p95，另报 graph 捕获常驻与峰值显存折算损失。先验全链输出和 KDA/DSA/indexer 缓存状态；既有 v1 TP8 scatter 错误只在 v2/TP2 修过，不能跳过 TP8 验收。若 8k/16k 只增显存而无省时，限于小块启用或否决。

### 4. P3：按现有通信账做局部融合/overlap，先排除已用路径

**算快，但必须逐条 collective 证实。** 当前 Fable 粗分类 all-reduce 约12%、mHC 约14%，不等于可相加的25%墙钟。SGLang `glm5_next.py` 已有 `LayerCommunicator` 的 MLP reduce-scatter 和 170/114 的输入 scatter；[上游 CP roadmap](https://github.com/sgl-project/sglang/issues/27252) 明确旧 CP 通信分散在 `communicator_dsa_cp.py`、模型与 indexer。可具体针对 profiler 上最长的相邻 `all_reduce`→mHC norm/prepare 或 CP gather→KDA projection pair，尝试已有 communicator 融合开关/重用缓冲，不写全模型重构。8k×4096×BF16 是128 MiB；按真实形状记每次 collective 字节、耗时、前后 GPU idle、是否关键路径，不能拿一个总百分比当收益。

最小试验：为 8k×P0/128k 和 4k 多请求采带 rank/stream/dependency 的 TP8 trace；只改一处通信或融合，记录每步总时间、各 rank 最大值、临时缓冲、decode 和 TPOT p95。若 collective 隐在 MoE/DSA 下，先不动。额外显存若消耗 KV 池，需用高并发链收益抵账。

## 明确暂缓/否决

1. **原生 FP8 tensor-core、Blackwell FP8 sparse attention 直接移植 A100**：A100 SM80 无原生 FP8 MMA；当前 FP8 专家权重走 BF16 计算。官方 [GLM-5.3-Flash cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/GLM/GLM-5.3-Flash.mdx) 的 DSA backend/FP8 KV 是特定新架构组合；本地 R24 已核 A100 tilelang FP8 KV 无现成 CUDA 路径。FP8 存储+BF16 读取需新 kernel 与质量验证，属于容量项目，不能称无代价提速。
2. **把 DCP2 或 4+4 PD 当冷 prefill 加速器**：DCP 缩 MLA KV，indexer 仍复制；TP4 双 target 约78.3 GiB/卡，KDA 状态/激活不可用。官方 [PD 文档](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/pd_disaggregation.mdx) 明确不同 TP 的布局传输问题，其 staging buffer 针对非 MLA；本模型又要同时搬 indexer 与 KDA/conv 状态且保留 MTP，当前设备不作为近线方案。
3. **重复上线 117 Humming、EP8、171 融合**：117 已用；EP8 局部慢；171 已有整档 1–2%/门不变负结果。除非 profile 指出新 batch/块形状或组合机制改变，不占新完整档。更深 MTP 可当 decode 资源交换，但需要真实接受长度、草稿状态/显存与 TPOT p95；它不减少 cold prefill target 计算。[模型官方资料与 MTP 边界](/workspace/Agentic_science_challenge/research/claude/R24_glm53flash_official.md)。
4. **只调 8k→16k 或任意大块**：已有初步约8%/token 下降，但单块时间增加，会延后 decode 与其他 chain；需看同 ID chain 修复/新增、fast/overall、TPOT p95 和显存峰。块长是调度交换，不是独立算子突破。

优先顺序依赖现行 DCP2 的**真实关键路径**：先做无改动的 TP8 P/c/batch 成本表与峰值账；如果 DSA/indexer 随 P 增长占比高，推 1；若小块 host 间隙显著，推 3；若长 P 上 CP 的 DSA 项可能盖过 KDA+通信，才推 2。上述局部百分比不得相加为 N@SLO 预测，最终用原 harness 全量闭合判分。
