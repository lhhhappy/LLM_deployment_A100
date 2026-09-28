# SGLang PR 审计：GLM-5.3-Flash / TP8 A100 的 prefill 执行层

> 2026-09-28 归档快照：技术正文保留原审查范围，仅将本地链接改为归档相对路径；本轮最终状态以 [R36](../../../research/codex/R36_prefill_kernel_options_0928.md) 为准。原件与归档件 SHA-256 见 [sources.json](sources.json)。

2026-09-28；审计对象为 `build/worktrees/mhc-moe-sm80-0928` 的 `5883845b`，引擎基于 `20a58da9`，MoE 后续生产提交 `eaf41653`。只读源码与上游 PR；本报告没有 GPU 新实测、没有改引擎或服务。**优先查整段 prefill 的主机发射和串行同步，而不是再优化 117 的单个 MoE kernel。** 当前 8k/16k 每卡 full rows；attention-TP 输入 scatter 默认关。`170` 有 breakable prefill graph 但未默认启用；`171` 已有 FP8 模型中免量化 KDA 投影的 6→2 调用融合；`114` 已有 indexer query-row TP 切分；`118` 已有 SM80 sparse prefill。以下收益均为筛选假设，必须在相同 TP8 服务配置量 `T(c,P,B)`、真实显存与 chain。

## 1. 首选：使 34 个 KDA extend 真正在 breakable prefill CUDA graph 中回放

上游 [SGLang #41105](https://github.com/sgl-project/sglang/pull/41105)（2026-09-28 仍 open，head [`0e54824a`](https://github.com/sgl-project/sglang/commit/0e54824a0178)）直接针对 GLM-5.3-Flash：原 breakable graph 在每个 KDA 层断开，KDA 的十余 Triton kernel 和 Python/host 表准备仍 eager；该 PR 按 bucket 预分配静态 chunk/conv/track 表，replay 前以一次 pinned 异步 H2D 更新，固定 kernel grid，在 DSA 层才断开。其 [描述与 8×B300 实测](https://github.com/sgl-project/sglang/pull/41105)给出 4093-token step 的 launch 1182→439、CPU profiler 116→42ms；主线 BCG→此 PR：单请求 1k TTFT 47→37ms，4k 152→113ms，16k 419→384ms。B300+FlashInfer MoE+TRT-LLM DSA 与本 A100+Humming+Triton DSA 不同，不能沿用百分比。上游 16k bucket 配置相对其默认 4k BCG **多约 22GB/卡图池**：这是全图的大 bucket 容量差，不是 KDA kernel 单独多 22GB，也不是 #41105 相对同样 16k BCG 的增量。

本地调用链：`Glm5NextKDA` (`engine/sglang/srt/models/glm5_next.py`) → `RadixLinearAttention.forward` (`.../layers/radix_linear_attention.py`) → `KDAAttnBackend.forward_extend` (`.../layers/attention/linear/kda_backend.py:740` 等) → `KDAKernelDispatcher.extend` → Triton `chunk_kda` (`engine/sglang/kernels/ops/attention/fla/kda.py`)；`prepare_chunk_indices` 位于 `.../fla/index.py`，从 GPU lengths 构表并缓存于 tensor 对象。`causal_conv1d_fn` 当前依 `extend_seq_lens_cpu` 决定 grid，track snapshot 还有 host 选行。170 的 `prefill_cuda_graph_runner.py` 只桥接了 KDA 的 eager break，不是此 PR 的静态表图内路径；本地没有 `kda_prefill_graph.py`、`kda_prefill_graph_track.py`、`SGLANG_DISABLE_KDA_PREFILL_GRAPH_EXTEND`，所以**明确未命中**。该 PR 的设计是可移植的 Triton/元数据组织，不调用 SM90/100 MMA；上游专门在 Triton KDA 条件下启用。不能照搬上游 backend/模型文件，因为本地 140 快照、170 scatter 修复、171 投影、MTP 状态契约要一起核。

实现成本：**高**，约 10+ 个生产文件、一个新 metadata 类及 track kernel；需要重放不同 `B/c/P`、前缀 cache 命中、未对齐快照、MTP verify/graph fallback、padding、TP8 输出/状态逐位或数值核对。验证先固定现有服务 `c=1024/4096` 比主线 170 BCG 与 #41105 BCG 的 CPU launches、GPU idle、每块 latency，随后才看 8k/16k：当前 170 文档默认只捕获到 4k，**原 8k/16k 行不会回放**，须显式扩 bucket 并实际量 graph reserved 与 KV/KDA 槽损失；若超 A100 80GB 容量，探索只覆盖高频小/尾块或较少 bucket。上游实现复用 runner 的现有 graph metadata 生命周期，但捕获的是两个 DSA break 之间的整段（KDA、mHC、MoE 等），并非仅 KDA kernel 私有图；若要“KDA-only 局部图”来压图池，须另加图边界/输入输出缓冲和状态重放契约，不能把它当上游 PR 现成功能。不可把改 chunk size 算作纯执行层收益。预期：在小块 host-bound 时有机会拿掉剩余 KDA eager launch 的显著部分；16k 已偏 GPU-bound，且大 bucket 显存风险高，因此整段 16k 收益可能远低于 4k。

## 2. 立即可做：#39688 剩余 KDA 元数据/β 融合，剔除 171 已有投影

已合并 [SGLang #39688](https://github.com/sgl-project/sglang/pull/39688)（head [`8d0fc756`](https://github.com/sgl-project/sglang/commit/8d0fc7560575)）是三个机制的整包：投影融合、β sigmoid 融入 gate/cumsum kernel、用已有 CPU 长度和 track 信息避免 GPU→host 同步。它在 8×H100 TP8/EP8 EAGLE 上报告 8k-input/128-output、c8 总吞吐 **+11%**，但包括本地 171 已有投影，且上游说明测量早于最后 rebase、warm-prefix 三轮全退化；**不能声称剩余子补丁仍有 11%**。本地 171 的单层投影已有 6 linear→1 linear+1 bmm；真实 N22 回放未确认净整档收益。

本地真实链：`ForwardBatch.init_new` → `HybridLinearAttnBackend._forward_metadata` (`engine/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`) → `KDAAttnBackend.forward_extend` (`.../linear/kda_backend.py:740`)；目前每层 `int(query_start_loc[-1])` 将 GPU scalar 取回 host，且 tracking 分支有 `.any()`/`.nonzero()`/`.cpu()`。上游 patch 在普通 extend、长度 CPU mirror 完整、且 `tbo_parent_token_range is None` 时令 `logical_num_tokens=sum(extend_seq_lens_cpu)`，存于 `ForwardMetadata`，否则回退原 GPU 读取；它还把 prefill tracking 的 CPU mask/length 从 `ScheduleBatch` 传入 `ForwardBatch`，只建一次逻辑计划，GPU 索引异步上传。`kda.py::kda_gate_chunk_cumsum` 则可顺手接 raw β 并写 sigmoid，消一轮独立 β op/中间读写。本地这些元数据/β路径**未命中**；不要再搬上游投影模块。#39688 的核心同步修复非 SM90+ 特性，可在 A100 执行。

实现成本：**低到中**，先只做有 guard 的 CPU `logical_num_tokens`，独立于 β/track；若 profile 确实显示同步在关键路径，再搬 CPU track plan 与 β kernel。必须保留 TBO/MIXED/CP/graph padding/snapshot 的回退和原完整状态输出。测量目标：同一个 8k/16k prefill 的每层 `cudaStreamSynchronize` 数、host self time、KDA段 GPU idle、整块耗时。上游的整包 +11% 仅是**宽松上界线索**，当前残余部分没有已公开 isolated speedup；若纯 scalar 修复只省亚毫秒，就停止。它与 #41105 有重叠：后者 graph 内重做 KDA 元数据；不能把两者收益相加。

## 3. 第二个独立热点：#39695 KPool planner 的每块同步及压缩准备重排

已合并 [SGLang #39695](https://github.com/sgl-project/sglang/pull/39695)（head [`bd92922c`](https://github.com/sgl-project/sglang/commit/bd92922c5ca5)）：将 CPU request-slot mirror 带进 `ForwardBatch`，planner 使用它并给 `repeat_interleave` 显式 `output_size`，避免长度/slot 的设备读回；eager 非 CP prefill 让 index-K 压缩和 cache 准备与 query quantization/head gate 并发。上游在 8×H100 TP8/EP8 EAGLE 做了功能与 lookup 验证，但**没有配对性能数据**；review 明确纠正“query rotate/head gate 与 K 准备重叠”说法：非 CP eager 中 `_get_q_k_bf16` 仍单 stream，真实新增重叠仅 compress/cache-write ∥ (act_quant+head-gate)。上游 FULL graph replay 曾有 `_pad_tensor_to_size` 类/实例调用错误，head 已修复；移植以修复后的版本为准。

本地路径：`DSABackend.init_forward_metadata` → `kpool_plan.py::init_kpool_extend_metadata`/`_kpool_cpu_plan`，以及每个 DSA 层的 `DSAIndexerKPool.forward_cuda` (`engine/sglang/srt/layers/attention/dsa/dsa_indexer_kpool.py:1618`) → `_compress_write`/`act_quant`/`_get_logits_head_gate` → `_get_topk_ragged`。当前 `kpool_plan.py:248` 仍对 GPU `forward_batch.req_pool_indices.tolist()`，而 `ScheduleBatch` 已有 `req_pool_indices_cpu`，但 `ForwardBatch` 未传递；本地 `_compress_write` 是 act_quant 后串行，未有 #39695 的 eager overlap，**明确未命中**。planner 在一个 prefill forward 的元数据初始化中运行一次，**不能把 `.tolist()` 按 11 层乘**；compression/query overlap 才在 11 个 DSA 层分别发生。`114` 只拆 query 行、`118` 只替 sparse attention，均不消除 planner 同步。#39695 使用普通 CUDA streams/CPU mirror，适用 SM80，attention input scatter off 与非 CP 条件吻合；若选中了 `skip_logits_computation`、非 plan fallback 或 graph bridge，命中范围会变，必须记录实际 dispatch。

实现成本：**中**，约 6 个运行时文件 + 精确镜像生命周期/stream 安全测试。先测 `kpool_plan.py` GPU `.tolist()` 的发生次数/等待时间及图前后时间；再移植 CPU mirror+`output_size`，与双 stream overlap 分臂计时。若一次 planning sync 并未在关键路径，则预计该部分整块几乎无收益；若 11 层 compress/query 串行确在关键路径，overlap 成立则可能有数毫秒级的整段改善。这里的“数毫秒”是待测量级而非公开实测/承诺；不得把单次 indexer 速度当 45 层或整段收益。

## 排除/暂缓，避免重复计算

- [#39350](https://github.com/sgl-project/sglang/pull/39350) 与 [#39248](https://github.com/sgl-project/sglang/pull/39248) 也是免量化 KDA 投影融合；本地 171 已做 4 个输入投影→1 与 f/g 二级→batched，最多可借加载/数值验收，不能再算新增 KDA GEMM 收益。
- [#40854](https://github.com/sgl-project/sglang/pull/40854) 已合并的 KPool query-row 分块是重要 **OOM/峰值内存**修复：上游 GB300 peak 283210→271382MiB、OOM retries 104→0，但 mean TTFT 132.3→134.2s。可为 A100 长链独立安全修复，不列为整段 prefill 加速候选。更大胆的 [#38469](https://github.com/sgl-project/sglang/pull/38469) 按请求切 K 列减少多请求矩阵空白、再切 Q，仍 open 且仅在超预算时走，公开 warm wall 0.6/0.6/0.7s 对 dense 0.6/0.6/0.6s；若改为常开才可探索 B>1 的 FLOP/带宽节省，但需要重做本地 114 行 shard 与绝对索引回映射，无已证实吞吐收益，先不占前三。
- [#40461](https://github.com/sgl-project/sglang/pull/40461) LiteTopK H100 indexer 1.21–1.40×很诱人，却只接 `dsa_indexer.py::_get_topk_ragged` 且 scorer 使用 SM90 WGMMA；本模型实际 `dsa_indexer_kpool.py`，硬件亦为 SM80，必须重新写 KPool+SM80 算法，不能直接移植或引用其倍率。
- [#41400](https://github.com/sgl-project/sglang/pull/41400) 的 FlashInfer KDA checkpoint prefill 明确只支持 SM100/SM103；[#34299](https://github.com/sgl-project/sglang/pull/34299) 的 CAKE 大收益也来自 GB300/SM100 kernel。[#39816](https://github.com/sgl-project/sglang/pull/39816) CuTe DSL AR fusion在 GB300 测的是 decode，不能认作 SM80 大块 prefill。
- 117 Humming W13/W2 只使完整 MoE 8k/16k 约 8.1/6.1% 快；这没有验证整段 prefill，更不能与 118 的约 8% 整块收益简单相加。mHC post+prenorm 融合 A100 原型 0.669→0.809ms 更慢；不以“融合”二字重立候选。

## 实际排序与验收

优先做 #39688 的 CPU 长度同步最小修复及 #39695 的 planner mirror（可快速抓到明确 bug/重复同步），同时设计 #41105 的 **≤4k bucket** TP8 图内 KDA 原型；若两种最小修复整块收益不足噪声，就把人力集中在 #41105。现有 8/16k block 必须分开报告；只捕到 4k 时，#41105 的收益只覆盖回放命中的尾块，不能以 4k 数字代替整段。所有候选按同一请求 ID 的 `chain_start` 超标增减与 `tpot_p95≤100ms` 门决定服务采纳；任何容量变化都报 KV/KDA 槽和 graph reserved，任何能力结论以真实权重 TP8 验证。

2026-09-28 后续原型：新 worktree `prefill-kernels-0928` 的 `engine/sglang/srt/layers/attention/dsa/kpool_plan.py` 新增显式 `req_pool_indices_cpu` 参数；不传时照旧读取设备 slot，且 `repeat_interleave` 保持原来的两个位置参数；仅传入普通 extend 的 CPU int64[B] mirror 时从 host 建压缩计划，并给 paged `repeat_interleave` 指定 CPU 已知输出长度。`scripts/analysis/check_kpool_req_mirror.py` 用标准库执行生产 planner 函数，比较普通/CP 局部行计划完全相等、CPU mirror 路径零设备 slot 读取、非法 mirror 拒绝、默认调用无新增 kwarg；`py_compile`、`git diff --check` 通过。当前公共 `ForwardBatch` 与 `DSABackend` 尚未接 flag，运行服务仍沿旧路径；root 负责公共文件接线与 TP8 验证。这是 CPU 路径合同测试，不是 GPU/整块性能数据。

2026-09-28 收束：考虑到此 mirror 每个 forward 只移除一次 planner 同步、目前没有整块证据，root 决定暂不接公共元数据。研究版已保存为 `build/scratch/kernel-next-0928/kpool-cpu-mirror-prototype.patch` 和同目录 `check_kpool_req_mirror.py`；工作树里的 `kpool_plan.py` 与探针已恢复/移除，不留未接线的生产接口。将来若复试，先将 patch 应用到对应 worktree，再把归档探针复制到该 worktree 的 `scripts/analysis/` 运行。
