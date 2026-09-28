# SM80 执行层审计（2026-09-28）

基线是独立候选 worktree `mhc-moe-sm80-0928` 的 `5883845b3fa2d76bcbbacd82481f684320be4d78`；该 worktree 未改。后续经 root 授权，KDA 原型写在 `prefill-kernels-0928` 的 `kda_backend.py`、`tests/test_kda_prefill_cpu_length.py`，局部图研究脚本为 `scripts/analysis/bench_kda_graph_sm80.py`。此轮不改调度政策，不联系 Fable。

先读了主仓库当前 `research/claude/base/04-model-kernels.md`、`R9_upstream_since_base.md`、`notes/knowledge.md`，再核对实际源码、117/118/171 说明及 L081p 既有 ledger。旧 L081p 仍是 Marlin MoE，且 target/draft span 混合平均问题已有修正，不能拿旧比例当当前上限。此文只保留两个有明确调用证据的机会，不为凑三个而重复已做的 tile/epilogue 扫描。

## 1. KDA 每层读取 GPU 标量；CPU 长度和更大的图捕获缺口

**确实走真实 SM80 路径。** GLM 配置有 34 个 KDA 层。`glm5_next.py:617-628` 调用 `RadixLinearAttention`，再由 hybrid backend 派给 `KDAAttnBackend.forward_extend`。`kda_backend.py` 基线第 739-740 行在每层做 `logical_num_tokens = int(query_start_loc[-1])`。`hybrid_linear_attn_backend.py:247-254` 明确在 `device=self.device` 创建 `query_start_loc`，末值来自 GPU `extend_start_loc[-1] + extend_seq_lens[-1]`；这不是 CPU tensor 的 harmless `int`。

普通 eager prefill 会执行这处 D2H 标量读取/同步，34 层重复。171 只改前面的投影；117/118 分别改 MoE/DSA attention；三个机制都不删除它。此处不是新发现的上游概念，但当前源码仍未应用：[SGLang #39688](https://github.com/sgl-project/sglang/pull/39688) 的 API diff 已核对，head `8d0fc7560575` 把 CPU logical length 缓存在 forward metadata。该 PR 同时含已有 171 投影、beta 融合及其他 tracking 工作，不能把整 PR 的加速归给这一行。

**同值来源及边界。**

- 普通主路径 `ForwardBatch.init_new`（`forward_batch_info.py:901-925`）从同一 `extend_seq_lens_cpu` 列表 H2D，再通过 `compute_position` 生成前缀和。因此 `sum(extend_seq_lens_cpu)` 等于 GPU query ends 的末值。
- **不能换成 `extend_num_tokens`**：DP 对齐在 `forward_batch_info.py:1491` 可把它改成物理 padding 行数。CPU lengths 保留逻辑数；补空请求时只附加 0（`:1596`）。
- 普通 input scatter 在 `communicator_mhc.py:490-492` 给没有 QKV hook 的 KDA all-gather 回完整 token 维，故 CPU 逻辑数仍正确；nondivisible physical padding 保留现有 trim/zero-output 行为。
- MIXED 也是 extend 元数据，decode 请求的增量长度为 1；应取所有 CPU 长度之和，不是只求长请求。
- 上游明确拒绝 `tbo_parent_token_range != None` 的直接 CPU sum。当前 TBO child 元数据可能有 parent-relative offset；不能仅凭 child 总长猜末值。
- CP 未在本原型中验证；初始化时 `get_parallel().attn_cp_size != 1` 或已有 `attn_cp_metadata` 均保留原路径。不能只查后者：`eager_runner.py` 先初始化 backend metadata，`GLM5Next.forward` 后设置 CP metadata；独立审查抓到了这个时序问题，已补 `metadata=None, cp_size=2` 的 CPU 测试。target verify 早返回自己的路径，decode/idle/draft/DLLM/split-prefill、GPU-only 缺失 CPU 列表均不启用原型。

**已实现的最小原型。** `SGLANG_AX_KDA_PREFILL_CPU_LENGTH=1` 默认关闭。KDA 自己的 `init_forward_metadata` 为合格 EXTEND/MIXED 缓存 CPU sum，未改共享 `ForwardMetadata` schema。校验 CPU list/tuple、batch 长度、query ends shape、非负 Python ints、无 TBO/CP；forward 另拒绝大于物理输入的缓存长度并回退原 GPU `int`。默认关闭沿用原标量读取。CPU focused test 已通过 4 组测试，含 70 个 normal/mixed 随机和 padding 组合、所有 mode 回退、TBO/CP/gpu-only、实际 metadata hook 的 AST 测试；py_compile 与 diff check 通过。

验证入口：

```sh
python3 tests/test_kda_prefill_cpu_length.py
PYTHONPATH=engine python3 tests/test_kda_prefill_cpu_length.py --gpu --rows 8192 16384
```

GPU 短测 `kda-cpu-length-a`（A100 80GB，GPU1，exit 0）只执行真实 KDA core、conv 与 SSM 更新，不模拟 TP 通信或模型投影。四例：8K、16K、ragged `[273,65]` + 6 physical pad、MIXED `[127,1,1]` + 7 pad，输出 BF16 bits、conv BF16 bits、SSM FP32 bits 全相同，padding 注入 NaN 后有效输出仍有限。热 FLA 缓存后 `aten::_local_scalar_dense` 四例均 1→0。7 轮随机交错、无 profiler 的 core CUDA-event median：8K 1.397760→1.301728 ms，16K 1.913824→1.872544 ms；host enqueue median 1.125831→1.023404、1.144197→1.057297 ms。样本少且时间包含 CPU enqueue 空隙，不能外推整模型收益。

原始 JSONL、stderr、exit、起止时间、launch JSON、加载源码与 SHA manifest 已归档到新 worktree 的 `evidence/prefill-kernels-0928/kda-cpu-length-a.*`。GPU 加载 backend SHA `901a408f`、probe `428236cc`，之后只补 CP 排除 guard，最终 backend `557315df`、CPU test `dd472def`，CPU 4 tests 再通过；此 guard 不改变已测 CP=1 的算术路径。GPU1 已释放。后续经 root 授权在 GPU0 独立研究局部 KDA graph，结果另记，不能合并成这里的 CPU 长度补丁收益。

**这还不是 KDA graph capture 修复。** `radix_linear_attention.py:244` 使用 `eager_on_graph(True, capture_stub=...)`，170 在 replay 仍把 KDA 本体放在 eager break，且 `:180-181` 可能把整个返回值 copy 到 graph 输出。`fla/index.py:17-27` 的 `prepare_chunk_indices` 首次还会对 GPU lengths `.tolist()`，其 tensor-identity cache 只减少后续重复调用，不能证明随请求变化的 CUDA graph 元数据正确。上游 #41105 的 graph 方案需要一起检查 shape/索引计划、tracking 快照、capture 占池和多次 fresh-input replay；本补丁只隔离每层 scalar sync。可区分验证是：先看本原型 scalar 计数和完整 prefill GPU 空隙，再分别做 KDA graph native break/captured kernel 对照，不把两种收益相加。

**局部图的最小接口和研究边界。** 已逐项读 [上游 #41105](https://github.com/sgl-project/sglang/pull/41105) diff（head `0e54824a0178`）。一般 bucket 复用至少需要显式传递、每次刷新 `chunk_indices/chunk_offsets`，padded chunk 的三个 early-return guards，以及 fixed conv block table；当前 causal conv 只支持从 CPU `max(seq_lens_cpu)` 定 launch grid。不能通过 warm tensor cache 后只改 cu_seqlens 来宣称支持动态 batch。

本轮先用独立脚本固定 `(logical M, physical M, sequence count, total ceil(length/64), conv bound)`，仅在 capture 的 `try/finally` 中绑定显式 GPU 表并恢复所有函数；不改任何生产 kernel。它刷新 sequence boundaries、state slots、prefix flags、输入、140 双快照 offsets/destinations；不支持的 topology 在 CPU 拒绝。每个 M 测单序列和三个变边界 ragged replay，ragged 保留 7 行 physical padding；整体输出/conv/SSM 和未触及 pool slot 都须 raw bits 一致。计时分 native eager、CPU-length eager、纯 replay、含 CPU 表构造 + pinned H2D + QKV/gate/beta 全量 D2D 的局部 graph。显存按当前单个 graph 的 allocator 私池、capture delta 和显式持久输入/状态缓冲分别记账，每个 case 结束销毁；不能把四图同时常驻的开销算成一图，也不能把一个图的内存直接当完整 34 层池需求。实测结果如下。

独立 CPU 审查还确认：本地 `prepare_chunk_offsets` 的 int32 输入经默认 `cumsum` 后为 int64，而上游式统一 packed 表用 int32。A 探针因此有 offset-pointer specialization 差异，虽值域安全、仍要求所有 raw-bit 门通过，不能把全部时间差纯归 launch overhead；如收益成立，保留单独 int64 offset buffer 的复测可以消除此混杂。

`kda-local-graph-a` 已 exit 0，4 图 × 3 次 fresh replay 全部通过，含变化的边界、prefix/slots、140 快照、NaN padding 和未触槽位的整池 raw bits。9 轮随机顺序、inner=4 的 CUDA-event median 如下（ms）；A 的 int32 offset 混杂保留，不当生产结论。

| logical M / 序列 / 快照 | native eager | CPU-length eager | graph replay | graph + 全 staging |
|---|---:|---:|---:|---:|
| 8192 / 1 / 无 | 1.194816 | 0.981408 | 0.890256 | 1.001384 |
| 8192 / 3 / 140 双快照 | 1.331304 | 1.208504 | 0.905856 | 1.041512 |
| 16384 / 1 / 无 | 2.030344 | 1.943272 | 1.881704 | 2.090408 |
| 16384 / 3 / 140 双快照 | 1.727632 | 1.599968 | 1.529704 | 1.749480 |

单图私池实际 reserved 依次 246/248/502/504 MiB；另显式持久输入与小状态池约 67.2/70.4/131.4/134.5 MiB。私池 live allocated 仅输出的 16/32 MiB，但释放给同池复用的临时块仍被 graph pool 保留，不能只报这个较小数。raw/loaded sources/launch/exit/manifest 和摘要均在 `evidence/prefill-kernels-0928/kda-local-graph-a.*`。局部图明显减 host enqueue，但 16K 全量 input staging 吞掉 GPU 收益；不做生产图接线。经 root 授权只复测一次保留 native int64 offsets 的 B，并利用现有短 profiler 输出 GPU kernel 时间分账，后续转查 core 的实际计算/内存热点。

**B 已闭合，offset 类型混杂已消除。** `kda-local-graph-b` exit 0，12 次 fresh replay/完整 state pool raw bits 全过，metadata 用独立 int64 offset buffer（第二次 pinned H2D 已计入 staging）。B 的四行 `[native, CPU-length, replay, 全 staging]` ms：8K 单序列 `[1.216400, 0.992384, 0.888352, 1.008232]`，8K 三序列 `[1.312584, 1.511040, 0.800296, 0.945648]`，16K 单序列 `[1.859824, 1.768984, 1.714128, 1.934752]`，16K 三序列 `[1.728960, 1.601088, 1.527536, 1.753616]`。8K 三序列的 CPU-length eager 比 native 反而慢，说明 host 时间易受负载影响，不把 A/B 差异归给 offset dtype，也不拿某次 native→CPU 变化承诺收益。图 core 对照后续只用同进程配对图。16K 的 full-staging 仍负，停止扩大 graph 实现。

**真实 GPU 热点。** B 的 warmed short profiler（不是 wall time）在 16K 单序列给出 GPU event 总和 1726.1 µs：`chunk_gated_delta_rule_fwd_kernel_h_blockdim64` 546.3 µs / 31.6%；三次 BF16 strided copy 219.0 µs / 12.7%；inter-solve 162.8、conv 162.6、recompute 158.0、output 157.9、intra 154.3 µs；两次 l2norm 合计 74.0 µs。三序列+140 时 recurrence 311.5 µs / 20.3%，三次 copy 219.0 µs / 14.3%。它们都是真实 SM80 Triton prefill core，117/118/171 没覆盖。

- `chunk_delta_h.py` 默认 BV32、warps4、stages2；grid 为 `(ceil(V/BV), N*H)`。真实 TP8 形状 H8/V128，N1 只有 32 CTAs，而每 CTA 沿 NT=128/256 顺序循环。V 切片独立，BV16 可给 64 CTAs；BV 定点研究用原 Autotuner 的 `.fn` 固定 launch 参数，避免原位 state autotune 污染，并用同进程 BV32 完整 core 图配对计时、真实 eager32 作 output/conv/SSM/snapshot raw-bit 参考。尚不写生产 kernel。
- `kda.py` 把 packed conv 的 Q/K/V 视图逐个 `.contiguous()`，再做两次 q/k l2norm；16K 三个 BF16 张量各 32 MiB，copy 合计 96 MiB 数据（读写约 192 MiB）。这不是核心算法必须的 FP32 中间值。root 独立负责研究一次 planar 输出准备/规范化是否能删重复搬运，必须保留 conv→BF16 的舍入与 l2norm 次序；我不改其文件。
- recurrence 内 FP32 live state 是数值契约，持久池也须支持 140 FP32 快照；每 chunk 的 `h` 中间实际是 BF16（`k.new_empty`）。不能把它误报为“多余 FP32 h 池”并降精度。删 `h` 往返需要把输出 dot 融进串行 recurrence，可能反而加重低并行，不优先实施。

`kda-state-bv16-a` 已 exit 0：4 种形状 × 3 次 fresh replay，BV16 candidate 与真实 BV32 eager 的 output/conv/SSM/snapshot/未触槽位均 raw-bit 相同，同进程 native32 图自身也逐位过门。配对完整 core 图 ms：8K 单序列 0.889720→0.845912（4.92%）；8K 三序列+140 0.797992→0.774080（3.00%）；16K 单序列 1.714008→1.627560（5.04%）；16K 三序列+140 1.526952→1.479848（3.08%）。profiler 中 16K 单序列 recurrence 546.0→457.7 µs，三序列 311.6→258.5 µs；其余 core 阶段基本相同。归因明确但不是大幅整模型收益。GPU1 的 layout preparation 研究由 root 独立负责，不合并计算收益。

最后的 `kda-state-bv8-a` 也 exit 0、12 次 fresh 整池 raw bits 全过，但配对完整 core 全慢：8K 单序列 0.888448→0.910840、8K 三序列 0.798024→0.820784、16K 单序列 1.715336→1.748376、16K 三序列 1.528320→1.563968 ms（慢约 1.9%–2.9%）。BV8 在当前 Triton BF16 min-dot 规则合法，但更多 CTA 没带来净收益；选择 BV16，停止扫点，GPU0 已释放。没有修改 production recurrence。若后续生产化，只考虑实际 KDA/SM80/BF16/H8/K=V128 与已验证 large-prefill family 的受控分支，不全局更改共享 GDN 环境变量；与 root 的 prepare 优化必须重新过组合门，收益不能相加。

收尾可复现入口：`scripts/analysis/bench_kda_graph_sm80.py --gpu --rows 8192 16384 --rounds 9 --inner 4 --state-bv 16`。所有新 GPU 任务均有完整 raw JSONL、stderr、exit、起止时间、launch JSON、加载源文件与 SHA manifest，位于 `prefill-kernels-0928/evidence/prefill-kernels-0928/kda-*.{jsonl,summary.json,...}`；已冻结的 `mhc-moe-sm80-0928` worktree 保持干净。所有这些结果都是 TP8 每卡形状的开发机完整 KDA core 诊断，不含投影、TP 通信或全模型/会话链证据。

## 2. 114 查询行切片发生太晚，query 投影/旋转/量化和 gate 仍 TP 重复

**确实走真实 SM80 路径。** 模型是 `index_kpool=4`、32 index heads×128、q_lora_rank=1536、hidden=4096。`IndexerKPool._forward_cuda_impl` 在 `dsa_indexer_kpool.py:1742` 先 `_get_q_k_bf16`，后 `act_quant(query)` 和 `_get_logits_head_gate(x,q_scale)`，最后才到 `_get_topk_ragged_kpool_plan`。`wq_b` 是 ReplicatedLinear（`:159`）；`_get_q_k_bf16` 的普通路径（`:677`）处理完整 `q_lora`，`:697` 再对完整 query 做 `rotate_activation`。head gate 在 `:199-204` 对完整 `x.float()` 作 FP32 projection。

114 在 `:1020-1031` 才选 `[r0,r1)`，`:1049-1053` 只把这一段 q/weights 送入 MQA logits。也就是说已经只计算 1/TP 的 logits/top-k，但上游 query 构造仍全量重复。输入 scatter 的 DSA pre-gather 也不消除它（`communicator_mhc.py:490-492`）。117/118/171 均不覆盖；此提案不重复计算 114 已省掉的 O(M×history) 工作。

**具体修改假设。** 对已有 114 合格的普通 kpool prefill，将同一 query row range 提前到 `wq_b`、query rotation、`act_quant` 和 head-gate 的输入处；K、compress_gate、kpool cache write 和其元数据仍保持当前全量契约，最终 top-k all-gather 保持原布局。为 top-k helper 明确传 full row count 和 local query offset，避免把 local q shape 误当完整 plan 行数；DCP、短序列 skip-logits、decode、verify、graph bridge 先保留 fallback。

**可量化而非收益承诺。** 8K 时完整 query BF16 输出为 64 MiB/卡，`x.float()` 临时为 128 MiB/卡，16K 翻倍。TP8 理论上可删掉这几项的 7/8 行工作和相应临时空间；不能称 indexer 或整模型八倍。尚无该调用链独立成本数据，可能被 GEMM/collective 隐藏，收益需从实际 profiler 的 query projection、Hadamard、quant、gate 四段取账。

**可区分验证。** 以当前 114 late-shard 为 A，只有 query/gate early-shard 为 B，固定真实 q_lora/x/positions/plan 和相同 K cache。依次比较投影、旋转/FP8量化、MQA scores、top-k indices/weights、完整 sparse attention 输出；M 改变可能换 cuBLAS kernel，进而改变 FP8/top-k 近 tie，不能因为 query 行独立就声称逐位相等。先在一个 GPU 模拟各 TP rank 的切片，覆盖 8191/8192/16384/16385、rows<tp、空尾 rank、首块/长前缀/多请求，最后真 TP8 核对 all-gather 的顺序和所有 rank 的结果。若 FP8 分数或 top-k 改变，必须解释是否来自投影舍入并过独立数值/真实权重门，不能只看 top-k overlap。

## 未作为新候选的内容

- **mHC prenorm split-K**：root 已独立实现/测量。基线 `mhc.py:1035` 的确只有 M≤2048 用 split-K，大形状用固定 unsplit；`MHCState.attn_to_mlp` 实际仍走 hc_post→hc_pre。split-K 改 FP32 求和分组，需 oracle，不承诺 bitexact。本审计不占该 GPU 工作。
- **FP8 dense**：SM80 `Fp8LinearMethod.apply` 仍无条件进入 Marlin（`fp8.py:979`），117 不替换 dense；Marlin M tile 最大 64，N≤4096 时一 launch 最多吃 8192 行（`gptq_marlin.cuh:611-625`）。这支持筛选大 prefill Humming/BF16 GEMM，但现有账上 dense Marlin 只是一小项、还要计反量化和显存，当前证据不足以排在上面两项之前，不能笼统宣称“FP8 在 A100 回退所以很慢”。
- **KDA 171/DSA 118**：已有投影融合和 attention index padding 消除不再列新收益；171 的旧 056 回放还有容量差与混合结果，不能直接重开开关并叫新优化。

目前没有新增 chain 或服务吞吐证据。一个同步修复即使单独收益小，也能作为后续 KDA capture 的可验证前置；是否继续由实际 scalar count、whole-prefill GPU 空隙和数值检查决定。
