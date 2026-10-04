# 最后三小时的 sparse decode 工程审查

2026-09-30 06:25 UTC，Codex。状态：只读源码和已有原始 trace 的审查；没有 GPU 新测、生产改动、queue/harness 改动或候选采用。未产生新的 chain 成绩。当前候选只支持局部预算判断。

结论：没有发现适合今晚直接替换 BF16 attention 的外部 drop-in。最小的新切口是消除 KPool selected index 的再次 padding，真实 profile 支持约 **42.20 µs/完整模型步**的 GPU 工作预算；它不解决约 5.9 ms 的正式 TPOT 差距。逐行有效长度可减少短行/补齐行的 attention 工作，但本次 profile 的短行是 **0/32**，不能从它推断已有净收益。已有 metadata fusion 的预算是每 forward 一次约 **0.31 ms**，不可乘 11 层或 8 卡。当前应优先让已决定的干扰臂开跑；以下工程点可作为中期候选，不阻塞。

## 身份与范围

- clean worktree：`build/worktrees/final-execution-0930`，HEAD `ca5d646c252688177480c7e67cec0a901e4b7069`；冻结 47266 baseline `bf6b66faf3ffbe10e593e858d3bbedd1c48cefb2`。
- 工作树还有未提交的 186 初始化日志 7 行；未改变这里的 kernel/数学。本文 backend 行号按当前工作树，相关后半段行号比 ca5d 多 7 行。ca5d backend SHA256 `5b28e712a31c7aa9639f0bd71be4fee0752905dd71f78ca01f9b55536c8e17b7`。
- 当前模型：A100 SM80，TP8 attention 本地 H8、D512，BF16 attention KV，KPool4/page64/topk2048；indexer 另用 D128 FP8 存储。两者不能混为 KV FP8。
- 当前 [question.md](../../question.md) 与 [profile03](pod/decode-profile/receipt.json)：32 个相同 8187-token prompt，五个孤立 decode 步，没有 mixed prefill、真实异长历史、池压力或 chain 分布。稳态 GPU 步约 18.74 ms；这是 profiler 下的诊断，不能与正式逐请求 TPOT 直接相减。
- 用户指定 [AppMana/forks-flash-mla-int](https://github.com/AppMana/forks-flash-mla-int) 已用网页与 git ls-remote 核查：默认 `appmana/vllm-ampere` HEAD 仍 `0d9ff279bfbc91104efd4c567d62f798144de47b`。现有独立 clone `build/external-review-0930/flashmla-ampere` 正好是此版本；直接只读复用，没有新增 clone 到生产路径。

## 实际链路及已经做过的工作

`dsa_backend.init_forward_metadata_out_graph/_apply_cuda_graph_metadata` 每完整 forward 准备一次 metadata；11 个 DSA 层各自 producer 更新 index cache、SM80 indexer 生成 logits、native KPool fused TopK 选择并将 pooled groups 展开/映射物理 token、`forward_decode` 将完整 BF16 Q 交给 `_forward_tilelang`，后者补 index 到 64 列倍数，再交给 `tilelang_sparse_fwd/sparse_attention_fwd_kernel_v1`。

- metadata 的 `dsa_cache_seqlens_int32` 已是 `min((seq//4)*4,2048)+seq%4`：`kernels/ops/attention/dsa_metadata.py:61–76`。不需要为逐行有效长度再新增 producer extent Tensor 或状态槽。
- 176 已处理 indexer static launch/context 不匹配；不将 page loop 重新计为新发现。181 已用完整连续 BF16 query view 去掉重复 CUDA Q concat；不重复计 Q copy。
- PAGED mapping 已在 TopK producer 融合：`kernels/jit/csrc/dsa/kpool_topk_transform.cuh:249–283`；无需再实现所谓 FlashMLA index gather fusion。
- TileLang 已共享 KV、使用 BF16 MMA、两阶段 pipeline；其 H8 至少 padded H16，当前一 CTA/row。单步 grid32 对 A100108SM 确有并行度不足的形状，但不能将 idle SM 比例直接当可兑现加速比例。
- 185 删除死 O_shared 已有探针与 eager 负优化疑点，本次不再包装成新候选。mHC、MoE up 与 KDA 的历史否决也不重复。
- CPU 发起已被 GPU 覆盖；本 profile 无 CPU starvation 证据。通信稳态 exclusive 约 2.08 ms/rank，但 sparse kernel 没有新的 collective；不能由 CPU annotation 或多卡重叠 duration 相加推导通信收益。

## 新切口 1：让索引 producer/consumer 对齐物理 padding，移除每层 fill + cat

源码事实：`kernels/ops/moe/kpool_topk_transform.py:56–57` 分配 2051 列（2048 + 3），`dsa_backend.py:4318–4329` 每层重新分配 61 列 -1 并 cat 成 2112。于是已经物理化的 selected indices 又完整写读一次。

原始证据：重新读 rank0 原 trace，找到 55 对连续的 `FillFunctor<int>` → `CatArrayBatchedCopy_alignedK_contig<OpaqueType<4u>>` → `main_kernel`，即 11 层 × 5 步。55 次 int32 cat 共 128.990 µs；稳态四步 fill+cat 均值 **42.19975 µs/step**，其中 cat 25.832 µs。配对在同一 GPU stream 串行，没有将重叠 duration 相加。source callsite 由 dtype、邻接序列与源码推断，trace 没有 shapes/stacks，不称直接 Python correlation。完整列表和 SHA 见 [sparse-padding-profile-recount.json](sparse-padding-profile-recount.json)。

最小实现有两种选择，均应独立默认 off：

1. producer 直接输出物理 2112 列，前 2051 语义完全相同、后 61 为 -1，consumer 不再 pad。**不能仅把 Python out_cols 改到 2112**：C++ `kpool_topk_transform.cuh:340–349` 现在由 `out_cols-tail_cols` 推导 token_topk，会改变/拒绝 2048 budget。必须明确区分编译 module 的 `kGroupTopK*pool_size=2048` 与物理输出列数，再核 indexer reuse、非 TileLang 和 fallback 宽度合同。
2. 让 BF16 attention 接受逻辑 2051 列，在最后 tile 对 index 地址进行真正 predicated load 并生成等价 -1 lane。这样避免改 TopK 的 ABI/输出宽度；需保留 33 tile 和原 MMA/softmax 顺序。当前 v1 `topk % block_I == 0` 的 assert 正是在保护越界 index，所以不能只删 assert。

预算：profile 下最多省约 0.042 ms GPU 工作，约 0.23% 的孤立 18.74 ms 步；producer 还要写 padding，净益可能更小。B32 index 流量按单层算：旧 cat 读 262528B selected + 7808B pad、写 270336B padded；这是数据量计算，不是带宽收益实测。不能再加上 181 已省的 BF16 Q cat。

最低验证：原 producer 与候选前 2051 全 tensor/顺序对照，并记录 native baseline-self tie/order 自发变化；追加 lane 必须每次 -1。B1/24/32/48、短/长/全空、seq%4、任意分页、req 变更、graph 缩空再 grow、poison padding、selected reuse。完整 TopK→attention boundary 输出/LSE、graph 与 eager 中位数/p95、临时/graph pool 显存；任何输入退化则保留原路。没有新测，不称今晚已赚 42 µs。

## 新切口 2：BF16 TileLang 使用已有 GPU selected 长度，避免空尾 tile

当前 v1：`tilelang_kernel.py:314` 固定 `NI=ceil(topk/64)=33`，`:375` 无条件遍历所有 tile；每 tile 重新读 index，gather KV，并执行 QK/softmax/PV。已有的 `metadata.dsa_cache_seqlens_int32` 没传到这个 BF16 kernel。外部函数 [sparse_mla_decode_mma_kernel](https://github.com/AppMana/forks-flash-mla-int/blob/0d9ff279bfbc91104efd4c567d62f798144de47b/csrc/flash_sparse_mla_decode_sm80.cu#L490) 直接从 per-row lens 算 tile 范围；可以借鉴这种 producer/consumer 合同，保持本地 BF16 数据与 selected 次序。

**本 profile 可省范围很小。** 32 行 prompt 都已 8187，故 selected history 都是满 2048，`<2048` 的行数 **0/32**。raw seq%4 没保存在 trace/receipt，不能报每步具体多少行无 tail；若 `%4==0`，可少最后一块全 -1 tile；否则 1–3 个 live tail token 仍需第 33 tile。对当前 profile 只给最多 1/33 tile 的工作上界，约 1.395 ms attention 分类成本 × 1/33 ≈ 0.042 ms/完整步；不是观测收益。不能假设 graph48 有 16 个 padded row：本 trace 实際 grid32，没有异长/空行证据。

实现风险：`T.Pipelined` 的循环界将从编译期 NI 变成 device-loaded 值；pipeline prologue/drain 与 zero-tile 分支要完整编译/资格，不能只改 Python range。长度是有效前缀上界，内含负页 holes 仍必须维持现有 mask；prefix 上界不要改变选中顺序或 split partition。空行旧 v1 是 `0/0` output、log2(0) LSE；不能顺手改成零输出而称逐位合同未变。应维持旧 NaN/Inf 行为，或另行证明 padded 输出不会参与后续状态/通信。停止尾部全 mask tile 对真实行通常是 no-op，但 signed zero、NaN、极值与 FP32 归约仍须实测。

最低验证：复用 metadata fusion 的实际 stage probe，在两臂只差 optional lengths 输入/有界循环；长度 0/1/3/4/17/64/65/2048/2049/2051、mixed/padded、negative holes、任意物理分页、poison seqlength 缩空/grow、Q/req 更替。完整 output/LSE 和 baseline-self 对照；CPU 无 item/sync、graph 静态地址；完整边界正负收益，且先核真实混合负载的短行占比。现在没有支持短行收益的 runtime 数据。

## 外部 SM80 思路的适配边界

外部 [selection_dequant kernel](https://github.com/AppMana/forks-flash-mla-int/blob/0d9ff279bfbc91104efd4c567d62f798144de47b/csrc/flash_sparse_mla_decode_sm80.cu#L122) 把选中 FP8/INT8 cache 解包为 BF16 scratch，供多个 head block 复用，再 double-buffer prefetch。当前本地 H8 只有一个 head block，BF16 本身不需要 dequant；照搬预pass会在 B32×2051×512×2 上新增约 **64.09 MiB** selected scratch 与一次写/读，没有复用证据。外部 FP8 448 + BF16 RoPE64 + UE8M0 scale 的 cache 布局与本地纯 BF16D512不同；INT8 也会改容量/数值输入。保持 BF16 MMA 不等于保持原 BF16 KV 数学输入，今晚不建议重做 FP8/INT8 cache。

外部被选中 lens 内的无效 slot 在预pass变零 KV 并可能以 score0 参与 softmax，本地负 index 被 mask 为 -inf；不能根据 README 的“masked”直接判语义相同。外部 return API 也没有本地 DCP 所需的 LSE 合同。

外部 split-KV 用更多 CTA 隐藏 gather latency，是真正值得借鉴的方向，但本地 [118 Triton](../../build/worktrees/final-execution-0930/engine/sglang/srt/layers/attention/dsa/sparse_attention_triton.py) 已有 `_num_splits`、resident-wave 选择、partial/merge 与单 split 不分配 workspace。该路径还每 split/head block 重复 last_valid 扫描 K_POW2=4096，可用现有长度避免，但当前 decode 并未用它，所以不是当前调用链的重复 work。要改善并行度先资格现有路径或有限 tile specialization，不新移植整个 FlashMLA CUDA 核。

split 会改变 softmax 分区/归约次序、增加 partial workspace 与 combine launch；external CUDA QK 还拆成两个256维半部再求和，不能称 local原FP32归约逐位相同。`flash_api.cpp:382–403` 每次仍分配 oaccum/mlse/counter/sel_kv，source本身不是更好的 graph lifetime 证据。当前三小时内没有证明它优于现有候选的完整 stage 数据。

## 本轮决策

先交付报告，不开始新 kernel/长 GPU 测试。若后续要选一个小工程实测，优先 **padding copy 消除**，因为真实 trace 有 42 µs 的可消除工作证据，合同比 split-KV 小；收益仍很小。**动态长度循环**先需要实际短/补齐行比例，否则当前 warm8k profile 几乎不受益。已有 fusion 可作为下一候选按当前已闭合开发资格进入 TP8，但单次 <0.35 ms 预算不应阻塞服务干扰臂，不承诺 TPOT/chain。
