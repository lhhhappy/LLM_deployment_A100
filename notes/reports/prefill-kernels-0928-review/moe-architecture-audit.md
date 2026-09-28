# SM80 MoE 执行组织审计：三个结构候选

> 2026-09-28 归档快照：技术正文保留原审查范围，仅将本地链接改为归档相对路径；本轮最终状态以 [R36](../../../research/codex/R36_prefill_kernel_options_0928.md) 为准。原件与归档件 SHA-256 见 [sources.json](sources.json)。

2026-09-28。初始源码审计固定 SGLang `mhc-moe-sm80-0928@5883845b` 和本地 `humming_kernels-0.1.12`；后续只在 `prefill-kernels-0928` 新建隔离研究脚本和证据，未修改生产文件。**chain 尚未验证。** 原完整 MoE 的开发机比较点为 up+down、reduce off：8192 行 3.199162 ms、16384 行 6.499830 ms。候选1已完成独立私有 header 的 GPU 对照，结果见收尾节；其余候选及两个追加结构问题只做源码和容量算账。所有未标为实测的操作数、字节数和收益上限均为计算或假设。

本轮优先验证了**把 FP8 解码的固定指数修正从每个权重移到 group scale**：数值门通过，但完整 MoE 配对 graph 仅快2.846%/1.967%，因此关闭该方向，不生产化、不继续扫 shape。另两项是成对 gate/up epilogue 和跳过纯 padding 的 MMA 子块，均只有较低的源码上限。N分条立即归并及整 FFN 同CTA的追加算账也不支持大幅收益；主攻方向转到 KDA 的真实 profile，并未用理论上限替代实测进展。

## 可核对的当前成本

形状为每卡 H=4096、I=256、E=289、top-k=9，其中 288 个 routed experts 选 8 个，另一个 shared expert 对每个 token 都执行。BF16 activation；原 FP8 e4m3 权重；Humming 转换后的 BF16 group scales，K128/N0；FP32 GEMM 累加；SwiGLU clamp 10；路由缩放 2.5。

当前执行链：专家对齐 → indexed W13 → 独立 clamp/SiLU/multiply → indexed W2 → weighted combine。W13 读取原 token 行，但写出 token-slot 顺序；W2 再按同一 sorted IDs gather 激活，最后也 scatter 到 token-slot 顺序。[runner:498](../../../engine/sglang/srt/layers/moe/moe_runner/humming.py#L498)、[runner:560](../../../engine/sglang/srt/layers/moe/moe_runner/humming.py#L560)、[scheduler:303](../../../evidence/moe-scale-fold-0928/cache/humming_scale_fold_headers/1fec70021765fb32becf11d8b30773eab91e9ea0f14ec64247e90d8f75396102/native/humming/scheduler.cuh#L303)、[writer:103](../../../evidence/moe-scale-fold-0928/cache/humming_scale_fold_headers/1fec70021765fb32becf11d8b30773eab91e9ea0f14ec64247e90d8f75396102/native/humming/epilogue/gmem_writer.cuh#L103)。

| 形状账本，每次每卡 | M=8192 | M=16384 |
| --- | ---: | ---: |
| 有用 GEMM FLOPs，`6×9M×H×I`，FMA 算 2 | 463.856 GFLOP | 927.713 GFLOP |
| 输入 BF16 `[M,H]` | 64 MiB | 128 MiB |
| gate/up BF16 `[9M,512]` | 72 MiB | 144 MiB |
| 激活 BF16 `[9M,256]` | 36 MiB | 72 MiB |
| W2 output BF16 `[9M,4096]` | 576 MiB | 1152 MiB |
| W2 output 写入＋combine 读取 | 1152 MiB | 2304 MiB |
| combine 输入＋最终输出 | 640 MiB | 1280 MiB |

这些是逻辑字节数，不是 DRAM counter。W13 的 N=512 被分为两个 N256 CTA，逻辑 A 请求为 1152/2304 MiB；W2 当前 N128 对同一个激活行块发出 32 份逻辑 A 请求，同为 1152/2304 MiB。不能把这些请求都算成 HBM 流量：indexed scheduler 的 N 是内层连续维，邻近 CTA 会使用同一 A 行块，已有 L2 复用。`loader_a` 以 16-byte 向量加载、128-byte swizzle，并不是逐元素非合并读。[scheduler:193](../../../evidence/moe-scale-fold-0928/cache/humming_scale_fold_headers/1fec70021765fb32becf11d8b30773eab91e9ea0f14ec64247e90d8f75396102/native/humming/scheduler.cuh#L193)、[loader_a:88](../../../evidence/moe-scale-fold-0928/cache/humming_scale_fold_headers/1fec70021765fb32becf11d8b30773eab91e9ea0f14ec64247e90d8f75396102/native/humming/memory/g2s_loader/loader_a.cuh#L88)。

旧 `moe-layer-c` profile 的 5 次调用除以 5 后，W13 / W2 / activation / combine 为 8k：1.849 / 1.241 / 0.087 / 0.389 ms，16k：3.430 / 2.473 / 0.179 / 0.771 ms。专家对齐的两个 GPU kernel 合计约 0.066/0.137 ms。这些是**优化前的另一组 profile**，只定位热点，不能直接相加到当前 3.20/6.50 ms，也不能将父 custom-op 和子 kernel 重复计费。[原始 profile](../../../evidence/mhc-moe-sm80-0928/moe-layer-c.jsonl)、[当前测量边界](../moe-mhc-sm80-0928.md#L9)。

## 1. 将固定 FP8 指数修正前移到 scale

**源码事实。** Ampere 路径每个 warp 先把 packed FP8 bits 转成 BF16 格式的未校正数，再乘固定指数修正，最后乘 group scale，然后做 BF16 MMA。BF16/FP8 e4m3/BF16-scale 的 `kExpOffset.x = 128−8 = 120`、y=0。关键位置：

- [FP8 bits 转换:85](../../../evidence/moe-scale-fold-0928/cache/humming_scale_fold_headers/1fec70021765fb32becf11d8b30773eab91e9ea0f14ec64247e90d8f75396102/native/humming/datatype/dequant_single.cuh#L85)。
- [指数推导:8](../../../evidence/moe-scale-fold-0928/cache/humming_scale_fold_headers/1fec70021765fb32becf11d8b30773eab91e9ea0f14ec64247e90d8f75396102/native/humming/arith/exp_offset.cuh#L8)、[mainloop offset:74](../../../evidence/moe-scale-fold-0928/cache/humming_scale_fold_headers/1fec70021765fb32becf11d8b30773eab91e9ea0f14ec64247e90d8f75396102/native/humming/arith/exp_offset.cuh#L74)。
- [每个权重乘 `2^120`:149](../../../evidence/moe-scale-fold-0928/cache/humming_scale_fold_headers/1fec70021765fb32becf11d8b30773eab91e9ea0f14ec64247e90d8f75396102/native/humming/arith/mainloop_arith.cuh#L149)、[再乘 group scale:208](../../../evidence/moe-scale-fold-0928/cache/humming_scale_fold_headers/1fec70021765fb32becf11d8b30773eab91e9ea0f14ec64247e90d8f75396102/native/humming/arith/mainloop_arith.cuh#L208)。
- [每个 K16 段再次 transform B:47](../../../evidence/moe-scale-fold-0928/cache/humming_scale_fold_headers/1fec70021765fb32becf11d8b30773eab91e9ea0f14ec64247e90d8f75396102/native/humming/mma/wmma.cuh#L47)、[流水循环:125](../../../evidence/moe-scale-fold-0928/cache/humming_scale_fold_headers/1fec70021765fb32becf11d8b30773eab91e9ea0f14ec64247e90d8f75396102/native/humming/kernel/humming.cuh#L125)。

**具体改法。** 保留 FP8 和 scale 张量、所有 tile、raster、K 累加顺序、FP32 累加以及 BF16 输出舍入；只在已加载 scale 的寄存器中准备 `s′=s×2^120`，把两次逐权重乘法变为一次。仅限 BF16 A/BF16 scale/FP8 e4m3/group K128，scale 范围符合下述条件；其他组合走原算法。当前 `S2RMemoryLoaderBS.load_layout1` 每次覆盖 scale buffer，所以可以在每次 `transform_b` 的 j=0 处准备一次，不能在每个 j 重复乘。[scale load:91](../../../evidence/moe-scale-fold-0928/cache/humming_scale_fold_headers/1fec70021765fb32becf11d8b30773eab91e9ea0f14ec64247e90d8f75396102/native/humming/memory/s2r_loader/loader_bs.cuh#L91)、[scale preparation hook:110](../../../evidence/moe-scale-fold-0928/cache/humming_scale_fold_headers/1fec70021765fb32becf11d8b30773eab91e9ea0f14ec64247e90d8f75396102/native/humming/arith/mainloop_arith.cuh#L110)。

令 `r` 为解码后尚未乘指数修正的 BF16。原权重操作为 `RN_BF16(RN_BF16(r×2^120)×s)`。有限 FP8 值可以被 BF16 精确表示，所以内层修正没有舍入；若 `s×2^120` 也有限且精确，两条路径最终相乘的实数值相同，最终 BF16 RN-even 结果应相同。允许 `0≤s≤255` 的 BF16 scale；256 的折叠溢出。负 scale 可按绝对值推广，但本轮未以负 scale 作为目标契约。不能把这些条件换成“所有 scale 都安全”，也不能把最终 BF16 scale 从原 FP32 scale 重新生成，绕过现有 BF16 转换。

**CPU 证据。** 用整数 mantissa/exponent 实现精确 dyadic BF16 RN-even，穷举 254 个有限 FP8 pattern × 17280 个sign bit为0的 BF16 scale pattern（+0至+255，含subnormal），4,389,120 对无差异；第一个超界 scale256 得到 folded BF16 `0x7f80`。CPU 模型本身不证明 CUDA intrinsic 的 FTZ 行为；后续实际 SM80 指令穷举及 Humming 位比较也已通过，见收尾节。scale负零和两个FP8 NaN编码不在已证明域，不能将数值范围描述误解为所有同范围bit pattern均已覆盖。

**工作量。** BM128/WarpM64 的两个 M-warp 组重复读取和解码同一个 B tile；`S2RMemoryLoaderB` 的地址取决于 N-warp，没有 M-warp 索引。[loader_b:65](../../../evidence/moe-scale-fold-0928/cache/humming_scale_fold_headers/1fec70021765fb32becf11d8b30773eab91e9ea0f14ec64247e90d8f75396102/native/humming/memory/s2r_loader/loader_b.cuh#L65)。取完全均衡的 640/1280 个 M blocks，两个 GEMM 合计每次约 4.027/8.053 billion 次 scalar 权重解码，也就是 2.013/4.027 billion 次对应的 BF16-pair 指数修正乘法。当前 WarpN64 的每线程 K16 段有 16 对逐权重指数修正；将修正放到 4 对 scale 后，修正乘法数减少75%，连同不可省的 group 乘法，共从32对降至20对，源码层 BF16-pair 乘法数减少37.5%。**这不是 kernel 时间减少37.5%**：MMA、loads、其余整数解码和 epilogue 不变，且尚未确认编译后的 SASS 是否已做等价优化。

**风险与判据。** 先审两份 cubin 的相同配置和真实不同 hash，再检查真实 W13、同激活 W2 及完整 MoE 原始 BF16 位，之后做随机顺序的配对 graph timing。只有完整层比当前 up+down 明显改善才继续；无收益就结束此方向。这里没有可诚实换算成 ms 的上限，缺少 decoder 指令占比/stall counter。该候选的优势是覆盖全部 GEMM、几乎没有新增流量或显存，而不是已知的速度幅度。

## 2. 成对 gate/up 输出，融合 SwiGLU 并保留专家块布局

**具体改法。** 加载期只重排权重 N 维：把每个 N256 tile 组织成同一128个 channel 的 gate 与 up，而不是当前一个 tile 全 gate、另一个全 up。每个 CTA 在 epilogue 的 BF16 shared staging 中配对，执行 clamp、SiLU、BF16 rounding、乘 up、再次 BF16 rounding，直接写 `[padded routed rows,256]` 激活。W2 从这个专家对齐布局连续读 A，仍用原 sorted IDs 把输出写回 token-slot 顺序。无需先把 `[M,4096]` 扩成 `[9M,4096]`，也不改变模型 channel、route 或权重数值。

当前 gate/up 与 activation 分别分配 `[9M,512]`、`[9M,256]`，[runner:264](../../../engine/sglang/srt/layers/moe/moe_runner/humming.py#L264)。激活明确是先 clamp，再 FP32 SiLU，再转 BF16，再与 BF16 up 相乘、写 BF16，不能直接在 FP32 GEMM accumulator 上做 fused SiLU。[activation:1119](../../../engine/sglang/kernels/ops/moe/fused_moe_triton_kernels.py#L1119)。当前 W13 scatter 和 W2 gather 使用同一个 `wr_row_index` / `rd_row_index`，需要为读布局和写布局拆开契约，不能只翻 `gemm_type` 开关。[scheduler:303](../../../evidence/moe-scale-fold-0928/cache/humming_scale_fold_headers/1fec70021765fb32becf11d8b30773eab91e9ea0f14ec64247e90d8f75396102/native/humming/scheduler.cuh#L303)、[writer:103](../../../evidence/moe-scale-fold-0928/cache/humming_scale_fold_headers/1fec70021765fb32becf11d8b30773eab91e9ea0f14ec64247e90d8f75396102/native/humming/epilogue/gmem_writer.cuh#L103)。

**收益账本。** 删除 gate/up 的写入与读取，逻辑流量减少144/288 MiB，删除一次独立激活 kernel；旧 profile 中这项 kernel 为0.087/0.179 ms。新激活计算并非消失，而是占用 W13 epilogue 的执行资源，所以不能把这段时间完整计为净节省。排序激活多存 padding，uniform 模型约多4.5/8.1 MiB；W2 连续读可能减少地址开销，但当前已有 N-major L2 复用，不能报32倍 DRAM节省。

该方案仍然是两个 N256 CTA，所以**不能声称 W13 input gather 减半**。强行改为 N512、M128，单是 FP32 accumulators 就要65536个32-bit registers，等于一个 SM 的全部 register file，还未计 A/B 和指针；改小 M 又增加 B 重载/解码，因此不纳入“免费融合”收益。此项大概率是几个百分点量级的候选，不能独自回答用户要求的大幅改进。

**风险与判据。** gate/up 配对跨 warp 的 shared-memory 读写需明确 barrier；CUDA sigmoid/exp 的近似与 Triton 可不同，必须保留舍入点并查独立参考。padding 行不能变成有效输出，EP/映射不在首版范围。只在指数修正前移已成立且本项有可衡量增量时投入，不先做整条 FFN 的片内驻留重写。

## 3. 原网格内跳过全 padding 的 M16 MMA 子块

**具体改法。** 保留 BM128、全部对齐数组及 W13/W2 共用的专家块映射。在 `fetch_moe_index_block` 后，为每个 warp 的 M16 子块生成“至少一行有效”的 warp-uniform mask；`WMMA.run` 对全 invalid 子块跳过对应 MMA，完全没有真实行的 M-warp 也可跳过其纯寄存器解码，但所有线程仍参加原 CTA barriers。输出仍由原 mask 控制。不改 routing、不把不同专家塞进同一 MMA、不重复已有 tile sweep，也不加第二次排序。

当前 padding 被 A load 和 C store 屏蔽，但固定展开的 MMA 循环仍算完整 M。[对齐契约:55](../../../engine/sglang/srt/layers/moe/moe_runner/triton_utils/moe_align_block_size.py#L55)、[padding sentinel](../../../engine/sglang/kernels/aot/csrc/moe/moe_align_kernel.cu#L258)、[A mask:108](../../../evidence/moe-scale-fold-0928/cache/humming_scale_fold_headers/1fec70021765fb32becf11d8b30773eab91e9ea0f14ec64247e90d8f75396102/native/humming/memory/g2s_loader/loader_a.cuh#L108)、[无条件 MMA:79](../../../evidence/moe-scale-fold-0928/cache/humming_scale_fold_headers/1fec70021765fb32becf11d8b30773eab91e9ea0f14ec64247e90d8f75396102/native/humming/mma/wmma.cuh#L79)。M16 mask 必须由整个子块的有效性导出；若依赖 padding 只在尾部，要用实际 align 输出验证此不变量。

**假设计算。** 每 token 独立、均匀选8/288路由专家，故单专家计数边际分布 `Binomial(M,8/288)`；shared 固定M行。不是从真实 checkpoint 路由直方图测得。

| uniform 模型期望 | 8192 | 16384 |
| --- | ---: | ---: |
| 每个 routed expert 平均行数 | 227.556 | 455.111 |
| 有用 routed+shared 行 | 73728 | 147456 |
| BM128 后行数 | 82935.182 | 163963.889 |
| 完美消除 padding 占原 GEMM FLOPs 的上限 | 11.10% | 10.07% |
| 仅舍入到 M16 后等效计算行数 | 75888 | 149616 |
| 本候选最多跳过原 MMA 的比例 | 8.50% | 8.75% |

MMA 时间只是整个 MoE 的一部分：权重/scale loads、仍有真实行的 warp 的完整 B 解码、epilogue、combine 都不会按这个比例下降。shared 的 M 在这两个形状没有 padding，也无收益。真实偏斜路由可使余量更小或更大；不能套 uniform 数字。尾块分支检查还可能拖慢所有满块，因此最好将满块检测移到持久循环外层，满块走原直线 MMA，只有尾块走 mask；不要复制K累加顺序或增加新launch。

本项的严格算术上限已经低于约9%完整层，实际必低于它；它是后备候选，不足以支撑“MoE 能再省一大截”的承诺。

## 已检查但不作为第四个候选的方向

- **共享专家单独稠密化/常驻 BF16。** 实际 GLM5 使用 `DeepseekV2MoE`，[模型接线:111](../../../engine/sglang/srt/models/glm5_next.py#L111)；共享专家追加进 fused expert/top-k，[DeepseekV2:584](../../../engine/sglang/srt/models/deepseek_v2.py#L584)。它占有用 GEMM 算量1/9，却承受与稀疏专家一样的 gather 与重复解码。缓存其等价 BF16 权重需要每层6 MiB（例如42层则252 MiB，若仍保留原 FP8）；额外路由分支、两个dense GEMM及激活launch、合并顺序不能省。即使把 shared 算量全部免费化也只有1/9的 GEMM 算量上限，实际远小于此；不优先于覆盖所有专家的候选1。
- **全专家 BF16 缓存。** 每层原始FP8参数约867 MiB，BF16为1734 MiB；全模型缓存会显著挤占 KV。若按42层估算，替换FP8的净膨胀约35.56 GiB，保留原FP8另加BF16则71.12 GiB。每次 forward 临时解码又至少新增约0.909 GB读＋1.818 GB写，尚未算BF16 GEMM多出的读取，不能当成免费去量化。
- **先显式 gather 全输入。** 生成 `[9M,H]` 本身多写576/1152 MiB，还要再读；indexed A load 已是向量化，不能仅凭“随机访问”断定完整 permute 更快。中间布局应优先在生产者 epilogue 改，不另外扩散原输入。
- **output-owner W2。** 让一个 CTA 拥有 token 输出能避免9个专家之间的锁，但随机路由下会丢失128个同专家token间的权重复用，退化为大量 per-token GEMV。只计FP8 W2权重的9路逐token读取，就达72/144 GiB，不能把删除576/1152 MiB中间输出当成净赚。原子、行锁、last-writer及完整MoE切小块已有负结果，本报告不重开。[已闭合负例](../moe-mhc-sm80-0928.md#L70)。

## 候选1收尾：位一致，但完整 MoE 收益不足

`moe-scale-fold-b` 在 A100-SXM4-80GB 上 exit0、stderr为空。与当前 up+down、reduce off 配置同进程比较：

| 9轮随机顺序配对 graph，每次16 replay | native median | fold median | 逐轮降幅中位数 |
| --- | ---: | ---: | ---: |
| W13，8192 | 1.643984 ms | 1.558350 ms | 4.690% |
| W2，8192 | 1.164268 ms | 1.092284 ms | 5.728% |
| 完整MoE，8192 | 3.318620 ms | 3.229828 ms | 2.846% |
| W13，16384 | 3.254070 ms | 3.119138 ms | 3.726% |
| W2，16384 | 2.300184 ms | 2.222146 ms | 3.044% |
| 完整MoE，16384 | 6.563152 ms | 6.421220 ms | 1.967% |

完整层两形状均9/9配对胜出，但这是一次开发机运行，未取独立复跑或频率轨迹；不能与前述另一轮3.199/6.500 ms的绝对数直接相减，也不代表 chain 成绩。原始 eager 结果保留为次级证据。

实际 BF16 `__hmul2` 指令使用两次乘法之间的 inline-asm barrier，穷举同样4,389,120个有限FP8/安全scale组合、每组合两BF16 lanes，零差异，覆盖subnormal/FTZ。256对全部254个有限FP8 pattern出现差异，research guard拒绝256、负数、Inf、NaN。实际 packed W13/W2 scale256变异选择native kernel并逐位相同，之后恢复scale；末尾scale SHA不变。实际117 fixture生成的合成checkpoint经Humming真实转换后，W13、同activation的W2、完整eager、fresh-input graph及graph对fresh eager的16条raw-bit记录全部通过。它不是生产checkpoint或TP8验证。

两份private header树只有`mainloop_arith.cuh`改变，私有include路径和带tree SHA的编译源码隔离disk cache，raw JSON whitespace隔离dispatch key但所有解析配置相同。只驱逐构造缓存、保留原注册ID，使两组binary同进程共存；准备后禁止JIT。W13/W2各自native与fold的ID、路径、cubin SHA全部不同。scale寄存器每轮在`load_stage_iter`重新加载，`j==0`只乘一次，未跨K重复放大。未获得SASS，故不宣称实测指令数下降。

证据为 [archive README](../../../evidence/moe-scale-fold-0928/README.md) 和 [raw b](../../../evidence/moe-scale-fold-0928/results/moe-scale-fold-b.jsonl)。archive SHA256为`21354aaca4f31ecaa1785d6597d244e78fe6799cf96180b273e3c8edd5ef8d09`，331161 bytes，156个manifest条目均核验。首次`a`因研究脚本误写`run_activation`失败，完整失败收据和当时脚本一并保留，未用其性能。[`sglang_pr_audit_sol`独立只读review](scale-fold-review.md)重算manifest、核对源码/编译命令/不同cubin和配对raw，确认小幅局部收益，并指出下述数值边界。GPU1已释放。

**独立review指出的全域边界。** 穷举scale是bit pattern `[0x0000,0x437f]`，只覆盖+0；research `scale_allowed`用`isfinite & >=0 & <=255`，所以未覆盖的BF16 -0也会进fold。权重穷举排除E4M3FN `0x7f/0xff`两个NaN编码，而当前research guard只查scales、未查weight bits。有限合成权重和严格正scale的实际位比较结论不受影响，但这不是全域生产安全证明。若未来再考虑采用，须补特殊bits证明/测试或显式guard拒绝；本轮不生产化，也不为这些边界继续跑GPU。

## 追加源码问题：W2按N分条后立即归并，不执行探针

**技术上可做，但不是现有view/API直接支持。** indexed scheduler用`n_id=mn_index%N_BLOCKS`、`m_id=mn_index/N_BLOCKS`，即N内层连续；[`scheduler.cuh:193`](../../../evidence/moe-scale-fold-0928/cache/humming_scale_fold_headers/1fec70021765fb32becf11d8b30773eab91e9ea0f14ec64247e90d8f75396102/native/humming/scheduler.cuh#L193)。B loader的global K-row stride和expert stride来自完整`ProblemShape::N`，N块仅加列偏移，[`loader_b.cuh:21`](../../../evidence/moe-scale-fold-0928/cache/humming_scale_fold_headers/1fec70021765fb32becf11d8b30773eab91e9ea0f14ec64247e90d8f75396102/native/humming/memory/g2s_loader/loader_b.cuh#L21)、[`:79`](../../../evidence/moe-scale-fold-0928/cache/humming_scale_fold_headers/1fec70021765fb32becf11d8b30773eab91e9ea0f14ec64247e90d8f75396102/native/humming/memory/g2s_loader/loader_b.cuh#L79)。所以可以保留E/M、K累加和物理N4096权重布局，只将scheduler的N范围缩为一条、B/BS地址加全局N偏移，C用局部N和紧凑row stride；它不会重复计算完整投影。

现有C++ launcher要求B、BS、C连续且exact shape。B是`[E,K/16,N*4]`，C是`[9M,N]`，缩小ProblemN会连B/BS契约一起缩小，直接传strided packed-weight view失败；[`tensor.h:29`](sources/humming-0.1.12/humming/csrc/launcher/tensor.h#L29)、[`:79`](sources/humming-0.1.12/humming/csrc/launcher/tensor.h#L79)。writer的row stride也是完整ProblemN，seek用同一全局n_block，[`gmem_writer.cuh:111`](../../../evidence/moe-scale-fold-0928/cache/humming_scale_fold_headers/1fec70021765fb32becf11d8b30773eab91e9ea0f14ec64247e90d8f75396102/native/humming/epilogue/gmem_writer.cuh#L111)、[`:146`](../../../evidence/moe-scale-fold-0928/cache/humming_scale_fold_headers/1fec70021765fb32becf11d8b30773eab91e9ea0f14ec64247e90d8f75396102/native/humming/epilogue/gmem_writer.cuh#L146)。因此至少需要scheduler N-range、compact writer、私有driver launcher绕过exact-C检查，以及带最终out列偏移/stride的reduce。预先将32份N128权重片`.contiguous()`也能研究，但额外复制约289MiB W2权重及4.5MiB scales，不能作为免费生产方案。

**决定不测的流量账。** 设`R=9M`、activation `A=2R×256`、W2中间`D=2R×4096`。原N内层组织有机会让32个N128 CTA通过L2复用同一A行块；全M的N条带则每条重新扫全部A。下表假设native A近似只从HBM读一次、分条A大多未命中，是不利场景，不是实测DRAM counter：

| 每层逻辑容量/流量 | 8192 | 16384 |
| --- | ---: | ---: |
| activation全体A | 36MiB | 72MiB |
| N128单条scratch | 18MiB | 36MiB |
| 最多删除D写＋读 | 1152MiB | 2304MiB |
| N128额外31次A扫描 | 1116MiB | 2232MiB |
| N64额外63次A扫描 | 2268MiB | 4536MiB |

N128仅A重读就抵消最大D流量节省的96.875%，N64为196.875%。16k的36MiB scratch虽然接近A100约40MB级L2，却同时竞争72MiB activation和每条约9.03MiB FP8权重；8k也是18+36+9.03MiB，不能从scratch大小推断全命中。要避免dirty scratch写回，还必须在被逐出前归并并覆写，kernel完成本身不保证这个条件。32/64次GEMM和reduce launch、反复persistent startup、scatter输出行的局部性进一步削弱收益。若A原先也常miss，上述抵消会减小，但没有counter支持相反乐观假设；不为一个靠L2碰运气的分条方案再开GPU。父任务确认不做该probe，本任务不再占用GPU1。

## 追加源码问题：同一CTA完成W13→SiLU→W2，不实现

**中间activation能放下，但原kernel不能直接拼接。** BM128的BF16 gate或activation `[128,256]` 各64KiB，完整gate+up `[128,512]` 为128KiB。Humming的stage storage含A、FP8 B和BF16 group scale，并与epilogue scratch组成union；[`storage.cuh:171`](../../../evidence/moe-scale-fold-0928/cache/humming_scale_fold_headers/1fec70021765fb32becf11d8b30773eab91e9ea0f14ec64247e90d8f75396102/native/humming/utils/storage.cuh#L171)、[`:179`](../../../evidence/moe-scale-fold-0928/cache/humming_scale_fold_headers/1fec70021765fb32becf11d8b30773eab91e9ea0f14ec64247e90d8f75396102/native/humming/utils/storage.cuh#L179)。用源码[`smem.py:74`](sources/humming-0.1.12/humming/utils/smem.py#L74)在CPU计算当前dtype/group/布局的结果：

| 阶段/形状 | 原shared，含索引 | FP32 accumulator最低数 |
| --- | ---: | ---: |
| W13 BM128/BN256/BK64，4 stages | 133KiB | 32768个32-bit registers |
| 同形状，3 stages | 100KiB | 32768 |
| 同形状，2 stages | 67KiB | 32768 |
| 一次计算全部gate+up，BM128/BN512/BK64，4 stages | 197KiB | 65536 |
| W2 BM128/BN128/BK64，3 stages | 76KiB | 16384 |

Humming SM80 tuner的单CTA上限为163KiB，[`sm8x.py:7`](sources/humming-0.1.12/humming/tune/sm8x.py#L7)。顺序算gate/up时，必须在up流水运行期间额外保留64KiB gate：当前s4需197KiB；s3需164KiB，仍超过该门；s2需131KiB，才可容纳。一次N512又把accumulator推到65536 registers，未计A/B和地址寄存器，且s4 SMEM已经超限。因此可行方式是改流水深度，或改M/成对channel组织并安排生命周期，均为新的融合kernel设计。

一种容量可行的组织是：同CTA顺序算两次N256；W13先按原BF16输出舍入，再clamp/SiLU并保留现有中间BF16舍入，gate stash原位覆写为64KiB activation；随后遍历32个W2 N128块，每块FP32累加并写BF16 routed output。activation从resident SMEM直接喂W2，需要定制loader、bank swizzle和阶段索引。直接沿用W2全部原staging再额外留activation是140KiB；去掉多余A staging可以更少，但整个kernel静态SMEM仍由最大W13阶段决定，不能在执行W2时自动释放SMEM给另一CTA。[`humming.cuh:55`](../../../evidence/moe-scale-fold-0928/cache/humming_scale_fold_headers/1fec70021765fb32becf11d8b30773eab91e9ea0f14ec64247e90d8f75396102/native/humming/kernel/humming.cuh#L55)、[`:94`](../../../evidence/moe-scale-fold-0928/cache/humming_scale_fold_headers/1fec70021765fb32becf11d8b30773eab91e9ea0f14ec64247e90d8f75396102/native/humming/kernel/humming.cuh#L94)。

完整W2 FP32 accumulator `[128,4096]` 是2MiB，无法片内一次保留，必须N分块。这样从W2原32个N CTA任务变为同一个M/expert CTA串行32次。uniform账本仍约648/1281个M块，够填108个SM，但该131–140KiB设计固定一CTA/SM，失去当前W2两CTA/SM的latency hiding；小I并没有消除资源冲突。一个CTA只处理某专家的M行块，也无法完成每token另外8个专家的归并，因此 `[9M,4096]` 大输出和最后reduce完整保留。

**逻辑字节上限。** 删除gate/up写＋读144/288MiB，再删除activation写＋一次唯一读72/144MiB，总计216/432MiB；可释放workspace为108/216MiB。W2中间写＋reduce读仍为1152/2304MiB。被删除量仅占这三种中间张量一次写读总量的15.79%，尚未计输入/权重读取；这是逻辑字节比例，不是时间加速上限。W2的32份逻辑A请求主要可能是L2流量，不能全数冒充HBM节省。旧独立activation约0.087/0.179ms的数学运算仍需在CTA中做，不能与字节估算再次相加。改变流水、占用率和N并行可能抵消收益；本轮不实现该大改。

审计已关闭MoE追加研究：候选1实测小收益且不生产化；其余为明确标注假设的源码上限/否决理由，无新GPU结果。任何未来开发机改善仍需真实权重TP8和同ID chain对照，不能换算为服务成绩。
