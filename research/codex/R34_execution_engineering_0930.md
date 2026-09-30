# A100 执行优化：外部实现怎样落到冻结引擎

2026-09-30，Codex 与 GPT-6.1 Sol 并行复核。审查基线为正式 47266 的
`bf6b66fa`，新执行分支为 `codex/final-execution-0930`。本页记录实现约束
和开发机证据；Pod 动态状态统一看 [queue](../../notes/queue.md)，服务成绩
必须另外闭合，不能把算子加速比例写成 TPOT 或晋档收益。

## 外部代码中真正可借鉴的部分

[TokenSpeed](https://github.com/lightseekorg/tokenspeed) 将执行输入放进地址稳定
的设备缓冲区，准备、捕获、重放与资源释放有明确边界。一个容易忽略的细节是
`InputBuffers._bulk_pinned` 每步分配新的整块 pinned staging：持久复用同一块
主机缓冲区会与异步 H2D、重叠调度发生竞争。它并不是“所有临时内存都应缓存”。
我们的对应检查应覆盖数据何时写完、GPU 何时读完、下一步何时允许复用。

其 CUDA mHC finalizer 将混合、Sinkhorn、残差聚合和归一化放进一个 launch。
移植前必须处理两个具体合同差异：projection leading width 有 24/32 两种，
我们的基线 RMS 分母使用未舍入的 FP32 weighted sum；原生实现使用舍入后值。
直接复制实现会悄悄改变计算。匹配它的 `--use_fast_math` 编译配置后，局部
速度才好于 TileLang；严格编译版反而更慢。SM80 无 PDL，不能借用新架构的
启动依赖机制。

[FlashInfer packed KDA](https://docs.flashinfer.ai/generated/flashinfer.kda_decode.packed_kda_decode.html)
当前公开接口的覆盖是 SM100a/103a、T1/H12/K128/V128，不能直接用于 A100。
可借鉴的是把 raw gate、beta sigmoid 和 recurrence 放到同一步，同时保留
物理状态槽与负补齐槽的合同。我们已有 Triton packed 实现；缺口在 GLM 的
safe lower-bound 路由。新 174 对 SM80 普通 decode 增加默认关闭的接线，
严格检查 dtype/layout/stride，并保持 conv update 与 checkpoint tracking。
CUDA packed 的归约结果不逐位相同，本轮选数值逐位一致的 Triton 路径。

[MonoMoE](https://arxiv.org/abs/2609.04244) 的源实现把 routing、up、activation、
down、combine 放进 persistent kernel，并规定 grid 不超过 SM 数，确保所有
等待跨块旗标的 block 同时驻留，避免死锁。当前公开 specialization 是
SM90a、E256/N512/K2048、batch≤8，使用 WGMMA/TMA；我们的 A100、E289/N256/K4096
与常见 batch24–48 不匹配。借鉴重点是减少短矩阵工作中的空转和调度成本，
不是照搬它的硬件指令或性能数字。

源代码固定提交见 [external-sources.json](../../evidence/execution-0930/external-sources.json)。
上述工程约束来自对应源码，不构成当前服务收益的证明。

## 已经实操验证的候选和反例

MoE 使用 Humming **0.1.12**，真实 TP8 几何与合成 FP8 权重。模型配置的
intermediate=2048，TP8 每卡为256；288 routed experts 加一个 shared，top-k
总共九。原生小 M W2 配置的区间是 `64 < routed_rows <= 4160`，不是 GPU
batch 的边界。候选统一将 K tile128 改64、CTA/SM1改2，保留 M16、N256、
FP32 累加、stream-K off 和输出布局。两个 GEMM 共享 expert block 索引，
因此只改 W2 的 M tile 会破坏索引合同。

完整 router→align→up→clamped activation→down→weighted reduction 的交错图测量，
normal 路由 B24/28/32/40/48 分别为 300.7→287.7、338.3→323.1、364.8→349.7、
394.1→378.0、439.4→420.6 微秒，整层约减少4–4.5%。不同路由分布与原生区间
边界也有正收益。仅 down 阶段约14%的收益不能当成整层、更不能当成完整服务
的收益。up 候选收益弱，M8 在 SM80 生成不支持的 MMA，均被舍弃。代码只匹配
原生完整配置和区间，未知新选项、其他输入区间保持原样。

证据：[full-candidates](../../evidence/execution-0930/moe/full-candidates.json)、
[bounds](../../evidence/execution-0930/moe/bounds.json)、
[真实模型几何](../../evidence/execution-0930/model-geometry.json)。

mHC 的已有 post→pre FMA helper 在 eager 中较快，但 CUDA graph 的 M32
从22.872增到34.140微秒，拒绝接线。尝试把 post 融进 tensor-core split-K
stage0 同样更慢。TokenSpeed CUDA finalizer 改配合同后，固定 block128 的
完整边界在 M24–48 约省0.8–1微秒，M2048却持平偏慢。现阶段不足以支持
大范围移植或重写。BF16 wide residual 是后续 FFN pre 和 FFN post 都需要的
输入，不能把这个中间写出直接删除。

证据：[mHC graph 对照](../../evidence/execution-0930/mhc/results.json)、
[TC 融合](../../evidence/execution-0930/mhc/tc-fusion-results.json)、
[匹配编译配置的 CUDA finalizer](../../evidence/execution-0930/mhc/ts-finalizer-fast-results.json)。

KDA 的2048步 recurrence、变更输入的图重放、负补齐槽、不同物理槽 pitch、
GQA 与精确 dispatcher 接线均做过对照，选中的 Triton 路径逐位一致。但
B32/36 局部略慢，不能写成每个 batch 都更快。证据见
[KDA](../../evidence/execution-0930/kda/graph-result.json)，实现说明在候选分支
`engine/docs/174-kda-safe-packed-decode.md`。

## 正在查的工程缺口

冻结引擎的 SM80 indexer 接收 DeepGEMM schedule placeholder，但实际 paged
kernel 不读取这个参数；普通 decode 每 batch 仍生成 zeros 并复制 schedule。
可跳过普通 schedule 更新，KPool4 的独立 pooled plan 与全部 live 长度/索引
更新必须原样保留。这个机会仅省每 batch、每 rank 的两次小操作，不能乘上
11层或八卡当作端到端收益。

更大的候选是把 graph 外的 attention metadata 准备捕获成图。已有 helper
默认关闭，正确性要确认真实 req→KV/状态槽、sequence length、track mask
与 destination 都按步更新；runner recapture 必须失效旧 metadata graph。
另外，首次捕获可能同步设备并触发回收；必须核实际 Torch 实现并控制捕获
时机，不能用测量前已捕获的局部成绩掩盖服务中第一次触发的停顿。

还有一个应当先核而非假定的布局事实：DSA indexer 的 query heads 由 replicated
linear 产生，不能仅因 TP8 就把 H32 除成 H4。下一轮 decode probe 会按实际
布局检查 page 读取、query 重用和完整 KPool stage 的成本。

同配置重跑的共同3108条请求中，chain 出现2条修好、5条新坏，TPOT mean
也有约0.48ms差异；小于这种服务波动的单次变化不足以宣布净收益。对照收据
见 [A1/A2 配对](../../evidence/execution-0930/a1-a2-paired/paired.json)。
