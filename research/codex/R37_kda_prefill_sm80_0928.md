# R37：KDA prefill 融合核与 BV16，转入 TP8 实测

2026-09-28，Codex 开发，GPT-6 Sol 独立源码与原始记录复核。
状态：**开发机数值、状态、预热和局部性能已验证；TP8 真实模型及 chain 待测。**
按用户要求结束本地调参，交付现有候选，不继续扩大形状表或重写其他算子。
实现与开关见 [170 说明](../../engine/docs/170-kda-sm80-prefill.md)。其他未采用路线见 [R36](R36_prefill_kernel_options_0928.md)。

## 做成了什么

- 新 SM80 准备核把 Q/K/V 三次 strided copy 和 Q/K 两次 L2 normalization 合成一次 GPU launch。读取卷积已舍入的 BF16，保留原 FP32 求和、epsilon、除 sqrt；V 直接复制 BF16 位。
- 原 recurrence 的 V tile 从 32 改成可选 16，增加长请求的 CTA 数；没有改递推公式、FP32 状态或快照算法。只从显式 KDA 调用进入，不改变共享 GDN 配置，不做会反复修改状态的多配置 autotune。
- 可独立启用 CPU logical length，减少每层重复读取 CUDA prefix sum 末元素；对 CP/TBO/非普通 extend 等不适用场景保留原路径。

生产准备核用三个独立输出存储。早期原型把三者放在一个 storage，最终原地写入 V 的 attention output 会连带保留已经用完的 Q/K，未 padding 的 16k 情形多保留 64 MiB。E 起已修正，继续只发一个 kernel。

## 实际选择与测量范围

单 A100-SXM4-80GB，Torch 2.13.0+cu130，Triton 3.7.1，TP8 对应的每卡 H=8、K=V=128。
合成输入下调用真实 `KDAAttnBackend.forward_extend`，边界包括卷积、准备、gate、recurrence、输出和状态写回；**不含模型投影、输出归一化、TP 通信、MoE、DSA、mHC或排队**。
三请求为不齐整长度，加 7 行 NaN padding，171 式投影输入的行间隙、已有前缀、140 快照、envelope-strided 状态池均有覆盖。

F 使用四个真实初始化的 Triton 实例，分别开原生、prepare、state16、二者组合；没有生产 kernel monkeypatch。
所有臂均开 CPU-length、预热相同 metadata cache，因此下面没有再计 CPU 同步优化的收益。
9 轮随机配对，每轮 8 次 stateful 调用；每臂计时前重置状态，计时内连续调用，不在每次调用之间重置。
eager GPU event 包含主机提交造成的 GPU 空档；host enqueue 与纯 captured core 另行报告。未锁 GPU 时钟，raw 保留 telemetry 和全部轮次。

| F 中最终选择的路径 | 原生 eager 中位 ms | 所选路径 ms | 配对耗时降幅 |
| --- | ---: | ---: | ---: |
| 8k，1 请求，prepare + BV16 | 0.951064 | 0.856020 | 10.20% |
| 8k，3 请求，只 prepare | 1.021416 | 0.953876 | 7.56% |
| 16k，1 请求，prepare + BV16 | 1.749476 | 1.490356 | 14.84% |
| 16k，3 请求，prepare + BV16 | 1.584952 | 1.342244 | 15.31% |

降幅是逐轮 `(baseline-candidate)/baseline` 的中位数，不一定等于两个中位数之比。
[F raw](../../evidence/prefill-kernels-0928/kda-prepare-f.jsonl)、[F summary](../../evidence/prefill-kernels-0928/kda-prepare-f.summary.json)、[源码封存](../../evidence/prefill-kernels-0928/kda-production-f.tar)。

**为什么最终收窄 BV16。** F 中 8k 三请求的组合相对 prepare-only，9/9 轮 eager 都慢，配对中位慢 2.87%；host enqueue 约 0.934→0.954 ms。其 captured core 仍快约 5%，但不能覆盖生产 eager 的主机代价。最终在实例入口做一个简单条件：8k 仅单请求用 BV16，16k 再由下游检查 N<=3。8k 混合请求不进入较重的 BV16 guard。
8k 单请求的 BV16 增量只有配对约 1.74%，且有两轮负波动；16k 两组增量约 5.52%/5.04%，九轮都正。此处不把所有形状描述成稳定同幅收益。

收窄后只补 [G 的实例 dispatch 边界门](../../evidence/prefill-kernels-0928/kda-state-dispatch-g.jsonl)，exit 0：8k 单请求传入 BV16、三请求明确回退，默认关闭保留原生。没有重复整套性能矩阵；上表明确引用 F 中被选择的执行路径，不冒称 G 又测了一次完整性能。
这些比例也不能写成“整模型 prefill 加速 15%”。是否值得上线由下一轮 TP8 块耗时和同 ID chain 对照决定。

## 数值、启动与内存

- Prepare E/F：真实 arm/off 与不支持布局回退；8k/16k Q/K 逐位同原生，V 穷举全部 65,536 BF16 位型；三个输出独立 storage。T16 启动预热后首次 8k/16k 调用及实例 extend 都是零 JIT miss、零编译、零磁盘载入。
- State B：12 个签名（index int32/int64、NT bucket 1/2、normal/snapshot-no-export/snapshot-export）及额外 8k ragged 均通过。BF16 h/v_new、完整 FP32 状态 envelope 逐位相同；四次 fresh graph 改输入、边界、slot、快照通过。正式调用和 capture 都是零 JIT/disk/compile/module load。18 项 fallback 含实际原生 Autotuner 调用。
- F：16 次实际 dispatch 选择符合开关，16 组 eager 和 16 组 fresh graph 的 output、conv envelope、SSM envelope 全部逐位一致，真实 int64 snapshot 与 int32/int64 slot indices 已覆盖。此处仍是合成输入，不替代真实权重 TP8 冒烟。
- CPU length 的四项合同测试和独立 GPU scalar-read 对照通过。CP metadata 在模型 forward 才建立，所以还检查启动时的 attention CP size；不靠当时 `attn_cp_metadata=None` 猜测没有 CP。

F 的返回输出 storage 字节等于逻辑输出字节：无 padding 的 8k/16k 为 16/32 MiB；padding case 另有 7 行输出。单次 core allocator peak 与原生基本相同，最大差 28 KiB；计量起点排除了已有 graph allocations，**不是整服务峰值或 KV 池容量结论**。

State B 初次 warm 临时 tensor 峰值 2,362,368 B，结束后无保留 tensor。初次 device-free 下降约 78 MiB 包含共用 driver/Triton 惰性初始化，不能算 BV16 独有成本。
后续 [memory-a](../../evidence/prefill-kernels-0928/kda-state-memory-a/) 先 warm 相同 12 个原生 BV32 签名，再 arm BV16：新增 12 次 module load，本次 allocated/reserved/device-free 增量均为 0；重复 warm 也为 0。它只证明该同进程基线下未观察到额外设备分配，不证明模块没有占用 driver 已分配池里的空间。缓存已存在时 BV16 warm 为 26.55 ms，不能当冷磁盘编译时间。

## 复核与失败记录

独立复核见 [prepare](../../notes/reports/prefill-kernels-0928-review/kda-prepare-review.md)、[state 与 F](../../notes/reports/prefill-kernels-0928-review/kda-state-production-review.md)、[早期 C raw](../../notes/reports/prefill-kernels-0928-review/kda-prepare-c-raw-review.md)。
所有开发 GPU 源码是逐文件上传；不声称远端整个引擎等于本地 HEAD。运行时 SHA、源码 capsule、完整 raw、stderr、exit 和时间收据保留在 [证据目录](../../evidence/prefill-kernels-0928/)。

失败也保留：prepare-b 因 Triton 分支里同名变量 BF16→FP32 重绑定而编译失败，c 改用 xf/y 后闭合；state-production-a 因测试 mock 的 16 MiB 引用迟到 GC，内存基线断言失败，b 在测量前清理测试引用后闭合。两者均不计性能成绩。BV8 比 BV16 慢，局部图包含输入 staging 后 16k 不赚，均不采用。

## TP8 实操交付

在当前同一服务基线、同一池大小、同一负载上只加入这组执行层开关：

```bash
SGLANG_AX_KDA_PREFILL_CPU_LENGTH=1
SGLANG_AX_KDA_PREFILL_PREPARE=1
SGLANG_AX_KDA_PREFILL_STATE_BV16=1
```

默认全关；任何一项可以单独撤回。保留当前 prefill graph 设置，不为了本候选另开 CUDA graph。
核启动的两个 `[ax] KDA prefill ...` armed 日志、引擎提交、池大小；先 TP8 冒烟，再沿用现有 N30 同 ID 对照，看完整块耗时、chain、TPOT 和实际峰值。
用户已决定能力门由线上检查，本地不再重复单独全套能力复核。当前没有新增 chain 成绩，也没有据此承诺更高线上档位。
