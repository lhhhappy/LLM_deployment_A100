# A100 大块 MoE：完整层约 6–8% 的执行收益

2026-09-28，Codex；独立复核见[审查记录](moe-mhc-sm80-0928-review.md)。**chain 尚未验证。** 开发分支 `codex/mhc-moe-sm80-0928`，引擎基线 `20a58da9`。开发机诊断已闭合，主候选为 117 的 W13 上投影与 W2 下投影联合优化；三个新开关均默认关闭。Fable 负责后续真实权重 TP8 与同 ID 服务对照。

生产代码、回归测试与机制说明提交：`eaf41653`。只摘取该提交即可获得候选执行代码；研究原型、报告和证据单独提交。引擎只改 MoE 归并、117 runner 子类和新配置 helper，没有调度或缓存改动。

## 最终可交付结果

实测使用 A100-SXM4-80GB、Humming **0.1.12**、torch 2.13.0+cu130、Triton 3.7.1。通过真实 `FusedMoE` 加载合成 block-FP8 checkpoint，采用 TP8 的每卡专家形状：289 个专家、hidden 4096、intermediate 256、top-k 9。每卡处理完整 8k/16k 行，保留共享专家、路由缩放 2.5、SwiGLU clamp 10 和所有权重。这里的“完整 MoE”包括路由对齐、两次专家 GEMM、激活与归并，不含 attention、mHC、TP 通信或整个 Transformer 层。

最后两组七臂对照独立在 GPU 0/1 内比较，每种形状每臂 9 轮、每轮 16 次，随机执行顺序。CUDA graph 消除 Python 提交间隙；另测 eager，方向一致。下表百分比为同轮配对耗时降低率的中位数，不是两个中位数之比。

| 数据与行数 | 原完整 MoE graph | 上、下投影联合 | 配对耗时降低 | 9 轮 bootstrap 95% 区间 |
| --- | ---: | ---: | ---: | ---: |
| D，均匀路由，8192 | 3.481870 ms | 3.199162 ms | **8.16%** | 7.53–9.57% |
| D，均匀路由，16384 | 6.916238 ms | 6.499830 ms | **6.12%** | 5.80–6.19% |
| E，偏斜路由、输入幅度 ×4，8192 | 4.065394 ms | 3.749558 ms | **8.12%** | 7.36–8.97% |
| E，偏斜路由、输入幅度 ×4，16384 | 7.395510 ms | 6.941980 ms | **6.15%** | 5.90–6.99% |

E 把路由偏向 32 个专家，仍保留全部 9 个专家贡献；执行和参考都包含 clamp，但没有记录被裁剪的元素比例。GPU 没有锁频：计时中 D 的 SM 频率为 1290–1395 MHz，E 为 1305–1395 MHz。候选没有占更高频率的优势，但区间只描述本次轮间波动，不是跨机器或服务收益的置信区间。原始记录：[D](../../evidence/mhc-moe-sm80-0928/moe-seven-arm-d.jsonl)、[E](../../evidence/mhc-moe-sm80-0928/moe-seven-arm-e.jsonl)；[D 摘要](../../evidence/mhc-moe-sm80-0928/moe-seven-arm-d.summary.json)、[E 摘要](../../evidence/mhc-moe-sm80-0928/moe-seven-arm-e.summary.json)。

W2 单改的 B/C 对照，在均匀与偏斜路由下测得完整 MoE graph 约 2.9–4.3% 的配对收益；D/E 新增 W13 后才得到上表联合结果。三个开关同时打开也测过，归并核的额外完整层收益仍不稳定。因此建议先以 **up+down** 做 TP8 候选：

```sh
SGLANG_AX_SM80_FP8_MOE_HUMMING=1
SGLANG_AX_SM80_MOE_UP_TUNE=1
SGLANG_AX_SM80_MOE_DOWN_TUNE=1
SGLANG_AX_SM80_MOE_REDUCE=0
```

这不是整段 prefill 快 6–8%。只有在 MoE 占总耗时 25%、其余不变且没有重叠变化的条件下，才可算出约 1.5–2% 的整体耗时节省；该算术不是新的 Pod 实测，更不能兑换为 chain 少几条或 N 档提升。

## 生产实现与验收

W2 保持 M128/K64 和 FP32 累加，把 N256 改为 N128，pipeline stages 4→3，`num_ctas_per_sm` 1→2。W13 只关闭原生 stream-K：Humming 0.1.12 原路径先把分片 FP32 结果转 BF16，再用 BF16 加法/原子操作合并；关闭后完整 K 维在 FP32 中累加，最后转 BF16。**W13 会改变舍入行为，不承诺逐位一致。** 权重、专家选择、clamp 和缩放均保留。

W13/W2 只覆盖 SM80、117 indexed 路径、BF16 输入/输出、FP8 e4m3 权重、BF16 K128/N0 scale，以及 **8192–16384 行（含边界）**。只认已测的原生配置，其他形状/配置保持原表。表与 JSON 同步复制，不原地修改原表或其他层缓存；开关在启动时读取，首 token warmup 预编译新配置。说明：[W13](../../engine/docs/117-sm80-moe-up-followup.md)、[W2](../../engine/docs/117-sm80-moe-down-followup.md)、[可选归并](../../engine/docs/117-sm80-moe-reduce-followup.md)。Pod 若开启 attention-TP 输入分片，须先核对 MoE 实际每卡行数，不能向分片后的不同形状外推或相加收益。

独立参考从原 FP8 checkpoint 和原 FP32 scale 反量化，用 FP32 GEMM（TF32 关闭），包含 clamp、SiLU、共享专家和路由缩放。它不是候选自己的中间结果。

- 最终 D/E 共 28 条完整层数值记录、56 条 fresh-input graph 记录通过。每次检查完整输出 finite、抽样 16 行独立参考。graph/eager 参考相对 L2 最大 **0.0059831**，同时满足固定 `<0.02` 门和相对原路径 `≤1.05×baseline+1e-4` 门。原路径重复误差只用于 graph/eager 比较，没有取代参考门。真实权重全模型能力尚未验证。
- 最终生产接线测试独立构建 off/down/up/both 四层，验证 48 条有效配置和 40 次实际 dispatch、15 项转换权重逐位一致、18 项同激活 W2 逐位一致，以及边界和错误 metadata 回退。首 token warmup 后禁止调用 Humming compiler，剩余 warmup 和实际 prefill 均通过。[收据](../../evidence/mhc-moe-sm80-0928/moe-up-down-production-c.jsonl)。
- 归并生产测试通过 61 项原始 BF16 位比较、19 个回退/默认关闭/FakeTensor 检查，包含 fresh graph、独立并发 stream、真实 Humming down-output。微型 warmup 后禁止 Triton 编译，实际大形状运行通过。[收据](../../evidence/mhc-moe-sm80-0928/moe-reduce-production-a.jsonl)。
- 最终七臂每次前向瞬时 allocated 峰值相同：8k **640.427 MiB**、16k **1280.711 MiB**，其中返回输出分别为 64/128 MiB。没有新常驻权重或工作区，也没有显存容量收益。graph allocator 的 live 值不能当作 graph pool 的 reserved 占用。

归并的最终生产 graph 为 8k **0.385512→0.377908 ms**、16k **0.765556→0.750916 ms**，约 2%；[原始轮次](../../evidence/mhc-moe-sm80-0928/moe-reduce-production-timing-a.jsonl)。CPU enqueue 的额外下降来自 capability 查询缓存，不能算 GPU 吞吐。

## 早期归并探索

所有数字均为单张 A100-SXM4-80GB、BF16 激活、FP32 路由权重、合成输入。MoE 形状为每 token 9 个专家（含共享专家）、hidden 4096；没有删专家、改变权重或降低精度。归并探针先逐位比较原 kernel，并抽样对 FP64 独立参考检查舍入界。计时用随机候选顺序、多轮 CUDA events，并另测 CUDA graph。

| 归并实验 | 8k 基线 → 最好候选 | 16k 基线 → 最好候选 | 范围 |
| --- | --- | --- | --- |
| 23 个原 kernel tile/warp 与扁平布局 | 0.39846 → 0.39366 ms | 0.77808 → 0.76925 ms | L2 清空后的事件计时 |
| CTA 次序、持久循环、缓存策略 | 0.39882 → 0.38979 ms | 0.77802 → 0.76234 ms | L2 清空后的事件计时 |
| 20 个向量宽度/warp 配置 | — | 0.77798 → 0.76074 ms | L2 清空后的事件计时 |

第二轮 16k 图回放中位数约 0.7653 → 0.7500 ms。上述是**归并单核约 2% 的收益**，不能写成整段 prefill 快 2%。16k 归并名义输入/输出流量约 1.343 GB；这是由形状计算，不是 DRAM 性能计数器读数。

证据：[第一轮](../../evidence/mhc-moe-sm80-0928/moe-sweep-a.jsonl)、[流式布局](../../evidence/mhc-moe-sm80-0928/moe-stream-a.jsonl)。脚本：[原型比较](../../scripts/analysis/bench_moe_reduce_sm80.py)、[缓存与执行顺序](../../scripts/analysis/bench_moe_stream_sm80.py)。

## 完整 MoE 的早期结果与分块负例

`moe-layer-c` 通过真实 `FusedMoE` 加载合成 block-FP8 权重，形状为 TP8 的每卡专家切片：289 个专家、hidden 4096、intermediate 256、top-k 9，保留共享专家、路由缩放 2.5 和 SwiGLU clamp 10。只替换归并 kernel，在同一份真实 down-output 上，8k/16k 均与原归并逐位相同。

7 轮完整层事件计时的中位数为 8k **3.72636 → 3.70737 ms**，16k **6.96378 → 6.88937 ms**。这是单次多轮观测：16k 有 2/7 轮候选更慢，且其中位数差值大于单归并 kernel 的差值，不能报成稳定整层收益。后续四臂/七臂 CUDA graph 对照仍未确认归并的独立完整层增量。

完整 16k 层输出也逐位相同；完整 8k 层原 Humming 的重复相对 L2 误差为 0.001511，候选相对基线为 0.001519。后者不是“8k 全层逐位相同”，也不是归并引入的误差；同一中间张量的归并已单独检查。`moe-layer-a/b` 因把完整 8k 层误设为逐位一致而失败，保留失败收据，不能计作通过。

完整 MoE 内部再拆成小块并拼接并不划算：8k 拆成 1k/2k/4k 后为 7.050/5.272/4.621 ms；16k 为 14.065/10.620/9.238 ms。16k 拆成 4k 可降低瞬时工作区，但吞吐显著变差，因此不采用。原 16k down-output 为 1152 MiB，前后写入/读取名义流量共 2304 MiB；这是每层复用的临时空间，不能乘 42 当成模型常驻内存。

证据：[完整层](../../evidence/mhc-moe-sm80-0928/moe-layer-c.jsonl)、[向量配置](../../evidence/mhc-moe-sm80-0928/moe-vector-a.jsonl)。

## mHC 的负结果

第一版把 post 与 prenorm 投影/square sum 合在一个 Triton kernel，仍写出后续残差所需的 BF16 residual。8k 下 residual 逐位一致，但最佳配置仍比 TileLang 原路径慢：原路径约 0.669 ms，最佳原型约 0.809 ms。多个配置寄存器占用达到 255 并溢出。另测只替换 prenorm 的对照、减少展开的循环及双 BF16 分量表示的 FP32 权重乘法，也没有获得净收益。**均未接入引擎。**

这里基线只包括 post + prenorm GEMM/square sum，不含最后的 mixing/Sinkhorn/RMSNorm，因此这些数据不是完整 mHC 边界计时。低于或等于 2048 行的原生路径另有 split-K 分支，不能用本大块基线外推。BF16 双分量原型的投影误差优于原 TF32 参考，但速度更慢，不能据此宣称模型质量变化。

证据：[融合](../../evidence/mhc-moe-sm80-0928/mhc-fused-a.jsonl)、[仅 prenorm](../../evidence/mhc-moe-sm80-0928/mhc-prenorm-a.jsonl)、[减少展开](../../evidence/mhc-moe-sm80-0928/mhc-compact-a.jsonl)。

## 下投影融合的负结果

隔离的 Humming 0.1.12 epilogue 原型保留每个专家原有的 BF16 GEMM 舍入，然后用 FP32 累加。下表范围均为下投影及归并，取专家优先顺序；输出列优先也测过，更慢。

| 方法 | 8k 基线 → 原型 | 16k 基线 → 原型 | 结论 |
| --- | --- | --- | --- |
| 逐元素 FP32 原子加 | 1.613 → 6.651 ms | 3.141 → 13.347 ms | 淘汰 |
| warp 行锁后向量累加 | 1.614 → 9.097 ms | 未测 | 淘汰 |
| 最后写入者读齐 9 个专家并归并 | 1.611 → 3.843 ms | 3.138 → 7.606 ms | 淘汰 |

原子/锁版本在抽样 FP64 舍入界内，最后写入者版本在该次运行测得全量 L2 差异为零（当时没有比较原始 BF16 位表示），但运行结果本身不构成跨 CTA 内存序的形式证明。独立审查指出 relaxed 计数器缺少显式 acquire、后续脚本误让锁错误字与输出重叠；现探针改为 acq_rel 发布与 warp 同步，并把错误字独立保存，上表仍是修正前的负结果，不给修正版预支性能结论。均未进生产引擎。探针只改私有缓存中的 Humming 头文件；没有改共享安装或 Pod。复用 launcher 的原型暂留原大小 workspace，**没有显存容量收益**，也没有可部署 ABI。

修正版 M256 的 119–122 四条路径数值、错误字、计数器以及最后写入者原始 BF16 位比较均通过，仍比原路径慢；[修正后 sanity](../../evidence/mhc-moe-sm80-0928/moe-epilogue-sanity-b.jsonl)。这些编号是探针 selector，不是新增机制编号。

证据：[原子加](../../evidence/mhc-moe-sm80-0928/moe-epilogue-a.jsonl)、[行锁](../../evidence/mhc-moe-sm80-0928/moe-epilogue-lock-a.jsonl)、[最后写入者](../../evidence/mhc-moe-sm80-0928/moe-epilogue-last-a.jsonl)。

## 下投影计算块调参

第一轮固定路由排序所用的 M 块高度 128，比较 18 个 GEMM 配置。8k 下投影＋归并从 **1.61287 → 1.52818 ms（5.25%）**；16k 从 **3.13561 → 3.02398 ms（3.56%）**。两者最佳配置不同：分别为 N128/warp64×64 和 N128/warp128×32，均 BK64、2 stages、`num_ctas_per_sm=2`。该参数同时影响编译器驻留提示和持久 CTA 启动数量，不等于实测同时驻留数量。

所有 BK64 候选的输出与原路径逐位相同，独立参考基于原 FP8 checkpoint 和 FP32 block scales；BK128 候选有约 3.5e-5 的相对差异但没有速度优势。第二轮选择两种行数共用的 N128/warp64×64/stages3/CTA2；W2+reduce graph 8k **1.593530→1.478561 ms**、16k **3.118844→2.976471 ms**。这些仍是阶段耗时，完整层以本文开头的最终对照为准。

证据：[18 配置](../../evidence/mhc-moe-sm80-0928/moe-down-tune-a.jsonl)、[第二轮](../../evidence/mhc-moe-sm80-0928/moe-down-tune-b.jsonl)、[W13 扫描](../../evidence/mhc-moe-sm80-0928/moe-up-tune-a.jsonl)。

## 失败留证与服务交接

`moe-down-production-a` 捕获了 Python shape tuple 与 JSON list 比较不等、导致优化静默未生效的问题；已经只对两个 shape 字段规范化，B 与最终 C 实际 dispatch 检查通过。`moe-four-arm-a` 同时受该 no-op 和过严原生 graph 重复断言影响，**不能用作收益证据**。B/C/D/E 已修正，源码快照及哈希随结果保存。原生 W13 自身的末位波动不能归给归并，也不能借此放宽直接数值门。

原始证据、状态与复现入口统一见[证据目录](../../evidence/mhc-moe-sm80-0928/README.md)。开发 GPU 测试已闭合，两卡释放；没有修改或停启 Pod。整理后的分支交 Fable 做真实权重 TP8 冒烟及 N30 同 ID 对照，先看 chain 超标修复/新增，区分开场与稳态，再检查覆盖率、错误率与 TPOT p95≤100 ms。任何 chain 回退否决；能力按用户要求由线上门判，不另起单独能力复核。
