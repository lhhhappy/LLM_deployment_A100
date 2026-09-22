# R13 — AgentX 与 MLPerf 双文精读：映射到本赛的候选与边界

2026-09-22，Codex main，T34。用户指定两篇文章后的定向研究；**仅查阅正文、公开 PR/API 和本地源码，无 GPU、安装、镜像、服务、提交或引擎修改**。本地参考 commit 已复核为 `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`。

标记：VERIFIED 表示确认文献/源码确有该内容，不表示我们复现了作者性能；INFERRED 表示本赛适用性判断或待验证假设。本报告不改变现有执行队列，不解除 E2 的数值门，也不是切换引擎的决定。

## 1. 阅读结论

**INFERRED**：最有价值的组合不是照抄参数，而是把三个问题分开：①是否有可恢复的缓存边界；②命中后短请求是否仍被长 prefill 挡住；③轮到它执行时，算子/通信/CPU 是否适配真实小 batch。D1、D2、DP 容量和小形状优化分别作用于不同部分，不能用单一 cache-hit 指标验收全部收益。

## 2. AgentX：哪些是已实现的机制

主文：[vLLM x AgentX，2026-09-08](https://vllm.ai/blog/2026-09-08-vllm-agentx)。以下为简述，不转载文章。

**VERIFIED / 作者报告**：文章讨论会话亲和路由、限制单请求 prefill 份额、DP/EP 组的 prefill 节拍，以及 MLA 的 DCP/DEP 取舍。其测试中 sticky routing 优于若干即时负载均衡策略；PP 在冷长输入与热多轮输入上的表现不同。其 TPGS 包含缓存 token，不能读成每秒新生成 token。硬件/模型和本赛不同，倍数不迁移。

### 2.1 快照保留有三个独立的实现阶段

**VERIFIED / GitHub REST 元数据、PR 描述，及 #47782 核心 diff**：

| 实现 | 确认的范围 | 合入日期 UTC |
|---|---|---|
| [#43447](https://github.com/vllm-project/vllm/pull/43447) | 滑动窗口选择性保留；未缓存块优先复用；不是完整的 Mamba 实现 | 2026-06-04 |
| [#45845](https://github.com/vllm-project/vllm/pull/45845) | 将 retention interval 接到 Mamba/线性注意力；减少过密快照占池 | 2026-06-23 |
| [#47782](https://github.com/vllm-project/vllm/pull/47782) | 稀疏保留时保护已发现的共享前缀边界，并贯通请求、匹配、调度和保留 mask | 2026-07-13 |

第三项的 `get_computed_blocks` 增加共享边界返回值；`Request.shared_prefix_boundary` 传入 `_mamba_block_aligned_split`。align 模式通过让 chunk 停在必要位置产出状态，保留逻辑再保护该位置。**不是单独改一个 eviction 参数就完成整个功能**。#43447 merge commit=`a6183563b6f604ef7b481ce8ce7af359c6dc1b74`；#47782=`8ac8375270c1c86b0a8eaba7e738908a9a7303e1`。网页读取超时时使用公开 REST API 核验，没有把搜索摘要当代码审阅。

**INFERRED / 与 D1 的差别**：共享边界策略根据已出现的复用补点；我们的 role 策略是在当前 prompt 内预判下轮可能分叉的位置。若边界只在相邻两轮出现、下一轮又向后移动，事后补点不保证避免第一次重算。role 策略则可能猜错并增加切块/状态成本。比较应包含 strict_append、reminder_heavy、共享 system prompt 分支，不能先认定语义边界策略胜过自适应策略。

**VERIFIED / 本地**：`arg_groups/fields/exec_.py:352` 的 `mamba_max_states_per_path` 默认 -1；`mem_cache/unified_cache/components/mamba.py:255` 对该上限的实现会保护 tail/fork/locked 等节点，因此是软上限。设成 2 不等于指定保护 role+end，更不等于每个会话只能占两个状态。

### 2.2 容量、路由与并行不能分开算

**VERIFIED / 架构资料**：[vLLM hybrid manager 设计](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/)解释共享块池与各层组可用前缀的交集；但该文明确基于旧 commit，仍含 Mamba prefix caching 为 WIP 的历史文字。当前实现状态以以上已合入 PR 为准，不能照搬那句 WIP。

**INFERRED / 本赛映射**：

- DP attention 按请求/请求 token 分工；DCP 按同一请求的上下文分工。两者都可能减少 MLA 重复 KV，但通信与状态分片路径不同。文章中的 TP8/DCP8 不是我们的 TP8/DP8 启动配方。
- 开 DP 后要同时检查 KDA 槽、attention 权重复制、每组 KV 余量和路由。沿用 [R5](R5_dp_memory_accounting.md) 的条件账，不能宣布近线性 N@SLO 增长。
- sticky 路由需要服务端实际可获取且稳定的 session 依据；不能假定 harness 会提供新 header，更不能用未来请求信息。若只有 token 前缀，应按合法可见的缓存信息设计另一个路由方案。
- “存有 MLA KV”不足以证明 GLM 混合前缀可恢复；KDA/conv/indexer 状态及深度必须齐全。增加 host offload 之前仍需完成 HiCache 的 hybrid-DSA 正确性和全层 flush 检查。

**VERIFIED / 本地边界**：`arg_groups/fields/memory.py:80` 已有 `enable_unified_memory`，默认关闭，并声明 Triton 后端要求以及 HiCache、prefill graph、speculation 限制。因此不能说 SGLang 完全没有共享池，也不能把它作为现有 DSA 配置的即插即用开关。静态池的 `mamba_full_memory_ratio` fallback 0.9 在 `arg_groups/fields/schedule.py:203`；调小之前应看实际瓶颈池和最终解析值。

### 2.3 两种调度手段不要混淆

**VERIFIED / 本地**：

1. [002 / D2](../../patches/002-spf-scheduling.md) 已移植短 prefill 优先及 continuation 预算预留；不是只给等待队列排序。对应公开 [SGLang #40024](https://github.com/sgl-project/sglang/pull/40024)。
2. 原生 `--prefill-decode-interval` 已存在（`arg_groups/fields/schedule.py:63`）。`scheduler.py:1343` 消耗计数器，`:1350` 在 extend 后装填；DP 使用同步的 `is_extend_in_batch`，`:3650` 附近决定跳过新 prefill。默认未设置时最终归零，见 `arg_groups/validation_hook.py:452`。

**INFERRED**：D2 给短 prefill 留出同轮预算；interval 则可能让全部新 prefill 多等若干步。它们针对不同干扰，组合可能改善 TPOT，也可能破坏 3 秒 fast TTFT。SGLang 这个触发式计数器与 vLLM 每 N 步准入并非逐项等价；也不要把它与 SSE 的 `stream_interval` 或 Mamba decode tracking 间隔混称。

文章的单请求 chunk 上限不能直接换算为 SGLang 的全局 `chunked_prefill_size`。例如把全局预算缩到 512 可能只让所有请求都变慢；应在实际 admission/continuation 预算上验证。所有修改还受单 partial、page/DSA 对齐和 starvation 约束。

## 3. AMD MLPerf：可迁移的是诊断方法

主文：[AMD MLPerf v5.1 技术复盘，2025-09-09](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inference-v5.1/README.html)。

**VERIFIED / 作者报告**：Llama2-70B Interactive 的 450ms TTFT/40ms TPOT 约束促使他们调优小 GEMM，并使用 vLLM V1 混合 prefill/decode。文中还介绍预编译、graph 和 CPU 开销优化；其他段落涉及离线排序、硬件变化和剪枝，不应合成“相同条件纯软件加速”。

**VERIFIED / 可复现入口**：[官方复现指南](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inference5.1-repro/README.html)提供 MI325X Llama2 的镜像、`interactive_mi325x` 配置入口，以及分开的 performance / accuracy 命令。本文只核对指南，不拉镜像，也没有逐个审阅镜像内部脚本或复現指标。

**INFERRED / 对我们的具体转化**：

- kernel 的形状分布应该取真实运行中每个 rank 的 decode batch 和 prefill 新增 token 数，不是把逻辑会话 N 当每卡 GEMM 的 M。DP 可能使单卡 batch 更小，缓存优化也会改变 prefill 形状；因此应在配置相对稳定后选 shape 调优。
- 先区分 GPU 真正在执行、等待通信、等待 CPU 派发的时间，再决定优化算子还是前端。不能只看 `nvidia-smi` 总利用率。
- 检查 CUDA graph 对真实形状的覆盖、padding 和显存开销；本地 `arg_groups/fields/exec_.py:482` 已有 decode/prefill 分开的 graph batch-size 参数，不能照抄旧版 vLLM 参数名。
- 预热只能消除编译/graph 构建成本；评测要求的 `/flush_cache` 仍必须真清请求缓存，不能借预热保留评测答案或前缀。
- Offline ZigZag/长短交错知道待测批次的全体请求；本赛只能调度已经到达的请求，不能预看未来链或改回放顺序。AITER/ROCm 的二进制算子不直接运行在 A100。
- 不采用删层/微调作为固定模型赛的加速方式；不照抄指南中的宿主机 BIOS、关 ASLR/NUMA 等操作。此类操作会影响共享机器，且不在本次研究授权内。
- `--enable-mixed-chunk` 不能因 AMD 报告受益就默认打开。SGLang 声明默认为 false；与 speculative algorithm 的兼容性检查见 `arg_groups/validation_hook.py:115`，混合 KDA/DSA 仍需独立正确性验证。

## 4. 待验证清单（建议，不是新实验派发）

下表是 **INFERRED / 实验设计**。不执行、不改队列；数值正确性和接口合规是前置门。参考 E2/E2b 的缓存结果不代表完整模型数值通过或 8 卡容量通过。

| 问题 | 最小对照 | 需要的观测 | 在哪里能确认 |
|---|---|---|---|
| D1 与 D2 是否互补 | 相同流量/顺序的 baseline、D1、D2、D1+D2；固定其余参数 | 逐请求实际重算、排队、fast/overall/chain TTFT、TPOT；不可只比命中率 | 替身查机制；完整模型8卡查SLO |
| 快照是否因压力失去价值 | 固定策略，增加活跃链；记录而非先强改 cap | KDA槽峰值、驱逐节点深度、KV余量、状态保留至下一轮的比例 | 替身可查生命周期；满模型查容量 |
| DP容量收益是否被迁移抵消 | DP1/2/4；有合法稳定路由依据后再比较亲和策略 | 每组请求/状态/KV分布、迁移、恢复字节、等待与decode步耗时 | 静态尺寸本地可算；结论须8卡 |
| prefill节拍是否值得开 | 保持D1/D2不变，小范围0/1/2候选 | TPOT变动与新增TTFT等待；无decode可运行时的空轮 | 替身查调度；真实混合负载查收益 |
| 小batch图/算子是否适配 | 固定正确配置后按实测shape选少量候选 | graph miss/padding、CPU间隙、通信/算子耗时及显存 | 2卡算子查sm80；8卡端到端 |

建议沿用已有 smoke / reminder_heavy / strict_append / cold_heavy 数据，不为了模仿外部 benchmark 改输出预算、截 prompt 或伪造 speculative acceptance。新诊断字段应独立记录，不能改变计分口径。

## 5. 最终判断

**INFERRED**：两篇强化了现有 D1/D2 研究的合理性，但没有证明现有补丁已安全、已提升 N@SLO。新增的最有用检查是“快照寿命与容量压力”和“小 batch 的真实执行形状”；prefill 节拍属于已有开关的有条件候选。暂不因阅读文章直接启用 HiCache、mixed chunk、统一池或改成 vLLM。

## 6. 用户追加：不止调参数，能否直接优化代码？

**VERIFIED / 现有产物**：001/004 修改角色快照的 prefill 切分，002 修改等待队列选择和 continuation 预算，000 修改接口/flush；它们已经是推理服务代码优化，并非原生开关搜索。当前没有交付自研 sm80 DSA 或多点 KDA kernel，不能把设计当实现。

**INFERRED / 代码级候选**：

| 层次 | 真正的代码改动 | 能避免的成本 | 主要风险 |
|---|---|---|---|
| 调度 | 选择短新增 prefill、为等待者预留预算、保持长请求进展 | 已命中的短请求仍排在长请求后面 | 饥饿、双partial、预算/对齐与DP同步 |
| 快照与缓存生命周期 | 选中间边界、保留策略、状态槽所有权与驱逐 | 丢失可恢复状态后的重复prefill | state/conv/indexer深度不一致、泄漏或错误命中 |
| KDA内部导出（Plan B） | 同一forward导出role与end两个持久点，贯通元数据、槽位及树插入 | 为取快照额外切一次forward带来的调度/launch及小batch成本 | 多点内存带宽、FP32精度、异步完成和失败清理 |
| 实际sm80算子与通信 | 根据真实形状调tile/warp、合并必要数据移动、减少重复索引工作 | 必须计算部分的算子/launch/通信耗时 | A100支持、数值变化、寄存器/共享内存限制；需先有profile |
| CPU/前端 | 减少长prompt重复处理、序列化和同步等待 | GPU等CPU的间隙 | 不能丢字段、改变内容或伪造计时 |

Plan B 不是空想：本轮再次只读核对 `layers/attention/linear/kda_backend.py:864`，已有单个 `track_state` / `track_chunk_idx` 和FP32 buffer路径；完整分析在[R7](R7_kda_internal_checkpoints.md)。**VERIFIED**的是已有单点入口；**INFERRED**的是把它安全扩展到role+end可能降低切块开销。运行中的final state不是自动保留下来的end checkpoint，不能仅将一个索引换成role后宣布双点完成。

建议按瓶颈区分“减少不必要的工作”和“加速必要的工作”。先证明代码改动改善端到端关键路径，再谈写更底层kernel；这不是要求止步于参数，也不是现在就授权实施上述候选。
