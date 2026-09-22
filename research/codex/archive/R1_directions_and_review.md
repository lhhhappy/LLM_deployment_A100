# R1 — 方向调研与 Claude F3 / R2 交叉审阅

作者：Codex。检索日期：2026-09-21 UTC（GPU 机 / 交接记录为 09-22）。

状态：**仅调研；未部署；不是双方已确认的方案，也没有新的 GPU 性能结论。** 已接受 [rule.md](../../rule.md)。本报告不覆盖 Claude 的文件，待 Claude 复核后再进入 `research/shared/`。

证据约定：`VERIFIED` 指配置、源码、已有产物或上游原始记录可直接支持；不等于本机实测。`INFERRED` 指优化判断、风险推断或条件估算。源码基准固定为 SGLang v0.5.20，commit `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`；不能据此断言主办方镜像版本相同。

## 1. 先对齐方向

**INFERRED — 最有希望的是同时减少额外重算、排队和缓存逐出；现在还不足以宣布 KV 容量是唯一主瓶颈，也不宜直接押注 DP8。** 建议保留以下候选顺序，尚未授权执行：

| 优先级 | 方向 | 想解决的问题 | 当前证据与主要风险 |
|---|---|---|---|
| P0 | 版本、接口与计分口径核对 | 避免耗时评测得到不可解释结果 | 原生 flush 返回格式及 DP 汇总有适配点；dev 少一项正式硬门 |
| P1 | 因果可实现的 KDA 角色边界快照 | frozen LCP 很长，实际可复用状态却更早 | F3 的机制成立；当前收益数字来自 oracle 模拟 |
| P1 | 短 prefill 保护 + 限制长请求续算预算 | 长 chunk 挡住 fast_intra | 上游 #40024 已合入，源码改动集中；仍需检查 hybrid 对齐、公平性 |
| P2 | KDA / KV 池配比、状态保留与 session 生命周期 | 有 KV 却缺可用状态，或 think gap 中逐出 | 不应一开始同时改精度、锁、lazy 和状态上限 |
| P2 | TP8 + DP-attention 2 / 4 | 减少 MLA KV 复制 | 真实容量收益取决于剩余显存、负载均衡和 A100 kernel；不要求先引入 DeepEP |
| P2，有正确性前置 | HiCache L2，之后再与 DP 组合 | 保留超过 GPU 容量的有效前缀 | hybrid DSA indexer 恢复存在源码疑点；启动成功不代表恢复正确 |
| P3 | MTP 深度、CUDA graph、通信后端 | TPOT 与内存之间取舍 | full-model / 8 卡才有代表性，不能只看独立 decode 吞吐 |

以上是 Codex 建议，不是新增实验计划的执行许可。

## 2. KV 容量：用户的方向成立，但模型不是纯 MLA

### 2.1 冻结配置与每 token 成本

**VERIFIED（配置）**：比赛 [config.json](../../s1-dev/glm_tok/config.json) 的架构是 `Glm5NextForConditionalGeneration`：45 层中 **34 层 KDA + 11 层 DSA/MLA**；`kv_lora_rank=512`、rope dim 0、index dim 128、index top-k 2048、`index_kpool=4`、288 专家 top-8、1 层 MTP。配置 SHA256：`bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f`。线上 [官方 config](https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/config.json) 只作交叉参考，以赛题冻结版本为准。

**VERIFIED（源码推导，非运行峰值）**：不含 MTP、页尾和额外 scratch，BF16 DSA 池每个 attention rank 的渐近分配量是：

```
MLA latent  = 11 × 512 × 2        = 11,264 B/token
indexer     = 11 × (128 + 4)      =  1,452 B/token
合计                               12,716 B/token ≈ 12.418 KiB/token
128 Ki tokens                      ≈ 1.552 GiB/rank
```

出处：`memory_pool.py:3876,4798`、`index_key_cache.py:32`、`model_executor/pool_configurator.py:452`，均相对 `src/sglang/python/sglang/srt/`。已有计算产物见 [audit.json](audit.json)。

**VERIFIED（重要区分）**：`index_kpool=4` 是计算/内容压缩语义；当前 `DSATokenToKVPool` 的 `index_buf_size` 缺省仍等于 token 池 size，`IndexKeyCache` 仍按完整物理 span 分配。不能把这个已分配 indexer 成本直接除以 4。尾部 buffer 还需另外计入。

**INFERRED**：如果能安全压缩 indexer 的物理分配，理论上可省约 `1452−363=1089 B/token`，约占上述合计的 8.6%；明显小于 DP 潜在收益，且牵涉寻址、搬移和恢复，不建议优先改。

### 2.2 DP 带来的收益与代价

**VERIFIED（源码 / 官方文档）**：MLA latent 在 attention TP 组内复制；KDA heads / state 按 **attention TP** 划分。因此 `TP=8, DP=D, enable_dp_attention` 时 attention TP 为 `8/D`。这是 MoE 保持大 TP、attention 按请求分组，不是把整个模型复制 D 份的普通数据并行。见 [SGLang DP / DPA 指南（固定版本）](https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/docs/docs/advanced_features/dp_dpa_smg_guide.mdx)、`models/glm5_next.py:306`、`configs/glm5_next.py:264`。

**VERIFIED（尺寸推导）**：FP32 recurrent state + BF16 conv、34 层、一个 KDA slot：

| TP | DP-attention | attention TP | 单 slot / rank | 等剩余显存、理想均衡下的 DSA 逻辑 token 容量 |
|---|---|---|---|---|
| 8 | 1 | 8 | 17.598 MiB | ×1 |
| 8 | 2 | 4 | 35.195 MiB | ×2 |
| 8 | 4 | 2 | 70.391 MiB | ×4 |
| 8 | 8 | 1 | 140.781 MiB | ×8 |

计算包含 `34 × (64/attention_TP) × 128 × 128 × 4` recurrent 和三路 conv 的 `34 × 3 × (64/attention_TP) × 128 × 3 × 2`。来源：`configs/mamba_utils.py:299–374`。每请求可能需要多个 slots，MTP 另加，不能把一 slot 当完整请求状态开销。

**INFERRED（审阅 R2）**：单 rank 的单 slot ×D，并不等于整机同样数量会话的 KDA 总成本也 ×D，因为各 rank 分到的会话约缩至 1/D。但 attention 权重复制、每 worker 预留、vision encoder 复制、分配不均和 scratch 都可能吃掉容量收益。正确的一阶模型是：

```
整机可缓存的逻辑 tokens ≈ D × 每 attention rank 实际分到 DSA 池的字节 / 12,716
```

“每 rank 实际分到的字节”必须在固定精度、MTP、graph、KDA pool 和 backend 后取值。F1 / R2 的“约 1.5M tokens”是基于约 20 GB 剩余池空间的条件估计，**不是已测容量**。N 也是逻辑会话数，不是同时活跃 HTTP 数；不能仅用 N×最大 prompt 长度认定已经溢出。

### 2.3 不要把已修复的 DP 问题当成永久不支持

**VERIFIED（上游报告者记录，非本机复现）**：[issue #36802 的最终更正](https://github.com/sgl-project/sglang/issues/36802#issuecomment-5466045592) 明确称在 8×H20 上解决。挂起来自 vision tower collective，报告使用 `--mm-enable-dp-encoder` 解决；错误输出对应 [#36884](https://github.com/sgl-project/sglang/pull/36884) 的 mHC reduce-scatter 分支，以及 [#36885](https://github.com/sgl-project/sglang/pull/36885) 的 KDA padding 写入 live state。标题中的 GEMM 归因被作者撤回。

**INFERRED**：这足以保留 DP 候选，不能推出 A100 / 本赛镜像 / MTP 已通过。后续应逐项核对两项修复和多模态 warmup 路径；`skip-server-warmup` 不修复文本错误。报告者用的 DeepGEMM + EP 是 H20 配方，不能搬到 A100，也不能据此认为 DPA 必須依赖 DeepEP。

**INFERRED（路由）**：优先考察 DP2 / DP4，之后才 DP8。同 session 的缓存亲和性有价值，但 `hash(session) % D` 不是均衡保证。既有公开 722 请求的离线汇总中，一个固定 blake2b 哈希在 DP8 的 uncached 总量最忙 / 平均为 3.36；它只说明静态哈希可能倾斜，**不是时间轴负载或 TTFT 预测**，也不能据数据挑最优 seed。[audit.json](audit.json)

更合理的研究对象是“新 session 按在线负载选择 + 后续稳定绑定 + 有界 TTL / flush epoch”；`routed_dp_rank` 可在入口传递，不必先造外部代理。多个 tokenizer worker 要共享一致的映射。不要把 session_id 盐化进 prompt / cache key，以免破坏合法的跨 session 前缀共享。

## 3. F3：同意机制，收紧量化结论

**VERIFIED（源码）**：hybrid cache 需要相同前缀上的 KDA state 才能恢复。更准确的说法是“命中截至 LCP 之前最近的可用状态”，不是必须恰好在 LCP。有 FULL KV、没有对应 KDA state，不能直接在 FULL 的最深命中处继续。`unified_cache/components/mamba.py:166` 会从 FULL hit 推出对齐后的 branching point。

**VERIFIED（源码）**：`schedule_batch.py:2913–2986` 每次 extend 的 track entry 只有一个位置；若本次选择 branching point，它会替换原本的 end track 位置。不能将“支持末尾和分叉”解释为“每个 forward 自动保存二者”。跨多个 chunk / forward 可以产生不同快照。

**VERIFIED（对模拟器的审阅）**：[sim_checkpoints.py](../../scripts/sim_checkpoints.py) 的 `boundary` 分支读取 `rs[i+1]['glm_lcp_with_prev']`，明确是下一请求 LCP oracle。其状态表示只是单调累积的长度集合，没有 token-path 身份、逐出、真实 slot 预算；branch 模型同时添加 prompt end 和 branch，亦非上述每 forward 单 track 的精确复刻。

**INFERRED**：F3 中 6918 → 3326 的 fast_intra 重算 p95 是理想化位置选择的提示，不是已实现策略的收益保证。长度集合在发生分叉后还保留旧路径更深位置，因此也不能当作严格上下界。它足以支持“值得研究”，不足以推出 TTFT 降幅或能过 N=22。

**INFERRED（可实现设计）**：只读取当前完整 rendered prompt，识别真正的末尾 role 控制 token，并选取允许的 KDA checkpoint grid 上、不超过该边界的位置。不能读取未来请求、phase / gate 标签或冻结的未来 LCP。仍保留正常 end checkpoint；不要删 reminder 或改 token。

设计审查需要覆盖：

- role token 与消息正文中的相似字符串不能混淆；tokenization 与评测输入一致。
- 边界在 cached prefix 内、本轮 extend 内、chunk 外的三种情况。
- cache page、KDA/FLA state grid、DSA kpool 的共同约束；不能一律假设任意位置可存。
- 切 chunk 增加额外 forward、MoE 通信和排队的成本。
- 替换 track 与另存 slot 的区别；保留边界和末尾需要真实状态生命周期设计。
- context reset / branch 后按 radix token 路径失效；不只按整数长度判命中。
- `mamba-max-states-per-path=1/2` 与额外边界保留可能冲突，不能默认早期状态永不复用。

现阶段只建议补全设计说明；本报告没有实现角色边界补丁。

## 4. R2 调度方向：#40024 值得优先，移植规模可控但不能只抄排序

**VERIFIED（上游 diff）**：[#40024](https://github.com/sgl-project/sglang/pull/40024) 于 2026-09-18 合入 `65ef55e`；本地固定源码尚无该策略。变动为 3 个生产文件共增加 82 行，以及 3 个测试文件；完整 diff 可从 [GitHub files API](https://api.github.com/repos/sgl-project/sglang/pulls/40024/files?per_page=100) 查阅。核心是 uncached-work 排序、对正在续算的长 prefill 预留短请求预算，以及拒绝第二个未完成 chunk（包括 host miss 后重算的情况）。

**VERIFIED（本地源码对照）**：本地已有 `_select_prefill_admission` 以及 host load-back 后重新 admission 的两个调用点，故不是需要先整体迁移调度器才能做。参考 `schedule_policy.py:1135,1345,1459`。本地还有共享 Mamba gap reserve、tile budget 等逻辑，必须保留。

**INFERRED（移植判断）**：代码范围小至中，正确性验证中等；暂未做 patch apply 检查或回归运行。审查至少保留上游五类保护：预算不透支、长请求继续前进、第二 unfinished chunk 不出现、host miss 后重新判断、原策略不改变。额外补 hybrid checkpoint 对齐与状态分配场景；上游所读 prefill-adder 测试中将 `supports_mamba` mock 为 false，不能视为 GLM hybrid 已验证。

**VERIFIED / INFERRED**：[#39717](https://github.com/sgl-project/sglang/pull/39717) 检索时仍开放，作者报告了自己的 A/B 数字；不是本赛数据。优先研究已合入的 #40024，不把两个补丁直接叠加。短请求优先仍要检查长 prefill 饥饿、chain_start 和 TPOT 门，不能以“chain_start 样本少”作为放任饥饿的依据。

## 5. HiCache + DP：先证明完整恢复，再谈组合收益

**VERIFIED（架构资料）**：[SGLang HiCache 博客](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/) 描述 GPU / host / storage 分层、层间传输与计算重叠、write-through/selective 等机制；[Unified Radix Cache 博客](https://www.lmsys.org/blog/2026-08-11-unified-radix-cache/) 则对应多组件缓存及 session-aware retention。它们支持研究方向，不证明 GLM-5.3-Flash 在 A100 上有相同收益。

**VERIFIED（源码路径）**：`hybrid_cache/hybrid_pool_assembler.py` 中 `_MambaStrategy` 匹配 HybridLinearKVPool + FULL/MAMBA；`build_hybrid_mamba_stack:835` 构建 KV、MAMBA 两个 pool，返回路径没有显式 INDEXER sidecar。对照 `_DsaStrategy:1644`，纯 DSA FULL 路径显式创建 `DSAIndexerPoolHost` 并注册 INDEXER sidecar。

**INFERRED（待验证疑点，不是已复现 bug）**：GLM 混合路径的默认 MLA host pool 是否遗漏 DSA indexer key / scale 或 kpool tail 的可恢复状态？这比泛泛问“支持 HiCache 吗”更值得追踪。普通 page_first / layer_first 的 latent 路径没有显式等价于纯 DSA 的 sidecar；仍需继续核对完整恢复 / 重建逻辑，不能仅凭这个 diff 判死刑。

**VERIFIED（其他模型的风险案例）**：[#39830](https://github.com/sgl-project/sglang/issues/39830) 是上游 GDN/Mamba 的 host-tier 错误输出报告，不能当作 GLM KDA 已知必现问题。也不能因为负载强制输出长度就说正确性“不影响本赛”；错误 state / indexer 可能改变执行路径，且能力门始终需要正确结果。

**INFERRED（未来最低证据要求，当前不执行）**：固定 token 序列，分别比较 no-cache、GPU prefix hit、强制逐出后的 host hit；必须覆盖 KDA recurrent + conv、MLA latent、DSA indexer + scale + tail 的恢复，检查输出 / logits 与实际 H2D 事件。仅“cached_tokens 变大”不能证明状态正确。先单拓扑 L2，再 DP2+L2，避免把路由、快照、搬移三种问题混在一起。

**INFERRED（资源预算）**：`hicache-size` 应按当前源码的 per-worker 分配语义累计计算整机 host RAM，并考虑 NUMA、pinned 内存、所有 rank 同时拷贝的带宽；不能直接把单条 1.2 GB / 20 GB/s = 60 ms 当整机恢复延迟。DP 也改变 host pool 复制与分布。主办方 host RAM 未知，暂不推荐具体 100–200 GB/rank。

**INFERRED**：首轮候选只用 L2。L3 与 HiSparse 不属于“顺手打开”的同一优化：L3 有隔离清理语义，HiSparse 的活跃稀疏 KV 工作集 offload 与跨请求 prefix retention 不同；本地 `arg_groups/hisparse_hook.py` 还要求关闭 radix cache，需独立论证。

## 6. 接口与评测：先防止慢评测浪费

**VERIFIED（源码）**：原生 `/generate` 已接收 rendered text 并走 SSE；启用 `--enable-metrics` 后具备所需请求 / prefill 时间字段的原生路径。应保留 tokenizer / scheduler 的真实时间，不要用外部包装器补造时间戳。见 `entrypoints/http_server.py:918`、`managers/tokenizer_manager.py` 的 timing/meta_info 组装路径。未在主办方镜像确认这一版本行为。

**VERIFIED（源码）**：`http_server.py:990` 的 `/flush_cache` 返回文本，而 [task.md](../../llm-challenge-arena-v1/task.md) 要求 JSON `{"success": true}`。更重要的是 `tokenizer_control_mixin.py:301` 等待 DP fan-out 后只取 `[0]`，不是合并所有 worker 成功状态；它可能掩盖其余 worker 的失败。应按全部 worker 成功才成功的语义适配，不可提前回成功。

**VERIFIED（源码）**：scheduler 只有 fully idle 才 reset；UnifiedRadixCache reset 涵盖 host L2 和控制器，但不是 L3 外部存储清空。L3 如将来启用，需要 namespace / epoch 隔离或受控清理本服务数据；不能清空用户共享 backend。

**VERIFIED（题面 / harness 对照）**：正式题面有 11 道硬门，包含 `tpot_p95 <= 0.10s/token`；当前 `s1_common.py:40` 的 GATES 只有 10 道，`s1_score.py:448` 的 PASS 未把 TPOT 算进去，TTFT 也是直接比 p95。故本地 `SERVICE_GATES_PASS` 不能直接当正式 PASS；不要改只读 harness，另行标注正式 TPOT 和统计口径。

**INFERRED（审阅 R2）**：CUDA graph 最大 batch size 不应机械设为 N；N 是逻辑会话数，实际并发 decode、DP 每 worker、MTP verify shapes 均不同。候选配置应由可观察的执行 batch 分布决定。

## 7. NVIDIA / 博客资料能借鉴什么

优先使用工程资料和源码；不把其他模型的 headline speedup 作为本赛收益。

- **VERIFIED（资料）→ INFERRED（应用）**：[NVIDIA Dynamo KV bottlenecks](https://developer.nvidia.com/blog/how-to-reduce-kv-cache-bottlenecks-with-nvidia-dynamo/) 的关键是降低重复计算、使用 host / storage 层和合理路由。对本赛可抽象为比较“恢复 + 等待”与“重算 + 等待”，未必需要引入完整 Dynamo 控制面；2025 博客对后端支持状态的描述不代表 2026 版本。
- **VERIFIED（资料）→ INFERRED（应用）**：[NVIDIA TensorRT-LLM KV reuse](https://developer.nvidia.com/blog/introducing-new-kv-cache-reuse-optimizations-in-nvidia-tensorrt-llm/) 给出 token-range retention priority / duration 与 cache events。可借鉴 session think-gap 的保留时长和可观测驱逐事件；不能据此推断 TRT-LLM 已支持本赛完整 hybrid 模型，或迁移即可提高命中率。
- **VERIFIED（资料）→ INFERRED（应用）**：前述 SGLang HiCache / UnifiedRadix 两篇与本地实现最直接相关。对我们更重要的是完整 state 生命周期、局部快照与 host restore，而不是“KV hit”这一个总数。

## 8. 小模型与裁层：能验证什么，不能验证什么

**VERIFIED（作者模型卡）**：存在 [malaiwah/glm5-next-tiny-random-bf16](https://huggingface.co/malaiwah/glm5-next-tiny-random-bf16)，约 0.836 MiB、5 层 KDA/KDA/KDA/DSA/KDA，独立随机初始化，不是官方训练小模型。作者的证据是 CPU Transformers / 工具链，不是 SGLang GPU 验证；HF 自动生成的 serve 示例不构成支持保证。

**VERIFIED（配置 / 源码）**：此前查阅 revision `e84b6a4844a36baecfaba7dbf0ca3cfea6662d24` 的 tiny 配置 index head dim 为 16，本地 SGLang `DSATokenToKVPool:4853` 则断言其为 128。因此不能直接把这个 tiny 模型当原生 DSA kernel 回归夹具。缩小维度会改变 kernel 可用性。

**INFERRED（更忠实的开发夹具）**：保留正式 head dimensions、MoE experts 和 tokenizer，仅裁前 5 层，可同时覆盖 dense KDA、MoE DSA、MoE KDA。适合未来验证 cache / scheduler / DP 控制路径；不适合证明能力分、长上下文误差或完整模型速度。MTP 需单独版本验证。当前目录里只有 [config-only manifest](fixtures/glm53-first5-config/fixture_manifest.json)，没有可加载裁层权重；计划选取 3958 tensors，涉及 62 个源分片中的 9 个，也不是几 MB 下载。

### 分层验证边界（将来获准后才执行）

| 层级 | 能确认 | 不能确认 |
|---|---|---|
| 现在只读配置 / 源码 | 实际结构、成本公式、API、特性约束、patch 接入点 | 真实可用显存、GPU 正确性、时延 |
| 之后 CPU 离线审计 / mock | 因果边界选择、路径失效模型、全 DP flush 汇总、SSE 合同 | kernel / NCCL / host restore 正确性 |
| 之后 2×A100 忠实裁层 | A100 kernel 路径、小规模 DP、强制 host restore、slot 生命周期 | 完整模型能力、TP8 通信、N@SLO |
| 最后 8×A100 完整模型 | pool 实际大小、正确性门、cache/queue/compute 分解、候选收益 | 单次随机波动下的可靠排名保证 |

**INFERRED（停止条件）**：无对应 cache miss / eviction / queue 证据，不继续扩大该分支；恢复错误先修正确性；短请求改善但长请求 / TPOT 退化，则不能升级到昂贵全档评测。每次只改变一个主要机制。

## 9. 请 Claude 复核的结论与仍待办事项

1. **F3 文案与模拟**：将“需要 AT LCP”修正为“截至 LCP 的最近可用状态”；将 boundary 数字标为 oracle，不再直接称 stock 实测重算或已实现收益。源码同一 forward 的 branch 覆盖 end，也需反映。
2. **R2 容量措辞**：KV binding 尚属假说；补 indexer 物理分配和 KDA/DP 总量视角。DP2/4 保留候选，不因 H20 旧 issue 标题排除。
3. **R2 HiCache**：请独立追踪 hybrid DSA indexer / tail 恢复；在证明前不要把 L2 作为只需打开开关的低风险项。
4. **PD 的排除理由**：权重 index 为 `328,326,771,576 bytes ≈ 305.78 GiB`。不能把“328 GB”直接和“4×80 GiB”相减得绝对放不下；若按 320 GiB 标称仅余约 14.2 GiB，尚未扣 runtime、KV、graph 和实际显存差额，仍很可能不实用。维持低优先级，但理由应是准确容量 / 收益预算，不是混用单位。
5. **底包版本**：`arena-sglang-glm53:260918` 的 tag 日期不是版本证据；registry 认证与各镜像 contents、A100 backend 仍待单独核验。vLLM main 支持已由下文 §11 补充确认；不要把 main 注册、底包 backport、可正确服务三者混为一谈。本报告未完成各镜像内容的独立调研。

## 10. 历史产物状态

本目录早先已有 `audit.py/json`、接口 probe、裁层辅助脚本和两个 patch 草案；它们保留供审阅，**不代表批准实施或已验证服务**。原生时间字段不需要再造。此前启动检查未得到成功的 GPU 服务合同 / 性能结果，相关自有进程已停止；没有正在运行的 Codex 实验。用户要求“先不需要跑”之后，本次对齐仅阅读源码、检索资料、写报告，未继续执行实验。

> 2026-09-22 T8归档说明：上述草案已原样移至[archive/](archive/README.md)，逐项状态见该目录README。此段描述R1产出时的历史状态；后续获批E1的结果另见[实验台账](../../notes/experiments.md)，不追溯改写R1结论。

## 11. 收到 Claude R1 / shared 方向清单后的补充审阅

本节在 Claude 新增 [R1](../claude/R1_model_and_engines.md) 和 [directions](../shared/directions.md) 后补充；更新上文的待核查状态，但保留调查过程。

**VERIFIED（更直接的上游证据）**：[#39156](https://github.com/sgl-project/sglang/pull/39156) 的动机与 §5 独立发现完全吻合：hybrid HiCache 恢复 FULL / recurrent，却缺 `index_k_with_scale_buffer`，修复增加 INDEXER sidecar。作者给出 GLM-5.3-Flash NVFP4、B300 的 host-restoration 对照，PR 检索时未合入。该补丁从较大的 [#38212](https://github.com/sgl-project/sglang/pull/38212) 拆出，不覆盖其全部 compressed-prefix / checkpoint / draft 问题。

**INFERRED（与 Claude 对齐）**：HiCache 暂不进入可直接启用的候选配置；先确认主办方镜像是否已有等价修复，再验证恢复正确性。现有证据已强于“仅有 GDN 风险”，但不能声称我们在 A100 复现过，或整个大 PR 可以未经审查一并移植。

**VERIFIED（上游 + 源码）**：[#39526](https://github.com/sgl-project/sglang/pull/39526) 针对 mixed chunk 合并时丢失 Mamba tracking tensors、却仍按已写入状态捐赠 slot。作者 GPU 对照为 Qwen3.5/GDN、L40，不是 GLM；本地 `schedule_batch.py:3016,3605` 同样有该 merge / 清 tracking 路径。**INFERRED**：赞同暂不启用 `--enable-mixed-chunk`，避免与 D1 新快照逻辑叠加；修复验证要覆盖真正 KDA。

**VERIFIED（vLLM）**：[#53906](https://github.com/vllm-project/vllm/pull/53906) 已于 2026-09-03 合入，merge commit `98ed0856f31fa3aaf5e27464e2b4ef5a8ee6b2f5`。认可 Claude 撤回“main 没有 Glm5Next”。wheel 内容与主办方 A100 backport 仍是不同问题。

**VERIFIED（D6 能确认的范围）**：本地 `dsa_indexer_kpool.py:767,846,925` 的 CUDA 路径依赖 DeepGEMM FP8 MQA，显式 TileLang 分支限定 `arch_major == 9`；`overrides.py:631` 又拒绝 CUDA 的 `triton` DSA backend。**INFERRED**：主办方镜像 / 依赖的 A100 实现必须作为 P0 核对，不能假设本地上游 checkout 可直接用于 A100。但“底包一定带私有补丁”超出了证据：也可能是公开 backport、依赖 fork 或其他 dispatch 路径；未拿到镜像差异，不应定性为私有。

对 shared 清单尚不同意的具体数字 / 断言：

- D1 的 7.5k → 3.3k 仍是 oracle 模拟；“额外调度几十毫秒”没有本模型的证据，应标未知。
- D1 的 `cached_tokens` 相符只是功能必要条件，不能替代 state / logits 正确性；tiny 模型原维度不满足 SGLang DSA 断言。
- “page=64 所以只能 extra_buffer”应写为 extra-buffer 系列；`mamba_hook.py:110` 也显式接受 `extra_buffer_lazy`，只是另有约束。
- D3 的 DP8 约 10M tokens 是条件容量账，不是 verified 启动值；DP2/4 应进入第一批候选。
- 裁层可测 kernel / 控制路径，不宜将每层时延线性乘到 45 层预测总性能；MoE dense/sparse 比例、TP 通信、graph、排队、访存均改变。
- `floor64(L−1536)` 是可因果实现的启发式，但不等于真实 role boundary；应与 role-token 方案分开描述，不能拿 oracle 数字为它背书。
