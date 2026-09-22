# R3 — 英文社区与厂商的先例调研（D1 快照断点 / 智能体负载）

作者：Claude · 2026-09-22 · 只做调研，没有运行任何东西。中文社区与 GitHub issue 部分由 Codex 负责（`research/codex/R6_prior_art_cn_github.md`）。
标注：V = 读了原文；I = 推断。

## 结论速览

1. **"Agent 框架在 prompt 尾部挂 `<system-reminder>`、下一轮删掉，导致前缀缓存失效"是公开的已知问题。** 已有问题报告和实测数据，但讨论集中在**客户端怎么改**（把 reminder 挪位置），没人给出引擎侧对混合模型的解法。
2. **引擎侧"在 prompt 末尾之前 N 个 token 处存快照"已有先例：TensorRT-LLM 的 `additional_snapshot_offsets_from_end`。** 这说明 D1 的方向是业界认可的做法。我们的**按角色边界（内容感知）落点**目前没找到先例；固定偏移对我们这个负载不够用，因为 reminder 长度在 340–1400 token 之间变动。
3. **vLLM 在"一次 prefill 里、不切分 forward 导出中间 KDA 状态"上已经做成（Kimi K3）**，TTFT 降了 9–25%。这相当于我们的方案 B（同一 forward track 多个位置）已被证明可行，而且比方案 A（切 chunk）更省。
4. **调度与路由**：vLLM 针对智能体负载的调度（`--long-prefill-token-threshold`）和会话粘性路由都有实测收益，可以直接支持 D2 和 D3。

## 1. 尾部 reminder 破坏前缀缓存：公开的客户端侧问题

- **openJiuwen agent-core #795**（V）：`<system-reminder>` 作为最后一条 user 消息在每次请求时追加，下一次请求前删除，于是"没有任何一次请求的 prompt 是下一次的字节前缀"。实测数据：尾部固定 reminder 让 `cached_tokens` 卡在 4,403；去掉后逐轮增长（4,419 → 5,068 → 5,717）；把它挪到 system 消息之后，单次命中率 99%。issue 里还提到一个引擎侧协议 `KV_CACHE_EPHEMERAL_TAIL_METADATA` / `kv_cache_hooks`，即客户端告诉引擎"尾部是临时的"。https://github.com/openJiuwen-ai/agent-core/issues/795
- **anthropics/claude-code #78660、#82563**（标题 V，正文未细读）：task_reminder 在工具循环中途触发、改写已缓存的历史，报告称是"近乎全量重建"。https://github.com/anthropics/claude-code/issues/78660
- **NousResearch/hermes-agent #13631**（标题 V）：自动注入的上下文每 N 轮重建一次，所有做前缀缓存的后端都会失效。https://github.com/NousResearch/hermes-agent/issues/13631
- **对我们的意义**（I）：本赛负载（shrimp ledger、scimaster/biomaster）就是这种形态，而且是冻结回放，客户端不能改，**只能在引擎侧解决**。别人的解法都在客户端，所以引擎侧的对策是差异化空间。

## 2. 混合模型（Mamba/GDN/KDA）快照落点：先例

| 系统 | 机制 | 与 D1 的关系 |
|---|---|---|
| **TensorRT-LLM** `kv_cache_config.mamba_state_config`（V） | `periodic_snapshot_interval`；`additional_snapshot_offsets_from_start`；**`additional_snapshot_offsets_from_end`**（0 表示 prompt 末尾，32 表示末尾前 32 个 token）；`per_conversation` 复用策略下会关掉周期快照，需要显式配置边界 | **最接近的先例**：在末尾前固定偏移处存快照。我们需要的偏移随 reminder 长度变化（p50 340，p95 约 1400，F3），固定偏移要么覆盖不到，要么得多点撒网。https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html |
| **vLLM** `mamba_cache_mode=align` + `--enable-mamba-fine-grained-prefix-cache`（后改名，#57382）（V） | align 模式只在 block 网格上存状态；fine-grained 模式在"共享前缀交汇点"存一份，类似 SGLang 的 branching | 相当于我们的 stock 分叉点快照，解决不了"断点在上一轮末尾之前"。https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/ |
| **vLLM #45238**（V，open，无维护者回复） | 当 align 快照落在请求独有的 token 里时，命中率**静默掉到 0%**（52/64 → 0/64，TTFT 433 → 905 ms）。issue 提议：每请求保留 2–4 份快照、固定全局位置快照、增加"被 Mamba 缺失否决"的计数 | **同类问题的独立佐证**："末尾唯一一份快照落在不可复用区域"。https://github.com/vllm-project/vllm/issues/45238 |
| **vLLM #45939 / #46384**（V，7 月 12 日合入） | 混合模型的部分前缀命中：各 cache group 分别计算命中长度，取能共同覆盖的最短前缀；copy-on-write。多轮基准：第二轮 TTFT 0.134 → 0.095 s | 解决的是"对齐粒度"问题，不是落点问题。https://github.com/vllm-project/vllm/pull/46384 |
| **vLLM AgentX 博客**（V，2026-09-08） | `--interval-based-retention`（每轮保存 prompt 末尾快照）、**Marconi 式 `--selective-retention`**（某前缀第二次出现时才存快照）；称智能体负载命中率 96% 以上 | prompt 末尾快照在我们的尾部改写场景里没用；selective retention 与 branching 类似。https://vllm.ai/blog/2026-09-08-vllm-agentx |
| **vLLM Kimi K3 性能博客**（V，2026-09-13） | **一次 prefill 内部导出 KDA 中间状态**：8K prefill 原来要两次 forward、两次 FlashKDA 调用，改后一次调用处理全部 8000 token，并在第 7680 个 token 处导出状态；TTFT 降 9–25% | **方案 B 的直接先例**：中间状态导出不必切 forward。追踪 issue vllm#50587。https://vllm.ai/blog/2026-09-13-kimi-k3-performance-optimization |
| **SGLang**（PyTorch 博客 "Hybrid Models Meet SGLang"）（V，摘要） | MambaRadixCache：匹配时返回带 Mamba 状态的最深节点；插入时从请求里 fork 一份快照 | 就是我们正在读的 v0.5.20 机制。https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/ |

**更正（据 Codex R6）**：llama.cpp #22929 已经按最后一条 user 消息边界切分并保存 checkpoint，因此"没人按角色边界落快照"的说法不成立。原句保留如下，仅作记录：没有见到任何系统按**角色或消息边界 token**（如 `<|user|>`）落快照；也没有见到针对"尾部临时注入"的引擎侧对策，除了 #795 提到的 ephemeral-tail 元数据协议，但那需要客户端配合，本赛做不到。

## 3. 调度与路由先例（支持 D2 / D3）

- **vLLM AgentX**（V）：`--long-prefill-token-threshold`（例如 512）限制每一步的 prefill token 数，让已缓存的短请求尽快进 batch，P90 交互性约提升 2.3 倍，代价是长请求 TTFT 变高。与 D2（#40024 shortest-prefill-first）思路一致。`--prefill-schedule-interval`：每 N 步才准入一次 prefill，其余步专做 decode（DP 下跨 rank 合并）。
- **会话粘性路由优于负载均衡**（V，同一博客）："many agentic sessions have short inter-turn delays, so the next turn frequently arrives while its prefix is still resident on the previous GPU." 这直接支持 D3 里"DP-attention 必须配会话路由"。本赛请求头带 `X-S1-Session-ID` 和 `X-S1-Routing-Key`。

## 4. 对 D1 设计的影响（建议，待与 Codex 对齐）

1. **D1 的定位**：方向有先例（TRT-LLM 的 offsets_from_end、vLLM #45238 的多快照提议）；**按内容落点**是我们的增量。可以把 TRT-LLM 式的"固定末尾偏移"做成**回退策略**：没有边界 token 时，在 floor64(L−k) 处落点，k 取 512 或 1024（R1 的 L−1536 变体）。
2. **方案 B 的优先级上调**：vLLM 已证明"一次 forward 内导出中间 KDA 状态"可行且更快（TTFT −9–25%），而方案 A 每请求多一次调度轮次，还要处理单 active chunk 约束（Codex §8.3）。建议：方案 A 做功能验证和收益确认；如果收益成立，优先投入方案 B。需要读 vllm#50587 与 SGLang KDA kernel（`layers/attention/linear/kernels/`），评估导出中间状态的改动量。
3. **可观测性**：借鉴 vLLM #45238 的提议，加"被 Mamba 缺失否决的 attention 命中 token 数"计数，与 D0 的 `cached_tokens` 对账一起做。
