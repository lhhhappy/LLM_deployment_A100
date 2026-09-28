# R35 — 长上下文多会话"显存放不下 KV"的业界方案调研（2026-09-26，Sonnet 子代理，fable 浓缩）

完整报告与六份分方向笔记（KV 量化、卸载与预取、序列/上下文并行、前缀缓存、准入与抢占、架构级减负）在会话沙盒 `scratchpad/reports/长上下文 KV 显存扩容方案调研.md` 与 `scratchpad/research_notes/…/`，引用约 100 条来源；本文只留能指导决策的结论。与 [R34](R34_kv_capacity_options.md)（我们这套栈上的落地清单）配套读。

## 一句话

在 8×A100 + SGLang + TP8 + MTP 上，没有一条公开手段能在几天内把有效 KV 容量整倍拉高而不带工程风险；能分层用的杠杆有四类：准入与抢占调度（不加容量，但最便宜、直接命中"接纳时不知未来占用"）；把"轮次之间在想下一句"的会话状态请出 GPU（降低必须同时常驻的会话数，只对 DSA/MLA 的 KV 有效）；DSA/MLA 部分的 FP8 量化（16.55 GB 池约 ×1.6–1.8，A100 无 FP8 张量核心、全部实测在 H100/B200）；消除 MLA 潜向量在 TP8 下 8 份复制的 DCP（天花板最高；**2026-09-28 更新：与 MTP 的兼容已经在我们自己的栈上做完并 8 卡跑通，见下）**。KDA 状态池（9.75 GB）是交叉证实的死角：卸载无正确性方案、量化有损（DAMP 任务分 ±1–2）、精确省略数学上做不到（残差 0.7–32.9%），只能按原生大小硬扛。

## 各方向要点（来源见完整报告）

- **KV 量化**：KIVI 2-bit 峰值显存 2.6× 更小；KVQuant nuq2 在 A100-80GB 上让 LLaMA-7B 跑到 1M 上下文（唯一同型号 GPU 的数字）；QServe W4A8KV4 在 A100 上吞吐 2.38×。SGLang 文档：写入时量化、读时反量化回 BF16，"反量化未与 attention kernel 融合会非常慢"，解码路径有未修复的每层每步 5 次 kernel launch 问题（issue 30815）。vLLM 2026-04 FP8 KV 博客（H100/B200）：解码 ITL 斜率降到 54%，但 head_dim=256 的长上下文 prefill 二次项系数升 1.6×，且 FlashMLA 路径出现系统性偏移、必须校准。MLA 特有：SnapMLA 与 DeepSeek-V3.2 的实现都把 RoPE 分量留 BF16、只量化内容（GLM 的 rope 维为 0，问题更小）。受控实验显示 8-bit KV 让 83% 的贪心解码序列与 fp16 不同（小模型），提示量化会改变 token 选择——本赛判分不比对输出 token（固定预算 + ignore_eos），只看能力题，故可用但必须重跑能力题。
- **卸载到主机**：HiCache 文档明确"要参与计算必须先写回 GPU 池"，主机层只省重算不提高同一瞬间的会话上限；HiCache 不支持线性注意力状态（issue 12826；PR 40310 审阅明确"不是 mamba 状态卸载"）。AttentionStore/CachedAttention（4×A100 实测 26 GB/s，2K token 5 GB 载入 192 ms）是最像我们场景的系统，但它依赖可见的作业队列——我们的回放是冻结数据集，链结构可提前统计（推断，未验证）。搬回耗时估算：平均会话 1.1 GB/卡约 43 ms、最大 5.5 GB 约 211 ms（单会话独占链路；开场并发争用无公开数字）。三大厂商都用固定 TTL + 访问刷新，没有按用户预测回来的生产系统。
- **DCP**：MLA 下只要 TP>1 潜向量就在每 rank 复制一份（vLLM 博客、TRT-LLM issue、Helix RFC 三方一致）；DCP 用 AllGather Q → 本地 attention → LSE 合并，8×B200 上打满显存前的并发从 64 提到 ≥512。SGLang 的 GLM-5.3-Flash DCP PR 30194 因 74 天无更新被关；投机解码只在 Kimi Linear + DSpark 组合声明支持（**上游状态未复查**）。见 R34 §3：我们自己的 115/116 已完成 8 卡验证并解决了 move_kv_cache 的 DCP 感知（`--dcp-size 2` 与 MTP 一起两次正式提交，46757/46758），比这里当时估的"数周以上、MTP 兼容未解"快得多；本地看到的容量收益（KV 池 ×1.7）线上没有对应的墙可解，所以没有带来 chain 判定上的提升。
- **前缀去重**：RadixAttention/APC 解决冗余计算不解决驻留容量（HiCache 设计文档原话）；命中 95% 已见顶。新论文（UNISON、Resident KV Claims）把"活跃会话被当冷数据淘汰"学术化，长链场景 TTFT 降 58–89%（未经同行评审）。
- **准入与抢占**：等待队列不占 KV 是 PagedAttention 类系统的默认属性；chunked prefill 只回答计算调度不回答准入；vLLM watermark 默认 0；PagedAttention 论文"重算开销不超过换出的 20%"。Llumnix 的"虚拟占用量"（排队请求记一个正的未来需求、不做物理预留）可移植到单实例准入；FastServe skip-join MLFQ 峰值 KV 开销可比 FCFS 高 7×；Andes 的"每请求最多抢占 1 次"安全阀值得借鉴；vLLM-LTR 只预测输出长度的相对排名（我们有同链历史，更好预测）。Sarathi 提醒块大小 257 vs 256 让 prefill 慢 32%（配置自查）。
- **有损减负**：Quest/MInference/DSA 的 top-k 只稀疏计算、不减存储（DSA indexer 还要一份额外 K 缓存）；H2O/StreamingLLM/SnapKV 等物理丢弃只在任务分层面"几乎不掉"，无逐 token 保证。

## 对我们的排序（容量效果 ÷ 工程代价）

1. 准入控制与抢占调优（含按同链历史预测输出长度的快车道）：数天，无精度风险，不加容量但少浪费。
2. 链感知地把轮次间会话的 DSA/MLA KV 请出 GPU 并预取（HiCache 现有 L2 改主动"停车场"）：1–2 周；并发搬回带宽是黑箱，需 8 卡实测。
3. FP8 量化 DSA/MLA 内容部分：数周；A100 净收益未知，prefill 可能变慢，需重跑能力题。
4. DCP：天花板最高；MTP 兼容已解决（见上），但本地测到的容量收益不对应线上瓶颈，未带来 chain 提升。
5. KDA 状态精度确认（我们已显式 float32；bf16 待 numcheck，见 R34 措施表）。
6. HiCache 参数与会话感知淘汰保护：小。

## 需要 8 卡实测才能回答的
A100 上 FP8 KV 的净收益（prefill/decode 分开）；FP8 对 MTP 接受率的影响；开场并发下多会话同时搬回的真实带宽与排队；8 卡是否共享同一 PCIe root complex；系统提示词/工具定义在 85k 平均上下文中的占比；"19 路同时活跃"里多少其实是轮次间空档；搬回耗时估算与实测的差距。**已回答（2026-09-28）**：DCP 在 GLM-5.3-Flash + TP8 + MTP 下的正确性与吞吐——8 卡跑通、能力门通过（AIME 28/30、GPQA 178/197）、稳态 TPOT 均值约 42 ms（带 MTP）；见 R34 §3 与 [knowledge.md](../../notes/knowledge.md)。
