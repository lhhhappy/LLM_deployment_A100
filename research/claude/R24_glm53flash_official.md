# R24 — GLM-5.3-Flash 一手资料：结构、显存账与对 serving 的含义

2026-09-24，Claude 子代理。只读调研，没有跑 GPU。一手资料的本地副本在 research/papers/（本地归档：`../papers/README.md`）。

**标注约定**
- **【官方原文】**：官方博客、模型卡、技术报告或 SGLang/vLLM 官方文档的原话或表格数字。
- **【按config推算】**：用 `s1-dev/glm_tok/config.json` 加底包源码里的存储格式算出来的。
- **【代码核实】**：在 `build/base_exact/sglang/` 读过源码，但没在 GPU 上跑过。
- **【推断】**：我的推理，需要实测。
- **【上游报告】**：GitHub PR/issue 作者的自述，不是官方发布，也未由我们复现。

**术语**（只解释一次）
- **KV**：注意力层给每个历史 token 存下的"键/值"，给后续 token 查询用。
- **KDA 状态**：线性注意力层（KDA）不存每个 token，而是把全部历史压成一个固定大小的矩阵。
- **快照**：在某个位置把 KDA 状态复制一份存下来。前缀缓存只有在有快照的位置才能接着用。
- **indexer**：DSA 稀疏注意力里的"挑选器"。它给每个历史 token 打分，挑出 2048 个最相关的去做真正的注意力计算。
- **TP8**：一个模型切到 8 张卡上。
- **HiCache**：把 GPU 放不下的前缀缓存挪到 CPU 内存（主机内存），需要时再搬回 GPU。
- **DCP**：decode 上下文并行，把 KV 按 token 分散到多卡存放。
- **MTP**：模型自带的"草稿层"，先猜几个 token，主模型再一次验证。

---

## 1. 来源清单

| # | 来源 | 实际读了什么 | 没取到或未读 |
|---|---|---|---|
| S1 | GLM-5 技术报告，https://arxiv.org/abs/2602.15763 （v2 PDF）；本地 papers/glm5-tech-report-2602.15763.pdf（本地归档：`../papers/glm5-tech-report-2602.15763.pdf`） | 全文抽成文本。精读 §2.1 架构（MLA-256、MTP 参数共享、DSA 续训、高效注意力消融）、§3.2 DSA RL、§3.3/§3.6 rollout 基础设施、DP 亲和路由段（§4.1 内）、§4.2.4 上下文管理、§5 国产芯片推理 | 报告写的是 **GLM-5（744B/40B 激活，80 层，全 DSA）**，不是 Flash。报告里没有 KDA，也没有 Flash 的数字 |
| S2 | GLM-5.3-Flash 官方博客，https://z.ai/blog/glm-5.3-flash ；本地 papers/glm53flash-blog.md（本地归档：`../papers/glm53flash-blog.md`） 和架构图 HyqVZw2wze.png（本地归档：`../papers/glm53flash-blog-img/HyqVZw2wze.png`） | 页面靠 JS 渲染，正文从 JS 包抽取。读了全部文字段落和架构图（含"每层 KV 大小""每层注意力计算"两张曲线） | 基座对比表和各基准图只有图片，没转文字。曲线的纵轴没写单位 |
| S3 | HF 模型卡，https://huggingface.co/zai-org/GLM-5.3-Flash ；本地 papers/glm53flash-model-card.md（本地归档：`../papers/glm53flash-model-card.md`） | raw README 全文 | 模型卡没有架构参数表。WebFetch 给的摘要说"支持 300k 上下文"，这是错的：300k 只出现在 HLE 评测脚注里，config 的 `max_position_embeddings` 是 1,048,576 |
| S4 | Kimi Linear（KDA）论文，https://arxiv.org/abs/2510.26692 ；本地 papers/kimi-linear-kda-2510.26692.pdf（本地归档：`../papers/kimi-linear-kda-2510.26692.pdf`） | 摘要、§3.2 效率、§4 架构（3:1 比例、MLA 不用位置编码）、§5.6 prefill/decode 速度 | 论文**没有讨论前缀缓存或状态复用**。快照语义要看 SGLang/vLLM 文档 |
| S5 | DeepSeek-V3.2（DSA）论文，https://arxiv.org/abs/2512.02556 ；本地 papers/deepseek-v3.2-dsa-2512.02556.pdf（本地归档：`../papers/deepseek-v3.2-dsa-2512.02556.pdf`） | §2.1 DSA 原型（公式 1–2）、§2.3 推理成本、附录 A（MHA 与 MQA 两种计算方式） | 图 3 的成本曲线只看了说明文字 |
| S6 | DeepSeek mHC 论文，https://arxiv.org/abs/2512.24880 ；本地 papers/deepseek-mhc-2512.24880.pdf（本地归档：`../papers/deepseek-mhc-2512.24880.pdf`） | §3.2 系统开销（表 2 访存量）、§4.3 kernel 融合、6.7% 开销 | 训练侧的重算和流水线部分没细读，与 serving 无关 |
| S7 | SGLang cookbook 与部署配置（本地参考版）：`refs/sglang-fe236ea6c3/docs/cookbook/autoregressive/GLM/GLM-5.3-Flash.mdx`、`docs/src/snippets/configs/zai-org/glm-5.3-flash{,-benchmarks}.jsx`；副本在 papers/sglang-docs/（本地归档：`../papers/sglang-docs/`） | 全文 | — |
| S8 | SGLang cookbook 线上版，https://cookbook.sglang.io/autoregressive/GLM/GLM-5.3-Flash ；本地 papers/sglang-cookbook-glm53flash-online.txt（本地归档：`../papers/sglang-cookbook-glm53flash-online.txt`） | 正文 | 交互面板生成的命令没展开。**与 S7 不同**：MTP 从"adaptive 5/1/6"改成固定 5/1/6；DCP 段没有了；新增 breakable prefill CUDA graph（PR #38522） |
| S9 | SGLang HiCache 文档 `docs/docs/advanced_features/hicache_design.mdx`、`hicache_best_practices.mdx`；skill `.claude/skills/compute-mamba-ratio/SKILL.md` | 全文 | `hisparse_guide.mdx` 只看了前提与 KV dtype 段 |
| S10 | SGLang PR/issue 共 19 个（正文，不含评论和 diff），见 papers/sglang-prs-glm53flash.md（本地归档：`../papers/sglang-prs-glm53flash.md`）。重点：#36507（支持，已合并 2026-09-06）、#40915、#40134、#38212、#38474、#39830、#40865、#36830、#37712、#41057、#40433、#40434、#39350 | 正文全文 | #38212 作者的实测评论和 H100 复核评论没抓 |
| S11 | vLLM recipe，https://recipes.vllm.ai/zai-org/GLM-5.3-Flash ；vLLM 混合 KV 管理与前缀缓存文档；副本在 papers/ 下 | 全文 | — |
| L | 本地已有结论：`research/README.md`、`research/claude/base/03-hybrid-cache.md`、`04-model-kernels.md`、`research/claude/R9_upstream_since_base.md`，以及它们引用的 `research/codex/R18_cache_loss_and_capacity.md` §8.3–8.4 | 全文 | — |

---

## 2. 模型结构与显存账

### 2.1 结构要点

| 项 | 值 | 出处 |
|---|---|---|
| 总参数 / 激活参数 | 320B / 18B | 【官方原文】S3 引言；S2 |
| 层数与排布 | 45 层：34 层 KDA，11 层 DSA（层号 3,7,…,43，即每 3 层 KDA 接 1 层 DSA）；另有 1 层 MTP | 【按config推算】`linear_attn_config.kda_layers/full_attn_layers`；博客原话"nearly halves … the number of layers (45 vs. 92)"（S2） |
| KDA | 64 头 × 128 维，短卷积核 4，`gate_lower_bound=-5` | config `linear_attn_config`；cookbook 要求不要改 lower bound（S7 GLM-5.3-Flash.mdx:115） |
| DSA 层 | MLA 压缩 KV `kv_lora_rank=512`，`qk_rope_head_dim=0`（没有 RoPE 部分，与 Kimi Linear 的"MLA 不用位置编码"一致，S4 §4）；indexer 32 头 × 128 维，每次挑 2048 个；`index_kpool=4` 且压缩 | config；博客："IndexPool, which compresses four indexer key vectors into one through weighted pooling"（S2） |
| MoE | 288 个路由专家，每 token 选 8 个，外加 1 个共享专家；前 3 层是稠密层 | config |
| mHC | `hc_mult=4`：残差流是 4 × 4096 | config；S6 |
| 权重 | FP8 e4m3（KDA 投影、indexer、router 等保留 BF16） | config `quantization_config`；本地 04 §1 |

### 2.2 每 token 的 KV（每张卡，TP8）

TP8 下 MLA 的压缩 KV 在每张卡上**完整复制一份**，不按卡切分（04 §1；HiCache 设计文档也写"for MLA … all ranks hold the complete and identical KV"，hicache_design.mdx:155）。下面是每张卡的量，也是整份的量。

| 组成 | BF16 KV | FP8 KV | 标注 |
|---|---|---|---|
| MLA 压缩 KV，每层 | 512 × 2 B = **1024 B** | 512 B + 4 个 fp32 缩放 = **528 B** | 【按config推算】；FP8 布局取自 R18 §8.4（`kv_cache_configurator.py:2472`） |
| indexer K，每层每 token 槽 | 128 B FP8 + 4 B 缩放 = **132 B**（无论 KV 用什么 dtype，indexer 都存 FP8） | 同左 | 【代码核实】`mem_cache/index_key_cache.py` 的 `_buffer_shape` = 页数 × 64 × (128 + 128/128×4) 字节，dtype 为 uint8 |
| 11 个 DSA 层合计 | **12,716 B/token ≈ 12.4 KiB** | **7,260 B/token** | 【按config推算】 |
| FP8 相对 BF16 能多装的 token 数 | — | **1.75×** | 【按config推算】与官方一致：GB300 上"FP8 pool holds 12.6M tokens per rank vs 7.0M at BF16 (1.8x)"（S7 benchmarks.jsx:177）；H20 上 #36830 实测 1.75× |
| MTP 草稿层自己的 KV | 约 +1 个 DSA 层，即 +1,156 B/token（约 +9%） | — | 【推断】草稿层是 1 层 DSA MLA（04 §4），草稿池的大小没核实 |

**新发现：indexer 缓冲约 3/4 没被用到。**【代码核实，未在 GPU 上验证】
1. `compute_pooled_write_locs` 把每 256 个 token 的 64 条合并后的 indexer 行，全写进这 256 个 token 的**第一页**（`layers/attention/dsa/kpool_fp8_index.py:344-356`，`BLOCK_SIZE_K=64`）。
2. 但 indexer 缓冲是按每个 KV 页都分配一页（`index_key_cache.py`）。
3. 所以每 4 页里有 3 页的 indexer 空间不会被写入。
4. 折算下来：每 token 约 1,089 B 空着，占 BF16 每 token 预算的 **8.6%**。

上游 #40134 的描述印证了这种打包方式："GLM pack indexer cache of 256 token into it's first 64 token"【上游报告】。要把这部分空间收回来，得给 indexer 单独做一个按 256 token 分配的池，这会碰到下面 §4.1 的分叉问题。所以它只是一个候选的容量来源，不是一改开关就能拿到的收益。

### 2.3 每个会话的 KDA 状态（每张卡）

【按config推算，与本地 03 §3 逐项一致】
- 每层每卡的状态：
  - SSM 状态：(64/8 =) 8 头 × 128 × 128 × 4 B（fp32）= 524,288 B；
  - 卷积状态：(4−1) × (q、k、v 各 64 × 128 / 8 = 1024 个通道，共 3072) × 2 B = 18,432 B；
  - 每层合计 542,720 B。
- 34 层合计 **18,452,480 B = 17.6 MiB/卡**；8 卡合计 141 MiB。
- 如果把 SSM 状态改用 bf16 存：约 9.5 MB/卡。

**一个状态槽相当于多少 token 的 KV**：18,452,480 / 12,716 ≈ **1,451 个 token**（BF16 KV）；FP8 KV 下约 2,542 个。

**一个正在跑的请求占几个槽**：底包默认 `extra_buffer`、overlap 开时是 5 个槽（`kv_cache_configurator.py:2100-2129`，03 §3）。MTP 的中间状态单独算，每个投机槽约 17.6 MiB × D（04 §4）。skill 文档的说法是"plain spec keeps D"，也就是把 D 并进每请求的槽数（S9 SKILL.md）。本地 04 则认为底包是分开预算的；两种说法我没逐行核对。

**量级对比**【按config推算】
- 一个 100k token 的会话，每卡 KV 约 1.18 GiB（BF16），状态 17.6 MiB × 5 槽 = 88 MiB，KV 占绝大头。
- 250k token 时每卡 KV 约 2.96 GiB。
- 只有上下文短于约 1.5k token 时，状态才比 KV 大。
- 按 skill 的公式 `r* = S·token_equiv/L`，L=100k、S=5 时 r* ≈ 0.07，远低于默认 0.9。skill 建议这种情况直接固定 `--max-mamba-cache-size = 目标并发 × S`（S9 SKILL.md "Procedure" 第 3 步）。本地 S0 已用 `--max-mamba-cache-size 200`（03 §3），方向一致。

### 2.4 indexer 与 mHC 的计算量

- **indexer**：DSA 原文说"lightning indexer still has a complexity of O(L²)"，但头少、可用 FP8，所以便宜（S5 §2.1、§2.3）。【按config推算】kpool=4 之后，每个查询 token 在每个 DSA 层要做 32 头 × 128 维 × (L/4) 次乘加。换算下来每 token 11 层合计约 0.8 GFLOP（36k 上下文）到 5.6 GFLOP（250k），和 04 §7 的估计一致。长上下文 prefill 时 indexer 的临时 logits 显存也会很大：#37712 在 B300 上遇到单次申请 73.65 GiB 导致显存不足【上游报告】，#40854 改成按查询行分块【上游报告，未合并】。
- **mHC**：残差流变成 n=4 路，每层每 token 的残差访存从 2C 读 + C 写变成约 (5n+1)C 读 + (3n+1)C 写（S6 表 2）。论文靠 kernel 融合把训练额外时间压到 6.7%（S6 §4.3）。它不影响 KV，但会增加小 batch decode 的访存和小 kernel 数（04 §1、§7 记录每次前向 90 个小调用）。

---

## 3. 官方的设计意图（原文）

1. **为什么要混合**："For the first time in the GLM series, we introduce a hybrid architecture combining sparse and linear attention, sharply reducing long-context serving costs while preserving precise long-context capabilities."（S2 第 2 段；S3 Introduction）
2. **两种注意力各管什么**："Linear attention captures local dependencies through state modeling, while sparse attention retrieves relevant global context through a lightweight indexer. To further reduce the latency and memory overhead of the indexer at a 1M-token context length, we introduce IndexPool…"（S2 "Architecture for Extreme Efficiency"）
3. **KV 与计算量**："Compared with GLM-5.3, GLM-5.3-Flash uses 3.0× less attention compute and a 4.4× smaller KV cache … The KV cache size is still slightly larger than Kimi-K3 and DeepSeek-V4-Flash, leaving further room for improvement."（S2 同段；架构图右侧标注 4.44× 和 3.01×）
   - 图的纵轴没写单位。按 config 算，Flash 每层平均每 token 约 250–266 B；GLM-5.3 的 config 我没读，所以没能复现 4.4 这个倍数。
4. **官方自己的推理栈**："Our stack combines intra-node tensor parallelism for linear attention and the LM head, ReplaySSM, W8A8 quantization, hybrid INT8/FP8/BF16 cache quantization, and Layer Split … Encode–Prefill–Decode (EPD) disaggregated architecture"（S2 "Serving at Scale on Chinese AI Chips"）。
   - 说明官方生产环境用了状态重放（ReplaySSM）、混合精度缓存和 P/D 分离。
   - 这些是在国产芯片上做的，没有公布 A100 的对应版本。
5. **多轮 agent 的前缀复用**（GLM-5 报告，非 Flash）：
   - 原文："In multi-turn agentic workloads, sequential requests from the same rollout share an identical prefix … all requests belonging to a given agent instance are routed to the same DP rank … As rollout length increases, prefill cost remains proportional to incremental tokens rather than total context length."（S1 §4.1 DP-aware routing 段）
   - 同一会话的请求要落在有它前缀缓存的同一处。
6. **prefill 干扰 decode**："a heavy prefill can preempt or disrupt ongoing decodes on the server … GLM-5, therefore, leverages slime's Prefill–Decode (PD) disaggregation."（S1 §3.6.2）这与本地 R21 观察到的 prefill 干扰 TPOT 是同一类问题。
7. **MTP 的定位**："MTP … is especially effective under the small-batch decoding regime … provides disproportionately large benefits on the long tail"（S1 §3.6.2）。
   - GLM-5 用 3 层共享参数训练 MTP，4 步投机时接受长度 2.76，DeepSeek-V3.2 是 2.55（S1 §2.1 表 2）。
   - 这是 GLM-5 的数字，Flash 没有公布。
8. **上下文管理会打断前缀**：GLM-5 的搜索 agent 用 keep-recent-k；上下文超过 T=32k 时丢掉整段工具历史重新开始（S1 §4.2.4）。模型卡的 HLE 评测也写了"using a context management strategy"（S3 脚注）。【推断】真实 agent 客户端会周期性改写历史，前缀在这些位置断开。本赛题的负载是固定回放，是否有这类断点要看数据本身（R20 的真实 LCP 账本）。

---

## 4. 对 serving 的含义

### 4.1 KDA 状态只能在快照处复用

- **官方语义**：cookbook 要求"Keep the prefix cache enabled for every strategy"（GLM-5.3-Flash.mdx:113）。vLLM 文档写"Mamba state is stored only on the Mamba block grid, so a prefix-cache hit can resume only at a block boundary"（S11 automatic_prefix_caching.md "Hybrid Mamba models"）。SGLang 线上 RFC #40865 描述当前行为是"retreats the *entire* hit to the checkpoint boundary and re-prefills everything after it"【上游报告，与本地 03 §1 代码阅读一致】。
- **本地已核实**（03 §1–§2）：命中长度被截到路径上最深的一个带状态节点。节点之后的 KV 虽然还在显存里，也要重算。
- **上游"只重放线性层"的提议不适用于本模型**【推断】：
  - #40865 提议只让线性层重跑缺口段、全注意力层直接用已缓存的 KV。
  - 但本模型每一层 KDA 的输入依赖前面 DSA 层和 MoE 对**同一批缺口 token** 的输出，而缓存 KV 给不出这些输出。
  - 要精确恢复状态，缺口 token 仍须走完整前向。能省的只是重复写 KV。
  - RFC 引的"93–99.9% 保真度"是 Qwen 类模型的近似，不能当作本模型的精确复用。
- **新风险：256 token 组内分叉时，indexer 行可能被另一分支覆盖。**【代码核实 + 上游报告，未复现】
  1. §2.2 说过，每 256 个 token 的合并行都写在该组第一页。
  2. 设请求 B 在一个 64 对齐、但不是 256 对齐的位置命中了请求 A 的前缀。B 续算时会按自己的页表把后续合并行写进**与 A 共享的那一页**。
  3. `dsa_indexer_kpool.py:470-480` 只要求起点 4 对齐，没有阻止这种写入。
  4. #38212 的描述是同一件事："divergent suffixes can share compressed index rows"。#38474 专门测"shared prefixes at 64/128/192 tokens inside a 256-token compressed-index group"。
  5. 结果是 A 分支后来的读者会拿到 B 的 indexer 键，挑选结果偏移，但不会报错。
  6. 对本负载：同链严格追加时，覆盖写入的内容与原内容相同（只差数值噪声），影响应该很小；真正分叉的边（剔除提示或思考内容后重写）才会写入不同内容。
  7. 本地补丁 101/140 没有处理这件事（grep 未见 kpool 相关改动）。

### 4.2 HiCache（主机内存卸载）是否适用

**官方立场**
- cookbook："Keep **HiCache** off when GPU memory is sufficient … These options remain selectable but are marked **Not Verified**"（GLM-5.3-Flash.mdx:129）。
- 部署面板只把"无 MTP 的 High Throughput + L1+L2"在 H100/H200/B200/B300 上标为 verified（glm-5.3-flash.jsx:659、710、756、802）。
- 这些验证只做了 GSM8K 准确率（benchmarks.jsx 的 H100/H200 注释）。GB300 另有一次速度测试，但注明"the random dataset has no prefix reuse, so L2 benefit was not exercised"（benchmarks.jsx:131）。
- 所以**官方没有证明过主机命中后输出正确，也没证明带 MTP 时可用**。

**我们的底包能否直接用**【代码核实】
1. 底包有 KDA 状态的主机池 `MambaPoolHost`（`mem_cache/pool_host/mamba.py`）。本模型会走 `_MambaStrategy → build_hybrid_mamba_stack`（`mem_cache/hybrid_cache/hybrid_pool_assembler.py:708-803,1307-1359`）。
2. 这条路径**只挂了 KV 和 KDA 状态两个池，没有 indexer 的附属池**。只有纯 DSA 模型的 `_DsaStrategy` 才挂 `DSAIndexerPoolHost`（同文件 :1497-1560）。
3. 上游 #40915 确认了后果【上游报告】："Hybrid Mamba plus DSA models (GLM-5.3-Flash) restore KV from host without the indexer on main, so a host hit runs attention on stale block selection. Host restore KL against warm is 0.06 to 0.71 on main."；修复后 KL 为 1.6e-3–3.5e-3。
4. #40134、#38212 从不同角度修同一问题，三者**都未合并**。
5. 另有 #39830：GDN 混合模型在 main 上"20 of 20 host-tier hits wrong"【上游报告，不是本模型】。
6. 结论：在底包上直接打开 `--enable-hierarchical-cache` 会让从主机恢复的请求输出错误。要用，至少得移植 #40915（或 #40134/#38212）并自己做 KL 与能力验证。

**搬回来 vs 重算的量级**【推算，未在 pod 上测】
- A100-SXM4 到主机是 PCIe Gen4 x16：单向理论约 32 GB/s，实际常见 20–25 GB/s（公开规格，本 pod 未实测；两卡是否共用一个 PCIe 交换机未知，共用则减半）。
- 100k token 每卡要搬 1.18 GiB（BF16）。按 12.5–25 GB/s 算约 **0.05–0.1 s**；KDA 状态 17.6 MiB 不到 1 ms。
- 本地实测冷 prefill：190k token 需 15.44 s（knowledge.md:22），约 1.23 万 token/s，所以重算 100k 约 **8 s**。
- 两者相差约 **80–160 倍**，搬回来远快于重算。HiCache 还会让搬第 N+1 层与算第 N 层重叠（hicache_design.mdx:150-153）。

**代价与限制**【推断】
- **主机内存**：MLA 的 KV 每卡一份相同副本。L2 主机池按卡分配，每卡要存自己那份，所以 8 卡的主机内存需求是单卡的 8 倍。例：每卡 64 GB 约存 540 万 token，共需 512 GB 主机内存。pod 的主机内存没有记录，要先查。
- **需要状态也被卸载**：只有快照节点能命中（§4.1），所以主机层也要保存这些位置的 KDA 状态（`MambaPoolHost` 负责），而且要和 KV 取交集（#39267 等）。
- **收益上限**：HiCache 只救"因显存淘汰而丢掉的前缀"。能救多少先看本地 R18/R20 的淘汰与真实 LCP 账本（R20 在 026/N18 测得可用 LCP 缺口占实际 prefill 的 8.0%）。在账本证明淘汰是主要损失之前，它只是"未测试"的方向。
- **赛规**：`/flush_cache` 必须真清；底包 `reset()` 会清主机池（03 §5），开启后要复核。

### 4.3 FP8 KV 在 sm80（A100）上能不能用

- **官方**：
  - "FP8 KV cache by default on Blackwell, BF16 KV cache on H100, H200"（GLM-5.3-Flash.mdx:72）。
  - "TileLang DSA with FP8 KV is not a valid CUDA combination"（:119）。
  - vLLM recipe："Hopper does not support FP8 KV cache for this model and must run BF16 KV"（S11）。
- **上游 #36830**【上游报告】：`index_kpool>1` 把 `flashmla_kv` 排除在外，而允许的 fa3/tilelang/trtllm 都没有"bf16 查询 × fp8 KV"的 CUDA 路径。trtllm 只支持 Blackwell（#40286 标题写明"trtllm SM100-only"）。
- **结论**：A100 上**没有任何现成后端**支持本模型的 FP8 KV。要用必须自己写"FP8 存储、读入时反量化成 BF16 计算"的 tilelang 稀疏 MLA kernel。这与 R18 §8.4 的判断一致。容量收益按 §2.2 是 1.75×，不是 2×；需要能力门与长上下文 logits 验证。

### 4.4 DCP 的官方支持范围

- **官方**：
  - refs 版 cookbook："DCP4, validated on 4x GB300 (TP4/EP4) … Other platforms and attention backends are unvalidated, and draft-extend v2 is unsupported under DCP"（GLM-5.3-Flash.mdx:125）。
  - 部署面板对其他硬件写"DCP is validated only on 4x GB300 TP4/EP4 for now"（glm-5.3-flash.jsx:129）。
  - 线上最新版 cookbook 已**没有 DCP 这一节**（S8）。
- **官方基准**：DCP4 + FP8 KV 在 1k 输入 / 256 输出、并发 16 下，吞吐比不用 DCP 低约 10%（benchmarks.jsx:64）。这个测试负载不考 KV 容量。
- **上游现状**【上游报告】：
  - #40433 恢复本模型的 target-only DCP，修复"global KV locations addressing rank-local buffers"（#36886）。它**复制 indexer、只切 MLA KV**，并在启动时**拒绝 DSA + DCP + 任何投机解码**。
  - #40434 在其上加 MTP 支持，只做了 CPU 替身测试，"GPU/e2e and GSM8K/AIME reruns remain pending"。
- **与本地的关系**：
  - "indexer 复制、只切 MLA"与 R18 §8.3 的容量上限（D8 约 4.45×，不是 8×）一致。
  - 正式 A 带 MTP，所以 DCP + MTP 在上游也还没验证过。本地 115 的 TP8 行数错误（knowledge.md:22）属于同一类地址问题。

### 4.5 MTP 的官方推荐配置

- refs 版 cookbook 的 Low Latency："adaptive MTP 5/1/6"，即 `--speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 --speculative-adaptive`（GLM-5.3-Flash.mdx:25；glm-5.3-flash.jsx:300-306 以及各 low-latency cell）。
- 线上最新版改成**固定深度** 5/1/6，没有 adaptive（S8："drafts from the checkpoint's MTP head at a fixed depth (5 steps, top-k 1, 6 draft tokens)"）。
- High Throughput 则完全关掉投机解码。官方建议 agent/chat 负载从 Low Latency 起步，大批量负载要实测 High Throughput（GLM-5.3-Flash.mdx:95）。
- 官方的 GB300 速度基准用的是 `SGLANG_SIMULATE_ACC_LEN=3`（模拟接受长度）："Simulated accept length makes this a throughput-mechanism number"（benchmarks.jsx:22）。所以**官方没有公布 Flash 的真实接受长度**。唯一的真实数字是 DCP 那一行"3.937 accept length"（benchmarks.jsx:64，1k 随机输入）。
- **与本地的关系**：
  - 正式 A 用的是 3/1/4、不开 adaptive（`scripts/pod/jobs/cal_offA_n14.sh:7`）。
  - 底包支持 `--speculative-adaptive`（`server_args.py:2270`）。
  - "5/1/6 或 adaptive"对本负载是**未测试的单变量方向**，不是已知收益。草稿更深会加重 verify 成本和 D 份中间状态（每份 17.6 MiB）。PD 模式下投机解码目前起不来（GLM-5.3-Flash.mdx:268）。

---

## 5. 与本地已有结论的冲突与补充

| 本地结论 | 官方或上游资料 | 判断 |
|---|---|---|
| 03 §3：indexer K+scale "~132 B, **unverified exact layout**" | `index_key_cache.py` 缓冲 = 64 × (128+4) B/页，uint8 | **补充**：132 B 已在代码中核实。另外发现 kpool 打包使约 3/4 的 indexer 缓冲不被写入（§2.2），约占每 token 预算 8.6% |
| 03 §1–§4 列出的缓存浪费里没有 indexer 行共享问题 | #38212、#38474、#40134 | **补充**：256 组内分叉会覆盖共享页的 indexer 行（§4.1），是正确性风险，不是容量问题；需在本地核实 |
| 03/04 没有分析 HiCache | S7、S9、#40915、#39830 | **补充**：底包的混合主机池缺 indexer，打开即出错；官方对主机命中和 MTP 组合都没验证（§4.2） |
| 04 §4：MTP 默认 (3,1,4)，正式 A 用 3/1/4 | 官方推荐 5/1/6（refs 版 adaptive，线上版固定） | **补充**：这是未测试方向；官方基准的接受长度是模拟的，不能当收益 |
| R18 §8.4：FP8 KV 7,260 B/token、1.7515×；sm80 需要写 kernel | GB300 实测 1.8×；#36830 H20 1.75×；cookbook"TileLang + FP8 KV 不合法" | **一致**，并由官方和上游证实 |
| R18 §8.3：DCP 下 indexer 复制时上限约 4.45× | #40433 同样复制 indexer；官方 DCP 只在 GB300 验证过，且吞吐 −10% | **一致**；另补充：上游启动时拒绝 DSA+DCP+投机解码 |
| R9 §3：BF16 KDA 投影融合是候选，不能照抄上游收益 | #39350（未合并）在 H20 上确认轮吞吐 +1.6%–2.7%、首轮 +14%；GSM8K 非劣性检验结论为"INCONCLUSIVE" | **一致**；上游自己也没证明质量无损 |
| R9 §2：上游 vLLM recipe 以 Hopper 为前提 | S11："supports NVIDIA Hopper and newer GPUs" | **一致** |
| 04 §7：indexer 约 0.8 GFLOP/token（36k）到 6（257k） | 按 kpool=4 重算得 0.8–5.6 | **一致** |
| 模型卡摘要的"300k 上下文" | config 1,048,576；300k 只是评测设置 | 纠正 WebFetch 摘要，本地文档没有这个错误 |
| R21：prefill 干扰 decode 的 TPOT | GLM-5 报告 §3.6.2 用 PD 分离解决同一问题 | **方向一致**。SGLang 的 PD 对本模型仍是 preview：只用假权重验证过，不能开投机解码（GLM-5.3-Flash.mdx:211,268） |

## 6. 没能核实的事

- Flash 真实的 MTP 接受长度，以及 adaptive 对本负载的效果（官方没有公布）。
- 博客 4.4× KV 曲线的单位和计算口径。
- 底包 MTP 草稿层 KV 池的实际大小；D 份中间状态在底包里到底怎么预算（skill 与 04 说法不同）。
- pod 的主机内存容量和 PCIe 拓扑（两卡是否共用交换机）。§4.2 的 0.05–0.1 s 是按公开规格推算的。
- §4.1 的 indexer 行覆盖和 §2.2 的 3/4 空置，都只读了代码，没在 GPU 上 dump 缓冲验证。
- #38212 作者的 H100 实测、其他 PR 的评论区，都没抓取。
