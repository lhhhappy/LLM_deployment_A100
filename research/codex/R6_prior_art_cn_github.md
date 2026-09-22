# R6 — 中文社区 / GitHub：agent 前缀改写与 hybrid 状态断点先例

作者：Codex。检索/源码核对截止：2026-09-22 UTC。范围：Claude 分工中的中文社区、六个引擎/缓存仓库，以及公开 harness 的 rid 问题。**仅调研；未运行实验、模拟、服务或 harness，未访问 GPU，未拉取镜像。**

## 0. 结论先行

1. **VERIFIED（已合并实现）**：D1 A 有非常直接的先例。llama.cpp [#22929](https://github.com/ggml-org/llama.cpp/pull/22929) 已于 2026-05-25 合并：从模板提取消息边界，在最后 user 消息前切 prefill 并保存 context checkpoint。不能再把“角色边界切 chunk”描述为没有公开实现的设想。
2. **VERIFIED（原作者故障报告）**：Claude Agent SDK [#269](https://github.com/anthropics/claude-agent-sdk-typescript/issues/269) 专门报告本轮注入、下轮剥除 system-reminder 导致历史前缀变化。它与本赛 F3 的机制相近，但不是对本赛数据来源的鉴定，也不是对所有 Claude Code 版本的证明。
3. **VERIFIED（提案/接口/修复证据）**：SGLang 的周期快照、vLLM 的应用指定断点、TensorRT-LLM 的尾部偏移快照均有公开先例；落地状态不同。它们支持探索 checkpoint 放置策略，**不证明我们的 KDA/A100/单 active chunk 实现已经正确**。
4. **INFERRED（证据适用范围）**：本轮没有找到同时满足“GLM-5.x / Kimi-Linear / Qwen3-Next + A100/H100 + agentic 闭环多轮 + 可比较调优前后数字”的完整公开证据。下面保留近邻案例，不把 L40S、H200、随机单轮或 kernel 微基准充当本赛收益。
5. **VERIFIED（本地源码）**：公开 dev 流程会跨 preflight、warmup、measure 复用同一数据请求的 rid；namespace 不同不会改变 rid。正常顺序执行不等于超时/异常时服务端必然清理完成。详见 §6；隐藏正式 runner 未提供，不能外推其具体实现。

### 证据标签

- `VERIFIED / SOURCE`：读到代码、合并记录或官方接口定义；只验证相应静态事实。
- `VERIFIED / REPORT`：确认原作者报告了该现象/数字，**不是我们复现成功**，因果解释仍可能待维护者确认。
- `INFERRED`：向本赛的迁移判断、待检验假说或检索范围内的缺口。
- `closed` 不等于 `merged` 或已修好。重要 PR 使用公开 GitHub API 核对 `merged_at`；不使用自动生成的 PR review 摘要代替 diff。

## 1. (a) agent 改写 prompt 的具体先例

### A1. Claude Agent SDK：最后 user 消息 inject / strip

**VERIFIED / REPORT**：[SDK #269](https://github.com/anthropics/claude-agent-sdk-typescript/issues/269)，报告环境 SDK 0.2.92 / CLI 2.1.92。作者展示本轮 user 内容含 reminder blocks、下一轮历史回放删除这些 blocks 的示例，并称 system/tools 可命中而 message history 反复重建。该 issue 被按 duplicate 关闭，主单 [#263](https://github.com/anthropics/claude-agent-sdk-typescript/issues/263) 仍 open；**不是已修复记录**。

**INFERRED**：这是“上一轮末尾的非稳定内容成为下一轮分叉点”的直接动机先例。但 JSON 的 array/string 形式变化本身不一定改变我们引擎收到的 token 序列；本赛应以已渲染 text 的 tokenizer/LCP 为准。也不能接受 issue 中“统一 array 就修好”的建议作为充分条件：如果 reminder 文本仍被删除，token 前缀仍可能变化。

### A2. Qwen Code：工具发现重建首条历史消息

**VERIFIED / SOURCE + REPORT**：[中文/英文 issue #6265](https://github.com/QwenLM/qwen-code/issues/6265) 指向 `refreshStartupContextReminder()` 重建 `history[0]`，工具发现同时更新 tool schema。[修复 #6420](https://github.com/QwenLM/qwen-code/pull/6420) 已合并（`4c884e47bdcfd7e50e59364fcab9c857ab3b5206`）：将易变 reminder 部分后置，并去掉一次发现工具后的 reminder 重建。

**INFERRED**：这证明实际 agent 工程确实遇到提示词稳定性问题，但它是**早段前缀变化**，不全是 D1 的 near-tail case；tool schema 仍变化时，不能宣称单改 reminder 就消灭所有 miss。本赛不应删除、移动 reminder 或 tools 来套用此应用层修复。

### A3. llama.cpp：agentic 工具负载触发的故障报告

**VERIFIED / REPORT**：[#21383](https://github.com/ggml-org/llama.cpp/issues/21383) 的 Qwen3.5-27B 故障复现描述每轮 system-reminder / 工具结果改变前缀。这是崩溃报告，不是优化成功数字；关闭状态不能证明该机制或全部版本已修复。

### A4. 中文第一人称记录

- **VERIFIED / REPORT**：[linux.do：Claude code 使用 GPT 模型无法命中缓存](https://linux.do/t/topic/2384126)。作者比较两个转发渠道，报告只命中系统段/低命中；先称关闭 attribution 或使用本地路由改善，后续又称效果不佳。保留这个反证，不把中间“已修复”回复摘成结论。没有充分隔离代理、模型端路由与 prompt 差异。
- **VERIFIED / REPORT**：[linux.do：/resume 后缓存失效](https://linux.do/t/topic/1853376)。楼主先转帖，再报告本地分析/脚本验证；回复指向恢复前后 tool 描述插入位置变化。属于 resume 生命周期案例，不能当成每轮 tail rewrite 的独立受控实验。没有执行帖内脚本或二进制修改建议。

### A5. 尚不能背书的泛化

**VERIFIED / SOURCE（仅提案内容）**：[OpenHands SDK #2827](https://github.com/OpenHands/software-agent-sdk/issues/2827) 讨论把 system prompt 分成稳定/动态段及相应 cache 标记；这不等于证明 OpenHands 每轮改写上一条 user 尾部。本轮也未取得足以把**同一种 inject/strip 根因**归到 Codex 所有版本的直接证据。报告不把用户举例中的框架名称自动升级为已验证根因；provider 文档留给 Claude 的英文分工。

**INFERRED：应区分三类输入变化。** 纯追加可以保留此前 token 前缀；历史中部/尾部改写只能重用分叉之前；system/tools 的早段改写可能使尾部 checkpoint 全部无效。标签名叫 `system-reminder` 并不自动造成 miss，关键是最终 token 前缀是否改变。

## 2. (b) 消息边界 / 显式断点 / 周期快照

### B1. 最贴近 D1 A：llama.cpp 已合并的最后 user 边界切分

**VERIFIED / SOURCE**：[PR #22929](https://github.com/ggml-org/llama.cpp/pull/22929) 明确以 agentic coding 响应速度为目标；从 chat template/autoparser 得到消息跨度，在最后 user 输入前切 batch，并在那里保存状态。历史提交 [e98cb517](https://github.com/ggml-org/llama.cpp/commit/e98cb517317bb0edd1eedf3045b24bb8499b3619) 可复核，PR 的 API merge SHA 为 `e2ef8fe42ccef597bfeab901dd6e39589613b71e`，两者不要混写。

**VERIFIED / SOURCE**：检索时 [server-context.cpp 固定版本](https://github.com/ggml-org/llama.cpp/blob/c550d2f60bde72df19fcef1fef627895095b8ba8/tools/server/server-context.cpp#L3520) 仍有以下路径：

- `message_spans.last_user_message_pos()` → 在最后 user 起点停止当前 prompt batch；其他 user 起点受最小间距约束。
- 下一批执行前保存已经计算好的前缀状态；源码说明当前尚未执行的 batch 不属于该 checkpoint。
- 尾部额外切分点使用 `4 + n_ubatch` 与 `4` 的偏移（实际分支还受 `min(n_batch, offset)` 等条件约束）。

**INFERRED**：它是“用调度切分换取可恢复中间状态”的实装先例；不是 SGLang 的可直接移植补丁。llama.cpp 从 chat API/template 获得结构化 spans，而本赛 `/generate` 收到渲染好的 text；我们仍须用正确 token 坐标、floor64、extend 范围与单 active chunk 约束。它也不直接覆盖 `<|observation|>` 策略、KDA ping-pong donation 或 TP/DP。

**VERIFIED / SOURCE + REPORT（不要只看成功侧）**：[#24055](https://github.com/ggml-org/llama.cpp/issues/24055) 报告 hybrid checkpoint 失效；修复方向 [#24035](https://github.com/ggml-org/llama.cpp/pull/24035) 仍 open，涉及 LCP 内合法状态的保留、候选选择及淘汰。另一个 [#26191](https://github.com/ggml-org/llama.cpp/pull/26191) 声称改善长 agent 会话的 RAM cache 选择，但 **closed 且 `merged_at=null`**，不能写成已合并成果。

### B2. SGLang：周期快照与预测性断点

**VERIFIED / SOURCE（提案）**：[#22326](https://github.com/sgl-project/sglang/issues/22326) 提议 prefill 按例如 1024 token 保存状态，区分 prefill/decode 策略，并提出“利用 chunk/cache_unfinished 原型”与“kernel 输出多个状态”的两条路线。issue 已 closed/completed；本轮未取得该具体方案对应的合并实现证据，因此不写成已有可用 flag，也不把示意图当 benchmark。

**VERIFIED / REPORT + SOURCE（open PR）**：[#37198](https://github.com/sgl-project/sglang/pull/37198)，head `227840965bb0ef751dafde45245b87cb5d89fb4a`。用可配置比例在尚无已知 branch 时预先存较早状态；示例长度 6000、系数 0.8 → 4800，使略短的后续共同前缀也能命中。作者报告生产负载 cache hit 约改善 7%，但未给出本赛适用的硬件/TTFT 和相对或百分点定义。

**INFERRED**：固定比例、固定尾部偏移、角色边界、周期网格属于不同放置策略；可作为后续对照，不应混为同一优化。#37198 中“多轮只存末尾足够”的前提仅适合相应 append-stable 负载，不能覆盖 F3。相比全程每 K token，D1 有机会用少量槽定位易变尾部之前，但需测 missed opportunity 与 retained-slot 成本。

### B3. vLLM：应用指定 semantic checkpoint，尚未合并

**VERIFIED / SOURCE + REPORT**：[RFC #55697](https://github.com/vllm-project/vllm/issues/55697) 让应用显式指定共享前缀边界；producer 在边界切 prefill，consumer 等状态 READY，再通过隔离的状态继续。实现分成 config/input [#55873](https://github.com/vllm-project/vllm/pull/55873)、scheduler [#55875](https://github.com/vllm-project/vllm/pull/55875)、GDN kernel [#55876](https://github.com/vllm-project/vllm/pull/55876)，本轮 API 核对三者均 **open、未合并**。性能数字见 §3。

**INFERRED**：显式 breakpoint 是 D1 的设计近邻，但其 marker 提取/剥除协议不应直接加入本赛输入路径；不能把真实 prompt 内容删除后仍称输入不变。我们可以仅在内部 metadata 上记录从当前 token 序列导出的边界，不依赖未来请求，也不要求改 harness。

**VERIFIED / SOURCE（另一种机制）**：[partial-tail RFC #45702](https://github.com/vllm-project/vllm/issues/45702) 区分 hash 匹配粒度与物理状态 block，并讨论 scheduler split / backend 内部导出两种 materialization 方式。当前 [APC 文档](https://github.com/vllm-project/vllm/blob/main/docs/features/automatic_prefix_caching.md) 的 `--enable-mamba-shared-prefix-checkpoint` 又有 align、MTP/EAGLE、match unit 等适用条件；不能将“文档有该 flag”误当成 #55697 的应用指定协议已上线。

### B4. TensorRT-LLM：显式尾部偏移，以及同 block 状态覆盖问题

**VERIFIED / SOURCE**：[KV cache 文档](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/kvcache.md) 暴露 Mamba 周期快照、`additional_snapshot_offsets_from_start/end`；`from_end=[0,32]` 表达末尾与末尾前 32 token 的位置。相关精确位置策略依赖 manager V2 / beam=1 等限制；per-conversation 策略也不是无条件套用所有默认值。

**VERIFIED / SOURCE（已合并 diff）**：[#18724](https://github.com/NVIDIA/TensorRT-LLM/pull/18724) 修复 branch snapshot 覆盖显式 placement：当已有配置点时，不再无条件追加 `prompt_len`，因为同一 tree block 的 recurrent page 可能被较晚状态替换。测试覆盖非零 end offset 保留。

**INFERRED**：这直接提醒 D1：声明“同时保留边界/end”不等于底层一定保留两个独立可恢复状态。必须核实位置、槽位 ownership、tree insertion 和淘汰；也解释为什么不能用 path cap=2 代替保留策略。TRT-LLM 只是策略/正确性先例，不是本轮另起引擎路线的建议。

### B5. LMCache：支持 hybrid offload，不负责凭空生成语义断点

**VERIFIED / SOURCE**：[MP hybrid 支持 #3613](https://github.com/LMCache/LMCache/pull/3613) 已于 2026-06-10 合并，`65c2ae814526289ec9086493e94606c87e88ba1a`。作者明确这些组是 byte-opaque，不能直接套 CacheGen/CacheBlend 或跨 backend 共享；这不是任意位置 KDA resume 的证明。

**VERIFIED / REPORT（未复现）**：[#4984](https://github.com/LMCache/LMCache/issues/4984) 与 open 修复 [#5004](https://github.com/LMCache/LMCache/pull/5004) 讨论 MTP 下把未写入有效 recurrent state 的 scratch block 当持久状态。另有 open [#5251](https://github.com/LMCache/LMCache/pull/5251) 提议按 retention group 延迟 offload，但 lookup 仍须验证 model-wide 完整性。

**INFERRED**：对本赛的价值是“保存/恢复所有必要组件、真实 checkpoint ready 后才可命中”的检查项；不是“接上 LMCache 即修复 D1”。这些 vLLM connector 问题也不能直接归因于 SGLang HiCache。

### B6. Mooncake：GLM-5.3 的 partial-hit 外部复用风险

**VERIFIED / REPORT**：open [#4153](https://github.com/kvcache-ai/Mooncake/issues/4153) 报告 GLM-5.3-Flash0831、8×H20 分为两个 TP4 实例、共享 Mooncake store，在 hash=64 / attention block=1152 的 partial-hit 路径出现错误回复；作者怀疑 external lookup 与实际可恢复 block 粒度不一致。本轮没有验证该根因或 workaround，不把它称为已确证的通用 Mooncake 缺陷。

**VERIFIED / SOURCE（提案）**：[#4255](https://github.com/kvcache-ai/Mooncake/issues/4255) 的 `LastHitOnly` lease 选择提案涉及多 checkpoint 查询只实际读取末端状态的资源成本，属于生命周期/lease 管理，不是角色边界选点算法。

**INFERRED**：细粒度 hash 命中、FA KV 可用、KDA 状态有效、DSA indexer 已恢复，必须同时满足。本赛若以后研究 HiCache/外部存储，不能只看“高 cached_tokens”或 load 成功日志。

## 3. (c) 数字：有哪些，不能外推到哪里

下表均为 **VERIFIED / REPORT：核实作者报告，不是本地复现**。百分比/时延只描述原比较，未归一化为本赛收益。

| 一手来源 | 模型、硬件与负载 | 原作者数字 | 对本题的边界 |
|---|---|---|---|
| [vLLM #55697](https://github.com/vllm-project/vllm/issues/55697) | Qwen3.5 35B，L40S；公共多模态前缀的 1→9 consumer 配对业务 | QPS 5.9→12.4；TTFT 482→248 ms；报告 BF16 最大差异 <1.2e-5 | 语义断点强近邻；不是 A100/H100、不是闭环 agent 多轮；非零 diff 不能称 bit-exact |
| [SGLang #37198](https://github.com/sgl-project/sglang/pull/37198) | hybrid/Kimi-K3 类生产共享前缀；硬件未说明 | cache hit 改善约 7% | 非 Kimi-Linear 的完整目标配置；无 TTFT/分位数/硬件基线 |
| [vLLM #35387](https://github.com/vllm-project/vllm/issues/35387) | Qwen3-Next-80B-A3B FP8，4×H100-80GB，TP4；`bench latency` | 无 MTP 均值 0.894s；MTP 1.578s；MTP+prefix 1.629s | **目标模型/H100 有数，但非多轮**；说明不能默认 MTP 必有收益，不能视为当前所有版本结论 |
| [SGLang #18195](https://github.com/sgl-project/sglang/pull/18195) | Qwen3-Coder-Next FP8，H100 TP2；随机 1024-in/1024-out，100 请求，concurrency=8 | MoE config 调整：输出 382.19→390.63 tok/s；median TTFT 571.22→338.33ms | Coder-Next 邻近变体；不是 agent trace，median 不能替代 fast_intra p95；本轮未把 closed 当 merged |
| [SGLang #37834](https://github.com/sgl-project/sglang/issues/37834) | Qwen3.8-27B，1×H200，TP1；2048-token 分组共享前缀，rate=8/concurrency≤48 | extra_buffer vs ReplaySSM/no_buffer：命中 88.2% vs 65.9%；mean TTFT 226.4 vs 1062.6ms | 非目标模型/硬件；切换同时影响 replay 与缓存策略，作者解释不能升级为已隔离的一般因果定律 |
| [Mooncake #4153](https://github.com/kvcache-ai/Mooncake/issues/4153) | GLM-5.3-Flash，8×H20，双 TP4 共享 store | 报告 external hit 59–65%，同时出现不相关输出 | 是正确性警报，不是成功部署收益；H20 不等于 H100/A100 |

**INFERRED：检索缺口不是“不存在”。** 本轮不足以给出某个公开参数组合能稳过 N=18/22 的承诺。尤其不要将 kernel 加速倍数、一次相同 prompt 的热命中或另一模型随机负载吞吐率映射到本赛固定分类门。上述案例支持我们的验证顺序：状态正确性 → 命中位置 → 槽/调度开销 → 闭环 p95/TPOT。

## 4. 中文渠道覆盖与可访问性

| 渠道 | 本轮取得的材料 | 处理方式 |
|---|---|---|
| linux.do | §1 两条第一人称/跟进记录 | VERIFIED / REPORT；包含作者后续反悔和未隔离变量，不把社区建议当修复证明 |
| 知乎 | [Tair 联手 SGLang 共建 DeepSeekV4 分层缓存架构](https://zhuanlan.zhihu.com/p/2048808646793023819) 的检索摘要命中 unified cache/Mamba 组件；直接抓取超时 | **仅导航线索，未完成正文核验**；不据此证明角色边界策略或任何性能数字 |
| 掘金 | [货拉拉技术：大模型推理加速工程化实践](https://juejin.cn/post/7613919418195804198)，正文可读，作者为团队账号 | VERIFIED / REPORT；真实工程材料但模型/负载/机型条件不满足本题完整交集。未将整体成本下降或推测解码收益作为 D1 证据 |
| 微信公众号 | 对 `mp.weixin.qq.com` 的 SGLang/vLLM、前缀缓存、Mamba/Qwen3-Next 等组合检索，未取得可直接核验且命中核心问题的原文 | 未覆盖到的部分明确留空；转载/搜索摘要不能伪装成已读公众号。Claude 负责的厂商英文博客不重复充数 |
| 飞书 | 对 `feishu.cn` 的相同主题检索未取得可直接核验的相关公开文档 | 没有绕过登录/权限。后续如有人提供公开直链，可继续补证；不能据此判断社区没有内部经验 |

还检索了六个目标 GitHub 仓库的 issues/PR 与 discussion 关键词；采用的强证据主要在 issues、PR、源码，**未找到比上述实现更强且满足目标多轮数字的 discussion**。没有全量抓取或登录搜索，结果不是穷尽清单。

筛选时排除了把 response-result cache 当 KV cache、用删历史/删 tools 换吞吐、缺硬件/模型版本的倍数宣传，以及与冻结模型 config 不符的“GLM-5-7B/32B”等页面。它们不进入方向收益表。

## 5. 给 Claude 的 D1/D2 设计交接（INFERRED，非新实施任务）

1. **D1 A 的先例强度提高**：优先读 llama.cpp #22929 的输入边界来源与切分时机，其次 vLLM #55875 的状态 READY/依赖处理。保留 F19 的 SGLang 单 chunk 检查，不能因为别家有实现就略过。
2. **策略比较要命名准确**：role-only、role+branch、near-tail offset、periodic K 是不同候选。F13 的已有数字仍只属于其模拟定义；不把全新的 role-over-branch 或 cap=2 方案套上同一数字。
3. **恢复点必须是有效前缀状态**：恢复长度是“不超过本次 LCP 且完整状态可用的最近 checkpoint”，不是“知道 LCP 就能直接跳过去”。同一物理 block 上的多个逻辑断点是否真的共存要逐项核对（TRT #18724）。
4. **HiCache / 外部 KV 后置的理由更充分**：LMCache/Mooncake 提供的是存储/传输能力；component 缺失、scratch 当状态、状态位置错配等会造成假命中。D1 本地正确性不依赖先引入这一层。
5. **不改 workload 语义**：不能使用应用层删除 reminder、调整 tool schema、剥除 thinking、缩短历史的“缓存优化”。不向 prompt 添加会被模型看到的 breakpoint marker；仅内部 metadata / 调度策略可以进入候选设计。

本轮没有运行任何上述验收；模型裁层/小模型/两卡验证属于另一个经授权的工作单，不是本报告的执行结果。

## 6. (d) D0 §2.4：rid 是否跨阶段复用？

### 6.1 明确答案与来源

**VERIFIED / SOURCE：会复用，同一 dataset row 的 rid 不含阶段、N 或运行时间。**

- [`s1_common.py:164`](../../s1-dev/harness/s1_common.py#L164)：`rid = pack + ':' + view + ':' + logical_call_id`。
- [`s1_loadgen.py:336`](../../s1-dev/harness/s1_loadgen.py#L336)、`:347`：从 chain 的 `req_ids` 原样放到 `ctx['req_id']`。
- [`s1_loadgen.py:108`](../../s1-dev/harness/s1_loadgen.py#L108)、`:112–114`：body `rid` 和 `X-S1-Request-ID` 都直接使用该 ID；namespace 在另一个 header。

| 阶段（同一 run_dev 调用） | 选择范围 | 发出的 cache namespace | rid |
|---|---|---|---|
| preflight | cohort 第一条 chain（`--max-chains 1`），no-gap | `<tag>-preflight` | 原始 dataset rid |
| warmup | shape-aware 选择的若干完整 chain | `<tag>-jit-warmup`；`-warmup` 由 loadgen 再追加 | 同一 row 仍是原始 rid |
| measure | 完整 cohort | `<tag>-measure` | 同一 row 仍是原始 rid |

源码：[`run_dev.py:187`](../../s1-dev/run_dev.py#L187)、`:209`、`:235`；[`s1_loadgen.py:487`](../../s1-dev/harness/s1_loadgen.py#L487)、`:529`、`:544`、`:592`。warmup 是否恰好选中 preflight 那一条，不影响结论：它们与 measure 的交集 row 必然复用 ID；没有声称三个阶段请求集合完全相同。

### 6.2 为什么正常流程不必然出现 active-rid 冲突

**VERIFIED / SOURCE**：`run_dev._run()` 用同步 `subprocess.run`（`:94–105`）；warmup 和测量分别 join 所有客户端线程（loadgen `:550–551`、`:599–600`）；`call_engine()` 正常读 SSE response 到结束（`:131–144`）。因此 runner 没有主动将三个阶段并行启动。measure 前调用一次 flush（run_dev `:229–232`）。

**INFERRED**：服务端正确结束/移除上一请求时，跨阶段重复 rid 本身不需要规避；但 namespace 改名**不是**服务端 active request ID 的隔离机制。

### 6.3 异常边界：不能用“子进程退出 / 调了 flush”证明已清干净

**VERIFIED / SOURCE**：

- HTTP/连接/timeout 错误在 `call_engine()` 中被转换为结果（loadgen `:145–151`）；warmup 最后仍 `return 0`（`:552–559`）。普通 loadgen 也以记录结果为主，末尾返回 0（`:626`）。因此 `_run` 的退出码为 0 不表示每条请求成功。
- `flush_kv()` 只检查 `urlopen` 是否异常，**不读 JSON success**（run_dev `:46–56`）；其返回值在 `:232` 被忽略。因此这份公开脚本可能在 flush 失败后继续 measure。这里只报告源码，没有修改 harness。
- 客户端超时/断开是否立即触发服务端 abort 与完整清理，不能只由这些客户端文件证明；本次没有将它标为已验证安全。

**INFERRED / D0 建议**：保证请求完成/abort 清理是服务端真实生命周期的一部分；flush 忙时必须诚实失败。未来由外部运行流程核对 flush 状态和各阶段错误，不得把失败的轮次当有效冷启动成绩。若入口时间戳加固需要映射，优先绑定每次 HTTP request 对象/内部唯一实例，不能用仅以可复用 rid 为键且未清理的全局时间表，否则可能污染下一阶段计时。无需为此改 benchmark rid 或原 harness。

### 6.4 “formal”用词范围

**VERIFIED / SOURCE**：这份 `run_dev.py:266–267` 调 scorer 时明确传 `--lane dev`。loadgen 注释中的“正式压测”指 warmup 之后的 measured replay；不是隐藏正式评测 runner 的源码。能确认的是**公开 dev 流程**及同一 loadgen 的 ID 构造；主办方隐藏 runner 是否另包 ID、如何重试，需要其接口说明或源码，不能从该文件推定。

本地只读证据 SHA-256（未运行文件）：

```text
run_dev.py             4e5fdbd479b22ba05173ddeb71b30c108b4bca0fb57993eb0ab91746275b783f
harness/s1_loadgen.py  440fd2589b00a3b891d5c9d1183c96e5dd4a264ffbc7a879d9f3cf1ebc86793f
harness/s1_common.py   a7238b5f1adf728bcccbeeb492f90f7e191dff68e3accac71f29dd8deb939f82
```

## 7. 检索可复核性与停止边界

检索组合包括：`system-reminder + prefix/cache`、`runtime injection + cache`、`Mamba + checkpoint + message/role/boundary`、`Qwen3-Next/Kimi-Linear/GLM-5.3 + A100/H100 + multi-turn/TTFT/prefix`；中文加“前缀缓存 / 角色边界 / 尾部 / 多轮”。分别使用上述中文域名过滤、GitHub 仓库过滤与公开 issues API。对关键 PR 的合并状态单独核对，不依赖搜索摘要。

llama.cpp 当前实现固定到 `c550d2f60bde72df19fcef1fef627895095b8ba8`；其余 `main/dev` 文档是检索时快照，不能保证比赛底包包含。未克隆新仓库、未执行社区脚本、未使用账号搜索、未修改引擎/评测源码。可访问性差的中文原文与目标硬件多轮数字明确标为缺口；不通过扩大到不相干模型来补齐表格。

## 8. R6 定向补充：KDA 内部导出 Plan B

Claude 请求的 vLLM #50587 与 SGLang kernel 深读已单列 [R7](R7_kda_internal_checkpoints.md)。**VERIFIED / SOURCE**：#52789 / #53614 已合并；本地 SGLang Triton 已有每序列单个内部 FP32 累加器快照。**INFERRED**：双点方案应扩展已有路径，工程重点是 conv、真实深度、槽位归属和 radix 插入，而不是先重写 KDA。单点 role replacement 不等于保留 role + end；F24 的保守策略可作为后续双点研究的对照。仍仅调研。
