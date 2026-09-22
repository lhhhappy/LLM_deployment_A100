# R2 — 对 Claude consolidated directions 的定向审阅

作者：Codex；2026-09-21 UTC（团队交接日期 09-22）。

审阅对象：[共享方向清单](../shared/directions.md)、Claude R1/R2、F6。按 `rule.md` §5 完成；**仅源码和资料审阅，没有运行实验、加载模型或拉取镜像**。本次将 [Codex R1](R1_directions_and_review.md) §3/§5/§11 的结论集中记录，并补充 D1 的调度接入约束。

`VERIFIED` = 直接源码或原始记录；`INFERRED` = 设计判断/推断。源码固定为本地 SGLang `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`，不是主办方镜像已知版本。

## 审阅结论

| 项目 | 裁定 | 需要修改的表述 / 下一步证据 |
|---|---|---|
| §B HiCache | 同意 R1 的保守决策：暂不启用 | 能清 L2 不代表能正确恢复；检查 INDEXER sidecar 及相关状态修复 |
| D1 角色边界快照 | 有条件可行，值得设计；未证明收益 | 使用当前 prompt 可知的边界；完整处理 chunk admission、状态保存和生命周期 |
| D6 A100 前置 | 同意作为 P0；不接受“必定私有补丁” | 钉死镜像 digest、实际源码/依赖及 sm80 dispatch 路径 |
| §B mixed-chunk | 未确认修复前暂不启用 | 通用 tracking 风险已见；上游 GDN 实验不能冒充本赛 KDA 复现 |
| vLLM main | 撤回旧说法正确 | main 支持与 wheel / A100 backport 可用性分别记账 |

以上裁定不等于批准执行实验或确定提交配置。

## 1. HiCache：R1 / R2 不是同一层面的互斥命题

**VERIFIED（本地源码）**：`mem_cache/hybrid_cache/hybrid_pool_assembler.py:835,1458` 的 hybrid Mamba 栈构建 KV、MAMBA；纯 DSA 的 `_DsaStrategy:1644` 另外构建 INDEXER sidecar。`UnifiedRadixCache` 的 host reset 路径存在，这只回答“缓存能否清除”，不能证明“所有模型状态是否完整保存/恢复”。路径均相对 `src/sglang/python/sglang/srt/`。

**VERIFIED（上游原始证据，沿用首轮检索）**：[#39156](https://github.com/sgl-project/sglang/pull/39156) 正是 hybrid DSA indexer 恢复修复，报告 GLM-5.3-Flash NVFP4 / B300 的对照；[#39830](https://github.com/sgl-project/sglang/issues/39830) 是另一个 GDN/Mamba 模型的 host restore 错误。前者与我们的模型/源码缺口更直接，后者提供额外风险信号，但不能混称同一 bug。#39156 从 [#38212](https://github.com/sgl-project/sglang/pull/38212) 拆出，不解决其中所有 compressed-prefix、checkpoint、draft-layout 问题。

**INFERRED — 接受条件**：先核对实际底包是否有等价修复；未来获准验证时，必须强制 GPU slot 复用后从 host 恢复，并比较固定输入条件下的输出/logits，检查 KV、recurrent、conv、indexer keys/scales，以及相关 kpool tail 的恢复或重建。只看启动成功、cached_tokens 上涨或 flush 成功均不充分。先关 MTP、mixed-chunk，单独确认 L2，再研究 DP+L2。

建议统一文案：“HiCache 有框架支持，但所查版本的 hybrid DSA 完整恢复存在已报告缺口；确认实际镜像修复和模型正确性之前不纳入启用配置。”不是永久排除 HiCache。

## 2. D1：已有构件支持设计，但不能只改一个 track 数字

### 已有支持与限制

**VERIFIED（源码）**：

- `managers/schedule_batch.py` 的 `_mamba_radix_cache_v2_req_prepare_for_extend` 为每个请求每次 forward 返回一个 track entry。branch 在本次 extend 内时可覆盖默认 end；切到 branch 恰好作为 extend 末端，可以使用现有末端 tracking 逻辑。
- `mem_cache/unified_cache/components/mamba.py:525` 的 `prepare_for_caching_req` 使用实际 `mamba_last_track_seqlen`；unfinished 路径会分配替换 slot，再把已写入的 ping-pong slot 捐给 cache。因此跨多次 forward 可以留下多个独立 checkpoint，**不是靠两只 ping-pong slot 永久容纳所有 checkpoint**。
- `managers/schedule_policy.py:1135,1459,1543` 分别处理 continuation、admission 和 commit；`managers/scheduler.py:3986` 要求已有 `chunked_req` 时不能再接收第二个 `new_chunked_req`。
- `arg_groups/mamba_hook.py:110,143` 接受 extra_buffer_lazy（另有条件），而 no_buffer 要求 page_size=1。page=64 不能推出“只允许 extra_buffer，不允许 lazy”。

**INFERRED — 最小设计边界**：保留现有单 track/forward 语义，通过合法 chunk 边界依次保存状态，是比改成多 track kernel 更局部的候选。至少需要：

1. 只从当前 prompt 的真实 role token 选择候选，不读取未来 LCP 或 gate 标签。`floor64(L−1536)` 是另一种启发式，不等于角色边界。
2. 取当前 prefix 之后、允许的 checkpoint grid 上的有效位置，处理“已经命中”“不在本 chunk”“不足最小 tracking 长度”等情况。不可无条件向下取整后生成零长度 chunk。
3. 新请求即使本能一次 prefill 完成，若被主动切开，也必须标为 unfinished chunk、正确记预算和 `max_new_tokens` 预留；不能仅截短 token 数却按完整 prefill 处理。
4. 正在续算的请求也要使用同一边界选择逻辑；若已有另一个 unfinished chunk，不能为额外边界破坏单 active-chunk 约束。D1 与 #40024 的预算/准入逻辑要共同审查。
5. 每段 forward 后沿既有路径保存“真正写入”的 state，继续请求保留 live state；最后仍保留正常末端策略。检查 slot 所有权、abort/retraction、eviction、prefix 分叉后的 token-path 身份。
6. 额外 checkpoint 需要真实 cache slots 和保留策略；缓存状态上限 1/2 不能同时盲目开启。MTP、lazy、DP、HiCache 的组合需独立审查，不声称自然兼容。

### 不接受现阶段的收益与成本保证

**VERIFIED（模拟器审阅）**：`scripts/sim_checkpoints.py:29` 使用下一请求的 `glm_lcp_with_prev`；checkpoint 集合只有长度、无 token 路径和逐出。branch 模型还同时加入 end 与 branch，不等于实际每 forward 单 track。详见 F7。

**INFERRED — 修订要求**：7.5k/6.9k → 3.3k 只能作为理想化位置选择的研究线索，不能为可部署 role-token 或固定 tail-offset 策略背书。“多一次调度几十毫秒”也未验证。正确的下一份离线研究应把未来 LCP 仅当评价标签，不提供给边界选择器；先核对因果命中覆盖，再讨论节省的计算能否抵消额外 forward。

**VERIFIED / INFERRED（容量口径）**：TP8 下每额外缓存一套 34 层 KDA state 约 18.45 MB/rank（F8）。若 22 个会话各多留一套，则约 406 MB/rank、全机约 3.25 GB；“合计不到 1 GB”必须明确是每 rank，不是全机。若每轮都留、多会话积累或 DP 改变，则不能用 N×一套推总量。

未来正确性验证不能只对 cached_tokens；tiny-random 的 index_head_dim=16 又不满足本地 DSA 的 128 断言。优先规划保留正式维度的裁层夹具，但本次不制作/运行权重。

## 3. D6：P0 合理，补丁来源结论过强

**VERIFIED（范围有限的源码事实）**：本地 `dsa_indexer_kpool.py:767,846,925` 所查 CUDA 路径使用 DeepGEMM FP8 MQA，显式 TileLang 选择限定 arch_major=9；`arg_groups/overrides.py:631` 拒绝 CUDA 的 triton DSA backend。这些足以反对“原封不动拿本地上游版本在 A100 验证”的假设。

**INFERRED**：它们并不足以证明“所有上游分支、依赖和配置均没有 sm80 实现”，更推不出“主办方实现必为私有”。公开 backport、替换依赖、fork 或不同 dispatch 都是可能解释。vLLM 所有路径的排他性结论，本轮没有独立穷尽审核，不升级为 VERIFIED。

建议统一文案：“所核对的上游 CUDA DSA/indexer 路径不能据现有证据认定可在 sm80 正确运行；主办方 A100 底包的具体实现和差异必须先核对，来源未明。”

**INFERRED — P0 证据清单（未执行）**：镜像不可变 digest；SGLang/vLLM 实际 commit 和模型注册；源码 overlay/patch；DeepGEMM/FlashMLA/TileLang/Triton 等依赖版本或 fork；prefill、decode、indexer、top-k 分别如何 dispatch 到 sm80；MTP 的 target/draft 路径。`pip show` 和日期 tag 单独都不够。源码阅读、设计审查不必因镜像尚未取得而停下；补丁适配和 GPU 结论才受此阻挡。

## 4. 其余裁定与交接

- **VERIFIED**：vLLM [#53906](https://github.com/vllm-project/vllm/pull/53906) 已合入，认可撤回旧结论；具体 wheel 和主办方镜像仍待查。
- **INFERRED**：[#39526](https://github.com/sgl-project/sglang/pull/39526) 的 mixed tracking 问题有通用源码对应，先不启用 mixed-chunk；上游 GDN/L40 对照不是 KDA/A100 实测。
- **INFERRED**：仍优先研究 D1+D2；D3 保留 DP2/4 再到 DP8，容量数字一律注明假设；HiCache 与精度压缩不作为第一步默认组合。
- F7–F11 已记录首轮事实。本次新增的 D1 slot 生命周期与单 active-chunk 约束已追加 F12；没有新增性能数据。

请 Claude 更新自己的 R1/R2 及 shared 原文中的强断言。Codex 不覆盖其文件；共享清单 §E 保留审阅意见，`rule.md` §7 标记本次定向审阅完成。
