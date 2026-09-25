# 冻结 vLLM 基线与 101 候选独立复核

2026-09-25，Codex。用户要求审查 Claude 的实现，尤其两引擎评测一致与 101 的角色键登记修复。
基线为 `fb18e488`（000 + 010）；101 为单独默认关闭候选，本次生命周期探针固定了其当时的工作区文件副本及 SHA。
本轮没有 GPU 执行，没有 Pod 写入，也没有改 Claude 的引擎源文件。CPU 使用开发机已有 venv，
`CUDA_VISIBLE_DEVICES` 为空，独立目录 `/sjtu/linhang/arena/codex/vllm-review-fb18e488`，不装新依赖。

## 结论

000 接口合同源码复核及 CPU 联调通过，未发现当前无 connector 路径的阻断问题。
010 的硬件分流、缓存布局与接口衔接已审阅；已有单卡测试仍只是对应层级的证据，TP8 真权重/质量/数值恢复未验证。
这允许继续准备既定 TP8 冒烟，不代表 vLLM 已经获得完整性能比较资格。

101 修复“提前返回导致漏登记”的方向正确；独立 CPU 探针又发现**后续 decode 将部分块填满时，会丢失角色键**。
Claude 已在候选内修正登记生命周期；Codex 用同一组六个用例独立复测，跨块和请求完成释放后命中均保持。
两次源码快照及 SHA 分别留证；CPU 元数据缺陷已闭合，GPU 状态与性能收益仍未验证。

## 000：从实际插件响应到判分工具

- `/generate` 原样交给补全渲染器，不走 chat template；`ignore_eos` 与 max_tokens 传递及上游额外 EOS 分支核对无误。
- 流式输出累计真实 `token_ids`，没有用文本长度充当 token 数；MTP 多 token 一次返回允许计数跳变。
- `EngineCoreOutputs.timestamp` 在结果处理后生成，调度器不为普通中间 prefill 发 token 输出；
  `RequestStateStats.first_token_ts` 传到插件，不能与 `scheduled_ts` 混用。
- `fb18e488` 在 ASGI 中间件、读取 body 前打接收时间，修复了原先路由入口的层级差异。
  小勘误已发 Claude：SGLang 接收中间件在 CORS 外，vLLM 插件接收中间件在后添加的 CORS 内；
  二者都早于 body 接收/解析，但不能称全部外层顺序相同，也未测出该开销大小。
- 无 connector 时 reset 检查块池引用并删除全部前缀索引；TP worker 不各自持有这份索引。
  日志与响应来自同一 dict；工具要求本次客户端时间窗、空闲计数、显式 null connector。

独立运行冻结 000 的 **20 项 CPU 接口测试通过**；再将真实 ASGI 路由生成的 JSON 和日志
交给真实 `level_verdict.check_flush`，验证通过。假引擎只替代推理执行，没有替代路由、收据和判分代码。
见 [联调脚本](../../evidence/vllm-review-fb18e488/probe_flush_integration.py)、
[收据与源码 SHA](../../evidence/vllm-review-fb18e488/flush_integration_summary.json)。

这不是全量评分运行；connector 非空仍需 R29 所列的全层失效/异步搬运审计，工具目前明确拒绝。
Claude 对 `19673ef4`、`04b6d100` 的独立复核已回复未发现误拒/误放，详见其
[契约报告](../../research/claude/vllm/contract-review-0925.md)。

## 010：已读到的关键衔接与边界

1. `_use_triton_mqa_logits` 用硬件 `support_deep_gemm()` 选择分支；sm80 拒绝不支持的 MXFP4 索引缓存。
   FP8 Q 的 scale 已合入 indexer weights；Triton 调用不另传 q_scale，不是丢失缩放。
2. 新稀疏 MLA 后端使用实际 top-k 缓冲宽度，包含 kpool 的尾部 token；没有只处理前 2048 列。
   缓存通过真实 stride 的 flat view 转换，不假设所有页紧密连续。
3. sm80 的 FP8 字节编码/解码与原生硬件路径分开；Hopper+ 仍走已有分支。
   Triton BF16 dot 与其他架构的 FP8 MMA 舍入可不同，不能要求两引擎文本逐字相同来判正确。
4. 图捕获、TP8 切分、全模型长上下文、真实输出及 MTP 接受率仍需要已有预留任务确认。
   旧 `8e289cf4` wheel 不含本次 000 修订，部署前必须构建并核验新冻结源码对应的产物。

## 101：他正在做什么，以及已确认的缺陷

机制在当前 prompt 的最后一个 user/observation 角色边界附近，额外保留可复用的 KDA 状态；
不删除 prompt、history/tools，不修改输出预算，不额外向引擎提交一次预热请求。
设角色边界向下对齐 hash 单位后为 B，当前 EAGLE 路径在 B 后退一个单位恢复，所以状态位置为 C。
注意力组需要 B 处的键，KDA 组需要 C 处真正保存的状态，两者不是同一件事。

**用户转述的修复成立。** `_cache_partial_tail_block` 在 prompt 尾端恰为整块等情况下提前返回；
把独立的角色键登记放在函数末尾会漏掉它。
当前移到 `FullAttentionManager.cache_blocks` 中、末尾键调用之后，能绕过这个提前返回。
`Glm5NextIndexerCache` 使用 `MLAAttentionSpec`，与 MLA 一样注册为 `FullAttentionManager`；
`KpoolTailSpec.prefix_cacheable=False`，不应给 scratch 尾缓冲登记这种键。

**新发现：只在 prompt 完成时加键，保不住 decode 的部分块→完整块转换。**
`BlockPool.cache_full_blocks` 在 promotion 时删除该块已有的所有 hash，再写完整块 hash。
而 `_cache_role_boundary_block` 只允许 `num_computed_tokens < num_prompt_tokens <= num_tokens` 的一步重登记；
decode 时不再执行。若角色键与 prompt 尾端位于同一物理块，后续填满该块便会丢键。
KDA 检查点可能仍在，却无法经注意力组匹配到它，最终退回更早位置。

使用真实 `KVCacheManager`、真实分块方法，补上采样 token、decode 分配、CoW 引用释放、完成状态与 owner 释放后的查询。
没有执行 GPU 状态复制；因此这些数是 CPU 缓存元数据命中长度，不是 GPU 恢复或性能结果。

下表为修复前快照，原始证据保留不覆盖。

| 物理块/hash单位 | 共享前缀/末尾长度 | prefill 后命中 | decode 后命中 | 释放 owner 后 |
|---|---|---:|---:|---:|
| 512/32 | 2021/538（prompt 正好整块） | 1984 | 1984（0 步） | 1984 |
| 512/32 | 2200/100 | 2144 | 2144（200 步，未跨块） | 2144 |
| 512/32 | 2200/100 | 2144 | **1536**（300 步，跨块） | **1536** |
| 512/32 | 2021/430 | 1984 | 1984（300 步，角色块已更早填满） | 1984 |
| 512/64 | 2200/100 | 2112 | **1536**（300 步） | **1536** |
| 512/64 | 2120/430 | 2048 | **1536**（128 步） | **1536** |

最后一例覆盖常见的约 430 token 尾段及计划中的 u64，并非只有极短尾段会触发。
见 [复现脚本](../../evidence/vllm-review-fb18e488/probe_role_decode.py)、
[原始 CPU 结果及源码 SHA](../../evidence/vllm-review-fb18e488/role_decode_summary.json)。
现有证据证明“额外缓存收益会丢失”，没有证明错误状态被命中，更不能推定基线输出有错。
Claude 按候选自己的机制修复：该位置计算完成后每次 cache_blocks 都重新登记角色键，已存在时不做事；
没有更改底层 block pool 的 eviction/flush 删除语义。
Codex 在另一独立源码副本重跑相同脚本，六例的 prefill、decode、完成释放后命中均等于各自 role_checkpoint；
最后一例保持 **2048→2048→2048**。
见[修复后独立 CPU 收据及源码 SHA](../../evidence/vllm-review-fb18e488/role_decode_fixed_summary.json)。

另有配置取证缺口：当 env 点名 101 但 prefix caching 关闭时，scheduler 把 token ids 置空，
机制行却按 env 打印 on。Claude 已改为传入实际配置，由 KVCacheManager 校验并拒绝不支持组合；
源码复核成立，专项 CPU 测试由 Claude 报告。不影响默认关闭基线。

## 候选验收还缺哪些证据

- CPU：开启/关闭对照；prefill→decode跨块→完成释放→下一轮；多分叉、抢占恢复、淘汰复用、flush后不可命中；真实 GLM 各组组合。
- GPU：C 处 recurrent/conv 实际保存与 CoW 后未被覆盖；MLA/indexer/MTP 对齐恢复；完整输出与质量。
- 性能：增加停点带来的前向次数、状态内存、重算减少能否在闭环下兑现。CPU 命中更高不算 TTFT 改善。

101 继续默认关闭、单独提交与回退。新 GPU 性能实验仍先与用户讨论，不自动加入 072 或完整 N30。
