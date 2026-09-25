# R29 — vLLM 替代候选：GLM 检查点、host 恢复与公平比较

2026-09-25，Codex。按用户最新要求：保留 SGLang 可提交/可回退栈，新增探索优先 vLLM；
**新的性能实验先与用户讨论配置、预算与判据**。本文是源码研究及本地 CPU 状态探针，不发布 GPU 任务，不改变 071/072 预留。

复核对象为仓库提交 `19673ef4` 中的 vLLM：上游 `a811738a6` + 已提交 000/010。
Claude 已报告 101 的 12 项 CPU 测试通过，正在做底包回归对照；尚未交接。
本文不审定其正确性，也不将工作区未提交代码算作可部署版本。
竞赛合同及已有结果复核见 [评估协议](../../notes/evaluation.md)、
[本轮实现审计](../../notes/reports/contract-implementation-review-0925.md)；调度初查见 [R28](../claude/vllm/R28_vllm_scheduler_vs_R27.md)。

## 1. 统一调度值得比较，但不能预支执行收益

`v1/core/sched/scheduler.py:Scheduler.schedule` 以已计算 token 和待推进 token 为统一状态，
同一步可安排运行中请求和新请求，支持多个 partial prefill。
但它仍先处理运行中请求，等待队首分配失败仍可 `break`；`scheduler_reserve_full_isl` 还影响准入容量。
因此“首次入批前耗时长”不能自动推出换引擎会改善。

更具体的执行边界在 `models/glm5next/common/kda.py`：spec 与 non-spec token 仍拆开处理再合并；
`_forward` 带 `eager_break_during_capture`，host prefill 分支不会直接收入 piecewise 图。
统一 token 调度不等于 KDA 的所有工作融合成一个 kernel，也不等于消除了 Python/launch 固定成本。
优势应由同实际工作量下的 step 时间、混合 batch 组成和真实 TTFT 来证明。
框架默认 A100 预算与我们的启动参数也要分开：`scripts/vllm/serve.sh` 已显式设 8192，不能按默认 2048 分析这份启动器。

## 2. GLM 没接入通用 FlashKDA 内部检查点导出器

在冻结源码中：

- `models/glm5next/common/kda.py:_resolve_kda_prefill_backend` 对 sm80 选择 Triton；FlashKDA 只接收指定的 sm90/100/120、dtype、head_dim 与 bounded-gate 组合。
- `model_executor/layers/mamba/kda_checkpoint.py` 的 `FlashKDAPrefillCheckpointExporter` 只有 `models/kimi_k3/nvidia/kda.py` 调用；GLM 类没有接入。
- GLM 继承 `GatedDeltaNetAttention → MambaBase.get_kv_cache_spec`，不声明内部 prefill checkpoint；`MambaSpec.num_prefill_checkpoint_blocks` 默认 0。
- GLM prefill 用 `causal_conv1d_fn` 更新卷积状态，随后将 `last_recurrent_state` scatter 到相应槽位；缓存位置由调度分块停点决定。

所以不能用“源码里有 KDA exporter”证明当前 GLM 可以在一次长 prefill 中任意导出中间状态；
同样不能据此说 A100 不支持前缀复用。它能复用已正确保留的分块末状态，问题是位置与数量。

迁移 180 的价值需要拆开验证：**算到了可复用位置 → 完整状态被保留 → L1 淘汰前搬到 host → 恢复到一致位置**。
只增加 host 容量无法补出从未保存过的 recurrent 状态；只增加停点又可能多出前向轮次并挤占状态池。
优先复核原生 retention/共享分叉点，再审 Claude 的 101。若未来做 sm80 内部分块检查点导出，
应独立保证 recurrent、conv、DSA/indexer、MTP 所代表的位置一致，不能只把缓存索引挂上去。

## 3. host 容量和命中粒度不是参数同名就相等

SGLang 069 的 host64 是每 rank 预算。vLLM `config/cache.py:kv_offloading_size` 是 TP 组总 GiB；
`v1/kv_offload/cpu/spec.py` 按每 worker 字节、物理副本数和对齐后的 chunk 字节分配。
64 × 8 是前者的全组名义预算换算，**不是建议现在给 vLLM 分配 512 GiB**；须结合真实主机余量、tmpfs、RSS 和状态布局核算。

vLLM 原生搬运器按 group 选择层、保存页的实际字节布局；已有 attention/Mamba 分支和 scratch group 排除路径。
仍需启动收据列出 MLA、索引器、KDA conv/recurrent、draft 的实际组、物理字节、block/hash 单位与可恢复边界。
物理总块数不能直接换成逻辑缓存 token 数。

`offloading/scheduler.py:SchedulerOffloadConfig` 的 `supports_partial_tail` 要求单物理块/chunk、统一 block size、
存在部分 recurrent group、无 EAGLE group、DCP=1 等条件。
因此 **L1 的细粒度命中不保证能原样存入 host**；MTP 路径若包含 EAGLE group，该部分尾块路径关闭。
这不等于完整块不能 offload。对照必须区分 L1 命中与 L1 驱逐后可从 host 恢复的命中。

上游已合入 [vLLM #57145](https://github.com/vllm-project/vllm/pull/57145)，不是只有路线图。
其 [recipe](https://github.com/vllm-project/recipes/blob/main/models/zai-org/GLM-5.3-Flash.yaml) 记录的 GLM 恢复验证为
4×H200、TP4、eager、16k 上下文、8 GiB CPU tier、每 chunk 一块，不能当作我们的 TP8/A100/长链/graph/MTP 证据。

## 4. flush：已传 connector 开关，仍有两个不同缺口

`entrypoints/generate_compat/api_router.py` **已显式 `reset_connector=True`**，这点无需再修一次。
CPU-only offload 的逻辑失效链是：清 host 管理器索引、提高 stale job 阈值、丢弃旧 completion，
随后 worker 在新一步的 `handle_preemptions` 中等待旧搬运。代码有防旧完成回填的机制，不能直接叫“flush 后旧数据回填”。

但成功回执不能直接描述为“所有 DMA 已 drain”：

1. `OffloadingConnectorScheduler.reset_cache` 先收集 `_current_batch_jobs_to_flush`，再清 `_jobs`；
   `EngineCore.reset_prefix_cache` 直接返回 scheduler 结果，没有在此等待 worker barrier。
2. `has_pending_push_work` 检查 `_jobs` 和 manager 工作，没有检查单独剩下的 flush IDs。
   在无其他请求/manager 工作的函数级场景中，reset 后可以同时出现“待 worker fence 非空”和 pending=False；
   在 fence 未被后续 step 消费前第二次 reset 会触发断言。
   是否完整 EngineCore 能走到该状态、是否有其他唤醒路径，尚需集成验证。不能把函数探针写成已复现线上 DMA 错误。

**持久化 tier 是另一件事**：`v1/kv_offload/tiering/manager.py:TieringOffloadingManager.reset_cache`
明确 drain secondary I/O、清 primary tier，故意保留文件/网络等 secondary 数据。
若启用这类 tier，插件当前“Drop every cached prefix, including connector-managed tiers”的说明超出了该 reset 的保证。
仅增加成功日志不能补齐合同；可以先限定经验证的纯 CPU tier，或给全部查询/发布引入一致的缓存 epoch。
尚未选择方案，没有直接删除任何持久数据。

后续启用 connector 时，flush 收据需区分：失效的层、cache epoch、逻辑失效完成时刻、
worker drain 是否实际完成、失败原因；不能虚报 worker 已等待。
Claude 已确认先交基础插件修订及契约审计。双方约定首版结构化响应体与服务端日志完全一致，
包括 success、engine/version、API起止epoch秒、reset_connector、kv_connector、起始在途请求数。
工具侧已实现并经CPU测试：时间属于本次flush窗口，队列空闲，connector显式null才接受；
connector非空暂记工具不支持/INVALID，不能叫官方SLO失败。API epoch秒不是缓存代际编号。
这些源码发现已发给Claude独立复核，本轮不抢写其引擎文件。

## 5. 先做哪些工作，何时才申请性能实验

| 现在无卡推进 | GPU 才能确认，先与用户讨论 |
|---|---|
| 同一冻结正文/token/输出预算/N/gap/预热/判分；计时字段来源审查 | TP8 真权重、能力与长上下文正确性；真实首 token 和累计计数 |
| 原生检查点的生成、保留、命中与 host 可恢复边界；101 默认关闭/边界复核 | MTP 下恢复数值、跨 rank 组一致、真实驱逐后重载 |
| 纯 CPU reset 状态转换、重复调用与 epoch 方案；结构化收据 | DMA 与 graph/异步调度交叠下 flush，冷重载与旧 epoch 不可命中 |
| 运行资源清单：host/GPU/根盘/tmpfs/venv/JIT/日志上限 | 安装前真实容量核验；8 卡实际内存与单步成本 |

CPU 源码探针见 [audit.py](../../evidence/vllm-contract-source-20260925/audit.py) 与
[summary.json](../../evidence/vllm-contract-source-20260925/summary.json)：取已提交方法，依赖使用明确的假对象，
核实 sm80 路径、reset 后待 fence 状态及 tier reset 调用顺序；没有运行完整引擎、真实文件缓存或 CUDA。
输出硬限 64 KiB，本轮结果约 1.6 KiB；无 Pod 写入。

公平比较工具已补成功请求固定输出/token 合同，并允许跨引擎不依赖 SGLang 专有 batch 日志；
17525 条历史记录重判结果不变。后续fb18e488已交接，000的20项接口测试和真实插件收据→判分工具CPU联调通过；
101另发现decode跨块后的角色键丢失，Claude修复后Codex六例CPU复测通过，见[独立复核](../../notes/reports/vllm-frozen-review-0925.md)。
TP8仍待验，当前还没有可信vLLM全量N30对照结果。
三类短场景（冷长 prefill、完整 L1 命中混合负载、L1 驱逐后 host 恢复）是待讨论的机制筛选方案；
其后才是同完整 N30 的全门比较。短测不能替代长链稳态，也不能当正式成绩。
