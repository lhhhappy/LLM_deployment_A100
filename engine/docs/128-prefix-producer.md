# 128p：共享前缀生产者与兄弟准入

2026-09-27，Codex。底座 `741f3eda`（S1 修正 + DCP + K3 + 冷暖冻结）；独立分支 `codex/128-prefix-producer`。目标是减少共享前缀链首的等待，服务排名以 N@SLO 为准。默认关闭；没有修改模型算子、DCP owner 映射或 MTP 提交语义。实现和开发验证见[报告](../../notes/reports/prefix-producer-admission-0927.md)。TP8/N34 收益尚待独立复核和开关对照。

## 启用与兼容

```bash
SGLANG_AX_PREFIX_PRODUCER=1
SGLANG_AX_DEADLINE_TIERS=1
SGLANG_AX_SCHED_PROTECT=1
SGLANG_AX_DEADLINE_FAMILY=0
```

机制行必须出现 `128p=on`、`128=off`；关闭臂要求 `128p=off`。新开关与旧 128 互斥，依赖 124/120、LPM、正常 TP 路径。保留 120 的模式限制，另拒绝优先级调度、122 pace、min-free-slot delayer。支持 125、126、DCP、MTP、HiCache L1/L2。带 session、beam、LoRA、embedding override、多模态输入、已产生输出的请求不参与家族发现，继续原生调度。

参与依赖的请求冻结初次观察到的**绝对 deadline**，起点使用原始接收时间；不因命中增长、停车、回退而重新计时。`SGLANG_AX_DEADLINE_FREEZE_CLASS=1` 仍可作为 741 的独立开关保留在两臂；新机制的参与者无需依赖全局开关才能保持 deadline。

| 新参数（均为 `SGLANG_AX_PREFIX_` 前缀） | 默认 | 用途 |
| --- | --- | --- |
| `PRODUCER` | 0 | 唯一启用开关 |
| `MAX_CANDIDATES` | 64 | 等待候选上限；超限停止新发现，继续处理已有依赖 |
| `MIN_SHARED` | 4096 | 相对消费者已命中前缀，最少新增共享 token |
| `MAX_HOLD_S` | 8 | 一次依赖的最长等待；原 deadline 可更早使其回退 |
| `STALL_S` | 2 | 活跃 partial 无进展上限；等待中的生产者受 MAX_HOLD 约束 |
| `MIN_PROGRESS` | 1024 | READY 同批时为 continuation 留的最低进度，按真实 grid 调整 |
| `MAX_READY` | 4 | 每轮优先尝试的 READY 数 |
| `TRACE_S` / `TRACE_ROUNDS` | 120 / 2048 | 每次 flush 后完整逐决策窗口，两者先到为止 |

短尾上限沿用 `SGLANG_AX_SCHED_SHORT_TOKENS`，并须完整放入当轮；S1 任务为 8192。与 126 同开时，最低进度还尊重 126 的冷块 floor。所有新参数仅在主开关开启时解析。

## 调用链与状态

CodeGraph 定位后逐处核对源码，包括独立的 `SchedulerBatchResultProcessor`：它不持有 Scheduler 的账本，完成日志必须从 Scheduler 的结果分发层进入。

```text
get_next_batch_to_run
  stash_chunked_request -> cache_unfinished_req -> 原生 KV/KDA 检查点发布
  _get_new_batch_prefill_raw
    calc_priority -> match_prefix_for_req -> PrefixReadiness + 原生 held 来源
    rank0: Tracker.prepare -> deadline/生产者/依赖排序
    rank0: _ax_prefix_plan -> READY 资源预览、续块 floor、126 普通短尾预算
    broadcast(version, epoch, sequence, RIDs, seats, budgets)
    _ax_prefix_admit_ready -> 实际 rematch/COW -> 锁内预检 -> 原生 add_one_req
    add_chunked_req -> 普通等待者 -> 准入/拒绝逐决策日志
process_batch_result -> 原生 prefill 结果处理 -> 原生完成时间日志
flush_cache -> 清空账本、pair cache、epoch 与计划序号
```

账本只保留 CPU 数据（请求身份、generation、deadline、前缀字节和长度），不保留 Req、树节点、GPU tensor 或额外 cache lock。原生接收/回收流程没有新常驻 GPU buffer 或额外 KDA snapshot。

每个兄弟最多一个依赖：`WAIT_PREFIX -> READY -> ADMITTED`，或 `FALLBACK`。生产者可从 waiting 转 partial，再转 running/完成；不会因为离开 waiting 就丢失关系。选定生产者不能同时依赖另一生产者，避免环。原生取消、retract、RID 重用、缓存域改变触发 generation/存活核对；producer 消失时，只有消费者**当轮真正匹配到**的可用检查点仍可消费。FALLBACK 不再重复建立等待；后来出现真实 READY 时可以重试短尾准入。

## 发现、排序与 held

使用请求实际 token、`extra_key`、`cache_salt`。哈希只作筛选，逐 token/字节确认真实 LCP；同域字节串排序后，利用相邻 LCP 的区间最小值求所有 pair，避免对每一对重复检查整段 200K prompt。pair 缓存在请求代际变化时更新，失活代际删除。家族仍是**直接生产者到兄弟**的关系，不能用连通分量把深浅不同的 LCP 合并。

预计成本包括到共享位置的剩余 prefill 与兄弟尾巴；只计入预计能在各自 deadline 前完成的成员，使用现有 124 块成本常数。每组按工作量/可完成成员排序；生产者自身能短尾完成时才计入其收益。相同收益优先较短生产者、再按队列顺序，不由随机 RID 决胜。既有饥饿救援保留。此估计不是执行时间保证。

记录 native in-batch LPM 实际插入的代表请求及 matched depth。只有可解释的依赖才解除 held：代表属于该家族，或所选生产者经真实 LCP 核对能够覆盖原 held 的整个共享跨度。后一种情况处理“相同系统提示把深家族挡在无关大请求后面”。无法解释的 held 保留；不会清空整个 held 集合。

## READY 与预算事务

- READY 来自消费者自己的 native `MatchResult.device_indices`；原生组件 validator 已要求有效 KV/KDA 检查点。`full_kv_hit_length`、生产者私有 KV 长度都不能证明 READY。
- FULL、KDA 或 SWA 任一部分需要 host reload，都不进入立即搭车。超过请求 logprob/最后一个 logits token 的匹配也不算；policy 在匹配前应用实际消费端 limit，避免截取一个更长命中长度而虚构 KDA 检查点。
- 不必等精确目标 G：更浅合法检查点若已让消费者能完整短尾完成，即可 READY。不强制改变检查点布局；没有 KDA 状态时继续等待或有界回退。
- rank0 先排序，再预览请求行、页向上取整 token、KV、输出 reserve、共享 Mamba gap/slot 与 continuation 最低进度。最终计划广播，rank 本地时钟不参与决策。
- READY 先做原生 COW/rematch，在持锁后的可淘汰容量上重检，再由 `add_one_req` 真正准入。所有 READY 必须完整结束；`has_chunked_req=True` 同时禁止 101 角色边界再分裂。全流程仍最多一个 partial。
- 每个计划中的 READY，各 rank 对实际匹配元数据作 CPU 共识；不一致共同拒绝该座位。原生提交结果若不一致则一致报错，不带不一致 batch 进入前向。
- 拒绝时清除 deferred COW/clear 元数据，只释放本次新分配的 Mamba 槽；不释放预先持有的状态。仅 READY 事务中的原生 COW 槽不足改为可恢复拒绝，其他路径保持原有断言。拒绝没有扣 adder token 预算，续块同轮使用剩余完整预算。
- 126 的普通短尾 reservation 移到最终排序后，READY 不进入旧的“上轮拒绝就屏蔽”集合。实际 READY 失败后冷 cap 按真实剩余预算重算，不留一个空座压小冷块。124 的有界停车仍生效，停车救援失败同轮恢复 owner。

## 观测和资源账

`[ax-prefix-decision]` 每轮一条 JSON，含完整计划、最多 64 条候选、真实 device/full/host 长度、deadline、held 来源/深度、依赖/READY 事件，以及实际准入、拒绝或停止原因。`[ax-prefix-finish]` 使用引擎原有 `prefill_finished_time`，记录接收、首次准入和完成时刻。它是服务端 TTFT，不能替代 harness 的端到端 TTFT。

逐决策窗口结束打印 `[ax-prefix-trace-end]`，之后每 30 秒保留累计计数；开启日志不加入 CUDA synchronize。窗口只限制新增诊断，不截断原始 harness、服务输出或评分数据。建议 N34 开场使用默认 120 秒/2048 轮；诊断最大约数十 MiB/flush，常见开场远小于此。超出 64 候选时通过 queue 数与 `discovery_bounded=false` 显示停止发现的边界。

无额外持久 GPU 显存。CPU prompt 字节缓存约 `8 * sum(candidate_prompt_tokens)`；合成 64×201024 token 用 102,924,288 字节，另有小量哈希、pair、日志元数据。生产者进入 decode 后释放字节缓存，flush 清空所有账本。更早放入兄弟会改变实际 KV/KDA 活跃峰值，继续受原生资源检查约束，TP8 仍须量峰值。

提取日志：

```bash
python3 scripts/analysis/prefix_producer_report.py server.log --output report-dir
```

它检查决策序列完整性，输出逐兄弟 CSV；不是判分器。正式成绩继续只用原 harness 和完整数据。
