# 审查：Codex 的共享前缀生产者/兄弟准入（codex/128-prefix-producer，c0fcd486，底 741f3eda）

2026-09-27 03:00 UTC，fable 委派的独立审查代理（只读；CPU 复现用 scratch 克隆）。**结论：对 TP8 N34 OFF/ON 开场探针不构成阻塞**；默认关闭时执行路径与 741f3eda 相同；就绪判定、COW 回退、预算和多卡一致性没有发现正确性缺陷。两条设计上限（1、2）大概率压低 S1 设置下的 ON 收益，要从 ON 日志里读；作者的“144 项 CPU 通过”在该提交上是 143/144。

## 发现（按严重度）
1. **续算旁边的 READY 座位在 S1 设置下很少真正生效**（有效性；用真实调度器方法在 CPU 复现）。计划遍历遇到排名更高、可运行的 device 短命中就停（`scheduler.py:1706-1710`）；S1 用 8192 块，130ez4 的 N34 开场 relief 分别持续约 50 s、90 s，relief 期间续算的上限就是整轮预算（`schedule_policy.py:1222`），短命中和 READY 兄弟都进不去（复现：relief 中到达的 512 token 暖命中，ON/OFF 批次完全相同，兄弟等了 5 轮）。124 停车时（续算剩余 >65536，长生产者常见）READY 头以 floor 0 运行、续算停一轮（`scheduler.py:1688, 4473`），ON/OFF 轨迹同样相同。
2. **可能形成永远无法 READY 的依赖。** 发现阶段（`ax_prefix_producer.py:293-305`）不检查生产者是否仍在目标之前、也不检查兄弟尾巴范围内会不会有 KDA 检查点；READY 需要 (目标 − 最近检查点) + 尾巴 ≤ short（`ax_prefix_readiness.py:28-37`），而检查点只在块尾或每次 extend 的一个分支快照（`schedule_batch.py:2866-2883`）。这类兄弟停在 WAIT_PREFIX（普通循环跳过，`scheduler.py:4480-4482`）直到 8 s 的 `MAX_HOLD_S`（`ax_prefix_producer.py:268`）到期——比开场排队短，没排在最前的家族会在生产者开跑前回退、失去分组排序（`282-294`），只剩 cache_ready → READY 一条路（`250-256`）。
3. **过时测试。** HEAD 上 `test_mechanism_report_official_a_env` 失败：`TREE_180 = tree_dir('HEAD')`（`tests/test_sched_protect_chain.py:641`）现在输出 `128p=off:…`，期望列表（`:724-729`）没有它；HEAD 在 741f3eda、候选文件未提交时 144 全过，说明证据日志里的 HiCacheTierTests 跑的是 741f3eda。
4. **角色拆分被抑制。** READY 准入总是传 `has_chunked_req=True`（`scheduler.py:1799`），无续算时也禁用了 101 的尾部角色拆分（`schedule_policy.py:1607`）；开发机 101 关、S1 模板 101 开，要盯兄弟下一轮的缓存命中。
5. **测试夹具伪影。** 假 `init` 给续算设 `extend_range`（`tests/test_prefix_producer.py:78-84`），停车后的续算在下一次 stash 后像已算完；CPU 停车结论不可靠。
6. **卫生。** 提交标题 `engine 128:` 让 `mech:128` 解析到 c0fcd486；`producer_admitted` 每轮都把续算计一次（`481-482`）。

## 已核实
- 默认关闭：`ENABLED` 为 False 时 key-limit/捕获/held 表都跳过（`schedule_policy.py:164, 212, 267, 380, 416, 433`；`schedule_batch.py:1473`）；tracker 为 None 时排序、停车、126、计划都是底座路径；关闭时只多每轮一个空集合/列表、每个等待请求一次 set 查找、每个 prefill 结果一次 getattr，机制行多 `128p=off:…`；没有额外前缀匹配、collective 或日志。
- (a) held 来源：`held_owner`（`ax_prefix_readiness.py:72-85`）记第一个共享原生深度的代表；`can_lift`（`ax_prefix_producer.py:218-232`）只在 owner 是其兄弟、或有存活且合格的 owner 共享至少该深度且组目标 ≥ 该深度时解除；浅系统提示只解除深生产者，无关请求保持 held 直到提示发布。
- (b) KDA：READY 用消费者自己的 device 长度；原生匹配在 Full 与 Mamba 校验都过的最深节点截断（`unified_tree_core.py:761-779, 844`；`mamba_component.py:138-149`）；任何 host 命中 → `host_reload`；没有路径会在未发布的 KDA 状态上准入兄弟。
- (c) COW 回退：失败在锁释放后抛出（`mamba_component.py:204-215`），预览复检在 `_lock_node` 下（`scheduler.py:1782-1792`），只释放本次新分配的槽（`1750-1755`），生产者的槽不释放。
- (d) 预算：READY 座位与续算都从 adder 实时预算扣，续算 floor 在预览里预留（`schedule_policy.py:1163-1186`，S1 下 floor 1024，READY 最多 7168/8192）；失败座位同轮归还（`4467-4475`）；停车且无人准入则恢复续算（`4599-4610`），无空转轮。
- (e) 一致性：prepare/should_park/log 只在 rank0 跑并广播；各 rank 校验版本/epoch/序号/续算（`4418-4423`）；每个 READY 尝试两次 all-gather 共识（`1794, 1803`）；retract/取消/RID 复用/flush 分别通过 identity/generation 失效（`100-120, 257-261, 1544-1546`）。
- TP8/DCP2/MTP/HiCache：无 128p 专属逻辑；DCP owner 为 `loc % dcp_size`（`memory_pool.py:4395`），READY 各 rank 相同；host-only 永不 READY；续算保持检查点网格对齐。
- 性能（CPU 实测真实代码）：34 等待、20–150k prompt：首轮计划 16.8 ms，之后每轮中位 1.1 ms/最大 3.7 ms，`held_owner` 1.8 ms；64 等待/48 运行：27.6 ms、2.3/5.2 ms、4.4 ms。O(n²) 受 64 候选上限约束；无 `.item()`/同步；每个计划轮多一次 gloo 广播（trace 窗口关闭后仍有，`4612`），每个 READY 座位两次 all-gather；逐决策日志 N34 约 20 KB/决策、每次 flush 最多约 40 MB，2048 轮上限可能早于 120 s 关窗。
- CPU 用例：c0fcd486 上 144 项 143 过 1 败（发现 3）。

## ON 臂必须给出的证据
机制行 `128p=on 128=off` 与 `[ax-prefix] on: Config(…)`；连续的 `[ax-prefix-decision]` 流、`prefix_producer_report.py` 能跑；`[ax-prefix-trace-end]` 的 elapsed 接近 120 s；有 `held=true` 且 `effective_held=false` 的 PRODUCER_WAIT 行；`[ax-prefix-stats]` 里 dependency/ready/reserved/admission_admitted/dependent_admitted 都 >0、回退原因分布、`admission_rank_match_disagreement`=0；统计“有 READY 行但 plan.ready 为空且有续算”的轮数（分停车 / 更高排名短命中）；逐请求：harness 同 ID chain 修复/新增（目标是 4 条 family_rider），`dependents.csv` 里依赖、生产者获选、READY 时间与 ready_device 对目标、兄弟首次获选、完成、回退原因；OFF 臂没有逐决策 trace，只有 harness TTFT；`ready_to_admit_ms` 是一轮内时间，不是发布到准入的时间。
