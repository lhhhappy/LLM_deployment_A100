# 122 调度候选独立审查

状态：已审；未发现阻止 048 启动的确定性崩溃或 TP collective 分叉。
冻结补丁：`cefdb2688cbc291742fc3c3ad188e343420fad01407d172f164ca6746d712d3b`（本地 SHA-256 核对）。
CPU 复核：`test_tpot_paced_prefill.py` 12/12、`test_sched_protect_chain.py` 27/27 通过。
范围：真实 `build/p122/baseA` 与 `candidate`、122 补丁、overlap/MTP/TP、`PrefillAdder`；未操作 GPU/队列。

## 结论与证据

1. **TP collective：未见确定性分叉。** `_ax_pace_now()` 只在有运行 decoder 且有等待或续算时进入；该路径在 `get_next_batch_to_run()` 的请求广播及上一轮结果处理之后。普通 TP 各 rank 的运行批、等待队列和完成态应一致；`tp_cpu_group` 已初始化。真实 TP8 仍需核对启动日志与前几轮是否同步推进。
2. **overlap 在飞预算：路径成立。** `event_loop_overlap()` 先为第 k+1 轮决策、再处理第 k 轮结果；122 用上轮预测结束时间计算 slack，并以 `ret.extend_num_tokens` 更新。模型低估、锚点偏差和 guard 放行已在补丁说明中披露，不能据 CPU 仿真声称 TPOT 保证。
3. **请求长度与准入：未见空 fill ids 崩溃。** 新等待请求尚未 `init_next_round_input()`，122 在初判和预留中使用 `Req.seqlen`；实际 `PrefillAdder` 先初始化再用 `full_untruncated_fill_ids`。`_ax_pace_limits()` 将预算送入 `rem_chunk_tokens`，冷续算 cap 及分页对齐仍由原 `PrefillAdder` 处理。
4. **观察边界：** 12 个 122 测试调用真实调度方法，但时钟、请求、batch、池均为假体；只覆盖一次假 TP reduce，不覆盖真实 Gloo/8 rank 或 MTP 结果处理。`mean_budget` 是决定时的预算统计，不等于实际块长；048 应从真实 extend 日志和 raw 记录核对块长、collective 等待和 KV 压力。

摘要：未找到需暂停 048 的明确代码缺陷。CPU 39 项通过只证明调度局部逻辑；真实 TP8 与 MTP 时序仍由 048 观察，性能与 TPOT 门按完整回放判定。
