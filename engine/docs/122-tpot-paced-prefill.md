# 122 — TPOT-paced prefill budget (candidate, on top of 120/121)

## 问题（044r 实测，正式 A 配置，开发集 N14）
- 0/722 条请求 TPOT 超过 0.10，p5 的请求仍有 2.4 s 没用掉的停顿预算（中位 11 s）。
- 挂的是首 token：chain_start 超标请求的 recv→exec 中位 55.7 s，overall_intra 为 7.3 s。
- 79% 的 prefill token 是在 ≤4096 的块里算的（121 的固定上限 + `--prefill-decode-interval 2`）。
- 有请求在等、也有请求在 decode 时，相邻两次 prefill 日志的间隔：4096 块中位 0.446 s（n=1933），8192 块 0.739 s（n=43）。这个间隔含 2 轮 decode，8192 的样本少。

假设（待 8 卡验证）：这一档 TPOT 还有余量，而首 token 在排队；更大的块能少付每块固定开销。

## 机制（借鉴 Sarathi-Serve 的"按 SLO 定每轮 token 预算"）
不同点：我们的门是每请求平均 TPOT，所以按每个请求实测的进度记账，而不是按单步 TBT。

- 每个正在 decode 的请求：`slack = t0 + τ·(已出 token − 锚定时 token) − now + 允许欠账`。t0 是它第一次出现在运行批中的时刻；首 token 尚未处理的按刚出首 token 计。
- 目标：让所有请求保持 `slack ≥ 0`；欠账默认 0.5 s，且不超过 (0.10−τ)·(n−1)。**这是调度目标，不是 TPOT 保证。** 以下情况会破坏它：成本模型低估；锚点晚于真实首 token（晚多少不受 τ 裕量约束）；连续 decode 达到 `MAX_DECODE` 后强制放行一整块（日志 `guard=` 计数）。TPOT 门只按实测判定。
- 每轮：预测成本 `C0 + c·(C1 + C2·上下文)` 放得进最小 slack 的整块（或剩余全部工作）就 prefill，否则 decode 一轮。连续 decode 最多 32 轮。
- 块内预算扣掉等待中完整缓存命中短请求所需的 token，余下给续算/冷请求。这取代 121 固定的 COLD_CAP。
- 开启时取代固定 interval 与 120 的单轮 decode。
- overlap 调度下，第 k+1 步在第 k 步 prefill 仍在 GPU 上时就已决策：以上一个受控 prefill 的预测结束时刻作为决策时间，避免同一份余量被连续两块重复使用。
- 8 个 TP rank 用一次 CPU all_reduce(MAX) 对齐时钟，保证决策一致。只在"有未完成的 decoder 且有 prefill 待做"时调用，这个条件在各 rank 上相同。

## 开关
全部在引擎启动时读一次，生效值打印在 `[ax-pace] on:` 日志行。

| 变量 | 默认 | 含义 |
|---|---|---|
| `SGLANG_AX_PACE_TPOT` | 0（关） | τ，目标 s/token；试验值 0.085 |
| `SGLANG_AX_PACE_DEFICIT_S` | 0.5 | 允许欠账 |
| `SGLANG_AX_PACE_FIXED_S` | 0.08 | C0；推断：满载周期截距 0.153 减去 2 轮 decode |
| `SGLANG_AX_PACE_PER_TOKEN_S` | 6e-5 | C1；8 卡 F87 实测 |
| `SGLANG_AX_PACE_PER_TOKEN_CTX_S` | 1.9e-10 | C2；TP1 替身斜率随上下文增长，推断 |
| `SGLANG_AX_PACE_MIN_CHUNK` | 0 | 0 表示整块 |
| `SGLANG_AX_PACE_MAX_DECODE` | 32 | 连续 decode 上限 |
| `SGLANG_AX_PACE_GATE` | 0.10 | TPOT 门限 |

每 30 s 打一行 `[ax-pace] decisions/forced_decode/mean_budget`，用来核对实际行为。

## 证据
**CPU（真实调度代码）：** `tests/test_tpot_paced_prefill.py` 12 个用例，加 `test_sched_protect_chain.py` 27 个，共 39 个全部通过。测试从源码树中抽出真实的 `get_next_batch_to_run`、`_get_new_batch_prefill_raw`、`PrefillAdder` 执行，模型、池与 batch 是假的。
- 关闭时与正式 A 逐步 trace 相同；
- rank 时钟取 max 且只在需要时调用；
- 新到请求 `full_untruncated_fill_ids` 为空时按 `seqlen` 计；
- overlap 下首 token 未处理的 decoder 仍计账；
- 上限放行整块；
- overlap 下在飞 prefill 计入。

**仿真（同一真实调度代码 + 假时钟与假成本模型）：** 仅说明机制按设计工作，不是性能预测。
- MTP 接受长度 3.3、约 7.7k token/s、5 个种子：冷请求首 token p95 降约 30–45%，TPOT p95 约 0.08；
- 成本低估 25% 时 TPOT 仍在门内；
- 无 MTP 时首 token 变差（TPOT 本无余量，机制优先守 TPOT）。

**8 卡：** 048（正式 A + 122，旧开发集 N22，对照 047）fast 33→10/23 转过，overall 55→28/27、chain 73→62/22 仍挂，TPOT 守住。
067（mem0.87 + 新版180 + 修复122，完整合成长链N30）5601条VALID FAIL，四类TTFT失败，TPOT mean/p95 .0335/.0675；组合结果不能单独归因122。
068保持067源码、负载、N30与rep16预热，只关闭122，验证整个机制的净取舍。关闭同时恢复固定decode interval与冷块上限，不能只解释为预算常数变化。
当前真实调度器CPU回归为122 15项、调度保护32项通过；仿真不是TP8收益证据。完整结果见[实验记录](../../notes/experiments.md)。

## 已知限制
- 锚点时序：在第一次看到该请求的决策时建立，时间取 max(同步时钟, 在飞受控 prefill 的预测结束)。
  - overlap 且产生它的 prefill 受控时，锚点 ≈ 预测结束；预测偏晚多少，锚点就晚多少。
  - 该 prefill 不受控（当时无 decoder）时，锚点取决策时刻，早于首 token，属保守方向。
  - 非 overlap 时，锚点晚于首 token 的时间是结果处理与调度耗时。
  - 这些偏差都没有上界保证。
- 成本模型不含 batch 内多请求差异；C0、C2 是推断值，需 A 配置 TP8 实测曲线替换。
- 每个需要决策的调度步多一次 TP CPU 组 all_reduce，开销未在 8 卡实测。
- 会抬高 tpot_mean（仿真中 0.06→0.08 量级）；只有 N 提升时才有排名意义。
