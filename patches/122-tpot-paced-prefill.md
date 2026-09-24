# 122 — TPOT-paced prefill budget (candidate, on top of 120/121)

## 问题（044r 实测，正式 A 配置，开发集 N14）
- 0/722 条请求 TPOT 超过 0.10，p5 的请求仍有 2.4 s 没用掉的停顿预算（中位 11 s）。
- 挂的是首 token：chain_start 超标请求的 recv→exec 中位 55.7 s，overall_intra 为 7.3 s。
- 79% 的 prefill token 是在 ≤4096 的块里算的（121 的固定上限 + `--prefill-decode-interval 2`）。
- 满载下一个周期实测：4096 块 0.446 s，8192 块 0.739 s（n=1933/43）。有效 prefill 9.2k→11.1k token/s。

结论：TPOT 余量闲着，TTFT 在排队；固定小块付了过多的每块固定开销。

## 机制（借鉴 Sarathi-Serve 的"按 SLO 定每轮 token 预算"）
不同点：我们的门是每请求平均 TPOT，所以按每个请求实测的进度记账，而不是按单步 TBT。

- 每个正在 decode 的请求：`slack = t0 + τ·(已出 token) − now + 允许欠账`。t0 是第一次看到它首 token 的时刻。
- 规则：让所有请求保持 `slack ≥ 0`，最终 TPOT ≤ τ + 欠账/(n−1)。欠账默认 0.5 s，且不超过 (0.10−τ)·(n−1)。
- 每轮：预测成本 `C0 + c·(C1 + C2·上下文)` 放得进最小 slack 的整块（或剩余全部工作）就 prefill，否则 decode 一轮。连续 decode 最多 32 轮。
- 块内预算扣掉等待中完整缓存命中短请求所需的 token，余下给续算/冷请求。这取代 121 固定的 COLD_CAP。
- 开启时取代固定 interval 与 120 的单轮 decode。
- 8 个 TP rank 用一次 CPU all_reduce(MAX) 对齐时钟，保证决策一致。只在"有 decoder 且有 prefill 待做"时调用。

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
CPU：`tests/test_tpot_paced_prefill.py`，8 个用例，跑在真实调度器代码上，另有 27 个 120 旧用例，全部通过。
- 关闭时与正式 A 逐步 trace 相同。
- rank 时钟一致性。
- 假时钟仿真：成本模型成立时所有请求 TPOT ≤0.10。

仿真（假成本模型，MTP 接受长度 3.3，负载约 7.7k token/s，5 个种子）：
- 冷请求 TTFT p95 下降约 30–45%，短命中 p95 0.6→0.67 s，TPOT p95 0.06→0.08。
- 成本低估 25% 时仍优于 A。
- 无 MTP 时反而变差：TPOT 本来没有余量，它会先守 TPOT。所以只用于 MTP 部署。

**仿真只说明机制按设计工作，不是性能预测。** 8 卡结果待测。

## 已知限制
- 首次锚点最多晚一步。
- 成本模型不含 batch 内多个请求的差异。
- 为了追求 N，会把 tpot_mean 从 0.017 左右抬到 0.06–0.08；排名第二键会变差。
