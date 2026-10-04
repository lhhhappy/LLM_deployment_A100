# 123 — shortest remaining prefill first, with aging (candidate, on top of 120)

## Why
In 037 (N22) 87 chain_start requests exceeded 30 s; 60 of them needed under 64k tokens and waited a median 54 s in the
queue while one large cold request was being prefilled chunk after chunk (only one chunked request can be active). Small
heads that needed 1.4k–10k tokens waited 49–75 s through 49–72 prefill batches of 8192+ tokens (scripts/analysis/badcase.py,
evidence/L037). Under LPM, cold requests are ordered by shared-prefix length and then arrival, not by the work they need.

## What it does (`srt/managers/schedule_policy.py`)
With `SGLANG_AX_SRPT_AGING` set, the LPM sort becomes: remaining prefill tokens (prompt − matched prefix) minus
`aging × seconds waited`, ascending. Cheap cache hits still go first (little remaining work), small cold requests are admitted
before large ones, and a large request gains `aging` tokens of priority per second so it is not starved (2000 tokens/s: a
100k-token request waiting 60 s ranks ahead of a fresh 3k one). Requests held back by LPM's in-batch prefix sharing stay last.
It changes only the admission order: an already active chunked request is not preempted.

## Switches
`SGLANG_AX_SRPT_AGING` (tokens per second; unset or 0 = plain LPM).

## Evidence
CPU: `tests/test_srpt_admission.py` (5 tests on the real sort and scheduler code with fakes); all 37 scheduler tests pass.
8 cards: pending (S1 + 122 + 123 at N22, compared with S1 + 122).

## 2026-09-25：TP 一致排序修复

各 rank 的 wait_queue_entry_time 是本地时间。共同 now 在两个排序键的比较中抵消，只同步 now 不能修复近邻键的次序分叉。启用 123 时，request-plane CPU group 的首 rank 计算最终 request ID 顺序并广播；其他 rank 对本地 Req 引用应用相同顺序。缓存匹配仍在各 rank 的真实路径执行。关闭 123 时不调用新广播，LPM 行为不变。

共享接口 Scheduler._ax_rank0_decide(compute) 返回所有 rank 相同的小型可序列化结果。各 rank 必须在同一分支进入；只有首 rank 执行 compute，回调副作用不会复制。使用 dp_tp_cpu_group 和该组首 rank 的 global rank，不硬编码世界 rank 0。广播增添 CPU 调度开销，需由真实 TP8 探针测量；不宣称零开销。

验证：真实排序函数复现 1 ms 入队时间差导致次序翻转；修复后双方采用首 rank 顺序；关闭分支不广播；非零 global source 的组广播接口校验。既有 admission 用例保持通过。
