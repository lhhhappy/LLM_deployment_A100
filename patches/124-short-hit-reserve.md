# 124 — short-hit reserve next to a partial (candidate, on top of 120/121)

## 问题（044r 实测，正式 A，dev N14；`scripts/analysis/intra_attrib.py`）
120/121 让续算的长请求占 COLD_CAP（A 为 4096）。完整缓存命中的短请求只能用剩下的 8192−4096=4096 token；新 token 超过 4096 的命中请求要等整个冷 prefill 做完。

| 缓存命中请求的新 token | 超 TTFT 限（fast 3 s / 其他 5 s） | 等待中位（超 / 未超） | 等待窗口里有他人 partial（超 / 未超） |
|---|---|---|---|
| ≤4096 | 0/274 | — / 0.27 s | — / 71% |
| 4097–8192 | 26/79 | 6.56 s / 0.30 s | 100% / 49% |

- 全程 91% 的 prefill batch 带 partial（基线）。
- 22 条 fast_intra 超 3 s 的请求里，19 条属于 4097–8192 这一类。
- 正式 B 的剩余预算为 0（COLD_CAP ≥ chunk）：045r 缓存命中 ≤8192 的请求 59/341 超限，且都在等 partial。这与 B 正式挂 fast 方向一致。

以上是描述性归因（日志 1 s 分辨率），不是因果证明。

## 机制
续算请求或冷请求首块的上限改为 `min(COLD_CAP, chunk − 等待中完整短命中请求的分页新 token 之和)`，至少一个 grid。
- 没有这类请求等待时，上限仍是 COLD_CAP，与 A 相同。
- batch 总 token 预算不变，所以单个 batch 的耗时不变。
- 前缀匹配用本轮 `calc_priority` 的结果；长度用 `seqlen`，因为真实等待请求在准入前 fill ids 为空。

补丁 122 含同一规则，但与 TPOT 进度控制耦合。124 用于单独测这个变量，与 122 互斥：两者改同一处，不能同时应用。胜出后合并为一个机制。

## 开关
`SGLANG_AX_SHORT_RESERVE`（默认 0 = 关，与正式 A 逐步相同）。

## 证据
CPU：`tests/test_short_hit_reserve.py` 5 个用例，运行真实调度方法（模型、池与 batch 为假）。
- 关闭时与正式 A 逐步相同；
- 7000 新 token 的命中请求与 partial 同批（partial 1024 + 命中 7000），A 中只有 partial 4096；
- 小命中请求与无命中请求时行为不变；
- 等待请求 fill ids 为空；
- 多个命中请求时 partial 仍至少前进一个 grid。

与 120/122 测试合计 44 个通过。8 卡未测。

## 风险
冷的长请求（链首）每让出一次，就多等一个短命中请求的 prefill 时间。A 的正式紧门正是 chain_start（p95 30.23），必须同看 chain 门。
