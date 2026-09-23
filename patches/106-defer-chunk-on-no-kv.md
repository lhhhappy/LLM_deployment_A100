# 106 — KV 无余量时推迟续算分块（修复 "Prefill out of memory" 引擎崩溃）
- 现象：8 卡梯子 024（KV 63 万）在 N10 warmup 期间，一个冷请求以 16384 分块续算，KV 占用 0.96→1.00，`mem_cache/allocation.py:217` 抛 "Prefill out of memory"，全部 TP rank 退出。
- 根因（底包）：`PrefillAdder.add_chunked_req` 在 `rem_total_tokens<=0` 时把预算强行改回整块 `rem_chunk_tokens`（注释称否则内存泄漏），导致在 KV 已满时硬分配。SWA 模型分支在同样情况下 `return req`（本轮跳过续算）。
- 改动：非 SWA 也走"本轮跳过续算"（`SGLANG_AX_DEFER_CHUNK_ON_NO_KV=1` 默认开，0 = 原行为）。decode 照常进行，完成或被撤回的请求释放 KV 后续算恢复。
- 验证：8 卡回归任务 027（与崩溃的 024 同配置 + 106）N10 不崩；026 起所有梯子带 106。
