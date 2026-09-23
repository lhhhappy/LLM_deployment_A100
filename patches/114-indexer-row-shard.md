# 114 — DSA indexer 按查询行切分到 TP 组（去掉 8 卡冗余）
- 问题：indexer 投影全是 ReplicatedLinear（`dsa_indexer.py:259-288`），每张卡都完整计算 32 头 × 全部历史的 logits 与 top-2048，结果相同，冗余 tp 倍；成本随上下文平方增长。
- 做法（`dsa_indexer_kpool.py::_get_topk_ragged_kpool_plan`，GLM kpool 预填充路径）：每卡只算连续行段 [r0,r1) 的 logits 与 top-k（行独立），`all_gather_into_tensor` 拼回，补齐行 -1（与原语义一致）。
- 开关：`SGLANG_AX_INDEXER_ROW_SHARD`（默认 1，0 = 原行为）；`SGLANG_AX_INDEXER_ROW_SHARD_MIN_ROWS`（默认 1024，小请求不切）；启用 DSA prefill CP 时自动跳过。首次生效打 INFO 日志。
- 验证（开发机 2×A100，TP2，SGLang 真实模型代码 + 随机权重的单卡份额缩小版，`scripts/analysis/extend_check.py`、`run_p114_check.sh`）：
  - 1.6 万 token：logits **逐位一致**，178.8 → 177.6 ms（indexer 很小）。
  - 6.5 万 token：logits **逐位一致**，733.0 → 705.0 ms（−3.8%，TP2 只省一半；推算 TP8 约 −6~7%，上下文越长越多）。
- 叠加：000→101→105→110→111→112→113→114 fuzz=0。decode 路径未改（后续可按 batch 行切）。
