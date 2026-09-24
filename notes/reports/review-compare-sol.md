# `compare_runs.py` 独立审查（047/044r 校准）

状态：047 对自身输出全部零差异；044r→047 的 722 个 req_id 配对和现有报告数字复现。工具可作探索，但以下两项在用于 048 主决策前须修。

## 必须修

1. **整段 server.log 混入测量前 prefill。** 脚本以全日志统计批数和 token。按 raw 最早 `t_recv_s` 切分，044r 有 928/3303 批、3.856/13.686M token 在测量前；047 有 919/3221 批、3.822/13.746M token 在测量前。约 28% token 来自 warmup/preflight，足以污染 048 的块长和 partial-only 对照。应只统计测量窗口，日志秒级时间戳使边界秒保守丢弃或单列。`[ax-pace]` 首/末行也应区分测量窗口。
2. **“same ids”不是完整有效对照。** `load()` 只查单份 raw 内重复、错误和缺时间；主函数只比两份 req_id 集合，完全不检查 `level_verdict.status`、expected cohort 或 run metadata。两边同缺一条时仍称“same ids”，甚至两边 verdict 均为 INVALID 也照常打印对照。应要求两份 verdict 均 VALID、raw 数量与预期 cohort 相等，并核对 run 的 `cohort_sha256_canonical`/`workload_hash` 等冻结负载标识；否则退出 INVALID。047/044r 的这两个哈希相同。

## 同步修正，避免临界误判

- TTFT 先 `round(..., 3)` 再用 `>3/5/15/30` 计超限；例如原值 3.0004 秒应超 3 秒，脚本判不超。TPOT 先 `round(..., 4)` 再判 `>0.10`；0.10004 秒同理。真实 047 在这些边界没有跨门请求，但 048 未知。展示可舍入，判门须使用 raw 值。
- `pairs_rendered.json` 只按 req_id 查表，没有验证对应 prompt、前驱及数据集。047 的 411 条 pair 覆盖 722 条请求中的一部分，311 条 gap 留空；现有 411 条 prompt 长度与 raw 一致。至少校验每条 pair 的 prompt 长度、chain/idx/前驱元数据及 pairs 来源的 cohort/hash；否则 gap 只可标注为未验证诊断值，不能据其做缓存归因。

反例只由上述源码分支直接给出，未改主脚本、报告、GPU 或队列。
