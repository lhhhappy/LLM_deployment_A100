# `compare_runs.py` 最终定向验收

状态：当前代码与测试复核完成；`test_compare_runs.py` 10/10 通过，其中 047 自比为 722/722、四道 TTFT 门 fixed/new 全零。只读审查，未碰 GPU/队列或作者脚本。

## 可用结论

- 双方必须为 VALID，raw 恰好覆盖冻结 cohort、无重复/错误/缺时间；run 的 cohort/workload 标识与逐请求元数据一致。两边同缺行、INVALID、异负载的反例均会退出。
- TTFT 与 TPOT 越门使用 raw 精度；舍入仅用于 CSV。3.0004 秒、0.10004 秒反例通过。
- server.log 只纳入整秒区间 `[ts, ts+1)` 完全落在 `[首条 recv, 最后首 token]` 内的批与 pace 行。047 自比排除测量前 919 批、末尾边界秒 1 批；边界计数为 1，温机日志不再混入块分布。
- T56 pairs 检查 prompt 长度、链位置、前驱长度，以及 `0 <= true_lcp <= min(prompt, previous_prompt)`；负值和 `10**9` 反例均拒绝。047 为 411 条 metadata-checked、311 条无 pair。

## 使用边界

`pairs_rendered.json` 尚未与 prompt 内容或 cohort 内容哈希绑定；metadata-checked 只证明长度和链元数据匹配，不能证明 true-LCP 值来自本次实际 prompt。LCP gap 应保持为“来源未验证的诊断值”，不能单独作为 048 缓存归因证据。日志仍只有秒级时间戳，边界秒被排除；批分布是窗口内近似，不是逐请求精确归属。其余主门、等待/执行时间与缓存命中差的同请求对照可用于 048 分析。
