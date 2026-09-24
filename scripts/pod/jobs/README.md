# 当前可运行任务

只保留两个候选及通用模板：

- `official_a_122_full_n30_70m.sh`：A参数 + 122。
- `official_a_122_180_full_n30_70m.sh`：同上，仅开启新版180主机缓存。

两项同源码提交、全量311链/5601请求、N30、70分钟准入后排空。
队列编号与实时安排见 [queue.md](../../../notes/queue.md)。
旧任务原文留在git和 `evidence/source-workflow-20260924/retired-jobs.tgz`；不能重新入队旧补丁任务。
新任务必须声明 `G_COMMIT`、`G_EXPECT`；默认走完整回放，设置 `G_MEASURE_SECONDS` 才走定时诊断。
