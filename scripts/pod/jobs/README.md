# 当前可运行任务

按用户新决定逐项比较三个配置，122全部关闭：

- `official_a_full_n30_70m.sh`：正式 A 参数基准（064）。
- `official_a_mem087_full_n30_70m.sh`：仅加 `--mem-fraction-static 0.87`（065）。
- `official_a_180_mem087_full_n30_70m.sh`：在065上仅加新版180的三个HiCache参数（066）。

三项固定源码提交 `c92acd57a61eb6f9eed3222cc048877eef7963d9`，全量311链/5601请求、N30、70分钟准入后排空。
启动、预热及排空不计入70分钟准入时长，正式测量前真实flush。
队列实时安排见 [queue.md](../../../notes/queue.md)。
061s在预热中发现122短命中预留导致64-token续块，已停；063s撤下。修复后的122另行验证。
旧任务原文留在git和 `evidence/source-workflow-20260924/retired-jobs.tgz`；不能重新入队旧补丁任务。
新任务必须声明 `G_COMMIT`、`G_EXPECT`；默认走完整回放，设置 `G_MEASURE_SECONDS` 才走定时诊断。
