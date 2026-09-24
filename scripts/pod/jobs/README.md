# 当前可运行任务

当前主线只有`official_b_full_n30_shortwarm.sh`（队列067）：正式A + mem0.87 + 新版180 + 修复122，
引擎759a6eb，固定rep16-v1短预热，真flush后全量N30回放5601请求，不设70分钟截止。

其余official_a_*_70m入口保留用于历史配置追溯，当前未排队，不自动重跑。
队列事实见[queue](../../../notes/queue.md)，统一规则见[evaluation](../../../notes/evaluation.md)。
所有新job声明固定G_COMMIT和G_EXPECT；原预热默认不变，日常诊断显式设G_WARMUP_PROFILE=rep16-v1。
当前rep16-v1入口用于单档完整回放，不与G_MEASURE_SECONDS叠加。
