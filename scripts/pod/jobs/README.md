# 当前可运行任务

当前候选`official_b_host64_full_n30_shortwarm.sh`（队列070）：沿用069的host64，只重新开启修复122。
源码759a6eb、正式A + mem0.87 + 新版180，MTP保留。069完整VALID FAIL，10/11门通过，仅chain31/29失败。
固定rep16-v1短预热，真flush后全量N30回放311链/5601请求，不设70分钟截止。

067/068/069入口保留用于完整对照溯源；其余official_a_*_70m入口仅作历史配置，不自动重跑。
实时状态见[queue](../../../notes/queue.md)，统一规则见[evaluation](../../../notes/evaluation.md)。
所有新job声明固定G_COMMIT和G_EXPECT；原预热默认不变，日常诊断显式设G_WARMUP_PROFILE=rep16-v1。
rep16-v1入口用于单档完整回放，不与G_MEASURE_SECONDS叠加。
