# 070临时盘驱逐与071恢复交接

070在2026-09-24 22:13:28 UTC被平台驱逐，原因是Pod本地临时存储总量超过20Gi，未进入测量。
具体目录贡献未知。067—069完整原始证据已取回本地，不假设旧Pod其他数据可恢复。

## 已完成

- 同service2102486579267252224已接受env-only更新，revision2=2103249397960679424，22:25:07 UTC。
- 仅新增AX_WORKSPACE_ROOT=/dev/shm/arena-runtime；镜像、模型版本、GPU product/8卡、CPU88/memory1509Gi/shm754Gi/ephemeral20Gi/replicas1不变。GPU显示名由平台补NVIDIA前缀。
- 原service无运行副本后才执行update，没有stop/delete/release。幂等key与收据在[evidence](../../evidence/L070-official_b_host64_full_n30_shortwarm/recovery-plan.json)。最新deploying/WaitingForAdmission，实际副本0，尚不能称服务恢复。
- scripts/pod/prepare_workspace.py已提交0cb9c09并同步GPU机；bootstrap首次mkdir/ppush前调用。8项CPU测试与独立review通过，源/目标非空或异链拒绝，正确既有链接可保留。
- JIT缓存暂留原路径，/dev/shm可能noexec；未核实不能移.so。SGLANG_JIT_CACHE_DIR默认硬编码，不跟随SGLANG_CACHE_DIR，未来完整迁移需单独覆盖。
- watch070仍在GPU机；本地桥PID21269健康，Pod可exec后monitor从retrying→up会触发消息。其cached running/startup不是实时任务状态。

## 新Pod就绪后按序做

1. 用pread与平台pods只读确认新Pod；检查revision2和实际8GPU。`scripts/pod/bootstrap`在GPU仓库执行，放gjob后台留日志，禁止重复执行不明状态的部署。
2. 必须看到WORKSPACE_LAYOUT applied=true，/tmp/ax真实指向/dev/shm/arena-runtime/ax；记录mount flags、cgroup与shm余量，不能接受env丢失的legacy输出。64GiB是bootstrap下限，host64启动后再次核验。
3. bootstrap还原原s1-dev/harness/data/verify_kit。GPU仓库原来没有长链data目录；本轮从本地打包传入已结束，manifest/cohort/requests三项SHA与本地一致，再通过ppush把data/s1-dev-longchain推到/tmp/ax/data；Pod再完整核manifest/cohort/requests SHA和311链5601请求。
4. 用同一任务入口scripts/pod/jobs/official_b_host64_full_n30_shortwarm.sh，qpush映射新队列名071-official_b_host64_full_n30_shortwarm.sh；引擎759a6eb，host64/122on/mem.87/MTP不变。队列保持空闲暂停到RUNTIME_DEPLOYED与DONE rc=0，再podq init/resume。
5. 挂watch071（1800秒、30分钟窗、first-report-s900）与新的window_notify桥；旧070标基础设施中断无分数，结束其两个自有后台进程，不碰Pod引擎或服务。
6. 实际G_EXPECT、device/host池、原rep16预算、真flush都核验后才称测量开始；首次15分钟、随后每30分钟取证。

## 比较与容量边界

071是新Pod，旧根盘编译缓存丢失，rep16不保证所有形状覆盖。与069的差异同时包含基础设施重建，不能全部归122。
shm与host64/模型/临时张量同计1509Gi内存，不是额外754Gi；实际启动峰值和稳态余量必须取证。
/tmp/ax主要写盘已迁RAM；根盘JIT与容器日志仍可能增长，不能宣称已根治20Gi驱逐。恢复后补磁盘/缓存目录、shm与cgroup的低频审计和外部证据归档。
071尚未入队。每步状态与命令收据写回本文件/短索引，避免重复提交服务修订或重复bootstrap。
