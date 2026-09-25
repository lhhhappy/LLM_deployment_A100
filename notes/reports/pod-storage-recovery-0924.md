# 070临时盘驱逐与071恢复交接

070在2026-09-24 22:13:28 UTC因Pod临时存储总量超过20Gi被驱逐，未进入测量。具体目录贡献未知；067—069完整证据已取回。host32→64使用CPU内存，当前无L3磁盘缓存，不能归因host扩容。

## 当前状态（2026-09-25 06:43 UTC）

- 同service `2102486579267252224`、revision2 `2103249397960679424`，新Pod已可exec；8×A100、CPU88、memory1509Gi、shm754Gi、ephemeral20Gi、原镜像/模型不变。未stop/delete/release服务。
- `/tmp/ax` 已实际链接到 `/dev/shm/arena-runtime/ax`；RAM执行探针通过。数据、源码、运行日志、JIT和临时缓存迁RAM，硬编码root缓存路径另做链接，小型原目录保留。见[容量收据](pod-capacity-0925.md)。
- 071冻结引擎 `759a6ebb8e31723519ad5daf438e26e24b32501a`、host64、122on、mem.87、MTP、rep16与完整N30数据；模型在队列暂停时提前加载，不发测量请求。
- 500MiB长链正文6374块已上传，传输DONE rc=0。Trisol bearer短暂过期后现有认证已刷新，login状态和Pod读取成功，未更换AK。
- 首次放行脚本因生成的Python换行转义错误退出，未放行回放；已修正并做语法检查，随后全部SHA校验通过。重试仅校验与放行，不重新传500MiB、不重新启动模型。
- watcher `watch071-20260925` 与本地通知桥已运行；旧070自有watcher已退出。开发机归档器每5分钟检查终态运行，完整拷贝并核验后清理Pod，目的地 `/sjtu/linhang/arena/archives/pod-runs`。

## 余下放行顺序

1. `pod071-release-retry-20260925` 在GPU开发机运行 `build/scratch/pod071-release.sh`：先验证32项运行文件SHA256、manifest/cohort和全部artifacts，确认311链5601请求。正文先上传到 `data/data/s1-dev-longchain`，校验通过后原子移到最终 `data/s1-dev-longchain`，不覆盖既有目录。
2. 校验成功后init worker，但队列仍暂停。等待已有 `preload.log` 出现 `PRELOAD_READY`，再resume；提前resume会因engine.sig尚未生成而重复启动引擎，禁止跳过等待。
3. 核对实际G_EXPECT、device/host池、rep16预算与真flush；进入测量后才记t0。全部原始数据保留，首次15分钟、随后每30分钟取证；不因局部FAIL自动停。
4. 终态按原harness全量11门判分，显式 `--data-root data/s1-dev-longchain`。完整文件归档SHA256/fsync后才清理Pod，有活动fd/当前engine日志/内容变化则推迟。

运行收据在 [L071](../../evidence/L071-official_b_host64_full_n30_shortwarm/)；放行脚本、校验脚本、预加载脚本和部署哈希都留副本。实际阶段以Pod日志及后台任务结果为准，不能把pending/startup当测量。

## 比较与容量边界

071是新Pod，冷编译缓存与运行路径变化是069对照的混杂；rep16不保证所有形状覆盖，差异不能全归122。
shm与host池、模型加载、临时张量共计1509Gi，不是额外754Gi。06:43快照仅为加载阶段，尚未得到启动峰值；根盘/tmp大小也不是整个20Gi配额用量。
持续核缓存目录、shm和cgroup余量。完整诊断不得为限日志而丢弃；通过正确写入位置及终态归档避免累计。

072仍在071终态之后预留；Claude在开发机准备vLLM独立环境及体积/ABI清单。Pod driver580.105.08、Ubuntu24.04.4/glibc2.39/Python3.12.3已回传。当前不换服务镜像、不覆盖SGLang依赖、不并驻两个完整模型。
