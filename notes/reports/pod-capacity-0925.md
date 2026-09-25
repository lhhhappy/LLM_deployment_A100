# 新 Pod 容量检查

2026-09-25 06:14 UTC，用户要求优先检查 Pod、避免再次超过 20Gi。
以下 06:14 初始检查全程只读；后续恢复操作单独记录在文末。
只读脚本通过 `pread capacity` 以内联 Python 执行，不在 Pod 写入脚本或结果。

## 实测

服务 `2102486579267252224` revision 2 已可 exec；Pod `lh-arena-sess-b-5bb8db8565-ktgp9`。
平台 get 返回 running、1/1 副本，但 runtime_stale=true；实时 exec 和 nvidia-smi 另证 Pod 当前可用。

| 项目 | 当前观察 |
|---|---|
| 临时存储配额 | 平台现行 revision 明确 **20Gi**，未扩容 |
| 根文件系统 | overlay，底层可用约 193.84GiB；不是 Pod 可用额度 |
| `/tmp/ax` | 约 8KiB，普通目录，仍在根盘；仅有 codex 子目录，queue 不存在 |
| RAM 工作目录 | env 已设 `/dev/shm/arena-runtime`，目录尚不存在，链接尚未建立 |
| `/dev/shm` | tmpfs，总量/可用均 754GiB；挂载选项 rw,relatime，未出现 noexec |
| 内存 | cgroup 上限 1509GiB，快照使用约 16.4MiB，failcnt=0；RAM 盘计入此预算 |
| 已检查磁盘目录 | `/tmp` 约 1.29MiB；`/root/.cache` 约 10.94MiB；`/root/.triton` 不存在 |
| GPU | 8×A100-SXM4-80GB，每卡 81920MiB，已用显示 0，无计算进程 |

目录 du 含可见镜像文件，不是容器可写层或整个 Pod 临时存储的计量。
现有平台 resources 接口只有磁盘 I/O，没有临时存储用量指标，因此**实际 20Gi 配额剩余量未知**，不能报成还有 20Gi。
当前未见已检查目录明显堆积；这不构成启动/安装峰值安全证明。

## 恢复边界

1. 先用已准备的工作目录迁移逻辑核验并建立 `/tmp/ax` → `/dev/shm/arena-runtime/ax`；确认落点后才上传。
2. JIT、pip/uv、安装临时目录、容器 stdout 仍分别计账；不能把一个工作目录迁 RAM 当作全部写入已迁移。
3. host64、模型加载、运行时缓冲与 tmpfs 共用 1509GiB，后续部署前算峰值，启动后重查。
4. 071/072 尚未入队；本次没有执行 bootstrap。旧 070 watcher 的 unknown/startup 是旧任务监控状态，不能当新 Pod 正在运行 070。

原始证据：[Pod 快照](../../evidence/pod-capacity-20260925/capacity.json)、
[平台现行资源配额](../../evidence/pod-capacity-20260925/platform.json)。

## 06:43 UTC 恢复进度

用户随后授权启动冻结的 host64 +122 配置。bootstrap 已完成，`/tmp/ax` 实际链接至 `/dev/shm/arena-runtime/ax`，RAM 上复制并运行 `/bin/true` 成功。数据、源码、完整日志、JIT、安装及编译临时缓存均指定到该工作目录；同时链接 `/root/.cache`、`.triton`、`.nv`、`.tilelang` 以覆盖部分库的硬编码路径，小型原镜像缓存保留。

071 同配置引擎正在加载权重，尚未开始测量。500MiB 冻结正文经6374块传输完成；认证短暂过期后已恢复，不更换访问密钥。启动前校验脚本的换行转义错误已修正；32项运行文件及manifest全部artifact校验已通过，现等待已有引擎READY才恢复队列。

当前快照：RAM 工作目录约0.80GiB，cgroup约123.28/1509GiB、failcnt=0，根盘 `/tmp` 约1.32MiB。驱动580.105.08，Ubuntu24.04.4，glibc2.39，Python3.12.3。此时尚未完成host池分配和捕图，不能视为启动峰值。

开发机归档器每5分钟检查终态运行，完整结果落到 `/sjtu/linhang/arena/archives/pod-runs`（GPFS磁盘，检查时约2.7TiB空闲）；逐文件SHA256/fsync验证后再清理Pod副本。有活动fd、当前引擎日志或内容变化时推迟，不截断原始统计。watch071已接替旧070监控；072仍预留，未安装vLLM环境。

证据：[加载阶段容量与ABI](../../evidence/L071-official_b_host64_full_n30_shortwarm/capacity-loading.json)、[运行文件哈希](../../evidence/L071-official_b_host64_full_n30_shortwarm/deployment-sha256.json)、[数据校验入口](../../evidence/L071-official_b_host64_full_n30_shortwarm/verify.py)、[等待就绪后放行](../../evidence/L071-official_b_host64_full_n30_shortwarm/release.sh)。
