# 新 Pod 容量检查

2026-09-25 06:14 UTC，用户要求优先检查 Pod、避免再次超过 20Gi。
全程只读：没有 bootstrap、上传数据、安装依赖、启动引擎或发布测试。
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
