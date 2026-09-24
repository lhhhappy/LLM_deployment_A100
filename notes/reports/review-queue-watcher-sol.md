# 队列 watcher：本地通知与 GPU 留档

`scripts/pod/watch_queue.py` 默认在本地只执行 `scripts/pod/pread status`，由现有主会话 watcher 通知 lead。`--gpu-record` 在 GPU 机本地经 `source scripts/pod/common.sh` + `bexec` 执行固定只读队列 status，事件仅追加到 GPU 机 `build/scratch/coordination/queue-watch-gpu/events.jsonl`，不调用 `pread`/`gssh` 或 `agent_message.py`。两层各有独立状态、锁与 heartbeat；本地断连时 GPU 层仍留档，恢复后本地层继续通知主会话。GPU 层不是第二个通知源。

两种模式默认每 60 秒轮询，单次 status 45 秒超时；本地通知 20 秒超时。`flock` 保证每个状态目录单实例。状态、待送事件、PID 与 heartbeat 放在各自的 `build/scratch/coordination/queue-watch[-gpu]/`；首次成功快照只建基线，不重播历史 failed/done。GPU 模式要求 ROOT、cwd、state-dir 都位于 `/sjtu/linhang/arena/` 下。

新出现或变更的 pending/running/done/failed 会通知主会话；即使 job 从 pending 直接变 failed 也会发现。远程不可用只告警一次，恢复时补报；通知返回非成功则待送事件持久保留、下轮重试。`SUBMITTED` 仅表示向终端提交，不证明主会话已读。

现有 `agent_message.py` 的 `--from-agent` 只允许 `lead/claude/data`。经主会话确认，本 watcher 使用合法 `--from-agent lead`，正文标明 `[queue watcher]`；未修改 CLI 或绕过 allowlist。

使用：

本地：

```sh
python3 scripts/pod/watch_queue.py --notify-test  # 显式发送一条通路测试；主会话确认收到
python3 scripts/pod/watch_queue.py --once         # 只读检查一次并写持久快照
python3 scripts/pod/watch_queue.py                # 前台持续监控；由主会话选择启动方式
```

GPU 机仓库 cwd（`/sjtu/linhang/arena/repo`）中：

```sh
python3 scripts/pod/watch_queue.py --gpu-record --once
python3 scripts/pod/watch_queue.py --gpu-record
```

主会话负责同步和启动 GPU 进程，检查 `process.json` heartbeat 与 `events.jsonl`。`--gpu-record --notify-test` 被拒绝，避免远端触发 relay。事件按持久 ID 防止进程在写档后、清空 outbox 前退出造成重复行。

CPU 测试 12/12 通过：`python3 -m unittest discover -s tests -p test_watch_queue.py`。覆盖初始历史抑制、快速失败、完成、远程故障与恢复去重、通知失败重试、状态解析失败、`pread` receipt、GPU 路径限制/本地 bexec 入口/事件防重。`process.json` 的 `started_at` 固定为进程启动时间，heartbeat 每轮刷新。进程退出不会操作服务或队列；锁随进程退出释放。
