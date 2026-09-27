# Codex 无 PTY 会话的收信登记修复

2026-09-27，Codex。状态：本机会话已登记，Fable 的实际回信与自发回环均进入当前会话，双向验证完成。使用 OpenAI Docs 技能核对[原生 App Server 协议](https://learn.chatgpt.com/docs/app-server)。

原 `agent_register.py` 要求代理 PID 的祖先中恰好有一个 Orca 终端进程。本次 Codex 是 managed app-server 会话，没有这样的 PTY；重跑原命令返回 `expected exactly one terminal owning pid 91599, found 0`。登记表中旧 terminal 条目也被发送器拒绝：`target terminal is not running the registered agent`。能给 Fable 发消息不能证明反向也通。

实现：[agent_codex_client.py](../../scripts/agent_codex_client.py) 使用标准库连接当前 app-server 已持有的 Unix WebSocket。它检查 `SO_PEERCRED` 的 PID/UID、WebSocket Upgrade、JSON-RPC initialize，并用 `thread/read` 验证目标 thread 已加载。注册记录保留 PID/start time/comm/thread/socket，写入采用锁与原子替换；没有保存凭据。

活跃会话按 `thread/turns/list` 返回的当前 turn ID 调用 `turn/steer`；空闲会话调用 `turn/start`。发生会话状态竞态时拒绝本次投递，不自动重发或重建会话；不调用 fork、resume、interrupt，也不直接改写 transcript。原终端传输及其身份校验保持有效。

本次登记：

```sh
python3 scripts/agent_register.py --name codex --pid 91599 \
  --session 01a0ddd3-03c7-78f0-bcd4-e30ccb9f9169 \
  --transport codex-app-server
```

PID 与 socket 属于本次存活进程，重启后重新登记。Fable 的发送命令不变：

```sh
python3 scripts/agent_message.py --to codex --from-agent fable --text '消息'
```

验证：三个脚本语法解析通过；对 Codex 与 Fable 的 `--check` 均返回 VALID；错误 PID 在初始化前被拒绝；不存在的 thread 在发送前被拒绝；Fable 的 `ACK native-0927: 双向收到` 与 Codex `SELFTEST native-0927` 均实际进入 `01a0ddd3-03c7-78f0-bcd4-e30ccb9f9169`。接收成功以本段实际确认算，不以 SUBMITTED 回执替代。未测试空闲会话启动的实发场景。

根工作区三个运行脚本已同步，方便 Fable 立即使用；代码提交归 `codex/prefill-sm80-0927`。本修复不改变引擎、GPU 运行或 Pod 队列。
