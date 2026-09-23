## 2026-09-23 | Task: 提供本地只读 pod 日志采集脚本（T57）

### 🤖 Execution Context

- **Agent ID**: Codex main
- **Base Model**: GPT-6
- **Runtime**: Codex workspace / Python 3

### 📥 User Query

> 给我一个本地脚本，我帮你执行；尝试禁用代理读取日志。

### 🛠 Changes Overview

**Scope:** scripts/pod 的独立诊断工具。

- 新增 collect_run_logs.py，单次通过现有 CPU-only pexec_codex 读取作业/引擎/测量日志、时间戳、已有退出码和 raw 行数。
- 默认 SSH 直连；--proxy 保留现有 SSH 配置；--on-gpu 可在开发机绕过本地 SSH。输入名称受限并用 shlex 转义，不向引擎发请求、不操作队列。
- T57-01 离线检查通过。直连实际探测退出255/连接超时；已有代理路径间歇可读，不保证本脚本的网络可达。

### 🧠 Design Intent (Why)

一次连接收齐诊断信息，允许用户在自己的终端执行。正在写入的raw仅报告快照，不把旧verdict或行数当作已验证成绩。

### 📁 Files Modified

- scripts/pod/collect_run_logs.py
- evidence/T57/validation.log、direct_connection.txt
- notes/dispatch.md、board.md、tests/TEST_PLAN.md
