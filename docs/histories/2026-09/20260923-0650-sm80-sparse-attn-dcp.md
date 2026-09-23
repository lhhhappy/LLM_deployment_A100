## [2026-09-23 06:50] | Task: 补丁 115：tilelang 稀疏注意力支持每卡 64 头（DCP）

### 🤖 Execution Context

- **Agent ID**: `claude-code`（统筹）；子任务由 Codex worker / Fable 顾问承担时在正文注明
- **Base Model**: `Claude Opus 5.5 (1M context)`
- **Runtime**: Claude Code CLI，本地容器 + GPU 开发机（2×A100）+ Trisol L2 pod（8×A100，经 bohr exec）

### 📥 User Query

> （DCP 探针启动失败：共享内存 262144）

### 🛠 Changes Overview

**Scope:** GLM-5.3-Flash on 8×A100 serving stack (patches / pod tooling / docs)

**Key Actions:**

- 删除 v1 核中写而不读的 O_shared；`max_heads_per_block` 参数化
- sm80 且头数≥64 时 block_I=64、num_stages=1（7 种组合扫描中最快可编译）
- 开发机：H=8 与原版逐位相同；H=64 误差 ≤3.4e-3、graph 正常；8 卡 DCP8 探针：KV 逻辑 ×7.8、19 万冷预填充 −19%，能力 12/12

### 🧠 Design Intent (Why)

DCP 同时解决 KV 容量（N6 实际峰值 92%）与 TP 按头切分重复读 KV 的结构问题。

### 📁 Files Modified

- `patches/115-sm80-sparse-attn-many-heads.{patch,md}`
- `build/p115_test.py`
- `build/p115_sweep.py`
