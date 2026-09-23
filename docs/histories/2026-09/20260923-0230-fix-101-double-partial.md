## [2026-09-23 02:30] | Task: 修复 101 双 partial 崩溃并加跑后存活检查

### 🤖 Execution Context

- **Agent ID**: `claude-code`（统筹）；子任务由 Codex worker / Fable 顾问承担时在正文注明
- **Base Model**: `Claude Opus 5.5 (1M context)`
- **Runtime**: Claude Code CLI，本地容器 + GPU 开发机（2×A100）+ Trisol L2 pod（8×A100，经 bohr exec）

### 📥 User Query

> （N6 首跑 722/722 connection refused）

### 🛠 Changes Overview

**Scope:** GLM-5.3-Flash on 8×A100 serving stack (patches / pod tooling / docs)

**Key Actions:**

- 定位引擎在 warmup 中 `assert self.chunked_req is None`：101 尾部切分使续算请求仍为 chunked，剩余预算又截断新请求 → 两个 partial
- **补丁 105**：角色切分开启且有续算分块时拒绝再截断新请求（F62）
- dev 任务模板加 ENGINE_DEAD_AFTER_RUN 检查；podq init 推送全部编号补丁

### 🧠 Design Intent (Why)

N6 基线必须先能完整跑完；105 不改变 stock 行为，仅堵住 101 引入的路径。

### 📁 Files Modified

- `patches/105-role-split-single-partial.{patch,md}`
- `scripts/pod/jobs/dev_template.sh`
- `scripts/pod/podq`
