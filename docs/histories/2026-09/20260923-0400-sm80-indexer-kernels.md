## [2026-09-23 04:00] | Task: sm80 DSA indexer 融合 kernel（112/113 v2）与行切分（114）

### 🤖 Execution Context

- **Agent ID**: `claude-code`（统筹）；子任务由 Codex worker / Fable 顾问承担时在正文注明
- **Base Model**: `Claude Opus 5.5 (1M context)`
- **Runtime**: Claude Code CLI，本地容器 + GPU 开发机（2×A100）+ Trisol L2 pod（8×A100，经 bohr exec）

### 📥 User Query

> profile 发现 torch 版 indexer 占 60% GPU；优化修复比跑通更重要（原文：后面的应该更重要，就是一些优化修复）

### 🛠 Changes Overview

**Scope:** GLM-5.3-Flash on 8×A100 serving stack (patches / pod tooling / docs)

**Key Actions:**

- 8 卡 /start_profile 定位 110 torch shim 占约 60% GPU（F63）
- Codex W17/W18/W21：112 decode 融合核（~4.5×）、113 prefill（再 ~6×，185 TFLOPS）、v2 去形状特化（200 随机形状 0 编译）
- **补丁 114**：indexer 按查询行切到 TP 组 + all-gather top-k，TP2 逐位一致，65k −3.8%

### 🧠 Design Intent (Why)

indexer 在 8 卡上完全冗余且是 A100 上的主要额外成本；融合核消除物化中间张量，行切分去掉 8× 冗余。

### 📁 Files Modified

- `patches/112-*.patch`
- `patches/113-*.patch`
- `patches/114-indexer-row-shard.{patch,md}`
- `scripts/kernels/`
- `scripts/analysis/extend_check.py`
