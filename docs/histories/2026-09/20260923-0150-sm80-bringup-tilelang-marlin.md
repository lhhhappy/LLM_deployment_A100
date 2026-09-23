## [2026-09-23 01:50] | Task: 让底包在 A100 上跑起来（tilelang DSA + FP8 MoE Marlin）

### 🤖 Execution Context

- **Agent ID**: `claude-code`（统筹）；子任务由 Codex worker / Fable 顾问承担时在正文注明
- **Base Model**: `Claude Opus 5.5 (1M context)`
- **Runtime**: Claude Code CLI，本地容器 + GPU 开发机（2×A100）+ Trisol L2 pod（8×A100，经 bohr exec）

### 📥 User Query

> 005 失败原因；要尽快在 8 卡上跑通（原文：看看 HANDOFF，我现在非常需要你）

### 🛠 Changes Overview

**Scope:** GLM-5.3-Flash on 8×A100 serving stack (patches / pod tooling / docs)

**Key Actions:**

- **tilelang 后端**：定位 fa3 仅 Hopper（`Only Hopper supports different V headdim`），改为 `--dsa-*-backend tilelang`（v1 核关 TMA/warp-spec，sm80 可编译，单卡数值+graph 通过，F57）
- **补丁 111**：Fp8MoEMethod 在 sm80 走 Marlin W8A16（底包已有 prepare_moe_fp8_layer_for_marlin 与 kFE4M3fn Marlin MoE JIT 核，接线 + `fp8_weights` 标志），单卡误差 6e-3（F58）
- 8 卡 b111 启动并通过功能探测（F59）

### 🧠 Design Intent (Why)

原版在 A100 起不来（DeepGEMM/fp8 Triton/FA3 均 sm90+）；复用底包已有但未接线的 sm80 能力，改动最小、数值可验证。

### 📁 Files Modified

- `patches/111-sm80-fp8-moe-marlin.{patch,md}`
- `scripts/pod/jobs/b111_start_probe.sh`
- `build/p110/test_tilelang_sparse_sm80.py`
- `build/p111/test_fp8_moe_marlin_sm80.py`
