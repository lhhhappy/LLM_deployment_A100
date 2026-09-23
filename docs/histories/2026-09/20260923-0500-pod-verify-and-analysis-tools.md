## [2026-09-23 05:00] | Task: 真机复核套件与系统分析工具

### 🤖 Execution Context

- **Agent ID**: `claude-code`（统筹）；子任务由 Codex worker / Fable 顾问承担时在正文注明
- **Base Model**: `Claude Opus 5.5 (1M context)`
- **Runtime**: Claude Code CLI，本地容器 + GPU 开发机（2×A100）+ Trisol L2 pod（8×A100，经 bohr exec）

### 📥 User Query

> 开发机测的和真机可能不一样，要留脚本在申请下来的机器上快速过一遍；评测之外的指标、系统瓶颈都要分析

### 🛠 Changes Overview

**Scope:** GLM-5.3-Flash on 8×A100 serving stack (patches / pod tooling / docs)

**Key Actions:**

- `scripts/pod/verify/`：run_verify（11 项 kernel 结论真机复核，11/11 PASS）、bench_moe_int8、coldprobe、component_table、logstat、analyze_run（harness 硬判定 + 正式规则估算、排队/执行拆解、缓存效率、最差请求）
- `scripts/analysis/`：单卡份额缩小版模型 + 真实 SGLang 代码的逐组件成本表（make_rank_model / rank_profile）、跨会话前缀复用潜力分析
- 8 卡任务：cap_smoke（12 题能力冒烟）、coldprobe_*、dev_generic_template、dev_ladder_template（像正式爬坡一样 N=10→26）

### 🧠 Design Intent (Why)

结论必须在比赛镜像上复现才算数；门禁之外要能解释每个请求为什么慢。

### 📁 Files Modified

- `scripts/pod/verify/*`
- `scripts/analysis/*`
- `scripts/pod/jobs/*`
