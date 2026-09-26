# 调研与源码地图

从仓库根部 [README](../README.md) 了解当前实验、队列和成绩；赛题以 [task.md](../llm-challenge-arena-v1/task.md) 为准。本目录只保留能支持现行代码或解释已核实实验结果的研究。历史版本留在 git，不作为当前结论。

| 文件 | 用途 |
|---|---|
| [底包与当前认识](claude/base/00-summary-mainline.md) | 底包来源、S0 与主要约束、过时结论的更正 |
| [请求入口](claude/base/01-request-path.md)、[调度器](claude/base/02-scheduler.md)、[混合缓存](claude/base/03-hybrid-cache.md)、[模型与算子](claude/base/04-model-kernels.md) | `build/base_exact/sglang` 的源码地图；运行时结论需结合现行补丁与实验 |
| [上游与替代引擎](claude/R9_upstream_since_base.md) | 09-24 覆盖更新：vLLM/TokenSpeed 等引擎筛选，KDA 融合、MoE 组织与 graph；标明 A100 证据边界 |
| [prefill 固定开销](claude/R10_prefill_fixed_overhead.md) | 开发机替身模型 profile 和 8 卡推断的边界 |
| [MTP 路径](codex/R17_nextn_sm80.md) | 补丁 160 的实现与兼容性 |
| [缓存与显存](codex/R18_cache_loss_and_capacity.md) | 早期 N6 账本；可修缺口的最新量化看 R20 |
| [分析与工具复核](codex/R19_progress_and_cache_review.md) | 纠正原先的缓存归因、发现会导致假通过的分析脚本问题 |
| [真实 LCP 归因](codex/R20_true_lcp_attribution.md) | 026/N18 的逐请求 LCP、fast 超时与短输出 TPOT |
| [035/N22 TPOT](codex/R21_N22_tpot_failures.md) | 205 条超标请求、时间聚集与 prefill 干扰证据 |
| [vLLM 路线](claude/vllm/README.md) | Claude 的 vLLM 路线：官方主分支底包、A100 移植、接口插件与验证收据；[R28](claude/vllm/R28_vllm_scheduler_vs_R27.md) 对照 R27 看 vLLM 调度 |
| [GLM-5.3-Flash 官方资料](claude/R24_glm53flash_official.md) | 技术报告、博客、模型卡、KDA/DSA/mHC 论文与 SGLang/vLLM 文档的通读，原文存于 [papers/](papers/README.md)。要点：底包 HiCache 在 CUDA 上不搬索引键，搬回后输出会错；每卡 KV 12,716 B/token（FP8 为 7,260）；KDA 状态 17.6 MiB/会话；DCP 官方只验证 GB300 且不支持配 MTP；官方推荐 MTP 5/1/6 |
| [A100 上的 FP8 MoE 路径](claude/R25_moe_sm80_path.md) | 从 111 到 Marlin 的分发与执行瓶颈；EP8 开发机每层反而慢 13–24%。Humming 候选已实现并做 TP8 服务筛选，当前有效结果看下一行 |
| [117 Humming 的服务收益](codex/R32_117_humming_effect.md) | 117 相对 081 的单改、A100 FP8 权重执行路径、同 ID N30 短测与前 60 分钟完成量；说明尚未证明 N34 晋档 |
| [turn_start 尾部退化](codex/R33_turn_start_regression.md) | 正式 46251→46364 同档 turn p95 上升 62%；本地 069→081 的同 ID 坏例互换，以及为何必须用更接近正式组成的负载校准 |
| [071 与线上的瓶颈](claude/R30_071_online_bottlenecks.md) | 09-25：chain 门卡在开场冷启动潮（prefill 吞吐墙加 LPM 排序），更正“缓存没接上”说法；分叉点抢占约 9% 重算；主机层 N34 容量瓶颈是当时的预测，须以新实测复核 |
| [N30→N34 等待归因](codex/R31_n34_waiting_bottleneck.md) | 081/082 同 ID 完整回放：N34 fast 大幅恶化主要发生在准入后、首次执行前；额外重算仅 +0.8%，full-KV 压力与等待相关，但尚未区分资源拒绝与可运行越过 |
| [显存容量墙的选项](claude/R34_kv_capacity_options.md) | 09-26：每卡显存账与配置公式（mem_fraction、mamba_full_memory_ratio、槽数→并发的静默耦合）、FP8 KV 存储层已在代码里而计算层缺 tilelang 分支、DCP 115/116 只差 8 卡验证且与 MTP 的缺口在 move_kv_cache、主机层 write_back 现成可试、准入侧无抢占；按容量÷工程量排序的清单 |
| [KV 容量方案的业界调研](claude/R35_kv_capacity_survey.md) | 09-26：量化、卸载、DCP、前缀去重、准入与抢占、有损减负六方向的公开证据与对我们的排序；KDA 状态池是死角；能落地的顺序是准入调度 → 链感知卸载 → FP8 → DCP |
| [驻留账本](../scripts/analysis/residency_ledger.py) | CPU 闭环 N 槽模拟：在飞 KV 需求对比池容量、淘汰与重算量；开发集校准未通过（prefill 多算 74%），只作数据集算术上界，结果 `evidence/longchain-design-20260924/residency_ledger_v2.json`；不用于选择当前配置 |
| [阻塞窗口与执行路线复核](../notes/codex-分析-阻塞归因与执行路线.md) | blocking.py 通用工具、042/037d 重算、实测时间/窗口重合/模型估计的区别 |

旧研究中把冻结 `uncached_expected` 当真实可复用量、把在飞 prompt 总长当物理 KV 驻留、把公开榜单当对手实现证据的推断已撤回。需要复核具体说法时查 R19–R21 与相应 `evidence/`，不要从 git 历史直接恢复旧结论。
