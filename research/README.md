# 调研与源码地图

从仓库根部 [README](../README.md) 了解当前实验、队列和成绩；赛题以 [task.md](../llm-challenge-arena-v1/task.md) 为准。本目录只保留能支持现行代码或解释已核实实验结果的研究。历史版本留在 git，不作为当前结论。

| 文件 | 用途 |
|---|---|
| [底包与当前认识](claude/base/00-summary-mainline.md) | 底包来源、S0 与主要约束、过时结论的更正 |
| [请求入口](claude/base/01-request-path.md)、[调度器](claude/base/02-scheduler.md)、[混合缓存](claude/base/03-hybrid-cache.md)、[模型与算子](claude/base/04-model-kernels.md) | `build/base_exact/sglang` 的源码地图；运行时结论需结合现行补丁与实验 |
| [上游候选](claude/R9_upstream_since_base.md) | 以底包为起点的上游差异；候选不代表已在本栈验证 |
| [prefill 固定开销](claude/R10_prefill_fixed_overhead.md) | 开发机替身模型 profile 和 8 卡推断的边界 |
| [MTP 路径](codex/R17_nextn_sm80.md) | 补丁 160 的实现与兼容性 |
| [缓存与显存](codex/R18_cache_loss_and_capacity.md) | 早期 N6 账本；可修缺口的最新量化看 R20 |
| [分析与工具复核](codex/R19_progress_and_cache_review.md) | 纠正原先的缓存归因、发现会导致假通过的分析脚本问题 |
| [真实 LCP 归因](codex/R20_true_lcp_attribution.md) | 026/N18 的逐请求 LCP、fast 超时与短输出 TPOT |
| [035/N22 TPOT](codex/R21_N22_tpot_failures.md) | 205 条超标请求、时间聚集与 prefill 干扰证据 |
| [高并发与达标请求数](../notes/reports/chain-goodput-objective-0927.md) | N@SLO 目标、124 现有可行性分层与长 partial 的缺口；含两个 6sol 子代理的外部调度/执行实践调研入口，均未作为新性能结果 |

旧研究中把冻结 `uncached_expected` 当真实可复用量、把在飞 prompt 总长当物理 KV 驻留、把公开榜单当对手实现证据的推断已撤回。需要复核具体说法时查 R19–R21 与相应 `evidence/`，不要从 git 历史直接恢复旧结论。
