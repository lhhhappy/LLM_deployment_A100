# Codex 调研

当前项目状态与操作入口见仓库根部 [README](../../README.md)；底包与研究索引见 [research/README](../README.md)。报告保留原始研究口径，运行时结论以更新的实测和复核为准。

| 报告 | 内容 |
|---|---|
| [R17](R17_nextn_sm80.md) | A100 上 NEXTN/MTP 的代码路径与补丁 160 |
| [R18](R18_cache_loss_and_capacity.md) | N6 缓存生命周期、静态显存池与容量账本 |
| [R19](R19_progress_and_cache_review.md) | 026 原始数据重算；分析、数值和判分工具缺陷 |
| [R20](R20_true_lcp_attribution.md) | 026/N18 真实 token LCP 的逐请求归因及 8.0% 上限 |
| [R21](R21_N22_tpot_failures.md) | 035/N22 的 205 条 TPOT 超标请求与时间聚集 |
| [R22](R22_kda_projection_fusion.md) | 171 KDA 投影融合：加载/状态/MTP 算子检查、单卡实测时间与显存，实际 TP8 待验 |

R18 的较早容量情景不能代替 R20 的真实 LCP 上限；R19 的工具缺陷描述的是发现时的版本，修复状态看当前脚本与测试。
