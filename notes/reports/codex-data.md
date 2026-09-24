# 数据 Codex 状态

- 仅数据与CPU工作，不操作8卡队列。唯一成品在[data/s1-dev-longchain](../../data/s1-dev-longchain/)，当前规模/验收进度见[data入口](../../data/README.md)。
- 新事件生成器已实现：继承各源链system/tools，保留公开输入和预算，跨session借query、续接完整工具组、明确重建并继续增长；正文均来自s1-dev，Phoenix生产正文不入成品。
- Phoenix千session采集完成，55,860 LLM span、44,119主调用、236条压缩观测；分层过采长尾，不能直接把原始计数称为线上比例。[结构证据](../../evidence/phoenix-longchain-20260924/expanded/behavior-profile.json)
- 新增gap目前是明确估计的end-to-start proxy，工具/用户等待分解未知；事件频率受源摘要约束，位置与内容组合为合成，不能声称还原隐藏集。Luna试制素材已归入[evidence素材目录](../../evidence/longchain-design-20260924/materials/)，本版未接入。
- 三路bug审查已覆盖生成状态、独立checker与原harness接口；真实token/LCP用原GLM Renderer验收，CPU替身只用于错误分支回归。[生成审查](../../evidence/longchain-design-20260924/review-v2-generation.md)、[checker审查](../../evidence/longchain-design-20260924/review-v2-checker.md)、[接口审查](../../evidence/longchain-design-20260924/review-v2-harness.md)
- 旧96链/1718请求候选原字节归档至`cache/longchain-legacy/`，冻结收据仍在[evidence](../../evidence/longchain-audit/frozen-candidate/README.md)。失败/过时生成中间件已清除，迁移记录见[data-relocation.json](../../evidence/longchain-design-20260924/data-relocation.json)。不再把旧候选作为当前集。
- 311链/5601请求为第一份完整扩展集目标，沿用源摘要链长，非正式341链/5150请求的恢复。没有新数据GPU成绩，也不预测正式N@SLO。

设计、复现命令和实现边界以[longchain.md](../../scripts/analysis/longchain.md)为准；文档索引见[交接](../codex-handoff-长链数据设计与文档索引.md)。
