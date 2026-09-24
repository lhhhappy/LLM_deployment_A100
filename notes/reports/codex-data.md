# 数据 Codex 状态

- 仅数据与CPU工作，不操作8卡队列。唯一成品在[data/s1-dev-longchain](../../data/s1-dev-longchain/)，当前规模/验收进度见[data入口](../../data/README.md)。
- 新事件生成器已实现：继承各源链system/tools，保留公开输入和预算，跨session借query、续接完整工具组、明确重建并继续增长；正文均来自s1-dev，Phoenix生产正文不入成品。
- Phoenix千session采集完成，55,860 LLM span、44,119主调用、236条压缩观测；分层过采长尾，不能直接把原始计数称为线上比例。[结构证据](../../evidence/phoenix-longchain-20260924/expanded/behavior-profile.json)
- 新增gap目前是明确估计的end-to-start proxy，工具/用户等待分解未知；事件频率受源摘要约束，位置与内容组合为合成，不能声称还原隐藏集。Luna试制素材已归入[evidence素材目录](../../evidence/longchain-design-20260924/materials/)，本版未接入。
- 三路bug审查已覆盖生成状态、独立checker与原harness接口；真实token/LCP用原GLM Renderer验收，CPU替身只用于错误分支回归。[生成审查](../../evidence/longchain-design-20260924/review-v2-generation.md)、[checker审查](../../evidence/longchain-design-20260924/review-v2-checker.md)、[接口审查](../../evidence/longchain-design-20260924/review-v2-harness.md)
- 旧96链/1718请求候选及父数据副本已按用户要求删除，冻结收据仍在[evidence](../../evidence/longchain-audit/frozen-candidate/README.md)。失败/过时生成中间件、Phoenix原始正文cache和旧工具下载缓存已清除，迁移记录见[data-relocation.json](../../evidence/longchain-design-20260924/data-relocation.json)。不再把旧候选作为当前集。
- 311链/5601请求完整扩展集已交付，沿用源摘要链长，非正式341链/5150请求的恢复。没有新数据GPU成绩，也不预测正式N@SLO。

设计、复现命令和实现边界以[longchain.md](../../scripts/analysis/longchain.md)为准；文档索引见[交接](../codex-handoff-长链数据设计与文档索引.md)。

## 本轮完整集事实

- 311链、5601请求：保留722请求，新增4879；新增139个用户事件、118次重建、4622次内部续跑。链长中位8、p95 82、最大240；119链只有1–4次调用，50条31+链贡献3627请求。
- 全量独立检查已VALID：5601正文/token/LCP，0错误；原harness self-check也已PASS、5601/5601、0失配；完整成品已放入data/s1-dev-longchain，小集与全部cache已删除。
- [完整分布审计](../../evidence/longchain-design-20260924/full-distribution.md)：跨session整份prompt重复只有原source已有的同一组；48,205对链首未新增相对source的长共享历史。普通素材在不同前缀下有复用，不等于物理cache命中。
- 代表性限制仍在：累计prompt393.887M，源摘要485.059M；缺失请求的源冻结新增预算18.936M，生成部分实际LCP新增5.109M。两种账本并非GPU测量，但差额说明仅匹配链长/phase不能宣称算量分布已恢复。总输出4.108M，源4.105M，3条源预算冲突已披露。
- 已做的workload ledger是静态token三本账；**尚未做N槽闭环驻留模拟，也未据此选N**。后者需引入请求耗时假设、共享前缀、淘汰及KV/KDA两类池，CPU模拟不能证明SLO通过档。
