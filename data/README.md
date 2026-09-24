# 可回放测试数据

唯一当前成品：**[s1-dev-longchain/](s1-dev-longchain/)**。

- 当前为24链、455请求的长程小集；正文来自s1-dev，Phoenix仅提供行为结构参考。
- 独立真实GLM渲染检查VALID，原harness全量self-check通过；没有GPU性能成绩。
- 完整311链、5601请求版本在`cache/s1-dev-longchain-build/`生成，验收后才替换成品。
- 生成方式及限制：[longchain.md](../scripts/analysis/longchain.md)。

`data/`只放成品，不放生成中间件。原始开发集仍在只读`s1-dev/data/dev-combined-v1/`；素材与验收记录在`evidence/longchain-design-20260924/`，旧冻结候选在`cache/longchain-legacy/`。清理迁移记录：[data-relocation.json](../evidence/longchain-design-20260924/data-relocation.json)。
