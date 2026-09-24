# 当前长程测试集

唯一成品：**[s1-dev-longchain/](s1-dev-longchain/)**。

可以开始回放；可直接复制的命令见[运行入口](../scripts/longchain/README.md#直接回放当前成品)。

- **311条链、5601个请求**：保留原722个请求，新增4879个；新增139次用户追问、118次历史重建。
- 中位链长8次，p95为82次，最长240次。次数指模型调用，包含工具续跑，不是用户轮数。
- 独立原GLM渲染检查 **VALID，5601/5601，0错误**；原harness全量自检 **PASS，0失配**。
- 验收状态见[acceptance.json](s1-dev-longchain/acceptance.json)。manifest的BUILT_UNVALIDATED是构建时冻结状态，后续验收由此独立收据记录。
- 没有GPU性能成绩，尚未做N槽闭环驻留模拟；合成测试集不能预测正式N@SLO。

| 每链模型调用 | 链数 | 请求数 |
|---|---:|---:|
| 1–4 | 119 | 253 |
| 5–8 | 48 | 316 |
| 9–15 | 64 | 734 |
| 16–30 | 30 | 671 |
| 31–99 | 43 | 2453 |
| 100+ | 7 | 1174 |

正文来自s1-dev素材；Phoenix只提供冻结的行为结构参考。链长/phase计数沿用源摘要，事件位置与等待分解仍有合成估计；新算量分布尚未恢复，详见[完整审计](../evidence/longchain-design-20260924/full-distribution.md)与[生成说明](../scripts/longchain/longchain.md)。

回放使用原runner，只换root=`data/s1-dev-longchain`、set=`s1-dev-longchain`、cohort=`data/s1-dev-longchain/cohort.json`。每档重新flush，使用新的运行输出目录，评分指向本集requests；其他流程不改。

`data/`只保留这一份成品和本说明。小集、旧副本、失败产物、Phoenix原文缓存及临时正文缓存已清除；`cache/`已清空并删除。原始开发集仍在只读`s1-dev/data/dev-combined-v1/`；生成命令与复现入口见[longchain.md](../scripts/longchain/longchain.md)，结构输入和检查记录保留在`evidence/`。
