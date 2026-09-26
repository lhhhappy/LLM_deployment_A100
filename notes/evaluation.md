# 本地对照与判分

赛规以 [task.md](../llm-challenge-arena-v1/task.md) 为准；实验顺序见 [queue.md](queue.md)。本页只定义本地证据如何判有效、如何比较。公开开发集只能比较自有配置，不能预测正式 N@SLO。

## 对照

每次运行以其冻结 cohort 和请求索引为准，不假定所有数据集请求数相同。成对实验应固定 cohort、正文与顺序、派发方式、预热和真 flush、引擎提交、资源池及其他开关，只改声明的变量。闭环回放中后续请求到达时间受前驱完成时间影响；按两次运行实际共有的 request ID 对照，并报告共有数、各桶超标数、修复与新增、TPOT、实际未命中 token、错误及等待/执行账。组合实验只评价组合本身。

短探针须注明派发范围；只对已派发请求排空后作诊断，不得称完整档判分。运行中 raw 只含已完成请求，窗口标记为 open，未完成请求和未扫描候选保持未知。差异小于同配置重跑噪声时保留不确定性。

## 有效性与判分

单档只有在预期 cohort 的每个 request ID 恰好出现一次、runner 成功、测量前 flush 有效且指标齐全时才可判分；否则为 INVALID。用 `scripts/pod/verify/level_verdict.py` 核对 N、cohort、runner 与 flush 收据，再按原 harness 和 [task.md](../llm-challenge-arena-v1/task.md) 的 TTFT 统计余量及 `tpot_p95` 门判 PASS/FAIL。TPM 按 harness 固定稳态窗口计算。客户端 TPOT 依实际首末 SSE 时间和输出 token 数逐请求计算，不能用引擎单步耗时或 token 加权均值替代。

`scripts/score_formal.py` 调用未修改的公开开发集 scorer，并给出本地估计项；它不是正式评分器或正式成绩。细节与完整实验结果分别记在 [knowledge.md](knowledge.md) 和 [experiments.md](experiments.md)，原始证据放在 `evidence/`。缓存损失须基于本轮真实合法前驱和完整状态；冻结标签 `uncached_expected` 不等于本轮可复用量。排队本身不能证明排序错误，GPU 利用率也不能单独证明效率。

跨 SGLang/vLLM 对照仍须使用相同原 harness、冻结负载、并发、预热和真 flush 合同；各自内部日志只用于归因。vLLM 运行前的接口与 flush 核对见[跨引擎合同审查](reports/contract-implementation-review-0925.md)。
