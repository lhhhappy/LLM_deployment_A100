# 现行评估口径

赛规只认 [task.md](../llm-challenge-arena-v1/task.md)。本页规定本地实验如何比较；实时任务和监控节奏见 [queue.md](queue.md)。正式排名按最高通过并发、该档较低TPOT均值、较高TPM、较早提交顺序。四道TTFT门按题面统计余量；TPOT p95不设余量。

## 实验与对照

完整本地长链集固定311链、5601请求、原顺序/正文/输出预算/gap；同档比较要保持数据、预热、flush、引擎commit、资源池与其他开关一致，只改变声明的变量。闭环回放使后续实际到达时间随前驱完成时刻变化；同ID比较不能假定请求在不同运行中同时到达。组合实验只评价组合。短探针须明确派发范围，已派发请求排空后只报诊断，不判完整档。

每轮先讲清问题、基线、改动、证据和资源代价。按正常工程判断选择源码审查、CPU、开发机或8卡验证；CPU探针不作强制前置。冻结运行不热改。记录源码与运行工具版本、完整启动参数、cohort哈希、预热、真flush、raw和服务日志。不同预热协议不能标成单变量性能对照。输出、thinking、tools、时间戳和token计数须如实保持。

## 判分和分析

只有5601个请求各恰好一次、runner成功、本轮flush有效且指标齐全，才用原harness（`scripts/score_formal.py`、`level_verdict.py`）加题面TTFT统计余量及TPOT p95门判VALID PASS/FAIL；缺数据是INVALID。TPM沿用原评分器固定稳态窗口。客户端TPOT从实际首末SSE与输出token数计算，不用引擎单步耗时或token加权均值替代。

运行中raw仅包含已完成请求，所有窗口标open。做同ID比较时报告四桶样本与超标、修复与新增、TPOT、实际未命中token、错误和等待/执行账；未完成请求及未扫描候选记未知。不能把`uncached_expected`直接称作本轮缓存损失，不能仅凭排队就归因排序，也不能把GPU高利用率直接当高效率。差异小于同配置重跑噪声时保留不确定性。完整结果进 [experiments.md](experiments.md)，可复用结论进 [knowledge.md](knowledge.md)，原始记录进`evidence/`。

## 运行监控

当前081/082每分钟只读检查队列、raw/服务日志推进及错误；测量后12/60/90分钟保存自动分析，之后每30分钟一次。到点用已完整验证的069 raw做相同ID对照。只因某个局部窗口超门不停止完整回放。若确认严重正确性或基础设施错误，先留原始证据，再按仓库规则停单个job；共享8卡服务不可停删释放。监控失联单独标明，不把旧快照当当前状态。自动分析与通知工具是 `scripts/analysis/window_watch.py`、`window_notify.py`。

SGLang与vLLM若跨引擎比较，仍用同一原harness、冻结负载、并发、预热和真flush合同；各自的资源配置与内部日志可不同。引擎专有日志缺失不构成另一引擎的失败；首次入批、缓存命中、MTP接受率等内部指标只供归因。vLLM运行前须核对接口和flush收据，现行审查见 [跨引擎合同](reports/contract-implementation-review-0925.md)。
