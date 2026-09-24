# 统一评估与迭代协议

2026-09-24 用户决定：当前由Codex单人持续优化SGLang，vLLM暂缓。
赛规以task.md及已记录的主办方排名更正为准。本文件定义本地比较条件，不改变官方规则。

## 一轮与两个评估用途

| 用途 | 预热 | 测量 | 结果边界 |
| --- | --- | --- | --- |
| 日常迭代 `longchain-n30-rep16-v1` | preflight一条链；固定rep16-v1的16个完整请求；真实flush | 全量311链/5601请求，固定N30，原顺序/正文/输出/gap | 全量闭合后可报本地VALID PASS/FAIL；运行中只作诊断 |
| 正式校准 | 与要复现的正式流程对齐，使用原harness预热 | 相同提交配置、对应并发档，记录数据差异 | 量化本地与正式差异，不能从本地直接换算正式N |

rep16-v1定义及覆盖边界见[组合与短预热](reports/sglang-shortwarm-mainline-0924.md)。
每轮记录引擎commit、运行工具commit/实际文件、命令/env、GPU型号与TP、N、数据/cohort哈希、
预热plan SHA、清缓存receipt、raw/run/summary。不同预热协议结果不能当单变量性能对照。
本地不压输出、不关thinking、不截历史、不删tools；预热不计分，测量仍跑每一个冻结请求。

## 运行中必须观察

- 阶段：加载/捕图、preflight、warmup、flush、measurement、scoring，分别记录耗时。
- 每分钟只读健康快照：任务状态、最新完成量、raw/server日志更新时间、最近错误与batch、8卡采样。
- 超过5分钟没有新的完成记录，或服务日志不再推进，记为调查提示；长请求/gap/尾段也可能造成，不能直接定为死锁。
- 每15分钟保存TTFT四桶的样本数、超标数、余量、p95；TPOT均值/p95；错误和token/cache账本。
- 当前raw只包含完成请求，没有完整在途发送台账；所有运行中窗口保持open，不能因为漏掉慢请求而判通过。
- 用户要求的30分钟重点检查：是否崩溃/OOM、持续碎块、明显无进展、元数据/清缓存/请求丢失等严重问题。
  证据明确时先写报告、保存原始文件和最近日志，再用stopjob停单个任务；不自动杀任务或释放8卡服务。
- 只因TTFT/TPOT超门不立即中止；先判断是否持续饱和/缓存容量/执行成本等性能问题，必要时继续到完整结果。

监控程序只读；连接失败自身重试，并记录监控不可用，不伪报引擎失败。
`window_watch.py`保存health.json、health_history.jsonl、alerts.jsonl和窗口JSON/HTML。
原metrics每10秒、GPU每5秒采样继续保留。服务日志不是完整内部trace：不能把重算差额全部说成淘汰，
不能仅凭排队说调度有错，也不能把GPU利用率100%说成高效。

## 全量判定

5601个请求各恰好一次、runner成功、本轮flush真实成功、指标齐全后，用原harness和score_formal：
coverage=100%、harness_data=0、harness_render=0、engine_error<1%、infra_error<1%、
四道TTFT门、gated_phases_have_samples、tpot_p95≤0.10，合计11门。
四道TTFT使用题面统计余量；TPOT p95无余量。缺数据INVALID，有效但任一门不过FAIL。
TPM使用原评分器固定稳态窗口及其有效性检查，不能用全程墙钟平均代替。
保留全部失败请求ID和逐请求CSV，关联服务日志；同请求跨运行配对，再报告整体差异。

## 迭代与仓库入口

1. 按“活变少、活变便宜、活排得更好”分类问题。这是三个可分工方向，收益相互耦合，不相加。
2. 每轮写问题/基线/唯一变量/预期证据/资源代价/停止条件；组合轮只评价组合整体。
3. CPU或开发机先验证，TP8做确认。小差异且没有噪声依据时保留不确定性。
4. 谁提交任务谁跟踪、取证和分析；任务安排只记[queue.md](queue.md)，个人过程只记[Codex迭代日志](iterations/codex.md)。
5. 确认的完整实验写experiments.md；成立且可复用的结论写knowledge.md/research/；代码按机制提交。
6. 当前运行库只支持SGLang。将来接其他引擎时先适配启动与三接口，再复用这套负载和评分；不直接复用SGLang启动器。

当前入口：README导航；queue实时调度；本页评估协议；iterations个人索引；reports长篇归因；
evidence原始证据；engine源码与机制文档。旧候选保留追溯，但不列为当前正在验证的结果。
