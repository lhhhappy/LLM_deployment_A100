# 数据方法与本地资料

Git 只保存生成方法、参数配方、校验脚本与必要的审计摘要。主办方请求正文、tokenizer 文件、生成的正文/requests/cohort/provenance 和缓存留在本地；本目录只跟踪这份说明。

| 要了解什么 | 入口 |
| --- | --- |
| 最新 v4 → v5 → 等待长尾 → rot150 怎样生成 | [可执行配方](../scripts/longchain/recipes/README.md)、[参数 JSON](../scripts/longchain/recipes/0930.json) |
| 基础事件、历史、token/LCP 与素材契约 | [生成设计](../scripts/longchain/longchain.md) |
| 生成、校验、发布、回放各脚本的职责 | [工具索引](../scripts/longchain/README.md) |
| 旧版为什么有偏差，元数据发布怎样修复 | [09-27 审查](../notes/reports/longchain-incremental-repair-0927.md) |
| 本地负载为何不能预测正式 N@SLO | [评估合同](../notes/evaluation.md)、[实验记录](../notes/experiments.md) |

官方公开包来自 task.md 指定的数据平台：`s1-dev/` 含311链的整链统计和722份公开请求正文，不含这些链全部5601次调用的正文。生成器保留公开部分，利用公开素材和冻结事件计划补齐缺失部分；新增内容属于合成或估计。正式隐藏集由平台独立运行。

本地最新探索负载名为 `s1-dev-longchain-v5g-tail-rot150`。输入和产物在 `s1-dev/`、`cache/`、`data/` 或指定开发机路径；输出须使用新目录，依赖保留的父正文分片。manifest、SHA256、来源账本与独立检查记录标识每一版，不能因规则相同就宣称数据字节相同。

首次克隆没有生产输入；可以运行不需原始素材的 CPU 合同测试，再按配方取得外部输入并生成。原始数据和生成数据均不进入本次 Git 提交。
