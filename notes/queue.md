# 实验队列

实时状态以 `scripts/pod/pread status` 为准。谁入队谁跟到完整结果；运行与协作规则见 [collaboration.md](collaboration.md)，历史实验见 [experiments.md](experiments.md)。

## 当前队列

| Job | 问题与单项变化 | 状态 |
| --- | --- | --- |
| `094-dev_opening_release0_n22` | 公开 dev 集/N22、46364 配置，原 harness 同时放链；核对开场冷请求与队列，须核对722条是否全部放出并排空 | 运行中，窗口监控已启动 |
| `096-calib_k9v2_117_n22` | 对089同一K9v2/N22的1865条请求，只开启117 Humming MoE；看四门与TPOT如何变化 | 已入队，排第二；117能力抽检开启 |

赛题[task.md](../llm-challenge-arena-v1/task.md)第270、322行明确公开dev与正式压测使用同一套harness和评分口径；公开harness开场同时放出至多N条链。此前认为线上可能错峰的推断撤回。095每2秒放链不再作为线上校准，尚未启动即从pending撤下；错峰代码与脚本保留为机制探针，不占这轮Pod时间。094保留以量公开dev原样负载的开场问题，但不能由dev分数预测正式N26失败门。

093已排空：对081同ID 3389条，chain超时23→21、turn7→5、fast238→241、overall190→186，TPOT均值33.14→31.87 ms，错误0；这是60分钟派发诊断，并非完整N30成绩。K9v2的089虽全量有效，仍与线上四桶分布不符；096只用于同数据上衡量117，不预测正式档。v3数据仍在补提醒轮换，冻结后再决定校准任务。既有完整实验见[experiments.md](experiments.md)，117见[R32](../research/codex/R32_117_humming_effect.md)。共享8卡服务保留；停单个测试只用`scripts/pod/stopjob`。
