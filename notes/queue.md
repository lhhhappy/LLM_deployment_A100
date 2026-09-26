# 实验队列

实时状态以 `scripts/pod/pread status` 为准。谁入队谁跟到完整结果；运行与协作规则见 [collaboration.md](collaboration.md)，历史实验见 [experiments.md](experiments.md)。

## 当前队列

| Job | 问题与单项变化 | 状态 |
| --- | --- | --- |
| `097-v3_open_base_n26` → `098-v3_open_124_n26` → `099-v3_open_124_125_n26` → `100-v3_open_124_cap8192_n26` → `101-v3_open_124_117_n26` | 同一冻结 v3、同一 stack2 提交；依次测试排序/停车、积压模式、固定大冷块、117。各 600 秒派发后排空，按同 ID 先看 chain 再看 turn | 已部署，097 运行中，其余待运行；自动开场报告已挂载 |
| `102-v3_46364_n26_full` | 46364 原引擎与配置，v3/N26 整集 5601 条全部派发并排空，检验本地是否复现线上 N26 失败 | 已排在 A 后；不设派发截止 |

赛题[task.md](../llm-challenge-arena-v1/task.md)第270、322行明确公开dev与正式压测使用同一套harness和评分口径；公开harness开场同时放出至多N条链。此前认为线上可能错峰的推断撤回。095每2秒放链不再作为线上校准，尚未启动即从pending撤下；错峰代码与脚本保留为机制探针，不占这轮Pod时间。094保留以量公开dev原样负载的开场问题，但不能由dev分数预测正式N26失败门。

v3 数据上传已收到 `PUSH_OK 7650 chunks`，启动时还会核 cohort、requests SHA256 与正文文件；全量渲染校验在开发机并行，若报错须作废实验。v2 的 124 探针未启动即撤队；六项任务按编号确保 worker 的文件名顺序符合计划。094 的公开 dev 原样开场短测已派发并排空 722/722 条；原 harness 报告四类 TTFT 均 FAIL，chain p95 为 69.69 秒。它是 18 分钟的开发集机制诊断，不是正式同档校准。附加的定时窗口 verdict 因找不到预期的 cohort 文件报 INVALID，不能用其判分；原 harness 报告和 raw 均保留。用户转向 v3/N26 后，096 在启动阶段按单 job 流程停止，未产生可比较结果，8 卡服务保留。124/125 复核见[审阅](../evidence/stack2-125-review-20260926.md)：探针可上卡，125 的慢请求护栏不是 TPOT 硬保证；stack2 默认还含 120 对齐修复。093已排空：对081同ID 3389条，chain超时23→21、turn7→5、fast238→241、overall190→186，TPOT均值33.14→31.87 ms，错误0；这是60分钟派发诊断，并非完整N30成绩。既有完整实验见[experiments.md](experiments.md)，117见[R32](../research/codex/R32_117_humming_effect.md)。
