# 实验队列

实时状态以 `scripts/pod/pread status` 为准。谁入队谁跟到完整结果；运行与协作规则见 [collaboration.md](collaboration.md)，历史实验见 [experiments.md](experiments.md)。

## 当前队列

| Job | 问题与单项变化 | 状态 |
| --- | --- | --- |
| `v2_open_124_n26` | v2 311 链/N26，stack2 开 124，600 秒派发后排空；看小链首排序与续块停车的 TP8 首次信号。相对 078 还含 120 对齐修复，只能作宽松参照 | 准备入队 |
| v3/N26 开场四轮 | 同一冻结 v3、同一 stack2 提交，依次关新开关、开 124、124+125、124+117；各 600 秒派发后排空，按同 ID 先看 chain 再看 turn | 待 v3 哈希冻结，脚本已合并 |

赛题[task.md](../llm-challenge-arena-v1/task.md)第270、322行明确公开dev与正式压测使用同一套harness和评分口径；公开harness开场同时放出至多N条链。此前认为线上可能错峰的推断撤回。095每2秒放链不再作为线上校准，尚未启动即从pending撤下；错峰代码与脚本保留为机制探针，不占这轮Pod时间。094保留以量公开dev原样负载的开场问题，但不能由dev分数预测正式N26失败门。

094 的公开 dev 原样开场短测已派发并排空 722/722 条；原 harness 报告四类 TTFT 均 FAIL，chain p95 为 69.69 秒。它是 18 分钟的开发集机制诊断，不是正式同档校准。附加的定时窗口 verdict 因找不到预期的 cohort 文件报 INVALID，不能用其判分；原 harness 报告和 raw 均保留。用户转向 v3/N26 后，096 在启动阶段按单 job 流程停止，未产生可比较结果，8 卡服务保留。124/125 复核见[审阅](../evidence/stack2-125-review-20260926.md)：探针可上卡，125 的慢请求护栏不是 TPOT 硬保证；stack2 默认还含 120 对齐修复。093已排空：对081同ID 3389条，chain超时23→21、turn7→5、fast238→241、overall190→186，TPOT均值33.14→31.87 ms，错误0；这是60分钟派发诊断，并非完整N30成绩。既有完整实验见[experiments.md](experiments.md)，117见[R32](../research/codex/R32_117_humming_effect.md)。
