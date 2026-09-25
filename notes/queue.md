# 实验队列

实时状态以 `scripts/pod/pread status` 为准。谁入队谁跟到完整结果；运行与协作规则见 [collaboration.md](collaboration.md)，历史实验见 [experiments.md](experiments.md)。

## 当前运行

| Job | 问题与单项变化 | 状态 |
| --- | --- | --- |
| `081-cap6144_full_n30` | 全量5601请求、N30。对完整069仅把 `SGLANG_AX_SCHED_COLD_CAP` 从4096改为6144；配置与正式attempt46364一致 | 运行中，12/60/90分钟自动分析 |
| `082-cap6144_full_n34` | 对081仅把回放并发N30改为N34；引擎参数不变，`max-running-requests=32` | 待081结束后自动运行，同样按12/60/90分钟分析 |

两轮固定引擎 `759a6ebb8e31723519ad5daf438e26e24b32501a`、host64、122/123关闭、prefill-decode-interval=2、mem.87、MTP、rep16-v1、同一冻结数据。完整回放不设时间截止，081即使SLO失败也不取消082。每分钟检查健康；12/60/90分钟之后每30分钟分析一次。运行中只观察已完成请求，**完整5601条闭合**后才按原harness判整档。081对069是单项参数比较；082对081是单项并发比较。配置与部署收据在 [full-cap6144-20260925](../evidence/full-cap6144-20260925/plan.json)。

正式attempt46364/job24605已提交078的cold6144配置，平台结果待核。今晚优先跑完081/082，明天结合正式结果再决定校准。Claude在开发机负责EP8执行层实验；Codex负责8卡队列和回放分析。124/125审阅结果另见 [review](../evidence/review-124-125-20260925/review.json)，未进入081/082。共享8卡服务保留，停单个任务用规定的 `scripts/pod/stopjob`。
