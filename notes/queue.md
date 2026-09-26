# 实验队列

实时状态以 `scripts/pod/pread status` 为准。谁入队谁跟到完整结果；运行与协作规则见 [collaboration.md](collaboration.md)，历史实验见 [experiments.md](experiments.md)。

## 当前队列

| Job | 问题与单项变化 | 状态 |
| --- | --- | --- |
| `090-k9v2_base_n26` | K9v2/N26 原46364配置；建立同一请求集合上的基线 | 运行中；脚本SHA256 `5b523af9` |
| `091-k9v2_117_n26` | 对090只开启117 Humming FP8 MoE；检验吞吐收益能否兑现为chain/turn | 已入队，排在090后；脚本SHA256 `77e30cef` |
| `092-k9v2_118_n26` | 对090只开启118 Triton DSA sparse kernel；与117独立横向比较 | 已入队，排在091后；脚本SHA256 `804915ae` |

三项固定每链前9请求、311链/1865请求、N26、冷块6144、8192预填块、host64、MTP、rep16预热与真flush；各跑完整K9v2并用原harness判分，按同ID比较四桶、TPOT、重算和完成时间。K9与正式分布未对齐，因此这里只筛选相对变化，不外推正式N26。用户已将预算参数实验推后，先看新执行路径。089 已全量完成：`VALID FAIL`，1865/1865请求、零错误；失败门为fast和overall，详见[实验记录](experiments.md)。

用户把优先级改为先对齐正式负载：086/117 N34 在1038条完成时主动停单个job，保留窗口证据；087/117+119 尚未启动，已从pending撤下，不自动续排。088因正文相对链接在Pod的`/tmp/ax`符号链接下指错，预热16/16均`HARNESS_DATA:body_missing`、测量未开始，标为INVALID。089只修K9数据包的路径生成，配置仍与46364相同。089的请求数与prompt总量接近正式一档，但四桶与TPOT没有对齐；不能将K9用于预测正式通过档。已完成的081/082/084等见[实验记录](experiments.md)，117见[R32](../research/codex/R32_117_humming_effect.md)，正式46364见[提交记录](submissions.md)，turn差异见[R33](../research/codex/R33_turn_start_regression.md)。共享8卡服务保留；停单个任务只用`scripts/pod/stopjob`。
