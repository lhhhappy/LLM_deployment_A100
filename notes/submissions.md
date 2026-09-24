# 正式提交

这里只记提交物、官方状态与可查的出处；开发集结果放 [experiments.md](experiments.md)。正式成绩需由 Playground/主办方返回，不能由开发集推断。下表前四项状态已用 [本地公开快照](../data/all_att_2026-09-23.json)中的 attempt changelog 核对；后两项截至 2026-09-24 的本地记录仍须刷新官方状态。

| Attempt | 日期 | 配置 / 镜像 | 已核实的官方状态 |
|---|---|---|---|
| 45734 | 09-22 | A，`lh-img:0922e` | 部署失败：image ref 格式 422；未进入题目评分、不计额度 |
| 45735 | 09-22 | B，`lh-img:0922f` | 同上 |
| 45766 | 09-22 | A，`lh-img:0922e`（修正镜像名格式） | Trisol service failed；未进入题目评分、不计额度 |
| 45767 | 09-22 | B，`lh-img:0922f`（修正镜像名格式） | 同上 |
| 45979 | 09-23 15:18 UTC | A，`lh-img:0923a`；MTP+114+v3、cap4096/interval2 | QUALIFIED，`n_at_slo`=**14**（用户转述 Playground，09-24）；失败档的门与 tpot_mean 待查 |
| 45980 | 09-23 15:18 UTC | B，同镜像；MTP+114、chunk8192、无 interval | QUALIFIED，`n_at_slo`=**10**（用户转述 Playground，09-24）；失败档的门与 tpot_mean 待查 |

0923a 的补丁清单在 [build/image/0923a.patches.txt](../build/image/0923a.patches.txt)；A/B 的本地提交 JSON 分别在 [official-0923-A.json](../submission/official-0923-A.json) 和 [official-0923-B.json](../submission/official-0923-B.json)。最终以上传的 Playground attempt 为准。A/B 不能直接与开发集 028/034 视为同一运行：需核对镜像中未启用的 150/170 是否改变默认代码路径。[审查依据](../evidence/T55/B1-notes.md)

新提交按日期追加一行：attempt、镜像/配置、官方终态、两科能力分、`n_at_slo`、`tpot_mean`、结果来源；尚未返回的字段留空，不写预测。每日使用多少提交额度以平台真实计数和用户安排为准。
