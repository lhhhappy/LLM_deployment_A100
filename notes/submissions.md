# 正式提交

这里只记提交物、官方状态与可查的出处。官方结果自己查：`scripts/official_status.sh <attempt_id>`（Playground CLI；stress 只含最高通过档）。开发集结果放 [experiments.md](experiments.md)。正式成绩需由 Playground/主办方返回，不能由开发集推断。下表前四项状态已用 [本地公开快照](../evidence/official/all_att_2026-09-23.json)中的 attempt changelog 核对；后两项于2026-09-24由Codex只读重新查询：[45979](../evidence/cost-audit-20260924/official-45979.json)、[45980](../evidence/cost-audit-20260924/official-45980.json)，均completed、能力门通过，最高通过档分别N14/N10。

| Attempt | 日期 | 配置 / 镜像 | 已核实的官方状态 |
|---|---|---|---|
| 45734 | 09-22 | A，`lh-img:0922e` | 部署失败：image ref 格式 422；未进入题目评分、不计额度 |
| 45735 | 09-22 | B，`lh-img:0922f` | 同上 |
| 45766 | 09-22 | A，`lh-img:0922e`（修正镜像名格式） | Trisol service failed；未进入题目评分、不计额度 |
| 45767 | 09-22 | B，`lh-img:0922f`（修正镜像名格式） | 同上 |
| 45979 | 09-23 15:18 UTC | A，`lh-img:0923a`；MTP+114+v3、cap4096/interval2 | QUALIFIED，能力 AIME 43/44、GPQA 153/156；`n_at_slo`=**14**：tpot_mean 0.0174、p95 0.0364，TTFT p95 fast 1.52/overall 3.87/turn 6.39/chain **30.23**（`scripts/official_status.sh`）；N18 失败档的分项平台不返回 |
| 45980 | 09-23 15:18 UTC | B，同镜像；MTP+114、chunk8192、无 interval | QUALIFIED，能力 AIME 42/44、GPQA 153/156；`n_at_slo`=**10**：tpot_mean 0.0121、p95 0.0206，TTFT p95 fast **3.46**/overall 3.72/turn 4.91/chain 10.89；N14 失败档的分项平台不返回 |

0923a 的补丁清单在 [build/image/0923a.patches.txt](../build/image/0923a.patches.txt)；A/B 的本地提交 JSON 分别在 [official-0923-A.json](../submission/official-0923-A.json) 和 [official-0923-B.json](../submission/official-0923-B.json)。最终以上传的 Playground attempt 为准。A/B 不能直接与开发集 028/034 视为同一运行：需核对镜像中未启用的 150/170 是否改变默认代码路径。[审查依据](../evidence/T55/B1-notes.md)

新提交按日期追加一行：attempt、镜像/配置、官方终态、两科能力分、`n_at_slo`、`tpot_mean`、结果来源；尚未返回的字段留空，不写预测。每日使用多少提交额度以平台真实计数和用户安排为准。
