# 正式提交

## 当前提交状态（2026-09-25）

**09-26 08:06 UTC只读核实：46364（job24605）执行completed，能力门通过，压测返回最高通过档N22，TPOT均值0.0226/p95 0.0438，tpm_all 1,729,474；最终`outcome=partial`、`score=98.7179`已写定（[原始输出](../evidence/official/attempt-46364-final-20260926.txt)）。** 沿用0925a/759a6eb，相对46251只把COLD_CAP 4096→6144；未启用123。N22同档chain p95 46.05→41.97秒，但fast/overall/turn p95 1.30/2.64/6.58→1.86/3.15/10.66秒。**turn 上升约 62%，虽仍低于 15 秒目标，已显著消耗余量；四桶必须同时观察。**正式并发档尚未提高。平台不返回失败档逐项，不能从本地N30 PASS推断正式N26原因。[配置与上传收据](../evidence/submission-0925-cap6144/README.md)、[turn 逐请求边界与校准计划](../research/codex/R33_turn_start_regression.md)。

**09-25 08:52 UTC核实：46251 / `lh-img:0925a` 已完成，scoreIsFinal=true，N22 PASS，能力门通过（AIME44/44、GPQA152/156）。** TPOT均值.023011343/p95 .049513025；四类TTFT p95为1.300266/2.642559/6.582520/46.049217秒。源码759a6eb、host64、122off、mem0.87、新版180、MTP，原样对应069；[提交核验](../evidence/submission-0925a/README.md)、[正式原始终态](../evidence/official/attempt-46251-final-20260925.json)、[与本地全部时间对照](reports/official-46251-local-comparison-0925.md)。正式不同N与数据的差值不能直接归因负载；更高失败档及统计余量明细未返回，不重复提交。

**22:58 UTC只读复核：两项均completed、scoreIsFinal=true，能力门通过，正式最高通过档为46173=N14、46174=N18。** 46173为A参数＋mem0.87＋新版180（0924c，源码c92acd5）；46174再加修复122（0924d，源码759a6eb，pace=.085）。两份提交都为host32GB/rank，未包含本地069的host64扩容。完整数值、能力题数与原始返回见[本轮正式结果](reports/official-46173-46174-0924.md)。

用户安排：本地继续N30研究，并按两项的冻结源码、host32和官方档位N14/N18做校准；本地数据不等于隐藏正式集，不承诺复现同一分数。原始上传证据见[提交归档](../evidence/submissions-0924/README.md)。064已停、065/066已撤销；070基础设施中断，071恢复安排见[队列](queue.md)。不重复正式提交。


这里只记提交物、官方状态与可查的出处。官方结果自己查：`scripts/official_status.sh <attempt_id>`（Playground CLI；stress 只含最高通过档）。开发集结果放 [experiments.md](experiments.md)。正式成绩需由 Playground/主办方返回，不能由开发集推断。下表前四项状态已用 [本地公开快照](../evidence/official/all_att_2026-09-23.json)中的 attempt changelog 核对；后两项于2026-09-24由Codex只读重新查询：[45979](../evidence/cost-audit-20260924/official-45979.json)、[45980](../evidence/cost-audit-20260924/official-45980.json)，均completed、能力门通过，最高通过档分别N14/N10。

| Attempt | 日期 | 配置 / 镜像 | 已核实的官方状态 |
|---|---|---|---|
| 45734 | 09-22 | A，`lh-img:0922e` | 部署失败：image ref 格式 422；未进入题目评分、不计额度 |
| 45735 | 09-22 | B，`lh-img:0922f` | 同上 |
| 45766 | 09-22 | A，`lh-img:0922e`（修正镜像名格式） | Trisol service failed；未进入题目评分、不计额度 |
| 45767 | 09-22 | B，`lh-img:0922f`（修正镜像名格式） | 同上 |
| 45979 | 09-23 15:18 UTC | A，`lh-img:0923a`；MTP+114+v3、cap4096/interval2 | QUALIFIED，能力 AIME 43/44、GPQA 153/156；`n_at_slo`=**14**：tpot_mean 0.0174、p95 0.0364，TTFT p95 fast 1.52/overall 3.87/turn 6.39/chain **30.23**（`scripts/official_status.sh`）；N18 失败档的分项平台不返回 |
| 45980 | 09-23 15:18 UTC | B，同镜像；MTP+114、chunk8192、无 interval | QUALIFIED，能力 AIME 42/44、GPQA 153/156；`n_at_slo`=**10**：tpot_mean 0.0121、p95 0.0206，TTFT p95 fast **3.46**/overall 3.72/turn 4.91/chain 10.89；N14 失败档的分项平台不返回 |
| 46173 | 09-24 14:26 UTC | A+mem0.87+新版180，`lh-img:0924c`，c92acd5；MTP保留、122 off、host32 | completed；能力AIME44/44、GPQA150/156，门通过；N14 PASS，TPOT均值0.016540932/p95 0.031841836；[原始结果](../evidence/official/attempt-46173-20260924.json)、[准确配置](../evidence/image-0924d/submission_slotA_mem087_hicache_0924c.json) |
| 46174 | 09-24 14:27 UTC | 同46173+修复122，`lh-img:0924d`，759a6eb；pace=.085、host32 | completed；能力AIME44/44、GPQA153/156，门通过；N18 PASS，TPOT均值0.020518278/p95 0.039681591；[原始结果](../evidence/official/attempt-46174-20260924.json)、[准确配置](../evidence/image-0924d/submission_slotB_mem087_hicache_122fix_0924d.json) |
| 46251 | 09-24 23:23 UTC（09-25 07:23 +08） | 069原样，`lh-img:0925a`，759a6eb；mem0.87、新版180、host64、122off、MTP | completed/final；能力AIME44/44、GPQA152/156，N22 PASS，TPOT均值.023011343/p95 .049513025；[原始终态](../evidence/official/attempt-46251-final-20260925.json)、[完整对照](reports/official-46251-local-comparison-0925.md) |
| 46364 | 09-25 15:18 UTC | 078配置；沿用0925a/759a6eb，相对46251仅cold cap4096→6144 | `exec=completed`、能力门通过，压测最高通过N22，TPOT均值.0226/p95 .0438；`outcome=partial`、score 98.7179（09-26 08:06 UTC 核实）；[收据](../evidence/submission-0925-cap6144/README.md) |

**源码对应（2026-09-24 迁移到 git 后）**：镜像 0923a（45979/45980）的引擎源码 = git 标签 `official-A-0923a`（也标为 `image-lh-img-0923a`）。依据：镜像源码与 13 补丁栈逐文件一致（T57，4690 个文件），该补丁栈与标签逐文件一致（迁移核对）。`git show official-A-0923a` 可直接查看，与候选的差异用 `git diff official-A-0923a HEAD -- engine/sglang`。09-22 的 0922e/0922f 部署失败、未评分，未迁移为标签，其补丁留在 git 历史。旧补丁清单在 [build/image/0923a.patches.txt](../build/image/0923a.patches.txt)；A/B 的本地提交 JSON 分别在 [official-0923-A.json](../submission/official-0923-A.json) 和 [official-0923-B.json](../submission/official-0923-B.json)。最终以上传的 Playground attempt 为准。A/B 不能直接与开发集 028/034 视为同一运行：需核对镜像中未启用的 150/170 是否改变默认代码路径。[审查依据](../evidence/T55/B1-notes.md)

**新提交怎么做与怎么记**：镜像用 `scripts/build_image.sh <提交>` 生成 Dockerfile（在我们已注册的镜像上只叠加增量，默认叠在 0923a 上），构建后给该提交打 `image-<镜像名>` 标签；本页每行写 attempt、引擎提交号（或标签）、镜像名、启动参数与 env 的出处。

新提交按日期追加一行：attempt、镜像/配置、官方终态、两科能力分、`n_at_slo`、`tpot_mean`、结果来源；尚未返回的字段留空，不写预测。每日使用多少提交额度以平台真实计数和用户安排为准。

**后续正式提交的写盘要求（用户确认）**：关闭请求正文、张量、性能trace等调试落盘；降低高频INFO/HTTP访问日志，保留启动配置收据、错误与警告；单独核验编译缓存和临时目录的磁盘预算。不能把host缓存算作磁盘，也不能靠关闭必要JIT或篡改评测打点省空间。本次0925a保持069配置，未追加日志参数变化。
| 46676 | 09-26 15:33 UTC（08:33 洛杉矶） | S1 = A′：`lh-img:0926b`（引擎 f546934e，tag `image-lh-img-0926b`，FROM 0925a）；46364 的命令逐字不变，env 新增 117 Humming、124 截止时间分层、激进 125（冷块 8192、interval 0、HIGH 15/LOW 5、MAX_SLOW 80、GATE 0.10）、`SGLANG_AX_DEADLINE_WARM_S=15`；引擎相对 4f9d1f0b 多了 84dcca0e（默认中性）和审查修正 124 停车/110 守卫/117 缓存/180 销毁 | **exec=failed（22:00 UTC 核实）**：平台记录“trisol deploy failed: service not running after 3600s (last status deploying)”，未进入评分、不计入每日额度；评测服务已被平台删除、日志不可得，用户决定暂不追查（可能是平台问题）；46677 同镜像仍 queued，其结果可作对照。本地依据 130b；本地依据 130b（60 分钟对 112：chain 26→30、turn 24→16）、130ea/130eb 探针与冒烟 12/12；[收据](../evidence/submission-0926-s1/README.md) |
| 46677 | 09-26 15:33 UTC | S2 = S1 + `SGLANG_AX_BACKLOG_MAX_SLOW=250`，其余逐字节相同（激进臂：125 护栏 80→250，让积压模式在 14 分钟后继续工作） | queued（15:35 UTC 核实）；未做 60 分钟确认，用户决定直接交，确认运行 130ec 排队中；[收据](../evidence/submission-0926-s2/README.md) |
| 46757 | 09-27 00:27 UTC（09-26 17:27 洛杉矶） | S3 = S1 + DCP2：`lh-img:0926c`（引擎 e464d8ab = f546934e + DCP 三补丁），命令加 `--dcp-size 2`、`--max-mamba-cache-size 418`、running/graph 48，env 加 DCP_COMPACT_TOPK=1、DSA_SPARSE_TRITON=0 | queued（00:30 UTC 核实）；本地依据 130ed/130ee、130ee5/130eez（DCP 同引擎对照）与 130eezzy（公开题 AIME 28/30、GPQA 178/197）；[收据](../evidence/submission-0926-s3/README.md) |
| 46758 | 09-27 00:31 UTC | S4 = 46757 + `SGLANG_AX_DCP_LOCAL_EXTEND=1`：`lh-img:0926d`（引擎 6976639e = e464d8ab + Codex 本地续算路径 + 审查修正），命令相同 | queued；实验臂，上传前未经 TP8（探针 130ez4/5/6 排队中）；[收据](../evidence/submission-0926-s4/README.md) |
