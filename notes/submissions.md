# 正式提交与当前有效版本

## 最新有效提交：47798 / CAP（2026-09-30）

2026-10-03，Codex 只读核实平台终态：`execStatus=completed`、`scoringState.scoreIsFinal=true`、能力门通过、`stress.passed=true`。这是当前发布入口；同日更新的 TPOT / 47800 未通过能力门，保留为失败探索。

| 项目 | 47798 / CAP | 47800 / TPOT |
| --- | --- | --- |
| 官方原始终态 | [47798](../evidence/official/attempt-47798-final-20261003.json) | [47800](../evidence/official/attempt-47800-final-20261003.json) |
| 正式 N@SLO | **42，PASS** | 无；能力门未通过，未做压测 |
| chain_start p95 | **44.646 s** | — |
| TPOT mean / p95 | **40.912 / 67.161 ms/token** | — |
| fast / overall / turn p95 | 1.202 / 1.605 / 3.822 s | — |
| AIME / GPQA | 44/44；151/156 | 36/44；152/156 |
| 能力门 | PASS | FAIL：AIME 81.82 ≤ 90 |
| 引擎 / 镜像 | `ca5d646c` / `lh-img:0930a` | 同源码、同镜像 |
| 冻结服务配置 | [official-0930-CAP.json](../submission/official-0930-CAP.json) | [official-0930-TPOT.json](../submission/official-0930-TPOT.json) |
| 实际上传物与构建来源 | [归档](../evidence/submission-0930-execution/README.md) | 同目录保留 TPOT 原始包和回执 |

chain 的 30 s 是 TTFT 点目标，官方判门使用超标率的 95% 单侧下界，因此 `chain p95 > 30 s` 与平台 N42 PASS 可以同时成立。此次返回没有更高失败档的明细或逐请求 raw，无法核算其具体超标条数，也无法断言更高档只卡 chain；规则见 [task.md](../llm-challenge-arena-v1/task.md)。TPOT p95 的 100 ms 硬门没有这项余量。

两份配置的完整 command、image、model_name 和其他 env 相同，仅以下两项不同：

| 环境变量 | CAP / 47798 | TPOT / 47800 |
| --- | --- | --- |
| `SGLANG_AX_SCHED_COLD_CAP` | 12288 | 16384 |
| `SGLANG_AX_BACKLOG_INTERVAL` | 0 | 1 |

两者均为 BF16 KV、MTP off、普通 PDI2、chain-risk1、running/graph48、Mamba400、host64、短阈值2048；请求启用175/176/178/179/181/182/184，174/177/metadata fusion关闭。实际形状是否走快路径仍受源码守卫约束。[源码地图](architecture.md)解释机制，[逐提交索引](read-history.md)解释改动过程。

47798 和 47800 的能力分不同是平台实测结果。现有终态没有逐题输出或控制随机性的成对比较，不能把失败直接归因于这两个配置变量或某个内核。

## 正式历史

各行是该配置的官方结果，不是同负载条件下逐步加补丁的因果实验。不同 N 的 TPOT 不能直接比较；同 N 的不同提交也只能描述已观察到的差异。原始源码标签和 commit 保留，完整本地对照见 [experiments.md](experiments.md)。

| Attempt / 日期 | 源码、镜像与主要变化 | 正式结果 | 来源 |
| --- | --- | --- | --- |
| 45979 / 09-23 | `official-A-0923a` / 0923a，MTP+114、cap4096/interval2 | N14 PASS；TPOT mean 17.435 ms | [查询摘要](../evidence/official/attempt-45979-20260924.json) |
| 45980 / 09-23 | 同镜像；chunk8192、无 interval | N10 PASS；TPOT mean 12.099 ms | [查询摘要](../evidence/official/attempt-45980-20260924.json) |
| 46173 / 09-24 | `c92acd5` / 0924c，mem0.87、新180、host32、122 off | N14 PASS；TPOT mean 16.541 ms | [官方终态](../evidence/official/attempt-46173-20260924.json) |
| 46174 / 09-24 | `759a6ebb` / 0924d，46173 + 修复122 | N18 PASS；TPOT mean 20.518 ms | [官方终态](../evidence/official/attempt-46174-20260924.json) |
| 46251 / 09-24 | `759a6ebb` / 0925a，host64、122 off、cold4096 | N22 PASS；TPOT mean 23.011 ms | [官方终态](../evidence/official/attempt-46251-final-20260925.json) |
| 46364 / 09-25 | 同0925a；仅cold4096→6144 | N22 PASS；TPOT mean约22.6 ms（查询文本舍入值） | [查询摘要](../evidence/official/attempt-46364-final-20260926.txt) |
| 46676 / 09-26 | `f546934e` / 0926b，S1：117+124+激进125 | N26 PASS；TPOT mean 25.240 ms | [官方终态](../evidence/official/attempt-46676-20260927.json) |
| 46677 / 09-26 | 同0926b，S2：BACKLOG_MAX_SLOW80→250 | N26 PASS；TPOT mean 25.099 ms | [官方终态](../evidence/official/attempt-46677-20260927.json) |
| 46757 / 09-27 | `e464d8ab` / 0926c，S3：S1 + DCP2 | N22 PASS；TPOT mean 22.209 ms | [官方终态](../evidence/official/attempt-46757-20260927.json) |
| 46758 / 09-27 | `6976639e` / 0926d，S4：DCP2 + local extend | N26 PASS；TPOT mean 25.658 ms | [官方终态](../evidence/official/attempt-46758-20260927.json) |
| 47043 / 09-28 | `20a58da9` / 0927a，chain-max、MTP off、131/132 | N30 PASS；TPOT mean 37.683 ms | [官方终态](../evidence/official/attempt-47043-final-20260928.json) |
| 47266 / 09-28 | `bf6b66fa` / 0928c，池400、118、分片、MoE调参、KDA prefill | N42 PASS；TPOT mean 43.266 ms | [官方终态](../evidence/official/attempt-47266-final-20260929.json) |
| 47606 / 09-29 | 同0928c，FAST：short4096/cold12288 | N42 PASS；TPOT mean 42.599 ms | [官方终态](../evidence/official/attempt-47606-final-20261003.json) |
| 47607 / 09-29 | 同0928c，PDI4：普通interval2→4 | 能力门 FAIL，无压测结果；AIME 36/44 | [官方终态](../evidence/official/attempt-47607-final-20260929.json) |
| 47798 / 09-30 | `ca5d646c` / 0930a，执行组合 + CAP 配置 | N42 PASS；TPOT mean 40.912 ms | [官方终态](../evidence/official/attempt-47798-final-20261003.json) |
| 47800 / 09-30 | 同0930a，执行组合 + TPOT 配置 | 能力门 FAIL，无压测结果；AIME 36/44 | [官方终态](../evidence/official/attempt-47800-final-20261003.json) |


09-22 的45734/45735因镜像地址格式拒绝，45766/45767部署失败；未形成评分成绩。早期平台快照保留在 [all_att_2026-09-23.json](../evidence/official/all_att_2026-09-23.json)。旧阶段的详细提交讨论与预期留在 Git 历史，本页只保留现行结果和复现身份。

## 对应源码、配置与上传物

- 引擎版本由 commit / 镜像标签固定；最新两包的源码均为 `image-lh-img-0930a` / `ca5d646c252688177480c7e67cec0a901e4b7069`。
- 冻结 JSON 记录 image、完整 command、env、model_name；原始 ZIP、构建 Dockerfile、SHA256 与上传回执见 [09-30 归档](../evidence/submission-0930-execution/README.md)。
- 正式成绩引用平台终态。本地 window / DRAINED 结果、不同派发时长的共同 ID 描述不能当完整档或正式成绩。
- 提交与查询流程见 [复现说明](reproduce.md)；新提交记录 attempt、日期、源码/镜像、配置入口、上传哈希、能力分、正式 N@SLO、TPOT 和原始返回。未返回字段留空。

用户要求的后续正式提交写盘约束保持：关闭请求正文、张量与性能 trace 调试落盘，降低高频 INFO / HTTP 访问日志，保留启动收据、错误与警告；按实际挂载核验编译缓存和临时目录预算，不修改评测时间戳、token 计数或 flush 行为。本次仓库整理没有创建新的正式 attempt 或改动服务。
