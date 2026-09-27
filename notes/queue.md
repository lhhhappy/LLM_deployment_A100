# 实验队列

这里仅记**正在执行、已排队和下一步待决**的任务。Pod 实时状态以 `scripts/pod/pread status` 为准；已闭合、停止或作废的运行及其证据见[实验记录](experiments.md)。入队者负责跟到结果闭合，协作边界见[collaboration.md](collaboration.md)。

| Job | 要回答的问题与对照 | 当前状态 |
| --- | --- | --- |
| 队列顺序（fable，17:50 UTC）：`130ezmb`（旧 v5g S1 锚点，在跑，不动）→ `130ezn1/ezn2/ezn3/ezn4`（**v5g-tail**，Codex c9c14153，fable 独立复核通过：S1 锚点 / S6 / chain-max 16k / chain-max 32k，N26 40 分钟派发+排空）。已撤：旧 v5g 的 ezmc/ezmd/ezme/ezmf 与 v4 的 ezp/ezq/ezr/ezs（旧 v5g gap 中位 7.3 s、>60 s 440 个超过源 raw 340 个，不作主数据；需要时再入队）。ezm7/ezm8/ezm9/ezma（118）已闭合，见 experiments.md。 | 复核内容（Pod 上 /tmp/ax/codex/review-0927/review.json）：三套 5601 行、ID 顺序与 v4 一致；v5-review 与冻结 v5、v5g-review 与冻结 v5g 逐字段 0 差异；tail 对 v5 只有 replay_gap_ms/gap_regap_factor 239 行，规则集合精确相等（合成、非 cohort 链首、原 gap>12551 ms、新值>max(60 s,原值)、≤310 s、等于旧 v5g 提案）；其中 2 条是 context_reset 前的等待；cohort 6 套全同（ff1dccae1087a798）；manifest 7 个产物 sha 全对；正文 5601/5601 命中 v4 分片；harness build_gap_plan 3600 s 封顶：v5/tail 0 链被压缩，v5g 1 链；Codex 结构 checker STRUCTURAL_OK 0 错误（无 tokenizer 全量渲染，非 complete PASS）。GPU 机容器仍不稳（17:15/17:18/17:21 重启），发布由本地循环驱动。 | ezmb 在跑；ezn1–ezn4 待本地发布循环在 ezmb 结束后插入 |
| `130ezl-v4_n26_S4_full`（第 43 分钟停，结论见 experiments/ledger）（fable） | 用户批准的 v4 校准：46758 配置整集 N26（对线上 41.1 s 的 chain p95，约 3 小时，完整判分）；随后 S5b 在 v4 上 N34 30 分钟对 130ezh | 已发布（queue-after 等 130ezk） |
| `130ezi-v3_n38_S1dcp_combo_nomtp_30m`（fable） | S5b（合并引擎 791453ca：128p + 本地续算，去 MTP，池 3.13M）在 N38 的 30 分钟窗口：稳态池峰值/排队/驱逐、四门与 TPOT>0.10，对 130ezh（同引擎 N34） | 跑（11:10 UTC 起） |
| `130ezj-v3_open_S1dcp_combo_nomtp_chunk16k_n34`（fable） | 单旋钮开场探针：S5b 配置把开场 prefill 块 8k→16k（--chunked-prefill-size 16384 + SGLANG_AX_BACKLOG_COLD_CAP=16384，稳态的 SCHED_COLD_CAP 6144 不变），对 130ezh 前 600 s 同 ID；看开场 chain 11 能否再少 | 发布者 queue-after-…095324 等 130ezi 结束 |
| 已闭合（结果在 experiments.md“130ez 系列”与 kanban.md）：130ez1、ez4–ez6、ez6z–ez6zzzz、ez7–ez9、ezd（停）、ezd5、eze2、ezf、ezg、ezh；已撤：130ef/ef5/ez2/ez3/130f/g/h/136–144、130eza/ezb（改名 ezd/eze2）、130eze（G_EXPECT 写错中止） | — | 脚本保留 |
| `dcp-mtp-20260926`（Codex） | 当前候选分支 `codex/dcp-mtp-stack2-n34`，原开发分支 `codex/dcp-mtp-n34`；保留MTP/HiCache推进N34/N38；技术方案在候选worktree的 `notes/plan-dcp-8card.md` | 两卡开发矩阵已闭合，结果见候选分支 `notes/experiments.md` 的DCP-20260926。W1/W2 N34配置已准备；TP8真实权重、另一参与者raw复核及N34/N38待验，尚未入队Pod |

已撤：114/115（改名 137/138）、116–119（改名 139–142）、120/121（N34 探针，已由 131/132 在 N30 覆盖）、122/123（改名 134/135）。撤下的名字在 Pod runs/ 下只有部署收据，无数据。
