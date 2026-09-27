# 实验队列

这里仅记**正在执行、已排队和下一步待决**的任务。Pod 实时状态以 `scripts/pod/pread status` 为准；已闭合、停止或作废的运行及其证据见[实验记录](experiments.md)。入队者负责跟到结果闭合，协作边界见[collaboration.md](collaboration.md)。

| Job | 要回答的问题与对照 | 当前状态 |
| --- | --- | --- |
| 队列顺序（fable，22:55 UTC）：`130ezn5`（chain-max 16k + 118，在跑，约 23:40 闭合）→ `130ezn6`（chain-max 16k 保 fast 变体：预留 2048→8192，单变量对 ezn3）→ `130ezn7`（chain-max 16k 在 N34，running 48）。已闭合（见 experiments.md 130ez 表）：ezmb、ezn1（S1 锚点）、ezn2（S6）、ezn2a（S6+128g：无净收益）、ezn3（chain-max 16k：chain 10→6 对 S1，p95 24.8 s，fast 代价）、ezn4（32k：不比 16k 好）。 | 目的：ezn6 决定上传配置里预留取多少（线上 fast 只有 1.9 s 对 3 s 的余量），ezn7 决定这套配置能不能冲 N30/N34。 | ezn5 在跑；ezn6/ezn7 待发布循环插入 |
| `130ezl-v4_n26_S4_full`（第 43 分钟停，结论见 experiments/ledger）（fable） | 用户批准的 v4 校准：46758 配置整集 N26（对线上 41.1 s 的 chain p95，约 3 小时，完整判分）；随后 S5b 在 v4 上 N34 30 分钟对 130ezh | 已发布（queue-after 等 130ezk） |
| `130ezi-v3_n38_S1dcp_combo_nomtp_30m`（fable） | S5b（合并引擎 791453ca：128p + 本地续算，去 MTP，池 3.13M）在 N38 的 30 分钟窗口：稳态池峰值/排队/驱逐、四门与 TPOT>0.10，对 130ezh（同引擎 N34） | 跑（11:10 UTC 起） |
| `130ezj-v3_open_S1dcp_combo_nomtp_chunk16k_n34`（fable） | 单旋钮开场探针：S5b 配置把开场 prefill 块 8k→16k（--chunked-prefill-size 16384 + SGLANG_AX_BACKLOG_COLD_CAP=16384，稳态的 SCHED_COLD_CAP 6144 不变），对 130ezh 前 600 s 同 ID；看开场 chain 11 能否再少 | 发布者 queue-after-…095324 等 130ezi 结束 |
| 已闭合（结果在 experiments.md“130ez 系列”与 kanban.md）：130ez1、ez4–ez6、ez6z–ez6zzzz、ez7–ez9、ezd（停）、ezd5、eze2、ezf、ezg、ezh；已撤：130ef/ef5/ez2/ez3/130f/g/h/136–144、130eza/ezb（改名 ezd/eze2）、130eze（G_EXPECT 写错中止） | — | 脚本保留 |
| `dcp-mtp-20260926`（Codex） | 当前候选分支 `codex/dcp-mtp-stack2-n34`，原开发分支 `codex/dcp-mtp-n34`；保留MTP/HiCache推进N34/N38；技术方案在候选worktree的 `notes/plan-dcp-8card.md` | 两卡开发矩阵已闭合，结果见候选分支 `notes/experiments.md` 的DCP-20260926。W1/W2 N34配置已准备；TP8真实权重、另一参与者raw复核及N34/N38待验，尚未入队Pod |

已撤：114/115（改名 137/138）、116–119（改名 139–142）、120/121（N34 探针，已由 131/132 在 N30 覆盖）、122/123（改名 134/135）。撤下的名字在 Pod runs/ 下只有部署收据，无数据。
