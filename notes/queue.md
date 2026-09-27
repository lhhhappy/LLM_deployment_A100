# 实验队列

这里仅记**正在执行、已排队和下一步待决**的任务。Pod 实时状态以 `scripts/pod/pread status` 为准；已闭合、停止或作废的运行及其证据见[实验记录](experiments.md)。入队者负责跟到结果闭合，协作边界见[collaboration.md](collaboration.md)。

| Job | 要回答的问题与对照 | 当前状态 |
| --- | --- | --- |
| 队列顺序（fable，18:50 UTC，按 worker 实际 en_US.UTF-8 排序核实）：`130ezmb` 已闭合 → `130ezn1` S1 锚点（在跑）→ `130ezn2a` S6+128g（Codex 873031fd，原生 LPM 零收益 hold 释放；worker 的 sort 忽略标点，所以它先于 ezn2）→ `130ezn2` S6 → `130ezn3` chain-max 16k → `130ezn4` chain-max 32k → `130ezn5` chain-max 16k + 118（合并引擎 cfd25d0f）。以后需要严格先后的名字不要用 `2a` 这种带标点歧义的后缀。全部 v5g-tail、N26、40 分钟派发+排空，每臂约 65 分钟。18:30 重发：Codex 发现 S6 系任务 G_EXPECT 里 dcp_local_max=512 是从 S5b 继承的错误期望（本地续算未开时引擎打印 0），已改 0 并重新打包；旧发布者已杀。128g 的同引擎 OFF 臂脚本已备（tail_n26_S6_lpm_guard_off_40m.sh），仅当 ON 对 ezn2 的结果不清时再入队。 | 128g 审查（fable）：默认关时代码路径与 cfd25d0f 相同（属性只在开启时赋值、报告多一个 128g=off 记号）；开启时只改“原生要 hold 且新增可复用上界为 0”的请求，释放者仍插模拟树；对 128p/132/停车/成本模型无改动；开启要求 LPM+批内检查开且非 bigram 树（即不能与 MTP 的 EAGLE 树同开，S6 无 MTP 不受影响）；四组 CPU 测试通过（128g 14、128p、调度、deadline）。 | ezmb 排空中；六臂待本地发布循环插入 |
| `130ezl-v4_n26_S4_full`（第 43 分钟停，结论见 experiments/ledger）（fable） | 用户批准的 v4 校准：46758 配置整集 N26（对线上 41.1 s 的 chain p95，约 3 小时，完整判分）；随后 S5b 在 v4 上 N34 30 分钟对 130ezh | 已发布（queue-after 等 130ezk） |
| `130ezi-v3_n38_S1dcp_combo_nomtp_30m`（fable） | S5b（合并引擎 791453ca：128p + 本地续算，去 MTP，池 3.13M）在 N38 的 30 分钟窗口：稳态池峰值/排队/驱逐、四门与 TPOT>0.10，对 130ezh（同引擎 N34） | 跑（11:10 UTC 起） |
| `130ezj-v3_open_S1dcp_combo_nomtp_chunk16k_n34`（fable） | 单旋钮开场探针：S5b 配置把开场 prefill 块 8k→16k（--chunked-prefill-size 16384 + SGLANG_AX_BACKLOG_COLD_CAP=16384，稳态的 SCHED_COLD_CAP 6144 不变），对 130ezh 前 600 s 同 ID；看开场 chain 11 能否再少 | 发布者 queue-after-…095324 等 130ezi 结束 |
| 已闭合（结果在 experiments.md“130ez 系列”与 kanban.md）：130ez1、ez4–ez6、ez6z–ez6zzzz、ez7–ez9、ezd（停）、ezd5、eze2、ezf、ezg、ezh；已撤：130ef/ef5/ez2/ez3/130f/g/h/136–144、130eza/ezb（改名 ezd/eze2）、130eze（G_EXPECT 写错中止） | — | 脚本保留 |
| `dcp-mtp-20260926`（Codex） | 当前候选分支 `codex/dcp-mtp-stack2-n34`，原开发分支 `codex/dcp-mtp-n34`；保留MTP/HiCache推进N34/N38；技术方案在候选worktree的 `notes/plan-dcp-8card.md` | 两卡开发矩阵已闭合，结果见候选分支 `notes/experiments.md` 的DCP-20260926。W1/W2 N34配置已准备；TP8真实权重、另一参与者raw复核及N34/N38待验，尚未入队Pod |

已撤：114/115（改名 137/138）、116–119（改名 139–142）、120/121（N34 探针，已由 131/132 在 N30 覆盖）、122/123（改名 134/135）。撤下的名字在 Pod runs/ 下只有部署收据，无数据。
