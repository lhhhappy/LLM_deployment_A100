# 实验队列

这里仅记**正在执行、已排队和下一步待决**的任务。Pod 实时状态以 `scripts/pod/pread status` 为准；已闭合、停止或作废的运行及其证据见[实验记录](experiments.md)。入队者负责跟到结果闭合，协作边界见[collaboration.md](collaboration.md)。

| Job | 要回答的问题与对照 | 当前状态 |
| --- | --- | --- |
| 队列顺序（fable，07:58 UTC）：`130ezng`（118 能力复核）07:29 按用户决定停掉 → `130eznh`（钉池候选在 N34 的天花板）→ `130ezni`（钉池候选 + 133 调参，对 eznc）→ `130eznj`（124m ON + 拒绝原因计数 + 钉池，对 eznc）→ `130eznk`（钉池候选 + attention-TP 输入分片，对 eznc）→ `130eznl`（钉池候选 + 118 prefill 稀疏注意力，引擎 52c694be，对 eznc；不做能力复核）。全部脱离发布者 + 看门狗（T065330/T065919/T072550/+eznl）。已闭合：eznc（钉池：对 eznb chain 12→7，稳态 0）、eznd（133 默认：chain 12→18 否决）、ezne/eznf（124m ON/OFF：零触发，chain 差在噪声内，Codex 9bc5119d 闭合）。线上 47043 排队中；Codex PIN 包（affb5edc，= 47043 + `--max-mamba-cache-size 400`）已复核，等用户定是否上传。 | 剩余 chain 逐条归因见台账 07:20：候选的 7 条全是开场第 22–29 位的吞吐尾巴。 | eznh 起跑中；ezni/eznj/eznk 待插入 |
| `130ezl-v4_n26_S4_full`（第 43 分钟停，结论见 experiments/ledger）（fable） | 用户批准的 v4 校准：46758 配置整集 N26（对线上 41.1 s 的 chain p95，约 3 小时，完整判分）；随后 S5b 在 v4 上 N34 30 分钟对 130ezh | 已发布（queue-after 等 130ezk） |
| `130ezi-v3_n38_S1dcp_combo_nomtp_30m`（fable） | S5b（合并引擎 791453ca：128p + 本地续算，去 MTP，池 3.13M）在 N38 的 30 分钟窗口：稳态池峰值/排队/驱逐、四门与 TPOT>0.10，对 130ezh（同引擎 N34） | 跑（11:10 UTC 起） |
| `130ezj-v3_open_S1dcp_combo_nomtp_chunk16k_n34`（fable） | 单旋钮开场探针：S5b 配置把开场 prefill 块 8k→16k（--chunked-prefill-size 16384 + SGLANG_AX_BACKLOG_COLD_CAP=16384，稳态的 SCHED_COLD_CAP 6144 不变），对 130ezh 前 600 s 同 ID；看开场 chain 11 能否再少 | 发布者 queue-after-…095324 等 130ezi 结束 |
| 已闭合（结果在 experiments.md“130ez 系列”与 kanban.md）：130ez1、ez4–ez6、ez6z–ez6zzzz、ez7–ez9、ezd（停）、ezd5、eze2、ezf、ezg、ezh；已撤：130ef/ef5/ez2/ez3/130f/g/h/136–144、130eza/ezb（改名 ezd/eze2）、130eze（G_EXPECT 写错中止） | — | 脚本保留 |
| `dcp-mtp-20260926`（Codex） | 当前候选分支 `codex/dcp-mtp-stack2-n34`，原开发分支 `codex/dcp-mtp-n34`；保留MTP/HiCache推进N34/N38；技术方案在候选worktree的 `notes/plan-dcp-8card.md` | 两卡开发矩阵已闭合，结果见候选分支 `notes/experiments.md` 的DCP-20260926。W1/W2 N34配置已准备；TP8真实权重、另一参与者raw复核及N34/N38待验，尚未入队Pod |

已撤：114/115（改名 137/138）、116–119（改名 139–142）、120/121（N34 探针，已由 131/132 在 N30 覆盖）、122/123（改名 134/135）。撤下的名字在 Pod runs/ 下只有部署收据，无数据。
