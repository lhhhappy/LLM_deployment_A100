09-30 Codex收尾：09/10均已DRAINED，末次Pod状态running/pending为空，服务保持。用户选择10 CAP+09 TPOT后，两包已于09:04–09:05 UTC上传成功，attempt分别47798/47800；配置、SHA和回执归[提交记录](submissions.md#09-30-最后两次正式提交)。用户随后要求结束，拟追加的N46 cold12k+relief1一小时试验尚未生成或入队，已取消后续安排。实验结果见[实验记录](experiments.md#09-30-codex最后两臂闭合与两包取舍)，清洁引擎分支保持`codex/final-execution-0930` / `ca5d646c`。

09-29 Codex 当前轮 `130ezny1–5` 已全部 DRAINED（各 N42 派发 1h 后排空），Pod running/pending 均空。五组零错误；汇总尚未发现 chain 与 TPOT 均值稳定优于对照的候选，同 ID raw 配对与实际形状核验待做。没有新增正式提交。[方案](reports/chain-night-0929.md)、[判定收据](../evidence/chain-night-0929/verdicts/)。下方历史记录不代表当前实时状态。

这里仅记**正在执行、已排队和下一步待决**的任务。Pod 实时状态以 `scripts/pod/pread status` 为准；已闭合、停止或作废的运行及其证据见[实验记录](experiments.md)。入队者负责跟到结果闭合，协作边界见[collaboration.md](collaboration.md)。

| Job | 要回答的问题与对照 | 当前状态 |
| --- | --- | --- |
| `130eznr-tail_rot150_n34_kda_night_pdi3_1h_ladder`（Codex，今晚） | bf6b66fa；--prefill-decode-interval 2 -> --prefill-decode-interval 3；N34 派发 1h 后排空，四 TTFT 门与 TPOT p95≤100ms 且参考同 ID chain 无新坏，才自动 N38 1h | 已恢复；eznr 冒烟12/12，N34 1h测量已开始，其余等待；四臂依次串行占用 TP8，CPU 分析并行；详见[今晚方案](reports/night-n34-tuning-0928.md) |
| `130ezns-tail_rot150_n34_kda_night_pdi4_1h_ladder`（Codex，今晚） | bf6b66fa；--prefill-decode-interval 2 -> --prefill-decode-interval 4；N34 派发 1h 后排空，四 TTFT 门与 TPOT p95≤100ms 且参考同 ID chain 无新坏，才自动 N38 1h | 已恢复；eznr 冒烟12/12，N34 1h测量已开始，其余等待；四臂依次串行占用 TP8，CPU 分析并行；详见[今晚方案](reports/night-n34-tuning-0928.md) |
| `130eznt-tail_rot150_n34_kda_night_relief1_1h_ladder`（Codex，今晚） | bf6b66fa；SGLANG_AX_BACKLOG_INTERVAL=0 -> SGLANG_AX_BACKLOG_INTERVAL=1；N34 派发 1h 后排空，四 TTFT 门与 TPOT p95≤100ms 且参考同 ID chain 无新坏，才自动 N38 1h | 已恢复；eznr 冒烟12/12，N34 1h测量已开始，其余等待；四臂依次串行占用 TP8，CPU 分析并行；详见[今晚方案](reports/night-n34-tuning-0928.md) |
| `130eznu-tail_rot150_n34_kda_night_repeat_1h_ladder`（Codex，今晚） | bf6b66fa；baseline repeat；N34 派发 1h 后排空，四 TTFT 门与 TPOT p95≤100ms 且参考同 ID chain 无新坏，才自动 N38 1h | 已恢复；eznr 冒烟12/12，N34 1h测量已开始，其余等待；四臂依次串行占用 TP8，CPU 分析并行；详见[今晚方案](reports/night-n34-tuning-0928.md) |
| 正式 47266 / job 25359（Codex，09-28 15:09 UTC） | 用户已选择并授权 0928c：47043 + 钉池 400 + 118 + attention-TP 分片 + MoE 调参 + KDA；不再单独提交 PIN | 已上传，末次独立查询 queued；[冻结包与收据](../evidence/submission-0928-kda/README.md)。eznq N34 已闭合，今日本地各臂结果与分母更正见[实验记录](experiments.md#09-28-codex-原始数据复盘与-eznq-闭合)和[复盘](reports/0928-official-local-synthesis.md)；本轮未新增 GPU 任务 |
| `130ezl-v4_n26_S4_full`（第 43 分钟停，结论见 experiments/ledger）（fable） | 用户批准的 v4 校准：46758 配置整集 N26（对线上 41.1 s 的 chain p95，约 3 小时，完整判分）；随后 S5b 在 v4 上 N34 30 分钟对 130ezh | 已发布（queue-after 等 130ezk） |
| `130ezi-v3_n38_S1dcp_combo_nomtp_30m`（fable） | S5b（合并引擎 791453ca：128p + 本地续算，去 MTP，池 3.13M）在 N38 的 30 分钟窗口：稳态池峰值/排队/驱逐、四门与 TPOT>0.10，对 130ezh（同引擎 N34） | 跑（11:10 UTC 起） |
| `130ezj-v3_open_S1dcp_combo_nomtp_chunk16k_n34`（fable） | 单旋钮开场探针：S5b 配置把开场 prefill 块 8k→16k（--chunked-prefill-size 16384 + SGLANG_AX_BACKLOG_COLD_CAP=16384，稳态的 SCHED_COLD_CAP 6144 不变），对 130ezh 前 600 s 同 ID；看开场 chain 11 能否再少 | 发布者 queue-after-…095324 等 130ezi 结束 |
| 已闭合（结果在 experiments.md“130ez 系列”与 kanban.md）：130ez1、ez4–ez6、ez6z–ez6zzzz、ez7–ez9、ezd（停）、ezd5、eze2、ezf、ezg、ezh；已撤：130ef/ef5/ez2/ez3/130f/g/h/136–144、130eza/ezb（改名 ezd/eze2）、130eze（G_EXPECT 写错中止） | — | 脚本保留 |
| `dcp-mtp-20260926`（Codex） | 当前候选分支 `codex/dcp-mtp-stack2-n34`，原开发分支 `codex/dcp-mtp-n34`；保留MTP/HiCache推进N34/N38；技术方案在候选worktree的 `notes/plan-dcp-8card.md` | 两卡开发矩阵已闭合，结果见候选分支 `notes/experiments.md` 的DCP-20260926。W1/W2 N34配置已准备；TP8真实权重、另一参与者raw复核及N34/N38待验，尚未入队Pod |

已撤：114/115（改名 137/138）、116–119（改名 139–142）、120/121（N34 探针，已由 131/132 在 N30 覆盖）、122/123（改名 134/135）。撤下的名字在 Pod runs/ 下只有部署收据，无数据。

- Codex 追加 eznv MTP（NEXTN 3/1/4，策略随之关闭101，DCP=1）：延迟发布器等待eznu完成且队列空闲；N34 1h过门再N38 1h。详见 [夜间计划](reports/night-n34-tuning-0928.md)。

- 用户优先级调整：130eznr1（1步轻量MTP，1/1/2）插在正在测的eznr之后、ezns之前；保留eznr完整测量及其条件N38。发布器 queue-after-0928-mtplite-priority 暂停领取后续任务，等待eznr闭合后保留pending发布并恢复；不停止当前引擎。

- Codex 用户授权fast五小时：130eznw1–5，短4096／间隔1风险0／冷块12k／短4096+冷12k／47266对照；均N34 1h、无自动N38；五项QUEUED、RUNTIME_DEPLOYED、resumed、DONE rc=0均已确认。[冻结方案](../evidence/fast-queue-0929/README.md)。

- 去重调整：撤销待执行130eznw5（与eznu完全相同的N34一小时原配置），复用eznu；保留w1–4共4h测量约4.5–5h墙钟。w1已运行。[去重审计](../evidence/fast-queue-0929/dedup-audit.md)。

- 09-29 08:35 UTC：fast w1–4均DRAINED闭合，w5已取消，队列空。四组性能统计门通过，但w1/w2/w4新增chain坏例；w3无新增但缺2个参考ID，不满足完整配对。[结果](../evidence/fast-queue-0929/results.md)。47266接口仍queued，详情redacted。

- 09-29第二轮用户授权：130eznx1–5（cold12 N38、short4096+cold12 N38、12612–16k N34、池320 N34、126 N38条件臂）；最多5h测量。第5只在第3通过后运行。五项QUEUED、RUNTIME_DEPLOYED、resumed、DONE rc=0已核验；第1组引擎已就绪、机制检查通过，尚未确认开始测量。[冻结计划](../evidence/fast-round2-0929/README.md)。
