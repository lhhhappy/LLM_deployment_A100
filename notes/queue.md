# 实验队列

这里仅记**正在执行、已排队和下一步待决**的任务。Pod 实时状态以 `scripts/pod/pread status` 为准；已闭合、停止或作废的运行及其证据见[实验记录](experiments.md)。入队者负责跟到结果闭合，协作边界见[collaboration.md](collaboration.md)。

| Job | 要回答的问题与对照 | 当前状态 |
| --- | --- | --- |
| `130eb-v3_open_A_warm15_fixes_n30`（fable） | A′ 环境不变，引擎 f546934e（46676/46677 所用）= 84dcca0e + 审查修正 124 停车/110 守卫/117 缓存/180 销毁；对 130a 同 ID；SMOKE_GATE=1 | running（冒烟 12/12、机制行正确），闭合后配对 130a 进看板 |
| `130ec-v3_n30_A_warm15_fixes_slow250_60m`（fable） | 46677（S2）的事后确认 | 开跑约 3 分钟后 stopjob（用户决定 S2 先不测）；脚本保留，需要时再排 |
| `130ed-dcp_stack2_w2_n30_60m` → `130ee-dcp_stack2_w1_n30_60m`（fable，脚本改自 Codex 的 dcp_mtp_stack2_w2/w1_n34_60m） | DCP 在 N30 有没有帮助：引擎 b261cbd9、A 调度、MTP 3/1/4、host64、418 槽、running/graph 32、60 分钟、SMOKE_GATE=1；两臂只差 `--dcp-size` | 130ed done 17:02 UTC：对 112 同 3076 条 chain 27→15、turn 24→2、overall 449→74、fast 506→72，TPOT 均值 +6 ms；池 2.38M 无排队。130ee pending（W1 对照） |
| `130ee5-dcp_stack2_w2_n34_60m` → `130eez-dcp_stack2_w1_n34_60m`（fable） | DCP 的 N34 对，同引擎 b261cbd9、A 调度、running/graph 48（Codex 的 w2/w1_n34 改到含 guard 修正的提交）：N34 的墙是否被推开。W2-N34 紧接 130ed（并发优先），W1-N34 排在 W1-N30 之后 | 已发布（queue-after-…，130ed 之后按名字顺序） |
| `130ef-v3_n30_A_warm15_fixes_60m` → `130eg-…_slow250_60m` → `130eh-…_pdi1_60m` → `130ei-…_high8_60m`（fable） | 超参组，全部 60 分钟 N30、引擎 f546934e（46676 引擎）、A′ 环境，各只改一处：130ef = 基线（S1 的本地参照，对 130b 看修正引擎的整窗效果）；130eg = MAX_SLOW 80→250（= 46677）；130eh = `--prefill-decode-interval` 2→1（稳态里 prefill 更频繁，换 chain/turn，付 TPOT）；130ei = BACKLOG_HIGH_S 15→8（更早进积压模式）。判定：对 130ef 同 ID 四门 + TPOT>0.10 + tpot p95；目标是 chain/turn 降、fast/TPOT 小涨可接受 | 已发布（queue-after-…160238，130ed 之后、130ee 之后按名字顺序） |
| `130ej-v3_open_A_warm15_fixes_multi5_n30` → `130ek-v3_n30_A_warm15_fixes_multi5_60m`（fable） | K3 = 新机制“按轮数分级的 warm 预算”（引擎 9be15822 = f546934e + 124 后续，默认中性；env WARM_S=15、WARM_MULTI_S=5）：单轮 warm 请求仍享 15 s 救援，多轮（>8192）warm 请求回到 5 s，不再压过冷链首。探针对 130eb，60 分钟对 130ef | 发布任务挂在 130ee 之后（排在 130f 前） |
| `130f-v3_n30_A_warm15_nomtp_60m`（fable） | A′ 去掉 MTP（钉住状态池），60 分钟对 130b：MTP 是否拖累 chain/turn | pending，130ec 之后 |
| `130g-v3g_n30_base_60m` → `130h-v3g_n30_A_warm15_60m`（fable；原 134/135） | 与 111/130b 除数据外逐字一致，数据换成 v3g（补插间隔按主办方真实链时长放大，均值 5.5→22.3 s，截断不摊；Pod `/tmp/ax/data/s1-dev-longchain-v3g`，requests SHA256 `1b14aab2…`）。问题：真实间隔下 N30 稳态是否退出显存墙，A′ 的相对收益是否不变；claude 建议用底座稳态每路 tpm_all 对线上 78.6k/分钟检验密度 | 原 134/135 被 worker 在插入前抢先启动，均已 stopjob（未测量）；由 `queue-after`（130 之后、130f 之后按名字顺序）重排 |
| `136-v3_n34_base48_60m` → `137-v3_n34_124_125x_117_48_60m` → `138-v3_n34_124_122_117_48_60m`（fable） | N34 三组（原 113–115，脚本不变）：底座 + 上限 48；A + 上限 48；A 把 125x 换 122 | pending，134/135 之后 |
| `139`–`144`（fable） | N34 容量单变量（各对 136；按 R34 修正）：139 KDA 状态池 418→256（不用 200：200÷5 会把并发压到 40）；140 mem 0.89 + 钉住 418 槽；141 hicache 96；142 去掉 MTP + 钉住 418 槽；143 主机层 write_back（独占式）；144 KDA 状态 bf16（能力冒烟开） | pending，最后 |

已撤：114/115（改名 137/138）、116–119（改名 139–142）、120/121（N34 探针，已由 131/132 在 N30 覆盖）、122/123（改名 134/135）。撤下的名字在 Pod runs/ 下只有部署收据，无数据。
