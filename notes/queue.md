# 实验队列

这里仅记**正在执行、已排队和下一步待决**的任务。Pod 实时状态以 `scripts/pod/pread status` 为准；已闭合、停止或作废的运行及其证据见[实验记录](experiments.md)。入队者负责跟到结果闭合，协作边界见[collaboration.md](collaboration.md)。

| Job | 要回答的问题与对照 | 当前状态 |
| --- | --- | --- |
| `130eb-v3_open_A_warm15_fixes_n30`（fable） | A′ 环境不变，引擎 f546934e（46676/46677 所用）= 84dcca0e + 审查修正 124 停车/110 守卫/117 缓存/180 销毁；对 130a 同 ID；SMOKE_GATE=1 | running（冒烟 12/12、机制行正确），闭合后配对 130a 进看板 |
| `130ec-v3_n30_A_warm15_fixes_slow250_60m`（fable） | 46677（S2）的事后确认 | 开跑约 3 分钟后 stopjob（用户决定 S2 先不测）；脚本保留，需要时再排 |
| `130ed-dcp_stack2_w2_n30_60m` → `130ee-dcp_stack2_w1_n30_60m`（fable，脚本改自 Codex 的 dcp_mtp_stack2_w2/w1_n34_60m） | DCP 在 N30 有没有帮助：引擎 b261cbd9、A 调度、MTP 3/1/4、host64、418 槽、running/graph 32、60 分钟、SMOKE_GATE=1；两臂只差 `--dcp-size` | 130ed done 17:02 UTC：对 112 同 3076 条 chain 27→15、turn 24→2、overall 449→74、fast 506→72，TPOT 均值 +6 ms；池 2.38M 无排队。130ee pending（W1 对照） |
| `130ee5-dcp_stack2_w2_n34_60m` → `130eez-dcp_stack2_w1_n34_60m` → `130eezy-dcp_stack2_w2_n38_60m` → `130eezz-dcp_stack2_w4_n38_60m`（fable） | DCP 梯子：N34 对（同引擎 b261cbd9、A 调度、running 48），然后 N38 用 W2 看失败形态、再用 W4（池 ×2.56，冒烟门先过）。130ee5 第 32 分钟：KV 峰值 86–88%、host 回载 4.3 GiB/2 分钟、TPOT>0.10 4.5%——N34 的约束已从 TTFT 转到 KV 余量和 TPOT 尾 | 130ee5 running；其余已发布 |
| `130eezzy-cap_full_S1dcp_w2` → `130ef5-cap_full_S1`（fable） | 能力门本地复核：公开 AIME 2026（30）+ GPQA-Diamond（197，未门控副本），请求形状与 12 题冒烟一致，并发 12 | 130eezzy done：AIME 28/30（93.3；2 道 60000 token 截断）、GPQA 178/197（90.4；6 道截断、13 道错），errors 0。GPQA 贴线 → 130ef5 用 S1 引擎（无 DCP）同题参照，已发布在 130ef 之后 |
| `130ef-v3_n30_A_warm15_fixes_60m`（fable） | S1（46676）引擎 + A′ 的 60 分钟本地参照，对 130b | pending（130eez 之后）。旧引擎上的旋钮组 130eg/eh/ei 与 K3 130ej/ek 已撤：DCP 后约束变了（TPOT 尾），旋钮要在 DCP 引擎、N34 上重测；K1 由线上 46677 直接回答 |
| `130ez1-v3_n34_S1dcp_w2_60m` → `130ez2-v3_n34_S1dcp_w1_60m` → `130ez3-v3_n30_S1dcp_w2_60m`（fable） | 下一次提交的候选引擎：e464d8ab（Codex 分支上 f546934e + DCP 三补丁，逐字节同 b261cbd9 的 DCP；不含未审完的本地续算路径）。S1 环境 + `--dcp-size 2`、COMPACT_TOPK=1、KDA 池钉 418；N34 对（running 48）+ N30 W2（对 130ef）。审查报告 notes/reports/review-dcp-local-extend-0926.md（f34c7ac4 的本地路径有两个阻塞项，默认关闭路径已核实不变） | 发布任务挂在 130ee5 之后（等 N38 发布完成再挂），排在 130ef 之后、130f 之前 |
| `130ez4-v3_open_S1dcp_local_offa_n34` → `130ez5-…_on_n34` → `130ez6-…_offb_n34`（fable） | Codex 的 DCP 本地续算路径（6976639e，两轮审查后不再阻塞）TP8 探针：S1 设置 + dcp 2、N34 开场、冒烟门；OFF-a/ON/OFF-b 三臂，G_EXPECT 带 dcp_local 三个 token；ON 臂用 scripts/analysis/dcp_route_audit.py 核路由快照。判定：ON 对 OFF-a 的四门/短暖续算时间/TPOT/MTP 接受，对照 OFF-b 对 OFF-a 的噪声 | 发布任务挂在能力复核之后（按名字排在 130ez3 后、130f 前） |
| `130f-v3_n30_A_warm15_nomtp_60m`（fable） | A′ 去掉 MTP（钉住状态池），60 分钟对 130b：MTP 是否拖累 chain/turn | pending，130ec 之后 |
| `130g-v3g_n30_base_60m` → `130h-v3g_n30_A_warm15_60m`（fable；原 134/135） | 与 111/130b 除数据外逐字一致，数据换成 v3g（补插间隔按主办方真实链时长放大，均值 5.5→22.3 s，截断不摊；Pod `/tmp/ax/data/s1-dev-longchain-v3g`，requests SHA256 `1b14aab2…`）。问题：真实间隔下 N30 稳态是否退出显存墙，A′ 的相对收益是否不变；claude 建议用底座稳态每路 tpm_all 对线上 78.6k/分钟检验密度 | 原 134/135 被 worker 在插入前抢先启动，均已 stopjob（未测量）；由 `queue-after`（130 之后、130f 之后按名字顺序）重排 |
| `136-v3_n34_base48_60m` → `137-v3_n34_124_125x_117_48_60m` → `138-v3_n34_124_122_117_48_60m`（fable） | N34 三组（原 113–115，脚本不变）：底座 + 上限 48；A + 上限 48；A 把 125x 换 122 | pending，134/135 之后 |
| `139`–`144`（fable） | N34 容量单变量（各对 136；按 R34 修正）：139 KDA 状态池 418→256（不用 200：200÷5 会把并发压到 40）；140 mem 0.89 + 钉住 418 槽；141 hicache 96；142 去掉 MTP + 钉住 418 槽；143 主机层 write_back（独占式）；144 KDA 状态 bf16（能力冒烟开） | pending，最后 |

已撤：114/115（改名 137/138）、116–119（改名 139–142）、120/121（N34 探针，已由 131/132 在 N30 覆盖）、122/123（改名 134/135）。撤下的名字在 Pod runs/ 下只有部署收据，无数据。
