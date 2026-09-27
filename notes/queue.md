# 实验队列

这里仅记**正在执行、已排队和下一步待决**的任务。Pod 实时状态以 `scripts/pod/pread status` 为准；已闭合、停止或作废的运行及其证据见[实验记录](experiments.md)。入队者负责跟到结果闭合，协作边界见[collaboration.md](collaboration.md)。

| Job | 要回答的问题与对照 | 当前状态 |
| --- | --- | --- |
| `130ez1-v3_n34_S1dcp_w2_60m`（fable） | 46757 的配置（S1 设置 + dcp 2、running 48）在 N34；也是后面全部探索旋钮的对照 | running（00:45 UTC 起） |
| `130ez4/ez5/ez6-v3_open_S1dcp_local_{offa,on,offb}_n34`（fable） | Codex 本地续算路径（6976639e，46758 多出的开关）TP8 三臂探针，N34 开场 | pending |
| `130ez6z-v3_open_S1dcp_w2_freeze_n34`（fable） | 新机制：124 的冷/暖类别在首次排序时冻结（`SGLANG_AX_DEADLINE_FREEZE_CLASS=1`，引擎 741f3eda = e464d8ab + K3（未开）+ 冻结）。来源：Codex 方案复核的反例 A，我用真实函数复现：36k 链首等 20 s 时命中 16k 可救（预算 30、slack +7.4）、命中 32k 后判为暖（预算 3 s、slack −17.8）反而排到最后，剩 3k 工作却超时——正是每次开场那 4 条家族兄弟的形态。探针 N34 开场对 130ez4（OFF-a，调度与 e464d8ab 相同），看 family_rider 类 chain 超时是否消失 | 已发布（130ez1 结束后插入，按名字排在 130ez6 后） |
| `130ez6zz-v3_open_S1dcp_prefix_off_n34` → `130ez6zzz-…_on_n34`（fable） | Codex 一口气实现的共享前缀生产者/兄弟准入（分支 codex/128-prefix-producer，c0fcd486，底 741f3eda，144 项 CPU 通过，两卡 TP2/DCP2+MTP+HiCache 输出与 OFF 逐 token 相同）：TP8 开场探针，OFF/ON 两臂只差 128p 开关，S1+dcp2、freeze=1、N34、冒烟门；ON 臂 trace 120 s/2048 轮。判定：同 ID chain 修复/新增（重点 family_rider），兄弟首次获选/READY/生产者获选时间戳。独立代码复核并行进行中，有阻塞则撤 | 已发布（130ez4 结束后插入，按名字排在 130ez6z 后、130ez7 前） |
| `130ez6zzzz-v3_open_S1dcp_combo_on_n34`（fable） | 合并探针：128p（c0fcd486）+ Codex 本地续算路径（f34c7ac4+6976639e）在同一引擎 791453ca（分支 claude/134-combo，摘取干净；CPU 143/144，唯一错误是机制报告测试的加载器解析不了新增的绝对导入，非运行缺陷；本地续算用例 OK）。S1+dcp2、freeze=1、N34 开场，对 130ez6zz（两者都关）。问题：两个各 −5 的 chain 收益是否叠加 | 发布者 T041752 连续错过（早期 PAUSE 被其他发布者收尾清掉）；05:15 UTC 手动 pause 队列，130ez9 结束后插入，按名序排在 130ezc 前 |
| `130ez7-…_pdi1`（闭）→ `130ez8-…_high8`（闭）→ `130ez9-…_slow250`（跑）→ `130ezc-…prefill_profile_dcp2` → `130ezd-…_multi5` → `130eze-…_nomtp`（fable） | 用户方向（09-27 00:40 UTC）：本地不再复现线上差异或重复已知结果，把值得探索的细节各调一次看效果。全部在 46757 的 DCP2 引擎、N34、60 分钟、对 130ez1 单变量：interval 2→1、HIGH_S 15→8、MAX_SLOW 80→250、K3 分级 warm 预算（引擎 f2b6425e = e464d8ab + 9be15822）、去 MTP。判定：chain/turn 降为主，TPOT/fast 涨可接受。05:47 UTC 重排：撤 130eza/130ezb，改名 130ezd/130eze 排到 prefill 剖析 130ezc 之后（发布者 queue-after-…054654 等 130ezc 结束），因为 N38 的杠杆是大上下文 prefill 吞吐，剖析结果越早越好 | 130ezd/eze 待 130ezc 后发布 |
| 已撤（09-27 00:45 UTC）：130ef（S1 参照，跑到 30 分钟停）、130ef5（GPQA 参照）、130ez2/ez3（W1 对照与 N30 重复）、130f、130g/h（v3g）、136–144（旧引擎容量旋钮，被 DCP 取代） | — | 脚本保留，需要时再排 |
| `dcp-mtp-20260926`（Codex） | 当前候选分支 `codex/dcp-mtp-stack2-n34`，原开发分支 `codex/dcp-mtp-n34`；保留MTP/HiCache推进N34/N38；技术方案在候选worktree的 `notes/plan-dcp-8card.md` | 两卡开发矩阵已闭合，结果见候选分支 `notes/experiments.md` 的DCP-20260926。W1/W2 N34配置已准备；TP8真实权重、另一参与者raw复核及N34/N38待验，尚未入队Pod |

已撤：114/115（改名 137/138）、116–119（改名 139–142）、120/121（N34 探针，已由 131/132 在 N30 覆盖）、122/123（改名 134/135）。撤下的名字在 Pod runs/ 下只有部署收据，无数据。
