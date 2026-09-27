# 实验队列

这里仅记**正在执行、已排队和下一步待决**的任务。Pod 实时状态以 `scripts/pod/pread status` 为准；已闭合、停止或作废的运行及其证据见[实验记录](experiments.md)。入队者负责跟到结果闭合，协作边界见[collaboration.md](collaboration.md)。

| Job | 要回答的问题与对照 | 当前状态 |
| --- | --- | --- |
| `130ez1-v3_n34_S1dcp_w2_60m`（fable） | 46757 的配置（S1 设置 + dcp 2、running 48）在 N34；也是后面全部探索旋钮的对照 | running（00:45 UTC 起） |
| `130ez4/ez5/ez6-v3_open_S1dcp_local_{offa,on,offb}_n34`（fable） | Codex 本地续算路径（6976639e，46758 多出的开关）TP8 三臂探针，N34 开场 | pending |
| `130ez6z-v3_open_S1dcp_w2_freeze_n34`（fable） | 新机制：124 的冷/暖类别在首次排序时冻结（`SGLANG_AX_DEADLINE_FREEZE_CLASS=1`，引擎 741f3eda = e464d8ab + K3（未开）+ 冻结）。来源：Codex 方案复核的反例 A，我用真实函数复现：36k 链首等 20 s 时命中 16k 可救（预算 30、slack +7.4）、命中 32k 后判为暖（预算 3 s、slack −17.8）反而排到最后，剩 3k 工作却超时——正是每次开场那 4 条家族兄弟的形态。探针 N34 开场对 130ez4（OFF-a，调度与 e464d8ab 相同），看 family_rider 类 chain 超时是否消失 | 已发布（130ez1 结束后插入，按名字排在 130ez6 后） |
| `130ez7-…_pdi1` → `130ez8-…_high8` → `130ez9-…_slow250` → `130eza-…_multi5` → `130ezb-…_nomtp`（fable） | 用户方向（09-27 00:40 UTC）：本地不再复现线上差异或重复已知结果，把值得探索的细节各调一次看效果。全部在 46757 的 DCP2 引擎、N34、60 分钟、对 130ez1 单变量：interval 2→1、HIGH_S 15→8、MAX_SLOW 80→250、K3 分级 warm 预算（引擎 f2b6425e = e464d8ab + 9be15822）、去 MTP。判定：chain/turn 降为主，TPOT/fast 涨可接受 | 已发布（queue-after-…004358） |
| 已撤（09-27 00:45 UTC）：130ef（S1 参照，跑到 30 分钟停）、130ef5（GPQA 参照）、130ez2/ez3（W1 对照与 N30 重复）、130f、130g/h（v3g）、136–144（旧引擎容量旋钮，被 DCP 取代） | — | 脚本保留，需要时再排 |

已撤：114/115（改名 137/138）、116–119（改名 139–142）、120/121（N34 探针，已由 131/132 在 N30 覆盖）、122/123（改名 134/135）。撤下的名字在 Pod runs/ 下只有部署收据，无数据。
