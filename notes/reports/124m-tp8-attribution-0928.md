# 124m：TP8 状态覆盖、数值边界与 N30 零触发归因

2026-09-28，Codex。运行引擎 `c4d01e56`；观察补丁 `6a86d57c`；分支 `codex/multiround-preempt-0928`。

**结论：多轮停车在专用探针中实际执行过；N30 40 分钟负载中一次都未执行。chain 14→12 不能归功于 124m。数值对照仍为 `NUMERICAL_REVIEW_REQUIRED`，不是数值 PASS。** 下一臂先测拒绝原因，不改触发条件。上传 PIN 包继续使用 `20a58da9` / `0927a`，不包含 124m。

## 原始记录与有效范围

短探针 ON 为 `130ezndz-tp8_124m_state_on`，OFF 为 `130eznfz-tp8_124m_state_off`。Pod 原始记录在各自 `/tmp/ax/runs/<job>/probe/receipt.json`；完整 JSON 已取回 [ON](../../evidence/T124m-tp8-0928/on/receipt.json) / [OFF](../../evidence/T124m-tp8-0928/off/receipt.json)，来源见同目录 `source.json`。仅为状态和输出诊断；两臂都将 cold 预算设为 8 秒以覆盖让路，不能当作 FINAL 30 秒策略的性能测试。

真实负载 ON 为 `130ezne-tail_rot150_n30_124m_on_40m`，OFF 为 `130eznf-tail_rot150_n30_124m_off_40m`。均为 v5g-tail-rot150、N30、2400 秒派发后排空、同引擎与同配置，只差 124m 开关；无 400 槽钉池。实际服务日志和 raw 在共享 checkout 的 `evidence/L<job>/N30/`；Pod 原件在 `/tmp/ax/runs/<job>/`。原始正确收据为 [ON timed_verdict](../../evidence/T124m-tp8-0928/on/timed_verdict.json) / [OFF timed_verdict](../../evidence/T124m-tp8-0928/off/timed_verdict.json)。

两臂分别完成 1679 / 1689 个请求，共同 1674，ON 独有 5、OFF 独有 15；共同集合的冻结 phase、chain index、输入/未缓存量、输出预算及 gap 一致。raw 无重复、无错误；收据派发集合与 raw 完全相等，flush 成功。状态为 **DRAINED 窗口诊断**，不是整集通过，不能套 341 条正式链的容错条数。

共享目录里的 `level_verdict.json=INVALID` 是下载工具用默认公开开发集重新判完整集产生的错口径：`fetch_status.json` 的 data_root 指向 `s1-dev/data/dev-combined-v1`，实际是 v5g-tail-rot150 的限时窗口。本报告保留该文件，不把它改成 PASS；使用上述原始 timed 收据及匹配 raw 核实窗口完整性。两份服务日志的 Traceback 均出现在排空之后的关闭阶段，不能作为测量中崩溃证据。

## 实测：状态探针成功覆盖了什么

ON 共 7 次 yield、6 次 resume、12 次 first_token、2 次 terminal；OFF 无 124m 事件。ON 客户端耗时 107.19 秒，OFF 150.33 秒（OFF 有重复对照，耗时不可用于速度比较）。

| 用例 | 实际覆盖 |
|---|---|
| `grid_minus/exact/plus` | 196608 token 的前后一个 token，跨页/256 网格边界；B 跨块完成，A 恢复 |
| `prefix_fork` | 4096 token 命中，前缀分叉后让路与续算 |
| `abort_A` / `abort_B` | 在服务器确认停车状态后取消对应角色；终止/恢复路径可见 |
| `reuse` | 前述请求清理后再次使用状态槽 |
| 每个 ON 用例 | parked 时 busy flush 被拒，排空后 idle flush 成功 |

这些证据证明已覆盖的状态路径能完成。未覆盖完整 KV/KDA 张量相等、高 KV 压力下回滚、强制服务时段到期、全局 abort_all 或 SLO；不扩大 `STATE_PASS` 的含义。

## 数值对照没有闭合

用现有探针的 `compare` 子命令对同输入 OFF/OFF/ON 做配对，结果为 [NUMERICAL_REVIEW_REQUIRED](../../evidence/T124m-tp8-0928/numeric-comparison.json)。8 组 case/role 都没有满足“全部输出步均落在两次 OFF 的 logprob 区间”这一严格检查。

| A 用例 | 分歧前共同输出步数 | OFF/OFF 最大 logprob 差 | ON/OFF 最大 logprob 差 |
|---|---:|---:|---:|
| grid_minus | 7 | 1.274807 | 0.948276 |
| grid_exact | 12 | 0.463740 | 0.343940 |
| grid_plus | 8 | 0.259326 | 0.551732 |
| prefix_fork | 2 | 0.080732 | 0.047667 |

B 的四组均在第一 token 分歧，不能直接比较“选中 token 的 logprob”作为同一个 token 的误差。第一步输入相同，top-8 仍可分析：例如 `grid_exact` 的两次 OFF 分别选 198、197，领先差都是 0.0625；ON 的 197/198 等分，选 197。说明固定输入的 OFF/OFF 本身就不逐位一致，但**不能据此排除停车造成额外误差**，尤其 grid_plus 的 A 已有较大差值。

下一份数值证据应固定生成历史，比较相同位置的共同 token 分数/状态，分别控制 batch 形状与停车开关，避免自由生成首 token 分歧后继续比较不同上下文。当前不放宽阈值、不把 smoke 12/12 当作这项证据。

复算：

```bash
python3 -B scripts/pod/verify/multiround_park_probe.py compare \
  --off evidence/T124m-tp8-0928/off/receipt.json \
  --on evidence/T124m-tp8-0928/on/receipt.json \
  --out evidence/T124m-tp8-0928/numeric-comparison.json
```

## 实测：N30 没有一次让路

直接读完整 ON `server.log`，并核对 Pod 原件：2,246,024 字节、10,722 行，`124m=on` 机制行存在，**`[ax-124m]` 事件数为 0**。OFF 同样为 0。因此 yield、resume、rollback、decline 均为 0；这也不能推出“KV 从未拦住救援”，因为 KV 预检发生在 begin/rollback 之前。

按原 harness `in_ttft_gate()` 与 `q()`，共同 1674 条的结果：

| 指标 | OFF | ON |
|---|---:|---:|
| chain >30 秒 | 14 | 12 |
| turn >15 秒 | 8 | 11 |
| overall >5 秒 | 200 | 194 |
| fast >3 秒 | 238 | 241 |
| TPOT >0.10 秒 | 76 | 96 |
| chain p95 | 31.663 秒 | 30.092 秒 |
| TPOT 均值 / p95 | 55.05 / 94.54 ms | 56.64 / 108.92 ms |

4 条改善、2 条新漏，均在稳态；开场漏数没有改善。由于机制没有实际切换，这 6 条变化不能归为“救回/抢占副作用”，只能报告为两次运行之间的变化。两臂首 token 时间改变还会改变后续闭环到达；同 ID 配对不意味着其他并发请求和缓存状态完全一致。

| 请求（尾缀；完整 ID 在 CSV） | 输入 / 缓存 OFF→ON | TTFT OFF→ON | 排队 OFF→ON |
|---|---|---:|---:|
| `lc_20260924_083:0009`，reset，新漏 | 153248 / 151040→151040 | 6.90→56.47 s | 6.15→55.69 s |
| `lc_20260924_098:0005`，reset，改善 | 22639 / 20992→20992 | 35.09→1.35 s | 34.55→0.88 s |
| `2f79d035…:9`，链首，改善 | 52749 / 33536→33536 | 32.21→8.76 s | 29.50→7.06 s |
| `dda45b2e…:0`，链首，改善 | 35314 / 33024→33792 | 59.66→5.16 s | 59.17→4.02 s |
| `ON3NogRU…:0`，链首，新漏 | 15859 / 13312→13312 | 5.06→51.67 s | 3.80→51.28 s |
| `nXmax-V7…:1`，链首，改善 | 202917 / 3840→3840 | 31.66→22.55 s | 2.83→2.23 s |

前 5 条的变化主要在等待，第 6 条主要在入执行后的耗时。不能只按“chain”标签把它们全归入大冷块抢占：其中两个新漏实际都高度命中缓存。`ax-prefix-decision` 仅记录前 120 秒，缺少这些稳态请求决策时的 held、KV 可分配量和冻结类别，现有日志不能进一步确认具体阻塞者。

逐请求证据：[全部共同 chain CSV](../../evidence/T124m-tp8-0928/n30/paired-chain.csv)、[6 条变化 CSV](../../evidence/T124m-tp8-0928/n30/changed-chain.csv)、[汇总与集合差异](../../evidence/T124m-tp8-0928/n30/comparison.json)。复算命令见 [证据 README](../../evidence/T124m-tp8-0928/README.md)。

## 150k A / 35k B：旧反事实与当前配置要分开

B 为 `biomaster:canon:member-v5-subagent-2f79d0356a87c796a247bf1f:llm:2`。35.997 秒坏例来自 **ezn9 的 S1 锚点**，不是 eznb 的 FINAL。相同 B 在 FINAL 已为 2.493 秒，在 124m ON/OFF 分别为 6.013 / 2.697 秒，均未漏。不要把已被其他配置修好的历史请求当作当前 124m 必须触发的依据。

S1 的占用者 A 为 `scimaster:canon:lc_20260924_293:llm:0003`，150249 输入、14848 缓存、TTFT 20.295 秒。它超过 harness 的 turn 15 秒门；但服务器 124 只见命中与冷暖代理，若冻结为冷类，其预算为 30 秒，**不等同于“已判死”**。同一服务器 `t_recv_s` 相减，B 在 A 接收 8.000929 秒后到达（更正初稿的约 8.39 秒）。[两请求原始行与四臂对照](../../evidence/T124m-tp8-0928/historical-35k-case.json)。

CPU 反事实使用真实长度/到达年龄及候选成本参数，甚至保守地假定 A 的全部 150249 token 尚未完成：`rescue_decision` 仍得到 `owner_rescuable`、A slack >9 秒。说明这个场景在首版“只有 A 已判死才能让路”的条件下可能本来就不触发。该计算不是历史决策日志：最初冻结类别、B 到达瞬间的复用量与 held/KV 状态缺失，不能宣称已还原当时第一条拒绝原因。

## 观察补丁：先知道哪条门挡住，再决定策略

`6a86d57c` 仅加观测，不改排序、成本、准入或让路条件。rank0 每 30 秒输出一次 `[ax-124m-plan]`；真正 flush 时补齐非空尾窗。每类拒绝每窗至多保留一条例子，没有请求正文或 GPU 同步。

- plan 结果：`no_owner`、`already_parked`、owner 已结束/不支持、batch/slot/running 空间不足、`selected`、`none`。
- 按原短路顺序的候选拒绝：`b_held`、`b_hostload`、`b_unsupported`、`kv_room`、`missing_receive_time`、`b_too_small`、`owner_tail`、`b_warm`、`owner_rescuable`、`b_late`、`age_limit`。
- 分母分别是 plan 调用次数和候选检查次数，不是唯一请求数；同一请求可以重复计数。未进入 plan 的上游返回不在统计内，最终不足 30 秒且无 flush 的尾窗也可能不输出。
- held/hostload/KV 预检的样例不提前调用 `deadline_cold()`，避免诊断反而改变冻结分类；不会为观测新增 collectives。rank0 plan 内也不能调用带广播的事件 emit。

已验证 137 项相关 CPU 用例，包括加计数前后决策相同、held 请求不被诊断提前冻结、30 秒限频以及上述 150k/35k 反事实。TP8 计数臂由 fable 审查并排队。若与 400 槽同臂，零触发问题可由计数定位，但性能收益仍必须对照同样钉池的 OFF，不能把扩池收益算进停车机制。

后续改动由计数决定：若多为 `owner_rescuable`，需讨论两请求都可完成的有界插入策略；若是 `kv_room`，先量化锁定容量；若是 `b_warm/b_too_small/held`，应核对其现有单轮服务路径。不能为了增加 yield 数量直接放宽这些门，也不能承诺靠这一机制解决全部 chain。
