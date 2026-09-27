# 原生 LPM 浅前缀暂缓：128g 修复与 S6 配对归因

2026-09-27，Codex。独立分支 `codex/lpm-reuse-guard-0927`，底 `cfd25d0f`，引擎补丁 `873031fd`。Fable 的代码独立审查已通过，[收据](../../evidence/lpm-reuse-guard-0927/fable-review.json)记录范围与残余风险；他也在 `89a35f40` 撤回修复前 19k 坏例的“128p 家族折算”解释。

**本轮结论：128g 在 S6 / v5g-tail / N26 的跨提交筛选中 chain 超时 8→8，修好 1 条、新坏 1 条，暂不进入候选。** 开场证据确认浅前缀代表轮换会压后可救冷头；稳态新增超时则有明显的闭环到达相位变化和 KV 压力，不能归成同一 held 缺陷。以下保留修复前的根因证据，再给出闭合探针的逐请求分析。汇总指标已对照 Fable 的配对结果复算；新增细节归因为 Codex 本次审计，尚未另行独立复核。

## 已证实的等待路径

对象为 `scimaster:canon:_XiHg9hppWn2KknKrPrj9:llm:0`，19,335 token，两个运行均 cache=0。以下 OFF/ON 指 **118**，两臂尚无 128g：

| 原始运行 | TTFT | 接收到首次执行 | 首次执行到首 token | held 决策数 | 首次非 held / 首次入批轮 |
|---|---:|---:|---:|---:|---:|
| 130ezm7：118 OFF | 25.420 s | 23.430 s | 1.991 s | 43 | 45 / 50 |
| 130ezm8：118 ON | 32.255 s | 30.401 s | 1.855 s | 61 | 64 / 71 |

raw 的 `t_admit_s` 是入口准入时间，不能作为第一次模型入批时间；表中等待使用 `t_exec_start_s - t_recv_s`，调度轮次使用实际 `admitted` 集合。

ON 原日志的 epoch 2：

| 决策轮 | 19k 的状态 | 同轮入批者 |
|---|---|---|
| 50 | ORDINARY；`held/effective_held=true`；深度 58；work=null | 40,483 token，FALLBACK / hold_timeout，work=null |
| 56 | 同上 | 60,848 token、device hit 18,176，ORDINARY，work=null |
| 63 | 同上 | 50,581 token，ORDINARY，work=null；也是当前 held owner |
| 64 | 首次解除 held，并排在 waiting 首位 | 50k 已经成为 continuation；19k 无法完整放入当前 8k 预算 |
| 71 | 首次真正入批 | 19k 自身 |

这些行另从 Pod 原件 `/tmp/ax/runs/130ezm8-v3_open_S5b_118prefill_n34/server.log` 复核，不是仅凭本地摘要。原生 owner 多次轮换，ON 第 3–63 轮一直 effective held。128p 的 `MAX_HOLD_S=8` 约束它自己建立的 Dependency，不约束这种 ORDINARY 原生 LPM 暂缓。因此“只在头 1 秒被 hold”与原记录不符；上述三个实际入批者也没有当轮 `work` 家族折算值。

## 源码解释与边界

调用关系为 `calc_priority -> _compute_prefix_matches -> simulated RadixCache -> ax_held -> Tracker.prepare/key -> PrefillAdder`。原生模拟树逐 token 匹配，>=32 token 暂缓；真实缓存通过 `registry -> UnifiedRadixCache -> unified_tree_core.match_prefix` 按页匹配，再由 Mamba component validator 验证状态。`MambaRadixCache` 老类不是本轮实际实例，不能据它单独下结论。

50/58 token 低于本配置的真实缓存页下界 64，不存在新增可复用页。KDA prefill checkpoint grid 与 decode tracking interval 不是一个量；修复使用 `mamba_checkpoint_grid(tree_cache.page_size)`，不把 256 写成通用常数。DCP 的扩大页由现有 cache builder 提供。

**可确认：**本例存在无可复用收益的原生 held，实际延后了可参与正常排序的轮次。**不能确认：**解除它后恰好少几条超时、其他冷头不会新超时、其余晚入批请求全是不可消除的产能尾部。排队超过 30 秒并不足以区分产能与排序问题。

逐决策相邻时间间隔中，ON 的浅 held 快照覆盖 27.082 秒、OFF 20.763 秒。这些是相邻观测间隔之和，不是持续持锁时间，更不是“修复必省 27 秒”；不能用它们回填反事实 TTFT。128g 实测另见下节，不能与这里的 118 OFF/ON 混用。

## 交付与验证

[机制说明](../../engine/docs/128-lpm-reuse-guard.md)给出开关、边界和日志。补丁只排除新增可复用量上界为零的原生暂缓；正上界不等于 READY，也不改变 128p 家族评分。单变量臂为 S6 + `SGLANG_AX_LPM_REUSE_GUARD=1`，要求 `128g=on`，同引擎控制臂显式置零。

CPU 新增 14 项、128p 35 项、原调度器 32 项均通过；核心复现运行生产策略、真实 RadixKey/模拟树和真实调度器/PrefillAdder，OFF 先选大头，ON 先选 19k，同时保留真正深前缀的兄弟依赖。没有新增 GPU 分配；临时模拟树 CPU 开销尚无专门计时，不能由整轮 TPOT 推出 host 开销。

Fable 另行审查并报告 guard、128p、调度和 deadline 四组测试通过。8 卡首轮 ON `130ezn2a` 对 S6 `130ezn2` 的跨提交筛选已闭合；严格同引擎 OFF 入口仍保留，未据此安排新实验。选择与比较边界见[交接](../handoffs/lpm-reuse-guard-0927.md)，运行状态只在共享队列维护。

## TP8 配对：两条换手到底发生了什么

### 证据范围与有效性

OFF 是 `L130ezn2-tail_n26_S6_40m/N26`，引擎 `450e8580`；ON 是 `L130ezn2a-tail_n26_S6_lpm_guard_on_40m/N26`，引擎 `873031fd`。两臂均为 **DCP1、无 MTP、running32、118/132 关闭**，不是 DCP2 实验。实际 ON 机制行是 `128g=on:256`，不能拿机制文档的旧示例 `64` 当实际网格。

完整 raw 分别为 1547 / 1551 条，无重复 ID、无错误，均与 dispatched / attempted 相等，12/12 冒烟、相同 rep16 预热计划、随后 flush 和排空收据齐全。run 中 workload hash 都是 `9d9cffd945ee6b1c`，cohort 都是 `ff1dccae1087a798`。同 ID 1542 条的冻结 phase、链内序号、prompt、输出预算和等待字段相同。按原 harness 的 `in_ttft_gate`、`q` 复算：chain 150 条中超时 **8→8**，p95 **41.344→38.256 s**；turn 5→5、overall 74→81、fast 75→78、TPOT>0.10 为 73→74，TPOT 均值 50.267→49.534 ms。p95 数值下降没有转换成少一条 chain 超时，也不足以证明更高档位可过。

有效性为 **DRAINED/DIAGNOSTIC**：40 分钟派发后的全部已准入请求闭合，不是完整 cohort 成绩；同 ID 子集也不能当固定到达时刻的负载。保留的本地 `level_verdict.json` 是 INVALID：`fetch_status.json` 显示取证后的重评分用了原始 `s1-dev/data/dev-combined-v1`，报 missing 467、extra 1292/1296。它不能判这两份合成短测 PASS；这里另核 raw/run/排空收据，只报告注明范围的诊断，不覆盖原判分文件。跨提交且只有一次配对，不能称稳定的单变量因果效应。

### 修好的 79k：去掉无收益的代表轮换，真正重算量没变

对象 `biomaster:canon:6thoYJhAH8oJW6nmGr9Hr:llm:1`，idx=0，因此即使数据 phase=intra，原 harness 仍归 chain。两臂 prompt=79,033、**实际 cached_tokens=0**，trace 的 device=0、left=79,033，始终 ORDINARY、producer=null、work=null。冻结的 `uncached_expected=1,584` 不是实际短尾计算量；本例不能写成“1.6k 暖兄弟得到了缓存复用”。

| 拆分（秒） | S6 OFF | S6 + 128g |
|---|---:|---:|
| 接收到首次模型执行 | 49.087 | 23.062 |
| 首次执行到首 token | 6.174 | 6.225 |
| TTFT | 55.261 | 29.287 |

OFF 从 seq4 到 seq57 有 54 个 effective-held 观测，深度仅 50/58；代表依次为 `7l9…`、`member…f3fc…`、`c371…`、19k 的 `_XiHg…`、最后是 **222,738-token 的 `z9-Jb…:llm:1`**。原日志完整 ID 与行号保存在逐决策证据，轮换不是推测：

| OFF 决策轮 | 实际发生的事 |
|---|---|
| 30–57 | 79k 被 222k 的浅前缀代表暂缓；seq57 先准入 222k |
| 58 | 79k 首次解除 held，距其 due 约 6.39 s；222k 已是 continuation |
| 58–85 | 222k 继续；79k 没有入批 |
| 86 / 92 / 98 | 又先准入剩余 35,136 / 35,143 / 56,095 token 的普通请求 |
| 105 | 79k 才真正入批 |

ON 从 seq5 第一次观测起即无 held：shared=58（后为 50）、grid=256、gain_upper=0，触发 `release_zero_gain`；seq57 就准入 79k。它在整个目标 trace 中没有 effective-held 观测。收益落在首次执行前，等待少 **26.025 s**，执行到首字反而多 0.051 s；并非 kernel 提速。

更具体的取舍是：222k 的 TTFT 从 **41.344→56.492 s**，两臂本来都超过 30 s；79k 从超时变为过门。开场这一处确实做到了“先保还能过门的一条”。解除 held 后仍可能等 continuation，原因可在 `Tracker.should_park` 与 `PrefillAdder.add_one_req` 看到：完整头必须放得进本轮预算，79k 冷头不能作为 8k 预算内的短命中插队。128g 没有修改这项约束，也没有改变家族成本评分。

### 新坏的 80k reset：闭环到达提前，稳态资源等待仍存在

对象 `scimaster:canon:lc_20260924_307:llm:0062`，idx=62、phase=context_reset，原 harness 归 chain；两臂 prompt=79,915、cached=78,592，新增计算约 1,323 token。

| 拆分（秒） | S6 OFF | S6 + 128g |
|---|---:|---:|
| 相对本轮首次派发的接收时刻 | 1314.828 | 1277.095 |
| 相对本轮首次派发的首次执行时刻 | 1320.100 | 1314.851 |
| 接收到首次模型执行 | 5.272 | 37.756 |
| 首次执行到首 token | 0.392 | 0.500 |
| TTFT | 5.664 | 38.256 |

**实测：它在 ON 提前 37.733 s 到达，首次执行也提前 5.249 s，但 TTFT 多了 32.593 s。** 同 ID 不等于相同的并发现场；这不是“执行相对整轮时钟变晚”的证据。

前驱 `…307:llm:0061` 更能解释相位变化：两臂同为 intra/fast，prompt=78,592、cached=76,288、输出 97；TTFT **35.739→0.461 s**，等待 **35.287→0.206 s**。它的完成时刻从相对 1313.075 s 提前到 1275.343 s，随后相同的 1.731 s 冻结 gap 把 reset 提早送入拥挤窗口。观测到的长等待从前驱 fast 移到了后继 chain；两条请求的服务时间并非逐项守恒，不能把这个描述当反事实证明，但它解释了为什么局部更快仍可能多一条 chain 超时。

稳态现场的证据与边界：

- **没有该请求的逐决策 trace。** 两臂 epoch2 分别在 elapsed 120.459 / 120.518 s 结束记录，无法提供它在 1277/1315 s 的 `held_by`、当次 device/host 命中或具体拒绝原因。
- ON 的 19:34:55、19:35:27、19:35:58 三个累计快照均为 keep=495、release=151；READY、dependency、wait_prefix 也不增加。覆盖其等待的窗口里**没有记录新的 guard 释放或家族 READY 事件**，不支持“当场又释放了兄弟，兄弟抢占了它”的解释。此前排序改变后续缓存驻留和到达节奏的间接影响仍不能排除。
- ON 等待区间内 10 秒采样的 KV 占用为 **95.2%–99.3%**，可用加可淘汰约 **9.3k–60.2k token**，running 15–16、queue 2–4、KDA 8.2%–8.7%。这是较强的 KV 压力证据；也不是所有瞬间都“满 100%”。同窗口累计普通准入 no_token +26、other +159、request_slots 截止 +21，属于**全部候选**，不能挂到这一个请求头上。
- 一个 254,821-token、仅命中 3,840 的请求 `…295:llm:0003` 在 ON 的目标到达前已开始，前向到首字跨度直到目标到达后 **4.245 s**，不能把 37.756 s 全部归为这一段 prefill 占满通道。它随后继续 decode，直到相对 1333.073 s 才结束，晚于目标首次执行；其 KV 驻留仍可能持续加剧容量约束。随后还有多批暖请求与一个新增约 32.5k 的 prefill；完整同请求区间已导出。raw 的 forward→first 是含交错的端到端区间，不是 kernel 占用时间。
- 最终 `cached_tokens=78,592` **不能证明它到达时就有 78,592 个可用 device token**。源码把 host 回载空间计入驻留准入预算，并扣除 running 输出预留、页和状态成本；实际可复用的 host 命中能省计算，仍可能等 device 空间。10 秒标量采样不能还原这一刻的锁与预算，不能仅用“只重算 1.3k”排除 KV 准入问题。

**推断：**这条新增 chain 超时更符合闭环相位变化下的稳态资源／准入等待，不像又一次浅前缀 held；缺少该窗口逐请求证据，尚不能在 host 回载、KV 预留、continuation 限制和候选排序之间定案。

### 计数更正与后续边界

传入摘要的“保留 32 / 释放 31”实际出自 **epoch1 预热**（ON server.log 第 1064、1162 行）。epoch2 开场 trace 是 keep=495 / release=151 个候选轮次观测，对应不同请求 20 / 5 条，其中 2 条在不同时刻两类都出现；最后保留的累计快照是 495 / 156，不能当排空瞬间精确终值。这些计数都不是修好的 chain 数。两臂最后快照的 dependency=6、dependent_admitted=6、READY=3，说明不能把开场收益泛化为“新增六条家族复用”。

此次结论只支持继续默认关闭、不进入当前候选，不否认零收益 held 的工程缺陷。若后续专门追稳态，应对同引擎两臂在相同条件下增加**有预算、由等待时间触发**的逐决策记录：真实 device/host 匹配、锁前后 KV 余量、输出预留、候选序位、拒绝原因和 continuation。先区分“短计算但需大量回载空间”与“设备短尾已有空间却被排序压后”，再决定预留驻留空间、准入救援或 prefill 抢占。不能只依据本例把 guard 再扩大，也不能用评测 req_id、phase 标签或丢弃超时请求换指标。本次只交归因和复算工具，没有修改引擎或队列。

### 本轮可复算材料

- [输入收据、原 harness 统计、两个目标与前驱时间线](../../evidence/lpm-reuse-guard-0927/tp8-s6/summary.json)
- [全部 150 条 common-ID chain 配对](../../evidence/lpm-reuse-guard-0927/tp8-s6/chain-pair.csv)
- [79k 逐轮代表、结果、同轮入批与 continuation](../../evidence/lpm-reuse-guard-0927/tp8-s6/target-decisions.jsonl)
- [按目标接收时间对齐的完整采样窗口](../../evidence/lpm-reuse-guard-0927/tp8-s6/target-metrics.csv)
- [与等待重叠的其他请求 forward→first 区间](../../evidence/lpm-reuse-guard-0927/tp8-s6/overlapping-requests.csv)
- [未被 trace 覆盖的稳态目标服务日志窗口，含原行号](../../evidence/lpm-reuse-guard-0927/tp8-s6/target-server-windows.jsonl)
- [审计脚本](../../scripts/analysis/lpm_guard_pair_audit.py)

```bash
python3 -B scripts/analysis/lpm_guard_pair_audit.py \
  --harness /workspace/Agentic_science_challenge/s1-dev/harness \
  --off /workspace/Agentic_science_challenge/evidence/L130ezn2-tail_n26_S6_40m/N26 \
  --on /workspace/Agentic_science_challenge/evidence/L130ezn2a-tail_n26_S6_lpm_guard_on_40m/N26 \
  --rid biomaster:canon:6thoYJhAH8oJW6nmGr9Hr:llm:1 \
  --rid scimaster:canon:lc_20260924_307:llm:0062 \
  --out evidence/lpm-reuse-guard-0927/tp8-s6
```

## 修复前 19k 根因的可复算证据

- [汇总及原始输入摘要](../../evidence/lpm-reuse-guard-0927/summary.json)
- [全部 chain 的 held 观测](../../evidence/lpm-reuse-guard-0927/chain-holds.csv)
- [19k 每轮状态、原日志行号及同轮入批者](../../evidence/lpm-reuse-guard-0927/target-decisions.jsonl)
- [审计脚本](../../scripts/analysis/lpm_hold_audit.py)

运行中不使用原始请求 ID 作任何决策；ID 只用于离线同请求归因。审计保留 <64/<128/<256 三档快照计数作为敏感性统计，不能据此假定该运行的 grid，也不是预测可挽救条数。只检查 raw 唯一性、错误与时间关系并按原 harness `in_ttft_gate` 选 chain；不冒充完整数据的正式判分。

```bash
python3 -B scripts/analysis/lpm_hold_audit.py \
  --harness /workspace/Agentic_science_challenge/s1-dev/harness \
  --level off=/workspace/Agentic_science_challenge/evidence/L130ezm7-v3_open_S5b_118off_n34/N34 \
  --level on=/workspace/Agentic_science_challenge/evidence/L130ezm8-v3_open_S5b_118prefill_n34/N34 \
  --rid scimaster:canon:_XiHg9hppWn2KknKrPrj9:llm:0 \
  --out evidence/lpm-reuse-guard-0927
```
