# 19k 冷头的原生浅前缀暂缓：归因修正与 128g

2026-09-27，Codex。原日志归因已获 Fable 独立复核，他在 `89a35f40` 撤回原“128p 家族折算”解释；本补丁 `873031fd` 的代码独立审查也已通过，[收据](../../evidence/lpm-reuse-guard-0927/fable-review.json)记录范围与残余风险，TP8 效果待验。独立分支 `codex/lpm-reuse-guard-0927`，底 `cfd25d0f`。

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

逐决策相邻时间间隔中，ON 的浅 held 快照覆盖 27.082 秒、OFF 20.763 秒。这些是相邻观测间隔之和，不是持续持锁时间，更不是“修复必省 27 秒”。修复需要真实 OFF/ON 重跑，不能回填反事实 TTFT。

## 交付与验证

[机制说明](../../engine/docs/128-lpm-reuse-guard.md)给出开关、边界和日志。补丁只排除新增可复用量上界为零的原生暂缓；正上界不等于 READY，也不改变 128p 家族评分。单变量臂为 S6 + `SGLANG_AX_LPM_REUSE_GUARD=1`，要求 `128g=on`，同引擎控制臂显式置零。

CPU 新增 14 项、128p 35 项、原调度器 32 项均通过；核心复现运行生产策略、真实 RadixKey/模拟树和真实调度器/PrefillAdder，OFF 先选大头，ON 先选 19k，同时保留真正深前缀的兄弟依赖。没有新增 GPU 分配，临时模拟树 CPU 开销及最终净 chain 收益待 TP8。

Fable 另行审查并报告 guard、128p、调度和 deadline 四组测试通过。8 卡首轮采用 ON `130ezn2a` 对 S6 `130ezn2` 的跨提交筛选；严格同引擎 OFF 入口仍保留。选择与比较边界见[交接](../handoffs/lpm-reuse-guard-0927.md)，运行状态只在共享队列维护。

## 可复算证据

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
