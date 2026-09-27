# 共享前缀生产者与兄弟准入：实现与验证

2026-09-27，Codex。实现底座 `741f3eda`，分支 `codex/128-prefix-producer`，worktree `build/worktrees/128-prefix-producer`。用户授权完整实现，Fable 已确认 124/128、held、就绪与逐决策接口由此分支修改。未更改共享 Pod 队列或运行服务；TP8 开关探针由 Fable 复核后安排。

## 交付与边界

完整机制在 `SGLANG_AX_PREFIX_PRODUCER=1` 后：缓存就绪视图、跨阶段生产者账本、held 协调、rank0 两阶段计划、READY 短尾预算与同轮回退、取消/retract/flush/RID 代际失效、逐决策及原生完成时间日志。与旧 `SGLANG_AX_DEADLINE_FAMILY` 互斥。源码接口、参数及资源账统一见[机制文档](../../engine/docs/128-prefix-producer.md)，不在报告复制实现说明。

本次没有给独有长历史制造算力收益，也不强制新增 KDA 检查点。目标是使现有可共享工作更早完成、使已经合法 READY 的尾巴及时获选。是否减少开场 chain 超时，必须由 N34 同 ID ON/OFF 证明。

用户提供的 46677 N26 结果将 chain 指为最紧的门；N30 失败档没有返回明细，不能把“因 chain 失败”写为平台实测结论。平均 TPOT 余量不是可直接花掉的四倍预算，仍需看正式 p95 与超时条数。

## 验证证据

**OBSERVED：CPU 144 项通过。** 使用真实 Scheduler/PrefillAdder 方法与原生 Mamba validator/COW 代码，替换模型执行、池与缓存数据。包括默认 OFF 对 741 的完整调度轨迹、不同 rank 时钟、两个 held 来源情况、哈希碰撞/深浅 LCP、host-only/KDA tombstone、实际 rematch 回退、锁后 KV 减少、COW/adder 拒绝、已持有状态不误释放、126 拒绝后恢复与重试、124 停车回退、S1 125/cap/短尾组合、取消/retract/flush/RID 重用、单 partial/总预算、独立结果处理器的完成时间记录。[完整输出](../../evidence/T128-prefix-producer-20260927/cpu-tests.log)

```bash
PYTHONPATH=tests python3 -m unittest test_ax_deadline test_demand_cap \
  test_ax_admission_scheduler test_sched_protect_chain test_prefix_producer -v
```

**OBSERVED：CPU 成本有界。** 合成每请求 201024 token，native `array('q')`，三次冷启动的中位数：34 候选 39.1 ms、64 候选 68.4 ms；已有账本每轮中位数分别 0.213/0.534 ms。字节缓存分别 54.68/102.92 MB。这是本机纯规划器结果，不是 TP8 服务开销。实现使用词典序相邻 LCP 加 pair 复用，避免每对请求重复验证长前缀；新 cohort 的首次 CPU 成本仍需在 TP8 上观察。[原始样本](../../evidence/T128-prefix-producer-20260927/cpu-planner.json)

```bash
python3 scripts/analysis/prefix_producer_cpu_bench.py --output /tmp/prefix-planner.json
```

**OBSERVED：TP2/DCP2 + MTP + CUDA graph + HiCache 真实引擎诊断。** GPU0/1，既有缩小的 8+1 层 H16 模型，确定性缩放 dummy 初始化，原生调度、cache、COW、前向和回收。21 请求覆盖预热、两次真 flush 后冷家族、暖重复；每请求 16 个输出 token，并检查有限 logprob。该结果不构成能力门或 N@SLO 成绩。

最终候选 `on-candidate` 与 `off-final` 的 **21/21 输出 token 一致，输出 logprob 最大绝对差为 0**。ON 有 17 条逐决策、9 次 READY 预留且全部实际准入、12 条参与者 prefill 完成日志；READY 到准入 6.15–6.85 ms。两次 flush 的代际为 2/3，均实际重建依赖并消费检查点；服务日志没有 Traceback、OOM 或 rank 共识拒绝。这些毫秒数是服务端准入间隔，不是 chain TTFT 或 ON/OFF 加速比。[完整日志、逐请求输出与复算入口](../../evidence/T128-prefix-producer-20260927/README.md)

开发机保持 short 4096、cold cap 8192、PDI 1，125/126 关闭；S1 的 short 8192、cap 6144/125 relief 8192 等交互由 CPU 场景覆盖，尚不能替代 TP8 的实际组合验证。

实机发现并修正了一个日志接线问题：结果处理器是独立 frozen dataclass，不持有 Scheduler 账本。现在在 Scheduler 调用原生结果处理器后采集已有完成时间；新增测试覆盖这条真实分发路径。首轮完整前向没有失败，但缺完成日志的试跑未作为完整观测验收。

## TP8 交接

两臂使用同一个新引擎提交，沿用 Fable 的 `v3_open_S1dcp_w2_freeze_n34.sh` 数据、冒烟、running/graph 48 和 S1 环境（冷 cap 6144、short 8192、warm 15、MAX_SLOW 80、DCP2、MTP、HiCache）。保留两臂 `SGLANG_AX_DEADLINE_FREEZE_CLASS=1`，不要夹带 PDI/HIGH_S/MAX_SLOW/K3/local-extend 变化。

| 臂 | 追加或显式设置 G_ENV | 追加 G_EXPECT |
| --- | --- | --- |
| OFF | `SGLANG_AX_PREFIX_PRODUCER=0 SGLANG_AX_DEADLINE_FAMILY=0` | `128p=off 128=off` |
| ON | `SGLANG_AX_PREFIX_PRODUCER=1 SGLANG_AX_DEADLINE_FAMILY=0 SGLANG_AX_PREFIX_TRACE_S=120 SGLANG_AX_PREFIX_TRACE_ROUNDS=2048` | `128p=on 128=off` |

看原 harness 完整性和冒烟，再按同 ID 比较 chain 修复/新增、生产者获选、合法 READY、兄弟获选、首 token、实际新算 token，以及 turn/fast/TPOT 和 KV/KDA 峰值。新 CSV 是服务端归因入口，不能替代 raw。短开场仅作机制诊断，不能称 N34 通过；有收益再做完整档/N38。

独立复核请求随提交发给 Fable，TP8 测量尚未执行；本报告不预先宣称 review PASS 或 chain 降低。
