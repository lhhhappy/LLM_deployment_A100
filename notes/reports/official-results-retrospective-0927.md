# 09-27 正式结果复盘：容量收益没有兑现为晋档，下一版须对准已通过的 S4

2026-09-27，Codex。状态：正式结果已只读查询；配置、六组既有本地 raw 与 125 调用链已独立核对。解释中的因果假设尚待另一位参与者复核。本轮没有运行 GPU 实验、修改引擎或调整队列。

## 结论

今天的两份正式提交没有超过昨日的 N26。需要反思的是选方向与验证之间的断层：本地 DCP 缓解容量压力、局部 chain 减少，并不自动构成线上晋档证据。更具体地说，**下一版 S5 的“chain 少五条”相对的是 S3；已经上线到 N26 的 S4 在本地也少这五条。继续用 S3 当唯一参照，会把已经兑现过的收益当成新增收益。**

这既不能证明 DCP 无用，也不能证明 DCP 无害；不能由返回的通过档判定失败档一定卡 chain，更不能断言只卡开场。当前应保留 46677 为正式性能参照、46758 为新路径参照，优先找相对二者真正新增的收益。

## 1. 官方终态究竟证明了什么

本轮用 `scripts/official_status.sh 46677 46757 46758` 重新只读查询，结果与保留的官方 JSON 一致。三份均完成、能力门通过；正式排名看 N@SLO、再看该档 TPOT 均值，不能用平台通用 score 的上涨替代晋档。

| 指标 | 46677，S2 | 46757，S3 | 46758，S4 |
|---|---:|---:|---:|
| N@SLO | **26** | **22** | **26** |
| chain p95，秒 | 39.13 | 29.14 | 41.11 |
| turn p95，秒 | 4.88 | 4.24 | 9.43 |
| overall p95，秒 | 2.76 | 2.16 | 2.36 |
| fast p95，秒 | 1.82 | 1.45 | 1.42 |
| TPOT 均值 / p95，毫秒 | 25.10 / 47.54 | 22.21 / 40.87 | 25.66 / 47.75 |

S2→S4 是同为 N26 的正式比较：chain p95 +5.05%，turn +93.48%，overall −14.64%，fast −21.73%，TPOT 均值 +2.23%。观察到的是短请求桶改善、chain 没改善、turn 明显变差，尚非统计显著性或单机制因果结论。S3 的 N22 chain 29.14 秒不能与 S2/S4 的 N26 横比并称“chain 改好了”。

平台只返回最高通过档；根据正式爬坡规则，S3 未过 N26，S2/S4 未过 N30，但这些失败档的十一门、错误率与逐请求记录未返回。通过档 chain p95 超 30 秒仍可因统计余量过门；正式 chain 桶还含非链首 context_reset，不能直接用 341 条链代替桶样本数。**不能从一个 p95 反推超时条数、发生时段或“过不过只是掷硬币”。**

原始依据：[46677](/workspace/Agentic_science_challenge/evidence/official/attempt-46677-20260927.json)、[46757](/workspace/Agentic_science_challenge/evidence/official/attempt-46757-20260927.json)、[46758](/workspace/Agentic_science_challenge/evidence/official/attempt-46758-20260927.json)、[正式规则](/workspace/Agentic_science_challenge/llm-challenge-arena-v1/task.md:495)。

## 2. 今天并没有隔离出 DCP 的正式净效果

直接解析三份 `submission.json` 的命令与 env，S2→S3 的变化为：

- DCP 1→2，启用 compact top-k，关闭 DSA sparse Triton 路由；镜像引擎增加 DCP 实现。
- running / CUDA graph 最大 batch 32→48，KDA 槽显式设为 418。
- **MAX_SLOW 250→80**，即回到了 S1 护栏，而非昨日通过 N26 的 S2 护栏。

因此 S3 是一个配置组合，N26→N22 不能单独归因于 DCP。S3→S4 的启动参数相同、只新增 LOCAL_EXTEND env，并更新含对应实现与修正的镜像；这是更接近单机制的比较，观察到档位恢复一档，但没有失败档数据或重复运行，不能量化它在线上修复了几条 chain。

配置源：[S2](/workspace/Agentic_science_challenge/evidence/submission-0926-s2/submission.json)、[S3](/workspace/Agentic_science_challenge/evidence/submission-0926-s3/submission.json)、[S4](/workspace/Agentic_science_challenge/evidence/submission-0926-s4/submission.json)。完整差异保存在本报告的 `audit.json`。

## 3. 新发现：S5 的参照必须包括 S4

以下均重新读取 raw，使用原 harness 的桶划分，在共同请求 ID 上计算。校验每条恰好一次、零错误、派发与完成数相同、flush 成功、相同 cohort/workload，以及共同 ID 的 prompt、输出 token 与桶标签相同。它们是已排空的限定窗口诊断，**不是全量档位成绩**；不同窗口的完成时间不可直接当吞吐提升。

| 比较 | 共同请求 | chain 超时 | 修复 / 新增 | 解释范围 |
|---|---:|---:|---:|---|
| 130ez5（本地续算）→130ez6zzzz（再加 128p/冻结） | 499 | **11→11** | 2 / 2 | 对应 S4→S5 的开场增量，没有净 chain 收益 |
| 130ez1（S3）→130ezf（S5 有 MTP） | 1588 | 16→11 | 8 / 3 | 对 S3 有收益；不能当对 S4 新增五条 |
| 130ezf（S5 有 MTP）→130ezh（去 MTP） | 1362 | **11→11** | 0 / 0 | 去 MTP 没进一步改善这批 chain |

我实现的 128p 有真实的依赖发现与准入收益，但收益与本地续算重叠，这一点必须落实到下一轮选臂。它还存在代价：S3→S5 同 1588 条中，overall 超时 43→57（修复 25、新增 39）、fast 47→55。新增坏例不等于已证明由某一行排序代码造成；需要沿 request ID 对照生产者优先、READY 预留、正常 waiter 的等待与实际缓存资格。

去 MTP 在同 1362 条中把 overall 45→19、fast 39→20，但 turn 1→2，TPOT>100ms 70→104，**TPOT p95 100.99→127.20 ms**。整个 30 分钟窗口与稳态子集是不同样本；不能只引用稳态 p95 86ms，就把开场代价排除。该窗口不判正式 FAIL，也不足以保证完整档会通过。对“chain 优先”的目标，去 MTP 当前更像稳态容量交换臂，还不是已经证实的 chain 攻坚臂。

## 4. 125 护栏有明确的工程改进点，但不能包办解释正式失败

CodeGraph 追到 `_ax_admission_plan → note_tpot → decide`，并核对 f546934e、e464d8ab、6976639e 的实际函数；这三版的相关逻辑一致：

1. 每次计划 prefill 时，对还在 running 的请求用 `(now - first_token)/(produced - 1)` 计算运行中的 TPOT。
2. 曾经超过 gate 的 RID 永久留在 `slow` 集合，即使最后完成时已恢复正常。
3. 达到 MAX_SLOW 后 `tripped` 锁存到下一次 flush；flush 确实重置它，未发现跨档遗留计数。
4. 日志只在 relief on/off 变化时打印 slow/seen，**如果本来就 off，后来触发 tripped，不会单独记录原因或触发时刻**。

对应源码：[状态定义与锁存](../../engine/sglang/srt/managers/ax_deadline.py:261)、[调度采样及日志](../../engine/sglang/srt/managers/scheduler.py:1652)、[flush](../../engine/sglang/srt/managers/scheduler.py:1530)。

直接从服务日志与 raw 起点对齐：

| 本地臂 | MAX_SLOW | 最后一次 relief off | relief_rounds 此后不再增加的首个采样 | 最终 TPOT>100ms |
|---|---:|---:|---:|---:|
| 130ez1，S3 | 80 | 175.68 秒，slow=73/86 | 181.68 秒；直到约第 60 分钟仍不增加 | 79 / 3219 |
| 130ez9，仅改护栏 | 250 | 1225.46 秒，slow=249/1058 | 1254.46 秒 | 78 / 1564 |

这些日志证明，两个阈值下实际服务行为并不相同，不能因为最终慢请求少就说“护栏基本没有作用”。`ever_slow=249` 和最终慢请求 78 也明确不是同一个量。锁存原因及精确时刻缺日志，不能把“约三分钟后 relief 不再执行”直接写成“第 181.68 秒发生锁存”，更不能外推官方也在这一时刻锁存。

同时，MAX_SLOW=250 在共同 1564 条中 chain 16→16（修 1、新 1）、turn 2→2、overall 42→38、fast 46→55。**简单把 80 放到 250 不是已经测出来的解法。** 值得做的是将“曾经慢”“当前在飞风险”“已完成慢请求”分开计数，给 guard 的触发与恢复补观测，再评价有滞回的恢复策略；不能为追 chain 直接删除 TPOT 护栏。这一领域由 Fable 协调，本文没有修改它。

## 5. 本地负载与产能推断需要收紧

本地 v3 的 311 条链首确实来自公开原始请求，但这不等于整场负载已校准。之前已核实公开 cohort 与 v3 的前 34 条只有五条重合；同一批家族共存、缓存可复用时刻和新链到达时刻都可能改变收益。

本轮读取 manifest 和元数据：v3 的 `uncached_expected` 总和 24,127,301，公开来源整链的标注总预算 30,912,302，比例 78.05%；合成新增量 12,151,265 对目标 18,936,266。manifest 自己标为 `DIAGNOSTIC_CANDIDATE_NOT_REPRESENTATIVE`。**这些是构造预算，不能直接称线上实际工作少了 22%，也不能据此按比例换算 N。** 同样，TPM 较高只表示逻辑负载需求，不能证明本地在 chain 关键维度上一定比线上更难。

还须撤掉三个过强解释：

- 通过档 turn 很低，不能证明正式失败档不撞 KV 墙；没有官方资源时间序列。
- 只有 TPOT 均值和不同 N 的摘要，不能证明线上 GPU/互联更慢，更不能按 TPOT 比例放大 prefill 耗时。
- 单请求 26.5k tok/s 与混跑差 1.85 倍的说法已被目标/草稿重复分母纠正。可复核的单请求输入速率约 13.15k tok/s；不能继续围绕不存在的差距投入开发。见[计时审计](leaderboard-prefill-audit-0927.md)。

## 6. 接下来的优先级

1. **先固定两个参照。** 正式最好配置 S2 与新执行路径 S4 都保留。下一版必须报告对 S4 的新增 chain 修复与新增坏例，避免继续从 S3 的 16 条起算。
2. **把正式退化拆开验证。** 同一个含 DCP 的引擎固定 MAX_SLOW、running/graph、KDA 槽、warm 策略、输出合同，仅切 W1/W2，先在 N26/N30 的同负载上对照。关注开场和整场后续 chain，尤其 turn 的尾部；不直接从本地 N34/N38 宣称线上可过。
3. **执行层攻坚选可泛化的 prefill 时间。** 正确按物理输入 batch 配对 target/draft，测块长 × 上下文 × batch 的成本；分清 DSA/indexer、MoE、通信和小块固定成本，优先改关键路径中可消除的时间。验收是相同未命中工作、相同上下文的毫秒下降，再看跨 cohort 的 chain 净改善。开场已连续 prefill 的时段，继续放松 decode interval 没有额外时间可换。
4. **调度侧保住代价边界。** 给 125 补明确的 guard 状态；128p 逐个查新增临界 chain 与暖请求等待，约束生产者推进对非家族请求的成本。只用在线可见的年龄、剩余工作、就绪状态，不按固定 ID 调参。
5. **校准数据后再谈档位。** 由数据维护者修正生成偏差并冻结新负载，同时保留旧集作机制回归。增加预先固定的开场组成对照，验证共享前缀收益能否泛化；新负载本身也不能等同隐藏正式集。候选最后仍须完整回放、原 harness 与十一门共同判定。

以上是工作建议，没有发布、入队或修改正在运行的任务。下一步最值得投入的是**相对 S4 真正新增的 prefill 进度与 chain 净收益**，而不是再证明 DCP 扩了池，或重复展示已有的五条收益。

## 复算与证据

在本分支执行：

```bash
python3 scripts/analysis/official_0927_retrospective.py \
  --repo-root /workspace/Agentic_science_challenge \
  --out evidence/official-0927-retrospective
```

[分析脚本](../../scripts/analysis/official_0927_retrospective.py)、[配置差异、原始来源、计数与日志摘录](../../evidence/official-0927-retrospective/audit.json)、[逐请求修复/新增表](../../evidence/official-0927-retrospective/paired-tail.csv)、[链首与开场核验](chain-opening-source-audit-0927.md)。本轮脚本已在上述真实数据运行；原 harness 的分桶及分位数函数直接引用，未重写判分规则，未向只读数据目录写入文件。
