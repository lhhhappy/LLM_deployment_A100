# 榜单指向并发档位；prefill 产能账需要区分模型调用与输入块

2026-09-27，Codex。榜单来源为用户提供的 11 行快照，未另行刷新榜单；自己的正式成绩引用 46677。另通过只读 `pread` 核对 130ezc 的既有 TP8 剖析收据，没有运行 GPU 测试或修改队列、引擎。

## 榜单能支持的判断

| 组别 | N@SLO | TPOT 均值 |
|---|---:|---:|
| 我们 | 26 | 25.10 ms |
| 其他 N26 中最快者 | 26 | 33.48 ms |
| N30 四份提交 | 30 | 46.20–60.17 ms |
| 第一名 | 34 | 50.34 ms |

同档 N26，我们比下一位的 TPOT 低 25.04%，说明当前已经有明确的同档解码延迟优势。跨档，第一名并发数比我们高 30.77%，TPOT 约是我们的 2.006 倍；但不能据此推断对方主动牺牲解码或用了某种调度、量化、DCP/MTP 配置。N 更高本身就可能改变 batch、驻留与延迟分布。

按 [正式排序规则](/workspace/Agentic_science_challenge/llm-challenge-arena-v1/task.md:597)，如果其他行不变、我们确实通过了对应档位：

| 假设我们达到 | 在这张快照中的位置 |
|---|---:|
| N26 / 20 ms | 仍第 6 |
| N30 / 45 ms | 第 2 |
| N30 / 50 ms | 第 5 |
| N34 / 50 ms | 第 1 |
| N34 / 60 ms | 第 2 |

这只是条件排序，不是性能预测。研发应先让完整档位通过，再优化同档 TPOT。N30 的 45 ms、N34 的 50 ms 可作为竞争目标参考，不能直接作为调度器阈值。

我们 TPM(all) 比第一名高 9.06%，TPM(decode) 只低 1.27%；CalvinCao 的两项 TPM 都比我们低，却通过 N30、排名在我们之前。TPM 包含缓存输入，且由固定稳态窗口内的回放需求决定，不是独占执行能力测量；也不参与正式排序。不能把更高 TPM 当作能晋档的证据。

能力分方面，我们两门均已通过；榜单中第一名两科都比我们少约一道正确题，但排名更高。不由这几个样本的分数差推断精度、输出长度或实现优劣，也不以降低能力为优化方向。

## TPOT 的可用余量应如何判断

46677 的线上 TPOT 是均值约 25.1 ms、p95 约 47.5 ms。解码门约束的是 **p95 ≤100 ms**，不能用 100/25 声称有四倍计算时间可以任意分配。并发提高会改变 p95，开场大块阻塞也可能只影响少数请求而不显著提高均值。

已有局部反例：130ez7 的同 ID 30 分钟诊断中，interval 2→1 只让均值从约 55.5→56.6 ms，但 p95 从约 98→112 ms，chain 不变。该窗口不是正式完整档成绩，却说明“均值只涨 1 ms”不足以证明代价很小。每次交换要同时报告同 ID chain 修复/新增、turn/fast/overall、完整测量范围内的 TPOT p95，而不只报平均值。

## 独立发现：130ezc 的 26.5k tok/s 推断使用了错误分母

待复核的现有解释是“负载下一个 8k 块约 0.57 秒，单请求只需 0.309 秒，因此混跑慢 1.85 倍”。我直接读取 Pod 上既有 `tp8_8k/ledger-rank0-rank1.json` 与 `receipt.json`，得到：

| 原始量 | TP0 数值 |
|---|---:|
| 请求实际输入、实际未命中 | 49,143 token，cached=0 |
| ≥8k 的 EXTEND 标记 | 10 次 × 8192 token，均值 309.411 ms |
| 尾块 EXTEND 标记 | 2 次 × 8183 token，均值 309.035 ms |
| EXTEND 标记 token 总和 | 98,286，恰好是输入的 2 倍 |
| EXTEND GPU span 总和 | 3.712184 s |
| GPU 标记覆盖窗口 | 3.749078 s |
| 首次入批 → prefill 完成 | 3.737644 s |
| 收到请求 → prefill 完成 | 3.839548 s |

单请求真实输入速率按 GPU 窗口是 **49,143 / 3.749078 ≈13,108 token/s**；按首次入批到 prefill 完成是 **≈13,148 token/s**。按 8k 对输入分块是六块，EXTEND 总 span 折合每输入块约 **618.7 ms**，而不是 309 ms。

CodeGraph 的 `build_step_span_name` callers 指向 `ModelRunner.forward`；[标记实现](/workspace/Agentic_science_challenge/engine/sglang/srt/utils/profile_utils.py:463) 使用一次模型 forward 的 `extend_num_tokens`，没有附带 target/draft 角色。[Eagle prefill 路径](/workspace/Agentic_science_challenge/engine/sglang/srt/speculative/eagle_worker_v2.py:1210) 先执行 target，再执行 draft；两者都经模型 runner 产生 EXTEND 标记。因此 `prof_ledger.py` 的 `mean_new_toks / mean_gpu_span` 不能不经角色与 batch 配对就当作请求输入速率。

**据这些记录，“单请求 26.5k 对混跑 12.8k，差 1.85 倍”不能成立。** 更正不是证明混跑没有额外成本：这里单请求上下文仅约 49k，带 profiler；混跑有不同上下文、decode、HiCache 和 batch。它们仍需相同口径的测量。也不能把两次模型调用理解为 MTP 开销占 50%——标记数量不是时间比例，必须在 trace 上区分实际 target 与 draft。

已保留 [原始 receipt 的只读副本](../../evidence/leaderboard-prefill-audit-20260927/tp8-8k-receipt.json)、[ledger 精确摘录与复算量](../../evidence/leaderboard-prefill-audit-20260927/tp8-8k-ledger-extract.json)。原 trace 和完整 ledger 仍在原 run 下，没有移动或清理。

## 对下一步的建议

1. 继续以 chain 与完整 N30/N34 晋档为主目标，不为进一步缩短 N26 的平均 TPOT 单开攻坚。达到 N34 已有其他提交证明在赛题上可行；这不证明我们的具体配置已足够。
2. 负载下 130ezd5 剖析仍有价值，但先按一个真实 prefill batch 归并 target/draft，并计入该 batch 间的 decode/host/通信时间；使用实际新输入 token 作分母。按关键路径时间决定优化 DSA、MoE、通信还是 host，不能按已撤回的 1.85 倍差距分配工程量。
3. 128p 在当前相同开场已从 16 降到 11；保留收益并检查两个新增边缘坏例的预算竞争。只使用线上可观察的缓存、剩余工作、年龄与依赖，不按榜单或固定请求 ID 调参。
4. 若增加 prefill 预算确实修复 chain，可以接受 TPOT 均值上涨，但必须观察 p95 与其他硬门。按请求压力动态分配预算比“先把 TPOT 人为拖到 50 ms”合理；现有几组全局旋钮对开场无收益，没有理由继续盲目放宽。

[用户榜单快照](../../evidence/leaderboard-prefill-audit-20260927/leaderboard-user-snapshot.json)、[chain 原始数据核验](chain-opening-source-audit-0927.md)。
