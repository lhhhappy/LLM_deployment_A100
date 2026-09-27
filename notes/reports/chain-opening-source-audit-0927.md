# 开场 chain：真实负载不等于不可避免的超时

2026-09-27，Codex。状态：公开数据、harness 源码与本地原始记录的独立核验；不是正式档位预测。没有运行新的 GPU 实验。仅形成只读分析，未修改引擎或发压数据。

**结论：长冷提示与空缓存开场确实是正式赛题要解决的问题；“必然有 15–16 条开场超时”“N38 的调度已到物理极限”没有得到证明。** v3 保留了公开链首，但重排了 cohort、合成了后续请求。它可以比较机制，不能据此给正式 N34/N38 的通过承诺。

## 数据与规则核验

- [正式题面](/workspace/Agentic_science_challenge/llm-challenge-arena-v1/task.md:555) 明确 chain_start 包括链首或上下文重建，prompt 常达十万 token、结构性零缓存命中，目标 30 秒；[flush 约束](/workspace/Agentic_science_challenge/llm-challenge-arena-v1/task.md:228) 要求每档测量前真清缓存。因此这种工作量并非我们合成出来的特殊题目。
- 用原 [load_index / phase_gate](/workspace/Agentic_science_challenge/s1-dev/harness/s1_common.py:107) 读取公开 722 请求 / 311 链与 v3 的 5601 请求 / 311 链。311 个链首 ID、token 数、输出预算、gap、phase、edge 和其余来源字段全部一致；仅 `session_id` 与 `body_ref` 改变。
- 流式读取公开数据的全部 722 份正文，重新计算 canonical JSON SHA256；全部同时匹配 v3 provenance 的 `source_body_sha256` 和 `body_sha256`。这是原正文对构建来源记录的核验；**本次没有重新读取 Pod 上实际部署的 v3 正文分片**。不能把来源账本一致写成现场逐字节验收。
- 公开链首 token：中位 35,654，p75 60,848，p90 102,171，最大 256,733；≥65,536 有 69 条，≥100,000 有 33 条，≥200,000 有 9 条。若阈值写 64,000，数量是 73，避免把“64k”的两种口径混用。
- 首轮 phase 为 session_start 的 104 条；其他 207 条分为 intra 115、turn_start 72、context_reset 20。148 个链首标记 system-tools-changed。这说明抽样链首会携带会话中段历史；标签不证明与其他在场请求完全没有可复用前缀。

## 本地与线上之间不能省略的边界

1. **开场组成已经改变。** 主办方公开 cohort 与 v3 cohort 的前 34 条链只有 **5 条重合**。两份开场 prompt 总量分别 2,017,921 与 1,627,244 token；这是逻辑 prompt 总量，不是实际 prefill 工作量，不能据此直接比较难度。二者的家族组成也不同。
2. **并非 34 个请求严格同时到达。** 原 [worker](/workspace/Agentic_science_challenge/s1-dev/harness/s1_loadgen.py:580) 开 N 个会话槽；[drive](/workspace/Agentic_science_challenge/s1-dev/harness/s1_loadgen.py:334) 保留每条请求自身的 gap、渲染与发送时间。v3 前 34 条中一条有 62,202 ms 的初始 gap；8 份 raw 均是相同的 33 个链首在首 30 秒到达，另一个在约 62 秒到达。正式 cohort 的到达细节未知。
3. **`split=hidden` 不证明当前正式集成员关系。** 178 是公开元数据中的来源标签计数。当前正式完整 cohort 没有提供，不能由此断言“至少 178 个链首就是本次线上请求”。题面及公开 README 明确开发集只用于自身 A/B，不能产生正式 N@SLO。
4. **chain 门样本量也不等于链数。** 原 harness 将非链首 context_reset 同样计入 chain 门：公开集是 311 链、314 个 chain 门请求；v3 是 311 链、432 个 chain 门请求。整轮统计余量按真实桶样本数计算，不能只凭会话链数锁定允许超时条数。
5. 线上 46677 的 chain p95 39.13 秒支持“chain 尾延迟需要优化”；平台未公开失败档逐请求时间，不能定位它们是否都发生在开场，更不能证明与本地是同一批大冷首轮。

## 15–16 条不是执行时间的硬下限

重算 [130ez1 raw](/workspace/Agentic_science_challenge/evidence/L130ez1-v3_n34_S1dcp_w2_60m/N34/raw_s1-dev-longchain-v3_N34_1790470437.jsonl)：首 30 秒到达的 33 个链首中，16 个 TTFT >30 秒。这 16 个请求从首次入批到首 token，**最长仅 13.649 秒**；入批前时间之和占它们 TTFT 总和的 **90.99%**。

代表例子：157,628-token 请求首字约 96.39 秒，但首次入批到首字约 13.65 秒；不能写成“它单独算也超过 30 秒”。另一方面，这次约第 3281 秒到达的 256,733-token 请求 TTFT 33.975 秒，其中首次入批到首字 33.373 秒，确实存在执行阶段本身超过目标的另一类坏例。两类应分别处理。

通过 CodeGraph 追到 `set_time_batch` → `ReqTimeStats.set_forward_entry_time`，并读 [真实 setter](/workspace/Agentic_science_challenge/engine/sglang/srt/observability/req_time_stats.py:759) 与 [harness 字段提取](/workspace/Agentic_science_challenge/s1-dev/harness/s1_loadgen.py:44)：该字段只记录首次入批，不会每块覆盖。后半段仍含分块调度、穿插 decode 和处理开销，**不是独占 GPU kernel 时间**；前半段也含其他请求占用通道，91% 等待不等于全部可由重排消除。

## 新合并探针的结果

所有行核对原始 ID 唯一性、错误字段、时间戳关系、相同 cohort、flush_success 与 runner_rc。相同的初始 34 个链首全部出现一次，其中首 30 秒的集合恰好相同，均为 33 个。

| 运行 | 改动 | 开场 chain 超时 / 33 |
|---|---|---:|
| 130ez4 / 130ez6 | 本地续算 OFF-a / OFF-b | 均为 16 |
| 130ez5 | 本地续算 ON | 11 |
| 130ez6zz | 128p OFF，类别冻结 ON | 16 |
| 130ez6zzz | 128p ON | 11 |
| 130ez6zzzz | 128p + 本地续算 ON | 11 |

以 130ez6zz 为共同参照：本地续算修复 6、新增 1；128p 修复 7、新增 2；合并同样修复 7、新增 2，且与 128p 的修复/新增 ID 集完全一致。本地续算修复的 6 条是 128p 修复集合的子集。**本次组合没有额外减少 chain 超时。** 这不证明本地续算在其他请求形状或负载下没有价值。

[合并 job 收据](/workspace/Agentic_science_challenge/evidence/L130ez6zzzz-v3_open_S1dcp_combo_on_n34/N34/job.log) 记录 600 秒派发后 DRAINED 505 请求、冒烟 12/12、机制核对通过。它是开场机制诊断；完整档 verdict 的 missing cohort 不能被抹掉或改写为 N34 PASS。

## 对攻坚方向的影响

- **先处理边缘坏例的取舍。** 128p 新增的两个 chain 坏例，OFF → 合并分别约 23.98→31.80 秒、28.46→36.30 秒；都是被推迟的可完成请求。下一步应用实测块成本，检查生产者优先与 READY 预留是否会让原本可达标的独立请求错过期限。目标是保留家族净收益同时减少这两条新增，而不是简单延长所有依赖等待或给家族无限优先。不能据这两个 ID 特调；只使用线上可观察的工作量、可用缓存、年龄、依赖与剩余预算。
- **prefill 提速仍值得攻坚，但先拿现行 DCP2 时间分解。** 130ezc 的脚本只测一个约 49k 冷提示的 8k/16k 块；它能分解该形状，却覆盖不了 100k–250k 上下文，也没有模拟多个链首与 decode 混跑。选择 DSA indexer、DCP gather、MoE 或 host/graph 优化之前，要确认它们在相关形状的关键路径占比。过去每卡 26 TFLOPS / “8% 利用率”是推导量，不能当作可获得 12 倍加速的证明。
- **不要把调度余量预先判为零。** “固定成本按最短先做，只差 1 条”没有建模前缀生产依赖、可复用成本随顺序变化、分块检查点与后续请求穿插；128p 同一开场从 16→11 本身已说明那个估算不是严格下界。N38 需要新证据，既不能承诺能过，也不能据此证明纯调度一定没用。
- **防止固定开场过拟合。** 机制固定后，优先使用主办方公开 cohort 原样作外部于 v3 顺序的回归；若设计多个盲选开场敏感性诊断，须保留完整正文/输出/自然 gap，另标“诊断”，不能替代原 harness 的正式完整档。N34/N38 的判断最终依赖正式或完整同负载结果，不把重复的 33 条链首当作总体保证。

## 复现

在本 worktree 执行：

```bash
python3 scripts/analysis/chain_opening_source_audit.py \
  --repo-root /workspace/Agentic_science_challenge \
  --out evidence/chain-opening-source-audit-20260927
```

[复算脚本](../../scripts/analysis/chain_opening_source_audit.py)、[机器可读结果](../../evidence/chain-opening-source-audit-20260927/audit.json)、[逐请求首次入批与首字时间](../../evidence/chain-opening-source-audit-20260927/initial34.csv)。原始 raw 保留在主证据 checkout；本报告没有复制、归档或清理运行证据。
