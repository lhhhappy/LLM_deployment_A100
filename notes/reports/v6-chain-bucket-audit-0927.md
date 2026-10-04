# v6 补小头提案：先区分 phase 与正式分桶

2026-09-27，Codex。审阅 Fable 的 v6 数据提案；状态：**Fable 已撤回补 207 个小头的任务**；公开数据和原 harness 合同已复算，正式隐藏集的逐请求分桶未知。本文保留合同核对和剩余偏差，不是待实施的生成计划。

**不能将“给 207 条切段链补小头”称为与正式一致的数据修复。** 主办方原 harness 本来就把每条回放链的第一个请求按 chain 判，优先于 `phase`。公开 722 行中 `phase=intra` 不意味着它一定走 5s 门。原本是用户业务会话的中段，也可以是评测回放链的第一条。

代码依据：`s1-dev/harness/s1_common.py:107` 的 `phase_gate`，先检查 `_idx_in_chain` / `idx_in_chain == 0`，随后检查 turn/reset；同文件 `load_index:153` 按原 `chain_id` 分组、按 dispatch 排序并分配 `_idx_in_chain`。这不是我们长链生成器另设的规则。`task.md:563` 也写明 chain 门覆盖“该链第一条请求，或上下文重建”，并说明 prompt 常达十万 token。`task.md:354` 说明公开开发集是链前缀抽样，不能据它预测正式 N。

直接在未修改的 `s1-dev/data/dev-combined-v1/requests.jsonl` 上调用原函数，结果如下：

| 原始 phase | 原 harness 实际桶 | 条数 | 未缓存 ≥50k | 未缓存 ≥100k |
|---|---|---:|---:|---:|
| session_start | chain | 104 | 1 | 0 |
| intra | chain | 115 | 49 | 17 |
| intra | overall-intra | 388 | 4 | 1 |
| turn_start | chain | 72 | 0 | 0 |
| turn_start | turn | 20 | 0 | 0 |
| context_reset | chain | 23 | 0 | 0 |

实际 chain 桶为 **314 条 = 311 个 idx=0 + 3 个非链首 reset**；20 个位于 idx=0 的 reset 不能重复计数。真正进入 overall-intra 的 388 条中，≥50k 为 4/388（1.03%），≥100k 为 1/388（0.26%），最大未缓存 103,207。把全部 503 个 phase=intra 合在一起算的 10.5%/3.6%，不能当成 5s 桶的长尾比例。

因此，目前能确认：长历史切段头存在、我们的本地 chain 超时集中于这些头；**不能确认**正式隐藏集把这些头恢复成业务会话中段，也不能确认正式 chain 超时只发生在小头/reset。仅按 phase 重判本地运行得到“chain 零超时”，不等于原 harness 的正式口径。

进一步逐链比对公开首行的 `dispatch_offset_ms` 与主办方 `chains.jsonl.first_dispatch_offset_ms`：

| 源链首是否公开 | 链数 | 相位 | prompt p50 / p95 / max | ≥100k |
|---|---:|---|---|---:|
| 是 | 258 | session_start 104 / intra 82 / turn_start 72 | 34,998 / 121,825 / 256,733 | 17 |
| 否 | 53 | intra 33 / reset 20 | 87,331 / 229,576 / 252,115 | 16 |

258 条真实源链首的边类型为 chain-head 104、system-tools-changed 148、compact-rebuild 6，其中 252 条的冻结未缓存量等于整个 prompt。`chain_index` 是链元数据，不能当作当前链内请求序号：例如 `...5f5b5fc16f0e43f5b26c6:109` 的 chain_index=109，但声明整链只有 3 个请求。

原官方 cohort 还明确把这个 256,733-token 请求列为 `head_uncached=256733`、`chain_head_phase=intra`、`prefix_len=1`；其 `gate_p95_alignment.chain_start.n_sample=314`，与原 harness 复算相同。这足以推翻“intra 标签的巨头必然不是主办方链首”的前提；该 cohort 的旧参考人口统计不用于推断当前隐藏集的精确分布。

**仍有真实的校准偏差：53 条源链首未公开的链。** 它们的公开首行都是 append-only，冻结未缓存量不等于整个 prompt，但空缓存回放时可能重算长历史。需要单列这 53 条影响，不能把它扩大为 207 条。按公开头逐条算，53 条中 ≥100k 为 **16 条**；“31 条”是另一分组口径，不能放在这一分母下。旧生成器沿用这些公开首行，不应描述成已经知道并重建了缺失的真实源链首。

原提案还会同时改变以下变量：

1. 新增 207 个请求及其输出/等待，改变链槽释放和所有后继到达时间。
2. 给原冷段首制造一个此前不存在的缓存生产者，使其少算一段前缀。原 chain 中有 system/tools 变更边，补算的是变更后的新版本前缀，不能假定源业务已经算过它。
3. 如果将全部原段首改为 intra，会一并改掉已知的 72 个 turn 和 20 个 reset 标签。保留这两类标签时，桶计数又不等于全部改 intra 的方案。
4. 按消息边界截取后，GLM prompt 末尾的 assistant generation prefix 与下一份完整历史未必逐 token 相同；LCP 必须真实渲染测量，不能写成“新头长度”。完整 system/tools 加第一条用户消息也不一定满足所抽的 18k–35k 长度，不能为匹配分布截掉既有字段。

**修订方向：** 现有冻结集继续作为带明确偏差的对照。若要验证“巨型 intra 占道时，小头和 reset 能否获救”，单独创建机制压力集，保留公开正文/原 phase，明确标出人为安排的到达关系、缓存初始状态与新增请求；用多个长度和到达偏移，不能宣称其频率已经校准线上。正式校准应优先对齐正式提交的配置、启动和评分合同，获取隐藏运行逐请求或至少按阶段分组的证据后再修正人口结构。

这是对 v6 校准前提的否定，不是否定独立压力场景的价值。本轮完成的是合同审计；未生成或发布“已校准 v6”，未替换冻结数据。没有用错误分桶结果选择新的 chain-max 候选。

复算命令（只读）：

```bash
python3 -B scripts/analysis/review_chainmax_bucket_contract.py \
  --source /workspace/Agentic_science_challenge/s1-dev/data/dev-combined-v1 \
  --harness /workspace/Agentic_science_challenge/s1-dev/harness \
  --out evidence/chainmax-review-0927/bucket_contract.json
```

[完整 722 行映射和输入收据](../../evidence/chainmax-review-0927/bucket_contract.json)，[脚本](../../scripts/analysis/review_chainmax_bucket_contract.py)，[chain-max 代码审查](chainmax-review-0927.md)。
