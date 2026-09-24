# 长链源数据的可恢复边界与素材复用

现行口径见[统一设计](../scripts/analysis/longchain.md)：允许A session的query/完整片段改写后续接B，构造合成轨迹；无需恢复原始故事。下文的“无法恢复”限定真实来源证据，不限制合成数量。正文可以组合/局部生成，时间与预算可明确估计，测量必须真实。

审查日期：2026-09-23。只读检查 `s1-dev/data/dev-combined-v1/`、旧候选（现归档至 `cache/longchain-legacy/longchain-screen/`）、harness、生成器和仓库公开数据索引；未改数据/生成器，未访问外部平台、GPU、队列或提交。以下源统计直接来自公开文件；重新计算项是推断/一致性检查，不代表恢复了隐藏原始轨迹。

## 同链历史的可恢复长度

`requests.jsonl` 每行对应一个可见真实 request ID 和一份完整 body；`chains.jsonl.n_requests` 是全源链的聚合轮数，不能据此生成缺失 request ID。对每链按 dispatch 排序取最后 body，统计消息对象及严格连续 assistant tool-call→tool-result 组：assistant 的每个 call ID 均须在后续连续 tool 消息中恰好匹配一次。工具组不是 LLM 请求数。

| 口径（311条源链） | 数量/分布 | 含义 |
|---|---:|---|
| 源 metadata `n_requests > 13` | 93链；p50=8、p90=45、max=240 | 源总轮数声明；隐藏轮没有逐轮 ID/body。 |
| 有真实 request ID/body 的可见轮数 >13 | 0链；p50=1、p90=5、max=13 | 可直接按原 harness 回放的位置；没有一条公开链能仅靠可见真实请求超过13轮。 |
| 最后可见 body 的 assistant 消息数 >13 | 67链；p50=3、p90=81、max=199 | 历史消息对象数，不等于不同请求数。 |
| 最后可见 body 中完整工具组 >13 | 56链；p50=2、p90=58、max=197 | 历史工具组数，不等于模型调用轮数。311个末快照合计4,294个完整组，另有801个 assistant tool-call 消息组无法按连续结果消息完整配对。 |
| 严格连续快照去重 | 411个同链相邻快照转移中仅146个是前一 `messages` 数组的精确前缀；仅55链有至少一个这样的转移。最长连续前缀段6个快照；跨快照最长段为5个快照、372条消息（184条 assistant 消息、114个完整工具组）。 | 只在完整消息对象精确前缀时安全去重；不精确时可能是重建、压缩或改写，不能把差异拼成确定时间线。 |

因此“比13轮长”有三种答案，必须分开：源 metadata 声称93链长于13轮；可见真实 request 位置为0链；最后 body 含超过13条 assistant 历史消息的有67链。后者包含多次工具交互/内部消息，不能转成 request 数。末 body 有完整历史文本，仍没有历史每轮独立的真实 request 行。

## 链级聚合能约束什么

全源311链为722条可见 request/body，但 `n_requests` 合计5,601。源 `sum_glm_tokens` 合计485.06M、可见 request prompt 合计34.42M；源 `sum_uncached_expected` 合计30.91M、可见合计11.98M。差额只有链级总数，没有逐轮 body。对缺失轮可计算请求数、prompt 总量和 expected-uncached 总量残差；prompt 总量减 uncached 总量还可给累计 LCP 约束。它们无法推出任一轮的长度、LCP、工具行为或时间分配，也不代表设备实际 cache miss。

`max_output_i_sum` 是输出上限的聚合，不是实际 completion token；真实 `source_completion_tokens` 仅见于可见请求。全源隐藏轮没有 completion 长度、每轮输出 cap 或 response 时长可供重建。不可为命中总预算拆工具块、裁文本或编零输出轮；聚合目标冲突就报告残差。

## 96链成品的联合分布检查

旧候选 `longchain-screen` 有1,718行：247条保留的真实可见请求、1,471条 synthetic。所选源 aggregate 的 prompt 总量147.876M，输出148.081M（差0.205M，约0.14%）；这只是 prompt **总和**相近。uncached 三本账如下：

| 口径 | 总量 | 解释 |
|---|---:|---|
| 所选源完整链 `sum_uncached_expected` | 8.970M | 全源聚合目标。 |
| 输出冻结 `uncached_expected` | 6.008M，其中 synthetic 1.935M | 源可见行4.074M加生成增量；相对源缺失余量4.896M仍差2.961M。 |
| cohort 每链首轮冷入场估计 | 约7.600M | 比冻结总量高1.592M（18个 head 有正 uplift）；这是冷 head 推算，不是运行时 GPU miss。 |

源 aggregate 缺失余量不能全解释为“应追加的未来轮”：17/96链的可见首 request 晚于 source `first_dispatch_offset_ms`，aggregate 还包括不可见的前置历史。若保留这些链，必须单列前缀缺失；不能把冻结字段低60%直接说成 GPU prefill 少60%。

edge 摘要也有入边口径。所选源 `total_edges=1,690`，比输出内部边数1,622多68；这68条正好对应 source 首行 `gap_valid=true` 的外部入边，其中17条是 append-only。剔除外部入边后，源内部目标 append 边1,544；输出内部实际 append 边1,613（99.4%），因为生成的1,471条新边全是 append。按源汇总减可见内部边，约有69条缺失边应为非 append；当前 append-only donor 延展不能恢复其具体 reset/断点位置。

我审查早期构建时确认过 `append_only_edges` 曾原样复制 source aggregate、`total_edges` 却按输出重算，二者不可混列。当前共享的 `cache/longchain-legacy/longchain-screen/chains.jsonl` 已重算输出内部 `append_only_edges`、`total_edges`、fraction（单请求链fraction为null）、prompt/uncached/cap sum、phase、当前首 dispatch；source 原摘要留在 `source_chain_targets`。source `first/last` offset 只描述原链。可见的真实相邻对中，411对有404对满足 `next.dispatch - previous.end == replay_gap_ms`；harness 的 gap 是前一请求完成后的等待，不是纯 dispatch 间隔。synthetic 前一轮没有观测 end，因此从 donor gap 推出的 synthetic dispatch offset只能排序，不能当真实时间戳。

源首末跨度是可用的“长链”筛选条件，不是可拆的逐轮时间表。所选96源链 `last_end-first_dispatch` 的中位数约3.1分钟、均值33.5分钟；仅4链≥2小时、3链≥3小时，最长26.4小时。链跨度、gap 总和和并发 replay 墙钟时间是不同量；现有合成 gap 不足以证明某条链是真实2–3小时。

## 公开数据查找与最小改进

仓库 `s1-dev/data/` 只有 `dev-combined-v1` 完整正文集；赛规说明正式压测集不公开。`evidence/official/all_att_2026-09-23.json` 的551条 attempt 元数据均无 raw message path、bundle、manifest、exec log 或 results 文件。`evidence/` 的 raw 是本地 harness 运行记录，不是新增的原始会话正文。没有发现第二套公开续轮语料。

现行改进：公开历史作为素材库，跨session借query与兼容工具片段是正常路径。保留B的system/tools和已有历史，借入引用适配到B，工具组闭合，事件由接收计划产生；不恢复未知原始顺序，不给每轮强加用户追问。局部正文可改写，gap/输出缺少直接关联时标明估计，不伪称观测。累计目标用于报告偏差，不为了对齐数字裁历史/压预算或加随机盐。

722份body中全局精确JSON去重得到433条非reminder user候选、371条纯assistant消息、2,381个可配对工具组，见[素材盘点](../evidence/longchain-design-20260924/material-inventory.json)。这些不是已验收query或独立轨迹；以311链/约5,601调用作为首个合成规模目标是计划，不能声称恢复源中缺失4,879次请求。跨度和工具等待用于行为参考，不能保证回放墙钟时长。
