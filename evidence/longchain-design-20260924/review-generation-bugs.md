# 生成器与素材逻辑：独立 bug review

2026-09-24。范围：当前追加式生成器、素材盘点、polish及相关来源校验；不审GPU，不联网，不改源数据或旧冻结成品。新跨session事件核心尚未实现，不能宣称已review通过。跨session借query/完整片段本身不是bug，也不要求完整故事语义。

结论：发现3类可复现实现bug，已交主会话集中修复；本报告的CPU反例再次运行，3类主要反例已通过。没有证据说明旧96链/1718请求因此损坏。仍需在新事件核心完成后验证新路径。

## P1：polish可把损坏父产物重新冻结为“有效”数据（已修）

- 修前位置：`longchain.py:693–703`只检查generator/status及来源覆盖，不验证父manifest声明的artifact hashes；之后重新写出hash。修后入口为[verify_parent_artifacts](../../scripts/analysis/longchain.py:405)，在polish读材料前调用。
- 触发：一份完整、已冻结的fixture，仅把requests中的`max_output_i`从10改成777，不改父manifest或chains摘要，然后合法编辑一条seed query并polish。
- 修前证据：父数据checker判INVALID（requests hash及chain输出汇总不符），polish仍完成；新产物checker判VALID，真实请求budget为777，manifest的`actual_output_sum`仍为10。冻结边界失效且账本可互相矛盾。
- 修后：在写出新成品前明确拒绝`polish parent artifact hash/path mismatch: requests.jsonl`。不应以更新父hash方式绕过。

## P1：工具ID改名会改坏结构，且有级联/漏改分支（已修主要反例）

- 位置：[rename_new_calls](../../scripts/analysis/longchain.py:376)；修前逐个对所有字符串执行`.replace()`，不区分结构字段。
- 短ID反例：call ID为`a`，生成后`role=assistant`变成`new_0ssistnew_0nt`，function name `calculate`与正文`data`一起损坏。
- 级联反例：两个旧ID为`abcdefgh`及`new`，prefix=`new`，原本第一个应得到`new_0`，却被第二轮替换成`new_1_0`。
- 漏改反例：调用使用`tool_call_id`而没有`id`；配对检查支持这种输入，旧renamer却完全不改名，复用时可能保留外链ID。
- 修后：单次有边界替换、结构字段保护、支持alternate ID；三个反例均得到预期结果。源素材中最短ID为6字符，未发现alternate-ID调用，故没有把短ID反例推断为旧成品受污染。
- 余下设计边界：工具ID/实体名不是同一种引用映射；局部适配不能仅依靠全字符串替换。新的跨session连续片段应按完整工具组作用域分配ID。源722份历史中48份存在不同组复用ID，这是源格式现象；相邻候选追加suffix内未发现重复ID。不能把旧历史跨组重复一刀切当损坏。

## P2：第二次polish会被checker错误拒绝（已修）

- 生成侧将`adapted_original.source_body_sha256/source_frozen_labels`保留为最初源body/标签，这符合来源账本；修前checker却总与直接parent的改写后body/标签比较。
- 位置：[来源校验](../../scripts/analysis/longchain_check.py:547)，生成侧[来源保留](../../scripts/analysis/longchain.py:780)。
- 反例：原fixture VALID→第一次seed query polish VALID→第二次seed query polish INVALID，错误为source body hash与frozen labels不匹配。与真实token计数无关，是祖先/直接parent含义不一致。
- 修后：如果parent已经是adapted_original，则比较其最初source字段；三代fixture均VALID。`chain_record`已解包nested source targets，没有发现逐次嵌套漂移，不能把该处误报成bug。

## 后续修复的独立复核

- 精确结构路径保护已复核：当旧ID恰好为`assistant`时，消息role与tool function.name保持`assistant`，但arguments里的name/type/role字段及结果content.name中的同值引用会改成`scoped_0`。不再全局跳过名为name/type/role的业务字段。
- 增加2源链/4公开请求的独立build fixture，两条源链故意共用同一个source session；生成后为2链/8请求。每链原样行和合成行都统一到自己的`lc_123_000`或`lc_123_001`，provenance完整保留`shared-source-session`。
- 第一合成请求排序offset从最后原请求的end=100加gap=5得到105；后一合成请求为110，合成end保持未知，provenance明确`synthetic_order_only_not_observed_time`，真实pacing仍取独立gap字段。
- 使用原`s1_common.load_index/freeze_cohort`接受8/8请求；独立checker在CharacterRenderer fixture下VALID、0错误。该结果证明身份与排序改动兼容原索引/冻结流程，不是实际GLM渲染或负载分布验收。
- 本轮再次重跑三个rename反例、两次polish与篡改父文件拒绝路径，结果均维持正确；最终结果收据包含`structure_path_rename`和`identity_time_build`。

## 素材容量及非bug项

- 独立重新运行素材盘点，得到722份body、311源链、136个(pack,session)、源摘要5601请求；433个非reminder user候选、371个无调用assistant、2381个完整工具组、69899个同pack跨session query×chain候选，与文档一致。
- 工具组精确JSON去重包含原ID，因此不等于2381种语义独立行为；user还可能含控制消息，433不是已核实真人问题数。现行文档已披露，计数没有发现bug。
- 原harness不执行工具、不会把当前模型回答回填下一冻结prompt，因此借其他session素材并不破坏teacher-forced回放契约。生成后必须重新渲染token/LCP与输出预算；任意改字不是“分数保证相同”。
- 历史编辑按整条消息hash传播、精确消息消失即停止；目前相关测试未发现原历史被无意逐轮重写。输出预算保留原请求，源aggregate矛盾明确记录而不压原输出。
- 新版缺用户事件计划、历史重建/重建后增长和全局新增重复审计，属于明示的实现缺口；不能用旧追加生成器的CPU通过代替新版验收。

## 复现与验证边界

运行：`python3 evidence/longchain-design-20260924/repro-generation-bugs.py`。修前对照：追加`--baseline`，只读取旧冻结源码快照。

[可执行脚本](repro-generation-bugs.py)、[修前结果](repro-generation-bugs-baseline.json)、[修后结果](repro-generation-bugs.json)。所有输入创建于临时目录；使用明确的CharacterRenderer测试double，只验证ID、来源及冻结控制流，不是GLM tokenizer验收，不是GPU结果。旧冻结目录未被写入。
