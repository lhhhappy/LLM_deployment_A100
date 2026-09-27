# 长链造数审查与增量修复

2026-09-27，Codex；分支 `codex/longchain-repair-0927`。用户要求直接修现有数据，不重新生成全套正文。
用户随后授权交给 Fable（Claude）review，通过后开测，并要求 Codex 持续监控、建议。未改引擎、原 harness 或冻结父数据。

结论：现有 v4/v5 可以复用正文、增量修订元数据。已经修复两个脚本的输入覆盖和发布契约问题，
完成 v5 对照、旧 v5g 压力参照、v5g-tail 主候选，已在 GPU 开发机及 Pod 上复用正文部署。真实等待只有汇总统计，不能恢复每个请求的间隔或位置；
补算工作量的偏差也不能通过改 `uncached_expected` 标签修好。
Fable 已独立复核接受三套发布及 tail 规则，后续同数据对照由他编排；当前状态只看共享[实验队列](/workspace/Agentic_science_challenge/notes/queue.md)。

## 1. 审查覆盖与调用图

已读主生成路径及其全部实际调用：`longchain.py build`、`EventCompiler`、`ReadCorpus`、
`rebudget.py`、`regap.py`、独立 checker 与 replay guard、原 harness 的 cohort、gap plan、drive。
未重新运行全量正文生成或 GLM token/LCP 渲染；未将未使用的旧 polish 分支计入完整审查。
按真实函数关系核对，当前环境没有可调用的 CodeGraph 服务：

```text
load_source → choose_chains → EventCompiler.plan
  → EventCompiler.event / ReadCorpus.lengthen
  → Renderer.render + tokenizer + LCP → 请求/来源/按链账本
  → s1_common.freeze_cohort → manifest

增量修复：现有 rows + 输出预算/显式 gap 补丁/指定 cohort
  → longchain_metadata.publish → 字段边界/父文件一致性检查
  → 复用 body shards + 更新 provenance/chain totals/manifest
  → 原 longchain_check / longchain_replay → 原 harness
```

源数据含 311 条链、722 个公开请求；合成部分为 4,879 个请求，完整回放共 5,601 个。
阶段数量按源链摘要补齐，位置随机；gap 从公开短前缀模板抽取，输出再按链分配总量；
v4 通过提前正文分歧位置补重算量。模板采样最初有输入/gap/输出关联，但后续缩放、分歧补量及 v5 重分配会改变关联。

## 2. 确认的工程 bug 与修复

| 问题 | 原代码复现 | 本分支修复 |
|---|---|---|
| 输出目录可覆盖输入 | 两请求临时样本：rebudget 先改写父 requests，再因 `SameFileError` 退出；regap 成功覆盖父集，连 parent hash/set 都记成修改后的自己 | 解析真实路径，拒绝已有目录、输入重叠、符号链接别名及受保护目录；发布到新目录，manifest 最后落地 |
| 衍生 manifest 不兼容统一入口 | v5 缺 `set`，真实 `longchain_replay.validate()` 立即拒绝；v5/v3g 缺顶层 `artifacts`、`max_context_tokens` | 从完整父 manifest 派生，保留必需字段，更新全部实际产物清单与元数据摘要 |
| 来源记录不随改写更新 | v5 沿用 v4 的 output origin；v3g 的发布未携带完整 provenance | 每个变更请求保存 before/after 与操作来源，gap 的生效值和分解是否已知单独记录；不改原模板记录来冒充源观测 |

这类错误不等于已证明了正式 chain 失败原因。原 runner 可能绕过上述入口直接加载文件，能发压不等于统一验收已通过。

新增边界：不允许元数据补丁改正文引用、请求身份、phase、token/LCP 标签；不允许漏请求或变更链内顺序。
输出预算检查上下文上限；gap 检查非负有限数、去重、位置与 ID 一致性。capped-replay 超过 310 秒直接拒绝。
默认继承原 cohort；显式换 cohort 只可重排同一组完整链。

## 3. 已确认的数据偏差，不能混称为代码 bug

| 项目 | 本次复算 | 对 chain / 并发分析的影响 |
|---|---|---|
| v3→v4 cohort 漂移 | 311 条链只 29 个位置相同；前 26/34/38 条共同链仅 8/12/12 | 跨版开场并非相同请求，不能把差值全归因补算量或引擎 |
| 全局总量掩盖逐链偏差 | v4 prompt 总量差 −2.10%，合成部分预期新增 token 总量差 −5.82%；245 条有合成请求的链，新增量绝对相对误差中位数 28.25%，151 条 >20%，76 条 >50% | 最难的工作可能落在错误的链、错误的位置；总量接近不能证明长尾或并发压力接近 |
| 内部 reset 工作量偏轻的可能性 | 121 个链内 reset，118 个合成，均为 append；v4 新增量 P50/P95 1,126/2,116 | 次数符合源摘要，但源事件位置、实际工作量未复原；不能把它们默认变成全冷，也不能默认它们确实这么轻 |
| 原始输出总量冲突 | 3 条链的公开预算和源摘要不能同时满足；v4 总输出多 3,491 | 生成器已记录例外，本次保留；不能宣称每条链输出都与源摘要精确相等 |
| 独立验收范围 | v4 保存的 check 为 STRUCTURAL_OK，rendered_prompts=0；构建阶段本身有实际渲染 | 结构通过不等于独立全量 token/LCP 复验，更不等于正式负载有代表性 |

以上新增 token 是冻结元数据的相邻前缀工作代理，不是运行时实际缓存命中或 GPU 重算量。
最严重的逐链欠填例子：631/72,208、1,405/127,159、2,932/118,606。不能修改计数标签让它们看起来达标；
应在拥有正文后仅修这些链的合法历史分歧，并重新计算受影响后缀的正文、LCP 与来源，不需要重造其余链。

生成器原 manifest 已声明 `DIAGNOSTIC_CANDIDATE_NOT_REPRESENTATIVE`。本分支保持这一边界。

## 4. 最新源 gap 摘要与可成立的判断

用户重新查询源业务历史数据库后提供；本次不访问数据库，临时源导出已清理。统计排除 311 个无前驱链首，
5,290 个有效非零正间隔，定义为 `next.started_at − previous.ended_at`，未拆工具/思考、未封顶。

| 指标 | 源业务原始间隔（用户查询） | v4 链中 replay gap（本地复算） |
|---|---:|---:|
| 平均 | 78.668 s | 5.647 s |
| P50 | 2.396 s | 2.653 s |
| P90 | 21.528 s | 12.551 s |
| P95 | 91.668 s | 22.110 s |
| P99 | 551.314 s | 33.210 s |
| 最大 | 26.06 小时 | 300.782 s |

源间隔 >60s / >300s / >3600s 分别 340 /148 /10。用户此前另确认，只按每链累计一小时上限比较时，
源累计等待约为 v4 的 3.5 倍；这些源摘要不包含足够的逐链信息，无法在本机独立重建这一计算。

这些事实确认缺少长尾，但不能据此指定处理后的 p95 或把 340 个事件固定分配到某些请求。
逐步规则是 `min(tool_union,300s)+min(net_think,10s)`，之后 harness 再按链压缩累计等待。
相同 91.668 秒原始间隔可能处理成 10 秒，也可能保留 91.668 秒。汇总分布还不包含长等待与
phase、上下文、输出长度及其他并发会话活动的关联。v3g 的中位数变成 7.319 秒，也不等于还原了这些关联。

长等待一方面降低即时在途请求数，另一方面可能在其他请求竞争缓存后导致回访重算；单凭等待不能断言缓存丢失。
后续只能做明确标注的敏感性臂，并报告回访命中、竞争期间 KV 写入、重算量和等入批时间；不把原始汇总分位数当作回放目标。

另一个边界：v3/v4 从公开数据继承了 207 个切段链首的非零 gap，合计 567.917 秒。
原 harness 的 `build_gap_plan → drive` 会等待这些 head gap，没有自动清零。
最长 62.202 秒的那个头位于 v3 cohort 第 30 /v4 第 26 位，因此 N34 开场前 60 秒经常只观察到 33 个头。
这是公开切段数据与用户源摘要的链首统计定义不同；本次保留原 harness 和公开元数据，不擅自清零。

## 5. 最终交付：v5 对照 + v5g-tail 主候选

用户明确：在 v5 的输出预算上应用等待长尾，不重新生成正文；review 全部通过后再开始测试。
GPU 与 Pod 均已生成 3 个完整目录，名称、路径和编排见[交接文档](../handoffs/chain-first-v5g-0927.md)。
三版都是 311 链、5,601 请求，正文链接到已有的 631,535,225 字节 v4 分片，没有复制第二份正文。

| 数据 | 修改范围 | gap P50 / P95 / P99 | >60 秒 |
|---|---|---|---:|
| v5-review | 修复发布契约，沿用已有 4,835 条输出预算改动 | 2.653 /22.110 /33.210 s | 18 |
| v5g-review | 修复已有按链时长缩放版的发布契约 | 7.319 /99.846 /310 s | 440 |
| v5g-tail-review | 从 v5 出发，只应用 239 个已在尾部的合成间隔延长 | 2.653 /33.210 /245.467 s | 250 |

旧 v5g 的 440 个 >60 秒间隔超过源 raw 的 340 个，且中位数明显变大，不能称为源数据的校准。
v5g-tail 的条件为：合成、非链首、v5 原 gap >全体链中 P90=12.551 秒，旧 regap 提案 >max(原 gap,60 秒)，提案不超过 310 秒。
保留其余所有原始等待，P50/P90 不变；链中等待总和 59,455.077 秒。
**这仍是有限假设的敏感性实验：没有恢复真实位置、工具/思考拆分或联合分布，不硬凑源 raw P95。**

复现（在有完整 v4/v5/v5g 的机器上，输出目录须新建）：

```bash
python3 -B scripts/longchain/finalize_v5g.py \
  --source-root /sjtu/linhang/arena/repo/cache \
  --out-root /sjtu/linhang/arena/codex/longchain-repair-0927/data
```

本轮 SSH 已恢复，GPU 与 Pod 都已完成发布；Pod 路径位于已经核验的 tmpfs，额外内容只含元数据、脚本和验收报告。
原先本机的 `METADATA_ONLY_INCOMPLETE` 文件只保留为构建演练，不作为最终交付目录。
Pod 原 checker 已完成主候选的全量结构验收：311 链、5,601 请求、5,601 正文，0 错误；
7/7 清单文件摘要一致，没有缺失或重复。收据是 `STRUCTURAL_OK`，不是独立全量 token/LCP 重渲染的 `VALID`。
完整报告保留在 Pod `codex/longchain-repair-0927/validation/v5g-tail-structure.json`；
[本地验收收据](../../evidence/longchain-incremental-repair-0927/pod-review-receipt.json) 记录其摘要、大小与检查结果。
开发机第一次检查没有留下完成报告，未把进程消失视为通过；Pod 补跑完成后 SSH 回传中断，
随后直接读取已写出的报告确认完成。GPU/Pod 三版 requests、cohort、正文摘要和改动计数逐项一致。

另直接调用原 `s1_loadgen.build_gap_plan`（每链 3,600 秒封顶）核验实际等待：v5 和 v5g-tail 都没有链触发压缩，
有效链中等待分别仍为 29,874.685 /59,455.077 秒；旧 v5g 有 1 条链触发，122,278.798→122,275.427 秒，
上表分位数与 >60 秒计数不变。可复算脚本是 [check_longchain_gap_plan.py](../../scripts/analysis/check_longchain_gap_plan.py)，
结果一并放在验收收据。此检查没有运行模型、修改 harness 或启动新性能测试。

## 6. 验证与交付边界

Fable 直接在 Pod 运行独立 `review_repair.py`，确认三套发布与 v4 的 ID 顺序、全部正文引用、cohort、
产物摘要和字段差异。tail 的 239 条实际变更与规则谓词精确相等，分布在 70 条链，其中 237 条 intra、
2 条 context_reset 前的等待；没有改 cohort 链首。审核同意作为敏感性臂使用。
[独立原始收据](../../evidence/longchain-incremental-repair-0927/fable-review-receipt.json) 保留全部 JSON 及原文件摘要。
其中 `changed_chain_index0=128` 指所属源会话切段编号为0，不能解释为改了128个链首；
`changed_context_resets=0` 原字段统计非空事件ID，phase=context_reset 的正确数量见 `changed_phase`，为2。
这两项字段解释已附在收据外层，不篡改独立原始记录，也不影响按cohort核实的非链首规则。

12 项针对性回归通过：实际改写 CLI 覆盖保护、符号链接别名、父文件变动、正文/分桶字段禁改、
gap 定位与边界、cohort 漏项/链内改序、真实 checker 对衍生数据的结构验收、缺正文拒绝、空缩放集以及只改等待尾部的边界。
两份完整真实元数据回归：修复后的 rebudget /regap 各输出 5,601 行，分别与既有 v5 /v3g 每一行逐字段完全一致，
证明本次发布修复没有偷偷改变已有工作负载。GPU 与 Pod 的发布脚本均按限定字段检查变更范围并验证父正文。

[修复入口](../../scripts/longchain/repair_metadata.py)、[发布实现](../../scripts/longchain/longchain_metadata.py)、
[回归测试](../../tests/test_longchain_metadata.py)、[原始审查复现](../../scripts/analysis/longchain_generation_audit.py)、
[逐链账本](../../evidence/longchain-generation-audit-0927/per_chain.csv)、
[原代码 bug 复现与汇总](../../evidence/longchain-generation-audit-0927/analysis.json)、
[部署与源统计收据目录](../../evidence/longchain-incremental-repair-0927/)。

本地与正式 chain 的 36 组记录重算另见
[chain 再分析](/workspace/Agentic_science_challenge/build/worktrees/prefill-sm80-0927/notes/reports/chain-local-official-reaudit-0927.md)。
本轮没有新的 GPU 性能结果，不据此宣称 chain p95 或 N@SLO 已改善。
