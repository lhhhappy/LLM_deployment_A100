# Codex 分析：旧长链候选的实现、验收与使用边界

本页保留96链/1718请求冻结候选的实现与验收事实，不是下一版生成规范。**现行规范统一见[longchain.md](../scripts/longchain/longchain.md)**：s1-dev为素材库，允许将A session的query/完整片段改写后续接到B；不要求恢复原故事，按事件计划构造长短链、工具续跑、等待和重建。Phoenix只提供行为结构。task.md的固定轨迹回放兼容这种离线组合；review聚焦负载和依赖，不逐条审文学/问答正确性。

交接入口及全部相关文档地址见 [长链数据 handoff](codex-handoff-长链数据设计与文档索引.md)，当前状态以 [数据状态页](reports/codex-data.md) 为准。旧候选曾生成并[冻结留档](../evidence/longchain-audit/frozen-candidate/README.md)，最新完整CPU校验为1718/1718、0错误；冻结时结构/哈希复查通过，主会话安排的独立验收已确认[ACCEPTED_DIAGNOSTIC_CPU_ARCHIVE](../evidence/longchain-audit/independent-freeze-acceptance.json)，本候选分析工作收尾；成品副本随后已按用户要求删除。**它不满足代表性长压测的质量要求，仅是诊断候选，不能替代原开发集或按当前数据安排代表性N22/N26评测。**本会话不操作 GPU 队列、不正式提交。执行层与编排线按 [coordination.md](coordination.md) 并行，原 [执行层 handoff](codex-handoff-执行层并行任务.md) 作为任务背景。

## 用户确认的交付标准

目标是更接近线上结构的长链负载，单档期望约2–3小时，保留链长长尾、上下文增长、输出、等待、前缀变化及共享家族的关联。正文允许组合与局部生成；时长只能实测，不能靠延长空等、循环克隆整链或伪造测量凑时间。正式341链/5150请求只是规模锚点，不意味着小集按请求数等比例缩放就等价。

质量优先于凑满96链/1718请求。允许跨session借query、切换任务及组合适配片段；来源可追溯、接收历史稳定、工具依赖闭合，不能只调准prompt/output均值却改变事件压力。基线回放前验收数据；回放后检查实际工作量、缓存与MTP、稳态覆盖。没有真实逐轮资料的字段明确估计，不把旧候选当默认评测。

## 为什么做；不拿它换算正式分数

044r 用正式 A 原样、含121，在开发集 N14 判 overall/chain 失败，TPOT mean/p95 为 .0426/.0783，而正式 A 同档通过（.0174/.0364）。045r 的正式 B 原样开发集 N10 也失败，TPOT .0345/.1182，正式为 .0121/.0206。两组 p95 比约2.2和5.7，不支持统一乘除系数。原始实验与官方出处见 [experiments](experiments.md) 和 [submissions](submissions.md)。

因此不要求开发集先过 N18/N22 才研究 N22/N26；原开发集仍用于同负载回归。新数据补充“长链、长上下文、持续周转”的机制覆盖，所有评分门照实报告。合成集有效 FAIL 不阻止有明确问题的高档诊断；缺记录/错误测量为 INVALID，先修工具。禁止改阈值、按成绩重新挑链或回写数据来获得 PASS。

## 旧候选实际生成规则（供复现）

唯一生成入口 [scripts/longchain/longchain.py](../scripts/longchain/longchain.py)：`build / polish / check`。不用 Pi、在线工具模拟、模型权重或外部 API；完整历史工具结果已经在公开数据中。三个 GPT-6 Luna、medium 辅助来源审计、独立校验和 query 审查；核心选择、生成、传播与冻结代码由本会话实现。

1. 对源 canon/serving 请求按精确 `pack:view:logical_call_id` 连 body；链内按原 harness 的 dispatch/id 顺序排列。不能从 req_id 字符串拆出 session，`chain_index` 也不是轮次。
2. 按 pack × 源链长度分层配额选96链；固定 seed=20260924，在128次候选抽样中选择源平均链长、prompt、输出与家族碰撞概率较接近者。选择过程完全不看引擎成绩。实际65个 system/tools 家族，家族成对相同概率1.95%，全来源为1.74%；这只是家族指标，不能当成实际 token 前缀命中率。
3. 每条选中链按自己的 `n_requests` 续长，不统一补成15轮，不截尾。公开请求正文保留，输出row的body_ref统一指向成品shard、原路径保留在provenance；缺失轮由完整历史 assistant 调用—tool 结果块构造。优先同链/同家族且增长量适合的块，允许同 pack、参数 schema 相同的跨家族块。固定 RNG 在前16个适合块中取样，重复使用有惩罚，不声称所有块唯一。
4. 新调用 ID 与结果中的对应引用一起重编号，仅修改复制的新块。缺工具结果或显式 ID 不匹配的 donor 拒绝；对每个调用的工具名和参数 schema 做匹配。不等于完整语义/文件状态一致性，也不执行历史工具。
5. 逐链累计 prompt 目标用于选完整块；输出总预算扣除已公开预算，再按 donor 观测输出权重分配缺失轮。没有把正文按 token 截断，没有减少公开输出预算。源统计冲突无法满足时显式记录偏差。
6. 真实相邻转移保留其 gap 和尾提醒替换方式；phase由实际新消息事件产生：新增普通user为turn_start，否则intra。当前不生成真实context rebuild；donor原phase仅记入provenance，原公开请求的源phase保留。历史块缺少真实gap/output时借配套观测转移并标为估计。没有按 N 缩放 gap，不假定固定20/28请求每分钟。仅做追加或最后提醒替换，未编造缺失 reset 的精确位置。
7. 使用原 Renderer 与 tokenizer 全段渲染后计 token、精确 LCP、新增与移除量；保存每条来源、donor、hash、输出/gap 估计说明。新请求的时间字段不伪造服务端实测时间，dispatch 只承担冻结排序。
8. `polish` 接受 Luna 建议经 root 复核的编辑。用户已授权重设计初始 user query；它在该消息仍存在的后续快照中一致传播，再全部重渲染。受影响公开快照改标 `adapted_original`，原 body hash 和源冻结标签另存；其余原请求保持不变。不改 system/tools、工具结果或参数，不插角色 token，不随机改旧历史制造缓存 miss。

每份数据在独立新目录创建，拒绝覆盖。原 `s1-dev/`、题面、harness 与 tokenizer 均只读。各 N 使用同一个 cohort、同一个顺序与 gap；链结束后空槽接新链，维持原闭环规则和周转。

## 分布：改善在哪里，缺口在哪里

以下为 raw 的实测离线统计，最终 query 编辑后的精确值见最终检查报告。源整链统计不是隐藏正式集的分布；正式仅已知用户转述341链/5150请求、平均约15.1轮。

| 指标 | 公开开发集 | 本次 raw | 源整链元数据 |
|---|---:|---:|---:|
| 链/请求数 | 311 / 722 | 96 / 1718 | 311 / 5601 |
| 平均链长 | 2.32 | 17.90 | 18.01 |
| 链长 p50/p90/p95/max | 1/5/6/13 | 8/40/84/228 | 8/45/82/240 |
| 平均 prompt token | 47,669 | 86,175 | 86,602 |
| prompt p50/p95/max | 36,576/114,376/256,733 | 82,293/173,421/259,112 | 缺逐轮值 |
| 平均输出预算 token | 299 | 738 | 733 |
| 输出 p50/p95/max | 198/915/5,644 | 531/1,865/11,775 | 缺逐轮值 |
| gap p50/p95，秒 | 2.00/18.67 | 1.50/9.23 | 缺逐轮值 |

93/96链的累计 prompt 与各自来源目标误差≤5%；另3链约7.3%、8.7%、13.5%。95/96链输出总量精确匹配；1链来源对缺失轮留下零预算，采用 donor 预算后多889 token并明确记录。全来源3链存在此类矛盾。53/311源链的可见起点晚于元数据首请求，本集17链；补长不等于恢复其真实顺序。

最终候选有227条原样请求、20条改写query的adapted_original，以及1,471条合成请求；6条query一致传播到488个快照。最终prompt合计148,080,545 token、mean86,193.6，输出预算仍1,268,003。1,471条补入请求中：299来自可见相邻转移，1,172来自历史块；952同家族、519跨家族。691个不同 donor 指纹被使用1,471次，同链最多复用一个块6次，跨集合最多16次。重复文本在不同前缀分叉之后不会直接创造跨链前缀命中，但可能影响 MTP 预测和专家选择。

最需要保留的偏差：

- **gap长尾不足。** 新集p95反而比dev短。原 harness 每链累计gap cap=3600秒，本集没有链触发压缩；输入/有效累计gap都是8,058.265秒，最大单链1,219.173秒。这些不证明正式节拍已对齐，也不能拿词句润色修复。
- **reset/新turn不足。** 所选源链 phase 总量为 session_start28 / intra1569 / turn_start76 / reset45；最终候选为28 / 1638 / 44 / 8。缺失的逐轮位置不可恢复。本集更偏追加命中，可能低估重建导致的prefill与缓存故障。
- **关联是部分约束。** 源链长度/累计量可对账，缺失轮的输出、gap及其与增长量的联合关系仍是估计；平均值接近不证明联合分布正确。报告提供按pack/family/phase/轮次/来源种类分组与联合直方图。
- **语义并未全修好。** Luna只对可自然衔接的query做编辑；相同工具家族不等于同一任务，跨主题、文件状态不一致和重复块仍应标记。此集不能用来证明能力正确或MTP与正式等价。

最终候选按原harness实际分桶为fast1524、overall1602、turn18、chain98，fast包含在overall。不能把 phase 的计数直接当 TTFT 分桶，不能沿用722集的超标允许数。链内理想可复用LCP为140.472M、剩余7.609M token：这是仅按前一请求计算的静态账本，不是GPU实际cache，也未扣除跨链共享或加入淘汰重算。

### 三本账：为什么8.97M与6.01M不能直接当作GPU差距

逐链明细见 [workload-ledger.csv](../evidence/longchain-audit/workload-ledger.csv)，总量见 [JSON](../evidence/longchain-audit/workload-ledger.json)。

| 口径 | token | 含义 |
|---|---:|---|
| 所选完整来源冻结uncached总量 | 8,969,783 | 每条源请求相对原来源前驱的冻结标签合计 |
| 其中公开247请求的原冻结总量 | 4,073,420 | 已有正文部分，不等于这些请求冷入场时必须算的量 |
| 未公开1471轮对应的来源剩余冻结预算 | 4,896,363 | 可能包含可见起点之前的缺失轮，不能一律称未来轮 |
| 当前1471条合成请求的LCP新增 | 1,934,529 | 合成全是后续追加/尾提醒替换；比剩余来源预算少60.5% |
| 成品全部请求冻结uncached合计 | 6,008,325 | 原请求保留来源标签，adapted/synthetic按可见前驱重新冻结，属于混合历史约定 |
| 成品全部请求按可见前驱算LCP新增 | 7,608,798 | 统一以该cohort前一请求计算、每链首LCP=0；不是物理GPU计数 |

6.008M与7.609M差1.600M，主要来自公开请求在本cohort冷入场、原冻结标签却参考来源前驱。其中约1.592M发生在链首。17条链的可见起点晚于来源起点，丢失早期工作会部分转成当前链首的冷prefill。**不能据60.5%宣称整档GPU少算60.5%，也不能用8.97M/6.01M作换算系数。** 跨链前缀共享和实际淘汰还会再次改变GPU工作量。

这仍然证明当前生成约束不足：累计prompt/output对上，并不保证工作出现在正确轮次、正确SLO桶。把原本中途的大段前缀变化移成链首冷算，会改变3/5秒门与30秒门之间的压力。下一版应保留真实可观察的前缀变化，并按来源阶段、轮次、增长与工作量一起对账；不能通过随机修改旧token来强造miss。

## 随机拼接会不会让计算变简单

语义简单不会让普通 prefill 少执行模型层。在相同shape/执行路径下，主要工作由token数、上下文、块、状态和通信决定；但 MoE 路由与 MTP 接受率受内容影响，不能承诺不同正文完全同速。不同历史分叉后的相同文本不等于可用的相同前缀；共享 system/tools 反而应保留，不能为了去重改掉。

当前最明确的“可能偏容易”来自缺少重建和较短gap；大量追加请求理论命中长前缀。润色可以改善话题衔接，不能恢复这些负载事件，也不能靠整段重写旧历史人为制造miss。正确检验是冻结后用基线测实际prefill、cached_tokens、KV/KDA占用/淘汰、MTP接受率与实际输出长度，并按来源kind/时间/轮次分组。对照关闭MTP只可作为独立机制诊断，不能混成同配置成绩。

## 校验与证据

- [原722请求检查](../evidence/longchain-audit/original-check.json)：完整Renderer与token验证。
- [raw早期检查](../evidence/longchain-audit/reviewed-raw-check.json)：1,718/1,718渲染与LCP核对通过；其当时的VALID不包含后来补入的body_ref/hash/phase事件检查，不作为最终验收依据。
- [原harness raw自检](../evidence/longchain-audit/raw-self-check.log)：失配0，共1718，PASS。原self-check会跳过缺body，因此必须配合独立ID全集检查。
- [前48语义审查](../evidence/longchain-audit/query-edits-a-review.md)、[后48语义审查](../evidence/longchain-audit/query-review-b.md)：建议不是自动批准，root再审后冻结。
- 校验器 [longchain_check.py](../scripts/longchain/longchain_check.py) 检查request/body/cohort覆盖与顺序、实际shard引用、冻结产物hash/cohort自校验、正输出/非负gap/上下文预算、完整新增工具调用组及synthetic实际事件phase、官方渲染、精确前驱LCP、再生成冻结标签和body/provenance；输出逐请求与联合分布。
- CPU `VALID` 只证明数据自洽，不证明引擎KV/KDA状态正确、能力过双90、SLO通过或正式负载等价。独立诊断归档验收已通过；该候选不安排GPU回放。历史生成阶段源码字节未归档，现行代码快照不能反证历史执行过程。

### 独立复核修复

- `body_ref`：原请求虽然body不变，仍必须指向本次实际输出shard。早期build/polish遗漏这一点；harness扫描ID能够回放，但元数据不正确。build和polish已统一所有row引用，并保留源路径/hash。
- synthetic phase：原数据的context_reset标签不能仅凭高LCP判断错误，我们不修改源phase。但生成器实际上只追加完整工具块或替换尾提醒，没有执行来源的隐藏重建事件；因此新请求按成品事件定intra/turn_start，donor_source_phase另外记录。两条旧synthetic会由context_reset改为intra，依据是生成操作，不是LCP阈值。
- 冻结检查：仅重渲染不能发现预算/gap被篡改；现在检查manifest全部文件hash、cohort自校验及数值/schema约束。旧raw只作输入并显式记录缺陷；最终出口修复且重新完整检查。
- 链摘要：早期把来源append计数/比例和起点混入成品摘要；现顶层按成品重算，来源完整摘要只保留在source_chain_targets。成品内部边1613/1622属于追加或末提醒替换。来源total_edges包含外部入边，不能直接与成品内部边数量比较。

## 归档与当前入口

旧成品与未润色父数据副本已按用户清理要求删除，不再保留在cache。原冻结hash与验收仍见 [归档收据](../evidence/longchain-audit/frozen-candidate/README.md)；其中历史绝对路径是当时环境，不是现行入口。

旧追加式构建命令已移除，避免用当前事件生成器冒充复现旧算法。当前唯一成品见 [data/README.md](../data/README.md)，生成/验收命令见 [longchain.md](../scripts/longchain/longchain.md)。本页保留旧候选的工作量分析与限制，不再维护第二套生成方案。
