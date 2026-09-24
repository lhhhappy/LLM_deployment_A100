# 长链数据交接与文档索引

2026-09-24更新。**唯一现行造数规范：[scripts/longchain/longchain.md](../scripts/longchain/longchain.md)**。s1-dev作为素材库，允许取A session的query或完整片段，原样或局部改写后接到B，构造B自己的长程历史。Phoenix只供行为/时间结构，生产正文不进入成品。session独立是依赖与历史独立，不是素材必须唯一，不是强制KV隔离。

已按[task.md](../llm-challenge-arena-v1/task.md)第167–222、488行及原harness的drive/call_engine核对：离线冻结组合与固定轨迹回放兼容，工具不执行，本次回答不回填下一轮，压测不评回答语义正确性。review聚焦长度、前缀、事件、间隔、预算及依赖；无需复原完整故事。文本仍可能影响tokenization、MTP和MoE路径，不能承诺任意改字后同分。

## 当前完成与待做

- 旧96链/1718请求候选已冻结、CPU独立归档验收通过；仅诊断，不是新版或代表性验收，没有GPU成绩。[冻结入口](../evidence/longchain-audit/frozen-candidate/README.md)
- Phoenix最近7完整日分层采集1000/1000 session完成，55,860个LLM span、44,119个主调用；236条连续边的压缩计数增加。分层过采长尾，需用抽样权重；不同模型的token不冒充GLM token。[结构统计](../evidence/phoenix-longchain-20260924/expanded/behavior-profile.json)
- 公共素材全量去重盘点完成：433条非reminder user候选、371条无tool_calls assistant、2,381个可配对工具组。user候选还可能含控制消息，工具组按完整JSON去重，均非已验收可用续接数。[盘点与定义](../evidence/longchain-design-20260924/material-inventory.json)
- Luna low已试制24条有来源的局部追问和12条通用追问；仅素材试制，尚未进入新版测试集。
- **事件核心已实现**：跨session追问、工具续跑、显式重建和重建后增长；完整311链/5601请求已生成并通过真实GLM全量检查与原harness自检；小集与cache已清除。当前唯一成品、完整集进度和生成命令见[data入口](../data/README.md)与[实现说明](../scripts/longchain/longchain.md)。三路审查的失败反例已修复，GPU尚未测量。

## 生成与验收顺序

1. 先冻结分布/事件计划，再取素材。B的旧历史保持稳定；借A的query不搬入A的整份prompt/system/tools；工具组闭合、局部引用适配，来源另存。
2. 小集24–32链检查接缝和关键机制；通过后以源311链/约5,601次调用为首个规模目标。该规模现已完成并通过CPU验收，不是正式分布声明；原722条正文不限制合成请求数量。
3. 长短链混合；一次借入的query可接多次工具调用，不把每次LLM请求都变成用户追问。重建应真改变输入；长等待不强制miss，不加盐、不每session flush。
4. 全量CPU检查真实GLM token/LCP、工具ID、稳定历史、预算、事件分布、跨session重复和原harness自检。按同一冻结成品A/B；后续由执行层安排真实缓存/MTP/时长与全部门的测量。

原开发集只用于同负载A/B/回归，合成集补充长程机制；两者均不能换算正式N@SLO。正式341链/5150请求只能提供规模/均值锚点，不推出长尾比例。合成多seed共享素材，不能冒充更多独立线上观测。

## 现行文档与复现入口

| 入口 | 用途 |
|---|---|
| [longchain.md](../scripts/longchain/longchain.md) | 唯一现行设计，含跨session组合、容量、task原理核对与review范围 |
| [数据状态](reports/codex-data.md) | 完成项、限制、下一步 |
| [四类TTFT与造数依据](codex-四类请求与造数建议.md) | 原dev/048、Phoenix及冻结候选缺口 |
| [来源约束](codex-长链来源约束.md) | 可见正文与不可恢复字段；不作为禁止合成的规则 |
| [对话素材审查](codex-长链对话审查.md)、[旧实现审查](codex-长链数据审查.md) | 可复用观察与结构风险；当前设计以上述唯一入口为准 |
| [旧候选实现/验收/命令](codex-方案-长链负载与N22-N26验证.md) | 96链/1718请求的历史实现事实 |
| [旧独立验收](reports/review-data-sol.md)、[机器收据](../evidence/longchain-audit/independent-freeze-acceptance.json) | 不代表新版通过 |
| [负载策略](fable-合成负载-2026-09-24.md) | 与部署实验的关系 |
| [longchain.py](../scripts/longchain/longchain.py)、[checker](../scripts/longchain/longchain_check.py) | 当前事件构建与独立检查代码；v2不使用旧polish后处理 |
| [可选回放防护](../scripts/longchain/longchain_replay.py) | 全量离线验收后调用原runner；只换root/set/cohort，防止缺正文假PASS与陈旧正文cache |
| [素材盘点脚本](../scripts/longchain/longchain_material_inventory.py) | 复算素材供应，不生成轨迹 |
| [协调](collaboration.md)、[队列](queue.md) | 本会话只做数据/CPU工作，不操作8卡队列 |

## 证据与数据

- [原始s1-dev](../s1-dev/data/dev-combined-v1/)、[原harness](../s1-dev/harness/)、[tokenizer](../s1-dev/glm_tok/)均只读。
- Phoenix：[抽样框](../evidence/phoenix-longchain-20260924/expanded/sampling-frame.json)、[1000份收集状态](../evidence/phoenix-longchain-20260924/expanded/collection.json)、[连续联合事件](../evidence/phoenix-longchain-20260924/expanded/joint-events.jsonl)、[事件窗](../evidence/phoenix-longchain-20260924/expanded/event-windows/)。原始正文cache已按用户清理要求删除；结构事件与抽样记录保留，不影响离线生成。
- 旧数据：旧成品字节已删除，仅保留[完整CPU校验](../evidence/longchain-audit/final-check.json)、[原harness自检](../evidence/longchain-audit/final-self-check.log)、[三本账](../evidence/longchain-audit/workload-ledger.csv)、[导出比较](../evidence/longchain-audit/export-audit.json)。旧冻结数据及父数据副本已按用户要求删除，仅留收据；阶段pilot收据不能代替最终收据。
- 原dev与正式A/B差异：[实验](experiments.md)、[提交](submissions.md)、[当前事实](knowledge.md)。不据此拟合数据让某个部署通过。
