# N34/N38 chain 与 130ez 实验计划复核

2026-09-27，Codex。状态：原始记录复算与源码反例已完成；机制建议未部署、待 Fable 复核。本轮没有改引擎、队列或 GPU 服务。目标仍是 N@SLO，接受 TPOT/fast 的合理取舍，正式硬门不变。

结论：保留 130ez1 → 130ez4/5/6 的先后关系，以及后续单变量探索。7/8/9 主要回答稳态调度取舍；攻开场 chain 还缺“家族前缀生产者如何获选、共享前缀何时真正可复用”的观测，以及大块 prefill 执行加速。124 成本常数接近实测，不能排除预算分类、前缀依赖和准入之间的错误交互。

## 1. 新落盘记录修正了开场范围

读取 130ee5、130eezy 全部 raw 与 server log，用原 harness 的 selector/quantile 重算；四门 p95 与各运行 summary 逐项一致。唯一 raw ID 数、run 的 dispatched/n_attempted、job 的 DRAINED 条数一致，零请求错误。本轮没有独立复核 dispatch ID 账本，也没有重新归档或做 SHA 检查。这些仍是 60 分钟派发并排空的诊断，完整 cohort 判分为 INVALID，不能称正式 N34/N38 通过。

| 记录 | raw | chain 样本 / 超时 | 开场 0–10 秒到达的坏 chain | 后续坏 chain | 首次获选前已等 >30 秒 |
|---|---:|---:|---:|---:|---:|
| 130ee5 / N34 | 3218 | 282 / 18 | 16 | 2 | 15 |
| 130eezy / N38 | 3373 | 311 / 23 | 19 | 4 | 18 |

N34 两条后续坏例在 911、3237 秒到达，首次获选后至首 token 分别 30.68、29.55 秒；N38 还有后续等待与长执行生命周期的问题。这个时段包含块间调度和 host 工作，不能称纯 GPU 时间。仅优化开场排序覆盖不了所有坏例。

两个旧运行的有效机制都是 128=off、126=off、warm=5；不能作为 128 修正版或新 S1 warm=15 的对照。130d 是 128=on、126=off、warm=15、DCP1 的旧探针，和新 DCP2 基线也不能混算净收益。

N34/N38 开场第一条 relief=on 分别在首派发后约 0.08/0.25 秒；时间精度受服务日志整秒时间戳限制。relief 配置 interval=0，HIGH=15，前一段分别约持续 90/121 秒。因此 HIGH 15→8 和普通 interval 2→1 在这段大部分时间已经没有进一步放开的空间；新 130ez1 是否相同要用它自己的日志确认。

“允许 24 条”依赖桶分母。现有 score_formal 的原规则实现，对 282、311、341、350 个样本分别允许 20、22、24、24 条。不能把全 cohort 的 24 套到任意中途窗口，更不能用窗口门替代完整成绩。

复算证据：[analysis.json](../../evidence/dcp-chain-plan-20260927/analysis.json)、[全部 chain 的逐请求时间账](../../evidence/dcp-chain-plan-20260927/chain.csv)。来源路径在 JSON 内；原始数据保留于主工作区 `evidence/L130ee5-dcp_stack2_w2_n34_60m/N34` 与 `evidence/L130eezy-dcp_stack2_w2_n38_60m/N38`。

## 2. 对每一臂的建议

逐个读取并静态解析现行 job 的 G_ARGS/G_ENV，确认 ez7/8/9 是各自相对 ez1 的单参数变化，ezb 仅去掉 NEXTN 参数组。ez4/5/6 使用 6976639e；eza 使用 f2b6425e（e464d8ab 加 K3），严格说是共同底座的派生引擎，归因时保留这个区别。

| 实验 | 验收重点 | 必须保留的解释边界 |
|---|---|---|
| ez4/5/6，短尾 OFF/ON/OFF | 测量窗口内 target/draft 各 rank 的 local_short 实际增量；符合路由条件的请求/批的执行生命周期、A/A 离散度、池容量与峰值显存 | LARGE_MAX=0，不覆盖冷首块或大块。以短尾收益验收；总 chain 不降不能否定这条路径。冒烟题没有足够缓存前缀，不能单独证明新路线数值正确，需带真实共享前缀的路径定向检查 |
| ez7，interval 2→1 | 分别看 relief 开/关时的 prefill 等待、冷请求完成与解码进度；稳态同 ID turn/chain 修复与新增 | relief=on 已是 interval0；普通 interval 改变的是其余时段。不要仅凭开场坏例总数评价 |
| ez8，HIGH 15→8 | 新增进入 relief 的时间段、停留时间、切换次数、这些时间段附近的等待与 TPOT | 若新基线开场仍立即开 relief，主要收益空间在稳态中小积压。LOW=5 保持不变，留意窄迟滞是否增加切换 |
| ez9，MAX_SLOW 80→250 | 基线护栏首次锁死的时点、锁死后冷积压与失败请求；候选在对应阶段是否减少等待 | 服务器累计 slow 集合不等于最终客户端 TPOT>0.10 集合，250 也不是正式允许条数。若基线根本未锁死，此臂没有触及想测的机制 |
| eza，K3 | 按 warm 新增 ≤4096、4097–8192、>8192 分组，同 ID 比较获选前等待与新/旧 turn、chain 坏例 | 源码实际是 3/15/5 秒，不是所有单轮 warm 都给15秒；多轮降为5秒可能再次产生负 slack 和长等。8192 是静态 token 分界，不代表实际剩余批预算或资源一定能容纳 |
| ezb，去 MTP | 完整开关效果：prefill/draft/verify 时间、输出完成量、TPOT、KV 容量和回载变化 | 已钉住418槽、running/graph48，但去 draft/verify 仍会改变显存与池容量。首轮可以评价整体效果；获益后才按需要分离计算与容量贡献。接受长度不能单独代表 MTP 净收益 |

各臂继续对 ez1 单变量比较，不把前一臂有效旋钮带入后一臂。筛选后只组合有证据的少数变化，再到 N38 验证；组合收益不得把单臂数字相加。60分钟闭合用于筛选，正式判断仍按完整数据和原 harness。

K3 的 G_EXPECT 目前只有通用 124=on，没有验证 warm_multi 的有效值。建议其运行核验明确检查初始化的 `DeadlineConfig` 中 warm=15、fast=3、warm_multi=5、warm_multi_tokens=8192。无需新增一套评分规则。

## 3. 128 应先查哪里：code graph 定位、逐函数复核

调用链为 `_get_new_batch_prefill_raw → calc_priority/_compute_prefix_matches → _ax_admission_plan → _ax_family_plan/family_plan → tier_order → PrefillAdder`。CodeGraph 定位了调度入口和 `_ax_family_plan` 调用方；动态模块调用有缺失、通用方法同名有误连，均以实际函数体补核，不能把图边当作运行证据。

源码中两个独立交互已通过真实 `ax_deadline.py` 函数复现（合成请求状态，**不是运行根因已经定案**）：

1. **缓存增加反而降级。** 同一36k请求等了20秒，匹配16k时剩20k，预算30秒，slack约+7.38秒；匹配32k后只剩4k，预算降成3秒，slack约−17.91秒。它从排在另一条22k冷请求之前变为之后。K3也保留这个≤4096的3秒分支。成本常数误差15%无法解决这种27秒的预算跳变。
2. **家族领头的加权分数不能越过原生 held。** `tier_order` 首先检查 LPM 的 `held`，直接给 tier3，再考虑128的工作量分数。即使领头36k按5人摊成7200，只要被原生 held 标记，它仍落在普通22k请求之后。不能直接删除 held：需要确认真正的前缀生产者、是否已有生产者在运行、KV和KDA检查点何时就绪，防止全家重复计算。

130d 的现有日志在相对3.51秒已经报告目标领头 `…895c6291ea:llm:0` 有3个rider、work=8849，到29.51秒仍出现在等待家族中。它证明“发现了家族”并未确保“领头早执行”；日志没有完整 tier/held/拒绝原因，不能在上述两个解释中定案。

建议诊断只在 rank0、沿实际广播计划记录，不增加 CUDA 同步：候选RID、原始接收时间、当前匹配长度、预算/slack、LPM held、family leader/rider、加权分数、排序前后位置、当前partial、剩余块预算、实际准入/拒绝原因，以及共享前缀可复用和首次获选的事件。窗口覆盖负载触发的冷积压，输出量按候选数×调度轮数预算，不输出完整prompt。诊断开启臂也需要考虑自身host开销。

若日志确认，最小修法是：对确实等待已知共享前缀生产者的请求保留原有绝对deadline，缓存增长只减少预计工作，不缩短deadline、更不重置接收时间；为真正的生产者协调原生held与家族held。只用已到达请求及真实token/cache key，不读冻结family标签、不主动预热。KV可共享并不意味着KDA状态已到对应检查点。

再往后才考虑家族级收益排序：比较“产生可复用前缀的成本 + 可完成兄弟尾巴的成本”与预计能按时完成的人数，避免仅用 `leader_remaining/(1+riders)` 乐观记账。实际批预算、前缀树分叉、cache_salt/extra_key、饥饿上限均需参与约束。优先验证能否稳定救回这组家族，而不是承诺全部四条都能救回。

## 4. 执行层应接哪一项

短尾TP8路线验证之后，另立大块DCP本地续算探针，开关与短尾分开。已有两卡缩小模型的8K完整前向交错计时约省7.2%，这是可测试假设，不是TP8收益。TP8测真实冷P0和后续P增长的2K/8K块，核对local_large路由、精度、每卡临时峰值与有效KV池；大块约0.75GiB/层的临时注意力缓冲可能抵消容量收益。

DSA indexer 后续应针对当前DCP2+Humming栈重新拆 logits、top-k、prefix gather、attention merge 的关键路径。114行分片已经在配置中，不能把再开114算作新方向。只有实测占用支持时才投入分块logits/top-k融合、减少中间矩阵或通信重叠。开场8K→16K旧实测未省总时且多占显存，不优先重复放大块长；多partial也不创造计算产能，先把依赖与准入解释清楚。

复算命令（在本分支worktree执行，读取主工作区原始数据）：

```bash
python3 scripts/analysis/dcp_chain_plan_review.py \
  --source-root /workspace/Agentic_science_challenge \
  --output evidence/dcp-chain-plan-20260927
```

执行层已有实现与证据见[本地续算审查修正](dcp-local-extend-review-fixes-0926.md)；原130ed逐请求归因见[DCP chain报告](dcp-chain-20260926.md)。
