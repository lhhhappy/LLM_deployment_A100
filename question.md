# 最后四小时：请专家判断两次提交最值得押什么

更新：2026-09-30 08:59 UTC，Codex。Astra max与6.1 Sol已完成代码、原始数据、波动、稀疏注意力及最终09/10结果的独立审查。两臂均DRAINED。专家建议07+09后，用户选择10+09，两包已成功上传47798/47800；用户随后结束工作，追加一小时组合试验尚未入队。闭合结果与取舍见末节及[实验记录](notes/experiments.md#09-30-codex最后两臂闭合与两包取舍)。不要把局部跑分直接换算成正式档位。

## 目标与时间

- 模型 GLM-5.3-Flash，8×A100-SXM4-80GB、TP8。排名先比 N@SLO，再比同档逐请求 TPOT 均值；TPM 不排名。
- 用户提供的榜单：第一名 N42、TPOT 37.352 ms；我们 N42、43.266 ms。追平需要省5.914 ms，约13.67%。第一名 decode TPM16985.7，我们18721.4；总吞吐更高不代表每路出字更快。
- 最后两次正式提交已上传：10 CAP为47798、09 TPOT为47800；上传回执均queued，正式成绩待平台返回。
- 用户最新取舍：本地原配置 N42 本身也不全过门，候选重点看同条件相对提升，不以本地绝对门或一次新增坏例强硬否决；正确性、数据完整性仍必须过。chain 优先，TPOT 均值也可作为主要收益，不隐去另一项的代价。
- 用户不希望把剩余时间用在相同条件重跑，也不接受没有代码理由的参数网格或收益承诺。

## 正式与本地口径

[比赛规则](llm-challenge-arena-v1/task.md)、[公开 harness](s1-dev/harness/) 保持只读。N是同时在场的逻辑会话槽，包含思考和工具间隙，不等于 HTTP 请求数或 GPU batch。

TPOT逐请求计算为 `(客户端末token SSE时刻－首token SSE时刻)/(输出token数－1)`，然后取均值和p95。首token之前的等待不直接计入TPOT；首token之后的prefill打断、通信/执行等待和实际SSE发送停顿会计入。输出长度固定，不能压输出或改计时。

四道TTFT目标是fast3s、overall5s、turn15s、chain30s；正式TTFT有统计余量，本地CP估计不等于已经知道平台的完整算法。TPOT p95100ms是正式硬门，但本地与正式负载不同。

[47266正式终态](evidence/official/attempt-47266-final-20260929.json) 已保存在本地，无需查询线上：N42 PASS；TPOT均值43.266ms、p9571.276ms；fast1.639s、overall2.056s、turn3.457s、chain45.075s。chain点p95超过30s而平台仍PASS，不能用点p95代替统计门；平台未给N46失败档逐项和逐请求记录，不能断言正式只卡fast或chain。

本地固定v5g-tail-rot150，311链、5601请求，requests SHA256 `6170fd8204db4de8be4f99861d8cfa20ba36eab8b3317b8a3cd1343c5aef8ebb`。以下除10为45min外，都是1h派发后排空的DRAINED窗口，保证已派发子集闭合，不是完整cohort PASS。不同配置会改变闭环到达与窗口完成请求数，所以另做同ID配对；10与旧1h臂仅作明确不同派发时长的共同ID描述，不能直接比较各自全窗均值。

## 当前代码与 Pod

- 47266引擎基线：`bf6b66faf3ffbe10e593e858d3bbedd1c48cefb2`。
- 当前干净提交：分支`codex/final-execution-0930`，worktree`build/worktrees/final-execution-0930/`，HEAD`ca5d646c252688177480c7e67cec0a901e4b7069`。
- 186日志/说明草稿已留存到`build/scratch/execution-0930/186-held-off/`，不纳入候选或Pod；当前worktree已逐字恢复到上述提交，`git status --short`为空。专家报告中的186审查是此前草稿的独立结论。
- 实际默认配置：BF16 KV、Triton KDA、MTP关闭、running/graph48、Mamba池400、host HiCache64GB、mem_fraction0.87、attention-TP input scattered、全局prefill块16k、常态/积压冷块16k、短请求预留2048、普通PDI2、relief interval0、chain-risk interval1。
- 07：N42，175/176+五项小修，DRAINED并已取回SHA/同ID复核；真实权重冒烟12/12、KV容量1810112tokens（21.44GB）与基线一致。测量开始约05:12 UTC。
- 08：原同组合N46待执行臂已按原脚本SHA移入cancelled，保留脚本与收据；改测有明确机制差异的09/10。
- 09 N42 relief1与10 N46常态cold12k均已DRAINED，分别3273/0错、2541/0错；两者真实权重smoke12/12、KV1810112tokens与基线一致。09于06:39:16–07:39:16 UTC派发，07:41:43排空；10于07:52:42–08:37:42派发，08:42:11排空。Pod running/pending均空，服务保持。
- 38个Pod共享runtime文件哈希未改。镜像`lh-img:0930a`（image id168511）已成功构建，源码对应`ca5d646c`；BASE/TPOT/CAP备选包均通过最终检查与dry-run，输出目录只有启动配置。本轮两次正式提交已成功上传，详情见[提交记录](notes/submissions.md#09-30-最后两次正式提交)。[构建收据](evidence/submission-0930-execution/image-build.json)、[包选择收据](evidence/submission-0930-execution/decision.json)。

## 已闭合的本地结果

| 实验 | N | 改动 | 完成/错误 | chain超标 | chain p95 | TPOT均值/p95(ms) | fast/overall/turn超标 |
|---|---:|---|---:|---:|---:|---:|---:|
| A1 |42|47266原配置|3133/0|14|23.63s|61.51/103.60|283/227/7|
| A2 |42|原配置重复，历史已有|3132/0|17|28.94s|62.03/105.65|244/195/10|
|02|46|47266原配置|3220/0|21|34.89s|64.99/105.02|441/396/12|
|05|42|175MoE decode W2+176DSA页复用|3257/0|15|26.45s|59.80/103.29|252/193/12|
|06|46|与05相同|3328/0|24|37.52s|62.67/111.28|412/376/12|
|07|42|05+178/179/181/182/184|3268/0|13/373|25.13s|59.88/103.66|303/244/10|
|09|42|07+仅relief interval0→1，60min|3273/0|16/374|28.55s|57.09/99.44|245/215/5|
|10|46|07组合+常态cold12k，45min|2541/0|21/324|39.06s|67.01/114.93|271/247/11|

同ID结果：05对A1修好2/新坏2，对A2修好3/新坏1；方向落在两次基线之间，不能称稳健chain改善。06与02有3220共同ID，chain修好3/新坏6；共同请求TPOT均值64.988→63.168ms（降2.8%），p95105.023→111.958ms。05/06均有平均收益，但没有显著降尾部的证据。

07对05有3241共同ID，chain修好2/新坏0，但TPOT均值59.907→59.940ms基本持平；对A1完整覆盖3133共同ID，修好2/新坏1，均值61.514→60.335ms；对A2完整覆盖3132共同ID，修好4/新坏0，均值62.028→60.314ms。所有共同ID输出计数一致。五项小修可保留底座，本次没有证明在175/176上追加显著TPOT收益。证据：[07对05](evidence/execution-0930/smallfix-vs-moe-dsa/paired.json)、[07对A1](evidence/execution-0930/smallfix-vs-a1/paired.json)、[07对A2](evidence/execution-0930/smallfix-vs-a2/paired.json)。

[独立原始数据复核](evidence/execution-0930/pod/closed-results-independent-review.json)、[05对A1](evidence/execution-0930/moe-dsa-vs-a1/paired.json)、[05对A2](evidence/execution-0930/moe-dsa-vs-a2/paired.json)、[06对02](evidence/execution-0930/moe-dsa-n46-vs-baseline/paired.json)。原始raw和SHA收据在`evidence/execution-0930/pod/`，不要仅引用本文表格。

## 五项小修是什么，为什么做，边界是什么

| 编号 | 省掉的工作 | 正确性与适用边界 | 开发机已观察到什么 |
|---|---|---|---|
|178|同一forward的`prefix_lens>0`比较，34个KDA层共享一次|当前forward/同一只读prefix对象；capture、verify、draft等保持原比较|mask准备+34次查找省257–260us；真实conv+scan stub，非完整模型|
|179|把prefill逐Req GPU标量取回CPU再上传，改为GPU gather；mask/长度用新pinned源上传|冻结helper修改前的track位置，保留copy/overlap生命周期；静态普通pool、B>=8才启用，小batch回退|完整prepare B8/12/32省约62/119/408us；首250个N46 prefill中238个B1，不能当普遍收益|
|181|无RoPE尾部时重复拼接完整BF16 query|完整连续、16字节对齐、只读query；不符合条件回原路径|实际attention完整graph边界B1/24/32/48省约1–4us，output/LSE相同|
|182|同一CPU槽列表被两次CUDA高级索引赋值重复转换|一个新pinned int64 index异步上传供两次原赋值共享；静态非lazy pool、PP1、无spec、非capture|完整prepare闲置边界省约10–26us；保留原返回list和stream依赖|
|184|HiCache多个pool重复拼接/上传同一索引|必须Tensor对象identity相同且member顺序完整；equal clone/子集/缺成员回退，descriptor/events/record_stream不变|normalize/合成完整transfer省约7–45us；非真实TP8页布局整服务|

代码与文档都在干净worktree的`engine/sglang/`、`engine/docs/178*、179*、181*、182*、184*`。证据分别在`evidence/execution-0930/kda-shared-prefix/`、`tokenspeed-host-review/`、`dsa-query-view/`、`cache-index-alias/`。这五项可以作为有代码依据的小修保留，但不能把不同探针的微秒直接相加或承诺13.7% TPOT收益。

## 新找到的DSA宽元数据融合：已验证，但宏观预算较小

真实TP8 profile03：8rank×5次纯decode，同一个约8k前缀、B32。稳态完整GPU步约18.74ms，GPU无活动空隙约0.16–0.19ms；CPU约3ms且graph提前提交，未证明主要是CPU供活饥饿。Humming up/down、DSA、通信、mHC、KDA都仍有成本，孤立纯decode不能直接与正式混合TPOT相减归因。

每完整forward每rank恰好一次1M宽元数据gather和128MiB DtoD；11次的是逐层attention/indexer。已有`SGLANG_EXPERIMENTAL_DSA_KPOOL_METADATA_FUSION=1`按device live长度生成元数据，省掉大部分无效宽复制；不是新attention算法、不删有效历史、不减少topk，不缩池。

开发补充资格已闭合：authoritative live metadata/logits一致、graph完整selected集合（含padding）一致；实际round_scale=True cache producer验证空行/无效req/混合行和padding不改状态；共同selected输入output/LSE逐位相同。native TopK原本atomic顺序不确定，native输出/LSE不是逐位相同，有限self/cross误差分布逐项保留，不能把某次有限baseline-self当绝对误差界。

完整单stage B32净省337.74us，但metadata每完整模型步只做一次，不能乘11或8。真实profile可消除GPU工作约0.31ms；整步可兑现预算保守0–0.35ms，约0–1.9%纯decode步，绝不等于TPOT下降21%。还没有当前TP8服务结果。

[完整资格](evidence/execution-0930/metadata-fusion/receipt.json)、[独立调用次数与预算](evidence/execution-0930/pod/decode-profile/metadata-callcount-independent.json)、[完整profile关键路径审查](evidence/execution-0930/pod/decode-profile/criticalpath-audit.json)。

## 暂不投入的路线，避免重复

- 额外graph档28/36/44：已测N42，chain对A1修3/新坏7、对A2修3/新坏5；graph池约0.74→0.79GB，稳态TPOT无改善。
- mHC现有helper：M32 graph22.9→34.1us，更慢；另一CUDA实现仅省约1us。
- 174 packed KDA：单步FP32状态等数值过，但B32/36略慢，off。
- 177 decode metadata glue：TP8输出资格INVALID，存在baseline同波不确定，因果未查明，off；不把它当prefill graph已过。
- 183 directstack：B32完整prepare重复观察退化，未合入。
- 185删attention死shared输出：graph省约3.5–4us，但B24/48 eager残差未证明无负，probe-only。
- MoE up128：已有normal/skew/clamp全MoE数据，多个形状退化，不重新包装成新方向。
- MTP/DCP/短prefill graph历史已有失败/未完成资格，四小时内需明确指出可复用的最小路径与验证成本，不能泛泛建议重写。

## 参数不是名字，而是实际覆盖分支

当前`Scheduler._arm_prefill_decode_interval`：先ordinary PDI2；125正在relief时覆盖为0；131有risk时取`min(interval,risk1)`。因此只调ordinary PDI对relief0和risk1轮次可能不起作用。

125 `MAX_SLOW=80`是累计曾经超过0.10的RID数；到阈值后直到flush才恢复。已有05/06日志显示relief主要出现在开场前两三分钟；后续没有再on，不能假定全程relief0是主要瓶颈。日志为[relief transitions](evidence/execution-0930/pod/moe-dsa-n42/relief-transitions.txt)。

历史同bf6/rot150/N34已试：relief0→1新坏chain2条；ordinary PDI3新坏1条；PDI4新坏0条且晋N38，N38 fast失败。PDI4 N34均值52.522 vsrepeat52.953ms，仅约0.8%，不是13.7%证明。PDI4旧包也已提交47607；新的当前执行包/N42若重试，必须明确新代码底座与实际生效比例。

历史cold12k臂只改常态`SGLANG_AX_SCHED_COLD_CAP=12288`，仍保留全局chunk16k与`BACKLOG_COLD_CAP=16384`开场积压产能。不是把所有块都改12k。cold12k在N34曾观察fast/TPOT改善且无新增chain，尚无当前五项组合/N46结果。

[参数与harness独立审查](evidence/execution-0930/final4h-parameter-review.json)。

## 请专家回答

1. **最必要的一项是什么？** 针对37.352ms与43.266ms的同档差距，真正可在四小时内影响逐请求TPOT均值的方向在哪里？哪些耗时是真可省、哪些只是不同负载或计量口径？
2. **怎么读harness和混合负载？** 逐请求非token加权TPOT、短输出的stall敏感性、闭环逻辑会话N、TTFT冻结uncached_expected分桶，会怎样影响最有价值的优化？请给具体代码路径与数据证据。
3. **剩余Pod怎么排？** 保留08原N46五项臂，还是用新N42 TPOT参数臂/新N46 cold12k臂替代？若只够两个1h臂，为什么选它们、保留哪些保护、以什么相对结果作决定？不要为了绝对本地PASS浪费时间。
4. **是否值得现在启用DSA元数据融合？** 数值/状态验证与每forward调用预算审查是否足够进入真实TP8冒烟和组合试验？请审查guard、padding、graph replay、pool生命周期及已有native排序不确定性，不把21%单stage等同宏观收益。
5. **有比PDI4更有力的最小改动吗？** PDI4既可能被覆盖，历史收益也小；能否从实际服务代码提出一个机制明确、可逆、小时内验证的修复？要列收益预算、失败模式和停止条件，不能只是调高/调低。
6. **两次正式提交押什么？** 结合正式chain统计余量未知、本地相对收益和旧正式配置，建议一个TPOT包和一个容量包，或指出这分工不合理。若没有足够证据，给最小补充实验；不要承诺N46一定过或追平第一。
7. **同配置波动来源在哪里？** A1与A2均值接近、chain p95差5.31s。请按同ID、到达时间、输出长度、开场与稳态、缓存/检查点、prefill块和GPU/CPU停顿分解；区分闭环负载反馈、系统漂移和数据集合变化。提出能改善服务稳定性的最小修复，以及不修改harness的观测或比较办法。
8. **现有稀疏注意力还能怎么改？** 请实际审查DSA索引器→TopK→稀疏attention→缓存更新与TP8调用链，对照FlashMLA SM80等外部实现的布局、访存与调度细节。保持有效历史、topk、输出和状态契约；指出小时内可验证的工程机会与应留作中期的FP8 KV适配，避免只引用论文成绩。

请优先阅读实际冻结代码与上述原始JSON，再给不超过两项的高价值尝试。你可以反对当前计划，明确哪些观察会让你改变判断。读代码和证据即可，不改harness、生产、队列，不发消息给外部。

## 专家初步建议与已准备的具体尝试

[Astra max最终审查](evidence/execution-0930/astra-final4h-review.md)已经独立复算07与两次基线，核对实际调度分支并逐项审查09/10冻结脚本。其建议与Codex选择一致：优先触及首token之后的prefill干扰，保留五项小修，不将不足0.35ms的单次metadata预算当主赌注。

| 臂 | 与当前小修底座相比 | 要验证什么 | 测量预算 |
|---|---|---|---|
|09 N42|仅`SGLANG_AX_BACKLOG_INTERVAL=0→1`；普通PDI2、risk1、冷块16k保持|开场长prefill间至少给一次decode，能否降低短输出请求的首token后stall及全体TPOT均值；同时记chain代价|派发60min后排空|
|10 N46|仅常态`SGLANG_AX_SCHED_COLD_CAP=16384→12288`；backlog冷块16k与relief0保持|减少常态单次占GPU时间，改善混合负载等待；不削开场积压的16k产能|派发45min后排空，为冻结与构包留时间；与旧1h臂仅做明确范围的共同ID诊断|

[冻结脚本与替换计划](evidence/execution-0930/pod/round6-publication/plan.json)。08只在确认idle+PAUSE、原脚本SHA一致后移入cancelled并留收据；不覆盖已用任务名或38个共享runtime。此处是Pod试验，不是消耗两次正式提交机会。

### 波动和稀疏实现的审查结论

[波动独立复核](evidence/execution-0930/repeat-variance-review.json)：A1/A2有3108共同ID，TPOT均值差+0.483ms、中位差+0.044ms；236条缓存量变化。2840个相邻请求的“下一派发差－上一完成差”p05/p95仅约−1.57/+2.11ms，而同ID到达漂移已达几十秒，说明主要是闭环继承服务完成差，再放大到排队与缓存竞争。五个steady chain新坏的新增延迟集中在admit→首次batch；首batch→首token没有同量级增长。当前不能唯一指定最初扰动来自GPU、CPU或通信。开场短输出停顿在两臂高度一致，是另一个稳定存在的工程问题。

缓解应针对不可抢占prefill与已开始出字请求的等待；对照保留全部正式统计，额外按共同ID、开场/稳态、到达偏移、缓存是否变化解释。不要改harness到达/评分来制造重复性，也不要因少数噪声请求否认明确的系统性代价。

[稀疏注意力实际源码审查](evidence/execution-0930/sparse-attention-final-review.md)：既有metadata融合预算约0.31ms/整步；selected index的2051→2112重复fill/cat在trace约42.20us/整步，尚无净益实测。已有GPU有效长度可限制短行循环，但profile中0/32行短于2048，当前收益证据有限。split-KV可改善B32仅32CTA的并行度，需资格当前已有118路径及partial/merge成本；不能直接搬FP8 selected-dequant预pass到BF16 H8路径，它会新增约64MiB scratch并改变布局/语义。今晚两臂不加入这些新kernel或fusion，优先验证毫秒级prefill干扰空间。

[实际发布与恢复收据](evidence/execution-0930/pod/round6-publication/receipt.json)：发布器DONE rc=0，38个共享runtime哈希保持；[末次状态](evidence/execution-0930/pod/round6-publication/queue-status-final.txt)确认09/10均done、running/pending均空。

### 最终闭合判断与两个具体包

09对07共同3259请求：TPOT均值59.942→57.151ms（−4.66%）、p95104.193→99.485ms；chain修好2/新坏5，fast超标301→240。开场五个旧90秒左右的首token后停顿降到13–37秒，贡献约94.7%的均值改善；双方600秒后稳态共同请求均值仅50.195→49.920ms。开场51个共同chain中11条TTFT增加超过5秒，0条改善超过5秒；其中两条新坏在四个旧臂均正常，不能把chain代价全部归于波动。09适合作为TPOT候选，另留relief0包保护chain。[同ID配对](evidence/execution-0930/relief1-vs-smallfix/paired.json)。

10对06共同2541请求：TPOT均值67.767→67.006ms（−1.12%）、p95119.770→114.926ms；chain22→21（修5/新4），但p9538.042→39.055秒；fast271→271、overall243→247、turn8→11。对02原配置共同ID则chain20→21、fast287→271。06不含五小修，且10/06时长45/60min，所以这不是cold12k的隔离实验，也不是完整N46通过。较弱相对收益不足以优先占用最后两次机会。[共同ID描述](evidence/execution-0930/cold12-vs-moe-dsa/paired.json)、[原配置对照](evidence/execution-0930/cold12-vs-baseline/paired.json)。

Codex、Astra max与6.1 Sol均建议**07 BASE（执行优化、relief0、冷16k）+09 TPOT（相同执行优化、relief1、冷16k）**；10 CAP（relief0、常态冷12k/积压16k）保留为明确的容量风险备选。三个包共用成功构建的`0930a`镜像，区别只有已实测的调度配置；174/177/metadata fusion均关闭。具体配置、ZIP SHA和未提交状态见[decision.json](evidence/submission-0930-execution/decision.json)，完整独立复核见[final-results-review.json](evidence/execution-0930/final-results-review.json)。用户最终选择10+09，已上传47798/47800；没有承诺N46或正式TPOT同比收益。用户随后结束工作，追加组合试验尚未入队。
