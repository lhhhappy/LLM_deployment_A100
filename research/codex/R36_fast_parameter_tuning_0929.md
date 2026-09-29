# Fast 首 token 调参：冻结 bf6b66fa（2026-09-29）

状态：研究、独立源码复核及两个候选job已完成，尚未入队、没有新参数性能结果。基线是47266全候选，非47043。目标是chain同ID不新增坏例、TPOT p95≤100ms前提下改善fast。

## 最新证据

原配置eznu/N38已闭合：fast 145/2289超过3秒。145坏例中142条首次执行前等待超过TTFT一半；坏例TTFT中位5.309s，recv→首次exec中位4.539s，首次exec→first中位0.414s，实际新计算量中位1174token。实际工作≤2048的106条、2049–4096的30条、>4096的9条，零缓存命中为0。时间戳不能拆分host恢复、资源资格、排序和排队，首次exec→first也不等于纯GPU时间。

同ID N34→N38：2180个共同fast，请求超标105→126，修好94、新坏115；115新坏中104缓存命中token完全相同，112条首次执行前等待占比过半。不能把全样本145与配对126混用，也不能将相同cached断言为同一设备/host层级。

MTP同ID N34对照2187个fast：105→298，修73、新坏266；新坏中143缓存命中相同、57实际工作>4096。因此MTP既有等待问题，也伴随部分缓存/工作量差异；尚不能定量归因显存下降或101关闭。

## 参数优先级

| 优先级 | 具体配置 | 源码作用 | 边界 |
|---|---|---|---|
|1|SGLANG_AX_SCHED_SHORT_TOKENS 2048→4096|120 _ax_short_hit、128p prefix readiness共享此阈值；扩大可完整执行的设备短命中资格|需device命中、无host reload、有剩余prefill预算/KV/请求槽；只可能覆盖部分30条，绝不承诺全部获救|
|2|prefill-decode-interval 2→1，同时CHAIN_RISK_INTERVAL 1→0|减少prefill间的decode轮数，风险chain和125 relief可连续prefill|两项联动组合，TPOT可能变差；保留risk1会因1<1不成立而启动失败|
|3备选|SCHED_COLD_CAP 16384→12288，BACKLOG_COLD_CAP维持16384|只在非relieved轮缩小冷工作块，留下同批余量、缩短单块阻塞|125也可能在稳态触发，不能叫纯开场/稳态切换；更多块可能伤chain，131 auto估计随块长改变|

普通间隔受125 relief、131 risk覆盖，改普通值不代表每轮都变。阈值4096是实际待算量；评测fast用冻结uncached_expected，二者不是同一分类。源码FAST_TOKENS本来就是4096，当前短命中资格才是2048。

暂不做：盲改DEADLINE_FAST_S（它只改变slack/hopeful排序，缩短反而可能更早扔进hopeless层）；关闭CHAIN_FIRST（明显有chain代价）；先降running48（可能把排队移到前端）；盲缩池400（先查真实峰值余量）；叠加MTP/DCP。当前chain_first注释中的fast有2–5倍余量已不符合47043官方事实，但不代表应直接关保护。

## 验证与产物

用bf6b66fa原ax_deadline模块实跑配置解析：normal1/risk1被拒，normal1/risk0加relief0被接受；两个job bash -n通过。未更改引擎或评分。候选job在scripts/pod/jobs/tail_rot150_n34_kda_fast_short4096_1h_ladder.sh与tail_rot150_n34_kda_fast_pdi1_1h_ladder.sh，继承N34一小时过门才N38的既有规则。

GPT-6 Sol独立源码复核确认优先short4096，发现并纠正131间隔联动限制，核实cold cap覆盖与准入限制。原始诊断与可复跑脚本在主工作区evidence/fast-tuning-0929/diagnosis.json、probe.py；probe只读闭合raw，核验原收据SHA和唯一请求数，fast桶按原harness phase_gate和冻结uncached_expected定义。
