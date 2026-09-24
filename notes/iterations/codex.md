# Codex 迭代索引

本页只保留当前判断、下一动作与证据；完整规则见[评估协议](../evaluation.md)，运行安排见[queue](../queue.md)。
当前单人优化SGLang，优先完整N30和四类TTFT；vLLM暂缓。允许偶尔独立代码审查。
用户提供线上榜单参照：N@SLO30、TPOT均值.046204、TPM1,557,557.2、decode TPM19,434；本地未过门样本不可直接排名。

## 当前轮：068已归档；069全量N30测量中（2026-09-24）

- 067/068同引擎759a6eb、mem0.87、新版180、MTP、311链/5601请求N30；068只关122。两轮每条恰好一次，0错误、原预算/gap一致、真flush，独立review通过。
- 067 VALID FAIL，四TTFT失败；068 VALID FAIL，fast/overall/chain失败，其余8门CP通过。turn13/13对统计方法敏感，Wilson允许12；不能当稳健胜出。
- 068 p95 fast/overall/turn/chain=8.3800/9.6509/20.8841/100.3816秒；超标/允许461/263、410/276、13/13、50/29。
- TPOT均值/p95：067 .033498/.067469 → 068 .034262/.072501；固定稳态TPM 2.788M→2.733M，关闭122未见整体优势。
- 唯一TTFT坏例616→589；fast修复282新增251、overall修复251新增260。完整[对照](../../evidence/L068-official_b_pace_off_full_n30_shortwarm/N30/compare_vs_067.txt)、[589坏例](../../evidence/L068-official_b_pace_off_full_n30_shortwarm/N30/ttft-cases.csv)、[哈希](../../evidence/L068-official_b_pace_off_full_n30_shortwarm/N30/case-list-manifest.json)；详见[experiments](../experiments.md)。
- FULL host占用中位99.903%，744样本中97.45%≥98%；只覆盖FULL，不含KDA host。evicted_tokens_total是device淘汰，dropped=0不排除host churn。
- lc139:0002在067/068均cached=0；069为75,520，TTFT265.63→8.143秒，距真LCP仍差1,792 token。缺口显著缩小，但有效状态保存/淘汰/恢复的具体事件仍未知。
- 当前069以068为基线，只改host预算32→64GB/rank，保持122off；测试容量假设，不把122off当最佳配置。KV/indexer与KDA host一起扩，8卡+256GB主机内存；CPU尺寸函数/内存余量筛查通过，独立review支持。
- 069源码759a6eb，冻结集manifest 19a7e5a6827f64a99695cba2d89b7efa2a0b05d207fc95da1568ec1d82280b2c，rep16/真flush/全量N30不变。工具2837b3c已部署并恢复队列，启动180秒、rep16预热112.832秒，20:06:39 UTC真flush后已进入全量测量。
- watch069与本地桥PID5264已启动；后台每分钟健康检查，测量t0+15/45/75分钟诊断；启动、16条原prompt长度/输出预算、同plan哈希和flush已核对；首测量dispatch 20:06:57.676 UTC，105分钟已复核；等待终态通知，若未结束则下一135分钟22:21:57 UTC。
- 启动实测device KV=1,397,760、KDA=418不变，host FULL=2,903,808 token、KDA host=23.73GB；122off/180on符合G_EXPECT。[启动收据](../../evidence/L069-official_b_pace_off_host64_full_n30_shortwarm/config-verification.json)。
- 069第105分钟冻结快照5585条，唯一、0错误、原预算/gap/时间戳核对通过，前4084条是原样字节前缀；较晚健康5594条不混入。四桶p95=2.710/3.644/13.641/40.336秒，TPOT均值/p95=.028889/.055932。
- 与068相同5585条相比，四桶超标461→222、410→195、13→7、50→31；未命中30.745M→19.067M token，TPOT均值.034343→.028889。仍非完整结果，保留选择/时序偏差。
- 新增完成1501条没有新增TTFT超标。308链已结束、剩3链16请求；82.135分钟起剩链<30，尾段不能当满N30。[105分钟审计](../../evidence/L069-official_b_pace_off_host64_full_n30_shortwarm/check105/audit.json)、[剩余请求](../../evidence/L069-official_b_pace_off_host64_full_n30_shortwarm/check105/remaining-requests.json)。
- chain累计31仍超过全量432样本CP允许29；若最终数据有效，该本地门无法通过，继续完整回放。31坏例均在前38.38分钟发出，20条queue_time≥80% TTFT，2条exec→first本身>30s；不能把执行段当纯kernel时间或简单扣队列推算可达收益。
- 当前fast修复367新增128、overall327/112、turn12/6、chain24/5；坏例数较75分钟不变，222个fast坏例中205条实际未命中≤4096、195条queue_time≥80% TTFT。[75分钟审计](../../evidence/L069-official_b_pace_off_host64_full_n30_shortwarm/check75/audit.json)、[31个chain坏例](../../evidence/L069-official_b_pace_off_host64_full_n30_shortwarm/check75/chain-badcases.csv)。
- 新坏例lc302:0017两轮cached45,824不变，TTFT.369→82.872秒，其中scheduler queue82.573秒；lc117:0032同cached71,168，.419→55.005秒。
- 源码120在partial存在时提前拒绝needs_host_load_back请求，180保留此规则；当前源码5项CPU HiCache调度测试通过，确认host候选会被暂缓；但最终cached不证明案例等待时层级或实际skip原因，仍需定向证据。[45分钟审计](../../evidence/L069-official_b_pace_off_host64_full_n30_shortwarm/check45/audit.json)、[持续坏例](../../evidence/L069-official_b_pace_off_host64_full_n30_shortwarm/check45/persistent-cases.json)、[新坏例与源码线索](../../evidence/L069-official_b_pace_off_host64_full_n30_shortwarm/check45/slow-fast-cases.json)。
- 同ID跟踪缓存量、等待/执行、prefill批与全部11门；容量无收益再补定向事件trace。缓存命中改善也不能忽略新坏例/TPOT回退。
- 067/068/069自然排空（剩链<30）分别始于97.88/99.02/82.14分钟；尾段不能抵消前段坏例，也不能单独量化满载吞吐。

## 已发现问题与处理边界

- 旧122持续碎块暂未复现：15→45分钟日志1856个prefill块，中位8064，仅5个≤64；不代表全程。
- 独立review发现预热输出校验过宽；3407498修复，067真实16条另行严格审计通过，不热改本轮运行库。
- 多N短预热收据/skip-warmup冲突已在模板修复；067单N不受影响。首checkpoint前误报无进展已修复监控。
- 发现6组历史metrics/nvidia采样器仍为孤儿进程，证据sampler-audit.json；额外开销尚未量化。
  本轮未清理、未声称瓶颈；后续修作业退出清理，避免误碰当前引擎/回放。
- 当前16请求不保证HiCache H2D恢复覆盖；只在主线出现相关异常时顺带补预热。
- compare_runs已兼容原harness缺省canonical：仍必需主hash/workload_hash，与显式cohort匹配并重算名单hash；新增逐请求gap检查和--no-pairs。18项回归、067真实自比较及独立review通过，不修改原始run或Pod。身份哈希不等于prompt正文哈希。

## 可能方向（待证据排序，不预先叠加）

| 假设 | 需要什么证据 | 下一步 |
| --- | --- | --- |
| 缓存能减少工作量（当前优先） | 全程FULL host高占用与重复缺口；尚不能归因淘汰 | 069先做容量干预，无收益再定向trace |
| 准入/批形成拖慢TTFT | 多数已完成fast坏例主要在queue；120在partial时拒绝host恢复候选 | CPU核验120/180交互，再用skip/restore事件辨因；不热改069 |
| 执行单位成本偏高 | 长链exec→first分解；同形状块成本、MoE与通信时间 | 测成本曲线，评估A内实现 |
| 122节奏过保守 | 解码余量与冷请求等待同步变化；forced-decode/实际块 | 校准成本模型，仍守TPOT尾部 |
| 测量或监控干扰 | 请求唯一性、原输出预算、flush、采样器寿命 | 先修工具；不把测量问题算引擎收益 |

## 观测与溯源

- GPU仓库：evidence/L<job>/window/保存健康与窗口原始证据；067/068已终态归档。
- 常规查询在GPU仓库运行：scripts/analysis/window_watch.sh <job> --status --changes-only；本地同名命令不读取GPU缓存，勿把本地空缓存当监控故障。
- 每分钟后台健康采样；前台只看变化与到点诊断。原始日志按请求/时间段读取，不进入常驻上下文。
- 每个run使用window_watch采样和本地window_notify桥接；后台轮询缓存，仅诊断/状态/异常变化唤醒当前会话。
  通路测试消息已由当前会话实际收到；无需Goal自动续跑。接收CLI/relay须保持运行，注册过期会拒发并留证重试。
- 全量闭合再判11门；不按5%或局部FAIL自动停。严重bug留证后只停单个job。
- 桥接通知补显示phase；启动/预热/测量状态切换原本有不同事件键，现在消息能区分。12项监控回归通过，保留去重状态重启桥，不触碰引擎。
- 当前代码核验：短预热5、评估14、队列5、调度32、122 15、窗口监控12项通过；CPU不代替TP8。
- 正式46173/46174已上传，15:13查询仍queued；成绩见[submissions](../submissions.md)，不重复提交。
