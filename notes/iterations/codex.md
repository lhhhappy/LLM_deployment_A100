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
- lc139:0002两轮cached=0，TTFT288.65→265.63秒；前驱结束至执行326.57→301.78秒。真LCP不证明有效混合检查点存在；保存/淘汰/恢复原因未知。
- 下一069以068为基线，只改host预算32→64GB/rank，保持122off；测试容量假设，不把122off当最佳配置。KV/indexer与KDA host一起扩，8卡+256GB主机内存；CPU尺寸函数/内存余量筛查通过，独立review支持。
- 069源码759a6eb，冻结集manifest 19a7e5a6827f64a99695cba2d89b7efa2a0b05d207fc95da1568ec1d82280b2c，rep16/真flush/全量N30不变。工具2837b3c已部署并恢复队列，启动180秒、rep16预热112.832秒，20:06:39 UTC真flush后已进入全量测量。
- watch069与本地桥PID5264已启动；后台每分钟健康检查，测量t0+15/45/75分钟诊断；启动、16条原prompt长度/输出预算、同plan哈希和flush已核对；首测量dispatch 20:06:57.676 UTC，首15分钟已复核；下一45分钟诊断20:51:57 UTC。
- 启动实测device KV=1,397,760、KDA=418不变，host FULL=2,903,808 token、KDA host=23.73GB；122off/180on符合G_EXPECT。[启动收据](../../evidence/L069-official_b_pace_off_host64_full_n30_shortwarm/config-verification.json)。
- 069首15分钟快照625条，唯一、0错误、原预算/gap/时间戳核对通过；较晚健康675条不混入快照。四桶p95=10.560/13.795/15.277/192.192秒，TPOT均值/p95=.043517/.084881。
- 与068相同625条相比，四桶超标90→69、95→61、8→3、33→23；未命中7.895M→5.291M token。仅已完成子集、有选择/时序偏差，不判整档或因果收益。
- fast修复61、新增40；剩余69坏例中58条实际未命中≤4096，57条queue_time占TTFT≥80%。lc002:0007仅1696新token，TTFT92.38s，其中scheduler queue91.32s；等待原因仍未知。
- 20:23:44补充样本FULL host占用99.894%、无retraction，不能单独证明容量瓶颈；[15分钟证据](../../evidence/L069-official_b_pace_off_host64_full_n30_shortwarm/check15/audit.json)、[慢例与服务区间](../../evidence/L069-official_b_pace_off_host64_full_n30_shortwarm/check15/slow-fast-cases.json)。继续全量，下一步结合坏例追准入/恢复/池预算。
- 同ID跟踪缓存量、等待/执行、prefill批与全部11门；容量无收益再补定向事件trace。缓存命中改善也不能忽略新坏例/TPOT回退。
- 068在99.02分钟后剩链<30，最后排空段不代表满N30；不拿尾窗好看抵消全量失败。

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
| 准入/批形成拖慢TTFT | 同请求recv→exec与actual prefill块；冷续块/短命中同批情况 | 复现调度反例，先CPU验证 |
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
