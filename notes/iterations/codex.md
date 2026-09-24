# Codex 迭代索引

只保留当前判断、下一动作与证据；规则见[评估协议](../evaluation.md)，运行安排见[queue](../queue.md)，完整结果见[experiments](../experiments.md)。
单人持续优化SGLang，优先完整N30和四类TTFT；vLLM暂缓，偶尔独立review。开发集结果不预测正式N。

## 当前：070启动时被驱逐；现有service revision2等待平台准入（2026-09-24）

- 067/068/069：引擎759a6ebb8e31723519ad5daf438e26e24b32501a，mem0.87、新版180、MTP，311链/5601请求N30；同rep16预热与真flush。
- 067开122、host32，四TTFT FAIL；068只关122，fast/overall/chain FAIL，没有整体优势证据。
- 069对照068只改host32→64GB/rank；KV/indexer与KDA host一起扩，8卡+256GB，GPU池不变。工具2837b3c，29个运行文件与068相同。
- 069完整5601条每条恰好一次、原prompt/输出预算/gap一致、0错误、TTFT全服务端，独立复核通过。测量6455.19秒（107.59分钟）。
- **069 VALID FAIL，10/11门通过，仅chain失败。** 四桶p95 fast/overall/turn/chain=2.7039/3.6268/13.6406/40.3355秒；超标/CP允许222/263、195/276、7/13、31/29。三种统计方法结论一致。
- TPOT均值/p95：068 .034262/.072501→069 .028824/.055902；固定[10,70)分钟有效窗TPM(all)2.733M→3.667M，decode29,532.55→41,081.13。前40分钟TPOT p95约.07311，仍需关注前段。
- 全量未命中prompt30.810M→19.132M；prefill批9707→7293、新token约30.98M→19.31M、单序列且pending批5168→2545。仅日志形态，不能证明逐请求skip原因。
- 唯一TTFT坏例589→311；同ID修复/新增fast367/128、overall327/112、turn12/6、chain24/5。一次完整对照支持保留host64继续研究，未测重跑噪声。
- [完整判分](../../evidence/L069-official_b_pace_off_host64_full_n30_shortwarm/N30/level_verdict.json)、[审计](../../evidence/L069-official_b_pace_off_host64_full_n30_shortwarm/N30/final-audit.json)、[全量对照](../../evidence/L069-official_b_pace_off_host64_full_n30_shortwarm/N30/compare_vs_068.txt)、[311坏例](../../evidence/L069-official_b_pace_off_host64_full_n30_shortwarm/N30/ttft-cases.csv)、[独立复核](../../evidence/L069-official_b_pace_off_host64_full_n30_shortwarm/N30/independent-review.json)。
- 31个chain坏例为30个cohort链首+1个非链首context_reset，均在前38.377分钟发出；20条queue≥80%TTFT，2条exec→first本身>30秒。执行段含分块/解码等待，不能当纯kernel或扣队列承诺收益。
- 新坏例lc302:0017两轮cached45,824相同，TTFT .369→82.872秒，queue82.573秒；lc117:0032 cached71,168相同，.419→55.005秒。保留为后续定位样本。
- 原坏例lc139:0002 cached0→75,520，TTFT265.630→8.143秒，真LCP77,312仍有1792缺口；LCP不证明有效混合状态存在/淘汰/恢复原因。
- FULL host占用中位99.903%，89.11%样本≥98%；不含KDA host。evicted_tokens_total是device淘汰，write_through dropped=0不排除host churn。
- 067/068/069自然排空（剩链<30）从97.88/99.02/82.14分钟开始；尾段不当满N30稳态产能。

## 070事故与恢复：现有service、新run071

- 入口scripts/pod/jobs/official_b_host64_full_n30_shortwarm.sh；源码759a6eb，host64、mem0.87、MTP、数据和rep16全部沿用069。
- 只变PACE0→.085及G_EXPECT122on；改变整套块预算/节奏，包含替代固定interval与冷块上限，不能单归因τ。
- 假设：缓存减少工作量后，prefill预算是否能减少剩余chain等待/执行；不承诺收益，不追求仅少两条临界过门。
- 比较全部5601同ID、311个固定坏例与新增坏例、四TTFT点估计/余量、TPOT与前段块时间账。CPU真实调度器47项通过，独立复核无阻断。
- 配置diff已核：[070计划](../../evidence/L070-official_b_host64_full_n30_shortwarm/config-plan.json)。工具1ba32ec部署已见RUNTIME_DEPLOYED/DONE rc=0并恢复队列；29运行文件与069相同。070尚未测量即被平台驱逐；running/startup是旧缓存，不能当实时状态。
- 平台22:13:28 UTC明确Evicted：本地临时存储超过20Gi；实际副本0/期望1。不能归因host64内存或122，也尚不知道哪个目录占满。[事故收据](../../evidence/L070-official_b_host64_full_n30_shortwarm/incident.json)。
- 恢复update已接受，同service revision2=2103249397960679424，deploying/WaitingForAdmission，实际副本0；GPU product/8卡/资源限额/镜像/模型相同，仅env新增AX_WORKSPACE_ROOT，GPU显示名平台补NVIDIA前缀。未调用stop/delete/release。
- 共享内存754Gi计入1509Gi总内存，不能视为额外资源；8项CPU工作目录回归通过，JIT暂留原路径，noexec未核实。新Pod冷编译缓存须作为071与069的比较限制，不能把差异全归122。
- 下一动作：等watch070从retrying恢复up的通知（新Pod可exec），先运行GPU仓库scripts/pod/bootstrap并强制核WORKSPACE_LAYOUT applied=true、tmpfs/mem；长链数据已补回GPU开发机并核三项SHA，Pod恢复后再还原并核哈希，然后qpush新run071/init/resume，挂watch071后结束旧070 watcher。步骤见[恢复交接](../reports/pod-storage-recovery-0924.md)。
- 数据manifest SHA256 19a7e5a6827f64a99695cba2d89b7efa2a0b05d207fc95da1568ec1d82280b2c；正式46173/46174不重复提交。

## 待定位问题与可能方向

| 问题 | 证据边界与下一动作 |
| --- | --- |
| 122预算与执行粒度 | 当前070单变量验证；成本模型含推断，实际TPOT才判门 |
| 120/180 host候选等待 | 源码partial存在时拒绝needs_host_load_back，CPU5项核实；未证明lc302实际走此分支。若仍突出再加定向skip/restore证据，不能直接删保护 |
| 长prefill单位成本 | 31个chain中2条exec→first>30秒；需要同形状块成本及调度分解，再改A内算子/graph |
| 最小预热覆盖 | rep16不保证所有并发形状或HiCache H2D覆盖；只在主线异常证据出现时顺带补 |
| 观测开销 | 6组历史孤儿采样器已记录、影响未量化；后续单独做按job清理，禁止全局pkill/误停当前引擎 |

## 观测与溯源

- GPU仓库/sjtu/linhang/arena/repo；GPU缓存evidence/L<job>/window/，本地build/scratch/window-notify/<job>/。
- 每分钟后台健康采样，测量t0+15/45/75/105…分钟或状态/异常变化才唤醒。前台读精简缓存，按需原始取证，写短记录后结束等待事件。
- 查询window_watch --status须在GPU仓库通过gssh执行；本地同名命令不读远程缓存，勿误报故障。当前codex relay通路已验证。
- 全量闭合后判11门；不按5%或局部FAIL自动停。严重bug先留证，只停单个job，8卡服务不可停删释放。
- 取证必须显式`fetch_level.sh <run> 30 --data-root data/s1-dev-longchain`。069漏传时误用旧dev集产生INVALID，保留更正收据后原数据重判；原raw/run未改。
- 完整raw/run/flush/日志与逐请求CSV留在evidence；窗口只作诊断。比较工具默认旧LCP不可套本集，使用显式cohort与--no-pairs。
- 正式46173/46174已上传；最后15:13查询queued，成绩只用official_status，不从本地推断。线上榜单N30/.046204只是参照，本地未过档不可直接排名。
