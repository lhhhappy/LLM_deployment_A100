# Codex 迭代索引

只保留当前判断、下一动作与证据；规则见[评估协议](../evaluation.md)，运行安排见[queue](../queue.md)，完整结果见[experiments](../experiments.md)。
Codex推进SGLang与共享Pod，Claude推进vLLM；优先同一评测合同和完整N30对照，复用有效设计，不要求不同引擎生成逐字一致。开发集结果不预测正式N。

09-24 23:23 UTC：按用户授权仅提交069一份，**46251 / 0925a**，上传回执queued、job24499。源码759a6eb、host64、122off；4692项源码和实际启动配置核验通过。后续正式版关闭调试落盘、收敛高频日志、核验编译缓存磁盘预算；本次保持069原配置。[提交与日志要求归档](../../evidence/submission-0925a/README.md)。

## 当前：071 N30测量中（2026-09-25）

- 07:54:11 UTC首批正式测量派发；rep16与069计划哈希、16条正文/输出预算完全相同，106.287秒（069112.832秒），真flush成功。后台15/45/75/105分钟检查已核实，首次08:09 UTC；测量开始时8条已完成、无异常告警。已告知用户。下一步按窗口分析全部坏例，特别核查测量后新增编译、122块预算与host恢复。
- 07:48:06 UTC原PID20041就绪，adopt核验后恢复071；engine reused、122on/180on、KV池1,397,760确认，未重载权重。07:52 preflight结束进入rep16。缓存恢复共14类，真实依赖逐项核验、二进制不变，恢复进程已退出；canonical路径用于后续部署。9项JIT CPU测试通过。[恢复说明](../reports/sglang-cold-jit-0925.md)、[正式配置对齐](../../evidence/L071-official_b_host64_full_n30_shortwarm/formal-alignment.json)。
- 用户新增授权已设goal：N30完整验收无严重问题后，同配置探索N38；否则继续坏例定因和修复。窗口仍15/45/75/105…分钟，明确bug先留证再stopjob，局部SLO FAIL不自动停。暂不提交N38、不取消072。测量开始后告知用户，交给既有后台唤醒。
- TF32 warning查到实际Inductor FP32 gate投影；069日志同为False，本轮未临时改精度或屏蔽warning。preflight已出现Triton首次编译日志，需核对测量t0之后是否仍有编译，不能把冷启动影响当122收益/退化。
- 07:45容量：RAM工作区1.07GiB，根/tmp1.32MiB，cgroup332.02/1509GiB、failcnt0。07:48新增物化正文1.46GiB仍在RAM；20Gi临时盘限制不变，终态归档核验后清理。

- 07:27 UTC：确认/tmp/ax符号链接触发底包JIT依赖过滤缺陷，临时cuda.cu入依赖后消失，Marlin已串行重编6份、每份约4分钟。45分钟等待已超时；PID20041实际argv/env/source核验后adopt-wait已接回，未重载模型。后续runtime改用真实缓存路径，当前071未热改。详见[根因](../reports/sglang-cold-jit-0925.md)。

- 07:03 UTC实时复查：07:00:12已完成KV分配（1,397,760 token），07:00:13开始target verify CUDA graph；尚无PRELOAD_READY/job.log，pending。cgroup327.14/1509GiB、failcnt0，RAM目录0.918GiB，根/tmp1.32MiB。最近日志有TileLang数据竞争检查warning与Triton弃用warning，未见异常栈/OOM；编译未完成，不宣称启动成功。[快照](../../evidence/L071-official_b_host64_full_n30_shortwarm/capacity-startup.json)
- 06:58—07:02访问通道短暂EOF/SSH断开，随后Pod直读与开发机缓存恢复；pread新增可选本机直连，不修改服务。用户强调极低观测开销，已写evaluation；071不新增trace，采样频率沿用069，实际开销未量化。
- 06:43 UTC：新Pod可exec、8卡权重加载进行中；数据500MiB传输完成。短暂Trisol凭据过期后认证已恢复，未更换AK。校验脚本换行转义错误已修正，32项运行文件与全部数据哈希已通过，现等待引擎READY。
- /tmp/ax实际指向/dev/shm/arena-runtime/ax；JIT、cache、tmp与常见硬编码缓存目录已迁RAM，执行探针通过。当前RAM工作目录0.80GiB、cgroup约123/1509GiB、根盘/tmp约1.32MiB；完整临时存储占用未知，不用底层df代替20Gi限额。
- 071 watcher与通知桥已替换旧070；开发机每5分钟归档终态运行，SHA256/fsync核验后清理，活动日志不删。目的地/sjtu/linhang/arena/archives/pod-runs，位于GPFS磁盘。
- vLLM在开发机准备独立环境；Pod driver580.105.08、Ubuntu24.04.4/glibc2.39/Python3.12.3已发Claude，072仍在071之后，不同时驻留两个完整模型。

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
- 恢复update已接受，同service revision2=2103249397960679424；09-25 06:14 UTC新Pod已可exec。GPU product/8卡/资源限额/镜像/模型相同，仅env新增AX_WORKSPACE_ROOT，GPU显示名平台补NVIDIA前缀。未调用stop/delete/release。
- 共享内存754Gi计入1509Gi总内存，不能视为额外资源；8项CPU工作目录回归通过，RAM执行探针通过，JIT与临时路径已迁移。新Pod冷编译缓存须作为071与069的比较限制，不能把差异全归122。
- 下一动作：完成071运行文件与全部数据SHA核验，待已加载的引擎PRELOAD_READY后恢复队列，不重复启动模型；核机制、池尺寸和真flush后才称测量开始。步骤见[恢复交接](../reports/pod-storage-recovery-0924.md)。
- 数据manifest SHA256 19a7e5a6827f64a99695cba2d89b7efa2a0b05d207fc95da1568ec1d82280b2c；正式46173/46174不重复提交。

## 待定位问题与可能方向

| 问题 | 证据边界与下一动作 |
| --- | --- |
| 122预算与执行粒度 | 当前071单变量验证；成本模型含推断，实际TPOT才判门 |
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
- 正式46173/46174于22:58 UTC复核均终态、能力通过，分别N14/.016541与N18/.020518；这两份都为host32，未包含069扩容。更高失败档明细未知；同配置同档校准用冻结N14/N18，不用本地N30直换分数。[正式结果](../reports/official-46173-46174-0924.md)
