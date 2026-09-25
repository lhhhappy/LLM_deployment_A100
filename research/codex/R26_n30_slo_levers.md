# R26 — 069 后的 fast / turn / chain 专项路线

2026-09-24，Codex。用户问题：除070/122外，怎样从工作量、单位成本、调度三个方向冲N34/N38？
用户随后要求推进无需卡的调研与开发，并强调CPU/GPU资源的有效协同。
当前由本会话独占修改120的准入诊断、对应CPU测试和本报告；不改队列/冻结运行配置。
以下候选未做新TP8实验，不是收益承诺。

## 当前事实

- 068/069同引擎759a6eb、同完整5601请求/N30、同rep16、同GPU池；只把host32扩至64GB/rank，122均关闭。
- 未命中prompt 30,809,642→19,131,690，少37.90%；TPOT mean .034262→.028824，p95 .072501→.055902。
- 069完整有效，10/11门通过，chain超标31/432、CP允许29。点估计目标30秒，p95 40.3355秒；不能把少修两条当稳健收益。
- 正式46173/46174分别N14/N18，后者带修复122、二者均host32。支持保留122候选；不同N不能比较TPOT退化。本地N30不换算正式N。
- 070启动被临时磁盘超限驱逐，无测量。071是恢复计划；本轮pread status仍返回无Running pod、deploying。071新Pod冷编译缓存是与069比较时的限制。

重新用原harness选择器和完整cohort读取068/069，桶数与超标数逐一断言等于既有VALID判定。
脚本和逐请求表：[analyze.py](../../evidence/slo-levers-20260924/analyze.py)、[summary.json](../../evidence/slo-levers-20260924/summary.json)、[badcases.csv](../../evidence/slo-levers-20260924/badcases.csv)。CSV一行是一个请求在一个超标桶中的记录，fast/overall有重叠，不能求和当唯一请求数。

| 069桶 | 超标数 | 接收→执行占TTFT≥80%的超标数 | 执行→首token本身超过目标的超标数 | 接收→执行p95 | 执行→首token p95 |
|---|---:|---:|---:|---:|---:|
| fast | 222 | 218 | 1 | 2.291s | .636s |
| overall | 195 | 179 | 7 | 3.005s | .719s |
| turn | 7 | 4 | 1 | 12.765s | 1.365s |
| chain | 31 | 20 | 2 | 35.358s | 13.476s |

这些分位数属于不同请求，不能相加。接收→执行包含准入和准备，执行→首token包含续块之间的decode/等待，均不是纯kernel计时。queue_time_s更窄：fast中195条达到80%，chain仍20条。

069 fast超标中205/222实际未命中≤4096；其最终cached不能说明到达时在GPU还是host。chain坏例30个cohort链首、1个非链首context_reset，全部在前38.377分钟发出。lc_20260924_239第0条：未命中13,812，TTFT163.799s，接收→执行161.717s，执行→首token2.082s；期间79条后到请求先开始执行。这证明等待和越过现象，不证明当时KV/状态/传输可满足其准入。

069剩余未命中量：chain 9.525M（49.78%）、overall intra 8.737M（45.67%）、turn .870M（4.55%）。这是token账，不是GPU时间比例。chain一半工作量仍需处理，小块固定成本和长上下文单位成本都值得研究。

## 三个方向可分实验，效果不独立

概念上：减少需算token；减少一次forward/每token/每个有效输出token的成本；改善准入、批形成、执行顺序。但实际存在闭环：

1. 缓存改善→重算少→长prefill占用少→其他请求更快准入、decode停顿减少。069同时改善四TTFT和TPOT正是实例，不能把收益再分别相加。
2. 块变小→打断粒度变细→fast更容易及时进入；每块固定成本付得更多→chain和总容量可能恶化。块变大相反。
3. 排序改变→驻留/淘汰和batch形状改变→工作量及单位成本也变。调度不只是重新分配等待。
4. 闭环会话更快完成请求会更早发出下一条；不同运行同ID并不代表同一到达压力。N是逻辑会话槽，不是decode batch。

单请求近似成本用于解释而非预测：

`TTFT = 接收至首次执行 + Σ各prefill块执行 + 块间等待 + 首token处理`

`块成本 ≈ 固定成本 + f(本块token数, 上下文, batch形状) + 未被覆盖的缓存恢复时间`

`TPOT ≈ (decode关键路径时间 + 被其他工作阻塞时间) / 实际有效输出token数`

MTP已在基线；其方向是降低每个有效输出token的成本，不能把“草稿更多”当作“成本更低”。122主要改块预算和prefill/decode交替，不解决所有候选资格与排序。

## 候选与验证次序

### A. host短命中的准入：优先定位，针对fast/turn

当前[PrefillAdder](../../engine/sglang/srt/managers/schedule_policy.py) `_ax_short_hit` 要求device前缀非空且`not needs_host_load_back()`；有active partial时`add_one_req`先用此条件拒绝，`init_load_back`位于后面。因此host候选可能始终错过续块旁的空位。122的short reserve同样排除host候选。

这是源码事实，但没有逐请求拒绝原因日志，不能据此断言lc302/全部218个fast坏例走了此分支。已有HiCache异步逐层恢复，不需要重复造传输机制；缺口是何时允许候选进入恢复与执行流程。

最小观测：按请求累计partial拒绝、NO_TOKEN、req-slot/KDA不足、预算不足、decode节奏等待；记录GPU命中、host命中、恢复排队/完成和首次执行。只在状态改变/请求结束写汇总，避免高频日志。

候选机制：对已经到达、host恢复后可在本轮预算内完成的短尾请求，预留真实KV/状态与传输预算，允许恢复并在事件就绪后搭车；保留单partial、资源记账、256组所有权与TP各rank一致决策。不能只删一行判断。新常驻buffer按字节折算损失的KV与状态槽。

CPU反例：partial+device hit、partial+host hit、host传输未完成、KV不足、恢复失败、跨256边界；随后开发机恢复数值，最后TP8同ID对照。成功判据是被这一原因阻塞的请求等待下降且chain/TPOT无实质回归。

### B. 冷请求的排序和完成机会：针对chain，并保住fast

当前LPM按绝对命中长度排序，并非按剩余执行时间排序；已有123按`剩余token - aging×等待秒数`排序，但只影响准入，不抢占正在续算的长请求。旧037c/037d为另一基线上的对照，当前host64+MTP没有独立收益证据。

先测现有123作为单独变量。若其不足，再设计按实测秒数估成本的排序，纳入上下文、host搬运、块数；随等待增加优先级，给长冷请求保证进展。若需要deadline，必须来自服务端可见信息或正式提供的SLO，不能使用未来轨迹/冻结评测标签。

另一独立候选：当前短命中保护也要求`prefix_indices>0`；无前缀但很短、可一次算完的冷请求，即使有剩余预算也不能在partial旁搭车。可尝试允许这种完整请求进批，严格不引入第二个partial。不与排序或host准入同时改。

不要直接做多长请求轮转：目前只有一个chunked_req；增加partial会占更多KV/KDA，可能把工作重新挤回host并引入额外恢复。

### C. 降低每块固定成本：170/host dispatch，连接fast、chain和TPOT

R10开发机45层TP1替身在P≈98k/c1024：profile步251ms，其中kernel100ms、GPU空闲约151ms，以CPU发射间隙为主。它不是当前TP8可兑现的151ms收益。

170 v2已修BCG/scatter捕获布局，TP2有数值证据；TP8、MTP、HiCache组合仍未完成验证，不能直接开到主线。[机制与限制](../../engine/docs/170-glm-bcg-prefill.md)、[R10](../claude/R10_prefill_fixed_overhead.md)。

方向是让1024/2048等小块便宜，缩短不可打断的执行时间，同时减少chain付出的累计固定成本。若图兼容性阻塞，可独立优化dispatch/临时张量/固定元数据准备。先同形状测chunk×context×batch成本、额外显存与decode步时间，再做真实权重TP8；图池减少KV的损失必须入账。

[Sarathi-Serve原论文](https://www.usenix.org/conference/osdi24/presentation/agrawal)以chunked prefill和与decode合批处理吞吐/时延取舍。这里只采用粒度与批形成思路，不移植其速度数字；我们的122并不是完整的stall-free mixed batching。

### D. 长prefill的GPU单位成本：chain的产能路线

小块偏host开销，大块应分别测MoE、DSA indexer、通信。R25已有同TP8每卡形状的Marlin/BF16约80TF量级，换成BF16不会自动解决窄矩阵问题。

依测量结果逐项考虑：减少MoE中间清零/读写；Humming或更合适的grouped GEMM；EP专家放置以改变矩阵形状；长上下文indexer。后两项需要通信/路由和共享专家验证。114行切分已在A，不能算作新收益。

Marlin权重加载后是私有布局，双留另一格式可能每卡增加约38GB（R25按形状推算），不能无预算声称小batch/大batch随意分流。任何kernel局部快多少都要按实际关键路径占比核算，不能直接宣布整档快多少。[R25](../claude/R25_moe_sm80_path.md)。

### E. 进一步减少工作量：容量与有效检查点

host FULL中位占用99.90%，是容量方向的线索；不是host churn的逐请求证明。host64→96/128是可筛选候选，按8rank分别多256/512GB总host预算；新Pod的RAM工作目录也算在同一cgroup，启动峰值和稳态须实测。不能从32→64两点推下一个收益。

同时观察KV/indexer与KDA host各自压力，再判断固定总预算下重分池是否优于扩容。当前指标不含KDA host占用；GPU mamba usage低不证明host状态池空闲。

turn专项还应核对角色边界的真实token LCP与可恢复KDA检查点。MTP当前关闭101/140，140也拒绝HiCache，不能直接重开；需要按MTP已接受token的位置保存、恢复完整混合状态且服从256网格。没有量化缺口前不优先做大改。[180契约](../../engine/docs/180-hicache-glm-dsa.md)、[R20方法](R20_true_lcp_attribution.md)。

## 面向N34/N38的实验结论标准

先完成host64+122对照；随后分别筛选host准入、123/短冷完成、小块固定成本。容量与kernel是另一组单变量候选，组合最后评价整体。

只有在每会话响应和gap相近、负载结构不变的粗略预算假设下，30→34/38可看作需求增加约13.3%/26.7%；它不是容量外推，更不是从正式N18到N38的保证。缓存抖动、闭环到达和batch效率都会改变比例。

先完整本地N30有余量，再固定配置走N34/N38；每档真flush、原harness、完整同ID，分别看前段与稳态和四TTFT/TPOT。原样正式N14/N18校准独立进行，最终N只认正式测量。当前已通过桶的p95余量fast仅.296s、turn1.359s，不能只抢救chain而耗光其余门。

本报告新统计由原始raw计算并与既有判定校准；候选机制仍需后续原始证据复核与分层实验，未作为部署结论。

## 用户补充：CPU/GPU全部资源的高效协同

资源利用是上述三条的共同约束，另立一张资源账；目标是完成有效工作，不是让所有核的利用率接近100%。

| 资源/阶段 | 要看什么 | 先做什么 | 判定边界 |
|---|---|---|---|
| CPU调度与发射 | 每rank主线程时间、run queue、kernel launch相关性、collective前的rank到达差 | 分析已有trace；把分词/日志等可并行工作与前向关键路径分开，检查NUMA亲和 | 88核空闲不代表关键Python线程没有瓶颈；盲目加线程可能放大TP慢rank |
| GPU计算 | EXTEND/DECODE步墙钟、kernel区间并集、MoE/DSA分项、MTP有效接受token | 按token×上下文×batch测曲线，先定位小块host-bound还是大块GPU-bound | 100% busy不能区分有效计算、重算、padding、搬运或通信等待 |
| 主机内存与H2D/D2H | KV与KDA分别占用/淘汰，恢复开始到就绪时间，与计算重合多少 | 先核池尺寸和生命周期；仅对已经到达的请求研究提前搬回 | 已有kernel backend搬运会占SM/带宽，不能当作免费DMA |
| GPU显存 | device FULL/KDA实际容量、活跃/可淘汰/锁定量，graph/workspace占用 | 每新增缓冲先列字节与少掉的KV/状态槽 | 稳态利用率低不代表峰值、checkpoint池或可回收量低 |
| TP通信 | 每rank时间差、通信与计算覆盖、分片矩阵形状 | CPU读源码/旧trace，开发机按每卡形状筛选 | 单卡替身不能验证TP8同步、链路与最慢rank |

用现有prof_ledger重新分析R10保留的45层TP1原trace：[CPU复算输出](../../evidence/slo-levers-20260924/historical-resource-ledger.json)。
一个1024-token EXTEND的GPU span为250.8ms，kernel区间并集99.6ms，区间内无kernel约60%；这是旧替身profile，非069或TP8性能。
prof_ledger按名字的分类有40% other，所以本次不拿它的组件占比做优化排序；R10的CPU发射归因来自另外的launch关联分析。

## 分层工作包：哪些现在做，哪些需要卡

划为7个工作包，全部可以先做CPU调研；前4项能先完成工具/逻辑实现与CPU验证，后3项的性能开发核心需要开发机GPU。任何服务收益最终均须TP8完整回放确认。

| 工作包 | 无卡先完成 | 开发机GPU | TP8 | 本轮状态/下一动作 |
|---|---|---|---|---|
| 1. 准入原因与资源账 | 真实分支诊断、原始记录统计、日志解析 | 可做开销筛选 | 短窗实采与诊断开销对照 | 已实现120默认关闭诊断与解析器，已过CPU回归 |
| 2. 排序与完整短请求准入 | 123已有代码/CPU测试；构造cold/host/device、KV不足与单partial反例 | 可校准成本曲线 | 完整同ID、四桶与TPOT | 下一单变量优先候选；不把模拟宣称加速 |
| 3. host恢复的准入与生命周期 | 真实缓存+假DMA验证锁、预算、就绪事件和失败回滚 | 真传输/混合状态恢复数值 | TP8就绪一致性、吞吐与tail | 先用工作包1区分host拒绝和资源不足，再设计最小改动 |
| 4. KV/KDA与host容量分配 | 池字节、峰值保留、256网格、真实LCP缺口账 | 可测搬回成本 | 启动峰值、长期churn、完整SLO | 64→96/128或重分池是独立候选，不按满占用直接判淘汰 |
| 5. 小块固定成本/170 | 旧trace、图状态契约、MTP/HiCache接口审查 | TP1/TP2数值、块成本与图池字节 | scatter/collective真实权重与回放 | 优先GPU研究方向；兼容性未过不能部署 |
| 6. MoE/DSA大块路径 | 张量形状、字节搬运、权重布局与探针输入 | kernel数值+时间，Humming/EP形状筛选 | 通信/路由/共享专家+整档 | 已有R25，不重复泛查kernel；先补分项实测 |
| 7. decode/MTP与通信 | 接受长度、每有效token成本、调用/通信依赖审查 | 小batch kernel与draft/verify计时 | 真实接受率、rank差与TPOT | 保留现MTP，依据每有效token成本再选变量 |

当前首个代码交付属于工作包1：
- [引擎诊断](../../engine/sglang/srt/managers/ax_admission_trace.py)及120原准入分支埋点；`SGLANG_AX_ADMISSION_TRACE=0`默认关闭，`120_trace`启动核验。
- [解析器](../../scripts/analysis/admission_trace.py)输出逐请求CSV与JSON；重复waiting快照不重复计数，缺日志保留unknown，重准入分episode。
- [调度回归](../../tests/test_admission_trace.py)、[解析回归](../../tests/test_admission_trace_analysis.py)覆盖真实逻辑、预算不变、开关/TP0限制、host与KV区分、未扫描队尾、周期快照、重入和累加口径。
- 未开启任何线上诊断或改动071的冻结配置；新日志的真实分支证据及开销均待TP8。
- 按用户“不要再打爆20Gi临时盘”约束，新增诊断设TP0进程生命周期8MiB总预算（每行预留128字节前缀），到限报告一次后停止采集/输出；解析器显式标记不完整观测。此限制只约束新增诊断，不能替代JIT/cache/工作目录的全盘容量核验；本轮不在Pod生成profile或上传新源码树。

建议顺序：CPU工作包1收尾→准备2/3最小反例与候选→开发机筛5/6；TP8恢复后先完成071，再协调诊断窗口和单变量确认。各项可以在准备阶段交错推进，但每次完整性能实验只改一处。

用户随后明确清理方式为“运行完移到本地存，Pod清理”。已实现独立的
[本地归档维护](../../scripts/analysis/archive_completed_runs.py)与[Pod校验/清理](../../scripts/pod/archive_run.py)，
每5分钟检查终态目录，逐文件读回本地，不在Pod生成第二份归档；本地SHA256+fsync后再次核对源manifest再清理。
活动日志、打开fd/cwd、其他运行的链接引用和文件变化均保护。它不改变冻结实验配置，也不负责清理JIT/模型。

本地维护进程已启动（每300秒，带互斥锁），状态在`build/scratch/archive-maintenance/status.json`。
首次连接返回service仍deploying、无Running pod，当前retrying，没有删除任何Pod文件。

## 用户提供的vLLM watermark脚本带来的可验证问题

用户贴出的脚本是固定KV余量的演示扫参：预留0/2/5/10/15%，观察抢占、吞吐与延迟。
实际默认值是Qwen、输入1000/输出5000、并发128、关闭prefix caching，注释开头的300/4000/200不是实际默认。
它针对decode增长引发过度准入→KV耗尽→抢占重算；不能证明其他参与者实际运行过或部署了这些参数。
`noisl`可能对应取消完整输入长度预留，但只有命名和该注释，仍不是其启动配置证据。

复查我们完整server.log，仅算测量时间窗内TP0的`KV cache pool is full. Retract requests`：067=3次/3请求、068=0、069=1次/1请求；
069在约24.569分钟发生，日志记释放2944 KV token（不是重算token量或可省耗时）。
[原始行与统计](../../evidence/slo-levers-20260924/retraction-audit.json)。这不支持当前存在频繁抢占抖动，因此不直接照搬固定10%/15%预留。
此前10秒采样的`num_retracted_reqs=0`不能推出全程没有回退：源码metrics_reporter在上报后清零该计数，短暂事件可能漏采。

更值得实验的是完整输入预留、当前块需要、后续decode增长三者的差异；太保守会排队，太激进会导致KV/KDA不够。
069实际`max_running_requests=32`；N34/N38前须检查request-slot约束是否变成新瓶颈，不能把逻辑会话N直接设成max_running_requests，
也不能不核KV/状态预算就增加上限。新增诊断已经区分request_slots、KV预算和未扫描队尾，先用它收真实证据。
