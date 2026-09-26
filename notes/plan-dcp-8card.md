# DCP：保留 MTP，验证 N34，再推进 N38

2026-09-26，Codex。当前8卡候选分支 **`codex/dcp-mtp-stack2-n34`**，独立worktree `build/worktrees/dcp-mtp-stack2-n34`；原开发分支 `codex/dcp-mtp-n34`、起点 `aebaff56`。工程实现及两卡定向验证已闭合；尚无TP8/N34/N38通过结论。运行状态只记 [queue.md](queue.md)，闭合结果只记 [experiments.md](experiments.md)。

## 交付目标

N34，继而 N38 的 chain、turn、fast、overall、TPOT 同时过门。开发集同条件对照，完整原 harness 判分；正式档位由官方结果确认。DCP 减少重复 KV 驻留，收益必须体现为准入/执行等待下降，不能用容量倍数代替交付。

[N30 阶段记录](program-n30-v3.md)：111 的 KV 使用率 p95 99.3%、81% 时间有队列；高命中的暖请求仍长期等待。112 改善部分等待，但部分 turn 被排序推迟约120秒。开场冷 prefill、排序饥饿与 KDA 状态上限不能仅靠扩容解决。v3/v3g 时序不同，实验两臂必须使用同一冻结数据，不能跨版本比较。

## 已核实的契约与基线问题

- 当前是 NEXTN→EAGLE V2，3 steps / topk 1 / 4 draft tokens。目标45层（34 KDA、11 DSA），单层 NextN 草稿无 KDA。草稿续算提供首个 proposal，另做两次 draft decode；目标验证四个位置，接受长度1–4。KDA提交对应步骤的SSM/conv，必要时保存跨tracking边界状态。详见 [R17](../research/codex/R17_nextn_sm80.md)。
- **当前链式 MTP 不需要接受路径的 KV 压实。** `eagle_worker_common.py:run_eagle_verify` 仅在topk>1调用 `_finalize_accept_tree_path`；前缀分支复制在topk≤1返回。160本身拒绝topk≠1。move仍需工程补全，但不是保留现有MTP的前置条件。
- **基线仍有041的33/40补齐错误**，与025高地址错误不同。[041日志](../evidence/queue-review-20260924/041-server-tail.log)指向真实33行的缓存接收40行补齐K；原函数CPU探针已复现33/40及33/34，本分支按planner真实行数修复。
- **草稿容量分配按复制计费，实际写入按DCP分片。** 共享MLA写kernel仍用全局owner规则；DRAFT_EXTEND_V2却没有完整进入Q gather与LSE合并。应修齐真实布局，不能依注释认定已有完整副本。
- **HiCache是上线必需修复。** 当前guard拒绝DCP+EAGLE；DSA indexer主机池沿用latent物理size，但应覆盖完整逻辑地址。删除guard不足以获得正确性。
- 旧r7只覆盖普通cold/extend/decode与普通decode graph，未验证完整MTP的target verify、draft decode、draft extend三类图。
- **Hybrid目标池绕过了原indexer扩容factory。** 容量校验必须进入`DSATokenToKVPool`构造器，同时覆盖target与NextN；低地址数值通过不代表高位地址安全。实际indexer字节数必须与预算逐项核对。

上游 [39638](https://github.com/sgl-project/sglang/pull/39638)区分链式MTP与树式搬运，[39639](https://github.com/sgl-project/sglang/pull/39639)区分分片latent与复制式indexer主机地址。它们是draft PR、主要证据来自Hopper；只供契约核对，不能替代本栈A100实测。

## 实现与提交边界

| 机制 | 具体实现 | 必须证明 |
|---|---|---|
| 115 补齐 | 区分真实extend行数与TP计算补齐；只写真实行，缺行报错 | 奇数长度、有/无前缀、混合请求、rope=0，025高地址与041错误均覆盖 |
| 115 MTP | 目标与DSA草稿latent统一owner分片；DECODE/TARGET_VERIFY/DRAFT_EXTEND_V2统一Q gather、局部attention、LSE合并；草稿分配和计费同步 | 三阶段eager/graph数值，TileLang log2 LSE，局部空key、全mask、批次变化 |
| 115 稀疏kernel | A100的DCP4/8使用能容纳32/64头的共享内存布局；可选按KPool列同余关系去掉确定不属于本rank的列 | 真实KPool输出的索引多重集、FP32参考、每行数值和图重放；单卡局部成本与TP8通信分开计量 |
| 180 HiCache | latent主机池分片；indexer主机池用逻辑容量并计入host总预算；校验packed NextN身份/行宽/页/层映射，再开放该组合guard | 真实淘汰→异址恢复→续算，latent/indexer逐字节一致、DSA与KDA状态对应，flush释放 |
| 115/160搬运 | 从源owner取值，写入目标owner；跨rank必须通信；先快照全部源，按页内key/scale布局搬indexer | 重叠/循环置换、重复源、高地址、空rank、跨页与padding；压缩kpool/tail语义另验，不能以普通copy宣称树式MTP支持 |
| 观测与部署 | 输出实际物理/逻辑容量、草稿布局、状态槽、graph与有效并发上限；固定commit与G_EXPECT | W=1保留原路径；未验证组合明确拒绝；可复现配置和回退 |

普通extend当前会每层gather完整前缀。记录通信和临时峰值，若成为fast/turn的新瓶颈，再以同输入实测决定改造；不能仅凭NVLink带宽认定开销可忽略。

开发起点`aebaff56`的engine目录与现行服务任务引用的stack2提交不同，不能直接把此分支和136/137的成绩作单变量比较。上线候选须将115/180机制提交整合到选定的服务基线，在同一整合提交上分别关闭/开启DCP，调度与117参数完全相同；保留两份源码差分收据。A′/A″的选择按现行阶段记录闭合结果确定，不随DCP实验偷偷更换。

已准备独立整合分支`codex/dcp-mtp-stack2-n34`：服务基线`84dcca0e`加115/180得到`e4d7ca68`。引擎补丁新增/删除行与开发分支完全相同；只有README表格发生合并冲突，保留了原117/118/128条目。[整合收据](../evidence/dcp-mtp-20260926/integration-patch-receipt.json)供复核。整合分支不改共享主分支，也不自动替换现有服务。

## 容量与成本预算

每DSA层每逻辑token：latent=1024/W B，复制式indexer=132 B。按相同KV总预算、11层target+1层draft计算：

| W | 草稿复制时容量比 | 草稿也分片时容量比 |
|---|---:|---:|
| 2 | 1.68× | 1.80× |
| 4 | 2.56× | 2.98× |
| 8 | 3.45× | 4.45× |

这是算术，未计页尾、KDA、图池与工作区变化。真实启动按各buffer字节数验账；不据此推断N100。映射表约4×逻辑槽数B；完整前缀buffer约1024×本批上下文token总数B，另加collective工作区。DCP不分片KDA持久状态或verify scratch。

MTP每轮成本=target verify+draft extend+两次draft decode+接受提交；除以加权接受长度才是每输出token成本。两卡能测局部耗时和显存，TP8通信、完整权重与服务吞吐必须上Pod验证。

### MTP 对 chain/turn 的影响如何归因

130f（A′去MTP）是独立消融，由现有队列负责。本方向先固定MTP比较DCP，再用130f闭合结果决定后续组合。MTP可能通过显存占用、prefill块间插入的decode实际占时、闭环请求推进速度影响首字门；三者不能由一次开关对照分别定量。记录每轮decode墙钟时间、每输出token成本、每分钟新增prefill token、KV准入等待和缓存重算。每轮更贵不等于总成本更贵，闭环回答变快也不保证请求到达密度同比增加。

约3GiB/卡、少23万KV token和平均接受2.7均须绑定对应运行/模型配置，不能外推到DCP后的草稿布局。TPOT约0.050不能证明其他参赛者关闭MTP；关闭后本服务TPOT是否约0.045及chain/turn净收益均待130f实测。首轮target结束后的publish fence和草稿prefill、CPU出字之间的依赖需结合源码/trace确认，不预设草稿对TTFT一定没有直接成本。

## 验证顺序

1. **复现契约错误。** 原函数CPU探针，保存源码SHA256；padding、阶段路由、分页indexer move、HiCache guard均留原错与修复收据。这不是GPU正确性证明。
2. **两卡数值。** 同TP2下DCP1/2，有量纲权重，cold、prefix hit、非整除长度、高slot、空本地key。缓存搬运逐字节检查；DSA分层输出有限，相对L∞参照既有1e-2门槛并以参考重跑量噪声。不能只看缩小模型logits。
3. **两卡完整MTP。** 真实EAGLE V2路径、接受长度1–4、eager和三类图实际replay；批次增删、64/128/256/512边界、host restore与flush。KDA提交必须匹配接受前缀。
4. **两卡性能。** SM80真实64头形状，batch/上下文/候选长度矩阵；预热后重复报告p50/p95、峰值显存、通信与计算时间。输出预算有限，原始记录完整。
5. **TP8正确性。** 真实权重能力冒烟、20k/60k/190k、缓存往返、高水位、三种graph；无错误/NaN/OOM/死锁，机制与容量收据一致。
6. **N34相对筛选。** 同冻结集、同MTP、调度、host预算和有效running/graph上限（建议48，核KDA槽不压低）；仅改变DCP实现与W。60分钟派发排空，按同ID比较四桶超时、TPOT、完成量和逐请求等待。W在2/4/8中选择足够解除容量压力、总成本合适的一档。
7. **N34→N38完整验证。** 胜出配置全量闭合，用原harness与题面规则判全部门；完整性不成立即INVALID。小于重跑噪声的差异不算提升。正式提交经另一参与者对照原始记录复核，用户决定。

每次回放区分开场/稳态chain，记录turn/fast准入前等待、准入后等待、执行成本；KV与host压力、重算、KDA槽、MTP接受长度。出现错误立即结束自己的探针留证。

## 已准备的8卡配置与回退

固定候选A的warm预算5秒、饥饿上限120秒、MTP 3/1/4、host64、FP32 KDA；两臂统一running/graph48及418个KDA槽。以同一`e4d7ca68`分别跑[W1对照](../scripts/pod/jobs/dcp_mtp_stack2_w1_n34_60m.sh)和[W2候选](../scripts/pod/jobs/dcp_mtp_stack2_w2_n34_60m.sh)。两臂只差DCP size及对应`G_EXPECT`；都显式请求列选择，W1保持基线路径。评价的是整套DCP实现的净效果，不将容量和kernel收益相加。

130e5闭合后10秒warm饥饿方案被否定，因此两臂显式恢复A的原预算与120秒上限；依据见现行阶段记录。这些是准备好的配置，不代表已经入队或通过TP8。因为状态槽和graph上限也固定了，旧130/136/137不能代替这次W1对照。首次启动须核真实容量、indexer/latent字节、状态槽、117及DCP日志；列选择要核实际`DCP top-k column stride`，不能只看请求的环境变量。数据5601条/311链及哈希在任务内联核验，保留原harness、warmup、真flush和完整日志。

短测胜出后使用[W2全量N34→N38](../scripts/pod/jobs/dcp_mtp_stack2_w2_n34_n38_full.sh)，N34失败即停止梯子。它仍是冻结开发集结论，正式N须由平台结果确认。若实测需要W4/W8，先冻结对应配置并补其TP8数值/通信收据，再安排完整档。

回退配置使用同一整合提交的`--dcp-size 1`；115关闭路径保留`84dcca0e`的服务机制。正式部署前另保留已有运行镜像、提交和启动收据；发生数值、容量或机制核对失败立即停止自己的新测试，保留原始记录。正式切换经原始数据复核后由用户决定。

## 进度目标与讨论点

今天完成最小实现、两卡数值/成本、主机往返和代码审查，再准备固定版本TP8任务。下一可用8卡窗口先正确性、再N34筛选，随后完整N34/N38。“明天上线”是进度目标，正确性和服务验收不能省略。

需要依据实测讨论的取舍：DCP2/4/8的容量与通信；统一分片草稿后的接受率与耗时；完整前缀gather是否侵占fast/turn余量；v3/v3g的负载选择；DCP后剩余调度/计算瓶颈。任何降低数值门槛、改变MTP/输出语义或调整上线范围的提议，先给出证据与影响。

开发机仅在`/sjtu/linhang/arena/`工作、复用环境。Pod上传前核真实绝对挂载与20Gi临时存储预算；不停止现有服务，不抢占当前测试。短测不冒充完整成绩，源码与运行证据归档。
