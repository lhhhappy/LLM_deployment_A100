# 统一评估与迭代协议

2026-09-24 用户最新决定：Codex继续SGLang，Claude Code接手vLLM，共同探索N38；vLLM先跑通开发/GPU流程并满足相同评测合同，见[交接](handoffs/vllm-claude-code.md)。两路共用以下评估口径，开发短探针不必等待固定70分钟。
赛规以task.md及已记录的主办方排名更正为准。本文件定义本地比较条件，不改变官方规则。

2026-09-25用户进一步明确：保留SGLang可提交/可回退栈，新增探索预算优先vLLM；
新的性能实验先讨论具体配置、资源/时间预算和判据。当前可继续源码研究与本地CPU校验，
不据此自行新增GPU任务或改动已冻结071/预留072。vLLM迁移技术边界见[R29](../research/codex/R29_vllm_contract_and_host.md)。

优先目标是完整本地N30全部门通过，当前优先定位TTFT与排队；TPOT历史余量不保证新组合仍有余量，所有门持续检查。正式A全部实现可改，但保留不可变tag和历史证据作对照。

## 一轮与两个评估用途

开发筛选另设固定请求/覆盖范围的短探针（例如就绪后10–15分钟初筛，非强制时长）。启动失败、OOM、状态/计数/flush错误确认后留证，结束自己的探针再修复；记录SMOKE_OK/SMOKE_FAIL，不判N@SLO。通过短测再安排下表的完整比较。SLO超时不等于程序错误；069的fast坏例持续至约68.583分钟，不能假设早期正常代表全程正常。完整测量不截成短窗来判PASS。

| 用途 | 预热 | 测量 | 结果边界 |
| --- | --- | --- | --- |
| 日常迭代 `longchain-n30-rep16-v1` | preflight一条链；固定rep16-v1的16个完整请求；真实flush | 全量311链/5601请求，固定N30，原顺序/正文/输出/gap | 全量闭合后可报本地VALID PASS/FAIL；运行中只作诊断 |
| 正式校准 | 与要复现的正式流程对齐，使用原harness预热 | 相同提交配置、对应并发档，记录数据差异 | 量化本地与正式差异，不能从本地直接换算正式N |

16请求是当前精简候选，不宣称数学意义的最小或完整形状覆盖。用户明确不单独优化预热；先保留当前方案，只有主线运行中发现与预热不足有关的异常才顺带补充。
rep16-v1定义及覆盖边界见[组合与短预热](reports/sglang-shortwarm-mainline-0924.md)。
每轮记录引擎commit、运行工具commit/实际文件、命令/env、GPU型号与TP、N、数据/cohort哈希、
预热plan SHA、清缓存receipt、raw/run/summary。不同预热协议结果不能当单变量性能对照。
本地不压输出、不关thinking、不截历史、不删tools；预热不计分，测量仍跑每一个冻结请求。

## 运行中必须观察

用户最新明确：预热完成并进入测量后，第15分钟启动首次诊断检查，随后每30分钟检查（15、45、75分钟……），不是运行期限。没有明显问题就结束本次检查并让回放继续；不因超标率超过5%、统计余量未过或一个窗口FAIL而自动停。四道TTFT原始值/坏例是重点，5%与CP仅保留参考；发生异常先定位原因和可行改进，证据明确且继续运行无价值时再留证停单个任务。

- 阶段：加载/捕图、preflight、warmup、flush、measurement、scoring，分别记录耗时。
- 每分钟只读健康快照：任务状态、最新完成量、raw/server日志更新时间、最近错误与batch、8卡采样。
- 超过5分钟没有新的完成记录，或服务日志不再推进，记为调查提示；长请求/gap/尾段也可能造成，不能直接定为死锁。
- 首次15分钟、随后每30分钟保存TTFT四桶的样本数、超标数、余量、p95；TPOT均值/p95；错误和token/cache账本。
- 当前raw只包含完成请求，没有完整在途发送台账（checkpoint的n_attempted也是完成记录数，不能当发送数）；所有运行中窗口保持open，不能因为漏掉慢请求而判通过。
- 用户要求的周期性重点检查：是否崩溃/OOM、持续碎块、明显无进展、元数据/清缓存/请求丢失等严重问题。
  证据明确时先写报告、保存原始文件和最近日志，再用stopjob停单个任务；不自动杀任务或释放8卡服务。
- 只因TTFT/TPOT超门不立即中止；先判断是否持续饱和/缓存容量/执行成本等性能问题，必要时继续到完整结果。

监控程序只读；连接失败自身重试，并记录监控不可用，不伪报引擎失败。
`window_watch.py`保存health.json、health_history.jsonl、alerts.jsonl和窗口JSON/HTML。
原metrics每10秒、GPU每5秒采样继续保留。服务日志不是完整内部trace：不能把重算差额全部说成淘汰，
不能仅凭排队说调度有错，也不能把GPU利用率100%说成高效。

## 全量判定

5601个请求各恰好一次、runner成功、本轮flush真实成功、指标齐全后，用原harness和score_formal。
成功请求的prompt_tokens必须等于冻结glm_tokens、output_tokens必须等于原逐请求预算，cached_tokens为合法整数且不超过输入；
违反回放合同记INVALID。失败请求仍按原错误率门统计，不要求失败响应也完整输出。这是本地有效性检查，不是新增官方SLO门。
判定仍为：
coverage=100%、harness_data=0、harness_render=0、engine_error<1%、infra_error<1%、
四道TTFT门、gated_phases_have_samples、tpot_p95≤0.10，合计11门。
四道TTFT使用题面统计余量；TPOT p95无余量。缺数据INVALID，有效但任一门不过FAIL。
TPM使用原评分器固定稳态窗口及其有效性检查，不能用全程墙钟平均代替。
保留全部失败请求ID和逐请求CSV，关联服务日志；同请求跨运行配对，再报告整体差异。

## SGLang / vLLM 评测一致性

用户2026-09-25澄清：目标是用同一套规范评测比较谁更好，并复用已有有效设计以减少vLLM开发和探索量。
评测工具、源码审查与机制适配可以并行；结果进入比较前核实评测合同，不要求先补齐所有SGLang功能。
Claude交接冻结vLLM提交后，由Codex独立审查。

- 共用只读原harness、冻结数据与模型/tokenizer；记录正文数据manifest、cohort、引擎和工具哈希。
  prompt原文走`/generate`，不再套chat模板；逐请求ignore_eos/输出预算一致。
- 同档对照使用同N、同链顺序、同gap规则、同rep16计划与真flush；短探针只报覆盖结果，不能混作完整档。
  闭环回放的实际到达时刻随前驱完成时间变化，不强行锁成同一串HTTP时间戳。
- 检查时间戳的真实产生位置：接收包含解析前阶段，首token须在结果实际可用后，首次入批不冒充计算完成。
  TPOT沿用客户端首末SSE与累计token；不得混用引擎内部单步耗时或token加权均值。
- 两路都用同四桶/统计余量/TPOT门、完整性和错误率检查。引擎专有日志只作诊断，缺少SGLang格式日志不能解释成vLLM性能或正确性失败。
  level_verdict现支持vLLM结构化flush：runner保存的响应体须与服务端`[ax] flush_cache JSON`一致，
  服务端起止epoch秒被本次客户端flush窗口包住，明确空闲、reset_connector=true且kv_connector=null。
  connector非空目前因tier/drain合同未完成审计而记INVALID（工具尚不支持），不能误称官方SLO失败。
  fb18e488插件与工具的无connector路径已完成CPU联调（真实ASGI/收据，假引擎）；TP8仍待验。
  不能直接跳过真清证明或将逻辑失效写成全部DMA已完成。见[冻结版复核](reports/vllm-frozen-review-0925.md)。
- 冷热缓存流程相同，实际命中可以不同，这是要比较的引擎行为；质量门各自通过，不设两路生成文本逐字相同的门。
  不要求调度顺序、检查点位置、MTP接受率、kernel、batch参数或KV/图池分配相同。
  在相同硬件和允许的CPU/GPU/内存资源边界内，各自选合适实现与分配；只在专门隔离某个机制时固定相应子预算。
  实际资源用量须记录。SGLang host预算按rank，vLLM offloading预算按TP组总量，不能照抄数字。
  首次入批等内部诊断可不同或缺失，不新增质量门；同档逐请求对照之外，最终各自按相同规则探索最高N@SLO。

已有评测、分析、有限日志与归档工具优先直接共用；缓存完整性、角色检查点、host恢复和调度经验按vLLM接口适配，
已有测试场景直接迁移。优先使用能满足需求的原生能力，只补真实缺口，不把复制SGLang实现当交付目标。

本轮源码复核、已修缺口及CPU证据见[实现审查](reports/contract-implementation-review-0925.md)。
完整且零请求错误的跨引擎逐请求对照使用：

```sh
python3 -B scripts/analysis/compare_runs.py BASE_N30 CAND_N30 --cross-engine --no-pairs \
  --data-root data/s1-dev-longchain --cohort data/s1-dev-longchain/cohort.json
```

该入口要求两侧独立完整verdict及flush/预热收据，不要求SGLang专有batch日志。
缺首次入批诊断保留unknown；时间戳语义与正文来源仍须上述源码/部署收据核验。

## 迭代与仓库入口

1. 按“活变少、活变便宜、活排得更好”分类问题。这是三个可分工方向，收益相互耦合，不相加。
2. 每轮写问题/基线/唯一变量/预期证据/资源代价/停止条件；组合轮只评价组合整体。
3. CPU或开发机先验证，TP8做确认；适时委托只读独立review，特别是评分、缓存状态正确性和调度改动。小差异且没有噪声依据时保留不确定性。
4. 谁提交任务谁跟踪、取证和分析；任务安排只记[queue.md](queue.md)，个人过程只记[Codex迭代日志](iterations/codex.md)。
5. 确认的完整实验写experiments.md；成立且可复用的结论写knowledge.md/research/；代码按机制提交。
6. 当前运行库只支持SGLang。Claude接入vLLM时先适配启动与三接口，再复用这套负载和评分；不直接复用SGLang启动器。

当前入口：README导航；queue实时调度；本页评估协议；iterations个人索引；reports长篇归因；
evidence原始证据；engine源码与机制文档。旧候选保留追溯，但不列为当前正在验证的结果。

## 上下文与轮询

- 常规只读`window_watch.sh JOB --status --changes-only`的缓存摘要；它不访问Pod。
  每分钟健康采样由一个常驻watcher负责。只有状态/告警变化或15/45/75…诊断到点才展开分析。
- 监控失联单独标明；超过180秒心跳过期会提示，不能把旧的健康快照当当前正常。
  同一连接错误在watcher终端最多每10分钟重复一次，完整时间记录留在文件。
- 不反复打印全量raw、server.log、git status或整篇文档；先看摘要，按坏例ID和时间段定向取证。
- 个人迭代索引控制在约60行：当前配置/最近判断/下一动作/候选方向/证据链接。
  长分析进入对应reports，原始日志和检查点进入evidence；更正旧结论，不层层追加重复说明。
- 压缩上下文前核对索引和队列；保留运行ID、提交号、测量起点、下个检查点、未决风险及证据路径。
- 程序唤醒：本地`window_notify.py JOB`每60秒读取GPU watcher缓存，普通轮询不进入会话。
  仅新诊断、状态或异常变化通过`agent_message.py --to codex`投递短消息；分析后结束本轮等待下一消息。
  桥接有单实例锁、持久待发消息和去重；终端提交可能在崩溃重试时重复，事件ID用于识别。
  SSH读取连续失败3次才通知，同一次失联不同报错不重复唤醒；恢复通知一次。真实服务告警与新诊断不延迟。
  当前会话已收到通路测试消息；接收CLI/relay需保持运行。进程/会话更换必须重新核对注册，不能借用旧lead。
