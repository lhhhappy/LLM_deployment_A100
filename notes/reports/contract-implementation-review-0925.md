# 从赛题到本地实现的复核（Codex，2026-09-25）

用户要求重新核对定义、实现与 N38 方向，并与 Claude 新 session 协作。
本轮只做本地源码审阅、CPU 探针和离线判分修复；不改引擎策略、不改冻结 071、不向 Pod 写日志。
用户后续收窄优先级：先保证SGLang/vLLM评测一致，Claude当前阶段交接后再深度审查冻结vLLM，
以此选择后续编排和host缓存优化；两路各用原harness，不另造有利于某个引擎的评分口径。
Claude session `e4faf351-6a00-4cf3-89bb-765b4c17abe2` 已核对活进程、终端与打开的 session 路径，
消息已投递，对方已确认收到 072 排队安排并接手 vLLM 契约复核。

## 判断

当前方向成立：减少重算（180/host64）、控制 prefill/decode 的相互等待（修复122），
并以独立 vLLM 基线作对照。成立的是继续验证这些假设，不是已经证明 N34/N38 可达。
最近已核实的正式结果是 46174 N18；本地069 N30仍为有效FAIL，只有chain门未过。
二者负载和并发不同，不能互换成绩，也不能由已通过档断定正式更高失败档的瓶颈。

本轮发现并修复一个本地有效性缺口：成功请求的实际输出数没有强制对齐冻结预算。
另补齐了工具侧vLLM结构化flush收据支持；服务端插件修订与联调由Claude接续。
没有发现可据此立即更改冻结SGLang调度策略的证据。

## 定义和源码逐项对应

排名依仓库已记录的主办方确认：N@SLO → tpot_mean → TPM → 提交时间。
题面旧文仍称TPM不排名，此处以[知识记录](../knowledge.md)说明的修订为准。

| 合同 | 本地实现与核验 | 结论/边界 |
|---|---|---|
| N是逻辑会话槽，不是在飞HTTP数或引擎batch大小 | 原loadgen的chain回放保持每链顺序及gap；完整性按冻结index核每条恰好一次 | 不把max_running_requests=32解释成最高只能N32 |
| TTFT含接收、分词、准入及全部prefill等待 | `http_server._ArenaRecvTimeMiddleware`在解析body前记录perf时间；Tokenizer映射epoch；`batch_result_processor`等结果拷贝完成并拿到token后，最终prefill块才设置首token时间 | `forward_entry_time`只是首次入批，不能充当GPU开始时间；首token输出路径未把它冒充完成时刻 |
| 四桶按冻结负载分类，fast是overall子集 | `score_formal`调用原harness的`in_ttft_gate`；`level_verdict`对冻结phase/edge/uncached/index作完整性核验 | 实际cache命中不能改变分桶；统计余量不是固定加秒数 |
| TPOT为逐请求客户端首末SSE差除以输出数减一，再无权平均/p95 | 原`s1_loadgen.call_engine`；`score_formal`使用raw的TPOT并检查有限/非负；p95≤.10无余量 | 不用GPU单步耗时、TPOT倒数或跨N的比值代替排名指标 |
| 原文不再套模板；ignore_eos下精确输出原预算；缓存数真实 | SGLang原generate路径保持；本轮对042/067/068/069逐条核prompt、output及cache范围，均符合 | 计数合法不独立证明数值质量；质量仍由能力门及必要数值测试判断 |
| 测量前真flush | HTTP返回JSON布尔success；tokenizer聚合worker应答；scheduler仅fully idle时重置树/池/草稿。HiCache在途write/load会使idle为False | CPU执行真实idle方法验证拒绝在途分支。不同TP/异步DMA的完整正确性仍需相应GPU证据 |
| CPU/GPU/存储预算 | 新准入诊断TP0进程总上限8MiB，离线输出另有上限；完成run先本地校验再清Pod | 不是整个容器8MiB上限；普通server日志、编译缓存和venv仍需独立总容量账 |

源码入口：
[HTTP计时](../../engine/sglang/srt/entrypoints/http_server.py)、
[首token就绪](../../engine/sglang/srt/managers/scheduler_components/batch_result_processor.py)、
[计时字段](../../engine/sglang/srt/observability/req_time_stats.py)、
[flush聚合](../../engine/sglang/srt/managers/tokenizer_control_mixin.py)、
[scheduler](../../engine/sglang/srt/managers/scheduler.py)、
[原loadgen](../../s1-dev/harness/s1_loadgen.py)、
[本地完整判分](../../scripts/pod/verify/level_verdict.py)。

## 发现一：成功EOF不等于完成固定输出（已修复，CPU）

原harness的call_engine在收到正token计数并正常结束HTTP流后，可返回error=None，
但不比较最终completion_tokens与max_new_tokens。之前本地score_files只验证完整请求名单、链位置，
level_verdict补验分桶、runner和flush，也没有挡住这种输出缩水。
这意味着完整cohort仍可能被当作有效比较，即使实际少做decode工作；不是已发现历史引擎偷减输出。

修复在[score_formal.validate_replay_tokens](../../scripts/score_formal.py)：

- 对成功请求从冻结index读取预算，要求整数output_tokens精确相等、prompt_tokens与glm_tokens相等，
  cached_tokens为合法整数且处于[0,prompt]。
- 不信raw自己声明的max_output_i，不改原harness，不把此项增加成第12道官方SLO门；违反回放合同记INVALID。
- 原本失败的请求保留原错误分类和错误率分母，不因没有完整输出额外判INVALID。
- 完整评分报告和level_verdict记录检查收据；score_records诊断子集接口保持原义。

新增回归覆盖短输出、超输出、浮点/布尔/缺失计数、错误prompt/cache、伪造raw预算、
错误请求豁免及原loadgen默认512。另在完整L042副本里只改一条成功请求的输出数，文件入口必须拒绝；不改原始证据。
042/067/068/069共17525条原始请求重新核查并判分，现有失败门及TPOT数值均不变。
[复现脚本与SHA收据](../../evidence/contract-review-20260925/audit.py)、
[CPU结果](../../evidence/contract-review-20260925/summary.json)。

## 发现二：vLLM的flush取证（工具已适配，服务端待交接）

本轮读取到的vLLM插件在reset成功后输出`flush_cache: prefix cache reset`，
而原level_verdict.check_flush要求`[YYYY-MM-DD HH:MM:SS ...] Cache flushed successfully!`。
因此正确的vLLM服务也可能被本地判INVALID。这是工具适配缺口，不是vLLM已经不能真清缓存的结论。

Claude已回复并约定结构化收据；工具侧现在要求响应体与服务端`[ax] flush_cache JSON`一致，
API起止epoch秒被客户端flush窗口包住，成功、空闲、reset_connector=true、kv_connector显式null。
测试覆盖成功、旧/不匹配日志、缺字段、错误时序、NaN、在途请求和未支持connector；不跳过server证明。
connector非空暂记工具不支持/INVALID，等各层失效与worker drain审计闭合再扩展。
API epoch秒是时钟口径，不是缓存代际编号。服务端插件与新Mamba检查点仍待Claude交接和独立复核，
更细的源码发现及CPU探针见[R29](../../research/codex/R29_vllm_contract_and_host.md)。

另一处绑定在`compare_runs.load`：原来没有SGLang prefill批次日志就直接INVALID。
已新增`--cross-engine --data-root DATA`，在独立完整verdict之上检查同N、显式预热profile、rep16计划SHA、
成功请求冻结token合同及既有逐请求gap/分桶/身份一致性；不再读取专有batch日志。
缺失的首次入批诊断保持unknown，不记零等待；有阶段打点时仅比较两侧都有观测的同请求子集。
该工具仍是零请求错误运行的逐请求诊断；官方允许的错误率仍由level_verdict原门处理，
不能把诊断工具拒绝输入解释为官方SLO失败。它也不单独证明时间戳实现或prompt正文来源一致。

22项对照测试包括：完整069收据下无SGLang日志仍可比较、缺首次入批只留未知、
旧VALID收据不能掩盖短输出、不同N或预热计划不能冒充同条件。

## 对优化机制的复核

**180/host64：保留最高优先级的系统证据。** 068→069完整同负载只改host容量，未命中输入少37.9%，
四类TTFT均改善；不是提高命中率就必然提分的泛化结论。GLM状态复用须同时满足MLA、indexer、KDA与草稿的检查点位置。
CPU字节/分叉/flush测试、正式能力门及完整回放已经有证据；专门的GPU恢复、DMA和跨rank收据仍不能用这些替代。
cached_tokens也是逻辑前缀计数，不等于全部实际GPU token工作；回退重算要结合prefill日志账。

**122：方向与赛题一致，成本模型还欠实测。** 它依据每请求decode进度分配prefill机会，
目标τ=.085并不是官方硬门。C0/C2部分来自推断，锚点是第一次被调度决策看到的时刻，
每次有竞争的决策多一次TP CPU all_reduce，MAX_DECODE还会强制放行；这些都使它不能提供TPOT保证。
应补本配置下块长×上下文×batch的实际耗时与同步成本，而不是继续用固定公式宣称能算出N38。
071有价值，因为它检验减少重算后122整体机制是否仍兑现；不是只检验一个τ常数。

**120扫描/123排序：CPU候选继续保留，暂不自动晋级。** 当前只剩chain门，而120_scan主要放行完整device短命中，
可能进一步延后冷链首。CPU反例证明某分支可改善，不证明069坏例由该分支导致；旧069没有分支诊断，311坏例仍保留未知归因。
先短窗分清候选不合资格、整池不足、host恢复、排序和持续有用计算；不能通过提前入批把等待移到后段来宣称TTFT改善。

**请求容量：32与N38不是同一个单位。** 069启动记录max_running_requests=32、pp_max_micro_batch_size=None；
scheduler按实际请求池容量/PP数补默认，因此PP1有效上限为32。真实方法CPU探针验证在该上限下running=32时不能再准入。
N38仍可能只有部分会话在飞；是否增加该限制需先观察request_slots分支、峰值驻留和GPU池余量。
提高限制会影响图、KV/KDA容量与计算竞争，不能仅修改一个数字就认定有更高产能。

**资源利用：追求有效工作，不追求忙碌百分比。** 已有069阶段记录中HTTP/IPC通常远小于调度等待，
暂无把CPU分词/HTTP重写放首位的证据。GPU busy高也不能分辨有用prefill、回退重算、DMA等待或较慢kernel。
CPU内存用于保留可复用状态已经有收益证据；CPU核数、NUMA、搬运重叠的收益仍应按对应时间账验证。

## 下一步的依赖与预算

跨引擎还要对齐实际资源账：SGLang `hicache_size=64` 是每rank预算；vLLM
`kv_offloading_size` 的文档与CPU池公式是整个TP组总量。相同数字不等于相同host容量。
vLLM原生OffloadingConnector已存在AttentionSpec/MambaSpec搬运分支，
但MTP/EAGLE下部分尾块offload支持仍有限制；有类型分支不证明GLM完整组合已正确。
优先验证原生机制，避免把SGLang的块大小、容量数字和准入条件原样复制。

“评测一致”的验收顺序：冻结数据/正文与token → 逐请求输出预算 → 原会话顺序与gap/N →
相同预热profile和真flush → 时间戳来源/客户端TPOT → 同四桶、统计余量与完整11门。
实际HTTP到达时刻和batch不要求相同：闭环回放会因上一请求完成时间不同而自然改变。
生成文本也不要求逐字相同；两路均须独立守住质量门。短探针没有完整cohort，不能报同档成绩。

| 工作 | 可以无卡先完成 | 需要什么GPU证据 |
|---|---|---|
| 输入/输出/计时/flush对齐 | 本轮补输出有效性、复算；与Claude约定flush收据；默认关闭路径与CPU反例 | vLLM TP8真实权重接口/质量冒烟；完整能力门仍独立 |
| 071 host64+122 | 冻结配置、manifest、G_EXPECT已备 | 新Pod容量与冷编译核验后，同负载完整N30；短窗口只筛错 |
| 等待归因 | R27分类器和限额诊断已备，保留未知 | 有限TP8窗口匹配坏例与分支；需要时测恢复依赖和GPU实际忙于什么 |
| 成本与请求容量 | 整理块长/上下文/批大小矩阵及资源账 | 开发机筛单块/host开销；TP8确认通信/池容量及整体延迟 |
| 存储防驱逐 | 现有归档/清理、tmpfs布置逻辑可CPU测试 | 安装venv前实际检查根盘/JIT/容器日志和cgroup；运行时低频总量监测 |

071→072仍按已预留队列；本轮pread再次返回无Running Pod、service deploying。
所以没有启动八卡任务，也没有新增Pod日志或安装环境。tmpfs计入主机内存，不是免费磁盘；
归档watcher在无Pod时只能记录本地错误并重试，不能声称完成了线上清理。

CPU验证：评估工具18项、短预热5项、跨运行比较22项、归档10项、工作目录8项，
以及调度保护/122/123/有限扫描/诊断112项均通过。仿真使用假时钟和成本，不当性能结果。
全部新增审计输出位于本地，audit.py单次输出有1MiB硬限；原始运行文件未改。
