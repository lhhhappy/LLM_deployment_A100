# 150 — 启动期请求预热（T46 / W20）

> T57（09-24）：105 已并入 101，112/113 已并入 110，116 已并入 115；文中的旧编号指这些现已合并的部分。现在是 `engine/` 里的 `engine 150:` 提交（`python3 scripts/engine/tree.py mech:150` 取源码树）。

内部启动参数增加 `--warmups ax_shapes` 即启用；不加时不发送请求。
未加入 RELEASE、构建脚本、L2 队列或提交配置。公开 command 仍用中性的 `/opt/ax/serve <profile>`，参数写内部 profile。

**结论范围：启动前预热代表性分支；不能保证任意服务形状零 JIT。**
112/113 的长度是 constexpr；有限请求集不能穷尽动态调度、上下文长度、stride 和量化/MoE路由组合。
下文逐项给出覆盖与缺口，不把发出的请求长度等同于实际 kernel launch shape。

## 生命周期与失败策略

底包 `http_server.py:383–420` 在 lifespan `yield` 前调用注册表。新增 `@warmup("ax_shapes")` 惰性导入实现。
所有请求经真实 `tokenizer_manager.generate_request(GenerateReqInput(input_ids=...), None)`；不用分词、HTTP回环或假状态。
完全 drain async generator；batch 使用 list input_ids 和 list sampling_params 一次提交。结束检查真实 `finish_reason`、prompt/completion计数和指定链的 `cached_tokens>0`；不改任何返回的 meta_info。

单请求/批次最多等待1800秒（冷编译可能很慢），真清等待scheduler idle最多60秒，IPC客户端70秒兜底。
请求异常、abort、缺少输出、截断/计数不符、预期链未命中、池未恢复，都记录日志、取消自有 rid、尝试最终真清并**让启动失败**。取消也走 finally。
选择失败退出而非静默跳过：这是显式开启的启动验收；继续启动会把冷编译或污染缓存带入评分期。没有自动无限重试。

组合多个custom warmup时应把ax_shapes放最后，否则后续warmup会重新创建自己的缓存。底包随后还会跑通用server warmup（France短句/8token），150不修改它；本补丁的空池保证是ax_shapes结束时，不能声称最终Up时所有其它启动请求也没有缓存。

MTP/任意 speculative_algorithm 非空：警告并跳过整个计划，无请求、无“已预热”声明。
PD、HiCache、DP/PP、多tokenizer模式在发请求前拒绝。池验证支持当前普通TP的FULL+MAMBA、固定页池；其它池接口不兼容即失败，不能冒充已验证。
140开关必须在进程启动前确定，不能在预热时切换。101/140都未启用时仍发送角色token，但明确记录角色快照覆盖不可用，不要求其分叉命中。

**诊断日志的边界**：`srt/utils/triton_load_watch.py:19–22` 明确说明，scheduler 的 `mark_serving_started()` **早于 request-driven warmup**。
所以启动期150仍可能打印 F59 同一条“after serving started”警告；应按HTTP readiness/`ax_shapes complete`时间切分，不能单凭该字符串判定发生在评分期。
不关闭诊断、不移动时间戳、不修改编译告警；`SGLANG_CRASH_ON_TRITON_LOAD_AFTER_READY=1` 不适合这里的request预热，会使冷加载失败。

## 请求计划

合成普通token在1000…30999，互不相关请求的第一token不同；仅角色token为154827/154829。
48组共564请求、139662输入token、最多5177输出token（另加600→2的真清探针）。所有请求temperature=0，单条max_new_tokens=2；并发decode为8…12（交错结束）。仅合成预热请求ignore_eos=True，确保decode实际执行；用户请求的thinking、输出预算、tools与历史完全不改。

| 组 | 输入 / 目的 |
|---|---|
| 冷短 | 600，无共享前缀 |
| 链 | 600→1600→8600→20600，精确共享input_ids；新增1000/7000/12000。后3条要求真实cached_tokens>0（对齐回退意味着实际extend略大于增量） |
| 冷长 | 独立20000，chunk8192时跨3块；120可能改变切块 |
| 边缘 | 1/63/64/65/128/2048/2049/4096/8192/8193，涵盖小grid、尾部和113小shape回退 |
| 角色 | 每种marker各2400，在下标1024放marker；随后分叉保持前1025 token，换后缀1301。101拆分或140角色快照；两特性关闭时只算普通输入 |
| ragged | 31×257=7967总token，若同一批准入则Σceil(L/64)=155，可到NT_BUCKET=2；不能保证调度合批 |
| decode | 每个batch宽度6…32均发送一批；各prompt长64/81/98/115，输出长8…12。实际活跃batch、graph padding和eager回退由原scheduler决定 |
| 真清探针 | 清理后重发第一条600 input_ids，要求cached_tokens=0；探针完成后再次真清 |

不能声称：20000 prompt就让KDA单次T=20000；chunk8192时一般NT≤128。
`NT_BUCKET=2` 可以由ragged分批尾部ceil开销产生，但若调度拆批或120 cap较小，本次启动可能到不了该bucket。
并发6…32也不能强迫eager decode：CUDA graph可能把它pad到已捕获batch。只有禁用graph的独立配置或自然fallback才能验证这些eager形状；150不临时修改graph策略。

## 真清与 metrics

流程：初始verified flush → 全计划 → verified flush → 同prompt零命中探针 → finally verified flush。
复用 `/flush_cache` 的 `TokenizerControlMixin.flush_cache → FlushCacheReqInput → SchedulerFlushWrapper → Scheduler.flush_cache` 原路径，保留000的全worker成功汇总。
新增内部可选 `verify_empty=False`（现有HTTP/其它调用默认不变）；选true时每个TP rank在真实清理后执行只读断言，用现有`tp_cpu_group`做CPU MAX all_reduce汇总失败，再由主rank回复；失败rank也必须参加collective。避免非主TP rank没有IPC输出而被000的DP worker汇总漏掉。断言失败转成`success=False`返回。
检查 idle、请求池free==size、KV free==页对齐capacity、Mamba free==capacity、可选checkpoint池、树token/state总量及所有FULL/MAMBA evictable/protected计数为0。显式异常不受 `python -O` 影响。
原树reset、request/KV/Mamba pool clear、辅助池reset、grammar clear、metrics reset、draft clear和CUDA empty_cache均保留；无假flush、无强写free计数。HiCache在发请求前拒绝，未新增绕过主机层的路径。

`log_metrics=False`复用底包请求指标门控，预热不会增加tokenizer的`cached_tokens_total`/请求直方图。底包可选RequestMetricsExporter原先忽略此开关；150补上守卫，避免异步export记录包含预热cached_tokens。
探针检查真实meta_info为零，再真清，保证后续真实请求不能命中预热KV/Mamba。
**边界**：scheduler运行日志和realtime compute/prefill-cache工作量指标仍诚实记录启动时的实际计算；不是承诺所有Prometheus生命周期计数器归零。没有重置/篡改用户指标。首次正常请求cached_tokens仍须L2核验。

## Autotune / JIT key审计

历史静态枚举脚本已清理；[evidence/T46/jit_inventory.json](../../evidence/T46/jit_inventory.json)保留当时对候选树全部显式 `triton.jit/autotune`（含赋值包装）的枚举，记录源码SHA、函数行号、key、constexpr、do_not_specialize、heuristics。
`evidence/T46/kernel_keys.md`逐函数展开FLA、Mamba conv与112/113重点文件；其它后端/辅助kernel明确不声称服务覆盖。
这是源码库存，不是模型实际调用图。GPU收据另列真实执行的少量kernel组；完整TP8运行取值尚未测得。

以下为GLM普通TP8 **预期取值/可达性（INFERRED调度覆盖）**，固定模型KDA H=8、K=V=S=128、chunk BT=64、BC=16、varlen=True；indexer H=32/D=128。dtype/stride/对齐也是JIT key，不仅是autotune的key列表。

| Kernel（详名见枚举） | autotune key | 计划可覆盖的取值 / 缺口 |
|---|---|---|
| inter_solve_fused | H,K,BC,V,FUSE_RECOMPUTE,FUSE_DIAGONAL | 140-off：小grid(8,128,16,128,True,True)，大grid(8,128,16,0,False,False)；600/长请求及边缘长度跨`B*NT*H<=256`。140-on固定后一种；不会预热关闭路径。safe-gate由真实模型决定，不伪造另一种 |
| intra_sub_chunk | BT,BC | (64,16)，safe-gate且非融合时调用；BK128，USE_GATHER按sm80选择。unsafe-gate分支不由同一模型两头覆盖 |
| intra_token_parallel | K,H | (128,8)，仅unsafe gate非融合时；safe-gate模型此分支不执行 |
| recompute_w_u（赋值包装） | H,K,V,BT,IS_VARLEN | (8,128,128,64,True)，长/非融合；STORE_KG=True，候选BK/BV/warps由autotune选，不把搜索候选误记为请求形状 |
| chunk_gla_fwd_o | BT,IS_VARLEN | (64,True)，KDA所有extend；固定H/K/V，T不specialize |
| kda_gate_chunk_cumsum | H,S,BT,IS_VARLEN | (8,128,64,True)，HAS_BIAS/HAS_SCALE=True；USE_LOWER_BOUND由GLM配置固定 |
| delta_h / delta_h_snapshot | H,K,V,BT,USE_GK,NT_BUCKET | (8,128,128,64,True,{0,1,2})；0≤32chunks、1≤128、2>128。前两项预期覆盖，2依赖ragged同批；单配置autotuner无多配置benchmark。140-on仅snapshot版，EXPORT_SNAPSHOTS=True；否则原版 |
| legacy scaled_dot intra_sub_inter / intra_sub_intra | BC,IS_VARLEN / BK,BT,IS_VARLEN | 当前chunk_kda_fwd走chunk_intra，不调用这两个旧入口；覆盖集合为空，不需为未选路径造请求 |
| cumsum vector | B,H,S,BT,IS_VARLEN,REVERSE,HAS_SCALE | 有A_log时走上述融合gate，当前路径集合为空；若改模型去掉A_log，需补(1,8,128,64,True,False,True) |
| chunk_fwd kkt_solve | H,Hg,K,BC | GDN路径，GLM KDA集合为空 |
| l2norm / fused_norm_gate | 无autotune；D/BT/BD、激活/残差/weight/bias旗标 | D128等模型固定值；真实extend/decode带入相应stride、dtype；未枚举其它模型/融合后端 |
| sigmoid recurrent decode | 无autotune；B,NP2_T,H,HV,K,V,BK,BV及状态/门控/树/stride旗标 | 普通decode NP2_T=1、H=HV=8、K=V=128；B为实际活跃或graph padded大小，不等于6…32提交宽度。spec verify/ReplaySSM相关旗标未覆盖 |
| recurrent packed / 通用recurrent | 无autotune；H/HV/K/V、stride等（通用版还含B） | lower_bound非空时backend拒绝packed分支，走上行；packed仅模型条件允许时。GPU微测不是声明全服务使用packed |
| conv fwd/update | 无autotune；dim、全部stride、state_len、kernel_width、HAS_INITIAL_STATES、cache/bias/SILU/连续batch；update另有seqlen/NP2与spec旗标 | dim=3*8*128、width=4、history3；冷/命中覆盖state内容与运行时has-initial数据；constexpr实际是否变化以调用wrapper为准。decode seqlen1；spec/intermediate树路径未覆盖 |
| 140 store_conv | 无autotune；WIDTH,HISTORY,BLOCK和raw/state全部stride | (3072,3,256)，140-on角色/末尾快照；offset/slot是运行时数据，不会按每个角色位置重编 |
| 112 _paged | 无autotune；N,H,D,P,S,PAGE,Q/W/C/table/cache全部stride,HH,DD | 普通N1/H32/D128/PAGE64/HH32/DD128；P为表宽、S为max_context_len。计划只采样实际返回的P/S；>20600上下文和不同表宽不可能从本计划保证 |
| 112 _ragged | 无autotune；NQ,NK,H,D、全部stride,CLEAN,BQ,BK,HH,DD | 小形状回退；H32/D128/BQ2/BK64。NQ/NK是精确长度，没有有限bucket！CLEAN仅实际调用值，不伪造另一模式 |
| 113 _unpack_prefill | 无autotune；R,H,D,XR,XH,XD,BLOCK | query R=NQ/H32、key R=NK/H1、D128/BLOCK1024；R和stride精确specialize |
| 113 _prefill | 无autotune；NQ,NK,H、全部stride,CLEAN,BQ,BK,HH,GROUP,LOOP | H32/BQ2/BK128/HH32/GROUP32/LOOP4；精确NQ/NK导致新长度编译。GPU单独验证601与600差别 |
| e4m3/op math helpers | 无独立launch；随caller编译 | 不把inline helper当作额外已观测请求shape |
| 其它模型、MoE/Marlin、topk、DSA attention、量化、通信、CUDA/C++/TileLang JIT | 全库静态清单给出显式Triton定义，其它编译器不在该枚举 | 本请求流会间接运行已选路径，但router token分布、各种长度/stride与后端配置不能从CPU证明穷尽；无全栈编译零miss保证 |

此外：B>32、长上下文到模型上限、取消/retract压力、不同KV页/TP/dtype、CUDA graph外分支、MTP、PD、HiCache、LoRA/多模态/grammar、特殊采样、其它量化后端均未证明覆盖。
若目标是“任意合法请求零服务期编译”，后续需要单独修改112/113的动态维度specialization/分桶并重做数值与性能验收，结合完整服务实际launch key观测；不能仅扩充几十条预热请求就宣布目标达成。

## 持久缓存（可选，未改启动环境）

在内部profile设置（示例中性路径）：

```sh
export TRITON_CACHE_DIR=/opt/ax/cache/triton
export FLA_CACHE_RESULTS=1
```

`fla/utils.py`在导入时检测triton.autotune是否支持`cache_results`，支持时仅对用了`autotune_cache_kwargs`的装饰器启用结果缓存。
底包本就默认FLA_CACHE_RESULTS=1；某些autotuner没有传该参数，不保证进程重启免benchmark。磁盘cubin命中也不等于已device-load、更不等于进程内JIT表命中。A100新进程实测首次57内存miss/57磁盘命中/0实际编译，仍42次benchmark（operators_persistent.log）；进程内重复才全部归零。
SGLANG_CACHE_DIR亦可统一第三方缓存（`environ.py:third_party_cache_defaults`）；必须在导入/进程启动前设置路径。

镜像构建机无GPU：先在**与最终镜像相同Python/Torch/Triton/CUDA/编译选项及sm80**环境的A100生成cache，保存版本与源码SHA收据，再经Claude授权的镜像构建复制进去。
运行期仍给该目录写权限以容纳新key；不同编译器版本/源码、stride、dtype、硬件、环境选项可能失效。开发机torch2.13/Triton3.7.1的cache不能不验版本就当作底包可移植产物。本任务没有打镜像、导入任何服务环境或预置cache。

## 验证与回滚

CPU：当时的21项mock/生产方法测试，以及两个真实CPU Gloo rank的健康/非主rank泄漏/拒绝flush验证，结果见[证据索引](../../evidence/T46/README.md)。旧的一次性 runner 和生成器已清理；历史验证还包括11补丁stack fuzz=0、确定性生成、Python编译、反向整栈还原、只读base不变。
GPU：历史开发机算子测试使用隔离cache，编译hook/benchmark计数与文件SHA变化；最终cold5首次12组113次JIT miss/实际编译、102次autotune benchmark，同形状重复12组各项全0，600→601新增2编译。原 runner 已清理，[日志与收据](../../evidence/T46/)保留；这些结果不能代替TP8服务。

L2待Claude：相同配置140 off/on分别冷启动，核对启动时长和峰值内存；逐case日志+实际kernel keys；确认HTTP ready后同计划/未见过的长度/原dev请求的新编译数量；首次真实请求cached_tokens=0、metrics无预热请求；服务flush忙拒绝/闲恢复；原dev所有TTFT/TPOT/能力门。
有graph与无graph须单列；MTP跳过不算通过。保留启动期编译日志和首个正常请求时刻。
回滚：删除 `--warmups ax_shapes` 并重启即可不预热；完全回滚时反向撤150（fuzz=0验证），现有HTTP flush默认行为保持基线。
