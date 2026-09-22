# T48 / W22 — GLM-5.3-Flash NEXTN on A100

依据仅为本仓库底包源码、公开模型配置与本轮开发机算子测试。没有访问 bohr/Trisol/pod，没有加载完整模型、启动8卡、构建镜像或提交。VERIFIED 指源码/特定测试成立；INFERRED 为尚未测量的服务影响。测试收据见 `evidence/T48/README.md`。

下文 `S/`=`build/base_exact/sglang/srt/`，`K/`=`build/base_exact/sglang/kernels/`。原始行号按只读底包；标“candidate”的行号按 `build/p160/candidate/sglang/`。完整指定栈：000→101→105→110→111→112(v2)→113(v2)→140→120→130→150→160。

## 1. 全路径（VERIFIED / source）

| 阶段 | 源码位置 | 实际行为及限制 |
|---|---|---|
| 参数入口 | S/arg_groups/speculative_hook.py:41,69,81,640,654,687,719,966 | NEXTN别名变成EAGLE；GLM不在自动draft-path模型白名单，必须显式同一个模型目录；默认steps3/topk1/draft_tokens4；未给max_running_requests才写48。48来自EAGLE族资源默认值，不是GLM固有限制。 |
| 160启动策略 | candidate/srt/arg_groups/ax_mtp_sm80.py:14；speculative_hook在auto_params后调用 | 仅GLM target + sm80 + resolved EAGLE；topk1；声明tilelang两DSA后端、BF16 KV、三阶段Triton KDA。父进程解析时设置110/111开关、关闭DeepGEMM HC/topk plan v2/101角色拆分/140；子进程继承。不改接受阈值或采样参数。 |
| worker选择 | S/speculative/spec_info.py:336；eagle_worker_v2.py:145,174,1106 | EAGLEWorkerV2 + EagleDraftWorker；target TpModelWorker与draft独立ModelRunner；draft_model_build_scope和speculative MoE backend上下文。普通路径支持overlap，非overlap同一v2实现。 |
| draft模型配置 | S/configs/model_config.py:741；configs/glm5_next.py:10 | 改architecture为Glm5NextForConditionalGenerationNextN，text.num_nextn_predict_layers=1，清text.linear_attn_config。GLM wrapper将text属性暴露到顶层，故index_share等从hf_config可读取。 |
| draft模块/权重 | S/models/glm5_next_nextn.py:24,34,62；deepseek_nextn.py:101,110,171,332,348,385；glm5_next.py:1556,1595,1640,1651 | 实际draft是DeepseekModelNextN，模块layer_id=0，checkpoint过滤model.layers.45（兼容language_model前缀）并映射model.decoder。先enorm/hnorm→fused_eh_norm concat→BF16 eh_proj→一层DeepseekV2DecoderLayer(is_nextn=True,skip_rope=True)→shared_head norm→LM head。只允许一个MTP层。 |
| quant配置 | glm5_next_nextn.py:34；deepseek_nextn.py:110,348；models/deepseek_common/utils.py:127 | 当前config为FP8 block128，modules_to_not_convert逐个列45层indexer/router/kv_b/eh/norm，**没有ignore整层45的通配项**，故不是整个draft BF16。FP8 linear自动Marlin；FP8 experts由111 Marlin。NVFP4→FP8 override只对modelopt_fp4，当前不走。Mapper保留checkpoint不量化模块列表，未改变权重。 |
| embedding/head | S/speculative/eagle_worker_v2.py:294 | 从target取embedding/head，按架构共享；draft的eh投影和一层专家仍有额外显存。不可把draft显存简单算45分之一。 |
| target结构 | S/models/glm5_next.py:632,648,688,775；s1-dev/glm_tok/config.json | target45层=34 KDA +11 DSA，mHC4；前3层dense其余MoE。draft **没有KDA，也没有target的4路mHC block**；其norm是普通NextN结构。target verify仍经过全部34层KDA与mHC。 |
| DSA draft后端 | S/speculative/draft_utils.py:159,174；S/layers/attention/dsa_backend.py:2920 | draft decode是DeepseekSparseAttnMultiStepBackend，每draft step各一个DSA metadata；draft extend单后端。TARGET_VERIFY/DRAFT_EXTEND_V2都选dsa_decode_impl，而普通prefill选prefill_impl。所以只设prefill后端不够。 |
| indexer verify | S/layers/attention/dsa/dsa_indexer_kpool.py:740,860,1466,1499,1575 | verify与draft_extend用kpool_write_plan；写所有候选tail，再按effective_n压缩已闭合pool；扩展每query长度，q转[R,1,32,128]，paged logits调用110代理→112。prefill/ragged→113/112。skip/reuse index仅免logits/topk，KV/tail仍需更新。 |
| pool plan/写缓存 | S/layers/attention/dsa/kpool_fp8_index.py:197,253,289,1139,1510,1618；kpool_plan.py | 一个验证窗可跨多个4-token pool；每query前缀长度不同；ring tail保留4+D，write plan从真实物理页表得压缩写地址。110已改该spec专用写函数的fp8指针/存储，160无需再实现量化。 |
| attention | S/layers/attention/dsa_backend.py:840,2920；K/ops/attention/dsa/tilelang_kernel.py | 先cache MLA KV，再以topk选择causal历史和tail；rope0、latent512使tilelang选择v1（禁TMA/warp specialization），没有Hopper tail分支。`dsa_backend.py:4437,4461`仅SM90/SM100/gfx95且非spec extend允许MHA_ONE_SHOT；sm80与verify/draft-extend均关闭，不绕回fa3。 |
| 共享topk | S/speculative/eagle_worker_v2.py:256,642,870,1009,1045,1103；S/layers/attention/index_topk_share.py:22,50,67；models/deepseek_nextn.py:285 | index_share_for_mtp_iteration且topk1才启用。取draft-extend最后已验证token对应的seed，width=2048+4−1=2051（model_config.py:292）。`mtp_iteration(keep_carry_seed=True)`使后续draft-decode复用同一seed；publish更新并在finally清除carry。不是复用target所有层索引，也不跨无关请求/轮永久缓存。 |
| draft循环 | S/speculative/eagle_worker_v2.py:600,638,651,696 | steps3已有draft-extend产生的第一个proposal；循环最后一步break，所以另做2个单层draft decode；topk1_postprocess argmax+positions+token写入。3个候选+1 target token组成4-token verify窗。 |
| verify/采样 | S/speculative/eagle_worker_common.py:461,563,590,613；eagle_utils.py:740,765,800,887 | target forward→原penalty/grammar→greedy或target-only采样；得到num_correct_drafts，再加1成accept_lens；commit_mamba_states_after_verify；返回predict/accept_lens/new_seq_lens。未改thinking、输出预算、tools、历史、时间戳或token计数。 |
| KDA dispatch | S/layers/attention/linear/kda_backend.py:101,283,436,717,834,914,989,1034,1075；linear/kernels/kda_triton.py:157 | prefill chunk_kda；普通decode causal_conv update+safe-gate recurrence；verify进入单独路径。T≥3且满足布局时 fused_kda_conv_gating_verify；T2/树/不满足条件回conv+fused_sigmoid_gating recurrence。safe gate lower_bound=-5要求Triton。 |
| KDA scratch | S/mem_cache/memory_pool.py:423,719,740,770,805；kv_cache_configurator.py:1070 | SpeculativeState.intermediate_ssm=[34,R+1,D,8,128,128] fp32；intermediate_conv_window=[34,R+1,D,3,3072] bf16。按最大running requests，不按全部radix槽；KDA显式禁止conv window去重（memory_pool.py:117），是dense窗。 |
| KDA回滚 | S/speculative/spec_utils.py:843,1066,1084；S/layers/attention/hybrid_linear_attn_backend.py:1267,1358；K/ops/speculative/eagle.py:95；K/ops/mamba/mamba_state_scatter_triton.py:676 | verify不提交SSM，每步写scratch；conv active暂时前移，接受后把accepted−1步SSM/conv散射到active槽，必要时把跨256边界那一步写原tracking槽；被拒绝候选不会成为下一轮初始SSM。fused accept/ReplaySSM不是本profile。 |
| CUDA graph | S/model_executor/runner/base_runner.py:423；speculative/eagle_draft_cuda_graph_runner.py:96,340,456；eagle_draft_extend_cuda_graph_runner.py:97,250,330,421；dsa_backend.py:1924,2155,2242,2561 | target图模式TARGET_VERIFY，另有draft decode与draft extend图；固定max_bs×D buffers，seed capture随图复制。图外/图内metadata残余调度仍经110代理；112/113 v2仅有限dtype/tile类别特化。算子graph通过不证明三个完整runner、overlap plan stream和TP collective集成通过。 |

## 2. sm80不兼容点与修复边界（VERIFIED/source，运行证据另列）

| 点/来源 | sm80故障 | 处理 |
|---|---|---|
| DeepGEMM MQA metadata/logits：dsa_backend.py:918,1656,2011,2561；dsa_indexer_kpool.py:765,801,920,998,1346 | sm80 Unsupported architecture；verify/draft-extend同样需要metadata | 110模块级lazy代理覆盖3入口与get_num_sms；112/113替换代理算子为软件FP8/BF16 MMA；160确保开关1。保留原metadata形状，只让sm80计算忽略DG schedule。 |
| FP8 Triton转换：kpool_fp8_index.py:1609,1668；K/ops/attention/dsa/triton_kernel.py:77 | fp8e4nv转换在sm80被拒 | 110 `_ax_fp8_view`传uint8 + `_ax_store_fp8`软件RNE；**spec多pool写入已被110覆盖**。T48测试T2/4/6及partial/pad/graph，不重复造一个160 kernel。 |
| FP8 W8A8 MoE：S/layers/quantization/fp8.py:1094,2050,2369,2428；K/ops/moe/fused_moe_triton_kernels.py:506 | tl.dot FP8无sm80 MMA | 111对FP8 block experts选Marlin W8A16、FP8 scalar type与本地专家数；target/draft均走Fp8MoEMethod，因此复用。BF16专家不强行量化。保留swiglu_limit=10路径（fused_marlin_moe.py:344）。 |
| FP8 dense：fp8.py:465,959,972；fp8_utils.py:2069 | 原FP8 MMA不适用 | 底包已自动80≤sm<89选Marlin；160不重复改。kv_b/indexer等checkpoint已标BF16，MLA absorbed BMM为BF16。 |
| fa3：S/layers/attention/dsa_backend.py:3434；F57 | qv512在sm80报Only Hopper supports different V headdim | 160同时设dsa_prefill/decode=tilelang；draft继承同一DSA策略。 |
| flashmla_sparse/flashmla_kv/trtllm：dsa_backend.py:840,858,2920；arg_groups/overrides.py:740 | 默认flashmla_sparse与kpool tail不兼容；原生kernel另有Hopper/Blackwell限制 | 当前profile全部路由BF16 tilelang；不是实现了sm80 FlashMLA。q8 sparse显式SM90检查也不会触发。 |
| mHC DeepGEMM prenorm：K/ops/layernorm/mhc.py:1009,1621；S/models/glm5_next.py:775 | target verify仍有mHC；DeepGEMM在sm80不可用 | 160设置SGLANG_OPT_DEEPGEMM_HC_PRENORM=0，使用既有TileLang/torch路径；不是只修draft。本轮M6/24×H4096×hc4的pre/post随机权重数值与graph通过；完整target block未加载。 |
| KDA FlashInfer/CuTe/PTX/FlashKDA：linear/kda_backend.py:101,283；linear/kernels/kda_*.py | FlashInfer/nvidia/CuTe需要SM100，PTX需要SM103；GLM safe gate有额外限制 | 160解析三阶段Triton；fused verify的PDL由真实is_arch_support_pdl关闭，无sm80不支持指令。Triton fallback和fused均实测。 |
| TOPK V2 plan：S/layers/attention/dsa/dsa_topk_backend.py:56；environ.py:1138 | 赛题要求A100关闭该优化 | 160固定SGLANG_OPT_USE_TOPK_V2=0；这关闭的是plan选择，不代表从不调用名字为fast_topk_v2的基础kernel。 |
| draft专属EH norm/argmax、accept/prologue/scatter | 不能由普通decode通过推出可用 | 底包C++/Triton实现按sm80编译，本轮单测；不更改采样算法。已知普通算子复用项与未覆盖项见证据矩阵。 |

本次静态检查没有发现需要在110已修spec写缓存之外新造FP8 kernel的证据。160是配置兼容补丁与测量补充，不冒称新增算子带来加速。开发机MoE小权重用33专家，真实模型为288 routed+1 shared，未验证全专家加载/分布。真实权重load、C++扩展版本与TP8是否一致仍是L2闸门。

## 3. 101 / 105 / 140 / 120 / 150交互

VERIFIED：140额外角色槽由普通Mamba allocator分配；spec intermediate是独立tensor、按request-row寻址。两者**没有已证实的物理别名冲突**。但是140仅在普通extend编排双点导出/所有权移交，configure显式拒绝spec；spec另走prepare_mamba_track_for_verify和接受后commit，缺少组合所有权/overlap/重入验证。不能删140的guard后直接开。

160在父进程参数解析时设置140=0并把101角色IDs置空。前者关闭原kernel的env分支及cache.configure，后者阻止101 admit/tail split及105对应限制。原extra_buffer ping-pong、prefix缓存、256-token验证tracking和真flush均保留。不是为了少算模型token；只放弃未验证的额外角色缓存策略。生产需重启切换，不能热改env。非GLM、非EAGLE、非sm80完全不应用160策略。

VERIFIED：120不读/写SSM scratch，改变的是get_next_batch_to_run与PrefillAdder准入。spec下其插入的一轮decode实际是一轮draft+target verify+draft extend，可能发出>1 token，时间也大于普通一步。关闭101/140并不关闭120独立单partial约束。首轮脚本固定120 off，与non-spec基线比较时也应固定off；之后单独120 on消融，观察长冷prefill/verify交替与overlap、槽耗尽、retract。尚无120+NEXTN调度轨迹的本轮端到端通过记录。

VERIFIED：150 `ax_shapes.run` 在args.speculative_algorithm非空时明确跳过。本脚本不加此warmup；harness原有warmup照常。不能把150的普通decode预热视为NEXTN预热，graph capture只预热捕获形状，服务期冷编译风险仍需L2记录。

## 4. 资源、TPOT与N@SLO估算（INFERRED；不是服务测量）

配置D=4（steps3、topk1）。每request每rank的全部34层KDA状态：
`34 × (8×128×128×4 + 3×3072×2) = 18,452,480 bytes = 17.59765625 MiB`。
R是最终有效请求槽cap（还可能被KDA主池K/ratio等限制），以下32/48对比假设主池足够、指定MR起效。verify scratch=`(R+1)×4×17.59765625 MiB`；R32约2.268GiB、R48约3.368GiB，差1.100GiB/rank。它与radix主状态池同时存在，另有draft权重/KV/indexer tail/图池/激活，不能把这张表当总显存账单。

源码 `S/mem_cache/kv_cache_configurator.py:2369` 的auto-fit按`(K+1)S+(K/ratio+1)DS=budget`解K；普通overlap extra_buffer ratio约5（实际skip-decode-lock开关会改变）。所以固定预算下D4的持久状态槽大致缩到原来的`1/(1+4/5)=55.6%`。以F59旧普通栈584槽做纯算术示例约322槽，而不是584不变；draft权重与本轮graph配置会改变预算，**不能作为预言的启动容量**。降低R不会保证自动把所有节约额重新分给Mamba槽：solver先按K/ratio解K，再用capped R核算scratch；其余可能进入KV预算。

显式R32允许N22与N26有请求cap余量，同时不为48个并发请求分配scratch。R6会严重限制后续N22/26，R48则多保留不必要的首轮scratch。真实准入还受KV、KDA主槽、prefix锁和retraction影响；32绝不是N@SLO≥26的保证。角色缓存关闭也可能加重intra预填充，抵消decode收益。

令普通目标模型每decode轮成本为C，投机总轮成本为`Cspec=Cverify(D4)+Cdraft_extend+2×Cdraft_decode+Cmetadata/accept`，`r=Cspec/C`；日志accept len=A包含target保证token。则稳态decode近似：`TPOTspec/TPOTbase=r/A`，`吞吐倍率=A/r`。**接受长度不能直接当加速倍率**。

| 假设r | A=1.6：TPOT倍率 / 吞吐倍率 | A=2.0：TPOT倍率 / 吞吐倍率 |
|---:|---:|---:|
| 1.2 | 0.750 / 1.333× | 0.600 / 1.667× |
| 1.4 | 0.875 / 1.143× | 0.700 / 1.429× |
| 1.6 | 1.000 / 1.000× | 0.800 / 1.250× |

这些r是敏感性假设，没有从2卡算子耗时外推。Marlin的4B verify、更宽KDA scratch写回和长冷prefill停顿可能使r>1.6，甚至退步。若普通TPOT假设35ms、r1.4，A1.6→30.63ms、A2.0→24.50ms；对用户给定第一名27.3ms，达标条件是`TPOTbase < 27.3ms×A/r`（r1.4时为31.2/39.0ms）。排名口径tpot_mean包含真实调度和请求分布，这些稳态数字不能直接当最终分数。

D4时A1.6–2.0对应candidate接受率`(A−1)/3=20.0%–33.3%`。采集用160新增的window tokens/rounds加权；没有这些计数时，只保留旧accept len/rate逐窗口值，不声称精确全局平均。

## 5. 8卡验证任务（仅准备；Claude审阅/执行）

`scripts/pod/jobs/dev_b160_mtp_n6.sh` 使用唯一源码名、完整12补丁栈、MR32/graph32、steps3/topk1/D4、两DSA tilelang。补充采集器需一同随scripts目录同步，脚本未入队、未改build_image/RELEASE。

1. 先核对真实45层target+45号draft权重、ignored modules映射、每rank后端、KDA主槽/verify scratch、三种graph捕获及首次ready日志。
2. eager与graph分别跑短prompt、长>2048、4-token池边界、接受长度1/2/3/4、批次增删/pad、前缀命中、跨256 tracking边界、abort/retract/并发finish、busy/idle flush。检查pool/tree sanity与flush后cached_tokens0。
3. 真实prompt上非spec/本profile（101/140两边都off）比较greedy输出/logits及能力门；随机采样比较分布/原阈值，不要求随机流逐token相同。不能以算子topk相近替代能力测试。
4. 原dev N6→10→14→18→22，用户允许后再N26。每档真flush，记录全部TTFT桶、tpot_mean/p95、吞吐、接受长度/率、decode ms/轮、ready后JIT、错误/重算/槽压力。120、130保持一致；120 on另做配对。
5. 当前脚本采集范围是整个harness运行（含preflight/warmup）；由harness测量开始/结束时间再裁剪server.log。旧lib的engine_current.log是启动时拷贝，非实时，脚本从自有engine PID的stdout找到实际日志；解析器只取TP0/无rank行，避免多rank重复计数。最后未打印的log interval不在加权总数里。

资源公式的机器可读算术表：`evidence/T48/inferred_resources.json`。JIT缓存配置的早期目录偏差、7个自有build迁移清单与纠正记录见证据README；最终开发机runner使用`SGLANG_JIT_CACHE_DIR`。
