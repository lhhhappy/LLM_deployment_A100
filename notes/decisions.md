# decisions.md — 决策记录（最新在上）

格式：日期 · 决策 · 理由/证据 · 决策人。被推翻的决策不删，写新条目并注明 "supersedes #n"。

| # | 日期 | 决策 | 理由 / 证据 | 决策人 |
|---|---|---|---|---|
| 37 | 2026-09-22 | T48/160仅GLM+sm80+NEXTN/EAGLE topk1启用兼容配置；复用110–113，双DSA tilelang/KDA Triton；父进程关闭101角色IDs与140，保留原verify rollback/extra_buffer；首个未执行脚本MR32、120/130 off| draft只有DSA/MoE且target verify仍有KDA/mHC；140与spec scratch无已证实别名冲突，但140显式拒绝spec且组合生命周期未验证。MR32相对48少约1.10GiB/rank D4 scratch；原日志新增tokens/rounds才能精确加权接受长度。R17、patch160与evidence/T48；不代表服务/能力/SLO通过，L2须Claude审阅 | Codex W22（T48实现范围） |
| 36 | 2026-09-22 | T46/150采用lifespan请求预热与全worker verified真flush；显式启用后请求/池验证失败即终止启动，MTP跳过；有限shape覆盖不承诺服务期零JIT| 避免静默带冷编译/残留KV/Mamba进入评分；112/113精确constexpr长度证明有限请求不能穷尽。保留原诊断/计数，log_metrics=False并修exporter守卫；P150-01…05/evidence/T46，L2待Claude，未入RELEASE | Codex W20（T46实现范围） |
| 35 | 2026-09-22 | T45/140叠加101/105，以SGLANG_AX_KDA_DUAL_SNAPSHOT启动开关替换split为双点fp32导出；额外角色槽可失败降级、tail先淘汰；开启固定非融合intra，关闭走原kernel字节| 跨small_grid阈值曾有8.535385e-5误差；固定路径后8组GPU逐元素一致，含7组有效角色边界。额外槽经请求→树移交，21CPU、32×30轮off调度一致；完整8卡验证另交Claude，未入RELEASE/队列。补丁说明与evidence/T45 | Codex W19（T45实现范围） |
| 34 | 2026-09-22 | T44选独立113叠加112：prefill先解码q/K到bf16，query-major 2×128、query复用4tile、GROUP32/4warps/stages1；保持fp32输出、110语义及112 decode/小形状回退| 190k扫描更大tile/更多warp较慢或spill，stages3无稳定收益；最终六档131–186等效TFLOPS、222数值/30graph及全栈通过。额外scratch110.39MiB；不改RELEASE，回滚反向113。见evidence/T44与F65 | Codex W18（T44实现范围） |
| 33 | 2026-09-22 | T43/112采用Triton uint8软件e4m3→bf16解码和bf16 MMA；decode页64/4warps，prefill BQ2/BK64/4warps（H32）；保留110每头bf16舍入、负页映射0和clean=False全宽语义，不修改tilelang入口| 现有tilelang依赖FP8 GEMM且限N=1；48组tile与后续布局/warp实测发现大tile寄存器溢出，小tile较稳且剪枝更细。evidence/T43；最终88组验收和整栈验证通过。无运行期autotune，回滚反向撤112 | Codex W17（T43实现范围） |
| 32 | 2026-09-22 | T41/120 默认启用普通 TP 调度保护：续算先保留对齐的 2048 token 上限预算，剩余预算按原 LPM 接完整短命中；prefill 后有存活 decoder 才交替一次；不启用 mixed、不重排 LPM。环境变量关可逐决策回退 | 保留原续算必入/槽位/KV 记账与 101；给已准入长请求严格进度保证，避免短请求无限抢走预算。27 CPU 测试与 3617 文件 py_compile 通过；SLO 收益待实测，patch 说明记录条件上界及原 LPM 排队饥饿边界 | Codex W15（T41 实现范围） |
| 31 | 2026-09-22 | T42/130 用每个 TokenizerManager 的单线程池执行原始完整分词；所有 regular text（含小输入）串行移出 loop；暂不加入可选前缀缓存。body routing_key 优先，再 Routing-Key 头，再 Session-ID 头 | 避免 HF padding/truncation 共享状态竞争；取消后保持槽位直到实际完成；完整分词天然避开前缀末端 BPE 合并风险。130 默认开启，SGLANG_AX_ASYNC_TOKENIZE=0 恢复同步分词；动态批处理保留原策略。CPU 测试/evidence/T42，L2 待 Claude 审阅安排 | Codex W16（T42 实现范围） |
| 30 | 2026-09-22 | 新主线（research/claude/base/00-summary-mainline.md）：M0 先证 A100 上 DSA 后端能跑（L1 装 base_exact + 带 DSA 替身）；M1 调度保护链中间请求（冷启动分块独占 GPU 是源码里的主瓶颈）；M2 KDA 双点 fp32 快照（照 vLLM#56960 在 kernel 内导出）+ 淘汰优先级，101 为过渡；M3 分词移出事件循环 + 路由键接入（#31170 思路用于单调度器会话感知排序）；M4 MTP；DP8 暂缓 | 四份底包源码地图（01–04）+ F53/F54；DP8 与负载不匹配 | 用户+Claude |
| 29 | 2026-09-22 | 底包 SGLang 以 `build/base_exact/`（= 公开提交 fe236ea6c3 + 两处多模态修复，4686 文件指纹全对）为唯一依据；v0.5.20 线（001/002/003/004、SPF/HRRN、模拟器结论）退役，只保留为 L1 替身参考；补丁一律照底包源码写 | F53：底包与公开提交逐字节一致；v0.5.20 补丁打不上底包，模拟器已失去可信度 | 用户+Claude |
| 28 | 2026-09-22 | 撤下 35 项纯调参 L2 任务（归档 tests/queue_archive/param_sweep_0922）；L2 只排"针对已测瓶颈的机制补丁"：先用 01–05 诊断 B 卡在哪道门、TTFT 花在哪、缓存命中率，再做 102（对话边界记状态、不拆分预填充，参考 vLLM#56960）、DP 亲和路由（X-S1-Routing-Key，参考 SGLang#31170）、MTP 等；参数只作为机制的配套 | 参数与 D1/缓存耦合，单项效果多在一档噪声内；用户判断纯调参到不了第一 | 用户+Claude |
| 27 | 2026-09-22 | L2 改为"底包 + 与提交镜像相同的补丁（000/1xx）+ 提交原文命令/环境"，统一入口 `scripts/l2.py`，入队即由 Claude 批准、自动排队；旧 5 项（v0.5.20 补丁、SPF）退役到 tests/queue_archive | 旧项在底包上无效；用户要求提测与正式提交同一份代码、Claude 可直接批准排队、新想法可一条命令提测（tests/L2.md） | Claude |
| 26 | 2026-09-22 | 定义三级验证体系：**L1 2 卡自测、L2 提测（Trisol 8 卡）、L3 正式提交**；逐级晋级，每级只回答它能回答的问题（tests/TIERS.md） | 用户要求把"测什么、测出来算不算数"讲清楚，保证提测结果和正式分数对得上 | 用户 |
| 25 | 2026-09-22 | **正式提交的批准权也交给 Claude**：只提交自测证明有效的配置，每天 2 次（稳定线 + 进取线） | 用户指示；GPU 机的 bohr、trisol、playground 已由用户登录完成 | 用户 |
| 24 | 2026-09-22 | T29 mixed-only诊断提高context到262144、KV池到524288；权重/补丁/page64/chunk8192/extra_buffer不变，旧E2矩阵不重跑、不混性能比较 | cold_heavy完整prompt达256733，131072配置在发送前拒绝；不能截断。IF-08与D1-10已在原配置取得收据；新目录单独补N4及池回收。初次D1-08工具漏查回放完成，旧raw中的pass作废，修工具并保留失败史 | Codex main（T29范围内诊断配置） |
| 23 | 2026-09-22 | **自测队列项的批准权交给 Claude**（由 Claude 放 APPROVED）；自测与提交两个守护进程**迁到 GPU 开发机**（/sjtu/linhang/arena/repo，tmux 常驻），以免合上笔记本后停掉 | 用户指示；本容器跑在用户笔记本上 | 用户 |
| 22 | 2026-09-22 | 〔已被 29 取代〕T25/003采用独立调度session字段、首见30s/续轮actual-miss 3或5s的保守代理；EDF/least-slack/weight2均默认关闭，交付草稿而不替代SPF默认候选 | F46/R12：仅header+cache不能判断turn/reset或冻结fast标签；896次固定R10 MODEL OUTPUT未显示胜SPF。CP条数余量不换算成固定秒数目标；live验证仍需SLO-08/09。本条不撤销I6 P0研究优先级 | Codex W12（T25实现范围） |
| 21 | 2026-09-22 | **每天 2 次正式提交，两次都要用上；每天准备两个镜像**：A = 稳定线（只含已通过审阅和验证的补丁），B = 进取线（加上当天最新的补丁）。自动提交守护进程每天上限 2 | 用户要求；用户确认每日额度是 2（较早的平台提示写的是 3，以用户确认为准） | 用户 |
| 20 | 2026-09-22 | **I6（SLO 感知 / EDF 调度）升为 P0**，与 D2 合并推进：优先保护链中间请求，允许链首在正式统计余量内排后 | 公开的 N=22 成绩（F40）：链中间门远低于阈值，链首 61s 仍然通过，这就是"牺牲链首、保链中间"的做法 | Claude（依据 F40） |
| 19 | 2026-09-22 | E2采用三组（原stock / D0+D1关闭 / D0+D1开启），统一context131072，先无trace回放再独立导出raw logits | reminder_heavy包含92236-token完整prompt，E1的65536不足；不截断、不换qfull权重。D0在两个补丁组相同，增加原stock排除“关闭仍改变缓存”。数值导出会同步GPU，不混作时延证据；当前冻结D1 v1.1 sha60f98ced，HiCache禁用，作者缺陷单独交接 | Codex main（T13/E2实施） |
| 18 | 2026-09-22 | 〔已被 29 取代〕**D2（调度，移植 #40024 SPF）升为与 D1 并列的最高优先级，立即开始写代码**；D1 继续做同引擎 A/B，在拿到实测之前**不宣称 D1 能提高临界档** | W5 校准（F38）：在拟合公开包络的参数区间里，调度策略的影响远大于 D1：FCFS 的临界档约 6–10，SPF 约 14–22；D1 让 prefill 总量少 22%，但模型里临界档没有稳定提高（fast_intra 主要卡在排队）。v0.5.20 默认就是 fcfs（`schedule.py:89-104`）。这是未经实测校准的模型，只用来排优先级，不作为结论 | Claude（依据 W5） |
| 17 | 2026-09-22 | T8早期Codex草案全部原样归档到`research/codex/archive/`，不晋升为当前工具；E1继续使用已有`scripts/` | 核对E1脚本无这些历史依赖；11个源码/输出/patch/fixture迁移后SHA256全部相同，逐项状态见archive/README。仅删除6个可再生成的pyc；历史脚本的默认相对路径不适配归档位置，标为不可直接执行。README当前状态仍由Claude根据E1交接更新 | 用户要求，Codex整理 |
| 16 | 2026-09-22 | E1/E2替身固定为`e1-kimi-linear-4l-qfull`，q_lora_rank=null；保留未修改stock源码与Python UnifiedTreeCore默认cache路径；F13/F24仅作理想预测，不作精确oracle | 原Kimi包装层的q_lora分支缺AttentionInputs上下文，qfull不改MLA的KV低秩池/KDA/cache/scheduler路径；E1已实测72请求通过。2处cache偏差提示模拟未建模驻留FULL-KV，源码与归因推断见experiments E1；任何新case须同权重重跑stock。仅替身验证范围，不改变完整GLM配置 | Codex main（E1实施与交接） |
| 15 | 2026-09-22 | 借鉴 harness-template-cn：采纳 `AGENTS.md`（Codex 自动读取的入口）+ `CLAUDE.md`、`plans/active` 执行计划（含验证方式与回滚）、`notes/tech-debt.md`、`notes/quality.md`、`scripts/check_records.py`（机械检查），以及定期归档；history 目录不单独建（并入 plan 决策记录与 patches 文档）；发布记录、前端、CI 类不采纳 | 模板有成型的迭代闭环（计划 → 验证 → 记录 → 清理）；我们的缺口是按任务的计划、技术债清单和机械检查（R6） | 用户提议，Claude 评估后采纳 |
| 14 | 2026-09-22 | 启动命令加 `--tool-call-parser glm47`，保留 `--reasoning-parser glm45` | chat template 以 `<\|assistant\|><think>` 结尾，glm45 的检测器允许省略起始标签；工具调用格式 arg_key/arg_value 与 glm47 一致；主办方验证过的示例也用这对组合（F32）。答案必须出现在 message.content，要实测确认 | Claude |
| 12 | 2026-09-22 | 按任务启动**完全权限的独立 Codex 实例**（`scripts/codex_worker.sh`，每个任务一个新上下文）；主 Codex 会话负责审阅和讨论；Claude 也可以派自己的 subagent | 单个 Codex 会话上下文有限；用户明确授权完全权限 | 用户 |
| 13 | 2026-09-22 | 借鉴 AgentX/ClawPerf 的经验，新增 I1–I5（directions §I） | AgentX 是与本赛最像的公开基准（闭环 agentic 多轮） | 用户要求，Claude 落实 |
| 11 | 2026-09-22 | 〔已被 29 取代〕D1 方案 A 维持为首选，并参考 llama.cpp #22929 的实现 | llama.cpp #22929（2026-05-25 合入）为 agentic coding 在最后一条 user 消息前切 prefill batch 并保存 checkpoint，与方案 A 同构（Codex R6 §2 B1）；TRT-LLM #18724 提醒："同时保留边界快照和末尾快照"必须核实底层确实存了两个独立状态 | Claude（依据 Codex R6） |
| 10 | 2026-09-22 | 分工：**Trisol 8 卡**做真实验证（底包内容、完整模型、开发集梯子、能力抽检）；**2 卡 GPU 机**只做小而快的事（替身模型功能测试、补丁开发与单测、数据分析、Dockerfile 预演） | 只有 Trisol 上有主办方底包、sm80 DSA 实现和完整模型；2 卡机器跑不了 GLM-5.3，但迭代成本低、不占配额 | 用户提出，Claude 采纳 |
| 9 | 2026-09-22 | 能力分抽检加入自测流程（task.md 没有明确要求这一项） | task.md 自查清单只要求 `/chat/completions` 往返一次、检查 thinking 和输出预算有没有被压掉（task.md:430-434）。D1/HiCache/int8 这类改动可能影响正确性，而能力门是硬门，所以额外加一道回退检测 | Claude 提议 |
| 8 | 2026-09-22 | 批准本地 2 卡验证（替身模型 E1/E2）；Trisol 8 卡和提交仍然不行 | 先用低成本方式确认 D1 的前提（stock 命中确实偏低），再动用 8 卡配额 | 用户 |
| 7 | 2026-09-22 | 本地验证用随机权重小 Kimi-Linear 做替身，不用裁层 GLM-5.3 | 上游在 sm80 上没有 DSA 实现（R1），GLM-5.3 在本地跑不起来；我们要验证的缓存和调度逻辑与权重无关 | Claude（待 E1 验证可行） |
| 6 | 2026-09-22 | 〔已被 29 取代〕D1 先做方案 A（在边界切 chunk），方案 B（同一 forward 导出中间状态）排在后面；若 A 验证有收益，B 优先级上调 | A 不用改 kernel；vLLM Kimi K3 已证明 B 可行且更快（R3）；Codex 指出 A 必须守住"单 active chunk"不变量（D1 §8.3） | Claude + Codex |
| 5 | 2026-09-22 | 暂不启用 HiCache 和 `--enable-mixed-chunk` | #39156：DSA 索引不会恢复；#39526：会破坏快照（R1，Codex F11） | Claude + Codex |
| 4 | 2026-09-22 | 〔已被 29 取代〕D2 先移植 #40024（SPF），#39717 以后再说 | #39717 以 #40024 为基线；#40024 改动小（生产代码 +82 行）（Codex R4） | Codex 提议，Claude 同意 |
| 3 | 2026-09-22 | 撤回"vLLM main 不支持 Glm5Next"的说法 | vLLM PR #53906 已于 9 月 3 日合入（R1/R2/F11） | Claude |
| 2 | 2026-09-22 | 撤回"MLA 在 TP8 下复制、改 DP-attention 就是最大杠杆"的初始判断，改为以 D1 为头号方向 | 模型是 34 KDA + 11 MLA/DSA 的混合结构（F1）；尾部 reminder 分叉的发现（F3/F13） | Claude |
| 1 | 2026-09-22 | 以 SGLang 为主路线 | v0.5.20 原生有 `glm5_next`；压测接口 `/generate` 是 SGLang 原生接口 | Claude |

### T12 工具实现选择（2026-09-22，Codex W2）

- 保持 dev 原始 verdict，默认按 dev 搜索；可显式选择 dev+tpot / estimated。TTFT 估计采用单侧 95% Clopper–Pearson（L=Beta⁻¹(0.05;k,n−k+1)，k=0 时 L=0），全程标 estimated，不宣称与隐藏实现一致；符合 D0 §5 审阅边界。
- 针对 F23，既做档前严格 flush，也用 S1_FLUSH_URL 本地 guard 验证原 runner 的预热后 flush；失败终止子进程而不修改 harness。VERIFIED：29 项 CPU 测试通过，证据 `evidence/T12/tools_validation.log`；具体接口/用法见脚本头与 experiments「Tools」。

## 决策 34（2026-09-23）— 8 卡服务被守护进程误删后的处置
- 事实：`lh-arena-sess-a`（2102309548588015616）于 2026-09-22T19:45:59Z 被 `scripts/trisol_test_daemon.py` 以 IDLE_RELEASE 删除。代码默认 `idle_hold_seconds=10800`，
  配置文件虽已改为 1e9，但**守护进程只在启动时读配置、改后未重启**，于是在旧队列最后一项结束 3 小时后自动释放。违反"停服务须经用户同意"。当时 pod 正在跑 dev N6 基线（019），结果与 pod 内所有状态丢失（profile 摘要已存 evidence/T43）。
- 处置：守护进程已停止且不再启用（pod 内队列 `scripts/pod/podq` 已承担全部实验；服务不需要"保活"动作）；
  按原参数重建 `lh-arena-sess-b`（2102486579267252224，底包镜像 arena-sglang-glm53:260918，命令 http.server 挂起，描述空），排队等待准入；`scripts/pod/common.sh` 默认 SID 已切换。
- 规则：任何会删/停服务的自动化一律不运行；改长驻进程配置后必须重启并在日志中核对生效值。

## 决策 38（2026-09-23）— 方向排序（请教 Fable 后，结合逐组件成本表）
- 依据：R8 §6/§7；逐组件成本表（findings 最新条）；Fable 审阅意见（核对了 dsa_indexer.py:259-288 ReplicatedLinear、113/MoE/tilelang 微基准、F31/F46 正式集桶大小与允许超标条数：chain_start 808/51、intra 9023/485、turn_start 65/6；底包 DCP `--dcp-size` 与 prefill CP 存在）。
- 分歧：Fable 认为 mHC（G4）FLOPs 可忽略；实测 mHC 占预填充 13%（带宽受限：4 倍宽残差流）→ G4 保留但靠后。
- **排序**：
  0. 服务回来先跑：pod 验证套件 → **能力门自测（AIME/GPQA 样题，含长输出 decode 路径）** → 单请求冷预填充分解（20k/60k/190k，/start_profile）。
  1. **G2 indexer 按查询行切分到 TP 组 + all-gather top-k**（O(L²) 且 8 卡冗余；114k 约 −4s、190k 约 −11s；零精度风险）。Claude 自做，开发机 TP2 先验证。
  2. 8 卡开关探针：`--enable-prefill-cp --attn-cp-size 8`（G3，= G1+G2 现成实现）与 `--dcp-size`（KV 容量 ×N）；各一个任务。
  3. 容量：C1（cuda-graph max bs）+ 读驱逐计数（N=26 时 fast_intra 可能先挂）。
  4. F3 EDF + 注定超时降级（合规，追平前排）。
  5. F1 仅角色边界快照（不做网格，避免吃容量）。
  6. G4 mHC、F2 MoE kernel 最后。

- 补记（09-23，lh-arena-sess-b 准入后）：发现旧守护进程被 tmux `arena-daemons:l2` 里的 `scripts/run_forever.sh` 自动拉起、已空转 8 小时（配置已指向新服务名，但事件日志对新服务 0 条记录）。已杀掉守护进程与 run_forever、关闭 l2 窗口，并把 `trisol_test_daemon.py`、`run_forever.sh`、其测试移到 `scripts/archive/retired/`（本地与 GPU 机）。
  核验：无任何进程调用 bohr/trisol 或旧守护进程；在用代码（scripts/pod、gjob、gssh）中 inference delete/stop 调用 0 处；`submit_daemon.py` 不碰推理服务、队列为空。

## 决策（2026-09-23，N6 基线后）— 按真实瓶颈重排
- 事实：我们在 N6 就卡 intra 两门，主因是排队（p95 6.4s）；chain_start 有余量；indexer 在 ≤6 万上下文只占 4.3%（114 价值下调）；allreduce 占 11%；KV 在 N6 已 50%。
- 排序：① 120 调度 A/B（队列中 013/014，N6 上直接看 intra 两门是否转 PASS）；② 140 双点快照 A/B（修短间隔丢缓存）+ 查长空闲丢缓存原因；
  ③ 容量（DCP 探针已排、cuda-graph bs、KV 预算）；④ 预填充：allreduce（自定义 allreduce/NCCL 协议）、prefill CP 探针、MoE；⑤ 114 仅在超长请求上有益，保留但不急。

## 决策（2026-09-23）— 采纳 Fable T49 审阅
- 120 改 v2（封顶仅在有等待时生效；chunk 16384 / cap 8192 / short 8192），v1 存 patches/drafts/。
- 新增并优先：`--enable-attn-tp-input-scattered`（mHC 每卡 1/8 token）；容量零风险杠杆（cuda-graph max bs 64、KDA 槽 200）。
- CP 不做主线（每轮只准入 1 请求、关闭 120、decode 头切分白名单无 Glm5Next）；DCP 仅作容量后备（需 115）。
- lib.sh：引擎复用键加入 SGLANG_AX_*/NCCL_* 环境变量（防止只差环境变量的实验误复用引擎）。

## 2026-09-23 — 梯子不降级；预填充主瓶颈定为 host 侧 eager 开销
- 用户：N18 失败就停下诊断修复，不再降到 N14/N10（模板 LADDER_DOWN 默认改为空）。
- 依据 F85–F87：调度参数只能在 TTFT 和 TPOT 之间换（027）；真正的杠杆是每块约 150ms 的 host 侧开销。顺序：170 的 8 卡 A/B（026j/k/l）→ 通过后做 170 + v3 小块 N18 → 下一步机制 T53（KDA/DSA 断点的 host 开销）→ MTP 叠加。

## 2026-09-23 晚 — 今天两次正式提交：A（028 配置）与 B（大块 + MTP）
- 用户批准：同一个镜像，打完直接提交，不先跑冒烟。依据 F89/F90：MTP 过了解码门（p95 0.082），TTFT 在 N18 未过 ⇒ 预期 N10–14。B 按"有可能的方向都保留"的原则提交。
- 明天迭代：先保住底线，再冲高；每次只改一个变量；8 卡复测 170 v2（scatter）与 116（DCP）；修正数值指纹工具的判据。
