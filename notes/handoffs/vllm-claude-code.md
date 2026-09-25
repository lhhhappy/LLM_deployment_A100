# Claude Code handoff：vLLM 路线、背景与开发规范

2026-09-24，会话交接。用户最新决定：**SGLang 与 vLLM 并行探索 N38；Codex 继续 SGLang，Claude Code 接手 vLLM。首阶段只要求跑通 vLLM 基本开发/GPU验证并对齐当前 SGLang 的模型、接口和评测行为。**
不预设哪个引擎更好，也不承诺 N38 可达。本文替代早先“Codex 单人、vLLM 暂缓”的工作安排；不改变赛规、既有冻结实验和共享资源保护规则。
用户已通知CC开始。Codex不再编辑vLLM实现；下文优化机制表作为后续参考，不是首轮必须移植的任务清单。

## 1. 接手时先知道的状态

- 仓库：`/workspace/Agentic_science_challenge`。交接前最新实现提交 `afa72d9`，属于 **SGLang 120**，不是 vLLM 版本。
- **截至Codex交出路线时，Codex未做vLLM移植、安装、构建或GPU调通。** 只读了资料和开发机资源，没有创建vLLM引擎目录、改源码、拉镜像、下载模型或启动服务。CC接手后的状态以其报告为准。
- Codex 刚启动的 vLLM 探查代理已经停止；它没有遗留后台进程、容器或 GPU 负载。Claude 可接手自己的文件和探针。
- 当前开发机只读检查为 **2×A100-SXM4-80GB**，每卡总显存 81920MiB、已用 7MiB，利用率 0%，无计算进程。**这是本轮快照，启动前重新检查；未采集精确时间戳。**
- 开发机 `/sjtu/linhang/arena` 文件系统可用 `2,922,147,479,552` 字节（约 2.66TiB）。Docker 根在 `/ebs/docker/165536.165536`，并非 arena 工作目录；不能拿 arena 的余量推断拉镜像安全。
- Docker 本地没有 vLLM/backport 镜像。arena 下存在 `models`、`env/sgl`、`env/m0` 等目录，内容与可用依赖尚未核实。两张卡不能假定能装下原始 FP8 全模型；先做单层/TP8 每卡形状探针，完整 TP8 另排资源。
- 用户提到可能有新八卡/V100机器，但**没有交付地址、型号/显存确认或可用性证据**，不能把它当现有资源。
- 共享八卡最后已知因070临时盘超20Gi被驱逐；恢复071计划在案，尚无恢复测量成绩。实时情况重新查 `scripts/pod/pread status`，不要把旧状态当实时。

## 2. 阅读顺序与目标

先读 [README](../../README.md)、[queue](../queue.md)、[AGENTS](../../AGENTS.md)、
[collaboration](../collaboration.md)、[evaluation](../evaluation.md)。赛规只认
[task.md](../../llm-challenge-arena-v1/task.md)。当前事实看[knowledge](../knowledge.md)，完整实验看[experiments](../experiments.md)。

技术入口按顺序：

1. [R9 引擎与上游筛选](../../research/claude/R9_upstream_since_base.md)：主办方 vLLM backport、官方底包、公开 fork 的区别。
2. [R26 四桶优化路线](../../research/codex/R26_n30_slo_levers.md)、[R27 首次入批](../../research/codex/R27_receive_to_admission.md)：真实坏例、机制和 CPU 反例。
3. [engine 机制表](../../engine/README.md)：哪些在正式配置里，哪些只有 CPU/局部证据。
4. [R17 MTP](../../research/codex/R17_nextn_sm80.md)、[R20 缓存账本](../../research/codex/R20_true_lcp_attribution.md)、[R21 TPOT 坏例](../../research/codex/R21_N22_tpot_failures.md)。
5. [模型/算子地图](../../research/claude/base/04-model-kernels.md)、[R10 固定开销](../../research/claude/R10_prefill_fixed_overhead.md)、[R25 MoE](../../research/claude/R25_moe_sm80_path.md)。历史实现结论要对照当前提交核实。

正式排名顺序：`n_at_slo` 越大越好 → `tpot_mean` 越小越好 → TPM 越大越好 → 先提交者靠前。
能力评测 AIME26、GPQA Diamond 都须严格高于90。N38是探索目标，不能用开发集成绩外推正式并发。
三个方向共同推进：减少重算、降低每次执行成本、改善批次/准入；CPU/GPU有效协同是手段，不把利用率100%当效率证明。

## 3. SGLang 已有结果与迁移时必须保留的认识

正式 A 的不可变标签为 `official-A-0923a`（attempt45979，N14，保留MTP）。后来正式46173是 A+mem0.87+新版180、host32、122off，N14 PASS；46174再加修复122，N18 PASS。
两份都过能力门；不同N的TPOT不可当作同负载退化比较。更高失败档明细未返回。

本地067/068/069固定引擎 `759a6ebb8e31723519ad5daf438e26e24b32501a`、完整311链/5601请求、N30：

| 运行 | 主要配置/唯一改动 | 完整结果 |
|---|---|---|
| 067 | A+mem0.87+180+修复122，host32 | VALID FAIL，四类TTFT失败 |
| 068 | 对067只关闭122 | VALID FAIL，fast/overall/chain失败，turn按统计余量通过 |
| 069 | 对068只将host32→64GB/rank | VALID FAIL，10/11门通过，仅chain失败 |
| 070 | 对069重新开122 | 启动阶段临时盘超20Gi被驱逐，**没有测量成绩** |
| 071 | 恢复070冻结配置，新Pod/新run | 最后已知未入队，交接后查实时状态；Claude不要擅自改此配置 |

069 p95：fast **2.704s** / overall **3.627s** / turn **13.641s** / chain **40.336s**，目标3/5/15/30s。
chain超标31/432，CP允许29；TPOT均值/p95 **.028824/.055902 s/token**。不能把少修两条的临界过门当稳定胜利。
068→069实际未命中prompt约30.810M→19.132M，少37.9%；这支持研究容量/复用，但不证明每条坏例都由host淘汰造成。

重要纠正：

- SGLang raw 的 `t_exec_start_s` 来自 `forward_entry_time`，是**首次选入批次**，早于batch准备与GPU launch；不是第一条kernel开始。
- 069坏例中接收→首次入批占TTFT≥80%：fast218/222、overall179/195、turn4/7、chain20/31。
  这不表示GPU有80%空闲，也不表示只在开场拥堵，更不表示调度一改就省80%。全程零请求错误，这里“坏例”是延迟超标。
- 068、069的服务日志只见0、1次retraction，尚无频繁抢占抖动证据。换到vLLM需重新测，不能先把watermark当主药。
- 069有311个唯一TTFT坏例；旧日志没有新准入诊断，归因仍未知。后到先入批只是现象，不等于错误排序。

最新 `afa72d9` 做了什么：

- `SGLANG_AX_ADMISSION_TRACE=1`：TP0有限诊断，整个进程累计8MiB上限；按队列episode对齐，避免预热/回退复用RID串账。
- `SGLANG_AX_SCHED_KV_SCAN=K`：默认0，上限16；候选未入批且KV预算不足时，有界检查队尾完整device短命中；保留原预算/锁/清理/full复位。
- 候选检查实际整页输入+完整输出预留+额外页+共享状态成本，避免检查与扣账量不一致；不搬host、不产生第二partial。
- 18项候选测试、合计112项相关CPU测试通过。**没有TP8正确性或性能成绩。** 它可能改善fast/overall，也可能推迟长chain。

参考实现：`engine/sglang/srt/managers/{scheduler.py,schedule_policy.py,ax_admission_trace.py}`；
`tests/test_kv_admission_scan.py`、`tests/test_admission_triage.py`；
`scripts/analysis/admission_triage.py`；`evidence/admission-wait-20260924/`。
这些是机制参考；vLLM没有同名时钟/请求状态合同，不能直接复用分类器字段含义。

## 4. vLLM 的技术起点：先锁定真实底包

优先使用 task.md 已验证接口的 A100 backport：

```text
registry.dp.tech/dptech/dp/native/prod-20675/vllm-backport:260918-sm80
```

task.md「示例提交」给了完整启动命令与env，包括TP8、原FP8权重、MTP3、prefix caching、
`FULL_AND_PIECEWISE`、`NCCL_ALGO=Ring`、`NCCL_PROTO=Simple`及GLM parser。
`/generate`、`/flush_cache` endpoint插件及白名单已在该镜像里；**先核验是否存在，不要先装第二份同路径插件。**
这只是主办方接口验证起点，不是我们的能力/吞吐成绩。示例`max-num-seqs=16`、graph只捕到16，不能原样用于N38探索。

严格区分另外两份来源：

- `registry.dp.tech/dptech/dp/native/prod-3732438/20675/arena-vllm-glm53:260918`是官方路线的另一镜像，参数不能混抄。
- [公开 wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport)提供SM80实现参考，未证明等于题面镜像。
  [本地README快照](../../evidence/engine-survey-20260924/vllm-backport-readme.md)可先读；4×A100示例使用AWQ4bit，不代表原FP8全模型可装入4卡。
- `refs/vllm-pr56960` 是只读上游参考，commit `7565389f848994d5271986f74aab2af7ede0cbab`，不是主办方backport。

第一份交付是底包收据：镜像digest、Python/vLLM/torch/CUDA版本、模型/权重身份、实际生效后端，
关键源码SHA（scheduler、request queue、KV manager、混合状态、模型、MTP、接口插件）与原始启动参数。
**先取真实源码再改；拿不到时可用固定公开提交做原型，但明确标为原型，不能直接盖到未知底包。**
源码或补丁应用需严格匹配哈希/上下文，失败即停；不靠行号替换、模糊补丁或运行时猜版本。

## 5. 首阶段对齐合同与后续技术路线

首轮以**最近有完整结果的069**为行为/负载对照：引擎`759a6eb`、MTP保留、host64、122off；
实际参数见[069任务](../../scripts/pod/jobs/official_b_pace_off_host64_full_n30_shortwarm.sh)。
071是同配置开启122的冻结恢复实验；`afa72d9`是仅CPU验证的新候选，都不是首轮必须移植清单。
跨引擎“对齐”不是文件或flag同名，先交付以下矩阵：

| 对齐项 | 首轮验收 |
|---|---|
| 模型与输入 | 相同原始FP8权重身份、tokenizer/chat template、thinking/tools/历史/输出预算 |
| 接口与统计 | 能力口最终答案、`/generate` SSE、真实token/时间戳、`ignore_eos`、真实flush；按原harness验 |
| MTP | 保留并核实际启用；SGLang steps3/topk1/draft4 与 vLLM speculative tokens3 的实际含义按源码核对，不能只比数字 |
| 缓存 | device前缀复用先跑通；列出host/混合状态恢复与069的功能差距。没有host tier时明确未完全对齐，不伪报host64等价 |
| 资源/调度 | 同硬件资源条件下记录实际KV/KDA、图池、batch/seq预算、运行策略；不机械复制mem0.87/interval2 |
| 开发流程 | 固定底包与源码提交，可复现安装/启动/短探针/取证，代码和数据路径独立 |

先得到稳定、可测试的vLLM原生版本和差异清单。除跑通/计数/状态正确性所必需的修复，
首轮不要求叠加120/122/123/180。下面是基线完成后再选的机制候选。

| 方向 | 可借鉴内容 | vLLM第一步及边界 |
|---|---|---|
| 原生基线 | MTP、prefix reuse、分块、显存池账本、完整评测 | 先查原生已有功能。可见上游调度器先处理running再处理waiting，与SGLang不同；以真实底包为准 |
| 120/有界准入 | 区分候选不适合与全局资源耗尽；有限扫描；拒绝回收 | 查`allocate_slots`失败是否已修改状态，队列跳过/回插顺序、异步恢复和preemption规则；全部预算用原生KV manager核验，不照搬SGLang单partial限制 |
| 122 | 用decode时延余量约束prefill干扰 | 先测vLLM每步时间、MTP产出、长块成本，再设计预算。SGLang的τ、固定开销/斜率不能直接移植；先试原生batch token预算/long-prefill threshold（若真实版本支持） |
| 123 | 剩余工作量与等待提升 | prefix token命中不等于剩余秒数；先在真实队列上验证优先级、完成/中断/取消；新请求不能无限压住长chain |
| 180/host缓存 | 少重算最有实证价值；KV与KDA检查点必须一致 | 查原生offload/LMCache支持的混合状态、DSA indexer、MTP、布局与恢复依赖；单独实验，不直接等同HiCache或照抄64GB/rank |
| watermark | 为运行中序列增长留块，减少抢占重算 | 先看抢占频次/重算成本与池峰值；确有抖动再单变量试。它可能增加准入等待，不是降低TTFT的通用开关 |
| 110/111/114/160 | SM80算子与MTP经验 | backport可能已有对应实现。先核实际dispatch、形状、正确性与通信；不重复安装SGLang补丁 |
| 170/171/172 | graph覆盖、融合、布局与显存收益账 | vLLM已有graph和KDA融合实现参考；测遗漏和TP8每卡形状，再判断增量，不能把两个框架相同功能当新增收益 |
| 有限诊断 | 接收→等待→首次调度→首token分段、拒绝原因 | 正确映射vLLM时钟与episode；缓存miss/队列等待不是GPU饱和证明。诊断默认关、单进程总上限、不开逐token无界日志 |

缓存路线的特别提醒：公开backport配套LMCache文档中，GLM的统一块为1152，MTP+align示例要求
`--prefix-cache-retention-interval 1152`和`--max-num-batched-tokens 1152`。
这是**该公开版本的合同线索**，必须在实际版本核实；更密集checkpoint会改变前向次数与固定开销，不能只报命中率。
启用外部缓存后，`/flush_cache`必须覆盖能重新恢复的缓存层，不能清GPU后又从host读回却声称冷缓存。

## 6. 按依赖执行，GPU只验证有明确问题的候选

**阶段A：仓库和资源。** 登记接手，读当前dirty状态与共享队列；检查GPU、模型、驱动、可执行盘/容器盘/host RAM。
锁定底包收据。全部探针在`/sjtu/linhang/arena/`自己的目录工作；不要为装包覆盖已有SGLang环境。

**阶段B：功能基线。** 开发机先做可容纳的单层/算子与调度验证；完整原FP8模型需要另排TP8，
不能为塞进两卡就静默换AWQ、关MTP或改模型后当作同一基线。小替身仅作接口/逻辑探针，结果明确标注。
有TP8后验`/chat/completions`、根路径`/generate` SSE、真实token计数/ignore_eos、前缀命中、真flush、长上下文、MTP和恢复路径。
核对thinking/tools/历史保持原样；不重新训练。记录加载/捕图/预热时间与峰值资源。

**阶段C：原生配置形成可靠对照。** 先建立未经新增策略修改的可复现vLLM基线。
N30/34/38需要评估足够请求槽（如48/64候选），但槽位、graph捕获、KV/KDA/草稿缓冲各有显存成本；
capture上限较小可能回退eager，具体按底包核实。不能只扩大槽位便承诺吞吐提升，也不要同时扫十个参数。
不要直接把SGLang mem0.87当vLLM最佳值；每次核实际权重、图池、KV与状态容量，保留资源对照。

**阶段D：单机制候选。** CPU真实调度器＋假资源验证默认关闭等价、次序、预算、取消/回退、异步恢复、状态回收。
GPU测TP8每卡形状的数值/速度/显存；两卡通信只证明两卡。先短窗观察预热与模式覆盖、服务健康及diagnostic开销。
若是明显错误，留证后停自己的任务；性能超门不自动截断完整实验。

**阶段E：完整比较。** 用同一冻结数据、原harness、rep16预热协议、真flush，先与现有完整N30建立对照；
然后按完整结果选择N34/N38及候选，不强制每种方案跑遍所有档。相同引擎内单变量，换引擎是整体组合比较。
报告四类TTFT、TPOT、TPM、全部11门、5601个同ID差异、新增坏例与资源代价。
正式提交仍由用户决定；“启动成功”“小窗快”“本地N38通过”都不等于正式N38。

### 快速开发循环：不等固定70分钟

用户最新同意按层级快速筛选：启动/捕图 → 功能冒烟 → 有明确请求集合与预算的短压力探针 → 值得保留的版本再完整回放。
探针按覆盖目标结束，或设置有限时间/请求数；例如服务就绪后10–15分钟的初筛，并非所有问题都需等满该时长。
启动失败、OOM、状态/输出错误、flush失效、缺请求、计数错误等确认后立即留证并结束自己的探针，修复后新run。
重点覆盖MTP、长上下文/切块、重复前缀、缓存恢复（若启用）、取消/回退与目标并发下的容量边界。

**短测是独立开发实验，不是把完整测量截到某个好看的窗口。** 固定探针清单和停止条件，对照时用同一份；
保持原请求内容/预算，可以明确挑机制样本，但不冒充完整cohort或正式分数。短测结束记`SMOKE_OK`/`SMOKE_FAIL`等开发结果，不报N@SLO。
069没有请求错误，fast延迟坏例最晚出现在约68.583分钟；所以“错误都在早期”尚未成立，SLO坏例也不等于程序报错。
完整判分仍须5601条闭合，不设70分钟截断；通过短测只取得跑完整对照的资格。

## 7. 仓库规范与两条路线的边界

用户本轮已授权新增vLLM路线。引擎目录约定扩展为：SGLang在`engine/sglang/`，vLLM在`engine/vllm/`；
不要把vLLM放进SGLang包，也不要修改`refs/`或`build/base_exact/`参考副本。

建议由Claude建立以下最小结构，**Codex没有创建这些vLLM实现目录，接手后先查现状**：

```text
engine/vllm/                 真实底包代码＋受git管理的改动/必要插件
engine/docs/vllm/            机制说明、开关、成本、CPU/GPU验证边界
scripts/vllm/               底包鉴别、导出/安装、启动、GPU探针
tests/test_vllm_*.py         CPU回归，不依赖加载全模型
research/claude/vllm/        个人路线索引与分析
evidence/vllm-<run-id>/      收据、原始结果、逐请求表、有限日志
```

- 不复制整仓库/权重/镜像层。追踪实际采用的源码和许可证、固定commit与SHA；构建物、下载缓存进忽略目录，原始实验不覆盖。
- 第一个提交固定vLLM底包与来源；再建立独立基线tag（名称由Claude登记，例如`vllm-base-<id>`）。现有`engine-base`、`official-A-0923a`仅属于SGLang，不移动。
- SGLang继续`engine NNN:`；vLLM使用独立提交命名`engine vllm NNN:`，与`engine/docs/vllm/NNN-*.md`对应。
  SGLang的120映射到vLLM什么机制，在文档中说明；编号不是等价实现证明。修复仍归所属机制，不另开编号。
- 每个策略默认关闭，关时保持**该vLLM固定底包**行为；不支持组合显式拒绝。启动打印引擎身份、源码哈希与有效机制，运行前核对期望值。
- `scripts/engine/tree.py`、`export.sh`、`build_image.sh`和现有`pod/lib.sh`按SGLang路径/底包设计。**不能直接拿它们部署vLLM。**
  先建独立vLLM导出/启动入口；若抽出共享接口，约定后修改并保留SGLang回归。原harness/评分/归档尽量复用。
- 共享主工作区已有大量与本任务无关的dirty文件。先看diff，按路径暂存自己的内容；不`git add -A`、不reset/clean、不提交别人的改动。
  若用worktree，登记路径/分支；GPU部署只传冻结的本路线文件，不能覆盖远端整个共享repo。
- Claude拥有vLLM路径；Codex拥有当前SGLang候选与071恢复。改共享notes或运行库前读最新版、声明接口和谁写；共享八卡排队，不在同一组卡同时做性能对照。
- 并行是研究/开发并行；只有确实有独立GPU资源时才能同时测性能。不要为抢卡停止、删除或释放共享八卡服务。
- 在[collaboration](../collaboration.md)登记真实会话/报告位置。接手席位目前是待登记，不假设旧Claude会话或消息通道仍有效。

## 8. GPU入口、日志和磁盘：必须避免再撞20Gi

短命令用`scripts/gssh`，长任务用`scripts/gjob`，GPU工作限制在`/sjtu/linhang/arena/`。
`gssh`只重试连接探测；真实命令断线后可能已经执行，先查状态，不重复启动。
`gjob`的job日志默认是普通文件，**不自带总量封顶**；为新探针配置有界输出/轮转后再开长期任务。
Docker镜像层会写DockerRootDir，不能仅通过把输出路径放arena就满足空间约束；先核容量与工作范围，必要时采用独立安装方案。

共享八卡审阅：`scripts/pod/pread status`。CPU远程分析用`pexec_codex`且只写`/tmp/ax/codex`。
队列入队前在[queue](../queue.md)登记引擎、基线、唯一变量、提交/哈希、数据和负责人；
当前qpush运行库只适配SGLang，接vLLM需先完成适配/核验。不要把vLLM直接塞进SGLang启动模板。
若需停某个已确认严重错误的测试，留证后用GPU机上的`scripts/pod/stopjob <job.sh>`；8卡service不可停删释放。

日志约束：新增准入诊断参考8MiB/进程生命周期，只有一个rank/调度进程输出；
GPU trace限事件数、时间窗与总字节，不连续长录；健康状态覆盖写，有界保留；原始测量raw不得抽样/截断来省空间。
额外GPU缓冲先算每卡字节及挤占的KV/状态槽；host缓存需核cgroup内存，tmpfs也计入它，不能当免费磁盘。
070之后的RAM工作区`AX_WORKSPACE_ROOT=/dev/shm/arena-runtime`是恢复方案，不能绕过bootstrap的容量/绑定核验；JIT可执行性和冷编译混杂单独记录。

已授权的清理流程是 **运行完成→取回本地→SHA256校验并持久化→只清理对应Pod副本**。
已有工具：

```sh
python3 -B scripts/analysis/archive_completed_runs.py --cleanup --watch 300
```

先检查现有单实例状态`build/scratch/archive-maintenance/status.json`，不要重复启动守护。
工具拒绝活动日志、被其他run引用、文件变化或仍打开的目录；不用Pod上先打巨大tar，也不删除模型/JIT/运行中任务。
vLLM若不在现有run布局内，先适配manifest与结束状态识别再清理，不能盲用删除命令。

## 9. 评测合同与验收交付

当前本地完整集：`data/s1-dev-longchain/`，311链/5601请求；manifest SHA256
`19a7e5a6827f64a99695cba2d89b7efa2a0b05d207fc95da1568ec1d82280b2c`。
先核文件，不重新合成、重排、截历史、删tools、压输出或降低thinking。
原公开dev是另一个较小集合，不能混用cohort和LCP账本；正式341链/5150请求也不能由本地换算。

同ID每条恰好一次、runner/真实flush/指标齐全才能判VALID。评分只用原harness与
`scripts/score_formal.py`的题面补充门；TTFT有统计余量，TPOT p95≤0.10 s/token没有余量。
运行中窗口只诊断，第15分钟首次、后每30分钟；不拿不完整窗口挑PASS，不把计数当等待秒数。
新增vLLM meta_info必须直接来自引擎：真实prompt/cache/累计completion token，真实接收与首token时间；
DELTA流要核累计计数，MTP一次多token不能把首事件误当单token；不伪造SGLang阶段字段。
能力口的最终答案须在`choices[0].message.content`；保持原thinking与tools，核对parser。

Claude首轮交付收窄为：

1. 真实底包/模型/硬件收据与可复现命令，明确两卡已验证什么、八卡还缺什么。
2. 独立vLLM代码路径、固定底包、开发/启动入口、必要修复及CPU回归；不要求首轮移植所有优化。
3. GPU探针原始证据：数值、耗时、峰值显存/host内存、实际后端、错误与限制。
4. 一个可靠原生基线与相对069的功能/配置差异矩阵；明确哪些已对齐、哪些尚缺，不能把缺host缓存隐藏掉。
5. 短测可快速重跑；条件具备后交付完整评测入口/冻结任务。两卡未能做的TP8验证保持待验证，不必先开发新的调度策略。

简短交付格式：**做了什么 → 实际验证层级 → 同条件收益/代价 → 未决问题 → 下一项唯一变量 → 证据路径**。
不用等待Codex批准已经授权的本地开发和安全探针；遇到共享文件/资源冲突，先协调避免互相覆盖。
