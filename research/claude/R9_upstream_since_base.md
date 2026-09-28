# R9 — 上游实现与替代引擎：A100 路线筛选

2026-09-24，Codex 覆盖更新。接续 09-23 的上游扫描和 Claude 的源码核验，替换旧的关键词排名表。当前研究范围包含换引擎；“属于另一个代码库”不再是降级理由。历史内容留在 git，原始实验仍在 evidence。

## 1. 先复用什么

| 旧研究 | 继续有效的部分 | 修正与补查 |
|---|---|---|
| 原 R9 底包溯源 | fe236ea6c3 不是当时 SGLang main 的祖先；底包是 GLM-5.3-Flash 早期定制实现 | 不能按日期直接升级，需比较实际模型和缓存路径 |
| 原 R9 SM80 核验 | vLLM #55737 的 FlashKDA 已查快路径仅 capability.major∈{9,10,12}；SGLang #35429 的 decode/indexer 与 Torch attention fallback 没有优于 110–113 的证据 | 沿用排除结果，不把 1.7–3.8× 宣称为 A100 可得收益 |
| [R10](R10_prefill_fixed_overhead.md) | TP1 替身的小块 eager 前向有大量 host 间隙；170 已有实现基础 | TP1 dummy 不能代表当前 TP8 各组件占比；170 v2 还需 TP8 数值复验 |
| [R17](../codex/R17_nextn_sm80.md) | MTP 的 draft、verify、状态提交和显存约束 | 正式 A 已通过 N14，保留 MTP；原“等 MTP 实测再看”已过时 |
| [R18](../codex/R18_cache_loss_and_capacity.md)、[R20](../codex/R20_true_lcp_attribution.md) | token 前缀与 KDA 状态位置必须同时有效；扩状态池会占内存 | 8% 是所测运行的相邻同链缺口占比，不是所有缓存优化的理论上限 |
| [R19](../codex/R19_progress_and_cache_review.md)、[R21](../codex/R21_N22_tpot_failures.md) | 核对评分、cohort、逐请求记录和执行窗口重合 | 不同配置、不同 N 的正式/本地差值不能当测试集偏差倍数 |

原扫描约覆盖 SGLang 916 个提交（204 个路径匹配）、vLLM 1204 个提交，多数只检查标题或路径，开放 PR 搜索未穷尽分页。旧 HIGH/MED/LOW 不是现行结论；本页也不是对所有引擎的穷尽测试。网页与源码均为时间快照。

## 2. 引擎对照：哪条有真实起点

赛规允许其他引擎，要求同样的 /generate、真实时间/token 计数及 /flush_cache。**有底包、跑通接口、通过双 90、通过某个并发档，是四种不同证据。** 依据：[task.md](../../llm-challenge-arena-v1/task.md) 引擎适配、基镜像与示例提交。

| 路线 | 已有证据 | 尚缺证据 | 优先级 |
|---|---|---|---|
| 当前 SGLang + 补丁 | 正式 A 已过双 90、N14，源码和回放齐全 | 没证明执行层接近硬件上限 | 保留精确 A 作对照，继续机制优化 |
| **主办方 vLLM sm80 backport** | 题面明确 8×A100 实测部署、MTP、prefix、流式计数与 flush | 没有可对照的双 90/N@SLO 成绩；graph 的实际捕获覆盖待查 | **首个替代引擎对照** |
| 官方主线 vLLM 底包 | 主办方另有镜像；上游有模型 recipe | 上游 NVIDIA recipe 前提是 Hopper 或更新架构，不能等同 sm80 backport | 分开审阅，不混用命令与依赖 |
| TokenSpeed | 主办方有 A100 底包；有 GLM-5.3-Flash recipe、C++ 控制路径 | 已查性能示例不证明本负载收益；该镜像实际 SM80 后端待查 | 第二引擎候选，先筛兼容性 |
| TensorRT-LLM | 支持表有 GLM-5/5.2/5.3 的 GlmMoeDsaForCausalLM | 已查表中没找到 Flash 的 Glm5Next，不能混同架构；缺本模型 sm80 路径证据 | 暂缓整引擎移植，可借鉴算子 |
| Dynamo | 有 GLM-5.3-Flash 聚合/PD recipe | recipe 底层是 vLLM，示例为 H200/GB200 | 部署和状态传输参考，不算独立 kernel 引擎 |
| KTransformers | 主办方有底包，项目有原生 FP8 模型教程 | 已查教程是 SGLang+KT 的 CPU/GPU 路线，列 SM89/SM120、≥350GB RAM；未证明 A100 延迟收益 | 容量与分层 prefill 备选 |
| Transformers | 主办方有底包，可做功能参考 | 本轮无本负载 TP8 吞吐优势证据 | 暂不占完整压测队列 |

一手来源：[vLLM recipe](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash)、[TokenSpeed](https://github.com/lightseekorg/tokenspeed)及[模型 recipe](https://lightseek.org/tokenspeed/recipes/models#glm-5-3-flash)、[TensorRT-LLM 支持表](https://nvidia.github.io/TensorRT-LLM/models/supported-models.html)、[Dynamo recipe](https://docs.nvidia.com/dynamo/dev/recipes/glm-5-3-flash)、[KTransformers 教程](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/GLM-5.3-Flash-Tutorial.md)。

**vLLM 的三份实现不能混淆：**

- 题面 backport 是 prod-20675/vllm-backport:260918-sm80，主办方已验证接口和缓存。示例只有 16 个序列槽、graph 只捕到 16；这是接口范例，不是 N26 推荐配置。
- 主办方官方底包 arena-vllm-glm53:260918 是另一份镜像，题面明确不能混用启动参数。
- 公开 [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport) 可参考 SM80/Marlin/混合缓存实现，但尚未证明与题面镜像等价。其 4×A100 示例是 AWQ 4bit，不能证明原始 FP8 的 TP4 显存够用。

vLLM 也可能继续用 FP8 权重、BF16 计算的 Marlin；换引擎不会消除 A100 无原生 FP8 MMA 的约束。社区 backport 的 LMCache 示例还涉及状态对齐与 1152-token chunk，不能只看 offload 命中而忽略额外前向次数。

## 3. 借实现，不一定换整个引擎

### KDA 投影融合：现有条件可能挡住可用路径

本底包 [glm5_next.py](../../build/base_exact/sglang/srt/models/glm5_next.py) 的 do_fuse_qkvbfg 要求 quant_config 为 None，且 head shard 与 TP 相同。当前 KDA 投影实际在 BF16 不量化清单中，但模型级 FP8 配置不为空，因此走分开的模块。底包已经有 MergedColumnParallelRepeatedLinear 和 ColumnParallelBatchedLinear，不必从头实现所有 GEMM。

已读 vLLM [common/kda.py](../../refs/vllm-pr56960/vllm/models/glm5next/common/kda.py)，参考 commit 为 7565389f848994d5271986f74aab2af7ede0cbab：KDA 初始化显式保留 BF16；q/k/v/b/f_a/g_a 合为一次投影，f_a/g_a 分片跨 TP 复制。这不是题面镜像源码等价证明。

**候选是我们当前输入侧 4 次投影→1 次**，因为 q/k/v 早已合并；不能照抄上游“6→1”算增量收益。先验证权重加载、复制分片、f/g 第二级投影、scatter 的非整齐行数、MTP/graph 和状态输出，再测实际 T(c,P,B)。这是可执行的优化线索，尚无新 GPU 收益数据。

### Graph 与元数据

170 v2 的 scatter 修复已有 TP2 证据，真实 TP8 仍待数值复验；旧 v1 的 TP8 错误不能视为自动消失。其价值首先在小块固定成本，不能当成 16k 加速承诺。

[SGLang #39422](https://github.com/sgl-project/sglang/pull/39422) 的 DSA draft metadata graph 可单独筛选。原核验已经说明收益来自 4×GB300、decode 密集负载，不沿用加速百分比；正式 A 现有 MTP，值得重查其覆盖路径。同组少同步、launcher 缓存和布局优化须查是否已存在；Triton FP8 MoE autotune 不直接适用于 111 Marlin。

### MoE：切分结构与大块实现都要看

当前 TP8 将 routed expert intermediate=2048 切成每卡 256。EP8 可使每卡约 36 个 routed experts 保留完整 intermediate，改变 GEMM 形状与分派方式；它可能只通过一个 flag 开启，但属于执行结构变化。收益需连同负载不均、dispatch/combine、共享专家和通信一起测。

**待验证的正确性风险：** [111](../../engine/docs/111-sm80-fp8-moe-marlin.md) 未向 MarlinMoeQuantInfo 传 expert_map/global_num_experts；[StandardDispatcher](../../build/base_exact/sglang/srt/layers/moe/token_dispatcher/standard.py) 在 EP 下可能已把非本地专家映射为 -1；[Marlin 包装](../../build/base_exact/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py) 却依 expert_map 是否存在决定 EP 标志及部分输出清零。须检查 -1 的排序、跳过和归零契约，直接补一遍映射可能二次映射。**尚未运行复现，也不是当前 TP8/EP1 成绩受此 bug 影响的证据。**

另一候选是大块 prefill 选择性反量化 BF16 grouped GEMM，decode 保留 Marlin。保留 FP8 时，全专家 BF16 副本额外约 35.56GiB/rank；单 MoE 层 BF16 缓冲约 1.69GiB/rank，双缓冲约 3.39GiB/rank。先测反量化+GEMM+同步总成本，扣除 KV/状态池损失。[显存推算](../../evidence/cost-audit-20260924/cost-estimates.json)不含所有其他缓冲。旧 F70 INT8 W8A8 负面结果沿用，没有新形状/kernel 假设不重复测试。

### DSA、布局与通信

114 已在适用形状下拆 indexer 行，S1 已开 attention 输入 scatter 及相应 reduce-scatter，不能把旧全量冗余重复算成未来收益。DCP 的 041 行数错误是独立正确性支线，修好不自动证明更快。

先列实际 collective 的形状、字节、流和依赖，再决定融合/重叠。16k×4096×BF16=128MiB，不是每条通信都可泛称 64MB。图、融合、布局与通信会互改关键路径，不同基线测得的省时不能相加。

## 4. 拓扑、精度与“物理上限”的边界

- 当前 FP8 权重和状态池下，4+4 两个完整 TP4 实例不满足显存账：专家约 71.12GiB/rank，另有 KDA 投影、状态、KV、graph 等。不是证明所有缩池/量化/offload 的 TP4 都不可能。PD 还需传 KV、KPool、KDA/conv 状态并维护缓存连续性。
- 量化/状态压缩改变容量、成本和误差，可研究，但须从题面指定权重和行为出发；社区 checkpoint 的硬件结果不是我们双 90 的证明。原生 Blackwell NVFP4 快路径不适用于 A100。
- 22ms decode 不是已证明的带宽下限。均匀独立路由仅作反例，B=32 每层预计访问约 171/288 个 routed experts，不是必读全部专家。需测真实专家分布、HBM 流量、KDA/通信及 MTP 每步产出。
- 排队是延迟发生的位置，不是机制归因。算得贵也会造成排队；缓存修复也会减少其他请求的等待。调度可通过批形成、减少空转及尾延迟控制提高 N@SLO，即使孤立算子吞吐没变。
- 新 blocking.py 的窗口重合支持长冷链首是重点排查对象，但不能把窗口当 GPU 独占，也不能把估计成本的残差当已测 interleave。详见[复核](../../notes/codex-分析-阻塞归因与执行路线.md)。

## 5. 当前筛选顺序

| 对照 | 第一阶段问题 | 进入完整回放的依据 |
|---|---|---|
| 已排的正式 A@devN14、B@devN10 与 S1 混合 profile（044r/045r/046r；旧044/045缺121已作废） | 校准同档差异，确认当前 TP8 执行账 | 状态以 queue/pod 为准，不新增重复任务 |
| **首个引擎：主办方 vLLM backport vs 精确 A** | 同模型/协议/flush/计数、正确性、graph 覆盖、成本与内存；记录启动成本 | 接口与数值成立，再同档完整比较；跨引擎先评整体，随后归因 |
| **首个局部改造：BF16 KDA 投影融合** | 加载/TP 分片/状态一致性、实际调用数与 T(c,P,B) | TP8 正确、同条件收益超出重复测量波动 |
| 既有候选 170 v2 | TP8 数值、小块成本、捕获内存 | 先有省时证据，再让调度采用新成本曲线 |
| MoE 结构与大块双实现，分别对照 | EP -1/共享专家契约；总成本与显存 | 不以缓存损失换孤立 GEMM 的好看数字 |
| 第二引擎：TokenSpeed A100 底包 | 实际 Glm5Next/SM80 后端、混合状态及接口 | 可运行证据齐备再占完整档 |

所有完整回放仍如实报告 11 门；双 90 是最终新实现候选的必要验证，12 题冒烟仅用于排错。开发集不是 N26 预测器。CPU/单算子筛选通过后由 Claude 统一安排 8 卡对照，正式提交按用户安排。

本轮没有新增 GPU 性能数据、入队或正式提交。网页快照与获取时间在 [engine-survey-20260924](../../evidence/engine-survey-20260924/)，总体进展看 [queue](../../notes/queue.md) 和 [codex-分析](../../notes/codex-分析-2026-09-24.md)。
