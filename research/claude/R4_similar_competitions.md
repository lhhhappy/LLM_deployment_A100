# R4 — 类似的比赛、基准与公开方案（作者：Claude，2026-09-22，只做调研）

标注：V = 读了原文；I = 推断。

> **更正（据 Codex W1 F27，2026-09-22）**：下文"decode 间隔 +141% 吞吐、−97.3% 延迟"并非 AgentX 负载上的结果，而是 SGLang PR #35017 在 8×GB300 上用**定长**负载（ISL 131072 / OSL 1024，并发 64，interval 0→4）测的，指标是 p99 ITL；同一测试中 TTFT p50 从 36.5s 升到 59.0s。该参数在 v0.5.20 里已有，对 GLM 的默认值解析为 0。另外，D1 切出来的 extend 也会触发这个间隔，两者要一起评估。

## 结论速览
1. **最像本赛的是 SemiAnalysis 的 AgentX / InferenceX v3**（V）：真实 agentic coding trace（393 个 Claude Code 会话），中位输入 142k token、中位输出 444 token，前缀命中率 96% 以上。它的测法和我们一样：**闭环、按并发 agent 客户端数扫描**，每个配置跑一小时。受测模型里**有 GLM 5.3**，但那是 744B 的大模型，不是我们的 Flash，只能参考方法，不能照搬结论。随基准一起合入了 50 多个上游 PR，覆盖 vLLM、SGLang、TRT-LLM、Dynamo、LMCache 等，可以当作"这种负载下大家修了什么"的清单。
2. **没有找到 Kaggle 上同类的"部署调优"比赛**（I，只做了有限检索）。Kaggle 的 LLM 比赛主要比模型效果。和本赛同类的是 HPC 学生赛（ASC24 的 LLM 推理优化题）和 MLPerf Inference（Interactive 场景对 TTFT/TPOT 设了门槛）。
3. **能直接用到本赛的新线索**（I，待核实）：
   - SGLang 在 DP-attention 下新增了**可配置的 decode 间隔**：强制在 prefill 之间插入 decode 轮次，报告输出吞吐 +141%、延迟 −97.3%，代价是 TTFT 升高。这对应 R2 提过的 `--prefill-decode-interval`，也对应本赛"解码门"和 `tpot_mean` 排名。
   - SGLang 已有**缓存感知的 DP 路由**和 **HiCache 混合模型卸载**（AgentX 优化清单）。后者要注意：本赛模型带 DSA，恢复路径有 #39156 的问题。
   - AMD ATOM 做了**按内容寻址的 checkpoint 生命周期**，思路是生成出来的 turn 自己留下可恢复点，与 D1 同一族。

## 1. SemiAnalysis AgentX（InferenceX v3）
- 负载（V）：393 个 Claude Code 会话，每个会话不少于 20 个请求；按 DAG 重建主 agent、并行子 agent 和辅助请求，保留工具执行间隔；有完整版（最长 1M token）和 256k 截断版两种。https://inferencex.semianalysis.com/agentx
- 测法（V）：并发数指"并发的 agent 客户端"，是闭环，配置越快、完成的请求越多；报告吞吐，同时报告 TTFT 和交互性（每用户 token/s），不只看单一延迟。**和本赛"N = 逻辑会话数"的定义一致。**
- 工程教训（V，方法论页）：
  - KV 容量压力大；**错误的前缀命中会产生错误状态**，漏掉的命中则要付出重算代价；
  - 从主机内存搬长前缀可能比重算快，但传输粒度和描述符开销很关键；
  - 路由器的 CPU 开销（前缀匹配、过期跟踪）会随在线会话数增长；
  - 注意力、稀疏注意力和 CUDA graph 的选择在不同序列长度下表现不一样。https://inferencex.semianalysis.com/blog/agentic-benchmark-agent-benchmark-guide
- 优化清单（V，只有条目、没有数字）：SGLang 做了滑窗分配、HiCache 混合卸载、运行时标量化上下文长度、缓存感知 DP 路由；vLLM 做了混合注意力前缀保留、混合模型 CPU KV 卸载（#45879、#45720、#45546）；TRT-LLM 做了边界感知的增量分词；AMD ATOM 做了稀疏 checkpoint 保留、循环状态 checkpoint。https://inferencex.semianalysis.com/agentx/optimizations
- 文章要点（V）：混合模型的循环状态"无法从周围 token 重建"，必须保留；DP-attention 下，一个 rank 被 chunked-prefill 续跑喂满时会一直赢得 prefill 优先权，其他 rank 的 batch 只能等，SGLang 因此加了可配置的 decode 间隔（+141% 输出吞吐、−97.3% 延迟，TTFT 变差）。https://newsletter.semianalysis.com/p/agentx-inferencexv3-does-cuda-moat
- vLLM 的对应博客（R3 已收录）：`--long-prefill-token-threshold`，以及会话粘性路由优于负载均衡。

**对本赛的用法**（I）：
1. AgentX 的"闭环 + 按并发扫描"就是本赛的 N@SLO 梯子，思路一致；
2. 它的优化清单可以当作 SGLang 侧的 PR 检查表，逐条确认 v0.5.20 和底包里有没有；
3. 本赛的门是按分桶 TTFT p95 卡的，比 AgentX 更严，所以 AgentX 里"用 TTFT 换吞吐"的优化（如 decode 间隔）要小心取舍，只有 `tpot_mean` 或 `tpot_p95` 成为瓶颈时才用。

## 2. 其他同类比赛与基准
| 名称 | 形式 | 可借鉴点 | 适用度 |
|---|---|---|---|
| **ClawPerf**（GitHub ucm-system/ClawPerf）（V，摘要） | 开源压测工具：模拟多用户、长对话、长上下文的 agent 负载；**按 SLO 做容量扫描**（并发先几何级增长，再二分细化出满足 TTFT/TPOT 约束的最大用户数） | 和本赛梯子同构；二分细化的思路可以用来**在开发集上快速找出临界 N**，省 8 卡时间 | 高（方法） |
| **MLPerf Inference** Interactive 场景（V） | 固定 TTFT 450ms / TPOT 40ms 的门限，比吞吐 | AMD 的 v5.1 提交：小 GEMM 调优，vLLM v1 混合 prefill/decode；TTFT 与 TPOT 此消彼长，由 `max-num-seqs` 调节 | 中（单轮负载，没有前缀复用） |
| **ASC24** 学生超算赛 LLM 推理题（V） | LLaMA2，比吞吐 | 冠军（华中科大）：TP、KV cache、PagedAttention、异步流水 | 低（比吞吐，无 SLO） |
| Kaggle | 未找到部署调优类比赛（I） | — | — |
| 学术：SCOOT（SLO 导向的引擎参数自动调优）、CacheOPT、AdaServe、Nitsum（V，标题与摘要） | 在 SLO 约束下自动调参、调度 | SCOOT 的"贝叶斯优化 + SLO 约束"可以用来系统地扫 chunk 大小、max-running-requests、MTP 深度 | 中（方法） |

## 3. 建议加入方向清单的项（待与 Codex 对齐）
- **D2 补充**：SGLang 的 decode 间隔参数（R2 的 `--prefill-decode-interval`，核实 v0.5.20 里的确切名字），加入调度扫描。
- **D3 补充**：DP-attention 必须配缓存感知或会话粘性路由（AgentX 和 vLLM 博客都支持这一点）。
- **方法**：参考 ClawPerf 的"几何扫描 + 二分细化"，在开发集上快速定位每个配置的临界 N；参考 SCOOT 的思路系统化扫参，减少 8 卡上的试错次数。
