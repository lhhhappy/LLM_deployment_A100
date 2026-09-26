# 审查：Codex 的 DCP 本地续算路径（codex/dcp-prefill-local-kv，f34c7ac4）

2026-09-26 18:00 UTC，fable 委派的独立审查代理（只读代码与证据，跑了 CPU 用例；未上 GPU/Pod）。worktree `build/worktrees/dcp-prefill-local-kv`。行号以该 worktree 为准。

**结论：还不能上 TP8 探针；修掉 B1、B2 之后可以。** 支持范围内没有发现本地路径的正确性错误。第 3950 行的 FAIL 是 top-k 选键近似平局的翻转，来源是舍入级噪声；证据排除了索引/偏移错误，但分不清是新路径引起还是底包自身的运行间噪声。

## 改动范围（引擎）
`kernels/ops/attention/dcp_local_indices.py`（新，Triton `_local_indices`、`local_dcp_indices`）；`layers/attention/dsa_backend.py`（`_should_return_dsa_dcp_lse` 新增 `local_extend` 参数、`forward_extend`）；`layers/dcp/local_extend.py`（新：`uses_local_extend`、`LocalExtendPolicy`）；`layers/dcp/metadata.py`（新字段 `dcp_local_extend`）；`model_executor/runner/eager_runner.py`（`__init__`、`_execute_extend`）；`models/deepseek_common/attention_forward_methods/forward_mla.py`（新 `is_dcp_mla_partial_phase`；`forward_absorb_prepare` 的 Q gather 与 `forward_absorb_core` 的合并）。分支基于 f546934e，三个 DCP 补丁逐字节搬入（即已是"S1 修正 + DCP"的整合）。

## 阻塞项
- **B1 每个 batch 尺寸都重编译 Triton kernel**：`dcp_local_indices.py:12-15` 把 `ROWS/COLS/S0/S1` 声明为 `tl.constexpr`，`ROWS` 是补齐后的查询行数，新的 T 就在 forward 里 JIT 一次，所有 rank 等；短尾窗口最多 512 种。开发机计时先预热 5 次，看不到。改成运行时参数。
- **B2 任务门看不见开关**：`scheduler.py:1295-1348` 的 `_ax_mechanism_report` 没有 `SGLANG_AX_DCP_LOCAL_EXTEND`/`LARGE_MAX` 的条目，G_EXPECT 无法证明 ON 臂真的走了该路径；只有两行 INFO（`local_extend.py:113`、`eager_runner.py:326-332`）。加机制 token 和按路由计数的 batch 计数器。

## 非阻塞
- N1 压缩前提只在 `SGLANG_AX_DCP_COMPACT_TOPK` 的设置里检查（`dsa_backend.py:600-613`），本地路径总以 `gcd(W, KPool)` 压缩列（`:3241`）；生产设 COMPACT_TOPK=1 所以今天成立，应在 `from_runner` 里再查一次。
- N2 只开主开关时，只有 ≤512 补齐 token 且前缀 ≥ max(4096, ~48·T) 的 batch 走本地路径（`local_extend.py:35-42`）；开场的冷链首没有前缀，永远不走。不要指望主开关改善 chain。
- N3 新 CPU 用例只覆盖 Python 策略；kernel 的索引数学由审查者用纯 Python 仿真：对 600 个位置与未压缩的 owner 过滤 0 处不一致（错位对照 496/600，说明检查灵敏）。

## 已核实
- 默认关闭时执行路径不变：`from_runner` 返回 None（`local_extend.py:49-50`）；planner 用关键字构造元数据、新字段默认 False（`metadata.py:40`）；116 的 gather 分支、LSE 谓词、`nan_to_num` 行为如前；`is_dcp_mla_partial_phase` 退化为 decode 判断；decode/verify/draft-extend 图不可能走该路（`forward_batch_info.py:144-153`）。关闭时无新分配、无新 collective、无 dtype 变化。
- owner 映射一致：写入 `loc % W` 的 rank、本地行 `loc // W`（`mla_buffer.py:110-114`）；decode 读取（`dsa_backend.py:193-207`）、搬运（`memory_pool.py:4535-4545`）、新 kernel（`dcp_local_indices.py:30-31`）同一规则；KPool 4 组展开与偶数页大小保证列奇偶一致；MTP 接受 token 搬回自身位置（`spec_utils.py:742-767`）；NextN 草稿同一写入/读取路径，自建 ForwardBatch 独立选路。
- 无跨阶段陈旧状态：只在 EXTEND/MIXED 生效，每次 forward 重选路，索引表每次新建，路由不一致时 116 守卫报错。
- 临时内存（按代码估算，T 补齐行数、H=8、W=2、D=512）：每层约 80–100 KiB/补齐 token；T=512 约 50 MiB，T=8192 约 0.65–0.8 GiB（作者实测 740 MiB 相符），逐层释放；LARGE_MAX=8192 时每层多约 0.75 GiB，须 TP8 实测。
- 合并是标准 fp32 log-sum-exp（复用 decode 的合并，`comm.py:112-150`、`dcp_kernels.py:187-283`），空本地分片和全 mask 行处理正确；比旧的单遍多一次 bf16 舍入。

## 第 3950 行
所有通过的对照（含底包对底包）都相差最多 1 个 bf16 ulp：底包自身就不是逐位可复现的（融合 top-k 用 atomicAdd 顺序写入选中组，`kpool_topk_transform.cuh:112,129,169,187`，累加顺序随运行变化）。本地路径 8K 在 7 次里翻 4 次、底包 A/A 一次没翻，样本太小；翻的永远是同一对（第 7 层 21028–31 ↔ 23340–43）。选键钉住后两 rank 都通过（max_row_rel 7.1e-3）。索引/偏移错误不成立。分数差没有量化（没存 indexer logits）。定案实验：底包 A/A 跑约 10 次并开捕获，两臂都导出第 3950 行第 7 层的 indexer logits（第 512/513 个值）和两组的 index-K 字节；底包也翻则是 (ii)，且没有 A/A 带宽的 1% 逐行门对稀疏注意力不是合格的通过判据。

## TP8/生产风险
只接受 W=2（`local_extend.py:56`）；owner 只看位置不看块边界；HiCache 逐层 host 等待仍在注意力前调用（`memory_pool.py:4354-4356`）；CUDA graph 不受影响；TileLang 的 16 头 1088 宽变体 decode 已在编译，无新编译；既有风险：top-k 阈值桶超过 4096 候选按原子顺序丢弃，<1024 行的 batch 跳过 indexer 行分片、各 rank 独立选键而合并假设选键相同——DCP decode 已如此，本地路径把它扩展到短尾。

## CPU 用例
13/13 通过（`test_dcp_local_extend` 6、`test_dcp_mtp_review` 7）。hicache180 的 harness 本容器无 torch 未计。

## 探针应记录
每条路由的 batch 数（含 T、P）；on/off 之外加 off/off 的 A/A，固定长前缀短尾 prompt，比较 greedy + top-k logprob 分布而非逐 token 相等（12 题冒烟太粗）；TP8 抽样行的选键捕获（集合重叠、第 512 组的 logit 差）；MTP 接受长度分布；每 rank 峰值显存、两臂相同 KV 池；每个新 T 首次出现的延迟离群。
