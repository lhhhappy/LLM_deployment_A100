# 170 — GLM-5.3-Flash 预填充 breakable CUDA graph（移植上游 PR #38522，T52；v2 修 scatter，T52b）

## 问题
- 底包把 KDA 模型的 prefill CUDA graph 默认关掉（`arg_groups/cuda_graph_hook.py::disable_breakable_cudagraph_if_incompatible` 的 "KDA hybrid linear attention" 规则），GLM 每个 chunked-prefill 前向都是 eager。8 卡上反推每块截距 ≈150ms（F 条目，INFERRED），限制了用小 chunk 保护 decode TPOT。
- 上游 PR #38522（`refs/pr38522.diff`）给 GLM 加了显式 opt-in 的 BCG：kpool indexer 走 eager 桥、线性注意力 capture 桩、warmup 后清 mamba 槽等。

## 底包事实（读源码确认）
- **显式 opt-in 底包已生效**：`parse_cuda_graph_config` 把 `--cuda-graph-backend-prefill` 记入 `_cuda_graph_config_locked`，`apply_cuda_graph_compatibility` 一见 `(PREFILL,"backend")` 被锁就直接 return，KDA/多模态规则都不跑。所以 170 不需要改锁语义，只补注释。
- BCG 在 replay 时由 `load_batch` 按**白名单**新建 `static_forward_batch`；eager break 拿到的是 `get_tc_piecewise_forward_context().forward_batch`，也就是这个 static batch，参数是 capture 时的张量（弱引用）。按请求变化的东西必须在 eager break 里从 context 读。
- sm80 上 `set_dsa_prefill_impl` 本来就不选 MHA one-shot（只有 SM90/100），BCG 强制 MLA 与 eager 路径一致。

## 改动（锚点 `[ax] 170`，7 个文件）
1. 上游原样移植：
   - `cuda_graph_hook.py`：KDA 规则注释；`apply_glm5_chunked_prefill_default`（只在没给 `--chunked-prefill-size` 时设 4096）；`apply_glm5_prefill_cuda_graph_policy`。
   - `pipeline.py`：接入以上两个函数。
   - `model_config.py`：Glm5Next 加进多模态 BCG 白名单。
   - 新文件 `dsa/kpool_prefill_cuda_graph.py`：kpool indexer eager 桥，从 context 取 live batch，截到真实 token 数，把结果写进静态 padded buffer，尾部填 -1。
   - `dsa_indexer_kpool.py`：`forward_cuda` / `_forward_cuda_impl` 拆分；BCG 预填充时关掉 dual stream。**桥调用的是我们的 `_forward_cuda_impl`**，所以 110/112/113 的 sm80 kernel 和 114 的行切分 all-gather 都在 eager break 里用真实 batch 跑；capture 时走桩，不做 collective。
   - `radix_linear_attention.py`：a/b 用 narrow 截 token 维；capture 桩 `output.zero_()`。
   - `prefill_cuda_graph_runner.py`：warmup 前后清掉 capture 用到的 mamba 槽。
2. **本栈补丁（上游没有）**：
   - `prefill_cuda_graph_runner.load_batch`：把 140 的 `ax_kda_snapshot_offsets/slots` 透传到 static batch。白名单里没有这两个字段；不透传的话，BCG 下 KDA break 读到 None，140 的角色快照会被静默跳过，调度器却照样把额外槽交给 radix 树（旧状态被当成缓存命中）。这两个张量只在 eager KDA break 里读，用 live 张量即可。没有 140 的栈上用 getattr，是空操作。
   - GLM 捕获上限改成 `min(4096, chunked_prefill_size)`（上游固定 4096）：本栈没有 mixed chunk，单次 prefill 前向不超过 chunk，大桶只浪费 capture 时间和池内存。只在用户没锁 `max_bs/bs` 时生效。
   - `_ax170_align_prefill_buckets_for_attn_tp_scatter`：开 `--enable-attn-tp-input-scattered` 且 tp>1 时，把桶向上对齐到 tp_size 的倍数，做法同 `apply_deepep_adjustments`。**这只是必要条件，不能单独修复 scatter**，真正的修复见 v2。

## v2（T52b）：`--enable-attn-tp-input-scattered` 下输出错误的根因与修复
- **现象（8 卡，coordinator 提供）**：026j（BCG + scatter）能力冒烟 0/12，输出看似流畅但错误；026k（eager + scatter）12/12；026l（BCG 无 scatter）12/12。v1 旧版只保留在 git 历史中。
- **开发机复现（实测，TP2，GPU0+1，上下文敏感测试权重）**：BCG+scatter 对 eager+scatter，22 个请求中首 token 只有 10 个相同，top-5 logprob 最大差 4.22。**正好是桶大小的长度（512/1024/4096）也错**，所以根因不是 n≠B 的 padding 问题。
- **根因（读代码确认）**：
  - `prefill_cuda_graph_runner.py::_run_forward` 的 BCG/Full 分支 capture 时直接调用 `self.layer_model.forward(...)`，绕过了外层 `Glm5NextForConditionalGeneration.forward`（glm5_next.py:1535）里的 `get_attn_tp_context().maybe_input_scattered(forward_batch)`。`attn_input_scattered` 只由各模型外层 forward 设置，默认 False（runtime_context.py:606）。
  - 所以 warmup 和 capture 都在**非 scatter 布局**下录制：all-reduce、完整 residual、第一层不做 reduce-scatter。
  - replay 时 `_execute_body_capture` 仍 eager 执行外层 forward，**scatter 为开**：`VocabParallelEmbedding.forward`（vocab_parallel_embedding.py:573）在 scatter 下不做 all-reduce，于是把每卡的部分和 embedding 拷进 input_embeds 槽，捕获的图当成完整 embedding 使用，每一层都算错。
  - 这是底包 BCG 的通用缺陷，DeepSeek-V2/V4 等所有在外层包 `maybe_input_scattered` 的模型都一样，不是 170 的桥或 slicing 引入的；只是 170 让 GLM 第一次走到这条路径。coordinator 怀疑的 eager break `x[:n]` 切片：scatter 下 DSA/线性层输入都已预先 all-gather 成 B 行（`dsa_pre_gather`；线性层 `qkv_latent_func=None`），n 是全局真实 token 数，所以切片没问题。
- **修复（v2，`[ax] 170 v2`）**：
  1. capture 时用 `maybe_input_scattered(forward_batch)` 包住 `layer_model.forward`，决策与 replay 时外层 forward 相同（`use_input_scattered`：extend 且不是 verify）。scatter 下断言桶是 tp_size 的倍数（v1 已按 tp 对齐）。
  2. `can_run_graph` 增加保护：replay 时的 scatter 决策与 capture 时不同就回退 eager（按逻辑不应触发，只作兜底）。
- **验证（实测，`evidence/T52b/SUMMARY_v2.txt`，22 个请求：冷启动长度 37/100/500/512/1000/1024/3000/4096/5000、P=2万/10万 前缀命中 × c=100/1000/3000、并发混合对）**：
  | 配置 | 首 token 相同 | 生成 logprob 最大差 | 首 token top-5 最大差 |
  |---|---|---|---|
  | TP2 scatter，eager 对 BCG，**v1** | **10/22** | **1.53** | **4.22** |
  | TP2 scatter，eager 对 BCG，**v2** | 22/22（全部 token 相同） | 0.141 | 0.277 |
  | TP2 无 scatter，eager 对 BCG，v2 | 22/22 | 0.067 | 0.106 |
  | TP1，eager 对 BCG，v2 | 22/22 | 3.8e-4 | 0.219 |
  | 参照：TP2 eager 无 scatter 对 eager scatter | 22/22 | 0.145 | 0.150 |
  所有 BCG 臂的 prefill 都走图（51/51）。v2 在 TP2 scatter 下的残差（0.14）与“只换通信布局”的参照（0.145）同量级；v1 是 10 倍以上的错误，并且首 token 翻转。
- **未做**：TP2 同配置 eager 两次重启的噪声底；DONE 已交给 T50b，GPU 让出，没有再跑。0.14 的残差按推断归为 TP2 下 NCCL 归约顺序差异，最终要以 8 卡能力冒烟 12/12 + numcheck 为准。
- **fuzz=0（实测）**：全栈（…160 170，含 MTP 顺序）、tier-1+140+120+170、tier-1+170、026 栈 B+114+140+120+170 都通过，全栈结果与移植树逐字节相同。
- **历史复跑条件**：TP2、GPU0/1，BCG/eager 与 scatter/no-scatter 四臂，使用只在测试树应用的[上下文敏感初始化补丁](../../evidence/T52/t52_test_dummy_init.patch)。原一次性 runner 和汇总脚本已清理；[T52b 原始结果](../../evidence/T52b/)保留，本段不是现行命令。

## 按请求变化的部分：graph-safe 还是 eager break（静态审查）
| 组件 | BCG 下 | 依据 |
|---|---|---|
| DSA 稀疏注意力（tilelang prefill、KV 写入） | eager break | `mla_bmm_then_unified_attention` / `breakable_unified_attention_with_output`；`_unified_attention_with_output_impl` 截到真实 token，padding 行清零 |
| kpool indexer（110/112/113/114） | eager break（170 桥） | `seq_lens_cpu.item()`、kpool plan、按请求循环都在 break 里；`get_is_capture_mode()` 在 BCG 内为 True，只会关掉 alt-stream 延迟写 cache（行为保守，仅性能差别） |
| KDA + conv + 140 快照 | eager break | `bcg_unified_linear_attention_with_output`；元数据在 replay 前由 `init_forward_metadata(live batch)` 重建；140 字段靠 170 透传 |
| mHC（hc_pre/hc_post） | 图内 | 逐 token 张量运算，无 host 同步 |
| MoE Marlin（111） | 图内 | 形状只依赖 bucket M；decode graph 已在用；padding 行只是多算 |
| attn-tp scatter 通信 | 图内 collective | bucket 对齐见上；TP8 未测 |
| logits / 采样 / logprob | eager 尾部 | BCG 只 capture 层体（body capture），`return_logprob` 仍可 replay |

## 验证（开发机 GPU1，TP1，8 层 rank 替身，dummy 权重，全栈 000…160+170，`SGLANG_AX_KDA_DUAL_SNAPSHOT=1`）
当时的运行脚本已清理；实验事实见[证据](../../evidence/T52/)。
- **fuzz=0**：全栈（000 101 105 106 110 111 112 113 114 115 140 120 130 150 160 170）、tier-1+140+120+170、tier-1+170 都能应用；全栈结果与移植树逐字节相同；开发机上也重新从补丁构建过。
- **启动/capture（实测）**：`--cuda-graph-backend-prefill breakable --chunked-prefill-size 4096` 时捕获 50 个桶 `[4,8,…,1024,1280,…,4096]`，17–27 s，池 1.25 GB。所有 prefill 行都显示 `cuda graph: True`（173/173、132/132、41/41）。
- **数值（实测）**：13 个请求。P∈{0, 2万, 10万}（实际命中 0/19968/99968）× c∈{100, 1000, 3000}（都不是桶大小，replay 时 padding 到最近的桶），外加两个预热请求，以及并发混合对（2万前缀+500，冷 1500）。每个请求 greedy 生成 32 个 token，带 top-5 logprob。
  - 默认 ±1e-3 dummy 权重下，BCG、eager、base（无 170）逐位相同。但**负对照**（eager、chunk 2048 对 4096）也逐位相同，说明 logits 只看当前 token，这组结果不能作为证据。
  - 改用仅测试用的[初始化补丁](../../evidence/T52/t52_test_dummy_init.patch)（norm=1，scale=1，矩阵 ±1/√fan_in，只在测试树，**不能进入产品补丁栈**）后：
    | 对比 | token 一致 | 生成 token logprob 最大差 | 首 token top-5 最大差 |
    |---|---|---|---|
    | eager170 对 BCG | 13/13（32/32） | 3.4e-4 | 0.156 |
    | base（无 170）对 eager170（同 kernel，只是重启） | 13/13 | 3.4e-4 | 0.156 |
    | base 对 BCG | 13/13 | 6.4e-4 | 0.218 |
    | 负对照 chunk 2048 对 4096 | 13/13 | 7.7e-4 | 0.141 |
  - 结论：BCG 与 eager 的差异落在重启间噪声带内（代码相同的两次 eager 也差 3.4e-4），小于换 chunk 的差异；10 万前缀、混合 batch、后续 decode 都没有分叉。局限：替身权重下 top-1 很尖，逐 token 比较不敏感；主要看 top-5 logprob。140 的角色快照没被触发：替身词表 19360，放不下角色 token 154827，只走了尾部快照路径。
- **速度（实测，chunkcost，`--enable-metrics`，服务端从收到请求到 prefill 完成，每点取 2 次中较好的一次）**：
  | c | P=0 eager → BCG (ms) | P=98304 eager → BCG (ms) |
  |---|---|---|
  | 512 | 40.1 → 27.4 | 63.4 → 48.3 |
  | 1024 | 40.1 → 30.2 | 64.4 → 52.3 |
  | 2048 | 44.4 → 37.9 | 71.7 → 64.4 |
  | 4096 | 64.2 → 62.8 | 98.6 → 96.1 |
  线性拟合：P=0 从 33.7ms+7.0µs/tok 变为 20.3ms+10.1µs/tok；P=98k 从 55.0+10.2 变为 39.2+13.6。曲线是凸的，拟合斜率会误导；应看小块固定开销：c≤1024 时 8 层省 9.9–15.1 ms（约 1.2–1.9 ms/层）。
- **decode（实测，同一 chunkcost）**：BCG 服务的 decode 步略慢，bs1 p50 2.95 对 2.74 ms，bs32 3.34 对 3.26 ms；另一轮是 2.79 对 2.72。原因未查（推测是更多 graph/闭包对象带来的 Python GC 或调度开销）。**TPOT 是决胜项，8 卡必须同时看 TPOT。**
- **单测**：当时的 140 快照对照 8/8 case 通过（逐位相等）；测试已迁至 `tests/gpu/test_kda_snapshot_140.py`，170 没有改 140 的任何文件。

## 推断（未测）
- 真模型 45 层（替身 8 层）。如果每层省的 launch 开销同比例放大，小块能省约 55–85 ms（1.2–1.9 ms/层 × 45），相对 8 卡约 150 ms 的截距是大头；但 TP8 的 kernel 和通信不同，必须 8 卡实测。capture 时间估计 ×5–6（约 1.5–2.5 分钟），池内存待测。
- 4096 块几乎不省；收益集中在小块，正好对应“用小 chunk 保护 TPOT”的方向。

## 风险
1. `--enable-attn-tp-input-scattered` + BCG：v1 在 8 卡上确实出错（026j），v2 已修，TP2 实测正确；TP8 需重跑能力冒烟 + numcheck 确认（命令见 v2 节）。
2. MTP（160）+ BCG：未验证。draft 的 prefill graph 与 kpool 桥条件（`is_extend_without_speculative`）的交互没审完；140 本身也禁止 NEXTN。先不要组合。
3. decode 步小幅变慢（见上）。
4. 超过捕获上限（`chunked_prefill_size`）的前向回退 eager；padding 超过 2 倍也回退 eager。
5. 170 在 eager 路径只多一层 `forward_cuda` 包装；实测 base 与 eager170 的差异在噪声内。

## 启动参数（8 卡首测）
在现有 G_ARGS 基础上：`--cuda-graph-backend-prefill breakable --chunked-prefill-size 4096`（或 2048/8192 做 A/B；上限自动取 min(4096, chunk)，要捕获更大的块需显式给 `--cuda-graph-max-bs-prefill`）。
建议先做 A/B：同一配置只切换 `--cuda-graph-backend-prefill breakable|disabled`，看 chunkcost 截距、decode 步和 TPOT。**scatter + BCG 必须用 v2**，并先过能力冒烟 12/12 和 `numcheck.py`（对 eager 参照做 `numcheck_cmp` wrong=0），再进梯子。不要同时开 MTP。
