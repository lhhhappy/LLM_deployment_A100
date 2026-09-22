# R7 — KDA 单次 forward 内部快照：vLLM #50587 与 SGLang Plan B

作者：Codex；2026-09-22 UTC。T10 / R6 补充。**仅网页、公开 GitHub API 和本地源码调研；没有实现补丁、运行测试、模拟、服务或 GPU 实验。**

本地基线：`src/sglang` HEAD `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`，工作树干净。以下 SGLang 文件位置均属于此 commit；不能据此断言主办方镜像包含相同实现。`VERIFIED / SOURCE` 表示源码事实，`VERIFIED / REPORT` 表示核实了上游报告而非复现，`INFERRED` 表示设计判断。

## 1. 结论与交接

1. **VERIFIED / SOURCE：本地 SGLang 已能在一次 KDA extend 中导出一个内部 FP32 recurrent 快照，且继续算到末尾。** 并不是必须先新增一套中间状态 kernel。入口已有 `track_state` / `track_chunk_idx`，Triton 从 FP32 累加状态直接写快照。
2. **VERIFIED / SOURCE：现有限制是每请求每次 extend 一个被 track 的缓存位置。** 正在运行的末尾 state 和已经进入 prefix cache 的末尾 checkpoint 不是一回事。只把原位置替换成 role 可以复用现有 kernel，但不等价于 D1 的 role + end，也不能套用 F13/F24 收益。
3. **INFERRED：真正的双点 Plan B 可从现有单点路径扩展，主要工作是多位置元数据、conv 窗口、独立 state slots、树插入及失败清理；Triton 侧改动可以限定为少量选定 chunk 的额外 store。** 不是把 ping-pong 槽从 2 改 3 就完成了。
4. **VERIFIED / SOURCE：#50587 的具体合并实现是 #52789 → #53614。** 合并代码使用 FlashKDA 内部导出接口；PR 首屏仍描述两次 FlashKDA 调用，不能以该图代表最终代码。单次 model forward / KDA 调用也不等于只有一个 CUDA kernel。
5. **INFERRED：继续维持 decisions #11 的 A 优先、B 后续。** F24 已补上保守策略证据；无需为了证明“能拿到内部状态”而扩写全部 kernel，但也不能绕过缓存正确性和真实 SLO 验证。本文不变更共同方案、不启动 T7。

角色边界本身已有 llama.cpp 合并先例，见 [R6 §2](R6_prior_art_cn_github.md)。它属于切分 batch 的 Plan A；这里调研的是不因 checkpoint 再切 model forward 的 Plan B。

## 2. vLLM：追到具体实现，而不是只引用性能标题

### 2.1 #50587 → #52789：内部 checkpoint 与性能适用范围

**VERIFIED / SOURCE**：[Kimi K3 性能追踪 #50587](https://github.com/vllm-project/vllm/issues/50587) 明确链接以下两项，不是一个可直接移植的补丁。

| 项目 | 核对状态 / 固定版本 | 与本题关系 |
|---|---|---|
| [vLLM #52789](https://github.com/vllm-project/vllm/pull/52789) | 2026-08-22 merged；`9eb9d9d3953959695108600c8ed33d36bc6a1e5f` | 不为 Mamba cache checkpoint 再拆完整 model forward |
| [FlashKDA 导出实现](https://github.com/vllm-project/FlashKDA/commit/ee0be888cd0e972f9409bf53756f8c38c6652173) | vLLM 所钉住的 `ee0be888cd0e972f9409bf53756f8c38c6652173` | recurrence 内部写一个指定位置的状态 |
| [vLLM #53614](https://github.com/vllm-project/vllm/pull/53614) | 2026-09-06 merged；`144e79c8106da23141ac010394b782f730cc7fe8` | 补充 partial prefix、spec/Eagle rewind、connector 恢复后的准确位置 |

**VERIFIED / REPORT**：#52789 报告 Kimi-K3、TP8、合成 8k 输入 / 1k 输出；并发 1/4/16 的 TTFT 为 435.28→394.19、1450.94→1080.78、2648.48→2099.69 ms。报告的吞吐增幅只有 4.0% / 1.4% / 2.4%。正文未明确 GPU 型号，且本次未建立所测 revision 与最终合并代码的一致性；不能称为 A100 上 agent 多轮负载的复现，也不能把 TTFT 百分比套到 N@SLO。[性能报告](https://github.com/vllm-project/vllm/pull/52789)

**VERIFIED / SOURCE：最终代码与正文示意图不同。** [合并版 KDA wrapper](https://github.com/vllm-project/vllm/blob/9eb9d9d3953959695108600c8ed33d36bc6a1e5f/vllm/models/kimi_k3/nvidia/kda.py#L861) 一次调用 `_flashkda_prefill`，传入完整 query、`checkpoint_state`、`checkpoint_offsets`，同时得到完整输出和 final state。随后 `_store_cache_checkpoints_kernel` 把 recurrent checkpoint 及该 offset 前的原始 QKV conv 窗口写入缓存槽。`num_prefill_checkpoint_blocks=1`：仍是**每序列一个内部点**，不是任意多点 API。

### 2.2 FlashKDA 内部到底导出了什么

**VERIFIED / SOURCE**：固定版本 [C++ API](https://github.com/vllm-project/FlashKDA/blob/ee0be888cd0e972f9409bf53756f8c38c6652173/csrc/flash_kda.cpp#L83) 要求 checkpoint 张量为 contiguous CUDA FP32 `[N,H,D,D]`，offset 为 `[N]`，整数类型与 `cu_seqlens` 一致。[recurrence kernel](https://github.com/vllm-project/FlashKDA/blob/ee0be888cd0e972f9409bf53756f8c38c6652173/csrc/smxx/fwd_kernel2.cuh#L488) 在 `(t+1)*CHUNK == offset` 时将当前状态经 shared memory 和线程屏障写出，然后继续推进。

**重要精度区别（VERIFIED / SOURCE）**：该 FlashKDA commit 的 `resident_state` / `state_acc` 是 **BF16**；导出时转 FP32，`StateFP32` 的输入/输出路径也包含 BF16 转换。不能把“FP32 输出 buffer”写成“保留了未舍入的 FP32 recurrent 累加器”。这是该 kernel 的数值实现，不等于其 checkpoint 必然错误；但它与下文 SGLang Triton 的 FP32 累加器快照不同。[状态定义与转换](https://github.com/vllm-project/FlashKDA/blob/ee0be888cd0e972f9409bf53756f8c38c6652173/csrc/smxx/fwd_kernel2.cuh#L290)

**VERIFIED / SOURCE**：[vLLM 的构建配置](https://github.com/vllm-project/vllm/blob/9eb9d9d3953959695108600c8ed33d36bc6a1e5f/cmake/external_projects/flashkda.cmake) 包含 sm90a / 后续架构，没有 sm80。这里能借的是设计，不能把依赖直接换上去就当 A100 路线成立。

### 2.3 #53614 最值得移植的是位置合同

**VERIFIED / SOURCE**：它把 checkpoint 的有效性统一给 scheduler、worker、cache manager 使用；只为确实能导出的状态分配/哈希槽，并处理 spec rewind、partial prefix 和 connector。回归案例是本来登记为 `hash@96` 的槽实际存 `state@64`，需要重设 key 并去掉旧 key。[PR 说明](https://github.com/vllm-project/vllm/pull/53614)

[固定版本的共享 helper](https://github.com/vllm-project/vllm/blob/144e79c8106da23141ac010394b782f730cc7fe8/vllm/v1/kv_cache_interface.py#L961) 区分 hash block、Mamba block、query 起止和 kernel alignment；检查 offset 相对本次 query 是否可导出。**INFERRED**：SGLang Plan B 也应有唯一的“真实 token 深度 → recurrent / conv / radix key”描述，不能由三处各自 floor 一遍，更不能把控制索引的 `+1` 写进真实 key。

**VERIFIED / REPORT**：#53614 的多轮测量是 TP8/EP8、两台各 4 GPU、8 streams、120k→164k 逐轮增长、带 spec 的 partial-match 对比。它增加了多轮证据，但不是 role-breakpoint A/B，也没有证明本赛 A100 的收益。[测试范围](https://github.com/vllm-project/vllm/pull/53614)

## 3. SGLang 已有能力：从调度字段一直追到 FP32 store

以下路径相对 `src/sglang/python/sglang/`。

| 层次 | VERIFIED / SOURCE 的现有行为 | 对 Plan B 的约束 |
|---|---|---|
| `srt/managers/schedule_batch.py:2893` | 每 req 返回一个 `_MambaRadixCacheV2TrackEntry`；branch 可以替换原 end-ish 位置 | 当前不是 track 列表 |
| `srt/layers/attention/hybrid_linear_attn_backend.py:349,374` | 分别生成 raw conv 窗口和 recurrent 索引；`track_chunk_idx` shape 为 `[batch]` | 多点要同时扩展两套索引 |
| `srt/layers/attention/linear/kda_backend.py:831` | 在原位 conv 前，从 raw `mixed_qkv` 抽取该位置的 conv state | 不能复制末尾 conv 冒充中间位置 |
| 同文件 `:864–931` | 为内部点分配 FP32 `h_track_buf`；检查实际执行 backend 的能力；传给 extend 再复制至 track slot | 单点路径已有精度保护，不能绕过 |
| `srt/layers/attention/linear/kernels/kda_triton.py:219` | 将 `track_state` / `track_chunk_idx` 传给 `chunk_kda` | Triton 无需第二次 KDA forward 来取一个点 |
| `kernels/ops/attention/fla/chunk_delta_h.py:137,167,188` | 每序列读一个 chunk index；在该 chunk 开始前把 FP32 `b_h` 写入 track buffer | 位置是“已处理多少 token”，不是包含边界 token |
| 同文件 `:330` | 最后仍把完整 extend 的末尾状态写入运行槽 | 运行槽会继续被 decode 更新，不等于持久末尾 checkpoint |
| `srt/layers/attention/hybrid_linear_attn_backend.py:911` | FP32 snapshot 只向 pool dtype 转换一次，再落入指定槽 | 不应从低精度 `h` 恢复“FP32” |

固定源码入口：[KDA backend](https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/python/sglang/srt/layers/attention/linear/kda_backend.py#L795)、[Triton recurrence](https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/python/sglang/kernels/ops/attention/fla/chunk_delta_h.py#L137)。本段结论来自本地对应文件，而非从远端版本号猜测。

### 3.1 普通 `h` 与专门的 FP32 track buffer 不能混淆

**VERIFIED / SOURCE**：`chunk_delta_h.py:396` 的 `h = k.new_empty(...)` 继承 activation dtype；kernel 每个 chunk 都将累加器舍入后写入 `h`。它用于计算输出，`fla/kda.py:1164–1194` 的 `output_intermediate_states` 主要决定是否把它返回给调用方，并不是打开该开关后才开始计算全部中间状态。

专用 `track_state` 则要求 FP32（`:373–378`），直接保存尚未舍入到 activation dtype 的状态。用 `h.float()` 代替，会丢掉这个精度保证。

**VERIFIED / SOURCE，不是本次执行结果**：已有 [kernel 回归测试](../../src/sglang/test/registered/kernel/ops/attention/test_kda_track_state.py) 比较“100-token extend 中第 64 个 token 后快照”和“只算 64 token 的 final state”，并检查 BF16 rounding 确实有损。它注册的是 `4-gpu-b200` CI，不能作为 A100 已验证的证据。[dtype 单测](../../src/sglang/test/registered/unit/layers/attention/test_mamba_track_state_dtype.py) 还覆盖 FP32→FP16 的双重舍入问题。本轮两者均未运行。

### 3.2 对齐、相对坐标和 `+1` 陷阱

**VERIFIED / SOURCE**：`_force_track_h(b)`（`schedule_batch.py:2905`）返回 `b+1`，是现有索引逻辑区分“内部状态”和“最终状态”的编码。真实缓存深度仍是 `b`。若直接把整齐对齐的内部 `b` 交给 `_init_track_ssm_indices`，它会走 aligned→final-state 分支，把末尾状态错误标在内部位置。

例子（**INFERRED / 由源码推演，不是实验**）：prefix `p=1024`，extend 256 token，到 `L=1280`，期望 role checkpoint `b=1152`。

```text
真实缓存深度 b = 1152；extend 内 offset = b-p = 128
Triton chunk index = 128/64 = 2（开始 chunk 2 前，已经处理 128 token）
复用旧单点控制字段时 track_seqlen = 1153；真实 key / last_track_seqlen = 1152
conv 窗口取本请求 raw QKV 的 [offset-3, offset)，即 [125,128)
完整 forward 的运行 state 最终在 1280，不能以它替代 state@1152
```

`ForwardBatch.mamba_track_aligned_lens()`（`forward_batch_info.py:1070`）会 floor 掉这个 `+1`，使 conv 与 recurrent 对齐。**INFERRED**：新增多点结构应直接区分 `depth / relative_chunk / source_kind`，避免把 sentinel 技巧扩散到树 key。

**VERIFIED / SOURCE**：`runtime_context.py:1996–2009` 的真实 checkpoint grid 是 `lcm(mamba_cache_chunk_size(), tree_page)`；前者还受 model chunk / page 配置影响。DCP 可放大 tree page。**INFERRED**：`floor64(role)` 只在对应基线下成立；还必须检查 `b-p` 落在 kernel chunk 网格上、`p<b`、conv 窗口不跨入别的 packed sequence。现有 conv helper 的全局 clamp 不能用来“修复”一个原本就非法的多点窗口。

## 4. 最大工程面：快照何时成为可复用缓存

### 4.1 原生单点并非只在请求结束时入树

**VERIFIED / SOURCE**：普通生成 prefill 的 result processing 会在未 finished、非混入的 decode req 情况下调用 `maybe_cache_unfinished_req`（`srt/managers/scheduler_components/batch_result_processor.py:383`）；长 chunk stash 也有该路径。**因此不能声称“内部 role state 必须等 decode 完成才可能入树”。** 已有单点路径能够在 prefill 后登记。

`srt/mem_cache/unified_radix_cache.py:1098–1238` 依次准备 component 数据、计算 effective cache length、构造 key、插入、重新 match、更新 KV mapping/锁，最后 cleanup。`components/mamba.py:525–646` 读取单个 `mamba_last_track_seqlen`：

```text
预留/持有 track slot → 各层写 conv + recurrent → forward result 就绪
→ unfinished-cache prepare 分配替换槽并捐赠已写好的槽
→ 按真实深度插入 / 处理已存在节点 → cleanup 更新 ownership
```

普通 prefill result 路径会等待 `copy_done`（result processor `:266`）；具体多点跨 stream / overlap 的写入完成条件仍需实现时证明，不能只看 Python 调用次序。节点不得在相关的全部 34 层 KDA state、conv 和 MLA/DSA/indexer 前缀数据尚未有效时对其他请求可见。

### 4.2 为什么不能简单连续调用两次原生 cache_unfinished_req

**VERIFIED / SOURCE**：原 API 会捐赠一个 ping-pong slot、替换 request 的槽映射、截取有效 KV 长度、重新 match/加减锁，并在 cleanup 清掉 `mamba_last_track_seqlen`。重复插入还需释放未被采纳的 state slot。它不是无副作用的 `insert(depth, tensor)`。

**INFERRED / 设计要求**：Plan B 需要显式的快照描述与 ownership，例如 `(真实 depth, request-owned slot, source, ready)`；各点有自己的完整层状态空间。应设计一次性提交多个点或有明确不变量的专用逐点插入接口，证明已有节点合并、KV 共享、锁引用和失败回滚。不要通过临时改 request 字段再反复调用现有 API 来省掉设计。

`extra_buffer_lazy` 只延迟原生 ping-pong 分配，不会自动提供第二个快照。原生 prefill 的 replacement slot 分配会 eviction/retry，仍失败则 assert（`mamba.py:489`）；不能把它写成“分配不到自然跳过”。新增可选 role 快照应在 forward 前明确 reserve 成功，否则退回原策略并计数；不要写完一半后才发现没有完整 destination，也不要释放仍在 stream 使用的槽。取消、retraction、flush、重复 prefix、decode 覆写、slot 重用均需要独立清理证明。

**INFERRED**：保留 `--mamba-max-states-per-path` 的原配置做初始对照；F18 的 soft cap 不保护语义 role 节点。暂不合并 HiCache、int8、MTP、prefill graph 等兼容面，也不能宣称这些开关与 Plan B 正交。

## 5. 把 Plan B 拆成不同假设，避免收益偷换

| 研究变体（INFERRED） | 同一 extend 的缓存点 | 是否需扩展多点 kernel | 能否直接援引 F24 |
|---|---|---|---|
| B0：single-point role replacement | 用 role 替换唯一 stock track 点；final 运行状态仍存在 | 对本地 Triton，选点正确时可沿用单点 kernel | 不能：未保证持久 end / branch |
| B1：conservative dual-track | 无 branch 冲突时 role + 原生对齐 end-ish；有冲突退回 stock | 最坏两个内部 FP32 点；需多点元数据和缓存生命周期 | 只对应 F24 的意图，运行时覆盖率/驱逐仍未知 |
| B2：保留所有三种点 | branch + role + 对齐 end-ish，去重后最多 3 点 | 最坏三个内部点；工程/内存更大 | 对应 role_all 意图，非当前优先方案 |

**容易遗漏的边界（VERIFIED / SOURCE + INFERRED）**：prompt 长度未对齐时，`end-ish = floor(L/G)*G` 本身也是**内部**点；不能用 `state@L` 假装 `state@end-ish`。因此“一个内部 FP32 点 + 复制最终运行状态”只覆盖末尾恰好可入树的子集，未普遍实现 B1。例：p=1024、L=1300、role=1152，B1 要导出 1152 和 1280；运行 state@1300 两者都不能代替。

**VERIFIED / 现有记录**：Claude 新增 F24，`role_conservative` 的 fast_intra real-uncached p95 为 3332，stock 为 7238，2 次 branch conflict 被跳过；`role_all` 为 3326。已读 v2 脚本：conservative 无冲突时确实同时记录 role/end，有冲突只记 stock branch。它回答了之前 F17 的策略差异质疑，但仍是无驱逐、无排队/时延的 token 模型；本轮没有重跑。[F24](../../notes/findings.md)、[v2 脚本](../../scripts/sim_role_boundary.py)

**INFERRED / 若后续选择 B1**：

- 限定 KDA + Triton + 普通 extend；保持旧单点路径和默认关闭的 feature gate。先沿用 A 的候选/冲突/准入范围做等策略对照，随后才研究 B 可否在 active chunk 存在时扩大覆盖率。
- 把选中点表示为小上限 `P=2` 的 indices `[batch,P]`、FP32 scratch `[batch,P,H,V,K]`、独立 destination slots；无点用 sentinel，重合点去重。kernel 在现有 chunk 循环中只对命中点写累加器，不新增全长重算，也不把全部 `h` 强制改成 FP32。
- 所有 conv 窗口从该层原始 QKV 在原位处理前保存；recurrent 由 kernel 导出，然后一次转为目标 pool dtype。保留 envelope-strided pool 的 stride / int64 索引，不假定槽是紧密排列的。
- 扩展 backend 能力声明，区分“支持现有单点”与“支持多点”；不支持时在准入前退回 stock/已验证路径，不能默默留下未写的 buffer。
- 不新增人为 partial prefill，因而避开 A 的额外 chunk 指针冲突；但正常长请求仍要分块，D2 的单 active chunk 与预算规则依旧适用。“不切 forward”指**不额外为 checkpoint 切分**，不是禁用 chunked prefill。

以上是设计选项，不是已经实现的接口名或通过验证的补丁。

## 6. A100 与显存：哪些能静态确认，哪些不能

**VERIFIED / SOURCE**：本地 backend 路由如下；这些是能力/分支检查，不是本次运行结果。

| backend | 本地内部快照路径 | A100 判断 |
|---|---|---|
| Triton KDA | 原生单点 FP32 track | 值得作为 sm80 研究基线；仍需编译、数值和服务测试 |
| FlashKDA wrapper | `return_intermediate_states` 时退回 Triton；并未直连上面 vLLM fork 的 checkpoint 导出接口 | 外部快路径要求 SM90+；不能作为 sm80 快路径 |
| NVIDIA KDA | 要内部快照时 fallback；final-only 与内部点有不同分支 | vendor 快路径要求 capability major 10，非 A100 |
| FlashInfer KDA | 当前这里是 decode / target_verify，并非 KDA prefill export | SM100 路径，不能借 GDN 的支持情况推定 KDA |

依据：`linear/kernels/kda_triton.py:31,219`、`kda_flashkda.py:80–92,135`、`kda_nvidia.py:81–89,259–269`、`kda_flashinfer.py:1–3,88–99`。Triton KDA 的可行性也**不能解决** GLM-5.3 完整模型 sm80 DSA/indexer 前置条件；D6 仍须核对真实底包。

**VERIFIED / 尺寸算术**：沿用 [R5](R5_dp_memory_accounting.md) 的 TP8/DP1、34 KDA、FP32 recurrent + BF16 conv 假设：每个完整持久 checkpoint 每 rank 为 18,452,480 B = 17.598 MiB。每请求多一个点，22 个逻辑请求的这一项约 406 MB/rank；历史缓存、分配余量和锁定节点另计，N=22 不是历史 checkpoint 总数上限。

每层、每个导出点的 FP32 recurrent scratch 为 `8×128×128×4 = 0.5 MiB / request / rank`；假设实际 prefill batch 恰为 22，P=2 时为 22 MiB/层。**不能**把逻辑闭环 N 自动当作 prefill batch size，也不能直接把临时 workspace 乘 34 当峰值或宣称必然完全复用；生命周期、allocator/overlap 需确认。持久 slot 消耗与 scratch 峰值是两笔账。

**INFERRED / 性能预期边界**：A 的两段不会把全部 prompt 算两遍；B 希望减少的是额外 model 调用/调度、较小 GEMM/MoE 批次和 collectives 的固定成本。合并后的 prefill 更长也可能延后其他请求 decode。要同时看 TTFT 尾部、tpot、队列及 N@SLO，不能只测单请求 checkpoint copy 开销。

## 7. 验证清单与停止边界

### 已由静态调研回答

- 内部状态无需拆完整 forward 就能得到；本地已有 FP32 单点接口与回归测试源码。
- 一个被 track 的缓存点、完整 forward 的运行末态、可复用树节点是三个不同对象。
- 普通 prefill 后有 unfinished-cache 插入路径；多点仍需重做 bookkeeping，不是只有 kernel 参数。
- kernel 内坐标相对 extend，radix key 是真实绝对深度；对齐网格和 conv 状态必须一致。
- vLLM 合并实现、上游报告适用范围、FlashKDA 的 dtype / sm80 边界已经区分。

### 将来获准后再做，本文没有执行

1. **单算子**：单/双内部点与截断 reference 的 state、conv、最终输出对照；有初态、packed varlen、短尾、重复/无效点、非连续 pool stride、不同 pool dtype、sentinel rows。保留 FP32 累加快照的精度回归，不能只比 tensor dtype。
2. **缓存恢复**：从每个节点恢复后续算，对照冷算 logits/若干 greedy 输出；实际状态深度、key、`cached_tokens` 三者一致。涵盖非整齐 prompt 尾部，防止 final-state 错标。
3. **生命周期**：现有 prefix 命中、重复节点、slot 耗尽、lazy、overlap、abort/retraction、flush 后再请求；统计 reserved / donated / freed，排除引用已复用状态。
4. **等策略 A/B**：先固定相同 checkpoint 集合/准入范围，再单独比较扩大覆盖率；记录 attempted/exported/inserted/retained/hit，不把“写过 tensor”当“下轮一定命中”。
5. **真实八卡**：模型/底包/DSA 后端确定后，测全部硬门、能力抽检及内存峰值。两卡小 Kimi-Linear 能做功能层验证，不证明完整 GLM-5.3 的容量、DSA 正确性或 N=22 成绩。

交接建议：Claude 修订 D1 §4 时，引用“**已有单点 FP32 导出，多点缺的是完整接口与生命周期**”；B1 可采用与 F24 一致的保守冲突策略。不要再使用普通 BF16 `h` 当作无损 FP32 源；也不要把 FlashKDA 的 FP32 输出 dtype 当成其内部 FP32 累加保证。A 优先级不变。
