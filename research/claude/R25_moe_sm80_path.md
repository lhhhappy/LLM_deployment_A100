# R25 — A100（sm80）上 FP8 MoE 的执行路径、大块低效的原因与替代方案

2026-09-24，Claude。只读调研：没有在开发机或 pod 上运行任何东西。

## 0. 读法

**标签**
- **实测**：本仓库已有的测量，并注明出处。
- **按形状推算**：由张量形状和硬件规格算出，没有实测。
- **推断**：带假设的判断，需要第 5 节的探针确认或推翻。

**路径缩写**
- `B/` = `build/base_exact/sglang/`，只读底包。
- `U/` = `refs/sglang-fe236ea6c3/`，上游 2026-09-01 的提交 fe236ea6c3。本文用到的 `fp8.py` 与 `layers/moe/` 两处与底包逐字节相同（`diff` 无输出）。
- `H/` = PyPI `humming-kernels==0.1.12` 的源码包（https://files.pythonhosted.org/packages/93/b9/53797fff4b1b31b4567453a365646d724469b1bb0d76c07f8746617d0cb6/humming_kernels-0.1.12.tar.gz），即镜像里装的版本（`evidence/base_manifest/pip_freeze.txt:88`）。
- `V/` = vLLM main 分支，2026-09-24 从 raw.githubusercontent.com 抓取；行号以抓取时为准。

**术语**（只解释一次）
- **Marlin**：一类"权重只量化"的 GEMM kernel。权重以低精度存储，在 kernel 内部即时还原成 BF16 后用张量核计算；原本为小批量、受显存带宽限制的场景设计。
- **W8A16**：权重 8 位（这里是 FP8 e4m3）、激活 BF16。A100 没有 FP8 张量核，只能这样算。
- **TP8 每卡切片**：8 张卡按专家的中间维切分。每个专家的中间维 2048 在每卡只剩 256，即 N=256。
- **grouped GEMM**：把几百个专家各自的小矩阵乘放进一次 kernel 启动里完成。
- **moe_block_size / block_size_m**：把分到同一专家的 token 按多少行一组对齐。不足一组的部分用空行补齐，补的行叫**填充（padding）**。
- **EP8（专家并行）**：不切专家，每卡放 288/8=36 个完整专家。
- **a2a**：all-to-all 通信，把 token 发往专家所在的卡。

## 1. 结论摘要（与决策相关）

1. **今天的路径**：Fp8MoEMethod 经 111 的分支进入 `MoeRunnerBackend.MARLIN`，再进入 `fused_marlin_moe`，一次 MoE 依次调用 7 个 GPU 步骤：
   1. 对齐排序；
   2. 把一块中间缓存整块清零，16k token 时为 1.21 GB；
   3. Marlin GEMM1（gate_up）；
   4. clamp+SwiGLU 激活（torch 多个小 op，开 172 后合成 1 个）；
   5. Marlin GEMM2（down）；
   6. `moe_sum_reduce` 把 9 个专家的输出相加；
   7. 必要时 `.to(dtype)`。

   共享专家作为第 289 个专家并入同一次调用（topk=9）。Marlin 每组最多 64 行（`thread_m_blocks≤4`），大 M 固定用 64×256×64 的块，每个 SM 只驻留 1 个线程块（8 个 warp）。
2. **约 80 TFLOPS 不是 Marlin 独有的问题**。已有实测（`evidence/INT8/bench_moe_int8_devbox.log:34`）：同形状下，未调参的 Triton BF16 grouped MoE 在 M=16384 只有 82.0 TF，Marlin 84.5 TF，INT8 张量核 W8A8 也只有 85.8–91.4 TF。因此把 FP8 反量化成 BF16，本身不会自动更快。Triton 那条线用的是默认小块配置（64×64×32），调参后的上限尚未测。
3. **非 GEMM 的显存搬运占可观比例（按形状推算）**：16k token 时，"清零 1.21 GB + 写 GEMM2 输出 1.21 GB + sum-reduce 读 1.21 GB"加起来约 1.5–2.3 ms，占 11.4 ms 的 13–20%。这部分改起来不涉及任何 kernel 数学，是风险最低的一刀。
4. **A100 上这些形状的现实上限（推断）**：约 150 TF 等效，即 16k 时 MoE 约 6 ms。MoE 最多快 1.9 倍，现实目标 1.3–1.5 倍。MoE 约占大块 prefill kernel 时间的 38%（`research/codex/R23_moe_swiglu_fusion.md:7`，只有 7 个样本），所以 prefill GPU 时间最多减 10–18%（推断，按 Amdahl 定律）。172 的经验提醒：局部 MoE 快 4.7% 在整档回放里没有显出净收益（R23 §057）。
5. **最值得先测的替代方案是 Humming，而且已在底包里**。
   - 底包自带 `HummingMoEMethod` 与 MoE runner，并能把 block-FP8 检查点转换成 Humming 格式（`B/srt/layers/quantization/humming.py:310-400,987-1250`），镜像已装 humming-kernels 0.1.12。
   - Humming 在 sm80 上对 BF16 激活用 128×256×64 的块（`H/humming/tune/sm8x.py:24-34`），MoE 的行块可以到 96/128（`H/humming/tune/base.py:60-85`）。Marlin 最多 64 行。
   - 缺的只是：Fp8Config 对 FusedMoE 在 sm80 上选 Humming 的那一小段胶水。底包已有同样写法：fp4 专家在 `fp8.py:405-410` 就是这样选 Humming 的。
6. **size 分流有一个此前没写出的硬约束**。111 在加载时把专家权重重排成 Marlin 私有布局（`B/srt/layers/quantization/marlin_utils_fp8.py:232-245`），原始 `[E,N,K]` 布局不再保留。两份都留，每卡要多 38 GB（按形状推算），做不到。所以"小 M 用 Marlin、大 M 用别的"只有两种走法：
   - 写一个能读 Marlin 布局的反量化 kernel；
   - 整条路径（decode 也算）都换成一种布局，例如 Humming。
7. **EP8 实测更慢（2026-09-25，开发机单卡，[证据](../../evidence/ep8-devbox-20260925/)）**：8192 token 每层 TP8 5.53 ms、EP8 6.35 ms（最慢卡加独立共享专家；偏斜路由 7.12 ms），16384 为 10.9 对 12.9 ms。每卡 Marlin 在 EP8 形状（N=2048）只有约 71 TF，TP8 形状约 84 TF：变宽不提高 Marlin 效率，下表原因 B 不成立。数值：EP8 相对误差 5.4e-3，与 TP8 的 5.7e-3 同级。111 的 EP 修复（传 `expert_map`、按全局专家数选块）在 `claude/ep8` 的 3df1b172，EP 不作为吞吐方向。以下为原先的源码分析，仍然成立：EP8 在本底包不需要 DeepEP。单机 TP 注意力下每卡已有全部 token，`--ep-size 8 --moe-a2a-backend none` 与 TP8 使用同一次 all-reduce，按源码推断没有额外通信。代价与风险：
   - 共享专家不能再融合（`B/srt/models/glm5_next.py:1349-1353`）；
   - 负载不均要靠真实路由来量；
   - 111 没有传 EP 映射（`expert_map`），EP 下有越界风险，这一点 R9:63 已提出。
8. **补丁 111 与上游规范有 6 处偏离**：环境变量、分支位置、EP、类型参数、测试、数值说明（第 4 节）。其中 EP 缺口和环境变量两条也适用于 172。
9. **decode 侧（推断）**：M=33/256 时 Marlin MoE 的有效带宽约 1.19 TB/s，约为 A100 80GB 峰值 2.039 TB/s 的 58%（按形状推算）。decode 本来就受权重带宽限制，还有约 20–30% 的理论空间。这关系到 `tpot_mean`（排名第二键）。前提是图模式下的实测值低于 eager 值，而 eager 计时含 host 间隙（R23）。

## 2. 今天在 sm80 上的精确调用路径（问题 1）

### 2.1 从量化方法到 kernel

| 步 | 位置 | 做什么 |
|---|---|---|
| 选法 | `engine/docs/111-sm80-fp8-moe-marlin.md`（`fp8.py` 基线 1094 行后） | 同时满足以下条件时置 `ax_sm80_marlin=True`：`_is_cuda`；`can_auto_enable_marlin_fp8()`，即 `80≤sm<89`（`B/srt/layers/quantization/fp8_utils.py:2069-2075`）；块量化；非 fp4；非 mxfp8；环境变量 `SGLANG_AX_SM80_FP8_MOE_MARLIN` 未关 |
| 加载后 | 111，`fp8.py:2052` 处改名并前置分支 | `prepare_moe_fp8_layer_for_marlin(layer, size_k_first=False)`，然后 `torch.cuda.empty_cache()` |
| 权重重排 | `B/srt/layers/quantization/marlin_utils_fp8.py:219-245` | 逐专家把 fp8 打包成 int32，再用 `gptq_marlin_repack(num_bits=8)` 改成 Marlin 私有布局，原布局被替换 |
| scale 处理 | 同文件 `:251-307` | 128×128 块的 fp32 `scale_inv` **转成 BF16**（`:254`），沿 N 复制 128 份成"每列、每 128 个 K 一组"（`:293-295`），做 Marlin 排列，再乘 2^120（`:29-40,304`）。之后 kernel 内把 fp8 位直接搬进 bf16 的位，省掉逐元素的指数修正 |
| runner | 111，`fp8.py:2365` 前置分支 | `MoeRunner(MoeRunnerBackend.MARLIN)`。底包原本对 marlin 后端什么也不做：`fp8.py:2393-2396` 写着 `TODO(cwan): refactor other backends` |
| apply | 111 | 构造 `MarlinMoeQuantInfo(weight_bits=8, fp8_weights=True)`，**没有传 `expert_map` / `global_num_experts`** |
| runner 函数 | `B/srt/layers/moe/moe_runner/marlin.py:115-212` | 每次调用都新建 workspace（`:152-154`，注释说明是为了避免 graph 捕获时锁别名）。clamp 限值取 `swiglu_limit`=10（`:200-204`） |
| 包装 | `B/srt/layers/moe/fused_marlin_moe.py:134-428` | 下文 2.2–2.4 |
| JIT 入口 | `B/kernels/ops/moe/moe_wna16_marlin.py:18-33,118-130` | 按 dtype/EP/bias 特化编译。每次调用新分配 fp32 的 `c_tmp`（上限 sms×4×64×256×4B ≈ 28 MB，按形状推算） |
| CUDA | `B/kernels/jit/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh`、`marlin_template.h` | FP8 权重只在 `BIGGROUP_GET_IF(kFE4M3fn)` 里实例化，即 group=-1 或 128（`.cuh:360-378,465`） |

模型：`Glm5NextMoE = DeepseekV2MoE`（`B/srt/models/glm5_next.py:110`）。检查点是 `quant_method=fp8`、`weight_block_size=[128,128]`（`s1-dev/glm_tok/config.json:290` 起）。MoE 层数 = 45 − 3 个 dense 层 = **42**。

### 2.2 共享专家与 topk

- EP=1 且 sm≥80 时，共享专家作为额外一个专家融合进来（`glm5_next.py:1339-1368`；`B/srt/models/deepseek_v2.py` 在 594、610、612 行附近：专家数 288+1，top_k 8+1）。
- 每卡本地专家数 `num_local_experts = 288 + 1 = 289`（`B/srt/layers/moe/fused_moe_triton/layer.py:356-373`）。
- 所以每次 Marlin 调用是 E=289、topk=9，**共享专家那一"组"有 M 行**，其余每个路由专家平均 M×8/288 行。
- 111 把 `e = layer.num_experts` 改成 `w13_weight.shape[0]`。TP 下两者相等，都是 289；EP 下不等。

### 2.3 分块、配置与填充

- **block_size_m**（`fused_marlin_moe.py:238-242`）：在 [8,16,32,48,64] 里取第一个满足 `M×topk/E/bs < 0.9` 的值，都不满足就取 64。上游 vLLM 用同一启发式（`V/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py:342-347`）。**上限 64 写死在两处**：`.cuh:885-888` 只接受 8 或 16–64 之间的 16 的倍数；`thread_m_blocks = ceil(bs/16) ≤ 4`（`.cuh:619`）。
- **线程配置**（`.cuh:140-152,479-577`）：
  - `thread_m_blocks>1` 时优先 `{thread_k=64, thread_n=256, 256 线程}`。
  - 共享内存按 `.cuh:197-243` 计算：A 32 KB + B 64 KB（fp8，4 级流水，`marlin/marlin.cuh:17`）+ scale 2 KB + 元数据 0.25 KB ≈ **98.3 KB/块**。
  - A100 每块可用上限约 163 KB（`H/humming/tune/sm8x.py:7`），`allow_count = ⌊163K/(98.3K+1.5K)⌋ = 1`，大 M 时还被 `min(...,2)` 截断（`.cuh:558-568`）。
  - 结论（按源码推算）：**每 SM 1 个线程块、8 个 warp，grid=108 个常驻块**。
- **条带调度**（`marlin_template.h:395-420`）：`iters = ceil(k_tiles × n_tiles × parallel / 108)`，每个常驻块依次处理若干（专家块，N 列）的 K 条带。
  - GEMM1：N=512 → `n_tiles=2`；K=4096 → `k_tiles=64`。
  - **GEMM2**：N=4096 → `n_tiles=16`；**K=256 → `k_tiles=4`**。每 4 次 K 迭代就要写出一个 64×256 的 bf16 块。
- **反量化**（`marlin/dequant.h:323-338`，`marlin_template.h:1255-1316`）：每 2 个 fp8 用约 3 条位运算放进 bf16，乘上已含 2^120 的 bf16 scale，再做 `mma` 16×8×16。M 方向是最内层循环，同一个解码后的 B 片段只复用 `thread_m_blocks≤4` 次，即 64 行。
- **对齐**（`B/srt/layers/moe/moe_runner/triton_utils/moe_align_block_size.py:87-161`）：排序缓冲长度 = `topk_ids.numel() + (E+1)(bs−1)`；专家 id 为 −1 的组记作 −1。L054 实测：底包排序用 atomicAdd，同一输入两次运行的块归属会变，造成末位差异（`evidence/L054-official_a_moe_numtrace/tensor-audit.json`；R23）。

**填充浪费**：按形状推算，路由用均匀随机 top-8 模拟，每点 3 次，真实路由偏斜会改变这些数字。

| M（每卡 token） | block_size_m | 真实行数 M×9 | 补齐后行数 | 浪费 | 路由专家平均行数 |
|---:|---:|---:|---:|---:|---:|
| 256 | 16 | 2,304 | ≈4,853 | 52% | 7.1 |
| 1,024 | 48 | 9,216 | ≈14,880 | 38% | 28.4 |
| 4,096 | 64 | 36,864 | ≈42,282 | 13% | 113.8 |
| 8,192 | 64 | 73,728 | ≈82,261 | 10% | 227.6 |
| 16,384 | 64 | 147,456 | ≈156,885 | 6% | 455.1 |

补齐的行虽然不读数据（`cp_async4_pred` 按 `block_num_valid_tokens` 做了屏蔽），但仍参与张量核计算。M ≥ 4k 时填充只解释 6–13%，不是 80 TF 的主因（推断）。

### 2.4 GEMM 以外的 kernel 与显存搬运（M=16384，每卡）

| 步 | 源码 | 搬运量（按形状推算） | 时间 |
|---|---|---|---|
| `intermediate_cache13 = torch.zeros(M×9×max(512,4096))` | `fused_marlin_moe.py:289-298` | 写 1.21 GB | 约 0.6–0.8 ms（推断，按 1.5–2.0 TB/s） |
| GEMM1 输出 | `:305-332` | 写 0.15 GB | 含在 GEMM1 内 |
| clamp+SiLU+mul | `:343-348` → `swiglu_limit_func`（`:106-119`） | 读 0.15 GB，写 0.075 GB | 未融合 0.660 ms，172 融合后 0.135 ms（**实测**，engine/docs/172，只测激活） |
| GEMM2 输出（每个 token×专家一行，已乘 topk 权重，`mul_topk_weights=True`） | `:368-395` | 写 1.21 GB | 含在 GEMM2 内 |
| `moe_sum_reduce`（×routed_scaling 2.5） | `:419-427` | 读 1.21 GB，写 0.13 GB | 约 0.85 ms（按形状推算：R10 在 c=1024 实测 2.4 ms/45 次 = 53 µs，`research/claude/R10_prefill_fixed_overhead.md:104`，按字节线性放大 16 倍） |
| 对齐排序、workspace/c_tmp 分配 | `:262-275`，`marlin.py:154` | 小 | 每次调用约 2–3 次 launch |

- sm80 上 bf16 不走原子累加：`use_atomic_add` 只对 fp16 或 sm≥90 开启（`:300-303`）。跨块的 K 切分因此走 fp32 `c_tmp` 的全局归约。
- 清零之所以必要，是因为"Marlin 跳过被屏蔽的行"（注释 `:289`）。TP 下只有 topk id 为 −1 的行不会被写，例如被 `num_token_non_padded` 屏蔽的补齐 token（`B/srt/layers/moe/topk.py` 的 `num_token_non_padded` 路径）。纯 prefill 大块里没有这样的行（推断），所以整块清零基本是冗余的。

## 3. 为什么大 M 只有约 80 TFLOPS（问题 2）

### 3.1 实测曲线

来源：`evidence/moe172/summary.json`，单卡 A100、随机权重、TP8 每卡形状、E=289、topk 9、eager 计时，可能含 host 间隙。FLOPs = tokens×9×3×4096×256×2。

| M | 0 号基线 gpu_ms（实测） | 等效 TFLOPS | 解读 |
|---:|---:|---:|---:|
| 33 | 0.463 | 4.0 | 受权重带宽限制。约 176/289 个专家被触及，读 0.55 GB，折合约 1.19 TB/s，是 HBM 峰值的 58%（按形状推算） |
| 256 | 0.766 | 18.9 | 全部专家被触及，读 0.909 GB，约 1.19 TB/s（58%，按形状推算） |
| 1,024 | 1.069 | 54.2 | 过渡区，填充 38% |
| 4,096 | 2.900 | 80.0 | |
| 8,192 | 5.805 | 79.9 | |
| 16,384 | 11.396 | 81.4 | 26% of 312 TF |

另一组交叉验证（`evidence/INT8/bench_moe_int8_devbox.log:30-34`，E=288、没有共享专家、topk 8、Triton 路径**均为默认配置**，日志第 3–12 行有 "Using default MoE kernel config"）：

| M=16384 | Marlin FP8 W8A16 | Triton INT8 块量化 W8A8 | Triton INT8 逐通道 | Triton BF16 |
|---|---|---|---|---|
| TF（实测） | 84.5 | 85.8 | 91.4 | 82.0 |

### 3.2 每卡 GEMM 形状

- **GEMM1**（gate_up）：每个路由专家 `[≈455 × 4096] × [4096 × 512]`；共享专家 `[16384 × 4096] × [4096 × 512]`。
- **GEMM2**（down）：`[≈455 × 256] × [256 × 4096]`，**K=256 很短**。一个 128×128 输出块只做 256 深的累加：约 8.4 MFLOP，却要写出 32 KB 结果。GEMM2 在 16k 时总共写 1.21 GB（每 token×专家一行 4096 维，不是先汇总到 token）：按 2.0 TB/s 约 0.6 ms；张量核峰值下计算约 1.0 ms。两者同量级，所以**GEMM2 天然难以跑满**（按形状推算）。
- 权重：每卡每层 fp8 共 0.909 GB，其中 w13 0.606、w2 0.303；42 层合计 38.2 GB（按形状推算）。R18 测得加载阶段增量 39.15 GiB/rank（`research/codex/R18_cache_loss_and_capacity.md:223`），同一量级。

### 3.3 A100 对这些形状的可达上限（推断）

- 规格：BF16 张量核 312 TF，HBM2e 2.039 TB/s，L2 40 MB，108 SM（NVIDIA A100 数据手册 https://www.nvidia.com/en-us/data-center/a100/ ，`evidence/cost-audit-20260924/cost-estimates.json` 亦引用）。大尺寸 cuBLAS BF16 GEMM 在 A100 上一般可到峰值的 70–85%。这是常识性推断，本仓库没有测过，见第 5 节 P2。
- **GEMM1** 的 K 长、每专家行数约 455，128×128 块就有约 5k 个块，足够填满 108 SM。可达约 200–230 TF，即 6.18e11 FLOP 约 2.7–3.1 ms。
- **GEMM2** 受 K=256 和 1.21 GB 输出限制，约 120–160 TF，即 3.09e11 FLOP 约 1.9–2.6 ms。
- 不可省的非 GEMM 部分：激活 0.135 ms（172 实测）、sum-reduce 约 0.85 ms、对齐约 0.1 ms。
- **合计约 5.7–6.8 ms，对应现在的 11.4 ms，理论提速 1.7–2.0 倍**，等效约 140–160 TF。考虑到还要做反量化，现实目标取 1.3–1.5 倍。

### 3.4 候选原因

| 解释 | 支持 | 反对 / 未证明 |
|---|---|---|
| A. Marlin 为小 M 设计：每组 ≤64 行，每 SM 1 块 8 warp；B 片段每次解码只复用 64 行 | 源码 2.3；Humming 在同一硬件上对 BF16 激活选 128×256（`H/humming/tune/sm8x.py:24-34`），说明作者认为更大的 M 块更好 | Triton BF16（64×64×32，没有反量化开销）同样只有 82 TF，所以"反量化本身"不是主因。块大小影响的是数据复用与流水线，还需要 ncu 看张量管线利用率 |
| B. TP8 切出的窄形状：GEMM1 的 N=512，**GEMM2 的 K=256** | 3.2 的按形状推算；GEMM2 每 4 个 K 迭代就做一次尾处理 | **已被实测推翻**：EP8 形状（N=2048、K=4096）下每卡 Marlin 更慢（71 对 84 TF，见第 1 节第 7 条） |
| C. 非 GEMM 搬运：清零、每个 top-k 各写一行、再单独求和 | 2.4 表：约 1.5–2.3 ms，占 13–20% | 有 R10 在 c=1024 的实测校准点；16k 的数值是线性外推 |
| D. 填充 | 16k 时 6%，4k 时 13% | 大 M 时不是主因 |
| E. 每次调用分配 workspace / c_tmp、启动约 7 次 kernel | 172 已证明小 op 合并有 3–5% | 大 M 时占比小；小 M 和 decode 才显著 |

**日志证明不了什么**：现有测量都是整次 MoE 的总时间，没有逐 kernel 拆分，也没有 ncu 计数器。上面 A 与 B 的比重无法区分，第 5 节 P1 专门补这块。随机权重和均匀路由也不能代表真实负载的专家偏斜。

## 4. 替代方案（问题 3）

表中收益一律是**推断**，以 16k token 单层 MoE 为准，基线 11.4 ms（实测）。显存按 R18 的每卡 12,716 B/token KV（`R18:217`）换算，缓存池中 KV 约占 52.6%（r=0.9，`R18:211`）；1 GiB 常驻显存约等于 4.4 万 KV token。

### (a) 按大小分流：小 M 用 Marlin，大 M 反量化成 BF16 临时副本再做 BF16 grouped GEMM

- **本底包已有**：
  - Triton BF16 fused_moe（未量化路径，`fused_experts_impl`），支持 `swiglu_limit` 分支（`B/srt/layers/moe/moe_runner/triton_utils/fused_moe.py:695-737`）；
  - A100 调参脚本在 `U/benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py`；
  - 配置文件按 E/N/设备查找（`B/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_config.py:36-53`）。镜像里**没有** E=289、N=256 的 A100 配置，会退回默认 64×64×32（`:246-261`）。
- **必须新写**：
  1. 反量化 kernel。难点在权重布局（结论 6）：111 加载后只剩 Marlin 布局。
     - 路线一：写一个"读 Marlin 布局 → 写 `[E,N,K]` BF16"的 kernel，需要 `gptq_marlin_repack` 8 位排列的逆，可用往返测试验证；
     - 路线二：放弃 Marlin 布局、保留原始 fp8，但 decode 就得换一个 kernel。
  2. 分流阈值与调用处（一个新的 runner 分支）。
- **显存**：
  - w13 与 w2 共用一块 BF16 缓冲，峰值 1.21 GB（1.13 GiB）/卡（按形状推算）；
  - 若各用一块，则为 1.82 GB（R9:65 给的是 1.69 GiB/层，双缓冲 3.39 GiB）；
  - 若只能从 KV 池里挤出来：1.13 GiB 约等于 5.0 万 KV token（约 5.3%，按形状推算）；
  - 若能放进 prefill 激活预留（R18:225 所说约 17.25 GiB 静态余量）则不损失 KV。现在 16k 时的 `cache13` 本身就是 1.21 GB 临时张量，是否还有余量要测峰值。
- **时间**：
  - 反量化每层读 0.909 GB、写 1.82 GB，约 1.4–1.8 ms（按形状推算）；
  - GEMM 部分取决于调参后的 Triton BF16 能达到多少，**目前唯一的实测是未调参的 82 TF**，这时 a 必然更慢；
  - 若调参后达到 150–200 TF，16k 约 4.6–6.2（GEMM）+1.6（反量化）+1.1（非 GEMM）≈ 7.3–8.9 ms，快 1.3–1.55 倍；8k 约 1.1–1.2 倍；**4k 左右打平或更慢**。
- **数值风险**：
  - 反量化时若 scale 保留 fp32、只在最后舍入一次 bf16，比 Marlin（scale 先舍入成 bf16，权重×scale 再舍入成 bf16）更接近真实权重，但会与现网输出逐位不同；
  - Triton 路径的 clamp/SiLU 舍入边界要逐位对齐 172 的契约：SiLU 结果先舍入到 bf16，再与已 clamp 的 up 相乘；
  - fp8 的 NaN 编码 0x7F：Marlin 的位运算会把它变成有限值 480（按 `dequant.h:323-338` 推算），110（`engine 110:` 提交）的 `_e4m3_to_bf16` 则输出 NaN。检查点里不应出现这个编码，但测试要固定语义。

### (b) Triton fused_moe 在 kernel 内做软件 FP8→BF16（与 110 在 indexer 上的做法相同）

- **本底包已有**：
  - `fused_moe_kernel_gptq_awq` 已支持 `use_int8_w8a16` 加 K 方向分组 scale（`B/kernels/ops/moe/fused_moe_triton_kernels.py:93-316`，派发在 `:924-935`，要求 `block_shape[1]>0`、scale 为 3 维）；
  - 主 kernel 有块量化 w8a8 分支，形式是 `acc += dot(a,b) * a_scale * b_scale`（`:574-590`）；
  - 110 的 `_e4m3_to_bf16`（`engine/docs/110-sm80-dsa-indexer.md`，`sm80_indexer_kernels.py` 开头）可以直接复用。
- **没有现成配置可开**：Triton 在 sm80 拒绝 `fp8e4nv`（F58；R17:36）。`use_int8_w8a16` 的解码是减 128 的整数零点（`:299-301`），不是 fp8。
- **必须新写**：一个 `use_fp8_w8a16` 常量分支，按 uint8 读取，做软件解码，块 scale 在每个 128-K 块上乘到 fp32 累加器；再加上派发、量化信息和调参配置，约 60–120 行（推断）。
- **布局约束同 (a)**：只能用原始布局，所以 decode 也得走 Triton。已有实测：M=64 时 Triton 各路径 0.75–0.94 ms，Marlin 0.60 ms（`evidence/INT8/...log:26`，均为未调参）。**decode 可能变慢**，而 decode 关系 TPOT。
- **显存**：不增加。
- **收益（推断，不确定性大）**：软件解码每元素约 4–7 条整数指令，只摊在 BLOCK_M（64–128）行上。按 A100 每 SM 每周期 64 条 INT32 对 1024 次 BF16 FMA 估算，解码可能与张量核时间同量级。大 M 预期 100–150 TF，未必胜过 (a) 或 (e)。**排在 (e) 和 Humming 之后**。

### (c) EP8（每卡 36 个完整专家）代替 TP8

- **本底包已有**：
  - `--ep-size 8`，a2a 可以是 `none`：要求 `ep×moe_dp == tp`（`B/srt/arg_groups/parallel_hook.py:101-108`）；
  - StandardDispatcher 把全局专家 id 映射为本地 id，非本地记作 −1（`B/srt/layers/moe/token_dispatcher/standard.py:186-228`）；
  - Marlin 包装支持 `expert_map` 与 EP 特化（`fused_marlin_moe.py:365-366`，`marlin_template.h:385-392`）。
  - 单机、TP 注意力时每卡已有全部 token，MoE 结果与 TP 一样用一次 all-reduce 合并（按源码推断），**通信量不变**。
- **不需要 DeepEP**。本底包中 DeepEP low-latency 带量化时要求 DeepGEMM（`B/srt/layers/moe/ep_moe/layer.py:170-180`，sm90）。DeepEP V1 旧版文档写着"A100 support (intranode only)"，需以 `DISABLE_SM90_FEATURES` 编译（https://raw.githubusercontent.com/deepseek-ai/DeepEP/main/docs/legacy.md 第 45–48、83、296 行）；V2 要求 SM90（README 第 65 行）。镜像里 `sgl-deep-ep==0.1.2`（`pip_freeze.txt:255`）是否带 sm80 编译未知，本方案用不到。
- **必须改**：
  1. 111 补上 EP 映射，照抄 `B/srt/layers/quantization/mxfp4_marlin_moe.py:20-43` 的 `build_marlin_moe_quant_info`。先核实 −1 的双重映射契约（R9:63）。
  2. 共享专家不能融合（`glm5_next.py:1349-1353`），改走独立的 TP 稠密 MLP：每卡中间维 256，用 dense FP8 Marlin，多几次 launch。
- **形状（按形状推算）**：
  - GEMM1 `[≈455×4096]×[4096×4096]`；
  - GEMM2 `[≈455×2048]×[2048×4096]`，**K 从 256 变成 2048**，正好针对 3.4 的 B；
  - M 块上限仍是 64，A 没有改变。
- **显存**：每卡 36×3×4096×2048 fp8 = 0.906 GB/层，与 TP8 相同。
- **负载不均（推断）**：
  - 均匀路由下每卡 token 数的标准差约为 √(0.875M)，16k 时约 1%；
  - 真实路由若有热门专家，最慢的卡决定整层时间，底包有 EPLB 可做冗余专家；
  - decode 小 batch 时，各卡被触及的专家数方差大，例如 40 token 时每卡约 24±3.5 个。**可能拖累 TPOT**。
- **收益**：未知，取决于 GEMM2 在 K=2048 时能快多少、真实偏斜有多大，是一个"一改就是结构变化"的候选（R9:61）。

### (d) 上游已经提供的 Ampere FP8 MoE 实现

- **vLLM**：FP8 MoE 后端按优先级排列（`V/.../fused_moe/oracle/fp8.py:75-133`）。Triton 块量化 FP8 只在 `supports_fp8()` 时可用（`V/.../experts/triton_moe.py:157-175`），所以 A100 自动落到 **MARLIN**（W8A16），**HUMMING** 可显式选择（`oracle/fp8.py:52-53,175-193,272-273,605-632`）。vLLM 的 Marlin MoE 与我们同源，block_size_m 也 ≤64（`V/.../experts/marlin_moe.py:342-347`），函数显式接收 `quant_type_id`（`:224-304`）。**vLLM 没有更好的大 M Marlin 块**。
- **上游 SGLang（U）**：Fp8MoEMethod 在 marlin 后端仍是 TODO（`fp8.py:2393-2396`），即上游本身不能在 A100 上跑 block-FP8 MoE，111 解决的就是这个缺口。它有 humming：
  - `HummingConfig.get_min_capability()=75`（`B/srt/layers/quantization/humming.py:488-489`）；
  - `_StackedBlockFp8CheckpointWeightSchema` 能转换 block FP8（`:310-400`）；
  - `HummingMoEMethod`（`:987-1250`）只接受 runner 后端 auto 或 humming（`:1230-1242`）；
  - runner 支持 `swiglu_limit`（`B/srt/layers/moe/moe_runner/humming.py:381-406`）；
  - GEMM 类型 indexed 或 grouped，由 `SGLANG_HUMMING_MOE_GEMM_TYPE` 选（`:58-69`）。
- **Humming 本身**（`H/README.md:27`）：BF16 激活、SM80+，支持"任意 ≤8 位的有符号浮点"权重。
  - sm80 对 16 位激活的基础块为 128×256×64（`H/humming/tune/sm8x.py:24-34`）；
  - MoE 按每专家行数把 M 块选为 48/64/96/128（`H/humming/tune/base.py:60-85`）；
  - scale 同样转成 param_dtype，即 bf16（`H/humming/schema/fp8.py:98-102`），与 Marlin 的 scale 舍入一致。
- **本底包里怎么接（必须新写，推断约 50–100 行）**：
  - 方式一：`--quantization humming`。会让所有 FP8 线性层也改走 `HummingLinearMethod`（`humming.py:656-688`），波及面大，可能与 110、171 等补丁冲突，不建议作为第一步。
  - 方式二（推荐）：仿 `fp8.py:405-410` 对 fp4 专家的写法，在 `Fp8Config.get_quant_method` 里，当 FusedMoE、块量化、sm80、`--moe-runner-backend humming` 同时成立时，构造 `HummingLayerQuantizationConfig`（fp8 块 schema 加 bf16 输入 schema）并返回 `HummingMoEMethod`。
  - 注意 `prepare_layer_meta(num_experts=layer.num_experts)`（`humming.py:1213-1225`）在 TP 下是 289；它会把 N pad 到 256 的倍数、K pad 到 128 的倍数，我们的 512/4096/256 都整除。
- **风险**：
  - JIT 首次编译耗时和 CUDA graph 可捕获性未验证；
  - 同一种 Humming 布局要同时服务 decode，所以 decode 必须一起测；
  - README 的"SOTA"只是作者自述，没有 A100 上我们形状的数据。

### (e) 不碰 GEMM 的清理（新增候选，风险最低）

1. **只在可能出现 −1 id 时才清零**，例如 decode 的 graph 补齐；或只清补齐 token 的行。16k 时省约 0.6–0.8 ms（5–7%，按形状推算）。风险在契约：任何把 id 置为 −1 的来源都必须被覆盖（topk 屏蔽、EP 映射），测试要构造这些情况。
2. **把 top-k 求和并进 GEMM2 的尾处理**（按 token 原子累加到 fp32 缓冲，再转 bf16），省去写 1.21 GB 与读 1.21 GB，约 0.8–1.2 ms（推断）。
   - 求和顺序会变且不确定，**数值与现网不逐位一致**；
   - sm80 上 bf16 原子加被刻意关掉（`:300-303`），若用 fp32 缓冲，16k 时需 M×4096×4 B = 268 MB 临时显存；
   - 需要改 Marlin 模板的尾处理，属于执行层 Codex 的范围。
3. 每次调用的 workspace 与 `c_tmp` 改为按最大尺寸预分配（`marlin.py:152-154` 的注释说明是为了 graph 安全，要保留每次调用独立）。收益主要在 decode 与小 M。

(e1)+(e2) 合计，16k 时约 1.4–2.0 ms，即快 12–18%（推断）。与 (a)/(c)/Humming 可以叠加：它们也都有"每个 top-k 各写一行，再单独求和"的结构，除非换成把求和并进尾处理的实现。

## 5. 对补丁 111 的规范审查（问题 4）

依据：
- `U/docs/docs/developer_guide/contribution_guide.mdx:161-182`（代码风格、新硬件）；
- `U/docs/docs/developer_guide/quantization_contribution_guide.mdx:9-54,61-94`；
- `U/.claude/skills/env-var-conventions/SKILL.md`（Rule 1/4/6）；
- `U/.claude/skills/add-jit-kernel/SKILL.md:487-578`（测试与基准）。

| # | 偏离 | 证据 | 上游质量的写法 |
|---|---|---|---|
| 1 | **环境变量**：新增了 `get_bool_env_var("SGLANG_AX_SM80_FP8_MOE_MARLIN","true")` 调用点 | SKILL Rule 1（:12，禁止新增此类调用，必须在 `environ.py` 定义 `EnvField`）；Rule 4（:132-156，第二个词必须是 ENABLE/DISABLE/USE/FORCE 之类的动词；`AX` 是私有标记）；Rule 6（:188-197，已有 CLI 时不要再造环境变量） | 不用环境变量。沿用已有的 `--moe-runner-backend marlin`（`B/srt/layers/moe/utils.py:184`），并让 `auto` 在 `can_auto_enable_marlin_fp8()` 时解析为 marlin，与稠密层的自动 Marlin（`fp8.py:465-471`）对称。确需强制开关时复用已登记的 `SGLANG_FORCE_FP8_MARLIN`（`B/srt/environ.py:922`）。**172 的 `SGLANG_AX_MOE_FUSE_SWIGLU` 同样违反 Rule 1/4** |
| 2 | **分支位置**：在 Fp8MoEMethod 的 `__init__`、`process_weights_after_loading`、`create_moe_runner`、`apply` 四处插入 `if self.ax_sm80_marlin` 提前返回，还把原方法改名为 `_ax_process_weights_after_loading_base` | 量化指南 :15-19,48-53（配置选方案，方案管权重，后端封装 kernel，不要写成一个大文件）；贡献指南 :179-182（新硬件放新文件，不大改现有代码，通用路径放第一个分支） | 新文件 `quantization/fp8_marlin_moe.py`，其中 `Fp8MarlinMoEMethod` 包住 fp8_method，写法照 `mxfp4_marlin_moe.py:45` 起的 `Mxfp4MarlinMoEMethod`；由 `Fp8Config.get_quant_method` 选择，位置在 `fp8.py:398-410` 同段；Fp8MoEMethod 本身不改 |
| 3 | **EP 缺口**：QuantInfo 没有 `expert_map` / `global_num_experts` | 对照 `mxfp4_marlin_moe.py:20-43`；R9:63 | 抽一个 `build_marlin_moe_quant_info`，fp8 与 mxfp4 共用；补 EP 测试（上游 `test_marlin_moe.py:328` 有 EP 用例的模板） |
| 4 | **类型参数**：在自定义算子上新增 `fp8_weights: bool` 并在内部改 scalar_type | vLLM 显式传 `quant_type_id`（`V/.../marlin_moe.py:224-304`）；布尔特判不能推广到 e5m2 等格式 | 传 `ScalarType` id；或在 QuantInfo 里带 `b_q_type`，由 runner 转交 |
| 5 | **测试**：没有登记到 CI 的测试。F58 的 `evidence/F58/test_fp8_moe_marlin_sm80.py` 在仓库外，而且用 33 个专家的小配置 | 贡献指南 :40-73；上游 `U/test/registered/quant/test_marlin_moe.py` 只覆盖整数类型（uint4/uint8），没有 fp8；指南还要求 accuracy 与带预热的基准 | 在 `test_marlin_moe.py` 增加 fp8 块 scale 用例：参考实现用现成的 `marlin_quant_fp8_torch`（`marlin_utils_fp8.py:344-374`），覆盖 E 含融合共享专家、EP、M 覆盖所有 block_size_m 档位、graph 捕获，用 `register_cuda_ci` 登记；PR 附 GSM8K 等 accuracy 与 warmup 基准 |
| 6 | **数值说明缺失**：scale 从 fp32 转成 bf16（`marlin_utils_fp8.py:254`），权重×scale 舍入到 bf16，fp8 的 NaN 编码被映射为有限值 | 量化指南 :72-77,91-92（PR 需说明数值变化） | PR 中写明这些与 FP8 硬件路径的差异，并给出相对误差（F58 实测约 6e-3，INT8 基准实测 5.9–6.1e-3，其中 BF16 权重的 Triton 为 4.7e-3） |
| 7 | 小项 | 每层 `torch.cuda.empty_cache()` 会拖慢加载；`e = w13_weight.shape[0]` 是一个独立的上游 bug 修复 | empty_cache 去掉或只在最后做一次；bug 修复单独提 PR 并附测试 |

## 6. 开发机测量计划（问题 5）

以下**只是计划，本文没有运行**。开发机：ssh 别名 GPU，只在 `/sjtu/linhang/arena/` 下工作，单卡 `CUDA_VISIBLE_DEVICES=0`。源码 = exact 底包 + 111 + 172，与 R23 的 `runs/moe172` 同一套构建方式；每次运行都打印源码和补丁的 SHA。结论只是临时的，需要 8 卡复核（规则见 AGENTS.md 与 R23）。

### P0 公共设置

- **形状**：E=289（288 路由 + 共享，id=288 的专家对每个 token 都选中）；hidden 4096；每卡 N=256；fp8 e4m3 权重加 128×128 fp32 `scale_inv`，由已知的 bf16 权重量化得到，保证真实权重已知。
- **M** = 33、256、1024、4096、8192、16384。前两个代表 decode：10–64 请求，开 MTP 验证时 ×4。
- **路由**：
  1. 均匀随机 top-8 加共享；
  2. **真实路由**：请主 Codex 排一个短的 pod 作业，复用 L054 的 numtrace 钩子，对 3、20、44 层在真实 8k/16k prefill 块和 decode 批次上保存 `topk_ids`；拿到之前，第二套路由只作偏斜敏感性分析，不用于决策。
- **计时**：CUDA event 与 CUDA graph replay 两种都记。预热 5 次，5 组×5 次，两种顺序互换，照 R23 的做法。报告中位数和分布，原始样本存 jsonl。

### P1 基线拆分（先做，约 30 分钟）

- 对 Marlin 基线逐段计时：对齐、清零、GEMM1、激活（172 开/关）、GEMM2、sum-reduce。
- 再在 M=16384 用 `ncu` 看 GEMM1/GEMM2 的这些计数器：
  - `sm__pipe_tensor_op_hmma_cycles_active`、`sm__throughput`；
  - `dram__bytes`、`lts__t_sector_hit_rate`；
  - 实际占用率、寄存器数、共享内存。
- **判定**：GEMM2 占比远高于它 1/3 的 FLOPs 份额 → 支持原因 B，优先 EP8 或 GEMM2 专项；GEMM1 张量管线利用率低 → 支持原因 A，优先 Humming；非 GEMM 超过 15% → 先做 (e)。

### P2 天花板

- 相同总 FLOPs 的稠密 BF16 cuBLAS：`[147456×4096]×[4096×512]` 和 `[147456×256]×[256×4096]`。
- 等大小 bmm：`[289,512,4096]` 各 ≈510 行。
- 调参后的 Triton BF16 fused_moe：`tuning_fused_moe_triton.py`，E=289、N=256，只调 M=4k/8k/16k。
- 得到"本卡本形状"的实际上限，替换 3.3 的推断值。

### P3 候选

每个都与 Marlin 同输入、同路由对比：

| 候选 | 做法 | 进入条件 |
|---|---|---|
| C1 Marlin（111+172） | 基线 | — |
| C2 Humming | 优先用底包的 `HummingMoEMethod`，喂 block-FP8 张量，免写胶水；indexed 与 grouped 两种都测；记 JIT 编译时间与 graph 可捕获性 | 直接测 |
| C3 反量化 + 调参 Triton BF16 | 先用 torch 原型测反量化成本，再测 Triton 反量化 kernel；总时间 = 反量化 + GEMM + 非 GEMM；记峰值显存 | P2 调参后的 Triton BF16 明显快于 Marlin 的 GEMM 部分 |
| C4 Triton 软 FP8 | 需要新写 kernel | 只在 C3 的 GEMM 部分明显胜出、但反量化成本吃掉收益时才写 |
| C5 EP8 形状 | 36 个专家、N=2048，Marlin 带 `expert_map`（即 111 修复版）与 Humming 两种；按路由算每"卡"负载，取 8 份中最慢的一份作为本层时间；加上独立共享专家 MLP 的时间 | 直接测 |
| C6 (e) 清理 | 跳过清零；融合求和 | 直接测，写成对 `fused_marlin_moe` 的单点修改 |

**decode 必测**：M=33/256 在 graph 下的时间。任何候选只要 decode 变慢，就必须报告，因为它关系 `tpot_mean`。

### P4 数值验收

必须能抓住已知错误的变体。

- **参考**：对原始 fp8 权重乘 fp32 块 scale 得到 fp32 真实权重，用 torch fp32 做完整 MoE，包括 clamp=10、SiLU、按 topk 权重乘 `routed_scaling_factor`。
- **固定路由**：缓存排序结果或固定 `topk_ids`，排除 L054 与 R23 已知的排序原子序波动。
- **指标**：rel_l2、max_abs、逐 token 余弦最小值、NaN/Inf 掩码。
- **判定办法**，不另造阈值（经验规则 5）：
  - 候选对参考的误差，与 Marlin 对同一参考的误差处于同一量级；已有实测参考值约 6e-3；
  - 同时报告 Marlin 自身重复运行的差异作为噪声；
  - **负对照必须明显落在外面**，否则测试无效。负对照包括：
    1. 块 scale 的 n/k 索引对调（真实可能犯的布局错误）；
    2. 专家 id 错位 1，把共享专家槽用成 287；
    3. 在构造 |gate|、|up|>10 的输入上去掉 clamp；
    4. GEMM2 漏乘 topk 权重。
- **覆盖**：激活含离群值；某专家 0 个 token；M 不是块的整数倍；graph 捕获后换输入重放；EP 下出现 −1 的行（C5/C6 必测）。

### P5 8 卡确认（由主 Codex 排队）

1. **TP8 真实权重的逐阶段对照**：复用 L054 的 numtrace，真实 token，候选与基线逐层比较。
2. **启动显存**：KV token 数、状态槽数与基线一致，对比 R18 字段；有差异须按 12,716 B/token 折算。
3. **原 harness 完整开发集回放**：单变量，以 047/057 为对照，按 `scripts/score_formal.py` 加 tpot_p95 门判分。只有差异超过同配置重跑噪声才算结果。

## 7. 仍未知 / 可能推翻本文的证据

- P1 若显示 GEMM2 其实不慢，EP8 的理由就弱了；若清零和求和实测远小于 2.4 节的推算，(e) 的收益就要下调。
- Humming 在我们形状上若不比 Marlin 快，结论 5 作废。它的 JIT 或 graph 若有问题，接入成本会上升。
- 真实路由偏斜会同时改变填充比例、EP 负载不均和 decode 时被触及的专家数，本文全部用均匀路由推算。
- moe172 的 eager 计时含 host 间隙；decode 在 graph 下的真实带宽利用率可能高于 58%，那么结论 9 的空间会变小。
