# 111 — sm80：块量化 FP8 MoE 专家改走 Marlin（fp8 e4m3 权重 + bf16 激活）

**问题**（任务 006，F58）：Fp8MoEMethod 在 A100 默认走 Triton `fused_moe_kernel`（fp8 w8a8），Triton 在 sm80 拒绝 `fp8e4nv` → CUDA graph 捕获前崩。
线性层在 sm80 已自动走 FP8 Marlin（`can_auto_enable_marlin_fp8`），但 MoE 没有对应分支。

**改动**（4 文件，锚点 `[ax] 111`）：
- `quantization/fp8.py`：`Fp8MoEMethod.ax_sm80_marlin`（sm80–88、块量化、非 fp4/mxfp8，环境变量 `SGLANG_AX_SM80_FP8_MOE_MARLIN=0` 可关）；
  开启时 `process_weights_after_loading` → `prepare_moe_fp8_layer_for_marlin(size_k_first=False)`；`create_moe_runner` → MARLIN；`apply` → `MarlinMoeQuantInfo(weight_bits=8, fp8_weights=True)`。
- `moe/moe_runner/marlin.py`：`MarlinMoeQuantInfo.fp8_weights` 透传。
- `moe/fused_moe_triton/fused_marlin_moe.py`：`fp8_weights: bool`（自定义算子 schema 需可注解类型）→ `scalar_types.float8_e4m3fn`。JIT 核 `moe_wna16_marlin.cuh` 原生支持 kFE4M3fn。
- `quantization/marlin_utils_fp8.py`：专家数取 `w13_weight.shape[0]`（本地专家，含融合的共享专家）。

**验证**：pod 单卡 A100，GLM TP8 形状（k=4096、n=256、block 128、top-8、33 专家），对 torch 反量化参考相对误差 ≈6e-3，CUDA graph 可捕获，M≤64 约 0.25 ms/层（`evidence/F58/`）。8 卡端到端 = pod 任务 008。
**性能备注**：权重仅量化，预填充大 M 走 bf16 tensor core；是否需要 A100 专用 MoE 调优留待基线数据。
