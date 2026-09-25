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

## 专家并行（`--ep-size 8 --moe-a2a-backend none`）

原先的 apply 没有把 EP 信息交给 Marlin。EP 下 `StandardDispatcher` 已把 topk_ids 映射成本卡编号，其他卡的专家为 −1（`token_dispatcher/standard.py` 的 `local_expert_mapping`）。Marlin 只有在 `is_ep` 时才跳过 −1 块；不传时会把 −1 当成专家，从权重和 scale 起点之前取数，结果是错数甚至非法访存。
- apply 传入 `expert_map=layer.dispatcher.local_expert_mapping` 与全局专家数，写法同 `mxfp4_marlin_moe.build_marlin_moe_quant_info`；id 不再映射第二次。EP 下若没有映射表（例如 top-k 输出不是标准格式），启动后第一次前向直接报错，不静默算错。
- `fused_marlin_moe` 的块大小按"每个专家的行数 = M×topk/全局专家数"选。原来除以本卡专家数，EP8 下高估 8 倍，小批量（decode）会选过大的块、白算填充行。TP 下全局数等于本卡数，行为不变。
- TP 路径不受影响：只有 `moe_ep_size > 1` 时分发器才生成映射表。
- 共享专家在 EP 下不再融合（`glm5_next.py`），作为 TP 切分的独立 MLP 与路由输出相加后做同一次 all-reduce。
- 验证：开发机单卡按 EP8 每卡形状（36 个完整专家、N=2048、真实 −1 id）对 fp32 参考比对数值，另设专家编号错位的负对照（`scripts/pod/verify/bench_moe_ep8.py`）。8 卡端到端、能力抽检与 MTP 接受率未验证。
