"""[ax] 117: block-FP8 MoE experts on sm80-88 through the Humming MoE runner.

These GPUs have no FP8 tensor cores, so the experts run weight-only: FP8 e4m3 weights are
decoded to BF16 inside the GEMM and multiplied with BF16 activations. 111 does this with the
Marlin MoE kernel (at most 64 rows per expert block); Humming's indexed MoE GEMM uses up to
128-row blocks and is faster for prefill-sized batches at the same accuracy.

Selected by Fp8Config.get_quant_method when SGLANG_AX_SM80_FP8_MOE_HUMMING=1 (see
Fp8MoEMethod.ax_sm80_humming). The checkpoint weights are created and loaded exactly as for
Fp8MoEMethod, then converted to Humming's layout; activation, routed scaling and fused shared
experts follow the same MoeRunnerConfig as the Marlin path.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import pynvml
import torch
from torch.nn import Module

from sglang.kernels.ops.moe.moe_fused_mul_sum import warmup_sm80_moe_reduce
from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.humming import (
    HummingMoeQuantInfo,
    HummingRunnerCore,
    HummingRunnerInput,
    get_standard_humming_moe_gemm_type,
)
from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
from sglang.srt.layers.moe.utils import get_moe_a2a_backend, get_moe_runner_backend
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.layers.quantization.humming_utils import prepare_humming_moe_layer
from sglang.srt.utils import get_bool_env_var, log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput
    from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod

logger = logging.getLogger(__name__)

# Layers served by this method in this process, for the scheduler's mechanism report.
_HUMMING_LAYERS: list[str] = []
_AX_SM80_MOE_DOWN_TUNE = get_bool_env_var("SGLANG_AX_SM80_MOE_DOWN_TUNE", "false")

# Token counts of the warm-up forwards: 1 builds the GEMM table; the others hit every size range of
# moe_fused_mul_sum's BLOCK_M heuristic above the decode graph sizes (<= 128, compiled at capture), each
# with a count divisible and not divisible by 16 (Triton specializes integer arguments on both).
_WARMUP_TOKENS = (1, 129, 144, 1025, 1040)


def humming_moe_layer_count() -> int:
    return len(_HUMMING_LAYERS)


class _AxFp8HummingRunnerCore(HummingRunnerCore):
    """117's shape metadata cache; no tensors or GPU workspaces are retained.

    The base prepare_buffers derives these dictionaries twice per forward.
    Reuse only concrete shapes; symbolic/fake shapes keep the original path.
    A bounded cache avoids growing with every ragged prefill length.
    """

    def get_humming_gemm_configs(self, humming_gemm_type):
        if not _AX_SM80_MOE_DOWN_TUNE:
            return super().get_humming_gemm_configs(humming_gemm_type)
        key = humming_gemm_type.value
        if key in self.humming_gemm_configs:
            return self.humming_gemm_configs[key]
        configs = super().get_humming_gemm_configs(humming_gemm_type)
        layer = self.layer
        if (
            key != "indexed"
            or layer.hidden_size != 4096
            or layer.intermediate_size_per_partition != 256
            or self.num_experts != 289
            or self.config.top_k != 9
            or layer.params_dtype != torch.bfloat16
            or layer.w2_weight.device.type != "cuda"
            or torch.version.hip is not None
        ):
            return configs
        from humming import dtypes

        meta = layer.humming_metas["w2"]
        if (
            meta.a_dtype != dtypes.bfloat16
            or meta.b_dtype != dtypes.float8e4m3
            or meta.c_dtype != dtypes.bfloat16
            or meta.weight_scale_group_size != 128
            # 117 expands each original 128-row block scale over its N rows.
            or meta.weight_scale_group_size_n != 0
            or meta.bs_dtype != dtypes.bfloat16
            or torch.cuda.get_device_capability(layer.w2_weight.device) != (8, 0)
        ):
            return configs
        from sglang.srt.layers.quantization.fp8_humming_tuning import (
            tune_sm80_prefill_down,
        )

        tuned = tune_sm80_prefill_down(configs)
        self.humming_gemm_configs[key] = tuned
        if tuned is not configs:
            log_info_on_rank0(
                logger,
                "SM80 Humming MoE W2 uses N128/stages3/CTA2 for 8192–16384 tokens.",
            )
        return tuned

    def get_buffer_metas(self, hidden_states, topk_ids, gemm_type):
        hidden_shape, topk_shape = hidden_states.shape, topk_ids.shape
        shape = (*hidden_shape, *topk_shape)
        if not all(isinstance(dim, int) for dim in shape):
            return super().get_buffer_metas(hidden_states, topk_ids, gemm_type)
        layer = self.layer
        meta = layer.humming_metas["w13"]
        key = (hidden_shape, topk_shape, gemm_type,
               self.num_experts, layer.hidden_size, layer.intermediate_size_per_partition,
               meta.a_dtype, meta.c_dtype)
        # prepare_buffers asks twice consecutively; matching the last key
        # also avoids repeatedly hashing Humming's dtype descriptors.
        if key == getattr(self, "_ax_buffer_meta_last_key", None):
            return self._ax_buffer_meta_last_value
        cache = getattr(self, "_ax_buffer_metas", None)
        if cache is None:
            cache = self._ax_buffer_metas = {}
        result = cache.get(key)
        if result is None:
            result = super().get_buffer_metas(hidden_states, topk_ids, gemm_type)
            if len(cache) >= 64:
                del cache[next(iter(cache))]
            cache[key] = result
        self._ax_buffer_meta_last_key = key
        self._ax_buffer_meta_last_value = result
        return result


class Fp8HummingMoEMethod(FusedMoEMethodBase):
    """Block-FP8 experts (FP8 e4m3 weights, 2-D block scales) with BF16 activations on Humming.

    Tensor-parallel experts only: expert parallelism, other all-to-all backends, biases and
    non-SwiGLU activations are refused when the layer is built instead of being run untested.
    """

    def __init__(self, fp8_method: Fp8MoEMethod, prefix: str):
        self._fp8 = fp8_method
        self.prefix = prefix

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self._fp8.create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )

    def create_moe_runner(self, layer: Module, moe_runner_config: MoeRunnerConfig):
        _check_supported(layer, moe_runner_config)
        # One runner core per layer for the model's lifetime. The base ("none", "humming")
        # fused func builds a new core on every call, which re-derives the tuning table.
        self.runner_core = _AxFp8HummingRunnerCore(moe_runner_config)
        self.gemm_type = get_standard_humming_moe_gemm_type()

    def process_weights_after_loading(self, layer: Module) -> None:
        weight_config = {
            "quant_method": "fp8",
            "weight_block_size": list(self._fp8.weight_block_size),
        }
        prepare_humming_moe_layer(layer, weight_config)
        self._prepare_kernels(layer)
        _HUMMING_LAYERS.append(self.prefix)

    def _prepare_kernels(self, layer: Module) -> None:
        """Derive the tuning table and JIT-compile its kernels now rather than in the first forward.

        Both are cached per process by layer shape, so only the first layer of a shape pays.
        Humming's heuristics initialise and shut down NVML twice per candidate batch size
        (~1.5k calls per layer shape, 71 s on an A100 dev box). NVML reference-counts
        initialisation, so holding one handle open makes them cheap (0.7 s, identical table).
        """
        start = time.perf_counter()
        self.runner_core.layer = layer
        pynvml.nvmlInit()
        try:
            configs = self.runner_core.get_humming_gemm_configs(self.gemm_type)
        finally:
            pynvml.nvmlShutdown()
        _check_block_heights(configs, self.prefix)
        # The first forward compiles every GEMM kernel of the table (all batch-size ranges); the rest
        # compile the Triton top-k sum for the prefill size ranges, so no request pays a JIT.
        device = layer.w13_weight.device
        top_k = self.runner_core.config.top_k
        for tokens in _WARMUP_TOKENS:
            self.runner_core.run(
                HummingRunnerInput(
                    hidden_states=torch.zeros(
                        tokens, layer.hidden_size, dtype=layer.params_dtype, device=device
                    ),
                    topk_weights=torch.zeros(tokens, top_k, dtype=torch.float32, device=device),
                    topk_ids=torch.arange(top_k, dtype=torch.int32, device=device).repeat(tokens, 1),
                    gemm_type=self.gemm_type,
                ),
                HummingMoeQuantInfo(layer=layer),
                running_state={},
            )
        warmup_sm80_moe_reduce(
            device,
            layer.params_dtype,
            layer.hidden_size,
            top_k,
            self.runner_core.config.routed_scaling_factor,
        )
        elapsed = time.perf_counter() - start
        if elapsed > 1.0:
            log_info_on_rank0(
                logger,
                f"Humming MoE tuning table and kernel JIT took {elapsed:.1f}s "
                f"({self.prefix}); later layers of this shape reuse them.",
            )

    def apply(self, layer: Module, dispatch_output: DispatchOutput) -> CombineInput:
        topk_output = dispatch_output.topk_output
        runner_output = self.runner_core.run(
            HummingRunnerInput(
                hidden_states=dispatch_output.hidden_states,
                topk_weights=topk_output.topk_weights,
                topk_ids=topk_output.topk_ids,
                gemm_type=self.gemm_type,
            ),
            HummingMoeQuantInfo(layer=layer),
            running_state={},
        )
        return StandardCombineInput(hidden_states=runner_output.hidden_states)


def _check_block_heights(configs: dict, prefix: str) -> None:
    """The runner sorts tokens into expert blocks with the w13 table's block height; the w2 kernel reads the
    same expert-block ids with its own table's height. Refuse a table where they differ for any batch size
    (Humming 0.1.12 on sm80 derives both from M and the expert count, so they agree today)."""

    def height(table, m):
        return next(config["block_shape"][0] for lo, hi, config in table if lo < m <= hi)

    w13, w2 = configs["w13_tuning_config"], configs["w2_tuning_config"]
    for m in sorted({lo + 1 for lo, _, _ in w13} | {lo + 1 for lo, _, _ in w2}):
        if height(w13, m) != height(w2, m):
            raise RuntimeError(
                f"[ax] 117: Humming w13/w2 tuning tables disagree on the expert block height at "
                f"{m} tokens ({height(w13, m)} vs {height(w2, m)}) for {prefix}"
            )


def _check_supported(layer: Module, config: MoeRunnerConfig) -> None:
    backend, a2a_backend = get_moe_runner_backend(), get_moe_a2a_backend()
    unsupported = {
        f"--moe-runner-backend {backend.value}": not (
            backend.is_auto() or backend.is_humming()
        ),
        f"--moe-a2a-backend {a2a_backend.value}": not a2a_backend.is_none(),
        f"expert parallelism (moe_ep_size={layer.moe_ep_size})": layer.moe_ep_size != 1,
        "expert biases": layer.with_bias,
        f"activation {config.activation!r} (gated={config.is_gated})": not (
            config.activation == "silu" and config.is_gated
        ),
        "gemm1_alpha (GPT-OSS style SwiGLU)": config.gemm1_alpha is not None,
        "apply_router_weight_on_input": config.apply_router_weight_on_input,
        "no_combine": config.no_combine,
        # Humming would quantize activations; this method is BF16-activation only.
        "SGLANG_HUMMING_INPUT_QUANT_CONFIG": bool(
            envs.SGLANG_HUMMING_INPUT_QUANT_CONFIG.get()
        ),
    }
    refused = [name for name, bad in unsupported.items() if bad]
    if refused:
        raise ValueError(
            "SGLANG_AX_SM80_FP8_MOE_HUMMING=1 supports tensor-parallel SwiGLU experts "
            f"with BF16 activations only; {layer.layer_name} uses {', '.join(refused)}. "
            "Unset the switch to use the Marlin path."
        )
