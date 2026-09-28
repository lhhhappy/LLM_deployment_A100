from functools import lru_cache

import torch
import triton
import triton.language as tl
from torch._subclasses.fake_tensor import FakeTensor

from sglang.srt.utils import get_bool_env_var, get_device_capability


# [ax] 117 follow-up: read once at import, before serving or graph capture. The
# disabled path retains the native heuristic and launch without device probes.
_AX_SM80_MOE_REDUCE = get_bool_env_var("SGLANG_AX_SM80_MOE_REDUCE", "false")


@lru_cache(maxsize=16)
def _is_sm80_device(device: torch.device) -> bool:
    # CUDA tensors carry an explicit index. Do not cache the current device's
    # capability globally: a process can use more than one kind of GPU.
    return (
        device.type == "cuda"
        and device.index is not None
        and torch.version.hip is None
        and torch.cuda.get_device_capability(device) == (8, 0)
    )


def _supported_sm80_reduce_scale(scale) -> bool:
    return scale is None or (type(scale) in (int, float) and scale in (1.0, 2.5))


def _can_use_sm80_moe_reduce(
    inputs, topk_weights, outputs, expert_map, routed_scaling_factor, is_ep
) -> bool:
    num_tokens, top_k, size = inputs.shape
    return (
        type(num_tokens) is int
        and num_tokens >= 4096
        and type(top_k) is int
        and top_k == 9
        and type(size) is int
        and size == 4096
        and not is_ep
        and expert_map is None
        and inputs.dtype == torch.bfloat16
        and topk_weights.dtype == torch.float32
        and outputs.dtype == torch.bfloat16
        and outputs.layout == torch.strided
        and outputs.is_contiguous()
        and inputs.device == topk_weights.device == outputs.device
        and not isinstance(topk_weights, FakeTensor)
        and not isinstance(outputs, FakeTensor)
        and _supported_sm80_reduce_scale(routed_scaling_factor)
        and _is_sm80_device(inputs.device)
    )


@triton.jit
def _moe_fused_mul_sum_sm80_kernel(
    inputs_ptr,
    topk_weights_ptr,
    outputs_ptr,
    num_tokens,
    routed_scaling_factor: tl.constexpr,
):
    # One CTA owns one token and 1024 columns. Adjacent CTAs visit adjacent
    # column tiles of that token. No persistent loop or cross-CTA reduction.
    pid = tl.program_id(0)
    offs_m = (pid // 4).to(tl.int64) + tl.arange(0, 1)
    offs_k = (pid % 4) * 1024 + tl.arange(0, 1024)
    m_mask = offs_m < num_tokens
    acc = tl.zeros((1, 1024), dtype=tl.float32)
    for n in tl.static_range(9):
        weight = tl.load(
            topk_weights_ptr + offs_m * 9 + n, mask=m_mask, other=0.0
        ).to(tl.float32)
        if routed_scaling_factor != 1.0:
            weight = weight * routed_scaling_factor
        value = tl.load(
            inputs_ptr + (offs_m[:, None] * 9 + n) * 4096 + offs_k[None, :],
            mask=m_mask[:, None],
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        # Preserve the native serial FP32 FMA order and final BF16 rounding.
        acc += value * weight[:, None]
    tl.store(
        outputs_ptr + offs_m[:, None] * 4096 + offs_k[None, :],
        acc.to(outputs_ptr.dtype.element_ty),
        mask=m_mask[:, None],
    )


@lru_cache(maxsize=16)
def _warmup_sm80_moe_reduce(device: torch.device, scale: float) -> None:
    # Compile both integer-divisibility specializations using only one row.
    # The grid, not num_tokens, bounds this nonpersistent kernel's accesses;
    # four CTAs touch only row 0 even with a representative large num_tokens.
    # Keep no GPU tensors in the cache, and do not allocate a full MoE workspace.
    with torch.cuda.device(device):
        inputs = torch.zeros((1, 9, 4096), dtype=torch.bfloat16, device=device)
        weights = torch.zeros((1, 9), dtype=torch.float32, device=device)
        outputs = torch.empty((1, 4096), dtype=torch.bfloat16, device=device)
        for num_tokens in (4096, 4097):
            _moe_fused_mul_sum_sm80_kernel[(4,)](
                inputs,
                weights,
                outputs,
                num_tokens,
                scale,
                num_warps=4,
                num_stages=1,
            )


def warmup_sm80_moe_reduce(
    device: torch.device,
    dtype: torch.dtype,
    hidden_size: int,
    top_k: int,
    routed_scaling_factor: float | None,
) -> None:
    """Compile the opt-in large-prefill reduction at 117's layer-load warmup."""
    if (
        _AX_SM80_MOE_REDUCE
        and dtype == torch.bfloat16
        and hidden_size == 4096
        and top_k == 9
        and _supported_sm80_reduce_scale(routed_scaling_factor)
        and _is_sm80_device(device)
    ):
        _warmup_sm80_moe_reduce(
            device, 1.0 if routed_scaling_factor is None else float(routed_scaling_factor)
        )


@triton.jit
def moe_fused_mul_sum_kernel(
    inputs_ptr,
    topk_weights_ptr,
    outputs_ptr,
    top_ids_ptr,
    expert_map_ptr,
    num_tokens,
    stride_m,
    has_expert_map: tl.constexpr,
    is_ep: tl.constexpr,
    top_k: tl.constexpr,
    size: tl.constexpr,
    routed_scaling_factor: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_m = tl.program_id(1).to(tl.int64)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    m_mask = offs_m < num_tokens
    k_mask = offs_k < size
    mask = m_mask[:, None] & k_mask[None, :]

    a_base = inputs_ptr + (offs_m * stride_m)[:, None] + offs_k[None, :]
    b_base = topk_weights_ptr + offs_m * top_k

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for n in tl.static_range(top_k):
        b_val = tl.load(b_base + n, mask=m_mask, other=0.0).to(tl.float32)
        if routed_scaling_factor != 1.0:
            b_val = b_val * routed_scaling_factor
        if has_expert_map:
            id_val = tl.load(top_ids_ptr + offs_m * top_k + n, mask=m_mask, other=0)
            expert_mask = tl.load(expert_map_ptr + id_val) >= 0
            a_vec = tl.load(
                a_base + n * size,
                mask=mask & expert_mask[:, None],
                other=0.0,
            ).to(tl.float32)
        elif is_ep:
            id_val = tl.load(top_ids_ptr + offs_m * top_k + n, mask=m_mask, other=0)
            expert_mask = id_val >= 0
            a_vec = tl.load(
                a_base + n * size,
                mask=mask & expert_mask[:, None],
                other=0.0,
            ).to(tl.float32)
        else:
            a_vec = tl.load(
                a_base + n * size,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
        acc += a_vec * b_val[:, None]

    out_ptrs = outputs_ptr + (offs_m * size)[:, None] + offs_k[None, :]
    tl.store(
        out_ptrs,
        acc.to(outputs_ptr.dtype.element_ty),
        mask=mask,
    )


def _heuristic_config(
    num_tokens: int,
    top_k: int,
    size: int,
    element_size: int,
):
    is_fp32 = element_size > 2
    major, _ = get_device_capability()
    is_sm90_plus = major is not None and major >= 9
    is_sm80_before = major is None or major < 8

    if is_sm90_plus:
        # SM90/SM100+: prefer small tiles + many CTAs.
        if is_fp32:
            BLOCK_M = 1 if num_tokens <= 4 else 2
        else:
            if num_tokens <= 4:
                BLOCK_M = 1
            elif num_tokens <= 128:
                BLOCK_M = 2
            else:
                BLOCK_M = 4
    elif is_fp32:
        if num_tokens <= 4:
            BLOCK_M = 1
        elif num_tokens <= 32:
            BLOCK_M = 2
        elif num_tokens <= 128:
            BLOCK_M = 4
        else:
            BLOCK_M = 4
    else:
        if num_tokens <= 4:
            BLOCK_M = 1
        elif num_tokens <= 32:
            BLOCK_M = 2
        elif num_tokens <= 128:
            BLOCK_M = 4
        elif num_tokens <= 1024:
            BLOCK_M = 16
        else:
            BLOCK_M = 8

    if is_fp32:
        max_block_k = 256
    elif is_sm80_before or is_sm90_plus:
        max_block_k = 512
    else:
        max_block_k = 1024
    BLOCK_K = min(triton.next_power_of_2(size), max_block_k)
    BLOCK_K = max(BLOCK_K, 256)

    total = BLOCK_M * BLOCK_K
    if is_fp32:
        num_warps = max(8, min(16, total // 64))
    else:
        num_warps = max(4, min(16, total // 256))

    if is_sm80_before:
        num_warps = min(num_warps, 8)
        num_stages = 2
    elif is_sm90_plus:
        num_warps = min(num_warps, 8)
        num_stages = 4 if total <= 2048 else 2
    else:
        num_stages = 4 if total <= 2048 else 2

    return BLOCK_M, BLOCK_K, num_warps, num_stages


def moe_fused_mul_sum(
    inputs: torch.Tensor,
    topk_weights: torch.Tensor,
    outputs: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    expert_map: torch.Tensor | None = None,
    routed_scaling_factor: float | None = None,
    is_ep: bool = False,
) -> torch.Tensor:
    """
    Fused kernel for MoE (Mixture of Experts) to perform weighted summation
    of expert outputs.

    Args:
        inputs: The output from experts.
            Shape: (num_tokens, top_k, hidden_size).
        topk_weights: The weights assigned to each expert for each token.
            Shape: (num_tokens, top_k).
        outputs: Optional pre-allocated output tensor.
            Shape: (num_tokens, hidden_size).
        topk_ids: Optional indices of the top-k experts. Used when
            `expert_map` is provided. Shape: (num_tokens, top_k).
        expert_map: Optional mapping for Expert Parallelism. A value < 0
            indicates an invalid token/expert pair that will be skipped.

    Returns:
        The fused weighted sum of expert outputs.
        Shape: (num_tokens, hidden_size).
    """
    assert inputs.ndim == 3
    assert topk_weights.ndim == 2
    assert inputs.is_contiguous()
    assert topk_weights.is_contiguous()
    assert inputs.dtype in (torch.float32, torch.float16, torch.bfloat16)
    assert topk_weights.dtype in (torch.float32, torch.float16, torch.bfloat16)

    num_tokens, top_k, size = inputs.shape
    output_shape = (num_tokens, size)
    if outputs is None:
        outputs = torch.empty(output_shape, dtype=inputs.dtype, device=inputs.device)

    assert outputs.shape == output_shape
    assert topk_weights.shape == (num_tokens, top_k)

    if not isinstance(inputs, FakeTensor):
        if _AX_SM80_MOE_REDUCE and _can_use_sm80_moe_reduce(
            inputs, topk_weights, outputs, expert_map, routed_scaling_factor, is_ep
        ):
            _moe_fused_mul_sum_sm80_kernel[(num_tokens * 4,)](
                inputs,
                topk_weights,
                outputs,
                num_tokens,
                1.0 if routed_scaling_factor is None else float(routed_scaling_factor),
                num_warps=4,
                num_stages=1,
            )
            return outputs

        BLOCK_M, BLOCK_K, num_warps, num_stages = _heuristic_config(
            num_tokens,
            top_k,
            size,
            inputs.element_size(),
        )
        grid = (triton.cdiv(size, BLOCK_K), triton.cdiv(num_tokens, BLOCK_M))
        moe_fused_mul_sum_kernel[grid](
            inputs,
            topk_weights,
            outputs,
            topk_ids,
            expert_map,
            num_tokens,
            top_k * size,
            expert_map is not None,
            is_ep,
            top_k,
            size,
            1.0 if routed_scaling_factor is None else routed_scaling_factor,
            BLOCK_M,
            BLOCK_K,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return outputs
