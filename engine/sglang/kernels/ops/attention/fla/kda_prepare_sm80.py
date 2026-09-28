"""Opt-in SM80 prefill preparation for the GLM KDA per-rank layout.

Read the BF16 convolution output once, normalize Q/K with the native L2
formula, and write three contiguous tensors. The native path performs
three strided copies and two normalization launches. No state or gate math
changes here. Only the measured 8k/16k, H=8, D=128 layout is dispatched.
"""

from functools import lru_cache

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["ROWS"])
def _prepare_qkv_kernel(Q, K, V, Q_OUT, K_OUT, V_OUT, ROWS, BT: tl.constexpr):
    rows = tl.program_id(0) * BT + tl.arange(0, BT)
    cols = tl.arange(0, 128)
    plane = tl.program_id(1)
    src = Q if plane == 0 else (K if plane == 1 else V)
    dst = Q_OUT if plane == 0 else (K_OUT if plane == 1 else V_OUT)
    offsets = (rows // 8)[:, None] * 3072 + (rows % 8)[:, None] * 128 + cols[None, :]
    x = tl.load(src + offsets, rows[:, None] < ROWS, 0)
    target = dst + rows[:, None] * 128 + cols[None, :]
    if plane < 2:
        # Match l2norm_fwd_kernel, including FP32 reduction, epsilon and
        # division by sqrt (not multiplication by rsqrt).
        xf = x.to(tl.float32)
        variance = tl.sum(xf * xf, axis=1)
        y = xf / tl.sqrt(variance + 1e-6)[:, None]
        tl.store(target, y, rows[:, None] < ROWS)
    else:
        # V is a bit copy. Avoid BF16 -> FP32 -> BF16 NaN canonicalization.
        tl.store(target, x, rows[:, None] < ROWS)


def _prepare_qkv(q, k, v):
    # The attention output overwrites V and outlives Q/K. Separate allocations
    # release Q/K at the native lifetime instead of retaining a 3-plane storage
    # through the returned V view (an extra 32/64 MiB at 8k/16k).
    output = tuple(torch.empty(q.shape, dtype=q.dtype, device=q.device) for _ in range(3))
    rows = q.shape[1] * 8
    _prepare_qkv_kernel[(triton.cdiv(rows, 64), 3)](
        q, k, v, *output, rows, BT=64, num_warps=8, num_stages=3
    )
    return output


@lru_cache(maxsize=None)
def warmup_prepare_sm80(device_index: int):
    """Compile once during backend setup, before any user request/capture.

    ROWS is not specialized, and all pointers are 16-byte aligned in both
    the tiny warmup and guarded serving path. No tensor is kept in this cache.
    """
    device = torch.device("cuda", device_index)
    with torch.cuda.device(device):
        packed = torch.zeros((16, 3, 8, 128), dtype=torch.bfloat16, device=device)
        q, k, v = (packed[:, i].unsqueeze(0) for i in range(3))
        _prepare_qkv(q, k, v)
        torch.cuda.current_stream(device).synchronize()


def prepare_qkv_if_supported(q, k, v, armed_device):
    """Return prepared planes, or None without reading tensor data.

    ``armed_device`` is set only after the selected Triton prefill backend
    has warmed this SM80 kernel. Unsupported shapes keep the native path.
    """
    if armed_device is None:
        return None
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[1] not in (8192, 16384):
        return None
    if q.shape[2:] != (8, 128):
        return None
    for tensor in (q, k, v):
        if (
            tensor.shape != q.shape
            or tensor.dtype != torch.bfloat16
            or tensor.device != armed_device
            or tensor.stride()[1:] != (3072, 128, 1)
            or tensor.data_ptr() % 16
        ):
            return None
    return _prepare_qkv(q, k, v)
