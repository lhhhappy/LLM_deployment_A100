"""Warmed BV16 recurrence for the measured SM80 KDA prefill layout.

The caller opts in explicitly. Shared GDN defaults and the recurrence math
are unchanged. The smaller V tile exposes more CTAs for long sequences;
the sequential chunk recurrence and FP32 persistent state stay intact.
"""

from functools import lru_cache

import torch


_WARMED_DEVICES = set()


def _native_configuration_is_default():
    from sglang.kernels.ops.attention.fla import chunk_delta_h, chunk_delta_h_snapshot

    return all(
        (module.GDN_CHUNK_H_BV, module.GDN_CHUNK_H_NUM_WARPS,
         module.GDN_CHUNK_H_NUM_STAGES) == (32, 4, 2)
        for module in (chunk_delta_h, chunk_delta_h_snapshot)
    )


def warmup_state_sm80(device_index: int) -> bool:
    """Compile and load the twelve serving signatures before graph capture.

    Refuse non-native GDN overrides rather than replacing a user's selected
    state tile. No model state is used, and no scratch tensor is retained.
    """
    if not _native_configuration_is_default():
        return False
    device = torch.device("cuda", device_index)
    if torch.cuda.get_device_capability(device) != (8, 0):
        return False
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            return False
        return _warmup_state_sm80(device_index)


@lru_cache(maxsize=None)
def _warmup_state_sm80(device_index: int) -> bool:
    from sglang.kernels.ops.attention.fla.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
    )
    from sglang.kernels.ops.attention.fla.chunk_delta_h_snapshot import (
        chunk_gated_delta_rule_fwd_kernel_h_blockdim64_snapshot,
    )

    device = torch.device("cuda", device_index)
    # T is not specialized. NT_BUCKET is only a compile/autotune key: the
    # kernel obtains its actual loop count from cu_seqlens. A single chunk
    # therefore executes both long-request signatures with about 2.3 MiB
    # of temporary storage. Actual launches also load CUDA modules/launchers;
    # JITFunction.warmup alone leaves those handles lazy.
    kvw = torch.zeros((1, 64, 8, 128), dtype=torch.bfloat16, device=device)
    v_new = torch.empty_like(kvw)
    gk = torch.zeros(kvw.shape, dtype=torch.float32, device=device)
    h = torch.empty((1, 1, 8, 128, 128), dtype=torch.bfloat16, device=device)
    state = torch.zeros((3, 8, 128, 128), dtype=torch.float32, device=device)
    cu = torch.tensor([0, 64], dtype=torch.int32, device=device)
    chunk_offsets = torch.tensor([0, 1], dtype=torch.int64, device=device)
    snapshot_offsets = torch.tensor([[64, 64]], dtype=torch.int64, device=device)
    snapshot_slots = torch.tensor([[1, 2]], dtype=torch.int64, device=device)
    common = dict(
        k=kvw, v=kvw, w=kvw, v_new=v_new, g=None, gk=gk, h=h,
        initial_state=state, stride_init_state=state.stride(0),
        cu_seqlens=cu, chunk_offsets=chunk_offsets, T=64,
        H=8, Hg=8, K=128, V=128, BT=64, BV=16,
        USE_G=False, USE_GK=True, USE_INITIAL_STATE=True,
        INPLACE_UPDATE=True, SAVE_NEW_VALUE=True, IS_VARLEN=True,
        USE_EXP2=True, num_warps=4, num_stages=2, num_ctas=1,
    )
    normal = chunk_gated_delta_rule_fwd_kernel_h_blockdim64.fn
    snapshot = chunk_gated_delta_rule_fwd_kernel_h_blockdim64_snapshot.fn
    for dtype in (torch.int32, torch.int64):
        indices = torch.zeros((1,), dtype=dtype, device=device)
        for bucket in (1, 2):
            args = dict(common, initial_state_indices=indices, NT_BUCKET=bucket)
            normal[(8, 8)](**args)
            snapshot[(8, 8)](
                **args, snapshot_offsets=None, snapshot_slots=None,
                EXPORT_SNAPSHOTS=False,
            )
            snapshot[(8, 8)](
                **args, snapshot_offsets=snapshot_offsets,
                snapshot_slots=snapshot_slots, EXPORT_SNAPSHOTS=True,
            )
    torch.cuda.current_stream(device).synchronize()
    _WARMED_DEVICES.add(device_index)
    return True


def can_use_state_sm80(
    k, w, u, g, gk, initial_state, indices, cu, chunk_offsets, NT,
    *, save_new_value=True, use_exp2=False,
    snapshot_offsets=None, snapshot_slots=None,
) -> bool:
    """Accept only warmed pointer/constexpr signatures without reading data.

    Sequence lengths and state-slot validity remain the caller's existing
    contract. Shapes suffice here; no GPU scalar read or synchronization is
    introduced. Ordinary and envelope-strided state pools share a runtime
    i32, divisible-by-16 slot pitch, so different pitches need no extra JIT.
    """
    if (
        g is not None or gk is None or not save_new_value or not use_exp2
        or not _native_configuration_is_default()
        or k.ndim != 4 or k.shape[0] != 1
        or k.shape[1] not in (8192, 16384) or k.shape[2:] != (8, 128)
        or k.device.type != "cuda" or k.device.index not in _WARMED_DEVICES
        or cu is None or cu.ndim != 1 or not 2 <= cu.numel() <= 4
    ):
        return False
    n = cu.numel() - 1
    if type(NT) is not int or not k.shape[1] // 64 <= NT <= k.shape[1] // 64 + n - 1:
        return False
    for tensor, dtype in (
        (k, torch.bfloat16), (w, torch.bfloat16), (u, torch.bfloat16),
        (gk, torch.float32),
    ):
        if tensor.shape != k.shape or tensor.dtype != dtype or not tensor.is_contiguous():
            return False
    if (
        initial_state is None or initial_state.ndim != 4
        or initial_state.shape[0] < 1 or initial_state.shape[1:] != (8, 128, 128)
        or initial_state.dtype != torch.float32
        or initial_state.stride()[1:] != (16384, 128, 1)
        or not 131072 <= initial_state.stride(0) < 2**31
        or initial_state.stride(0) % 16
        or indices is None or indices.shape != (n,)
        or indices.dtype not in (torch.int32, torch.int64) or not indices.is_contiguous()
        or cu.dtype != torch.int32 or not cu.is_contiguous()
        or chunk_offsets is None or chunk_offsets.shape != (n + 1,)
        or chunk_offsets.dtype != torch.int64 or not chunk_offsets.is_contiguous()
    ):
        return False
    tensors = (k, w, u, gk, initial_state, indices, cu, chunk_offsets)
    if snapshot_offsets is not None:
        if snapshot_slots is None:
            return False
        for tensor in (snapshot_offsets, snapshot_slots):
            if tensor.shape != (n, 2) or tensor.dtype != torch.int64 or not tensor.is_contiguous():
                return False
        tensors += (snapshot_offsets, snapshot_slots)
    elif snapshot_slots is not None:
        return False
    return all(t.device == k.device and t.data_ptr() % 16 == 0 for t in tensors)
