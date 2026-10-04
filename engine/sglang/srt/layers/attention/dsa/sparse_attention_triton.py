"""[ax] 118: Triton DSA sparse attention (absorbed MLA, no RoPE tail) for A100.

Replaces TileLang ``sparse_attention_fwd_kernel_v1`` (``tilelang_sparse_fwd`` with ``tail_dim == 0``) when
``SGLANG_AX_DSA_SPARSE_TRITON=1``; see ``DeepseekSparseAttnBackend._forward_tilelang`` in dsa_backend.py and
engine/docs/118-dsa-sparse-triton.md.

Contract (same as ``tilelang_sparse_fwd``): ``q`` [M, H, D] bf16, ``kv`` [P, 1, D] bf16 addressed one token per row,
``indices`` [M, K] or [M, 1, K] int32 (-1 = masked, at any position); returns ``out`` [1, M, H, D] bf16 and, with
``return_lse``, ``lse`` [1, M, H] fp32 in log2 units (log2 of the sum of 2^(score * sm_scale * log2(e))), which is
what the DCP log2/exp2 combine consumes. ``K`` needs no padding to a multiple of 64.

Design: one program per (query row, block of 16 heads, split). Heads are zero-padded to 16 rows, the smallest
mma.m16n8k16 tile; for H = 8 (64 heads over TP8) that wastes half the MMA work but not the bandwidth, which is what
bounds this kernel: every row gathers its own ~2k KV rows of 1 KiB. Each tile of 32 KV rows is gathered with masked
cp.async into a single shared-memory buffer. With num_stages=3 Triton prefetches the index tiles two tiles ahead
and issues the next KV gather right after the current tile's two MMAs, so a program overlaps little of its own
gather; latency is hidden by the programs resident on each SM (three on A100: 128 registers, 49 KiB of shared
memory). Tiles after the row's last valid index are skipped, so short histories (a -1 suffix) cost what they read.

When the grid would leave SMs idle (decode, target verify, draft, short extends), each row's tiles are split over up
to 16 programs and a second kernel merges the partial softmax states. The split count depends only on the host-side
shape and the device, so it is fixed for a captured CUDA graph; but a row's summation order, and so its last bits,
depends on how many rows share the call (not batch-invariant, unlike the TileLang kernel).

Deliberate differences from the TileLang kernel, both only for rows or entries serving never produces:
- A row with no valid index returns 0 and LSE -inf (TileLang: 0/0 = NaN, LSE -inf). Every real query row holds at
  least its own token; fully masked rows are padding from ``_pad_topk_indices``, whose output is discarded.
- An index >= P is masked like -1 (TileLang zero-fills the KV row but still counts it with logit 0).

Every kernel specialization serving can launch is compiled and loaded by ``warmup_sparse_attention_fwd`` at backend
construction, never during CUDA-graph capture or mid-serving; the warmup also measures how many programs fit on
the GPU at once from the compiled kernel. Lengths, pool size and index strides are runtime integers, so new shapes
do not recompile.
"""

import functools
import math

import torch
import triton
import triton.language as tl

_LOG2E = math.log2(math.e)
_BLOCK_H = 16  # mma M tile; fewer heads are zero-padded
_BLOCK_I = 32  # KV rows gathered per tile
_NUM_WARPS = 4
_NUM_STAGES = 3
_MAX_SPLITS = 16
# (device index, H, D, K_POW2) -> programs of the single-split kernel resident on the whole GPU at once
_WAVE = {}


@triton.jit
def _head_offsets(H: tl.constexpr, BLOCK_H: tl.constexpr, AXIS: tl.constexpr):
    # The head-block grid axis is compiled in only when there is more than one block: with a runtime head
    # offset the kernel took 148 instead of 128 registers and 7% longer at the 8-head prefill shape on A100.
    if H > BLOCK_H:
        return tl.program_id(AXIS) * BLOCK_H + tl.arange(0, BLOCK_H)
    return tl.arange(0, BLOCK_H)


@triton.jit(do_not_specialize=["P", "K", "stride_im"])
def _sparse_attention_fwd_kernel(
    Q,
    KV,
    IDX,
    O,
    LSE,
    O_PART,
    ML_PART,
    P,
    K,
    sm_scale_log2,
    stride_qm,
    stride_qh,
    stride_kvp,
    stride_im,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
    K_POW2: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    offs_h = _head_offsets(H, BLOCK_H, 2)
    offs_d = tl.arange(0, D)
    offs_i = tl.arange(0, BLOCK_I)
    h_mask = offs_h < H

    q = tl.load(
        Q + row * stride_qm + offs_h[:, None] * stride_qh + offs_d[None, :],
        mask=h_mask[:, None],
        other=0.0,
    )

    # Visit only the tiles up to the one holding the row's last valid index.
    idx_row = IDX + row * stride_im
    offs_k = tl.arange(0, K_POW2)
    row_idx = tl.load(idx_row + offs_k, mask=offs_k < K, other=-1)
    last_valid = tl.max(tl.where((row_idx >= 0) & (row_idx < P), offs_k, -1), 0)
    num_tiles = (last_valid + BLOCK_I) // BLOCK_I
    tiles_per_split = (num_tiles + NUM_SPLITS - 1) // NUM_SPLITS
    tile_begin = split * tiles_per_split
    tile_end = tl.minimum(tile_begin + tiles_per_split, num_tiles)

    m_i = tl.full([BLOCK_H], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    acc = tl.zeros([BLOCK_H, D], tl.float32)
    for t in range(tile_begin, tile_end):
        cols = t * BLOCK_I + offs_i
        kv_idx = tl.load(idx_row + cols, mask=cols < K, other=-1)
        valid = (kv_idx >= 0) & (kv_idx < P)
        kv = tl.load(
            KV + kv_idx.to(tl.int64)[:, None] * stride_kvp + offs_d[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        s = tl.dot(q, tl.trans(kv)) * sm_scale_log2
        s = tl.where(valid[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        # Nothing valid yet: shift by 0 so exp2(-inf - 0) = 0 instead of exp2(-inf + inf) = NaN.
        m_ref = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_ref)
        p = tl.exp2(s - m_ref[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new

    if NUM_SPLITS == 1:
        l_div = tl.where(l_i == 0.0, 1.0, l_i)
        tl.store(
            O + (row * H + offs_h[:, None]) * D + offs_d[None, :],
            (acc / l_div[:, None]).to(O.dtype.element_ty),
            mask=h_mask[:, None],
        )
        if RETURN_LSE:
            lse = tl.where(l_i == 0.0, float("-inf"), tl.log2(l_div) + m_i)
            tl.store(LSE + row * H + offs_h, lse, mask=h_mask)
    else:
        part = row * NUM_SPLITS + split
        tl.store(
            O_PART + (part * H + offs_h[:, None]) * D + offs_d[None, :],
            acc,
            mask=h_mask[:, None],
        )
        tl.store(ML_PART + (part * 2) * H + offs_h, m_i, mask=h_mask)
        tl.store(ML_PART + (part * 2 + 1) * H + offs_h, l_i, mask=h_mask)


@triton.jit
def _sparse_attention_combine_kernel(
    O_PART,
    ML_PART,
    O,
    LSE,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs_h = _head_offsets(H, BLOCK_H, 1)
    offs_d = tl.arange(0, D)
    h_mask = offs_h < H

    m = tl.full([BLOCK_H], float("-inf"), tl.float32)
    for s in tl.static_range(NUM_SPLITS):
        part = row * NUM_SPLITS + s
        m_part = tl.load(ML_PART + (part * 2) * H + offs_h, mask=h_mask, other=float("-inf"))
        m = tl.maximum(m, m_part)
    m_ref = tl.where(m == float("-inf"), 0.0, m)
    l = tl.zeros([BLOCK_H], tl.float32)
    acc = tl.zeros([BLOCK_H, D], tl.float32)
    for s in tl.static_range(NUM_SPLITS):
        part = row * NUM_SPLITS + s
        # An empty split has m = -inf, l = 0, acc = 0 and weight 0.
        m_part = tl.load(ML_PART + (part * 2) * H + offs_h, mask=h_mask, other=float("-inf"))
        l_part = tl.load(ML_PART + (part * 2 + 1) * H + offs_h, mask=h_mask, other=0.0)
        o_part = tl.load(
            O_PART + (part * H + offs_h[:, None]) * D + offs_d[None, :],
            mask=h_mask[:, None],
            other=0.0,
        )
        w = tl.exp2(m_part - m_ref)
        l += l_part * w
        acc += o_part * w[:, None]

    l_div = tl.where(l == 0.0, 1.0, l)
    tl.store(
        O + (row * H + offs_h[:, None]) * D + offs_d[None, :],
        (acc / l_div[:, None]).to(O.dtype.element_ty),
        mask=h_mask[:, None],
    )
    if RETURN_LSE:
        lse = tl.where(l == 0.0, float("-inf"), tl.log2(l_div) + m)
        tl.store(LSE + row * H + offs_h, lse, mask=h_mask)


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def _resident_programs(kernel, props) -> int:
    """Programs of a compiled kernel that fit on one SM at once, by the CUDA occupancy rules (registers are allocated
    per warp in units of 256; each block reserves 1 KiB of shared memory on sm80+)."""
    warps = kernel.metadata.num_warps
    regs_per_warp = triton.cdiv(kernel.n_regs * 32, 256) * 256
    return max(
        1,
        min(
            props.max_threads_per_multi_processor // (warps * 32),
            props.regs_per_multiprocessor // (regs_per_warp * warps),
            props.shared_memory_per_multiprocessor // (kernel.metadata.shared + 1024),
        ),
    )


def _wave(num_heads: int, dim: int, width: int, device: torch.device) -> int:
    key = (_device_index(device), num_heads, dim, triton.next_power_of_2(width))
    if key not in _WAVE:  # direct use without the backend's warmup
        assert not torch.cuda.is_current_stream_capturing(), (
            "call warmup_sparse_attention_fwd before CUDA-graph capture"
        )
        warmup_sparse_attention_fwd(num_heads, dim, width, device)
    return _WAVE[key]


def _num_splits(programs: int, wave: int) -> int:
    """Largest power-of-two split (<= 16) whose grid still fits in one wave of resident programs: a second,
    partial wave costs more than the split saves (measured on A100 for 1-500 rows, see the 118 doc)."""
    splits = 1
    while splits < _MAX_SPLITS and programs * splits * 2 <= wave:
        splits *= 2
    return splits


def _launch(q, kv, indices, out, lse, sm_scale, num_splits):
    num_tokens, num_heads, dim = q.shape
    head_blocks = triton.cdiv(num_heads, _BLOCK_H)
    return_lse = lse is not None
    if num_splits > 1:
        o_part = torch.empty(
            (num_tokens * num_splits, num_heads, dim), dtype=torch.float32, device=q.device
        )
        ml_part = torch.empty(
            (num_tokens * num_splits, 2, num_heads), dtype=torch.float32, device=q.device
        )
    else:
        o_part = ml_part = out  # unused by the single-split specialization
    lse_arg = lse if return_lse else out  # unused without RETURN_LSE
    kernel = _sparse_attention_fwd_kernel[(num_tokens, num_splits, head_blocks)](
        q,
        kv,
        indices,
        out,
        lse_arg,
        o_part,
        ml_part,
        kv.shape[0],
        indices.shape[1],
        sm_scale * _LOG2E,
        q.stride(0),
        q.stride(1),
        kv.stride(0),
        indices.stride(0),
        H=num_heads,
        D=dim,
        BLOCK_H=_BLOCK_H,
        BLOCK_I=_BLOCK_I,
        K_POW2=triton.next_power_of_2(indices.shape[1]),
        NUM_SPLITS=num_splits,
        RETURN_LSE=return_lse,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    if num_splits > 1:
        _sparse_attention_combine_kernel[(num_tokens, head_blocks)](
            o_part,
            ml_part,
            out,
            lse_arg,
            H=num_heads,
            D=dim,
            BLOCK_H=_BLOCK_H,
            NUM_SPLITS=num_splits,
            RETURN_LSE=return_lse,
            num_warps=4,
        )
    return kernel


def sparse_attention_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    return_lse: bool = False,
):
    """Drop-in for ``tilelang_sparse_fwd`` with ``tail_dim == 0``; see the module docstring.

    ``indices`` may be any width; -1 entries anywhere are skipped, and so is any entry >= ``kv.shape[0]``, which
    TileLang would read as a zero KV row that still takes part in the softmax (callers never pass one; the kernel
    does not check). A row without a valid entry gives 0 and LSE -inf."""
    if indices.dim() == 3:
        assert indices.shape[1] == 1, "one KV group"
        indices = indices.squeeze(1)
    assert q.dim() == 3 and kv.dim() == 3 and kv.shape[1] == 1 and indices.dim() == 2
    num_tokens, num_heads, dim = q.shape
    assert dim == d_v == kv.shape[2], "no RoPE tail: q, kv and the output share one width"
    assert dim == triton.next_power_of_2(dim), f"dim={dim} must be a power of 2"
    assert q.dtype == kv.dtype == torch.bfloat16 and indices.dtype == torch.int32
    assert indices.shape[0] == num_tokens
    assert q.stride(2) == 1 and kv.stride(2) == 1 and indices.stride(1) == 1

    out = torch.empty((1, num_tokens, num_heads, dim), dtype=q.dtype, device=q.device)
    lse = (
        torch.empty((1, num_tokens, num_heads), dtype=torch.float32, device=q.device)
        if return_lse
        else None
    )
    if num_tokens > 0:
        programs = num_tokens * triton.cdiv(num_heads, _BLOCK_H)
        wave = _wave(num_heads, dim, indices.shape[1], q.device)
        _launch(q, kv, indices, out, lse, sm_scale, _num_splits(programs, wave))
    if return_lse:
        return out, lse
    return out


@functools.cache
def warmup_sparse_attention_fwd(
    num_heads: int, dim: int, width: int, device: torch.device, return_lse: bool = False
) -> None:
    """Compile and load every specialization that ``sparse_attention_fwd`` can launch for this shape (all split
    counts; with ``return_lse`` also the LSE variants), so none is first compiled or loaded during CUDA-graph
    capture or while serving, and record how many programs fit on the GPU at once. Pointer alignment and the q/kv
    strides match serving's contiguous tensors."""
    q = torch.zeros((1, num_heads, dim), dtype=torch.bfloat16, device=device)
    kv = torch.zeros((1, 1, dim), dtype=torch.bfloat16, device=device)
    indices = torch.full((1, width), -1, dtype=torch.int32, device=device)
    out = torch.empty((1, 1, num_heads, dim), dtype=torch.bfloat16, device=device)
    lse = torch.empty((1, 1, num_heads), dtype=torch.float32, device=device)
    for with_lse in (False, True) if return_lse else (False,):
        splits = 1
        while splits <= _MAX_SPLITS:
            kernel = _launch(q, kv, indices, out, lse if with_lse else None, 1.0, splits)
            if splits == 1 and not with_lse:
                index = _device_index(device)
                props = torch.cuda.get_device_properties(index)
                key = (index, num_heads, dim, triton.next_power_of_2(width))
                _WAVE[key] = _resident_programs(kernel, props) * props.multi_processor_count
            splits *= 2
    torch.cuda.synchronize(device)
