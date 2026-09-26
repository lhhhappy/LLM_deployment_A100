"""Owner filtering, KPool column compaction and TileLang padding in one pass."""

import torch
import triton
import triton.language as tl


@triton.jit
def _local_indices(
    src,
    dst,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    OUT_COLS: tl.constexpr,
    WIDTH: tl.constexpr,
    RANK: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = offset // OUT_COLS, offset % OUT_COLS
    source_col = col * STRIDE + RANK % STRIDE
    loc = tl.load(
        src + row * S0 + source_col * S1,
        (row < ROWS) & (source_col < COLS),
        other=-1,
    )
    own = (loc >= 0) & (loc % WIDTH == RANK)
    value = tl.where(own, loc // WIDTH, -1).to(tl.int32)
    tl.store(dst + offset, value, row < ROWS)


def local_dcp_indices(indices, *, width: int, rank: int, kpool_stride: int):
    """Convert virtual locs to physical rows; tail/padding stays masked.

    Stride > 1 requires KPool-expanded columns and aligned allocator pages.
    No tensor is read on the host; noncontiguous input tables are supported.
    """
    rows, columns = indices.shape
    live_columns = (columns - rank % kpool_stride + kpool_stride - 1) // kpool_stride
    padded_columns = triton.cdiv(live_columns, 64) * 64
    out = torch.empty((rows, padded_columns), dtype=torch.int32, device=indices.device)
    if out.numel():
        _local_indices[(triton.cdiv(out.numel(), 256),)](
            indices,
            out,
            rows,
            columns,
            *indices.stride(),
            padded_columns,
            width,
            rank,
            kpool_stride,
            256,
        )
    return out
