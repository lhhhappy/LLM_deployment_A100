"""140: snapshot raw KDA convolution history before the in-place conv forward.

KDA layout is [slot, history, channel]; offsets are relative to each sequence,
[batch, 2], and -1 slots are disabled. Only full 64-token chunks are exported.
"""
import triton
import triton.language as tl


@triton.jit
def _store_conv(raw, state, cu, offsets, slots,
                raw_s0: tl.constexpr, raw_s1: tl.constexpr,
                state_s0: tl.constexpr, state_s1: tl.constexpr, state_s2: tl.constexpr,
                WIDTH: tl.constexpr, HISTORY: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    seq = row // 2
    dst = tl.load(slots + row).to(tl.int64)
    offset = tl.load(offsets + row)
    start = tl.load(cu + seq)
    end = tl.load(cu + seq + 1)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = (dst >= 0) & (offset >= HISTORY) & (offset <= end - start) & (offset % 64 == 0)
    pos = start + offset - HISTORY + col // WIDTH
    channel = col % WIDTH
    value = tl.load(raw + pos * raw_s0 + channel * raw_s1,
                    mask=valid & (col < HISTORY * WIDTH), other=0)
    tl.store(state + dst * state_s0 + (col // WIDTH) * state_s1 + channel * state_s2,
             value, mask=valid & (col < HISTORY * WIDTH))


def store_conv(raw, state, cu, offsets, slots):
    history, width = state.shape[1:]
    assert history <= 64 and raw.shape[-1] == width
    assert offsets.shape == slots.shape == (cu.numel() - 1, 2)
    _store_conv[(offsets.numel(), triton.cdiv(history * width, 256))](
        raw, state, cu, offsets, slots, *raw.stride(), *state.stride(), width, history, 256)
