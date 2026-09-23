# [ax] 110: software fp8 (e4m3fn) stores for devices whose Triton cannot convert to fp8e4nv (sm80/A100).
# On devices below sm89, fp8 output buffers are passed to Triton kernels as uint8 (_ax_fp8_view) and
# kernels write through _ax_store_fp8, which encodes float32 -> e4m3fn bits in software with
# round-to-nearest-even (bit-exact vs torch's float32 -> float8_e4m3fn for |x| <= 448, incl. -0.0 and
# subnormals; callers clamp to +-448 before storing). On sm89+ the pointers stay fp8 and _ax_store_fp8
# is a plain tl.store, so behaviour there is unchanged.
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_SOFT = {}


def _ax_fp8_view(t: torch.Tensor) -> torch.Tensor:
    dev = t.device.index if t.device.index is not None else torch.cuda.current_device()
    if dev not in _SOFT:
        major, minor = torch.cuda.get_device_capability(dev)
        _SOFT[dev] = major * 10 + minor < 89
    return t.view(torch.uint8) if _SOFT[dev] else t.view(torch.float8_e4m3fn)


@triton.jit
def _ax_f32_to_e4m3_bits(x):
    sign = (x.to(tl.int32, bitcast=True) >> 31) & 1          # from the bits: keeps -0.0
    a = tl.minimum(tl.abs(x), 448.0)
    ai = a.to(tl.int32, bitcast=True)
    e = ((ai >> 23) & 0xFF) - 127
    frac = (ai & 0x7FFFFF).to(tl.float32) * (1.0 / 8388608.0)
    m = libdevice.rint(frac * 8.0).to(tl.int32)               # 3 mantissa bits, RNE
    e = e + (m == 8).to(tl.int32)
    m = tl.where(m == 8, 0, m)
    bits = ((e + 7) << 3) | m                                 # normal: exponent bias 7
    sub = libdevice.rint(a * 512.0).to(tl.int32)              # subnormal step 2^-9 (a < 2^-6)
    bits = tl.where(a < 0.015625, sub, bits)
    bits = tl.where(a == 0.0, 0, bits) | (sign << 7)
    return bits.to(tl.uint8)


@triton.jit
def _ax_store_fp8(ptrs, x, mask):
    if ptrs.dtype.element_ty == tl.uint8:
        tl.store(ptrs, _ax_f32_to_e4m3_bits(x), mask=mask)
    else:
        tl.store(ptrs, x, mask=mask)
