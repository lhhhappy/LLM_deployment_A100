"""Bit-exact test of the patch-110 software e4m3fn encoder (sm80) against torch's float32->float8_e4m3fn.
Covers: every representable e4m3 value, every midpoint between neighbours (round-to-nearest-even ties),
subnormals, +-448 saturation, signed zero, and 200k random values in [-448, 448]. Also exercises the
_ax_store_fp8 branch through a uint8 pointer. Usage: python test_soft_fp8.py <patched sglang parent dir>"""
import importlib.util
import sys

import torch
import triton
import triton.language as tl

path = sys.argv[1] + "/sglang/kernels/ops/attention/dsa/ax_soft_fp8.py"
spec = importlib.util.spec_from_file_location("ax_soft_fp8_patched", path)
kp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kp)


@triton.jit
def _enc_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    kp._ax_store_fp8(out_ptr + offs, x, mask)


def encode(x):
    out = torch.empty(x.numel(), dtype=torch.uint8, device=x.device)
    _enc_kernel[(triton.cdiv(x.numel(), 1024),)](x, out, x.numel(), BLOCK=1024)
    return out


dev = "cuda"
print("device", torch.cuda.get_device_name(), "capability", torch.cuda.get_device_capability())
allbits = torch.arange(256, dtype=torch.uint8, device=dev)
vals = allbits.view(torch.float8_e4m3fn).float()
finite = vals[torch.isfinite(vals)]
finite = torch.unique(finite)                        # sorted, includes -0/+0 as one
mids = (finite[1:] + finite[:-1]) / 2                # exact ties in fp32
rnd = (torch.rand(200000, device=dev) * 2 - 1) * 448.0
tiny = (torch.rand(20000, device=dev) * 2 - 1) * 0.02  # subnormal region
edge = torch.tensor([448.0, -448.0, 447.99, -447.99, 0.0, -0.0, 0.015625, 0.0156249, 0.001953125, 0.0009765625,
                     0.0009765624, 0.0029296875], device=dev)
x = torch.cat([finite, mids, rnd, tiny, edge]).contiguous()
ours = encode(x)
ref = x.to(torch.float8_e4m3fn).view(torch.uint8)
bad = (ours != ref)
mism = bad
print(f"values tested {x.numel()}, mismatches {int(mism.sum())}")
if mism.any():
    i = mism.nonzero()[:10].flatten()
    for j in i.tolist():
        print(f"  x={x[j].item()!r} ours=0x{ours[j].item():02x} ref=0x{ref[j].item():02x}")
    raise SystemExit(1)
print("ALL BIT-EXACT")
