"""act_quant on sm80 (patch 110) vs a torch reference of the same formula: y = clamp(x/scale, +-448) -> e4m3."""
import sys
import torch
sys.path.insert(0, sys.argv[1])
from sglang.kernels.ops.attention.dsa.triton_kernel import act_quant  # patched tree
import sglang.srt.layers.attention.dsa.kpool_fp8_index as kp  # import check only
x = (torch.randn(1000, 64, 128, device="cuda") * 3).to(torch.bfloat16).contiguous()
for fmt in (None, "ue8m0"):
    y, s = act_quant(x, 128, fmt)
    xf = x.float().view(-1, 128)
    amax = xf.abs().amax(1).clamp(min=1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0))) if fmt else amax / 448.0
    ref = (xf / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    same = (y.view(-1, 128).view(torch.uint8) == ref.view(torch.uint8)).float().mean().item()
    print(f"scale_fmt={fmt}: y dtype {y.dtype}, bytes equal {same*100:.4f}%, scale max rel err {((s.view(-1)-scale)/scale).abs().max().item():.2e}")
    assert same > 0.9999
print("ACT_QUANT OK")
