# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``fwht128_quant_fp8`` (GLM-5.3-Flash indexer Q) against a torch reference.

The reference repeats the kernel's fp32 butterflies in the same order, so the
rotated values are bit-identical; with the kernel's own power-of-two scale the
quantized bytes must then equal ``torch``'s e4m3fn cast exactly. This covers
the software encode on sm80 and the hardware convert on sm89+ alike.
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA Triton kernel"
)

INV_SQRT_128 = 0.08838834764831845


def _fwht128_ref(x: torch.Tensor) -> torch.Tensor:
    y = x.float()
    stride = 1
    while stride < 128:
        pairs = y.view(-1, 128 // (2 * stride), 2, stride)
        a, b = pairs[:, :, 0, :], pairs[:, :, 1, :]
        y = torch.stack((a + b, a - b), dim=2).reshape(-1, 128)
        stride *= 2
    return (y * INV_SQRT_128).to(torch.bfloat16).float()


@pytest.mark.parametrize("n_rows", [1, 31, 32, 33, 1000])
@pytest.mark.parametrize("magnitude", [1e-3, 1.0, 300.0])
def test_fwht128_quant_fp8_matches_reference(n_rows, magnitude):
    from vllm.models.glm5next.nvidia.ops.kpool_compress import fwht128_quant_fp8

    torch.manual_seed(n_rows)
    q = (torch.randn(n_rows, 128, device="cuda") * magnitude).to(torch.bfloat16)
    q_fp8, scale = fwht128_quant_fp8(q)
    assert q_fp8.dtype == torch.float8_e4m3fn and scale.shape == (n_rows, 1)

    y = _fwht128_ref(q)
    absmax = y.abs().amax(dim=1, keepdim=True).clamp_min(1e-4)
    # Power-of-two scale that brings absmax into (224, 448].
    assert torch.equal(torch.exp2(torch.log2(scale).round()), scale)
    assert torch.all(absmax / scale <= 448.0)
    assert torch.all(absmax / scale > 224.0)

    ref = (y / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    assert torch.equal(q_fp8.view(torch.uint8), ref.view(torch.uint8))
