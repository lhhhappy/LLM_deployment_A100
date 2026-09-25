# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang mHC pre/post kernels at GLM-5.3-Flash shapes vs the torch reference.

GLM-5.3-Flash runs ``MHCPreOp``/``MHCPostOp`` with hidden 4096, 4 residual
streams and 20 Sinkhorn iterations. On GPUs without DeepGEMM (A100) the pre
block's projection takes the TileLang fp32 GEMM path, so this pins that path's
numerics on every architecture the TileLang kernels support.
"""

import pytest
import torch

from vllm.model_executor.layers.mhc import HAS_TILELANG_MHC, MHCPostOp, MHCPreOp
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda() and HAS_TILELANG_MHC),
    reason="CUDA with TileLang mHC required",
)

HIDDEN = 4096
HC = 4
MIX = (2 + HC) * HC
GLM_EPS = dict(
    rms_eps=1e-5,
    hc_pre_eps=1e-6,
    hc_sinkhorn_eps=1e-6,
    hc_post_mult_value=2.0,
    sinkhorn_repeat=20,
)


def _pre_inputs(num_tokens: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    residual = torch.randn(num_tokens, HC, HIDDEN, generator=g, device="cuda").to(
        torch.bfloat16
    )
    fn = (
        torch.randn(MIX, HC * HIDDEN, generator=g, device="cuda")
        * (HC * HIDDEN) ** -0.5
    )
    hc_scale = torch.rand(3, generator=g, device="cuda") + 0.5
    hc_base = torch.randn(MIX, generator=g, device="cuda") * 0.1
    return residual, fn, hc_scale, hc_base


@pytest.mark.parametrize("num_tokens", [1, 7, 128, 2048])
def test_mhc_pre_matches_torch(num_tokens, default_vllm_config):
    op = MHCPreOp()
    args = _pre_inputs(num_tokens, seed=num_tokens)
    post, comb, layer_input = op.forward_cuda(*args, **GLM_EPS)
    post_ref, comb_ref, layer_input_ref = op.forward_native(*args, **GLM_EPS)
    torch.testing.assert_close(post, post_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(comb, comb_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(
        layer_input.float(), layer_input_ref.float(), atol=2e-2, rtol=2e-2
    )


@pytest.mark.parametrize("num_tokens", [1, 7, 128, 2048])
def test_mhc_post_matches_torch(num_tokens, default_vllm_config):
    op = MHCPostOp()
    g = torch.Generator(device="cuda").manual_seed(num_tokens)
    x = torch.randn(num_tokens, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    residual = torch.randn(num_tokens, HC, HIDDEN, generator=g, device="cuda").to(
        torch.bfloat16
    )
    post = torch.rand(num_tokens, HC, 1, generator=g, device="cuda") * 2
    comb = torch.softmax(
        torch.randn(num_tokens, HC, HC, generator=g, device="cuda"), -1
    )
    out = op.forward_cuda(x, residual, post, comb)
    ref = op.forward_native(x, residual, post, comb)
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)
