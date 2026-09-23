# [ax] 110: DSA indexer on sm80 (A100). DeepGEMM's MQA-logits kernels are Hopper+ only and raise
# "Unsupported architecture" at CUDA-graph capture on A100, so the stock base cannot start there.
# This shim keeps every DeepGEMM attribute except the three indexer entry points, which get
# fused implementations with the same contracts (fp8 inputs, fp32 accumulation per head).
# Enabled automatically when the device major capability is < 9; set SGLANG_AX_SM80_INDEXER=0 to
# force the original module.
import os

import torch

_FP8 = torch.float8_e4m3fn
_CHUNK_BYTES = int(os.environ.get("SGLANG_AX_SM80_INDEXER_CHUNK_BYTES", str(1 << 30)))


_NEED = None


def _need_shim() -> bool:
    # Decided on first use (not at import: scheduler subprocesses must pick their device first).
    global _NEED
    if _NEED is None:
        _NEED = (os.environ.get("SGLANG_AX_SM80_INDEXER", "1") != "0" and torch.cuda.is_available()
                 and torch.cuda.get_device_capability()[0] < 9)
    return _NEED


# [ax] 112: fused software-fp8/bf16 kernels; preserve the 110 contracts.
from sglang.srt.layers.attention.dsa.sm80_indexer_kernels import (
    fp8_mqa_logits,
    fp8_paged_mqa_logits,
)


def get_paged_mqa_logits_metadata(context_lens, block_kv, num_sms, indices=None):
    # Scheduling metadata for the DeepGEMM kernel; the fused path ignores it.
    return torch.zeros((max(1, int(num_sms)) + 1, 2), dtype=torch.int32, device=context_lens.device)


_OVERRIDES = {
    "fp8_mqa_logits": fp8_mqa_logits,
    "fp8_paged_mqa_logits": fp8_paged_mqa_logits,
    "get_paged_mqa_logits_metadata": get_paged_mqa_logits_metadata,
    "get_num_sms": lambda: torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count,
}


class _Sm80DeepGemm:
    """Proxy: on sm80 the indexer entry points resolve to the fused versions; everything else (and
    everything on sm90+) resolves to the real module."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        if name in _OVERRIDES and _need_shim():
            return _OVERRIDES[name]
        if isinstance(self._real, Exception):
            raise self._real
        return getattr(self._real, name)


_INSTALLED = False


def _install_on_module(mod):
    """Patch the real deep_gemm module in place with lazy dispatchers, so every importer that calls
    deep_gemm.<fn> at call time (e.g. dsa/kpool_plan.py) reaches the sm80 path too."""
    global _INSTALLED
    if _INSTALLED or isinstance(mod, Exception):
        return
    for name, ours in _OVERRIDES.items():
        orig = getattr(mod, name, None)

        def dispatch(*a, _ours=ours, _orig=orig, **k):
            return _ours(*a, **k) if (_need_shim() or _orig is None) else _orig(*a, **k)

        setattr(mod, name, dispatch)
    _INSTALLED = True


def maybe_wrap(deep_gemm_module):
    _install_on_module(deep_gemm_module)
    return _Sm80DeepGemm(deep_gemm_module)
