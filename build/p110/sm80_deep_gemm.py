# [ax] 110: DSA indexer on sm80 (A100). DeepGEMM's MQA-logits kernels are Hopper+ only and raise
# "Unsupported architecture" at CUDA-graph capture on A100, so the stock base cannot start there.
# This shim keeps every DeepGEMM attribute except the three indexer entry points, which get
# torch implementations with the same contracts (fp8 inputs, fp32 accumulation per head).
# Enabled automatically when the device major capability is < 9; set SGLANG_AX_SM80_INDEXER=0 to
# force the original module.
import os

import torch
import torch.nn.functional as F

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


def fp8_mqa_logits(q, kv, weights, ks, ke, clean_logits=False, max_seqlen_k=0):
    """q [nq,H,D] fp8, kv=(k [nk,D] fp8, k_scale [nk] f32), weights [nq,H] f32, ks/ke [nq] int32.
    Returns [nq, nk] f32: sum_h relu(q_h . k) * w_h * k_scale; -inf outside [ks, ke) if clean."""
    assert max_seqlen_k == 0, "compressed-width mode not used on the GLM path"
    k, k_scale = kv
    nq, H, _ = q.shape
    nk = k.shape[0]
    out = torch.zeros((nq, nk), dtype=torch.float32, device=q.device)
    if nq == 0 or nk == 0:
        return out
    kT = k.to(torch.bfloat16).t().contiguous()
    qb = q.to(torch.bfloat16)
    w = weights.to(torch.float32)
    for h in range(H):
        s = torch.mm(qb[:, h, :], kT).float()          # fp32 accumulate
        out.add_(torch.relu_(s).mul_(w[:, h : h + 1]))
    out.mul_(k_scale.to(torch.float32).view(1, nk))
    if clean_logits:
        pos = torch.arange(nk, device=q.device).view(1, nk)
        bad = (pos < ks.view(nq, 1)) | (pos >= ke.view(nq, 1))
        out.masked_fill_(bad, float("-inf"))
    return out


def fp8_paged_mqa_logits(q, kv_cache, weights, context_lens, block_table, schedule_meta,
                         max_context_len, clean_logits=False, indices=None):
    """q [B,N,H,D] (or [R,H,D]) fp8; kv_cache [blocks, 64, 1, D+4] bytes (D fp8 values + f32 scale);
    weights [B*N, H]; context_lens [B] or [B,N]; block_table [B, P]. Returns [B*N, max_context_len]
    f32 (positions >= context length are 0; callers mask with lengths). CUDA-graph safe: no host sync."""
    assert indices is None
    if q.dim() == 3:
        q = q.unsqueeze(1)
    B, N, H, D = q.shape
    R = B * N
    blk = kv_cache.shape[1]
    P = block_table.shape[1]
    width = P * blk
    ctx = context_lens.reshape(B, -1).to(torch.int32)
    ctx = ctx.expand(B, N) if ctx.shape[1] == 1 else ctx
    ctx = ctx.reshape(R)
    bt = block_table.repeat_interleave(N, dim=0) if N > 1 else block_table
    qr = q.reshape(R, H, D).to(torch.bfloat16)
    w = weights.reshape(R, H).to(torch.float32)
    flat = kv_cache.view(torch.uint8).reshape(kv_cache.shape[0], blk * (D + 4))
    vals_off = blk * D
    out = torch.zeros((R, max_context_len), dtype=torch.float32, device=q.device)
    rows_per = max(1, _CHUNK_BYTES // max(1, width * D * 2))
    pos = torch.arange(width, device=q.device).view(1, width)
    use = min(width, max_context_len)
    for r0 in range(0, R, rows_per):
        r1 = min(R, r0 + rows_per)
        g = flat[bt[r0:r1].clamp(min=0)]                                  # [r, P, blk*(D+4)]
        kvals = g[..., :vals_off].contiguous().view(_FP8).to(torch.bfloat16).reshape(r1 - r0, width, D)
        kscale = g[..., vals_off:].contiguous().view(torch.float32).reshape(r1 - r0, width)
        s = torch.bmm(kvals, qr[r0:r1].transpose(1, 2)).float()            # [r, width, H]
        s = (F.relu(s) * w[r0:r1].unsqueeze(1)).sum(dim=2) * kscale        # [r, width]
        s = s.masked_fill(pos >= ctx[r0:r1].view(-1, 1), 0.0)
        out[r0:r1, :use] = s[:, :use]
    return out


def get_paged_mqa_logits_metadata(context_lens, block_kv, num_sms, indices=None):
    # Scheduling metadata for the DeepGEMM kernel; the torch path ignores it.
    return torch.zeros((max(1, int(num_sms)) + 1, 2), dtype=torch.int32, device=context_lens.device)


_OVERRIDES = {
    "fp8_mqa_logits": fp8_mqa_logits,
    "fp8_paged_mqa_logits": fp8_paged_mqa_logits,
    "get_paged_mqa_logits_metadata": get_paged_mqa_logits_metadata,
    "get_num_sms": lambda: torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count,
}


class _Sm80DeepGemm:
    """Proxy: on sm80 the indexer entry points resolve to the torch versions; everything else (and
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
