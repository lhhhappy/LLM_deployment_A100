"""112: fused A100 DSA logits. Inputs are passed as bytes, never fp8 pointers.

The 110 oracle rounds each head's GEMM result to bf16 BEFORE relu/weighting.
Keep that rounding, and disable FMA in the epilogue. No tensor data reaches Python.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _e4m3_to_bf16(x):
    u = x.to(tl.int32)
    mag = u & 127
    exponent = mag >> 3
    mantissa = mag & 7
    normal = ((exponent + 120) << 23) | (mantissa << 20)
    subnormal = (mantissa.to(tl.float32) * (1.0 / 512.0)).to(tl.int32, bitcast=True)
    bits = tl.where(exponent == 0, subnormal, normal)
    bits = tl.where(mag == 127, 0x7FC00000, bits) | ((u & 128) << 24)
    return bits.to(tl.float32, bitcast=True).to(tl.bfloat16)


@triton.jit
def _paged(Q, K, W, C, BT, O,
           N: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
           P: tl.constexpr, S: tl.constexpr, PAGE: tl.constexpr,
           QB: tl.constexpr, QN: tl.constexpr, QH: tl.constexpr, QD: tl.constexpr,
           WB: tl.constexpr, WH: tl.constexpr, CB: tl.constexpr, CN: tl.constexpr,
           TB: tl.constexpr, TP: tl.constexpr, KB: tl.constexpr,
           HH: tl.constexpr, DD: tl.constexpr):
    row = tl.program_id(0)
    page = tl.program_id(1)
    b, n = row // N, row % N
    tok = tl.arange(0, PAGE)
    pos = page * PAGE + tok
    ctx = tl.load(C + b * CB + n * CN).to(tl.int32)
    # Launch bounds depend on shape only. Empty pages do no KV/query work;
    # every output is still written, including on a shrinking graph replay.
    result = tl.full((PAGE,), 0, tl.float32)
    if (page < P) & (page * PAGE < ctx):
        physical = tl.maximum(tl.load(BT + b * TB + page * TP), 0).to(tl.int64)
        ds = tl.arange(0, DD)
        hs = tl.arange(0, HH)
        kval = _e4m3_to_bf16(tl.load(
            K + physical * KB + tok[:, None] * D + ds[None, :],
            mask=ds[None, :] < D, other=0))
        qval = _e4m3_to_bf16(tl.load(
            Q + b * QB + n * QN + ds[:, None] * QD + hs[None, :] * QH,
            mask=(ds[:, None] < D) & (hs[None, :] < H), other=0))
        dots = tl.dot(kval, qval).to(tl.bfloat16).to(tl.float32)
        weight = tl.load(W + row * WB + hs * WH, hs < H, other=0).to(tl.float32)
        scale_ptr = (K + physical * KB + PAGE * D).to(tl.pointer_type(tl.float32))
        scale = tl.load(scale_ptr + tok)
        result = tl.sum(tl.maximum(dots, 0.0) * weight[None, :], 1) * scale
        result = tl.where(pos < ctx, result, 0.0)
    tl.store(O + row.to(tl.int64) * S + pos, result, pos < S)


@triton.jit
def _ragged(Q, K, SC, W, KS, KE, O,
            NQ: tl.constexpr, NK: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
            QQ: tl.constexpr, QH: tl.constexpr, QD: tl.constexpr,
            KK: tl.constexpr, KD: tl.constexpr, SS: tl.constexpr,
            WQ: tl.constexpr, WH: tl.constexpr, KSS: tl.constexpr, KES: tl.constexpr,
            CLEAN: tl.constexpr, BQ: tl.constexpr, BK: tl.constexpr,
            HH: tl.constexpr, DD: tl.constexpr):
    qi = tl.program_id(0) * BQ + tl.arange(0, BQ)
    ki = tl.program_id(1) * BK + tl.arange(0, BK)
    if CLEAN:
        lo = tl.load(KS + qi * KSS, qi < NQ, other=0)
        hi = tl.load(KE + qi * KES, qi < NQ, other=0)
        active = (qi < NQ) & (lo < hi) & (lo < (tl.program_id(1) + 1) * BK) & (hi > tl.program_id(1) * BK)
        compute = tl.sum(active.to(tl.int32), 0) > 0
    else:
        compute = True
    result = tl.full((BK, BQ), float('-inf'), tl.float32)
    if compute:
        ds = tl.arange(0, DD)
        qh = tl.arange(0, BQ * HH)
        qrow = tl.program_id(0) * BQ + qh // HH
        head = qh % HH
        kval = _e4m3_to_bf16(tl.load(
            K + ki[:, None] * KK + ds[None, :] * KD,
            (ki[:, None] < NK) & (ds[None, :] < D), other=0))
        qval = _e4m3_to_bf16(tl.load(
            Q + qrow[None, :] * QQ + head[None, :] * QH + ds[:, None] * QD,
            (qrow[None, :] < NQ) & (head[None, :] < H) & (ds[:, None] < D), other=0))
        dots = tl.dot(kval, qval).to(tl.bfloat16).to(tl.float32)
        weight = tl.load(W + qrow * WQ + head * WH,
                         (qrow < NQ) & (head < H), other=0).to(tl.float32)
        weighted = tl.maximum(dots, 0.0) * weight[None, :]
        result = tl.sum(tl.reshape(weighted, (BK, BQ, HH)), 2)
        scale = tl.load(SC + ki * SS, ki < NK, other=0).to(tl.float32)
        result = result * scale[:, None]
        if CLEAN:
            result = tl.where((ki[:, None] >= lo[None, :]) & (ki[:, None] < hi[None, :]),
                              result, float('-inf'))
    tl.store(O + qi[None, :].to(tl.int64) * NK + ki[:, None], result,
             (qi[None, :] < NQ) & (ki[:, None] < NK))


def fp8_paged_mqa_logits(q, kv_cache, weights, context_lens, block_table, schedule_meta,
                         max_context_len, clean_logits=False, indices=None):
    """110 contract: [B*N,S] fp32; clean is ignored; negative pages clamp to zero."""
    assert indices is None
    if q.dim() == 3:
        q = q.unsqueeze(1)
    B, N, H, D = q.shape
    page = kv_cache.shape[1]
    assert page == 64 and D == 128
    assert q.dtype == torch.float8_e4m3fn
    ctx = context_lens.reshape(B, -1)
    assert ctx.shape[1] in (1, N)
    weights = weights.reshape(B * N, H)
    # Same packed layout as 110: all PAGE*D fp8 bytes, THEN PAGE f32 scales.
    cache = kv_cache.view(torch.uint8).reshape(kv_cache.shape[0], page * (D + 4))
    out = torch.empty((B * N, max_context_len), device=q.device, dtype=torch.float32)
    if B * N == 0 or max_context_len == 0:
        return out
    with torch.cuda.device(q.device):
        _paged[(B * N, triton.cdiv(max_context_len, page))](
            q.view(torch.uint8), cache, weights, ctx, block_table, out,
            N, H, D, block_table.shape[1], max_context_len, page,
            *q.stride(), *weights.stride(), ctx.stride(0),
            ctx.stride(1) if ctx.shape[1] == N else 0,
            *block_table.stride(), cache.stride(0),
            max(16, triton.next_power_of_2(H)), triton.next_power_of_2(D),
            num_warps=4, enable_fp_fusion=False)
    return out


def fp8_mqa_logits(q, kv, weights, ks, ke, clean_logits=False, max_seqlen_k=0):
    """110 contract: clean=True masks to -inf; clean=False computes ALL columns."""
    assert max_seqlen_k == 0, "compressed-width mode not used on the GLM path"
    k, scale = kv
    nq, H, D = q.shape
    nk = k.shape[0]
    assert D == 128 and q.dtype == k.dtype == torch.float8_e4m3fn
    out = torch.empty((nq, nk), device=q.device, dtype=torch.float32)
    if nq == 0 or nk == 0:
        return out
    scale, ks, ke = scale.reshape(nk), ks.reshape(nq), ke.reshape(nq)
    hh = max(16, triton.next_power_of_2(H))
    bq = max(1, 128 // hh)
    with torch.cuda.device(q.device):
        _ragged[(triton.cdiv(nq, bq), triton.cdiv(nk, 64))](
            q.view(torch.uint8), k.view(torch.uint8), scale, weights, ks, ke, out,
            nq, nk, H, D, *q.stride(), *k.stride(), scale.stride(0),
            *weights.stride(), ks.stride(0), ke.stride(0), clean_logits,
            bq, 64, hh, triton.next_power_of_2(D), num_warps=8, enable_fp_fusion=False)
    return out
