"""Prototype replacement for tilelang_sparse_fwd (sparse_attention_fwd_kernel_v1, tail_dim = 0) on sm80.

Same inputs/outputs: q [M, H, D] bf16, kv [P, 1, D] bf16 (page_size 1 addressing), indices [M, K] int32 (-1 = masked,
any position), sm_scale; returns out [1, M, H, D] bf16 (and lse [1, M, H] fp32 in log2 units like the served kernel
when return_lse=True). Differences, both deliberate: fully masked rows return 0 (served: 0/0 = NaN), and indices
>= P are masked (served: zero-filled but still counted in the softmax with logit 0).
"""
import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634
KMETA = {}


@triton.jit
def _row_tiles(IDX, row, stride_im, K, KP2: tl.constexpr, BI: tl.constexpr):
    offs = tl.arange(0, KP2)
    ii = tl.load(IDX + row.to(tl.int64) * stride_im + offs, mask=offs < K, other=-1)
    last = tl.max(tl.where(ii >= 0, offs, -1), 0)
    return (last + BI) // BI  # tiles that contain at least the last valid entry


@triton.jit
def _sparse_fwd_tri(Q, KV, IDX, O, LSE, OP, LP,
                    M, P, K, sm_scale_log2,
                    stride_qm, stride_qh, stride_kvp, stride_im, stride_om, stride_oh,
                    H: tl.constexpr, BH: tl.constexpr, D: tl.constexpr, BI: tl.constexpr, KP2: tl.constexpr,
                    NSPLIT: tl.constexpr, RETURN_LSE: tl.constexpr, NUM_STAGES: tl.constexpr):
    row = tl.program_id(0)
    split = tl.program_id(1)
    offs_h = tl.arange(0, BH)
    offs_d = tl.arange(0, D)
    offs_i = tl.arange(0, BI)
    q = tl.load(Q + row.to(tl.int64) * stride_qm + offs_h[:, None] * stride_qh + offs_d[None, :],
                mask=offs_h[:, None] < H, other=0.0)
    n_tiles = _row_tiles(IDX, row, stride_im, K, KP2, BI)
    per = (n_tiles + NSPLIT - 1) // NSPLIT
    t0 = split * per
    t1 = tl.minimum(t0 + per, n_tiles)
    m_i = tl.full([BH], float("-inf"), tl.float32)
    l_i = tl.zeros([BH], tl.float32)
    acc = tl.zeros([BH, D], tl.float32)
    idx_row = IDX + row.to(tl.int64) * stride_im
    for t in tl.range(t0, t1, num_stages=NUM_STAGES):
        cols = t * BI + offs_i
        ii = tl.load(idx_row + cols, mask=cols < K, other=-1)
        valid = (ii >= 0) & (ii < P)
        kv = tl.load(KV + ii.to(tl.int64)[:, None] * stride_kvp + offs_d[None, :], mask=valid[:, None], other=0.0)
        s = tl.dot(q, tl.trans(kv)) * sm_scale_log2
        s = tl.where(valid[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_use)
        p = tl.exp2(s - m_use[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kv)
        m_i = m_new
    if NSPLIT == 1:
        l_safe = tl.where(l_i > 0, l_i, 1.0)
        out = acc / l_safe[:, None]
        tl.store(O + row.to(tl.int64) * stride_om + offs_h[:, None] * stride_oh + offs_d[None, :], out.to(tl.bfloat16),
                 mask=offs_h[:, None] < H)
        if RETURN_LSE:
            lse = tl.where(l_i > 0, tl.log2(l_safe) + m_i, float("-inf"))
            tl.store(LSE + row.to(tl.int64) * H + offs_h, lse, mask=offs_h < H)
    else:
        base = (row.to(tl.int64) * NSPLIT + split)
        tl.store(OP + base * (BH * D) + offs_h[:, None] * D + offs_d[None, :], acc)
        tl.store(LP + base * (2 * BH) + offs_h, m_i)
        tl.store(LP + base * (2 * BH) + BH + offs_h, l_i)


@triton.jit
def _sparse_fwd_tri_t(Q, KV, IDX, O, LSE, OP, LP,
                      M, P, K, sm_scale_log2,
                      stride_qm, stride_qh, stride_kvp, stride_im, stride_om, stride_oh,
                      H: tl.constexpr, D: tl.constexpr, BI: tl.constexpr, KP2: tl.constexpr,
                      NSPLIT: tl.constexpr, RETURN_LSE: tl.constexpr, NUM_STAGES: tl.constexpr):
    """transposed form: S^T = KV_tile . Q^T ([BI, H]) and O^T += KV_tile^T . P^T ([D, H]); H = 8 fills mma n8, no head padding."""
    row = tl.program_id(0)
    split = tl.program_id(1)
    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    offs_i = tl.arange(0, BI)
    qT = tl.load(Q + row.to(tl.int64) * stride_qm + offs_d[:, None] + offs_h[None, :] * stride_qh)  # [D, H]
    n_tiles = _row_tiles(IDX, row, stride_im, K, KP2, BI)
    per = (n_tiles + NSPLIT - 1) // NSPLIT
    t0 = split * per
    t1 = tl.minimum(t0 + per, n_tiles)
    m_i = tl.full([H], float("-inf"), tl.float32)
    l_i = tl.zeros([H], tl.float32)
    accT = tl.zeros([D, H], tl.float32)
    idx_row = IDX + row.to(tl.int64) * stride_im
    for t in tl.range(t0, t1, num_stages=NUM_STAGES):
        cols = t * BI + offs_i
        ii = tl.load(idx_row + cols, mask=cols < K, other=-1)
        valid = (ii >= 0) & (ii < P)
        kv = tl.load(KV + ii.to(tl.int64)[:, None] * stride_kvp + offs_d[None, :], mask=valid[:, None], other=0.0)
        sT = tl.dot(kv, qT) * sm_scale_log2  # [BI, H]
        sT = tl.where(valid[:, None], sT, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sT, 0))
        m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_use)
        p = tl.exp2(sT - m_use[None, :])
        l_i = l_i * alpha + tl.sum(p, 0)
        accT = accT * alpha[None, :] + tl.dot(tl.trans(kv), p.to(tl.bfloat16))
        m_i = m_new
    if NSPLIT == 1:
        l_safe = tl.where(l_i > 0, l_i, 1.0)
        outT = accT / l_safe[None, :]
        tl.store(O + row.to(tl.int64) * stride_om + offs_h[None, :] * stride_oh + offs_d[:, None], outT.to(tl.bfloat16))
        if RETURN_LSE:
            lse = tl.where(l_i > 0, tl.log2(l_safe) + m_i, float("-inf"))
            tl.store(LSE + row.to(tl.int64) * H + offs_h, lse)
    else:
        base = (row.to(tl.int64) * NSPLIT + split)
        tl.store(OP + base * (H * D) + offs_h[None, :] * D + offs_d[:, None], accT)
        tl.store(LP + base * (2 * H) + offs_h, m_i)
        tl.store(LP + base * (2 * H) + H + offs_h, l_i)


@triton.jit
def _combine(OP, LP, O, LSE, stride_om, stride_oh,
             H: tl.constexpr, BH: tl.constexpr, D: tl.constexpr, NSPLIT: tl.constexpr, RETURN_LSE: tl.constexpr):
    row = tl.program_id(0)
    offs_h = tl.arange(0, BH)
    offs_d = tl.arange(0, D)
    m_g = tl.full([BH], float("-inf"), tl.float32)
    for s in range(NSPLIT):
        base = row.to(tl.int64) * NSPLIT + s
        m_g = tl.maximum(m_g, tl.load(LP + base * (2 * BH) + offs_h))
    m_use = tl.where(m_g == float("-inf"), 0.0, m_g)
    l_g = tl.zeros([BH], tl.float32)
    acc = tl.zeros([BH, D], tl.float32)
    for s in range(NSPLIT):
        base = row.to(tl.int64) * NSPLIT + s
        m_s = tl.load(LP + base * (2 * BH) + offs_h)
        l_s = tl.load(LP + base * (2 * BH) + BH + offs_h)
        w = tl.where(l_s > 0, tl.exp2(m_s - m_use), 0.0)
        l_g += l_s * w
        acc += tl.load(OP + base * (BH * D) + offs_h[:, None] * D + offs_d[None, :]) * w[:, None]
    l_safe = tl.where(l_g > 0, l_g, 1.0)
    tl.store(O + row.to(tl.int64) * stride_om + offs_h[:, None] * stride_oh + offs_d[None, :],
             (acc / l_safe[:, None]).to(tl.bfloat16), mask=offs_h[:, None] < H)
    if RETURN_LSE:
        tl.store(LSE + row.to(tl.int64) * H + offs_h, tl.where(l_g > 0, tl.log2(l_safe) + m_g, float("-inf")),
                 mask=offs_h < H)


def _pick_split(M, n_sm=108):
    if M >= 4 * n_sm:
        return 1
    s = 1
    while M * s < 2 * n_sm and s < 16:
        s *= 2
    return s


def triton_sparse_fwd(q, kv, indices, sm_scale, d_v=512, return_lse=False, BI=64, num_warps=4, num_stages=2, nsplit=None,
                      transposed=False):
    if indices.dim() == 3:
        indices = indices.squeeze(1)
    assert q.dim() == 3 and kv.dim() == 3 and kv.shape[1] == 1 and indices.dim() == 2
    M, H, D = q.shape
    assert D == d_v and kv.shape[2] == D and indices.shape[0] == M
    assert q.stride(2) == 1 and kv.stride(2) == 1 and indices.stride(1) == 1
    K = indices.shape[1]
    BH = H if transposed else max(16, triton.next_power_of_2(H))
    if transposed:
        assert H in (8, 16, 32, 64), H
    KP2 = triton.next_power_of_2(K)
    out = torch.empty((1, M, H, D), dtype=q.dtype, device=q.device)
    lse = torch.empty((1, M, H), dtype=torch.float32, device=q.device) if return_lse else out
    ns = _pick_split(M) if nsplit is None else nsplit
    if ns > 1:
        op = torch.empty((M, ns, BH, D), dtype=torch.float32, device=q.device)
        lp = torch.empty((M, ns, 2, BH), dtype=torch.float32, device=q.device)
    else:
        op = lp = out
    o2 = out.view(M, H, D)
    common = (q, kv, indices, o2, lse, op, lp,
              M, kv.shape[0], K, sm_scale * LOG2E,
              q.stride(0), q.stride(1), kv.stride(0), indices.stride(0), o2.stride(0), o2.stride(1))
    if transposed:
        ck = _sparse_fwd_tri_t[(M, ns)](*common, H=H, D=D, BI=BI, KP2=KP2, NSPLIT=ns, RETURN_LSE=return_lse,
                                        NUM_STAGES=num_stages, num_warps=num_warps, num_stages=num_stages)
    else:
        ck = _sparse_fwd_tri[(M, ns)](*common, H=H, BH=BH, D=D, BI=BI, KP2=KP2, NSPLIT=ns, RETURN_LSE=return_lse,
                                      NUM_STAGES=num_stages, num_warps=num_warps, num_stages=num_stages)
    KMETA[(transposed, BI, num_warps, num_stages, ns)] = dict(n_regs=getattr(ck, "n_regs", None), n_spills=getattr(ck, "n_spills", None),
                                                  shared=getattr(getattr(ck, "metadata", None), "shared", None))
    if ns > 1:
        _combine[(M,)](op, lp, o2, lse, o2.stride(0), o2.stride(1), H=H, BH=BH, D=D, NSPLIT=ns,
                       RETURN_LSE=return_lse, num_warps=4)
    if return_lse:
        return out, lse
    return out


# variants for the bench (name -> callable(q, kv, idx, sm_scale))
def tri_b64_w4_s2(q, kv, idx, sm_scale):
    return triton_sparse_fwd(q, kv, idx, sm_scale, BI=64, num_warps=4, num_stages=2)


def tri_b32_w4_s3(q, kv, idx, sm_scale):
    return triton_sparse_fwd(q, kv, idx, sm_scale, BI=32, num_warps=4, num_stages=3)


def tri_b32_w4_s4(q, kv, idx, sm_scale):
    return triton_sparse_fwd(q, kv, idx, sm_scale, BI=32, num_warps=4, num_stages=4)


def tri_b64_w8_s2(q, kv, idx, sm_scale):
    return triton_sparse_fwd(q, kv, idx, sm_scale, BI=64, num_warps=8, num_stages=2)


def tri_b16_w4_s4(q, kv, idx, sm_scale):
    return triton_sparse_fwd(q, kv, idx, sm_scale, BI=16, num_warps=4, num_stages=4)


def tri_b32_w8_s3(q, kv, idx, sm_scale):
    return triton_sparse_fwd(q, kv, idx, sm_scale, BI=32, num_warps=8, num_stages=3)


def _mk(BI, w, st, tr):
    def f(q, kv, idx, sm_scale):
        return triton_sparse_fwd(q, kv, idx, sm_scale, BI=BI, num_warps=w, num_stages=st, transposed=tr)
    return f


for _BI in (16, 32, 64):
    for _w in (2, 4, 8):
        for _st in (2, 3, 4):
            globals()["trt_b%d_w%d_s%d" % (_BI, _w, _st)] = _mk(_BI, _w, _st, True)


def _mkp(BI, w, st, ns):
    def f(q, kv, idx, sm_scale):
        return triton_sparse_fwd(q, kv, idx, sm_scale, BI=BI, num_warps=w, num_stages=st, nsplit=ns)
    return f


for _ns in (1, 2, 4, 8, 16, 32):
    globals()["tri_b32_w4_s3_ns%d" % _ns] = _mkp(32, 4, 3, _ns)
    globals()["tri_b64_w4_s2_ns%d" % _ns] = _mkp(64, 4, 2, _ns)


def dsa_sparse_fwd(q, kv, indices, sm_scale, d_v=512, return_lse=False):
    """prototype default: padded heads, 32-row tiles, 4 warps; split-K + combine when M is small."""
    return triton_sparse_fwd(q, kv, indices, sm_scale, d_v=d_v, return_lse=return_lse, BI=32, num_warps=4, num_stages=3)


# ---- served TileLang kernel (unmodified factory) with other launch configs, for the diagnosis
def _mktl(BI, stages, threads, stage_output):
    def f(q, kv, idx, sm_scale):
        from sglang.kernels.ops.attention.dsa.tilelang_kernel import sparse_attention_fwd_kernel_v1

        pad = (-idx.shape[-1]) % 64
        if pad:
            idx = torch.cat((idx, idx.new_full((*idx.shape[:-1], pad), -1)), dim=-1)
        M, H, D = q.shape
        k = sparse_attention_fwd_kernel_v1(H, D, 0, idx.shape[-1], sm_scale=sm_scale, block_I=BI, num_stages=stages,
                                           threads=threads, stage_output=stage_output)
        lse = torch.empty((1, M, H), dtype=torch.float32, device=q.device)
        return k(q.unsqueeze(0), kv.unsqueeze(0), idx.unsqueeze(1).unsqueeze(0), lse)
    return f


for _cfg in [(64, 2, 256, True), (64, 2, 256, False), (32, 2, 128, False), (32, 3, 128, False), (32, 2, 256, False),
             (16, 2, 128, False), (16, 3, 128, False), (64, 1, 256, False), (32, 4, 128, False)]:
    globals()["tl_b%d_s%d_t%d%s" % (_cfg[0], _cfg[1], _cfg[2], "" if _cfg[3] else "_noO")] = _mktl(*_cfg)


# ---- diagnostic: gather-only floor (same index walk and masked 16B loads, no attention math; output is NOT attention)
@triton.jit
def _gather_floor(KV, IDX, O, P, K, stride_kvp, stride_im, D: tl.constexpr, BI: tl.constexpr, KP2: tl.constexpr,
                  NUM_STAGES: tl.constexpr):
    row = tl.program_id(0)
    offs_d = tl.arange(0, D)
    offs_i = tl.arange(0, BI)
    n_tiles = _row_tiles(IDX, row, stride_im, K, KP2, BI)
    acc = tl.zeros([BI, D], tl.float32)
    idx_row = IDX + row.to(tl.int64) * stride_im
    for t in tl.range(0, n_tiles, num_stages=NUM_STAGES):
        cols = t * BI + offs_i
        ii = tl.load(idx_row + cols, mask=cols < K, other=-1)
        valid = (ii >= 0) & (ii < P)
        kv = tl.load(KV + ii.to(tl.int64)[:, None] * stride_kvp + offs_d[None, :], mask=valid[:, None], other=0.0)
        acc += kv.to(tl.float32)
    tl.store(O + row.to(tl.int64) * D + offs_d, tl.sum(acc, 0).to(tl.bfloat16))


def _mkg(BI, w, st):
    def f(q, kv, idx, sm_scale):
        M = q.shape[0]
        out = torch.zeros((1, M, q.shape[1], q.shape[2]), dtype=q.dtype, device=q.device)
        _gather_floor[(M,)](kv, idx, out, kv.shape[0], idx.shape[1], kv.stride(0), idx.stride(0), D=q.shape[2], BI=BI,
                            KP2=triton.next_power_of_2(idx.shape[1]), NUM_STAGES=st, num_warps=w, num_stages=st)
        return out
    return f


gather_b32_w4_s3 = _mkg(32, 4, 3)
gather_b16_w4_s4 = _mkg(16, 4, 4)
gather_b64_w8_s2 = _mkg(64, 8, 2)


@triton.jit
def _sparse_fwd_tri2(Q, KV, IDX, O, LSE, OP, LP,
                     M, P, K, sm_scale_log2,
                     stride_qm, stride_qh, stride_kvp, stride_im, stride_om, stride_oh,
                     H: tl.constexpr, BH: tl.constexpr, D: tl.constexpr, BI: tl.constexpr, KP2: tl.constexpr,
                     NSPLIT: tl.constexpr, RETURN_LSE: tl.constexpr, NUM_STAGES: tl.constexpr, EVICT: tl.constexpr):
    """as _sparse_fwd_tri, but the next tile's indices are prefetched into a loop-carried register (manual stage)."""
    row = tl.program_id(0)
    split = tl.program_id(1)
    offs_h = tl.arange(0, BH)
    offs_d = tl.arange(0, D)
    offs_i = tl.arange(0, BI)
    q = tl.load(Q + row.to(tl.int64) * stride_qm + offs_h[:, None] * stride_qh + offs_d[None, :],
                mask=offs_h[:, None] < H, other=0.0)
    n_tiles = _row_tiles(IDX, row, stride_im, K, KP2, BI)
    per = (n_tiles + NSPLIT - 1) // NSPLIT
    t0 = split * per
    t1 = tl.minimum(t0 + per, n_tiles)
    m_i = tl.full([BH], float("-inf"), tl.float32)
    l_i = tl.zeros([BH], tl.float32)
    acc = tl.zeros([BH, D], tl.float32)
    idx_row = IDX + row.to(tl.int64) * stride_im
    cols = t0 * BI + offs_i
    ii_n = tl.load(idx_row + cols, mask=cols < K, other=-1)
    for t in tl.range(t0, t1, num_stages=NUM_STAGES):
        ii = ii_n
        cols_n = (t + 1) * BI + offs_i
        ii_n = tl.load(idx_row + cols_n, mask=cols_n < K, other=-1)
        valid = (ii >= 0) & (ii < P)
        if EVICT:
            kv = tl.load(KV + ii.to(tl.int64)[:, None] * stride_kvp + offs_d[None, :], mask=valid[:, None], other=0.0,
                         cache_modifier=".cg", eviction_policy="evict_last")
        else:
            kv = tl.load(KV + ii.to(tl.int64)[:, None] * stride_kvp + offs_d[None, :], mask=valid[:, None], other=0.0)
        s = tl.dot(q, tl.trans(kv)) * sm_scale_log2
        s = tl.where(valid[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_use)
        p = tl.exp2(s - m_use[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kv)
        m_i = m_new
    l_safe = tl.where(l_i > 0, l_i, 1.0)
    out = acc / l_safe[:, None]
    tl.store(O + row.to(tl.int64) * stride_om + offs_h[:, None] * stride_oh + offs_d[None, :], out.to(tl.bfloat16),
             mask=offs_h[:, None] < H)


def _mk2(BI, w, st, ev):
    def f(q, kv, idx, sm_scale):
        M, H, D = q.shape
        out = torch.empty((1, M, H, D), dtype=q.dtype, device=q.device)
        o2 = out.view(M, H, D)
        _sparse_fwd_tri2[(M, 1)](q, kv, idx, o2, out, out, out, M, kv.shape[0], idx.shape[1], sm_scale * LOG2E,
                                 q.stride(0), q.stride(1), kv.stride(0), idx.stride(0), o2.stride(0), o2.stride(1),
                                 H=H, BH=16, D=D, BI=BI, KP2=triton.next_power_of_2(idx.shape[1]), NSPLIT=1,
                                 RETURN_LSE=False, NUM_STAGES=st, EVICT=ev, num_warps=w, num_stages=st)
        return out
    return f


for _BI, _w, _st, _ev in [(32, 4, 2, False), (32, 4, 3, False), (32, 4, 3, True), (64, 8, 2, False), (64, 4, 3, False),
                          (16, 4, 3, False), (32, 8, 3, False)]:
    globals()["tri2_b%d_w%d_s%d%s" % (_BI, _w, _st, "_ev" if _ev else "")] = _mk2(_BI, _w, _st, _ev)


@triton.jit
def _sparse_fwd_tri3(Q, KV, IDX, O, P, K, sm_scale_log2,
                     stride_qm, stride_qh, stride_kvp, stride_im, stride_om, stride_oh,
                     H: tl.constexpr, BH: tl.constexpr, BI: tl.constexpr, KP2: tl.constexpr, NUM_STAGES: tl.constexpr):
    """D = 512 handled as 4 column chunks of 128 (separate q/kv/acc tensors) to let the compiler keep q in registers."""
    DC: tl.constexpr = 128
    row = tl.program_id(0)
    offs_h = tl.arange(0, BH)
    offs_c = tl.arange(0, DC)
    offs_i = tl.arange(0, BI)
    qb = Q + row.to(tl.int64) * stride_qm + offs_h[:, None] * stride_qh + offs_c[None, :]
    hm = offs_h[:, None] < H
    q0 = tl.load(qb, mask=hm, other=0.0)
    q1 = tl.load(qb + DC, mask=hm, other=0.0)
    q2 = tl.load(qb + 2 * DC, mask=hm, other=0.0)
    q3 = tl.load(qb + 3 * DC, mask=hm, other=0.0)
    n_tiles = _row_tiles(IDX, row, stride_im, K, KP2, BI)
    m_i = tl.full([BH], float("-inf"), tl.float32)
    l_i = tl.zeros([BH], tl.float32)
    a0 = tl.zeros([BH, DC], tl.float32)
    a1 = tl.zeros([BH, DC], tl.float32)
    a2 = tl.zeros([BH, DC], tl.float32)
    a3 = tl.zeros([BH, DC], tl.float32)
    idx_row = IDX + row.to(tl.int64) * stride_im
    for t in tl.range(0, n_tiles, num_stages=NUM_STAGES):
        cols = t * BI + offs_i
        ii = tl.load(idx_row + cols, mask=cols < K, other=-1)
        valid = (ii >= 0) & (ii < P)
        kb = KV + ii.to(tl.int64)[:, None] * stride_kvp + offs_c[None, :]
        vm = valid[:, None]
        k0 = tl.load(kb, mask=vm, other=0.0)
        k1 = tl.load(kb + DC, mask=vm, other=0.0)
        k2 = tl.load(kb + 2 * DC, mask=vm, other=0.0)
        k3 = tl.load(kb + 3 * DC, mask=vm, other=0.0)
        s = tl.dot(q0, tl.trans(k0))
        s = tl.dot(q1, tl.trans(k1), s)
        s = tl.dot(q2, tl.trans(k2), s)
        s = tl.dot(q3, tl.trans(k3), s)
        s = tl.where(valid[None, :], s * sm_scale_log2, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_use)
        p = tl.exp2(s - m_use[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        pb = p.to(tl.bfloat16)
        a0 = tl.dot(pb, k0, a0 * alpha[:, None])
        a1 = tl.dot(pb, k1, a1 * alpha[:, None])
        a2 = tl.dot(pb, k2, a2 * alpha[:, None])
        a3 = tl.dot(pb, k3, a3 * alpha[:, None])
        m_i = m_new
    inv = 1.0 / tl.where(l_i > 0, l_i, 1.0)
    ob = O + row.to(tl.int64) * stride_om + offs_h[:, None] * stride_oh + offs_c[None, :]
    tl.store(ob, (a0 * inv[:, None]).to(tl.bfloat16), mask=hm)
    tl.store(ob + DC, (a1 * inv[:, None]).to(tl.bfloat16), mask=hm)
    tl.store(ob + 2 * DC, (a2 * inv[:, None]).to(tl.bfloat16), mask=hm)
    tl.store(ob + 3 * DC, (a3 * inv[:, None]).to(tl.bfloat16), mask=hm)


def _mk3(BI, w, st):
    def f(q, kv, idx, sm_scale):
        M, H, D = q.shape
        assert D == 512
        out = torch.empty((1, M, H, D), dtype=q.dtype, device=q.device)
        o2 = out.view(M, H, D)
        ck = _sparse_fwd_tri3[(M,)](q, kv, idx, o2, kv.shape[0], idx.shape[1], sm_scale * LOG2E,
                                    q.stride(0), q.stride(1), kv.stride(0), idx.stride(0), o2.stride(0), o2.stride(1),
                                    H=H, BH=16, BI=BI, KP2=triton.next_power_of_2(idx.shape[1]), NUM_STAGES=st,
                                    num_warps=w, num_stages=st)
        KMETA[("tri3", BI, w, st)] = dict(n_regs=getattr(ck, "n_regs", None), n_spills=getattr(ck, "n_spills", None),
                                          shared=getattr(getattr(ck, "metadata", None), "shared", None))
        return out
    return f


for _BI, _w, _st in [(32, 4, 2), (32, 4, 3), (64, 4, 2), (64, 8, 2), (16, 4, 3), (32, 8, 2)]:
    globals()["tri3_b%d_w%d_s%d" % (_BI, _w, _st)] = _mk3(_BI, _w, _st)
