"""DSA sparse-attention (MLA latent, sm80) kernel bench at the TP8 per-rank shape.

Reproduces the served call `tilelang_sparse_fwd` (sparse_attention_fwd_kernel_v1, tail_dim=0) as
dsa_backend._forward_tilelang issues it for GLM-5.3-Flash on A100:
  q  [M, 8, 512] bf16 (64 heads / TP8, absorbed latent q, qk_rope_head_dim = 0)
  kv [P, 1, 512] bf16 (one DSA layer's MLA pool, page_size 1 addressing, pages of 64 in the allocator)
  indices [M, 1, 2112] int32 (2048 pooled-group tokens + up to 3 kpool tail tokens, -1 padded to 64x)
Index rows are built like kpool_topk_transform.cuh: short rows = all history in order + tail + -1 suffix;
long rows = 512 selected groups of 4 consecutive tokens (arbitrary order) + tail + -1 suffix.

Usage (dev box, GPU 0 only): python bench_dsa_sparse.py --suite main --out res.jsonl
"""
import argparse
import json
import math
import os
import sys
import time

import torch

H_LOCAL = 8
D = 512
TOPK = 2048
KPOOL = 4
WIDTH = TOPK + KPOOL - 1  # 2051 columns from the indexer
WIDTH_PAD = WIDTH + (-WIDTH) % 64  # 2112 after _forward_tilelang's -1 padding
PAGE = 64
SM_SCALE = 1.0 / math.sqrt(256.0)  # scalar; any positive value exercises the same code


# ----------------------------------------------------------------------------- inputs
def build_page_table(ctx_len, pool_tokens, gen, device, contiguous=False):
    """token position -> pool location, allocator pages of 64; page 0 is the reserved dummy page."""
    n_pages = (ctx_len + PAGE - 1) // PAGE
    total_pages = pool_tokens // PAGE
    assert n_pages < total_pages
    if contiguous:
        pages = torch.arange(1, n_pages + 1)
    else:
        pages = torch.randperm(total_pages - 1, generator=gen)[:n_pages] + 1
    loc = (pages[:, None] * PAGE + torch.arange(PAGE)[None, :]).reshape(-1)[:ctx_len]
    return loc.to(device=device, dtype=torch.int64)


def make_rows(q_pos, page_table, pattern, gen_seed, sort_groups=False, batch=512):
    """indices [len(q_pos), WIDTH] int32 for one request whose query tokens sit at positions q_pos.

    pattern: 'corr'   = static relevance + recency/sink bonus + small per-query noise (neighbours overlap)
             'uniform'= independent random groups per query (no reuse between queries)
             'same'   = every long row selects the same groups (maximal reuse)
    """
    device = page_table.device
    g = torch.Generator(device=device)
    g.manual_seed(gen_seed)
    M = q_pos.numel()
    out = torch.full((M, WIDTH), -1, dtype=torch.int32, device=device)
    n_groups_total = int(q_pos.max().item() + 1) // KPOOL + 1
    static = torch.randn(n_groups_total, generator=g, device=device)
    grp_ar = torch.arange(n_groups_total, device=device)
    for s in range(0, M, batch):
        p = q_pos[s : s + batch].to(device)
        rows = p.numel()
        G = (p + 1) // KPOOL  # complete pooled groups visible to this query
        tail_n = (p + 1) % KPOOL
        gmax = int(G.max().item())
        valid_g = grp_ar[None, :gmax] < G[:, None]
        if gmax <= TOPK // KPOOL:
            sel = grp_ar[:gmax].expand(rows, gmax).clone()
            sel_valid = valid_g
            nsel = gmax
        else:
            if pattern == "corr":
                dist = (G[:, None] - grp_ar[None, :gmax]).float()
                score = 1.0 * static[None, :gmax] + 3.0 * (dist <= 64).float() + 3.0 * (grp_ar[None, :gmax] < 4).float()
                score = score + 0.35 * torch.randn(rows, gmax, generator=g, device=device)
            elif pattern == "uniform":
                score = torch.rand(rows, gmax, generator=g, device=device)
            elif pattern == "same":
                score = static[None, :gmax].expand(rows, gmax).clone()
                score = score + 1e3 * (grp_ar[None, :gmax] < 4).float()
            else:
                raise ValueError(pattern)
            score = score.masked_fill(~valid_g, float("-inf"))
            nsel = TOPK // KPOOL
            top = torch.topk(score, nsel, dim=1)
            sel = top.indices
            sel_valid = torch.isfinite(top.values)
            if sort_groups or pattern == "same":
                key = torch.where(sel_valid, sel, torch.full_like(sel, 1 << 30))
                order = torch.argsort(key, dim=1)
            else:  # radix top-k emits groups in arbitrary (atomicAdd) order
                order = torch.argsort(torch.rand(rows, nsel, generator=g, device=device), dim=1)
            sel = torch.gather(sel, 1, order)
            sel_valid = torch.gather(sel_valid, 1, order)
            # rows with <= 512 complete groups take the in-order branch of the transform kernel
            short = G <= nsel
            if bool(short.any()):
                ar = grp_ar[:nsel].expand(rows, nsel)
                sel = torch.where(short[:, None], ar, sel)
                sel_valid = torch.where(short[:, None], ar < G[:, None], sel_valid)
        # valid groups first (kernel output is a valid prefix), keep relative order
        vo = torch.argsort((~sel_valid).to(torch.int8), dim=1, stable=True)
        sel = torch.gather(sel, 1, vo)
        sel_valid = torch.gather(sel_valid, 1, vo)
        tok = sel[:, :, None] * KPOOL + torch.arange(KPOOL, device=device)[None, None, :]
        tok_valid = sel_valid[:, :, None].expand_as(tok)
        tok = tok.reshape(rows, -1)
        tok_valid = tok_valid.reshape(rows, -1)
        hist_len = torch.clamp(G * KPOOL, max=TOPK)
        loc = torch.where(tok_valid, page_table[tok.clamp(max=page_table.numel() - 1)], torch.full_like(tok, -1))
        blk = torch.full((rows, WIDTH), -1, dtype=torch.int64, device=device)
        ncol = min(loc.shape[1], TOPK)
        blk[:, :ncol] = loc[:, :ncol]
        # tail tokens right after the history (col hist_len + j), like the transform kernel
        for j in range(KPOOL - 1):
            has = j < tail_n
            col = hist_len + j
            tokpos = G * KPOOL + j
            val = torch.where(has, page_table[tokpos.clamp(max=page_table.numel() - 1)], torch.full_like(tokpos, -1))
            blk.scatter_(1, col[:, None], torch.where(has[:, None], val[:, None], blk.gather(1, col[:, None])))
        out[s : s + batch] = blk.to(torch.int32)
    return out


def pad_width(idx):
    pad = (-idx.shape[-1]) % 64
    if pad:
        idx = torch.cat([idx, idx.new_full((*idx.shape[:-1], pad), -1)], dim=-1)
    return idx


def make_case(case, device="cuda", seed=0):
    """returns dict(q, kv, idx[M, WIDTH_PAD], meta)"""
    gen = torch.Generator()
    gen.manual_seed(seed)
    kind = case["kind"]
    pool = case.get("pool_tokens", 1 << 20)
    pattern = case.get("pattern", "corr")
    rows = []
    if kind == "prefill":
        ctx_end = case["ctx_end"]
        M = case["M"]
        pt = build_page_table(ctx_end, pool, gen, device, contiguous=case.get("contiguous", False))
        q_pos = torch.arange(ctx_end - M, ctx_end, device=device)
        rows.append(make_rows(q_pos, pt, pattern, seed + 1, sort_groups=case.get("sorted", False)))
    elif kind == "decode":  # B requests x T query tokens each (target verify / draft decode)
        B, T = case["B"], case["T"]
        lo, hi = case.get("ctx_lo", 20000), case.get("ctx_hi", 50000)
        ctxs = torch.randint(lo, hi + 1, (B,), generator=gen)
        need = int(sum((int(c) + PAGE - 1) // PAGE for c in ctxs) + 1) * PAGE
        while pool < 2 * need:  # the served pool holds other requests' KV too; keep the batch's pages scattered
            pool *= 2
        used = 0
        perm = torch.randperm(pool // PAGE - 1, generator=gen) + 1
        for b in range(B):
            c = int(ctxs[b])
            npg = (c + PAGE - 1) // PAGE
            pages = perm[used : used + npg]
            used += npg
            pt = (pages[:, None] * PAGE + torch.arange(PAGE)[None, :]).reshape(-1)[:c].to(device=device, dtype=torch.int64)
            q_pos = torch.arange(c - T, c, device=device)
            rows.append(make_rows(q_pos, pt, pattern, seed + 7 * b + 1))
        M = B * T
    else:
        raise ValueError(kind)
    idx = torch.cat(rows, 0)
    if case.get("neg_rows"):  # fully masked rows (e.g. CUDA-graph padding rows)
        idx[-case["neg_rows"] :] = -1
    if case.get("holes"):  # -1 anywhere inside rows (other producers than the kpool transform)
        hg = torch.Generator(device=device)
        hg.manual_seed(seed + 3)
        idx = torch.where(torch.rand(idx.shape, generator=hg, device=device) < case["holes"], torch.full_like(idx, -1), idx)
    if case.get("suffix_only"):  # valid entries only in the last columns of the row
        idx[:, : idx.shape[1] - case["suffix_only"]] = -1
    idx_raw = idx.contiguous()
    idx = pad_width(idx).contiguous()
    g = torch.Generator(device=device)
    g.manual_seed(seed + 99)
    qscale = case.get("qscale", 1.0)
    q = (torch.randn(M, H_LOCAL, D, generator=g, device=device) * qscale).to(torch.bfloat16)
    kv = torch.randn(pool, 1, D, generator=g, device=device).to(torch.bfloat16)
    return dict(q=q, kv=kv, idx=idx, idx_raw=idx_raw, M=M, pool=pool)


# ----------------------------------------------------------------------------- reference
def ref_sparse_fwd(q, kv, idx, sm_scale, rows=None, chunk=32):
    """fp32 reference: softmax over valid (idx >= 0) entries; fully-masked rows -> 0 (reported separately)."""
    M = q.shape[0]
    rows = torch.arange(M, device=q.device) if rows is None else rows
    out = torch.empty(rows.numel(), q.shape[1], D, dtype=torch.float32, device=q.device)
    lse = torch.empty(rows.numel(), q.shape[1], dtype=torch.float32, device=q.device)
    kv2 = kv.view(kv.shape[0], -1)
    for s in range(0, rows.numel(), chunk):
        r = rows[s : s + chunk]
        ii = idx[r].long()
        valid = ii >= 0
        k = kv2[ii.clamp(min=0)].float()  # [c, K, 512]
        qq = q[r].float()  # [c, H, 512]
        sc = torch.einsum("chd,ckd->chk", qq, k) * sm_scale
        sc = sc.masked_fill(~valid[:, None, :], float("-inf"))
        m = sc.amax(-1, keepdim=True)
        m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        p = torch.exp(sc - m)
        l = p.sum(-1, keepdim=True)
        o = torch.einsum("chk,ckd->chd", p, k[..., :D]) / l.clamp(min=1e-30)
        out[s : s + chunk] = torch.where(l > 0, o, torch.zeros_like(o))
        lse[s : s + chunk] = (torch.log(l) + m).squeeze(-1)
    return out, lse


# ----------------------------------------------------------------------------- impls
def served_fwd(q, kv, idx, sm_scale, return_lse=False):
    from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd

    return tilelang_sparse_fwd(q=q, kv=kv, indices=idx.unsqueeze(1), sm_scale=sm_scale, d_v=D, return_lse=return_lse)


def served_e2e(q, kv, idx_raw, sm_scale):
    # dsa_backend._forward_tilelang: -1-pad the 2051-wide table to a multiple of 64, then the TileLang call
    pad = (-idx_raw.shape[-1]) % 64
    t = torch.cat((idx_raw, idx_raw.new_full((*idx_raw.shape[:-1], pad), -1)), dim=-1) if pad else idx_raw
    return served_fwd(q, kv, t, sm_scale)


def get_impl(name):
    """returns (fn(q, kv, idx, sm_scale), wants_raw_indices)"""
    if name == "served":
        return served_fwd, False
    if name == "served_e2e":
        return served_e2e, True
    import dsa_sparse_proto as P

    return getattr(P, name), True


# ----------------------------------------------------------------------------- timing
_FLUSH = None


def l2_flush():
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.empty(96 << 20, dtype=torch.uint8, device="cuda")
    _FLUSH.random_(0, 255) if False else _FLUSH.fill_(1)


def bench(fn, iters=20, warmup=3, flush=True):
    ts = []
    for i in range(warmup + iters):
        if flush:
            l2_flush()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        if i >= warmup:
            ts.append(s.elapsed_time(e))
    ts.sort()
    return dict(med_ms=ts[len(ts) // 2], min_ms=ts[0], max_ms=ts[-1])


def bench_graph(fn, n=8, reps=5, flush=True):
    """CUDA-graph timing (as decode/verify run in serving): per-call ms = (graph[flush+fn]*n - graph[flush]*n) / n."""
    fn()
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g, gf = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            if flush:
                l2_flush()
            fn()
    with torch.cuda.graph(gf):
        for _ in range(n):
            if flush:
                l2_flush()

    def t(gr):
        gr.replay()
        torch.cuda.synchronize()
        best = []
        for _ in range(reps):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            gr.replay()
            e.record()
            e.synchronize()
            best.append(s.elapsed_time(e))
        best.sort()
        return best[len(best) // 2]

    tf, tg = t(gf), t(g)
    del g, gf
    return dict(med_ms=(tg - tf) / n)


def err_stats(out, ref, valid_rows_mask=None):
    o = out.float()
    diff = (o - ref).abs()
    finite = torch.isfinite(o).all(dim=-1).all(dim=-1)
    rel = diff.norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-6)
    res = dict(
        max_abs=float(torch.nan_to_num(diff, nan=float("inf")).max()),
        mean_abs=float(torch.nan_to_num(diff, nan=0.0).mean()),
        max_rowrel=float(torch.nan_to_num(rel, nan=float("inf")).max()),
        nonfinite_rows=int((~finite).sum()),
    )
    return res


def gathered_bytes(idx):
    return int((idx >= 0).sum()) * D * 2


# ----------------------------------------------------------------------------- suites
PREFILL_M = 8192
SUITES = {
    "main": [
        dict(name="prefill_ctx49k_corr", kind="prefill", M=PREFILL_M, ctx_end=49152, pattern="corr"),
        dict(name="prefill_ctx49k_uniform", kind="prefill", M=PREFILL_M, ctx_end=49152, pattern="uniform"),
        dict(name="prefill_ctx49k_same", kind="prefill", M=PREFILL_M, ctx_end=49152, pattern="same"),
        dict(name="prefill_ctx16k_corr", kind="prefill", M=PREFILL_M, ctx_end=16384, pattern="corr"),
        dict(name="prefill_first8k", kind="prefill", M=PREFILL_M, ctx_end=8192, pattern="corr"),
        dict(name="prefill_ctx131k_corr", kind="prefill", M=PREFILL_M, ctx_end=131072, pattern="corr"),
        dict(name="prefill_ctx131k_uniform", kind="prefill", M=PREFILL_M, ctx_end=131072, pattern="uniform"),
        dict(name="verify_32x4", kind="decode", B=32, T=4, pattern="corr"),
        dict(name="decode_32x1", kind="decode", B=32, T=1, pattern="corr"),
        dict(name="verify_8x4", kind="decode", B=8, T=4, pattern="corr"),
        dict(name="verify_1x4", kind="decode", B=1, T=4, pattern="corr"),
        dict(name="verify_16x4", kind="decode", B=16, T=4, pattern="corr"),
    ],
    "edge": [
        dict(name="edge_first_short", kind="prefill", M=1000, ctx_end=1000, pattern="corr"),
        dict(name="edge_M_odd", kind="prefill", M=333, ctx_end=30011, pattern="corr"),
        dict(name="edge_M1", kind="prefill", M=1, ctx_end=5, pattern="corr"),
        dict(name="edge_boundary2051", kind="prefill", M=64, ctx_end=2080, pattern="corr"),
        dict(name="edge_negrows", kind="decode", B=5, T=4, pattern="corr", neg_rows=3),
        dict(name="edge_peaky", kind="prefill", M=512, ctx_end=40000, pattern="corr", qscale=6.0),
        dict(name="edge_holes", kind="prefill", M=777, ctx_end=40000, pattern="corr", holes=0.3),
        dict(name="edge_suffix_only", kind="prefill", M=100, ctx_end=40000, pattern="corr", suffix_only=5),
        dict(name="edge_verify_holes", kind="decode", B=3, T=4, pattern="corr", holes=0.5),
    ],
}


def run_case(case, impls, args):
    torch.cuda.synchronize()
    t0 = time.time()
    c = make_case(case, seed=args.seed)
    q, kv, idx = c["q"], c["kv"], c["idx"]
    M = c["M"]
    nvalid = (idx >= 0).sum(1)
    rec = dict(case=case["name"], M=M, width=idx.shape[1], valid_mean=float(nvalid.float().mean()),
               valid_min=int(nvalid.min()), gathered_GB=gathered_bytes(idx) / 1e9, gen_s=round(time.time() - t0, 2))
    ref_rows = None
    if M > args.ref_rows:
        g = torch.Generator(device="cuda")
        g.manual_seed(5)
        ref_rows = torch.cat([torch.arange(0, 64, device="cuda"), torch.randperm(M, generator=g, device="cuda")[: args.ref_rows - 128],
                              torch.arange(M - 64, M, device="cuda")]).unique()
    ref, ref_lse = ref_sparse_fwd(q, kv, idx, SM_SCALE, rows=ref_rows)
    outs = {}
    idx_raw = c["idx_raw"]
    for name in impls:
        fn, raw = get_impl(name)
        try:
            out = fn(q, kv, idx_raw if raw else idx, SM_SCALE)
            torch.cuda.synchronize()
        except Exception as ex:  # report and continue
            rec[name] = dict(error=repr(ex)[:400])
            continue
        o = out.view(M, H_LOCAL, D)
        outs[name] = o
        sel = o if ref_rows is None else o[ref_rows]
        st = err_stats(sel, ref)
        # fully-masked rows: reference defines 0; report what the impl does
        allneg = (idx < 0).all(1)
        if bool(allneg.any()):
            st["masked_rows_nonfinite"] = int((~torch.isfinite(o[allneg].float())).any(-1).any(-1).sum())
            st["masked_rows_maxabs_finite"] = float(torch.nan_to_num(o[allneg].float(), nan=0.0).abs().max())
            good = ~allneg if ref_rows is None else ~allneg[ref_rows]
            st.update({k + "_validrows": v for k, v in err_stats(sel[good], ref[good]).items()})
        rec[name] = st
    # timing: impls interleaved, --reps rounds; report the median of per-round medians (and the best round)
    rounds = {n: [] for n in outs}
    for _ in range(args.reps):
        for name in outs:
            fn, raw = get_impl(name)
            ii = idx_raw if raw else idx
            call = lambda: fn(q, kv, ii, SM_SCALE)
            if args.eager:
                rounds[name].append(bench(call, iters=args.iters, flush=not args.hot)["med_ms"])
            else:
                rounds[name].append(bench_graph(call, n=args.graph_n, flush=not args.hot)["med_ms"])
    for name, ts in rounds.items():
        ts = sorted(ts)
        rec[name].update(med_ms=ts[len(ts) // 2], min_ms=ts[0], max_ms=ts[-1], GBps=rec["gathered_GB"] / (ts[len(ts) // 2] / 1e3))
    if args.lse:  # LSE (log2 units, as the served kernel returns it) of the prototype default vs served vs fp32 reference
        import dsa_sparse_proto as P

        _, l_srv = served_fwd(q, kv, idx, SM_SCALE, return_lse=True)
        _, l_pro = P.dsa_sparse_fwd(q, kv, idx_raw, SM_SCALE, return_lse=True)
        l_srv, l_pro = l_srv.view(M, H_LOCAL), l_pro.view(M, H_LOCAL)
        good = ~(idx < 0).all(1)
        l_ref2 = ref_lse * 1.4426950408889634
        sel_good = good if ref_rows is None else good[ref_rows]
        pr = l_pro if ref_rows is None else l_pro[ref_rows]
        rec["lse"] = dict(proto_vs_served=float((l_pro[good] - l_srv[good]).abs().max()) if good.any() else None,
                          proto_vs_ref=float((pr[sel_good] - l_ref2[sel_good]).abs().max()) if sel_good.any() else None,
                          proto_masked_rows=[float(x) for x in l_pro[~good].flatten()[:4]])
    if "served" in outs:
        for name, o in outs.items():
            if name != "served":
                d = (o.float() - outs["served"].float())
                both = torch.isfinite(d)
                rec[name]["vs_served_max_abs"] = float(d[both].abs().max()) if both.any() else None
    del c, q, kv, idx, idx_raw, outs
    torch.cuda.empty_cache()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="main")
    ap.add_argument("--cases", default="")
    ap.add_argument("--impls", default="served")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--eager", action="store_true", help="event timing of eager calls (includes host launch gaps)")
    ap.add_argument("--graph-n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ref-rows", type=int, default=2048)
    ap.add_argument("--hot", action="store_true", help="no L2 flush between calls")
    ap.add_argument("--lse", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    torch.manual_seed(0)
    impls = [s for s in args.impls.split(",") if s]
    cases = SUITES[args.suite]
    if args.cases:
        want = set(args.cases.split(","))
        cases = [c for s in SUITES.values() for c in s if c["name"] in want]
    print(json.dumps(dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, impls=impls, hot=args.hot)), flush=True)
    for case in cases:
        rec = run_case(case, impls, args)
        line = json.dumps(rec)
        print(line, flush=True)
        if args.out:
            with open(args.out, "a") as f:
                f.write(line + "\n")



def dump_served(outdir, topk=WIDTH_PAD):
    """write the served TileLang kernel's generated CUDA (same factory args as tilelang_sparse_fwd)."""
    from sglang.kernels.ops.attention.dsa.tilelang_kernel import sparse_attention_fwd_kernel_v1

    k = sparse_attention_fwd_kernel_v1(H_LOCAL, D, 0, topk, sm_scale=SM_SCALE, return_lse=False)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "served_v1.cu"), "w") as f:
        f.write(k.get_kernel_source())
    return k


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--dump-served":
        dump_served(sys.argv[2])
    else:
        main()
