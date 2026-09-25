#!/usr/bin/env python3
"""EP8 vs TP8 MoE layer time on one A100, GLM-5.3-Flash shapes, Marlin FP8 W8A16 (111's serving path).

TP8 (served today): every rank runs all 288 routed experts plus the shared one (fused as expert 288, top-k 9)
on its 1/8 slice of the intermediate dim (N=256). Each rank does the same work, so rank 0's slice is timed.
EP8 (--ep-size 8 --moe-a2a-backend none): rank r holds experts 36r..36r+35 whole (N=2048); routed ids outside
its range are -1 (the dispatcher's mapping) and fused_marlin_moe runs with is_ep. Every rank is timed with the
same routing; the layer waits for the slowest. The shared expert is no longer fused under EP: it is timed
as a 1-expert Marlin layer on its TP slice (N=256, all tokens), an approximation of the dense path.
Both layouts need the same all-reduce of the MoE output afterwards, so communication is not timed.

Numerics: 64 tokens against an fp32 reference built from the dequantized fp8 weights (the model's weights).
Negative control: EP ids shifted by one expert must fail visibly, or the check is void.
Usage: python3 bench_moe_ep8.py [M list, default 64,128,256,2048,8192,16384]
"""
import json
import os
import sys
import time

import torch

torch.manual_seed(0)
dev = "cuda"
E, K, NF, TOPK, B = 288, 4096, 2048, 8, 128  # routed experts, hidden, expert intermediate, top-k, fp8 block
TP = EP = 8
NT, EL = NF // TP, E // EP  # 256 per TP slice, 36 experts per EP rank
MS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "64,128,256,2048,8192,16384").split(",")]

from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe  # noqa: E402
from sglang.srt.layers.quantization.marlin_utils_fp8 import prepare_moe_fp8_layer_for_marlin  # noqa: E402


def qblock_fp8(w):  # [e, n, k] float -> fp8 [e, n, k], scale_inv [e, n/B, k/B]
    e, n, k = w.shape
    wb = w.float().view(e, n // B, B, k // B, B)
    s = wb.abs().amax(dim=(2, 4)).clamp(min=1e-8) / 448.0
    return (wb / s[:, :, None, :, None]).view(e, n, k).to(torch.float8_e4m3fn), s


def deq(q, s):
    e, n, k = q.shape
    return (q.float().view(e, n // B, B, k // B, B) * s[:, :, None, :, None]).view(e, n, k)


# Full-width fp8 weights for 288 routed + 1 shared expert, generated per expert to bound memory.
q13 = torch.empty(E + 1, 2 * NF, K, dtype=torch.float8_e4m3fn, device=dev)
s13 = torch.empty(E + 1, 2 * NF // B, K // B, device=dev)
q2 = torch.empty(E + 1, K, NF, dtype=torch.float8_e4m3fn, device=dev)
s2 = torch.empty(E + 1, K // B, NF // B, device=dev)
for e in range(E + 1):
    a, sa = qblock_fp8(torch.randn(1, 2 * NF, K, device=dev) * 0.02 * (1 + 3 * torch.rand(1, 2 * NF, 1, device=dev)))
    b, sb = qblock_fp8(torch.randn(1, K, NF, device=dev) * 0.02 * (1 + 3 * torch.rand(1, K, 1, device=dev)))
    q13[e], s13[e], q2[e], s2[e] = a[0], sa[0], b[0], sb[0]


def marlin_layer(w13, sc13, w2, sc2, n):
    layer = torch.nn.Module()
    for k_, v_ in dict(num_experts=w13.shape[0], hidden_size=K, intermediate_size_per_partition=n,
                       weight_block_size=[B, B], orig_dtype=torch.bfloat16).items():
        setattr(layer, k_, v_)
    p = lambda t: torch.nn.Parameter(t.contiguous().clone(), requires_grad=False)  # noqa: E731
    layer.w13_weight, layer.w2_weight = p(w13), p(w2)
    layer.w13_weight_scale_inv, layer.w2_weight_scale_inv = p(sc13), p(sc2)
    prepare_moe_fp8_layer_for_marlin(layer, size_k_first=False)
    return layer


def tp_slice(r, experts):  # rank r's slice of gate, up and down for the given experts
    g, u = slice(r * NT, (r + 1) * NT), slice(NF + r * NT, NF + (r + 1) * NT)
    gb, ub = slice(r * NT // B, (r + 1) * NT // B), slice((NF + r * NT) // B, (NF + (r + 1) * NT) // B)
    w13 = torch.cat([q13[experts, g], q13[experts, u]], 1)
    sc13 = torch.cat([s13[experts, gb], s13[experts, ub]], 1)
    return w13, sc13, q2[experts, :, g], s2[experts, :, gb]


t0 = time.time()
TP0 = marlin_layer(*tp_slice(0, slice(0, E + 1)), NT)  # 289 experts incl. shared, N=256
EPL = [marlin_layer(q13[r * EL:(r + 1) * EL], s13[r * EL:(r + 1) * EL], q2[r * EL:(r + 1) * EL],
                    s2[r * EL:(r + 1) * EL], NF) for r in range(EP)]
SH0 = marlin_layer(*tp_slice(0, slice(E, E + 1)), NT)  # shared expert, TP slice, EP layout
print(json.dumps(dict(setup_s=round(time.time() - t0, 1), torch=torch.__version__,
                      gpu=torch.cuda.get_device_name(0))), flush=True)


def call(layer, x, tw, ti, ep):
    return fused_marlin_moe(hidden_states=x, w1=layer.w13_weight, w2=layer.w2_weight, w1_scale=layer.w13_weight_scale,
                            w2_scale=layer.w2_weight_scale, gating_output=torch.empty(x.shape[0], 1, device=dev),
                            topk_weights=tw, topk_ids=ti, num_bits=8, fp8_weights=True,
                            global_num_experts=layer.num_experts,
                            expert_map=torch.empty(0, device=dev) if ep else None)


def timeit(f, n=20, reps=5):
    f(); torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        for _ in range(n):
            f()
        b.record(); torch.cuda.synchronize()
        out.append(a.elapsed_time(b) / n)
    return sorted(out)[len(out) // 2]


def routing(M, skew):
    logits = torch.randn(M, E, device=dev)
    if skew:  # expert popularity ~ 1/(rank+1)^0.6 in a random expert order (Gumbel top-k sampling)
        pop = (torch.arange(E, device=dev) + 1.0) ** -0.6
        logits = torch.log(pop[torch.randperm(E, device=dev)])[None] - torch.log(-torch.log(torch.rand(M, E, device=dev)))
    ids = logits.topk(TOPK, dim=1).indices.int()
    return ids, torch.softmax(torch.randn(M, TOPK, device=dev), -1)


def ep_ids(ids, r, shift=0):
    local = ids - r * EL + shift
    return torch.where((ids >= r * EL) & (ids < (r + 1) * EL), local, torch.full_like(ids, -1)).int()


def reference(x, tw, ids, rows=64):
    x = x[:rows].float()
    out = torch.zeros(rows, K, device=dev)
    for e in list(range(E + 1)):
        sel = (ids[:rows] == e).nonzero() if e < E else None
        if e < E and sel.numel() == 0:
            continue
        w13f, w2f = deq(q13[e:e + 1], s13[e:e + 1])[0], deq(q2[e:e + 1], s2[e:e + 1])[0]
        tok = sel[:, 0] if e < E else torch.arange(rows, device=dev)
        wt = tw[:rows][sel[:, 0], sel[:, 1]] if e < E else torch.ones(rows, device=dev)
        h = x[tok] @ w13f.T
        a = torch.nn.functional.silu(h[:, :NF]) * h[:, NF:]
        out.index_add_(0, tok, wt[:, None] * (a @ w2f.T))
    return out


def rel(o, r):
    return float(((o.float() - r).norm() / r.norm()).item())


rows = []
for M in MS:
    for skew in (False, True):
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        ids, tw = routing(M, skew)
        tp_ids = torch.cat([ids, torch.full((M, 1), E, dtype=torch.int32, device=dev)], 1)
        tp_tw = torch.cat([tw, torch.ones(M, 1, device=dev)], 1)
        ones = torch.ones(M, 1, device=dev)
        zeros = torch.zeros(M, 1, dtype=torch.int32, device=dev)
        t_tp = timeit(lambda: call(TP0, x, tp_tw, tp_ids, False))
        t_ep = [timeit(lambda r=r: call(EPL[r], x, tw, ep_ids(ids, r), True)) for r in range(EP)]
        t_sh = timeit(lambda: call(SH0, x, ones, zeros, False))
        load = [int(((ids >= r * EL) & (ids < (r + 1) * EL)).sum()) for r in range(EP)]
        row = dict(M=M, routing="skew" if skew else "uniform", tp8_ms=round(t_tp, 3),
                   ep8_rank_ms=[round(t, 3) for t in t_ep], shared_ms=round(t_sh, 3),
                   ep8_layer_ms=round(max(t_ep) + t_sh, 3), speedup=round(t_tp / (max(t_ep) + t_sh), 3),
                   rank_load=load, load_max_over_mean=round(max(load) / (sum(load) / EP), 3))
        if M >= 64 and not skew:
            ref = reference(x, tw, ids)
            ep_out = sum(call(EPL[r], x[:64].contiguous(), tw[:64].contiguous(), ep_ids(ids[:64], r), True).float()
                         for r in range(EP))
            sh_full = marlin_layer(q13[E:E + 1], s13[E:E + 1], q2[E:E + 1], s2[E:E + 1], NF)
            ep_out = ep_out + call(sh_full, x[:64].contiguous(), ones[:64], zeros[:64], False).float()
            tp_out = sum(call(marlin_layer(*tp_slice(r, slice(0, E + 1)), NT), x[:64].contiguous(),
                              tp_tw[:64].contiguous(), tp_ids[:64].contiguous(), False).float() for r in range(TP))
            bad = sum(call(EPL[r], x[:64].contiguous(), tw[:64].contiguous(),
                           ep_ids(ids[:64], r, shift=1).clamp(max=EL - 1), True).float() for r in range(EP))
            bad = bad + call(sh_full, x[:64].contiguous(), ones[:64], zeros[:64], False).float()
            row.update(rel_err_ep8=round(rel(ep_out, ref), 5), rel_err_tp8=round(rel(tp_out, ref), 5),
                       rel_err_negative_control=round(rel(bad, ref), 3))
        rows.append(row)
        print(json.dumps(row), flush=True)

print("\n| M | routing | TP8 ms | EP8 slowest rank ms | shared ms | EP8 layer ms | speedup | load max/mean |")
for r in rows:
    print(f"| {r['M']} | {r['routing']} | {r['tp8_ms']} | {max(r['ep8_rank_ms'])} | {r['shared_ms']} | "
          f"{r['ep8_layer_ms']} | {r['speedup']} | {r['load_max_over_mean']} |")
print("DONE")
