#!/usr/bin/env python3
"""Dequantize-then-BF16 grouped-GEMM MoE vs the served Marlin W8A16 path, GLM-5.3-Flash TP8 per-rank shape, one A100.

Shape (rank 0 of TP8, as bench_moe_ep8.py): 289 experts = 288 routed + the shared expert fused as id 288 (top-k 8+1),
hidden K=4096, intermediate slice N=256: w13 [289, 512, 4096] (gate rows, then up rows), w2 [289, 4096, 256];
FP8 e4m3 weights with 128x128 fp32 block scales; uniform routing (top-8 of randn logits); BF16 activations; SiLU
without the clamp on both paths unless a row says clamp10.

Phases (one process each; weights and inputs are seeded, so every phase sees the same tensors):
  base  : Q1 ceilings (cuBLAS dense pair, bmm), served Marlin, engine Triton fused_moe with BF16 weights and the
          default config (whole call and per stage), Q3 dequant cost, Q4 numerics with negative controls,
          and the shared expert's share of the Marlin call.
  tune  : Q2 search over fused_moe_kernel configs, GEMM1 (N=512, K=4096) and GEMM2 (N=4096, K=256) separately,
          M in 2048/8192/16384, bounded by TUNE_MINUTES; writes the engine's config files (up + _down) to ./moecfg.
  final : the engine call fused_experts_impl reading those files (SGLANG_MOE_CONFIG_DIR), dequant + tuned total vs
          Marlin in the same process, clamp-10 variants, peak memory, numerics of the tuned path.
Timing: CUDA events, eager, warmup then median of 5 reps x n calls. FLOPs = M*9*(2*4096*512 + 2*256*4096).
Usage: python3 bench_moe_bf16.py base|tune|final
"""
import json
import math
import os
import random
import sys
import time

PHASE = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
CFGDIR = os.path.join(HERE, "moecfg")
if PHASE == "final":
    os.environ["SGLANG_MOE_CONFIG_DIR"] = CFGDIR

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

dev = "cuda"
E, K, N, TOPK, B = 289, 4096, 256, 9, 128
MS = [2048, 8192, 16384]
BF = torch.bfloat16

from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler  # noqa: E402

set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
from sglang.srt.distributed.parallel_state import init_distributed_environment, initialize_model_parallel  # noqa: E402

init_distributed_environment(world_size=1, rank=0, local_rank=0,
                             distributed_init_method=f"tcp://127.0.0.1:{29500 + os.getpid() % 1000}", backend="nccl")
initialize_model_parallel(tensor_model_parallel_size=1)

from sgl_kernel import moe_sum_reduce  # noqa: E402
from sglang.kernels.ops.activation.activation import silu_and_mul  # noqa: E402
from sglang.kernels.ops.moe.fused_moe_triton_kernels import invoke_fused_moe_kernel  # noqa: E402
from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe  # noqa: E402
from sglang.srt.layers.moe.moe_runner.triton_utils import (  # noqa: E402
    get_config_file_name, moe_align_block_size, override_config, try_get_optimal_moe_config)
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts_impl  # noqa: E402
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import get_default_config  # noqa: E402
from sglang.srt.layers.quantization.marlin_utils_fp8 import prepare_moe_fp8_layer_for_marlin  # noqa: E402


def emit(**row):
    print(json.dumps(row), flush=True)
    return row


def flops(M):
    return M * TOPK * (2 * K * 2 * N + 2 * N * K)


def tf(M, ms):
    return round(flops(M) / ms / 1e9, 1)


# ---------------------------------------------------------------- weights (seeded, identical in every phase)
def qblock_fp8(w):  # [e, n, k] float -> fp8 [e, n, k], scale_inv [e, n/B, k/B]
    e, n, k = w.shape
    wb = w.float().view(e, n // B, B, k // B, B)
    s = wb.abs().amax(dim=(2, 4)).clamp(min=1e-8) / 448.0
    return (wb / s[:, :, None, :, None]).view(e, n, k).to(torch.float8_e4m3fn), s


torch.manual_seed(0)
t0 = time.time()
q13 = torch.empty(E, 2 * N, K, dtype=torch.float8_e4m3fn, device=dev)
s13 = torch.empty(E, 2 * N // B, K // B, device=dev)
q2 = torch.empty(E, K, N, dtype=torch.float8_e4m3fn, device=dev)
s2 = torch.empty(E, K // B, N // B, device=dev)
for e in range(E):
    a, sa = qblock_fp8(torch.randn(1, 2 * N, K, device=dev) * 0.02 * (1 + 3 * torch.rand(1, 2 * N, 1, device=dev)))
    b, sb = qblock_fp8(torch.randn(1, K, N, device=dev) * 0.02 * (1 + 3 * torch.rand(1, K, 1, device=dev)))
    q13[e], s13[e], q2[e], s2[e] = a[0], sa[0], b[0], sb[0]
assert not (q13.view(torch.uint8) & 127 == 127).any() and not (q2.view(torch.uint8) & 127 == 127).any()  # no fp8 NaN


# ---------------------------------------------------------------- Q3 dequant: fp8 block -> bf16, fp32 scale, one rounding
@triton.jit
def _e4m3_to_bf16(x):  # verbatim from sglang/srt/layers/attention/dsa/sm80_indexer_kernels.py (engine 110/113)
    u = x.to(tl.uint16)
    mag = u & 127
    normal = (mag << 4) + 0x3C00
    sub = (mag.to(tl.float32) * (1.0 / 512.0)).to(tl.bfloat16).to(tl.uint16, bitcast=True)
    bits = tl.where(mag < 8, sub, normal)
    bits = tl.where(mag == 127, 0x7FC0, bits) | ((u & 128) << 8)
    return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)


@triton.jit
def _deq_fp8_block(Q, S, Y, n, k, s_e, s_n, s_k, TN: tl.constexpr, TK: tl.constexpr):
    pn, pk, e = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rn = pn * TN + tl.arange(0, TN)
    rk = pk * TK + tl.arange(0, TK)
    off = e.to(tl.int64) * n * k + rn[:, None] * k + rk[None, :]
    s = tl.load(S + e * s_e + (pn * TN // 128) * s_n + (pk * TK // 128) * s_k)
    y = (_e4m3_to_bf16(tl.load(Q + off)).to(tl.float32) * s).to(tl.bfloat16)
    tl.store(Y + off, y)


DEQ = dict(TN=32, TK=128, warps=4)


def deq_triton(q, s, out, TN=None, TK=None, warps=None):
    TN, TK, warps = TN or DEQ["TN"], TK or DEQ["TK"], warps or DEQ["warps"]
    e, n, k = q.shape
    _deq_fp8_block[(n // TN, k // TK, e)](q.view(torch.uint8), s, out, n, k, s.stride(0), s.stride(1), s.stride(2),
                                          TN=TN, TK=TK, num_warps=warps)
    return out


def deq_torch(q, s):
    e, n, k = q.shape
    return (q.float().view(e, n // B, B, k // B, B) * s[:, :, None, :, None]).view(e, n, k).to(BF)


w13b = torch.empty(E, 2 * N, K, dtype=BF, device=dev)
w2b = torch.empty(E, K, N, dtype=BF, device=dev)
deq_triton(q13, s13, w13b)
deq_triton(q2, s2, w2b)
deq_exact = bool(torch.equal(w13b, deq_torch(q13, s13)) and torch.equal(w2b, deq_torch(q2, s2)))


# ---------------------------------------------------------------- served Marlin layer (patch 111 path)
def marlin_layer():
    layer = torch.nn.Module()
    for k_, v_ in dict(num_experts=E, hidden_size=K, intermediate_size_per_partition=N,
                       weight_block_size=[B, B], orig_dtype=BF).items():
        setattr(layer, k_, v_)
    p = lambda t: torch.nn.Parameter(t.contiguous().clone(), requires_grad=False)  # noqa: E731
    layer.w13_weight, layer.w2_weight = p(q13), p(q2)
    layer.w13_weight_scale_inv, layer.w2_weight_scale_inv = p(s13), p(s2)
    prepare_moe_fp8_layer_for_marlin(layer, size_k_first=False)
    return layer


ML = marlin_layer()
emit(phase=PHASE, setup_s=round(time.time() - t0, 1), torch=torch.__version__, triton=triton.__version__,
     gpu=torch.cuda.get_device_name(0), deq_triton_equals_torch=deq_exact,
     commit=open(os.path.join(os.environ.get("PYTHONPATH", ""), "COMMIT")).read().strip()
     if os.path.exists(os.path.join(os.environ.get("PYTHONPATH", ""), "COMMIT")) else None)


def marlin(x, tw, ids, clamp=None):
    return fused_marlin_moe(hidden_states=x, w1=ML.w13_weight, w2=ML.w2_weight, w1_scale=ML.w13_weight_scale,
                            w2_scale=ML.w2_weight_scale, gating_output=torch.empty(x.shape[0], 1, device=dev),
                            topk_weights=tw, topk_ids=ids, num_bits=8, fp8_weights=True, global_num_experts=E,
                            clamp_limit=clamp)


def triton_moe(x, tw, ids, w13=None, w2=None, clamp=None):
    return fused_experts_impl(x, w13b if w13 is None else w13, w2b if w2 is None else w2, tw, ids, inplace=False,
                              gate_up_interleaved=False, swiglu_limit=clamp)


# ---------------------------------------------------------------- inputs
def inputs(M):
    g = torch.Generator(device=dev).manual_seed(1000 + M)
    x = torch.randn(M, K, device=dev, dtype=BF, generator=g)
    ids = torch.randn(M, E - 1, device=dev, generator=g).topk(TOPK - 1, dim=1).indices.int()
    tw = torch.softmax(torch.randn(M, TOPK - 1, device=dev, generator=g), -1)
    ids9 = torch.cat([ids, torch.full((M, 1), E - 1, dtype=torch.int32, device=dev)], 1).contiguous()
    tw9 = torch.cat([tw, torch.ones(M, 1, device=dev)], 1).contiguous()
    return x, tw9, ids9


def timeit(f, n=10, reps=5):
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


def ncalls(M):
    return {2048: 20, 8192: 8, 16384: 5}.get(M, 10)


# ---------------------------------------------------------------- Q4 reference: fp32 MoE from the dequantized fp8 weights
def reference(x, tw, ids, rows=64):
    x = x[:rows].float()
    out = torch.zeros(rows, K, device=dev)
    for e in ids[:rows].unique().tolist():
        sel = (ids[:rows] == e).nonzero()
        tok, wt = sel[:, 0], tw[:rows][sel[:, 0], sel[:, 1]].float()
        d13 = (q13[e].float().view(2 * N // B, B, K // B, B) * s13[e][:, None, :, None]).view(2 * N, K)
        d2 = (q2[e].float().view(K // B, B, N // B, B) * s2[e][:, None, :, None]).view(K, N)
        h = x[tok] @ d13.T
        out.index_add_(0, tok, wt[:, None] * ((torch.nn.functional.silu(h[:, :N]) * h[:, N:]) @ d2.T))
    return out


def rel(o, r):
    return float(((o[: r.shape[0]].float() - r).norm() / r.norm()).item())


def numerics(M, label, extra_paths=True):
    """Errors of the full-M call's first 64 rows (so the config actually used at M is the one checked)."""
    x, tw, ids = inputs(M)
    ref = reference(x, tw, ids)
    row = dict(kind="numerics", M=M, label=label, rel_err_bf16=round(rel(triton_moe(x, tw, ids), ref), 6),
               rel_err_marlin=round(rel(marlin(x, tw, ids), ref), 6))
    row["bf16_repeat_bitwise_equal"] = bool(torch.equal(triton_moe(x, tw, ids), triton_moe(x, tw, ids)))
    if extra_paths:
        bad13 = deq_triton(q13, s13.roll(1, 0), torch.empty_like(w13b))  # scale of the neighbouring expert
        row["neg_scale_of_other_expert"] = round(rel(triton_moe(x, tw, ids, w13=bad13), ref), 4)
        del bad13
        shifted = torch.cat([(ids[:, :-1] + 1) % (E - 1), ids[:, -1:]], 1).contiguous()  # routed ids off by one
        row["neg_expert_id_shift"] = round(rel(triton_moe(x, tw, shifted), ref), 4)
    return emit(**row)


# ---------------------------------------------------------------- per-stage runner (direct kernel calls, engine code)
class Stages:
    def __init__(self, M, x, tw, ids):
        self.M, self.x, self.tw, self.ids = M, x, tw, ids
        self.c1 = torch.empty(M * TOPK, 2 * N, device=dev, dtype=BF)
        self.c2 = torch.randn(M * TOPK, N, device=dev, dtype=BF)
        self.c3 = torch.empty(M, TOPK, K, device=dev, dtype=BF)
        self.out = torch.empty(M, K, device=dev, dtype=BF)
        self.align_cache = {}

    def align(self, bm):
        if bm not in self.align_cache:
            self.align_cache[bm] = moe_align_block_size(self.ids, bm, E)
        return self.align_cache[bm]

    def _inv(self, A, W, C, mul, topk, cfg):
        st, ei, npad = self.align(cfg["BLOCK_SIZE_M"])
        invoke_fused_moe_kernel(A, W, None, C, None, None, None, self.tw, self.ids, st, ei, npad, mul, topk, cfg,
                                compute_type=tl.bfloat16, use_fp8_w8a8=False, use_int8_w8a8=False,
                                use_int8_w8a16=False, use_int4_w4a16=False, per_channel_quant=False,
                                block_shape=None, filter_expert=True)

    def g1(self, cfg):
        self._inv(self.x, w13b, self.c1, False, TOPK, cfg)

    def g2(self, cfg):
        self._inv(self.c2, w2b, self.c3, True, 1, cfg)

    def split(self, cfg1, cfg2):
        n = ncalls(self.M)
        bm = cfg1["BLOCK_SIZE_M"]
        t = dict(align=timeit(lambda: moe_align_block_size(self.ids, bm, E), n),
                 gemm1=timeit(lambda: self.g1(cfg1), n),
                 act=timeit(lambda: silu_and_mul(self.c1, self.c2, expert_ids=self.ids.view(-1), expert_step=1), n),
                 gemm2=timeit(lambda: self.g2(cfg2), n),
                 sum=timeit(lambda: moe_sum_reduce(self.c3, self.out, 1.0), n))
        return {k: round(v, 3) for k, v in t.items()}


def default_cfg(M):
    return get_default_config(M, E, N, K, TOPK, None, False, None)


# ================================================================= phase base
def phase_base():
    # Q3: dequant variants (both weights of one layer), and torch
    best = None
    for TN in (16, 32, 64, 128):
        for warps in (2, 4, 8):
            ms = timeit(lambda: (deq_triton(q13, s13, w13b, TN, 128, warps), deq_triton(q2, s2, w2b, TN, 128, warps)),
                        10)
            emit(kind="dequant_variant", TN=TN, TK=128, warps=warps, ms=round(ms, 3))
            if best is None or ms < best[0]:
                best = (ms, TN, warps)
    ms_w13 = timeit(lambda: deq_triton(q13, s13, w13b, best[1], 128, best[2]), 10)
    ms_w2 = timeit(lambda: deq_triton(q2, s2, w2b, best[1], 128, best[2]), 10)
    gb = (q13.numel() + q2.numel()) * 3 / 1e9  # 1 B read + 2 B write per element (scales negligible)
    ms_torch = timeit(lambda: (deq_torch(q13, s13), deq_torch(q2, s2)), 3, 3)
    emit(kind="dequant", best_TN=best[1], best_warps=best[2], ms_both=round(best[0], 3), ms_w13=round(ms_w13, 3),
         ms_w2=round(ms_w2, 3), GB_moved=round(gb, 3), TBps=round(gb / best[0], 3), torch_ms_both=round(ms_torch, 3),
         bf16_buffer_GB=round((w13b.numel() + w2b.numel()) * 2 / 1e9, 3))
    deq_triton(q13, s13, w13b); deq_triton(q2, s2, w2b)

    for M in MS:
        x, tw, ids = inputs(M)
        n = ncalls(M)
        R = M * TOPK
        # Q1 ceilings
        A1 = torch.randn(R, K, device=dev, dtype=BF); A2 = torch.randn(R, N, device=dev, dtype=BF)
        W1 = w13b[0].contiguous(); W2 = w2b[0].contiguous()
        d1 = timeit(lambda: torch.mm(A1, W1.t()), n); d2 = timeit(lambda: torch.mm(A2, W2.t()), n)
        del A1, A2
        r = math.ceil(R / E)
        B1 = torch.randn(E, r, K, device=dev, dtype=BF); B2 = torch.randn(E, r, N, device=dev, dtype=BF)
        b1 = timeit(lambda: torch.bmm(B1, w13b.transpose(1, 2)), n); b2 = timeit(lambda: torch.bmm(B2, w2b.transpose(1, 2)), n)
        del B1, B2
        rr = math.ceil(M * (TOPK - 1) / (E - 1))  # 288 routed experts x rows, plus the shared expert on all M rows
        C1 = torch.randn(E - 1, rr, K, device=dev, dtype=BF); C2 = torch.randn(E - 1, rr, N, device=dev, dtype=BF)
        S1 = torch.randn(M, K, device=dev, dtype=BF); S2 = torch.randn(M, N, device=dev, dtype=BF)
        c1 = timeit(lambda: torch.bmm(C1, w13b[:-1].transpose(1, 2)), n)
        c2 = timeit(lambda: torch.bmm(C2, w2b[:-1].transpose(1, 2)), n)
        sh1 = timeit(lambda: torch.mm(S1, w13b[-1].t()), n); sh2 = timeit(lambda: torch.mm(S2, w2b[-1].t()), n)
        del C1, C2, S1, S2
        emit(kind="ceiling", M=M, dense_gemm1_ms=round(d1, 3), dense_gemm2_ms=round(d2, 3), dense_ms=round(d1 + d2, 3),
             dense_TF=tf(M, d1 + d2), dense_gemm1_TF=round(flops(M) * 2 / 3 / d1 / 1e9, 1),
             dense_gemm2_TF=round(flops(M) / 3 / d2 / 1e9, 1),
             bmm_uniform_rows=r, bmm_uniform_ms=round(b1 + b2, 3), bmm_uniform_TF=tf(M, b1 + b2),
             bmm_routed_rows=rr, bmm_routed_plus_shared_ms=round(c1 + c2 + sh1 + sh2, 3),
             bmm_routed_plus_shared_TF=tf(M, c1 + c2 + sh1 + sh2), shared_dense_ms=round(sh1 + sh2, 3))
        # served Marlin, and the shared expert's share of it
        tm = timeit(lambda: marlin(x, tw, ids), n)
        ids8, tw8 = ids[:, :-1].contiguous(), tw[:, :-1].contiguous()
        tm8 = timeit(lambda: marlin(x, tw8, ids8), n)
        # engine Triton BF16 fused_moe, default config
        cfg = default_cfg(M)
        used = try_get_optimal_moe_config(w13b.shape, w2b.shape, TOPK, None, M, return_down_config=True)
        td = timeit(lambda: triton_moe(x, tw, ids), n)
        st = Stages(M, x, tw, ids).split(cfg, cfg)
        emit(kind="paths", M=M, marlin_ms=round(tm, 3), marlin_TF=tf(M, tm), marlin_top8_no_shared_ms=round(tm8, 3),
             triton_default_ms=round(td, 3), triton_default_TF=tf(M, td), default_cfg=cfg, cfg_engine_used=used[0],
             triton_default_stages=st, flops=flops(M))
        numerics(M, "default")


# ================================================================= phase tune
BMS = (64, 128, 256)


def space():
    out = []
    for bm in BMS:
        for bn in (64, 128, 256):
            for bk in (32, 64, 128):
                for w in (4, 8):
                    for s in (2, 3, 4, 5):
                        if s * (bm * bk + bk * bn) * 2 > 166912:
                            continue
                        out.append(dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk, GROUP_SIZE_M=1,
                                        num_warps=w, num_stages=s))
    return out


def phase_tune():
    minutes = float(os.environ.get("TUNE_MINUTES", "38"))
    t_start = time.time()
    stage1_deadline = t_start + 60 * minutes * 0.8
    runners = {M: Stages(M, *inputs(M)) for M in MS}
    log = open(os.path.join(HERE, os.environ.get("TUNE_LOG", "tune_results.jsonl")), "a")
    res = {}  # (gemm, M, key) -> ms

    def key(c):
        return json.dumps(c, sort_keys=True)

    def trial(gemm, cfg, reps=3):
        rows = {}
        for M in MS:
            rn = runners[M]
            f = (lambda: rn.g1(cfg)) if gemm == 1 else (lambda: rn.g2(cfg))
            try:
                ms = timeit(f, ncalls(M), reps)
            except Exception as ex:  # out of resources etc.: record and skip this config
                rec = dict(gemm=gemm, cfg=cfg, error=f"{type(ex).__name__}: {str(ex)[:120]}")
                log.write(json.dumps(rec) + "\n"); log.flush()
                return None
            res[(gemm, M, key(cfg))] = ms
            rows[M] = round(ms, 4)
        log.write(json.dumps(dict(gemm=gemm, cfg=cfg, ms=rows, t=round(time.time() - t_start))) + "\n"); log.flush()
        return rows

    configs = space()
    random.Random(0).shuffle(configs)
    done = 0
    for cfg in configs:
        if time.time() > stage1_deadline:
            break
        for gemm in (1, 2):
            trial(gemm, cfg)
        done += 1
    emit(kind="tune_stage1", configs_total=len(configs), configs_done=done, minutes=round((time.time() - t_start) / 60, 1))

    # stage 2: GROUP_SIZE_M variants and longer re-timing of the top configs per (gemm, M, BLOCK_SIZE_M)
    final = {}
    for gemm in (1, 2):
        for M in MS:
            for bm in BMS:
                cand = sorted(((ms, k) for (g, m, k), ms in res.items()
                               if g == gemm and m == M and json.loads(k)["BLOCK_SIZE_M"] == bm))[:3]
                best = None
                for _, k in cand:
                    for grp in (1, 8, 16, 32):
                        c = dict(json.loads(k), GROUP_SIZE_M=grp)
                        rn = runners[M]
                        f = (lambda: rn.g1(c)) if gemm == 1 else (lambda: rn.g2(c))
                        try:
                            ms = timeit(f, ncalls(M), 7)
                        except Exception:
                            continue
                        log.write(json.dumps(dict(stage=2, gemm=gemm, M=M, cfg=c, ms=round(ms, 4))) + "\n"); log.flush()
                        if best is None or ms < best[0]:
                            best = (ms, c)
                if best:
                    final[(gemm, M, bm)] = best
    # joint BLOCK_SIZE_M per M (both GEMMs share one alignment), then the engine's up and _down files
    up, down, summary = {}, {}, []
    for M in MS:
        opts = [(final[(1, M, bm)][0] + final[(2, M, bm)][0], bm) for bm in BMS
                if (1, M, bm) in final and (2, M, bm) in final]
        tot, bm = min(opts)
        up[str(M)], down[str(M)] = final[(1, M, bm)][1], final[(2, M, bm)][1]
        summary.append(emit(kind="tuned", M=M, BLOCK_SIZE_M=bm, gemm1_ms=round(final[(1, M, bm)][0], 3),
                            gemm2_ms=round(final[(2, M, bm)][0], 3), gemm_sum_ms=round(tot, 3),
                            per_bm={str(b): round(t, 3) for t, b in opts}, up=up[str(M)], down=down[str(M)]))
    vdir = os.path.join(CFGDIR, "configs", f"triton_{triton.__version__.replace('.', '_')}")
    os.makedirs(vdir, exist_ok=True)
    for d, down_moe in ((up, False), (down, True)):
        fn = os.path.join(vdir, get_config_file_name(E, N, None, None, False, down_moe=down_moe))
        with open(fn, "w") as f:
            json.dump(d, f, indent=2)
        emit(kind="config_file", path=fn)
    emit(kind="tune_done", minutes=round((time.time() - t_start) / 60, 1))


# ================================================================= phase final
def phase_final():
    for M in [int(v) for v in os.environ.get("FINAL_MS", "2048,4096,6144,8192,16384").split(",")]:
        x, tw, ids = inputs(M)
        n = ncalls(M)
        cfg, (dcfg, _) = try_get_optimal_moe_config(w13b.shape, w2b.shape, TOPK, None, M, return_down_config=True)
        tm = timeit(lambda: marlin(x, tw, ids), n)
        tm_c = timeit(lambda: marlin(x, tw, ids, clamp=10.0), n)
        tt = timeit(lambda: triton_moe(x, tw, ids), n)
        tt_c = timeit(lambda: triton_moe(x, tw, ids, clamp=10.0), n)
        with override_config(default_cfg(M)):
            tdef = timeit(lambda: triton_moe(x, tw, ids), n)
        tdq = timeit(lambda: (deq_triton(q13, s13, w13b), deq_triton(q2, s2, w2b)), n)

        def deq_then_moe():
            deq_triton(q13, s13, w13b); deq_triton(q2, s2, w2b)
            return triton_moe(x, tw, ids)
        ttot = timeit(deq_then_moe, n)
        st = Stages(M, x, tw, ids).split(cfg, dcfg or cfg)
        # peak extra memory of one call (weights and inputs already resident)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
        triton_moe(x, tw, ids); torch.cuda.synchronize(); pk_t = torch.cuda.max_memory_allocated() - base
        torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
        marlin(x, tw, ids); torch.cuda.synchronize(); pk_m = torch.cuda.max_memory_allocated() - base
        emit(kind="final", M=M, cfg_up=cfg, cfg_down=dcfg, marlin_ms=round(tm, 3), marlin_TF=tf(M, tm),
             marlin_clamp10_ms=round(tm_c, 3), triton_default_ms=round(tdef, 3), triton_tuned_ms=round(tt, 3),
             triton_tuned_TF=tf(M, tt), triton_tuned_clamp10_ms=round(tt_c, 3), tuned_stages=st,
             dequant_ms=round(tdq, 3), dequant_plus_tuned_ms=round(ttot, 3), dequant_plus_tuned_TF=tf(M, ttot),
             speedup_vs_marlin_persistent_bf16=round(tm / tt, 3), speedup_vs_marlin_with_dequant=round(tm / ttot, 3),
             call_peak_extra_GB_triton=round(pk_t / 1e9, 3), call_peak_extra_GB_marlin=round(pk_m / 1e9, 3))
        numerics(M, "tuned")
    # persistent BF16 copy (shape arithmetic): 42 MoE layers + 1 MTP layer, per rank
    per_layer = (w13b.numel() + w2b.numel())
    emit(kind="memory", fp8_GB_per_layer=round(per_layer / 1e9, 3), bf16_GB_per_layer=round(per_layer * 2 / 1e9, 3),
         layers=43, bf16_copy_GB_all_layers=round(per_layer * 2 * 43 / 1e9, 1),
         bf16_copy_GiB_all_layers=round(per_layer * 2 * 43 / 2**30, 1), fp8_GB_all_layers=round(per_layer * 43 / 1e9, 1))


{"base": phase_base, "tune": phase_tune, "final": phase_final}[PHASE]()
print("DONE", flush=True)
