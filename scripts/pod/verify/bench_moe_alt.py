#!/usr/bin/env python3
"""MoE kernel alternatives at GLM-5.3-Flash's TP8 per-rank shape on one A100 (rank 0's slice).

Shape: 288 routed experts + the shared expert fused as id 288 (top-k 8+1 per token), hidden K=4096,
intermediate slice N=256 per rank, FP8 e4m3 weights with 128x128 fp32 block scales. Serving
parameters: swiglu_limit=10 (torch clamp path, 172 off), routed_scaling_factor=2.5, shared weight 1/2.5.
Weights are generated like bench_moe_ep8.py but only rank 0's slice (same distribution per slice).

Candidates (all fed the same weights, tokens and routing):
  marlin   fused_marlin_moe exactly as the serving runner calls it (engine source, unmodified)
  c6       a local copy of fused_marlin_moe whose intermediate_cache13 is torch.empty (non-EP path)
  hum_idx  base HummingMoEMethod (block-FP8 checkpoint -> Humming) + HummingRunnerCore, indexed GEMM
  hum_grp  same layer, grouped_contiguous GEMM

Modes: marlin (marlin+c6 timing, stage breakdown, bitwise checks, numerics), humming (humming +
marlin re-timed in the same process, numerics), contract (-1 routed ids: why upstream zeroes).
Usage: python3 bench_alt.py MODE [M list, default 128,2048,8192,16384]   (MODE also: extra)
Layout: run from <dir>/alt0/ next to <dir>/src (engine tree with COMMIT); humming-kernels 0.1.12 sdist
(PyPI blake2b 93b95379...) unpacked at alt0/hk/humming_kernels-0.1.12. Dev-box copy: runs/ep8/alt0/bench_alt.py.
"""
import hashlib
import json
import os
import sys
import time

ALT = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HUMMING_CACHE_DIR", os.path.join(ALT, "hcache"))
os.environ.setdefault("HUMMING_TMP_DIR", os.path.join(ALT, "htmp"))
sys.path.insert(0, os.path.join(ALT, "hk", "humming_kernels-0.1.12"))
if len(sys.argv) > 1 and sys.argv[1] == "contract":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # name the faulting kernel
if len(sys.argv) > 1 and sys.argv[1] in ("humming", "extra") and os.environ.get("CUDA_HOME"):
    # Humming's nvrtc_compile helper runs as a child process; its libnvrtc dlopens libnvrtc-builtins by name,
    # which lives next to it in $CUDA_HOME/lib (run.sh only puts the compat libcuda on LD_LIBRARY_PATH).
    os.environ["LD_LIBRARY_PATH"] = ":".join(
        x for x in (os.environ.get("LD_LIBRARY_PATH"), os.path.join(os.environ["CUDA_HOME"], "lib")) if x)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

MODE = sys.argv[1]
MS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "128,2048,8192,16384").split(",")]
dev = "cuda"
E, K, NF, TOPK, B = 288, 4096, 2048, 8, 128
NT, EG, TK = NF // 8, E + 1, TOPK + 1
CLAMP, RSF = 10.0, 2.5
REF_ROWS = 64


def log(**kw):
    print(json.dumps(kw), flush=True)


import sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe as FM  # noqa: E402
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size  # noqa: E402
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace  # noqa: E402
from sglang.srt.layers.quantization.marlin_utils_fp8 import prepare_moe_fp8_layer_for_marlin  # noqa: E402

log(event="start", mode=MODE, ms=MS, torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
    commit=open(os.path.join(os.path.dirname(ALT), "src", "COMMIT")).read().strip()
    if os.path.exists(os.path.join(os.path.dirname(ALT), "src", "COMMIT")) else None,
    fused_marlin_moe_sha256=hashlib.sha256(open(FM.__file__, "rb").read()).hexdigest()[:16])


# ---------------------------------------------------------------- weights (rank 0 slice)
def qblock_fp8(w):  # [e, n, k] float -> fp8 [e, n, k], scale_inv [e, n/B, k/B]
    e, n, k = w.shape
    wb = w.float().view(e, n // B, B, k // B, B)
    s = wb.abs().amax(dim=(2, 4)).clamp(min=1e-8) / 448.0
    return (wb / s[:, :, None, :, None]).view(e, n, k).to(torch.float8_e4m3fn), s


def deq(q, s):
    e, n, k = q.shape
    return (q.float().view(e, n // B, B, k // B, B) * s[:, :, None, :, None]).view(e, n, k)


gen = torch.Generator(device=dev).manual_seed(0)
q13 = torch.empty(EG, 2 * NT, K, dtype=torch.float8_e4m3fn, device=dev)  # [gate 256; up 256]
s13 = torch.empty(EG, 2 * NT // B, K // B, device=dev)
q2 = torch.empty(EG, K, NT, dtype=torch.float8_e4m3fn, device=dev)
s2 = torch.empty(EG, K // B, NT // B, device=dev)
for e in range(EG):
    a, sa = qblock_fp8(torch.randn(1, 2 * NT, K, device=dev, generator=gen) * 0.02
                       * (1 + 3 * torch.rand(1, 2 * NT, 1, device=dev, generator=gen)))
    b, sb = qblock_fp8(torch.randn(1, K, NT, device=dev, generator=gen) * 0.02
                       * (1 + 3 * torch.rand(1, K, 1, device=dev, generator=gen)))
    q13[e], s13[e], q2[e], s2[e] = a[0], sa[0], b[0], sb[0]
log(event="weights", q13_sum=float(q13.float().sum()), s13_sum=float(s13.double().sum()),
    q2_sum=float(q2.float().sum()), s2_sum=float(s2.double().sum()))

# Negative controls: plausible block-scale layout bugs.
S13_GU_SWAP = torch.cat([s13[:, 2:4], s13[:, 0:2]], 1).contiguous()  # gate/up scale halves swapped
S13_E_SHIFT = s13.roll(1, dims=0).contiguous()  # scales taken from the neighbouring expert


def marlin_layer(sc13):
    layer = torch.nn.Module()
    for k_, v_ in dict(num_experts=EG, hidden_size=K, intermediate_size_per_partition=NT,
                       weight_block_size=[B, B], orig_dtype=torch.bfloat16).items():
        setattr(layer, k_, v_)
    p = lambda t: torch.nn.Parameter(t.contiguous().clone(), requires_grad=False)  # noqa: E731
    layer.w13_weight, layer.w2_weight = p(q13), p(q2)
    layer.w13_weight_scale_inv, layer.w2_weight_scale_inv = p(sc13), p(s2)
    prepare_moe_fp8_layer_for_marlin(layer, size_k_first=False)
    return layer


# ---------------------------------------------------------------- inputs
def routing(M, g, masked_rows=0):
    ids = torch.randn(M, E, device=dev, generator=g).topk(TOPK, dim=1).indices.int()
    tw = torch.softmax(torch.randn(M, TOPK, device=dev, generator=g), -1)
    ids = torch.cat([ids, torch.full((M, 1), E, dtype=torch.int32, device=dev)], 1).contiguous()
    tw = torch.cat([tw, torch.full((M, 1), 1.0 / RSF, device=dev)], 1).float().contiguous()
    if masked_rows:
        ids[M - masked_rows:] = -1  # what _mask_topk_ids_padded_region does to padded rows
    return ids, tw


def reference(x, tw, ids, sc13=None):  # fp32, dequantized fp8 weights, clamp + SiLU, scaled
    sc13 = s13 if sc13 is None else sc13
    x = x.float()
    out = torch.zeros(x.shape[0], K, device=dev)
    for e in ids.unique().tolist():
        if e < 0:
            continue
        sel = (ids == e).nonzero()
        tok = sel[:, 0]
        wt = tw[sel[:, 0], sel[:, 1]].float()
        h = x[tok] @ deq(q13[e:e + 1], sc13[e:e + 1])[0].T
        act = F.silu(h[:, :NT].clamp(max=CLAMP)) * h[:, NT:].clamp(min=-CLAMP, max=CLAMP)
        out.index_add_(0, tok, wt[:, None] * (act @ deq(q2[e:e + 1], s2[e:e + 1])[0].T))
    return out * RSF


def rel(o, r):
    return float(((o.float() - r).norm() / r.norm()).item())


def flops(M):
    return M * TK * 3 * K * NT * 2


# ---------------------------------------------------------------- candidates
def marlin_serving(layer, x, tw, ids, fn=FM.fused_marlin_moe, **extra):
    """Exactly the arguments fused_experts_none_to_marlin passes (moe_runner/marlin.py)."""
    ws = marlin_make_workspace(x.device, max_blocks_per_sm=4)
    return fn(hidden_states=x, w1=layer.w13_weight, w2=layer.w2_weight, w1_scale=layer.w13_weight_scale,
              w2_scale=layer.w2_weight_scale, gating_output=torch.empty(x.shape[0], 1, device=dev),
              topk_weights=tw, topk_ids=ids, global_num_experts=layer.num_experts, expert_map=None,
              workspace=ws, num_bits=8, is_k_full=True, inplace=False, routed_scaling_factor=RSF,
              clamp_limit=CLAMP, gemm1_alpha=None, activation="silu", is_gated=True, fp8_weights=True,
              **extra).to(x.dtype)


def fused_marlin_moe_local(hidden_states, w1, w2, w1_scale, w2_scale, gating_output, topk_weights, topk_ids,
                           global_num_experts=-1, expert_map=None, workspace=None, num_bits=8, is_k_full=True,
                           inplace=False, routed_scaling_factor=None, clamp_limit=None, gemm1_alpha=None,
                           activation="silu", is_gated=True, fp8_weights=False,
                           empty_cache13=True, poison=False, events=None):
    """Copy of fused_marlin_moe (engine 37e90023) restricted to the branches this model takes
    (fp8 W8A16, gated SiLU with clamp_limit, no bias/zeros/act-order, not mxfp4/nvfp4).
    Change C6(a): intermediate_cache13 is torch.empty when expert_map is None.
    poison=True fills the fresh buffer with NaN so any row read before being written shows up.
    events: optional list; CUDA events are recorded between stages for a breakdown."""
    from sgl_kernel.scalar_type import scalar_types

    def rec():
        if events is not None:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            events.append(ev)

    assert fp8_weights and num_bits == 8 and is_gated and activation == "silu" and gemm1_alpha is None
    assert clamp_limit is not None
    M, Kd = hidden_states.shape
    Ee = w1.shape[0]
    N = w2.shape[1] * 16
    topk = topk_ids.shape[1]
    gemm1_n = 2 * N
    for block_size_m in [8, 16, 32, 48, 64]:
        if M * topk / Ee / block_size_m < 0.9:
            break
    if global_num_experts == -1:
        global_num_experts = Ee
    rec()
    assert M > 1  # the single-token align path is not exercised here
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_size_m, global_num_experts)
    if workspace is None:
        max_workspace_size = (max(2 * N, Kd) // 64) * (sorted_token_ids.size(0) // block_size_m)
        sms = torch.cuda.get_device_properties(hidden_states.device).multi_processor_count
        workspace = torch.zeros(min(max_workspace_size, sms * 4), dtype=torch.int, device=hidden_states.device)
    scalar_type = scalar_types.float8_e4m3fn
    rec()
    intermediate_cache2 = torch.empty((M * topk, N), device=hidden_states.device, dtype=hidden_states.dtype)
    alloc = torch.empty if (empty_cache13 and expert_map is None) else torch.zeros
    intermediate_cache13 = alloc((M * topk * max(gemm1_n, Kd),), device=hidden_states.device,
                                 dtype=hidden_states.dtype)
    if poison:
        intermediate_cache13.fill_(float("nan"))
        intermediate_cache2.fill_(float("nan"))
    intermediate_cache1 = intermediate_cache13[: M * topk * gemm1_n].view(-1, gemm1_n)
    intermediate_cache3 = intermediate_cache13[: M * topk * Kd].view(-1, Kd)
    use_atomic_add = (hidden_states.dtype == torch.half
                      or torch.cuda.get_device_capability(hidden_states.device)[0] >= 9)
    rec()
    intermediate_cache1 = FM.moe_wna16_marlin_gemm(
        hidden_states, intermediate_cache1, w1, None, w1_scale, None, None, None, None, workspace,
        sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, moe_block_size=block_size_m,
        top_k=topk, mul_topk_weights=False, is_ep=expert_map is not None, b_q_type=scalar_type, size_m=M,
        size_n=gemm1_n, size_k=Kd, is_k_full=is_k_full, use_atomic_add=use_atomic_add, use_fp32_reduce=True,
        is_zp_float=False)
    rec()
    FM.swiglu_limit_func(intermediate_cache2, intermediate_cache1.view(-1, gemm1_n), clamp_limit)
    rec()
    if expert_map is not None:
        intermediate_cache3.zero_()
    intermediate_cache3 = FM.moe_wna16_marlin_gemm(
        intermediate_cache2, intermediate_cache3, w2, None, w2_scale, None, None, None, None, workspace,
        sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, moe_block_size=block_size_m,
        top_k=1, mul_topk_weights=True, is_ep=expert_map is not None, b_q_type=scalar_type, size_m=M * topk,
        size_n=Kd, size_k=N, is_k_full=is_k_full, use_atomic_add=use_atomic_add, use_fp32_reduce=True,
        is_zp_float=False).view(-1, topk, Kd)
    rec()
    output = FM.zero_copy_context.get_moe_output(hidden_states)
    if output is None:
        output = hidden_states if inplace else torch.empty_like(hidden_states)
    FM.moe_sum_reduce(intermediate_cache3, output, 1.0 if routed_scaling_factor is None else routed_scaling_factor)
    rec()
    return output


def c6_call(layer, x, tw, ids, **kw):
    return marlin_serving(layer, x, tw, ids, fn=fused_marlin_moe_local, **kw)


def zeros_call(layer, x, tw, ids, **kw):  # local copy with the original torch.zeros (checks the copy itself)
    return marlin_serving(layer, x, tw, ids, fn=fused_marlin_moe_local, empty_cache13=False, **kw)


class fixed_align:
    """moe_align_block_size orders tokens within an expert with atomics, so the serving path itself is not
    bitwise repeatable (diag_det.log: 6/128, ~70/2048, ~490/16384 rows differ between calls; identical once
    the align output is fixed). Inside this context both the engine function and the local copy get the
    same memoized align output for the same topk_ids tensor."""

    def __enter__(self):
        import sglang.srt.layers.moe.fused_moe_triton as FMT
        self.FMT, self.real, cache = FMT, FMT.moe_align_block_size, {}

        def memo(topk_ids, block_size, num_experts, *a, **k):
            key = (topk_ids.data_ptr(), tuple(topk_ids.shape), block_size, num_experts)
            if key not in cache:
                cache[key] = self.real(topk_ids, block_size, num_experts, *a, **k)
            return cache[key]

        FMT.moe_align_block_size = memo
        globals()["moe_align_block_size"] = memo
        return self

    def __exit__(self, *a):
        self.FMT.moe_align_block_size = self.real
        globals()["moe_align_block_size"] = self.real


# ---------------------------------------------------------------- timing helpers
def timeit(f, n=20, reps=5, warm=3):
    for _ in range(warm):
        f()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        for _ in range(n):
            f()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b) / n)
    return sorted(out)[len(out) // 2], out


def peak_bytes(f):
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    o = f()
    torch.cuda.synchronize()
    p = torch.cuda.max_memory_allocated() - base
    del o
    return p


def row(M, name, ms, samples, **kw):
    r = dict(event="time", M=M, cand=name, ms=round(ms, 4), tflops=round(flops(M) / ms / 1e9, 1),
             samples=[round(s, 4) for s in samples])
    r.update(kw)
    log(**r)
    return r


def inputs(M, seed):
    g = torch.Generator(device=dev).manual_seed(1000 + seed)
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16, generator=g)
    ids, tw = routing(M, g)
    return x, ids, tw


# ---------------------------------------------------------------- modes
def run_marlin():
    t0 = time.time()
    L = marlin_layer(s13)
    Lgu, Les = marlin_layer(S13_GU_SWAP), marlin_layer(S13_E_SHIFT)
    log(event="marlin_layers_built", s=round(time.time() - t0, 1),
        weight_bytes=int(sum(p.numel() * p.element_size() for p in L.parameters())))
    # numerics on 64 tokens (seed 0 inputs of M=2048, first 64 rows made contiguous)
    x, ids, tw = inputs(2048, 0)
    xs, ids_s, tw_s = x[:REF_ROWS].contiguous(), ids[:REF_ROWS].contiguous(), tw[:REF_ROWS].contiguous()
    ref = reference(xs, tw_s, ids_s)
    with fixed_align():
        o_m = marlin_serving(L, xs, tw_s, ids_s)
        o_c = c6_call(L, xs, tw_s, ids_s)
    log(event="numerics", rows=REF_ROWS, marlin=round(rel(o_m, ref), 6), c6=round(rel(o_c, ref), 6),
        c6_bitwise_vs_marlin=bool(torch.equal(o_m, o_c)),
        neg_gate_up_scale_swap_marlin=round(rel(marlin_serving(Lgu, xs, tw_s, ids_s), ref), 4),
        neg_gate_up_scale_swap_c6=round(rel(c6_call(Lgu, xs, tw_s, ids_s), ref), 4),
        neg_expert_shift_scale_marlin=round(rel(marlin_serving(Les, xs, tw_s, ids_s), ref), 4),
        neg_expert_shift_scale_c6=round(rel(c6_call(Les, xs, tw_s, ids_s), ref), 4),
        marlin_repeat_bitwise=bool(torch.equal(o_m, marlin_serving(L, xs, tw_s, ids_s))))
    del Lgu, Les
    torch.cuda.empty_cache()

    # bitwise: C6 (empty, explicit NaN poison) vs engine fused_marlin_moe, several seeds, all M
    for M in MS:
        for seed in range(3):
            x, ids, tw = inputs(M, seed)
            unfixed = [marlin_serving(L, x, tw, ids) for _ in range(2)]
            with fixed_align():
                base = marlin_serving(L, x, tw, ids)
                again = marlin_serving(L, x, tw, ids)
                poisoned = c6_call(L, x, tw, ids, poison=True)
                local_zeros = zeros_call(L, x, tw, ids)
                # allocator pre-filled with NaN (no explicit fill inside): release cached blocks, then grab
                # and free one big NaN block so the next call's buffers are carved out of NaN-filled memory
                torch.cuda.empty_cache()
                junk = torch.full((M * TK * K * 2,), float("nan"), device=dev, dtype=torch.bfloat16)
                del junk
                plain = c6_call(L, x, tw, ids)
            log(event="bitwise", M=M, seed=seed,
                unfixed_align_rows_differing=int((unfixed[0] != unfixed[1]).any(1).sum()),
                marlin_repeat=bool(torch.equal(base, again)),
                c6_poisoned=bool(torch.equal(base, poisoned)), c6_after_nan_prefill=bool(torch.equal(base, plain)),
                local_copy_zeros=bool(torch.equal(base, local_zeros)),
                nan_in_poisoned=bool(torch.isnan(poisoned).any()), finite_base=bool(torch.isfinite(base).all()))

    # timing (interleaved per M), peak memory, stage breakdown, moe_sum_reduce alone
    for M in MS:
        x, ids, tw = inputs(M, 0)
        for name, f in [("marlin", lambda: marlin_serving(L, x, tw, ids)),
                        ("c6_empty", lambda: c6_call(L, x, tw, ids)),
                        ("marlin", lambda: marlin_serving(L, x, tw, ids)),
                        ("c6_empty", lambda: c6_call(L, x, tw, ids))]:
            ms, smp = timeit(f)
            row(M, name, ms, smp, peak_gb=round(peak_bytes(f) / 1e9, 3))
        for empty in (False, True):
            stages = []
            for _ in range(23):
                evs = []
                marlin_serving(L, x, tw, ids, fn=fused_marlin_moe_local, empty_cache13=empty, events=evs)
                stages.append(evs)
            torch.cuda.synchronize()
            names = ["align+ws", "alloc(+zero)", "gemm1", "act(clamp torch)", "gemm2", "sum_reduce"]
            per = [[st[i].elapsed_time(st[i + 1]) for st in stages[3:]] for i in range(len(names))]
            log(event="breakdown", M=M, cache13="empty" if empty else "zeros",
                **{n: round(sorted(v)[len(v) // 2], 4) for n, v in zip(names, per)},
                total=round(sum(sorted(v)[len(v) // 2] for v in per), 4))
        c3 = torch.randn(M, TK, K, device=dev, dtype=torch.bfloat16)
        out = torch.empty(M, K, device=dev, dtype=torch.bfloat16)
        ms_sr, smp = timeit(lambda: FM.moe_sum_reduce(c3, out, RSF))
        ms_z, smp_z = timeit(lambda: torch.zeros(M * TK * K, device=dev, dtype=torch.bfloat16))
        log(event="standalone", M=M, moe_sum_reduce_ms=round(ms_sr, 4),
            sum_reduce_GBps=round((c3.numel() + out.numel()) * 2 / ms_sr / 1e6, 0),
            zeros_cache13_ms=round(ms_z, 4), cache13_GB=round(M * TK * K * 2 / 1e9, 3))
        del c3, out
    log(event="done", mode="marlin")


def build_humming(sc13):
    import sglang.srt.layers.quantization.humming_utils as hu
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.humming import HummingRunnerCore
    from sglang.srt.layers.quantization.humming import HummingConfig, HummingMoEMethod

    hu.configure_humming_deepep_dispatch = lambda layer: (setattr(layer, "_humming_uses_deepep_fp8_dispatch", False)
                                                          or False)
    cfg = HummingConfig(full_config={"quant_method": "fp8", "activation_scheme": "dynamic",
                                     "weight_block_size": [B, B]})
    qc = cfg.get_quant_config_for_layer("model.layers.3.mlp.experts", "moe")
    method = HummingMoEMethod(qc)
    layer = torch.nn.Module()
    method.create_weights(layer, num_experts=EG, hidden_size=K, intermediate_size_per_partition=NT,
                          params_dtype=torch.bfloat16, weight_loader=lambda *a, **k: None)
    names = sorted(n for n, _ in layer.named_parameters())
    src = {"w13_weight": q13, "w13_weight_scale_inv": sc13, "w2_weight": q2, "w2_weight_scale_inv": s2}
    assert set(names) == set(src), names
    for n in names:
        p = getattr(layer, n)
        assert tuple(p.shape) == tuple(src[n].shape) and p.dtype == src[n].dtype, (n, p.shape, p.dtype)
        setattr(layer, n, torch.nn.Parameter(src[n].clone(), requires_grad=False))
    layer.locks = torch.zeros(1024, dtype=torch.int32, device=dev)
    layer.hidden_size, layer.intermediate_size_per_partition, layer.params_dtype = K, NT, torch.bfloat16
    t0 = time.time()
    method.process_weights_after_loading(layer)
    torch.cuda.synchronize()
    conv_s = time.time() - t0
    rc = MoeRunnerConfig(num_experts=EG, num_local_experts=EG, hidden_size=K, intermediate_size_per_partition=NT,
                         top_k=TK, params_dtype=torch.bfloat16, activation="silu", is_gated=True,
                         routed_scaling_factor=RSF, swiglu_limit=CLAMP)
    core = HummingRunnerCore(rc)
    core.layer = layer
    meta = {k: str(v) for k, v in layer.humming_metas.items()}
    return layer, core, method, conv_s, meta


def hum_call(layer, core, x, tw, ids, gemm_type):
    from sglang.srt.layers.moe.moe_runner.humming import HummingMoeQuantInfo, HummingRunnerInput
    ri = HummingRunnerInput(hidden_states=x, topk_weights=tw, topk_ids=ids, gemm_type=gemm_type)
    return core.run(ri, HummingMoeQuantInfo(layer=layer), {}).hidden_states


def run_humming():
    from humming.config import GemmType
    t0 = time.time()
    L = marlin_layer(s13)
    log(event="marlin_layer_built", s=round(time.time() - t0, 1))
    H, core, method, conv_s, meta = build_humming(s13)
    log(event="humming_layer_built", convert_s=round(conv_s, 2),
        weight_bytes=int(sum(p.numel() * p.element_size() for p in H.parameters())),
        params={n: [list(p.shape), str(p.dtype)] for n, p in H.named_parameters()}, metas=meta)
    GT = {"hum_idx": GemmType.INDEXED, "hum_grp": GemmType.GROUPED_CONTIGUOUS}

    # JIT compile time: first call per (gemm type, M)
    for M in MS:
        x, ids, tw = inputs(M, 0)
        for name, gt in GT.items():
            torch.cuda.synchronize()
            t1 = time.time()
            try:
                o = hum_call(H, core, x, tw, ids, gt)
                torch.cuda.synchronize()
                log(event="first_call", M=M, cand=name, s=round(time.time() - t1, 2), finite=bool(torch.isfinite(o).all()))
            except Exception as ex:  # report and continue with the other type
                log(event="first_call_fail", M=M, cand=name, err=repr(ex)[:2000])
                GT[name] = None
        GT = {k: v for k, v in GT.items() if v is not None}

    # numerics: 64 tokens vs fp32 reference; negative controls built through the same Humming path
    x, ids, tw = inputs(2048, 0)
    xs, ids_s, tw_s = x[:REF_ROWS].contiguous(), ids[:REF_ROWS].contiguous(), tw[:REF_ROWS].contiguous()
    ref = reference(xs, tw_s, ids_s)
    o_m = marlin_serving(L, xs, tw_s, ids_s)
    res = dict(event="numerics", rows=REF_ROWS, marlin=round(rel(o_m, ref), 6))
    for name, gt in GT.items():
        o_h = hum_call(H, core, xs, tw_s, ids_s, gt)
        res[name] = round(rel(o_h, ref), 6)
        res[name + "_vs_marlin"] = round(rel(o_h, o_m.float()), 6)
        res[name + "_repeat_bitwise"] = bool(torch.equal(o_h, hum_call(H, core, xs, tw_s, ids_s, gt)))
    for tag, sc in (("gate_up_scale_swap", S13_GU_SWAP), ("expert_shift_scale", S13_E_SHIFT)):
        Hb, coreb, _, _, _ = build_humming(sc)
        for name, gt in GT.items():
            res[f"neg_{tag}_{name}"] = round(rel(hum_call(Hb, coreb, xs, tw_s, ids_s, gt), ref), 4)
        del Hb, coreb
        torch.cuda.empty_cache()
    log(**res)

    # large-M agreement with Marlin (no fp32 reference at this size)
    for M in MS:
        x, ids, tw = inputs(M, 0)
        o_m = marlin_serving(L, x, tw, ids)
        for name, gt in GT.items():
            o_h = hum_call(H, core, x, tw, ids, gt)
            log(event="agree", M=M, cand=name, rel_vs_marlin=round(rel(o_h, o_m.float()), 6),
                finite=bool(torch.isfinite(o_h).all()))

    # CUDA graph capture + replay at the smallest M (decode-like), both candidates in this process
    M = MS[0]
    x, ids, tw = inputs(M, 0)
    gcands = [("marlin", lambda: marlin_serving(L, x, tw, ids))]
    gcands += [(n, lambda gt=gt: hum_call(H, core, x, tw, ids, gt)) for n, gt in GT.items()]
    for name, f in gcands:
        try:
            ref_out = f().clone()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    f()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                gout = f()
            g.replay()
            torch.cuda.synchronize()
            ms, smp = timeit(g.replay)
            log(event="graph", M=M, cand=name, ms=round(ms, 4), samples=[round(v, 4) for v in smp],
                replay_bitwise_vs_eager=bool(torch.equal(gout, ref_out)))
            del g
        except Exception as ex:
            log(event="graph_fail", M=M, cand=name, err=repr(ex)[:2000])

    # timing, interleaved with Marlin in the same process
    for M in MS:
        x, ids, tw = inputs(M, 0)
        cands = [("marlin", lambda: marlin_serving(L, x, tw, ids))]
        for name, gt in GT.items():
            cands.append((name, lambda gt=gt: hum_call(H, core, x, tw, ids, gt)))
        for name, f in cands + cands:
            ms, smp = timeit(f)
            row(M, name, ms, smp, peak_gb=round(peak_bytes(f) / 1e9, 3))
    log(event="done", mode="humming")


def run_contract():
    """Routed ids of -1 (padded rows masked by _mask_topk_ids_padded_region, enabled when ep_size>1 or
    under the FULL prefill CUDA graph) are skipped by Marlin; zeros keep those rows' output finite."""
    L = marlin_layer(s13)
    for M in MS:
        g = torch.Generator(device=dev).manual_seed(7)
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16, generator=g)
        ids, tw = routing(M, g, masked_rows=16)
        with fixed_align():
            base = marlin_serving(L, x, tw, ids)
            pois = c6_call(L, x, tw, ids, poison=True)
        live = slice(0, M - 16)
        log(event="contract_masked_ids", M=M, masked_rows=16,
            live_rows_bitwise=bool(torch.equal(base[live], pois[live])),
            zeros_padded_rows_all_zero=bool((base[M - 16:] == 0).all()),
            empty_padded_rows_nan=bool(torch.isnan(pois[M - 16:]).any()))
    log(event="done", mode="contract")


def m172c6_call(layer, x, tw, ids, **kw):  # C6 plus engine 172's fused clamped SwiGLU (off in serving today)
    FM._AX_FUSE_CLAMPED_SWIGLU = True
    try:
        return c6_call(layer, x, tw, ids, **kw)
    finally:
        FM._AX_FUSE_CLAMPED_SWIGLU = False


def run_extra():
    """Follow-ups: where Humming's first call spends its time with a warm kernel cache; numerics of every
    candidate at the kernel configs large M actually uses (first 64 rows of the full-M output vs fp32);
    decode-sized M; Marlin+172+C6 as the best Marlin variant."""
    import cProfile
    import io
    import pstats

    from humming.config import GemmType
    L = marlin_layer(s13)
    H, core, _, conv_s, _ = build_humming(s13)
    x, ids, tw = inputs(128, 0)
    pr = cProfile.Profile()
    t1 = time.time()
    pr.enable()
    hum_call(H, core, x, tw, ids, GemmType.INDEXED)
    torch.cuda.synchronize()
    pr.disable()
    first = time.time() - t1
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(30)
    open(os.path.join(ALT, "first_call_profile.txt"), "w").write(buf.getvalue())
    log(event="first_call_warm_cache", cand="hum_idx", s=round(first, 2), profile="alt0/first_call_profile.txt")
    log(event="humming_tuning", configs={k: v["w13_tuning_config"] for k, v in core.humming_gemm_configs.items()},
        w2={k: v["w2_tuning_config"] for k, v in core.humming_gemm_configs.items()})
    GI = GemmType.INDEXED
    cands = {"marlin": lambda x, tw, ids: marlin_serving(L, x, tw, ids),
             "c6_empty": lambda x, tw, ids: c6_call(L, x, tw, ids),
             "m172_c6": lambda x, tw, ids: m172c6_call(L, x, tw, ids),
             "hum_idx": lambda x, tw, ids: hum_call(H, core, x, tw, ids, GI),
             "hum_grp": lambda x, tw, ids: hum_call(H, core, x, tw, ids, GemmType.GROUPED_CONTIGUOUS)}
    for M in [64, 128, 2048, 8192, 16384]:
        x, ids, tw = inputs(M, 3)
        ref = reference(x[:REF_ROWS], tw[:REF_ROWS], ids[:REF_ROWS])
        with fixed_align():
            outs = {n: f(x, tw, ids) for n, f in cands.items()}
        log(event="numerics_bigM", M=M, rows=f"first {REF_ROWS} of {M}",
            **{n: round(rel(o[:REF_ROWS], ref), 6) for n, o in outs.items()},
            m172_c6_bitwise_vs_marlin=bool(torch.equal(outs["m172_c6"], outs["marlin"])))
    for M in [32, 64]:
        x, ids, tw = inputs(M, 0)
        for name in ("marlin", "hum_idx", "marlin", "hum_idx"):
            f = (lambda n=name: cands[n](x, tw, ids))
            ms, smp = timeit(f)
            g = torch.cuda.CUDAGraph()
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(3):
                    f()
            torch.cuda.current_stream().wait_stream(st)
            with torch.cuda.graph(g):
                f()
            gms, gsmp = timeit(g.replay)
            row(M, name, ms, smp, graph_ms=round(gms, 4), graph_samples=[round(v, 4) for v in gsmp])
            del g
    for M in [2048, 8192, 16384]:
        x, ids, tw = inputs(M, 0)
        for name in ("marlin", "m172_c6", "hum_idx", "marlin", "m172_c6", "hum_idx"):
            f = (lambda n=name: cands[n](x, tw, ids))
            ms, smp = timeit(f)
            row(M, name, ms, smp, peak_gb=round(peak_bytes(f) / 1e9, 3))
    log(event="done", mode="extra")


{"marlin": run_marlin, "humming": run_humming, "contract": run_contract, "extra": run_extra}[MODE]()
