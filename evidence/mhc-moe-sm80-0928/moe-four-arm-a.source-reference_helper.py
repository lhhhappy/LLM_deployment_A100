#!/usr/bin/env python3
"""117 on one A100: block-FP8 MoE experts through the engine's FusedMoE layer, Humming vs Marlin.

One TP=1 FusedMoE with GLM-5.3-Flash's TP8 per-rank expert shape: 288 routed experts + the shared
expert fused as expert 288 (top-k 8+1), hidden 4096, intermediate 256, FP8 e4m3 weights with
128x128 fp32 block scales, swiglu_limit 10, routed_scaling_factor 2.5; random weights and routing.
Everything behind the switch is real engine code: Fp8Config.get_quant_method, FusedMoE.weight_loader,
process_weights_after_loading, FusedMoE.forward (dispatch, quant method, combine).

Checks (one JSON object per line; exits nonzero on the first failed assertion):
  numerics   switch off (111 Marlin) and on (117 Humming) vs an fp32 reference and vs each other at
             M = 1, 32, 128, 2048, 8192, incl. inputs that drive the clamp; negative controls
  graph      CUDA graph capture and replay with fresh inputs at decode sizes, both paths
  timing     layer-path wall/host/GPU time and transient memory at M = 128, 2048, 8192, 16384
  identity   switch off vs the reference commit's fp8.py (--base-fp8): same method, bitwise-equal
             weights after loading and bitwise-equal outputs with the routing alignment held fixed
Also: expert parallelism with the switch on is refused; run-to-run repeatability is reported.
Not covered: TP8 communication, real checkpoint weights and routing, MTP acceptance.

Usage (one GPU; candidate engine and humming-kernels 0.1.12 on PYTHONPATH, HUMMING_CACHE_DIR set):
  python test_fp8_moe_humming_117.py --base-fp8 <fp8.py of the reference commit> [--only numerics,graph,...]
"""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

E_ROUTED, TOP_K, HIDDEN, INTER, BLOCK = 288, 8, 4096, 256, 128
E, TK = E_ROUTED + 1, TOP_K + 1  # the shared expert is fused as expert 288, selected by every token
LIMIT, RSF = 10.0, 2.5
SWITCH = "SGLANG_AX_SM80_FP8_MOE_HUMMING"
QUANT_CONFIG = {"quant_method": "fp8", "activation_scheme": "dynamic", "fmt": "e4m3",
                "weight_block_size": [BLOCK, BLOCK]}  # GLM-5.3-Flash config.json
DEV = "cuda"


def emit(**kw):
    print(json.dumps(kw), flush=True)


def check(ok, **kw):
    emit(kind="check", ok=bool(ok), **kw)
    if not ok:
        raise AssertionError(kw)


# ------------------------------------------------------------------ runtime (single rank, TP=1)
def open_runtime(stack):
    from sglang.srt.distributed import init_distributed_environment, initialize_model_parallel
    from sglang.srt.layers.moe import MoeA2ABackend, MoeRunnerBackend
    from sglang.srt.runtime_context import get_context, get_flags, get_parallel

    stack.enter_context(get_context().override_server_args(model_path="dummy"))
    stack.enter_context(get_flags().moe.override(runner_backend=MoeRunnerBackend.AUTO,
                                                 a2a_backend=MoeA2ABackend.NONE))
    stack.enter_context(get_parallel().override(moe_ep_size=1, moe_ep_rank=0, moe_tp_size=1, moe_tp_rank=0,
                                                tp_size=1, tp_rank=0))
    init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="nccl",
                                 distributed_init_method=f"tcp://127.0.0.1:{29500 + os.getpid() % 1000}")
    initialize_model_parallel(tensor_model_parallel_size=1)


# ------------------------------------------------------------------ checkpoint (rank-0 slice)
def quantize_blocks(w):
    """[e, n, k] float -> fp8 e4m3 weights and fp32 per-128x128-block scale_inv, as in the checkpoint."""
    e, n, k = w.shape
    blocks = w.float().view(e, n // BLOCK, BLOCK, k // BLOCK, BLOCK)
    scale = blocks.abs().amax(dim=(2, 4)).clamp(min=1e-8) / 448.0
    return (blocks / scale[:, :, None, :, None]).view(e, n, k).to(torch.float8_e4m3fn), scale


def dequantize(q, s):
    e, n, k = q.shape
    return (q.float().view(e, n // BLOCK, BLOCK, k // BLOCK, BLOCK) * s[:, :, None, :, None]).view(e, n, k)


def make_checkpoint(seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    w13 = torch.empty(E, 2 * INTER, HIDDEN, dtype=torch.float8_e4m3fn, device=DEV)  # [gate; up]
    s13 = torch.empty(E, 2 * INTER // BLOCK, HIDDEN // BLOCK, device=DEV)
    w2 = torch.empty(E, HIDDEN, INTER, dtype=torch.float8_e4m3fn, device=DEV)
    s2 = torch.empty(E, HIDDEN // BLOCK, INTER // BLOCK, device=DEV)
    for e in range(E):  # rows get different magnitudes so block scales differ within an expert
        a = torch.randn(1, 2 * INTER, HIDDEN, device=DEV, generator=g) * 0.02
        b = torch.randn(1, HIDDEN, INTER, device=DEV, generator=g) * 0.02
        a = a * (1 + 3 * torch.rand(1, 2 * INTER, 1, device=DEV, generator=g))
        b = b * (1 + 3 * torch.rand(1, HIDDEN, 1, device=DEV, generator=g))
        w13[e:e + 1], s13[e:e + 1] = quantize_blocks(a)
        w2[e:e + 1], s2[e:e + 1] = quantize_blocks(b)
    return dict(w13=w13, s13=s13, w2=w2, s2=s2)


def build_layer(ckpt, humming, fp8_module=None):
    """FusedMoE built, loaded and post-processed the way the model loader does it."""
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    fp8_module = fp8_module or importlib.import_module("sglang.srt.layers.quantization.fp8")
    os.environ[SWITCH] = "1" if humming else "0"
    with torch.device(DEV):  # the model loader builds modules under the target device
        layer = FusedMoE(num_experts=E, top_k=TK, hidden_size=HIDDEN, intermediate_size=INTER, layer_id=3,
                         num_fused_shared_experts=1, params_dtype=torch.bfloat16,
                         quant_config=fp8_module.Fp8Config.from_config(QUANT_CONFIG),
                         prefix="model.layers.3.mlp.experts", routed_scaling_factor=RSF, swiglu_limit=LIMIT)
    rows = INTER // BLOCK
    for e in range(E):
        for shard, name, w, s in (("w1", "gate_proj", ckpt["w13"][e, :INTER], ckpt["s13"][e, :rows]),
                                  ("w3", "up_proj", ckpt["w13"][e, INTER:], ckpt["s13"][e, rows:]),
                                  ("w2", "down_proj", ckpt["w2"][e], ckpt["s2"][e])):
            param = "w2" if shard == "w2" else "w13"
            layer.weight_loader(getattr(layer, f"{param}_weight"), w, f"experts.{e}.{name}.weight", shard, e)
            layer.weight_loader(getattr(layer, f"{param}_weight_scale_inv"), s,
                                f"experts.{e}.{name}.weight_scale_inv", shard, e)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    start = time.perf_counter()
    layer.quant_method.process_weights_after_loading(layer)
    torch.cuda.synchronize()
    process_s = time.perf_counter() - start
    tensors = list(layer.parameters()) + list(layer.buffers())
    emit(kind="layer", humming=humming, method=type(layer.quant_method).__name__, process_s=round(process_s, 3),
         weight_bytes=sum(t.numel() * t.element_size() for t in tensors),
         load_transient_mb=round((torch.cuda.max_memory_allocated() - before) / 2**20, 1),
         params={n: [list(p.shape), str(p.dtype)] for n, p in layer.named_parameters()})
    return layer


# ------------------------------------------------------------------ inputs, forward, reference
def make_inputs(m, seed, x_scale=1.0):
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    g = torch.Generator(device=DEV).manual_seed(1000 + seed)
    x = (torch.randn(m, HIDDEN, device=DEV, generator=g) * x_scale).to(torch.bfloat16)
    routed = torch.randn(m, E_ROUTED, device=DEV, generator=g).topk(TOP_K, dim=1).indices.int()
    weights = torch.softmax(torch.randn(m, TOP_K, device=DEV, generator=g), -1)
    # What biased_grouped_topk emits with a fused shared expert: renormalized routed weights and
    # the shared expert as the last column with weight 1/routed_scaling_factor.
    ids = torch.cat([routed, torch.full((m, 1), E_ROUTED, dtype=torch.int32, device=DEV)], 1).contiguous()
    tw = torch.cat([weights, torch.full((m, 1), 1.0 / RSF, device=DEV)], 1).float().contiguous()
    return x, StandardTopKOutput(topk_weights=tw, topk_ids=ids, router_logits=torch.empty(m, E_ROUTED, device=DEV))


class Reference:
    """fp32 MoE on the dequantized checkpoint: clamp + SiLU, top-k weights, routed scaling once."""

    def __init__(self, ckpt):
        self.d13 = dequantize(ckpt["w13"], ckpt["s13"])
        self.d2 = dequantize(ckpt["w2"], ckpt["s2"])

    def __call__(self, x, topk, clamp=True, rsf=RSF, drop_shared=False):
        x, ids, tw = x.float(), topk.topk_ids, topk.topk_weights.float()
        out = torch.zeros(x.shape[0], HIDDEN, device=DEV)
        for e in ids.unique().tolist():
            if drop_shared and e == E_ROUTED:
                continue
            rows, slots = (ids == e).nonzero(as_tuple=True)
            h = x[rows] @ self.d13[e].T
            gate, up = h[:, :INTER], h[:, INTER:]
            if clamp:
                gate, up = gate.clamp(max=LIMIT), up.clamp(min=-LIMIT, max=LIMIT)
            out.index_add_(0, rows, tw[rows, slots, None] * ((F.silu(gate) * up) @ self.d2[e].T))
        return out * rsf


def rel(out, ref):
    return float((out.float() - ref.float()).norm() / ref.float().norm())


@contextlib.contextmanager
def fixed_alignment():
    """Hold moe_align_block_size's output fixed per routing tensor.

    The alignment kernel orders tokens within an expert with atomics, so the same MoE call is not
    bitwise repeatable; inside this context every call on the same topk_ids reuses the first result.
    """
    import sglang.srt.layers.moe.fused_moe_triton as fmt

    real, memo = fmt.moe_align_block_size, {}

    def memoized(topk_ids, block_size, num_experts, *args, **kwargs):
        key = (topk_ids.data_ptr(), tuple(topk_ids.shape), block_size, num_experts, args,
               tuple(sorted(kwargs.items())))
        if key not in memo:
            memo[key] = real(topk_ids, block_size, num_experts, *args, **kwargs)
        return memo[key]

    fmt.moe_align_block_size = memoized
    try:
        yield
    finally:
        fmt.moe_align_block_size = real


# ------------------------------------------------------------------ checks
def run_numerics(layers, ref, bad_layer):
    for m in (1, 32, 128, 2048, 8192):
        for tag, x_scale in (("normal", 1.0), ("clamp_heavy", 4.0)):
            if tag == "clamp_heavy" and m not in (128, 2048):
                continue
            x, topk = make_inputs(m, seed=m, x_scale=x_scale)
            r = ref(x, topk)
            outs = {name: layer(x, topk) for name, layer in layers.items()}
            err = {name: rel(o, r) for name, o in outs.items()}
            row = dict(kind="numerics", M=m, inputs=tag, marlin=round(err["marlin"], 6),
                       humming=round(err["humming"], 6), humming_vs_marlin=round(rel(outs["humming"], outs["marlin"]), 6),
                       finite=all(bool(torch.isfinite(o).all()) for o in outs.values()))
            h = outs["humming"]
            # Negative controls: wrong semantics must be far from what the layer returns.
            row["neg_no_shared_expert"] = round(rel(h, ref(x, topk, drop_shared=True)), 4)
            row["neg_no_routed_scaling"] = round(rel(h, ref(x, topk, rsf=1.0)), 4)
            if tag == "clamp_heavy":
                row["neg_no_clamp"] = round(rel(h, ref(x, topk, clamp=False)), 4)
            if m in (32, 2048):
                row["neg_expert_shifted_scales"] = round(rel(bad_layer(x, topk), r), 4)
            emit(**row)
            if m == 2048 and tag == "normal":
                emit(kind="repeatability", M=m, **{name: repeat_rows_differing(layer, x, topk)
                                                   for name, layer in layers.items()})
            check(row["finite"], M=m, inputs=tag, what="finite outputs")
            check(err["marlin"] < 2e-2 and err["humming"] < 2e-2, M=m, inputs=tag, what="error vs fp32 reference")
            check(err["humming"] <= 1.25 * err["marlin"] + 1e-4, M=m, inputs=tag, what="humming error within 1.25x marlin")
            negatives = {k: v for k, v in row.items() if k.startswith("neg_")}
            check(all(v > 0.1 for v in negatives.values()), M=m, inputs=tag, what="negative controls rejected",
                  **negatives)


def repeat_rows_differing(layer, x, topk):
    """Rows that differ between two identical calls, with the routing alignment free and held fixed."""
    free = [layer(x, topk) for _ in range(2)]
    with fixed_alignment():
        fixed = [layer(x, topk) for _ in range(2)]
    return dict(free=int((free[0] != free[1]).any(1).sum()), fixed_alignment=int((fixed[0] != fixed[1]).any(1).sum()),
                rows=x.shape[0])


def run_refusal(ckpt):
    """Switch on with expert parallelism must refuse at layer construction, naming the reason."""
    from sglang.srt.runtime_context import get_parallel

    try:
        with get_parallel().override(moe_ep_size=2, moe_ep_rank=0):
            build_layer(ckpt, humming=True)
    except ValueError as err:
        check("expert parallelism (moe_ep_size=2)" in str(err), what="EP refused with a clear error", error=str(err))
    else:
        check(False, what="EP refused with a clear error", error=None)


def capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    return graph, out


def run_graph(layers, ref):
    for m in (1, 8, 32, 64, 128):
        x_static, topk_static = make_inputs(m, seed=500 + m)
        graphs = {name: capture(lambda layer=layer: layer(x_static, topk_static)) for name, layer in layers.items()}
        stale = {}  # outputs for the capture-time inputs, to show that later replays recompute
        for name, (graph, out) in graphs.items():
            graph.replay()
            stale[name] = out.clone()
        for trial in range(2):  # replay on fresh inputs copied into the captured buffers
            x, topk = make_inputs(m, seed=600 + 10 * m + trial)
            x_static.copy_(x)
            topk_static.topk_ids.copy_(topk.topk_ids)
            topk_static.topk_weights.copy_(topk.topk_weights)
            r = ref(x, topk)
            row = dict(kind="graph", M=m, trial=trial)
            for name, (graph, out) in graphs.items():
                graph.replay()
                eager = layers[name](x, topk)
                row[name] = dict(replay=round(rel(out, r), 6), eager=round(rel(eager, r), 6),
                                 replay_vs_eager=round(rel(out, eager), 6),
                                 stale_output=round(rel(stale[name], r), 4))
                check(row[name]["replay"] <= 1.25 * row[name]["eager"] + 1e-4 and row[name]["stale_output"] > 0.1,
                      M=m, trial=trial, path=name, what="graph replay recomputes fresh inputs", **row[name])
            emit(**row)
        timing = {}
        for name, (graph, _) in list(graphs.items()) * 2:
            timing.setdefault(name, []).append(measure(graph.replay)["gpu_ms"])
        emit(kind="graph_time", M=m, **{name: round(statistics.median(v), 4) for name, v in timing.items()})
        del graphs


def measure(fn, inner=20, repeats=5):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    host, wall, gpu = [], [], []
    for _ in range(repeats):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        a.record()
        for _ in range(inner):
            fn()
        t1 = time.perf_counter()
        b.record()
        b.synchronize()
        t2 = time.perf_counter()
        host.append((t1 - t0) * 1e3 / inner)
        wall.append((t2 - t0) * 1e3 / inner)
        gpu.append(a.elapsed_time(b) / inner)
    return dict(host_ms=statistics.median(host), wall_ms=statistics.median(wall), gpu_ms=statistics.median(gpu))


def transient_mb(fn):
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    del out
    return round(peak / 2**20, 1)


def run_timing(layers):
    from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend
    from sglang.srt.layers.moe.moe_runner.humming import HummingMoeQuantInfo
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput

    for m in (128, 2048, 8192, 16384):
        x, topk = make_inputs(m, seed=900 + m)
        res = {}
        for name, layer in list(layers.items()) * 2:  # ABAB
            res.setdefault(name, []).append(measure(lambda layer=layer: layer(x, topk)))
        row = dict(kind="timing", M=m)
        for name, samples in res.items():
            row[name] = {k: round(statistics.median(s[k] for s in samples), 4) for k in samples[0]}
            row[name]["transient_mb"] = transient_mb(lambda: layers[name](x, topk))
        row["gpu_speedup"] = round(row["marlin"]["gpu_ms"] / row["humming"]["gpu_ms"], 3)
        emit(**row)
    # Host cost of the base ("none", "humming") fused func, which builds a runner core per call,
    # against this method's persistent core, both on the same converted layer.
    layer = layers["humming"]
    base_runner = MoeRunner(MoeRunnerBackend.HUMMING, layer.moe_runner_config)
    x, topk = make_inputs(128, seed=77)
    dispatch = StandardDispatchOutput(hidden_states=x, hidden_states_scale=None, topk_output=topk)
    quant_info = HummingMoeQuantInfo(layer=layer)
    emit(kind="host_overhead", M=128,
         persistent_core=measure(lambda: layer.quant_method.apply(layer, dispatch)),
         base_per_call_core=measure(lambda: base_runner.run(dispatch, quant_info)))


def run_identity(ckpt, marlin_layer, base_fp8_path):
    """Switch off must equal the reference commit's Fp8MoEMethod: same method, weights and outputs."""
    spec = importlib.util.spec_from_file_location("sglang.srt.layers.quantization._fp8_reference", base_fp8_path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    ref_layer = build_layer(ckpt, humming=False, fp8_module=base)
    mine = marlin_layer.quant_method
    check(type(mine).__name__ == "Fp8MoEMethod" and mine.ax_sm80_marlin and not mine.ax_sm80_humming
          and type(ref_layer.quant_method).__name__ == "Fp8MoEMethod" and ref_layer.quant_method.ax_sm80_marlin
          and mine.runner.runner_backend == ref_layer.quant_method.runner.runner_backend,
          what="switch off selects the reference method and Marlin runner",
          fp8_sha256=hashlib.sha256(open(base_fp8_path, "rb").read()).hexdigest()[:16])
    mine_params, ref_params = dict(marlin_layer.named_parameters()), dict(ref_layer.named_parameters())
    check(mine_params.keys() == ref_params.keys()
          and all(torch.equal(mine_params[n].view(torch.uint8), ref_params[n].view(torch.uint8)) for n in ref_params),
          what="bitwise-equal weights after loading", params=sorted(ref_params))
    for m in (1, 32, 128, 2048, 8192):
        x, topk = make_inputs(m, seed=3000 + m)
        free = [marlin_layer(x, topk) for _ in range(2)]
        with fixed_alignment():
            a, b = marlin_layer(x, topk), ref_layer(x, topk)
        emit(kind="identity", M=m, bitwise_equal_fixed_alignment=bool(torch.equal(a, b)),
             repeat_rows_differing_free_alignment=int((free[0] != free[1]).any(1).sum()))
        check(torch.equal(a, b), M=m, what="bitwise-equal outputs vs reference fp8.py")
    del ref_layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-fp8", required=True, help="fp8.py of the reference commit for the identity check")
    ap.add_argument("--only", default="numerics,graph,timing,identity")
    args = ap.parse_args()
    only = set(args.only.split(","))
    torch.cuda.set_device(0)
    emit(kind="start", torch=torch.__version__, gpu=torch.cuda.get_device_name(0), argv=sys.argv[1:],
         commit=os.environ.get("TREE_COMMIT"))
    with contextlib.ExitStack() as stack:
        open_runtime(stack)
        ckpt = make_checkpoint()
        layers = {}
        for name in ("humming", "marlin"):  # startup cost of the first layer, then its first forward
            start = time.perf_counter()
            layers[name] = build_layer(ckpt, humming=name == "humming")
            built = time.perf_counter()
            x, topk = make_inputs(4096, seed=1)
            layers[name](x, topk)
            torch.cuda.synchronize()
            emit(kind="first_layer", path=name, build_s=round(built - start, 2),
                 first_forward_M4096_s=round(time.perf_counter() - built, 3))
        check(type(layers["humming"].quant_method).__name__ == "Fp8HummingMoEMethod"
              and type(layers["marlin"].quant_method).__name__ == "Fp8MoEMethod", what="switch selects the method")
        run_refusal(ckpt)
        ref = Reference(ckpt)
        if "numerics" in only:
            bad = dict(ckpt, s13=ckpt["s13"].roll(1, dims=0).contiguous())  # a converter reading the wrong expert
            bad_layer = build_layer(bad, humming=True)
            run_numerics(layers, ref, bad_layer)
            del bad_layer
            torch.cuda.empty_cache()
        if "graph" in only:
            run_graph(layers, ref)
        if "timing" in only:
            run_timing(layers)
        if "identity" in only:
            run_identity(ckpt, layers["marlin"], args.base_fp8)
    emit(kind="done", ok=True)


if __name__ == "__main__":
    main()
