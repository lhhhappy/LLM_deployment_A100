"""Full routed MoE decode at the frozen GLM TP8 per-card shape on one A100."""
import argparse
import contextlib
import copy
import importlib.util
import json
import os
import statistics
import time
from pathlib import Path

import torch


def emit(**record):
    print(json.dumps(record), flush=True)


def graph_timing(fn, repeats=7, iterations=30):
    for _ in range(8):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    for _ in range(10):
        graph.replay()
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iterations)
    return dict(median_us=statistics.median(samples), samples_us=samples), graph, output


def selected(table, tokens):
    matches = [config for lo, hi, config in table if lo < tokens * 9 <= hi]
    assert len(matches) == 1
    return copy.deepcopy(matches[0])


def tune_stages(layer, core, helper, reference, inputs, topk, records):
    """Change only runtime schedules; retain weights, routing and GEMM row height."""
    from humming.config import GemmType
    from humming.layer import HummingMethod
    buffers = core.prepare_buffers(inputs, topk.topk_ids, GemmType.INDEXED)
    kwargs = dict(zip(("w13", "w2"), core._prepare_indexed_gemm_kwargs(topk.topk_ids)))
    HummingMethod.forward_layer(layer=layer, inputs=inputs, outputs=buffers["gate_up_output"], sublayer_name="w13", **kwargs["w13"])
    core.apply_activation(buffers["gate_up_output"], buffers["activation_output"])
    batch = inputs.shape[0]
    ids = topk.topk_ids.reshape(-1)
    oracle_up = torch.zeros(batch * 9, 512, device="cuda")
    oracle_down = torch.zeros(batch * 9, 4096, device="cuda")
    for expert in ids.unique().tolist():
        rows = (ids == expert).nonzero().flatten()
        oracle_up[rows] = inputs[rows // 9].float() @ reference.d13[expert].T
        oracle_down[rows] = buffers["activation_output"][rows].float() @ reference.d2[expert].T
    tables = core.get_humming_gemm_configs(GemmType.INDEXED)
    schedules = []
    for n, k, stages, ctas in ((256,128,4,1),(256,128,3,1),(256,128,2,1),(256,128,2,2),
                              (256,64,3,1),(256,64,3,2),(256,64,4,2),
                              (128,128,4,1),(128,128,3,2),(128,128,2,2),
                              (128,64,4,2),(128,64,3,2),(128,64,2,2),
                              (64,128,4,2),(64,128,3,2),(64,64,4,2)):
        schedules.append(dict(block_shape=[16,n,k], warp_shape=[16,64 if n==256 else 32,64],
                              num_stages=stages, num_ctas_per_sm=ctas))
    for stage in ("w13", "w2"):
        native = selected(tables[stage+"_tuning_config"], batch)
        oracle = oracle_up if stage == "w13" else oracle_down
        source = inputs if stage == "w13" else buffers["activation_output"]
        target = buffers["gate_up_output"] if stage == "w13" else buffers["down_output"].view(-1,4096)
        for schedule in schedules:
            for stream_k in ((True,False) if stage == "w13" else (False,)):
                config = dict(native, **schedule, use_stream_k=stream_k)
                kw = dict(kwargs[stage], tuning_config=json.dumps([(0,2**31-1,config)]))
                def launch():
                    return HummingMethod.forward_layer(layer=layer, inputs=source, outputs=target, sublayer_name=stage, **kw)
                try:
                    launch()
                    torch.cuda.synchronize()
                    relative = helper.rel(target, oracle)
                    assert bool(target.isfinite().all()) and relative < 0.025, relative
                    timing, _, _ = graph_timing(launch, repeats=5, iterations=30)
                    record = dict(type="stage_candidate", stage=stage, batch=batch, config=config,
                                  reference_relative_l2=relative, **timing)
                except (ValueError, RuntimeError) as error:
                    record = dict(type="stage_candidate_rejected", stage=stage, batch=batch, config=config,
                                  error=str(error)[:500])
                records.append(record)
                emit(**record)


def full_candidates(layer, core, helper, reference, router, gate_weight, bias, records, batches=(24,28,32,40,48), explore=False):
    """Capture independent schedules, then rotate replay order on identical inputs."""
    from humming.config import GemmType
    from sglang.kernels.ops.attention.dsv4 import linear_bf16_fp32
    original = core.get_humming_gemm_configs
    native_tables = original(GemmType.INDEXED)
    native_up, native_down = selected(native_tables["w13_tuning_config"],32), selected(native_tables["w2_tuning_config"],32)
    best_up = dict(native_up, block_shape=[16,128,128], warp_shape=[16,32,64], num_stages=3, num_ctas_per_sm=2)
    best_down = dict(native_down, block_shape=[16,256,64], num_stages=4, num_ctas_per_sm=2)
    options = {"native": (native_up,native_down), "down64": (native_up,best_down), "up128_down64": (best_up,best_down)}
    try:
        for batch in batches:
            batch_up,batch_down=selected(native_tables["w13_tuning_config"],batch),selected(native_tables["w2_tuning_config"],batch)
            eligible=(batch_down==native_down)
            options={"native":(batch_up,batch_down),"down64":(batch_up,best_down if eligible else batch_down)}
            if explore:
                options["down_cta3"]=(batch_up,dict(best_down,num_stages=2,num_ctas_per_sm=3))
                options["down_cta4"]=(batch_up,dict(best_down,num_stages=2,num_ctas_per_sm=4))
            torch.manual_seed(batch+1900)
            inputs = torch.randn(batch,4096,device="cuda",dtype=torch.bfloat16)
            initial = inputs.clone()
            graphs, outputs = {}, {}
            for name,(up,down) in options.items():
                table = dict(native_tables)
                for stage,config in (("w13",up),("w2",down)):
                    entries = [(0,2**31-1,config)]
                    table[stage+"_tuning_config"] = entries
                    table[stage+"_tuning_config_str"] = json.dumps(entries)
                core.get_humming_gemm_configs = lambda gt,t=table: t
                def full():
                    topk = router(inputs,linear_bf16_fp32(inputs,gate_weight))
                    return layer(inputs,topk)
                capture_start=time.monotonic()
                _,graphs[name],outputs[name] = graph_timing(full,repeats=1,iterations=10)
                capture_record=dict(type="capture_cost",batch=batch,candidate=name,seconds=time.monotonic()-capture_start)
                records.append(capture_record);emit(**capture_record)
            saved_bias=bias.clone()
            for distribution in ("normal","skew","clamp"):
                inputs.copy_(initial)
                bias.copy_(saved_bias)
                if distribution=="skew": bias[:8].add_(100)
                if distribution=="clamp": inputs.mul_(50)
                topk = router(inputs,linear_bf16_fp32(inputs,gate_weight))
                expected=reference(inputs,topk)
                errors={}
                for name in options:
                    graphs[name].replay()
                    torch.cuda.synchronize()
                    errors[name]=helper.rel(outputs[name],expected)
                    assert bool(outputs[name].isfinite().all()) and errors[name]<0.025,(name,errors)
                assert all(x<=errors["native"]*1.20+0.0003 for x in errors.values()),errors
                # Warm all schedules together to avoid startup clocks skewing the comparison.
                for _ in range(80):
                    for graph in graphs.values(): graph.replay()
                times={name:[] for name in options}
                names=list(options)
                for rep in range(12):
                    order=names[rep%len(names):]+names[:rep%len(names)]
                    for name in order:
                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(50): graphs[name].replay()
                        end.record(); end.synchronize()
                        times[name].append(start.elapsed_time(end)*1000/50)
                for name,(up,down) in options.items():
                    record=dict(type="full_candidate",batch=batch,distribution=distribution,candidate=name,
                                unique_routed_experts=int(topk.topk_ids[:,:-1].unique().numel()),
                                reference_relative_l2=errors[name],median_us=statistics.median(times[name]),
                                samples_us=times[name],up_config=up,down_config=down)
                    records.append(record);emit(**record)
            bias.copy_(saved_bias)
    finally:
        core.get_humming_gemm_configs=original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--profile-dir", required=True)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--full-candidates", action="store_true")
    parser.add_argument("--batches", default="24,28,32,40,48")
    parser.add_argument("--explore", action="store_true")
    args = parser.parse_args()
    torch.cuda.set_device(0)
    assert torch.cuda.get_device_capability() == (8, 0)
    spec = importlib.util.spec_from_file_location("moe117_fixture", args.fixture)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    records = []
    with contextlib.ExitStack() as stack:
        helper.open_runtime(stack)
        from sglang.kernels.ops.attention.dsv4 import linear_bf16_fp32
        from sglang.srt.layers.moe.topk import TopK
        from humming.config import GemmType
        from humming.layer import HummingMethod
        from sglang.kernels.ops.moe.moe_fused_mul_sum import moe_fused_mul_sum

        checkpoint = helper.make_checkpoint()
        layer = helper.build_layer(checkpoint, humming=True)
        core = layer.quant_method.runner_core
        configs = core.get_humming_gemm_configs(GemmType.INDEXED)
        metadata = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                        geometry=dict(hidden=4096, per_rank_intermediate=256, routed_experts=288,
                                      fused_shared=1, topk=9, clamp=10, routed_scaling=2.5),
                        source="bf6b66fa clean exported engine; 117 real FusedMoE, synthetic quantized checkpoint",
                        configs=configs)
        emit(type="metadata", **metadata)
        reference = helper.Reference(checkpoint)
        torch.manual_seed(20260930)
        gate_weight = (torch.randn(288, 4096, device="cuda") * 0.02).bfloat16()
        bias = torch.randn(288, device="cuda") * 0.02
        router = TopK(top_k=9, layer_id=3, use_grouped_topk=True, topk_group=1, num_expert_group=1,
                      renormalize=True, num_fused_shared_experts=1, scoring_func="sigmoid", correction_bias=bias,
                      routed_scaling_factor=2.5, apply_routed_scaling_factor_on_output=False,
                      allow_routed_experts_capture=False)
        try:
            if args.full_candidates:
                full_candidates(layer,core,helper,reference,router,gate_weight,bias,records,batches=tuple(map(int,args.batches.split(","))),explore=args.explore)
                return
            if args.tune:
                torch.manual_seed(1932)
                inputs = torch.randn(32,4096,device="cuda",dtype=torch.bfloat16)
                topk = router(inputs, linear_bf16_fp32(inputs, gate_weight))
                tune_stages(layer, core, helper, reference, inputs, topk, records)
                return
            for batch in (24, 28, 32, 40, 48):
                torch.manual_seed(batch + 1900)
                inputs = torch.randn(batch, 4096, device="cuda", dtype=torch.bfloat16)
                def full():
                    logits = linear_bf16_fp32(inputs, gate_weight)
                    topk = router(inputs, logits)
                    return layer(inputs, topk)
                topk = router(inputs, linear_bf16_fp32(inputs, gate_weight))
                assert bool((topk.topk_ids[:, -1] == 288).all())
                assert torch.allclose(topk.topk_weights[:, -1], torch.full((batch,), 0.4, device="cuda"), atol=1e-6)
                expected = reference(inputs, topk)
                actual = full()
                relative = helper.rel(actual, expected)
                assert bool(actual.isfinite().all()) and relative < 0.02
                unique = int(topk.topk_ids[:, :-1].unique().numel())
                perf, graph, graph_out = graph_timing(full)
                # Graph replay sees new input buffers, not capture-time values.
                inputs.add_(torch.randn_like(inputs) * 0.02)
                fresh_topk = router(inputs, linear_bf16_fp32(inputs, gate_weight))
                fresh_ref = reference(inputs, fresh_topk)
                graph.replay()
                torch.cuda.synchronize()
                replay_rel = helper.rel(graph_out, fresh_ref)
                assert bool(graph_out.isfinite().all()) and replay_rel < 0.02
                record = dict(type="full_layer", batch=batch, unique_routed_experts=unique,
                              reference_relative_l2=relative, replay_relative_l2=replay_rel,
                              up_config=selected(configs["w13_tuning_config"], batch),
                              down_config=selected(configs["w2_tuning_config"], batch), **perf)
                records.append(record)
                emit(**record)
                # Stage costs use the actual full-layer activations and align metadata.
                buffers = core.prepare_buffers(inputs, fresh_topk.topk_ids, GemmType.INDEXED)
                kw1, kw2 = core._prepare_indexed_gemm_kwargs(fresh_topk.topk_ids)
                stages = {
                    "gate_gemm": lambda: linear_bf16_fp32(inputs, gate_weight),
                    "router": lambda: router(inputs, linear_bf16_fp32(inputs, gate_weight)),
                    "align": lambda: core._prepare_indexed_gemm_kwargs(fresh_topk.topk_ids),
                    "up": lambda: HummingMethod.forward_layer(layer=layer, inputs=inputs, outputs=buffers["gate_up_output"], sublayer_name="w13", **kw1),
                    "activation": lambda: core.apply_activation(buffers["gate_up_output"], buffers["activation_output"]),
                    "down": lambda: HummingMethod.forward_layer(layer=layer, inputs=buffers["activation_output"], outputs=buffers["down_output"].view(-1, 4096), sublayer_name="w2", **kw2),
                    "reduce": lambda: moe_fused_mul_sum(inputs=buffers["down_output"].view(batch, 9, 4096), topk_weights=fresh_topk.topk_weights,
                                                        topk_ids=fresh_topk.topk_ids, is_ep=False, routed_scaling_factor=2.5, outputs=buffers["output"]),
                }
                for name, fn in stages.items():
                    timing, _, _ = graph_timing(fn)
                    record = dict(type="stage", batch=batch, stage=name, **timing)
                    records.append(record)
                    emit(**record)
                if batch == 32:
                    profile_dir = Path(args.profile_dir)
                    profile_dir.mkdir(parents=True, exist_ok=True)
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
                        for _ in range(3):
                            graph.replay()
                        torch.cuda.synchronize()
                    profile.export_chrome_trace(str(profile_dir / "moe_B32_graph.json"))
        finally:
            Path(args.out).write_text(json.dumps(dict(metadata=metadata, records=records), indent=2) + "\n")


if __name__ == "__main__":
    main()
