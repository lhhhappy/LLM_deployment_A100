#!/usr/bin/env python3
"""Four-arm SM80 MoE probe: native, reduction, down-GEMM, and both.

Uses the real 117 FusedMoE layer with synthetic TP8 per-rank weights and routes.
This measures one GPU; it does not establish TP8, real-checkpoint, or chain gains.
Example: --rows 8192 16384 --down-config '{"warp_shape":[128,32,64]}'
By default the down arm uses the production tuning helper, including its range
and exact-native-family guards. A nonempty config selects a research override
and may instead map row counts (and optionally "default") to overrides. Graphs
use separate pools; per-arm eager memory is measured before any graph exists.
Only the diagnostic process's dispatch settings are changed.
"""

import argparse
import contextlib
import copy
import gc
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import time

import torch


ARM_NAMES = ("baseline", "reduce", "down", "both")
DEFAULT_DOWN = {
    "block_shape": [128, 128, 64],
    "warp_shape": [64, 64, 64],
    "num_stages": 3,
    "num_ctas_per_sm": 2,
    "use_stream_k": False,
    "use_f16_accum": False,
}


def emit(**record):
    print(json.dumps(record), flush=True)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bf16_bits_equal(left, right):
    assert left.dtype == right.dtype == torch.bfloat16
    return torch.equal(left.view(torch.int16), right.view(torch.int16))


def git_head(root):
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


class DeviceTelemetry:
    """Read clocks without changing shared GPU settings; telemetry is optional."""

    def __init__(self):
        self.handle = None
        self.nvml = None
        self.error = None
        self.mapping = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml = pynvml
            uuid = getattr(torch.cuda.get_device_properties("cuda"), "uuid", None)
            if uuid is not None:
                uuid = uuid.decode() if isinstance(uuid, bytes) else str(uuid)
                # PyTorch may expose the bare UUID; NVML expects GPU-/MIG-.
                candidates = [uuid] if uuid.startswith(("GPU-", "MIG-")) else ["GPU-" + uuid, uuid]
                for candidate in candidates:
                    try:
                        self.handle = pynvml.nvmlDeviceGetHandleByUUID(candidate)
                        self.mapping = "cuda_uuid"
                        break
                    except pynvml.NVMLError:
                        pass
            if self.handle is None:
                visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
                index = torch.cuda.current_device()
                name = visible[index] if visible != [""] else str(index)
                self.handle = (
                    pynvml.nvmlDeviceGetHandleByIndex(int(name))
                    if name.isdecimal()
                    else pynvml.nvmlDeviceGetHandleByUUID(name)
                )
                self.mapping = "CUDA_VISIBLE_DEVICES" if visible != [""] else "cuda_ordinal"
        except Exception as exc:
            self.error = str(exc)

    def read(self):
        if self.handle is None:
            return {"unavailable": self.error}
        try:
            n, h = self.nvml, self.handle
            uuid = n.nvmlDeviceGetUUID(h)
            return {
                "sm_mhz": n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM),
                "memory_mhz": n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_MEM),
                "power_w": n.nvmlDeviceGetPowerUsage(h) / 1000,
                "temperature_c": n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU),
                "pstate": n.nvmlDeviceGetPerformanceState(h),
                "device_mapping": self.mapping,
                "nvml_uuid": uuid.decode() if isinstance(uuid, bytes) else str(uuid),
            }
        except Exception as exc:
            return {"unavailable": str(exc)}

    def close(self):
        if self.nvml is not None:
            self.nvml.nvmlShutdown()


def pick_config(table, valid_rows):
    return next(config for low, high, config in table if low < valid_rows <= high)


class Arms:
    def __init__(self, layer, reduce_module, gemm_type, rows, overrides, tune_down):
        self.core = layer.quant_method.runner_core
        self.reduce_module = reduce_module
        self.key = gemm_type.value
        native = self.core.get_humming_gemm_configs(gemm_type)
        tuned = tune_down(native)
        self.tables = {
            name: copy.deepcopy(tuned if name in ("down", "both") else native)
            for name in ARM_NAMES
        }
        self.native_up = pick_config(native["w13_tuning_config"], rows * 9)
        self.native_down = pick_config(native["w2_tuning_config"], rows * 9)
        self.config_mode = "research_override" if overrides else "production_helper"
        if overrides and rows >= 4096:
            up = pick_config(native["w13_tuning_config"], rows * 9)
            assert up["block_shape"][0] == self.native_down["block_shape"][0] == 128
            assert not self.native_down["use_stream_k"]
            chosen = overrides
            if overrides and all(str(k).isdecimal() or k == "default" for k in overrides):
                chosen = overrides.get(str(rows), overrides.get("default", {}))
            config = dict(self.native_down, **DEFAULT_DOWN)
            config.update(chosen)
            assert config["block_shape"][0] == 128 and config["block_shape"][2] == 64
            assert config["warp_shape"][2] == 64
            assert all(b % w == 0 for b, w in zip(config["block_shape"], config["warp_shape"]))
            assert not config["use_stream_k"] and not config["use_f16_accum"]
            for name in ("down", "both"):
                table = self.tables[name]["w2_tuning_config"]
                table[:] = [
                    (low, high, copy.deepcopy(config) if low < rows * 9 <= high else old)
                    for low, high, old in table
                ]
                self.tables[name]["w2_tuning_config_str"] = json.dumps(table)
        self.candidate_down = pick_config(self.tables["down"]["w2_tuning_config"], rows * 9)
        self.enabled_down = self.candidate_down != self.native_down
        normalized_native = dict(self.native_down)
        for key in ("block_shape", "warp_shape"):
            normalized_native[key] = tuple(normalized_native.get(key, ()))
        self.standard_native_family = normalized_native == {
            "block_shape": (128, 256, 64), "warp_shape": (64, 64, 64),
            "use_stream_k": False, "use_f16_accum": False, "num_sms": 108,
            "num_stages": 4, "num_ctas_per_sm": 1, "num_write_splits": 1,
        }
        if not overrides and self.standard_native_family and 8192 <= rows <= 16384:
            assert self.enabled_down, "production tuning silently left the supported native family unchanged"

    @contextlib.contextmanager
    def select(self, name):
        old_table = self.core.humming_gemm_configs[self.key]
        old_reduce = self.reduce_module._AX_SM80_MOE_REDUCE
        self.core.humming_gemm_configs[self.key] = self.tables[name]
        self.reduce_module._AX_SM80_MOE_REDUCE = name in ("reduce", "both")
        try:
            yield
        finally:
            self.core.humming_gemm_configs[self.key] = old_table
            self.reduce_module._AX_SM80_MOE_REDUCE = old_reduce


def make_inputs(helper, rows, seed, route_kind, args):
    x, topk = helper.make_inputs(rows, seed, x_scale=args.x_scale)
    if route_kind == "skew":
        generator = torch.Generator(device="cuda").manual_seed(seed + 710117)
        logits = torch.randn(rows, helper.E_ROUTED, generator=generator, device="cuda")
        logits[:, : args.hot_experts] += args.skew_bias
        topk.topk_ids[:, : helper.TOP_K].copy_(logits.topk(helper.TOP_K, dim=1).indices.int())
    topk.router_logits.zero_()
    return x, topk


def slice_topk(topk, ids):
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    return StandardTopKOutput(
        topk_weights=topk.topk_weights[ids],
        topk_ids=topk.topk_ids[ids],
        router_logits=topk.router_logits[ids],
    )


def route_summary(topk, helper):
    counts = torch.bincount(topk.topk_ids[:, : helper.TOP_K].flatten().long(), minlength=helper.E_ROUTED)
    values = sorted(counts.cpu().tolist())
    return {
        "routed_expert_min": values[0],
        "routed_expert_median": statistics.median(values),
        "routed_expert_max": values[-1],
        "routed_expert_counts": counts.cpu().tolist(),
        "shared_expert_rows": topk.topk_ids.shape[0],
    }


def check_intermediates(layer, x, topk, arms, helper, case):
    """Capture one actual down input before its workspace is reused by combine."""
    from humming.layer import HummingMethod

    real = HummingMethod.forward_layer
    own_descriptor = HummingMethod.__dict__.get("forward_layer")
    captured = {}

    def hook(cls, *positional, **kwargs):
        is_down = kwargs.get("layer") is layer and kwargs.get("sublayer_name") == "w2"
        if is_down:
            assert not positional
            captured["inputs"] = kwargs["inputs"].clone()
            captured["kwargs"] = {
                key: value.clone() if isinstance(value, torch.Tensor) else value
                for key, value in kwargs.items()
                if key not in ("inputs", "outputs", "layer")
            }
        result = real(*positional, **kwargs)
        if is_down:
            captured["output"] = kwargs["outputs"].clone()
        return result

    HummingMethod.forward_layer = classmethod(hook)
    try:
        with arms.select("baseline"), helper.fixed_alignment():
            layer(x, topk)
    finally:
        if own_descriptor is None:
            delattr(HummingMethod, "forward_layer")
        else:
            HummingMethod.forward_layer = own_descriptor
    assert captured
    original = captured["output"]
    output = torch.empty_like(original)
    checks = {}
    for name in (("baseline", "down") if arms.enabled_down else ()):
        kwargs = dict(captured["kwargs"], tuning_config=arms.tables[name]["w2_tuning_config_str"])
        real(layer=layer, inputs=captured["inputs"], outputs=output, **kwargs)
        checks[name] = {
            "exact": bf16_bits_equal(output, original),
            "relative_l2": helper.rel(output, original),
        }
        assert checks[name]["exact"], f"{name} changed down-GEMM for the same actual activation"
    reduced = {}
    for name in ("baseline", "reduce"):
        with arms.select(name):
            reduced[name] = arms.reduce_module.moe_fused_mul_sum(
                original.view(x.shape[0], 9, 4096), topk.topk_weights,
                topk_ids=topk.topk_ids, routed_scaling_factor=helper.RSF,
            )
    reduce_exact = bf16_bits_equal(reduced["baseline"], reduced["reduce"])
    emit(kind="intermediate_identity", **case, down_enabled=arms.enabled_down, down=checks,
         reduce_exact=reduce_exact, reduce_relative_l2=helper.rel(reduced["reduce"], reduced["baseline"]),
         padded_routed_rows=int(captured["kwargs"]["num_tokens_padded"].item()))
    assert reduce_exact, "reduction changed the same actual down-GEMM output"


def check_full_layer(layer, x, topk, arms, reference, ids, helper, case, max_ref_l2):
    oracle = reference(x[ids], slice_topk(topk, ids))
    with helper.fixed_alignment():
        with arms.select("baseline"):
            baseline = layer(x, topk)
            repeat = layer(x, topk)
        baseline_error = max(helper.rel(baseline[ids], oracle), helper.rel(repeat[ids], oracle))
        repeat_error = helper.rel(repeat, baseline)
        for name in ARM_NAMES:
            with arms.select(name):
                output = baseline if name == "baseline" else layer(x, topk)
            reference_error = helper.rel(output[ids], oracle)
            finite = bool(torch.isfinite(output).all())
            emit(kind="layer_numerics", **case, arm=name, finite=finite, alignment="fixed",
                 exact_vs_baseline=bf16_bits_equal(output, baseline),
                 baseline_repeat_l2=repeat_error, candidate_vs_baseline_l2=helper.rel(output, baseline),
                 sample_rows=ids.cpu().tolist(), sample_reference_l2=reference_error,
                 baseline_sample_reference_l2=baseline_error, reference="original FP8 checkpoint + FP32 scales; FP32 matmul, TF32 disabled")
            assert finite and reference_error < max_ref_l2
            assert reference_error <= 1.05 * baseline_error + 1e-4


def eager_memory(layer, x, topk, arms, case):
    # No graphs have been created for this shape, and no outputs are retained.
    for name in ARM_NAMES:
        with arms.select(name):
            for _ in range(3):
                layer(x, topk)
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            output = layer(x, topk)
            torch.cuda.synchronize()
            emit(kind="eager_memory", **case, arm=name,
                 transient_mib=(torch.cuda.max_memory_allocated() - before) / 2**20,
                 output_storage_mib=output.untyped_storage().nbytes() / 2**20,
                 baseline_allocated_mib=before / 2**20, graph_pools_present=False)
            del output


def capture_graphs(layer, x, topk, arms, case):
    graphs = {}
    order = list(ARM_NAMES)
    random.Random(case["seed"] + 914).shuffle(order)
    emit(kind="graph_capture_order", **case, order=order)
    for name in order:
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        before_reserved = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        with arms.select(name):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    layer(x, topk)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            # Omitting pool= deliberately gives this arm an independent pool.
            with torch.cuda.graph(graph, stream=stream):
                output = layer(x, topk)
        torch.cuda.synchronize()
        graphs[name] = (graph, output)
        emit(kind="graph_memory", **case, arm=name,
             retained_delta_mib=(torch.cuda.memory_allocated() - before) / 2**20,
             capture_peak_delta_mib=(torch.cuda.max_memory_allocated() - before) / 2**20,
             reserved_delta_mib=(torch.cuda.memory_reserved() - before_reserved) / 2**20,
             total_allocated_mib=torch.cuda.memory_allocated() / 2**20,
             total_reserved_mib=torch.cuda.memory_reserved() / 2**20,
             independent_pool=True, note="total includes earlier arms; use eager_memory for per-arm transient comparisons")
    return graphs


def check_graphs(layer, x, topk, arms, graphs, reference, ids, helper, case, args):
    stale = {}
    for name, (graph, output) in graphs.items():
        graph.replay()
        stale[name] = output[ids].clone()
    for trial in range(args.graph_trials):
        seed = case["seed"] + 1000000 + trial
        fresh_x, fresh_topk = make_inputs(helper, x.shape[0], seed, case["routes"], args)
        assert not torch.equal(fresh_x, x) and not torch.equal(fresh_topk.topk_ids, topk.topk_ids)
        x.copy_(fresh_x)
        topk.topk_ids.copy_(fresh_topk.topk_ids)
        topk.topk_weights.copy_(fresh_topk.topk_weights)
        topk.router_logits.copy_(fresh_topk.router_logits)
        del fresh_x, fresh_topk
        oracle = reference(x[ids], slice_topk(topk, ids))
        with arms.select("baseline"):
            baseline = layer(x, topk)
            baseline_repeat = layer(x, topk)
        repeat_l2 = helper.rel(baseline_repeat, baseline)
        # Native ordering/stream-K noise can differ between graph and eager.
        # Calibrate the graph comparison using both modes, while the independent
        # checkpoint oracle and actual-intermediate bit tests remain hard gates.
        controls = {"eager0": baseline, "eager1": baseline_repeat}
        baseline_graph, baseline_graph_output = graphs["baseline"]
        for name in ("graph0", "graph1"):
            baseline_graph.replay()
            controls[name] = baseline_graph_output.clone()
        control_names = list(controls)
        pairs = {
            f"{left}_vs_{right}": helper.rel(controls[left], controls[right])
            for index, left in enumerate(control_names)
            for right in control_names[index + 1 :]
        }
        control_reference_errors = {name: helper.rel(value[ids], oracle) for name, value in controls.items()}
        baseline_error = max(control_reference_errors.values())
        baseline_graph_eager_envelope = max(pairs.values())
        emit(kind="baseline_graph_control", **case, trial=trial, fresh_seed=seed,
             alignment="free_recomputed", pairwise_l2=pairs, sample_reference_l2=control_reference_errors,
             graph_eager_envelope_l2=baseline_graph_eager_envelope)
        assert baseline_error < args.max_ref_l2
        for name, (graph, output) in graphs.items():
            graph.replay()
            replay = output.clone()
            with arms.select(name):
                eager = layer(x, topk)
            replay_error = helper.rel(replay[ids], oracle)
            eager_error = helper.rel(eager[ids], oracle)
            replay_vs_eager = helper.rel(replay, eager)
            stale_error = helper.rel(stale[name], oracle)
            finite = bool(torch.isfinite(replay).all()) and bool(torch.isfinite(eager).all())
            emit(kind="graph_numerics", **case, arm=name, trial=trial, fresh_seed=seed,
                 changed_inputs_and_routes=True, finite=finite, replay_reference_l2=replay_error,
                 eager_reference_l2=eager_error, replay_vs_eager_l2=replay_vs_eager,
                 baseline_repeat_l2=repeat_l2, baseline_reference_l2=baseline_error,
                 baseline_graph_eager_envelope_l2=baseline_graph_eager_envelope,
                 stale_reference_l2=stale_error)
            assert finite and max(replay_error, eager_error) < args.max_ref_l2
            assert max(replay_error, eager_error) <= 1.05 * baseline_error + 1e-4
            assert replay_vs_eager <= 1.25 * baseline_graph_eager_envelope + 1e-4
            assert stale_error > 0.1, "fresh-input graph check did not reject the capture-time output"
            del replay, eager
        del controls, baseline, baseline_repeat, oracle
    return seed


def measure_block(fn, inner):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    start.record()
    for _ in range(inner):
        fn()
    end.record()
    enqueued = time.perf_counter()
    end.synchronize()
    finished = time.perf_counter()
    return {
        "gpu_ms": start.elapsed_time(end) / inner,
        "host_ms": (enqueued - t0) * 1000 / inner,
        "wall_ms": (finished - t0) * 1000 / inner,
    }


def paired_summary(samples, seed):
    baseline = samples["baseline"]
    result = {}
    rng = random.Random(seed)
    for name, records in samples.items():
        gains = [(b["gpu_ms"] - r["gpu_ms"]) / b["gpu_ms"] * 100 for b, r in zip(baseline, records)]
        resampled = sorted(statistics.median(rng.choices(gains, k=len(gains))) for _ in range(5000))
        result[name] = {
            "median_ms": {metric: statistics.median(r[metric] for r in records) for metric in ("gpu_ms", "host_ms", "wall_ms")},
            "paired_gain_percent": gains,
            "paired_median_gain_percent": statistics.median(gains),
            "paired_median_bootstrap_95pct": [resampled[125], resampled[4874]],
        }
    return result


def benchmark(layer, x, topk, arms, graphs, telemetry, case, args, timing_seed):
    rng = random.Random(case["seed"] + 317)
    for mode in ("eager", "graph"):
        deadline = time.monotonic() + args.settle_seconds
        while time.monotonic() < deadline:
            for name in ARM_NAMES:
                with arms.select(name):
                    if mode == "eager":
                        layer(x, topk)
                    else:
                        graphs[name][0].replay()
            torch.cuda.synchronize()
        samples = {name: [] for name in ARM_NAMES}
        for round_index in range(args.rounds):
            order = list(ARM_NAMES)
            rng.shuffle(order)
            for name in order:
                fn = (lambda: layer(x, topk)) if mode == "eager" else graphs[name][0].replay
                with arms.select(name):
                    # The switch itself is outside the measured region in every arm.
                    for _ in range(3):
                        fn()
                    before = telemetry.read()
                    record = measure_block(fn, args.inner)
                    record["telemetry_before"] = before
                    record["telemetry_after"] = telemetry.read()
                samples[name].append(record)
            emit(kind="timing_round", **case, mode=mode, round=round_index,
                 order=order, samples={name: samples[name][-1] for name in ARM_NAMES})
        emit(kind="timing", **case, mode=mode, inner=args.inner, rounds=args.rounds,
             timing_input_seed=timing_seed, summary=paired_summary(samples, case["seed"]),
             event_scope="elapsed CUDA stream time; eager can include CPU enqueue gaps; graph replays one full layer")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=int, default=[4096, 8192, 16384])
    parser.add_argument("--ragged", action="store_true", help="also test one row below and above each requested size")
    parser.add_argument("--routes", nargs="+", choices=["uniform", "skew"], default=["uniform"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[928])
    parser.add_argument("--down-config", default="{}", help="JSON overrides, or row-count to overrides mapping")
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--inner", type=int, default=16)
    parser.add_argument("--graph-trials", type=int, default=2)
    parser.add_argument("--reference-rows", type=int, default=16)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument("--hot-experts", type=int, default=32)
    parser.add_argument("--skew-bias", type=float, default=3.0)
    parser.add_argument("--x-scale", type=float, default=1.0)
    parser.add_argument("--max-ref-l2", type=float, default=0.02)
    args = parser.parse_args()
    overrides = json.loads(args.down_config)
    assert isinstance(overrides, dict)
    assert min(args.rows) > 0 and min(args.rounds, args.inner, args.graph_trials, args.reference_rows) > 0
    assert 8 <= args.hot_experts <= 288 and args.settle_seconds >= 0
    rows = sorted(set(args.rows + ([m + d for m in args.rows for d in (-1, 1) if m + d > 0] if args.ragged else [])))
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("moe117_candidate_reference", root / "tests/gpu/test_fp8_moe_humming_117.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    reduce_module = importlib.import_module("sglang.kernels.ops.moe.moe_fused_mul_sum")
    runner_module = importlib.import_module("sglang.srt.layers.moe.moe_runner.humming")
    fp8_module = importlib.import_module("sglang.srt.layers.quantization.fp8_humming_moe")
    tuning_module = importlib.import_module("sglang.srt.layers.quantization.fp8_humming_tuning")
    from humming.config import GemmType
    from humming.jit.compiler import Compiler
    import humming
    import triton

    assert torch.cuda.get_device_capability() == (8, 0), "this candidate is restricted to SM80"
    assert Path(reduce_module.__file__).resolve().is_relative_to(root / "engine")
    assert Path(runner_module.__file__).resolve().is_relative_to(root / "engine")
    header_root = Path(Compiler.humming_include_dir()).resolve()
    headers = {str(p.relative_to(header_root)): digest(p) for p in sorted(header_root.rglob("*.cuh"))}
    source_files = {
        "probe": Path(__file__), "reference_helper": Path(helper.__file__),
        "reduce": Path(reduce_module.__file__), "runner": Path(runner_module.__file__),
        "fp8_method": root / "engine/sglang/srt/layers/quantization/fp8_humming_moe.py",
        "down_tuning": Path(tuning_module.__file__),
        "humming_layer": Path(humming.__file__).parent / "layer.py",
        "humming_forward": Path(humming.__file__).parent / "forward.py",
    }
    emit(kind="environment", gpu=torch.cuda.get_device_name(), capability=[8, 0],
         cuda_uuid=str(getattr(torch.cuda.get_device_properties("cuda"), "uuid", None)),
         sms=torch.cuda.get_device_properties("cuda").multi_processor_count,
         torch=torch.__version__, cuda=torch.version.cuda, triton=triton.__version__,
         humming_source=humming.__file__, arguments=vars(args), effective_rows=rows,
         git_head=git_head(root), sources={k: {"path": str(p.resolve()), "sha256": digest(p)} for k, p in source_files.items()},
         humming_headers_sha256=hashlib.sha256(json.dumps(headers, sort_keys=True).encode()).hexdigest(),
         visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), synthetic_checkpoint_seed=0,
         inherited_reduce_flag=reduce_module._AX_SM80_MOE_REDUCE,
         inherited_down_tune_flag=fp8_module._AX_SM80_MOE_DOWN_TUNE,
         baseline_build_flags={"reduce": False, "down_tune": False},
         reference_tf32=False, graph_pools="independent; graphs are serially replayed")
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_precision = torch.get_float32_matmul_precision()
    old_reduce = reduce_module._AX_SM80_MOE_REDUCE
    old_down_tune = fp8_module._AX_SM80_MOE_DOWN_TUNE
    telemetry = DeviceTelemetry()
    started = time.monotonic()
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        reduce_module._AX_SM80_MOE_REDUCE = False
        fp8_module._AX_SM80_MOE_DOWN_TUNE = False
        with contextlib.ExitStack() as stack, torch.inference_mode():
            helper.open_runtime(stack)
            checkpoint = helper.make_checkpoint()
            layer = helper.build_layer(checkpoint, True)
            reference = helper.Reference(checkpoint)
            del checkpoint
            for m in rows:
                for routes in args.routes:
                    for seed in args.seeds:
                        case = dict(rows=m, routes=routes, seed=seed + m)
                        x, topk = make_inputs(helper, m, case["seed"], routes, args)
                        arms = Arms(layer, reduce_module, GemmType.INDEXED, m, overrides, tuning_module.tune_sm80_prefill_down)
                        ids = torch.linspace(0, m - 1, min(m, args.reference_rows), device="cuda").long().unique()
                        emit(kind="case", **case, native_up=arms.native_up,
                             native_down=arms.native_down, candidate_down=arms.candidate_down,
                             down_enabled=arms.enabled_down, config_mode=arms.config_mode,
                             standard_native_family=arms.standard_native_family,
                             native_shape_container_types={name: type(arms.native_down[name]).__name__ for name in ("block_shape", "warp_shape")},
                             routing=route_summary(topk, helper), telemetry=telemetry.read())
                        check_intermediates(layer, x, topk, arms, helper, case)
                        check_full_layer(layer, x, topk, arms, reference, ids, helper, case, args.max_ref_l2)
                        eager_memory(layer, x, topk, arms, case)
                        graphs = capture_graphs(layer, x, topk, arms, case)
                        try:
                            timing_seed = check_graphs(layer, x, topk, arms, graphs, reference, ids, helper, case, args)
                            benchmark(layer, x, topk, arms, graphs, telemetry, case, args, timing_seed)
                        finally:
                            torch.cuda.synchronize()
                            for graph, output in graphs.values():
                                graph.reset()
                            graphs.clear()
                            del graph, output
                            gc.collect()
                        del x, topk, ids, arms
            emit(kind="complete", elapsed_s=time.monotonic() - started)
    finally:
        reduce_module._AX_SM80_MOE_REDUCE = old_reduce
        fp8_module._AX_SM80_MOE_DOWN_TUNE = old_down_tune
        torch.set_float32_matmul_precision(old_precision)
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        telemetry.close()
        if torch.distributed.is_initialized():
            from sglang.srt.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment

            destroy_model_parallel()
            destroy_distributed_environment()


if __name__ == "__main__":
    main()
