#!/usr/bin/env python3
"""Bounded W13 tuning probe at 8k/16k TP8 per-rank shapes; unchanged FP32 MMA.

Keep the existing indexed M-block height. Original checkpoint scales and an
independent sampled FP64 projection gate numerical acceptance. This probe does
not modify production dispatch, shared Humming sources, or model weights.
"""

import argparse
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import statistics
import time

import torch


def emit(**row):
    print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=int, default=[8192, 16384])
    parser.add_argument("--rounds", type=int, default=7)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("reference117", root / "tests/gpu/test_fp8_moe_humming_117.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    from humming.config import GemmType
    from humming.layer import HummingMethod
    import humming

    emit(kind="environment", gpu=torch.cuda.get_device_name(), torch=torch.__version__,
         humming_source=humming.__file__, arguments=vars(args),
         probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        with contextlib.ExitStack() as stack:
            helper.open_runtime(stack)
            checkpoint = helper.make_checkpoint()
            layer = helper.build_layer(checkpoint, True)
            del checkpoint["w2"], checkpoint["s2"]
            core = layer.quant_method.runner_core
            table = core.get_humming_gemm_configs(GemmType.INDEXED)["w13_tuning_config"]
            for m in args.rows:
                x, topk = helper.make_inputs(m, 117928 + m)
                kwargs, _ = core._prepare_indexed_gemm_kwargs(topk.topk_ids)
                baseline = next(cfg for lo, hi, cfg in table if lo < m * 9 <= hi)
                output = torch.empty(m * 9, 512, dtype=torch.bfloat16, device="cuda")
                emit(kind="config", rows=m, baseline=baseline)

                def run(config):
                    HummingMethod.forward_layer(layer=layer, inputs=x, outputs=output,
                                                sublayer_name="w13", **dict(kwargs, tuning_config=json.dumps(config)))
                    return output

                run(baseline)
                expected = output.clone()
                run(baseline)
                repeat = helper.rel(output, expected)
                tokens = torch.linspace(0, m - 1, 8, device="cuda").long().unique()
                selected = (tokens[:, None] * 9 + torch.arange(9, device="cuda")).flatten()
                oracle = torch.empty(selected.numel(), 512, device="cuda", dtype=torch.float64)
                for row, token in enumerate(tokens.cpu().tolist()):
                    for slot, expert in enumerate(topk.topk_ids[token].cpu().tolist()):
                        weight = helper.dequantize(checkpoint["w13"][expert:expert + 1],
                                                   checkpoint["s13"][expert:expert + 1])[0].double()
                        oracle[row * 9 + slot] = weight @ x[token].double()
                baseline_error = ((expected[selected].double() - oracle).norm() / oracle.norm()).item()
                emit(kind="baseline_oracle", rows=m, relative_l2=baseline_error, baseline_repeat_l2=repeat)
                assert baseline_error < 0.01
                variants = {"baseline": baseline}
                bm = baseline["block_shape"][0]
                shapes = [(256, 64, bm // 2), (128, 64, bm // 2), (128, 32, bm)]
                for n, wn, wm in shapes:
                    for stages, ctas in ((3, 1), (4, 1), (2, 2), (3, 2)):
                        for stream_k in (False, True):
                            config = dict(baseline, block_shape=[bm, n, 64], warp_shape=[wm, wn, 64],
                                          num_stages=stages, num_ctas_per_sm=ctas, use_stream_k=stream_k)
                            if config == baseline:
                                continue
                            name = f"n{n}_wm{wm}_wn{wn}_s{stages}_c{ctas}_sk{int(stream_k)}"
                            try:
                                run(config)
                                torch.cuda.synchronize()
                            except (AssertionError, ValueError) as exc:
                                emit(kind="config_rejected", rows=m, name=name, error=str(exc))
                                continue
                            error = ((output[selected].double() - oracle).norm() / oracle.norm()).item()
                            finite = bool(torch.isfinite(output).all())
                            emit(kind="numerics", rows=m, name=name, config=config, finite=finite,
                                 reference_l2=error, versus_baseline_l2=helper.rel(output, expected))
                            assert finite and error <= baseline_error * 1.05 + 1e-4
                            variants[name] = config
                for config in variants.values():
                    for _ in range(3):
                        run(config)
                deadline = time.monotonic() + 1
                while time.monotonic() < deadline:
                    run(baseline)
                    torch.cuda.synchronize()
                times = {name: [] for name in variants}
                rng = random.Random(m)
                for _ in range(args.rounds):
                    order = list(variants)
                    rng.shuffle(order)
                    for name in order:
                        times[name].append(helper.measure(lambda: run(variants[name]), inner=8, repeats=1)["gpu_ms"])
                medians = {name: statistics.median(values) for name, values in times.items()}
                emit(kind="timing", rows=m, rounds=times, medians=medians)
                best = sorted(medians, key=medians.get)[:3]
                graphs = {}
                for name in dict.fromkeys(["baseline"] + best):
                    graph, _ = helper.capture(lambda: run(variants[name]))
                    graphs[name] = graph
                graph_times = {name: [] for name in graphs}
                for _ in range(args.rounds):
                    order = list(graphs)
                    rng.shuffle(order)
                    for name in order:
                        graph_times[name].append(helper.measure(graphs[name].replay, inner=16, repeats=1)["gpu_ms"])
                emit(kind="graph_timing", rows=m, rounds=graph_times,
                     medians={name: statistics.median(values) for name, values in graph_times.items()})
                for graph in graphs.values():
                    graph.reset()
                del graph, graphs, expected, output, x, topk, oracle
            emit(kind="complete")
    finally:
        if torch.distributed.is_initialized():
            from sglang.srt.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
            destroy_model_parallel()
            destroy_distributed_environment()


if __name__ == "__main__":
    main()
