#!/usr/bin/env python3
"""118 full-KV prefill: actual dispatch, fp32 reference, paired time and memory.

Synthetic operator inputs, including the contiguous KV layout after DCP gather.
This does not execute DCP collectives, real model weights or a request workload.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import random
import statistics
import time
from types import SimpleNamespace as NS

CASES = {
    "cold_8k": [(8192, 8192)],
    "ctx16k_8k": [(16384, 8192)],
    "ctx49k_8k": [(49152, 8192)],
    "ctx131k_8k": [(131072, 8192)],
    "ctx262k_8k": [(262144, 8192)],
    "ctx49k_1k": [(49152, 1024)],
    "ctx30k_333": [(30011, 333)],
    "ctx49k_16k": [(49152, 16384)],
    "mixed4_8k": [(49152, 2048), (131072, 2048), (262144, 2048), (8192, 2048)],
}


def emit(**record):
    print(json.dumps(record, sort_keys=True), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--calls", type=int, default=5)
    ap.add_argument("--seed", type=int, default=921)
    args = ap.parse_args()
    if args.rounds < 3 or args.calls < 1:
        ap.error("need at least 3 interleaved rounds and 1 call per round")

    import torch
    import triton
    import triton.knobs as knobs
    from sglang.srt.layers.attention import dsa_backend as backend
    from sglang.srt.layers.attention.dsa import sparse_attention_triton as kernel

    # Reuse the existing numerical oracle and kpool input generator.
    test_path = Path(__file__).resolve().parents[2] / "tests/gpu/test_dsa_sparse_118.py"
    spec = importlib.util.spec_from_file_location("oracle118", test_path)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    heads, dim, width, device = oracle.H, oracle.D, oracle.WIDTH, oracle.DEV
    loaded = []
    knobs.runtime.kernel_load_start_hook.add(
        lambda module, function, name, *_: loaded.append(name)
    )
    nloads = lambda: sum("sparse_attention" in name for name in loaded)
    kernel.warmup_sparse_attention_fwd(heads, dim, width, device)
    warmed = nloads()
    backend._AX_DSA_SPARSE_TRITON = False
    owner = NS(_ax118_prefill_seen=True)
    rng = random.Random(args.seed)
    flush = torch.empty(96 << 20, device=device, dtype=torch.uint8)
    emit(kind="environment", gpu=torch.cuda.get_device_name(), torch=torch.__version__,
         triton=triton.__version__, heads=heads, dim=dim, width=width,
         layout="contiguous full KV after gather", warmed_kernels=warmed,
         scope="synthetic operator, no DCP collectives or service claim")

    for case in args.cases:
        reqs = CASES[case]
        cpu_gen = torch.Generator().manual_seed(args.seed)
        tables, base = [], 0
        for context, rows in reqs:
            table = torch.arange(base, base + context)
            tables.append(oracle.kpool_rows(range(context - rows, context), table, cpu_gen))
            base += context
        idx = torch.cat(tables).to(device)
        gpu_gen = torch.Generator(device=device).manual_seed(args.seed + 1)
        q = torch.randn(len(idx), heads, dim, device=device, generator=gpu_gen).bfloat16()
        kv = torch.randn(base, 1, dim, device=device, generator=gpu_gen).bfloat16()

        def call(enabled):
            backend._AX_DSA_SPARSE_TRITON_PREFILL = enabled
            return backend.DeepseekSparseAttnBackend._forward_tilelang(
                owner, q, kv, dim, idx, oracle.SM_SCALE, is_prefill=True
            )

        # Include request boundaries, kpool transition and both ends of the chunk.
        sample = {0, len(idx) - 1}
        offset = 0
        for context, rows in reqs:
            sample.update((offset, offset + rows - 1))
            for pos in (63, 64, 2046, 2047, 2048, 2049):
                row = pos - (context - rows)
                if 0 <= row < rows:
                    sample.add(offset + row)
            offset += rows
        sample.update(torch.linspace(0, len(idx) - 1, min(128, len(idx))).long().tolist())
        sample = torch.tensor(sorted(sample), device=device)
        ref, _, vmax = oracle.reference(q[sample], kv, idx[sample])
        outputs = {label: call(enabled)[0][sample] for label, enabled in (("off", False), ("on", True))}
        all_rows = torch.ones(len(sample), dtype=torch.bool, device=device)
        error = {label: {
            "bound_ratio": oracle.bound_ratio(out, ref, vmax, all_rows),
            "mean_abs": (out.float() - ref).abs().mean().item(),
            "max_abs": (out.float() - ref).abs().max().item(),
        } for label, out in outputs.items()}
        assert all(e["bound_ratio"] <= 1 for e in error.values()), error
        assert error["on"]["mean_abs"] <= 1.25 * error["off"]["mean_abs"] + 1e-7, error
        del outputs, ref, vmax

        for _ in range(10):
            call(False)
            call(True)
        torch.cuda.synchronize()
        times = {label: {"device_ms": [], "wall_ms": [], "temporary_bytes": []} for label in ("off", "on")}
        for _ in range(args.rounds):
            order = [("off", False), ("on", True)]
            rng.shuffle(order)
            for label, enabled in order:
                for _ in range(args.calls):
                    flush.fill_(1)
                    torch.cuda.synchronize()
                    memory = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    wall = time.perf_counter()
                    start.record()
                    out = call(enabled)
                    end.record()
                    end.synchronize()
                    times[label]["wall_ms"].append((time.perf_counter() - wall) * 1000)
                    times[label]["device_ms"].append(start.elapsed_time(end))
                    times[label]["temporary_bytes"].append(torch.cuda.max_memory_allocated() - memory)
                    del out
        for record in times.values():
            record["device_median_ms"] = statistics.median(record["device_ms"])
            record["wall_median_ms"] = statistics.median(record["wall_ms"])
            record["temporary_max_bytes"] = max(record["temporary_bytes"])
        late = nloads() - warmed
        assert late == 0, f"{late} Triton kernels loaded after warmup"
        emit(kind="case", case=case, requests=reqs, query_rows=len(idx), kv_rows=base,
             reference_rows=len(sample), error=error, timings=times, late_kernel_loads=late,
             device_speedup=times["off"]["device_median_ms"] / times["on"]["device_median_ms"])
        del q, kv, idx, tables
    emit(kind="complete", cases=args.cases, late_kernel_loads=nloads() - warmed)


if __name__ == "__main__":
    main()
