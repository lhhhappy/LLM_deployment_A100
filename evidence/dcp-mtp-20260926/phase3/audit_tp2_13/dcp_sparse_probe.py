#!/usr/bin/env python3
"""Measure a KPool-aware DCP column selection prototype on A100.

Compare the engine's full masked list against its optional column selection,
checking identical owned-index multisets, attention, log2-LSE, FP32 samples and
actual graph replay. Both timing arms include index mapping and mask padding.
TP8 per-rank head shapes are emulated on one GPU; no collective/SLO claim.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from dcp_contract_probe import source_function


def padded(indices):
    pad = (-indices.shape[-1]) % 64
    return torch.nn.functional.pad(indices, (0, pad), value=-1) if pad else indices


def stats(samples):
    t = torch.tensor(samples)
    return {"p50_ms": float(t.median()), "p95_ms": float(torch.quantile(t, .95)),
            "samples_ms": samples}


def measure_pair(reference, candidate, *, numerics_only=False):
    functions = (reference, candidate)
    for _ in range(5):
        for fn in functions:
            fn()
    torch.cuda.synchronize()
    graphs, graph_copies = [], []
    for fn in functions:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = fn()
        graph.replay()
        torch.cuda.synchronize()
        graphs.append(graph)
        graph_copies.append(tuple(t.clone() for t in graph_output))
    results = [{}, {}]
    if not numerics_only:
        for mode, calls in (("eager", functions), ("graph", [g.replay for g in graphs])):
            samples = [[], []]
            # Alternate order within each pair to expose, and reduce, drift
            # that a complete baseline sweep followed by a candidate hides.
            for iteration in range(16):
                for arm in (iteration % 2, 1 - iteration % 2):
                    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                    start.record()
                    for _ in range(10):
                        calls[arm]()
                    end.record()
                    end.synchronize()
                    samples[arm].append(start.elapsed_time(end) / 10)
            for arm in range(2):
                results[arm][mode] = stats(samples[arm])
    return results, graph_copies


def check(a, b):
    x, lx = a; y, ly = b
    # Empty local attention legitimately has LSE=-inf and may return NaNs.
    # Nonempty rows must be finite; never hide their NaNs with nan_to_num.
    nonempty_finite = bool(
        (torch.isfinite(x) | torch.isneginf(lx)[..., None]).all()
        and (torch.isfinite(y) | torch.isneginf(ly)[..., None]).all())
    x = torch.nan_to_num(x, nan=0., posinf=0., neginf=0.).float()
    y = torch.nan_to_num(y, nan=0., posinf=0., neginf=0.).float()
    rel = float((x-y).abs().max() / y.abs().max().clamp_min(1e-30))
    per_row = float(((x-y).abs().amax((-1, -2)) /
                     y.abs().amax((-1, -2)).clamp_min(1e-30)).max())
    same_empty = torch.equal(torch.isneginf(lx), torch.isneginf(ly))
    finite = torch.isfinite(lx) & torch.isfinite(ly)
    lse_error = float((lx[finite]-ly[finite]).abs().max()) if finite.any() else 0.
    invalid = bool((~(torch.isfinite(lx) | torch.isneginf(lx))).any()
                   or (~(torch.isfinite(ly) | torch.isneginf(ly))).any())
    return {"relative_linf": rel, "max_row_relative_linf": per_row,
            "log2_lse_max_abs": lse_error,
            "same_empty_rows": same_empty, "invalid_lse": invalid,
            "nonempty_finite": nonempty_finite,
            "passed": rel <= 1e-2 and per_row <= 1e-2 and lse_error <= 1e-2 and same_empty
                      and not invalid and nonempty_finite}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[4, 136, 152])
    parser.add_argument("--numerics-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from sglang.kernels.ops.attention.dsa.tilelang_kernel import sparse_attention_fwd_kernel_v1
    from sglang.srt.layers.attention.dsa.kpool_fp8_index import topk_from_pooled_history_logits

    hashes, cases = {}, []
    parallel = SimpleNamespace(attn_dcp_size=1, attn_dcp_rank=0)
    original = source_function(
        args.source_root, "srt/layers/attention/dsa_backend.py",
        "_ax116_dcp_local_indices", {"get_parallel": lambda: parallel}, hashes,
    )
    source = args.source_root / "kernels/ops/attention/dsa/tilelang_kernel.py"
    hashes[str(source.relative_to(args.source_root))] = hashlib.sha256(source.read_bytes()).hexdigest()
    result = {"validity": "SINGLE_GPU_KERNEL_PROTOTYPE", "source_sha256": hashes,
              "gpu": torch.cuda.get_device_name(), "cases": cases,
              "index_source": "real topk_from_pooled_history_logits/fused KPool kernel",
              "paired_timing": not args.numerics_only,
              "scope": "Engine index mapping with real KPool output; no full model or TP8 communication. H32/H64 both arms use one stage and no output staging."}
    for path in ("srt/layers/attention/dsa/kpool_fp8_index.py",
                 "kernels/ops/moe/kpool_topk_transform.py",
                 "kernels/jit/csrc/dsa/kpool_topk_transform.cuh"):
        hashes[path] = hashlib.sha256((args.source_root / path).read_bytes()).hexdigest()
    torch.manual_seed(3109)
    # Probe the dispatch used by the unmodified engine at TP8/DCP4. If it
    # cannot fit A100 shared memory, record that precise startup failure;
    # subsequent H32 timings compare two usable one-stage layouts only.
    try:
        q32 = torch.randn(1, 4, 32, 512, device="cuda", dtype=torch.bfloat16)
        kv32 = torch.randn(1, 64, 1, 512, device="cuda", dtype=torch.bfloat16)
        i32 = torch.arange(64, device="cuda", dtype=torch.int32).view(1, 1, 1, 64).expand(1, 4, 1, 64).contiguous()
        l32 = torch.empty(1, 4, 32, device="cuda", dtype=torch.float32)
        kernel32 = sparse_attention_fwd_kernel_v1(32, 512, 0, 64, return_lse=True)
        kernel32(q32, kv32, i32, l32)
        torch.cuda.synchronize()
        result["h32_default_layout"] = {"supported": True}
    except RuntimeError as exc:
        if "shared memory" not in str(exc).lower():
            raise
        result["h32_default_layout"] = {"supported": False, "error": str(exc)}
    print(json.dumps({"h32_default_layout": result["h32_default_layout"]}), flush=True)
    # Exercise the production H32 dispatcher, including graph capture; the
    # comparison above deliberately bypasses it to retain the failing witness.
    from sglang.kernels.ops.attention.dsa import tilelang_kernel as module
    with patch.object(module, "get_parallel", lambda: SimpleNamespace(dcp_enabled=True)):
        def dispatched32():
            return module.tilelang_sparse_fwd(q32[0], kv32[0], i32[0],
                                             512**-.5, return_lse=True)
        expected_lse = torch.empty_like(l32)
        expected_out = sparse_attention_fwd_kernel_v1(
            32, 512, 0, 64, sm_scale=512**-.5, return_lse=True,
            num_stages=1, stage_output=False)(q32, kv32, i32, expected_lse)
        dispatched = dispatched32()
        _, (_, captured) = measure_pair(dispatched32, dispatched32, numerics_only=True)
        result["h32_engine_dispatch"] = {
            "eager": check(dispatched, (expected_out, expected_lse)),
            "graph": check(captured, (expected_out, expected_lse)),
        }
        assert all(v["passed"] for v in result["h32_engine_dispatch"].values())
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for width in (2, 4, 8):
        parallel.attn_dcp_size = width
        heads, dim, topk, pool = 8 * width, 512, 2048, 4
        stride = math.gcd(width, pool)
        capacity, page = 16384, 64 * width
        kv = torch.randn(capacity // width, 1, dim, device="cuda", dtype=torch.bfloat16)
        use_small_layout = heads >= 64 or (heads == 32 and not result["h32_default_layout"]["supported"])
        layout = dict(num_stages=1, stage_output=False) if use_small_layout else {}
        for rank in (0, width - 1):
            parallel.attn_dcp_rank = rank
            for rows in args.rows:
                q = torch.randn(rows, heads, dim, device="cuda", dtype=torch.bfloat16) * .2
                # Page permutations exercise high virtual locations and page gaps.
                pages = torch.randperm(capacity // page - 1, device="cuda") + 1
                positions = torch.arange(8195, device="cuda")
                virtual = pages[positions // page] * page + positions % page
                if rows == args.rows[0]:
                    boundaries = [0, 1, 2, 3, 4, 63, 64, 65, 127, 128, 129,
                                  255, 256, 257, 2047, 2048, 2049, 8192, 8193, 8194, 8195]
                    b_lens = torch.tensor(boundaries, device="cuda", dtype=torch.int32)
                    b_index = topk_from_pooled_history_logits(
                        torch.randn(len(boundaries), 2048, device="cuda"),
                        b_lens // pool, pool_size=pool, topk=topk,
                        page_table=virtual.int().expand(len(boundaries), -1).contiguous(),
                        seq_lens=b_lens)
                    for a, b in zip(original(b_index), original(b_index, kpool_stride=stride)):
                        assert torch.equal(a[a >= 0].sort().values, b[b >= 0].sort().values), "Boundary dropped an owned key"
                    result.setdefault("boundary_contracts", []).append(
                        {"width": width, "rank": rank, "lengths": boundaries, "passed": True})
                lengths = torch.full((rows,), 8195, dtype=torch.int32, device="cuda")
                lengths[0] = 0
                if rows > 1:
                    lengths[1] = 1
                indices = topk_from_pooled_history_logits(
                    torch.randn(rows, 2048, dtype=torch.float32, device="cuda"),
                    lengths // pool, pool_size=pool, topk=topk,
                    page_table=virtual.int().expand(rows, -1).contiguous(), seq_lens=lengths)
                full = original(indices)
                selected = original(indices, kpool_stride=stride)
                for a, b in zip(full, selected):
                    assert torch.equal(a[a >= 0].sort().values, b[b >= 0].sort().values), "Dropped or duplicated owned key"

                def run(compact):
                    index = padded(original(indices, kpool_stride=stride if compact else 1))
                    kernel = sparse_attention_fwd_kernel_v1(
                        heads, dim, 0, index.shape[-1], sm_scale=dim**-.5,
                        return_lse=True, **layout)
                    lse = torch.empty((1, rows, heads), dtype=torch.float32, device="cuda")
                    out = kernel(q.unsqueeze(0), kv.unsqueeze(0), index[:, None].unsqueeze(0), lse)
                    return out, lse

                ref, test = run(False), run(True)
                numeric = check(test, ref)
                # Independent FP32 reference on beginning/end rows.
                oracle_errors = []
                for row in sorted({0, min(1, rows-1), rows-1}):
                    locs = full[row][full[row] >= 0].long()
                    if not locs.numel():
                        continue
                    keys = kv[locs, 0].float()
                    scores = q[row].float() @ keys.T * dim**-.5
                    expected = scores.softmax(-1) @ keys
                    actual = test[0][0, row].float()
                    oracle_errors.append(float((actual-expected).abs().max() / expected.abs().max().clamp_min(1e-30)))
                (baseline_time, candidate_time), (ref_graph, test_graph) = measure_pair(
                    lambda: run(False), lambda: run(True), numerics_only=args.numerics_only)
                graph_numeric = check(test_graph, ref_graph)
                case = {"width": width, "rank": rank, "heads": heads, "rows": rows,
                        "kernel_layout_overrides": layout,
                        "original_columns": padded(full).shape[-1], "selected_columns": padded(selected).shape[-1],
                        "owned_multisets_equal": True, "numerical": numeric,
                        "graph_numerical": graph_numeric, "fp32_relative_linf": oracle_errors,
                        "baseline": baseline_time, "candidate": candidate_time,
                        "passed": numeric["passed"] and graph_numeric["passed"] and max(oracle_errors, default=0.) <= 1e-2}
                cases.append(case)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps({k:v for k,v in case.items() if k not in ("baseline","candidate")}), flush=True)
    result["passed"] = all(c["passed"] for c in cases)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
