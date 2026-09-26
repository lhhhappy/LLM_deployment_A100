"""Run inside dcp_check's TP2 runtime; compare complete DCP attention steps.

Both arms include communication, index conversion, sparse attention and merging.
The H=8 cases reproduce TP8 per-rank attention shapes on a two-rank DCP group;
they do not reproduce TP8 MoE/allreduce or a request-service workload.
"""
import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch


@torch.inference_mode()
def run(output):
    from sglang.srt.layers.attention.dsa_backend import (
        DeepseekSparseAttnBackend,
        _ax116_dcp_local_indices,
    )
    from sglang.srt.layers.dcp.comm import (
        all_gather_kv_cache_for_mla_extend,
        all_gather_q_for_mla_decode,
        cp_lse_ag_out_rs_mla,
    )
    from sglang.kernels.ops.attention.dcp_local_indices import local_dcp_indices
    from sglang.srt.runtime_context import get_parallel

    ps = get_parallel()
    assert ps.attn_dcp_size == 2
    rank, width, dim = ps.attn_dcp_rank, 2, 512
    sparse = DeepseekSparseAttnBackend._forward_tilelang
    results = []
    # Exact index oracle: high virtual slots, noncontiguous input, tail widths,
    # padding and both owner residues. No attention tolerance hides index errors.
    gen = torch.Generator(device="cuda").manual_seed(703)
    for columns in (1, 2, 3, 4, 63, 64, 65, 2048, 2051):
        source = torch.randint(-3, 200000, (7, columns * 2), device="cuda", generator=gen).int()[:, ::2]
        for stride in (1, 2):
            actual = local_dcp_indices(source, width=width, rank=rank, kpool_stride=stride)
            selected = source[:, rank % stride::stride]
            expected = torch.where((selected >= 0) & (selected % width == rank), selected // width, -1).int()
            assert torch.equal(actual[:, :expected.shape[1]], expected)
            assert bool((actual[:, expected.shape[1]:] == -1).all())


    def measure_all(arms):
        stats = {name: {"wall_ms": [], "device_ms": [], "peak_temporary_bytes": 0}
                 for name, _ in arms}
        for _, fn in arms:
            for _ in range(4):
                fn()
        torch.cuda.synchronize()
        # Paired, interleaved observations; every rank uses the same order.
        for repetition in range(30):
            ordered = arms if repetition % 2 == 0 else arms[::-1]
            for name, fn in ordered:
                base_memory = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start = time.perf_counter()
                a.record(); fn(); b.record(); b.synchronize()
                st = stats[name]
                st["wall_ms"].append((time.perf_counter() - start) * 1000)
                st["device_ms"].append(a.elapsed_time(b))
                st["peak_temporary_bytes"] = max(st["peak_temporary_bytes"],
                                                torch.cuda.max_memory_allocated() - base_memory)
        for st in stats.values():
            st["wall_median_ms"] = statistics.median(st["wall_ms"])
            st["device_median_ms"] = statistics.median(st["device_ms"])
        return stats

    large = os.environ.get("AX_BENCH_LARGE", "0") == "1"
    for heads in ((8,) if large else (8, 32)):
        for prefix in ((0, 8192, 32768, 131072, 262144) if large else (8192, 32768, 65536)):
            for tokens in ((2048, 8192) if large else (8, 32, 128, 512)):
                gen = torch.Generator(device="cuda").manual_seed(456)
                full = torch.randn(prefix + tokens, 1, dim, generator=gen,
                                   device="cuda", dtype=torch.bfloat16)
                local = full[rank::width].contiguous()
                indices = torch.empty(tokens, 2051, device="cuda", dtype=torch.int32)
                indices.fill_(-1)
                # Exact grouped KPool column order, including every live tail
                # residue. Future positions are never selected.
                ends = prefix + torch.arange(1, tokens + 1, device="cuda")
                history = torch.arange(512, device="cuda")[None, :].expand(tokens, -1)
                # Select up to 512 complete causal groups from each row's
                # available history, including the no-prefix cold case.
                available = ends[:, None] // 4
                history = torch.where(history < available,
                                      (history * 7919) % available.clamp_min(1), -1)
                columns = history[:, :, None] * 4 + torch.arange(4, device="cuda")
                indices[:, :2048] = torch.where(history[:, :, None] >= 0, columns, -1).reshape(tokens, -1)
                tail_cols = torch.arange(3, device="cuda")[None, :]
                indices[:, 2048:] = torch.where(tail_cols < ends[:, None] % 4,
                                               ends[:, None] - ends[:, None] % 4 + tail_cols, -1)
                # Include an empty rank and a completely padded query row.
                indices[0].fill_(-1); indices[0, 0] = 0
                if tokens > 1:
                    indices[1].fill_(-1)
                gen.manual_seed(999 + rank)
                q = torch.randn(tokens, heads, dim, device="cuda", dtype=torch.bfloat16,
                                generator=gen) * .2
                rope = q.new_empty((tokens, heads, 0))
                pool = SimpleNamespace(get_mla_kv_buffer=lambda layer, idx: (local[idx], None))
                local_pre = torch.arange(prefix // width, device="cuda", dtype=torch.int32)
                gathered = torch.empty_like(full)
                # A real virtual->gathered map lookup (nonzero virtual base).
                base = 65536
                row_table = torch.cat((indices.new_full((base,), -1),
                                       torch.arange(len(full), device="cuda", dtype=torch.int32)))
                virtual = torch.where(indices >= 0, indices + base, -1)

                def legacy():
                    all_gather_kv_cache_for_mla_extend(
                        pool, None, [prefix], local_pre, prefix, gathered, dim,
                        full[prefix:], full.new_empty((tokens, 1, 0)))
                    rows = torch.where(virtual >= 0, row_table[virtual.clamp(min=0).long()], -1).int()
                    return sparse(None, q, gathered, dim, rows, dim ** -.5).view(tokens, heads, dim)

                def candidate(fused=True):
                    qa, qr = all_gather_q_for_mla_decode(q, rope)
                    # base is W-aligned; the local buffer starts at row zero.
                    rows = (local_dcp_indices(indices, width=width, rank=rank, kpool_stride=2)
                            if fused else _ax116_dcp_local_indices(indices, kpool_stride=2))
                    out, lse = sparse(None, torch.cat((qa, qr), -1), local, dim,
                                      rows, dim ** -.5, return_lse=True)
                    out = torch.nan_to_num(out.view(tokens, heads * width, dim), nan=0., posinf=0., neginf=0.)
                    return cp_lse_ag_out_rs_mla(out, lse, ps.dcp_group).transpose(0, 1)

                expected, actual = legacy(), candidate()
                # Legacy has no normalization of padded all-masked outputs;
                # compare live rows, and require finite zero candidate padding.
                valid = (indices >= 0).any(1)
                x, y = actual[valid], expected[valid]
                relative = ((x-y).abs().flatten(1).amax(1)
                            / y.abs().flatten(1).amax(1).clamp_min(1e-8)).max().item()
                finite = bool(torch.isfinite(actual).all())
                empty_zero = bool((actual[~valid] == 0).all())
                assert finite and empty_zero and relative <= .01, (heads, prefix, tokens, relative)
                # Alternate order across shapes to avoid a fixed thermal/order bias.
                arms = [("gather_kv", legacy), ("local_kv", candidate),
                        ("local_torch_indices", lambda: candidate(False))]
                if len(results) % 2:
                    arms.reverse()
                timings = measure_all(arms)
                result = {"heads_per_rank": heads, "prefix": prefix, "tokens": tokens,
                          "rank": rank, "row_relative_linf": relative, "finite": finite,
                          "empty_rank_and_padding_passed": empty_zero, **timings}
                results.append(result)
                Path(output).write_text(json.dumps(results, indent=2) + "\n")
                print("DCP_STEP", rank, heads, prefix, tokens,
                      {k: round(v["wall_median_ms"], 4) for k, v in timings.items()}, flush=True)
    return results
