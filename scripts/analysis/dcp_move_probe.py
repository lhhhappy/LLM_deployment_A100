#!/usr/bin/env python3
"""Run the real MLA/indexer move bodies with a two-rank NCCL collective.

torchrun --standalone --nproc-per-node=2 dcp_move_probe.py --source-root .../sglang --output result.json
Tiny synthetic pools isolate relocation from attention. This proves neither MTP
state transitions nor compressed-indexer tree support. Source hashes are saved.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from dcp_contract_probe import source_function


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank, width = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    hashes, cases = {}, []
    parallel = SimpleNamespace(dcp_enabled=True, attn_dcp_size=width,
                               attn_dcp_rank=rank,
                               dcp_group=SimpleNamespace(device_group=dist.group.WORLD))
    move = source_function(
        args.source_root, "srt/mem_cache/memory_pool.py", "move_kv_cache",
        {"get_parallel": lambda: parallel, "maybe_detect_oob": lambda *a: None},
        hashes, owner="MLATokenToKVPool",
    )
    move_index = source_function(
        args.source_root, "srt/mem_cache/index_key_cache.py", "move",
        {"get_parallel": lambda: parallel},
        hashes, owner="IndexKeyCache",
    )
    physical_size, page, dim, key_dim, scale_dim = 128, 64, 512, 128, 4
    logical_rows = (physical_size + page) * width
    fixtures = {
        "owner_change": ([65, 66], [66, 65]),
        "overlapping_cycle": ([321, 322, 323], [322, 323, 321]),
        "repeated_source": ([255, 256, 257], [321, 321, 321]),
        "one_rank_no_writes": ([64, 128, 330], [65, 129, 331]),
        "padding": ([0, 0, 64], [1, 2, 65]),
        "empty": ([], []),
    }
    for dtype in (torch.bfloat16, torch.uint8):
        generator = torch.Generator(device="cuda").manual_seed(1709)
        # Random bytes include NaNs and signed zeros; relocation must preserve bits.
        original = torch.randint(0, 256, (logical_rows, 1, dim * dtype.itemsize),
                                 dtype=torch.uint8, device="cuda", generator=generator)
        index_original = torch.randint(
            0, 256, (logical_rows // page, page * (key_dim + scale_dim)),
            dtype=torch.uint8, device="cuda", generator=generator,
        )
        for name, (destinations, sources) in fixtures.items():
            dst = torch.tensor(destinations, device="cuda", dtype=torch.int64)
            src = torch.tensor(sources, device="cuda", dtype=torch.int64)
            kv = original[rank::width].clone().view(dtype)
            expected = original.clone()
            valid = dst != 0
            expected[dst[valid]] = original[src[valid]]
            pool = SimpleNamespace(size=physical_size, page_size=page, kv_buffer=[kv])
            move(pool, dst, src)
            latent_ok = torch.equal(kv.view(torch.uint8), expected[rank::width])

            actual_index, expected_index = index_original.clone(), index_original.clone()
            for d, s in zip(destinations, sources):
                if d == 0:
                    continue
                for offset, stride in ((0, key_dim), (page * key_dim, scale_dim)):
                    d_start, s_start = offset + d % page * stride, offset + s % page * stride
                    expected_index[d // page, d_start:d_start + stride] = index_original[
                        s // page, s_start:s_start + stride]
            index = SimpleNamespace(buffer=[actual_index, actual_index[:0]],
                                    pool=SimpleNamespace(page_size=page,
                                                         index_head_dim=key_dim,
                                                         quant_block_size=128))
            move_index(index, dst, src)
            index_ok = torch.equal(actual_index, expected_index)
            cases.append({"case": name, "dtype": str(dtype), "rank": rank,
                          "latent_bit_exact": latent_ok, "indexer_bit_exact": index_ok})
    torch.cuda.synchronize()
    by_rank = [None] * width
    dist.all_gather_object(by_rank, cases)
    passed = all(c["latent_bit_exact"] and c["indexer_bit_exact"]
                 for cs in by_rank for c in cs)
    if rank == 0:
        result = {"validity": "DISTRIBUTED_PRIMITIVE", "passed": passed,
                  "gpu": torch.cuda.get_device_name(), "world_size": width,
                  "torch": torch.__version__, "source_sha256": hashes,
                  "cases": [c for cs in by_rank for c in cs],
                  "limitations": "Actual function bodies and NCCL, synthetic pools; no attention, graph, MTP acceptance, or compressed kpool proof."}
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"passed": passed, "cases": len(result["cases"]),
                          "output": str(args.output)}), flush=True)
    dist.destroy_process_group()
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
