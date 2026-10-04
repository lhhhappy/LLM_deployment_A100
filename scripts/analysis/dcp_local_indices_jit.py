#!/usr/bin/env python3
"""Cold-shape JIT and full owner-filter oracle on both developer GPUs."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time

import torch


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(module, rank, cases):
    kernel = module._local_indices
    device = torch.cuda.current_device()
    cache = kernel.device_caches[device][0]
    records = []
    for rows, columns, strided in cases:
        groups = torch.randint(16384, 32768, (rows, (columns + 3) // 4), device="cuda") * 4
        groups[groups % 44 == 0] = -4
        data = (groups.unsqueeze(-1) + torch.arange(4, device="cuda")).flatten(1)[:, :columns].int()
        data[data < 0] = -1
        if strided:
            storage = torch.empty((rows * 2, columns * 2), dtype=torch.int32, device="cuda")
            storage[::2, ::2] = data
            data = storage[::2, ::2]
        expected = torch.where((data >= 0) & (data % 2 == rank), data // 2, -1)
        torch.cuda.synchronize()
        before = len(cache)
        start = time.perf_counter()
        actual = module.local_dcp_indices(data, width=2, rank=rank, kpool_stride=2)
        torch.cuda.synchronize()
        first_ms = (time.perf_counter() - start) * 1000
        # Independent oracle: preserve every owned token from the UNCOMPRESSED
        # table, its order within each row, and the per-row owned-token count.
        assert torch.equal((actual >= 0).sum(1), (expected >= 0).sum(1))
        assert torch.equal(actual[actual >= 0], expected[expected >= 0])
        assert bool((actual >= -1).all())
        assert actual.shape[1] % 64 == 0
        records.append(dict(rows=rows, columns=columns, strides=list(data.stride()),
                            first_call_ms=first_ms, variants_before=before, variants_after=len(cache)))
    return dict(variants=len(cache), cases=records, oracle_passed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--legacy", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    torch.manual_seed(115)
    output = Path(str(args.output) + f".rank{rank}.json")
    if output.exists():
        raise FileExistsError(output)
    result = dict(rank=rank, scope="isolated index kernel; first calls include JIT, not a model benchmark")
    if args.legacy:
        result["legacy"] = check(load(args.legacy, "legacy_indices"), rank,
                                 [(t, 2051, False) for t in range(1, 9)])
    cases = [(t, 2051, False) for t in range(1, 513)]
    cases += [(t, 2051, False) for t in (513, 1024, 2048, 8192)]
    cases += [(t, c, True) for t in (1, 17, 129) for c in (1, 3, 5, 63, 64, 65, 127, 2048, 4099)]
    result["runtime"] = check(load(args.source, "runtime_indices"), rank, cases)
    result["passed"] = result["runtime"]["variants"] == 1
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(rank=rank, passed=result["passed"],
                         variants=result["runtime"]["variants"], cases=len(cases))), flush=True)
    assert result["passed"], "shape/stride change created a new compiled variant"


if __name__ == "__main__":
    main()
