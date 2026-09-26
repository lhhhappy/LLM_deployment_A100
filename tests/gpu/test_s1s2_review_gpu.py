#!/usr/bin/env python3
"""Single A100 checks for the S1/S2 review; run with the candidate on PYTHONPATH.

Real CUDA tensors, Humming layer forwards/graphs, and registered host pools.
No complete model, TP8 collectives, harness labels, or SLO estimate.
"""
import argparse
import contextlib
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import MethodType, SimpleNamespace as NS
from unittest.mock import patch
import weakref

import torch


def emit(**result):
    print(json.dumps(result), flush=True)


def host_lifetime():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/tests/hicache180"))
    import _cpu_harness as H
    from sglang.srt.mem_cache.pool_host import dsa

    assert H.DEVICE == "cuda", "set HC180_DEVICE=cuda: do not emulate the CUDA path"
    H.publish_args()
    req_pool, kv, draft = H.glm_like_pools(with_draft=True)
    for layout in ("layer_first", "page_first", "page_first_direct"):
        for repeat in range(3):
            anchor = NS(page_size=64, layout=layout, size=8192, page_num=128)
            pool = dsa.DSAIndexerPoolHost(
                decl=dsa.make_dsa_indexer_pool_decl(kv.full_kv_pool), anchor_host=anchor,
                packed_draft_device_pools=(draft.full_kv_pool,),
            )
            owner = pool.index_k_with_scale_buffer
            assert owner.is_pinned()
            refs = [weakref.ref(t) for t in getattr(pool, "index_k_data_refs", [])]
            refs.append(weakref.ref(owner))
            stage = pool.staging_buffer
            staging_bytes = 0 if stage is None else stage.numel() * stage.element_size()
            if stage is not None:
                refs.append(weakref.ref(stage))
            host_bytes = owner.numel() * owner.element_size()
            torch.cuda.synchronize()
            with patch.object(dsa, "_cuda_host_unregister", wraps=dsa._cuda_host_unregister) as unregister:
                pool.destroy()
                assert unregister.call_count == 1
                assert not owner.is_pinned(), "cudaHostUnregister must run before releasing the allocation"
                pool.destroy()
                assert unregister.call_count == 1
            # The spy's call_args holds the tensor too. Discard that test
            # instrumentation before checking the pool's actual ownership.
            del unregister
            del owner, stage
            gc.collect()
            assert all(ref() is None for ref in refs), "pool still owns a host view or staging allocation"
            # Destroying a host pool must not destroy its device pool's storage.
            assert all(t.numel() for t in kv.full_kv_pool.index_k_with_scale_buffer)
            emit(kind="host_lifetime", layout=layout, repeat=repeat, ok=True,
                 released_host_bytes=host_bytes, released_staging_bytes=staging_bytes)


def humming_cache():
    import test_fp8_moe_humming_117 as H
    from sglang.srt.layers.moe.moe_runner.humming import HummingRunnerCore

    with contextlib.ExitStack() as stack:
        H.open_runtime(stack)
        checkpoint = H.make_checkpoint()
        layer = H.build_layer(checkpoint, humming=True)
        core, kind = layer.quant_method.runner_core, layer.quant_method.gemm_type
        cached = core.get_buffer_metas
        uncached = MethodType(HummingRunnerCore.get_buffer_metas, core)
        for count in (1, 128, 2048, 8192):
            x, topk = H.make_inputs(count, seed=count)
            assert cached(x, topk.topk_ids, kind) == uncached(x, topk.topk_ids, kind)
            metadata = {"base": [], "candidate": []}
            layer_time = {"base": [], "candidate": []}
            for trial in range(7):
                arms = [("base", uncached), ("candidate", cached)]
                if trial % 2:
                    arms.reverse()
                for name, fn in arms:
                    core.get_buffer_metas = fn
                    for _ in range(3):
                        layer(x, topk)
                    torch.cuda.synchronize()
                    start = time.perf_counter_ns()
                    for _ in range(10000):
                        fn(x, topk.topk_ids, kind)
                        fn(x, topk.topk_ids, kind)
                    metadata[name].append((time.perf_counter_ns() - start) / 10000e3)
                    start = time.perf_counter_ns()
                    for _ in range(40):
                        layer(x, topk)
                    torch.cuda.synchronize()
                    layer_time[name].append((time.perf_counter_ns() - start) / 40e6)
            core.get_buffer_metas = cached
            emit(kind="humming_cache_timing", M=count, metadata_pair_us=metadata, layer_wall_ms=layer_time,
                 median_metadata_pair_us={k: statistics.median(v) for k, v in metadata.items()},
                 median_layer_wall_ms={k: statistics.median(v) for k, v in layer_time.items()})
        emit(kind="humming_cache", entries=len(core._ax_buffer_metas), ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("host", "humming"), required=True)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    import sglang
    root = Path(sglang.__file__).resolve().parent
    paths = ("srt/layers/quantization/fp8_humming_moe.py", "srt/mem_cache/pool_host/dsa.py")
    emit(kind="start", test=args.only, torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
         visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), engine=str(root),
         source_sha256={p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths})
    if args.only == "host":
        host_lifetime()
    else:
        humming_cache()
    torch.cuda.synchronize()
    emit(kind="done", ok=True, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
