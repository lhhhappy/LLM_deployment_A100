#!/usr/bin/env python3
"""CPU-only 124m ownership transitions through real Gloo (no inference).

Eight single-threaded workers, <32 KiB receipt; writes only --output and its
rendezvous. --source contains ax_deadline.py, ax_rank0_decision.py and
ax_multiround_park.py. Pools/Reqs are fakes; decision/state/transport are real.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from datetime import timedelta
from types import ModuleType, SimpleNamespace as NS


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def worker(rank, world, rendezvous, source, output):
    import torch
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=world,
                            init_method="file://" + rendezvous, timeout=timedelta(seconds=45))
    try:
        source = Path(source)
        for name in ("sglang", "sglang.srt", "sglang.srt.managers"):
            sys.modules[name] = ModuleType(name)
        dl = load("ax_deadline", source / "ax_deadline.py")
        sys.modules["sglang.srt.managers"].ax_deadline = dl
        multi = load("ax_multiround_park", source / "ax_multiround_park.py")
        transport = load("ax_rank0_decision", source / "ax_rank0_decision.py")
        cfg = dl.DeadlineConfig(load_factor=1.05, max_wait_s=600)
        clock = [1000.0]
        multi.time = NS(perf_counter=lambda: clock[0])
        results = []

        def req(rid, left, prefix, waited):
            return NS(rid=rid, seqlen=left + prefix, origin_input_ids=[0] * (left + prefix),
                      prefix_indices=[0] * prefix, num_matched_prefix_tokens=0, output_ids=[],
                      time_stats=NS(scheduler_recv_time=1000.0 - waited),
                      kv=NS(req_pool_idx=1 if prefix else None, cache_protected_len=prefix,
                            kv_allocated_len=prefix, mamba_pool_idx=None), inflight_middle_chunks=0,
                      extend_range=NS(start=prefix, end=prefix, length=0),
                      needs_host_load_back=lambda: False, finished=lambda: False, beam_group=None)

        for case in ("completion", "lease", "one_rank_stale_prefix", "admission_divergence"):
            clock[0] = 1000.0
            state = multi.State()
            a = req("A", 200000, 16384, 20 if rank == 0 else 0)
            b = req("B", 35000, 0, 2 if rank == 0 else 90)
            running = NS(reqs=[], batch_is_full=False)
            s = NS(chunked_req=a, waiting_queue=[b], running_batch=running, policy=NS(),
                   _ax_admission_cfgs=lambda: (cfg, None), max_running_requests=48,
                   get_num_allocatable_reqs=lambda *a, **kw: 48,
                   forward_stream=NS(synchronize=lambda: None),
                   _ax_rank0_decide=lambda f: transport.rank0_decide(dist.group.WORLD, f),
                   _ax_prefix_consensus=lambda v: transport.same_decision(dist.group.WORLD, v))
            adder = NS(_request_total_tokens=lambda r, work: work + 4096, rem_total_tokens=1000000)
            decision = s._ax_rank0_decide(lambda: state.plan(s, adder, running, 16384, None))
            assert decision is not None
            if case == "one_rank_stale_prefix" and rank == world - 1:
                a.prefix_indices.append(0)
            transaction = state.begin(s, decision)
            if case == "one_rank_stale_prefix":
                assert transaction is None and not state.parked and s.chunked_req is a
            else:
                assert transaction is not None and state.reqs() == (a,)
                b.extend_range = NS(start=0, end=16384, length=16384)
                adder.can_run_list, adder.new_chunked_req = [b], b
                if case == "admission_divergence" and rank == world - 1:
                    b.extend_range.end += 256
                rejected = False
                try:
                    state.finish(s, adder, transaction, running, False)
                except RuntimeError:
                    rejected = True
                assert rejected == (case == "admission_divergence")
                if not rejected:
                    s.chunked_req = b
                    if case == "completion":
                        b.output_ids.append(1)
                        s.chunked_req = None
                    else:
                        clock[0] = decision["lease_end"] + 1 if rank == 0 else 1000.0
                    state.prepare_step(s)
                    assert s.chunked_req is a
                    assert state.reqs() == (() if case == "completion" else (b,))
            rows = [None] * world
            dist.all_gather_object(rows, dict(rank=rank, case=case, status="PASS"))
            results.append(rows)
        assert not torch.cuda.is_initialized()
        if rank == 0:
            receipt = dict(status="PASS", world_size=world, gpu_used=False,
                           scope="124m state transitions and real CPU collectives; no model/KV numerical proof",
                           cases=results)
            Path(output).write_text(json.dumps(receipt, indent=2) + "\n")
            print(json.dumps({k: v for k, v in receipt.items() if k != "cases"}), flush=True)
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    import torch.multiprocessing as mp
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rendezvous = str(output.with_suffix(f".{os.getpid()}.rendezvous"))
    mp.spawn(worker, args=(8, rendezvous, str(args.source.resolve()), str(output)), nprocs=8, join=True)


if __name__ == "__main__":
    main()
