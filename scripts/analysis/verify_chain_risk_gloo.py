#!/usr/bin/env python3
"""Run production 131 arming + rank0_decide through real CPU Gloo processes.

No engine/model import, GPU, server request, or queue change. Dependencies are
torch and three source files under --engine/sglang/srt/managers (or --engine
pointing directly at engine/sglang). Structural scheduler state is synthetic;
the arming, risk model, deferred-round counter and transport are production ASTs.
Writes only --output and the rendezvous beside it. Budget: <64 KiB output, eight
single-threaded CPU workers plus torch imports; no inference tensors.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
from datetime import timedelta
from types import ModuleType, SimpleNamespace as NS


METHODS = {'_arm_prefill_decode_interval', '_ax_chain_risk_cfg',
           '_ax_chain_risk_interval', '_should_defer_prefill', '_ax_rank0_decide'}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_scheduler(engine):
    managers = Path(engine) / 'srt/managers'
    for name in ('sglang', 'sglang.srt', 'sglang.srt.managers'):
        sys.modules[name] = ModuleType(name)
    load_module('sglang.srt.managers.ax_rank0_decision', managers / 'ax_rank0_decision.py')
    ax = load_module('ax_deadline', managers / 'ax_deadline.py')
    source = managers / 'scheduler.py'
    cls = next(n for n in ast.parse(source.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == 'Scheduler')
    nodes = [n for n in cls.body if getattr(n, 'name', '') in METHODS]
    assert len(nodes) == len(METHODS)
    cls = ast.ClassDef(name='Scheduler', bases=[], keywords=[], decorator_list=[], body=nodes)
    tree = ast.Module(body=[ast.ImportFrom(module='__future__', level=0,
                       names=[ast.alias(name='annotations')]), cls], type_ignores=[])
    ns = dict(ax_deadline=ax, time=NS(perf_counter=lambda: 100.0),
              logger=logging.getLogger('chain-risk-gloo'))
    exec(compile(ast.fix_missing_locations(tree), str(source), 'exec'), ns)
    return ns['Scheduler'], ax


def worker(rank, world, rendezvous, engine, output):
    import torch
    import torch.distributed as dist

    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized(), 'CPU-only verifier initialized CUDA'
    dist.init_process_group('gloo', rank=rank, world_size=world,
                            init_method='file://' + rendezvous, timeout=timedelta(seconds=45))
    try:
        cls, ax = load_scheduler(engine)
        dl = ax.DeadlineConfig(freeze_class=True, load_factor=1.05)
        base = dl.arrival_offset_s + ax.service_s(200000, 16384, dl) + ax.service_s(16384, 16384, dl)
        boundary = dl.cold_budget_s - 8.0 - base
        cases = [
            ('root_comfortable_peer_at_risk', boundary-.01 if rank == 0 else boundary+.01, None, 2),
            ('root_at_risk_peer_comfortable', boundary+.01 if rank == 0 else boundary-.01, None, 1),
            ('root_hopeless_peer_at_risk', 40.0 if rank == 0 else boundary+.01, None, 2),
            ('frozen_class_on_root_only', 24.0, None, 1),
            ('125_lower_interval', boundary+.01, 0, 0),
            ('131_off', boundary+.01, None, 2),
        ]
        results = []
        for name, waited, relief, expected in cases:
            s = cls()
            s.prefill_decode_interval = 2
            s._prefill_decode_interval_remaining = 0
            s.require_mlp_sync = False
            s.dp_tp_cpu_group = dist.group.WORLD
            s._ax_chain_risk_cfg_ = None if name == '131_off' else ax.ChainRiskConfig(interval=1)
            s._ax_chain_risk_stats = dict(rounds=0, log_t=0.0)
            s._ax_admission_cfg = (dl, NS(relaxed_interval=relief) if relief is not None else None)
            s._ax_admission_cfgs = lambda: s._ax_admission_cfg
            s._ax_backlog_relieved = relief is not None
            prefix, prompt = (99840, 160000) if name.startswith('frozen') else (0, 216384)
            req = NS(rid='synthetic-head', origin_input_ids=[0] * prompt, output_ids=[],
                     num_matched_prefix_tokens=prefix, prefix_indices=[0] * prefix,
                     extend_input_len=16384, seqlen=prompt,
                     time_stats=NS(scheduler_recv_time=100.0-waited))
            if name.startswith('frozen') and rank == 0:
                req._ax_deadline_cold = True
            batch = NS(reqs=[req], extend_num_tokens=16384, _ax_chain_risk_chunk=16384,
                       forward_mode=NS(is_extend=lambda: True))
            calls = dict(broadcast_entry=0, risk_eval=0)
            actual_decide, actual_risk = s._ax_rank0_decide, s._ax_chain_risk_interval
            def decide(f):
                calls['broadcast_entry'] += 1
                return actual_decide(f)
            def risk(b):
                calls['risk_eval'] += 1
                return actual_risk(b)
            s._ax_rank0_decide, s._ax_chain_risk_interval = decide, risk
            s._arm_prefill_decode_interval(batch)
            interval = s._prefill_decode_interval_remaining
            defer = [s._should_defer_prefill() for _ in range(3)]
            local = dict(rank=rank, interval=interval, defer=defer, **calls)
            rows = [None] * world
            dist.all_gather_object(rows, local)
            on = name != '131_off'
            assert all(r['interval'] == expected and r['defer'] == [i < expected for i in range(3)] for r in rows), rows
            assert all(r['broadcast_entry'] == int(on) for r in rows), rows
            assert [r['risk_eval'] for r in rows] == [int(on)] + [0] * (world-1), rows
            results.append(dict(case=name, ranks=rows))
        if rank == 0:
            managers = Path(engine) / 'srt/managers'
            receipt = dict(status='PASS', backend='gloo', world_size=world, gpu_used=False,
                           scope='CPU control flow and real transport; not TP8 model execution or latency',
                           source_sha256={name: hashlib.sha256((managers/name).read_bytes()).hexdigest()
                                          for name in ('scheduler.py', 'ax_deadline.py', 'ax_rank0_decision.py')},
                           clock_boundary_s=boundary, cases=results)
            Path(output).write_text(json.dumps(receipt, indent=2) + '\n')
            print(json.dumps({k: receipt[k] for k in ('status', 'backend', 'world_size', 'gpu_used', 'scope')}), flush=True)
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--engine', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--world-size', type=int, choices=(2, 8), default=8)
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    import torch.multiprocessing as mp
    engine, output = args.engine.resolve(), args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rendezvous = str(output.with_suffix(f'.{os.getpid()}.rendezvous'))
    mp.spawn(worker, args=(args.world_size, rendezvous, str(engine), str(output)),
             nprocs=args.world_size, join=True)


if __name__ == '__main__':
    main()
