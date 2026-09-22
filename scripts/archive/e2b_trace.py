#!/usr/bin/env python3
"""T26 read-only cache mechanism trace; no timing/SLO claims from these runs."""
import json
import os
from pathlib import Path
import sys
import time

from sglang.srt.managers import schedule_policy as policy
from sglang.srt.managers.scheduler import Scheduler

ROOT = Path(os.environ['E2B_TRACE_ROOT']).resolve()
if ROOT.parent.parent != Path('/sjtu/linhang/arena/runs') or not ROOT.parent.name.startswith('E2b_'):
    raise RuntimeError('T26 must stay in dedicated arena/runs/E2b_*/variant')
ROOT.mkdir(parents=True, exist_ok=True)


def emit(kind, **fields):
    with (ROOT/'scheduler_trace.jsonl').open('a') as out:
        out.write(json.dumps(dict(kind=kind, pid=os.getpid(), at=time.time(), **fields))+'\n')


def pools(s):
    return dict(kv_free=s.token_to_kv_pool_allocator.available_size(),
                mamba_free=s.req_to_token_pool.mamba_allocator.available_size(),
                request_free=s.req_to_token_pool.available_size())


_init = Scheduler.__init__
def init(self, *args, **kwargs):
    _init(self, *args, **kwargs)
    emit('startup', pools=pools(self), truncation_align=self.truncation_align_size)
Scheduler.__init__ = init

_flush = Scheduler.flush_cache
def flush(self, *args, **kwargs):
    before = pools(self)
    result = _flush(self, *args, **kwargs)
    emit('flush', success=result, before=before, after=pools(self),
         role_stats=dict(policy.ROLE_BOUNDARY_STATS))
    return result
Scheduler.flush_cache = flush

if hasattr(policy.PrefillAdder, '_role_boundary_final_chunk_len'):
    _tail = policy.PrefillAdder._role_boundary_final_chunk_len
    def tail(self, req, prefix_len, extend_len):
        result = _tail(self, req, prefix_len, extend_len)
        if result is not None:
            emit('tail_split', rid=req.rid, prefix=prefix_len,
                 depth=prefix_len+result, original_end=prefix_len+extend_len)
        return result
    policy.PrefillAdder._role_boundary_final_chunk_len = tail

if __name__ == '__main__':
    from sglang.launch_server import load_plugins, prepare_server_args, run_server, kill_process_tree
    load_plugins()
    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
