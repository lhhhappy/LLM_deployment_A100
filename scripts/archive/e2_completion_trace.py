#!/usr/bin/env python3
"""T29 TP1 diagnostic entrypoint: read-only pool/partial/D1 counters tap.

Process-local wrappers only; no changes to tensors, admissions, or results.
Never use timings from this trace as a performance score. Installed at module
import so multiprocessing spawn children get identical instrumentation.
"""
import json
import os
from pathlib import Path
import sys
import time

from sglang.srt.managers import schedule_policy as policy
from sglang.srt.managers.scheduler import Scheduler

ROOT = Path(os.environ['E2_COMPLETION_ROOT']).resolve()
if not ROOT.is_relative_to(Path('/sjtu/linhang/arena/runs')) or not ROOT.name.startswith('E2_T29_'):
    raise RuntimeError('T29 trace must use a dedicated arena/runs/E2_T29_* directory')
ROOT.mkdir(parents=True, exist_ok=True)
ACTIVE = None
ROUND = 0


def emit(kind, **fields):
    with (ROOT/'scheduler_trace.jsonl').open('a') as out:
        out.write(json.dumps({'kind': kind, 'pid': os.getpid(), 'at': time.time(), **fields})+'\n')


def pools(scheduler):
    return {'kv_free': scheduler.token_to_kv_pool_allocator.available_size(),
            'mamba_free': scheduler.req_to_token_pool.mamba_allocator.available_size(),
            'request_free': scheduler.req_to_token_pool.available_size()}


_init = Scheduler.__init__
def init(self, *args, **kwargs):
    _init(self, *args, **kwargs)
    emit('startup', pools=pools(self), role_stats=dict(policy.ROLE_BOUNDARY_STATS))
Scheduler.__init__ = init

_flush = Scheduler.flush_cache
def flush(self, *args, **kwargs):
    before = pools(self)
    result = _flush(self, *args, **kwargs)
    emit('flush', success=result, before=before, after=pools(self),
         role_stats=dict(policy.ROLE_BOUNDARY_STATS))
    return result
Scheduler.flush_cache = flush

_commit = policy.PrefillAdder._commit_prefill_admission
def commit(self, req, admission, *args, **kwargs):
    result = _commit(self, req, admission, *args, **kwargs)
    if ACTIVE is not None:
        ACTIVE['commits'].append({'rid': req.rid, 'kind': 'new',
                                  'partial': admission.is_chunked})
    return result
policy.PrefillAdder._commit_prefill_admission = commit

_one = policy.PrefillAdder.add_one_req
def one(self, req, *args, **kwargs):
    if ACTIVE is not None:
        ACTIVE['attempts'] += 1
    return _one(self, req, *args, **kwargs)
policy.PrefillAdder.add_one_req = one

_chunk = policy.PrefillAdder.add_chunked_req
def chunk(self, req, *args, **kwargs):
    before = len(self.can_run_list)
    result = _chunk(self, req, *args, **kwargs)
    if ACTIVE is not None and len(self.can_run_list) > before:
        ACTIVE['commits'].append({'rid': req.rid, 'kind': 'continuation',
                                  'partial': result is not None})
    return result
policy.PrefillAdder.add_chunked_req = chunk

_split = policy.PrefillAdder._maybe_role_boundary_split
def split(self, req, admission, has_chunked_req):
    result = _split(self, req, admission, has_chunked_req)
    if result is not None:
        emit('split', rid=req.rid, depth=result.prefix_len+result.extend_len,
             prefix=result.prefix_len, original_end=admission.prefix_len+admission.extend_len)
    return result
policy.PrefillAdder._maybe_role_boundary_split = split

_prefill = Scheduler.get_new_batch_prefill
def prefill(self, *args, **kwargs):
    global ACTIVE, ROUND
    ROUND += 1
    ACTIVE = {'round': ROUND, 'attempts': 0, 'commits': []}
    before = dict(policy.ROLE_BOUNDARY_STATS)
    try:
        result = _prefill(self, *args, **kwargs)
        batch = result.batch_to_run
        if batch is not None or ACTIVE['attempts'] or ACTIVE['commits']:
            after = dict(policy.ROLE_BOUNDARY_STATS)
            ACTIVE['role_delta'] = {k: after.get(k, 0)-before.get(k, 0) for k in after}
            ACTIVE['partial_count'] = sum(c['partial'] for c in ACTIVE['commits'])
            ACTIVE['batch_rids'] = [r.rid for r in batch.reqs] if batch is not None else []
            emit('round', **ACTIVE)
        return result
    except Exception as exc:
        emit('round_error', error_type=type(exc).__name__, **ACTIVE)
        raise
    finally:
        ACTIVE = None
Scheduler.get_new_batch_prefill = prefill


if __name__ == '__main__':
    from sglang.launch_server import load_plugins, prepare_server_args, run_server, kill_process_tree
    load_plugins()
    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
