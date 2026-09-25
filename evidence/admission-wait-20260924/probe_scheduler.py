#!/usr/bin/env python3
"""CPU counterexamples on measured 069 source and current source.

Uses production scheduler/adder ASTs with fake cache, pools and forward.
This proves branch behavior, not occurrence/frequency or savings in 069.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests'))
from test_sched_protect_chain import Req, make_scheduler, step, tree_dir

FROZEN_COMMIT = '759a6ebb8e31723519ad5daf438e26e24b32501a'
CURRENT = ROOT / 'engine/sglang'


def build(root, **kw):
    s, ns = make_scheduler(root, **kw)
    s.enable_hierarchical_cache = True
    s.max_running_requests = 32
    ns['get_parallel'] = lambda: NS(pp_max_micro_batch_size=32)
    return s


def cases(root):
    out = {}
    # A failure of the high-LPM candidate stops scanning before a feasible tail.
    s = build(root, waiting=[Req('large_hit', 6000, cached=8192),
                             Req('small_hit', 256, cached=4096)],
              running=[Req('decoder', 1, output=100)], available=5000)
    first = step(s)
    assert first['mode'] == 'decode' and first['full']
    assert first['verdicts'] == [('large_hit', 'NO_TOKEN')]
    assert first['waiting'] == ['large_hit', 'small_hit']
    reference = build(root, waiting=[Req('small_hit', 256, cached=4096)],
                      running=[Req('decoder', 1, output=100)], available=5000)
    feasible = step(reference)
    assert feasible['mode'] == 'prefill' and feasible['reqs'][0][0] == 'small_hit'
    out['no_token_head_blocks_feasible_tail'] = {'blocked': first, 'tail_alone_same_pool_budget': feasible}

    # Hypothetical pool growth, without a request finishing or a reset event.
    # The full latch prevents rechecking. This is not a claim that 069 grew a pool.
    prior_adders = len(s.adders)
    s.token_to_kv_pool_allocator.available_size.return_value = 100000
    following = step(s)
    assert following['mode'] == 'decode' and following['full']
    assert len(s.adders) == prior_adders
    out['full_latch_skips_rechecking'] = {'following': following, 'new_adder_calls': 0,
                                        'scope': 'hypothetical resource recovery, no finish/reset'}

    # A short host hit is inspected before a device hit, but it cannot join.
    host = Req('host', 4096 + 64, cached=4096, host=4096)
    host.num_matched_prefix_tokens = 8192
    s = build(root, chunk=Req('partial', 30000), waiting=[host, Req('device', 256, cached=4096)])
    trace = step(s)
    assert [r[0] for r in trace['reqs']] == ['partial', 'device']
    assert trace['waiting'] == ['host']
    s.tree_cache.init_load_back.assert_not_called()
    out['host_short_tail_excluded_before_restore'] = trace

    # Spare token budget alone does not make a short cold request eligible.
    s = build(root, chunk=Req('partial', 30000), waiting=[Req('cold_short', 128)])
    trace = step(s)
    assert [r[0] for r in trace['reqs']] == ['partial']
    assert trace['waiting'] == ['cold_short']
    assert s.adders[-1].rem_chunk_tokens >= 128
    out['short_cold_excluded_despite_spare_chunk_budget'] = trace
    return out


with patch.dict(os.environ, {'SGLANG_AX_SCHED_PROTECT': '1',
                            'SGLANG_AX_SCHED_COLD_CAP': '4096',
                            'SGLANG_AX_SCHED_SHORT_TOKENS': '8192',
                            'SGLANG_AX_PACE_TPOT': '0', 'SGLANG_AX_SRPT_AGING': '0',
                            'SGLANG_AX_ADMISSION_TRACE': '0'}):
    frozen = cases(tree_dir(FROZEN_COMMIT))
    current = cases(CURRENT)
assert frozen == current
result = {
    'scope': '4 CPU branch counterexamples; fake resources and forwards, no GPU',
    'frozen_engine': FROZEN_COMMIT, 'current_matches_frozen': True,
    'limits': ['not evidence these branches caused a given measured bad request',
               'not evidence for performance gains', 'no TP8, DMA or MTP numerical validation'],
    'current_source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (CURRENT / 'srt/managers/scheduler.py',
                                       CURRENT / 'srt/managers/schedule_policy.py')},
    'cases': current,
}
body = json.dumps(result, indent=2) + '\n'
assert len(body.encode()) <= 128 * 1024
Path(__file__).with_name('scheduler-probes.json').write_text(body)
print(json.dumps({'cases': len(current), 'source_trees': 2, 'all_assertions_passed': True,
                  'output_bytes': len(body.encode())}))
