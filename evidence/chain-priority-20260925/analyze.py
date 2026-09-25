#!/usr/bin/env python3
"""Chain triage and actual 123 ordering probes. No performance prediction."""
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tests'), str(ROOT / 's1-dev/harness')]
from s1_common import in_ttft_gate
from test_sched_protect_chain import load_source


def main():
    ns = load_source(ROOT / 'engine/sglang')
    ns['time'] = NS(perf_counter=lambda: 10000.0)
    sort = ns['SchedulePolicy']._ax_sort_by_remaining_work
    cases = []
    with patch.dict(os.environ, {'SGLANG_AX_SRPT_AGING': '2000'}):
        for work, age, expected in (
            (100000, 20, ['fresh', 'old']),
            (100000, 48.4, ['fresh', 'old']),
            (100000, 48.6, ['old', 'fresh']),
            (250000, 123.4, ['fresh', 'old']),
            (250000, 123.6, ['old', 'fresh']),
        ):
            q = [NS(rid='old', origin_input_ids=range(work), output_ids=[],
                    num_matched_prefix_tokens=0,
                    time_stats=NS(wait_queue_entry_time=10000.0-age)),
                 NS(rid='fresh', origin_input_ids=range(63000), output_ids=[],
                    num_matched_prefix_tokens=60000,
                    time_stats=NS(wait_queue_entry_time=10000.0))]
            sort(q, set())
            actual = [r.rid for r in q]
            assert actual == expected, (work, age, actual)
            cases.append(dict(old_remaining=work, old_wait_s=age,
                              fresh_remaining=3000, actual_order=actual))
    runs = []
    for name, rel in (
        ('069-full', 'evidence/L069-official_b_pace_off_host64_full_n30_shortwarm/N30/raw_s1-dev-longchain_N30_1790280416.jsonl'),
        ('071-window45', 'evidence/L071-official_b_host64_full_n30_shortwarm/check45/raw.jsonl'),
    ):
        data = (ROOT / rel).read_bytes()
        rows = [json.loads(s) for s in data.splitlines() if s.strip()]
        bad = [r for r in rows if not r.get('error') and in_ttft_gate(r, 'chain_start') and r['ttft_s'] > 30]
        groups = []
        for lo, hi in ((0, 4096), (4097, 32768), (32769, 65536), (65537, 10**9)):
            group = [r for r in bad if lo <= r['prompt_tokens']-r['cached_tokens'] <= hi]
            groups.append(dict(uncached_range=[lo, hi], n=len(group),
                               queue_ge80=sum(r['queue_time_s'] >= .8*r['ttft_s'] for r in group)))
        runs.append(dict(run=name, raw_sha256=hashlib.sha256(data).hexdigest(),
                         raw_rows=len(rows), chain_bad=len(bad), groups=groups))
    policy = ROOT / 'engine/sglang/srt/managers/schedule_policy.py'
    out = dict(scope='CPU mechanism probes and descriptive completed-request diagnostics; no counterfactual TTFT/N prediction',
               policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
               aging_tokens_per_second=2000, ordering_cases=cases, runs=runs,
               caveats=['Sorting cannot override active-partial, host eligibility or resource limits.',
                        'Cached tokens come from the first token-producing response event, not a snapshot of every admission attempt.',
                        '071 contains only completed requests; not a level verdict.',
                        'Aging crossover against one fresh request is not a queue-wait guarantee.'])
    dest = Path(__file__).with_name('result.json')
    dest.write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(dict(ordering_cases=len(cases), chain_bad=[r['chain_bad'] for r in runs], output=str(dest))))


if __name__ == '__main__':
    main()
