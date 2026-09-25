#!/usr/bin/env python3
"""Reproduce incomplete 071 window diagnoses, using local files only."""
import argparse
import collections
import csv
import hashlib
import io
import json
import re
import sys
from pathlib import Path

DEFAULT_WINDOW = Path(__file__).resolve().parent
ROOT = DEFAULT_WINDOW.parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--window', type=Path, default=DEFAULT_WINDOW)
OUT = parser.parse_args().window.resolve()
sys.path[:0] = [str(ROOT / 'scripts/analysis'), str(ROOT / 's1-dev/harness')]
import admission_triage as triage
import badcase
import compare_window
import window_gates

raw = OUT / 'raw.jsonl'
base_root = ROOT / 'evidence/L069-official_b_pace_off_host64_full_n30_shortwarm/N30'
base = window_gates.load_raw(base_root / 'raw_s1-dev-longchain_N30_1790280416.jsonl')
rows = window_gates.load_raw(raw)
scorer = window_gates.score_formal.load_harness()
summary, paired = compare_window.compare(base, rows, scorer)
by = triage.validate_raw(rows)
bb = triage.validate_raw(base)
paired = {r['req_id']: r for r in paired}
t0 = min(r['client_dispatch_at_s'] for r in rows)
t1 = max(r['client_finish_at_s'] for r in rows)
log = (OUT / 'server.log').read_bytes()
receipt = json.loads((OUT / 'server-transfer.json').read_text())
assert len(log) == receipt['raw_size']
assert hashlib.sha256(log).hexdigest() == receipt['raw_sha256']
assert receipt['observed_at'] > t1


def phases(r):
    p = dict(api_s=r['t_admit_s'] - r['t_recv_s'],
             dispatch_to_queue_s=r['t_exec_start_s'] - r['queue_time_s'] - r['t_admit_s'],
             queue_s=r['queue_time_s'],
             admission_to_first_s=r['t_first_token_s'] - r['t_exec_start_s'])
    assert min(p.values()) >= -1e-5
    assert abs(sum(p.values()) - r['ttft_s']) < 1e-5
    return p


phase = {r['req_id']: phases(r) for r in rows}
order = triage.overtaking(rows)
gates = {}
bad_ids = set()
for _, gate, limit in scorer.TTFT_GATE_SPECS:
    group = [r for r in rows if scorer.in_ttft_gate(r, gate)]
    bad = [r for r in group if r['ttft_s'] > limit]
    bad_ids.update(r['req_id'] for r in bad)
    gates[gate] = dict(n=len(group), over=len(bad),
        bad_queue_share_of_summed_ttft=sum(r['queue_time_s'] for r in bad) / sum(r['ttft_s'] for r in bad) if bad else None,
        bad_queue_ge80pct=sum(r['queue_time_s'] >= .8*r['ttft_s'] for r in bad),
        bad_post_admission_over_limit=sum(phase[r['req_id']]['admission_to_first_s'] > limit for r in bad),
        bad_arrival_first_minute=sum(r['client_dispatch_at_s']-t0 < 60 for r in bad))

stamp = re.compile(r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(?: TP\d+)?\]')
batches, pace, compilation = [], [], []
for line in log.decode(errors='replace').splitlines():
    m = stamp.match(line)
    if not m:
        continue
    t = badcase.epoch(m[1])
    if 'after serving started' in line:
        compilation.append((t, line))
    if not t0 - 1 <= t <= t1 + 1:
        continue
    b = badcase.LINE.match(line)
    if b:
        batches.append(dict(t=t, **dict(zip(('seqs','new','cached','running','queue'),map(int,b.groups()[1:])))))
    if '[ax-pace] decisions=' in line:
        counts = {k:int(v) for k,v in re.findall(r'(decisions|forced_decode|budgeted_prefill|guard)=(\d+)',line)}
        pace.append(dict(t=t, **counts))

details = []
for rid in sorted(bad_ids):
    r, b, d = by[rid], bb[rid], paired[rid]
    inw = [x for x in batches if r['t_recv_s']-1 <= x['t'] <= r['t_exec_start_s']+1]
    old_phase = phases(b)
    details.append(dict(req_id=rid, gates=d['candidate_failed'], baseline_gates=d['baseline_failed'],
        arrival_s=r['client_dispatch_at_s']-t0, ttft_s=r['ttft_s'], baseline_ttft_s=b['ttft_s'],
        **phase[rid], **{'baseline_'+k:v for k,v in old_phase.items()},
        cached_tokens=r['cached_tokens'], baseline_cached_tokens=b['cached_tokens'],
        prompt_tokens=r['prompt_tokens'], baseline_output_tokens=b['output_tokens'], output_tokens=r['output_tokens'],
        later_arrivals_admitted_earlier_observed=order[rid]['count'],
        other_prefill_reports_before_admission=len(inw),
        other_prefill_new_tokens_before_admission=sum(x['new'] for x in inw),
        branch_attribution='unknown: frozen 759a6eb has no admission branch trace'))

new_chains = [r for r in details if 'chain_start' in r['gates'] and 'chain_start' not in r['baseline_gates']]
analysis = dict(scope=summary['scope'], completed=len(rows), unique_ttft_bad=len(details),
    raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(), measured_from_s=t0, measured_to_s=t1,
    gates=gates,
    new_chain_count=len(new_chains),
    new_chain_same_cached=sum(r['cached_tokens']==r['baseline_cached_tokens'] for r in new_chains),
    new_chain_admission_delay_increased=sum(r['queue_s']>r['baseline_queue_s'] for r in new_chains),
    batches=dict(count=len(batches), new_tokens=sum(b['new'] for b in batches),
        new_token_hist=dict(sorted(collections.Counter(b['new'] for b in batches).items())),
        at_least8192=sum(b['new']>=8192 for b in batches),
        at_least8192_with_decoders_empty_queue=sum(b['new']>=8192 and b['running']>0 and b['queue']==0 for b in batches),
        one_sequence_with_waiters=sum(b['seqs']==1 and b['queue']>0 for b in batches)),
    pace_first=pace[0], pace_last=pace[-1],
    pace_counter_deltas={k:pace[-1][k]-pace[0][k] for k in ('decisions','forced_decode','budgeted_prefill','guard')},
    compilation_warnings=dict(total=len(compilation), during_measurement=sum(t0<=t<=t1 for t,l in compilation),
                              latest_utc=compilation[-1][1].split(']')[0]+']' if compilation else None),
    branch_attribution_unknown=len(details),
    caveats=['Completed requests only; later-arrival counts are lower bounds.',
             'Batch timestamps have one-second resolution; counts are approximate context, not causal delay.',
             'Pace counters are decisions, not seconds; logged permission need not yield an actual prefill batch.',
             'No >1s Triton compilation warning does not exclude other or shorter compilation.',
             'Queue time is not proof of unused GPU capacity, host gating, or an ordering defect.'])
buffer = io.StringIO()
writer = csv.DictWriter(buffer, fieldnames=list(details[0]), lineterminator='\n')
writer.writeheader(); writer.writerows(details)
serialized = json.dumps(analysis,indent=2)+'\n'
assert len(buffer.getvalue().encode())+len(serialized.encode()) < 2*1024*1024
(OUT/'admission-details.csv').write_text(buffer.getvalue())
(OUT/'admission-summary.json').write_text(serialized)
print(json.dumps({k:analysis[k] for k in ('completed','unique_ttft_bad','gates','pace_counter_deltas','compilation_warnings')},indent=2))
