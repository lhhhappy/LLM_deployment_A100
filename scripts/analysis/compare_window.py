#!/usr/bin/env python3
"""Pair a completed-request snapshot with a baseline; never issue a run verdict.

Use immutable snapshot files. Unfinished requests are absent, including slow
ones; the paired subset must never be presented as the whole candidate run.
"""
import argparse
import csv
import json
from pathlib import Path

import window_gates as gates

META = ('chain_id', 'phase', 'idx_in_chain', 'edge_type', 'glm_tokens',
        'uncached_expected', 'max_output_i', 'effective_replay_gap_ms')


def duration(row, start, finish):
    a, b = row.get(start), row.get(finish)
    return b-a if gates.finite(a) and gates.finite(b) and b >= a else None


def fail_set(row, scorer):
    if row.get('error'):
        return set()
    ttft = row.get('ttft_s')
    return {selector for _, selector, limit in scorer.TTFT_GATE_SPECS
            if scorer.in_ttft_gate(row, selector) and gates.finite(ttft) and ttft > limit}


def compare(base, candidate, scorer):
    by = {r['req_id']:r for r in base}
    assert len(by) == len(base), 'duplicate baseline request'
    assert len({r['req_id'] for r in candidate}) == len(candidate), 'duplicate candidate request'
    assert candidate, 'empty candidate snapshot'
    paired, details = [], []
    changes = {selector:dict(repaired=0,new=0,persistent=0)
               for _,selector,_ in scorer.TTFT_GATE_SPECS}
    for r in candidate:
        b = by[r['req_id']]
        assert all(k in b and k in r and b[k]==r[k] for k in META), 'frozen metadata mismatch: '+r['req_id']
        for row in (b,r):
            if row.get('error'):
                continue
            assert gates.finite(row.get('ttft_s')) and row['ttft_s'] >= 0, 'missing/invalid TTFT'
            assert type(row.get('output_tokens')) is int and row['output_tokens']==row['max_output_i'], 'output contract mismatch'
            assert type(row.get('prompt_tokens')) is int and row['prompt_tokens']==row['glm_tokens'], 'prompt contract mismatch'
        paired.append(b)
        bf, cf = fail_set(b,scorer),fail_set(r,scorer)
        if not b.get('error') and not r.get('error'):
            for gate, counts in changes.items():
                counts['repaired'] += gate in bf-cf
                counts['new'] += gate in cf-bf
                counts['persistent'] += gate in bf&cf
        d = dict(req_id=r['req_id'], phase=r['phase'], idx_in_chain=r['idx_in_chain'],
                 baseline_failed=';'.join(sorted(bf)),candidate_failed=';'.join(sorted(cf)),
                 baseline_error=bool(b.get('error')),candidate_error=bool(r.get('error')),
                 prompt=r.get('prompt_tokens'),uncached_expected=r.get('uncached_expected'))
        for prefix,row in [('base',b),('candidate',r)]:
            for key in ('ttft_s','tpot_s','cached_tokens','output_tokens'):
                d[prefix+'_'+key]=row.get(key)
            d[prefix+'_recv_to_exec_s']=duration(row,'t_recv_s','t_exec_start_s')
            d[prefix+'_exec_to_first_s']=duration(row,'t_exec_start_s','t_first_token_s')
        for key in ('ttft_s','tpot_s','cached_tokens','recv_to_exec_s','exec_to_first_s'):
            x,y=d['base_'+key],d['candidate_'+key]
            d['delta_'+key]=y-x if gates.finite(x) and gates.finite(y) else None
        details.append(d)
    summary=dict(scope='completed_requests_only; all candidate windows open; not a verdict',
                 baseline_total=len(base),candidate_completed=len(candidate),
                 baseline_requests_absent=len(base)-len(candidate),
                 same_request_baseline=gates.stats(paired,scorer),
                 candidate=gates.stats(candidate,scorer),gate_changes=changes,
                 timing_note='recv-to-exec is not a pure scheduler timer; exec-to-first is not GPU-only time',
                 unique_candidate_ttft_bad=sum(bool(d['candidate_failed']) for d in details),
                 unknown_candidate_timing=sum(d['candidate_recv_to_exec_s'] is None or
                                              d['candidate_exec_to_first_s'] is None for d in details))
    return summary, details


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('baseline',type=Path)
    ap.add_argument('snapshot',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    base=gates.load_raw(args.baseline)
    candidate=gates.load_raw(args.snapshot)
    summary,rows=compare(base,candidate,gates.score_formal.load_harness())
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'paired-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (args.out/'paired-completed.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:summary[k] for k in ('scope','candidate_completed','gate_changes',
                                         'unique_candidate_ttft_bad','unknown_candidate_timing')},indent=2))


if __name__=='__main__':
    main()
