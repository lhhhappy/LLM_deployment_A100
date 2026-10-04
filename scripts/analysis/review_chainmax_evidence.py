#!/usr/bin/env python3
"""Audit the closed chain-max windows, using the original harness buckets/q.

No PASS/FAIL or full-cohort score is produced. Common IDs do not imply equal
arrival times or interference in a closed-loop replay.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

RUNS = {
    'S1': 'L130ezn1-tail_n26_S1_40m',
    'S6': 'L130ezn2-tail_n26_S6_40m',
    'chainmax16k': 'L130ezn3-tail_n26_chainmax16k_40m',
    'chainmax32k': 'L130ezn4-tail_n26_chainmax32k_40m',
}
GATES = {'chain_start': 30, 'turn_start': 15, 'overall_intra': 5, 'fast_intra': 3}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def one(root, pattern):
    paths = list(root.glob(pattern))
    require(len(paths) == 1, f'{root}: expected one {pattern}, got {paths}')
    return paths[0]


def load(root):
    raw_path, run_path = one(root, 'raw*.jsonl'), one(root, 'run_*.json')
    rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line]
    raw = {r['req_id']: r for r in rows}
    run = json.loads(run_path.read_text())
    flush = json.loads((root / 'flush_evidence.json').read_text())
    job = (root / 'job.log').read_text()
    require(len(raw) == len(rows) == run['n_attempted'] == run['dispatched'],
            f'{root}: duplicate/missing rows')
    require(flush['runner_rc'] == 0 and flush['flush_success']
            and flush['short_warmup']['status'] == 'COMPLETE', f'{root}: flush/runner')
    require((root / 'rundev_exit_code').read_text().strip() == '0', f'{root}: exit')
    require(f'TIMED_DIAGNOSTIC DRAINED {len(rows)} requests' in job
            and 'CAP_SMOKE correct=12/12' in job and 'MECHANISMS OK' in job,
            f'{root}: missing drain/smoke/mechanism receipt')
    origin = min(r['client_dispatch_at_s'] for r in rows)
    require(flush['flush_finished_s'] < origin
            and max(r['client_finish_at_s'] for r in rows) <= flush['runner_finished_s'],
            f'{root}: incomplete drain or wrong flush order')
    for r in rows:
        require(not r.get('error') and not r.get('error_class'), f'{root}: error row')
        a, b, c = (r[k] for k in ('t_recv_s', 't_exec_start_s', 't_first_token_s'))
        require(a <= b <= c and abs(c - a - r['ttft_s']) < 1e-5,
                f'{root}: timestamp mismatch {r["req_id"]}')
        require(r['tpot_s'] is None or math.isfinite(r['tpot_s']), f'{root}: TPOT')
    receipt = dict(directory=str(root.resolve()), rows=len(rows),
        validity='DRAINED/DIAGNOSTIC, not full cohort', config=run['config'],
        workload_hash=run['workload_hash'], origin_s=origin,
        warmup_plan=flush['short_warmup']['plan_sha256'],
        files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in (raw_path, run_path, root / 'server.log', root / 'job.log')},
        job_receipts=[line for line in job.splitlines() if any(s in line for s in
            ('ENGINE_COMMIT ', 'TIMED_DIAGNOSTIC DRAINED', 'MECHANISMS OK', 'CAP_SMOKE correct='))])
    return dict(raw=raw, origin=origin, receipt=receipt)


def stats(level, ids, gate_fn, quantile):
    rows = [level['raw'][rid] for rid in sorted(ids)]
    result = dict(n_common=len(rows), gates={})
    for name, limit in GATES.items():
        bucket = [r for r in rows if gate_fn(r, name)]
        result['gates'][name] = dict(n=len(bucket), over=sum(r['ttft_s'] > limit for r in bucket),
                                     p95_s=quantile([r['ttft_s'] for r in bucket], .95))
    chains = [r for r in rows if gate_fn(r, 'chain_start')]
    steady = [r for r in chains if r['t_recv_s'] - level['origin'] >= 60]
    result['steady_chain_after_60s'] = dict(n=len(steady),
        over_30=sum(r['ttft_s'] > 30 for r in steady),
        between_10_and_30=sum(10 <= r['ttft_s'] <= 30 for r in steady))
    tpot = [r['tpot_s'] for r in rows if r['tpot_s'] is not None]
    result['tpot'] = dict(n=len(tpot), mean_s=sum(tpot)/len(tpot), p95_s=quantile(tpot, .95),
                          over_010=sum(x > .10 for x in tpot))
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence-root', type=Path, required=True)
    p.add_argument('--harness', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.harness))
    from s1_common import in_ttft_gate, q
    levels = {k: load(args.evidence_root / v / 'N26') for k, v in RUNS.items()}
    require(len({x['receipt']['workload_hash'] for x in levels.values()}) == 1, 'workload differs')
    require(len({x['receipt']['warmup_plan'] for x in levels.values()}) == 1, 'warmup differs')
    args.out.mkdir(parents=True, exist_ok=True)
    groups = {}
    for name, names in [('three_way', ('S1', 'S6', 'chainmax16k')),
                        ('chunk_pair', ('chainmax16k', 'chainmax32k'))]:
        ids = set.intersection(*(set(levels[n]['raw']) for n in names))
        for rid in ids:
            for field in ('phase', 'idx_in_chain', 'uncached_expected', 'replay_gap_ms', 'max_output_i'):
                require(len({levels[n]['raw'][rid][field] for n in names}) == 1,
                        f'Frozen metadata differs: {rid}, {field}')
        groups[name] = dict(n_common=len(ids),
            arms={n: stats(levels[n], ids, in_ttft_gate, q) for n in names})
        with (args.out / f'{name}.csv').open('w', newline='') as fh:
            columns = ['req_id', 'phase', 'idx_in_chain', 'chain_bucket', 'fast_bucket']
            fields = ['arrival_s', 'ttft_s', 'wait_s', 'execution_to_first_s', 'uncached_actual', 'tpot_s']
            writer = csv.DictWriter(fh, fieldnames=columns + [n+'_'+f for n in names for f in fields],
                                    lineterminator='\n')
            writer.writeheader()
            for rid in sorted(ids):
                first = levels[names[0]]['raw'][rid]
                row = {k: first[k] for k in columns[:3]}
                row.update(chain_bucket=in_ttft_gate(first, 'chain_start'), fast_bucket=in_ttft_gate(first, 'fast_intra'))
                for n in names:
                    r = levels[n]['raw'][rid]
                    v = [r['t_recv_s'] - levels[n]['origin'], r['ttft_s'],
                         r['t_exec_start_s'] - r['t_recv_s'], r['t_first_token_s'] - r['t_exec_start_s'],
                         r['prompt_tokens'] - r['cached_tokens'], r['tpot_s']]
                    row.update({n+'_'+f: x for f, x in zip(fields, v)})
                writer.writerow(row)
    common = set.intersection(*(set(levels[n]['raw']) for n in ('S1', 'S6', 'chainmax16k')))
    bad = [levels['chainmax16k']['raw'][rid] for rid in common
           if in_ttft_gate(levels['chainmax16k']['raw'][rid], 'fast_intra')
           and levels['chainmax16k']['raw'][rid]['ttft_s'] > 3]
    waiting = [r['t_exec_start_s'] - r['t_recv_s'] for r in bad]
    execution = [r['t_first_token_s'] - r['t_exec_start_s'] for r in bad]
    fast = dict(n=len(bad), wait_p50_s=q(waiting, .5), execution_p50_s=q(execution, .5),
                aggregate_wait_share=sum(waiting)/sum(r['ttft_s'] for r in bad),
                actual_tail_le={str(k): sum(r['prompt_tokens'] - r['cached_tokens'] <= k for r in bad)
                                for k in (2048, 4096, 8192)},
                caveat='Final cached_tokens does not prove device/KDA eligibility at arrival.')
    # Preserve full matching decisions and line numbers, not invented reasons for untraced waits.
    log_path = args.evidence_root / RUNS['chainmax16k'] / 'N26/server.log'
    selected, configs = [], []
    for line_no, line in enumerate(log_path.read_text().splitlines(), 1):
        if 'TP0]' in line and ('[ax] 124 ' in line or '[ax] 131 chain-risk' in line):
            configs.append(dict(line=line_no, text=line))
        if '[ax-prefix-decision] ' not in line:
            continue
        d = json.loads(line.split('[ax-prefix-decision] ', 1)[1])
        if d['epoch'] != 2:
            continue
        matches = [r['rid'] for r in d['candidates']
                   if 0 < r.get('left', 0) <= 2048 and r.get('device', 0) > 0
                   and r.get('host', 0) == 0 and not r.get('effective_held')
                   and r.get('due_in', 0) > 0
                   and 'queue_stop_chunk_budget' in r.get('result', '')]
        if matches:
            selected.append(dict(line=line_no, matches=matches, decision=d))
    summary = dict(scope=__doc__.strip(), receipts={k: v['receipt'] for k, v in levels.items()},
        full_windows={k: stats(v, v['raw'], in_ttft_gate, q) for k, v in levels.items()},
        paired=groups, chainmax16k_common_fast_misses=fast,
        short_warm_budget_observations=len(selected))
    write_json(args.out / 'pair_audit.json', summary)
    write_json(args.out / 'budget_trace.json', dict(source=str(log_path.resolve()),
               configs=configs, observations=selected, scope='epoch 2, bounded first-120s trace only'))
    print(json.dumps(dict(paired=groups, fast=fast, matching_decisions=len(selected)), indent=2))


if __name__ == '__main__':
    main()
