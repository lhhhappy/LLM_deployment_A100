#!/usr/bin/env python3
"""Audit a closed, timed LPM-guard pair. This is not a full-cohort score.

Read complete raw files and server logs. Preserve epoch and source line numbers;
an absent target decision is a coverage limit, never an inferred admission reason.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

def require(condition, message):
    if not condition:
        raise ValueError(message)


def one(directory, pattern):
    paths = list(directory.glob(pattern))
    require(len(paths) == 1, f"Expected one {pattern} in {directory}: {paths}")
    return paths[0]


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)


def load_level(label, directory, epoch):
    raw_path, run_path = one(directory, 'raw*.jsonl'), one(directory, 'run_*.json')
    rows, run = read_jsonl(raw_path), read_json(run_path)
    raw = {r['req_id']: r for r in rows}
    require(len(raw) == len(rows), f"Duplicate request IDs in {label}")
    require(len(rows) == run['n_attempted'] == run['dispatched'],
            f"Attempted/dispatched/raw count mismatch in {label}")
    for r in rows:
        require(not r.get('error') and not r.get('error_class'), f"Error row in {label}")
        recv, forward, first = (r[k] for k in ('t_recv_s', 't_exec_start_s', 't_first_token_s'))
        require(recv <= forward <= first, f"Bad timestamp order: {label} {r['req_id']}")
        require(abs(first - recv - r['ttft_s']) < 1e-5, f"Bad TTFT: {label} {r['req_id']}")
        require(r['tpot_s'] is None or math.isfinite(r['tpot_s']), f"Bad TPOT: {label}")
    flush = read_json(directory / 'flush_evidence.json')
    require(flush['flush_success'] and flush['runner_rc'] == 0, f"Flush/runner failure: {label}")
    require(flush['short_warmup']['status'] == 'COMPLETE', f"Incomplete warmup: {label}")
    origin = min(r['client_dispatch_at_s'] for r in rows)
    require(flush['flush_finished_s'] < origin, f"Flush after measured dispatch: {label}")
    require(max(r['client_finish_at_s'] for r in rows) <= flush['runner_finished_s'],
            f"Undrained row: {label}")
    require((directory / 'rundev_exit_code').read_text().strip() == '0', f"Runner exit: {label}")
    job = (directory / 'job.log').read_text()
    require(f'TIMED_DIAGNOSTIC DRAINED {len(rows)} requests' in job, f"Missing drain receipt: {label}")
    require('CAP_SMOKE correct=12/12' in job and 'MECHANISMS OK' in job, f"Missing mechanism/smoke receipt: {label}")

    decisions, stats, log, trace_end, mechanisms = [], [], [], [], []
    sequences, guard_counts = {}, Counter()
    guard_ids = defaultdict(set)
    for n, line in enumerate((directory / 'server.log').read_text().splitlines(), 1):
        stamp = re.match(r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)', line)
        wall = (datetime.strptime(stamp[1], '%Y-%m-%d %H:%M:%S')
                .replace(tzinfo=timezone.utc).timestamp()) if stamp else None
        record = dict(source_line=n, wall_s=wall, text=line)
        log.append(record)
        if '[ax-prefix-decision] ' in line:
            d = json.loads(line.split('[ax-prefix-decision] ', 1)[1])
            e, seq = d['epoch'], d['sequence']
            require(seq == sequences.get(e, 0) + 1, f"Trace sequence gap: {label}:{n}")
            sequences[e] = seq
            if e == epoch:
                decisions.append(dict(source_line=n, wall_s=wall, **d))
                for c in d['candidates']:
                    if c.get('lpm_hold'):
                        kind = c['lpm_hold']['decision']
                        guard_counts[kind] += 1
                        guard_ids[kind].add(c['rid'])
        match = re.search(r'\[ax-prefix-stats\] epoch=(\d+) (\{.*\})', line)
        if match:
            stats.append(dict(source_line=n, wall_s=wall, epoch=int(match[1]),
                              counts=json.loads(match[2])))
        if '[ax-prefix-trace-end]' in line:
            trace_end.append(record)
        if '[ax] mechanisms:' in line:
            mechanisms.append(record)

    require(mechanisms, f"Missing actual engine mechanism line: {label}")

    paths = [raw_path, run_path] + [directory / f for f in
            ('server.log', 'metrics.jsonl', 'flush_evidence.json', 'job.log',
             'summary.json', 'rundev_exit_code', 'level_verdict.json', 'fetch_status.json')]
    receipt = dict(label=label, directory=str(directory.resolve()), raw_rows=len(rows),
        scope='DRAINED/DIAGNOSTIC; complete admitted window, not full cohort',
        inputs={p.name: digest(p) for p in paths}, config=run['config'],
        warmup_plan=flush['short_warmup']['plan_sha256'], dispatch_origin_s=origin,
        engine_commit=re.search(r'^ENGINE_COMMIT (\w+)', job, re.M)[1],
        mechanisms=mechanisms, trace_epoch=epoch, trace_rounds=len(decisions),
        trace_end=trace_end, traced_guard_observations=dict(guard_counts),
        traced_guard_distinct_requests={k: len(v) for k, v in guard_ids.items()},
        last_stats_by_epoch={e: next(s for s in reversed(stats) if s['epoch'] == e)
                             for e in sorted({s['epoch'] for s in stats})},
        fetched_verdict=read_json(directory / 'level_verdict.json'),
        fetch_data_root=read_json(directory / 'fetch_status.json')['data_root'],
        original_summary_scope=read_json(directory / 'summary.json')['scope'])
    return dict(label=label, raw=raw, origin=origin, receipt=receipt,
                decisions=decisions, stats=[s for s in stats if s['epoch'] == epoch],
                log=log, metrics=read_jsonl(directory / 'metrics.jsonl'))


def timing(row, origin):
    fields = ('req_id', 'chain_id', 'prefix_family_id', 'idx_in_chain', 'phase',
              'prompt_tokens', 'cached_tokens', 'uncached_expected', 'output_tokens',
              'ttft_s', 'tpot_s', 'replay_gap_ms', 'effective_replay_gap_ms',
              'client_dispatch_at_s', 'client_finish_at_s', 't_recv_s',
              't_exec_start_s', 't_first_token_s')
    return dict({k: row[k] for k in fields},
                arrival_relative_s=row['t_recv_s'] - origin,
                forward_relative_s=row['t_exec_start_s'] - origin,
                first_relative_s=row['t_first_token_s'] - origin,
                finish_relative_s=row['client_finish_at_s'] - origin,
                wait_to_forward_s=row['t_exec_start_s'] - row['t_recv_s'],
                forward_to_first_s=row['t_first_token_s'] - row['t_exec_start_s'])


def target_context(run, rid):
    raw, origin, label = run['raw'], run['origin'], run['label']
    require(rid in raw, f"Target missing from raw: {label} {rid}")
    r = raw[rid]
    recv, forward = r['t_recv_s'], r['t_exec_start_s']
    excerpts, owners, last_owner = [], [], object()
    for d in run['decisions']:
        c = next((c for c in d['candidates'] if c['rid'] == rid), None)
        if c is None:
            continue
        excerpts.append(dict(label=label, target_rid=rid, source_line=d['source_line'],
            epoch=d['epoch'], sequence=d['sequence'], t=d['t'], target=c,
            rank=next(i for i, item in enumerate(d['candidates']) if item['rid'] == rid),
            admitted=d['admitted'], continuation=d['plan']['continuation'],
            admitted_candidates=[c for c in d['candidates'] if c['rid'] in d['admitted']]))
        owner = c['held_by'] if c['effective_held'] else None
        if owner != last_owner:
            owners.append(dict(sequence=d['sequence'], owner=owner,
                               held_depth=c['held_depth'], source_line=d['source_line']))
            last_owner = owner
    metrics = []
    for m in run['metrics']:
        if recv - 10 <= m['t'] <= forward + 10:
            keys = ('t', 'full_token_usage', 'kv_available_tokens', 'kv_evictable_tokens',
                    'num_running_reqs', 'num_queue_reqs', 'mamba_usage')
            metrics.append(dict(label=label, target_rid=rid, relative_to_recv_s=m['t']-recv,
                                **{k: m.get(k) for k in keys}))
    before = [s for s in run['stats'] if s['wall_s'] + 1 <= recv]
    after = [s for s in run['stats'] if s['wall_s'] > forward]
    bracketing = ([before[-1]] if before else []) + [s for s in run['stats']
                  if recv <= s['wall_s'] <= forward] + ([after[0]] if after else [])
    delta = ({k: bracketing[-1]['counts'].get(k, 0) - bracketing[0]['counts'].get(k, 0)
              for k in sorted(bracketing[0]['counts'].keys() | bracketing[-1]['counts'].keys())}
             if before and after else None)
    peers = [dict(label=label, target_rid=rid, **timing(p, origin),
                  forward_relative_to_target_s=p['t_exec_start_s']-recv,
                  first_relative_to_target_s=p['t_first_token_s']-recv)
             for p in sorted(raw.values(), key=lambda p: (p['t_exec_start_s'], p['req_id']))
             if p['req_id'] != rid and p['t_exec_start_s'] < forward and p['t_first_token_s'] > recv]
    predecessor = next((p for p in raw.values() if p['chain_id'] == r['chain_id']
                        and p['idx_in_chain'] == r['idx_in_chain'] - 1), None)
    # Untraced targets need the complete timestamped window. Traced targets
    # already have decision excerpts; do not duplicate their large JSON lines.
    logs = [dict(label=label, target_rid=rid, **entry) for entry in run['log']
            if not excerpts and entry['wall_s'] is not None
            and recv-1 <= entry['wall_s'] <= forward+1]
    context = dict(timing=timing(r, origin), observed_decisions=len(excerpts),
        effective_held_observations=sum(e['target']['effective_held'] for e in excerpts),
        owner_transitions=owners,
        first_unheld_sequence=next((e['sequence'] for e in excerpts if not e['target']['effective_held']), None),
        first_admitted_sequence=next((e['sequence'] for e in excerpts if rid in e['admitted']), None),
        stats_bracketing=bracketing, stats_delta=delta,
        predecessor=timing(predecessor, origin) if predecessor else None,
        overlapping_forward_to_first_intervals=len(peers))
    return context, excerpts, metrics, peers, logs


def gate_summary(raw, ids, gate_specs, in_gate, quantile):
    result = {}
    for _, selector, limit in gate_specs:
        values = [raw[rid]['ttft_s'] for rid in sorted(ids) if in_gate(raw[rid], selector)]
        result[selector] = dict(n=len(values), over_limit=sum(v > limit for v in values),
                                limit_s=limit, p95_s=quantile(values, .95))
    values = [raw[rid]['tpot_s'] for rid in sorted(ids) if raw[rid]['tpot_s'] is not None]
    result['tpot'] = dict(n=len(values), over_0_10=sum(v > .1 for v in values),
                          mean_s=sum(values)/len(values), p95_s=quantile(values, .95))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--off', type=Path, required=True)
    parser.add_argument('--on', type=Path, required=True)
    parser.add_argument('--harness', type=Path, required=True)
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--rid', action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.harness.resolve()))
    from s1_common import TTFT_GATE_SPECS, in_ttft_gate, q

    off, on = runs = [load_level(label, path, args.epoch)
                      for label, path in (('off', args.off), ('on', args.on))]
    for field in ('set', 'N', 'cohort_sha256', 'workload_hash', 'pacing', 'chain_gap_cap_ms'):
        require(off['receipt']['config'][field] == on['receipt']['config'][field],
                f"Workload mismatch: {field}")
    require(off['receipt']['warmup_plan'] == on['receipt']['warmup_plan'], 'Warmup plan mismatch')
    common = set(off['raw']) & set(on['raw'])
    require(common, 'No common requests')
    frozen = ('chain_id', 'idx_in_chain', 'phase', 'prompt_tokens', 'glm_tokens',
              'uncached_expected', 'max_output_i', 'replay_gap_ms', 'effective_replay_gap_ms')
    for rid in common:
        require(all(off['raw'][rid][k] == on['raw'][rid][k] for k in frozen),
                f"Frozen fields mismatch: {rid}")
    paired = []
    for rid in sorted(common):
        if not in_ttft_gate(off['raw'][rid], 'chain_start'):
            continue
        a, b = off['raw'][rid], on['raw'][rid]
        change = ('fixed' if a['ttft_s'] > 30 >= b['ttft_s'] else
                  'new_failure' if b['ttft_s'] > 30 >= a['ttft_s'] else 'unchanged')
        row = dict(req_id=rid, change=change, phase=a['phase'], idx_in_chain=a['idx_in_chain'])
        for run, r in ((off, a), (on, b)):
            t = timing(r, run['origin'])
            for k in ('prompt_tokens', 'cached_tokens', 'arrival_relative_s',
                      'forward_relative_s', 'first_relative_s', 'ttft_s',
                      'wait_to_forward_s', 'forward_to_first_s'):
                row[f"{run['label']}_{k}"] = t[k]
        paired.append(row)

    excerpts, metrics, peers, logs, targets = [], [], [], [], {}
    for rid in args.rid:
        targets[rid] = {}
        for run in runs:
            context, e, m, p, log = target_context(run, rid)
            targets[rid][run['label']] = context
            excerpts.extend(e); metrics.extend(m); peers.extend(p); logs.extend(log)
    summary = dict(scope='DIAGNOSTIC: common IDs from drained 40-minute runs; not a formal score',
        limitations=['Cross-commit screening pair, not same-engine OFF/ON or repeat evidence.',
                     'Raw cached_tokens does not identify the device/host cache tier at arrival.',
                     'Forward-to-first intervals include interleaving, not GPU kernel duration.',
                     'Cumulative stats cover all candidates; no per-request attribution outside trace.',
                     'Fixed-duration common-ID subset does not prove full-cohort SLO attainment.'],
        inputs=[r['receipt'] for r in runs], common_requests=len(common),
        unmatched_ids={r['label']: sorted(set(r['raw'])-common) for r in runs},
        gates={r['label']: gate_summary(r['raw'], common, TTFT_GATE_SPECS, in_ttft_gate, q) for r in runs},
        chain_swaps=[r for r in paired if r['change'] != 'unchanged'], targets=targets)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    for name, rows in (('chain-pair.csv', paired), ('target-metrics.csv', metrics),
                       ('overlapping-requests.csv', peers)):
        write_csv(args.out / name, rows)
    for name, rows in (('target-decisions.jsonl', excerpts), ('target-server-windows.jsonl', logs)):
        (args.out / name).write_text(''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':'))+'\n'
                                            for r in rows))
    print(json.dumps(dict(common_requests=len(common), gates=summary['gates'],
                          chain_swaps=summary['chain_swaps']), indent=2))


if __name__ == '__main__':
    main()
