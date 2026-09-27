#!/usr/bin/env python3
"""Join actual prefix decisions to raw chain requests; never a capacity score."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def analyze(level, label, epoch, target, in_ttft_gate):
    raws = list(level.glob('raw*.jsonl'))
    if len(raws) != 1:
        raise ValueError(f'expected exactly one measured raw file: {level}')
    raw_path, log_path = raws[0], level / 'server.log'
    raw = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
    by_id = {r['req_id']: r for r in raw}
    if len(raw) != len(by_id) or any(r.get('error') or r.get('error_class') for r in raw):
        raise ValueError(f'duplicate/error raw rows: {level}')
    traces, selected, last, excerpts = {}, {}, {}, []
    digest = hashlib.sha256()
    for line_number, line in enumerate(log_path.open('rb'), 1):
        digest.update(line)
        marker = b'[ax-prefix-decision] '
        if marker not in line:
            continue
        decision = json.loads(line.split(marker, 1)[1])
        e, seq = decision['epoch'], decision['sequence']
        if seq != last.get(e, 0) + 1:
            raise ValueError(f'decision sequence gap at {log_path}:{line_number}')
        last[e] = seq
        if e != epoch:
            continue
        for rid in decision['admitted']:
            selected.setdefault(rid, seq)
        for item in decision['candidates']:
            if item['rid'] not in by_id:
                continue
            record = dict(label=label, sequence=seq, t=decision['t'],
                          source_line=line_number, **item)
            traces.setdefault(item['rid'], []).append(record)
            if item['rid'] == target:
                excerpts.append(dict(label=label, source_line=line_number,
                    epoch=e, sequence=seq, t=decision['t'], target=item,
                    admitted=decision['admitted'], continuation=decision['plan']['continuation'],
                    admitted_candidates=[r for r in decision['candidates']
                                         if r['rid'] in decision['admitted']]))
    chains = []
    for rid, row in by_id.items():
        if not in_ttft_gate(row, 'chain_start'):
            continue
        recv, forward, first = (row[k] for k in ('t_recv_s', 't_exec_start_s', 't_first_token_s'))
        if not recv <= forward <= first or abs(first-recv-row['ttft_s']) > 1e-5:
            raise ValueError(f'inconsistent server timing: {label} {rid}')
        samples = traces.get(rid, [])
        held = [r for r in samples if r['effective_held']]
        result = dict(label=label, req_id=rid, prompt=row['prompt_tokens'],
            cached=row['cached_tokens'], ttft_s=row['ttft_s'],
            wait_to_forward_s=forward-recv, forward_to_first_s=first-forward,
            observed_decisions=len(samples), effective_held_decisions=len(held),
            first_unheld_sequence=next((r['sequence'] for r in samples if not r['effective_held']), None),
            admitted_sequence=selected.get(rid),
            max_held_depth=max((r['held_depth'] for r in held), default=0))
        for threshold in (64, 128, 256):
            result[f'held_depth_below_{threshold}_samples'] = sum(0 < r['held_depth'] < threshold for r in held)
            # Only adjacent global decision rounds where this request appears
            # in both snapshots. This is an observed interval, not causal time.
            intervals = [b['t']-a['t'] for a, b in zip(samples, samples[1:])
                         if b['sequence'] == a['sequence']+1 and a['effective_held']
                         and 0 < a['held_depth'] < threshold]
            if any(dt < 0 for dt in intervals):
                raise ValueError('non-monotonic decision clock')
            result[f'observed_intervals_below_{threshold}_s'] = sum(intervals)
        chains.append(result)
    if target not in traces:
        raise ValueError(f'target absent from requested epoch: {level}')
    receipt = dict(label=label, level=str(level), raw_rows=len(raw),
                   raw_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                   server_sha256=digest.hexdigest(), decision_epochs=last,
                   target=next(r for r in chains if r['req_id'] == target))
    return receipt, chains, excerpts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--harness', type=Path, required=True)
    parser.add_argument('--level', action='append', required=True, help='label=/path/to/Nxx')
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--rid', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.harness.resolve()))
    from s1_common import in_ttft_gate
    receipts, chains, excerpts = [], [], []
    for value in args.level:
        label, directory = value.split('=', 1)
        receipt, rows, points = analyze(Path(directory), label, args.epoch, args.rid, in_ttft_gate)
        receipts.append(receipt)
        chains.extend(rows)
        excerpts.extend(points)
    summary = dict(scope='DIAGNOSTIC: observed trace windows, no full-cohort score',
        limitations='Snapshot intervals are not continuous holds or causal delay. '
                    'Threshold columns are sensitivity counts, not proof of the active cache grid. '
                    'Neither a counterfactual schedule nor saved chain counts are predicted.',
        target=args.rid, inputs=receipts)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    write_csv(args.out / 'chain-holds.csv', chains)
    with (args.out / 'target-decisions.jsonl').open('w') as handle:
        for row in excerpts:
            handle.write(json.dumps(row, separators=(',', ':')) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
