#!/usr/bin/env python3
"""Extract 128p decision evidence; diagnostic only, never a harness/SLO scorer."""
import argparse
import csv
import json
from pathlib import Path


def analyze(path):
    decisions, finishes = [], []
    for line in path.read_text().splitlines():
        for marker, records in (('[ax-prefix-decision] ', decisions), ('[ax-prefix-finish] ', finishes)):
            if marker in line:
                records.append(json.loads(line.split(marker, 1)[1]))
    last = {}
    rows, selected = {}, {}
    reserved = admitted = 0
    for event in decisions:
        epoch, seq = event['epoch'], event['sequence']
        if seq != last.get(epoch, 0) + 1:
            raise ValueError(f'Incomplete or duplicated decision stream: epoch={epoch} sequence={seq}')
        last[epoch] = seq
        for rid in event['admitted']:
            selected.setdefault((epoch, rid), event['t'])
        reserved += len(event['plan']['ready'])
        admitted += sum(reason == 'admitted' for _, reason in event['result'])
        for item in event['candidates']:
            if not item['producer']:
                continue
            key = epoch, item['rid']
            row = rows.setdefault(key, dict(epoch=epoch, rid=item['rid'], producer=item['producer'],
                received=item.get('received'), dependency_t=event['t'], ready_t=None,
                admitted_t=None, finish_t=None, target=item['target'], ready_device=None))
            if item['state'] == 'READY' and row['ready_t'] is None:
                ready_event = next((e for e in event.get('events', [])
                                   if e['event'] == 'cache_ready' and e['rid'] == item['rid']), None)
                row['ready_t'] = ready_event['t'] if ready_event else event['t']
                row['ready_device'] = item['device']
    for event in finishes:
        row = rows.get((event['epoch'], event['rid']))
        if row is not None:
            row['finish_t'] = event['t']
            row['received'] = event.get('received', row['received'])
    for key, row in rows.items():
        row['admitted_t'] = selected.get(key)
        row['producer_admitted_t'] = selected.get((row['epoch'], row['producer']))
        row['ready_to_admit_ms'] = (1000 * (row['admitted_t'] - row['ready_t'])
            if row['admitted_t'] is not None and row['ready_t'] is not None else None)
        row['server_ttft_s'] = (row['finish_t'] - row['received']
            if row['finish_t'] is not None and row['received'] is not None else None)
    summary = dict(validity='DIAGNOSTIC', decisions=len(decisions), epochs=last,
        dependents=len(rows), reserved_seats=reserved, ready_admissions=admitted,
        prefill_finish_records=len(finishes),
        ready_to_admit_ms=[r['ready_to_admit_ms'] for r in rows.values() if r['ready_to_admit_ms'] is not None],
        limitations='Explicit trace window only. Server completion stamps exclude HTTP/network delay; no SLO verdict.')
    return summary, list(rows.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary, rows = analyze(args.log)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    if rows:
        with (args.output / 'dependents.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
