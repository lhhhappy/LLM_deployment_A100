#!/usr/bin/env python3
"""Read-only E2 numeric cross-check; compare raw tensors and decoded token IDs."""
import argparse
import json
from pathlib import Path
import torch


def first_difference(a, b):
    assert len(a) == len(b) == 32
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_root', type=Path)
    args = p.parse_args()
    reports = {v: json.loads((args.run_root / f'trace-{v}/numeric_summary.json').read_text())
               for v in ('on', 'off')}
    assert reports['on']['selected_pairs'] == reports['off']['selected_pairs']
    assert all(len(r['results']) == 3 for r in reports.values())
    summary = []
    for i in range(3):
        row, raw = {'pair': reports['on']['selected_pairs'][i]}, {}
        for v, report in reports.items():
            r = report['results'][i]
            for stage_index, stage in ((0, 'cold0'), (1, 'cold1'), (2, 'warm')):
                raw[v, stage] = torch.load(args.run_root / f'trace-{v}' / r['runs'][stage_index]['raw_file'],
                                          map_location='cpu', weights_only=True)['logits'].float()
            row[v] = {k: r[k] for k in ('cold_repeat_max_abs', 'tolerance_2x_cold_noise',
                       'cold_warm_max_abs', 'cold_warm_rms', 'greedy_32_equal',
                       'warm_restored_previous_role_split', 'passed')}
            row[v]['first_greedy_difference_zero_based'] = first_difference(r['runs'][0]['greedy_ids'], r['runs'][2]['greedy_ids'])
        row['cross_on_off_cold_max_abs'] = (raw['on', 'cold0'] - raw['off', 'cold0']).abs().max().item()
        row['cross_on_off_warm_max_abs'] = (raw['on', 'warm'] - raw['off', 'warm']).abs().max().item()
        row['cross_on_off_warm_first_greedy_difference_zero_based'] = first_difference(
            reports['on']['results'][i]['runs'][2]['greedy_ids'],
            reports['off']['results'][i]['runs'][2]['greedy_ids'])
        summary.append(row)
    print(json.dumps({'metric': 'full-vocabulary first-output-token raw logits', 'pairs': summary}, indent=2))


if __name__ == '__main__':
    main()
