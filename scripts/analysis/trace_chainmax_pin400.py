#!/usr/bin/env python3
"""Extract actual scheduling decisions for selected requests; no replay/model.

The output contains all candidate metadata/plan fields in the relevant opening
window, including held representatives. No prompt bodies are read or exported.
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--log', type=Path, required=True)
    ap.add_argument('--targets', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    targets = json.loads(args.targets.read_text())
    decisions = []
    for line_no, line in enumerate(args.log.open(), 1):
        marker = '[ax-prefix-decision]'
        if marker not in line:
            continue
        row = json.loads(line.split(marker, 1)[1])
        if any(r['rid'] in targets for r in row['candidates']):
            row['source_line'] = line_no
            decisions.append(row)
    assert decisions, 'no matching request decisions'
    t0 = min(r['received'] for d in decisions for r in d['candidates'] if r.get('received'))
    epoch = decisions[0]['epoch']
    decisions = [d for d in decisions if d['epoch'] == epoch and d['t']-t0 < 65]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(log=str(args.log),t0=t0,epoch=epoch,targets=targets,
                                          decisions=decisions),separators=(',',':'))+'\n')
    aliases = {r:str(i) for i,r in enumerate(targets)}
    last = {}
    for d in decisions:
        order = d['plan']['order']
        for r in d['candidates']:
            if r['rid'] not in targets:
                continue
            key = (r['held'], r['held_by'], r['effective_held'],r['state'],r['result'])
            if last.get(r['rid']) == key and r['rid'] not in d['admitted']:
                continue
            last[r['rid']] = key
            print(json.dumps(dict(target=aliases[r['rid']],seq=d['sequence'],t=round(d['t']-t0,3),
                position=order.index(r['rid']) if r['rid'] in order else None,
                state=r['state'],left=r['left'],device=r['device'],due_in=r['due_in'],
                held=r['held'],held_by=r['held_by'],held_depth=r['held_depth'],effective_held=r['effective_held'],
                result=r['result'],continuation=d['plan']['continuation'],admitted=d['admitted'])))


if __name__ == '__main__':
    main()
