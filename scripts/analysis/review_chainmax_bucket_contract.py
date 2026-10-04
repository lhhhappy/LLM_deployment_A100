#!/usr/bin/env python3
"""Compare source phase labels with the original harness's position-first gates.

This does not infer hidden/official dataset composition and never changes data.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--harness', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.harness))
    from s1_common import load_index, in_ttft_gate, q
    rows, chains, grouped = load_index(str(args.source))
    cells = defaultdict(list)
    mapped = []
    for rid, r in rows.items():
        bucket = next(g for g in ('chain_start', 'turn_start', 'overall_intra')
                      if in_ttft_gate(r, g))
        cells[(r['phase'], bucket)].append(r)
        mapped.append(dict(req_id=rid, chain_id=r['chain_id'], phase=r['phase'],
                           idx_in_chain=r['_idx_in_chain'], bucket=bucket,
                           prompt=r['glm_tokens'], uncached=r['uncached_expected']))
    summary = []
    for (phase, bucket), rs in sorted(cells.items()):
        summary.append(dict(phase=phase, harness_bucket=bucket, n=len(rs),
            uncached_ge_50k=sum(r['uncached_expected'] >= 50000 for r in rs),
            uncached_ge_100k=sum(r['uncached_expected'] >= 100000 for r in rs),
            uncached_max=max(r['uncached_expected'] for r in rs),
            prompt_p50=q([r['glm_tokens'] for r in rs], .5),
            prompt_p95=q([r['glm_tokens'] for r in rs], .95)))
    alignment = {}
    for same in (True, False):
        heads = [rs[0] for cid, rs in grouped.items()
                 if (rs[0]['dispatch_offset_ms'] == chains[cid]['first_dispatch_offset_ms']) == same]
        alignment['source_first_visible' if same else 'source_first_not_visible'] = dict(
            n=len(heads), phases=dict(Counter(r['phase'] for r in heads)),
            edges=dict(Counter(r['edge_type'] for r in heads)),
            prompt_p50=q([r['glm_tokens'] for r in heads], .5),
            prompt_p90=q([r['glm_tokens'] for r in heads], .9),
            prompt_p95=q([r['glm_tokens'] for r in heads], .95),
            prompt_max=max(r['glm_tokens'] for r in heads),
            prompt_ge_100k=sum(r['glm_tokens'] >= 100000 for r in heads),
            completely_uncached=sum(r['uncached_expected'] == r['glm_tokens'] for r in heads))
    output = dict(scope=__doc__.strip(), source=str(args.source.resolve()),
        harness=str(args.harness.resolve()), n_requests=len(rows), n_chains=len(grouped),
        input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (args.source / 'requests.jsonl', args.source / 'chains.jsonl', args.harness / 's1_common.py')},
        head_phases=dict(Counter(rs[0]['phase'] for rs in grouped.values())),
        source_head_alignment=alignment,
        actual_buckets=dict(Counter(r['bucket'] for r in mapped)),
        phase_bucket_cells=summary, request_mapping=mapped)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in output.items() if k != 'request_mapping'}, indent=2))


if __name__ == '__main__':
    main()
