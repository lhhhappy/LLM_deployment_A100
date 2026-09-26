#!/usr/bin/env python3
"""Freeze a whole-chain subset. No rendering, timing changes or GPU calls."""
import argparse
import collections
import copy
import gzip
import hashlib
import json
import random
from pathlib import Path

import longchain as lc

LABELS = ('1-4', '5-8', '9-15', '16-30', '31-99', '100+')


def bucket(n):
    return sum(n > edge for edge in (4, 8, 15, 30, 99))


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def lines(path, rows):
    with path.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')


def build(parent, out, name, count, seed):
    manifest = json.loads((parent / 'manifest.json').read_text())
    cohort = json.loads((parent / 'cohort.json').read_text())
    chains = list(lc.read_jsonl(parent / 'chains.jsonl'))
    rows = list(lc.read_jsonl(parent / 'requests.jsonl'))
    if not 1 <= count <= len(chains):
        raise ValueError('count outside parent chain count')
    by_id = {c['chain_id']: c for c in chains}
    bins = [[c for c in chains if bucket(c['n_requests']) == i] for i in range(6)]
    quotas = lc.apportion(count, [len(x) for x in bins])
    strata = []
    for group, quota in zip(bins, quotas):
        packs = sorted({c['pack'] for c in group})
        groups = [[c['chain_id'] for c in group if c['pack'] == pack] for pack in packs]
        if groups:
            strata.extend(zip(groups, lc.apportion(quota, [len(g) for g in groups])))
    # Match metadata totals without selecting on any engine performance result.
    features = collections.defaultdict(collections.Counter)
    for r in rows:
        f = features[r['chain_id']]
        for key in ('glm_tokens', 'uncached_expected', 'max_output_i', 'replay_gap_ms'):
            f[key] += r.get(key) or 0
        f['requests'] += 1
        f[r['phase']] += 1
        f['gap_over_30s'] += (r.get('replay_gap_ms') or 0) > 30000
        f['input_over_128k'] += r['glm_tokens'] > 131072
        f['output_over_4k'] += r['max_output_i'] > 4096
    total = sum(features.values(), collections.Counter())
    target = {k: v * count / len(chains) for k, v in total.items()}
    rng = random.Random(seed)
    best = None
    for _ in range(256):
        ids = {cid for group, q in strata for cid in rng.sample(sorted(group), q)}
        values = sum((features[cid] for cid in sorted(ids)), collections.Counter())
        score = sum(((values[k] - v) / max(v, 1)) ** 2 for k, v in target.items())
        if best is None or score < best[0]:
            best = score, ids, values
    score, ids, actual = best
    ordered = [c for c in cohort['chains'] if c['chain_id'] in ids]
    selected = [r for r in rows if r['chain_id'] in ids]
    wanted = {rid for c in ordered for rid in c['req_ids']}
    if len(ordered) != count or len(wanted) != len(selected) or wanted != {lc.req_id(r) for r in selected}:
        raise ValueError('parent cohort/request coverage mismatch')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'bodies').mkdir()
    (out / 'samples').mkdir()
    shard = f'bodies/{name}.jsonl.gz'
    found = set()
    with (out / shard).open('wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, compresslevel=1, mtime=0) as dst:
        for ref in sorted({r['body_ref'] for r in selected}):
            with gzip.open(parent / ref, 'rb') as src:
                for line in src:
                    rid = json.loads(line)['req_id']
                    if rid in wanted:
                        if rid in found:
                            raise ValueError(f'duplicate body: {rid}')
                        found.add(rid)
                        dst.write(line)
    if found != wanted:
        raise ValueError('missing selected bodies')
    for row in selected:
        row['body_ref'] = shard
    selected_chains = [c for c in chains if c['chain_id'] in ids]
    prov = [p for p in lc.read_jsonl(parent / 'provenance.jsonl') if p['req_id'] in wanted]
    lines(out / 'requests.jsonl', selected)
    lines(out / 'chains.jsonl', selected_chains)
    lines(out / 'provenance.jsonl', prov)
    lines(out / 'event-plans.jsonl', (p for p in lc.read_jsonl(parent / 'event-plans.jsonl') if p['chain_id'] in ids))
    lines(out / f'samples/{name}.jsonl', ({'chain_id': c['chain_id'], 'set_role': 'whole-chain-stratified-subset'} for c in ordered))
    keys = sorted({f"{by_id[cid]['pack']}|{LABELS[bucket(by_id[cid]['n_requests'])]}" for cid in ids})
    child = dict(cohort, set=name, seed=seed, n_chains=count, n_requests=len(selected), chains=ordered,
                 n_strata=len(keys), strata_keys=keys)
    child['cohort_sha256'] = hashlib.sha256(json.dumps(ordered, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
    dump(out / 'cohort.json', child)
    receipt = dict(method='whole-chain-length-then-pack-stratified-256-metadata-balanced-draws',
                   seed=seed, parent=str(parent), parent_manifest_sha256=lc.file_digest(parent / 'manifest.json'),
                   subset_script_sha256=lc.file_digest(Path(__file__)),
                   bins=[dict(length=label, parent=len(group), subset=q) for label, group, q in zip(LABELS, bins, quotas)],
                   target_totals=target, actual_totals=actual, selection_score=score,
                   selected_chain_ids=[c['chain_id'] for c in ordered],
                   preserved='Whole chains, parent cohort order, body bytes, phases, output budgets and gaps unchanged.',
                   validation='NOT_SELF_CHECKED; extraction coverage only; no GPU measurement or official representativeness claim.')
    dump(out / 'subset-selection.json', receipt)
    result = copy.deepcopy(manifest)
    result.update(set=name, status='BUILT_UNVALIDATED', n_chains=count, n_requests=len(selected),
                  selected_chain_ids=[c['chain_id'] for c in ordered],
                  chain_summaries=[c for c in manifest['chain_summaries'] if c['chain_id'] in ids],
                  original_requests=sum(p['kind'] == 'original' for p in prov),
                  donor_usage=dict(collections.Counter(p['donor_mode'] for p in prov if p.get('donor_mode'))),
                  actual_prompt_sum=sum(r['glm_tokens'] for r in selected),
                  actual_output_sum=sum(r['max_output_i'] for r in selected),
                  selected_source_aggregate=lc.aggregate([c['source_chain_targets'] for c in selected_chains]),
                  subset_derivation=receipt)
    result['artifacts'] = {str(p.relative_to(out)): lc.file_digest(p) for p in sorted(out.rglob('*')) if p.is_file()}
    dump(out / 'manifest.json', result)
    print(json.dumps(dict(root=str(out), chains=count, requests=len(selected), bins=receipt['bins'],
                          phase_counts=dict(collections.Counter(r['phase'] for r in selected)),
                          mean_gap_s=actual['replay_gap_ms']/len(selected)/1000), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=Path('data/s1-dev-longchain'))
    parser.add_argument('--out', type=Path, default=Path('data/s1-dev-longchain-lite'))
    parser.add_argument('--set', default='s1-dev-longchain-lite')
    parser.add_argument('--chains', type=int, default=64)
    parser.add_argument('--seed', type=int, default=20260924)
    args = parser.parse_args()
    build(args.parent, args.out, args.set, args.chains, args.seed)
