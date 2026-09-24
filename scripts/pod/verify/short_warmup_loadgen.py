#!/usr/bin/env python3
"""Fixed representative warmup only; never changes measured replay or payloads.

rep16-v1 selects eight original adjacent request pairs from distinct chains,
nearest to fixed context targets, with each original output budget <=1024.
This is intentionally incomplete shape coverage for local diagnostics.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

PROFILE = 'rep16-v1'
TARGETS = (8192, 16384, 32768, 65536, 98304, 131072, 196608, 245760)


def select_pairs(chains, rows):
    selected, used = [], set()
    for target in TARGETS:
        candidates = []
        for chain in chains:
            if chain['chain_id'] in used:
                continue
            for i in range(len(chain['req_ids']) - 1):
                ids = chain['req_ids'][i:i+2]
                a, b = (rows[rid] for rid in ids)
                budgets = [int(r.get('max_output_i') or 512) for r in (a, b)]
                if b.get('phase') == 'context_reset' or max(budgets) > 1024:
                    continue
                key = (abs(int(a['glm_tokens']) - target), sum(budgets), chain['chain_id'], i)
                candidates.append((key, chain, ids))
        if not candidates:
            raise ValueError('dataset cannot supply eight distinct representative pairs')
        _, chain, ids = min(candidates, key=lambda item: item[0])
        used.add(chain['chain_id'])
        selected.append(dict(chain, req_ids=list(ids)))
    return selected


def plan_for(chains, rows, shape_key):
    pairs = select_pairs(chains, rows)
    all_shapes = {shape_key(rows[rid]) for c in chains for rid in c['req_ids']}
    covered = {shape_key(rows[rid]) for c in pairs for rid in c['req_ids']}
    manifest = dict(profile=PROFILE, diagnostic_only=True, n_pairs=len(pairs),
                    n_requests=sum(len(c['req_ids']) for c in pairs),
                    targets=list(TARGETS), dataset_shape_buckets=len(all_shapes),
                    selected_shape_buckets=len(covered),
                    missing_shape_buckets=len(all_shapes-covered),
                    pairs=[dict(chain_id=c['chain_id'], req_ids=c['req_ids'],
                                prompt_tokens=[rows[r]['glm_tokens'] for r in c['req_ids']],
                                output_budgets=[rows[r].get('max_output_i') or 512 for r in c['req_ids']])
                           for c in pairs])
    manifest['plan_sha256'] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return pairs, manifest


def validate_records(plan, records):
    expected = [rid for c in plan['pairs'] for rid in c['req_ids']]
    if Counter(r.get('req_id') for r in records) != Counter(expected):
        raise ValueError('representative warmup request census mismatch')
    if any(r.get('error') or r.get('error_class') for r in records):
        raise ValueError('representative warmup contains failed requests')
    budgets = {rid: budget for pair in plan['pairs']
               for rid, budget in zip(pair['req_ids'], pair['output_budgets'], strict=True)}
    # Original call_engine sets ignore_eos=True. An early successful EOF must
    # not count as exercising the unchanged output budget (especially MTP).
    if any(type(r.get('output_tokens')) is not int or r['output_tokens'] != budgets[r['req_id']]
           for r in records):
        raise ValueError('representative warmup output count differs from original budget')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--loadgen', type=Path, required=True)
    args, rest = ap.parse_known_args(argv)
    if rest[:1] == ['--']: rest = rest[1:]
    if '--warmup' not in rest:
        ap.error('this wrapper may only run the unscored warmup phase')
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--out-dir', type=Path, required=True)
    opts, _ = p.parse_known_args(rest)
    opts.out_dir.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('short_original_loadgen', args.loadgen)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    evidence = dict(profile=PROFILE, status='RUNNING', started_s=time.time())
    records = []
    raw = (opts.out_dir/'short_warmup_raw.jsonl').open('w')
    def select(chains, rows, minimum=1):
        pairs, plan = plan_for(chains, rows, module.warmup_shape_key)
        evidence.update(plan)
        (opts.out_dir/'short_warmup_plan.json').write_text(json.dumps(plan, indent=2)+'\n')
        print('SHORT_WARMUP_PLAN '+json.dumps(plan), flush=True)
        return pairs, plan['selected_shape_buckets'], plan['missing_shape_buckets']
    original_drive = module.drive
    def done(rec):  # original drive invokes this under its common lock
        row = module.raw_record(rec)
        records.append(row)
        raw.write(json.dumps(row, ensure_ascii=False)+'\n'); raw.flush()
        print('SHORT_WARMUP_PROGRESS completed=%d req_id=%s error=%s' %
              (len(records), row['req_id'], bool(row.get('error'))), flush=True)
    def drive(*args, **kwargs):
        kwargs['on_done'] = done
        return original_drive(*args, **kwargs)
    module.select_warmup_chains, module.drive = select, drive
    previous_argv = sys.argv
    rc = 2
    try:
        sys.argv = [str(args.loadgen), *rest]
        rc = int(module.main() or 0)
        if rc:
            raise ValueError('original warmup exited unsuccessfully')
        validate_records(evidence, records)
        evidence['status'] = 'COMPLETE'
        return 0
    except Exception as exc:
        evidence['status'] = 'FAILED'
        evidence['error_type'] = type(exc).__name__
        print('SHORT_WARMUP_FAILED '+type(exc).__name__, file=sys.stderr, flush=True)
        return 2
    finally:
        sys.argv = previous_argv
        raw.close()
        evidence.update(finished_s=time.time(), n_completed=len(records))
        evidence['elapsed_s'] = evidence['finished_s']-evidence['started_s']
        (opts.out_dir/'short_warmup_receipt.json').write_text(json.dumps(evidence, indent=2)+'\n')
        print('SHORT_WARMUP_END status=%s completed=%d elapsed_s=%.3f' %
              (evidence['status'], len(records), evidence['elapsed_s']), flush=True)


if __name__ == '__main__':
    sys.exit(main())
