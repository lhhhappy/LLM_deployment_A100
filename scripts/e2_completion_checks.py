#!/usr/bin/env python3
"""T29 L2 residual checks on an exclusive, already-running trace server.

Reuses original Renderer and replay_chains; full dev prompts unchanged, output4
only for random-weight mechanism tests. IF-08 uses W6's bounded live checker.
All original evidence remains intact; new data only under E2_T29_* run root.
"""
import argparse
from collections import Counter
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import subprocess
import sys

from serving_probe import client_for, flush, harness, payload, require


def read_trace(root):
    return [json.loads(line) for line in (root/'scheduler_trace.jsonl').read_text().splitlines()]


def merge_cases(cold, reminder):
    # Ordered union; retain each complete prefix and use the longer if shared.
    result = {}
    for pair in zip_longest(cold, reminder):
        for row in pair:
            if row is None:
                continue
            old = result.get(row['chain_id'])
            if old:
                n = min(len(old['req_ids']), len(row['req_ids']))
                require(old['req_ids'][:n] == row['req_ids'][:n], 'inconsistent chain prefixes')
            if old is None or len(row['req_ids']) > len(old['req_ids']):
                result[row['chain_id']] = row
    return list(result.values())


def summarize_rounds(events):
    rounds = [e for e in events if e['kind'] == 'round']
    counts = Counter()
    for e in rounds:
        counts.update(e['role_delta'])
    commits = [c for e in rounds for c in e['commits']]
    return {'rounds': len(rounds), 'max_partial': max((e['partial_count'] for e in rounds), default=None),
            'max_batch_size': max((len(e['batch_rids']) for e in rounds), default=0),
            'new_admission_attempts': sum(e['attempts'] for e in rounds),
            'new_commits': sum(c['kind'] == 'new' for c in commits),
            'continuation_commits': sum(c['kind'] == 'continuation' for c in commits),
            'all_commits': len(commits), 'role_stats': dict(counts),
            'role_stats_sum': sum(counts.values()),
            'round_errors': sum(e['kind'] == 'round_error' for e in events)}


def pool_verdict(replay_complete, startup, before, final):
    if not replay_complete:
        return 'blocked'  # empty/equal pools prove nothing if workload never ran
    return 'pass' if final['success'] and final['after'] == before == startup else 'fail'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--case-root', type=Path, required=True)
    p.add_argument('--comparison', type=Path, required=True)
    p.add_argument('--dev-root', type=Path, required=True)
    p.add_argument('--base-url', default='http://127.0.0.1:31000')
    p.add_argument('--max-prompt-tokens', type=int, default=131072)
    p.add_argument('--mixed-only', action='store_true', help='IF-08/role flush already tested in prior run')
    args = p.parse_args()
    root = args.run_root.resolve()
    require(root.is_relative_to(Path('/sjtu/linhang/arena/runs')) and root.name.startswith('E2_T29_'),
            'dedicated T29 run root required')
    report_file = root/'completion_report.json'
    require(not report_file.exists(), 'refuse overwrite')
    report = {'purpose': 'T29 random qfull L2 functional diagnostic; not performance', 'cases': {},
              'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    def save():
        report_file.write_text(json.dumps(report, indent=2)+'\n')
    save()
    client = client_for(args.base_url, 1200)
    startup = [e for e in read_trace(root) if e['kind'] == 'startup']
    require(len(startup) == 1, 'expected exactly one TP1 scheduler startup')
    report['startup_pools'] = startup[0]['pools']

    if not args.mixed_only:
        # Existing, CPU/mock-validated W6 tool; not inferred from exit=0 alone.
        with (root/'if08.log').open('x') as log:
            completed = subprocess.run([sys.executable, str(Path(__file__).with_name('if_checks.py')),
                '--base-url', args.base_url, '--cases', 'IF-08', '--long-tokens', '4096',
                '--flush-wait', '180', '--timeout', '600', '--out', str(root/'if08.json')],
                stdout=log, stderr=subprocess.STDOUT, timeout=900)
        report['cases']['IF-08'] = {'exit_code': completed.returncode, 'report': 'if08.json'}
        save()

        common = harness(args.dev_root)
        data = args.dev_root/'data/dev-combined-v1'
        _, _, groups = common.load_index(str(data))
        renderer = common.Renderer(str(args.dev_root/'glm_tok'))
        comparison = json.loads(args.comparison.read_text())
        pair = next(r for r in comparison['reminder_heavy']['requests'] if r['on_cached'] > r['stock_cached'])
        rows = groups[pair['chain_id']][:pair['idx_in_chain']+1]
        bodies = common.materialize_bodies(str(data), [r['_req_id'] for r in rows])
        texts = [renderer.render(bodies[r['_req_id']]) for r in rows]
        require(hashlib.sha256(texts[-1].encode()).hexdigest() == pair['prompt_sha256'], 'target changed')
        flush(client, 120)
        history = []
        for idx, text in enumerate(texts):
            rid = f't29-role-{idx}'
            result = client.json('/generate', payload(text, 4, rid=rid))
            meta = result['meta_info']
            require(meta['completion_tokens'] == 4, 'generation length mismatch')
            history.append({'rid': rid, 'prompt_tokens': meta['prompt_tokens'], 'cached_tokens': meta['cached_tokens']})
        warm = history[-1]
        previous_rid = history[-2]['rid']
        splits = [e for e in read_trace(root) if e['kind'] == 'split' and e['rid'] == previous_rid]
        proven = warm['cached_tokens'] == pair['on_cached'] and any(e['depth'] == warm['cached_tokens'] for e in splits)
        flush_receipt = flush(client, 120)
        cold = client.json('/generate', payload(texts[-1], 4, rid='t29-role-after-flush'))['meta_info']
        report['cases']['D1-10'] = {'status': 'pass' if proven and cold['cached_tokens'] == 0 else 'fail',
            'role_restore_proven': proven, 'pair': pair, 'history': history, 'previous_splits': splits,
            'flush': flush_receipt, 'after_flush_cached_tokens': cold['cached_tokens']}
        save()

    names = ('cold_heavy', 'reminder_heavy')
    case_paths = [args.case_root/(name+'.json') for name in names]
    original = [json.loads(path.read_text()) for path in case_paths]
    mixed = merge_cases(*original)
    mixed_path = root/'mixed_case.json'
    mixed_path.write_text(json.dumps(mixed, indent=2)+'\n')
    report['mixed_input'] = {'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in case_paths},
        'chains': len(mixed), 'requests': sum(len(row['req_ids']) for row in mixed),
        'concurrency': 4, 'cache_isolation': 'shared-cache; flush before/after, not between active chains'}
    flush(client, 120)
    before = [e for e in read_trace(root) if e['kind'] == 'flush'][-1]['after']
    offset = len(read_trace(root))
    with (root/'mixed_replay.log').open('x') as log:
        completed = subprocess.run([sys.executable, str(Path(__file__).with_name('replay_chains.py')),
            '--dev-root', str(args.dev_root), '--base-url', args.base_url,
            '--case-file', str(mixed_path), '--concurrency', '4', '--max-prompt-tokens', str(args.max_prompt_tokens),
            '--require-json-flush', '--output', str(root/'mixed_n4')],
            stdout=log, stderr=subprocess.STDOUT, timeout=1800)
    measured = summarize_rounds(read_trace(root)[offset:])
    summary_path = root/'mixed_n4/summary.json'
    measured['replay_summary'] = json.loads(summary_path.read_text()) if summary_path.exists() else None
    measured['exit_code'] = completed.returncode
    replay_complete = (completed.returncode == 0 and measured['replay_summary'] is not None and
        measured['replay_summary']['requests'] == report['mixed_input']['requests'] and
        measured['replay_summary']['errors'] == 0)
    measured['replay_complete'] = replay_complete
    stable = (replay_complete and measured['rounds'] > 0 and
              measured['max_partial'] <= 1 and measured['round_errors'] == 0)
    measured['partial_invariant_passed'] = stable
    measured['stats_equal_all_commits'] = measured['role_stats_sum'] == measured['all_commits']
    measured['stats_equal_new_commits'] = measured['role_stats_sum'] == measured['new_commits']
    measured['stats_equal_admission_attempts'] = measured['role_stats_sum'] == measured['new_admission_attempts']
    # Registry says total admissions: retain all commits AND new-only numbers,
    # do not silently redefine denominator to make incomplete counters pass.
    measured['status'] = 'pass' if stable and measured['stats_equal_all_commits'] else 'fail'
    report['cases']['D1-05'] = measured
    save()
    flush(client, 120)
    final = [e for e in read_trace(root) if e['kind'] == 'flush'][-1]
    report['cases']['D1-08'] = {'status': pool_verdict(replay_complete, report['startup_pools'], before, final),
        'startup': report['startup_pools'], 'before_mixed_after_flush': before,
        'before_final_flush': final['before'], 'after_final_flush': final['after']}
    save()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
