#!/usr/bin/env python3
"""D1-04: first-output-token logprobs/top-k and greedy 32-ID cache comparison.

SGLang exposes normalized logprobs, NOT raw logits or the whole vocabulary.
The measured max difference covers the union of observed first-token top-k IDs.
Use --expected-cached-tokens B (aligned boundary depth) to verify recovery at B;
without it a cache hit is required but checkpoint identity remains unverified.
One server compares cold/cold/warm; --other-url repeats on a second server and
also compares servers. --calibrate measures only cold twice (no previous prompt).
All runs flush the whole server: use isolated, already authorized endpoints.

Pair JSON: {"previous": "full rendered prompt", "next": "full rendered prompt"}
or --chain-id ID --pair-index 0 to render adjacent original dev requests.
No thinking/history/tool modification. Fixed 32-token ignore_eos is only this
numerical probe, not a capability run or change to the serving/harness budget.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from serving_probe import (REPO, CheckFailed, client_for, flush, harness, integer,
                           payload, require, write_report)


def read_pair(pair_file=None, chain_id=None, index=0, dev_root=REPO/'s1-dev'):
    if pair_file:
        pair = json.loads(Path(pair_file).read_text())
    else:
        common = harness(dev_root)
        root = Path(dev_root) / 'data/dev-combined-v1'
        _, _, groups = common.load_index(str(root))
        require(chain_id in groups and 0 <= index < len(groups[chain_id])-1, 'invalid adjacent dev pair')
        rows = groups[chain_id][index:index+2]
        ids = [r['_req_id'] for r in rows]
        bodies = common.materialize_bodies(str(root), ids)
        renderer = common.Renderer(str(Path(dev_root) / 'glm_tok'))
        pair = dict(zip(('previous', 'next'), (renderer.render(bodies[r]) for r in ids)))
    require(isinstance(pair, dict) and all(isinstance(pair.get(k), str) and pair[k]
            for k in ('previous', 'next')), 'pair needs nonempty previous and next text')
    require(pair['previous'] != pair['next'], 'pair must contain distinct consecutive prompts')
    return pair


def logprob_map(entries):
    require(isinstance(entries, list) and bool(entries), 'missing first-token logprobs')
    result = {}
    for item in entries:
        require(isinstance(item, list) and len(item) >= 2, 'malformed logprob tuple')
        lp, token = item[:2]
        require(isinstance(lp, (int, float)) and not isinstance(lp, bool) and math.isfinite(lp)
                and integer(token), 'invalid logprob/token ID')
        require(token not in result, 'duplicate token ID in logprobs')
        result[token] = lp
    return result


def measure(client, pair, *, warm=False, k=10, token_ids=None, wait=120, previous_tokens=32,
            expected_cached=None):
    flush(client, wait)
    if warm:
        prior = client.json('/generate', payload(pair['previous'], previous_tokens))
        require(prior.get('meta_info', {}).get('cached_tokens') == 0, 'previous prompt was not cold after flush')
        require(prior.get('meta_info', {}).get('completion_tokens') == previous_tokens, 'previous prompt incomplete')
    extra = {'return_logprob': True, 'logprob_start_len': -1, 'top_logprobs_num': k,
             'return_text_in_logprobs': False}
    if token_ids is not None:
        extra['token_ids_logprob'] = token_ids
    response = client.json('/generate', payload(pair['next'], 32, **extra))
    meta = response.get('meta_info', {})
    require(meta.get('completion_tokens') == 32, 'greedy probe did not generate exactly 32 tokens')
    cache = meta.get('cached_tokens')
    require(integer(cache) and (cache > 0 if warm else cache == 0), 'warm hit/cold-zero precondition failed')
    if warm and expected_cached is not None:
        require(cache == expected_cached, 'warm request did not restore expected boundary depth')
    top = meta.get('output_top_logprobs')
    selected = meta.get('output_token_logprobs')
    require(isinstance(top, list) and bool(top), 'output_top_logprobs missing')
    require(isinstance(selected, list) and len(selected) == 32, '32 selected-token logprobs required')
    ids = [entry[1] for entry in selected if isinstance(entry, list) and len(entry) >= 2]
    require(len(ids) == 32 and all(integer(i) for i in ids), 'greedy output token IDs missing')
    if 'output_ids' in response:
        require(response['output_ids'] == ids, 'output IDs/logprob IDs disagree')
    top_map = logprob_map(top[0])
    require(len(top_map) == k and ids[0] in top_map, 'top-k missing requested entries/greedy first token')
    fixed = None
    if token_ids is not None:
        fixed_rows = meta.get('output_token_ids_logprobs')
        require(isinstance(fixed_rows, list) and bool(fixed_rows), 'requested token_ids_logprob not returned')
        fixed = logprob_map(fixed_rows[0])
        require(set(fixed) == set(token_ids), 'fixed-ID logprob coverage incomplete')
        require(set(top_map) <= set(fixed), 'top-k changed outside discovered union; rerun/increase k')
    return {'cached_tokens': cache, 'top_k': top_map, 'fixed_logprobs': fixed,
            'first_selected_logprob': selected[0][0], 'greedy_ids': ids}


def difference(a, b):
    x, y = a['fixed_logprobs'], b['fixed_logprobs']
    require(x is not None and y is not None and set(x) == set(y), 'cannot compare unmatched token IDs')
    return {'max_abs_logprob_diff': max(abs(x[t] - y[t]) for t in x),
            'greedy_32_equal': a['greedy_ids'] == b['greedy_ids'],
            'top_k_set_equal': set(a['top_k']) == set(b['top_k']),
            'compared_token_ids': sorted(x)}


def run_comparison(clients, pair, *, calibrate=False, k=10, wait=120, previous_tokens=32,
                   expected_cached=None, other_expected_cached=None, tolerance_floor=0.0):
    # Discovery then fresh independent measurements over a COMMON union, including
    # each side of a top-k rank crossing. Intersections can hide the biggest error.
    modes = [False, False] if calibrate else [False, False, True]
    union = set()
    depths = [expected_cached, other_expected_cached]
    for server_index, client in enumerate(clients):
        for warm in modes:
            result = measure(client, pair, warm=warm, k=k, wait=wait,
                             previous_tokens=previous_tokens, expected_cached=depths[server_index])
            union.update(result['top_k'])
    reports, all_runs = [], []
    for server_index, client in enumerate(clients):
        runs = [measure(client, pair, warm=warm, k=k, token_ids=sorted(union), wait=wait,
                        previous_tokens=previous_tokens, expected_cached=depths[server_index]) for warm in modes]
        noise = difference(runs[0], runs[1])
        tolerance = max(2 * noise['max_abs_logprob_diff'], tolerance_floor)
        row = {'cold_noise': noise, 'tolerance': tolerance, 'measurements': runs,
               'expected_cached_tokens': depths[server_index],
               'checkpoint_depth_verified': depths[server_index] is not None and not calibrate,
               'passed': noise['greedy_32_equal']}
        if not calibrate:
            comparison = difference(runs[0], runs[2])
            row['cold_vs_warm'] = comparison
            row['passed'] &= comparison['greedy_32_equal'] and comparison['max_abs_logprob_diff'] <= tolerance
        reports.append(row)
        all_runs.append(runs)
    cross = []
    if len(clients) == 2:
        tolerance = max(r['tolerance'] for r in reports)
        for i, label in ((0, 'cold'),) if calibrate else ((0, 'cold'), (2, 'warm')):
            comparison = difference(all_runs[0][i], all_runs[1][i])
            cross.append({'mode': label, **comparison, 'tolerance': tolerance,
                          'passed': comparison['greedy_32_equal'] and comparison['max_abs_logprob_diff'] <= tolerance})
    return {'case': 'D1-04', 'calibration_only': calibrate,
            'metric': 'first-token normalized logprobs on union of observed top-k; NOT full raw logits',
            'checkpoint_depth_verified': all(r['checkpoint_depth_verified'] for r in reports),
            'expected_cached_tokens': expected_cached, 'tolerance_floor': tolerance_floor,
            'servers': reports, 'cross_server': cross,
            'passed': all(r['passed'] for r in reports + cross)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--base-url', default='http://127.0.0.1:8000')
    p.add_argument('--other-url')
    group = p.add_mutually_exclusive_group()
    group.add_argument('--pair-file', type=Path)
    group.add_argument('--chain-id')
    p.add_argument('--pair-index', type=int, default=0)
    p.add_argument('--dev-root', type=Path, default=REPO/'s1-dev')
    p.add_argument('--top-k', type=int, default=10)
    p.add_argument('--calibrate', action='store_true')
    p.add_argument('--expected-cached-tokens', type=int)
    p.add_argument('--other-expected-cached-tokens', type=int,
                   help='optional independent boundary-depth assertion for --other-url')
    p.add_argument('--previous-tokens', type=int, default=32)
    p.add_argument('--tolerance-floor', type=float, default=0.0,
                   help='explicit absolute logprob tolerance floor; default exactly 2x measured cold noise')
    p.add_argument('--flush-wait', type=int, default=120)
    p.add_argument('--timeout', type=float, default=600)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    if min(a.top_k, a.previous_tokens, a.flush_wait) <= 0 or a.timeout <= a.flush_wait:
        p.error('positive budgets required; timeout must exceed flush wait')
    if not math.isfinite(a.tolerance_floor) or a.tolerance_floor < 0:
        p.error('tolerance floor must be finite/nonnegative')
    if any(n is not None and n <= 0 for n in (a.expected_cached_tokens, a.other_expected_cached_tokens)):
        p.error('expected cached depth must be positive')
    if a.dry_run:
        print(json.dumps({'dry_run': True, 'case': 'D1-04', 'servers': 2 if a.other_url else 1,
                          'calibrate': a.calibrate, 'top_k': a.top_k,
                          'steps': ['discover top-k union', 'flush + cold twice',
                                    'unless calibration: flush + previous + next',
                                    'compare union-ID logprobs within 2x cold noise and greedy 32 IDs'],
                          'note': 'no HTTP, pair reads, downloads or file writes'}, indent=2))
        return 0
    if not (a.pair_file or a.chain_id):
        p.error('--pair-file or --chain-id is required')
    try:
        pair = read_pair(a.pair_file, a.chain_id, a.pair_index, a.dev_root)
        clients = [client_for(a.base_url, a.timeout)]
        if a.other_url:
            clients.append(client_for(a.other_url, a.timeout, 'S1_OTHER_API_KEY'))
        report = run_comparison(clients, pair, calibrate=a.calibrate, k=a.top_k, wait=a.flush_wait,
            previous_tokens=a.previous_tokens, expected_cached=a.expected_cached_tokens,
            other_expected_cached=a.other_expected_cached_tokens,
            tolerance_floor=a.tolerance_floor)
    except CheckFailed as e:
        report = {'case': 'D1-04', 'passed': False, 'reason': str(e)}
    except Exception:
        report = {'case': 'D1-04', 'passed': False, 'reason': 'local/HTTP failure; details suppressed'}
    write_report(report, a.out)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
