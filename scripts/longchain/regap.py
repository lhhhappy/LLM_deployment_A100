#!/usr/bin/env python3
"""Derive a duration-based gap sensitivity variant: same chains, prompts, output budgets and
replay order; imputed replay gaps are rescaled using an estimated service cost and source chain duration
(chains.jsonl: last_end_offset_ms - first_dispatch_offset_ms). This does not reconstruct observed request gaps.

Why: the public-prefix gap pool omits the full source history's long tail. A per-chain stretch is useful as a
sensitivity experiment, but also shifts the median and cannot recover event placement or tool/think decomposition.
Its effects on instantaneous activity and cache reuse must be measured. Non-imputed public gaps are kept unchanged.

Rule per chain (all quantities in seconds):
  target = clamp(real_duration - sum(ttft_s + max_output_i * tpot_s), 0, chain_cap)   # harness caps a chain at 3600
  imputed gaps are multiplied by f = (target - valid_gaps) / imputed_gaps, f >= 1 unless --allow-scale-down;
  every gap is then capped at gap_cap (the organizer's min(tool,300)+min(think,10) formula gives at most 310 s);
  the time cut by the cap is dropped, as in the organizer's own replay (their raw gaps over 310 s are truncated and
  not redistributed: 3 of 411 public gaps, 4069 s raw replayed as 318 s), unless --redistribute is given.
Chains without an organizer duration, with a single request, or with no imputed gap are unchanged.

Usage: regap.py --src cache/s1-dev-longchain-v3 --organizer s1-dev/data/dev-combined-v1/chains.jsonl
               --out cache/s1-dev-longchain-v3g --set s1-dev-longchain-v3g [--ttft-s 1.5 --tpot-s 0.03]
Writes a fresh derivative with updated request provenance and a complete manifest. Body shards are linked to
the parent, not copied. --metadata-only permits missing source assets but marks the result incomplete.
"""
import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
if __package__:
    from .longchain_metadata import checked_output, publish
else:
    from longchain_metadata import checked_output, publish


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def q(v, p):
    s = sorted(v)
    return s[min(len(s) - 1, int(p * len(s)))] if s else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True)
    ap.add_argument('--organizer', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--set', required=True)
    ap.add_argument('--ttft-s', type=float, default=1.5)
    ap.add_argument('--tpot-s', type=float, default=0.03)
    ap.add_argument('--gap-cap-s', type=float, default=310.0)
    ap.add_argument('--chain-cap-s', type=float, default=3600.0)
    ap.add_argument('--allow-scale-down', action='store_true')
    ap.add_argument('--redistribute', action='store_true', help='spread the time cut by gap_cap over the other imputed gaps (not the organizer rule)')
    ap.add_argument('--metadata-only', action='store_true', help='allow missing parent assets; output remains explicitly incomplete')
    a = ap.parse_args()
    src, out = checked_output(a.src, a.out)
    rows = [json.loads(l) for l in open(src / 'requests.jsonl')]
    by_id = {f"{r['pack']}:{r['view']}:{r['logical_call_id']}": r for r in rows}
    cohort = json.loads((src / 'cohort.json').read_text())
    organizer = {c['chain_id']: c for c in (json.loads(l) for l in open(a.organizer))}
    stats = dict(chains=0, changed=0, no_duration=0, single=0, no_imputed=0, capped_chain=0, capped_gaps=0,
                 scaled_down_skipped=0, factors=[], real_durations=[], before=[], after=[])
    for ch in cohort['chains']:
        stats['chains'] += 1
        reqs = [by_id[rid] for rid in ch['req_ids']]
        org = organizer.get(ch['chain_id'])
        if org is None:
            stats['no_duration'] += 1
            continue
        if len(reqs) < 2:
            stats['single'] += 1
            continue
        real = (org['last_end_offset_ms'] - org['first_dispatch_offset_ms']) / 1000.0
        service = sum(a.ttft_s + (r.get('max_output_i') or 0) * a.tpot_s for r in reqs)
        target = min(a.chain_cap_s, max(0.0, real - service))
        if real - service > a.chain_cap_s:
            stats['capped_chain'] += 1
        tail = reqs[1:]  # the head has no gap
        valid = [r for r in tail if not r.get('gap_imputed')]
        imputed = [r for r in tail if r.get('gap_imputed')]
        valid_sum = sum((r.get('replay_gap_ms') or 0) for r in valid) / 1000.0
        imputed_sum = sum((r.get('replay_gap_ms') or 0) for r in imputed) / 1000.0
        before = valid_sum + imputed_sum
        stats['real_durations'].append(real)
        stats['before'].append(before)
        if not imputed or imputed_sum <= 0:
            stats['no_imputed'] += 1
            stats['after'].append(before)
            continue
        f = (target - valid_sum) / imputed_sum
        if f < 1.0 and not a.allow_scale_down:
            stats['scaled_down_skipped'] += 1
            stats['after'].append(before)
            continue
        f = max(0.0, f)
        new = {}
        overflow = 0.0
        for r in imputed:
            g = (r.get('replay_gap_ms') or 0) / 1000.0 * f
            if g > a.gap_cap_s:
                overflow += g - a.gap_cap_s
                g = a.gap_cap_s
                stats['capped_gaps'] += 1
            new[r['logical_call_id']] = g
        room = [r for r in imputed if new[r['logical_call_id']] < a.gap_cap_s]
        if a.redistribute and overflow > 0 and room:
            share = overflow / len(room)
            for r in room:
                new[r['logical_call_id']] = min(a.gap_cap_s, new[r['logical_call_id']] + share)
        for r in imputed:
            r['replay_gap_ms'] = int(round(new[r['logical_call_id']] * 1000))
            r['gap_regap_factor'] = round(f, 4)
        stats['changed'] += 1
        stats['factors'].append(f)
        stats['after'].append(valid_sum + sum(new.values()))
    summary = dict(
        kind='duration-based-gap-sensitivity', set=a.set, parent_set=cohort['set'],
        parent_requests_sha256=sha(src / 'requests.jsonl'), parent_cohort_sha256=json.loads((src / 'cohort.json').read_text())['cohort_sha256'],
        organizer_chains_sha256=sha(a.organizer),
        rule=dict(ttft_s=a.ttft_s, tpot_s=a.tpot_s, gap_cap_s=a.gap_cap_s, chain_cap_s=a.chain_cap_s, allow_scale_down=a.allow_scale_down,
                  redistribute=a.redistribute),
        chains=stats['chains'], changed=stats['changed'], unchanged=dict(no_duration=stats['no_duration'], single=stats['single'],
                                                                          no_imputed=stats['no_imputed'], would_scale_down=stats['scaled_down_skipped']),
        chains_over_harness_cap=stats['capped_chain'], gaps_capped=stats['capped_gaps'],
        factor=dict(median=q(stats['factors'], .5), p90=q(stats['factors'], .9), max=max(stats['factors']) if stats['factors'] else None),
        chain_gap_seconds=dict(before_mean=statistics.mean(stats['before']) if stats['before'] else None,
                               after_mean=statistics.mean(stats['after']) if stats['after'] else None,
                               organizer_duration_mean=statistics.mean(stats['real_durations']) if stats['real_durations'] else None),
        cohort_sha256=cohort['cohort_sha256'],
    )
    gaps_after = [(r.get('replay_gap_ms') or 0) / 1000.0 for r in rows if r.get('replay_gap_ms')]
    summary['per_request_gap_s'] = dict(median=q(gaps_after, .5), mean=statistics.mean(gaps_after) if gaps_after else None,
                                       p90=q(gaps_after, .9), max=max(gaps_after) if gaps_after else None)
    manifest = publish(src, out, rows, name=a.set, operation=summary, metadata_only=a.metadata_only)
    summary['status'] = manifest['status']
    summary['requests_sha256'] = manifest['artifacts']['requests.jsonl']
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
