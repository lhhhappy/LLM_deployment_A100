"""CPU-only, reproducible request features for completed 035/N22 vs 026/N18.

No engine calls. Log reports are counted inside conservative decode windows;
their durations are NOT interpreted as GPU time or individual token stalls.
Run from any directory: python3 /path/to/evidence/T59/analyze.py
"""
import collections
import csv
import datetime as dt
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RAW = ROOT / 'evidence/L035/raw_dev-combined-v1_N22_1790184432.jsonl'
REF = ROOT / 'evidence/T53/026_N18_raw.jsonl'
LOG = OUT / '035_server.log'
EXPECTED = {
    RAW: '3f8bfa0d683b5ade8caba09d6829754693042589f607a15e7fc3fb7a21a7bd9b',
    REF: '486f041027abde2a1702e0c19190ce11e7001eb0cc4055d3773aff8c95bfdb59',
    LOG: '28c3a9255b5424888d2ab99dfde62e00cce4415baff233584dd5af96b7bb0f1e',
}


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def write_csv(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def utc(t):
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat(timespec='milliseconds')


def minute(t):
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime('%H:%M')


def group_stats(rows, key):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    result = {}
    for name, members in sorted(groups.items()):
        failed = sum(r['fail'] for r in members)
        result[name] = dict(n=len(members), failed=failed, fail_pct=100*failed/len(members),
                            mean_tpot_s=statistics.mean(r['tpot_s'] for r in members))
    assert sum(g['n'] for g in result.values()) == len(rows)
    assert sum(g['failed'] for g in result.values()) == sum(r['fail'] for r in rows)
    return result


def bin_key(value, bounds):
    for lo, hi in zip(bounds, bounds[1:]):
        if lo <= value < hi:
            return f'[{lo},{hi})'
    raise AssertionError(value)


def main():
    for path, digest in EXPECTED.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path
    raw = [json.loads(s) for s in RAW.read_text().splitlines()]
    reference = [json.loads(s) for s in REF.read_text().splitlines()]
    ref = {r['req_id']: r for r in reference}
    assert len(raw) == len(ref) == len(reference) == 722
    assert len({r['req_id'] for r in raw}) == len(raw)
    assert {r['req_id'] for r in raw} == set(ref)
    assert all(r['output_tokens'] > 1 and r['output_tokens'] == r['max_output_i']
               and not r.get('error_class') and not r.get('error')
               and math.isfinite(r['tpot_s']) and r['tpot_s'] >= 0 for r in raw)
    assert all((r['prompt_tokens'], r['idx_in_chain'], r['output_tokens']) ==
               (ref[r['req_id']]['prompt_tokens'], ref[r['req_id']]['idx_in_chain'],
                ref[r['req_id']]['output_tokens']) for r in raw)

    # Split on POSIX newlines: startup progress bars contain carriage returns.
    prefill = []
    for lineno, text in enumerate(LOG.read_bytes().decode().split('\n'), 1):
        if 'Prefill batch' not in text:
            continue
        match = re.match(r'\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) TP0\]', text)
        assert match, (lineno, text)
        second = dt.datetime.strptime(match[1], '%Y-%m-%d %H:%M:%S').replace(tzinfo=dt.timezone.utc).timestamp()
        values = {k: int(v) for k, v in re.findall(r'#([\w-]+): (\d+)', text)}
        prefill.append(dict(line=lineno, utc=match[1], second=second, **values))

    # Validate the UTC offset from unique single-request, complete-batch shapes.
    # Only this whole-second alignment is used, not the failed T56 subsecond map.
    anchors = []
    for i, r in enumerate(raw, 1):
        new = r['prompt_tokens'] - r['cached_tokens']
        if new > 16384 or r['t_first_token_s'] - r['t_exec_start_s'] > 2:
            continue
        candidates = [b for b in prefill if b['new-seq'] == 1 and b['pending-token'] == 0
                      and b['cached-token'] == r['cached_tokens'] and new <= b['new-token'] < new+64
                      and -1.05 < b['second'] - r['t_first_token_s'] < .10]
        if len(candidates) == 1:
            anchors.append(dict(raw_line=i, log_line=candidates[0]['line'],
                                log_minus_first_s=candidates[0]['second']-r['t_first_token_s']))
    assert len(anchors) == 331

    rows, overlaps = [], {}
    for i, r in enumerate(raw, 1):
        old = ref[r['req_id']]
        first = r['client_first_token_at_s']
        duration = r['tpot_s'] * (r['output_tokens'] - 1)
        end = first + duration
        # Keep entire [log_second, log_second+1) inside the window and leave
        # an extra second after its first token to avoid its own finish report.
        events = [b for b in prefill if b['second'] > first+1 and b['second']+1 < end]
        row = dict(raw_line=i, req_id=r['req_id'], pack=r['pack'],
                   chain_id=r['chain_id'], idx_in_chain=r['idx_in_chain'],
                   replay_head=r['idx_in_chain'] == 0, edge_type=r['edge_type'],
                   prompt_tokens=r['prompt_tokens'], cached_tokens=r['cached_tokens'],
                   uncached_tokens=r['prompt_tokens']-r['cached_tokens'],
                   hit_fraction=r['cached_tokens']/r['prompt_tokens'], output_tokens=r['output_tokens'],
                   ttft_s=r['ttft_s'], tpot_s=r['tpot_s'], fail=r['tpot_s'] > .10,
                   decode_duration_s=duration, first_token_utc=utc(first),
                   decode_end_estimated_utc=utc(end), first_token_minute_utc=minute(first),
                   n18_tpot_s=old['tpot_s'], n18_fail=old['tpot_s'] > .10,
                   n18_cached_tokens=old['cached_tokens'], cached_delta=r['cached_tokens']-old['cached_tokens'],
                   tpot_ratio_to_n18=r['tpot_s']/old['tpot_s'],
                   interior_prefill_reports=len(events),
                   interior_prefill_ge8192=sum(b['new-token'] >= 8192 for b in events),
                   interior_prefill_ge16384=sum(b['new-token'] >= 16384 for b in events),
                   interior_prefill_budget_tokens=sum(b['new-token'] for b in events),
                   first_interior_prefill_log_line=events[0]['line'] if events else '',
                   last_interior_prefill_log_line=events[-1]['line'] if events else '')
        rows.append(row)
        if row['fail']:
            overlaps[str(i)] = dict(req_id=r['req_id'], log_lines=[b['line'] for b in events])
    bad = sorted((r for r in rows if r['fail']), key=lambda r: -r['tpot_s'])
    assert len(bad) == 205
    cohorts = group_stats(rows, lambda r: r['first_token_minute_utc'])
    lo = min(r['t_recv_s'] for r in raw)
    hi = max(r['client_finish_at_s'] for r in raw)
    measured_batches = [b for b in prefill if lo < b['second'] and b['second']+1 < hi]
    for key, value in cohorts.items():
        events = [b for b in measured_batches if minute(b['second']) == key]
        value.update(prefill_reports=len(events), prefill_budget_tokens=sum(b['new-token'] for b in events),
                     prefill_ge8192=sum(b['new-token'] >= 8192 for b in events))
    counts = dict(bad_short_output_lt100=sum(r['output_tokens'] < 100 for r in bad),
                  bad_own_uncached_le4096=sum(r['uncached_tokens'] <= 4096 for r in bad),
                  bad_hit_ge90pct=sum(r['hit_fraction'] >= .9 for r in bad),
                  bad_cached_equal_n18=sum(r['cached_delta'] == 0 for r in bad),
                  bad_cached_better_n18=sum(r['cached_delta'] > 0 for r in bad),
                  bad_cached_worse_n18=sum(r['cached_delta'] < 0 for r in bad),
                  bad_same_n18=sum(r['n18_fail'] for r in bad),
                  bad_new_n22=sum(not r['n18_fail'] for r in bad),
                  n18_bad_now_good=sum(r['n18_fail'] and not r['fail'] for r in rows),
                  good_both=sum(not r['n18_fail'] and not r['fail'] for r in rows))
    assert counts['bad_same_n18'] + counts['bad_new_n22'] == 205
    assert counts['bad_same_n18'] + counts['n18_bad_now_good'] == 136
    assert sum(counts[k] for k in ('bad_same_n18','bad_new_n22','n18_bad_now_good','good_both')) == 722
    summary = dict(n=722, failed=205, failed_pct=205/722*100,
                   tpot_mean=statistics.mean(r['tpot_s'] for r in rows),
                   tpot_p95=sorted(r['tpot_s'] for r in rows)[int(.95*len(rows))],
                   actual_prefill_tokens=sum(r['uncached_tokens'] for r in rows),
                   n18_actual_prefill_tokens=sum(r['prompt_tokens']-r['cached_tokens'] for r in reference),
                   counts=counts,
                   by_pack=group_stats(rows, lambda r: r['pack']),
                   by_replay_head=group_stats(rows, lambda r: 'head' if r['replay_head'] else 'followup'),
                   by_edge_type=group_stats(rows, lambda r: r['edge_type']),
                   by_output=group_stats(rows, lambda r: bin_key(r['output_tokens'], [0,60,100,256,512,1024,10**9])),
                   by_prompt=group_stats(rows, lambda r: bin_key(r['prompt_tokens'], [0,32768,65536,131072,10**9])),
                   by_uncached=group_stats(rows, lambda r: bin_key(r['uncached_tokens'], [0,1024,4096,16384,65536,10**9])),
                   by_minute=cohorts,
                   selected_cases=[rows[i-1] for i in (656,652,655,650,647)],
                   method='Observed request features; conservative whole-second log report overlap. '
                          'Decode absolute windows approximate; no GPU duration or causal stall estimate. '
                          'N18/N22 closed-loop arrival times differ. Minutes group by first token, not all active requests.')
    write_csv('all_722.csv', rows)
    write_csv('failures_205.csv', bad)
    write_csv('minute_cohorts.csv', [dict(minute_utc=k, **v) for k,v in cohorts.items()])
    dump('summary.json', summary)
    dump('prefill_report_events.json', prefill)
    dump('failure_log_overlaps.json', overlaps)
    dump('alignment_whole_seconds.json', dict(utc_offset_s=0, anchors=anchors))
    dump('validation.json', dict(passed=True, input_sha256={str(p.relative_to(ROOT)): h for p,h in EXPECTED.items()},
                                 unique_matched_requests=722, failed=205, all_feature_partitions_close=True,
                                 same_prompt_and_output_tokens=True, utc_anchors=len(anchors),
                                 scope='CPU descriptive analysis of existing raw/logs, not a new run or causal A/B.',
                                 excluded_method='T56 subsecond request-batch map failed at raw line 63; '
                                                 'not used. mapping_failure.txt preserves the failure.'))
    print(json.dumps(dict(passed=True, n=722, failed=205, counts=counts,
                          selected_cases=summary['selected_cases'], by_minute=cohorts), indent=2))


if __name__ == '__main__':
    main()
