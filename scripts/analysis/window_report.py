#!/usr/bin/env python3
"""Automatic same-ID comparisons and bounded alignment evidence for a saved window."""
import argparse
import csv
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import time

import compare_window
import window_gates as gates


def collect_alignment(path, start, end):
    """Runs read-only on the pod; counts all events, bounds returned request details."""
    import datetime
    import hashlib
    import json
    import re
    from pathlib import Path

    requests, pending, counts = {}, {}, {}
    malformed = 0
    digest = hashlib.sha256()
    p = Path(path)
    size = p.stat().st_size
    with p.open('rb') as f:
        while f.tell() < size:
            line = f.readline(size-f.tell())
            if not line.endswith(b'\n') or b'[ax-chunk-alignment]' not in line:
                continue
            text = line.decode('utf-8')
            stamp = re.match(r'\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) TP0\]', text)
            if not stamp:
                malformed += 1
                continue
            at = datetime.datetime.strptime(stamp[1], '%Y-%m-%d %H:%M:%S').replace(
                tzinfo=datetime.timezone.utc).timestamp()
            if at > int(end):
                continue
            if at < int(start):
                if '"budget_exhausted"' in text:
                    counts['budget_exhausted'] = 1
                continue
            digest.update(line)
            try:
                event = json.loads(text.split('[ax-chunk-alignment] ', 1)[1])
                kind = event['event']
                counts[kind] = counts.get(kind, 0)+1
                if kind == 'budget_exhausted':
                    continue
                rid = event['rid']
                r = requests.setdefault(rid, dict(rid=rid, defers=0, resumes=0,
                    deferred_decisions=0, elapsed_s=0., max_elapsed_s=0.,
                    checkpoint_plans=0, unaligned_plans=0))
                if kind == 'defer':
                    r['defers'] += 1
                    pending[rid] = at
                elif kind == 'resume':
                    r['resumes'] += 1
                    r['elapsed_s'] += event['elapsed_s']
                    r['max_elapsed_s'] = max(r['max_elapsed_s'], event['elapsed_s'])
                    r['deferred_decisions'] += event['deferred_decisions']
                    pending.pop(rid, None)
                elif kind == 'checkpoint_plan':
                    r['checkpoint_plans'] += bool(event['track_mask'])
                    r['unaligned_plans'] += bool(event['unaligned_prefix'])
                    r['last_checkpoint_plan'] = event
            except (KeyError, ValueError, TypeError):
                malformed += 1
    for rid, at in pending.items():
        requests[rid]['unmatched_defer_age_s_approx'] = max(0., end-at)
    details = sorted(requests.values(), key=lambda r: (
        r.get('unmatched_defer_age_s_approx', 0), r['max_elapsed_s']), reverse=True)
    return dict(scope='measurement through raw snapshot time; log timestamps have second precision',
        observed_through_s=end, source_bytes=size, matched_lines_sha256=digest.hexdigest(),
        counts=counts, malformed=malformed, requests_seen=len(requests),
        resumed_episodes=sum(r['resumes'] for r in details),
        unmatched_defers=len(pending),
        max_resume_s=max((r['max_elapsed_s'] for r in details), default=0),
        unaligned_plans=sum(r['unaligned_plans'] for r in details),
        budget_exhausted=bool(counts.get('budget_exhausted')),
        requests=details[:128], omitted_request_details=max(0, len(details)-128),
        limitation='checkpoint_plan is planned, not published/consumed state; unmatched defer is not proof of a stall')


def read_alignment(meta, job, call):
    # No new engine instrumentation or writes; the existing TP0 trace is scanned once per report.
    source = inspect.getsource(collect_alignment)
    code = (source+'\nimport json,sys\nprint("ALIGNMENT " + json.dumps(collect_alignment('
            'sys.argv[1],float(sys.argv[2]),float(sys.argv[3]))))')
    output = call(code, '/tmp/ax/runs/'+job+'/server.log',
                  meta['flush']['flush_finished_s'], meta['observed_at'])
    lines = [s[len('ALIGNMENT '):] for s in output.splitlines() if s.startswith('ALIGNMENT ')]
    if len(lines) != 1:
        raise ValueError('missing/ambiguous alignment receipt')
    return json.loads(lines[0])


def baseline_rows(path):
    rows = gates.load_raw(path)
    receipt = path.parent/'level_verdict.json'
    if receipt.is_file():
        verdict = json.loads(receipt.read_text())
        raw_name = path.name
    else:
        snapshot = json.loads((path.parent/'snapshot.json').read_text())
        verdict, raw_name = snapshot.get('verdict', {}), snapshot['raw']
    if verdict.get('status') != 'VALID' or verdict.get('rows') != len(rows) or verdict.get('raw') != raw_name:
        raise ValueError('baseline is not a matching VALID full-data run')
    differences = gates.check_score(gates.stats(rows, gates.score_formal.load_harness()),
                                   path.parent/'score_formal.json')
    if differences:
        raise ValueError('baseline raw/score mismatch: '+str(differences))
    return rows


def build_report(out, baseline, label, job, call=None, alignment=False):
    meta = json.loads((out/'snapshot.json').read_text())
    expected_sha = meta.get('downloaded_raw_sha256') or meta.get('raw_sha256')
    if expected_sha is None:
        # Upgrade an older saved window through its original hashed gzip, never
        # by recompression (zlib versions can produce different gzip bytes).
        from window_watch import download
        download(meta, out, call)
        (out/'snapshot.json').write_text(json.dumps(meta, ensure_ascii=False)+'\n')
        expected_sha = meta['downloaded_raw_sha256']
    raw = (out/'raw.jsonl').read_bytes()
    # Bind the local snapshot to the pod's immutable, SHA-checked transfer receipt.
    raw_sha = hashlib.sha256(raw).hexdigest()
    if raw_sha != expected_sha:
        raise ValueError('raw does not match snapshot transfer SHA256')
    base, rows = baseline_rows(baseline), gates.load_raw(out/'raw.jsonl')
    scorer = gates.score_formal.load_harness()
    summary, details = compare_window.compare(base, rows, scorer)
    run = json.loads((out/'window_gates.json').read_text())[job]
    if run['n_rows'] != len(rows) or summary['candidate'] != run['whole']:
        raise ValueError('comparison differs from saved window statistics')
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(meta['observed_at']))
    dest = out/'analysis'/(stamp+'-'+meta['sha256'][:8])
    dest.mkdir(parents=True, exist_ok=True)
    (dest/'raw.jsonl.gz').write_bytes(gzip.compress(raw, mtime=0))
    (dest/'snapshot.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2)+'\n')
    with (dest/'paired-completed.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    b, c = summary['same_request_baseline'], summary['candidate']
    bg, cg = b['gates'], c['gates']
    bad_chain = [d for d in details if 'chain_start' in d['candidate_failed'].split(';')]
    timed = [d for d in bad_chain if gates.finite(d['candidate_recv_to_exec_s'])]
    summary['chain_wait'] = dict(bad=len(bad_chain), known_timing=len(timed),
        ttft_s=sum(d['candidate_ttft_s'] for d in timed),
        recv_to_exec_s=sum(d['candidate_recv_to_exec_s'] for d in timed),
        queue_dominant_80pct=sum(d['candidate_recv_to_exec_s'] >= .8*d['candidate_ttft_s'] for d in timed))
    summary['baseline_label'] = label
    summary['baseline_sha256'] = hashlib.sha256(baseline.read_bytes()).hexdigest()
    summary['raw_sha256'] = hashlib.sha256(raw).hexdigest()
    summary['complete_data'] = run.get('complete_data', False)
    trace = None
    if alignment:
        try:
            trace = read_alignment(meta, job, call)
            (dest/'alignment.json').write_text(json.dumps(trace, ensure_ascii=False, indent=2)+'\n')
            by_id = {d['req_id']:d for d in details}
            summary['traced_completed_requests'] = [by_id[r['rid']] for r in trace['requests'] if r['rid'] in by_id]
            chains = {}
            for row in base:
                chains.setdefault(row['chain_id'], []).append(row)
            successors = {}
            for chain in chains.values():
                ordered = sorted(chain, key=lambda row: row['idx_in_chain'])
                successors.update({a['req_id']:b['req_id'] for a, b in zip(ordered, ordered[1:])})
            summary['traced_successors'] = [dict(predecessor=r['rid'],
                successor=successors.get(r['rid']),
                comparison=by_id.get(successors.get(r['rid'])),
                note='same-chain next frozen request; association alone is not causal attribution')
                for r in trace['requests']]
        except Exception as exc:
            summary['alignment_error'] = str(exc)[:400]
    summary['alignment'] = trace
    gates_text = '，'.join(f'{k} {bg[k]["over"]}→{cg[k]["over"]}'
                           for k in ('chain', 'fast', 'overall', 'turn'))
    change = summary['gate_changes']['chain_start']
    delta = c['tokens']['uncached']-b['tokens']['uncached']
    percent = 100*delta/b['tokens']['uncached'] if b['tokens']['uncached'] else 0
    bt, ct = b['tpot'], c['tpot']
    def ms(value):
        return f'{value*1000:.2f}' if value is not None else '未知'
    brief = (f'对{label}同ID {len(rows)}条：超时{gates_text}；chain修复{change["repaired"]}/新增{change["new"]}。'
             f'TPOT均值/p95 {ms(bt["mean"])}/{ms(bt["p95"])}→{ms(ct["mean"])}/{ms(ct["p95"])} ms；'
             f'重算{delta:+,} tokens（{percent:+.1f}%）；错误{c["errors"]}。')
    if trace is not None:
        brief += (f'对齐暂缓{trace["counts"].get("defer", 0)}次、已见恢复{trace["resumed_episodes"]}次，'
                  f'最长{trace["max_resume_s"]:.3f}s、未见恢复{trace["unmatched_defers"]}次；'
                  f'不对齐计划{trace["unaligned_plans"]}条。')
        if trace['budget_exhausted'] or trace['malformed']:
            brief += '观测不完整，详见alignment.json。'
    elif alignment:
        brief += '对齐观测暂不可用，自动重试。'
    brief += '全量已闭合，判分见原harness。' if summary['complete_data'] else '仅已完成样本，未完成请求未知，不判整档。'
    summary['brief'] = brief
    (dest/'analysis.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    (dest/'brief.txt').write_text(brief+'\n')
    latest = dict(snapshot_sha256=meta['sha256'], observed_at=meta['observed_at'],
        rows=len(rows), brief=brief, path=str(dest), alignment_error=summary.get('alignment_error'))
    tmp = out/'auto-analysis.json.part'
    tmp.write_text(json.dumps(latest, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(out/'auto-analysis.json')
    return latest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('job')
    ap.add_argument('--baseline-raw', type=Path, required=True)
    ap.add_argument('--baseline-label', default='baseline')
    ap.add_argument('--alignment-trace', action='store_true')
    args = ap.parse_args()
    from window_watch import ROOT, remote
    result = build_report(ROOT/'evidence'/('L'+args.job)/'window', args.baseline_raw,
                          args.baseline_label, args.job, remote, args.alignment_trace)
    print(result['brief'])


if __name__ == '__main__':
    main()
