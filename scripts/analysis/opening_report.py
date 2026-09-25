#!/usr/bin/env python3
"""Automatic, drained opening-window diagnosis. No engine requests or GPU work.

Request counts and logged token work are observations. Decode time is a local
estimate from nearby decode-only windows, with unsupported windows left unknown.
Batch log throughput measures wall intervals, not isolated GPU kernel time.
"""
import base64
import bisect
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'s1-dev/harness'))
from s1_common import in_ttft_gate
from s1_score import q as harness_quantile
from window_gates import stats as gate_stats, score_formal


def collect_sources(job, n):
    import base64, gzip, hashlib, json
    from pathlib import Path
    root = Path('/tmp/ax/runs')/job
    level = root/('N'+str(n))
    files = {}
    for name in ['timed_window.json', 'timed_verdict.json', 'dispatch_ledger.jsonl',
                 'flush_evidence.json', 'short_warmup_receipt.json', 'metrics.jsonl']:
        files[name] = base64.b64encode((level/name).read_bytes()).decode()
    with (root/'server.log').open('rb') as f:
        size = (root/'server.log').stat().st_size
        if size > 50*1024*1024:
            raise ValueError('opening server log exceeds bounded collector budget')
        files['server.log'] = base64.b64encode(f.read(size)).decode()
    blob = gzip.compress(json.dumps(files).encode(), mtime=0)
    sha = hashlib.sha256(blob).hexdigest()
    path = Path('/tmp/ax/codex')/('opening_'+job+'_'+sha+'.gz')
    if not path.exists():
        path.write_bytes(blob)
    return dict(archive=str(path), size=len(blob), sha256=sha)


def save(path, obj):
    tmp = path.with_suffix(path.suffix+'.part')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def quantile(values, q):
    if not values:
        return None
    return harness_quantile(values, q)


def overlap(intervals, a, b):
    return sum(max(0, min(b,y)-max(a,x)) for x,y in intervals)


def merged(intervals):
    result = []
    for a,b in sorted(intervals):
        if b <= a: continue
        if result and a <= result[-1][1]: result[-1][1] = max(result[-1][1], b)
        else: result.append([a,b])
    return result


def batch_events(path):
    ts = re.compile(r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) TP0\] (Prefill batch|Decode batch), (.*)$')
    kv = re.compile(r'(#?[a-zA-Z][a-zA-Z \-()/]*?): ([-\d.]+|True|False)')
    events = []
    for line, text in enumerate(path.open(errors='replace')):
        m = ts.match(text.strip())
        if not m: continue
        values = {k:float(v) for k,v in kv.findall(m[3]) if v not in ['True','False']}
        kind = 'P' if m[2].startswith('Prefill') else 'D'
        tokens = values['#new-token' if kind=='P' else 'spec tokens']
        rate = values['input throughput (token/s)' if kind=='P' else 'gen throughput (token/s)']
        events.append(dict(line=line, kind=kind, stamp=datetime.strptime(m[1],'%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp(),
                           gap=tokens/rate if rate>0 else None, values=values))
    calibration = {}
    for kind in 'PD':
        seq = [e for e in events if e['kind']==kind]
        if not seq: continue
        cumul = 0
        for i,e in enumerate(seq):
            if i: cumul += e['gap'] if e['gap'] is not None else e['stamp']-seq[i-1]['stamp']
            e['cum'] = cumul
        low = max(e['stamp']-e['cum'] for e in seq)
        high = min(e['stamp']+1-e['cum'] for e in seq)
        reconstructed = high-low >= -0.05
        for e in seq: e['t'] = (low+high)/2+e['cum'] if reconstructed else e['stamp']+0.5
        calibration[kind] = dict(n=len(seq), feasible_offset_width_s=high-low,
                                timing='rounded throughput reconstruction' if reconstructed else 'second-resolution log midpoint')
    return events, calibration


def reconstruct_backlog(rows, events, receipt, output):
    """Retrospective work balance; logged planned tokens are page-rounded.

    Correct only the known final-page padding at first-token completion. Extra
    chunk padding/reprocessing and log boundary uncertainty remain residuals.
    Never clamp negative balances or silently force the drained endpoint to 0.
    """
    start, end = receipt['first_dispatch_at_s'], receipt['drained_at_s']
    deltas = [(start, 0, 0, 0, 0, 0)]
    for r in rows:
        work = r['prompt_tokens'] - r['cached_tokens']
        assert work >= 0
        rounded = (work+63)//64*64
        deltas.append((r['client_dispatch_at_s'], work, rounded, 0, 0, 0))
        first = r.get('t_first_token_s') or r.get('client_first_token_at_s')
        if first is None:
            raise ValueError('backlog completion ledger requires first-token timestamp')
        deltas.append((first, 0, 0, 0, rounded-work, work))
    selected = [e for e in events if e['kind']=='P' and start<=e['t']<=end]
    for e in selected:
        deltas.append((e['t'], 0, 0, int(e['values']['#new-token']), 0, 0))
    deltas.append((end, 0, 0, 0, 0, 0))
    a = ap = service = padding = completed = 0
    points = []
    for t, da, dap, ds, dp, dc in sorted(deltas):
        a += da; ap += dap; service += ds; padding += dp; completed += dc
        points.append(dict(time_s=t-start, arrived_actual_tokens=a,
            arrived_page_rounded_tokens=ap, logged_page_rounded_service_tokens=service,
            completed_request_tail_padding_tokens=padding,
            estimated_actual_service_tokens=service-padding,
            estimated_backlog_tokens=a-service+padding,
            page_rounded_balance_tokens=ap-service,
            request_work_not_yet_at_first_token=a-completed))
    with (output/'backlog.csv').open('w') as f:
        writer=csv.DictWriter(f, fieldnames=list(points[0]))
        writer.writeheader(); writer.writerows(points)
    peak=max(points, key=lambda p:p['estimated_backlog_tokens'])
    first_clear=next((p['time_s'] for p in points if p['time_s']>peak['time_s']
                     and p['estimated_backlog_tokens']<=0), None)
    residual=points[-1]['estimated_backlog_tokens']
    # A bounded standalone SVG preserves the signed curve, including residuals.
    width, height, left, top = 1000, 360, 85, 40
    low=min(0,min(p['estimated_backlog_tokens'] for p in points))
    high=max(1,peak['estimated_backlog_tokens'])
    duration=max(1,end-start)
    def xy(p):
        return (left+850*p['time_s']/duration, top+250*(high-p['estimated_backlog_tokens'])/(high-low))
    path=' '.join(f'{x:.2f},{y:.2f}' for x,y in map(xy,points))
    zero=top+250*high/(high-low)
    svg=(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
         '<rect width="100%" height="100%" fill="white"/>'
         '<g font-family="sans-serif" font-size="14" fill="#222">'
         '<text x="85" y="23">Retrospective B(t): actual arrivals minus logged service, with tail-padding correction</text>'
         f'<line x1="{left}" y1="{zero}" x2="935" y2="{zero}" stroke="#999"/>'
         f'<polyline points="{path}" fill="none" stroke="#2166ac" stroke-width="1.4"/>'
         f'<text x="5" y="50">{high/1000:.1f}k tok</text><text x="10" y="{zero:.1f}">0</text>'
         f'<text x="85" y="315">0 s</text><text x="860" y="315">{duration:.1f} s</text>'
         f'<text x="85" y="344">Residual {residual:+d} tokens; batch-log timing, not exact CUDA completion. No forced zero.</text></g></svg>')
    (output/'backlog.svg').write_text(svg)
    return dict(method='offline actual request arrivals minus page-rounded batch work, plus known final-page padding at first token',
        page_size=64, arrived_actual_tokens=a, arrived_page_rounded_tokens=ap,
        logged_page_rounded_service_tokens=service, known_tail_padding_tokens=padding,
        terminal_residual_tokens=residual, terminal_request_work_not_at_first_token=a-completed,
        peak_estimated_tokens=peak['estimated_backlog_tokens'], peak_time_s=peak['time_s'],
        minimum_signed_tokens=min(p['estimated_backlog_tokens'] for p in points),
        first_nonpositive_after_peak_s=first_clear,
        exact_clear_time_s=None,
        limitations='Final cached_tokens known only after replay; batch logs have coarse timing and page-rounded planned work. '
                    'Residual can include extra chunk padding/reprocessing or boundary error. '
                    'First nonpositive balance is not certified queue empty; later arrivals can grow it again.',
        csv='backlog.csv', plot='backlog.svg')


def analyze(raw, source, output):
    receipt = json.loads((source/'timed_window.json').read_text())
    verdict = json.loads((source/'timed_verdict.json').read_text())
    rows = [json.loads(l) for l in raw.read_text().splitlines() if l]
    ledger_data = (source/'dispatch_ledger.jsonl').read_bytes()
    ledger = [json.loads(l) for l in ledger_data.splitlines()]
    sent = [e['req_id'] for e in ledger if e['event']=='dispatch']
    ids = [r['req_id'] for r in rows]
    assert receipt['status']==verdict['status']=='DRAINED'
    assert len(ids)==len(set(ids))==len(sent)==receipt['n_dispatched']
    assert set(ids)==set(sent)
    assert hashlib.sha256(raw.read_bytes()).hexdigest()==verdict['raw_sha256']
    assert hashlib.sha256(ledger_data).hexdigest()==receipt['ledger_sha256']
    successful = [r for r in rows if not r.get('error')]
    assert all(r['prompt_tokens']==r['glm_tokens'] and r['output_tokens']==r['max_output_i'] for r in successful)
    start, end = receipt['first_dispatch_at_s'], receipt['admission_deadline_s']
    chain = [r for r in rows if in_ttft_gate(r,'chain_start')]
    cohort_stats = []
    for cutoff in [30,60,300,600]:
        subset = [r for r in chain if r['client_dispatch_at_s']<start+cutoff]
        valid = [r for r in subset if not r.get('error')]
        cohort_stats.append(dict(arrived_before_s=cutoff, requests=len(subset),
            over30=sum(r['ttft_s']>30 for r in valid), errors=len(subset)-len(valid),
            uncached_tokens=sum(r['prompt_tokens']-r['cached_tokens'] for r in valid),
            ttft_p95_s=quantile([r['ttft_s'] for r in valid],.95)))
    details=[]
    for r in chain:
        details.append(dict(req_id=r['req_id'], arrival_s=r['client_dispatch_at_s']-start,
            ttft_s=r.get('ttft_s'), queue_s=r.get('queue_time_s'),
            receive_to_first_batch_s=(r['t_exec_start_s']-r['t_recv_s']) if r.get('t_exec_start_s') and r.get('t_recv_s') else None,
            first_batch_to_first_token_s=(r['t_first_token_s']-r['t_exec_start_s']) if r.get('t_first_token_s') and r.get('t_exec_start_s') else None,
            prompt_tokens=r.get('prompt_tokens'), cached_tokens=r.get('cached_tokens'), error=r.get('error')))
    with (output/'chain.csv').open('w') as f:
        if details:
            w=csv.DictWriter(f,fieldnames=list(details[0]));w.writeheader();w.writerows(details)
    events, calibration = batch_events(source/'server.log')
    prefill = [e for e in events if e['kind']=='P' and start<=e['t']<end]
    # Queue occupancy below is observed waiting-to-first-batch wall time. It is
    # not eligibility, avoidable delay, nor a per-request resource diagnosis.
    waiting = merged([(r['t_recv_s'],r['t_exec_start_s']) for r in successful if r.get('t_recv_s') and r.get('t_exec_start_s')])
    decoding = merged([(r['t_first_token_s'],r['t_first_token_s']+(r['output_tokens']-1)*(r.get('tpot_s') or 0))
                       for r in successful if r.get('t_first_token_s')])
    dec = [e for e in events if e['kind']=='D']
    windows=[]
    for a,b in zip(dec,dec[1:]):
        if a['t']<start or b['t']>end or not b['gap']: continue
        if overlap(decoding,a['t'],b['t']) < .98*(b['t']-a['t']): continue
        ins = [p for p in prefill if a['line']<p['line']<b['line']]
        windows.append(dict(a=a['t'],b=b['t'],seconds=b['gap'],batch_size=b['values']['spec rounds']/40,
                            prefill_batches=len(ins),prefill_tokens=sum(p['values']['#new-token'] for p in ins)))
    pure = [w for w in windows if w['prefill_batches']==0 and 0<w['seconds']<12]
    reference_path=ROOT/'evidence/opening-0925a-n22-n26-20260925/decode-reference.json'
    reference=json.loads(reference_path.read_text()) if reference_path.exists() else None
    estimated=[]
    for w in windows:
        if not w['prefill_batches']: continue
        near=sorted([p for p in pure if abs(p['batch_size']-w['batch_size'])<=3],key=lambda p:abs(p['batch_size']-w['batch_size']))[:9]
        method='same-run nearby batch sizes'
        if len(near)>=3:
            prediction=statistics.median(p['seconds'] for p in near)
        elif reference:
            # Report prior-fit transfer explicitly. Never relabel this as a
            # measured decode fraction or as an isolated GPU forward duration.
            model=reference['model']; mid=(w['a']+w['b'])/2
            ctx=0
            for r in successful:
                first=r.get('t_first_token_s'); span=(r['output_tokens']-1)*(r.get('tpot_s') or 0)
                if first and first<=mid<=first+span and span>0:
                    ctx+=r['prompt_tokens']+r['output_tokens']*(mid-first)/span
            prediction=40*(model['a_s']+model['b_s_per_req']*w['batch_size']+model['c_s_per_Mtok_ctx']*ctx/1e6)
            method='069 same-engine empirical fit transferred to this window'
        else: continue
        if not 0<prediction<w['seconds']: continue
        estimated.append(dict(**w,decode_estimate_s=prediction,method=method,reference_windows=len(near),
                              reference_p10_s=quantile([p['seconds'] for p in near],.1),reference_p90_s=quantile([p['seconds'] for p in near],.9)))
    minute_stats=[]
    for m in range(int(receipt['duration_s']/60)):
        a,b=start+60*m,start+60*(m+1)
        ps=[p for p in prefill if a<=p['t']<b]
        wait_s=overlap(waiting,a,b)
        busy=[p for p in ps if any(x<=p['t']<=y for x,y in waiting)]
        minute_stats.append(dict(minute=m,prefill_new_tokens=int(sum(p['values']['#new-token'] for p in ps)),
            prefill_batches=len(ps),prefill_tokens_per_wall_second=sum(p['values']['#new-token'] for p in ps)/60,
            queue_nonempty_union_s=wait_s,queue_period_prefill_tokens_per_s=sum(p['values']['#new-token'] for p in busy)/wait_s if wait_s>0 else None,
            kv_usage_max=max((p['values']['full token usage'] for p in ps),default=None)))
    # Same engine's no-decoder consecutive prefill intervals are the closest
    # available isolated reference; require a continuing owner and no idle gap.
    isolated=[]
    all_p=[e for e in events if e['kind']=='P']
    for a,b in zip(all_p,all_p[1:]):
        if not start<=a['t']<b['t']<end or not b['gap'] or b['gap']>=5: continue
        if a['values']['#pending-token']<=0 or a['values']['#running-req']!=0 or b['values']['#running-req']!=0: continue
        if any(a['line']<d['line']<b['line'] for d in dec): continue
        isolated.append(dict(tokens=b['values']['#new-token'],seconds=b['gap']))
    covered=sum(w['seconds'] for w in estimated)
    measured=gate_stats(rows, score_formal.load_harness())
    tp=measured['tpot']
    tp['max']=max((r['tpot_s'] for r in successful if (r.get('output_tokens') or 0)>1 and r.get('tpot_s') is not None), default=None)
    tp['over_rate']=tp['over_0.10']/tp['n'] if tp['n'] else None
    backlog=reconstruct_backlog(successful, events, receipt, output)
    result=dict(scope='drained local opening diagnostic; not full-cohort or official verdict', n=verdict['n'],
        n_completed=len(rows),errors=len(rows)-len(successful), admission_seconds=receipt['duration_s'],
        drain_seconds=receipt['drained_at_s']-end,chain_cohorts=cohort_stats,
        tpot_mean_s=tp['mean'], tpot_p95_s=tp['p95'], tpot=tp, ttft_gates=measured['gates'],
        percentile_method='original harness sorted[floor(q*n)]', backlog=backlog,
        tpot_reference_only=dict(assumed_requests=1788, approximate_over_0_10_reference=89,
            note='Not this window allowance; official denominator and selected cohort unconfirmed.'),
        minutes=minute_stats,log_timing_calibration=calibration,
        pure_decode_wall_step_reference=dict(n_windows=len(pure),
            p50_s=quantile([w['seconds']/40 for w in pure],.5),
            p95_s=quantile([w['seconds']/40 for w in pure],.95),
            windows=pure, note='40-step log windows without prefill, continuously decoding; differing batch sizes/contexts, not isolated GPU kernel time.'),
        isolated_prefill_reference=dict(n_intervals=len(isolated),new_tokens=sum(w['tokens'] for w in isolated),
            seconds=sum(w['seconds'] for w in isolated),tokens_per_s=sum(w['tokens'] for w in isolated)/sum(w['seconds'] for w in isolated) if isolated else None),
        decode_time_estimate=dict(method='nearby same-run decode-only windows; if unavailable, explicit 069 same-engine fitted reference; includes CPU overhead',
            prior_reference=reference,
            pure_reference_windows=len(pure),mixed_windows=len([w for w in windows if w['prefill_batches']]),supported_mixed_windows=len(estimated),
            supported_wall_s=covered,decode_share=sum(w['decode_estimate_s'] for w in estimated)/covered if covered else None,
            limitations='Estimate only; unsupported windows unknown. Not GPU busy time. Changes in context and graph shape can bias this estimate.',windows=estimated),
        evidence_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [raw,*source.iterdir()] if p.is_file()})
    save(output/'analysis.json',result)
    early=cohort_stats[1];whole=cohort_stats[-1]
    estimate=result['decode_time_estimate']['decode_share']
    gate_brief='，'.join(f"{g} {v['over']}/{v['n']}" for g,v in measured['gates'].items())
    brief=(f"N{verdict['n']}开场短测已排空：{len(rows)}条、错误{result['errors']}；前60秒到达chain {early['over30']}/{early['requests']}超过30秒，"
           f"前10分钟到达chain {whole['over30']}/{whole['requests']}超时；各桶超时/样本 {gate_brief}；"
           f"TPOT均值/p95/最大 {tp['mean']*1000:.2f}/{tp['p95']*1000:.2f}/{tp['max']*1000:.2f}ms，>0.10秒 {tp['over_0.10']}/{tp['n']}（{tp['over_rate']:.2%}）。"
           f"离线B峰值约{backlog['peak_estimated_tokens']/1000:.1f}k，排空残差{backlog['terminal_residual_tokens']:+d} tokens。"
           +(f"有参考覆盖的混合时段decode墙钟占比估计{estimate:.1%}（覆盖{covered:.0f}秒，非GPU实测）。" if estimate is not None else '现有日志不足以估计decode占时。')
           +'本地短测，不判完整档或线上失败原因。')
    (output/'brief.txt').write_text(brief+'\n')
    return dict(brief=brief,path=str(output),result=result)


def compare_report(report, out, target, reference_job):
    baseline=out.parent.parent/('L'+reference_job)
    previous=json.loads((baseline/'opening/analysis.json').read_text())
    a={r['req_id']:r for r in map(json.loads,(baseline/'window/raw.jsonl').read_text().splitlines())}
    b={r['req_id']:r for r in map(json.loads,(out/'raw.jsonl').read_text().splitlines())}
    common=sorted(a.keys() & b.keys())
    for rid in common:
        assert all(a[rid].get(k)==b[rid].get(k) for k in ['glm_tokens','max_output_i','phase','idx_in_chain','replay_gap_ms'])
    scorer=score_formal.load_harness()
    left=gate_stats([a[r] for r in common],scorer)
    right=gate_stats([b[r] for r in common],scorer)
    changes={}
    for _,selector,limit in scorer.TTFT_GATE_SPECS:
        good=[r for r in common if in_ttft_gate(a[r],selector) and not a[r].get('error') and not b[r].get('error')]
        changes[selector]=dict(n=len(good),reference_over=sum(a[r]['ttft_s']>limit for r in good),
            candidate_over=sum(b[r]['ttft_s']>limit for r in good),
            fixed=sum(a[r]['ttft_s']>limit>=b[r]['ttft_s'] for r in good),
            new=sum(a[r]['ttft_s']<=limit<b[r]['ttft_s'] for r in good))
    chain=changes['chain_start']
    pair=dict(reference_job=reference_job,reference_n=previous['n'],candidate_n=report['result']['n'],
        common_completed=len(common),common_chain=left['gates']['chain']['n'],common_chain_without_errors=chain['n'],
        reference_chain_over30=chain['reference_over'],candidate_chain_over30=chain['candidate_over'],
        chain_fixed=chain['fixed'],chain_new=chain['new'],gates=changes,reference=left,candidate=right,
        reference_only=len(a.keys()-b.keys()),candidate_only=len(b.keys()-a.keys()),
        scope='Both ten-minute admission cohorts drained. Closed-loop arrivals and reached IDs can change; paired requests alone do not isolate service order.')
    detail=[]
    for rid in common:
        item=dict(req_id=rid)
        for field in ['client_dispatch_at_s','ttft_s','tpot_s','prompt_tokens','cached_tokens','error']:
            item['reference_'+field]=a[rid].get(field);item['candidate_'+field]=b[rid].get(field)
        detail.append(item)
    if detail:
        with (target/'paired.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(detail[0]));w.writeheader();w.writerows(detail)
    save(target/'comparison.json',pair)
    gate_text='，'.join(f"{g} {left['gates'][g]['over']}→{right['gates'][g]['over']}" for g in left['gates'])
    report['brief']+=(f" 对074同ID {len(common)}条：{gate_text}；chain修复{chain['fixed']}/新增{chain['new']}；"
                      f"TPOT>0.10 {left['tpot']['over_0.10']}/{left['tpot']['n']}→{right['tpot']['over_0.10']}/{right['tpot']['n']}。"
                      if reference_job=='074-official_0925a_opening_n26' else
                      f" 对N{previous['n']}同ID {len(common)}条：{gate_text}；chain修复{chain['fixed']}/新增{chain['new']}。")
    (target/'brief.txt').write_text(report['brief']+'\n')
    return pair


def build_report(out, job, meta, call, reference_job=None):
    from window_watch import marked, READ_CODE, CHUNK
    target=out.parent/'opening';target.mkdir(exist_ok=True)
    source=target/'source';source.mkdir(exist_ok=True)
    code=inspect.getsource(collect_sources)+'\nimport json,sys\nprint("OPENING_META "+json.dumps(collect_sources(sys.argv[1],int(sys.argv[2]))))'
    info=json.loads(marked(call(code,job,meta['n']),'OPENING_META '))
    parts=[]
    for offset in range(0,info['size'],CHUNK):
        data=base64.b64decode(marked(call(READ_CODE,info['archive'],offset,min(CHUNK,info['size']-offset)),'WINDOW_DATA '),validate=True)
        parts.append(data)
    blob=b''.join(parts)
    assert len(blob)==info['size'] and hashlib.sha256(blob).hexdigest()==info['sha256']
    (target/'sources.json.gz').write_bytes(blob)
    for name,value in json.loads(gzip.decompress(blob)).items():
        assert Path(name).name==name
        (source/name).write_bytes(base64.b64decode(value,validate=True))
    save(target/'transport.json',info)
    report=analyze(out/'raw.jsonl',source,target)
    if reference_job:
        compare_report(report, out, target, reference_job)

    return report


if __name__=='__main__':
    result=analyze(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]))
    if len(sys.argv)>4:
        compare_report(result,Path(sys.argv[1]).parent,Path(sys.argv[3]),sys.argv[4])
    print(result['brief'])
