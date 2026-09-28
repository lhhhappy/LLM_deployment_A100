#!/usr/bin/env python3
"""Read-only audit of drained diagnostic runs; does not produce a formal score.

Run on archived raw/server/metrics/GPU samples. GPU utilization is sampled busy
time, not compute efficiency; receive/execute/first intervals are lifecycle wall
time, not isolated GPU time. Logs have one-second resolution. No engine calls.
"""
import argparse
import ast
import bisect
import collections
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import sys


def quantile(xs, q):
    a = sorted(x for x in xs if x is not None and math.isfinite(x))
    if not a:
        return None
    # Match s1-dev/harness and scripts/score_formal.py (no interpolation).
    return a[min(len(a) - 1, math.floor(q * len(a)))]


def stats(xs):
    a = [x for x in xs if x is not None and math.isfinite(x)]
    return dict(n=len(a), mean=sum(a)/len(a) if a else None,
                p50=quantile(a, .5), p95=quantile(a, .95),
                max=max(a) if a else None, min=min(a) if a else None)


def epoch(s):
    return dt.datetime.strptime(s, '%Y-%m-%d %H:%M:%S').replace(tzinfo=dt.timezone.utc).timestamp()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric_at(samples, times, at):
    return samples[max(0, bisect.bisect_right(times, at)-1)]


def audit(level, h, out):
    summary = json.loads((level/'summary.json').read_text())
    raw = level/Path(summary['raw']).name
    runpath = level/Path(summary['run']).name
    run = json.loads(runpath.read_text())
    rows = [json.loads(s) for s in raw.open() if s.strip()]
    assert len(rows) == len({r['req_id'] for r in rows}), 'duplicate raw IDs'
    assert len(rows) == run['dispatched'] == run['n_attempted'], 'not drained'
    assert not any(r.get('error') or r.get('error_class') for r in rows)
    for r in rows:
        ts = [r[k] for k in ['t_recv_s','t_admit_s','t_exec_start_s','t_first_token_s']]
        assert all(math.isfinite(x) for x in ts) and ts == sorted(ts), r['req_id']
        assert abs(ts[-1]-ts[0]-r['ttft_s']) < 1e-5
        assert r['prompt_tokens'] == r['glm_tokens'] and r['output_tokens'] == r['max_output_i']
    start = min(r['t_recv_s'] for r in rows)
    end = max(r['client_finish_at_s'] for r in rows)
    logpath = level/'server.log'
    args, mechanism, pools = {}, [], []
    pre, dec, failures, outside_failures = [], [], [], []
    log_time = None
    pre_re = re.compile(r'^\[(.*?) TP0\] Prefill batch[,.] #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+).*?#running-req: (\d+), #queue-req: (\d+).*?cuda graph: (\w+), input throughput \(token/s\): ([\d.]+)')
    dec_re = re.compile(r'^\[(.*?) TP0\] Decode batch, #running-req: (\d+).*?cuda graph: (\w+), gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)')
    for l in logpath.read_text(errors='replace').splitlines():
        stamp=re.match(r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(?: TP\d+)?\]',l)
        if stamp: log_time=epoch(stamp[1])
        if 'server_args=' in l:
            args = ast.literal_eval(l.split('server_args=',1)[1])
        if 'TP0]' in l and ('mechanisms:' in l or 'max_total_num_tokens=' in l or 'Mamba Cache is allocated.' in l):
            (mechanism if 'mechanisms:' in l else pools).append(l)
        m = pre_re.match(l)
        if m:
            s,n,new,cached,running,queue,graph,through = m.groups()
            t = epoch(s)
            if math.floor(start) <= t <= math.ceil(end):
                pre.append(dict(t=t, new=int(new), cached=int(cached), seqs=int(n), running=int(running), queue=int(queue), graph=graph=='True', throughput=float(through)))
        m = dec_re.match(l)
        if m:
            s,n,graph,through,queue = m.groups()
            t = epoch(s)
            if math.floor(start) <= t <= math.ceil(end):
                dec.append(dict(t=t, n=int(n), graph=graph=='True', throughput=float(through), queue=int(queue)))
        if 'server_args=' not in l and any(w in l for w in ['CUDA out of memory','Traceback (most recent','retracted_reqs=','JIT compile']):
            target=failures if log_time is not None and math.floor(start)<=log_time<=math.ceil(end) else outside_failures
            target.append(dict(epoch=log_time,line=l[:350]))
    metrics = [json.loads(s) for s in (level/'metrics.jsonl').open() if s.strip()]
    metrics.sort(key=lambda r:r['t']); mt = [r['t'] for r in metrics]
    for r in metrics:
        r['kv_pool'] = sum(r.get(k,0) for k in ['kv_available_tokens','kv_evictable_tokens','kv_used_tokens'])
        r['mamba_pool'] = sum(r.get(k,0) for k in ['mamba_available_tokens','mamba_evictable_tokens','mamba_used_tokens'])
        r['mamba_resident'] = r.get('mamba_evictable_tokens',0)+r.get('mamba_used_tokens',0)
    gpu=[]
    for parts in csv.reader((level/'gpu_util.csv').open()):
        try:
            t=dt.datetime.strptime(parts[0].strip(), '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=dt.timezone.utc).timestamp()
            gpu.append(dict(t=t, rank=int(parts[1]), util=float(parts[2].split()[0]), mib=float(parts[3].split()[0])))
        except (ValueError,IndexError):
            continue
    chain = [r for r in rows if h.in_ttft_gate(r,'chain_start')]
    cbad = [r for r in chain if r['ttft_s'] > 30]
    def detail(r):
        i=metric_at(metrics,mt,r['t_exec_start_s'])
        return dict(req_id=r['req_id'], phase=r['phase'], edge_type=r['edge_type'], arrive_s=r['t_recv_s']-start,
                    prompt=r['prompt_tokens'], cached=r['cached_tokens'], uncached=r['prompt_tokens']-r['cached_tokens'],
                    ttft_s=r['ttft_s'], recv_admit_s=r['t_admit_s']-r['t_recv_s'],
                    admit_exec_s=r['t_exec_start_s']-r['t_admit_s'],
                    before_exec_s=r['t_exec_start_s']-r['t_recv_s'],
                    exec_first_s=r['t_first_token_s']-r['t_exec_start_s'],
                    effective_prefill_tok_s=(r['prompt_tokens']-r['cached_tokens'])/max(1e-9,r['t_first_token_s']-r['t_exec_start_s']),
                    output=r['output_tokens'], tpot_s=r['tpot_s'],
                    kv_used_at_exec=i.get('kv_used_tokens'), kv_evictable_at_exec=i.get('kv_evictable_tokens'),
                    mamba_used_at_exec=i.get('mamba_used_tokens'), mamba_evictable_at_exec=i.get('mamba_evictable_tokens'))
    def decode_overlap(r):
        a,b=r['client_first_token_at_s'],r['client_finish_at_s']
        pp=[x for x in pre if a<=x['t']<b]
        dd=[x for x in dec if a<=x['t']<b]
        return dict(**detail(r),client_first_s=a-start,client_finish_s=b-start,
                    client_decode_elapsed_s=b-a,
                    logged_prefill_blocks_during_decode=len(pp),
                    logged_prefill_tokens_during_decode=sum(x['new'] for x in pp),
                    prefill_events=[dict(at_s=x['t']-start,tokens=x['new'],running=x['running'],queue=x['queue'],graph=x['graph']) for x in pp],
                    decode_events=[dict(at_s=x['t']-start,batch=x['n'],logged_tok_s=x['throughput'],graph=x['graph']) for x in dd])
    case_ids={'biomaster:canon:lc_20260924_194:llm:0003',
              'scimaster:canon:lc_20260924_241:llm:0047',
              'biomaster:canon:lc_20260924_001:llm:0028'}
    windows={}
    for label,lo,hi in [('opening_0_120s',0,120),('steady_120_2400s',120,2400),('full_including_drain',0,end-start)]:
        a,b=start+lo,start+hi
        rr=[r for r in rows if a <= r['t_recv_s'] < b]
        pp=[x for x in pre if a <= x['t'] < b]
        dd=[x for x in dec if a <= x['t'] < b]
        mm=[x for x in metrics if a <= x['t'] < b]
        gg=[x for x in gpu if a <= x['t'] < b]
        left,right=metric_at(metrics,mt,a),metric_at(metrics,mt,b)
        keys=['load_back_bytes_total','load_back_duration_seconds_sum','hicache_backup_bytes_total','hicache_backup_duration_seconds_sum','evicted_tokens_total','generation_tokens_total','num_retracted_reqs']
        deltas={k:right.get(k,0)-left.get(k,0) for k in keys}
        tpot=[r['tpot_s'] for r in rr if r['tpot_s'] is not None]
        windows[label]=dict(requests=len(rr), chain=len([r for r in rr if h.in_ttft_gate(r,'chain_start')]),
            chain_over=sum(r['ttft_s']>30 for r in rr if h.in_ttft_gate(r,'chain_start')),
            tpot=stats(tpot),tpot_over=sum(x>.1 for x in tpot),
            actual_uncached_tokens=sum(r['prompt_tokens']-r['cached_tokens'] for r in rr),
            logged_prefill_tokens=sum(x['new'] for x in pp),logged_prefill_blocks=len(pp),
            prefill_block_tokens=stats([x['new'] for x in pp]),prefill_graph_count=sum(x['graph'] for x in pp),
            prefill_small_le1024=sum(x['new']<=1024 for x in pp),
            prefill_small_le4096=sum(x['new']<=4096 for x in pp),
            prefill_small_le4096_tokens=sum(x['new'] for x in pp if x['new']<=4096),
            prefill_large_ge14336=sum(x['new']>=14336 for x in pp),
            logged_prefill_tok_s=sum(x['new'] for x in pp)/(hi-lo),
            logged_decode_count=len(dd),decode_graph_count=sum(x['graph'] for x in dd),
            sampled_gpu_util=stats([x['util'] for x in gg]),sampled_gpu_mib=stats([x['mib'] for x in gg]),
            metrics={k:stats([x.get(k) for x in mm]) for k in ['full_token_usage','kv_used_tokens','kv_available_tokens','kv_evictable_tokens','kv_pool','mamba_used_tokens','mamba_evictable_tokens','mamba_available_tokens','mamba_resident','mamba_pool','num_queue_reqs','num_running_reqs','hicache_host_used_tokens','hicache_host_total_tokens']},
            metric_counter_deltas=deltas,
            kv_locked_fraction=stats([x.get('kv_used_tokens',0)/x['kv_pool'] for x in mm if x['kv_pool']]),
            mamba_free_zero_samples=sum(x.get('mamba_available_tokens')==0 for x in mm),
            metric_sample_count=len(mm),metric_full_usage_gt95_samples=sum(x.get('full_token_usage',0)>.95 for x in mm))
    # One-second log intervals are approximate. Do not label the differences as kernel durations.
    timeline=[]
    for lo in range(0,math.ceil(end-start),10):
        hi=min(lo+10,end-start); a,b=start+lo,start+hi
        mm=[x for x in metrics if a<=x['t']<b]; gg=[x for x in gpu if a<=x['t']<b]
        timeline.append(dict(start_s=lo,end_s=hi,prefill_tokens=sum(x['new'] for x in pre if a<=x['t']<b),
            prefill_blocks=sum(a<=x['t']<b for x in pre),decode_log_count=sum(a<=x['t']<b for x in dec),
            gpu_util_mean=stats([x['util'] for x in gg])['mean'],gpu_mib_max=stats([x['mib'] for x in gg])['max'],
            kv_usage_max=stats([x.get('full_token_usage') for x in mm])['max'],
            queue_max=stats([x.get('num_queue_reqs') for x in mm])['max'],
            running_max=stats([x.get('num_running_reqs') for x in mm])['max']))
    stem=level.parent.name
    with (out/(stem+'-timeline.csv')).open('w') as f:
        w=csv.DictWriter(f,list(timeline[0]));w.writeheader();w.writerows(timeline)
    cd=[detail(r) for r in sorted(chain,key=lambda r:r['t_recv_s'])]
    with (out/(stem+'-chain.csv')).open('w') as f:
        w=csv.DictWriter(f,list(cd[0]));w.writeheader();w.writerows(cd)
    selected=['enable_attn_tp_input_scattered','chunked_prefill_size','max_prefill_tokens','max_running_requests','max_mamba_cache_size','mem_fraction_static','speculative_algorithm','dcp_size','cuda_graph_config']
    return dict(scope='DRAINED_DIAGNOSTIC; full_cohort_complete=false; not formal score',
        path=str(level),raw_sha256=sha(raw),server_log_sha256=sha(logpath),metrics_sha256=sha(level/'metrics.jsonl'),
        rows=len(rows),dispatched=run['dispatched'],start_epoch=start,end_epoch=end,
        config=run['config'],server_args={k:args.get(k) for k in selected},mechanisms=mechanism,pool_lines=pools,
        errors_in_measurement=failures,error_markers_outside_measurement=outside_failures,windows=windows,
        selected_decode_overlap=[decode_overlap(r) for r in rows if r['req_id'] in case_ids],
        cache_pressure_samples=[dict(at_s=x['t']-start,kv_used=x.get('kv_used_tokens'),kv_pool=x['kv_pool'],
            mamba_used=x.get('mamba_used_tokens'),mamba_evictable=x.get('mamba_evictable_tokens'),
            mamba_available=x.get('mamba_available_tokens'),mamba_pool=x['mamba_pool'],queue=x.get('num_queue_reqs'))
            for x in metrics if start<=x['t']<=end and (x.get('full_token_usage',0)>.95 or x.get('mamba_available_tokens')==0)],
        chain_bad=[detail(r) for r in cbad],chain_long=[detail(r) for r in chain if r['prompt_tokens']>=100000],
        chain_bad_before_exec_gt30=sum(r['t_exec_start_s']-r['t_recv_s']>30 for r in cbad),
        tpot_worst=[detail(r) for r in sorted(rows,key=lambda r:r['tpot_s'] or 0,reverse=True)[:12]]),rows


def pair(a,b,h):
    A={r['req_id']:r for r in a};B={r['req_id']:r for r in b};ids=sorted(A.keys()&B.keys())
    fields=['prompt_tokens','max_output_i','chain_id','idx_in_chain','phase','uncached_expected','edge_type']
    assert all(A[r][k]==B[r][k] for r in ids for k in fields),'frozen fields differ'
    result=dict(common=len(ids),only_a=len(A.keys()-B.keys()),only_b=len(B.keys()-A.keys()))
    starts=[min(r['t_recv_s'] for r in T.values()) for T in [A,B]]
    for gate,limit in [('chain_start',30),('turn_start',15),('overall_intra',5),('fast_intra',3)]:
        ii=[r for r in ids if h.in_ttft_gate(A[r],gate)]
        bad_a={r for r in ii if A[r]['ttft_s']>limit};bad_b={r for r in ii if B[r]['ttft_s']>limit}
        result[gate]=dict(n=len(ii),over=[len(bad_a),len(bad_b)],p95=[quantile([t[r]['ttft_s'] for r in ii],.95) for t in [A,B]],
            fixed=sorted(bad_a-bad_b),new=sorted(bad_b-bad_a))
        if gate=='chain_start':
            def categories(keys):
                return dict(collections.Counter(
                    ('opening' if A[k]['t_recv_s']-starts[0]<120 else 'steady')+'|'+str(A[k]['phase'])+'|'+str(A[k]['edge_type']) for k in keys))
            result[gate]['fixed_categories']=categories(bad_a-bad_b)
            result[gate]['new_categories']=categories(bad_b-bad_a)
    result['tpot']=[dict(**stats([T[r]['tpot_s'] for r in ids]),over=sum((T[r]['tpot_s'] or 0)>.1 for r in ids)) for T in [A,B]]
    result['uncached_tokens']=[sum(T[r]['prompt_tokens']-T[r]['cached_tokens'] for r in ids) for T in [A,B]]
    return result


def audit_capability(path):
    """Separate transport failure and capped output from completed answers.

    Accept a pread text capture, which can have an exit_code footer after JSON.
    The returned counts are diagnostic and do not declare a quality gate pass.
    """
    rows,_=json.JSONDecoder().raw_decode(path.read_text())
    assert isinstance(rows,list) and len({(r['kind'],r['id']) for r in rows})==len(rows)
    result={'source':str(path),'sha256':sha(path),'quality_gate_pass':None,'datasets':{}}
    for kind in sorted({r['kind'] for r in rows}):
        rr=[r for r in rows if r['kind']==kind]
        completed=[r for r in rr if not r.get('error')]
        truncated=[r for r in completed if r.get('finish')=='length']
        result['datasets'][kind]=dict(attempts=len(rr),transport_success=len(completed),
            correct=sum(bool(r['ok']) for r in rr),error_count=len(rr)-len(completed),
            truncated=len(truncated),wrong_non_error=sum(not r['ok'] for r in completed),
            wrong_non_truncated=sum(not r['ok'] and r.get('finish')!='length' for r in completed),
            error_types=dict(collections.Counter(r['error'] for r in rr if r.get('error'))),
            truncated_metadata=[{k:r.get(k) for k in ['id','finish','completion_tokens','secs']} for r in truncated])
    result['scope']='INCOMPLETE_CAPABILITY_CHECK' if any(v['error_count'] or v['truncated'] for v in result['datasets'].values()) else 'COMPLETE_TRANSPORT; quality requires matched reference'
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cap-results',type=Path,help='Optional capability JSON or pread capture; diagnostics only')
    p.add_argument('--runs',nargs='+',default=['130eznb','130eznc','130ezne','130eznf','130ezn3','130ezn5'])
    args=p.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(args.repo/'s1-dev/harness'))
    import s1_common as h
    result={'runs':{},'pairs':{},'scope':'execution/memory diagnostic; no GPU/engine mutations',
            'percentile_method':'sorted[min(n-1, floor(q*n))], matching harness'}; raw={}
    for run in args.runs:
        ds=list((args.repo/'evidence').glob('L'+run+'-*'));assert len(ds)==1,(run,ds)
        levels=list(ds[0].glob('N[0-9]*'));assert len(levels)==1
        result['runs'][run],raw[run]=audit(levels[0],h,args.out)
    for a,b in [('130eznb','130eznc'),('130ezne','130eznf'),('130ezn3','130ezn5')]:
        if a in raw and b in raw:result['pairs'][a+'_'+b]=pair(raw[a],raw[b],h)
    if args.cap_results:
        result['capability']=audit_capability(args.cap_results)
        (args.out/'cap-validity-audit.json').write_text(json.dumps(result['capability'],ensure_ascii=False,indent=2)+'\n')
    (args.out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    for n,r in result['runs'].items():
        w=r['windows']['full_including_drain'];s=r['windows']['steady_120_2400s'];m=w['metrics']
        print(json.dumps(dict(run=n,rows=r['rows'],chain_over=w['chain_over'],tpot_p95=w['tpot']['p95'],tpot_over=w['tpot_over'],
            steady_tpot_p95=s['tpot']['p95'],kv_used_peak=m['kv_used_tokens']['max'],kv_pool=m['kv_pool']['p50'],
            mamba_used_peak=m['mamba_used_tokens']['max'],mamba_resident_peak=m['mamba_resident']['max'],
            gpu_util=w['sampled_gpu_util']['mean'],gpu_mib_peak=w['sampled_gpu_mib']['max'],
            prefill_blocks=w['logged_prefill_blocks'],small_blocks=w['prefill_small_le4096'],
            restore_GiB=w['metric_counter_deltas']['load_back_bytes_total']/2**30),ensure_ascii=False))
    for name, pair_result in result['pairs'].items():
        brief={k:v for k,v in pair_result.items() if k not in ['chain_start','turn_start','overall_intra','fast_intra']}
        brief['chain']={k:(len(v) if k in ['fixed','new'] else v) for k,v in pair_result['chain_start'].items()}
        print('PAIR',name,json.dumps(brief,ensure_ascii=False))


if __name__=='__main__':
    main()
