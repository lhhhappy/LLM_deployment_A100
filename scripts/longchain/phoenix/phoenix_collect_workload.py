#!/usr/bin/env python3
"""Collect a frozen DB-selected session cohort; resume with audited local files.

Uses production Phoenix only for observation. Never invokes inference or tools.
Raw content stays in ignored cache/ with mode 0600. A window is not a lifetime
session, and a complete API query is not proof of complete instrumentation.
"""
from __future__ import annotations
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import threading
import time

from phoenix_longchain_observe import audit, iso, parsed


def dump(path, value, private=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    if path.name.endswith('.gz'):
        with gzip.open(tmp, 'wt', encoding='utf-8', compresslevel=3) as f:
            json.dump(value, f, ensure_ascii=False, separators=(',', ':'))
    else:
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    if private:
        tmp.chmod(0o600)
    tmp.replace(path)


def enriched(rows):
    summary, calls, edges = audit(rows)
    main = [c for c in calls if c['call_type'] == 'mainagent' and c['agent'] == 'scimaster']
    lengths, byrun = [], collections.defaultdict(list)
    for c in main:
        if c['run_id']:
            byrun[c['run_id']].append(c)
    fragments = []
    for run, cs in byrun.items():
        group = []
        for c in cs:
            consecutive = group and isinstance(c['call_index'], int) and isinstance(group[-1]['call_index'], int) and c['call_index'] == group[-1]['call_index']+1
            if group and not consecutive:
                fragments.append(group)
                group = []
            group.append(c)
        if group:
            fragments.append(group)
    spans = []
    for cs in fragments:
        lengths.append(len(cs))
        spans.append({'run_id':cs[0]['run_id'], 'first_span':cs[0]['span_id'],
                      'last_span':cs[-1]['span_id'], 'n_calls':len(cs),
                      'first_call_index':cs[0]['call_index'], 'last_call_index':cs[-1]['call_index'],
                      'start':cs[0]['start'], 'end':cs[-1]['end'],
                      'user_message_ids':sorted({c['message_id'] for c in cs if c['message_id']}),
                      'complete_run_proven':False})
    caps = missing_usage = 0
    for r in rows:
        gen = r.get('attributes.gen_ai') or {}
        msgs = parsed((gen.get('input') or {}).get('messages'))
        for m in msgs if isinstance(msgs, list) else []:
            for part in m.get('parts') or []:
                caps += any(isinstance(v,str) and len(v)==8000 for v in part.values())
    compression = [e for e in edges if e['index_consecutive'] and (e['compression_count_delta'] or 0)>0]
    longest = sorted((e for e in edges if e['index_consecutive'] and (e['gap_s'] or 0)>=0), key=lambda e:e['gap_s'], reverse=True)[:5]
    summary.update({'max_consecutive_main_calls':max(lengths,default=0),
                    'consecutive_fragments':len(fragments), 'input_parts_at_8000_char_cap':caps,
                    'observed_compression_edges':len(compression),
                    'direct_exact_replay_ready':False,
                    'direct_exact_replay_reason':'Complete bodies/tools and GLM rendering are not verified; exported messages are bounded. Behavioral fragments remain usable for synthesis.'})
    return {'summary':summary, 'calls':calls, 'edges':edges,
            'continuous_fragments':spans, 'compression_events':compression, 'longest_wait_edges':longest}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frame',type=Path,required=True)
    p.add_argument('--raw-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--limit',type=int,default=500)
    p.add_argument('--request-interval',type=float,default=.3)
    p.add_argument('--take',type=int)
    a=p.parse_args()
    if not 1<=a.workers<=8 or a.limit<2:p.error('workers 1..8; limit >=2')
    frame=json.loads(a.frame.read_text()); records=frame['records']
    if a.take:records=records[:a.take]
    start,end=iso(frame['window_start']),iso(frame['window_end'])
    for k in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
        os.environ.pop(k,None)
    from phoenix.client import Client
    from phoenix.client.types.spans import SpanQuery
    local=threading.local()
    pace_lock=threading.Lock()
    next_request=[0.0]
    def client():
        if not hasattr(local,'client'):
            local.client=Client(base_url='https://scimaster-phoenix.bohrium.com',api_key=os.environ.get('PHOENIX_API_KEY') or None)
        return local.client
    def retry(fn):
        for attempt in range(4):
            try:
                with pace_lock:
                    delay=max(0,next_request[0]-time.monotonic())
                    next_request[0]=max(time.monotonic(),next_request[0])+a.request_interval
                if delay:time.sleep(delay)
                return fn()
            except Exception:
                if attempt==3:raise
                time.sleep(5*(attempt+1))
    def collect(record):
        sid=record['session_id']
        target=a.out/'sessions'/f'{sid}.json'
        rawpath=a.raw_dir/f'{sid}.json.gz'
        if target.exists() and rawpath.exists():
            old=json.loads(target.read_text())
            if old['window_start']==frame['window_start'] and old['window_end']==frame['window_end'] and old['raw_sha256']==hashlib.sha256(rawpath.read_bytes()).hexdigest():
                return {**old['summary'],'session_id':sid,'collection_status':'cached'}
        meta_path=a.raw_dir/'session-index'/f'{sid}.json'
        if meta_path.exists():meta=json.loads(meta_path.read_text())
        else:
            meta=retry(lambda:client().sessions.get(session_id=sid,timeout=45))
            dump(meta_path,meta,True)
        traces=meta.get('traces') or []
        tids=sorted({t['trace_id'] for t in traces if t.get('trace_id')
                     and (not t.get('start_time') or iso(t['start_time'])<end)
                     and (not t.get('end_time') or iso(t['end_time'])>=start)})
        if not tids:raise RuntimeError('Phoenix session contains no trace IDs')
        pages=[]
        def fetch(ids,lo,hi):
            condition=' or '.join(f"trace_id == '{tid}'" for tid in ids)
            query=SpanQuery().where(f"span_kind == 'LLM' and ({condition})")
            df=retry(lambda:client().spans.get_spans_dataframe(query=query,project_identifier='default',start_time=lo,end_time=hi,limit=a.limit,timeout=60))
            n=0 if df is None else len(df)
            if n>=a.limit:
                if len(ids)>1:
                    mid=len(ids)//2
                    return fetch(ids[:mid],lo,hi)+fetch(ids[mid:],lo,hi)
                if hi-lo<=timedelta(milliseconds=1):raise RuntimeError('Saturated minimal time window; refuse truncation')
                mid=lo+(hi-lo)/2
                return fetch(ids,lo,mid)+fetch(ids,mid,hi)
            pages.append({'trace_ids':len(ids),'start':lo.isoformat(),'end':hi.isoformat(),'rows':n})
            if not n:return []
            if 'context.span_id' not in df.columns:df=df.reset_index()
            df=df.loc[:,~df.columns.duplicated()]
            result=json.loads(df.to_json(orient='records',date_format='iso'))
            if any(r.get('attributes.session.id')!=sid or r.get('context.trace_id') not in ids for r in result):
                raise RuntimeError('Server query filter mismatch; stop instead of accepting unrelated spans')
            if any(not(lo<=iso(r['start_time'])<=hi) for r in result):
                raise RuntimeError('Server time filter mismatch')
            return result
        rows=[]
        for offset in range(0,len(tids),60):rows.extend(fetch(tids[offset:offset+60],start,end))
        unique={r['context.span_id']:r for r in rows}
        if not unique:raise RuntimeError('No LLM spans in selected window; not treated as zero behavior')
        vals=list(unique.values())
        dump(rawpath,vals,True)
        doc=enriched(vals)
        doc.update({'sampling':record,'window_start':frame['window_start'],'window_end':frame['window_end'],
                    'fetched_at':datetime.now(timezone.utc).isoformat(),'session_trace_ids_returned':len(tids),
                    'pages':pages,'duplicates_removed':len(rows)-len(unique),
                    'raw_path':str(rawpath.resolve()),'raw_sha256':hashlib.sha256(rawpath.read_bytes()).hexdigest()})
        dump(target,doc)
        return {**doc['summary'],'session_id':sid,'collection_status':'fetched'}
    a.raw_dir.mkdir(parents=True,exist_ok=True,mode=0o700);a.out.mkdir(parents=True,exist_ok=True)
    results=[]; t0=time.monotonic()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(collect,r):r for r in records}
        for f in as_completed(futures):
            rec=futures[f]
            try:result=f.result()
            except Exception as exc:
                # Error type and bounded text only, never dump HTTP headers.
                result={'session_id':rec['session_id'],'collection_status':'failed','error_type':type(exc).__name__,'error':str(exc)[:350]}
            results.append(result)
            with (a.out/'progress.jsonl').open('a') as log:log.write(json.dumps(result,ensure_ascii=False)+'\n')
            if len(results)%10==0 or result['collection_status']=='failed':
                print(json.dumps({'done':len(results),'target':len(records),'failed':sum(r['collection_status']=='failed' for r in results),
                    'llm_spans':sum(r.get('span_count',0) for r in results),
                    'main_calls':sum(r.get('main_calls',0) for r in results),'elapsed_s':round(time.monotonic()-t0,1),
                    'last_status':result['collection_status'],'last_error':result.get('error')},ensure_ascii=False),flush=True)
    dump(a.out/'collection.json',{'requested':len(records),'results':results,'elapsed_s':time.monotonic()-t0})
    print(json.dumps({'complete':True,'requested':len(records),'success':sum(r['collection_status']!='failed'for r in results),'failed':sum(r['collection_status']=='failed'for r in results)},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
