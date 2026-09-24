#!/usr/bin/env python3
"""Inspect actual tool/compression spans around selected LLM transitions.

Diagnostic selection by observed gap/compression, never population weighting.
Only structural attributes are exported. Tool unions require matching run IDs.
"""
import argparse
import json
import os
from datetime import timedelta
from pathlib import Path
import time

from phoenix_longchain_observe import iso


FIELDS=['name','span_kind','context.span_id','context.trace_id','parent_id','start_time','end_time',
        'attributes.session.id','attributes.muse.agent.run_uuid','attributes.muse.agent.llm.run_uuid',
        'attributes.muse.agent.tool.name','attributes.muse.agent.tool.status',
        'attributes.muse.agent.tool.result.bytes','attributes.muse.agent.tool.result.estimated_tokens',
        'attributes.muse.agent.compression.phase','attributes.muse.agent.compression.event_id',
        'attributes.muse.agent.compression.tokens_before','attributes.muse.agent.compression.tokens_after',
        'attributes.muse.agent.compression.first_diff_message_index',
        'attributes.muse.agent.compression.prev_llm_span_id']


def union(intervals):
    total=0;end=None
    for lo,hi in sorted(intervals):
        if hi<=lo:continue
        total+=(hi-max(lo,end)).total_seconds() if end is not None and hi>end else (hi-lo).total_seconds() if end is None else 0
        end=max(end,hi) if end else hi
    return total


def summarize_window(c, rows):
    """Prefer observed common parent membership when custom projections are null."""
    before,after=c['before'],c['after'];run=c['edge']['run_id']
    common_parent=before['parent_id'] if before['parent_id']==after['parent_id'] else None
    gaplo,hi=iso(before['end']),iso(after['start'])
    matched=[r for r in rows if r['name'].startswith('muse.agent.tool_call.') and r.get('end_time') and
             (r.get('attributes.muse.agent.run_uuid')==run or
              (common_parent is not None and r.get('parent_id')==common_parent))]
    intervals=[(max(gaplo,iso(r['start_time'])),min(hi,iso(r['end_time'])))for r in matched]
    compression=[r for r in rows if r.get('attributes.muse.agent.compression.event_id') or
                 (common_parent is not None and r.get('parent_id')==common_parent and
                  'compress' in r['name'] and r.get('span_kind')=='CHAIN')]
    return {'current_span':c['edge']['current'],'session_id':c['session_id'],'gap_s':c['edge']['gap_s'],
            'rows':len(rows),'matched_parent_run_tool_spans':len(matched),
            'matched_tool_union_within_gap_s':union(intervals) if matched else None,
            'matched_tool_names':sorted({r['name'] for r in matched}),
            'compression_event_spans':len(compression),'status':'ok',
            'matching_basis':'Same observed parent as both adjacent LLM calls, or matching projected run ID.',
            'custom_attribute_projection_nonnull':sum(any(r.get(k)is not None for k in FIELDS if k.startswith('attributes.')) for r in rows),
            'caveat':'Union only spans started after preceding LLM start and belonging to same parent/run; not proof all blocking tools were captured. Null custom projections do not prove missing original attributes.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--count',type=int,default=40);p.add_argument('--limit',type=int,default=500)
    a=p.parse_args(); candidates=[]
    for path in sorted((a.root/'sessions').glob('*.json')):
        d=json.loads(path.read_text());calls={c['span_id']:c for c in d['calls']}
        for e in d['longest_wait_edges'][:1]+d['compression_events'][:1]:
            if not e['index_consecutive'] or (e['gap_s']or 0)<=0:continue
            before,after=calls[e['previous']],calls[e['current']]
            candidates.append({'session_id':d['sampling']['session_id'],'edge':e,'before':before,'after':after})
    waits=sorted((c for c in candidates if not(c['edge']['compression_count_delta']or 0)),key=lambda c:c['edge']['gap_s'],reverse=True)
    compress=sorted((c for c in candidates if (c['edge']['compression_count_delta']or 0)>0),key=lambda c:abs(c['edge']['prompt_delta']or 0),reverse=True)
    selected={c['edge']['current']:c for c in waits[:a.count//2]+compress[:a.count-a.count//2]}
    for k in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):os.environ.pop(k,None)
    from phoenix.client import Client
    from phoenix.client.types.spans import SpanQuery
    client=Client(base_url='https://scimaster-phoenix.bohrium.com',api_key=os.environ.get('PHOENIX_API_KEY')or None)
    out=a.root/'event-windows';out.mkdir(exist_ok=True)
    results=[]
    for sid,c in selected.items():
        target=out/f'{sid}.json'
        if target.exists():
            old=json.loads(target.read_text());old['summary']=summarize_window(old['transition'],old['spans'])
            target.write_text(json.dumps(old,ensure_ascii=False,indent=2)+'\n')
            results.append(old['summary']);continue
        tid=c['before']['trace_id'];pages=[]
        def fetch(lo,hi):
            query=SpanQuery().where(f"trace_id == '{tid}' and span_kind != 'LLM'").select(*FIELDS)
            for attempt in range(4):
                try:
                    time.sleep(.5)
                    df=client.spans.get_spans_dataframe(query=query,project_identifier='default',start_time=lo,end_time=hi,limit=a.limit,timeout=60)
                    break
                except Exception:
                    if attempt==3:raise
                    time.sleep(5*(attempt+1))
            n=0 if df is None else len(df)
            if n>=a.limit:
                if hi-lo<timedelta(milliseconds=1):raise RuntimeError('Refuse saturated time slice')
                mid=lo+(hi-lo)/2;return fetch(lo,mid)+fetch(mid,hi)
            pages.append({'start':lo.isoformat(),'end':hi.isoformat(),'n':n})
            if not n:return []
            if 'context.span_id'not in df.columns:df=df.reset_index()
            rs=json.loads(df.to_json(orient='records',date_format='iso'))
            assert all(r['context.trace_id']==tid and lo<=iso(r['start_time'])<=hi for r in rs),'Query filter mismatch'
            return rs
        try:
            lo,hi=iso(c['before']['start']),iso(c['after']['start'])
            rs=fetch(lo,hi);rows=list({r['context.span_id']:r for r in rs}.values())
            gaplo=iso(c['before']['end']);run=c['edge']['run_id']
            matched=[r for r in rows if r['name'].startswith('muse.agent.tool_call.') and
                     r.get('attributes.muse.agent.run_uuid')==run and r.get('end_time')]
            intervals=[(max(gaplo,iso(r['start_time'])),min(hi,iso(r['end_time'])))for r in matched]
            compression=[r for r in rows if r.get('attributes.muse.agent.compression.event_id')]
            summary=summarize_window(c,rows)
            doc={'summary':summary,'selection':'largest waits and prompt drops, diagnostic not representative',
                 'transition':c,'pages':pages,'spans':rows}
            target.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n');results.append(summary)
        except Exception as exc:results.append({'current_span':sid,'status':'failed','error':str(exc)[:200]})
        print(json.dumps({'done':len(results),'target':len(selected),'last':results[-1]},ensure_ascii=False),flush=True)
    (out/'index.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
