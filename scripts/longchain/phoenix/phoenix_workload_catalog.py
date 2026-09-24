#!/usr/bin/env python3
"""Build a text-free, filterable inventory of collected Phoenix trajectories."""
import argparse
import collections
import hashlib
import html
import json
from pathlib import Path


def q(values, probability):
    values=sorted(x for x in values if isinstance(x,(float,int)))
    return values[min(int(len(values)*probability),len(values)-1)] if values else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--verify-raw',action='store_true')
    a=p.parse_args()
    frame=json.loads((a.root/'sampling-frame.json').read_text())
    catalogue=[]; spans=set(); events=[]; fragments=[]; stats=collections.Counter()
    per_stratum=collections.defaultdict(collections.Counter)
    models=collections.Counter(); tools=collections.Counter(); phases=[]
    missing=[]; raw_hash_errors=[]; days=collections.Counter()
    for rec in frame['records']:
        sid=rec['session_id'];path=a.root/'sessions'/f'{sid}.json'
        if not path.exists():missing.append(sid);continue
        d=json.loads(path.read_text());s=d['summary']
        if a.verify_raw:
            actual=hashlib.sha256(Path(d['raw_path']).read_bytes()).hexdigest()
            if actual!=d['raw_sha256']:raw_hash_errors.append(sid)
        assert d['window_start']==frame['window_start'] and d['window_end']==frame['window_end']
        main=[c for c in d['calls'] if c['call_type']=='mainagent' and c['agent']=='scimaster']
        assert len(main)==s['main_calls']
        for c in d['calls']:
            if c['span_id'] in spans:raise ValueError('Duplicate span across sessions')
            spans.add(c['span_id'])
        valid=[c for c in main if c['run_id'] and isinstance(c['call_index'],int)
               and c['start'] and c['end'] and isinstance(c['prompt_tokens_source'],(int,float))
               and isinstance(c['output_tokens_source'],(int,float)) and c['status']!='ERROR']
        for c in main:
            models[c['model']]+=1
            days[c['start'][:10]]+=1  # explicitly UTC, not a BJT daily estimate
        tools.update(s['output_tool_calls'])
        for k in ['span_count','main_calls','main_runs','distinct_main_user_messages','observed_compression_edges',
                  'consecutive_nonoverlap_edges','input_parts_at_8000_char_cap']:
            stats[k]+=s[k]
        stats['calls_with_required_metadata_and_no_error']+=len(valid)
        stats['llm_error_spans']+=s['status'].get('ERROR',0)
        gaps=[e['gap_s'] for e in d['edges'] if e['index_consecutive'] and e['gap_s'] is not None and e['gap_s']>=0]
        if s['observed_compression_edges']:stats['sessions_with_compression']+=1
        if max(gaps,default=0)>=300:stats['sessions_with_5min_wait']+=1
        if s['max_consecutive_main_calls']>=50:stats['sessions_with_50call_fragment']+=1
        if s['max_consecutive_main_calls']>=100:stats['sessions_with_100call_fragment']+=1
        if s['max_consecutive_main_calls']>=2:stats['sessions_with_continuous_fragment']+=1
        per_stratum[rec['calls_bucket']]['sessions']+=1
        per_stratum[rec['calls_bucket']]['main_calls']+=len(main)
        per_stratum[rec['calls_bucket']]['source_usage_rows']+=rec['usage_rows']
        per_stratum[rec['calls_bucket']]['compression_edges']+=s['observed_compression_edges']
        for e in d['compression_events']:
            events.append({'session_id':sid,'weight':rec['expansion_weight'],**e})
        for f in d['continuous_fragments']:
            fragments.append({'session_id':sid,'weight':rec['expansion_weight'],**f})
        tags=[]
        if s['max_consecutive_main_calls']>=50:tags.append('长连续段')
        if s['observed_compression_edges']:tags.append('压缩前后')
        if max(gaps,default=0)>=300:tags.append('长等待')
        if len(valid)<len(main):tags.append('异常或字段缺失')
        if not tags:tags.append('常规连续段' if s['max_consecutive_main_calls']>=2 else '短调用')
        catalogue.append({'session_id':sid,'bucket':rec['calls_bucket'],'db_rows':rec['usage_rows'],
            'main_calls':len(main),'all_llm':s['span_count'],'user_messages':s['distinct_main_user_messages'],
            'runs':s['main_runs'],'max_contiguous':s['max_consecutive_main_calls'],
            'compressions':s['observed_compression_edges'],'max_gap_s':round(max(gaps,default=0),3),
            'prompt_p50':s['main_prompt_tokens_source']['p50'],'prompt_max':s['main_prompt_tokens_source']['max'],
            'output_p95':s['main_output_tokens_source']['p95'],'body_capped_parts':s['input_parts_at_8000_char_cap'],
            'weight':rec['expansion_weight'],'models':list(s['models']),'tags':tags,
            'metadata_usable_calls':len(valid),'sample_path':f'sessions/{sid}.json',
            'raw_path':d['raw_path'],'raw_sha256':d['raw_sha256']})
    summary={'requested_sessions':len(frame['records']),'collected_sessions':len(catalogue),
        'window':[frame['window_start'],frame['window_end']],'sampling_unit':frame['unit'],
        'raw_unweighted_sample_counts':dict(stats),'population':frame['population'],'quotas':frame['quotas'],
        'per_stratum':dict(per_stratum),'missing_sessions':missing,'raw_hash_errors':raw_hash_errors,
        'models':dict(models),'top_output_tools':tools.most_common(25),'observed_main_calls_by_utc_day':dict(sorted(days.items())),
        'fragment_length':{'n':len(fragments),'p50':q([f['n_calls'] for f in fragments],.5),
                           'p95':q([f['n_calls'] for f in fragments],.95),'max':q([f['n_calls'] for f in fragments],1)},
        'caveats':['Counts are raw stratified sample totals, not population percentages.',
                   'DB usage rows differ from canonical LLM calls; source model tokens differ from GLM.',
                   'Continuous run fragments are not automatically replay chains or complete runs.',
                   'Recorded compression counts do not provide exact removed-prefix positions.',
                   'Inputs are telemetry-bounded. Missing text can be synthesized; behavior boundaries cannot be silently invented.',
                   'This inventory has not been rendered or benchmarked. No GPU score is claimed.']}
    assert not raw_hash_errors,raw_hash_errors
    (a.root/'catalog.json').write_text(json.dumps(catalogue,ensure_ascii=False,indent=2)+'\n')
    (a.root/'corpus-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    for name,data in [('continuous-fragments',fragments),('compression-events',events)]:
        with (a.root/(name+'.jsonl')).open('w') as f:
            for r in data:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    # The catalogue carries only structural counts and local evidence links.
    # All identifiers originate in typed exports; escape HTML/JSON boundaries.
    payload=json.dumps(catalogue,ensure_ascii=False).replace('<','\\u003c')
    template=Path(__file__).with_name('phoenix_workload_catalog.html').read_text()
    page=template.replace('__CATALOG_DATA__',payload).replace('__COUNT__',str(len(catalogue)))
    (a.root/'catalog.html').write_text(page)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
