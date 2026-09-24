#!/usr/bin/env python3
"""Weighted source behavior profile and observable joint event records.

No scoring phases are inferred. Source token deltas are not token LCP.
"""
import argparse
import collections
import json
from pathlib import Path


def weighted(values):
    pairs=sorted((float(v),float(w)) for v,w in values if isinstance(v,(int,float)))
    total=sum(w for _,w in pairs)
    if not pairs:return {'observed_n':0}
    result={'observed_n':len(pairs),'expanded_weight':total,'mean':sum(v*w for v,w in pairs)/total}
    for p in (.5,.9,.95,.99):
        acc=0
        for value,w in pairs:
            acc+=w
            if acc>=p*total:
                result['p'+str(int(p*100))]=value
                break
    result['max']=pairs[-1][0]
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    frame=json.loads((a.root/'sampling-frame.json').read_text())
    hist=collections.defaultdict(list);categories=collections.Counter();paircounts=collections.Counter()
    data=[];session_profiles=[]
    for rec in frame['records']:
        path=a.root/'sessions'/f"{rec['session_id']}.json"
        if not path.exists():continue
        d=json.loads(path.read_text());w=rec['expansion_weight'];sid=rec['session_id']
        calls={c['span_id']:c for c in d['calls']}
        main=[c for c in d['calls'] if c['call_type']=='mainagent' and c['agent']=='scimaster']
        hist['calls_per_observed_session_window'].append((len(main),w))
        hist['distinct_user_messages_per_session_window'].append((d['summary']['distinct_main_user_messages'],w))
        for c in main:
            hist['prompt_tokens_source'].append((c['prompt_tokens_source'],w))
            hist['output_tokens_source'].append((c['output_tokens_source'],w))
        for f in d['continuous_fragments']:hist['continuous_fragment_call_count'].append((f['n_calls'],w))
        for e in d['edges']:
            if not e['index_consecutive'] or e['gap_s'] is None or e['gap_s']<0:continue
            before,after=calls[e['previous']],calls[e['current']]
            if before['status']=='ERROR' or after['status']=='ERROR':continue
            event=('reported_compression' if (e['compression_count_delta']or 0)>0 else
                   'system_or_tools_change' if e['system_changed']or e['tools_changed'] else
                   'prompt_drop_without_compression_evidence' if (e['prompt_delta']or 0)<0 else
                   'continuation_without_observed_rebuild')
            row={'session_id':sid,'run_id':e['run_id'],'previous_span':e['previous'],'current_span':e['current'],
                 'weight':w,'event_observation':event,'before_prompt_tokens_source':before['prompt_tokens_source'],
                 'after_prompt_tokens_source':after['prompt_tokens_source'],'prompt_delta_source':e['prompt_delta'],
                 'previous_output_tokens_source':before['output_tokens_source'],'next_output_tokens_source':after['output_tokens_source'],
                 'gap_s':e['gap_s'],'compression_count_delta':e['compression_count_delta'],
                 'same_system_hash':e['system_changed'] is False,'same_tools_hash':e['tools_changed'] is False,
                 'model_changed':e['model_changed'],'model_source':after['model'],
                 'before_cache_ratio_source':before['cache_ratio'],'after_cache_ratio_source':after['cache_ratio'],
                 'before_payload_token_estimates':before['payload_token_estimates'],
                 'after_payload_token_estimates':after['payload_token_estimates'],
                 'tool_wait_decomposition_proven':False,'competition_context_reconstructed':False,
                 'glm_lcp_known':False}
            data.append(row);categories[event]+=w;paircounts[event]+=1
            for key,value in [('gap_s',e['gap_s']),('prompt_delta_source',e['prompt_delta']),('next_output_tokens_source',after['output_tokens_source'])]:
                hist[event+'/'+key].append((value,w))
        session_profiles.append({'session_id':sid,'weight':w,'user_messages':d['summary']['distinct_main_user_messages'],
                                 'runs':d['summary']['main_runs'],'calls':len(main)})
    with (a.root/'joint-events.jsonl').open('w') as f:
        for row in data:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    profile={'window':[frame['window_start'],frame['window_end']],
             'collected_sessions':len(session_profiles),'requested_sessions':len(frame['records']),
             'weighting':'inverse inclusion probability within source DB weekly usage-count strata; descriptive source-population estimates, not contest targets',
             'valid_adjacent_events':len(data),'observed_event_counts':dict(paircounts),
             'expanded_event_counts':dict(categories),'distributions':{k:weighted(v)for k,v in hist.items()},
             'limitations':['No imputation or nonresponse correction for failed sessions.',
                            'Session and fragment lengths are window-censored.',
                            'Some user messages can originate in automated testing; identity type is not verified.',
                            'No exact GLM LCP or competition state is inferred from provider cache counters.',
                            'Event observations do not define contest phases.']}
    (a.root/'behavior-profile.json').write_text(json.dumps(profile,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'sessions':len(session_profiles),'events':len(data),'observed_event_counts':dict(paircounts)},ensure_ascii=False))


if __name__=='__main__':main()
