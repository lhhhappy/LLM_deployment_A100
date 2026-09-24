#!/usr/bin/env python3
"""Read-only Phoenix sample export and structural audit; never a replay dataset.

Run with uv --with arize-phoenix-client --with pandas. Raw bodies belong under
ignored cache/, summaries contain no message text. All counts are window-limited.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path


def parsed(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def quantiles(values):
    values = sorted(v for v in values if isinstance(v, (int, float)))
    return {"n": len(values), **{k: values[min(len(values)-1, int(q*len(values)))]
            if values else None for k, q in (("p50", .5), ("p90", .9), ("p95", .95), ("max", 1))}}


def iso(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def audit(rows):
    calls = []
    part_types = collections.Counter()
    output_tools = collections.Counter()
    coverage = collections.Counter()
    for r in sorted(rows, key=lambda r: (r.get("start_time") or "", r.get("context.span_id") or "")):
        muse = r.get("attributes.muse") or {}
        agent = muse.get("agent") or {}
        llm = agent.get("llm") or {}
        gen = r.get("attributes.gen_ai") or {}
        msgs = parsed((gen.get("input") or {}).get("messages"))
        outputs = parsed((gen.get("output") or {}).get("messages"))
        msgs = msgs if isinstance(msgs, list) else []
        outputs = outputs if isinstance(outputs, list) else []
        for m in outputs:
            for part in m.get('parts') or []:
                if part.get('name'):
                    output_tools[part['name']] += 1
        cache = llm.get("cache") or {}
        summaries = parsed(llm.get("cache_summary")) or {}
        for m in msgs:
            for p in m.get("parts") or []:
                part_types[p.get("type", "unknown")] += 1
        for k, v in {"input_value": r.get("attributes.input.value"),
                     "gen_ai_input_messages": msgs, "gen_ai_output_messages": outputs,
                     "input_system_message": any(m.get("role") in ("system", "developer") for m in msgs),
                     "invocation_parameters": r.get("attributes.llm.invocation_parameters"),
                     "llm_tools": r.get("attributes.llm.tools")}.items():
            if v:
                coverage[k] += 1
        prompt = r.get("attributes.llm.token_count.prompt")
        cached = r.get("attributes.llm.token_count.prompt_details.cache_read")
        calls.append({"span_id": r.get("context.span_id"), "trace_id": r.get("context.trace_id"),
                      "parent_id": r.get("parent_id"), "start": r.get("start_time"), "end": r.get("end_time"),
                      "status": r.get("status_code"), "run_id": llm.get("run_uuid") or agent.get("run_uuid"),
                      "message_id": llm.get("user_message_uuid"), "call_index": llm.get("call_index_in_run"),
                      "call_type": llm.get("call_type"), "agent": muse.get("agent.name") or gen.get("agent.name"),
                      "model": r.get("attributes.llm.model_name"), "request_id": llm.get("request_id"),
                      "prompt_tokens_source": prompt, "output_tokens_source": r.get("attributes.llm.token_count.completion"),
                      "cached_tokens_source": cached, "cache_ratio": cached/prompt if isinstance(cached,(int,float)) and prompt else None,
                      "reported_gap_ms": llm.get("gap_since_prev_call_ms"), "system_hash": llm.get("system_prompt_hash"),
                      "tools_hash": llm.get("tool_schemas_hash"), "tools_changed_flag": llm.get("tool_schemas_hash_changed"),
                      "is_compression": llm.get("is_compression"), "compression_count": llm.get("compress_count_in_run"),
                      "miss_bucket": cache.get("miss_bucket"), "miss_reasons": summaries.get("top_miss_reasons") if isinstance(summaries,dict) else None,
                      "input_message_count": len(msgs), "input_roles": dict(collections.Counter(m.get("role") for m in msgs)),
                      "input_chars": sum(len(json.dumps(m,ensure_ascii=False)) for m in msgs),
                      "tool_schema_count": llm.get("tool_schemas_count"), "payload_token_estimates": llm.get("payload"),
                      "finish_reasons": parsed(llm.get("finish_reasons")),
                      "attempt_count": len(parsed(llm.get("attempts")) or []),
                      "duration_s": (iso(r["end_time"])-iso(r["start_time"])).total_seconds() if r.get("end_time") and r.get("start_time") else None})
    main = [c for c in calls if c["call_type"] == "mainagent" and c["agent"] == "scimaster"]
    groups = collections.defaultdict(list)
    for c in main:
        if c["run_id"]:
            groups[c["run_id"]].append(c)
    edges = []
    for run, cs in groups.items():
        for a,b in zip(cs,cs[1:]):
            gap=(iso(b["start"])-iso(a["end"])).total_seconds() if a["end"] and b["start"] else None
            ia,ib=a["call_index"],b["call_index"]
            ca,cb=a["compression_count"],b["compression_count"]
            edges.append({"run_id":run,"previous":a["span_id"],"current":b["span_id"],"gap_s":gap,
                          "index_consecutive": (ib==ia+1) if isinstance(ia,int) and isinstance(ib,int) else None,
                          "compression_count_before":ca,"compression_count_after":cb,
                          "compression_count_delta":cb-ca if isinstance(ca,int) and isinstance(cb,int) else None,
                          "system_changed": a["system_hash"]!=b["system_hash"] if a["system_hash"] and b["system_hash"] else None,
                          "tools_changed": a["tools_hash"]!=b["tools_hash"] if a["tools_hash"] and b["tools_hash"] else None,
                          "model_changed": a["model"]!=b["model"],
                          "prompt_delta": b["prompt_tokens_source"]-a["prompt_tokens_source"] if isinstance(a["prompt_tokens_source"],(int,float)) and isinstance(b["prompt_tokens_source"],(int,float)) else None,
                          "next_cache_ratio":b["cache_ratio"]})
    valid_edges=[e for e in edges if e["index_consecutive"] and e["gap_s"] is not None and e["gap_s"]>=0]
    summary={"span_count":len(rows),"main_calls":len(main),"main_runs":len(groups),
             "distinct_main_user_messages":len({c["message_id"] for c in main if c["message_id"]}),
             "call_types":dict(collections.Counter(c["call_type"] for c in calls)),
             "models":dict(collections.Counter(c["model"] for c in main)),
             "status":dict(collections.Counter(c["status"] for c in calls)),
             "field_coverage_all_llm":dict(coverage),"input_part_types":dict(part_types),
             "output_tool_calls":dict(output_tools),
             "main_calls_per_run":quantiles([len(cs) for cs in groups.values()]),
             "main_prompt_tokens_source":quantiles([c["prompt_tokens_source"] for c in main]),
             "main_output_tokens_source":quantiles([c["output_tokens_source"] for c in main]),
             "main_cache_ratio_source":quantiles([c["cache_ratio"] for c in main]),
             "main_missing_cache_count":sum(c["cache_ratio"] is None for c in main),
             "main_cache_below_half":sum(c["cache_ratio"] is not None and c["cache_ratio"]<.5 for c in main),
             "within_run_adjacent_edges":len(edges),"consecutive_nonoverlap_edges":len(valid_edges),
             "within_run_consecutive_gap_s":quantiles([e["gap_s"] for e in valid_edges]),
             "overlapping_edges":sum(e["gap_s"] is not None and e["gap_s"]<0 for e in edges),
             "system_hash_changes_observed":sum(e["system_changed"] is True for e in edges),
             "tools_hash_changes_observed":sum(e["tools_changed"] is True for e in edges),
             "prompt_drops_gt_10k_observed":sum(e["prompt_delta"] is not None and e["prompt_delta"]< -10000 for e in edges),
             "compression_calls":sum(c["is_compression"] is True for c in calls),
             "compression_calls_definition":"LLM calls flagged is_compression, not context compression events; zero does not establish absence of compression.",
             "consecutive_edges_with_compression_count_increase":sum(
                 e["index_consecutive"] is True and (e["compression_count_delta"] or 0)>0 for e in edges),
             "compression_count_increases_on_consecutive_edges":sum(
                 max(0,e["compression_count_delta"] or 0) for e in edges if e["index_consecutive"] is True),
             "main_calls_after_reported_compression":sum((c["compression_count"] or 0)>0 for c in main),
             "main_miss_buckets":dict(collections.Counter(str(c["miss_bucket"]) for c in main)),
             "complete_session_proven":False,
             "caveat":"Selected session window, not a representative sample or a complete replay chain. Source-model tokens and provider cache hits are not GLM tokens or TP8 cache behavior. Adjacent starts do not prove causal dependencies."}
    return summary,calls,edges


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',action='append',required=True)
    p.add_argument('--start',required=True);p.add_argument('--end',required=True)
    p.add_argument('--raw-dir',type=Path,required=True);p.add_argument('--report-dir',type=Path,required=True)
    p.add_argument('--limit',type=int,default=2000)
    a=p.parse_args()
    start,end=iso(a.start),iso(a.end)
    if start.tzinfo is None or end.tzinfo is None or start>=end or a.limit<2:
        p.error('Require timezone-aware start < end and limit >= 2')
    for k in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
        os.environ.pop(k,None)
    from phoenix.client import Client
    from phoenix.client.types.spans import SpanQuery
    client=Client(base_url='https://scimaster-phoenix.bohrium.com',api_key=os.environ.get('PHOENIX_API_KEY') or None)
    a.raw_dir.mkdir(parents=True,exist_ok=True,mode=0o700);a.report_dir.mkdir(parents=True,exist_ok=True)
    for sid in a.session:
        # Restrict interpolation to UUID syntax; never put free text into DSL.
        import uuid
        if str(uuid.UUID(sid))!=sid: p.error('Expected canonical session UUID')
        # The production dataframe endpoint returned zero for session attribute
        # filters even on a known positive sample. Resolve via sessions -> traces,
        # as in the supplied session exporter. Trace-ID filtering was calibrated.
        turns=client.sessions.get_session_turns(session_id=sid,timeout=45)
        trace_ids=sorted({t['trace_id'] for t in turns if t.get('trace_id')})
        if not trace_ids: raise RuntimeError(f'No session turns for {sid}; cannot prove absence')
        condition=' or '.join(f"trace_id == '{tid}'" for tid in trace_ids)
        query=SpanQuery().where(f"span_kind == 'LLM' and ({condition})")
        pages=[]
        def fetch(lo,hi):
            df=client.spans.get_spans_dataframe(query=query,project_identifier='default',start_time=lo,end_time=hi,limit=a.limit,timeout=60)
            n=0 if df is None else len(df)
            if n>=a.limit:
                if hi-lo<timedelta(seconds=1): raise RuntimeError('Saturated subsecond window; refuse truncation')
                mid=lo+(hi-lo)/2
                return fetch(lo,mid)+fetch(mid,hi)
            pages.append({'start':lo.isoformat(),'end':hi.isoformat(),'rows':n})
            if not n:return []
            # Span ID is normally both index and a column; preserve the column.
            if 'context.span_id' not in df.columns: df=df.reset_index()
            df=df.loc[:,~df.columns.duplicated()]
            return json.loads(df.to_json(orient='records',date_format='iso'))
        rows=fetch(start,end)
        if not rows: raise RuntimeError(f'No LLM spans for {sid} in requested window')
        unique={r['context.span_id']:r for r in rows}
        if any(r.get('attributes.session.id')!=sid for r in unique.values()):
            raise RuntimeError('Session mismatch in trace-filtered export')
        raw=a.raw_dir/f'{sid}.json.gz'
        with gzip.open(raw,'wt',encoding='utf-8') as f:json.dump(list(unique.values()),f,ensure_ascii=False)
        raw.chmod(0o600)
        summary,calls,edges=audit(list(unique.values()))
        summary.update({'session_id':sid,'start':a.start,'end':a.end,'pages':pages,'duplicates_removed':len(rows)-len(unique),
                        'session_turns_returned':len(turns),'session_trace_ids_returned':len(trace_ids),
                        'raw_path':str(raw.resolve()),'raw_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),
                        'fetched_at':datetime.now(timezone.utc).isoformat()})
        (a.report_dir/f'{sid}.json').write_text(json.dumps({'summary':summary,'calls':calls,'edges':edges},ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__': main()
