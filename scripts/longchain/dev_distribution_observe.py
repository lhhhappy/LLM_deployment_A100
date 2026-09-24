#!/usr/bin/env python3
"""Read-only workload audit; source chain aggregates are not hidden formal data.

Print structural statistics only. No prompt text, engine calls or dataset edits.
"""
import collections
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "s1-dev/data/dev-combined-v1"


def rid(row):
    return f"{row['pack']}:{row['view']}:{row['logical_call_id']}"


def stats(values):
    values = sorted(x for x in values if x is not None)
    if not values:
        return {"n": 0}
    return {"n": len(values), "mean": sum(values) / len(values),
            **{f"p{q}": values[min(len(values)-1, int(q/100*len(values)))]
               for q in (50, 90, 95, 99)}, "max": values[-1]}


def lengths(values):
    groups = [("1-4", 1, 4), ("5-8", 5, 8), ("9-15", 9, 15),
              ("16-30", 16, 30), ("31-99", 31, 99), ("100+", 100, float('inf'))]
    return {"summary": stats(values), "bins": [
        {"bin": name, "chains": sum(lo <= v <= hi for v in values),
         "calls": sum(v for v in values if lo <= v <= hi)}
        for name, lo, hi in groups]}


def families(values):
    c = collections.Counter(values)
    assert None not in c
    n = sum(c.values())
    return {"unique_exact_hashes": len(c), "largest_counts": sorted(c.values(), reverse=True)[:10],
            "pair_same_hash_probability_without_replacement":
                sum(v*(v-1) for v in c.values())/(n*(n-1)) if n > 1 else None}


def main():
    rows = [r for r in map(json.loads, (DATA / 'requests.jsonl').open())
            if r.get('view') == 'canon' and r.get('in_serving_load')]
    chains = {c['chain_id']: c for c in map(json.loads, (DATA / 'chains.jsonl').open())}
    grouped = collections.defaultdict(list)
    for r in rows:
        grouped[r['chain_id']].append(r)
    for rs in grouped.values():
        rs.sort(key=lambda r: (r.get('dispatch_offset_ms') or 0, r['logical_call_id']))
    assert len({rid(r) for r in rows}) == len(rows) == 722
    assert len(grouped) == 311
    heads = {rid(rs[0]) for rs in grouped.values()}
    raw_file = ROOT / 'evidence/L048-official_a_122_n22/N22/raw_dev-combined-v1_N22_1790225665.jsonl'
    raw = {r['req_id']: r for r in map(json.loads, raw_file.open())}
    assert set(raw) == {rid(r) for r in rows}
    raw_by_position = {(r['chain_id'], r['idx_in_chain']): r for r in raw.values()}
    # Prior audit used the same immutable 722 inputs; only token LCP is reused,
    # never its run-specific cache/latency measurements.
    lcp = {r['req_id']: int(r['true_lcp']) for r in csv.DictReader(
        (ROOT / 'evidence/T56/pairs_attributed.csv').open())}
    assert set(lcp) == set(raw) - heads
    out = {"provenance": {"dataset": str(DATA.relative_to(ROOT)),
        "runtime": str(raw_file.relative_to(ROOT)),
        "lcp": 'evidence/T56/pairs_attributed.csv'},
        "caveats": ["Source n_requests is metadata, not observed full bodies or formal cohort.",
                    "Exact sys/tools hashes do not measure partial common prefixes.",
                    "Tool union is aggregate; no per-tool timing attribution.",
                    "LCP/cache deficit is not proof of eviction; runtime results are N22 run 048."]}
    for pack in ['all', 'biomaster', 'scimaster']:
        cs = [rs for rs in grouped.values() if pack == 'all' or rs[0]['pack'] == pack]
        rs = [r for c in cs for r in c]
        edges = [r for r in rs if rid(r) not in heads]
        top = []
        for r in sorted(edges, key=lambda r: r.get('tool_union_ms') or 0, reverse=True)[:5]:
            rr = raw[rid(r)]
            prev = raw_by_position[rr['chain_id'], rr['idx_in_chain']-1]
            start, end = prev['client_finish_at_s'], rr['client_dispatch_at_s']
            others = [v for v in raw.values() if v['chain_id'] != rr['chain_id']
                      and start <= v['client_dispatch_at_s'] < end]
            top.append({**{k: r.get(k) for k in ["pack", "logical_call_id", "phase", "edge_type",
                         "tool_union_ms", "net_think_ms", "replay_gap_ms", "gap_valid"]},
                        "true_lcp": lcp[rid(r)], "prompt": rr['prompt_tokens'],
                        "cached_048": rr['cached_tokens'], "ttft_048_s": rr['ttft_s'],
                        "effective_gap_048_ms": rr['effective_replay_gap_ms'],
                        "previous_cached_048": prev['cached_tokens'],
                        "observed_client_gap_048_s": end-start,
                        "other_dispatches_during_gap_048": len(others),
                        "other_chains_during_gap_048": len({v['chain_id'] for v in others}),
                        "other_uncached_tokens_dispatched_during_gap_048":
                            sum(v['prompt_tokens']-v['cached_tokens'] for v in others),
                        "lcp_cache_deficit_048": max(0, lcp[rid(r)]-rr['cached_tokens'])})
        out[pack] = {"visible_lengths": lengths([len(c) for c in cs]),
            "source_declared_lengths": lengths([chains[c[0]['chain_id']]['n_requests'] for c in cs]),
            "head_sys_tools": families([c[0]['sys_tools_hash'] for c in cs]),
            "request_sys_tools": families([r['sys_tools_hash'] for r in rs]),
            "visible_edges": len(edges),
            "edge_tool_seconds": stats([r['tool_union_ms']/1000 for r in edges if r.get('tool_union_ms') is not None]),
            "edge_replay_seconds": stats([r['replay_gap_ms']/1000 for r in edges if r.get('replay_gap_ms') is not None]),
            "edge_tool_ge_seconds": {str(t): sum((r.get('tool_union_ms') or 0) >= t*1000 for r in edges)
                                      for t in (10, 30, 60, 300)},
            "nonzero_head_gaps": sum((c[0].get('replay_gap_ms') or 0) > 0 for c in cs),
            "source_gap_formula_mismatches": sum(r['replay_gap_ms'] != min(r['tool_union_ms'],300000)+min(r['net_think_ms'],10000)
                 for r in rs if all(r.get(k) is not None for k in ['replay_gap_ms','tool_union_ms','net_think_ms'])),
            "top_within_chain_tool_waits": top}
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
