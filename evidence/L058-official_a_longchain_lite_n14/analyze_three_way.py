"""Reproduce aggregate comparison of two workloads on official A, N14.

Different workloads: these are distribution diagnostics, not an optimization A/B.
Official A exposes aggregate stress results only. Never invent its raw percentiles.
"""
import collections
import datetime as dt
import json
import math
import pathlib
import re
import statistics
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "s1-dev/harness"))
from s1_common import in_ttft_gate, q


def jl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def dist(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                median=statistics.median(values) if values else None,
                **{f"p{n}": q(values, n / 100) for n in (50, 90, 95, 99)},
                max=max(values) if values else None)


def occupancy(intervals, start, end):
    events = collections.Counter({start: 0, end: 0})
    for a, b in intervals:
        a, b = max(a, start), min(b, end)
        if b > a:
            events[a] += 1
            events[b] -= 1
    last, count = start, 0
    seconds = collections.Counter()
    for t, delta in sorted(events.items()):
        seconds[count] += t - last
        count += delta
        last = t
    return {"mean": sum(n * s for n, s in seconds.items()) / (end - start),
            "fraction_at_least_10": sum(s for n, s in seconds.items() if n >= 10) / (end - start),
            "fraction_at_most_3": sum(s for n, s in seconds.items() if n <= 3) / (end - start),
            "seconds_by_count": dict(sorted(seconds.items()))}


def sampled_metrics(path, start, end):
    rows = sorted(jl(path), key=lambda r: r["t"])
    keys = ("num_running_reqs", "num_queue_reqs", "token_usage", "full_token_usage",
            "mamba_usage", "kv_available_tokens", "kv_evictable_tokens", "kv_used_tokens",
            "mamba_available_tokens", "mamba_evictable_tokens", "mamba_used_tokens",
            "evicted_tokens_total", "spec_accept_length")
    result = {}
    for key in keys:
        values = []
        for a, b in zip(rows, rows[1:]):
            if key not in a or b["t"] - a["t"] > 30:
                continue
            duration = min(b["t"], end) - max(a["t"], start)
            if duration > 0:
                values.append((a[key], duration))
        if values:
            seconds = sum(w for _, w in values)
            result[key] = {"time_mean": sum(v * w for v, w in values) / seconds,
                           "observed_fraction": seconds / (end - start),
                           "sample_min": min(v for v, _ in values),
                           "sample_max": max(v for v, _ in values)}
    active = [r for r in rows if start <= r["t"] <= end]
    for pool in ("kv", "mamba"):
        keys3 = [pool + s for s in ("_available_tokens", "_evictable_tokens", "_used_tokens")]
        usable = [r for r in active if all(k in r for k in keys3)]
        if usable:
            result[pool + "_pool_samples"] = {
                "immediately_free_plus_evictable": dist([r[keys3[0]] + r[keys3[1]] for r in usable]),
                "sum_free_evictable_used": dist([sum(r[k] for k in keys3) for r in usable])}
    return result


def gpu_util(path, start, end):
    by_gpu = collections.defaultdict(list)
    for line in path.read_text().splitlines():
        fields = [v.strip() for v in line.split(",")]
        if len(fields) != 4:
            continue
        t = dt.datetime.strptime(fields[0], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()
        by_gpu[fields[1]].append((t, float(fields[2].split()[0])))
    means = []
    for rows in by_gpu.values():
        acc = seconds = 0.0
        for (t, value), (t2, _) in zip(rows, rows[1:]):
            duration = min(t2, end) - max(t, start)
            if duration > 0 and t2 - t < 15:
                acc += duration * value
                seconds += duration
        if seconds:
            means.append(acc / seconds)
    return {"gpu_count": len(means), "time_mean_across_gpus": statistics.mean(means) if means else None}


def analyze(path):
    summary = json.loads((path / "summary.json").read_text())
    verdict = json.loads((path / "level_verdict.json").read_text())
    score = json.loads((path / "score_formal.json").read_text())
    assert verdict["status"] == "VALID" and verdict["n"] == 14
    rows = jl(path / pathlib.Path(summary["raw"]).name)
    assert len(rows) == len({r["req_id"] for r in rows}) == verdict["rows"]
    assert not any(r["error_class"] or r["error"] for r in rows)
    assert all(r["prompt_tokens"] == r["glm_tokens"] and r["output_tokens"] == r["max_output_i"] for r in rows)
    start = min(r["client_dispatch_at_s"] for r in rows)
    end = max(r["client_finish_at_s"] for r in rows)
    chain_rows = collections.defaultdict(list)
    for r in rows:
        chain_rows[r["chain_id"]].append(r)
    requests = [(r["client_dispatch_at_s"], r["client_finish_at_s"]) for r in rows]
    chains = [(min(r["client_dispatch_at_s"] for r in rs), max(r["client_finish_at_s"] for r in rs)) for rs in chain_rows.values()]
    prompt = sum(r["prompt_tokens"] for r in rows)
    cached = sum(r["cached_tokens"] for r in rows)
    output = sum(r["output_tokens"] for r in rows)
    tpot = dist([r["tpot_s"] for r in rows])
    assert abs(tpot["mean"] - verdict["tpot_mean"]) < 1e-12
    assert tpot["p95"] == verdict["tpot_p95"]
    total_decode_seconds = sum(r["client_finish_at_s"] - r["client_first_token_at_s"] for r in rows)
    gates = {}
    for name, limit in (("fast_intra", 3), ("overall_intra", 5), ("turn_start", 15), ("chain_start", 30)):
        group = [r for r in rows if in_ttft_gate(r, name)]
        bad = [r for r in group if r["ttft_s"] > limit]
        assert len(group) == verdict["ttft_gates"][name]["n"]
        assert len(bad) == verdict["ttft_gates"][name]["over_limit"]
        gates[name] = dict(verdict["ttft_gates"][name], share=len(group) / len(rows),
            over_rate=len(bad) / len(group), ttft=dist([r["ttft_s"] for r in group]),
            bad_recv_to_exec=dist([r["t_exec_start_s"] - r["t_recv_s"] for r in bad]),
            bad_exec_to_first=dist([r["t_first_token_s"] - r["t_exec_start_s"] for r in bad]),
            bad_wait_exceeds_limit=sum(r["t_exec_start_s"] - r["t_recv_s"] > limit for r in bad),
            bad_arrived_first_120s=sum(r["t_recv_s"] < start + 120 for r in bad))
    chain_gate = [r for r in rows if in_ttft_gate(r, 'chain_start')]
    startup = []
    for label, group in [('first_40s', [r for r in chain_gate if r['t_recv_s'] < start+40]),
                         ('after_40s', [r for r in chain_gate if r['t_recv_s'] >= start+40])]:
        startup.append(dict(window=label, n=len(group), over_30s=sum(r['ttft_s']>30 for r in group),
            actual_new_tokens=sum(r['prompt_tokens']-r['cached_tokens'] for r in group),
            frozen_uncached_expected=sum(r['uncached_expected'] for r in group)))
    segments = []
    for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 30), (30, 60)):
        a, b = start + lo * 60, min(start + hi * 60, end)
        if a >= b:
            continue
        rr = [r for r in rows if a <= r["client_dispatch_at_s"] < b]
        segments.append({"start_min": lo, "end_min": (b - start) / 60, "n_dispatched": len(rr),
            "tpot_by_dispatch": dist([r["tpot_s"] for r in rr]),
            "inflight_requests": occupancy(requests, a, b)["mean"],
            "live_chains_including_gaps": occupancy(chains, a, b)["mean"]})
    result = dict(set=summary["set"], n=14, requests=len(rows), chains=len(chains),
        error_count=0, prompt_and_output_match=len(rows), server_ttft_count=sum(r["ttft_source"] == "server" for r in rows),
        wall_s=verdict["wall_s"], dispatch_to_last_finish_s=end-start,
        start_utc=dt.datetime.fromtimestamp(start, dt.timezone.utc).isoformat(),
        end_utc=dt.datetime.fromtimestamp(end, dt.timezone.utc).isoformat(),
        tpot=tpot, tpot_over_100ms=sum(r["tpot_s"] > .1 for r in rows),
        first_20min_dispatch_tpot=dist([r['tpot_s'] for r in rows if r['client_dispatch_at_s'] < start+1200]),
        per_request_ttft_attainment=sum(all(not in_ttft_gate(r,g) or r['ttft_s']<=lim
            for g,lim in [('fast_intra',3),('overall_intra',5),('turn_start',15),('chain_start',30)])
            for r in rows)/len(rows),
        token_weighted_tpot_diagnostic=total_decode_seconds / sum(r["output_tokens"] - 1 for r in rows),
        prompt_tokens=prompt, cached_tokens=cached, uncached_tokens_actual=prompt-cached,
        frozen_uncached_expected=sum(r["uncached_expected"] for r in rows), output_tokens=output,
        token_weighted_cache_hit=cached/prompt, requests_per_chain=dist([len(rs) for rs in chain_rows.values()]),
        prompt_length=dist([r["prompt_tokens"] for r in rows]),
        output_length=dist([r["output_tokens"] for r in rows]),
        actual_new_prompt_tokens=dist([r["prompt_tokens"]-r["cached_tokens"] for r in rows]),
        high_cache_hit_requests=sum(r["cached_tokens"] / r["prompt_tokens"] >= .9 for r in rows),
        gap_seconds=sum(r["effective_replay_gap_ms"] for r in rows)/1000,
        gap_per_request_seconds=dist([r["effective_replay_gap_ms"]/1000 for r in rows]),
        gates=gates, chain_gate_startup=startup, flush=verdict["flush"], budget_attainment=score["dev"]["budget_attainment"],
        formal_tpm_window_valid=score["dev"]["tpm_steady_window"]["validity"],
        inflight_requests=occupancy(requests,start,end), live_chains_including_gaps=occupancy(chains,start,end),
        time_to_90pct_completion_min=(q([r["client_finish_at_s"] for r in rows],.9)-start)/60,
        segments=segments, sampled_engine_metrics=sampled_metrics(path/'metrics.jsonl',start,end),
        gpu_util=gpu_util(path/'gpu_util.csv',start,end))
    return result


if __name__ == "__main__":
    runs = {
        "original_dev_044r": analyze(REPO / 'evidence/L044r-cal_offA_n14/N14'),
        "longchain_lite_058": analyze(OUT / 'N14'),
        "official_A_45979": json.loads((REPO / 'evidence/cost-audit-20260924/official-45979.json').read_text()),
    }
    (OUT/'three-way-metrics.json').write_text(json.dumps(runs,indent=2,ensure_ascii=False)+'\n')
    for name in ('original_dev_044r','longchain_lite_058'):
        r = runs[name]
        print(name, json.dumps({k:r[k] for k in ('requests','chains','wall_s','tpot','tpot_over_100ms','prompt_tokens','cached_tokens','uncached_tokens_actual','output_tokens','token_weighted_cache_hit','gpu_util','segments')},ensure_ascii=False))
