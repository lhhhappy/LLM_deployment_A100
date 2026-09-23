# T51: summarize ONE torch-profiler trace of a single extend forward (sglang /start_profile ... /stop_profile window).
# Prints JSON: GPU span / busy / idle, idle gaps split into "CPU-starved" (next kernel's launch call issued only after
# the previous kernel ended => GPU waited on the host) vs queued, launch counts, host-sync calls, top kernels,
# component table (rules from component_table.py), and (stack traces) inclusive time of key python functions.
# Usage: python t51_trace_stats.py <trace.json[.gz]> [--top 15] [--kernels-out k.json]
import gzip, json, re, sys, collections
src = open(__file__.replace("t51_trace_stats.py", "../pod/verify/component_table.py")).read()
exec(src[:src.index("root = sys.argv[1]")])  # RULES + comp()
f = sys.argv[1]; TOP = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 15
t = json.load(gzip.open(f, "rt") if f.endswith(".gz") else open(f)); ev = t["traceEvents"] if isinstance(t, dict) else t
K = sorted([e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")], key=lambda e: e["ts"])
RT = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("cuda_runtime", "cuda_driver")]
rt_by_corr = {e["args"].get("correlation"): e for e in RT if "args" in e}
out = {"trace": f}
# restrict to the EXTEND step: GPU kernels inside the gpu_user_annotation 'step[EXTEND ...]' (drops the trailing overlap decode step)
AN = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("user_annotation", "gpu_user_annotation")]
W = [e for e in AN if e["cat"] == "gpu_user_annotation" and e["name"].startswith("step[EXTEND")]
if W:
    w0, w1 = W[0]["ts"], W[0]["ts"] + W[0]["dur"]; K = [e for e in K if w0 <= e["ts"] < w1]
    corr = {e.get("args", {}).get("correlation") for e in K}; RT = [r for r in RT if r.get("args", {}).get("correlation") in corr or not ("Launch" in r["name"] or "Memcpy" in r["name"])]
    RT = [r for r in RT if ("Launch" in r["name"] or "Memcpy" in r["name"]) or (W[0]["ts"] - 60000 <= r["ts"] < w1)]
out["cpu_annotations_ms"] = {}
for e in AN:
    if e["cat"] == "user_annotation" and not e["name"].startswith(("scheduler.recv", "scheduler.process_input", "scheduler.get_next")):
        out["cpu_annotations_ms"][e["name"][:50]] = round(out["cpu_annotations_ms"].get(e["name"][:50], 0) + e["dur"] / 1e3, 2)
if not K: print(json.dumps({"trace": f, "error": "no kernels"})); sys.exit()
t0, t1 = K[0]["ts"], max(e["ts"] + e["dur"] for e in K)
busy = 0; cur_s, cur_e = None, None; gaps = []  # union of kernel intervals (single stream expected)
starved = 0.0; queued_gap = 0.0; big = []
prev_end = None; prev_name = None
for e in K:
    s, d = e["ts"], e["dur"]
    if prev_end is not None and s > prev_end:
        g = s - prev_end
        r = rt_by_corr.get(e.get("args", {}).get("correlation"))
        launch_issue = (r["ts"] + r["dur"]) if r else None
        is_starved = launch_issue is not None and launch_issue >= prev_end  # host issued the launch after GPU went idle
        if is_starved: starved += g
        else: queued_gap += g
        gaps.append(g); big.append((g, prev_name, e["name"], bool(is_starved)))
    prev_end = max(prev_end or 0, s + d); prev_name = e["name"]
    busy += d
ksum = sum(e["dur"] for e in K)
span = t1 - t0
out.update(gpu_span_ms=round(span / 1e3, 2), kernel_sum_ms=round(ksum / 1e3, 2), gpu_idle_ms=round(sum(gaps) / 1e3, 2),
           idle_cpu_starved_ms=round(starved / 1e3, 2), idle_other_ms=round(queued_gap / 1e3, 2),
           n_kernels=sum(1 for e in K if e.get("cat") == "kernel"), n_memcpy=sum(1 for e in K if e.get("cat") == "gpu_memcpy"),
           n_memset=sum(1 for e in K if e.get("cat") == "gpu_memset"),
           gaps_hist={k: sum(1 for g in gaps if lo <= g < hi) for k, lo, hi in [("<5us", 0, 5), ("5-20us", 5, 20), ("20-100us", 20, 100), ("0.1-1ms", 100, 1000), (">=1ms", 1000, 1e18)]},
           gaps_sum_by_bucket_ms={k: round(sum(g for g in gaps if lo <= g < hi) / 1e3, 2) for k, lo, hi in [("<5us", 0, 5), ("5-20us", 5, 20), ("20-100us", 20, 100), ("0.1-1ms", 100, 1000), (">=1ms", 1000, 1e18)]},
           biggest_gaps=[dict(us=round(g), after=a[:60], before=b[:60], cpu_starved=s) for g, a, b, s in sorted(big, reverse=True)[:12]])
# runtime API calls inside the GPU window (+ the host lead-in: first launch of the batch)
rc = collections.Counter(); rd = collections.Counter()
first_launch = min((r["ts"] for r in RT if "Launch" in r["name"]), default=t0)
for r in RT: rc[r["name"]] += 1; rd[r["name"]] += r["dur"]
out["runtime_calls"] = {n: dict(n=rc[n], ms=round(rd[n] / 1e3, 2)) for n, _ in rd.most_common(12)}
out["host_first_launch_to_gpu_end_ms"] = round((t1 - first_launch) / 1e3, 2)
# kernels
top = collections.Counter(); cnt = collections.Counter(); comp_t = collections.Counter(); comp_n = collections.Counter()
for e in K:
    n = re.sub(r"<.*", "", e["name"])[:90]; top[n] += e["dur"]; cnt[n] += 1; c = comp(e["name"]); comp_t[c] += e["dur"]; comp_n[c] += 1
out["top_kernels"] = [dict(ms=round(v / 1e3, 3), n=cnt[k], comp=comp(k), name=k) for k, v in top.most_common(TOP)]
out["components"] = {c: dict(ms=round(v / 1e3, 2), n=comp_n[c]) for c, v in comp_t.most_common()}
if "--kernels-out" in sys.argv:
    json.dump({k: dict(us=v, n=cnt[k]) for k, v in top.items()}, open(sys.argv[sys.argv.index("--kernels-out") + 1], "w"))
# python functions (only present with with_stack=True)
PY = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "python_function"]
if PY:
    keys = ["run_batch", "forward_batch_generation", "model_runner.py.*: forward$", "_forward_raw", "_execute_extend", "init_forward_metadata",
            "prepare_for_extend", "get_next_batch_to_run", "get_new_batch_prefill", "process_batch_result", "cache_unfinished_req",
            "cache_finished_req", "init_new", "recv_requests", "process_input_requests", "match_prefix", "resolve_future",
            "event_loop_overlap", "glm5_next.py.*: forward$", "sample", "logits_processor", "copy_to_cpu", "alloc", "mamba", "snapshot", "handle_generate_request", "add_one_req", "compute_position"]
    agg = collections.defaultdict(lambda: [0, 0])
    for e in PY:
        for k in keys:
            if re.search(k, e["name"]): agg[(k, re.sub(r"^.*/sglang/", "", e["name"])[:110])][0] += e["dur"]; agg[(k, re.sub(r"^.*/sglang/", "", e["name"])[:110])][1] += 1
    out["py_inclusive"] = [dict(ms=round(v[0] / 1e3, 2), n=v[1], fn=k[1]) for k, v in sorted(agg.items(), key=lambda kv: -kv[1][0])[:45]]
print(json.dumps(out, indent=1))
