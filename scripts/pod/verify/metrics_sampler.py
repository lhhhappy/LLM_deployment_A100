# Runtime monitor: every INTERVAL s append one JSON line of key /metrics gauges (running/queued reqs, token usage,
# throughput, cache hit rate, spec accept) to <out>. Runs until killed. Usage: python3 metrics_sampler.py <out.jsonl> [interval=10]
import json, os, re, sys, time, urllib.request
out = sys.argv[1]; iv = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
url = f"http://127.0.0.1:{os.environ.get('PORT', '30000')}/metrics"
KEEP = re.compile(r"^sglang:(num_running_reqs|num_queue_reqs|token_usage|gen_throughput|cache_hit_rate|spec_accept_length|"
                  r"num_used_tokens|mamba_usage|swa_token_usage|num_retracted_reqs|pending_prealloc_token_usage|"
                  r"prompt_tokens_total|generation_tokens_total|num_requests_total)\{[^}]*\} ([0-9.eE+-]+)", re.M)
with open(out, "a") as f:
    while True:
        try:
            txt = urllib.request.urlopen(url, timeout=5).read().decode(); row = {"t": round(time.time(), 1)}
            for k, v in KEEP.findall(txt): row[k] = round(row.get(k, 0) + float(v), 4) if k.endswith("_total") else float(v)
            f.write(json.dumps(row) + "\n"); f.flush()
        except Exception as e: f.write(json.dumps({"t": round(time.time(), 1), "err": str(e)[:60]}) + "\n"); f.flush()
        time.sleep(iv)
