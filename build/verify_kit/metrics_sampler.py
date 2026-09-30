# Runtime monitor: every INTERVAL s append one JSON line of key /metrics gauges (running/queued reqs, token usage,
# throughput, cache hit rate, spec accept) to <out>. Runs until killed. Usage: python3 metrics_sampler.py <out.jsonl> [interval=10]
import json, os, re, sys, time, urllib.request
out = sys.argv[1]; iv = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
url = f"http://127.0.0.1:{os.environ.get('PORT', '30000')}/metrics"
KEEP = re.compile(r"^sglang:(num_running_reqs|num_queue_reqs|token_usage|gen_throughput|cache_hit_rate|spec_accept_length|"
                  r"num_used_tokens|mamba_usage|swa_token_usage|num_retracted_reqs|pending_prealloc_token_usage|"
                  r"full_token_usage|kv_available_tokens|kv_evictable_tokens|kv_used_tokens|"
                  r"mamba_available_tokens|mamba_evictable_tokens|mamba_used_tokens|evicted_tokens_total|"
                  r"eviction_duration_seconds_count|eviction_duration_seconds_sum|"
                  r"hicache_host_used_tokens|hicache_host_total_tokens|prefill_effective_tokens_total|"
                  r"load_back_tokens_total|load_back_bytes_total|"
                  r"load_back_duration_seconds_count|load_back_duration_seconds_sum|"
                  r"hicache_backup_tokens_total|hicache_backup_bytes_total|hicache_dropped_tokens_total|"
                  r"hicache_backup_duration_seconds_count|hicache_backup_duration_seconds_sum|"
                  r"prompt_tokens_total|generation_tokens_total|num_requests_total)(\{[^}]*\})? ([0-9.eE+-]+)", re.M)
with open(out, "a") as f:
    while True:
        try:
            txt = urllib.request.urlopen(url, timeout=5).read().decode(); row = {"t": round(time.time(), 1)}
            samples = KEEP.findall(txt)
            for k, labels, v in samples:
                # New tier/pool counters stay labeled: KDA slots and KV tokens
                # must not be added into one flat "token" number.
                if k in {"prefill_effective_tokens_total", "load_back_tokens_total",
                         "hicache_backup_tokens_total", "hicache_dropped_tokens_total"}: continue
                row[k] = round(row.get(k, 0) + float(v), 4) if k.endswith("_total") else float(v)
            # Preserve labels for per-pool analysis; legacy flat fields remain compatible.
            # Missing metrics stay absent, never interpreted as zero evictions.
            row["samples"] = [{"name": k, "labels": labels, "value": float(v)} for k, labels, v in samples]
            f.write(json.dumps(row) + "\n"); f.flush()
        except Exception as e: f.write(json.dumps({"t": round(time.time(), 1), "err": str(e)[:60]}) + "\n"); f.flush()
        time.sleep(iv)
