#!/usr/bin/env python3
"""M0 probe: /generate with prompts of 500, 1500, 3000, 9000 and 20000 tokens (crossing the DSA
top-k=2048 threshold and the 8192 chunk size), streaming like the harness; then /flush_cache.
Records per request: HTTP status, first-token latency, token counts, error text."""
import argparse
import json
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("--port", type=int, required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
base = f"http://127.0.0.1:{a.port}"
# Token ids are irrelevant for kernel paths; send input_ids so the lengths are exact.
results = []
for n in (500, 1500, 3000, 9000, 20000):
    body = {"input_ids": [(i * 7919) % 150000 + 100 for i in range(n)],
            "sampling_params": {"max_new_tokens": 16, "temperature": 0, "ignore_eos": True},
            "stream": True, "rid": f"m0-{n}"}
    req = urllib.request.Request(base + "/generate", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); first = None; last = None; err = None; status = None
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            status = r.status
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                d = json.loads(line[5:])
                if "error" in d:
                    err = d["error"]; break
                first = first or time.time()
                last = d
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    meta = (last or {}).get("meta_info", {})
    res = {"prompt_tokens": n, "status": status, "ttft_s": round(first - t0, 3) if first else None,
           "completion_tokens": meta.get("completion_tokens"), "cached_tokens": meta.get("cached_tokens"),
           "error": err}
    results.append(res); print(json.dumps(res), flush=True)
try:
    with urllib.request.urlopen(urllib.request.Request(base + "/flush_cache", data=b"", method="POST"), timeout=60) as r:
        flush = {"status": r.status, "body": r.read().decode()[:200]}
except Exception as e:
    flush = {"error": f"{type(e).__name__}: {e}"}
print(json.dumps({"flush_cache": flush}), flush=True)
json.dump({"results": results, "flush": flush}, open(a.out + "/probe.json", "w"), indent=1)
ok = all(r["error"] is None and r["completion_tokens"] for r in results)
raise SystemExit(0 if ok else 1)
