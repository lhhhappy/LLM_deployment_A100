# Single-request cold prefill timing on the running engine: random token ids of given lengths via /generate
# (flush before each), TTFT from server meta_info; optional torch profile of one request (component table).
# Usage: python3 coldprobe.py <out_dir> <len1,len2,...> [profile_len]
import json, os, random, sys, time, urllib.request
out, lens = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
prof_len = int(sys.argv[3]) if len(sys.argv) > 3 else 0
port = os.environ.get("PORT", "30000"); base = f"http://127.0.0.1:{port}"
def post(path, body, timeout=1800):
    return urllib.request.urlopen(urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=timeout).read()
rows = []
for L in lens:
    post("/flush_cache", {}); rnd = random.Random(L)
    ids = [rnd.randrange(1000, 150000) for _ in range(L)]
    if L == prof_len:
        os.makedirs(f"{out}/prof_{L}", exist_ok=True)
        post("/start_profile", {"output_dir": f"{out}/prof_{L}", "num_steps": 3, "activities": ["GPU"]})
    t = time.time()
    r = json.loads(post("/generate", {"input_ids": ids, "sampling_params": {"max_new_tokens": 8, "temperature": 0, "ignore_eos": True}}))
    mi = r["meta_info"]; wall = time.time() - t
    ttft = (mi.get("prefill_finished_time") or 0) - (mi.get("request_received_ts") or 0)
    row = dict(len=L, ttft_s=round(ttft, 3), wall_s=round(wall, 3), prompt_tokens=mi.get("prompt_tokens"), cached=mi.get("cached_tokens"),
               first_ids=r.get("output_ids", [])[:4] if isinstance(r.get("output_ids"), list) else None)
    rows.append(row); print("COLDPROBE", json.dumps(row), flush=True)
json.dump(rows, open(f"{out}/coldprobe.json", "w"), indent=1)
