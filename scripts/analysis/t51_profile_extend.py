# T51: torch-profile ONE extend forward (c new tokens on a cached prefix of P) on a running sglang server.
# For each P: two traces — "nostack" (accurate timing) and "stack" (python call stacks for CPU attribution; inflated).
# Usage: python t51_profile_extend.py <out_root> [P list=98304,0] [c=1024]
import json, os, random, sys, time, urllib.request
out = sys.argv[1]; PS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "98304,0").split(",")]
C = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
base = f"http://127.0.0.1:{os.environ.get('PORT', '31000')}"
def post(path, body, timeout=3600):
    return urllib.request.urlopen(urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=timeout).read()
VMAX = int(os.environ.get("VOCAB_MAX", "19000"))
def ids(n, seed): r = random.Random(seed); return [r.randrange(1000, VMAX) for _ in range(n)]
def gen(prompt):
    m = json.loads(post("/generate", {"input_ids": prompt, "sampling_params": {"max_new_tokens": 1, "temperature": 0, "ignore_eos": True}})).get("meta_info", {})
    return round(1000 * ((m.get("prefill_finished_time") or 0) - (m.get("request_received_ts") or 0)), 1), m.get("cached_tokens")
summary = []
for P in PS:
    post("/flush_cache", {}); time.sleep(1)
    prefix = ids(P, 500 + P) if P else []
    gen(ids(2048, 1))
    if P: gen(prefix)
    for seed in (1, 2):  # unprofiled reference timings
        if P: gen(prefix)
        summary.append(dict(P=P, c=C, mode="plain", ms_cached=gen(prefix + ids(C, 7000 + seed))))
    for mode in os.environ.get("MODES", "nostack,stack").split(","):
        if P: gen(prefix)
        d = f"{out}/P{P}_c{C}_{mode}"; os.makedirs(d, exist_ok=True)
        post("/start_profile", {"output_dir": d, "activities": ["CPU", "GPU"], "with_stack": mode == "stack", "record_shapes": False, "profile_prefix": f"P{P}_{mode}"})
        time.sleep(0.5)
        r = gen(prefix + ids(C, 8000 + len(summary)))
        time.sleep(0.3)
        post("/stop_profile", {})
        summary.append(dict(P=P, c=C, mode=mode, ms_cached=r, dir=d)); print(json.dumps(summary[-1]), flush=True)
        time.sleep(3)
json.dump(summary, open(f"{out}/profile_summary.json", "w"), indent=1)
print("PROFILE_SUMMARY", json.dumps(summary))
