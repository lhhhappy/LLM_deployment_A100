# Cost-model probe (~3-4 min, no restart): measures the two numbers every scheduling decision depends on.
#   1) extend cost T(c, P): one forward of c new tokens on top of a cached prefix of P tokens (engine otherwise idle).
#      Fit T = a_P + b_P * c per P -> a = fixed per-chunk overhead, b = per-token cost; b vs P = context-dependent part.
#   2) decode step d(bs): median inter-token gap with bs concurrent streams (8k cached prompts).
# Usage: python3 chunkcost.py <out_dir> [prefixes=0,32768,98304,180224] [chunks=512,1024,2048,4096,8192,16384]
import json, os, random, sys, threading, time, urllib.request
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
PS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0,32768,98304,180224").split(",")]
CS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "512,1024,2048,4096,8192,16384").split(",")]
BS = [1, 4, 8, 12, 16, 20, 24, 32]
base = f"http://127.0.0.1:{os.environ.get('PORT', '30000')}"
def post(path, body, timeout=3600):
    return urllib.request.urlopen(urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=timeout)
def flush():
    for _ in range(30):
        try: post("/flush_cache", {}).read(); return
        except Exception: time.sleep(2)
VMAX = int(os.environ.get("VOCAB_MAX", "150000"))  # dev-box surrogate model has vocab/8 -> set VOCAB_MAX=19000
def ids(n, seed): r = random.Random(seed); return [r.randrange(1000, VMAX) for _ in range(n)]
def gen(prompt, max_new=1):
    d = json.loads(post("/generate", {"input_ids": prompt, "sampling_params": {"max_new_tokens": max_new, "temperature": 0, "ignore_eos": True}}).read())
    m = d.get("meta_info", {})
    return (m.get("prefill_finished_time") or 0) - (m.get("request_received_ts") or 0), m.get("cached_tokens")
def fit(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / max(1e-9, sum((x - mx) ** 2 for x in xs)); return my - b * mx, b
res = {"extend": [], "fit": {}, "decode": []}
flush()
gen(ids(2048, 1))  # warm-up forward
seed = 10
for P in PS:
    prefix = ids(P, 500 + P) if P else []
    if P: gen(prefix)  # cache the prefix
    xs, ys = [], []
    for c in CS:
        best = None
        for rep in range(2):
            seed += 1
            if P: gen(prefix)  # re-touch so it stays cached
            t, cached = gen(prefix + ids(c, seed))
            best = t if best is None else min(best, t)
        row = dict(P=P, c=c, ms=round(1000 * best, 1), cached=cached, us_per_tok=round(1e6 * best / c, 1))
        res["extend"].append(row); print("EXTEND", json.dumps(row), flush=True)
        xs.append(c); ys.append(1000 * best)
    a, b = fit(xs, ys); res["fit"][P] = dict(fixed_ms=round(a, 1), us_per_tok=round(1000 * b, 1))
    print("FIT", P, res["fit"][P], flush=True)
    flush()
# decode step curve
prompts = [ids(8192, 9000 + i) for i in range(max(BS))]
for p in prompts: gen(p)
def stream(prompt, max_new, rec):
    times = []
    with post("/generate", {"input_ids": prompt, "stream": True, "sampling_params": {"max_new_tokens": max_new, "temperature": 0, "ignore_eos": True}}) as r:
        last = 0
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            c = json.loads(line[5:]).get("meta_info", {}).get("completion_tokens", 0)
            if c > last: times += [time.time()] * (c - last); last = c
    rec["times"] = times
for bs in BS:
    recs = [dict() for _ in range(bs)]
    ths = [threading.Thread(target=stream, args=(prompts[i] + ids(16, 70000 + 100 * bs + i), 160, recs[i])) for i in range(bs)]
    for t in ths: t.start()
    for t in ths: t.join()
    gaps = sorted(b - a for r in recs for a, b in list(zip(r["times"], r["times"][1:]))[20:])  # skip ramp-up
    row = dict(bs=bs, step_ms_p50=round(1000 * gaps[len(gaps) // 2], 2), step_ms_p90=round(1000 * gaps[int(.9 * len(gaps))], 2))
    res["decode"].append(row); print("DECODE", json.dumps(row), flush=True)
json.dump(res, open(f"{out}/chunkcost.json", "w"), indent=1)
print("CHUNKCOST", json.dumps({"fit": res["fit"], "decode": res["decode"]}), flush=True)
