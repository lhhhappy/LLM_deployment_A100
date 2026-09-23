# Interference probe — the hard core of the workload in ~2-3 minutes:
#   D decode streams are generating (cached prompts), then ONE long cold prefill arrives, and while it runs a short
#   cache-hit request arrives every second. Measures exactly what fails the gates:
#     - decode streams: inter-token gaps during the cold prefill (p50/p95/max) and per-stream TPOT  -> tpot gate / tpot_mean
#     - short cache-hit requests: TTFT (server timestamps)                                          -> fast/overall_intra
#     - the cold request itself: TTFT                                                               -> chain_start
# Usage: python3 interference.py <out_dir> [decode_streams=12] [cold_len=190000] [short_new=1500]
import json, os, random, sys, threading, time, urllib.request
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
D = int(sys.argv[2]) if len(sys.argv) > 2 else 12
COLD = int(sys.argv[3]) if len(sys.argv) > 3 else 190000
SHORT_NEW = int(sys.argv[4]) if len(sys.argv) > 4 else 1500
base = f"http://127.0.0.1:{os.environ.get('PORT', '30000')}"
def post(path, body, timeout=3600):
    return urllib.request.urlopen(urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=timeout)
def flush():
    for _ in range(30):
        try: post("/flush_cache", {}).read(); return
        except Exception: time.sleep(2)
def ids(n, seed): r = random.Random(seed); return [r.randrange(1000, 150000) for _ in range(n)]
def stream(prompt_ids, max_new, rec):
    t0 = time.time(); times = []; meta = {}
    with post("/generate", {"input_ids": prompt_ids, "stream": True,
                            "sampling_params": {"max_new_tokens": max_new, "temperature": 0, "ignore_eos": True}}) as r:
        last = 0
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            d = json.loads(line[5:]); mi = d.get("meta_info", {}); c = mi.get("completion_tokens", 0)
            if c > last: times += [time.time()] * (c - last); last = c; meta = mi
    rec.update(t0=t0, times=times, meta=meta)
flush()
# warm prefixes for the decode streams and the short requests (so they are cache hits)
prefixes = [ids(4000, 1000 + i) for i in range(D)]
short_prefix = ids(30000, 7)
for p in prefixes + [short_prefix]:
    post("/generate", {"input_ids": p, "sampling_params": {"max_new_tokens": 1, "temperature": 0}}).read()
# 1) start decode streams
recs = [dict() for _ in range(D)]
ths = [threading.Thread(target=stream, args=(prefixes[i] + ids(64, 2000 + i), 3000, recs[i])) for i in range(D)]
for t in ths: t.start()
time.sleep(4)
# 2) cold long prefill
cold = {}; tc = threading.Thread(target=stream, args=(ids(COLD, 99), 16, cold)); tc.start(); cold_start = time.time()
# 3) short cache-hit requests every second while the cold prefill runs
shorts = []
k = 0
while tc.is_alive() and k < 120:
    rec = {}; th = threading.Thread(target=stream, args=(short_prefix + ids(SHORT_NEW, 3000 + k), 8, rec)); th.start(); shorts.append((rec, th)); k += 1
    time.sleep(1.0)
tc.join(); cold_end = time.time()
for _, th in shorts: th.join()
for t in ths: t.join()
def q(v, p): v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")
gaps_during, gaps_before, tpots = [], [], []
for r in recs:
    tt = r.get("times", [])
    if len(tt) > 2: tpots.append((tt[-1] - tt[0]) / (len(tt) - 1))
    gaps_during += [b - a for a, b in zip(tt, tt[1:]) if cold_start <= a <= cold_end]
    gaps_before += [b - a for a, b in zip(tt, tt[1:]) if a < cold_start]  # pure decode, no prefill interference
def ttft(r): m = r.get("meta", {}); return (m.get("prefill_finished_time") or 0) - (m.get("request_received_ts") or 0)
st = [ttft(r) for r, _ in shorts if r.get("meta")]
res = dict(decode_streams=D, cold_len=COLD, cold_ttft=round(ttft(cold), 2), cold_wall=round(cold_end - cold_start, 2),
           decode_gap_before_cold_ms=dict(p50=round(1000 * q(gaps_before, .5), 1), p95=round(1000 * q(gaps_before, .95), 1)),
           decode_gap_during_cold_ms=dict(p50=round(1000 * q(gaps_during, .5), 1), p95=round(1000 * q(gaps_during, .95), 1), max=round(1000 * max(gaps_during or [0]), 1)),
           stream_tpot=dict(mean=round(sum(tpots) / max(1, len(tpots)), 4), max=round(max(tpots or [0]), 4)),
           short_hit_ttft=dict(n=len(st), p50=round(q(st, .5), 2), p95=round(q(st, .95), 2), max=round(max(st or [0]), 2)))
try:  # speculative decoding stats if any (MTP)
    import re as _re
    txt = urllib.request.urlopen(base + "/metrics", timeout=10).read().decode()
    res["spec"] = {k: float(v) for k, v in _re.findall(r'^sglang:(spec_accept_(?:length|rate))\{[^}]*\} ([0-9.eE+-]+)', txt, _re.M)}
except Exception as e: res["spec"] = str(e)[:80]
json.dump(res, open(f"{out}/interference.json", "w"), indent=1)
print("INTERFERENCE", json.dumps(res), flush=True)
