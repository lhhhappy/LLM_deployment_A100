# Engine-side bottleneck summary from the SGLang server log (TP0 scheduler lines) + optional /metrics snapshot.
# Usage: python3 logstat.py <server.log> [port]
import re, sys, json, collections, urllib.request
L = sys.argv[1]; port = sys.argv[2] if len(sys.argv) > 2 else None
P = re.compile(r"\[(\S+ \S+) TP0\] Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+), full token usage: ([\d.]+), mamba usage: ([\d.]+), #running-req: (\d+), #queue-req: (\d+).*?input throughput \(token/s\): ([\d.]+)")
D = re.compile(r"\[(\S+ \S+) TP0\] Decode batch, #running-req: (\d+), #full token: (\d+), full token usage: ([\d.]+), mamba num: (\d+), mamba usage: ([\d.]+).*?gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)")
pre, dec, prev = [], collections.defaultdict(list), None
maxq = maxu = maxm = 0; kinds = collections.Counter(); ev = collections.Counter()
for line in open(L, errors="replace"):
    if "TP0]" not in line and "Triton kernel" not in line: continue
    if "compile after serving started" in line: ev["serving_compile"] += 1
    if re.search(r"retract", line, re.I): ev["retract"] += 1
    if re.search(r"Traceback|Scheduler hit an exception", line): ev["exception"] += 1
    m = P.search(line)
    if m:
        _, ns, nt, ct, fu, mu, run, q, thr = m.groups()
        pre.append(dict(ns=int(ns), new=int(nt), cached=int(ct), run=int(run), q=int(q), thr=float(thr)))
        maxq = max(maxq, int(q)); maxu = max(maxu, float(fu)); maxm = max(maxm, float(mu)); kinds["prefill"] += 1; prev = "P"; continue
    m = D.search(line)
    if m:
        _, run, ft, fu, mn, mu, thr, q = m.groups()
        if prev == "D": dec[int(run)].append(float(thr))
        maxq = max(maxq, int(q)); maxu = max(maxu, float(fu)); maxm = max(maxm, float(mu)); kinds["decode_log"] += 1; prev = "D"
def pct(v, p): v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))] if v else None
print(f"== {L}")
print(f"prefill batches={len(pre)}  decode log lines={kinds['decode_log']}  max queue={maxq}  max KV usage={maxu:.2f}  max KDA-state usage={maxm:.2f}  events={dict(ev)}")
if pre:
    tot_new = sum(p["new"] for p in pre); tot_cached = sum(p["cached"] for p in pre)
    print(f"prefill tokens: new={tot_new:,} cached(reported per batch)={tot_cached:,}  batches with >1 seq={sum(p['ns']>1 for p in pre)}")
    for lo, hi in ((1, 1024), (1024, 4096), (4096, 8192), (8192, 10**9)):
        s = [p["thr"] for p in pre if lo <= p["new"] < hi]
        if s: print(f"  new-token [{lo},{hi}): n={len(s):4d}  input tok/s p50={pct(s,.5):8.0f} p90={pct(s,.9):8.0f}")
    full = [p for p in pre if p["new"] >= 8192]
    for lo, hi in ((0, 1), (1, 32768), (32768, 98304), (98304, 10**9)):
        s = [p["thr"] for p in full if lo <= p["cached"] < hi]
        if s: print(f"  full 8192-chunks with cached in [{lo},{hi}): n={len(s):4d} tok/s p50={pct(s,.5):8.0f}  (=> {1000*8192/pct(s,.5):.0f} ms/chunk)")
    qs = [p["q"] for p in pre]; print(f"  queue at prefill: p50={pct(qs,.5)} p90={pct(qs,.9)} max={max(qs)}")
print("decode (consecutive decode lines only): ms/step = 1000*bs/throughput")
for bs in sorted(dec):
    v = dec[bs]; t = pct(v, .5)
    print(f"  bs={bs:3d} n={len(v):4d}  tok/s p50={t:7.1f}  => {1000*bs/t:6.1f} ms/step  ({1000/t*bs/bs if t else 0:5.1f} ms per token per req)")
if port:
    try:
        txt = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10).read().decode()
        keep = [l for l in txt.splitlines() if not l.startswith("#") and re.search(r"cache_hit_rate|num_running|num_queue|token_usage|_sum |_count |spec_accept", l)]
        print("metrics:"); [print("  " + l[:160]) for l in keep[:40]]
    except Exception as e: print("metrics unavailable:", e)
