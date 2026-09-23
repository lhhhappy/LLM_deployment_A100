# Compare ladder runs per phase: n, TTFT p50/p95, share of requests that lost expected prefix cache (cached < 50% of
# expected), mean uncached tokens actually prefilled vs frozen expectation. Usage: cachecmp.py <raw.jsonl> [...]
import json, sys
def q(v, p): v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")
for f in sys.argv[1:]:
    rows = []
    for l in open(f):
        l = l.strip()
        if l.startswith("{"):
            try: rows.append(json.loads(l))
            except Exception: pass
    if not rows: print(f, "no rows"); continue
    if f is sys.argv[1]: print("fields:", sorted(rows[0].keys())[:60])
    ph = {}
    for r in rows: ph.setdefault(r.get("phase") or r.get("bucket") or "?", []).append(r)
    print("==", f.split("/runs/")[-1][:60], "rows", len(rows))
    for k, rs in sorted(ph.items()):
        tt = [r.get("ttft_s") or r.get("ttft") or 0 for r in rs]
        lost = act = exp = 0
        for r in rs:
            pt = r.get("prompt_tokens") or 0; ue = r.get("uncached_expected"); ct = r.get("cached_tokens")
            if ue is None or ct is None: continue
            e = max(0, pt - ue); act += pt - ct; exp += ue
            if e > 1000 and ct < 0.5 * e: lost += 1
        print(f"  {k:14s} n={len(rs):4d} ttft p50={q(tt,.5):6.2f} p95={q(tt,.95):6.2f}  lost_cache={lost:3d}  prefilled/expected={act/max(1,exp):.2f}")
