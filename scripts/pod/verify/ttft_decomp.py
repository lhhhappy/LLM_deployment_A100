# Decompose TTFT per phase into queue wait (recv->exec_start) and execution (exec_start->first token), for all
# requests and for the ones over their gate; plus the time profile of load. Usage: ttft_decomp.py <raw.jsonl>
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.startswith("{")]
LIM = {"intra": 5, "turn_start": 15, "session_start": 30, "context_reset": 30}
def q(v, p): v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")
print(f"rows={len(rows)} wall={max(r['t_first_token_s'] or 0 for r in rows) - min(r['t_recv_s'] for r in rows):.0f}s")
for ph in sorted({r["phase"] for r in rows}):
    rs = [r for r in rows if r["phase"] == ph and r.get("t_exec_start_s") and r.get("t_first_token_s")]
    wq = [r["t_exec_start_s"] - r["t_recv_s"] for r in rs]; ex = [r["t_first_token_s"] - r["t_exec_start_s"] for r in rs]
    over = [r for r in rs if r["ttft_s"] > LIM.get(ph, 30)]
    oq = sum(r["t_exec_start_s"] - r["t_recv_s"] for r in over); oe = sum(r["t_first_token_s"] - r["t_exec_start_s"] for r in over)
    print(f"  {ph:13s} n={len(rs):3d} queue p50/p95={q(wq,.5):5.2f}/{q(wq,.95):6.2f}s exec p50/p95={q(ex,.5):5.2f}/{q(ex,.95):6.2f}s | over-limit {len(over):3d}: time share queue={oq/max(1e-9,oq+oe):.0%}")
# load timeline: requests arriving per minute and prompt tokens arriving per minute
t0 = min(r["t_recv_s"] for r in rows); b = {}
for r in rows: m = int((r["t_recv_s"] - t0) // 60); b.setdefault(m, [0, 0]); b[m][0] += 1; b[m][1] += r["prompt_tokens"] - (r.get("cached_tokens") or 0)
print("  per-minute arrivals / uncached-prefill Mtok:", " ".join(f"{m}:{v[0]}/{v[1]/1e6:.2f}" for m, v in sorted(b.items())))
