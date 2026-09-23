# Per-request system analysis of one harness run (beyond the gate verdict). Uses the harness's own gate bucketing.
# Usage: python3 analyze_run.py <harness_dir> <raw_*.jsonl> [out.json]
# Sections: gates (p95, #over limit, allowed-over at the 95% lower-bound rule), TTFT decomposition
# (queue = exec_start - recv, prefill = ttft - (exec_start - recv)), cache efficiency (actual cached vs frozen expected),
# prefill rate vs uncached size, TPOT distribution, worst requests.
import json, math, sys, collections
sys.path.insert(0, sys.argv[1]); import s1_common as C
rows = [json.loads(l) for l in open(sys.argv[2])]
ok = [r for r in rows if r.get("ttft_s") is not None and not r.get("error")]
def q(v, p): v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")
def allowed_over(n, p=0.05):
    # largest k such that the one-sided 95% Clopper-Pearson lower bound of k/n is still <= p
    from math import comb
    def lower(k):
        if k == 0: return 0.0
        lo, hi = 0.0, 1.0
        for _ in range(60):
            m = (lo + hi) / 2; tail = sum(comb(n, i) * m**i * (1 - m)**(n - i) for i in range(k, n + 1))
            lo, hi = (m, hi) if tail < 0.05 else (lo, m)
        return lo
    k = 0
    while k < n and lower(k + 1) <= p: k += 1
    return k
LIM = {"fast_intra": 3.0, "overall_intra": 5.0, "turn_start": 15.0, "chain_start": 30.0}
print(f"requests={len(rows)} ok={len(ok)} errors={len(rows) - len(ok)}")
print("\n== gates  (dev harness = hard p95<=limit; formal-est = task.md rule: FAIL only if 95% lower bound of exceed-rate > 5%)")
verdict = {"gates": {}, "errors": len(rows) - len(ok)}
for g, lim in LIM.items():
    rs = [r for r in ok if C.in_ttft_gate(r, g)]; t = [r["ttft_s"] for r in rs]; over = sum(x > lim for x in t)
    ao = allowed_over(len(rs)) if rs else 0; p95 = q(t, .95) if t else 0.0
    hard = (not t) or p95 <= lim; formal = over <= ao
    verdict["gates"][g] = dict(n=len(rs), p95=round(p95, 3), over=over, allowed_over=ao, harness_pass=hard, formal_est_pass=formal)
    print(f"  {g:14s} n={len(rs):4d} p50={q(t,.5):6.2f} p95={p95:6.2f} max={max(t) if t else 0:6.1f} over={over:3d} allowed={ao:3d} | harness {'PASS' if hard else 'FAIL'} | formal-est {'PASS' if formal else 'FAIL'} (margin {ao - over:+d})")
tp_ = sorted(r["tpot_s"] for r in ok if r.get("tpot_s"))
verdict["tpot_mean"] = sum(tp_) / max(1, len(tp_)); verdict["tpot_p95"] = q(tp_, .95) if tp_ else 0
err_ok = verdict["errors"] <= 0.01 * len(rows) and verdict["tpot_p95"] <= 0.10
verdict["harness_all_pass"] = err_ok and all(v["harness_pass"] for v in verdict["gates"].values())
verdict["formal_est_all_pass"] = err_ok and all(v["formal_est_pass"] for v in verdict["gates"].values())
print(f"  => harness ALL_PASS={verdict['harness_all_pass']}  formal-est ALL_PASS={verdict['formal_est_all_pass']}  tpot_mean={verdict['tpot_mean']:.4f} tpot_p95={verdict['tpot_p95']:.4f}")
print("\n== TTFT decomposition (server timestamps)")
dec = []
for r in ok:
    if r.get("t_recv_s") and r.get("t_exec_start_s"):
        qw = max(0.0, r["t_exec_start_s"] - r["t_recv_s"]); dec.append((r, qw, max(0.0, r["ttft_s"] - qw)))
for g in LIM:
    d = [(qw, pf) for r, qw, pf in dec if C.in_ttft_gate(r, g)]
    if d: print(f"  {g:14s} queue p50={q([a for a,_ in d],.5):6.2f} p95={q([a for a,_ in d],.95):6.2f} | exec->first p50={q([b for _,b in d],.5):6.2f} p95={q([b for _,b in d],.95):6.2f}")
print("\n== cache efficiency (actual cached vs frozen expectation)")
tot_exp = sum(max(0, r["prompt_tokens"] - (r.get("uncached_expected") or 0)) for r in ok if r.get("prompt_tokens"))
tot_act = sum(r.get("cached_tokens") or 0 for r in ok)
lost = [(r, max(0, r["prompt_tokens"] - (r.get("uncached_expected") or 0)) - (r.get("cached_tokens") or 0)) for r in ok if r.get("prompt_tokens")]
print(f"  expected cached={tot_exp:,}  actual cached={tot_act:,}  ratio={tot_act / max(1, tot_exp):.3f}")
big = sorted(lost, key=lambda x: -x[1])[:8]
print(f"  requests with actual cached < expected-1024: {sum(1 for _, l in lost if l > 1024)}; lost tokens total={sum(max(0,l) for _, l in lost):,}")
for r, l in big[:5]: print(f"    lost={l:7d} prompt={r['prompt_tokens']:7d} cached={r.get('cached_tokens')} exp_uncached={r.get('uncached_expected')} phase={r.get('phase')} edge={r.get('edge_type')} ttft={r['ttft_s']:.2f}")
print("\n== prefill rate vs actual uncached tokens (exec->first token)")
for lo, hi in ((0, 1024), (1024, 4096), (4096, 16384), (16384, 65536), (65536, 10**9)):
    d = [(r["prompt_tokens"] - (r.get("cached_tokens") or 0)) / pf for r, qw, pf in dec if lo <= r["prompt_tokens"] - (r.get("cached_tokens") or 0) < hi and pf > 0.05]
    pfs = [pf for r, qw, pf in dec if lo <= r["prompt_tokens"] - (r.get("cached_tokens") or 0) < hi]
    if pfs: print(f"  uncached [{lo:6d},{hi:9d}) n={len(pfs):4d} exec->first p50={q(pfs,.5):6.2f}s p95={q(pfs,.95):6.2f}s  eff tok/s p50={q(d,.5):8.0f}")
print("\n== TPOT")
tp = [r["tpot_s"] for r in ok if r.get("tpot_s")]
print(f"  n={len(tp)} mean={sum(tp)/max(1,len(tp)):.4f} p50={q(tp,.5):.4f} p95={q(tp,.95):.4f} max={max(tp) if tp else 0:.3f}")
print("\n== worst requests vs their gate")
worst = sorted(ok, key=lambda r: -r["ttft_s"] / C.phase_gate(r))[:10]
for r in worst:
    qw = (r.get("t_exec_start_s") or 0) - (r.get("t_recv_s") or 0)
    print(f"  ttft={r['ttft_s']:6.1f}/{C.phase_gate(r):4.0f}s queue={qw:6.1f} prompt={r['prompt_tokens']:7d} cached={r.get('cached_tokens'):7d} exp_unc={r.get('uncached_expected'):7d} phase={r.get('phase')} idx={r.get('idx_in_chain')} edge={r.get('edge_type')}")
if len(sys.argv) > 3: json.dump(verdict, open(sys.argv[3], "w"), indent=1)
