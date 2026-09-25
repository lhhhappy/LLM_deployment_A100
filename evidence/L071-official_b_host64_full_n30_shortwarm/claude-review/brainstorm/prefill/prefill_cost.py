#!/usr/bin/env python3
"""Prefill cost / lane occupancy / zero-wait floor for one N30 run (CPU only, stdlib only).

Inputs: events CSV from parse_log.py (TP0 Prefill/Decode lines with exact interval gap_s) and the run's raw jsonl.
Method (all timing from the server's own perf_counter intervals, not the 1 s log stamps):
 1. Exact timeline: cumulative gap_s per series, offset pinned so every event lies in [stamp, stamp+1).
    Checked: order of the merged P/D series equals the log line order (0 inversions) and raw t_first_token_s
    matches the P event of the finishing batch (median 4 ms).
 2. A Decode event closes exactly 40 decode forwards; decode windows (between consecutive D events) contain
    40 decode steps plus the prefill batches whose P event falls inside. No forward straddles a D event.
 3. Decode step model d = a + b*bs + c*ctx_dec (bs = spec_rounds/40 exact mean; ctx_dec = sum of prompt+generated
    tokens of requests decoding at the window midpoint, from raw) fitted on windows with no prefill.
 4. Prefill time per window = W - 40*d_hat. Prefill batch cost model f = alpha + beta*T + gamma*T*ctx + delta*(seqs-1),
    fitted by least squares on those residuals (T = new tokens, ctx = context before the tokens; chunked request
    depth from raw cached + earlier shares, short requests' ctx = their cached tokens).
Usage: parse_log.py RUN_DIR events_TAG.csv; prefill_cost.py events_TAG.csv RAW_JSONL OUT_DIR TAG (events CSVs are regenerated, not kept)
"""
import bisect, collections, csv, json, math, sys

sys.dont_write_bytecode = True
sys.path.insert(0, "/workspace/Agentic_science_challenge/s1-dev/harness")
from s1_common import in_ttft_gate  # original harness bucket selector (read-only import)

EV, RAW, OUT, TAG = sys.argv[1:5]
MEAS_T0 = None  # measurement start = first raw dispatch


def fl(v):
    return None if v in ("", None) else float(v)


def lstsq(X, y, w=None):
    n = len(X[0]); A = [[0.0] * n for _ in range(n)]; b = [0.0] * n
    for i, row in enumerate(X):
        wi = 1.0 if w is None else w[i]
        for p in range(n):
            b[p] += wi * row[p] * y[i]
            for q in range(n):
                A[p][q] += wi * row[p] * row[q]
    M = [A[i] + [b[i]] for i in range(n)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c])); M[c], M[piv] = M[piv], M[c]
        for r in range(n):
            if r != c and M[c][c] != 0:
                f = M[r][c] / M[c][c]
                M[r] = [M[r][k] - f * M[c][k] for k in range(n + 1)]
    return [M[i][n] / M[i][i] for i in range(n)]


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else None


# ---------- 1. timeline ----------
ev = list(csv.DictReader(open(EV)))
for x in ev:
    for k in list(x):
        if k != "kind":
            x[k] = fl(x[k])
calib = {}
for kind in "PD":
    s = [x for x in ev if x["kind"] == kind]
    cum, lo, hi = 0.0, -1e18, 1e18
    for i, x in enumerate(s):
        if i:
            cum += x["gap_s"]
        x["cum"] = cum
        lo, hi = max(lo, x["ts"] - cum), min(hi, x["ts"] + 1 - cum)
    off = (lo + hi) / 2
    for x in s:
        x["t"] = off + x["cum"]
    calib[kind] = {"offset_window_s": hi - lo, "n": len(s)}
ev.sort(key=lambda x: x["line"])
calib["order_inversions_vs_line"] = sum(1 for a, b in zip(ev, ev[1:]) if a["t"] > b["t"] + 1e-6)

R = [json.loads(l) for l in open(RAW)]
R_by = {r["req_id"]: r for r in R}
MEAS_T0 = min(r["client_dispatch_at_s"] for r in R) - 1.0
MEAS_T1 = max(r["client_finish_at_s"] for r in R) + 1.0
ev = [x for x in ev if MEAS_T0 <= x["t"] <= MEAS_T1]
P = [x for x in ev if x["kind"] == "P"]
Pt = [x["t"] for x in P]
for r in R:
    r["unc"] = r["prompt_tokens"] - r["cached_tokens"]
    r["gate"] = next((g for g in ("fast_intra", "turn_start", "chain_start") if in_ttft_gate(r, g)),
                     "overall_intra" if in_ttft_gate(r, "overall_intra") else None)

# ---------- 2. request <-> prefill batch matching ----------
lag = []
for r in R:  # finishing batch: first P event at/after t_first_token (post-processing lag grows with prompt size)
    i = bisect.bisect_left(Pt, r["t_first_token_s"] - 0.005)
    r["fin"] = i if i < len(P) and Pt[i] - r["t_first_token_s"] < 0.5 else None
    if r["fin"] is not None:
        lag.append((r["prompt_tokens"], Pt[i] - r["t_first_token_s"]))
calib["unmatched_first_token"] = sum(1 for r in R if r["fin"] is None)
comp = collections.defaultdict(list)
for r in R:
    if r["fin"] is not None:
        comp[r["fin"]].append(r)
owner, conflicts = {}, 0
for r in R:
    if r["fin"] is None:
        continue
    i = bisect.bisect_right(Pt, r["t_exec_start_s"])
    if r["fin"] > i:  # multi-batch request: every batch in its window carries one of its chunks
        for k in range(i, r["fin"]):
            conflicts += k in owner
            owner[k] = r
calib["owner_conflicts"] = conflicts
depth_state = {}
for k, x in enumerate(P):
    r = owner.get(k)
    fin_here = comp.get(k, [])
    others = [q for q in fin_here if q is not r and owner.get(k) is not q]
    # the chunked request that finishes in batch k is its own last chunk
    last = [q for q in fin_here if q["req_id"] in depth_state]
    chunk_req = r if r is not None else (last[0] if last else None)
    others = [q for q in fin_here if q is not chunk_req]
    S = sum(math.ceil(q["unc"] / 64) * 64 for q in others)
    tctx = sum(math.ceil(q["unc"] / 64) * 64 * q["cached_tokens"] for q in others)
    x["chunk_req"], x["share"], x["depth"] = None, None, None
    if chunk_req is not None:
        d0 = depth_state.setdefault(chunk_req["req_id"], chunk_req["cached_tokens"])
        share = max(0.0, x["new_token"] - S)
        x["chunk_req"], x["share"], x["depth"] = chunk_req, share, d0
        depth_state[chunk_req["req_id"]] = d0 + share
        tctx += share * d0
    x["tctx"] = tctx
    x["long_cont"] = chunk_req is not None
# requests finishing in one batch whose owner is not set: their own batch
for k, x in enumerate(P):
    x["nD_before"] = 0
nd = 0
for x in ev:
    if x["kind"] == "D":
        nd += 1
    else:
        x["nD_before"], nd = nd, 0

# ---------- 3. decode step model ----------
dec_iv = []  # (start,end,prompt+max_out) of each request's decode phase
for r in R:
    dec_iv.append((r["t_first_token_s"], r["t_first_token_s"] + (r["output_tokens"] - 1) * (r["tpot_s"] or 0),
                   r["prompt_tokens"], r["output_tokens"]))
dec_iv.sort()
starts = [a[0] for a in dec_iv]


def ctx_dec(t):
    s = 0.0
    j = bisect.bisect_right(starts, t)
    for a, b, p, o in dec_iv[max(0, j - 400):j]:
        if a <= t <= b:
            s += p + o * (t - a) / max(b - a, 1e-6)
    return s


D = [x for x in ev if x["kind"] == "D"]
idx = {id(x): i for i, x in enumerate(ev)}
win = []
for a, b in zip(D, D[1:]):
    inside = [x for x in ev[idx[id(a)] + 1: idx[id(b)]] if x["kind"] == "P"]
    W = b["t"] - a["t"]
    bs = b["spec_rounds"] / 40.0
    win.append(dict(t0=a["t"], t1=b["t"], W=W, bs=bs, ctx=ctx_dec((a["t"] + b["t"]) / 2), P=inside,
                    acc=b["accept_len"]))
pure = [w for w in win if not w["P"]]
X = [[1.0, w["bs"], w["ctx"] / 1e6] for w in pure]
y = [w["W"] / 40 for w in pure]
dm = lstsq(X, y)
res = sorted(abs(y[i] - sum(p * q for p, q in zip(dm, X[i]))) / y[i] for i in range(len(y)))
dmodel = {"a_s": dm[0], "b_s_per_req": dm[1], "c_s_per_Mtok_ctx": dm[2], "n_windows": len(pure),
          "rel_abs_err_p50": pct(res, .5), "rel_abs_err_p90": pct(res, .9)}


def dhat(bs, ctx):
    return dm[0] + dm[1] * bs + dm[2] * ctx / 1e6


# ---------- 4. prefill cost model ----------
Xp, yp = [], []
for w in win:
    if not w["P"] or w["W"] > 30:  # very long windows can hold idle time
        continue
    resid = w["W"] - 40 * dhat(w["bs"], w["ctx"])
    w["pref_s"] = resid
    n = len(w["P"]); T = sum(x["new_token"] for x in w["P"]); TC = sum(x["tctx"] for x in w["P"])
    ns = sum(x["new_seq"] - 1 for x in w["P"])
    Xp.append([n, T / 1e4, TC / 1e9, ns]); yp.append(resid)
pm = lstsq(Xp, yp)
pres = [yp[i] - sum(p * q for p, q in zip(pm, Xp[i])) for i in range(len(yp))]
pmodel = {"alpha_s_per_batch": pm[0], "beta_s_per_10k_tok": pm[1], "gamma_s_per_1e9_tok_ctx": pm[2],
          "delta_s_per_extra_seq": pm[3], "n_windows": len(yp),
          "resid_p10": pct(pres, .1), "resid_p50": pct(pres, .5), "resid_p90": pct(pres, .9)}


def fhat(T, ctx, extra_seqs=0):
    return pm[0] + pm[1] * T / 1e4 + pm[2] * T * ctx / 1e9 + pm[3] * extra_seqs


# ---------- 5. chunk cost curve (Q1) ----------
chunk_rows = []
for k, x in enumerate(P):
    if k == 0 or not x["long_cont"]:
        continue
    chunk_rows.append(dict(t=x["t"], line=int(x["line"]), req=x["chunk_req"]["req_id"], depth=x["depth"],
                           share=x["share"], new_token=x["new_token"], new_seq=int(x["new_seq"]),
                           running=int(x["running"]), queue=int(x["queue"]), gap_s=x["gap_s"],
                           nD_logs_before=x["nD_before"], f_model=fhat(x["new_token"], x["depth"], x["new_seq"] - 1)))
with open(f"{OUT}/chunks_{TAG}.csv", "w", newline="") as f:
    w_ = csv.DictWriter(f, fieldnames=list(chunk_rows[0])); w_.writeheader(); w_.writerows(chunk_rows)
curve = []
for lo in range(0, 262144, 32768):
    for rlo, rhi in ((0, 7), (8, 15), (16, 23), (24, 40)):
        g = [c["gap_s"] for c in chunk_rows if c["new_seq"] == 1 and c["share"] == 8192
             and lo <= c["depth"] < lo + 32768 and rlo <= c["running"] <= rhi]
        if len(g) >= 5:
            curve.append(dict(depth_k=f"{lo // 1024}-{(lo + 32768) // 1024}", running=f"{rlo}-{rhi}", n=len(g),
                              gap_p10=pct(g, .1), gap_p50=pct(g, .5), gap_mean=sum(g) / len(g), gap_p90=pct(g, .9),
                              f_model_8192=fhat(8192, lo + 16384)))
# per-request exec->first-token vs pure model (interleaving overhead)
req_rows = []
for r in R:
    if r["fin"] is None or r["unc"] <= 8192:
        continue
    n_ch = math.ceil(r["unc"] / 8192)
    fpure = sum(fhat(min(8192, r["unc"] - i * 8192), r["cached_tokens"] + i * 8192) for i in range(n_ch))
    req_rows.append(dict(req=r["req_id"], gate=r["gate"], unc=r["unc"], cached=r["cached_tokens"],
                         exec_to_first=r["t_first_token_s"] - r["t_exec_start_s"], f_pure_model=fpure,
                         n_chunks=n_ch))

# ---------- 6. occupancy (Q2) ----------
# per decode window: prefill seconds = W - 40*d_hat (clip >= 0), split long-continuation vs other by model share
occ = []  # (t0,t1,pref_s,cont_s,decode_s)
for w in win:
    if w["W"] > 30:
        continue
    dec_s = 40 * dhat(w["bs"], w["ctx"])
    pref = max(0.0, w["W"] - dec_s)
    fm = [fhat(x["new_token"], (x["tctx"] / x["new_token"]) if x["new_token"] else 0, x["new_seq"] - 1)
          for x in w["P"]]
    cont = sum(f for f, x in zip(fm, w["P"]) if x["long_cont"])
    tot = sum(fm) or 1.0
    occ.append((w["t0"], w["t1"], pref, pref * cont / tot, min(dec_s, w["W"])))
span = sum(o[1] - o[0] for o in occ)
lane = {"covered_s": span, "prefill_s": sum(o[2] for o in occ), "long_cont_s": sum(o[3] for o in occ),
        "decode_s": sum(o[4] for o in occ)}
lane["prefill_frac"] = lane["prefill_s"] / span
lane["long_cont_frac"] = lane["long_cont_s"] / span
lane["uncovered_note"] = "windows >30 s (no decoders / idle) excluded"
lane["excluded_long_windows_s"] = sum(w["W"] for w in win if w["W"] > 30)
# model-based prefill seconds (sum f_hat over all P events) as a cross-check
lane["prefill_s_model_all_batches"] = sum(fhat(x["new_token"], (x["tctx"] / x["new_token"]) if x["new_token"] else 0,
                                               x["new_seq"] - 1) for x in P)
lane["wall_s_measure"] = MEAS_T1 - MEAS_T0

cs = [r for r in R if r["gate"] == "chain_start"]
cs_bad = sorted(cs, key=lambda r: -r["ttft_s"])
lim = 30.0
bad = [r for r in cs if r["ttft_s"] > lim]


def occ_during(intervals):
    tot = pre = cont = 0.0
    for t0, t1, pref, cs_, dec in occ:
        L = t1 - t0
        if L <= 0:
            continue
        ov = sum(max(0.0, min(t1, b) - max(t0, a)) for a, b in intervals)
        ov = min(ov, L)
        tot += ov; pre += pref * ov / L; cont += cs_ * ov / L
    return tot, pre, cont


def merge(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


wait_iv = merge([(r["t_recv_s"], r["t_exec_start_s"]) for r in bad if r["t_exec_start_s"] > r["t_recv_s"]])
wt, wp, wc = occ_during(wait_iv)
lane_bad = {"union_wait_s": sum(b - a for a, b in wait_iv), "covered_s": wt, "prefill_frac": wp / wt if wt else None,
            "long_cont_frac": wc / wt if wt else None}
# pending uncached work while bad requests waited: pending-token from P lines in those intervals (includes full
# prompts of waiting requests, i.e. an upper bound on uncached work)
pend = [x["pending"] for x in P if any(a <= x["t"] <= b for a, b in wait_iv)]
lane_bad["pending_token_p50"] = pct(pend, .5); lane_bad["pending_token_p90"] = pct(pend, .9)
# measured uncached throughput in those intervals: new tokens / time
tok = sum(x["new_token"] for x in P if any(a <= x["t"] <= b for a, b in wait_iv))
lane_bad["prefill_tok_per_s"] = tok / lane_bad["union_wait_s"] if lane_bad["union_wait_s"] else None
lane["run_prefill_tok_per_s"] = sum(x["new_token"] for x in P) / lane["wall_s_measure"]

# ---------- 7. zero-wait floor (Q3) ----------
def floor_pure(unc, cached):
    n = math.ceil(unc / 8192) if unc > 0 else 0
    return sum(fhat(min(8192, unc - i * 8192), cached + i * 8192) for i in range(n))


# effective per-chunk time under load: measured exec->first / pure model over multi-chunk requests
ratios = [q["exec_to_first"] / q["f_pure_model"] for q in req_rows if q["f_pure_model"] > 0]
eff = pct(ratios, .5)
floor_rows = []
for r in cs:
    ue = r["uncached_expected"]; ce = r["prompt_tokens"] - ue
    fa = floor_pure(r["unc"], r["cached_tokens"]); fe = floor_pure(ue, ce)
    floor_rows.append(dict(req=r["req_id"], phase=r["phase"], edge=r["edge_type"], prompt=r["prompt_tokens"],
                           unc_actual=r["unc"], unc_expected=ue, ttft=r["ttft_s"],
                           wait=r["t_exec_start_s"] - r["t_recv_s"], exec_to_first=r["t_first_token_s"] - r["t_exec_start_s"],
                           floor_pure_actual=fa, floor_pure_expected=fe,
                           floor_loaded_actual=fa * eff, floor_loaded_expected=fe * eff, bad=r["ttft_s"] > lim))
with open(f"{OUT}/chain_start_floor_{TAG}.csv", "w", newline="") as f:
    w_ = csv.DictWriter(f, fieldnames=list(floor_rows[0])); w_.writeheader(); w_.writerows(floor_rows)


def cnt(rows, key, th=30.0):
    return sum(1 for q in rows if q[key] > th)


fb = [q for q in floor_rows if q["bad"]]
floor = {"n_chain_start": len(floor_rows), "n_bad": len(fb), "loaded_factor_p50": eff,
         "loaded_factor_p10_p90": [pct(ratios, .1), pct(ratios, .9)],
         "over30_pure_actual": cnt(floor_rows, "floor_pure_actual"),
         "over30_pure_expected": cnt(floor_rows, "floor_pure_expected"),
         "over30_loaded_actual": cnt(floor_rows, "floor_loaded_actual"),
         "over30_loaded_expected": cnt(floor_rows, "floor_loaded_expected"),
         "bad_over30_pure_actual": cnt(fb, "floor_pure_actual"),
         "bad_over30_loaded_actual": cnt(fb, "floor_loaded_actual"),
         "bad_over30_pure_expected": cnt(fb, "floor_pure_expected"),
         "bad_over30_loaded_expected": cnt(fb, "floor_loaded_expected"),
         "max_unc_actual": max(q["unc_actual"] for q in floor_rows),
         "tokens_for_30s_pure_from0": None}
# tokens that fit into 30 s pure prefill starting from context 0
T = 0
while floor_pure(T + 8192, 0) <= 30.0:
    T += 8192
floor["tokens_for_30s_pure_from0"] = T
T = 0
while floor_pure(T + 8192, 0) * eff <= 30.0:
    T += 8192
floor["tokens_for_30s_loaded_from0"] = T

# ---------- 8. interleaving (Q4) ----------
inter = {}
cont_windows = [w for w in win if w["P"] and all(x["long_cont"] for x in w["P"]) and w["W"] <= 30]
if cont_windows:
    Wt = sum(w["W"] for w in cont_windows)
    dt = sum(40 * dhat(w["bs"], w["ctx"]) for w in cont_windows)
    inter["windows_only_long_cont"] = len(cont_windows)
    inter["decode_share_of_time"] = dt / Wt
    inter["mean_bs"] = sum(w["bs"] for w in cont_windows) / len(cont_windows)
    inter["decode_steps_per_prefill_batch"] = 40 * len(cont_windows) / sum(len(w["P"]) for w in cont_windows)
inter["exec_to_first_over_pure_p50"] = eff
inter["tpot_view"] = "each 8192 chunk (~f s) stalls every decoder once; see summary text"

summary = dict(tag=TAG, calibration=calib, decode_model=dmodel, prefill_model=pmodel, curve=curve, lane=lane,
               lane_while_bad_waited=lane_bad, floor=floor, interleave=inter,
               lag_vs_prompt=[{"prompt_bin_k": b, "n": len(v), "lag_p50": pct(v, .5), "lag_p90": pct(v, .9)}
                              for b, v in sorted(collections.defaultdict(list, {}).items())])
lb = collections.defaultdict(list)
for p, l in lag:
    lb[int(p // 65536) * 64].append(l)
summary["lag_vs_prompt"] = [{"prompt_k": f"{b}-{b + 64}", "n": len(v), "lag_p50": pct(v, .5), "lag_p90": pct(v, .9)}
                            for b, v in sorted(lb.items())]
with open(f"{OUT}/req_exec_{TAG}.csv", "w", newline="") as f:
    w_ = csv.DictWriter(f, fieldnames=list(req_rows[0])); w_.writeheader(); w_.writerows(req_rows)
json.dump(summary, open(f"{OUT}/summary_{TAG}.json", "w"), indent=1, default=float)
print(json.dumps({k: summary[k] for k in ("calibration", "decode_model", "prefill_model", "lane",
                                          "lane_while_bad_waited", "floor", "interleave")}, indent=1, default=float))
for c in curve:
    print(c)
for l in summary["lag_vs_prompt"]:
    print(l)

# ---------- 9. continuation-dominated decode windows: per-chunk cost without decode ----------
# windows whose prefill batches all carry a chunk of the single chunked request (possibly with short riders):
# prefill seconds = W - 40*d_hat = n*alpha + beta*sum(T) + gamma*sum(share*depth)
cw = []
for w in win:
    if w["P"] and w["W"] <= 30 and all(x["long_cont"] for x in w["P"]):
        n = len(w["P"]); T = sum(x["new_token"] for x in w["P"])
        cw.append(dict(t0=w["t0"], n_batches=n, tokens=T, share_tokens=sum(x["share"] for x in w["P"]),
                       depth_mean=sum(x["depth"] for x in w["P"]) / n,
                       tdepth=sum(x["share"] * x["depth"] for x in w["P"]), bs=w["bs"], W=w["W"],
                       decode_s=40 * dhat(w["bs"], w["ctx"]), pref_s=w["W"] - 40 * dhat(w["bs"], w["ctx"])))
if cw:
    with open(f"{OUT}/cont_windows_{TAG}.csv", "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=list(cw[0])); w_.writeheader(); w_.writerows(cw)
    fit = lstsq([[c["n_batches"], c["tokens"] / 1e4, c["tdepth"] / 1e9] for c in cw], [c["pref_s"] for c in cw])
    res = [c["pref_s"] - (fit[0] * c["n_batches"] + fit[1] * c["tokens"] / 1e4 + fit[2] * c["tdepth"] / 1e9) for c in cw]
    per8k = [c["pref_s"] / (c["tokens"] / 8192) for c in cw]
    summary["chunk_from_cont_windows"] = {
        "n_windows": len(cw), "n_batches": sum(c["n_batches"] for c in cw), "tokens": sum(c["tokens"] for c in cw),
        "fit_alpha_s_per_batch": fit[0], "fit_beta_s_per_10k": fit[1], "fit_gamma_s_per_1e9_tok_depth": fit[2],
        "fit_resid_p10_p50_p90": [pct(res, .1), pct(res, .5), pct(res, .9)],
        "f8192_at_depth": {f"{d}k": fit[0] + fit[1] * 0.8192 + fit[2] * 8192 * d * 1024 / 1e9 for d in (0, 64, 128, 192, 256)},
        "pref_s_per_8192_tokens_p25_p50_p75": [pct(per8k, .25), pct(per8k, .5), pct(per8k, .75)],
        "decode_share_p50": pct([c["decode_s"] / c["W"] for c in cw], .5),
        "by_depth": [{"depth_k": f"{lo // 1024}-{(lo + 65536) // 1024}", "n_win": len(v),
                      "pref_per_8192_p25": pct(v, .25), "p50": pct(v, .5), "p75": pct(v, .75)}
                     for lo in range(0, 262144, 65536)
                     for v in [[c["pref_s"] / (c["tokens"] / 8192) for c in cw if lo <= c["depth_mean"] < lo + 65536]] if v]}
    json.dump(summary, open(f"{OUT}/summary_{TAG}.json", "w"), indent=1, default=float)
    print(json.dumps(summary["chunk_from_cont_windows"], indent=1, default=float))

# ---------- 10. per-minute lane occupancy and uncached backlog ----------
T0r = min(r["t_recv_s"] for r in R)
nmin = int((MEAS_T1 - T0r) // 60) + 1
agg = [[0.0] * 6 for _ in range(nmin)]  # covered, prefill, long_cont, decode, new_tokens, bs*time
for t0, t1, pref, cont, dec in occ:
    L = t1 - t0
    m0, m1 = int((t0 - T0r) // 60), int((t1 - T0r) // 60)
    for m in range(max(0, m0), min(nmin - 1, m1) + 1):
        a, b = max(t0, T0r + 60 * m), min(t1, T0r + 60 * (m + 1))
        if b > a and L > 0:
            fr = (b - a) / L
            agg[m][0] += b - a; agg[m][1] += pref * fr; agg[m][2] += cont * fr; agg[m][3] += dec * fr
for x in P:
    m = int((x["t"] - T0r) // 60)
    if 0 <= m < nmin:
        agg[m][4] += x["new_token"]
bsw = [(w["t0"], w["t1"], w["bs"]) for w in win]
for t0, t1, bs in bsw:
    m = int(((t0 + t1) / 2 - T0r) // 60)
    if 0 <= m < nmin:
        agg[m][5] += bs * (t1 - t0)
minute_rows = []
for m in range(nmin):
    tm = T0r + 60 * m + 30
    backlog = sum(r["unc"] if r["t_exec_start_s"] > tm else
                  r["unc"] * max(0.0, 1 - (tm - r["t_exec_start_s"]) / max(1e-6, r["t_first_token_s"] - r["t_exec_start_s"]))
                  for r in R if r["t_recv_s"] <= tm < r["t_first_token_s"])
    arrived = sum(r["unc"] for r in R if T0r + 60 * m <= r["t_recv_s"] < T0r + 60 * (m + 1))
    cov = agg[m][0] or 1e-9
    bad_recv = sum(1 for r in bad if T0r + 60 * m <= r["t_recv_s"] < T0r + 60 * (m + 1))
    minute_rows.append(dict(minute=m, covered_s=agg[m][0], prefill_frac=agg[m][1] / cov, long_cont_frac=agg[m][2] / cov,
                            decode_frac=agg[m][3] / cov, prefill_tok_s=agg[m][4] / 60.0, mean_bs=agg[m][5] / cov,
                            arrived_unc=arrived, backlog_unc_mid=backlog, bad_chain_start_received=bad_recv))
with open(f"{OUT}/minutes_{TAG}.csv", "w", newline="") as f:
    w_ = csv.DictWriter(f, fieldnames=list(minute_rows[0])); w_.writeheader(); w_.writerows(minute_rows)
for r_ in minute_rows[:12]:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r_.items()})

# ---------- 11. per bad chain-start request: what the GPU did while it waited ----------
bad_rows = []
for r in sorted(bad, key=lambda r: r["t_recv_s"]):
    iv = [(r["t_recv_s"], r["t_exec_start_s"])]
    tot, pre, cont = occ_during(iv)
    # which chunked requests occupied the lane during the wait (seconds of their chunks, by model share)
    occupants = collections.Counter()
    for x in P:
        if iv[0][0] <= x["t"] <= iv[0][1] and x["long_cont"]:
            occupants[x["chunk_req"]["req_id"]] += fhat(x["new_token"], x["depth"] or 0, x["new_seq"] - 1)
    top = occupants.most_common(3)
    ahead_unc = sum(R_by[q]["unc"] for q in occupants)
    bad_rows.append(dict(req=r["req_id"], recv_min=(r["t_recv_s"] - T0r) / 60, unc=r["unc"],
                         unc_expected=r["uncached_expected"], ttft=r["ttft_s"],
                         wait=r["t_exec_start_s"] - r["t_recv_s"], exec_to_first=r["t_first_token_s"] - r["t_exec_start_s"],
                         wait_prefill_frac=(pre / tot if tot else None), wait_longcont_frac=(cont / tot if tot else None),
                         n_long_reqs_served_during_wait=len(occupants), their_unc_sum=ahead_unc,
                         top_occupant=(top[0][0] if top else ""), top_occupant_unc=(R_by[top[0][0]]["unc"] if top else 0),
                         floor_pure=floor_pure(r["unc"], r["cached_tokens"]),
                         recv_to_admit=r["t_admit_s"] - r["t_recv_s"], admit_to_exec=r["t_exec_start_s"] - r["t_admit_s"],
                         kv_usage_mean_wait=(lambda u: sum(u) / len(u) if u else None)(
                             [x["full_usage"] for x in ev if iv[0][0] <= x["t"] <= iv[0][1]]),
                         kv_usage_max_wait=max([x["full_usage"] for x in ev if iv[0][0] <= x["t"] <= iv[0][1]] or [None]),
                         regime=("opening_prefill_bound" if pre / max(tot, 1e-9) >= 0.6 else "other")))
with open(f"{OUT}/bad_chain_start_{TAG}.csv", "w", newline="") as f:
    w_ = csv.DictWriter(f, fieldnames=list(bad_rows[0])); w_.writeheader(); w_.writerows(bad_rows)
for b_ in bad_rows:
    print("%5.1f unc %6d wait %6.1f exec %5.1f pref %.2f cont %.2f nlong %2d ahead %8d top %6d" % (
        b_["recv_min"], b_["unc"], b_["wait"], b_["exec_to_first"], b_["wait_prefill_frac"] or -1,
        b_["wait_longcont_frac"] or -1, b_["n_long_reqs_served_during_wait"], b_["their_unc_sum"], b_["top_occupant_unc"]),
          "admit %.1f exec_wait %.1f kv %.2f/%.2f" % (b_["recv_to_admit"], b_["admit_to_exec"], b_["kv_usage_mean_wait"] or -1,
                                                     b_["kv_usage_max_wait"] or -1))
summary["bad_regimes"] = collections.Counter(b_["regime"] for b_ in bad_rows)
kvu = [x["full_usage"] for x in ev]
summary["kv_full_usage_frac_time_ge_0.95_events"] = sum(1 for u in kvu if u >= 0.95) / len(kvu)
json.dump(summary, open(f"{OUT}/summary_{TAG}.json", "w"), indent=1, default=float)
print(summary["bad_regimes"], summary["kv_full_usage_frac_time_ge_0.95_events"])

# ---------- 12. opening burst: scheduling-independent lower bound on chain-start misses vs prefill rate ----------
# Single server of rate C tok/s doing only these chain-start prefills (no decode, no intra work, preemptive EDF on a
# size-greedy chosen set): an optimistic bound, i.e. no scheduler at that rate can miss fewer (estimate, not a run).
import heapq


def max_on_time(jobs, C, D=30.0):
    chosen = []
    for j in sorted(jobs, key=lambda j: j[1]):
        cand = chosen + [j]
        evs = sorted(cand); rem = {i: x[1] / C for i, x in enumerate(evs)}; h = []; t = 0.0; i = 0; ok = True
        while i < len(evs) or h:
            if not h:
                t = max(t, evs[i][0])
            while i < len(evs) and evs[i][0] <= t:
                heapq.heappush(h, (evs[i][0] + D, i)); i += 1
            dl, k = heapq.heappop(h)
            nxt = evs[i][0] if i < len(evs) else 1e18
            run = min(rem[k], nxt - t); t += run; rem[k] -= run
            if rem[k] > 1e-9:
                heapq.heappush(h, (dl, k))
            elif t > dl + 1e-9:
                ok = False; break
        if ok:
            chosen = cand
    return len(chosen)


op = [r for r in cs if r["t_recv_s"] - T0r < 60]
jobs_a = [(r["t_recv_s"] - T0r, r["unc"]) for r in op]
jobs_e = [(r["t_recv_s"] - T0r, r["uncached_expected"]) for r in op]
summary["opening_bound"] = {
    "n_chain_start_first_60s": len(op), "unc_actual": sum(j[1] for j in jobs_a), "unc_expected": sum(j[1] for j in jobs_e),
    "bad_among_them": sum(1 for r in op if r["ttft_s"] > lim),
    "min_misses": {str(C): {"actual": len(op) - max_on_time(jobs_a, C), "expected": len(op) - max_on_time(jobs_e, C)}
                   for C in (7000, 9000, 10000, 11700, 15000, 20000, 23400)}}
json.dump(summary, open(f"{OUT}/summary_{TAG}.json", "w"), indent=1, default=float)
print(json.dumps(summary["opening_bound"]))
