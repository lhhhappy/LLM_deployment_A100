# How much cold prefill could a hybrid (KDA+DSA) engine avoid with better state checkpoints?
# For each dev-set request (dispatch order), LCP in tokens with ANY earlier request (cross-session), rendered and
# tokenized exactly like the harness (s1_common.Renderer). Compares reuse policies:
#   frozen   : harness 'uncached_expected' (what the gates' bucket labels assume)
#   end_only : reuse only when an earlier prompt is fully a prefix of this one (stock-like: one state at the end)
#   grid_G   : reuse up to the largest multiple of G <= LCP (KDA state checkpoints every G tokens)
#   exact    : reuse the full LCP (state at every divergence point; upper bound)
import sys, json, gzip, glob, os, collections
S1 = sys.argv[1]; sys.path.insert(0, os.path.join(S1, "harness"))
from s1_common import Renderer
import numpy as np
from s1_common import load_index
_rows, _chains, _grp = load_index(f"{S1}/data/dev-combined-v1")   # same selection as the harness (canon, in_serving_load)
R = list(_rows.values())
bodies = {}
for f in glob.glob(f"{S1}/data/dev-combined-v1/bodies/**/*.jsonl.gz", recursive=True):
    for l in gzip.open(f, "rt"):
        b = json.loads(l); bodies[b["req_id"]] = b
ren = Renderer(f"{S1}/glm_tok")
R.sort(key=lambda r: r["dispatch_offset_ms"])
toks = []
for r in R:
    t = np.array(ren.tokenizer.encode(ren.render(bodies[r["_req_id"]]), add_special_tokens=False), dtype=np.int32)
    toks.append(t)
def lcp(a, b):
    n = min(len(a), len(b)); 
    if n == 0: return 0
    d = np.nonzero(a[:n] != b[:n])[0]
    return int(d[0]) if len(d) else n
rows = []
for i, r in enumerate(R):
    t = toks[i]; best = 0; endreuse = 0
    for j in range(i):
        l = lcp(t, toks[j]); best = max(best, l)
        if l == len(toks[j]): endreuse = max(endreuse, l)
    rows.append(dict(i=i, n=len(t), glm=r["glm_tokens"], frozen=r["uncached_expected"], exact_hit=best, end_hit=endreuse,
                     start=(r["chain_index"] == 0 or r["edge_type"] in ("chain-head", "system-tools-changed") or r["phase"] == "context_reset"),
                     edge=r["edge_type"], same_session_prev=r["glm_lcp_with_prev"]))
def unc(row, pol):
    n = row["n"]
    if pol == "frozen": return row["frozen"]
    if pol == "end_only": return n - row["end_hit"]
    if pol == "exact": return n - row["exact_hit"]
    G = int(pol.split("_")[1]); return n - (row["exact_hit"] // G) * G
pols = ["frozen", "end_only", "grid_8192", "grid_4096", "grid_1024", "exact"]
out = {}
for grp, sel in (("all", lambda r: True), ("chain_start", lambda r: r["start"]), ("intra", lambda r: not r["start"])):
    rs = [r for r in rows if sel(r)]
    out[grp] = {"n": len(rs), **{p: int(sum(unc(r, p) for r in rs)) for p in pols}}
    v = sorted(unc(r, "grid_4096") for r in rs); f = sorted(r["frozen"] for r in rs)
    out[grp]["p50_frozen"] = f[len(f)//2]; out[grp]["p50_grid4096"] = v[len(v)//2]
    out[grp]["p90_frozen"] = f[int(.9*len(f))]; out[grp]["p90_grid4096"] = v[int(.9*len(v))]
by_edge = collections.defaultdict(lambda: [0, 0, 0])
for r in rows:
    if r["start"]:
        e = by_edge[r["edge"]]; e[0] += 1; e[1] += r["frozen"]; e[2] += unc(r, "grid_4096")
out["chain_start_by_edge(n,frozen,grid4096)"] = dict(by_edge)
print(json.dumps(out, indent=1, ensure_ascii=False))
json.dump(rows, open(sys.argv[2], "w"))
