# [T52 / patch 170] Deterministic extend+decode probe for comparing BCG prefill vs eager on the same server config.
#   PORT=<port> python3 bcg_correctness.py <out.json> [prefixes=0,20000,100000] [chunks=100,1000,3000]
# For each cached prefix P (cached first by a 1-token request) and each new-token count c, sends prefix+c random ids,
# greedy 32 tokens, return_logprob with top-5 per output position. Then a 2-request concurrent (mixed) batch.
# COLD env: comma list of cold exact-length prompts (default numcheck set). Records output ids, per-token logprobs, first-token top-5, cached_tokens. Compare two runs with bcg_compare.py.
import json, os, random, sys, threading, urllib.request

out = sys.argv[1]
PS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0,20000,100000").split(",")]
CS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "100,1000,3000").split(",")]
NEW = int(os.environ.get("NEW_TOKENS", "32"))
VMAX = int(os.environ.get("VOCAB_MAX", "19000"))  # dev-box surrogate vocab 19360; real model: 150000
base = f"http://127.0.0.1:{os.environ.get('PORT', '30000')}"


def post(path, body, timeout=3600):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read() or b"{}")


def ids(n, seed):
    r = random.Random(seed)
    return [r.randrange(1000, VMAX) for _ in range(n)]


def gen(prompt, max_new=NEW):
    d = post("/generate", {"input_ids": prompt, "return_logprob": True, "top_logprobs_num": 5,
                           "sampling_params": {"max_new_tokens": max_new, "temperature": 0, "ignore_eos": True}})
    m = d["meta_info"]
    return dict(
        n_in=len(prompt), cached=m.get("cached_tokens"), out_ids=d.get("output_ids"),
        out_lp=[x[0] for x in m["output_token_logprobs"]],
        top1=[[x[0], x[1]] for x in m["output_top_logprobs"][0]],
    )


post("/flush_cache", {})
rows = []
# cold single requests at exact lengths: graph buckets (512/1024/4096) AND padded sizes (37/100/500/1000/3000/5000)
COLD = [int(x) for x in os.environ.get("COLD", "37,100,500,512,1000,1024,3000,4096,5000").split(",") if x]
for L in COLD:
    r = gen(ids(L, 3000 + L)); r.update(tag=f"cold L={L}"); rows.append(r)
    print(json.dumps({k: r[k] for k in ("tag", "n_in", "cached")}), flush=True)
post("/flush_cache", {})
for P in PS:
    prefix = ids(P, 500 + P) if P else []
    if P:
        r = gen(prefix, 1); r.update(tag=f"warm P={P}"); rows.append(r)
    for c in CS:
        r = gen(prefix + ids(c, 7000 + P + c)); r.update(tag=f"P={P} c={c}"); rows.append(r)
        print(json.dumps({k: r[k] for k in ("tag", "n_in", "cached")}), flush=True)
# mixed batch: two requests sent together (one with a cached 20k prefix, one cold)
pre = ids(20000, 500 + 20000) if 20000 in PS else ids(20000, 520000)
reqs = {"mixA P=20000 c=500": pre + ids(500, 91), "mixB cold 1500": ids(1500, 92)}
res = {}
def run(tag, p):
    res[tag] = gen(p)
ths = [threading.Thread(target=run, args=kv) for kv in reqs.items()]
for t in ths: t.start()
for t in ths: t.join()
for tag, r in res.items():
    r.update(tag=tag); rows.append(r); print(json.dumps({k: r[k] for k in ("tag", "n_in", "cached")}), flush=True)
json.dump(rows, open(out, "w"), indent=0)
print("DONE", out)
