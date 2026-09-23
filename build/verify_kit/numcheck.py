# Numerics fingerprint (~1-2 min): greedy continuations + per-token logprobs for prompts of chosen exact lengths
# (graph-bucket sizes AND padded sizes), a prefix-hit extend, a mixed 2-request batch, and real-text chat prompts.
# Run on two engines (reference eager vs candidate) and diff with numcheck_cmp.py. Detects wrong-but-fluent outputs
# that throughput probes never see.
# Usage: python3 numcheck.py <out_dir> [lengths=37,100,500,512,1000,1024,2000,2048,3000,4096,5000]
import json, os, random, sys, threading, urllib.request
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
LENS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "37,100,500,512,1000,1024,2000,2048,3000,4096,5000").split(",")]
VMAX = int(os.environ.get("VOCAB_MAX", "150000"))
base = f"http://127.0.0.1:{os.environ.get('PORT', '30000')}"
def post(path, body, timeout=1800):
    return json.loads(urllib.request.urlopen(urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=timeout).read())
def flush():
    for _ in range(30):
        try: urllib.request.urlopen(urllib.request.Request(base + "/flush_cache", b"{}", {"Content-Type": "application/json"}), timeout=60).read(); return
        except Exception: __import__("time").sleep(2)
def ids(n, seed): r = random.Random(seed); return [r.randrange(1000, VMAX) for _ in range(n)]
def gen(prompt=None, text=None, n=48):
    body = {"sampling_params": {"max_new_tokens": n, "temperature": 0, "ignore_eos": True}, "return_logprob": True}
    if prompt is not None: body["input_ids"] = prompt
    else: body["text"] = text
    d = post("/generate", body); m = d["meta_info"]
    toks = [t[1] for t in m["output_token_logprobs"]]; lps = [t[0] for t in m["output_token_logprobs"]]
    return dict(tokens=toks, logprobs=lps, cached=m.get("cached_tokens"))
res = {}
flush()
for L in LENS:                                   # cold, single request, exact length
    res[f"cold_{L}"] = gen(ids(L, 1000 + L)); print("NUM", L, res[f"cold_{L}"]["tokens"][:8], flush=True)
pre = ids(8000, 7)                               # prefix-hit extend with a padded tail
gen(pre, n=1); res["hit_8000+333"] = gen(pre + ids(333, 8)); res["hit_8000+512"] = gen(pre + ids(512, 9))
flush()                                          # 2 concurrent requests -> one mixed extend batch
box = {}
th = [threading.Thread(target=lambda k, p: box.__setitem__(k, gen(p)), args=(k, p)) for k, p in (("mix_a_700", ids(700, 11)), ("mix_b_1500", ids(1500, 12)))]
for t in th: t.start()
for t in th: t.join()
res.update(box)
TEXT = ("The committee reviewed the proposal in detail. Each member described the risks, the costs and the expected "
        "benefits, and then they voted. ")
for k, reps in (("text_short", 3), ("text_mid", 40), ("text_long", 160)):
    res[k] = gen(text=TEXT * reps + "\nSummarize the discussion above in one sentence:")
json.dump(res, open(f"{out}/numcheck.json", "w"))
print("NUMCHECK_DONE", len(res), flush=True)
