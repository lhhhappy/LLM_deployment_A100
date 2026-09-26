#!/usr/bin/env bash
# Full capability check body (run by a job after the engine is up, like cap_smoke_body.sh): the public AIME 2026 set and the
# public GPQA-Diamond set, served through /v1/chat/completions with the same request shape as the platform's capability
# stage as far as task.md tells it (reasoning on, no output cap that truncates answers). Prints one line
#   CAP_FULL aime=<correct>/<n> gpqa=<correct>/<n> ...
# and writes cap_full_results.json in RUN_DIR. The platform's exact question subset, order and sampling are not public,
# so this is a numerics/regression check between engines and a rough absolute reading, not the platform's score.
# Inputs: CAP_DIR/aime26.jsonl {id, problem, answer}, CAP_DIR/gpqa_diamond.jsonl {id, question, choices[4], answer} (letter).
set -u
curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null || { echo "ENGINE_NOT_RUNNING"; exit 2; }
python3 - "$RUN_DIR" "${CAP_DIR:?}" "${CAP_CONCURRENCY:-24}" <<'PY'
import json, re, sys, time, urllib.request, concurrent.futures as cf, os
d, cap, conc = sys.argv[1], sys.argv[2], int(sys.argv[3]); port = os.environ.get("PORT", "30000")
def load(p): return [json.loads(l) for l in open(p) if l.strip()]
aime = load(f"{cap}/aime26.jsonl"); gpqa = load(f"{cap}/gpqa_diamond.jsonl")
items = [("aime", x) for x in aime] + [("gpqa", x) for x in gpqa]
def prompt(kind, x):
    if kind == "aime":
        return x["problem"] + "\n\nPut the final integer answer in \\boxed{}."
    opts = "\n".join(f"({l}) {c}" for l, c in zip("ABCD", x["choices"]))
    return x["question"] + "\n\n" + opts + "\n\nThink it through, then give only the letter of the correct option in \\boxed{}."
def ask(k):
    kind, x = items[k]
    body = {"model": "default", "messages": [{"role": "user", "content": prompt(kind, x)}], "max_tokens": 60000}
    t = time.time()
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600))
    except Exception as e:
        return dict(k=k, kind=kind, id=x["id"], error=str(e)[:200], ok=False, got=None, want=x["answer"])
    c = r["choices"][0]; txt = c["message"].get("content") or ""; m = re.findall(r"\\boxed\{([^}]*)\}", txt)
    got = m[-1].strip() if m else None
    if kind == "aime":
        ok = got is not None and re.sub(r"[^0-9-]", "", got) == str(x["answer"])
    else:
        letter = re.sub(r"[^A-Da-d]", "", got or "")[:1].upper() if got else ""
        ok = letter == x["answer"]
    return dict(k=k, kind=kind, id=x["id"], want=x["answer"], got=got, ok=ok, finish=c.get("finish_reason"),
                completion_tokens=r["usage"]["completion_tokens"], secs=round(time.time() - t, 1),
                reasoning_chars=len(c["message"].get("reasoning_content") or ""))
t0 = time.time()
with cf.ThreadPoolExecutor(conc) as ex: res = sorted(ex.map(ask, range(len(items))), key=lambda r: r["k"])
json.dump(res, open(f"{d}/cap_full_results.json", "w"), indent=1)
def summ(kind):
    rr = [r for r in res if r["kind"] == kind]
    return sum(r["ok"] for r in rr), len(rr), sum(1 for r in rr if r.get("error")), sum(1 for r in rr if r.get("finish") == "length"), (sum(r.get("completion_tokens", 0) for r in rr) // max(1, len(rr)))
a, g = summ("aime"), summ("gpqa")
print(f"CAP_FULL aime={a[0]}/{a[1]} gpqa={g[0]}/{g[1]} errors={a[2]+g[2]} truncated={a[3]+g[3]} mean_tokens aime={a[4]} gpqa={g[4]} wall={time.time()-t0:.0f}s")
PY
