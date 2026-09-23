CP_NAME=b113dcp
CP_PATCHES="000-interface-compliance.patch 101-d1v12-on-base.patch 105-role-split-single-partial.patch 110-sm80-dsa-indexer.patch 111-sm80-fp8-moe-marlin.patch 112-sm80-indexer-kernels.patch 113-sm80-prefill-indexer.patch"
CP_ARGS="--dcp-size 8"
# Cold-prefill probe: start/reuse engine VARIANT, then single-request cold prefills (20k/60k/190k) + profile of 60k,
# + capability smoke subset (correctness guard for structural variants). Wrapper sets:
#   CP_NAME (unique src name), CP_PATCHES (patch list), CP_ARGS (extra launch args), CP_ENV (extra env assignments)
source $AX/bin/scripts/pod/lib.sh
prepare_src "$CP_NAME" $CP_PATCHES || exit 1
export SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=154827,154829 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
[ -n "${CP_ENV:-}" ] && export $CP_ENV
ensure_engine "$CP_NAME" --schedule-policy lpm --dsa-prefill-backend tilelang --dsa-decode-backend tilelang $CP_ARGS || exit 1
grep -h "KV Cache is allocated\|max_total_num_tokens\|row-shard active\|cp_size\|dcp" $AX/engine_current.log | head -6
python3 $AX/verify_kit/coldprobe.py $RUN_DIR 20000,60000,190000 60000
python3 $AX/verify_kit/component_table.py $RUN_DIR/prof_60000 2>/dev/null | head -14
curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null || { echo "ENGINE_NOT_RUNNING"; exit 2; }
cat > $RUN_DIR/problems.json <<'JSON'
[{"q": "Find the remainder when $2^{2026}$ is divided by $1000$.", "a": 864}, {"q": "How many positive divisors does $10!$ have?", "a": 270}, {"q": "Find the number of ordered pairs of positive integers $(a,b)$ such that $a+b=1000$ and neither $a$ nor $b$ has a zero digit.", "a": 738}, {"q": "How many trailing zeros does $1000!$ have?", "a": 249}, {"q": "Find the least positive integer $n$ such that $2^n \\equiv 1 \\pmod{125}$.", "a": 100}, {"q": "How many lattice paths from $(0,0)$ to $(6,6)$ using unit steps right or up never pass above the line $y=x$?", "a": 132}, {"q": "Find the remainder when $7^{2026}$ is divided by $1000$.", "a": 649}, {"q": "How many subsets of $\\{1,2,\\dots,12\\}$ (including the empty set) have a sum divisible by $4$?", "a": 1024}, {"q": "What is the sum of the digits of $2^{100}$?", "a": 115}, {"q": "How many integers $n$ with $1\\le n\\le 1000$ are divisible by neither $2$, $3$, nor $5$?", "a": 266}, {"q": "Find the number of positive integers $n\\le 2026$ such that $n^2+1$ is divisible by $5$.", "a": 810}, {"q": "A fair coin is flipped $10$ times. The probability that no two consecutive flips are both heads is $m/n$ in lowest terms. Find $m+n$.", "a": 73}]
JSON
python3 - $RUN_DIR <<'PY'
import json, re, sys, time, urllib.request, concurrent.futures as cf
d = sys.argv[1]; probs = json.load(open(f"{d}/problems.json")); port = __import__("os").environ.get("PORT", "30000")
def ask(i):
    body = {"model": "default", "messages": [{"role": "user", "content": probs[i]["q"] + " Put the final integer answer in \\boxed{}."}], "max_tokens": 60000}
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600))
    c = r["choices"][0]; txt = c["message"].get("content") or ""; m = re.findall(r"\\boxed\{([^}]*)\}", txt)
    got = m[-1].strip() if m else None
    return dict(i=i, want=probs[i]["a"], got=got, ok=(got is not None and re.sub(r"[^0-9-]", "", got) == str(probs[i]["a"])),
                finish=c.get("finish_reason"), completion_tokens=r["usage"]["completion_tokens"], secs=round(time.time() - t, 1),
                reasoning_chars=len(c["message"].get("reasoning_content") or ""))
with cf.ThreadPoolExecutor(12) as ex: res = sorted(ex.map(ask, range(len(probs))), key=lambda x: x["i"])
json.dump(res, open(f"{d}/cap_results.json", "w"), indent=1)
for x in res: print(json.dumps(x))
print(f"CAP_SMOKE correct={sum(x['ok'] for x in res)}/{len(res)} finish={sorted(set(x['finish'] for x in res))} "
      f"mean_completion_tokens={sum(x['completion_tokens'] for x in res)//len(res)}")
PY
