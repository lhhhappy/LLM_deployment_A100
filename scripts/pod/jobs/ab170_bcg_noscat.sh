# 170 A/B (T52): prefill backend=breakable, scatter=, chunk 4096. chunkcost + interference + cap smoke (real weights).
source $AX/bin/scripts/pod/lib.sh
prepare_src "b170" 000-interface-compliance.patch 101-d1v12-on-base.patch 105-role-split-single-partial.patch 106-defer-chunk-on-no-kv.patch 110-sm80-dsa-indexer.patch 111-sm80-fp8-moe-marlin.patch 112-sm80-indexer-kernels.patch 113-sm80-prefill-indexer.patch 140-kda-dual-snapshot.patch 120-sched-protect-chain.patch 170-glm-bcg-prefill.patch || exit 1
export SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=154827,154829 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 SGLANG_AX_SCHED_COLD_CAP=2048 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_KDA_DUAL_SNAPSHOT=1
ensure_engine "b170" --schedule-policy lpm --dsa-prefill-backend tilelang --dsa-decode-backend tilelang --chunked-prefill-size 4096 --mem-fraction-static 0.75  --cuda-graph-max-bs-decode 64 --max-mamba-cache-size 200 --prefill-decode-interval 3 --cuda-graph-backend-prefill breakable || exit 1
grep -h "KV Cache is allocated\|prefill CUDA graph\|Capture.*prefill" $AX/engine_current.log | tail -4 | cut -c1-220
python3 $AX/verify_kit/chunkcost.py $RUN_DIR
python3 $AX/verify_kit/interference.py $RUN_DIR 12 190000 1500
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
