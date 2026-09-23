# 116 DCP probe (T50b): --dcp-size 8 with 116 (latent really sharded, DSA reads translated, indexer K replicated in the
# virtual loc space). Covers: capacity line, numcheck vs a non-DCP reference engine, a prefix-hit extend on VIRTUAL locs
# past the old per-rank row count (818,112 on 025), and cap smoke. Two engine loads (~21 min each). Real weights.
# Pass: no ENGINE_DIED/illegal memory; NUMCMP wrong=0 for numcheck AND hiload; hiload hit shows cached_tokens>0 and
# max live tokens > 818112; CAP_SMOKE >= 10/12. Expected capacity: ~450k tokens/rank (x8 logical ~3.6M), see 116 .md.
source $AX/bin/scripts/pod/lib.sh
P="000-interface-compliance.patch 101-d1v12-on-base.patch 105-role-split-single-partial.patch 106-defer-chunk-on-no-kv.patch 110-sm80-dsa-indexer.patch 111-sm80-fp8-moe-marlin.patch 112-sm80-indexer-kernels.patch 113-sm80-prefill-indexer.patch 114-indexer-row-shard.patch 115-sm80-sparse-attn-many-heads.patch 116-dcp-dsa-address.patch"
prepare_src "b116" $P || exit 1
export SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=154827,154829 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
ARGS="--schedule-policy lpm --dsa-prefill-backend tilelang --dsa-decode-backend tilelang --chunked-prefill-size 16384 --mem-fraction-static 0.75 --cuda-graph-max-bs-decode 64"
mkdir -p $RUN_DIR/dcp $RUN_DIR/ref

# hiload: 5 cold prompts of 180k random tokens (900k live > 818,112 -> the allocator hands out virtual locs past the old
# per-rank rows), then prefix-hit extends of the LAST (highest locs) and FIRST prompt. Output format = numcheck.json.
cat > $RUN_DIR/hiload.py <<'PY'
import json, os, random, sys, urllib.request
out = sys.argv[1]; base = f"http://127.0.0.1:{os.environ.get('PORT', '30000')}"
def post(path, body, timeout=3600):
    return json.loads(urllib.request.urlopen(urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=timeout).read())
def ids(n, seed): r = random.Random(seed); return [r.randrange(1000, 150000) for _ in range(n)]
def gen(p, n):
    m = post("/generate", {"input_ids": p, "sampling_params": {"max_new_tokens": n, "temperature": 0, "ignore_eos": True}, "return_logprob": True})["meta_info"]
    return dict(tokens=[t[1] for t in m["output_token_logprobs"]], logprobs=[t[0] for t in m["output_token_logprobs"]], cached=m.get("cached_tokens"))
urllib.request.urlopen(urllib.request.Request(base + "/flush_cache", b"{}", {"Content-Type": "application/json"}), timeout=120).read()
P = [ids(180000, 500 + i) for i in range(5)]
res = {}
for i, p in enumerate(P):
    res[f"cold_{i}"] = gen(p, 1); print("HILOAD cold", i, res[f"cold_{i}"]["cached"], flush=True)
for i in (4, 0):
    res[f"hit_{i}"] = gen(P[i] + ids(600, 900 + i), 32); print("HILOAD hit", i, "cached", res[f"hit_{i}"]["cached"], flush=True)
json.dump(res, open(f"{out}/hiload.json", "w"))
PY

# ---- 1. DCP8 engine under test
ensure_engine "b116" $ARGS --dcp-size 8 || exit 1
grep -h "KV Cache is allocated\|max_total_num_tokens" $AX/engine_current.log | tail -2 | cut -c1-200   # capacity line
python3 $AX/verify_kit/numcheck.py $RUN_DIR/dcp >/dev/null || echo "NUMCHECK_FAILED dcp"
python3 $RUN_DIR/hiload.py $RUN_DIR/dcp || echo "HILOAD_FAILED dcp"
grep -ho "full token usage: [0-9.]*" $RUN_DIR/server.log | awk '{if($4>m)m=$4}END{print "HILOAD max_full_token_usage="m" (x logical capacity above; must exceed 818112 live tokens)"}'
curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null || { echo "ENGINE_DEAD after hiload"; grep -m3 "illegal memory\|Error" $RUN_DIR/server.log; exit 3; }
PORT=$PORT RUN_DIR=$RUN_DIR bash $AX/verify_kit/cap_smoke_body.sh
cp $RUN_DIR/server.log $RUN_DIR/dcp/server.log

# ---- 2. non-DCP reference (same code; 116 is the identity without --dcp-size)
ensure_engine "b116" $ARGS || exit 1
grep -h "KV Cache is allocated\|max_total_num_tokens" $AX/engine_current.log | tail -2 | cut -c1-200
python3 $AX/verify_kit/numcheck.py $RUN_DIR/ref >/dev/null || echo "NUMCHECK_FAILED ref"
python3 $RUN_DIR/hiload.py $RUN_DIR/ref || echo "HILOAD_FAILED ref"

# ---- 3. verdict
echo "== numcheck (ref vs dcp)"; python3 $AX/verify_kit/numcheck_cmp.py $RUN_DIR/ref/numcheck.json $RUN_DIR/dcp/numcheck.json | tee $RUN_DIR/numcmp.txt | tail -1
echo "== hiload (ref vs dcp)";  python3 $AX/verify_kit/numcheck_cmp.py $RUN_DIR/ref/hiload.json $RUN_DIR/dcp/hiload.json | tee $RUN_DIR/hiloadcmp.txt
