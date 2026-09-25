# Upstream scheduler/prefix-cache tests on the unmodified base tree vs our tree, same env.
# Runs on the dev box (e.g. under scripts/gjob), CPU only, offline: the OPT stand-in is served
# through VLLM_MODEL_REDIRECT_PATH. Base tree = official a811738a6 sources + installed binaries.
set -uo pipefail
V=/sjtu/linhang/arena/vllm
source /sjtu/linhang/arena/env.sh; unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export PATH=$V/env/bin:$PATH TMPDIR=$V/cache/tmp HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=
export VLLM_MODEL_REDIRECT_PATH=$V/models/redirect.json VLLM_CACHE_ROOT=$V/cache/vllm
OUT=$V/runs/regress-$(date -u +%H%M%S); mkdir -p $OUT
TESTS="tests/v1/core/test_scheduler.py tests/v1/core/test_prefix_caching.py tests/v1/core/test_mamba_align_chunk_split.py tests/v1/core/prefix_cache"
for name in base ours; do
  if [ $name = base ]; then D=$V/build/base-a811738a6; else D=$V/src; fi
  (cd $D && PYTHONPATH=$D python -c "import vllm; print('$name vllm from', vllm.__file__)" &&
   PYTHONPATH=$D python -m pytest -q -p no:cacheprovider -p no:warnings -rA $TESTS > $OUT/$name.log 2>&1)
  grep -E "^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS) tests/" $OUT/$name.log | sed 's/ - .*//' | sort > $OUT/$name.results
  tail -1 $OUT/$name.log
done
python3 - $OUT <<'PY'
import sys, collections
out = sys.argv[1]
def load(n):
    d = {}
    for line in open(f"{out}/{n}.results"):
        st, tid = line.split(" ", 1)
        d[tid.strip()] = st
    return d
b, o = load("base"), load("ours")
new_fail = sorted(t for t in o if o[t] in ("FAILED", "ERROR") and b.get(t) not in ("FAILED", "ERROR"))
fixed = sorted(t for t in b if b[t] in ("FAILED", "ERROR") and o.get(t) == "PASSED")
only_ours = sorted(t for t in o if t not in b)
print("base:", collections.Counter(b.values()), "ours:", collections.Counter(o.values()))
print("new failures in ours:", len(new_fail)); [print("  ", t) for t in new_fail[:40]]
print("tests only in ours:", len(only_ours), collections.Counter(o[t] for t in only_ours))
print("base-failing now passing:", len(fixed))
PY
echo REGRESS_DONE $OUT
