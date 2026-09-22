#!/usr/bin/env bash
# env/m0 = env/sgl (system site packages: torch 2.13.0+cu130, triton 3.7.1, ...) + the image pins that
# differ (evidence/base_manifest/pip_freeze.txt). Wheels are fetched with curl from the aliyun mirror
# (pip via pypi.org measured ~10-250 kB/s on this box; aliyun ~18 MB/s), then installed from local files.
set -euo pipefail
source /sjtu/linhang/arena/repo/scripts/e1_env.sh
cd /sjtu/linhang/arena
MIRROR=https://mirrors.aliyun.com/pypi
WH=/sjtu/linhang/arena/cache/wheels; mkdir -p "$WH" "$TMPDIR"
[ -x env/m0/bin/python ] || env/sgl/bin/python -m venv --system-site-packages env/m0
fetch() {  # fetch <project> <exact wheel filename prefix>
  local href
  href=$(curl -s --max-time 30 "$MIRROR/simple/$1/" | grep -o "href=\"[^\"]*$2[^\"]*\.whl[^\"]*\"" | head -1 | sed 's/^href="//; s/"$//; s/#.*//')
  [ -n "$href" ] || { echo "ERROR no wheel for $2"; exit 1; }
  local url="$MIRROR/simple/$1/$href" name; name=$(basename "${href%%#*}")
  curl -sS -L --max-time 900 -o "$WH/$name" "$url"
  echo "fetched $name $(stat -c %s "$WH/$name") bytes"
  WHEELS+=("$WH/$name")
}
WHEELS=()
fetch sglang-kernel "sglang_kernel-0.4.6.post1-cp310-abi3-manylinux2014_x86_64"
fetch sgl-deep-gemm "sgl_deep_gemm-0.1.5.post3-py3-none-manylinux2014_x86_64"
fetch tilelang "tilelang-0.1.12-cp38-abi3-manylinux_2_27_x86_64"
env/m0/bin/python -m pip install --no-deps --no-index "${WHEELS[@]}"
env/m0/bin/python -m pip list 2>/dev/null | grep -iE "^(sglang-kernel|sgl-deep-gemm|tilelang|torch|triton) "
env/m0/bin/python /sjtu/linhang/arena/repo/scripts/m0/make_standin.py /sjtu/linhang/arena/repo/s1-dev/glm_tok/config.json /sjtu/linhang/arena/repo/s1-dev/glm_tok /sjtu/linhang/arena/models/m0-glm5-dsa-4l
echo SETUP_DONE
