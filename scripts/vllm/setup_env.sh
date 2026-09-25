#!/usr/bin/env bash
# Build the vLLM dev environment on the GPU dev box (run there; idempotent).
#   bash /sjtu/linhang/arena/vllm/tools/vllm/setup_env.sh
# Everything lives under /sjtu/linhang/arena/vllm (the root filesystem has ~2 GB free).
#
# 1. Official prebuilt wheel of the base commit (engine/docs/vllm/README.md).
# 2. Its dependency set resolved against pypi.org into deps/lock.txt, installed from the
#    Aliyun mirror with pypi.org as fallback (the default dev-box mirror lags by months).
# 3. CUDA compiler packages pinned to 13.0, matching torch's CUDA 13.0 runtime headers and the
#    13.0 forward-compat driver: the resolver otherwise picks nvcc 13.4, and FlashInfer's JIT
#    (sampling kernels) then fails "CUDA compiler and CUDA toolkit headers are incompatible".
# 4. Build and test tools. The editable install of src/ is scripts/vllm/devbox.sh install.
set -euo pipefail
source /sjtu/linhang/arena/env.sh
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY PIP_PROXY
export PIP_CONFIG_FILE=/dev/null  # the user pip.conf points at a proxy that is not running

V=/sjtu/linhang/arena/vllm
COMMIT=a811738a6051b5b12a7fcf800f465f2dd2df0a0e
WHL=vllm-0.30.1rc1.dev114+ga811738a6-cp38-abi3-manylinux_2_28_x86_64.whl
MIRROR=https://mirrors.aliyun.com/pypi/simple
export TMPDIR=$V/cache/tmp UV_CACHE_DIR=$V/cache/uv UV_HTTP_TIMEOUT=300 PIP_CACHE_DIR=$V/cache/pip
mkdir -p $V/wheels $V/deps $V/cache/tmp $V/runs $V/models

cd $V/wheels
if [ ! -f "$WHL" ]; then
  curl -sSL --retry 3 -o "$WHL.part" \
    "https://wheels.vllm.ai/$COMMIT/vllm-0.30.1rc1.dev114%2Bga811738a6-cp38-abi3-manylinux_2_28_x86_64.whl"
  mv "$WHL.part" "$WHL"
fi
sha256sum "$WHL" | tee "$WHL.sha256"

[ -x $V/env/bin/python ] || /sjtu/linhang/arena/env/sgl/bin/python3.12 -m venv $V/env
PY=$V/env/bin/python
$PY -m pip install -q --index-url $MIRROR --upgrade pip uv
UV="$PY -m uv"

echo "$V/wheels/$WHL" > $V/deps/in.txt
[ -s $V/deps/lock.txt ] || $UV pip compile --python $PY --python-version 3.12 \
  --index-url https://pypi.org/simple $V/deps/in.txt -o $V/deps/lock.txt --no-header --no-annotate
cat > $V/deps/cuda-toolchain.txt <<'EOF'
nvidia-cuda-nvcc==13.0.88
nvidia-cuda-crt==13.0.88
nvidia-nvvm==13.0.88
nvidia-cuda-cccl==13.0.85
EOF
$UV pip install --python $PY --extra-index-url $MIRROR --index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match -r $V/deps/lock.txt --override $V/deps/cuda-toolchain.txt
$UV pip install --python $PY --index-url $MIRROR \
  'cmake>=3.26.1' ninja 'packaging>=24.2' 'setuptools>=77.0.3,<81.0.0' 'setuptools-scm>=8' \
  'setuptools-rust>=1.9.0' wheel 'jinja2>=3.1.6' \
  pytest pytest-asyncio pytest-timeout pytest-forked httpx tblib 'ruff==0.14.0'
# A CUDA_HOME for JIT builds (FlashInfer links with -L$CUDA_HOME/lib64 -lcudart -lcuda): the pip
# CUDA 13 packages ship bin/include/lib with versioned .so names only, so expose them through
# symlinks here instead of editing the packages. libcuda comes from the 13.0 compat driver.
CU=$V/env/lib/python3.12/site-packages/nvidia/cu13
CH=$V/cuda-home
mkdir -p $CH/lib64/stubs
for d in bin include nvvm; do ln -sfn $CU/$d $CH/$d; done
for f in $CU/lib/*.so*; do ln -sfn "$f" $CH/lib64/$(basename "$f"); done
for f in $CU/lib/*.so.*; do
  base=$(basename "$f"); stem=${base%%.so.*}.so
  [ -e $CH/lib64/$stem ] || ln -sfn "$f" $CH/lib64/$stem
done
ln -sfn /sjtu/linhang/arena/env/cuda-compat-13-0/usr/local/cuda-13.0/compat/libcuda.so $CH/lib64/stubs/libcuda.so
$PY -m pip list 2>/dev/null | grep -i -E \
  "^(vllm|torch|triton|flashinfer-python|tilelang|transformers|xgrammar|nvidia-cuda-nvcc|nvidia-cuda-runtime) "
echo ENV_READY
