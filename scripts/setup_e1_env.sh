#!/usr/bin/env bash
# Install the pinned E1 text-serving subset. Source/harness remain unmodified.
set -euo pipefail
source /sjtu/linhang/arena/code/e1_env.sh
E1_RUN=/sjtu/linhang/arena/runs/E1_20260922
E1_SOURCE=/sjtu/linhang/arena/code/sglang-v0.5.20
test "$(python -c 'import sys; print(sys.prefix)')" = /sjtu/linhang/arena/env/sgl
mkdir -p "$E1_RUN" /sjtu/linhang/arena/cache/cuda
install_e1() {
    # Prefer the mirror's compatible wheels. pip with an extra index preferred
    # the public CDN even for the same version, making large downloads very slow.
    uv -v pip install --python /sjtu/linhang/arena/env/sgl/bin/python \
      --index https://mirrors.ivolces.com/pypi/simple/ \
      --default-index https://mirrors.aliyun.com/pypi/simple/ \
      --index-strategy unsafe-first-match "$@"
}
E1_CUBLAS=/sjtu/linhang/arena/cache/nvidia_cublas-13.1.1.3-py3-none-manylinux_2_27_x86_64.whl
if [[ -f "$E1_CUBLAS" ]]; then
    test "$(sha256sum "$E1_CUBLAS" | cut -d ' ' -f 1)" = 37936a16db8fe4ac1f065c2139360608a543a09275cb1a1af612e08cfa065436
    install_e1 --no-deps "$E1_CUBLAS"
fi
install_e1 'torch==2.13.0' 'cuda-toolkit[nvcc,cccl]==13.0.3' setuptools-scm setuptools-rust wheel
# SGLang's nvcc JIT links CUDA_HOME/lib64, whereas pip's cu13 tree uses lib.
if [[ ! -e "$CUDA_HOME/lib64" ]]; then
    ln -s lib "$CUDA_HOME/lib64"
fi
if [[ ! -e "$CUDA_HOME/lib/libcudart.so" ]]; then
    test -f "$CUDA_HOME/lib/libcudart.so.13"
    ln -s libcudart.so.13 "$CUDA_HOME/lib/libcudart.so"
fi
# ivolces lacks this wheel. Aliyun's copy has the official PyPI SHA256.
E1_KERNEL=/sjtu/linhang/arena/cache/sglang_kernel-0.4.7-cp310-abi3-manylinux2014_x86_64.whl
if [[ ! -f "$E1_KERNEL" ]]; then
    curl -fL --retry 2 -o "$E1_KERNEL" \
      https://mirrors.aliyun.com/pypi/packages/63/db/45fcc5dc66de8e8f2e63c908b30bc309f20086d6b063a1aaa813c6dc478f/sglang_kernel-0.4.7-cp310-abi3-manylinux2014_x86_64.whl
fi
test "$(sha256sum "$E1_KERNEL" | cut -d ' ' -f 1)" = 666f39de214a1558c5a98f43e6562e1032d8d08823b61f6a307830882f3241ce
install_e1 --no-deps "$E1_KERNEL"
install_e1 \
  'transformers==5.12.1' 'tokenizers==0.22.2' 'flashinfer-python==0.6.18' \
  'apache-tvm-ffi==0.1.11' 'compressed-tensors==0.18.0' \
  'cuda-tile==1.6.0rc5' 'cuda-python>=13.0' 'numba==0.65.1' \
  'torch_memory_saver>=0.0.9.post1' 'nvidia-cutlass-dsl[cu13]==4.6.2' \
  'openai==2.6.1' 'openai-harmony==0.0.4' 'outlines==0.1.11' 'xgrammar==0.2.1' \
  'llguidance>=1.7.6,<2' 'mistral_common>=1.11.5' \
  aiohttp anthropic 'blobfile==3.0.0' build datasets distro easydict einops fastapi gguf \
  interegular ipython msgspec ninja numpy nvidia-ml-py orjson packaging \
  partial_json_parser pillow prometheus-client psutil pybase64 pydantic \
  python-multipart pyzmq requests scipy sentencepiece setproctitle tiktoken \
  tqdm uvicorn uvloop watchfiles xxhash zstandard safetensors pytest torchvision \
  'soundfile==0.13.1' 'sgl-deep-gemm==0.2.0'
test "$(git -C "$E1_SOURCE" rev-parse HEAD)" = 94602c9c2b7cbdb8efd5c52802dac6a1c180089e
git -C "$E1_SOURCE" diff --quiet HEAD -- python
SGLANG_BUILD_RUST_EXTS=none python -m pip install --no-deps --no-build-isolation -e "$E1_SOURCE/python"
python -m pip freeze > "$E1_RUN/pip-freeze.txt"
python -m pip check > "$E1_RUN/pip-check.txt" || true
python -c 'import torch,sglang; print("torch",torch.__version__,"cuda",torch.version.cuda,"sglang",sglang.__version__); print("cuda_available",torch.cuda.is_available()); print(torch.ones(1,device="cuda"))'
