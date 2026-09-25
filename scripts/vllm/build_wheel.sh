#!/usr/bin/env bash
# Build one deployable vLLM wheel from a committed engine/vllm tree (never the working tree).
#   scripts/vllm/build_wheel.sh <commit>
# The tree is exported with `git archive` to the dev box and built there with the official
# wheel's compiled parts (VLLM_USE_PRECOMPILED), so the result is: our Python source + the
# base commit's binaries. Version: 0.30.1rc1.dev114+ga811738a6.ax.<commit12>.
# Output: /sjtu/linhang/arena/vllm/dist/<wheel> and <wheel>.sha256, plus a build receipt.
set -euo pipefail
cd "$(dirname "$0")/../.."
commit=$(git rev-parse --verify "${1:?commit required}^{commit}")
short=${commit:0:12}
V=/sjtu/linhang/arena/vllm
BASE_WHL=vllm-0.30.1rc1.dev114+ga811738a6-cp38-abi3-manylinux_2_28_x86_64.whl
version=0.30.1rc1.dev114+ga811738a6.ax.$short
build=$V/build/$short

git archive --format=tar "$commit" engine/vllm | GSSH_TIMEOUT=600 scripts/gssh \
  "rm -rf $build && mkdir -p $build $V/dist && tar -x -C $build --strip-components=2"
GSSH_TIMEOUT=1800 scripts/vllm/devbox.sh sh "cd $build &&
  VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_LOCATION=$V/wheels/$BASE_WHL \
  VLLM_VERSION_OVERRIDE=$version \
    python -m pip wheel --no-deps --no-build-isolation -w $V/dist . 2>&1 | tail -3
  whl=\$(ls $V/dist/vllm-${version/+/%2B}*.whl $V/dist/vllm-$version-*.whl 2>/dev/null | head -1)
  test -n \"\$whl\"
  sha256sum \"\$whl\" | tee \"\$whl.sha256\"
  python - \"\$whl\" $commit <<'EOF'
import json, os, sys, zipfile
whl, commit = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(whl)
names = z.namelist()
receipt = {
    'wheel': os.path.basename(whl),
    'commit': commit,
    'bytes': os.path.getsize(whl),
    'files': len(names),
    'shared_objects': sorted(n for n in names if n.endswith('.so')),
    'has_generate_compat': any('generate_compat/api_router.py' in n for n in names),
    'has_triton_mla_sparse': any('triton_mla_sparse.py' in n for n in names),
}
with open(whl + '.receipt.json', 'w') as fh:
    json.dump(receipt, fh, indent=1)
print(json.dumps({k: v for k, v in receipt.items() if k != 'shared_objects'}))
EOF
  rm -rf $build"
