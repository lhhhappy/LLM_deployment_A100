#!/usr/bin/env bash
# Build a runtime bundle of an engine/vllm commit on the dev box: the organizer vLLM image's
# Python environment (its dist-packages, from the rootfs copy made by image_export.sh) patched by
# the same apply.sh the submission Dockerfile embeds, plus a relocatable venv that runs it with the
# target's /usr/bin/python3.12. Meant for the 8-GPU pod, whose SGLang image has the same Ubuntu
# 24.04 system packages (glibc 2.39, Python 3.12.3-1ubuntu0.16, CUDA 13.0 runtime; evidence/
# vllm-backport-identity-20260925), so nothing is installed there: unpack into a RAM-disk directory,
#   export AX_RUNTIME_DIR=<RAM-disk dir>; source <bundle>/vllm-env.sh; vllm serve --model ...
#   scripts/vllm/bundle.sh <commit>        (run locally; builds under scripts/gjob on the dev box)
# Output on the dev box: $OUT/<commit12>/ (tree) and $OUT/<commit12>.tar.gz (+ .sha256, MANIFEST).
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOTFS=/sjtu/linhang/arena/vllm-backport/260918-sm80/rootfs
OUT=/sjtu/linhang/arena/vllm-backport/bundles
DEV_PY=/sjtu/linhang/arena/env/sgl/bin/python3.12   # py_compile only; same bytecode as 3.12.3
COMMIT=$(git rev-parse --verify "${1:?commit required}^{commit}")
SHORT=${COMMIT:0:12}
STAGE=build/scratch/vllm-bundle/$SHORT
rm -rf "$STAGE" && mkdir -p "$STAGE/env/bin"
python3 scripts/vllm/make_apply.py "$COMMIT" -o "$STAGE/apply.sh"

cat > "$STAGE/env/pyvenv.cfg" <<'EOF'
home = /usr/bin
include-system-site-packages = false
version = 3.12.3
EOF
ln -s /usr/bin/python3.12 "$STAGE/env/bin/python3"
ln -s python3 "$STAGE/env/bin/python"
cat > "$STAGE/env/bin/vllm" <<'EOF'
#!/bin/sh
# The image's `vllm` console script, bound to this venv's interpreter.
exec "$(dirname "$0")/python3" -c 'import sys; from vllm.entrypoints.cli.main import main; sys.argv[0] = "vllm"; sys.exit(main())' "$@"
EOF
chmod +x "$STAGE/env/bin/vllm"
cat > "$STAGE/vllm-env.sh" <<EOF
# Source to run engine/vllm $COMMIT (organizer vLLM image environment) with /usr/bin/python3.12.
# AX_RUNTIME_DIR must be a RAM-disk directory: every cache and temp file goes there, none to the
# root disk. Variables are the organizer image's ENV that affect vLLM, plus the cache locations.
_b=\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)
: "\${AX_RUNTIME_DIR:?set AX_RUNTIME_DIR to a RAM-disk directory}"
mkdir -p "\$AX_RUNTIME_DIR"/tmp "\$AX_RUNTIME_DIR"/cache "\$AX_RUNTIME_DIR"/config
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export PATH="\$_b/env/bin:/usr/local/cuda/bin:\$PATH"
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib
export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9"
export VLLM_PLUGINS=s1_generate,lora_filesystem_resolver,lora_hf_hub_resolver
export VLLM_ENABLE_CUDA_COMPATIBILITY=0 VLLM_USAGE_SOURCE=production-docker-image
export AX_ENGINE_COMMIT=$COMMIT
export TMPDIR="\$AX_RUNTIME_DIR/tmp" XDG_CACHE_HOME="\$AX_RUNTIME_DIR/cache"
export VLLM_CACHE_ROOT="\$AX_RUNTIME_DIR/cache/vllm" VLLM_CONFIG_ROOT="\$AX_RUNTIME_DIR/config"
export TRITON_CACHE_DIR="\$AX_RUNTIME_DIR/cache/triton" TORCHINDUCTOR_CACHE_DIR="\$AX_RUNTIME_DIR/cache/inductor"
export FLASHINFER_WORKSPACE_BASE="\$AX_RUNTIME_DIR/cache" TILELANG_CACHE_DIR="\$AX_RUNTIME_DIR/cache/tilelang"
export HF_HOME="\$AX_RUNTIME_DIR/cache/huggingface" HF_HUB_OFFLINE=1
unset _b
EOF

GSSH_TIMEOUT=60 scripts/gssh "mkdir -p $OUT"
rsync -a --delete "$STAGE/" "GPU:$OUT/$SHORT.stage/"
scripts/gjob run "vllm-bundle-$SHORT" "set -euo pipefail
  S=$OUT/$SHORT.stage B=$OUT/$SHORT
  test -f $ROOTFS/usr/local/lib/python3.12/dist-packages/vllm-0.13.1.dist-info/RECORD
  rm -rf \$B && mkdir -p \$B/env/lib/python3.12
  cp -al $ROOTFS/usr/local/lib/python3.12/dist-packages \$B/env/lib/python3.12/site-packages
  cp -a \$S/env/pyvenv.cfg \$S/env/bin \$B/env/ && cp -a \$S/apply.sh \$S/vllm-env.sh \$B/
  sh \$B/apply.sh \$B/env/lib/python3.12/site-packages $DEV_PY
  cmp $ROOTFS/usr/local/lib/python3.12/dist-packages/vllm-0.13.1.dist-info/RECORD \
      \$B/env/lib/python3.12/site-packages/vllm-0.13.1.dist-info/RECORD
  printf '{\"commit\": \"$COMMIT\", \"base\": \"vllm-base-backport-v0.13.1\", \"image\": \"vllm-backport:260918-sm80\", \"apply_sha256\": \"%s\", \"files\": %s, \"bytes\": %s}\n' \
    \$(sha256sum < \$B/apply.sh | cut -c1-64) \$(find \$B -type f | wc -l) \$(du -sb \$B | cut -f1) > \$B/MANIFEST.json
  rm -rf \$S
  tar -C $OUT -cf - $SHORT | gzip -4 > $OUT/$SHORT.tar.gz.partial && mv $OUT/$SHORT.tar.gz.partial $OUT/$SHORT.tar.gz
  (cd $OUT && sha256sum $SHORT.tar.gz > $SHORT.tar.gz.sha256)
  cat \$B/MANIFEST.json; ls -l $OUT/$SHORT.tar.gz; echo BUNDLE_DONE"
echo "started vllm-bundle-$SHORT; follow with: scripts/gjob wait vllm-bundle-$SHORT"
