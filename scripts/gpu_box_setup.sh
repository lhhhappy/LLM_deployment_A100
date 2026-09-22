#!/usr/bin/env bash
# One-time setup of the always-on daemon host (GPU dev box). Run ON the GPU box:
#   bash /sjtu/linhang/arena/repo/scripts/gpu_box_setup.sh
# Installs bohr CLI + trisol extension + playground CLI under /sjtu (root disk is ~3 GB),
# and writes /sjtu/linhang/arena/daemon_env.sh. Logins are done by the human afterwards.
set -euo pipefail
A=/sjtu/linhang/arena
mkdir -p $A/npm-global $A/cache/npm
export npm_config_prefix=$A/npm-global npm_config_cache=$A/cache/npm
export PATH=$A/npm-global/bin:$PATH
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY   # this box reaches npm/trisol/bohrium directly
npm install -g @dptech-corp/bohr-cli@latest
curl -fsSL http://play.bohrium.com/install.sh | bash || true   # playground CLI (may re-install bohr; fine)
bohr --version; playground --version || true
cat > $A/daemon_env.sh <<ENV
# source /sjtu/linhang/arena/daemon_env.sh   (daemon host environment)
export PATH=$A/npm-global/bin:\$PATH
export npm_config_prefix=$A/npm-global
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export ARENA_ROOT=$A/repo
cd $A/repo
ENV
echo "OK. Next: source $A/daemon_env.sh; then do the logins (see README)."
