# Shared settings for the pod tools (run on the GPU box). See scripts/pod/README.md.
_pod_cwd=$PWD; source /sjtu/linhang/arena/daemon_env.sh >/dev/null 2>&1; cd "$_pod_cwd"   # daemon_env.sh cd's into the repo
SID=${L2_SERVICE_ID:-2102309548588015616}
POD_AX=/tmp/ax                     # pod workspace: src/, queue/, runs/, worker.log
bexec() { timeout "${PEXEC_TIMEOUT:-120}" bohr trisol inference exec "$SID" --team arena -- bash -c "$1"; }
