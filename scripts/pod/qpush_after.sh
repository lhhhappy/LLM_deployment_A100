#!/usr/bin/env bash
# qpush, deferred: the GPU-box job first waits until the pod queue has no running and no pending job and the named
# job is in done/failed, then pauses the queue, deploys the bundle (runtime + engine diffs + jobs) and resumes.
# Use it while another job (or another deferred publish) is in flight; plain qpush needs an idle, paused queue.
#   scripts/pod/qpush_after.sh <done-job-name-substring> queue-name.sh=scripts/pod/jobs/file.sh ...
# Watch with scripts/gjob tail <DEPLOYMENT_JOB>; RUNTIME_DEPLOYED then "resumed" means the jobs are queued.
set -euo pipefail
cd "$(dirname "$0")/../.."
wait_for=${1:?done-job-name substring to wait for}; shift
ignore_pending=0
if [ "${1:-}" = "--ignore-pending" ]; then ignore_pending=1; shift; fi   # insert among pending jobs (name order decides)
[ "$#" -gt 0 ] || { echo 'qpush_after <wait-for> queue-name.sh=scripts/pod/jobs/file.sh ...'; exit 2; }
bash scripts/pod/verify/make_kit.sh >/dev/null
id="queue-after-$(date -u +%Y%m%dT%H%M%S)-$$"
bundle="build/queue/$id"
python3 -B scripts/pod/queue_bundle.py build "$bundle" "$@"
cat > "$bundle/deploy.sh" <<DEPLOY
set -euo pipefail
cd /sjtu/linhang/arena/repo
source scripts/pod/common.sh
state() { PEXEC_TIMEOUT=60 bexec 'cd /tmp/ax/queue && echo "running=\$(ls running 2>/dev/null | tr "\n" " ")" && echo "pending=\$(ls pending 2>/dev/null | tr "\n" " ")" && echo "finished=\$(ls done failed 2>/dev/null | tr "\n" " ")"'; }
pushed=0
allow_flag=""; [ "$ignore_pending" = 1 ] && allow_flag="--allow-pending"
# Inserting among pending jobs: pause now, so the worker does not start the next pending job in the seconds
# between the target job ending and this publish (it did on 2026-09-26: 134 started before 130a could be queued).
# Early pause only on request: it blocks the worker from starting the pending jobs the publisher is waiting for
# (three deadlocks on 2026-09-27); ordered job names make it unnecessary.
[ "$ignore_pending" = 1 ] && [ "${QPUSH_EARLY_PAUSE:-0}" = 1 ] && bexec 'touch /tmp/ax/queue/PAUSE && echo paused-early'
while :; do
  s=\$(state) || { echo "[wait] \$(date -u +%FT%TZ) state read failed; retry in 120 s"; sleep 120; continue; }
  echo "[wait] \$(date -u +%FT%TZ) \$(tr '\n' ' ' <<<"\$s")"
  if grep -q '^running= *\$' <<<"\$s" && { [ "$ignore_pending" = 1 ] || grep -q '^pending= *\$' <<<"\$s"; } && grep -q "^finished=.*$wait_for" <<<"\$s"; then
    # idle: pause, deploy, publish; if another publisher got in first the publish asserts and we resume and retry
    bexec 'touch /tmp/ax/queue/PAUSE && echo paused'
    [ "\$pushed" = 1 ] || { scripts/pod/ppush /tmp/ax/staging/$id $bundle && pushed=1; }
    if bexec "python3 /tmp/ax/staging/$id/$bundle/queue_bundle.py publish /tmp/ax/staging/$id/$bundle \$allow_flag"; then
      bexec 'rm -f /tmp/ax/queue/PAUSE && echo resumed'
      break
    fi
    echo "[wait] publish refused (queue changed?); resuming and retrying in 120 s"
    bexec 'rm -f /tmp/ax/queue/PAUSE && echo resumed'
  fi
  sleep 120
done
DEPLOY
tar cf - "$bundle" scripts/pod | GSSH_TIMEOUT=${QPUSH_TAR_TIMEOUT:-120} scripts/gssh 'tar xf - -C /sjtu/linhang/arena/repo'
scripts/gjob run "$id" "cd /sjtu/linhang/arena/repo && bash $bundle/deploy.sh"
echo "DEPLOYMENT_JOB=$id (gjob tail; waits for '$wait_for' in done/failed and an empty queue, then publishes and resumes)"
