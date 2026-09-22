#!/usr/bin/env bash
# Wait until the L2 service is running, then bootstrap the pod and queue the first jobs. Never stops/deletes
# the service. Run on the GPU box in tmux:  bash scripts/pod/autostart.sh job1.sh job2.sh ...
D=$(cd "$(dirname "$0")" && pwd); source "$D/common.sh"; cd "$D/../.."
LOG=/sjtu/linhang/arena/runs/autostart.log
echo "$(date -u +%FT%TZ) waiting for $SID" >> $LOG
until bohr trisol inference list --team arena 2>/dev/null | grep "$SID" | grep -q " running "; do sleep 60; done
echo "$(date -u +%FT%TZ) running; waiting for exec" >> $LOG
until bexec "nvidia-smi -L | wc -l" 2>/dev/null | grep -q 8; do sleep 30; done
"$D/bootstrap" >> $LOG 2>&1 && echo "$(date -u +%FT%TZ) bootstrap ok" >> $LOG
for j in "$@"; do "$D/podq" submit "$j" >> $LOG 2>&1; done
echo "$(date -u +%FT%TZ) queued: $*" >> $LOG
