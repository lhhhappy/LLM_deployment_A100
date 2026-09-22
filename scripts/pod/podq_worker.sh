#!/usr/bin/env bash
# Runs inside the pod. Executes /tmp/ax/queue/pending/*.sh one at a time in name order.
# Job output: /tmp/ax/runs/<job>/job.log, exit code in /tmp/ax/runs/<job>/exit_code.
AX=/tmp/ax; mkdir -p $AX/queue/{pending,running,done,failed} $AX/runs
echo "$$" > $AX/worker.pid
while true; do
  [ -e $AX/queue/PAUSE ] && { sleep 10; continue; }
  job=$(ls $AX/queue/pending 2>/dev/null | sort | head -1)
  [ -z "$job" ] && { sleep 5; continue; }
  name=${job%.sh}; mkdir -p $AX/runs/$name
  mv $AX/queue/pending/$job $AX/queue/running/$job
  echo "$(date -u +%FT%TZ) start $name" >> $AX/worker.log
  ( cd $AX/runs/$name && RUN_DIR=$AX/runs/$name AX=$AX bash $AX/queue/running/$job > job.log 2>&1 ); rc=$?
  echo $rc > $AX/runs/$name/exit_code
  [ $rc -eq 0 ] && mv $AX/queue/running/$job $AX/queue/done/$job || mv $AX/queue/running/$job $AX/queue/failed/$job
  echo "$(date -u +%FT%TZ) end $name rc=$rc" >> $AX/worker.log
done
