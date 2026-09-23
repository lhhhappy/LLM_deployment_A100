#!/usr/bin/env bash
# (runs INSIDE the pod) Stop ONE queue job and the dev-set load it started. Never touches the service, the queue
# worker or the engine. Usage: bash /tmp/ax/bin/stopjob_inpod.sh <job-file-name, e.g. 026-ladder_best140.sh>
J=$1; me=$$; par=$PPID
[ -n "$J" ] || { echo "usage: stopjob <job.sh>"; exit 2; }
jobpids=$(ps -eo pid,args | awk -v j="$J" -v me=$me -v par=$par '$0 ~ j && $1!=me && $1!=par && $0 !~ /stopjob_inpod/ {print $1}')
[ -n "$jobpids" ] || { echo "job $J not running"; exit 0; }
kids=""; for p in $jobpids; do kids="$kids $(pgrep -P $p)"; done
# dev-set load processes (run_dev / s1_loadgen) are descendants of the job; kill them and the job shell only
desc=$(ps -eo pid,args | awk '/run_dev\.py|s1_loadgen\.py|s1_score\.py/ && !/awk/ {print $1}')
echo "stopping job $J: job=$jobpids load=$desc"
kill $desc $jobpids 2>/dev/null; sleep 3; kill -9 $desc $jobpids 2>/dev/null; echo done
