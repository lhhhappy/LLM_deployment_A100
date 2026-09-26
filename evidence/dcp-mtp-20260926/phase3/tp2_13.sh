set -euo pipefail
DCP_R=/sjtu/linhang/arena/runs/dcp-mtp-20260926
export DCP_DEVICES=GPU-0fd597b1-c02f-3a31-8fa7-8ecb15791ca8,GPU-14c26f76-e5a4-9d1f-5d33-9ff397af4933
export DCP_TP=2 DCP_MTP_MODEL="$DCP_R/model-mtp-r2"
export OMP_NUM_THREADS=8
export LD_LIBRARY_PATH=/sjtu/linhang/arena/env/cuda-compat-13-0/usr/local/cuda-13.0/compat:/sjtu/linhang/arena/cache/T48/deps/z3/lib
DCP_PY=/sjtu/linhang/arena/env/m0/bin/python
preflight() {
  python3 - "$DCP_R/tp2_13-resources.jsonl" <<'PY'
import subprocess,json,sys,time
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
receipt={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'rows':rows}
with open(sys.argv[1],'a') as f:f.write(json.dumps(receipt)+'\n')
assert len(rows)==2 and all(int(row.rsplit(',',1)[-1].strip())>65000 for row in rows), receipt
PY
}
arm() {
  local label=$1 width=$2 graph=$3
  preflight
  echo "START $label"
  timeout --signal=TERM --kill-after=10s 420s bash "$DCP_R/run_dcp_devbox.sh" "$DCP_R" "$DCP_R/fix5/engine" "$label" "$width" "$graph" 0 > "$DCP_R/$label.log" 2>&1
  echo "COMPLETE $label"
}
compare() {
  "$DCP_PY" "$DCP_R/dcp_mtp_compare.py" "$DCP_R/$1" "$DCP_R/$2" --output "$DCP_R/$3.json" > "$DCP_R/$3.log"
}
DCP_PROBE=move arm move13 2 0
DCP_PROBE=mtp arm mtp_tp2_ref13 1 0
DCP_PROBE=mtp arm mtp_tp2_dcp13 2 1
compare mtp_tp2_dcp13 mtp_tp2_ref13 compare_mtp13
DCP_PROBE=mtp DCP_ACCEPT_ORACLE="$DCP_R/mtp_tp2_ref13/responses.json" arm mtp_tp2_accept_ref13 1 1
DCP_PROBE=mtp DCP_ACCEPT_ORACLE="$DCP_R/mtp_tp2_ref13/responses.json" arm mtp_tp2_accept_dcp13 2 1
compare mtp_tp2_accept_dcp13 mtp_tp2_accept_ref13 compare_accept13
DCP_PROBE=mtp DCP_ACCEPT_ORACLE="$DCP_R/mtp_tp2_ref13/responses.json" SGLANG_AX_DCP_COMPACT_TOPK=1 arm mtp_tp2_compact13 2 1
compare mtp_tp2_compact13 mtp_tp2_accept_dcp13 compare_compact13
DCP_PROBE=mtp DCP_HICACHE=1 DCP_RESTORE=1 arm mtp_tp2_host_ref13 1 1
DCP_PROBE=mtp DCP_HICACHE=1 DCP_RESTORE=1 arm mtp_tp2_host_dcp13 2 1
compare mtp_tp2_host_dcp13 mtp_tp2_host_ref13 compare_host13
preflight
DCP_PROBE=attention timeout --signal=TERM --kill-after=10s 420s bash "$DCP_R/run_dcp_devbox.sh" "$DCP_R" "$DCP_R/fix5/engine" high_graph13 2 1 0.6 > "$DCP_R/high_graph13.log" 2>&1
"$DCP_PY" "$DCP_R/dcp_compare.py" "$DCP_R/high_graph13.pt" "$DCP_R/ref4.pt" --require-high-loc --require-graph > "$DCP_R/compare_high_graph13.log"
echo 'ALL_DIAGNOSTICS_COMPLETED'
