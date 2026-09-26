set -uo pipefail
DCP_R=/sjtu/linhang/arena/runs/dcp-mtp-20260926
export DCP_DEVICES=GPU-0fd597b1-c02f-3a31-8fa7-8ecb15791ca8
export DCP_TP=1 DCP_MTP_MODEL="$DCP_R/model-mtp-r2"
nvidia-smi --id="$DCP_DEVICES" --query-gpu=timestamp,uuid,memory.used,utilization.gpu,clocks.sm,power.draw --format=csv,noheader -l 5 > "$DCP_R/gpu10-occupancy.csv" &
DCP_MONITOR_PID=$!
trap 'kill "$DCP_MONITOR_PID" 2>/dev/null || true' EXIT
DCP_PROBE=sparse bash "$DCP_R/run_dcp_devbox.sh" "$DCP_R" "$DCP_R/fix4/engine" sparse10 1 0 0 > "$DCP_R/sparse10.log" 2>&1
DCP_SPARSE_RC=$?
echo "sparse rc=$DCP_SPARSE_RC"
DCP_PROBE=mtp bash "$DCP_R/run_dcp_devbox.sh" "$DCP_R" "$DCP_R/fix4/engine" mtp_tp1_ref10 1 0 0 > "$DCP_R/mtp_tp1_ref10.log" 2>&1
DCP_REF_RC=$?
echo "mtp natural rc=$DCP_REF_RC"
[[ $DCP_REF_RC = 0 ]] || exit "$DCP_REF_RC"
DCP_PROBE=mtp DCP_ACCEPT_ORACLE="$DCP_R/mtp_tp1_ref10/responses.json" bash "$DCP_R/run_dcp_devbox.sh" "$DCP_R" "$DCP_R/fix4/engine" mtp_tp1_accept10 1 1 0 > "$DCP_R/mtp_tp1_accept10.log" 2>&1
DCP_ACCEPT_RC=$?
echo "mtp acceptance rc=$DCP_ACCEPT_RC"
DCP_PROBE=mtp DCP_HICACHE=1 DCP_RESTORE=1 bash "$DCP_R/run_dcp_devbox.sh" "$DCP_R" "$DCP_R/fix4/engine" mtp_tp1_host10 1 1 0 > "$DCP_R/mtp_tp1_host10.log" 2>&1
DCP_HOST_RC=$?
echo "mtp host restore rc=$DCP_HOST_RC"
[[ $DCP_SPARSE_RC = 0 && $DCP_ACCEPT_RC = 0 && $DCP_HOST_RC = 0 ]]
