# Submitted B exactly (base + 000 + 101, D1 on, lpm): does it start and serve >2048-token prompts?
source $AX/bin/scripts/pod/lib.sh
prepare_src b 000-interface-compliance.patch 101-d1v12-on-base.patch
export SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=154827,154829
start_engine --schedule-policy lpm || exit 1
python3 $AX/bin/scripts/m0/probe.py --port $PORT --out $RUN_DIR; rc=$?
stop_engine; exit $rc
