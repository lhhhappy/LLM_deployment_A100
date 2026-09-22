# M0 on real hardware: stock base image code + the submitted command (no patches). Expect: startup
# failure in DSA indexer CUDA-graph capture (DeepGEMM "Unsupported architecture" on sm80).
source $AX/bin/scripts/pod/lib.sh
prepare_src stock
start_engine --schedule-policy lpm || exit 1
python3 $AX/bin/scripts/m0/probe.py --port $PORT --out $RUN_DIR; rc=$?
stop_engine; exit $rc
