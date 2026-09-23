# Pod verification of all dev-box kernel conclusions on the served image (run BEFORE the engine starts).
source $AX/bin/scripts/pod/lib.sh
stop_engine   # GPUs must be free; the engine is restarted by the next job (ensure_engine)
prepare_src bverify 000-interface-compliance.patch 101-role-boundary-split.patch 106-defer-chunk-on-no-kv.patch \
  110-sm80-dsa-indexer.patch 111-sm80-fp8-moe-marlin.patch 120-sched-protect-chain.patch \
  140-kda-dual-snapshot.patch 160-nextn-sm80.patch || exit 1
bash $AX/verify_kit/run_verify.sh bverify 0
