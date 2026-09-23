# Pod verification of all dev-box kernel conclusions on the served image (run BEFORE the engine starts).
source $AX/bin/scripts/pod/lib.sh
stop_engine   # GPUs must be free; the engine is restarted by the next job (ensure_engine)
prepare_src bverify 000-interface-compliance.patch 101-d1v12-on-base.patch 105-role-split-single-partial.patch \
  110-sm80-dsa-indexer.patch 111-sm80-fp8-moe-marlin.patch 112-sm80-indexer-kernels.patch 113-sm80-prefill-indexer.patch \
  140-kda-dual-snapshot.patch 120-sched-protect-chain.patch 130-async-tokenize.patch 150-startup-warmup.patch 160-nextn-sm80.patch || exit 1
bash $AX/verify_kit/run_verify.sh bverify 0
