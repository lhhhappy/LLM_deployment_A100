#!/usr/bin/env bash
set -uo pipefail
cd /workspace/Agentic_science_challenge
scripts/gssh 'set -e
cd /sjtu/linhang/arena/runs/hc180
source /sjtu/linhang/arena/env.sh >/dev/null 2>&1
C=/sjtu/linhang/arena/env/sgl/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME=$C CUDA_PATH=$C PATH=$C/bin:/sjtu/linhang/arena/env/sgl/bin:$PATH LD_LIBRARY_PATH=/sjtu/linhang/arena/env/cuda-compat-13-0/usr/local/cuda-13.0/compat:$C/lib CUDA_VISIBLE_DEVICES=0 HC180_DEVICE=cuda
date -u +START_UTC=%Y-%m-%dT%H:%M:%SZ
sha256sum src/patches/180-hicache-glm-dsa.patch src/scripts/tests/hicache180/test_180_glm_host_pools.py src/scripts/tests/hicache180/test_180_tree_hicache_e2e.py
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory --format=csv,noheader
/sjtu/linhang/arena/env/sgl/bin/python3 -c "import torch; print(\"torch\",torch.__version__,\"cuda\",torch.cuda.is_available(),\"device\",torch.cuda.get_device_name(0))"
cd src/scripts/tests/hicache180
PYTHONPATH=/sjtu/linhang/arena/runs/hc180/t180:. /sjtu/linhang/arena/env/sgl/bin/python3 -m unittest -v test_180_glm_host_pools.TestGlmHostTier.test_round_trip_restores_every_component_byte_exact test_180_tree_hicache_e2e.TestTreeHiCacheIndexerOwnership.test_fork_inside_group_does_not_overwrite_sibling test_180_tree_hicache_e2e.TestTreeHiCacheIndexerOwnership.test_flush_with_write_through_in_flight
date -u +END_UTC=%Y-%m-%dT%H:%M:%SZ' > evidence/HC180-GPU-contract/targeted.log 2>&1
rc=$?
printf 'SSH_AND_TEST_RC=%s\n' "$rc" >> evidence/HC180-GPU-contract/targeted.log
exit "$rc"
