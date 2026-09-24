#!/usr/bin/env bash
set -euo pipefail
cd /sjtu/linhang/arena/repo
scripts/pod/ppush /tmp/ax/staging/059-hicache-3b63d9c8 patches/180-hicache-glm-dsa.patch scripts/pod/jobs/official_a_180_hicache_lite_n14.sh evidence/L059-official_a_180_hicache_lite_n14/config-audit.json evidence/L059-official_a_180_hicache_lite_n14/enqueue_inpod.py
source scripts/pod/common.sh
bexec 'python3 /tmp/ax/staging/059-hicache-3b63d9c8/evidence/L059-official_a_180_hicache_lite_n14/enqueue_inpod.py'
