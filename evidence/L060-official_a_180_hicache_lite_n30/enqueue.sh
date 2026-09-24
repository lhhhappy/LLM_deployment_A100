#!/usr/bin/env bash
set -euo pipefail
cd /sjtu/linhang/arena/repo
scripts/pod/ppush /tmp/ax/staging/060-hicache-n30 scripts/pod/jobs/official_a_180_hicache_lite_n30.sh evidence/L060-official_a_180_hicache_lite_n30/config-audit.json evidence/L060-official_a_180_hicache_lite_n30/enqueue_inpod.py
source scripts/pod/common.sh
bexec 'python3 /tmp/ax/staging/060-hicache-n30/evidence/L060-official_a_180_hicache_lite_n30/enqueue_inpod.py'
