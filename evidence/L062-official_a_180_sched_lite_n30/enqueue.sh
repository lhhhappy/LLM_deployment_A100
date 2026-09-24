#!/usr/bin/env bash
set -euo pipefail
cd /sjtu/linhang/arena/repo
scripts/pod/ppush /tmp/ax/staging/062-180-sched-lite-n30 scripts/pod/jobs/official_a_180_sched_lite_n30.sh patches/180-hicache-glm-dsa.patch evidence/L062-official_a_180_sched_lite_n30/config-audit.json evidence/L062-official_a_180_sched_lite_n30/enqueue_inpod.py
source scripts/pod/common.sh
bexec 'python3 /tmp/ax/staging/062-180-sched-lite-n30/evidence/L062-official_a_180_sched_lite_n30/enqueue_inpod.py'
