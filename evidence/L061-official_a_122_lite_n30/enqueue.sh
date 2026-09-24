#!/usr/bin/env bash
set -euo pipefail
cd /sjtu/linhang/arena/repo
scripts/pod/ppush /tmp/ax/staging/061-122-lite-n30 scripts/pod/jobs/official_a_122_lite_n30.sh evidence/L061-official_a_122_lite_n30/config-audit.json evidence/L061-official_a_122_lite_n30/enqueue_inpod.py
source scripts/pod/common.sh
bexec 'python3 /tmp/ax/staging/061-122-lite-n30/evidence/L061-official_a_122_lite_n30/enqueue_inpod.py'
