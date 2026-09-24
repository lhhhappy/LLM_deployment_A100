#!/usr/bin/env bash
set -euo pipefail
cd /sjtu/linhang/arena/repo
scripts/pod/ppush /tmp/ax/staging/060b-baseline-lite-n30 scripts/pod/jobs/official_a_longchain_lite_n30.sh evidence/L060b-official_a_longchain_lite_n30/config-audit.json evidence/L060b-official_a_longchain_lite_n30/enqueue_inpod.py
source scripts/pod/common.sh
bexec 'python3 /tmp/ax/staging/060b-baseline-lite-n30/evidence/L060b-official_a_longchain_lite_n30/enqueue_inpod.py'
