#!/usr/bin/env bash
# Drive an already-running, separately authorized session A hang pod.
set +x
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONDONTWRITEBYTECODE=1
exec python3 -B "$SCRIPT_DIR/runner.py" "$@"
