#!/usr/bin/env bash
# First container check, AFTER a separately authorized engine has started.
# No service creation, Trisol CLI or submission. --dry-run has no side effects.
set +x
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONDONTWRITEBYTECODE=1
exec python3 -B "$SCRIPT_DIR/preflight_8gpu.py" "$@"
