#!/usr/bin/env bash
# Fetch and verify ONE measured N. See fetch_level.py --help.
set -euo pipefail
exec python3 -B "$(dirname "$0")/fetch_level.py" "$@"
