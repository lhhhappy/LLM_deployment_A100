#!/usr/bin/env bash
# Read-only snapshots: JOB [interval_s=1500] [window_min=25] [--once] [--notify]
# Pending jobs and transport failures keep waiting. Only verified complete runs close windows.
set -euo pipefail
exec python3 -B "$(dirname "$0")/window_watch.py" "$@"
