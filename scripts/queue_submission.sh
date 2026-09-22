#!/usr/bin/env bash
# T22: snapshot a four-field candidate into the queue; never grant approval.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 CANDIDATE_JSON SLUG [TRACE_JSONL] [NOTES_MD]" >&2
  exit 2
fi
queue_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
queue_args=(--enqueue "$1" --slug "$2")
if [[ $# -ge 3 ]]; then queue_args+=(--trace "$3"); fi
if [[ $# -ge 4 ]]; then queue_args+=(--notes "$4"); fi
exec python3 -B "$queue_root/scripts/submit_daemon.py" "${queue_args[@]}"
