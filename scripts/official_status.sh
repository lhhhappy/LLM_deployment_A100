#!/usr/bin/env bash
# Official attempt results (read-only).  scripts/official_status.sh <attempt_id> [...]
# Uses the Playground CLI (`playground status --attempt-id ID --json`); the REST path /api/challenges/.../attempts/ID
# returns the web app HTML, not JSON. Attempt IDs: notes/submission_attempts.log. The stress block is the highest
# PASSING level only.
set -euo pipefail; cd "$(dirname "$0")/.."
. ./env.sh >/dev/null 2>&1 || true
set -a; . ~/.config/playground/credentials.env; set +a
for id in "$@"; do
  timeout 120 playground status --attempt-id "$id" --json | python3 scripts/official_status.py
done
