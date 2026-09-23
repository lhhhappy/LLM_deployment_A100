#!/usr/bin/env bash
# Launch a headless Codex worker for ONE task (fresh context per task).
# The user explicitly authorized full-permission Codex workers (2026-09-22).
# Usage: [CODEX_MODEL=gpt-5.6-luna] [CODEX_EFFORT=medium] scripts/codex_worker.sh <worker-name> <prompt-file>
# Choose CODEX_MODEL and CODEX_EFFORT for the task; defaults are below.
# Logs: logs/codex/<worker>.log (full), logs/codex/<worker>.last.md (final message)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
name="$1"; prompt_file="$2"
MODEL="${CODEX_MODEL:-gpt-6-astra}"; EFFORT="${CODEX_EFFORT:-xhigh}"
echo "worker=$name model=$MODEL effort=$EFFORT" >&2
exec codex exec -m "$MODEL" -c model_reasoning_effort="$EFFORT" --skip-git-repo-check \
  -s danger-full-access -C "$ROOT" \
  -o "logs/codex/${name}.last.md" \
  "$(cat "$prompt_file")" > "logs/codex/${name}.log" 2>&1
