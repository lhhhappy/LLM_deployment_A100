#!/usr/bin/env bash
# Official (L3) submission, working around playground-cli 0.1.39 hardcoding trace="[]"
# on attempt creation. Usage (run by the user): bash scripts/submit_official.sh A|B
# Steps: dry-run bundle -> create attempt with the stub trace -> upload bundle via CLI.
# The token is read from ~/.config/playground/credentials.env and never printed.
set -euo pipefail
cd "$(dirname "$0")/.."
. ./env.sh   # NODE_USE_ENV_PROXY=1: the Node CLI ignores the proxy without it
ARM="${1:?usage: submit_official.sh A|B [existing_attempt_id]}"
ID="${2:-}"
CH=llm-challenge-arena-v1
DAY="${DAY:-0922}"
OUT="build/submit_${DAY}_${ARM}"
ZIP="build/submit_${DAY}_${ARM}.zip"
TRACE=submission/stub-trace.jsonl
set -a; . ~/.config/playground/credentials.env; set +a

python3 scripts/check_submission.py --final "$OUT/submission.json" >/dev/null
[ -f "$ZIP" ] || playground submit --challenge-id "$CH" --outputs "$OUT" --trace "$TRACE" --dry-run --bundle-out "$ZIP" >/dev/null
MANIFEST=$(mktemp -d)
unzip -o -q "$ZIP" arm_manifest.json -d "$MANIFEST"
TRACE_JSON=$(python3 -c 'import json,sys;print(json.dumps([json.loads(l) for l in open(sys.argv[1]) if l.strip()]))' "$TRACE")

if [ -z "$ID" ]; then
RESP=$(curl -sS -X POST "https://play.bohrium.com/api/challenges/$CH/attempts" \
  -H "Authorization: Bearer $PLAYGROUND_TOKEN" \
  -F "method=Playground CLI submission" -F "model=claude-opus-5" -F "harness=claude-code" \
  -F "type=agent" -F "status=submitted" -F "detail=official ${ARM} $(date -u +%F)" \
  -F "manifest_json=<$MANIFEST/arm_manifest.json" \
  --form-string "trace=$TRACE_JSON" -F "author_name=Playground CLI")
ID=$(python3 -c 'import json,sys;print(json.loads(sys.argv[1]).get("id",""))' "$RESP" 2>/dev/null || true)
if [ -z "$ID" ]; then echo "attempt create failed: $(echo "$RESP" | head -c 400)"; exit 1; fi
echo "attempt_id=$ID"
echo "$(date -u +%FT%TZ) arm=$ARM attempt_id=$ID created" >> notes/submission_attempts.log
fi
playground submit --challenge-id "$CH" --attempt-id "$ID" --bundle "$ZIP" --trace "$TRACE"
echo "$(date -u +%FT%TZ) arm=$ARM attempt_id=$ID uploaded" >> notes/submission_attempts.log
