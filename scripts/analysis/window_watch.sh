#!/usr/bin/env bash
# Live window-gate curves for a pod replay job (read-only; diagnostic, never a verdict).
#   scripts/analysis/window_watch.sh <job-name> [interval_s=1500] [window_min=25]
# <job-name> is the queue entry without .sh, e.g. 060-official_a_180_hicache_lite_n30.
# Prints ONE summary line per interval (tables/curves go to files). Every interval it fetches the measured raw file (named by the level's newest checkpoint) through
# pexec_codex (gzip|base64 to stdout; nothing is written on the pod), then renders
#   evidence/L<job>/window/window_gates.{html,json,txt}
# with scripts/analysis/window_gates.py --live. When the job leaves `running`, it renders once more
# without --live and, if the pod saved score_formal.json, checks the whole-run row against it.
set -uo pipefail
job="${1:?job name, e.g. 060-official_a_180_hicache_lite_n30}"; every="${2:-1500}"; win="${3:-25}"
cd "$(dirname "$0")/../.."
run="/tmp/ax/runs/$job"; out="evidence/L$job/window"; mkdir -p "$out"

fetch() {  # -> $out/raw.jsonl (+ $out/score_formal.json when present); returns 1 if no measured raw yet
  local ck raw
  ck=$(timeout 300 scripts/pod/pexec_codex "ls -t $run/N*/checkpoint_*.json 2>/dev/null | head -1" 2>/dev/null | grep '^/tmp/ax/runs/' | head -1)
  [ -n "$ck" ] || return 1
  raw="$(dirname "$ck")/$(basename "$ck" .json | sed 's/^checkpoint_/raw_/').jsonl"
  timeout 600 scripts/pod/pexec_codex "gzip -c $raw | base64 -w0" 2>/dev/null \
    | grep -E '^[A-Za-z0-9+/=]{16,}$' | base64 -d 2>/dev/null | gunzip > "$out/raw.jsonl.tmp" || return 1
  [ -s "$out/raw.jsonl.tmp" ] || return 1
  mv "$out/raw.jsonl.tmp" "$out/raw.jsonl"
  timeout 300 scripts/pod/pread cat "$(dirname "$ck")/score_formal.json" 2>/dev/null \
    | python3 -c 'import json,sys; d=json.loads(sys.stdin.read().split("exit_code:")[0]); json.dump(d, open(sys.argv[1], "w"))' \
      "$out/score_formal.json" 2>/dev/null || rm -f "$out/score_formal.json"
}

render() {  # $1 = --live or empty
  local check=()
  [ -z "$1" ] && [ -s "$out/score_formal.json" ] && check=(--check-score "$out/score_formal.json")
  python3 scripts/analysis/window_gates.py "$out/raw.jsonl" --label "$job" --window-min "$win" $1 \
    > "$out/window_gates.txt"
  # Only one line goes to stdout (it may land in an agent's context); tables and curves stay in files.
  python3 scripts/analysis/window_gates.py "$out/raw.jsonl" --label "$job" --window-min "$win" $1 "${check[@]}" \
    --summary --out-json "$out/window_gates.json" --out-html "$out/window_gates.html" \
    | sed "s/^/[$(date -u +%H:%MZ)] ${1:-final} /"
}

while :; do
  running=$(timeout 300 scripts/pod/pread status 2>/dev/null | grep '^running:' || true)
  if ! grep -q "$job" <<<"$running"; then
    fetch && render "" || echo "[$(date -u +%H:%M:%SZ)] $job not running and no measured raw found"
    echo "WATCH_END $job"; exit 0
  fi
  fetch && render --live || echo "[$(date -u +%H:%M:%SZ)] $job running, no measured raw yet"
  sleep "$every"
done
