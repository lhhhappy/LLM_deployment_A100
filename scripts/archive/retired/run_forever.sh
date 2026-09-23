#!/usr/bin/env bash
# T27 unattended supervisor. It never invents queue items; STOP is honoured.
set -u
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
mkdir -p logs data
PIDFILE="$ROOT/data/trisol_test_forever.pid"
STOPFILE="$ROOT/tests/queue/STOP"
if [ -s "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "already running: $(cat "$PIDFILE")" >&2
  exit 1
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT INT TERM
while [ ! -e "$STOPFILE" ]; do
  python3 -u -B scripts/trisol_test_daemon.py --config tests/trisol_test_config.json \
    >> logs/trisol_test_daemon.log 2>&1 || true
  [ -e "$STOPFILE" ] && break
  sleep 60
done
