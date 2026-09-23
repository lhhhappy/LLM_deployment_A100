#!/usr/bin/env bash
# Fetch one finished dev level from the pod and analyze it locally (read-only on the pod; CPU only here).
#   scripts/analysis/fetch_level.sh <run dir, e.g. 037-s0_i16_n22> <N> [reference raw for per-request comparison]
# Writes evidence/L<run>/{raw,run,summary,server.log}, then prints:
#   - the verdict (scripts/pod/verify/level_verdict.py: every request once + harness scorer + task.md rules)
#   - the bad-case analysis (scripts/analysis/badcase.py)
set -euo pipefail
cd "$(dirname "$0")/../.."
run=$1; N=$2; ref=${3:-}
out=evidence/L${run%%-*}; mkdir -p "$out"
D=/tmp/ax/runs/$run; A=/tmp/ax/codex/fetch_${run}.tgz; px() { GSSH_TIMEOUT=600 scripts/gssh "cd /sjtu/linhang/arena/repo && scripts/pod/pexec_codex '$1'" 2>/dev/null; }
# 1. pack on the pod (pexec_codex may only write /tmp/ax/codex), report size + sha256
meta=$(px "cd $D/N$N && r=\$(python3 -c \"import json;print(json.load(open(\\\"summary.json\\\"))[\\\"raw\\\"].split(\\\"/\\\")[-1])\") && j=\$(python3 -c \"import json;print(json.load(open(\\\"summary.json\\\"))[\\\"run\\\"].split(\\\"/\\\")[-1])\") && tar czf $A \$r \$j summary.json -C $D server.log job.log && echo META \$(stat -c %s $A) \$(sha256sum $A | cut -c1-64)" | grep -o "META [0-9]* [0-9a-f]*")
[ -n "$meta" ] || { echo "packing failed (run dir / level finished?)"; exit 2; }
size=$(echo $meta | cut -d" " -f2); sha=$(echo $meta | cut -d" " -f3); chunk=120000; n=$(( (size + chunk - 1) / chunk ))
# 2. fetch in chunks, 3. verify the sha256 before extracting (a truncated transfer must fail, not be patched over)
: > "$out/.fetch.tgz"
for i in $(seq 0 $((n - 1))); do
  px "dd if=$A bs=$chunk skip=$i count=1 status=none | base64 -w0; echo; echo END" | awk '/^END$/{exit} {print}' | tr -d '\n' | base64 -d >> "$out/.fetch.tgz"
done
got=$(sha256sum "$out/.fetch.tgz" | cut -c1-64)
[ "$got" = "$sha" ] || { echo "transfer corrupt: sha $got != $sha (size $(stat -c %s "$out/.fetch.tgz") of $size)"; exit 2; }
tar xzf "$out/.fetch.tgz" -C "$out" && rm -f "$out/.fetch.tgz"
raw=$(ls "$out"/raw_*.jsonl | head -1)
k=$(mktemp -d); cp scripts/pod/verify/level_verdict.py scripts/score_formal.py "$k/"
python3 -B "$k/level_verdict.py" "$out" "$N" --harness-dir s1-dev/harness --data-root s1-dev/data/dev-combined-v1 | tee "$out/verdict.txt" || true
rm -rf "$k"
python3 scripts/analysis/badcase.py "$raw" "$out/server.log" ${ref:+--ref "$ref"} --csv "$out/badcases.csv" | tee "$out/badcase.txt"
