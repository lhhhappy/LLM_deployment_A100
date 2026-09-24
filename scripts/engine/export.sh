#!/usr/bin/env bash
# Export one engine commit for the pod: the diff from the organizer base to that commit.
#   scripts/engine/export.sh <ref>        e.g. official-A-0923a, HEAD, a branch or a commit
# Prints the full commit id and writes build/engine/<commit>.diff. The pod applies it to its own copy of the
# base (scripts/pod/lib.sh prepare_src) exactly like the old patch stack: patch -p3 --fuzz=0 in the package dir.
# Refuses a dirty engine/ tree so the id always names the exact code.
set -euo pipefail
cd "$(dirname "$0")/../.."
ref="${1:?ref, e.g. official-A-0923a or HEAD}"
commit=$(git rev-parse --verify "$ref^{commit}")
if [ "$ref" = HEAD ] && [ -n "$(git status --porcelain -- engine)" ]; then
  echo "engine/ has uncommitted changes; commit them first" >&2; exit 2
fi
mkdir -p build/engine
git diff --no-color --binary engine-base "$commit" -- engine/sglang > "build/engine/$commit.diff"
echo "$commit"
