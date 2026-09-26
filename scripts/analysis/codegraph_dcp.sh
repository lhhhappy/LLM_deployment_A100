#!/usr/bin/env bash
# Query the local CodeGraph index of this worktree's SGLang source.
# The standalone binary lives outside the branch under the shared build/tools.
set -euo pipefail
DCP_WORKTREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DCP_COMMON_GIT=$(git -C "$DCP_WORKTREE" rev-parse --path-format=absolute --git-common-dir)
DCP_CODEGRAPH="$(dirname "$DCP_COMMON_GIT")/build/tools/bin/codegraph"
[[ -x "$DCP_CODEGRAPH" ]] || { echo "CodeGraph is not installed at $DCP_CODEGRAPH" >&2; exit 2; }
export CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 CODEGRAPH_NO_DAEMON=1
cd "$DCP_WORKTREE/engine/sglang"
exec "$DCP_CODEGRAPH" "$@"
