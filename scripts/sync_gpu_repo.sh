#!/usr/bin/env bash
# Sync committed, managed files to the GPU-box mirror after the pod queue is idle.
# Default: rsync dry run. --apply changes only managed directories and listed exact stale paths.
# Never touches pod/Trisol or GPU-box runs, models, env, cache, evidence, or protected source inputs.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
case "${1:-}" in
  "") RSYNC_ARGS=(-aivn) ;;
  --apply) RSYNC_ARGS=(-aiv) ;;
  *) echo "usage: scripts/sync_gpu_repo.sh [--apply]" >&2; exit 2 ;;
esac
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "Commit the local cleanup before syncing: this tool sends committed HEAD." >&2
  exit 2
fi

DEST=/sjtu/linhang/arena/repo
MANAGED=(scripts engine patches tests notes research docs plans cases submission data)
ROOT_FILES=(README.md AGENTS.md CLAUDE.md .gitignore)
ARCHIVE=()
for path in "${MANAGED[@]}" "${ROOT_FILES[@]}"; do
  if git cat-file -e "HEAD:$path" 2>/dev/null; then ARCHIVE+=("$path"); fi
done
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
git archive HEAD -- "${ARCHIVE[@]}" | tar -xf - -C "$STAGE"

echo "Source: committed HEAD $(git rev-parse --short HEAD)"
echo "Destination: GPU:$DEST"
echo "Managed directories: ${MANAGED[*]}"
echo "Preserved outside managed directories: s1-dev, llm-challenge-arena-v1, build (except three exact stale paths), refs (except the exact harness clone), src, evidence, runs, models, env, cache, logs"

for path in "${MANAGED[@]}"; do
  mkdir -p "$STAGE/$path"
  rsync "${RSYNC_ARGS[@]}" --delete \
    -e 'ssh -o BatchMode=yes -o ConnectTimeout=15' \
    "$STAGE/$path/" "GPU:$DEST/$path/"
done
for path in "${ROOT_FILES[@]}"; do
  [ -f "$STAGE/$path" ] || continue
  rsync "${RSYNC_ARGS[@]}" -e 'ssh -o BatchMode=yes -o ConnectTimeout=15' \
    "$STAGE/$path" "GPU:$DEST/$path"
done

echo "Exact stale paths: $DEST/rule.md $DEST/HANDOFF.md $DEST/board.md $DEST/refs/harness-template-cn $DEST/build/scratch $DEST/build/podtools $DEST/build/diag10"
if [ "${1:-}" = --apply ]; then
  ssh -o BatchMode=yes -o ConnectTimeout=15 GPU \
    'set -e; cd /sjtu/linhang/arena/repo; rm -f -- rule.md HANDOFF.md board.md; for path in refs/harness-template-cn build/scratch build/podtools build/diag10; do if [ -e "$path" ]; then rm -r -- "$path"; fi; done'
else
  echo "Dry run only. Review the rsync deletions and current pod queue; then rerun with --apply after the queue is idle."
fi
