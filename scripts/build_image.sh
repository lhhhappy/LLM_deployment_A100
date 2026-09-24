#!/usr/bin/env bash
# build_image.sh — generate (NOT build) an inline Dockerfile that turns one of our registered images into the engine
# source of a git commit, for `bohr image build`.
#
#   scripts/build_image.sh <ref> [--from-image IMAGE --from-ref REF] [--name NAME:TAG] [--project-id ID]
#
# Platform constraints (llm-challenge-arena-v1/task.md, 制备并注册镜像): the Dockerfile is uploaded inline, at most
# 64 KiB, with no build context (no COPY); --name must not end with `latest`. The full base->HEAD diff no longer fits,
# so an image is built FROM one of our registered images whose engine source equals a known commit (--from-ref), and
# embeds only the diff from that commit to <ref>:
#   default FROM = lh-img:0923a (attempts 45979/45980), whose source equals tag official-A-0923a (T57 + migration check).
# Inside the image the RUN step applies the diff to the installed sglang package with `patch -p3 --fuzz=0` (paths are
# a/engine/sglang/srt/..., so srt/... lands in the package dir; any failed hunk fails the build), byte-compiles the
# changed files and records the commit in /opt/ax/engine_commit.
# Local checks (always): the embedded payload decodes to the exact diff, and applying it to the --from-ref tree gives
# the <ref> tree file for file (scripts/engine/tree.py).
# After building, tag the commit so later images can layer on this one: git tag image-<name>-<tag> <commit>.
# This script never builds or publishes; it prints the build command.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FROM_IMAGE="registry.dp.tech/dptech/dp/native/prod-4727808/4650601/lh-img:0923a"
FROM_REF="official-A-0923a"
NAME="lh-img:$(date -u +%m%d)a"   # image names/tags are visible to all arena members: keep them neutral
PROJECT_ID="<PROJECT_ID>"
LIMIT=65536
REF=""

die() { echo "ERROR: $*" >&2; exit 1; }
while [ $# -gt 0 ]; do
  case "$1" in
    --from-image) FROM_IMAGE="$2"; shift 2 ;;
    --from-ref) FROM_REF="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --project-id) PROJECT_ID="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    -*) die "unknown option $1" ;;
    *) [ -z "$REF" ] || die "one ref only"; REF="$1"; shift ;;
  esac
done
[ -n "$REF" ] || die "usage: build_image.sh <ref> [options]"
[[ "$NAME" == *:* ]] || die "--name must be name:tag"
[[ "$NAME" == *latest ]] && die "--name must not end with 'latest' (task.md)"
cd "$ROOT"
COMMIT="$(git rev-parse --verify "$REF^{commit}")"
FROM="$(git rev-parse --verify "$FROM_REF^{commit}")"
git merge-base --is-ancestor "$FROM" "$COMMIT" || die "$FROM_REF is not an ancestor of $REF"
[ "$REF" != HEAD ] || [ -z "$(git status --porcelain -- engine)" ] || die "engine/ has uncommitted changes"

STAGE="$(mktemp -d)"; trap 'rm -rf "$STAGE"' EXIT
git diff --no-color --binary "$FROM" "$COMMIT" -- engine/sglang > "$STAGE/engine.diff"
[ -s "$STAGE/engine.diff" ] || die "no engine change between $FROM_REF and $REF"
bad="$(grep -E '^(\+\+\+|---) ' "$STAGE/engine.diff" | awk '{print $2}' | grep -vE '^([ab]/engine/sglang/|/dev/null$)' || true)"
[ -z "$bad" ] || die "diff has paths outside engine/sglang: $bad"
B64="$(gzip -9nc "$STAGE/engine.diff" | base64 -w0)"

mkdir -p build/image
DF=build/image/Dockerfile
{
  echo "# generated"
  echo "FROM $FROM_IMAGE"
  echo "RUN set -eu; \\"
  echo "    command -v patch >/dev/null 2>&1 || (apt-get update && apt-get install -y --no-install-recommends patch) || { echo 'patch(1) unavailable' >&2; exit 1; }; \\"
  echo "    mkdir -p /tmp/ax /opt/ax; \\"
  echo "    echo '$B64' | base64 -d | gunzip > /tmp/ax/engine.diff; \\"
  echo "    PKG=\"\${AX_PKG:-/sgl-workspace/sglang/python/sglang}\"; test -d \"\$PKG/srt\"; \\"
  echo "    patch -p3 -d \"\$PKG\" --force --fuzz=0 --no-backup-if-mismatch -r - < /tmp/ax/engine.diff || { echo 'engine diff FAILED' >&2; exit 1; }; \\"
  echo "    for f in \$(grep '^+++ b/' /tmp/ax/engine.diff | cut -d/ -f4- | grep '\\.py\$'); do python3 -S -I -m py_compile \"\$PKG/\$f\"; done; \\"
  echo "    echo $COMMIT > /opt/ax/engine_commit; rm -rf /tmp/ax; \\"
  echo "    echo 'engine source applied'"
} > "$DF"
SIZE="$(wc -c < "$DF")"
echo "Dockerfile: $DF ($SIZE bytes, limit $LIMIT)"
[ "$SIZE" -le "$LIMIT" ] || { mv "$DF" "$DF.too-big"; die "over the inline limit; build a newer base image first (--from-image/--from-ref)"; }

# Local check 1: the embedded payload is exactly the diff.
sed -n "s/^    echo '\([A-Za-z0-9+\/=]*\)' | base64 -d | gunzip > \/tmp\/ax\/engine.diff.*/\1/p" "$DF" | base64 -d | gunzip \
  | cmp -s - "$STAGE/engine.diff" || die "embedded payload differs from the diff"
# Local check 2: from-ref tree + payload == ref tree, file for file.
python3 scripts/engine/tree.py "$FROM" "$STAGE/applied" >/dev/null
python3 scripts/engine/tree.py "$COMMIT" "$STAGE/want" >/dev/null
patch -p3 -d "$STAGE/applied/sglang" --fuzz=0 --no-backup-if-mismatch -s < "$STAGE/engine.diff" || die "diff does not apply to $FROM_REF"
diff -r -q "$STAGE/applied/sglang" "$STAGE/want/sglang" >/dev/null || die "applied tree differs from $REF"
echo "checks OK: payload = diff; $FROM_REF + diff = $REF ($COMMIT)"
echo
echo "Not built. To build and register:"
echo "  bohr image build --dockerfile $DF --name $NAME --project-id $PROJECT_ID --wait --yes -o json"
echo "Then: git tag image-${NAME//:/-} $COMMIT ; use the returned image URL with a unique tag in submission.json."
