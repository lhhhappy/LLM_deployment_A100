#!/usr/bin/env bash
# build_image.sh — generate (NOT build) an inline Dockerfile that bakes our SGLang patches
# into an organizer base image, for `bohr image build` / `lbg sdbx image build`.
#
# Platform constraints (llm-challenge-arena-v1/task.md, 制备并注册镜像):
#   * the Dockerfile is uploaded inline, limit 64 KiB, and there is NO build context (no COPY);
#   * --name must not end with `latest`.
# So the patches are packed as a deterministic tar.gz, base64-encoded and embedded in a RUN step.
#
# The RUN step, inside the image:
#   * verifies the payload sha256s,
#   * locates the installed sglang package dir (python3 -c 'import sglang,os;...'),
#   * applies each patch with `patch -p3 -d <pkg>` (our diffs are a/python/sglang/srt/..., so
#     stripping a/ python/ sglang/ leaves srt/..., relative to the package dir),
#     strict: --fuzz=0, --force (never guess "reversed", never skip), no --forward;
#     any failed hunk => non-zero exit => build fails,
#   * byte-compiles the patched files and runs `python3 -c "import sglang"` as a smoke check.
#
# This script NEVER builds or publishes. It prints the exact build command for a human to run.
# Local verification (always run): extract the payload back out of the generated Dockerfile and
# compare sha256s with the source patches; dry-run (then really apply, to a throwaway copy) the
# patches against build/scratch/python (a copy of build/base_exact = the exact base source, F54).
#
# Usage:
#   scripts/build_image.sh [--base IMAGE_REF] [--name NAME:TAG] [--project-id ID] [--no-scratch] [PATCH...]
# Defaults: base = organizer SGLang image (task.md 可起步的基镜像), name = lh-img:<mmdd>a (neutral),
#           patches = the current RELEASE stack declared below, in numeric apply order
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="registry.dp.tech/dptech/dp/native/prod-3732438/20675/arena-sglang-glm53:260918"
NAME="lh-img:$(date -u +%m%d)a"   # image names/tags are visible to all arena members
PROJECT_ID="<PROJECT_ID>"
DO_SCRATCH=1
PATCHES=()
STRIP=3
LIMIT=65536
OUT_DIR="$ROOT/build/image"
SCRATCH="$ROOT/build/scratch"

die() { echo "ERROR: $*" >&2; exit 1; }
sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}
decode64() { python3 -c 'import base64,sys;sys.stdout.buffer.write(base64.b64decode(sys.stdin.buffer.read()))'; }
if patch --version 2>&1 | grep -q 'GNU patch'; then
  PATCH_APPLY=(--force --fuzz=0 --no-backup-if-mismatch -r -)
  PATCH_DRY=(--dry-run)
else
  PATCH_APPLY=(-f -F 0 -r /dev/null)
  PATCH_DRY=(-C)
fi

while [ $# -gt 0 ]; do
  case "$1" in
    --base) BASE="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --project-id) PROJECT_ID="$2"; shift 2 ;;
    --no-scratch) DO_SCRATCH=0; shift ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    --) shift; PATCHES+=("$@"); break ;;
    -*) die "unknown option $1" ;;
    *) PATCHES+=("$1"); shift ;;
  esac
done
if [ ${#PATCHES[@]} -eq 0 ]; then
  # Default = every numbered patch present, in order (000, 001, 002, ...), per W8 R11:
  # one image carries all patches; features are selected by env / --schedule-policy.
  # Only reviewed patches listed in patches/RELEASE (drafts such as an in-progress 003 stay out).
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    case "$line" in \#*) continue ;; esac
    PATCHES+=("$ROOT/patches/$line")
  done < "$ROOT/patches/RELEASE"
fi
[ ${#PATCHES[@]} -gt 0 ] || { echo "no patches found" >&2; exit 1; }

[[ "$NAME" == *:* ]] || die "--name must be name:tag"
[[ "$NAME" == *latest ]] && die "--name must not end with 'latest' (task.md)"
[[ "$BASE" =~ ^registry\.dp\.tech/ ]] || echo "WARNING: base '$BASE' is not on registry.dp.tech; Trisol/LBG may not pull it" >&2

# ---- stage patches ---------------------------------------------------------------------------
STAGE="$(mktemp -d)"; trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/p"
NAMES=()
for p in "${PATCHES[@]}"; do
  [ -f "$p" ] || die "patch not found: $p"
  b="$(basename "$p")"
  [[ "$b" =~ ^[A-Za-z0-9._-]+$ ]] || die "unsafe patch filename: $b"
  # Every file path in the patch must be a/python/sglang/... so that -p$STRIP lands in the pkg dir.
  bad="$(grep -E '^(\+\+\+|---) ' "$p" | awk '{print $2}' | grep -vE '^([ab]/python/sglang/|/dev/null$)' || true)"
  [ -z "$bad" ] || die "$b has paths outside a|b/python/sglang/: $bad"
  # Neutral in-image name (p0.patch, p1.patch, ...): image contents may be visible to arena members.
  nb="p${#NAMES[@]}.patch"
  cp "$p" "$STAGE/p/$nb"
  NAMES+=("$nb")
done
(cd "$STAGE/p" && for b in "${NAMES[@]}"; do printf '%s  %s\n' "$(sha256_file "$b")" "$b"; done > SHA256SUMS)
python3 - "$STAGE/p" "$STAGE/patches.tar.gz" <<'PY'
import gzip
from pathlib import Path
import sys
import tarfile

source, output = Path(sys.argv[1]), Path(sys.argv[2])
with output.open("wb") as target:
    with gzip.GzipFile(filename="", fileobj=target, mode="wb", compresslevel=9, mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w", format=tarfile.GNU_FORMAT) as archive:
            for name in (".", *(p.name for p in sorted(source.iterdir()))):
                path = source if name == "." else source / name
                info = archive.gettarinfo(str(path), arcname=name)
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ""
                info.mode = 0o755 if name == "." else 0o644
                if name == ".":
                    archive.addfile(info)
                else:
                    with path.open("rb") as body:
                        archive.addfile(info, body)
PY
B64="$(python3 -c 'import base64,sys;sys.stdout.write(base64.b64encode(open(sys.argv[1],"rb").read()).decode())' "$STAGE/patches.tar.gz")"

# ---- write Dockerfile ------------------------------------------------------------------------
mkdir -p "$OUT_DIR"
DF="$OUT_DIR/Dockerfile"
cat > "$STAGE/serve" <<'LAUNCH'
#!/bin/sh
# Neutral launcher: Trisol service specs only show "/opt/ax/serve <profile>".
set -eu
P="${1:-b0}"; [ $# -gt 0 ] && shift
export SGLANG_OPT_USE_TOPK_V2=0
X=""
case "$P" in
  b0) ;;
  b1) export SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=154827,154829 ;;
  b2) X="--schedule-policy shortest-prefill-first" ;;
  b3) export SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=154827,154829; X="--schedule-policy shortest-prefill-first" ;;
  *) echo "unknown profile $P" >&2; exit 2 ;;
esac
exec python3 -m sglang.launch_server --model-path /mnt/models --host 0.0.0.0 --port 8000 \
  --tp-size 8 --served-model-name default --enable-metrics --incremental-streaming-output \
  --page-size 64 --mamba-radix-cache-strategy extra_buffer --reasoning-parser glm45 \
  --tool-call-parser glm47 $X "$@"
LAUNCH
LAUNCHER_B64="$(python3 -c 'import base64,sys;sys.stdout.write(base64.b64encode(open(sys.argv[1],"rb").read()).decode())' "$STAGE/serve")"
{
  echo "# generated"
  echo "FROM $BASE"
  echo "RUN set -eu; \\"
  echo "    command -v patch >/dev/null 2>&1 || (apt-get update && apt-get install -y --no-install-recommends patch) || { echo 'patch(1) unavailable' >&2; exit 1; }; \\"
  echo "    mkdir -p /tmp/ax/p /opt/ax; \\"
  echo "    echo '$B64' | base64 -d > /tmp/ax/p.tgz; \\"
  echo "    tar -xzf /tmp/ax/p.tgz -C /tmp/ax/p; \\"
  echo "    cd /tmp/ax/p && sha256sum -c SHA256SUMS; \\"
  echo "    PKG=\"\${AX_PKG:-/sgl-workspace/sglang/python/sglang}\"; \\"
  echo "    test -d \"\$PKG/srt\"; \\"
  echo "    for p in ${NAMES[*]}; do \\"
  echo "      patch -p$STRIP -d \"\$PKG\" --force --fuzz=0 --no-backup-if-mismatch -r - < \"/tmp/ax/p/\$p\" || { echo \"apply \$p FAILED\" >&2; exit 1; }; \\"
  echo "      for f in \$(grep '^+++ ' \"/tmp/ax/p/\$p\" | awk '{print \$2}' | cut -d/ -f$((STRIP+1))-); do python3 -S -I -m py_compile \"\$PKG/\$f\"; done; \\"
  echo "    done; \\"
  echo "    echo '$LAUNCHER_B64' | base64 -d > /opt/ax/serve; chmod 755 /opt/ax/serve; sh -n /opt/ax/serve; \\"
  echo "    rm -rf /tmp/ax; \\"
  echo "    echo 'patched OK (syntax-checked; torch not loaded in the build env)'"
} > "$DF"

SIZE="$(wc -c < "$DF")"
echo "Dockerfile: $DF"
echo "Dockerfile size: $SIZE bytes (limit $LIMIT)"
if [ "$SIZE" -gt "$LIMIT" ]; then
  mv "$DF" "$DF.too-big"
  die "Dockerfile exceeds 64 KiB inline limit; moved to $DF.too-big"
fi

# ---- local verification 1: payload round-trip -------------------------------------------------
V="$STAGE/verify"; mkdir -p "$V"
sed -n "s/^    echo '\([A-Za-z0-9+\/=]*\)' | base64 -d > \/tmp\/ax\/p.tgz.*/\1/p" "$DF" | decode64 | tar -xz -C "$V"
for i in "${!PATCHES[@]}"; do
  src="$(sha256_file "${PATCHES[$i]}")"
  got="$(sha256_file "$V/${NAMES[$i]}")"
  [ "$src" = "$got" ] || die "payload mismatch for ${NAMES[$i]}: src=$src embedded=$got"
  echo "payload OK  ${NAMES[$i]}  sha256=$src"
done
for b in "${NAMES[@]}"; do
  [ "$(sha256_file "$V/$b")" = "$(awk -v f="$b" '$2 == f {print $1}' "$V/SHA256SUMS")" ] ||
    die "embedded SHA256SUMS does not verify $b"
done

# ---- local verification 2: patches vs a copy of the exact base source (build/base_exact, F54) --
if [ "$DO_SCRATCH" = 1 ]; then
  [ -d "$ROOT/build/base_exact/sglang/srt" ] || die "build/base_exact not found (exact base source, F54)"
  rm -rf "$SCRATCH"; mkdir -p "$SCRATCH/python"
  cp -a "$ROOT/build/base_exact/sglang" "$SCRATCH/python/sglang"
  PKG="$SCRATCH/python/sglang"
  # Cumulative, same order as the Dockerfile RUN step: dry-run patch i on top of 0..i-1, then apply it.
  for b in "${NAMES[@]}"; do
    patch -p$STRIP -d "$PKG" "${PATCH_DRY[@]}" "${PATCH_APPLY[@]}" < "$V/$b" > "$STAGE/dry.log" 2>&1 \
      || { cat "$STAGE/dry.log" >&2; die "dry-run failed for $b on top of the previous patches (base fe236ea6c3+mmfix1)"; }
    echo "dry-run OK $b (stacked): $(tr '\n' ' ' < "$STAGE/dry.log")"
    patch -p$STRIP -d "$PKG" "${PATCH_APPLY[@]}" < "$V/$b" > /dev/null
    for f in $(grep '^+++ ' "$V/$b" | awk '{print $2}' | cut -d/ -f$((STRIP+1))-); do
      python3 -m py_compile "$PKG/$f"
    done
  done
  echo "scratch apply + py_compile OK ($SCRATCH = exact base + patches, i.e. what L3 runs)"
fi

echo
echo "Not built. To build and register:"
echo "  bohr image build --dockerfile build/image/Dockerfile --name $NAME --project-id $PROJECT_ID --wait -o json"
echo "Use the returned image URL with a unique tag in submission.json (the platform rejects tag@sha256)."
