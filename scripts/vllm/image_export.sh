#!/usr/bin/env bash
# Copy an image's root filesystem to the dev box through a Bohrium CPU sandbox: the sandbox
# platform pulls the image (our access keys cannot pull the organizer's registry projects), a
# tar of / is written inside the sandbox in 2000 MB gzip parts, and the dev box downloads,
# verifies (sha256 computed inside the sandbox) and unpacks them. No image is built.
# Run on the dev box (bohr from daemon_env.sh), e.g. under scripts/gjob:
#   image_export.sh start <image> <cpu profile, e.g. 8c32g> <project id>   -> prints the sandbox id
#   image_export.sh fetch <sandbox id> <out dir>    parts -> <out>/parts, tree -> <out>/rootfs
#   image_export.sh delete <sandbox id>
# Creating a sandbox is billable; the sandbox ends on its own after its timeout (7200 s).
set -euo pipefail
source /sjtu/linhang/arena/daemon_env.sh >/dev/null 2>&1
EXPORT=/root/ax-export
TAR_CMD="mkdir -p $EXPORT && cd / && (tar --numeric-owner --xattrs -cpf - --one-file-system \
--exclude=./proc --exclude=./sys --exclude=./dev --exclude=./run --exclude=./tmp --exclude=.$EXPORT . \
2>$EXPORT/tar.err | gzip -4 | split -b 2000M -d -a 3 - $EXPORT/rootfs.tar.gz.part-; \
echo tar_exit=\${PIPESTATUS[0]} > $EXPORT/STATUS; cd $EXPORT && sha256sum rootfs.tar.gz.part-* > SHA256SUMS; \
echo DONE >> $EXPORT/STATUS) > $EXPORT/run.log 2>&1"

sx() {  # run a command in the sandbox, print its stdout
  bohr sandbox exec --timeout 120 -o json --command "$2" "$1" |
    python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["data"]["stdout"], end="")'
}

case ${1:-} in
  start)
    sb=$(bohr sandbox create --image "$2" --cpu "$3" --project-id "$4" --timeout 7200 --wait --yes -o json |
      python3 -c 'import json,sys; print(json.load(sys.stdin)["data"]["sandboxID"])')
    bohr sandbox exec --timeout 0 --background -o json --command "$TAR_CMD" "$sb" >/dev/null
    echo "$sb"
    ;;
  fetch)
    sb=$2 out=$3
    mkdir -p "$out/parts"
    while :; do
      state=$(sx "$sb" "cat $EXPORT/STATUS 2>/dev/null; ls $EXPORT | grep part-")
      parts=$(echo "$state" | grep part- || true)
      finished=$(echo "$state" | grep -c '^DONE$' || true)
      last=$(echo "$parts" | tail -1)
      for p in $parts; do
        [ "$finished" = 0 ] && [ "$p" = "$last" ] && continue   # still being written
        [ -f "$out/parts/$p.ok" ] && continue
        t0=$(date +%s)
        bohr sandbox files read "$sb" "$EXPORT/$p" --destination "$out/parts/$p" --timeout 3600 >/dev/null
        touch "$out/parts/$p.ok"
        echo "$(date -u +%H:%M:%S) $p $(stat -c %s "$out/parts/$p") bytes in $(( $(date +%s) - t0 ))s"
      done
      [ "$finished" -ge 1 ] && break
      sleep 30
    done
    for f in SHA256SUMS STATUS tar.err; do
      bohr sandbox files read "$sb" "$EXPORT/$f" --destination "$out/parts/$f" >/dev/null
    done
    grep -q '^tar_exit=0$' "$out/parts/STATUS" || { echo "tar failed: $(cat "$out/parts/STATUS")"; exit 1; }
    (cd "$out/parts" && sha256sum -c SHA256SUMS)
    rm -rf "$out/rootfs.partial" && mkdir -p "$out/rootfs.partial"
    cat "$out"/parts/rootfs.tar.gz.part-* | tar -xzpf - --numeric-owner -C "$out/rootfs.partial"
    mv "$out/rootfs.partial" "$out/rootfs"
    du -sh "$out/parts" "$out/rootfs"
    echo EXPORT_DONE
    ;;
  delete)
    bohr sandbox delete "$2" --yes
    ;;
  *) sed -n 2,11p "$0"; exit 2 ;;
esac
