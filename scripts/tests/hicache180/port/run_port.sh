#!/bin/bash
# Regenerate the 180 change from refs/pr*.diff: upstream hunks + the adaptations listed in engine/docs/180-hicache-glm-dsa.md.
# usage: run_port.sh SRC_TREE_DIR OUT_DIR   (trees contain sglang/; SRC = the deployed stack 000..170)
# then compare with the 180 commit: c=$(python3 scripts/engine/tree.py --mechanism 180); git diff $c^ $c -- engine/sglang
# (the scheduler.py guard change in that commit is a local addition, not produced by these port steps).
set -e
H=$(dirname $(readlink -f $0))
rm -rf "$2"; mkdir -p "$2"; cp -r "$1/sglang" "$2/sglang"
for s in step1_40913 step2_40914_40915 step3_38212 step4_ax; do
  [ -f $H/$s.py ] || continue
  python3 $H/$s.py "$2/sglang" > "$2/$s.log" 2>&1 || { tail -5 "$2/$s.log"; echo "FAIL $s"; exit 1; }
  tail -1 "$2/$s.log"
done
find "$2" -name "*.rej" -o -name "*.orig"
