#!/bin/bash
# Run the patch-180 tests against one SGLang tree (a directory that contains sglang/).
#   run_tests.sh TREE [PYTHON] [unittest args...]      (default: every test_180_*.py)
# CPU by default (byte copies emulated, see _cpu_harness.py). On the dev box:
#   HC180_DEVICE=cuda run_tests.sh TREE python   -> same tests, real sgl_kernel/JIT copies, pinned host memory.
# Get a tree from the engine git history, e.g. the 180 commit:
#   python3 scripts/engine/tree.py mech:180          (prints the cached path .../sglang; pass its parent as TREE)
# PYTHON needs torch and the sglang import deps.
set -uo pipefail
TREE=$(readlink -f "$1"); shift
PY=${1:-python3}; [ $# -gt 0 ] && shift
HERE=$(dirname "$(readlink -f "$0")")
if [ $# -eq 0 ]; then set -- discover -s "$HERE" -p 'test_180_*.py'; fi
cd "$HERE" && PYTHONPATH="$TREE:$HERE" "$PY" -m unittest "$@" 2>&1 | grep -v "UserWarning\|warnings.warn"
