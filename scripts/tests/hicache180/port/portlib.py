"""Port step 1: upstream #40913 onto a tree (arg: path to sglang package dir).

Applies the source part of refs/pr40913.diff with `patch -p3` (fuzz allowed), then
hand-resolves the rejected hunks against our base (fe236ea6c3 + our stack).
"""
import os
import re
import subprocess
import sys
from pathlib import Path

PKG = sys.argv[1]
REPO = str(Path(__file__).resolve().parents[4])


def src_part(n):
    t = open(f"{REPO}/refs/pr{n}.diff").read()
    parts = re.split(r"(?m)^(?=diff --git )", t)
    return "".join(p for p in parts if p.startswith("diff --git a/python/"))


def apply(n):
    r = subprocess.run(
        ["patch", "-p3", "-f", "--no-backup-if-mismatch", "-d", PKG],
        input=src_part(n), text=True, capture_output=True,
    )
    print(r.stdout[-3000:], r.stderr[-2000:])


def rd(p):
    return open(os.path.join(PKG, p)).read()


def wr(p, s):
    open(os.path.join(PKG, p), "w").write(s)


def sub_once(s, old, new, what):
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"{what}: expected 1 match, got {n}")
    return s.replace(old, new)


