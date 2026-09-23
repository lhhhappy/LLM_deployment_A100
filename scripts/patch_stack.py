#!/usr/bin/env python3
"""Build and compare patched SGLang trees on the CPU (never touches the pod or a GPU).

  patch_stack.py apply OUT P1 P2 ...               copy build/base_exact/sglang to OUT/sglang, apply P1.. in order
  patch_stack.py same TREE_A TREE_B                exit 0 iff TREE_A/sglang and TREE_B/sglang are byte-identical
  patch_stack.py make OLD NEW OUT.patch F1 F2 ...  write a -p3 patch that turns OLD/sglang into NEW/sglang for the
                                                   listed package-relative files (e.g. srt/managers/scheduler.py)

Patch names are relative to patches/ with or without ".patch". Patches apply with `patch -p3 --fuzz=0`, exactly as
scripts/pod/lib.sh and scripts/build_image.sh do. The base is build/base_exact (the byte-exact L3 base, F54).
Used to prove that a patch-stack refactor leaves the final source tree of every recorded configuration unchanged.
"""
import argparse
import filecmp
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "build", "base_exact", "sglang")


def patch_path(name):
    name = name if name.endswith(".patch") else name + ".patch"
    return os.path.join(ROOT, "patches", name)


def apply(out, names):
    pkg = os.path.join(out, "sglang")
    if os.path.exists(out):
        shutil.rmtree(out)
    shutil.copytree(BASE, pkg, symlinks=True)
    for name in names:
        with open(patch_path(name), "rb") as f:
            r = subprocess.run(["patch", "-p3", "-d", pkg, "--fuzz=0", "--no-backup-if-mismatch", "-s"],
                               stdin=f, capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"FAIL applying {name}:\n{r.stdout}{r.stderr}")
    stray = [os.path.join(d, f) for d, _, fs in os.walk(pkg) for f in fs if f.endswith((".orig", ".rej"))]
    if stray:
        sys.exit(f"stray patch files: {stray[:5]}")
    print(f"OK {out}: {len(names)} patches")


def files_of(tree):
    pkg = os.path.join(tree, "sglang")
    return {os.path.relpath(os.path.join(d, f), pkg) for d, _, fs in os.walk(pkg) for f in fs}


def same(a, b):
    fa, fb = files_of(a), files_of(b)
    diff = sorted(fa ^ fb)
    diff += sorted(f for f in fa & fb
                   if not filecmp.cmp(os.path.join(a, "sglang", f), os.path.join(b, "sglang", f), shallow=False))
    for f in diff:
        print("DIFFERS", f)
    print("IDENTICAL" if not diff else f"{len(diff)} files differ", f"({len(fa)} files)")
    return 0 if not diff else 1


def make(old, new, out, rels):
    chunks = []
    for rel in rels:
        a, b = os.path.join(old, "sglang", rel), os.path.join(new, "sglang", rel)
        la = "a/python/sglang/" + rel if os.path.exists(a) else "/dev/null"
        lb = "b/python/sglang/" + rel if os.path.exists(b) else "/dev/null"
        r = subprocess.run(["diff", "-u", "--label", la, "--label", lb,
                            a if os.path.exists(a) else "/dev/null", b if os.path.exists(b) else "/dev/null"],
                           capture_output=True, text=True)
        if r.returncode > 1:
            sys.exit(f"diff failed for {rel}: {r.stderr}")
        chunks.append(r.stdout)
    with open(out, "w") as f:
        f.write("".join(chunks))
    print(f"wrote {out}: {sum(c.count(chr(10)) for c in chunks)} lines, {sum(1 for c in chunks if c)} files changed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command")
    apply_parser = commands.add_parser("apply", help="copy the exact base and apply patches")
    apply_parser.add_argument("out")
    apply_parser.add_argument("patches", nargs="*")
    same_parser = commands.add_parser("same", help="compare two patched trees")
    same_parser.add_argument("tree_a")
    same_parser.add_argument("tree_b")
    make_parser = commands.add_parser("make", help="write a patch between two trees")
    make_parser.add_argument("old")
    make_parser.add_argument("new")
    make_parser.add_argument("out")
    make_parser.add_argument("files", nargs="+")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()
    if args.command == "apply":
        apply(args.out, args.patches)
    elif args.command == "same":
        sys.exit(same(args.tree_a, args.tree_b))
    elif args.command == "make":
        make(args.old, args.new, args.out, args.files)
