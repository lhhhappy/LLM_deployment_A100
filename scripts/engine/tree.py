#!/usr/bin/env python3
"""Engine source trees from git.

  python3 scripts/engine/tree.py <ref> [OUT]      export engine/sglang at <ref> to OUT/sglang
  python3 scripts/engine/tree.py --mechanism 120  print the commit that introduced mechanism 120

<ref> is any git ref (official-A-0923a, HEAD, a commit), `mech:NNN` for the latest commit of mechanism NNN
(subject "engine NNN: ..."; a mechanism may have follow-up commits), or `before:NNN` for the parent of its
first commit. Without OUT the tree is cached under
build/engine/trees/<commit>/ and reused. Library use (tests): tree_dir(ref) -> Path to .../sglang.
"""
from __future__ import annotations

import argparse
import io
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "build/engine/trees"


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True).stdout.strip()


def mechanism_commits(num: str) -> list[str]:
    """Commits of mechanism NNN, newest first."""
    hits = git("log", "--format=%H", "--grep", f"^engine {num}:", "engine-base..HEAD").splitlines()
    if not hits:
        raise SystemExit(f"mechanism {num}: no commit")
    return hits


def mechanism_commit(num: str) -> str:
    return mechanism_commits(num)[0]


def resolve(ref: str) -> str:
    if ref.startswith("mech:"):
        return mechanism_commit(ref[5:])
    if ref.startswith("before:"):
        return git("rev-parse", mechanism_commits(ref[7:])[-1] + "^")
    return git("rev-parse", "--verify", ref + "^{commit}")


def export(commit: str, out: Path) -> Path:
    data = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", commit, "engine/sglang"],
                          check=True, capture_output=True).stdout
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        subprocess.run(["rm", "-rf", str(tmp)], check=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(tmp)
    out.mkdir(parents=True, exist_ok=True)
    (tmp / "engine/sglang").rename(out / "sglang")
    subprocess.run(["rm", "-rf", str(tmp)], check=True)
    (out / "COMMIT").write_text(commit + "\n")
    return out / "sglang"


def tree_dir(ref: str) -> Path:
    commit = resolve(ref)
    out = CACHE / commit
    if (out / "COMMIT").exists():
        return out / "sglang"
    return export(commit, out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref", nargs="?")
    ap.add_argument("out", nargs="?", type=Path)
    ap.add_argument("--mechanism")
    a = ap.parse_args()
    if a.mechanism:
        print(mechanism_commit(a.mechanism))
        return 0
    if not a.ref:
        ap.error("ref required")
    commit = resolve(a.ref)
    path = export(commit, a.out) if a.out else tree_dir(commit)
    print(commit, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
