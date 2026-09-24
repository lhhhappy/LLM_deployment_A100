#!/usr/bin/env python3
"""One-time migration: numbered patches -> git-managed engine source (engine/sglang).

  python3 scripts/engine/import_patches.py REPO [--dry-run]

In REPO (a git work tree) it creates, one commit each:
  1. engine/sglang = build/base_exact/sglang verbatim           (tag engine-base)
  2. each patch of official A (0923a, attempt 45979), in order,
     with its .md moved to engine/docs/                          (tag official-A-0923a after 170)
  3. each candidate, in order, likewise (all default-off)
The .patch files are applied with the same rule as scripts/patch_stack.py
(patch -p3 --fuzz=0 inside the package directory), so the final tree equals
`patch_stack.py apply` of the same list. 124 is not migrated: it conflicts with
122/123 and its short-hit reserve lives in 122.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
BASE = HERE / "build/base_exact/sglang"
PATCHES = HERE / "patches"
OFFICIAL_A = ["000-interface-compliance", "101-role-boundary-split", "106-defer-chunk-on-no-kv",
              "110-sm80-dsa-indexer", "111-sm80-fp8-moe-marlin", "114-indexer-row-shard",
              "120-sched-protect-chain", "121-sched-cap-while-decoding", "130-async-tokenize",
              "140-kda-dual-snapshot", "150-startup-warmup", "160-nextn-sm80", "170-glm-bcg-prefill"]
CANDIDATES = ["115-dcp-sm80", "122-tpot-paced-prefill", "123-srpt-admission",
              "171-kda-bf16-proj-fusion", "172-moe-clamped-swiglu", "180-hicache-glm-dsa"]
TRAILER = "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def title(md: Path, name: str) -> str:
    for line in md.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return re.sub(r"^\d{3}\s*[—-]+\s*", "", line[2:].strip())
    return name


def commit(repo: Path, subject: str, body: str) -> str:
    git(repo, "add", "-A", "engine")
    on_disk = sum(1 for p in (repo / "engine/sglang").rglob("*") if p.is_file() or p.is_symlink())
    tracked = len(git(repo, "ls-files", "engine/sglang").splitlines())
    if on_disk != tracked:
        sys.exit(f"{on_disk - tracked} engine files are not tracked (ignore rules?)")
    git(repo, "commit", "-q", "-m", f"{subject}\n\n{body}\n\n{TRAILER}")
    return git(repo, "rev-parse", "--short", "HEAD")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo", type=Path)
    a = ap.parse_args()
    repo = a.repo.resolve()
    pkg, docs = repo / "engine/sglang", repo / "engine/docs"
    if pkg.exists():
        sys.exit(f"{pkg} exists; this migration runs once")
    # The repo's top-level ignore rules (src/, cache/) would silently drop base files; re-include them here.
    pkg.parent.mkdir(parents=True, exist_ok=True)
    (pkg.parent / ".gitignore").write_text(
        "# Undo the repository-wide src/ and cache/ rules for the engine source; keep bytecode ignored.\n"
        "!src/\n!cache/\n")
    shutil.copytree(BASE, pkg, symlinks=True)
    docs.mkdir(parents=True)
    print("base", commit(repo, "engine: import organizer base source verbatim",
                         "engine/sglang is build/base_exact/sglang (the byte-exact L3 base) unchanged.\n"
                         "Every later engine change is a commit on top of this one."))
    git(repo, "tag", "-f", "engine-base")
    for group, names in (("official A", OFFICIAL_A), ("candidate", CANDIDATES)):
        for name in names:
            with open(PATCHES / f"{name}.patch", "rb") as f:
                r = subprocess.run(["patch", "-p3", "-d", str(pkg), "--fuzz=0", "--no-backup-if-mismatch", "-s"],
                                   stdin=f, capture_output=True, text=True)
            if r.returncode:
                sys.exit(f"FAIL {name}: {r.stdout}{r.stderr}")
            stray = [p for p in pkg.rglob("*") if p.suffix in (".orig", ".rej")]
            if stray:
                sys.exit(f"stray files after {name}: {stray[:3]}")
            md = PATCHES / f"{name}.md"
            shutil.copy(md, docs / md.name)
            num = name.split("-", 1)[0]
            sha = commit(repo, f"engine {num}: {title(md, name)}"[:200],
                         f"Migrated from patches/{name}.patch ({group}); design notes in engine/docs/{md.name}.")
            print(num, sha)
            if name == OFFICIAL_A[-1]:
                git(repo, "tag", "-f", "official-A-0923a")
    return 0


if __name__ == "__main__":
    sys.exit(main())
