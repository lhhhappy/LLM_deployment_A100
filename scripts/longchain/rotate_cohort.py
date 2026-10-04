#!/usr/bin/env python3
"""Rotate complete frozen chains; preserve request values, bodies and chain order."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile

if __package__:
    from .longchain_metadata import publish, read_rows
else:
    from longchain_metadata import publish, read_rows


def rotate(parent, out, offset, name):
    parent = Path(parent)
    cohort = json.loads((parent / "cohort.json").read_text())
    chains = cohort["chains"]
    if not chains:
        raise ValueError("cannot rotate an empty cohort")
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ValueError("offset must be a nonnegative integer")
    offset %= len(chains)
    rotated = deepcopy(cohort)
    rotated["chains"] = chains[offset:] + chains[:offset]
    operation = {"kind": "cohort-rotation", "offset": offset,
                 "rule": "chains[offset:] + chains[:offset]",
                 "scope": "same complete chains and within-chain order; diagnostic opening order"}
    with tempfile.TemporaryDirectory(prefix="longchain-cohort-") as temporary:
        replacement = Path(temporary) / "cohort.json"
        replacement.write_text(json.dumps(rotated, ensure_ascii=False) + "\n")
        manifest = publish(parent, out, read_rows(parent / "requests.jsonl"),
                           name=name, operation=operation, cohort_path=replacement)
    actual = json.loads((Path(out) / "cohort.json").read_text())
    digest = hashlib.sha256(json.dumps(actual["chains"], ensure_ascii=False,
                                      sort_keys=True).encode()).hexdigest()[:16]
    return {"set": name, "offset": offset, "chains": len(chains),
            "cohort_sha256": digest, "status": manifest["status"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=150)
    parser.add_argument("--set", required=True)
    args = parser.parse_args()
    print(json.dumps(rotate(args.parent, args.out, args.offset, args.set), ensure_ascii=False))


if __name__ == "__main__":
    main()
