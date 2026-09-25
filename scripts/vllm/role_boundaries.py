#!/usr/bin/env python3
"""Position of the last role-boundary token in every replayed prompt.

Renders each request as the load generator does and records, per request, the
token count and the index of the last `<|user|>` / `<|observation|>` token (the
start of the trailing turn block; -1 if absent). Output: JSON lines
{req_id, glm_tokens, role_pos}. Input for kda_reuse_model.py --roles.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for path in (
    os.path.join(REPO, "s1-dev", "harness"),
    os.environ.get("HARNESS_DIR", ""),
):
    if path:
        sys.path.insert(0, path)

ROLE_TOKENS = ("<|user|>", "<|observation|>")
_w = {}


def _init(tok_dir: str) -> None:
    from s1_common import Renderer

    r = Renderer(tok_dir)
    _w["r"] = r
    _w["ids"] = {r.tokenizer.convert_tokens_to_ids(t) for t in ROLE_TOKENS}


def _scan(body: dict) -> tuple[int, int]:
    ids = _w["r"].tokenizer.encode(_w["r"].render(body), add_special_tokens=False)
    for i in range(len(ids) - 1, -1, -1):
        if ids[i] in _w["ids"]:
            return len(ids), i
    return len(ids), -1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--tok-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from s1_common import load_index, materialize_bodies

    rows, _c, _b = load_index(args.data_root)
    rids = sorted(rows)
    bodies = materialize_bodies(args.data_root, rids)
    with ProcessPoolExecutor(
        args.workers, initializer=_init, initargs=(args.tok_dir,)
    ) as pool:
        res = list(pool.map(_scan, [bodies[r] for r in rids], chunksize=8))
    with open(args.out, "w", encoding="utf-8") as fh:
        for rid, (n, pos) in zip(rids, res):
            assert n == rows[rid]["glm_tokens"], (rid, n, rows[rid]["glm_tokens"])
            fh.write(
                json.dumps({"req_id": rid, "glm_tokens": n, "role_pos": pos}) + "\n"
            )
    print(f"{len(rids)} requests; no role token in {sum(p < 0 for _, p in res)}")


if __name__ == "__main__":
    main()
