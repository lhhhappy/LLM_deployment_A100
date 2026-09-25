#!/usr/bin/env python3
"""Multi-turn prefix reuse probe: replay whole chains in order, one at a time.

After a flush, each selected chain's requests are sent sequentially (no gaps,
no concurrency), exactly as rendered by the harness. For every request the
receipt records prompt_tokens, the engine's cached_tokens and the frozen
glm_lcp_with_prev (the token LCP with the previous prompt of the same chain).
With nothing else competing for the cache, lcp - cached is the recompute
caused by cache granularity (block/state alignment), which is the quantity a
finer prefix-match unit is meant to shrink. Summary lines are exact sums over
the non-head requests; chain heads (lcp 0) are listed but excluded.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_interface import generate, post_json, summarize  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for path in (
    os.path.join(REPO, "s1-dev", "harness"),
    os.environ.get("HARNESS_DIR", ""),
):
    if path:
        sys.path.insert(0, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--tok-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chains", type=int, default=3)
    ap.add_argument("--turns", type=int, default=8, help="requests per chain")
    ap.add_argument("--max-prompt", type=int, default=120000)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    base = args.base_url.rstrip("/").removesuffix("/v1")

    from s1_common import Renderer, load_index, materialize_bodies

    rows, _chains, by_chain = load_index(args.data_root)
    chosen = []
    for cid in sorted(by_chain):
        reqs = by_chain[cid][: args.turns]  # replay order, as the load generator
        if len(reqs) == args.turns and all(
            r["glm_tokens"] <= args.max_prompt for r in reqs
        ):
            chosen.append(reqs)
        if len(chosen) == args.chains:
            break
    wanted = [r["_req_id"] for reqs in chosen for r in reqs]
    bodies = materialize_bodies(args.data_root, wanted)
    renderer = Renderer(args.tok_dir)

    status, body = post_json(base + "/flush_cache", None, 120)
    assert status == 200 and body.get("success") is True, (status, body)

    records = []
    for reqs in chosen:
        for r in reqs:
            text = renderer.render(bodies[r["_req_id"]])
            out = summarize(
                generate(base, text, args.max_new_tokens, r["_req_id"], 1800)
            )
            records.append(
                {
                    **out,
                    "chain_id": r["chain_id"],
                    "idx_in_chain": r["_idx_in_chain"],
                    "glm_tokens": r["glm_tokens"],
                    "glm_lcp_with_prev": r["glm_lcp_with_prev"],
                }
            )
    follow = [x for x in records if x["glm_lcp_with_prev"]]
    lcp = sum(x["glm_lcp_with_prev"] for x in follow)
    cached = sum(x["cached_tokens"] or 0 for x in follow)
    summary = {
        "label": args.label,
        "time": time.time(),
        "chains": len(chosen),
        "follow_requests": len(follow),
        "sum_lcp_with_prev": lcp,
        "sum_cached": cached,
        "sum_lcp_minus_cached": lcp - cached,
        "mean_lcp_minus_cached": (lcp - cached) / max(1, len(follow)),
        "errors": sum(1 for x in records if x["errors"]),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(
            {"summary": summary, "requests": records}, fh, ensure_ascii=False, indent=1
        )
    print(json.dumps(summary))
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
