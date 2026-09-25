#!/usr/bin/env python3
"""Estimate prefix-cache recompute caused by where KDA states are kept.

vLLM's hybrid prefix cache can resume a request only at a position where a KDA
(recurrent) state was kept. With MTP the attention side also drops one hash
unit, so a follow-up request whose prompt shares `lcp` tokens with an earlier
prompt of its chain hits

    max{ s in kept states of earlier requests : 0 < s <= floor(lcp/u)*u - u }

(u = prefix-match unit; by default the attention block size B). This script
applies that rule to every follow-up request of a data set for several state
placement policies, assuming no eviction and no cross-chain sharing, and
reports the tokens inside the frozen LCP that would be recomputed.

  calibrate: --calibrate probe.json [--block B]   (output of probe_prefix_reuse.py)
  estimate:  --data-root data/s1-dev-longchain --block 576 --unit 64
Numbers from `estimate` are model estimates, not measurements.
"""

import argparse
import collections
import json
import os


def hit(kept: list[int], lcp: int, unit: int) -> int:
    cand = (lcp // unit) * unit - unit
    ok = [s for s in kept if 0 < s <= cand]
    return max(ok) if ok else 0


def policies(block: int, unit: int, chunk: int):
    return {
        f"default: tail state at floor(L/B)B-B, B={block}": (
            block,
            lambda L: [(L // block) * block - block],
        ),
        f"tail at u={unit}: floor(L/u)u-u": (
            unit,
            lambda L: [(L // unit) * unit - unit],
        ),
        f"tail at u={unit} + a state every {chunk} tokens (dense retention)": (
            unit,
            lambda L: [(L // unit) * unit - unit] + list(range(chunk, L, chunk)),
        ),
    }


def load_chains(data_root: str):
    rows = []
    with open(os.path.join(data_root, "requests.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["view"] == "canon" and r.get("in_serving_load"):
                rows.append(r)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["chain_id"]].append(r)
    for rs in by.values():  # the load generator's replay order
        rs.sort(
            key=lambda x: (
                x.get("dispatch_offset_ms") or 0,
                x.get("logical_call_id") or "",
            )
        )
    return by


def estimate(by, unit, place):
    lost = n = zero = 0
    for rs in by.values():
        kept: list[int] = []
        for i, r in enumerate(rs):
            lcp = r["glm_lcp_with_prev"]
            if i > 0 and lcp:
                h = hit(kept, lcp, unit)
                lost += lcp - h
                zero += h == 0
                n += 1
            kept += place(r["glm_tokens"])
    return {
        "follow_requests": n,
        "recomputed_tokens": lost,
        "mean": lost / max(1, n),
        "zero_hit": zero / max(1, n),
    }


def calibrate(path: str, block: int, unit: int) -> None:
    reqs = json.load(open(path, encoding="utf-8"))["requests"]
    ok = n = 0
    kept: list[int] = []
    for r in reqs:
        if r["idx_in_chain"] == 0:
            kept = [(r["prompt_tokens"] // block) * block - block]
            continue
        pred = hit(kept, r["glm_lcp_with_prev"], unit)
        got = r["cached_tokens"] or 0
        n += 1
        ok += pred == got
        if pred != got:
            print(
                f"  mismatch: lcp {r['glm_lcp_with_prev']} measured {got} model {pred}"
            )
        kept.append((r["prompt_tokens"] // block) * block - block)
    print(
        f"calibration {path}: model matches {ok}/{n} measured hits (B={block}, u={unit})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root")
    ap.add_argument("--calibrate")
    ap.add_argument("--block", type=int, default=576)
    ap.add_argument("--unit", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=8192)
    args = ap.parse_args()
    if args.calibrate:
        calibrate(args.calibrate, args.block, args.block)
    if args.data_root:
        by = load_chains(args.data_root)
        for name, (unit, place) in policies(args.block, args.unit, args.chunk).items():
            e = estimate(by, unit, place)
            print(
                f"{name}: {e['recomputed_tokens'] / 1e6:.1f}M tokens over "
                f"{e['follow_requests']} follow-ups (mean {e['mean']:,.0f}, "
                f"zero-hit {e['zero_hit']:.1%})"
            )


if __name__ == "__main__":
    main()
