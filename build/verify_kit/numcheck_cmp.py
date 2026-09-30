# Compare complete numerical fingerprints. Any missing/extra case, token drift, truncation,
# or non-finite/large logprob difference is a mismatch. Kept output-compatible with ladder jobs.
# Usage: python3 numcheck_cmp.py <ref.json> <cand.json>  -> NUMCMP wrong=M/N
import json
import math
import sys

ref, cand = (json.load(open(p)) for p in sys.argv[1:3])
bad = 0
for name in sorted(set(ref) | set(cand)):
    if name not in ref or name not in cand:
        print(f"{name:16s} {'EXTRA' if name not in ref else 'MISSING'} WRONG")
        bad += 1
        continue
    a, b = ref[name], cand[name]
    ta, tb = a.get("tokens", []), b.get("tokens", [])
    la, lb = a.get("logprobs", []), b.get("logprobs", [])
    div = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
    if div is None and len(ta) != len(tb):
        div = min(len(ta), len(tb))
    diffs = [abs(x - y) for x, y in zip(la, lb)]
    max_diff = max(diffs, default=float("inf"))
    wrong = (not ta or div is not None or len(la) != len(ta) or len(lb) != len(tb)
             or not all(math.isfinite(x) for x in la + lb) or max_diff > 0.5)
    bad += wrong
    print(f"{name:16s} first_div={str(div):5s} max_lp_diff={max_diff:.3g} {'WRONG' if wrong else 'ok'}")
print(f"NUMCMP wrong={bad}/{len(set(ref) | set(cand))}")
