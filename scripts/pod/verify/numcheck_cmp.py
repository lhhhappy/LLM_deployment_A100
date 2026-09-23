# Diff two numcheck.json files: per case, first diverging token index and max |logprob diff| over the common prefix.
# Usage: python3 numcheck_cmp.py <ref.json> <cand.json>   -> prints a table and NUMCMP summary line
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
bad = 0
for k in a:
    if k not in b: print(f"{k:16s} MISSING"); bad += 1; continue
    ta, tb = a[k]["tokens"], b[k]["tokens"]
    div = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
    upto = div if div is not None else min(len(ta), len(tb))
    lp = max([abs(x - y) for x, y in zip(a[k]["logprobs"][:upto], b[k]["logprobs"][:upto])] or [0.0])
    first = abs(a[k]["logprobs"][0] - b[k]["logprobs"][0]) if ta and tb and ta[0] == tb[0] else float("inf")
    # token 0 differing, or a large first-token logprob gap, means the prefill itself is wrong
    verdict = "WRONG" if (div == 0 or first > 0.5) else ("drift" if div is not None and div < 8 else "ok")
    bad += verdict == "WRONG"
    print(f"{k:16s} first_div={str(div):5s} first_lp_diff={first:.3g} max_lp_diff_common={lp:.3g} {verdict}")
print(f"NUMCMP wrong={bad}/{len(a)}")
