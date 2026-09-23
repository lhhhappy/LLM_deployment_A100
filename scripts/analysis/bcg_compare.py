# [T52 / patch 170] Compare two bcg_correctness.py outputs (e.g. eager vs BCG). Prints per-request diffs + summary JSON.
#   python3 bcg_compare.py <ref.json> <test.json> [out.json]
import json, sys

a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
a, b = sorted(a, key=lambda r: r["tag"]), sorted(b, key=lambda r: r["tag"])  # mixed pair finishes in any order
assert [r["tag"] for r in a] == [r["tag"] for r in b], "request lists differ"
rows = []
for x, y in zip(a, b):
    ta = {t: lp for lp, t in x["top1"]}; tb = {t: lp for lp, t in y["top1"]}
    common = set(ta) & set(tb)
    first_top_diff = max((abs(ta[t] - tb[t]) for t in common), default=None)
    n = min(len(x["out_ids"]), len(y["out_ids"]))
    div = next((i for i in range(n) if x["out_ids"][i] != y["out_ids"][i]), None)
    same_upto = n if div is None else div
    lp_diff = max((abs(p - q) for p, q in zip(x["out_lp"][:same_upto], y["out_lp"][:same_upto])), default=0.0)
    rows.append(dict(tag=x["tag"], n_in=x["n_in"], cached=[x["cached"], y["cached"]],
                     first_token_same=x["out_ids"][0] == y["out_ids"][0],
                     first_top5_same_set=set(ta) == set(tb), first_top5_max_abs_lp_diff=first_top_diff,
                     tokens_identical=f"{same_upto}/{n}", max_abs_lp_diff_on_common_prefix=lp_diff))
for r in rows:
    print(json.dumps(r))
summ = dict(
    n=len(rows),
    first_token_same=sum(r["first_token_same"] for r in rows),
    all_tokens_identical=sum(r["tokens_identical"].split("/")[0] == r["tokens_identical"].split("/")[1] for r in rows),
    max_first_top5_lp_diff=max(r["first_top5_max_abs_lp_diff"] or 0 for r in rows),
    max_lp_diff_common_prefix=max(r["max_abs_lp_diff_on_common_prefix"] for r in rows),
)
print("SUMMARY", json.dumps(summ))
if len(sys.argv) > 3:
    json.dump(dict(rows=rows, summary=summ), open(sys.argv[3], "w"), indent=1)
