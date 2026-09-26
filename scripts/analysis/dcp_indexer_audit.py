#!/usr/bin/env python3
"""Compare captured row 3950: exact scores, valid index-K bytes and selections.

This is a scoped numerical diagnosis, not a replacement model-quality gate.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch


def observations(root, arm, rank):
    path = root / (arm + ".pt" + (f".rank{rank}" if rank else ""))
    index = torch.load(str(path) + ".indexer", map_location="cpu")
    forward = torch.load(path, map_location="cpu")
    # This fixture has exactly two sparse layers. File existence alone must
    # not turn a missing hook or duplicate observation into a complete matrix.
    if sorted(r["layer"] for r in index) != [3, 7]:
        raise ValueError(f"Incomplete/duplicate indexer layers in {path}")
    info = forward["info"]
    row = 0 if info.get("attention_capture_row") == 3950 else 3950
    for r in index:
        name = f"model.layers.{r['layer']}.self_attn"
        r["attention"] = forward["attn"]["ext"][name][row].clone()
        r["greedy"] = forward["ext"].argmax(-1).tolist()
    return {r["layer"]: r for r in index}


def groups(record):
    return set(int(v) for v in record["selected"][:2048:4] if v >= 0)


def scores(record):
    start, end = int(record["key_start"]), int(record["key_end"])
    return record["logits"][start:end]


def changed_bytes(a, b):
    return int((a.contiguous().view(torch.uint8) != b.contiguous().view(torch.uint8)).sum())


def compare(record, ref):
    start, end = int(record["key_start"]), int(record["key_end"])
    assert (start, end) == (int(ref["key_start"]), int(ref["key_end"]))
    selected, reference = groups(record), groups(ref)
    if len(selected) != 512 or len(reference) != 512:
        raise ValueError("This fixture must select 512 distinct KPool groups")
    a, b = record["attention"], ref["attention"]
    result = dict(computed_here=record["computed_here"], selected_groups=len(selected),
                  selected_only=sorted(selected-reference), reference_only=sorted(reference-selected),
                  pair_present={str(g): g in selected for g in (21028, 23340)},
                  valid_key_bytes_changed=int((record["pooled_key_bytes"][start:end] != ref["pooled_key_bytes"][start:end]).sum()),
                  valid_key_scales_changed=int((record["pooled_key_scales"][start:end] != ref["pooled_key_scales"][start:end]).sum()),
                  valid_key_scale_bytes_changed=changed_bytes(record["pooled_key_scales"][start:end], ref["pooled_key_scales"][start:end]),
                  query_bytes_changed=int((record["query_bytes"] != ref["query_bytes"]).sum()),
                  weights_bytes_changed=changed_bytes(record["weights"], ref["weights"]),
                  weights_max_abs_delta=float((record["weights"]-ref["weights"]).abs().max()),
                  attention_row_relative_linf=float((a-b).abs().max()/b.abs().max().clamp_min(1e-8)),
                  greedy_equal=record["greedy"] == ref["greedy"])
    if record["computed_here"]:
        x, y = scores(record), scores(ref)
        assert bool(torch.isfinite(x).all()) and len(x) > 512
        values, indices = x.topk(513)
        table = record["page_table"][:len(x)*4:4]
        result.update(boundary_512_513=values[-2:].tolist(), boundary_gap=float(values[-2]-values[-1]),
                      boundary_groups=table[indices[-2:]].tolist(),
                      logits_max_abs_delta=float((x-y).abs().max()), logits_bytes_changed=changed_bytes(x, y))
        pair_columns = [int(torch.nonzero(table == g).flatten().item()) for g in (21028, 23340)]
        result["pair_scores"] = {str(g): float(x[c]) for g,c in zip((21028,23340),pair_columns)}
        ka, kb = [record["pooled_key_bytes"][start+c] for c in pair_columns]
        sa, sb = [record["pooled_key_scales"][start+c] for c in pair_columns]
        result["pair_keys_equal"] = bool(torch.equal(ka, kb) and sa == sb)
        # An exact tie may choose either member; check membership at the true
        # FP32 threshold without changing the production selection.
        selected_mask = torch.tensor([int(v) in selected for v in table])
        result["selection_regret"] = float(x[~selected_mask].max()-x[selected_mask].min())
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--pairs", type=int, default=10)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.pairs < 1:
        p.error("pairs must be positive")
    refs = {rank: observations(args.root, "review_base_01", rank) for rank in (0,1)}
    rows, missing, variants = [], [], {}
    for i in range(1, args.pairs+1):
        for arm in ("base", "local"):
            name = f"review_{arm}_{i:02}"
            for rank in (0,1):
                path = args.root / (name + ".pt" + (f".rank{rank}" if rank else ""))
                if not path.is_file() or not Path(str(path)+".indexer").is_file():
                    missing.append(str(path.name))
                    continue
                obs = observations(args.root, name, rank)
                for layer, record in obs.items():
                    row = dict(arm=name, kind=arm, rank=rank, layer=layer, **compare(record, refs[rank][layer]))
                    rows.append(row)
                    key = f"{arm}:rank{rank}:layer{layer}"
                    variants.setdefault(key, Counter())[tuple(sorted(groups(record)))] += 1
    result = dict(scope="row-3950 diagnosis with identical capture in both arms; not a TP8 quality verdict",
                  complete=not missing, missing=missing,
                  selection_variants={k:dict(variants=len(v),counts=sorted(v.values(),reverse=True)) for k,v in variants.items()},
                  rows=rows)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: v for k,v in result.items() if k != "rows"}, indent=2))
    raise SystemExit(0 if result["complete"] else 1)


if __name__ == "__main__":
    main()
