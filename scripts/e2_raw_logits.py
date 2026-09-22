#!/usr/bin/env python3
"""D1-04 raw full-vocab cold/cold/resume check on three actual improved pairs.

Requires the E2 trace server, serial TP1, exclusive localhost endpoint. Warms the
entire ORIGINAL chain prefix, not just the previous prompt: the latter can miss
D1's warm-extend-only hook. Output is a functional diagnostic, never an SLO score.
Raw .pt tensors remain on the dev box. No prompt content is persisted here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

import torch


def post(url, path, payload):
    req = urllib.request.Request(url + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as response:
        assert response.status == 200
        return json.load(response)


def flush(url):
    result = post(url, "/flush_cache?timeout=120", {})
    assert result.get("success") is True, result


def generate(url, text, rid, tokens):
    result = post(url, "/generate", {
        "text": text, "rid": rid, "stream": False,
        "return_logprob": True, "logprob_start_len": -1,
        "return_text_in_logprobs": False,
        "sampling_params": {"temperature": 0, "max_new_tokens": tokens, "ignore_eos": True}})
    meta = result["meta_info"]
    assert meta["completion_tokens"] == tokens
    ids = [x[1] for x in meta["output_token_logprobs"]]
    assert len(ids) == tokens
    return {"prompt_tokens": meta["prompt_tokens"], "cached_tokens": meta["cached_tokens"],
            "greedy_ids": ids}


def select_pairs(comparison):
    selected, chains = [], set()
    for row in comparison["reminder_heavy"]["requests"]:
        if row["on_cached"] > row["stock_cached"] and row["chain_id"] not in chains:
            selected.append(row)
            chains.add(row["chain_id"])
            if len(selected) == 3:
                return selected
    raise RuntimeError("Need three distinct chains with an actual D1 cache improvement")


def compare_tensors(cold, repeat, warm):
    assert cold.shape == repeat.shape == warm.shape
    assert torch.isfinite(cold).all() and torch.isfinite(repeat).all() and torch.isfinite(warm).all()
    c, r, w = (x.float() for x in (cold, repeat, warm))
    noise = (c - r).abs().max().item()
    diff = (c - w).abs()
    return {"shape": list(c.shape), "saved_dtype": str(cold.dtype),
            "cold_repeat_max_abs": noise, "tolerance_2x_cold_noise": 2 * noise,
            "cold_warm_max_abs": diff.max().item(),
            "cold_warm_rms": diff.square().mean().sqrt().item(),
            "different_vocab_entries": int(torch.count_nonzero(diff)),
            "logits_within_declared_tolerance": diff.max().item() <= 2 * noise}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison", type=Path, required=True)
    p.add_argument("--trace-root", type=Path, required=True)
    p.add_argument("--dev-root", type=Path, required=True)
    p.add_argument("--variant", choices=("on", "off"), required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:31000")
    args = p.parse_args()
    root = args.trace_root.resolve()
    if not root.is_relative_to(Path("/sjtu/linhang/arena/runs/E2_20260922")):
        p.error("trace root outside E2 run")
    report_path = root / "numeric_summary.json"
    if report_path.exists():
        p.error("will not overwrite prior numeric run")
    selected = select_pairs(json.loads(args.comparison.read_text()))
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.dev_root / "harness"))
    from s1_common import Renderer, load_index, materialize_bodies
    data = args.dev_root / "data/dev-combined-v1"
    _, _, groups = load_index(str(data))
    renderer = Renderer(str(args.dev_root / "glm_tok"))
    report = {"case": "D1-04", "variant": args.variant,
              "purpose": "raw first-output-token logits, all vocabulary; not a timing run",
              "selected_pairs": selected, "results": []}
    control = root / "control.json"
    for pi, pair in enumerate(selected):
        rows = groups[pair["chain_id"]][:pair["idx_in_chain"] + 1]
        assert rows[-1]["_req_id"] == pair["req_id"]
        bodies = materialize_bodies(str(data), [r["_req_id"] for r in rows])
        texts = [renderer.render(bodies[r["_req_id"]]) for r in rows]
        prompt_tokens = len(renderer.tokenizer.encode(texts[-1], add_special_tokens=False))
        assert prompt_tokens == pair["prompt_tokens"]
        assert hashlib.sha256(texts[-1].encode()).hexdigest() == pair["prompt_sha256"]
        runs, tensors = [], []
        for stage in ("cold0", "cold1", "warm"):
            if control.exists():
                control.unlink()  # only this run's ephemeral arming file
            flush(args.base_url)
            history = []
            if stage == "warm":
                for ri, text in enumerate(texts[:-1]):
                    rid = f"e2num-{args.variant}-p{pi}-history{ri}"
                    measured = generate(args.base_url, text, rid, 4)
                    if ri == 0:
                        assert measured["cached_tokens"] == 0
                    history.append({"rid": rid, **measured})
            name = f"{args.variant}_pair{pi}_{stage}"
            rid = "e2num-" + name
            spec = {"name": name, "rid": rid, "prompt_tokens": prompt_tokens}
            temporary = root / "control.tmp"
            temporary.write_text(json.dumps(spec))
            temporary.replace(control)
            measured = generate(args.base_url, texts[-1], rid, 32)
            control.unlink()
            assert measured["prompt_tokens"] == prompt_tokens
            expected_cache = pair[args.variant + "_cached"] if stage == "warm" else 0
            assert measured["cached_tokens"] == expected_cache, (name, measured, expected_cache)
            raw_path = root / (name + ".pt")
            if not raw_path.exists():
                raise RuntimeError(f"Raw tensor not captured for {name}; do not count API logprobs as logits")
            raw = torch.load(raw_path, map_location="cpu", weights_only=True)
            assert raw["rid"] == rid and raw["seq_lens"] == [prompt_tokens]
            tensors.append(raw.pop("logits"))
            runs.append({"stage": stage, **measured, "history": history,
                         "raw_file": raw_path.name, "raw_metadata": raw})
        differences = compare_tensors(*tensors)
        splits_path = root / "splits.jsonl"
        splits = [json.loads(s) for s in splits_path.read_text().splitlines()] if splits_path.exists() else []
        previous_rid = runs[-1]["history"][-1]["rid"]
        boundary_proven = any(s["rid"] == previous_rid and
                              s["split_depth"] == runs[-1]["cached_tokens"] for s in splits)
        greedy_equal = runs[0]["greedy_ids"] == runs[1]["greedy_ids"] == runs[2]["greedy_ids"]
        result = {"pair": pair, "runs": runs, **differences,
                  "warm_restored_previous_role_split": boundary_proven,
                  "greedy_32_equal": greedy_equal,
                  "passed": differences["logits_within_declared_tolerance"] and greedy_equal and
                            (boundary_proven if args.variant == "on" else True)}
        report["results"].append(result)
        report["passed"] = len(report["results"]) == 3 and all(r["passed"] for r in report["results"])
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k not in ("runs", "pair")}), flush=True)
    flush(args.base_url)


if __name__ == "__main__":
    main()
