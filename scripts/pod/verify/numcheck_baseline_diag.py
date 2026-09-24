#!/usr/bin/env python3
"""Record repeated cold greedy requests on one unchanged engine for diagnosis.

Usage: numcheck_baseline_diag.py OUT_DIR [--mode both|first|full]
This is not a numerical acceptance or capability test.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import urllib.request


TEXT = ("The committee reviewed the proposal in detail. Each member described the risks, the costs and the expected "
        "benefits, and then they voted. ")
TEXT_SHORT = TEXT * 3 + "\nSummarize the discussion above in one sentence:"


def ids(length):
    rng = random.Random(1000 + length)
    return [rng.randrange(1000, int(os.environ.get("VOCAB_MAX", "150000")))
            for _ in range(length)]


def cases(mode):
    first = [(f"cold_{length}", {"input_ids": ids(length)}, 1, 3)
             for length in (37, 256, 1024)]
    first.append(("text_short", {"text": TEXT_SHORT}, 1, 3))
    full = [("cold_37", {"input_ids": ids(37)}, 48, 2),
            ("text_short", {"text": TEXT_SHORT}, 48, 2)]
    return (first if mode in ("both", "first") else []) + (full if mode in ("both", "full") else [])


def request(path, body, timeout):
    base = f"http://127.0.0.1:{os.environ.get('PORT', '30000')}"
    req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout).read()


def flush():
    # A failed flush invalidates the cold-request comparison immediately.
    request("/flush_cache", {}, 60)


class InvalidResponse(RuntimeError):
    def __init__(self, message, body, raw_response):
        super().__init__(message)
        self.body = body
        self.raw_response = raw_response.decode("utf-8", errors="replace")


def generate(prompt, count):
    body = dict(prompt)
    body.update(sampling_params={"max_new_tokens": count, "temperature": 0,
                                 "ignore_eos": True},
                return_logprob=True, top_logprobs_num=10)
    raw = request("/generate", body, 1800)
    try:
        result = json.loads(raw)
        meta = result["meta_info"]
        output = meta["output_token_logprobs"]
        tops = meta["output_top_logprobs"]
        if len(output) != count or len(tops) != count:
            raise ValueError(f"truncated diagnostic output: wanted {count}, got "
                             f"logprobs={len(output)} top_logprobs={len(tops)}")
        if meta.get("cached_tokens") != 0:
            raise ValueError(f"cold diagnostic request used cache: {meta.get('cached_tokens')}")
        if not tops[0] or len(tops[0]) < 10:
            raise ValueError("diagnostic response lacks first-token top 10")
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        raise InvalidResponse(str(exc), body, raw) from exc
    return body, result


def summarize(records):
    summary = []
    groups = {}
    for record in records:
        groups.setdefault((record["case"], record["max_new_tokens"]), []).append(record)
    for (name, count), group in groups.items():
        anchor = group[0]["response"]["meta_info"]["output_token_logprobs"]
        comparisons = []
        for record in group[1:]:
            other = record["response"]["meta_info"]["output_token_logprobs"]
            first = next((i for i, (a, b) in enumerate(zip(anchor, other)) if a[1] != b[1]), None)
            prefix_end = count if first is None else first
            comparisons.append({"repeat": record["repeat"], "first_token_divergence": first,
                                "first_logprob_delta": other[0][0] - anchor[0][0],
                                "max_abs_logprob_delta": max(abs(b[0] - a[0]) for a, b in zip(anchor, other)),
                                "max_abs_logprob_delta_before_divergence": (
                                    max(abs(other[i][0] - anchor[i][0]) for i in range(prefix_end))
                                    if prefix_end else None),
                                "logprob_delta_scope": (
                                    "all positions share token history" if first is None
                                    else "full delta is descriptive after token divergence")})
        summary.append({"case": name, "max_new_tokens": count,
                        "first_token_ids": [r["response"]["meta_info"]["output_token_logprobs"][0][1] for r in group],
                        "first_token_top10": [r["response"]["meta_info"]["output_top_logprobs"][0] for r in group],
                        "comparisons_to_first": comparisons})
    return summary


def run(out_dir, mode):
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with (out_dir / "responses.jsonl").open("w") as handle:
        for name, prompt, count, repeats in cases(mode):
            prompt_hash = hashlib.sha256(json.dumps(prompt, sort_keys=True).encode()).hexdigest()
            for repeat in range(1, repeats + 1):
                flush()
                try:
                    body, response = generate(prompt, count)
                except InvalidResponse as exc:
                    (out_dir / "failure.json").write_text(json.dumps({
                        "case": name, "repeat": repeat, "error": str(exc),
                        "request": exc.body, "raw_response": exc.raw_response,
                    }, indent=2) + "\n")
                    raise
                record = {"case": name, "repeat": repeat, "max_new_tokens": count,
                          "prompt_sha256": prompt_hash, "request": body, "response": response}
                handle.write(json.dumps(record, allow_nan=False) + "\n")
                handle.flush()
                records.append(record)
    summary = {"mode": mode, "requests": len(records), "cases": summarize(records),
               "scope": "diagnostic only; no numerical or capability verdict"}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print("DIAGNOSTIC_COMPLETE", json.dumps({"mode": mode, "requests": len(records)}, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--mode", choices=("both", "first", "full"), default="both")
    args = parser.parse_args()
    run(args.out_dir, args.mode)


if __name__ == "__main__":
    main()
