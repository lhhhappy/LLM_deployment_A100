#!/usr/bin/env python3
"""E1/E2 random-stand-in cache replay, NOT an arena score or replacement harness.

Uses unchanged harness Renderer + call_engine; full prompts and ordered chains.
Default: sequential chains, a verified flush before each chain, tiny output.
F13/F24 predictions intentionally remain an ideal token-only model (no decode,
eviction, finite state slots, resident FULL-KV truncation, or cross-chain sharing).
E1 observed 2/72 stock mismatches; predictions are not an engine oracle.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def lcp(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def predict_policy(prompts, role_ids, policy="stock", page_size=64, chunk_size=8192):
    """Port of scripts/sim_role_boundary.py v2 semantics on the selected sequence.

    Deliberately NOT an engine simulator. Preserves F13/F24's idealized per-chain
    tree and chunk-point loop, including their omissions of resident FULL-KV
    lifetime and per-extend branch placement (see E1 in notes/experiments.md).
    """
    if policy not in ("stock", "role_conservative", "role_over_branch", "role_all"):
        raise ValueError(policy)
    if page_size <= 0 or chunk_size <= 0 or chunk_size % page_size:
        raise ValueError("positive, page-aligned chunk_size required")
    floor = lambda x: x // page_size * page_size
    store = [(0, None)]
    results = []
    for i, tokens in enumerate(prompts):
        hit = 0
        for depth, previous in store:
            if depth > hit and (previous is None or
                    (len(tokens) >= depth and prompts[previous][:depth] == tokens[:depth])):
                hit = depth
        common = lcp(prompts[i - 1], tokens) if i else 0
        end = floor(len(tokens))
        branch = floor(common) if common > hit and floor(common) > hit else None
        candidates = [j for j in range(hit, len(tokens)) if tokens[j] in role_ids]
        role = floor(candidates[-1]) if candidates else None
        if role is not None and not hit < role < end:
            role = None
        added = []
        pos = hit
        while pos + chunk_size < len(tokens):
            pos += chunk_size
            added.append(floor(pos))
        conflict = branch is not None
        if policy == "stock" or role is None:
            added.append(branch if conflict else end)
        elif policy == "role_conservative" and conflict:
            added.append(branch)
        else:
            added.extend([role, end])
            if policy == "role_all" and conflict:
                added.append(branch)
        store.extend((depth, i) for depth in added)
        results.append({"cached_tokens": hit, "uncached_tokens": len(tokens) - hit,
                        "previous_lcp": common, "role_boundary": role,
                        "branch_boundary": branch, "checkpoints_added": added})
    return results


def flush_cache(base_url, require_json=False):
    """Verify the response, accepting the exact native stock text only in E1."""
    req = urllib.request.Request(base_url.rstrip("/") + "/flush_cache", method="POST")
    with urllib.request.urlopen(req, timeout=60) as response:
        status = response.status
        raw = response.read().decode("utf-8")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    json_success = isinstance(parsed, dict) and parsed.get("success") is True
    native_success = raw.startswith("Cache flushed.\n")
    success = status == 200 and (json_success or (not require_json and native_success))
    record = {"kind": "flush", "http_status": status, "body": raw,
              "json_contract": json_success, "success": success, "at": time.time()}
    if not success:
        raise RuntimeError(f"Flush did not confirm success: {record}")
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dev-root", type=Path, required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:31000")
    p.add_argument("--output", type=Path, required=True, help="new run directory")
    p.add_argument("--num-chains", type=int, default=3)
    p.add_argument("--chain-id", action="append", default=[])
    p.add_argument("--case-file", type=Path,
                   help="INTERNAL ordered chain-prefix JSON from make_case_sets.py; not an official cohort")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=4)
    p.add_argument("--max-prompt-tokens", type=int, default=65536,
                   help="select only chains entirely within limit; never truncate")
    p.add_argument("--chunk-size", type=int, default=8192)
    p.add_argument("--page-size", type=int, default=64)
    p.add_argument("--request-timeout", type=int, default=1200)
    p.add_argument("--require-json-flush", action="store_true")
    p.add_argument("--smoke", action="store_true", help="cold/repeat/flush/cold first")
    p.add_argument("--plan-only", action="store_true", help="render/predict, no HTTP calls")
    args = p.parse_args()
    if min(args.num_chains, args.concurrency, args.max_new_tokens, args.max_prompt_tokens) < 1:
        p.error("counts and limits must be positive")
    if args.output.exists():
        p.error("output must be a new directory; will not overwrite prior results")
    if args.case_file and args.chain_id:
        p.error("--case-file and --chain-id are mutually exclusive")
    # Imports of the read-only harness must not create __pycache__ files.
    sys.dont_write_bytecode = True
    harness = args.dev_root / "harness"
    sys.path.insert(0, str(harness.resolve()))
    from s1_common import Renderer, load_index, materialize_bodies, q
    from s1_loadgen import call_engine

    root = args.dev_root / "data/dev-combined-v1"
    _, _, groups = load_index(str(root))
    case = None
    if args.case_file:
        from make_case_sets import validate_case
        case = validate_case(json.loads(args.case_file.read_text()), groups)
        groups = {c["chain_id"]: groups[c["chain_id"]][:len(c["req_ids"])] for c in case}
    renderer = Renderer(str(args.dev_root / "glm_tok"))
    role_ids = set(renderer.tokenizer.convert_tokens_to_ids(["<|user|>", "<|observation|>"]))
    if None in role_ids or len(role_ids) != 2:
        raise RuntimeError("Expected two distinct role token IDs")
    requested = ([c["chain_id"] for c in case] if case else
                 args.chain_id or sorted(cid for cid, rows in groups.items() if len(rows) >= 2))
    if any(cid not in groups for cid in requested):
        p.error("unknown or non-serving chain-id")
    wanted = [r["_req_id"] for cid in requested for r in groups[cid]]
    bodies = materialize_bodies(str(root), wanted)
    missing = set(wanted) - bodies.keys()
    if missing:
        raise RuntimeError(f"Missing bodies: {len(missing)}; refusing incomplete replay")
    selected, excluded = [], []
    policies = ("stock", "role_conservative", "role_over_branch", "role_all")
    for cid in requested:
        rows = groups[cid]
        texts = [renderer.render(bodies[r["_req_id"]]) for r in rows]
        token_lists = [renderer.tokenizer.encode(t, add_special_tokens=False) for t in texts]
        longest = max(map(len, token_lists))
        if longest > args.max_prompt_tokens:
            excluded.append({"chain_id": cid, "reason": "entire_chain_over_limit",
                             "max_prompt_tokens": longest})
            if args.chain_id or case:
                raise RuntimeError(f"Explicit chain {cid} has {longest} tokens; increase limit")
            continue
        predictions = {name: predict_policy(token_lists, role_ids, name,
                                            args.page_size, args.chunk_size)
                       for name in policies}
        selected.append((cid, rows, texts, token_lists, predictions))
        if len(selected) >= (len(requested) if args.chain_id or case else args.num_chains):
            break
    if not selected or (not (args.chain_id or case) and len(selected) != args.num_chains):
        raise RuntimeError("Not enough complete eligible chains")
    args.output.mkdir(parents=True)
    run_id = args.output.name
    manifest = {
        "purpose": "random-stand-in functional replay; NOT a benchmark score",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "selection": [{"chain_id": cid, "req_ids": [r["_req_id"] for r in rows],
                       "prompt_lengths": list(map(len, tokens))}
                      for cid, rows, _, tokens, _ in selected],
        "excluded": excluded, "ignore_eos": True, "pacing": "none(functional)",
        "cache_isolation": "flush-per-chain" if args.concurrency == 1 else "shared-cache",
        "prediction_scope": "F13/F24 ideal per-chain tokens; no decode/eviction/resident-FULL-KV-truncation/per-extend-branch-placement/cross-chain sharing",
        "hashes": {name: sha256(harness / name)
                   for name in ("s1_common.py", "s1_loadgen.py")},
        "tokenizer_sha256": sha256(args.dev_root / "glm_tok/tokenizer.json"),
        "replay_sha256": sha256(__file__),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    os.environ["S1_ENGINE_URL"] = args.base_url
    os.environ["S1_REQUEST_TIMEOUT_S"] = str(args.request_timeout)
    lock, records = threading.Lock(), []
    log_path = args.output / "requests.jsonl"
    events_path = args.output / "events.jsonl"

    def emit(record, events=False):
        with lock:
            with (events_path if events else log_path).open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if not events:
                records.append(record)
            print(json.dumps({k: v for k, v in record.items() if k != "predictions"}, ensure_ascii=False), flush=True)

    def call(text, rid, cid):
        return call_engine({"req_id": rid, "session_id": cid,
                            "cache_namespace": run_id, "routing_key": cid},
                           text, args.max_new_tokens)

    if args.smoke and not args.plan_only:
        emit(flush_cache(args.base_url, args.require_json_flush), events=True)
        smoke_text = selected[0][2][0]
        smoke = []
        for stage in ("cold", "repeat", "after_flush"):
            if stage == "after_flush":
                emit(flush_cache(args.base_url, args.require_json_flush), events=True)
            result = call(smoke_text, f"{run_id}:smoke:{stage}", "E1-smoke")
            emit({"kind": "smoke", "stage": stage, **result}, events=True)
            smoke.append(result)
        if any(r.get("error") or r.get("output_tokens") != args.max_new_tokens for r in smoke):
            raise RuntimeError("Smoke request failed; see events.jsonl")
        if not (smoke[0]["cached_tokens"] == 0 and smoke[1]["cached_tokens"] > 0
                and smoke[2]["cached_tokens"] == 0):
            raise RuntimeError("Cold/repeat/flush/cold cache invariant failed")

    def run_chain(item):
        cid, rows, texts, token_lists, predictions = item
        if not args.plan_only and args.concurrency == 1:
            emit({**flush_cache(args.base_url, args.require_json_flush), "chain_id": cid}, events=True)
        for i, (row, text, tokens) in enumerate(zip(rows, texts, token_lists)):
            record = {"chain_id": cid, "idx_in_chain": i, "req_id": row["_req_id"],
                      "phase": row.get("phase"), "uncached_expected": row.get("uncached_expected"),
                      "rendered_prompt_tokens": len(tokens),
                      "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                      "predictions": {name: predictions[name][i] for name in policies}}
            if not args.plan_only:
                record.update(call(text, f"{run_id}:{row['_req_id']}", cid))
                if record.get("error"):
                    emit(record)
                    raise RuntimeError("Request failed; stopped its chain without retry")
                actual = record.get("cached_tokens")
                if (actual is None or not 0 <= actual <= len(tokens)
                        or record.get("prompt_tokens") != len(tokens)
                        or record.get("output_tokens") != args.max_new_tokens):
                    record["error"] = "token_count_missing_or_tokenization_mismatch"
                    emit(record)
                    raise RuntimeError(record["error"])
                record["actual_uncached_tokens"] = len(tokens) - actual
                record["stock_cached_delta"] = actual - predictions["stock"][i]["cached_tokens"]
            emit(record)
        return cid

    if not args.plan_only and args.concurrency > 1:
        emit(flush_cache(args.base_url, args.require_json_flush), events=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(run_chain, selected))
    fast = [r for r in records if r["idx_in_chain"] > 0 and
            r["phase"] not in ("turn_start", "context_reset") and
            (r["uncached_expected"] or 0) <= 4096]
    summary = {
        "plan_only": args.plan_only,
        "requests": len(records), "chains": len(selected), "fast_intra_requests": len(fast),
        "errors": sum(bool(r.get("error")) for r in records),
        "exact_stock_prediction_matches": None if args.plan_only else sum(r.get("stock_cached_delta") == 0 for r in records),
        "stock_cached_delta_min": min((r["stock_cached_delta"] for r in records if "stock_cached_delta" in r), default=None),
        "stock_cached_delta_max": max((r["stock_cached_delta"] for r in records if "stock_cached_delta" in r), default=None),
        "fast_intra_actual_uncached_p95": q([r.get("actual_uncached_tokens") for r in fast], .95),
        "fast_intra_predicted_stock_uncached_p95": q([r["predictions"]["stock"]["uncached_tokens"] for r in fast], .95),
        "ttft_source_counts": {s: sum(r.get("ttft_source") == s for r in records)
                               for s in ("server", "client_proxy")},
        "limitations": manifest["prediction_scope"],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
