#!/usr/bin/env python3
"""Short synthetic endpoint probe. Flushes the specified DEVELOPMENT server.

Not an arena score, model-quality check, or HiCache eviction/reload check.
Stdlib only; no API keys or prompt contents are saved in receipts.
"""
import argparse
import json
import math
import os
import time
import urllib.request
import uuid
from pathlib import Path


def events(lines):
    """Parse SSE payloads, including comments and multiline data fields."""
    parts = []
    for line in lines:
        line = line.decode("utf-8").rstrip("\r\n")
        if not line:
            if parts:
                yield "\n".join(parts)
                parts = []
        elif line.startswith("data:"):
            parts.append(line[5:].removeprefix(" "))
    if parts:
        yield "\n".join(parts)


def generate(base, prompt, limit, rank, timeout, headers, expected_prompt_tokens=None):
    body = {"text": prompt, "stream": True, "rid": uuid.uuid4().hex,
            "sampling_params": {"max_new_tokens": limit, "temperature": 0, "ignore_eos": True}}
    if rank is not None:
        body["routed_dp_rank"] = rank
    req = urllib.request.Request(base + "/generate", json.dumps(body).encode(), headers=headers)
    start = time.monotonic()
    first = last = None
    first_s = last_s = None
    previous = 0
    done = False
    n_events = 0
    with urllib.request.urlopen(req, timeout=timeout) as response:
        assert response.headers.get_content_type() == "text/event-stream", "not SSE"
        for data in events(response):
            if data == "[DONE]":
                done = True
                break
            obj = json.loads(data)
            assert "error" not in obj, obj.get("error")
            meta = obj["meta_info"]
            for key in ("completion_tokens", "prompt_tokens", "cached_tokens"):
                assert type(meta.get(key)) is int and meta[key] >= 0, (key, meta.get(key))
            n = meta["completion_tokens"]
            assert previous <= n <= limit, ("invalid cumulative count", previous, n)
            assert meta["cached_tokens"] <= meta["prompt_tokens"]
            if expected_prompt_tokens is not None:
                assert meta["prompt_tokens"] == expected_prompt_tokens, "prompt token mismatch"
            if n > 0 and first is None:
                first, first_s = dict(meta), time.monotonic() - start
            if n > previous:
                last_s = time.monotonic() - start
            previous, last = n, dict(meta)
            n_events += 1
    assert done and first is not None and last["completion_tokens"] == limit, "incomplete generation"
    assert last.get("finish_reason", {}).get("type") == "length", "not length-limited"
    recv = last.get("request_received_ts")
    prefill = first.get("prefill_finished_time")
    server_timing = all(isinstance(t, (int, float)) and math.isfinite(t) and t > 1e9 for t in (recv, prefill))
    if server_timing:
        assert prefill >= recv, "negative server TTFT"
    return {"rank_requested": rank, "rank_reported": first.get("dp_rank"),
            "events": n_events, "prompt_tokens": first["prompt_tokens"],
            "cached_tokens": first["cached_tokens"], "completion_tokens": last["completion_tokens"],
            "has_native_server_timing": server_timing,
            "server_ttft_s": prefill - recv if server_timing else None,
            "client_ttft_s": first_s, "wall_s": time.monotonic() - start,
            "client_tpot_s": (last_s - first_s) / (limit - 1) if limit > 1 else None}


def run(args):
    base = args.base_url.rstrip("/").removesuffix("/v1")
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("ARENA_PROBE_API_KEY")
    if key:
        headers["Authorization"] = "Bearer " + key
    # Stable across cold/warm/post-flush, unique across separate probe runs.
    prompt = args.prompt_file.read_text() if args.prompt_file else (
        "Synthetic serving contract check " + uuid.uuid4().hex + ".\n" +
        "The following paragraph is a repeated context for a cache test. " * 128 + "\nContinue:")
    ranks = list(range(args.dp_size)) if args.dp_size else [None]
    result = {"kind": "synthetic_contract_probe_not_score", "checks": [], "generations": [], "flushes": []}

    def check(name, ok):
        result["checks"].append({"name": name, "passed": bool(ok)})

    def flush():
        req = urllib.request.Request(base + "/flush_cache", b"{}", headers=headers)
        with urllib.request.urlopen(req, timeout=args.timeout) as response:
            raw = response.read()
            try:
                data = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                data = None
            ok = isinstance(data, dict) and data.get("success") is True
            result["flushes"].append({"status": response.status, "json_success": ok})
            check("flush_json_contract", ok)

    try:
        with urllib.request.urlopen(urllib.request.Request(base + "/v1/models", headers=headers), timeout=args.timeout) as response:
            check("models", response.status == 200)
        flush()
        for rank in ranks:
            pair = []
            for label in ("cold", "warm"):
                item = generate(base, prompt, args.tokens, rank, args.timeout, headers, args.expected_prompt_tokens)
                item["case"] = label
                result["generations"].append(item)
                pair.append(item)
                check(f"native_timing_{rank}_{label}", item["has_native_server_timing"])
                if item["rank_reported"] is not None and rank is not None:
                    check(f"rank_{rank}_{label}", item["rank_reported"] == rank)
            check(f"cold_cache_{rank}", pair[0]["cached_tokens"] == 0)
            check(f"warm_cache_{rank}", pair[1]["cached_tokens"] > pair[0]["cached_tokens"])
        flush()
        for rank in ranks:
            item = generate(base, prompt, args.tokens, rank, args.timeout, headers, args.expected_prompt_tokens)
            item["case"] = "after_global_flush"
            result["generations"].append(item)
            check(f"flush_clears_rank_{rank}", item["cached_tokens"] == 0)
        # Leave development server cold for a subsequent independent run.
        flush()
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["passed"] = "error" not in result and bool(result["checks"]) and all(c["passed"] for c in result["checks"])
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", required=True)
    p.add_argument("--dp-size", type=int, default=0, help="0: normal routing; positive: probe each rank explicitly")
    p.add_argument("--tokens", type=int, default=17)
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--prompt-file", type=Path)
    p.add_argument("--expected-prompt-tokens", type=int)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.tokens < 1 or args.dp_size < 0:
        p.error("tokens must be positive and dp-size nonnegative")
    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
