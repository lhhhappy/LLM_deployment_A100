#!/usr/bin/env python3
"""Profile one frozen, cold long-chain prompt on the running TP8 SGLang service.

This is an execution-cost probe, not a replay or SLO measurement. The caller
must have warmed the engine. This script flushes KV, profiles exactly one
request with max_new_tokens=1, and records the response and trace inventory.
"""

import argparse
import gzip
import json
import sys
import time
import urllib.request
from pathlib import Path


def post(url, payload=None, timeout=180):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status} from {url}")
    return json.loads(raw) if raw.startswith(b"{") else raw.decode("utf-8", "replace")


def select(root, rid):
    for line in (root / "requests.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        current = f"{row['pack']}:{row['view']}:{row['logical_call_id']}"
        if current != rid:
            continue
        with gzip.open(root / row["body_ref"], "rt", encoding="utf-8") as bodies:
            for body_line in bodies:
                body = json.loads(body_line)
                if body["req_id"] == rid:
                    return row, body
        raise RuntimeError(f"body missing for {rid}")
    raise RuntimeError(f"request missing for {rid}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tok-dir", type=Path, required=True)
    parser.add_argument("--rid", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    profile_dir = out / "traces"
    profile_dir.mkdir(exist_ok=False)
    row, body = select(args.root, args.rid)
    sys.path.insert(0, str(Path(__file__).parent.parent / "s1" / "s1-dev" / "harness"))
    # In the pod the kit lives at /tmp/ax/verify_kit, not under the source repo.
    from s1_common import Renderer

    renderer = Renderer(str(args.tok_dir))
    prompt = renderer.render(body)
    rendered_tokens = renderer.n_tokens(prompt)
    if not 48000 <= rendered_tokens <= 49152:
        raise RuntimeError(f"expected 48–49k cold prompt, got {rendered_tokens} tokens")
    flush = post(args.base_url + "/flush_cache")
    (out / "flush_response.txt").write_text(str(flush) + "\n", encoding="utf-8")
    if not isinstance(flush, dict) or flush.get("success") is not True:
        raise RuntimeError("cold profile aborted: /flush_cache did not acknowledge success=true")
    profile_req = {
        "output_dir": str(profile_dir), "profile_id": args.tag,
        "profile_prefix": args.tag, "activities": ["CPU", "GPU"],
        "with_stack": False, "record_shapes": False,
        "merge_profiles": False,
    }
    started = time.monotonic()
    post(args.base_url + "/start_profile", profile_req)
    try:
        result = post(args.base_url + "/generate", {
            "text": prompt,
            "sampling_params": {"max_new_tokens": 1, "temperature": 0, "ignore_eos": True},
            "stream": False, "rid": f"tp8-profile-{args.tag}",
        }, timeout=600)
    finally:
        stopped = post(args.base_url + "/stop_profile", timeout=600)
    elapsed = time.monotonic() - started
    traces = sorted(profile_dir.glob("*.trace.json.gz"))
    ranks = {rank for rank in range(8) if any(f"TP-{rank}" in p.name for p in traces)}
    if not {0, 1}.issubset(ranks):
        raise RuntimeError(f"rank 0/1 traces missing; found ranks {sorted(ranks)}")
    meta = result.get("meta_info", {}) if isinstance(result, dict) else {}
    cached = meta.get("cached_tokens")
    if cached is not None and int(cached) > 64:
        raise RuntimeError(f"cold profile was not cold: cached_tokens={cached}")
    emitted = meta.get("completion_tokens", meta.get("output_tokens"))
    if emitted is not None and int(emitted) != 1:
        raise RuntimeError(f"profile output length changed: output_tokens={emitted}")
    receipt = {
        "tag": args.tag, "rid": args.rid,
        "frozen_glm_tokens": row["glm_tokens"], "rendered_tokens": rendered_tokens,
        "profiled_wall_s": elapsed, "flush_response": flush, "stop_response": stopped,
        "response_meta": meta,
        "traces": [{"name": p.name, "bytes": p.stat().st_size} for p in traces],
        "ranks": sorted(ranks),
        "scope": "single cold request; profiled; not a replay or performance verdict",
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in
                      ("tag", "rendered_tokens", "profiled_wall_s", "ranks", "response_meta")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
