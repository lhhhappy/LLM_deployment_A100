#!/usr/bin/env python3
"""Interface probe against a running server (task.md "提交前自查" plus accounting).

Checks, in order, and writes a JSON receipt:
  1. GET  /v1/models returns 200.
  2. POST /generate SSE for real replay prompts: every event has meta_info,
     completion_tokens never decreases and ends at max_new_tokens (ignore_eos),
     prompt_tokens equals the frozen glm_tokens, request_received_ts <=
     prefill_finished_time <= the client's first-event time.
  3. The same prompt again: cached_tokens rises.
  4. POST /flush_cache returns 2xx {"success": true}; the prompt again: cached_tokens == 0.
  5. CONCURRENCY parallel /generate requests all complete without errors.
  6. POST /v1/chat/completions round trip returns a message.
Exit 0 only if every check passes. Prompts are rendered with the harness
Renderer, exactly as the load generator sends them.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for path in (os.path.join(REPO, "s1-dev", "harness"), os.environ.get("HARNESS_DIR", "")):
    if path:
        sys.path.insert(0, path)


def post_json(url: str, payload: dict | None, timeout: float) -> tuple[int, dict]:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, method="POST", headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read(2048).decode("utf-8", "replace")}


def generate(base: str, text: str, max_new_tokens: int, rid: str, timeout: float) -> dict:
    """One streamed /generate; returns every meta_info plus client timings."""
    payload = {
        "text": text,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0,
            "ignore_eos": True,
        },
        "stream": True,
        "rid": rid,
    }
    req = urllib.request.Request(
        base + "/generate",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    metas, first_client, done, errors = [], None, False, []
    sent = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    done = True
                    continue
                event = json.loads(body)
                if "error" in event:
                    errors.append(event["error"])
                    continue
                meta = event.get("meta_info")
                if meta is None:
                    errors.append("event without meta_info")
                    continue
                if first_client is None and int(meta.get("completion_tokens") or 0) > 0:
                    first_client = time.time()
                metas.append(meta)
    except urllib.error.HTTPError as e:
        errors.append(f"HTTP {e.code}: {e.read(512).decode('utf-8', 'replace')}")
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        errors.append(repr(e))
    return {
        "rid": rid,
        "sent": sent,
        "first_client": first_client,
        "done": done,
        "errors": errors,
        "metas": metas,
    }


def check_stream(r: dict, max_new_tokens: int, glm_tokens: int | None) -> list[str]:
    bad = list(r["errors"])
    metas = r["metas"]
    if not r["done"]:
        bad.append("no [DONE]")
    if not metas:
        return bad + ["no events"]
    counts = [int(m["completion_tokens"]) for m in metas]
    if counts != sorted(counts):
        bad.append(f"completion_tokens decreased: {counts[:20]}")
    if counts[-1] != max_new_tokens:
        bad.append(f"final completion_tokens {counts[-1]} != {max_new_tokens}")
    first = next(m for m in metas if int(m["completion_tokens"]) > 0)
    recv, pft = first.get("request_received_ts"), first.get("prefill_finished_time")
    if recv is None or pft is None:
        bad.append("missing request_received_ts/prefill_finished_time")
    else:
        if not recv <= pft:
            bad.append(f"prefill_finished_time {pft} < request_received_ts {recv}")
        if r["first_client"] is not None and pft > r["first_client"] + 0.05:
            bad.append("prefill_finished_time later than the client saw the token")
    if metas[-1].get("request_received_ts") != recv:
        bad.append("request_received_ts differs between first and final event")
    if glm_tokens is not None and int(first["prompt_tokens"]) != glm_tokens:
        bad.append(f"prompt_tokens {first['prompt_tokens']} != glm_tokens {glm_tokens}")
    return bad


def summarize(r: dict) -> dict:
    metas = r["metas"]
    first = next((m for m in metas if int(m["completion_tokens"]) > 0), {})
    return {
        "rid": r["rid"],
        "events": len(metas),
        "prompt_tokens": first.get("prompt_tokens"),
        "cached_tokens": first.get("cached_tokens"),
        "completion_tokens": metas[-1]["completion_tokens"] if metas else None,
        "server_ttft_s": (
            first["prefill_finished_time"] - first["request_received_ts"]
            if first.get("prefill_finished_time") and first.get("request_received_ts")
            else None
        ),
        "queue_time_s": first.get("queue_time"),
        "errors": r["errors"],
    }


def pick_requests(data_root: str, max_prompt: int, tok_dir: str, n: int) -> list[dict]:
    """Deterministic spread of prompt lengths up to max_prompt tokens."""
    from s1_common import Renderer, load_index, materialize_bodies

    rows, _chains, _by_chain = load_index(data_root)
    ok = sorted(
        (r for r in rows.values() if r["glm_tokens"] <= max_prompt),
        key=lambda r: (r["glm_tokens"], r["_req_id"]),
    )
    step = max(1, len(ok) // n)
    chosen = ok[::step][:n]
    bodies = materialize_bodies(data_root, [r["_req_id"] for r in chosen])
    renderer = Renderer(tok_dir)
    return [
        {
            "rid": r["_req_id"],
            "glm_tokens": r["glm_tokens"],
            "max_new_tokens": int(r.get("max_output_i") or 128),
            "text": renderer.render(bodies[r["_req_id"]]),
        }
        for r in chosen
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--tok-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-prompt", type=int, default=120000)
    ap.add_argument("--requests", type=int, default=6)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-new-tokens-cap", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=1800)
    args = ap.parse_args()
    base = args.base_url.rstrip("/").removesuffix("/v1")
    checks: dict[str, dict] = {}

    try:
        with urllib.request.urlopen(base + "/v1/models", timeout=30) as resp:
            models = json.loads(resp.read())
        checks["models"] = {"ok": True, "ids": [m["id"] for m in models["data"]]}
    except Exception as e:  # noqa: BLE001 - recorded in the receipt
        checks["models"] = {"ok": False, "error": repr(e)}

    reqs = pick_requests(args.data_root, args.max_prompt, args.tok_dir, args.requests)
    for q in reqs:
        q["max_new_tokens"] = min(q["max_new_tokens"], args.max_new_tokens_cap)

    status, body = post_json(base + "/flush_cache", None, 120)
    checks["flush_before"] = {"ok": status == 200 and body.get("success") is True}

    results = []
    for q in reqs:
        r = generate(base, q["text"], q["max_new_tokens"], q["rid"], args.timeout)
        bad = check_stream(r, q["max_new_tokens"], q["glm_tokens"])
        results.append({**summarize(r), "glm_tokens": q["glm_tokens"], "bad": bad})
    checks["generate"] = {"ok": all(not x["bad"] for x in results), "requests": results}

    q = max(reqs, key=lambda x: x["glm_tokens"])
    again = generate(base, q["text"], q["max_new_tokens"], q["rid"] + "#again", args.timeout)
    cold = next(x for x in results if x["rid"] == q["rid"])
    warm = summarize(again)
    checks["cache_hit"] = {
        "ok": not check_stream(again, q["max_new_tokens"], q["glm_tokens"])
        and (warm["cached_tokens"] or 0) > (cold["cached_tokens"] or 0),
        "first": cold["cached_tokens"],
        "second": warm["cached_tokens"],
        "prompt_tokens": q["glm_tokens"],
    }

    status, body = post_json(base + "/flush_cache", None, 120)
    after = summarize(generate(base, q["text"], q["max_new_tokens"], q["rid"] + "#flushed", args.timeout))
    checks["flush"] = {
        "ok": status == 200 and body.get("success") is True and after["cached_tokens"] == 0,
        "status": status,
        "body": body,
        "cached_after_flush": after["cached_tokens"],
    }

    batch = [reqs[i % len(reqs)] for i in range(args.concurrency)]
    with ThreadPoolExecutor(args.concurrency) as pool:
        outs = list(
            pool.map(
                lambda iq: generate(
                    base, iq[1]["text"], iq[1]["max_new_tokens"], f"{iq[1]['rid']}#c{iq[0]}", args.timeout
                ),
                enumerate(batch),
            )
        )
    conc = [
        {**summarize(r), "bad": check_stream(r, q2["max_new_tokens"], q2["glm_tokens"])}
        for r, q2 in zip(outs, batch)
    ]
    checks["concurrent"] = {"ok": all(not x["bad"] for x in conc), "requests": conc}

    model_id = (checks["models"].get("ids") or ["glm-5-3-flash"])[0]
    status, body = post_json(
        base + "/v1/chat/completions",
        {"model": model_id, "messages": [{"role": "user", "content": "1+1=?"}], "max_tokens": 64},
        600,
    )
    msg = (body.get("choices") or [{}])[0].get("message", {}) if status == 200 else {}
    checks["chat"] = {"ok": status == 200 and "content" in msg, "status": status, "message": msg}

    verdict = "PASS" if all(c["ok"] for c in checks.values()) else "FAIL"
    receipt = {"base_url": base, "time": time.time(), "verdict": verdict, "checks": checks}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(receipt, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: v["ok"] for k, v in checks.items()}), verdict)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
