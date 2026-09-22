#!/usr/bin/env python3
"""Container checklist helper for preflight_8gpu.sh; never launches an engine.

Dry run: scripts/preflight_8gpu.sh --dry-run
Live, only on an authorized running engine:
  scripts/preflight_8gpu.sh --base-url http://127.0.0.1:8000 --out runs/preflight.json
The engine-root URL may end in /v1; chat defaults to /v1/chat/completions with
model=default. Override --chat-path /chat/completions for a root-mounted API.
API key is read only from S1_API_KEY, never from argv or written to logs.
Only safe check summaries are printed/saved; response bodies, exception text,
subprocess output, request prompts and environment are never emitted.
Requirements are installed only if their version/import check fails. Pytest
uses no cache provider; unittest is the fallback. Neither writes in s1-dev/.
This exercises the task.md endpoint checklist, not image pull/revision provenance
or proof of all-tier cache clearing under every workload. The repeat/flush probe
verifies an actual cold result, not merely a successful HTTP status (F23).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import uuid

REPO = Path(__file__).resolve().parents[1]
CHECKS = [
    "Import/version-check harness requirements; pip install -r s1-dev/harness/requirements.txt only if needed",
    "python3 -m pytest -q -p no:cacheprovider s1-dev/harness/test_s1_harness.py (unittest fallback if pytest absent)",
    "GET /v1/models returns 200, nonempty model list containing default",
    "POST /flush_cache?timeout=120 returns 2xx JSON with success exactly true",
    "POST /generate SSE: every event has completion_tokens/prompt_tokens/cached_tokens; first prefill_finished_time and final request_received_ts",
    "ignore_eos=true produces exactly the requested output length; initial cached_tokens=0",
    "Repeat the identical prompt: cached_tokens increases",
    "Verified JSON flush, identical prompt again: cached_tokens=0",
    "POST /v1/chat/completions model=default: nonempty choices[0].message.content and matching response model",
    "Verified final JSON flush; save safe JSON check report (no secrets or response text)",
]


class CheckFailed(Exception):
    """Messages contain only fixed diagnostics, never remote content."""


def require(condition, message):
    if not condition:
        raise CheckFailed(message)


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def timestamp(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


class Client:
    def __init__(self, base_url, timeout=180, api_key=None):
        parsed = urllib.parse.urlsplit(base_url)
        require(parsed.scheme in ("http", "https") and bool(parsed.hostname), "invalid engine URL")
        require(not (parsed.username or parsed.password or parsed.query or parsed.fragment),
                "engine URL must not contain credentials, query or fragment")
        path = parsed.path.rstrip("/")
        if path.endswith("/v1"):
            path = path[:-3]
        self.base = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
        self.timeout, self.api_key = timeout, api_key

    def open(self, path, payload=None):
        # Only fixed local endpoint paths; do not allow a remote URL override.
        require(path.startswith("/") and not path.startswith("//"), "invalid endpoint path")
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        req = urllib.request.Request(self.base+path,
            data=json.dumps(payload).encode() if payload is not None else None, headers=headers,
            method="POST" if payload is not None else "GET")
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except Exception:
            raise CheckFailed("HTTP connection/status failure (remote details suppressed)") from None

    def json(self, path, payload=None, expected_status=None):
        try:
            with self.open(path, payload) as response:
                require(200 <= response.status < 300, "HTTP status not successful")
                if expected_status is not None:
                    require(response.status == expected_status, "unexpected HTTP success status")
                return json.load(response)
        except CheckFailed:
            raise
        except Exception:
            raise CheckFailed("invalid or unreadable JSON response") from None


def verified_flush(client, server_wait=120):
    reply = client.json(f"/flush_cache?timeout={server_wait}", {})
    require(isinstance(reply, dict) and reply.get("success") is True,
            "flush must return JSON object with boolean success=true")
    return {"success": True, "server_wait_s": server_wait}


def sse_payloads(response):
    """Parse SSE event blocks, including multiline data and heartbeat comments."""
    data = []
    for raw in response:
        try:
            line = raw.decode("utf-8").rstrip("\r\n")
        except UnicodeError:
            raise CheckFailed("non-UTF8 SSE response") from None
        if not line:
            if data:
                payload = "\n".join(data)
                if payload == "[DONE]":
                    return
                yield payload
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if data and "\n".join(data) != "[DONE]":
        yield "\n".join(data)


def generate(client, prompt, tokens):
    body = {"text": prompt, "stream": True,
            "sampling_params": {"max_new_tokens": tokens, "temperature": 0, "ignore_eos": True}}
    count, first, final, previous = 0, None, None, -1
    try:
        with client.open("/generate", body) as response:
            require("text/event-stream" in response.headers.get("Content-Type", ""),
                    "generate did not return text/event-stream")
            for payload in sse_payloads(response):
                event = json.loads(payload)
                require(isinstance(event, dict), "SSE event must be an object")
                meta = event.get("meta_info")
                require(isinstance(meta, dict), "SSE missing meta_info")
                require(all(integer(meta.get(k)) for k in
                            ("completion_tokens", "prompt_tokens", "cached_tokens")),
                        "SSE token counters must be nonnegative integers on every event")
                require(0 <= meta["cached_tokens"] <= meta["prompt_tokens"] and meta["prompt_tokens"] > 0,
                        "invalid prompt/cache counters")
                require(previous <= meta["completion_tokens"] <= tokens, "invalid completion counter sequence")
                if first is None:
                    first = dict(meta)
                require(meta["prompt_tokens"] == first["prompt_tokens"] and
                        meta["cached_tokens"] == first["cached_tokens"], "prompt/cache counters changed mid-stream")
                previous = meta["completion_tokens"]
                final, count = dict(meta), count+1
    except CheckFailed:
        raise
    except Exception:
        raise CheckFailed("invalid/interrupted generate SSE (details suppressed)") from None
    require(count > 0, "no SSE data events")
    require(final["completion_tokens"] == tokens, "ignore_eos exact output length failed")
    require(timestamp(first.get("prefill_finished_time")) and timestamp(final.get("request_received_ts")),
            "required first/final server timestamps missing or invalid")
    require(first["prefill_finished_time"] >= final["request_received_ts"], "negative server TTFT")
    return {"events": count, "completion_tokens": final["completion_tokens"],
            "prompt_tokens": final["prompt_tokens"], "cached_tokens": final["cached_tokens"],
            "server_ttft_s": first["prefill_finished_time"]-final["request_received_ts"]}


def run_http_checks(client, report, tokens=16, chat_tokens=1024,
                    chat_path="/v1/chat/completions", server_wait=120):
    def record(name, details):
        report.append({"check": name, "passed": True, **details})
    models = client.json("/v1/models", expected_status=200)
    require(isinstance(models, dict) and isinstance(models.get("data"), list), "models response lacks data array")
    require(any(isinstance(m, dict) and m.get("id") == "default" for m in models["data"]),
            "models endpoint does not advertise model=default")
    record("models", {"default_advertised": True})
    record("initial_json_flush", verified_flush(client, server_wait))
    prompt = "Preflight cache probe " + uuid.uuid4().hex + "\n" + (
        "Read this fixed context and reply with a short greeting. The context tests prefix reuse.\n"*256)
    cold = generate(client, prompt, tokens)
    require(cold["cached_tokens"] == 0, "initial flush did not produce cold cached_tokens=0")
    record("cold_generate_sse_exact_length", cold)
    repeat = generate(client, prompt, tokens)
    require(repeat["prompt_tokens"] == cold["prompt_tokens"], "repeat prompt token count changed")
    require(repeat["cached_tokens"] > cold["cached_tokens"], "cached_tokens did not rise on repeat")
    record("repeat_cache_hit", repeat)
    record("repeat_json_flush", verified_flush(client, server_wait))
    after = generate(client, prompt, tokens)
    require(after["prompt_tokens"] == cold["prompt_tokens"] and after["cached_tokens"] == 0,
            "verified JSON flush did not restore cold cached_tokens=0")
    record("after_flush_cold", after)
    chat = client.json(chat_path, {"model": "default", "stream": False,
        "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
        "max_tokens": chat_tokens, "temperature": 0})
    require(isinstance(chat, dict) and isinstance(chat.get("choices"), list) and bool(chat["choices"]),
            "chat response lacks choices")
    choice = chat["choices"][0]
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    content = message.get("content") if isinstance(message, dict) else None
    require(isinstance(content, str) and bool(content.strip()), "chat content is empty (increase chat budget if reasoning exhausts it)")
    require(chat.get("model") == "default", "chat response model does not match default")
    record("chat_roundtrip", {"model_matches": True, "nonempty_content": True})
    record("final_json_flush", verified_flush(client, server_wait))


def harness_checks(dev_root, report):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    check = [sys.executable, "-B", "-c", "import transformers; from importlib.metadata import version; "
             "v=tuple(map(int,version('transformers').split('.')[:2])); assert (4,51)<=v<(6,0)"]
    def run(command, timeout=300):
        try:
            return subprocess.run(command, env=env, cwd=REPO, capture_output=True, timeout=timeout).returncode
        except Exception:
            raise CheckFailed("harness subprocess failed or timed out (output suppressed)") from None
    installed = False
    if run(check) != 0:
        require(run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                     "-r", str(dev_root/"harness/requirements.txt")], 900) == 0,
                "harness requirements installation failed (output suppressed)")
        installed = True
        require(run(check) == 0, "harness requirements import/version check still fails")
    report.append({"check": "harness_requirements", "passed": True, "pip_installed": installed})
    if importlib.util.find_spec("pytest") is not None:
        command = [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                   str(dev_root/"harness/test_s1_harness.py")]
        runner = "pytest"
    else:
        command = [sys.executable, "-B", "-m", "unittest", "discover", "-s",
                   str(dev_root/"harness"), "-p", "test_s1_harness.py", "-v"]
        runner = "unittest"
    require(run(command) == 0, "harness self-tests failed (output suppressed)")
    report.append({"check": "harness_self_tests", "passed": True, "runner": runner})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--extended-if", action="store_true", help="also run T19 IF checks (including 4096-token probes and 20 full dev prompts)")
    p.add_argument("--same-host", action="store_true", help="confirm direct client shares server wall clock, required for extended IF-12")
    p.add_argument("--dev-root", type=Path, default=REPO/"s1-dev")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--chat-path", choices=("/v1/chat/completions", "/chat/completions"), default="/v1/chat/completions")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--chat-max-tokens", type=int, default=1024)
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--flush-server-wait", type=int, default=120)
    p.add_argument("--out", type=Path, default=REPO/"runs"/f"preflight_{int(time.time())}.json")
    a = p.parse_args()
    if a.dry_run:
        print("DRY RUN: no installs, tests, HTTP requests, services or file writes.")
        for i, check in enumerate(CHECKS, 1):
            print(f"{i}. {check}")
        if a.extended_if:
            from if_checks import DEFAULT_CASES, DESCRIPTIONS
            for case in DEFAULT_CASES:
                print(f"{case}: {DESCRIPTIONS[case]}")
        return 0
    if min(a.max_new_tokens, a.chat_max_tokens, a.timeout) <= 0 or not math.isfinite(a.timeout):
        p.error("token counts and timeout must be positive and finite")
    if a.flush_server_wait < 0 or a.timeout <= a.flush_server_wait:
        p.error("HTTP timeout must exceed nonnegative flush server wait")
    if a.dev_root.resolve() == a.out.resolve() or a.dev_root.resolve() in a.out.resolve().parents:
        p.error("output must be outside read-only s1-dev")
    report = {"purpose": "container endpoint checklist; not a benchmark", "checks": [], "passed": False}
    try:
        client = Client(a.base_url, a.timeout, os.environ.get("S1_API_KEY"))
        harness_checks(a.dev_root.resolve(), report["checks"])
        run_http_checks(client, report["checks"], a.max_new_tokens, a.chat_max_tokens,
                        a.chat_path, a.flush_server_wait)
        if a.extended_if:
            from if_checks import DEFAULT_CASES, run_checks
            from serving_probe import client_for
            extended = run_checks(client_for(a.base_url, a.timeout), DEFAULT_CASES,
                                  dev_root=a.dev_root, same_host=a.same_host, wait=a.flush_server_wait)
            report["extended_if"] = extended
            require(extended["passed"], "extended IF checks failed or prerequisites missing")
        report["passed"] = True
    except CheckFailed as e:
        report["checks"].append({"check": "aborted", "passed": False, "reason": str(e)})
    except Exception:
        report["checks"].append({"check": "aborted", "passed": False, "reason": "unexpected local failure; details suppressed"})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
