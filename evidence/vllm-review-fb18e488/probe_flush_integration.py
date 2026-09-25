"""CPU-only frozen endpoint -> actual level_verdict flush integration."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

ROOT = Path(sys.argv[1]).resolve()
TREE = ROOT / "engine/vllm"
for name in ("vllm.entrypoints", "vllm.entrypoints.generate_compat"):
    package = importlib.import_module(name)
    package.__path__.insert(0, str(TREE.joinpath(*name.split("."))))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


test_file = TREE / "tests/entrypoints/generate_compat/test_generate_compat.py"
fixture = load("frozen_endpoint_fixture", test_file)
level = load("review_level_verdict", ROOT / "scripts/pod/verify/level_verdict.py")
router = importlib.import_module("vllm.entrypoints.generate_compat.api_router")
events = []
with patch.object(router.logger, "info", side_effect=lambda msg, *args: events.append(msg % args)):
    start = time.time()
    response = fixture.make_client(fixture.FakeEngine(fixture.MTP_STEPS)).post("/flush_cache")
    finish = time.time()
assert response.status_code == 200
payload = response.json()
with tempfile.TemporaryDirectory(dir=ROOT) as temp:
    out = Path(temp)
    raw, run = out / "raw.jsonl", out / "run.json"
    (out / "run_dev.log").write_text("flushed KV via checked runner\n")
    (out / "server.log").write_text("\n".join(events) + "\n")
    receipt = {
        "flush_success": True, "runner_rc": 0, "n": 30,
        "raw": raw.name, "run": run.name,
        "runner_started_s": start, "flush_started_s": start,
        "flush_finished_s": finish, "flush_response": payload,
    }
    (out / "flush_evidence.json").write_text(json.dumps(receipt))
    verdict = level.check_flush(
        out, raw, run, {}, [{"client_dispatch_at_s": finish + 1}], 30
    )
assert verdict["verified"] and verdict["server_receipt"] == payload

import pytest

rc = pytest.main([str(test_file), "-q", "--confcutdir", str(test_file.parent),
                  "-p", "no:cacheprovider"])
result = {
    "endpoint_revision": "fb18e4887d5fef87811b36aeae2d7cbd31e0b7bb",
    "fake_engine_real_asgi_and_verdict": True,
    "gpu_used": False, "unit_suite_exit_code": rc,
    "flush_mode": verdict["mode"], "response": payload,
    "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [*sorted((TREE / "vllm/entrypoints/generate_compat").glob("*.py")),
                                test_file, ROOT / "scripts/pod/verify/level_verdict.py"]},
}
body = json.dumps(result, indent=2) + "\n"
assert len(body) < 64 * 1024
(ROOT / "flush_integration_summary.json").write_text(body)
print(json.dumps({"verified": True, "unit_suite_exit_code": rc}))
raise SystemExit(rc)
