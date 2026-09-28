"""Evidence checks only; these tests do not simulate GPU state correctness."""

import asyncio
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock


SOURCE = Path(__file__).resolve().parents[1] / "scripts/pod/verify/multiround_park_probe.py"
SPEC = importlib.util.spec_from_file_location("park_probe", SOURCE)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class EvidenceTest(unittest.TestCase):
    def test_abort_location_uses_server_event_order(self):
        events = [{"event": "yield", "a": "A", "b": "B"}]
        terminal = {"event": "terminal", "rid": "A", "reason": "abort"}
        self.assertEqual(probe.abort_location(events + [terminal], "A"), "parked")
        events.append({"event": "resume", "rid": "A", "park": None})
        self.assertEqual(probe.abort_location(events + [terminal], "A"), "active")

    def test_fragmented_live_log_and_old_events(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "server.log"
            log.write_text('[ax-124m] {"event":"yield","a":"old"}\n')
            reader = probe.Probe(SimpleNamespace(server_log=log, out=root), None)
            try:
                with log.open("a") as f:
                    f.write('[ax-124m] {"event":"yield",')
                reader.poll()
                self.assertEqual(reader.events, [])
                with log.open("a") as f:
                    f.write('"a":"new"}\n')
                reader.poll()
                self.assertEqual(reader.events, [{"event": "yield", "a": "new"}])
            finally:
                reader.log.close()
                reader.event_out.close()
                reader.response_out.close()

    def test_numerics_never_compare_after_token_divergence(self):
        arms = []
        for mode in ("off", "on"):
            records = []
            for repeat in range(2 if mode == "off" else 1):
                for name, _, _ in probe.CASES:
                    for role in ("A", "B"):
                        records.append(dict(case=f"{name}.r{repeat}", role=role,
                            input_sha256=name + role, meta={"output_token_logprobs":
                                [[-1.0, 42] for _ in range(probe.TOKENS)]}))
            arms.append(dict(mode=mode, status="STATE_PASS", responses=records))
        self.assertEqual(probe.compare(*arms)["status"], "WITHIN_OBSERVED_OFF_NOISE")
        changed = copy.deepcopy(arms)
        changed[1]["responses"][0]["meta"]["output_token_logprobs"][2] = [-1.0, 43]
        result = probe.compare(*changed)
        self.assertEqual(result["status"], "NUMERICAL_REVIEW_REQUIRED")
        self.assertEqual(result["cases"][0]["common_conditioning_steps"], 2)
        changed[1]["responses"].pop(0)
        with self.assertRaises(KeyError):
            probe.compare(*changed)  # No missing case can disappear into PASS.

    def test_cancelled_http_task_still_aborts_its_server_request(self):
        async def scenario():
            client = object.__new__(probe.Probe)
            client.prefix, client.records = "own-rids-", []
            task = asyncio.create_task(asyncio.sleep(60))
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            client.tasks = {"own-rids-A": task}
            client.post = AsyncMock(return_value=(200, {}))
            await client.cleanup()
            client.post.assert_awaited_once_with(
                "/abort_request", {"rid": "own-rids-"}, timeout=3)
        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
