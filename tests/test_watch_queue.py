#!/usr/bin/env python3
"""CPU-only checks for the persistent read-only queue watcher."""
import importlib.util
import json
from unittest.mock import patch
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("watch_queue", ROOT / "scripts/pod/watch_queue.py")
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


def status(**parts):
    return "\n".join(f"{key}: {' '.join(parts.get(key, []))}" for key in w.STATES) + "\n"


class QueueWatcher(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "state.json"
        self.state = w.empty_state()
        self.sent = []

    def step(self, output, sender=None):
        return w.poll_once(self.state, self.path, lambda: output,
                           sender or self.sent.append)

    def test_initial_history_is_baseline_without_spam(self):
        self.assertTrue(self.step(status(done=["047.sh"], failed=["049.sh"], pending=["050.sh"])))
        self.assertEqual(self.sent, [])
        self.assertEqual(w.load_state(self.path)["jobs"]["049.sh"], "failed")

    def test_pending_to_failed_and_unseen_fast_failure(self):
        self.step(status(pending=["050.sh"]))
        self.step(status(failed=["050.sh", "051.sh"]))
        self.assertEqual(len(self.sent), 2)
        self.assertIn("050.sh: pending → failed", self.sent[0])
        self.assertIn("051.sh: new → failed", self.sent[1])

    def test_new_completion_and_running_transition(self):
        self.step(status(pending=["050.sh"]))
        self.step(status(running=["050.sh"], done=["051.sh"]))
        self.assertEqual(len(self.sent), 2)
        self.assertTrue(any("pending → running" in x for x in self.sent))
        self.assertTrue(any("new → done" in x for x in self.sent))

    def test_failed_send_stays_in_durable_outbox_and_retries(self):
        self.step(status(pending=["050.sh"]))
        def reject(_):
            raise RuntimeError("relay unavailable")
        self.assertFalse(self.step(status(failed=["050.sh"]), reject))
        self.assertEqual(len(w.load_state(self.path)["outbox"]), 1)
        # Simulate restart: load the durable state, then retry with unchanged remote status.
        self.state = w.load_state(self.path)
        self.assertTrue(self.step(status(failed=["050.sh"])))
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(w.load_state(self.path)["outbox"], [])

    def test_remote_outage_one_alert_and_recovery(self):
        self.step(status(running=["048.sh"]))
        def offline():
            raise RuntimeError("ssh timeout")
        self.assertFalse(w.poll_once(self.state, self.path, offline, self.sent.append))
        self.assertFalse(w.poll_once(self.state, self.path, offline, self.sent.append))
        self.assertEqual(len([x for x in self.sent if "不可用" in x]), 1)
        self.step(status(failed=["048.sh"]))
        self.assertEqual(len([x for x in self.sent if "已恢复" in x]), 1)
        self.assertTrue(any("running → failed" in x for x in self.sent))

    def test_partial_or_duplicate_status_is_remote_failure(self):
        self.assertRaises(ValueError, w.parse_status, "running: 048.sh\npending:\n")
        self.assertRaises(ValueError, w.parse_status,
                          status(running=["048.sh"], failed=["048.sh"]))

    def test_unexpected_status_output_fails_closed(self):
        self.assertRaises(ValueError, w.parse_status, status() + "SSH ERROR\n")
        self.assertRaises(ValueError, w.parse_status, status(pending=["bad/name.sh"]))

    def test_pread_success_receipt_only_at_end(self):
        raw = status(running=["048-offA_122_n22.sh"], failed=["049-171_num.sh"]) + "exit_code: 0\n"
        self.assertEqual(w.parse_status(raw), {"048-offA_122_n22.sh": "running", "049-171_num.sh": "failed"})
        self.assertRaises(ValueError, w.parse_status, raw + "exit_code: 0\n")
        self.assertRaises(ValueError, w.parse_status, "exit_code: 0\n" + status())
        self.assertRaises(ValueError, w.parse_status, status() + "exit_code: 1\n")

    def test_state_file_is_valid_after_each_poll(self):
        self.step(status(pending=["048.sh"]))
        self.step(status(done=["048.sh"]))
        on_disk = json.loads(self.path.read_text())
        self.assertEqual(on_disk["jobs"], {"048.sh": "done"})
        self.assertEqual(on_disk["outbox"], [])

    def test_gpu_mode_restricts_all_paths(self):
        good = Path("/sjtu/linhang/arena/repo")
        w.validate_gpu_location(good, good, good / "build/scratch/coordination/queue-watch-gpu")
        with self.assertRaises(ValueError):
            w.validate_gpu_location(ROOT, good, good / "state")
        with self.assertRaises(ValueError):
            w.validate_gpu_location(good, Path("/tmp"), good / "state")
        with self.assertRaises(ValueError):
            w.validate_gpu_location(good, good, Path("/tmp/state"))

    def test_gpu_fetch_uses_local_bexec_and_no_pread(self):
        with patch.object(w, "run_bounded", return_value=status()) as call:
            self.assertEqual(w.gpu_status(45), status())
        argv, timeout = call.call_args.args
        self.assertEqual(argv[:2], ["bash", "-c"])
        self.assertIn("source scripts/pod/common.sh", argv[2])
        self.assertIn("bexec", argv[2])
        self.assertNotIn("pread", argv[2])
        self.assertEqual(timeout, 45)
        self.assertEqual(call.call_args.kwargs["cwd"], ROOT)

    def test_gpu_event_append_is_durable_and_deduplicated(self):
        path = Path(self.tmp.name) / "events.jsonl"
        item = {"id": 7, "text": "[queue watcher] 051.sh: running → failed", "at": "2026-09-24T00:00:00+00:00"}
        seen = w.event_ids(path)
        w.append_event(path, item, seen)
        w.append_event(path, item, seen)
        self.assertEqual(len(path.read_text().splitlines()), 1)
        w.append_event(path, item, w.event_ids(path))  # retry after a process restart
        self.assertEqual(len(path.read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
