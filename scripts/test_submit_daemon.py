#!/usr/bin/env python3
"""T22 offline unit tests: mocked Playground CLI + API, no APPROVED files.

Approval file reads/existence are mocked in memory; no test grants real approval.
Run: PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts -p test_submit_daemon.py -v
"""
import contextlib
import datetime as dt
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import submit_daemon as sd

WHEN = dt.datetime(2026, 9, 22, 4, 0, tzinfo=dt.timezone.utc)
CANDIDATE = {"image": "registry.dp.tech/test/arena@sha256:" + "0" * 64,
             "command": "vllm serve --model /mnt/models --served-model-name default",
             "env": {}, "model_name": "default"}
TRACE = b'{"type":"session_start","source":"stub"}\n{"role":"user","message":{"content":[{"type":"text","text":"placeholder"}]}}\n'


def row(aid, state="submitted", created=WHEN, author="test-user"):
    return {"id": str(aid), "authorId": author, "createdAt": sd.timestamp(created), "status": state}


def result_card():
    return {"scorewheel_raw_result": {"aime26": {"points": 97.7}, "gpqa-diamond": {"points": 96.8},
        "gate_passed": True, "stress": {"n_at_slo": 14, "tpot_mean": .021, "tpot_p95": .041,
        "chain_start_p95": 18.0, "turn_start_p95": 4.1, "overall_intra_p95": 4.5,
        "fast_intra_p95": 2.1, "evaluation_status": "PASS", "reason": None}}}


class FakePlayground(sd.Backend):
    """Keep production CLI argument construction and JSON parsers in the path."""
    def __init__(self, root, clock):
        super().__init__(root)
        self.clock, self.rows, self.calls = clock, [], []
        self.fail_check = self.fail_dry = self.fail_submit = self.fail_api = self.fail_status = False
        self.after_dry = self.before_submit = None
        self.next_result = None
        self.status_override = None
        self.next_id = 1000

    def get_json(self, route):
        self.calls.append(("api", route))
        if self.fail_api:
            raise sd.SafeError("api-unavailable")
        if route == "/auth/me":
            return {"id": "test-user"}
        page = int(route.rsplit("=", 1)[1])
        return {"attempts": self.rows[(page - 1) * 100:page * 100]}

    def command(self, argv, cwd=None):
        self.calls.append(("validator", argv))
        assert "--final" in argv and "--trace" in argv
        if self.fail_check:
            raise sd.SafeError("mock-validator-failed")
        return b""

    def cli(self, args, cwd=None):
        self.calls.append(("cli", args.copy()))
        if args[0] == "status":
            if self.fail_status:
                raise sd.SafeError("mock-status-failed")
            value = self.status_override or next(a for a in self.rows if str(a["id"]) == args[-1])
            return json.dumps(value).encode()
        assert args[:3] == ["submit", "--challenge-id", sd.CHALLENGE]
        outputs = Path(args[args.index("--outputs") + 1])
        trace = Path(args[args.index("--trace") + 1])
        assert [p.name for p in outputs.iterdir()] == ["submission.json"]
        assert trace.read_bytes() == TRACE
        assert "APPROVED" not in [p.name for p in outputs.iterdir()]
        if "--dry-run" in args:
            if self.after_dry:
                self.after_dry()
            if self.fail_dry:
                raise sd.SafeError("mock-dry-run-failed")
            return b'{"status":"dry_run"}'
        if self.before_submit:
            self.before_submit()
        if self.fail_submit:
            raise sd.SafeError("mock-submit-timeout")
        self.next_id += 1
        self.rows.append(row(self.next_id, created=self.clock()))
        return self.next_result if self.next_result is not None else json.dumps({"attempt_id": str(self.next_id)}).encode()

    def submits(self):
        return [args for kind, args in self.calls if kind == "cli" and args[0] == "submit" and "--dry-run" not in args]


class DaemonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="test-submit-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.now = WHEN
        self.backend = FakePlayground(self.root, lambda: self.now)
        self.daemon = sd.Daemon(self.root, self.backend, clock=lambda: self.now)
        self.approvals = {}
        original_is_file = Path.is_file
        def is_file(path):
            if path.name == "APPROVED":
                return path.parent.name in self.approvals
            return original_is_file(path)
        def read_approval(item, submission, trace):
            if item.name not in self.approvals:
                raise sd.SafeError("approval-required")
            return sd.parse_approval(self.approvals[item.name], item.name, submission, trace)
        self.patches = [patch.object(Path, "is_file", is_file),
                        patch.object(sd, "read_approval", read_approval),
                        patch.object(sd, "credential_env"),
                        patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("network forbidden"))]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def item(self, name="01-test", approve=True):
        item = self.root / "submission/queue" / name
        item.mkdir(parents=True)
        (item / "submission.json").write_text(json.dumps(CANDIDATE))
        (item / "stub-trace.jsonl").write_bytes(TRACE)
        (item / "notes.md").write_text("Synthetic test item\n")
        if approve:
            self.approvals[name] = "\n".join([
                "approved_by: MOCK USER", "approved_at: " + sd.timestamp(WHEN), "what: " + name,
                "submission_sha256: " + sd.sha((item / "submission.json").read_bytes()),
                "trace_sha256: " + sd.sha(TRACE)])
        return item

    def restart(self):
        self.daemon = sd.Daemon(self.root, self.backend, clock=lambda: self.now)
        self.daemon.tick()

    def finish(self, aid=None):
        target = self.backend.rows[-1] if aid is None else next(a for a in self.backend.rows if a["id"] == aid)
        target.update(status="scored", execStatus="completed", scoringState={"state": "final", "scoreIsFinal": True},
                      scorecard=result_card())

    def test_unapproved_queue_never_calls_cli_or_api(self):
        self.item(approve=False)
        self.daemon.tick()
        self.assertEqual(self.backend.calls, [])
        self.assertEqual(self.daemon.records[0]["status"], "awaiting-approval")

    def test_empty_queue_never_reads_credentials(self):
        self.daemon.tick()
        sd.credential_env.assert_not_called()
        self.assertFalse(self.daemon.ledger_path.exists())

    def test_stop_pauses_everything_including_poll_and_ledger(self):
        self.item()
        self.daemon.tick()
        before = self.daemon.ledger_path.read_bytes()
        self.backend.calls.clear()
        (self.daemon.queue / "STOP").touch()
        self.daemon.tick()
        self.assertEqual(self.backend.calls, [])
        self.assertEqual(self.daemon.ledger_path.read_bytes(), before)

    def test_broken_stop_symlink_also_pauses(self):
        self.item()
        (self.daemon.queue / "STOP").symlink_to("nonexistent")
        self.daemon.tick()
        self.assertEqual(self.backend.calls, [])

    def test_submit_reserves_first_and_output_dir_is_minimal(self):
        self.item()
        def check_reservation():
            data = json.loads(self.daemon.ledger_path.read_text())
            self.assertEqual(data[0]["status"], "submitting")
            self.assertEqual(data[0]["submitted_at"], sd.timestamp(WHEN))
        self.backend.before_submit = check_reservation
        self.daemon.tick()
        record = self.daemon.records[0]
        self.assertEqual(record["attempt_id"], "1001")
        self.assertEqual(record["approved_by"], "MOCK USER")
        self.assertEqual(record["status"], "submitted")
        self.assertEqual(len(self.backend.submits()), 1)
        kinds = [a for a, _ in self.backend.calls]
        self.assertLess(kinds.index("validator"), kinds.index("cli"))
        self.assertIn("SUBMITTED 01-test attempt=1001", self.daemon.events.read_text())
        self.assertFalse(any(p.name == "APPROVED" for p in self.root.rglob("*")))

    def test_restart_polls_and_never_resubmits(self):
        self.item()
        self.daemon.tick()
        self.restart()
        self.assertEqual(len(self.backend.submits()), 1)
        self.assertIn(("cli", ["status", "--attempt-id", "1001"]), self.backend.calls)

    def test_terminal_scorecard_and_summary_exactly_once(self):
        self.item()
        self.daemon.tick()
        self.finish()
        self.restart()
        self.restart()
        record = self.daemon.records[0]
        self.assertEqual(record["status"], "succeeded")
        card = record["final_scorecard"]
        self.assertEqual(card["aime26"]["points"], 97.7)
        self.assertEqual(card["gpqa"]["points"], 96.8)
        self.assertTrue(card["gate_passed"])
        self.assertEqual(set(card["stress"]), set(sd.STRESS_FIELDS))
        self.assertEqual(card["stress"]["n_at_slo"], 14)
        self.assertEqual(self.daemon.summary.read_text().count("<!-- submission:01-test -->"), 1)
        self.assertEqual(self.daemon.events.read_text().count("RESULT 01-test"), 1)

    def test_summary_and_result_repaired_after_crash(self):
        self.item()
        self.daemon.tick()
        self.finish()
        with patch.object(self.daemon, "terminal_output", side_effect=RuntimeError("crash")):
            with self.assertRaises(RuntimeError):
                self.daemon.tick()
        self.assertEqual(json.loads(self.daemon.ledger_path.read_text())[0]["status"], "succeeded")
        self.restart()
        self.assertEqual(self.daemon.summary.read_text().count("submission:01-test"), 1)

    def test_default_one_in_flight(self):
        self.item()
        self.item("02-test")
        self.daemon.tick()
        self.assertEqual(len(self.backend.submits()), 1)
        self.assertEqual(self.daemon.records[1]["status_history"][-1]["detail"], "in-flight-limit")

    def test_two_daily_limit_with_configured_concurrency(self):
        self.item()
        self.item("02-test")
        self.item("03-test")
        self.daemon.max_in_flight = 3
        self.daemon.tick()
        self.assertEqual(len(self.backend.submits()), 2)
        self.assertEqual(self.daemon.records[2]["status_history"][-1]["detail"], "daily-limit")

    def test_api_manual_inflight_blocks_even_with_empty_ledger(self):
        self.item()
        self.backend.rows = [row("manual")]
        self.daemon.tick()
        self.assertEqual(self.backend.submits(), [])

    def test_old_api_inflight_still_blocks(self):
        self.item()
        self.backend.rows = [row("manual", created=WHEN-dt.timedelta(days=12))]
        self.daemon.tick()
        self.assertEqual(self.backend.submits(), [])

    def test_other_authors_do_not_consume_our_limit(self):
        self.item()
        self.backend.rows = [row("other", author="someone-else")]
        self.daemon.tick()
        self.assertEqual(len(self.backend.submits()), 1)

    def test_two_manual_api_submissions_count_toward_daily_quota(self):
        self.item()
        self.backend.rows = [row("manual1", "scored"), row("manual2", "failed")]
        self.daemon.tick()
        self.assertEqual(self.backend.submits(), [])

    def test_ledger_and_api_count_same_attempt_once(self):
        self.item()
        self.daemon.tick()
        self.finish()
        self.item("02-test")
        self.daemon.tick()
        self.assertEqual(len(self.backend.submits()), 2)

    def test_shanghai_calendar_midnight_not_rolling_24h(self):
        self.item()
        self.backend.rows = [row("a", "scored", created=WHEN), row("b", "failed", created=WHEN)]
        self.now = dt.datetime(2026, 9, 22, 15, 59, 59, tzinfo=dt.timezone.utc)
        self.daemon.tick()
        self.assertEqual(len(self.backend.submits()), 0)
        self.now += dt.timedelta(seconds=1)
        self.daemon.tick()
        self.assertEqual(len(self.backend.submits()), 1)

    def test_local_ledger_inflight_survives_api_eventual_consistency(self):
        self.item()
        self.daemon.tick()
        self.backend.rows.clear()
        self.backend.fail_status = True
        self.item("02-test")
        self.daemon.tick()
        self.assertEqual(len(self.backend.submits()), 1)

    def test_api_unavailable_blocks_submission(self):
        self.item()
        self.backend.fail_api = True
        with self.assertRaises(sd.SafeError):
            self.daemon.tick()
        self.assertEqual(self.backend.submits(), [])

    def test_precheck_error_terminal_no_retry(self):
        for source in ("fail_check", "fail_dry"):
            with self.subTest(source=source):
                name = "01-check" if source == "fail_check" else "02-dry"
                self.item(name)
                setattr(self.backend, source, True)
                self.daemon.tick()
                setattr(self.backend, source, False)
        self.restart()
        self.assertTrue(all(r["status"] == "failed-precheck" for r in self.daemon.records))
        self.assertEqual(self.backend.submits(), [])

    def test_unknown_submit_blocks_globally_across_days_and_restarts(self):
        self.item()
        self.backend.fail_submit = True
        self.daemon.tick()
        self.backend.fail_submit = False
        self.item("02-test")
        self.now += dt.timedelta(days=5)
        self.restart()
        self.assertEqual(len(self.backend.submits()), 1)
        self.assertEqual(self.daemon.records[0]["status"], "submission-unknown")
        self.assertEqual(self.daemon.records[1]["status_history"][-1]["detail"], "unresolved-submission")

    def test_crash_after_reservation_does_not_retry(self):
        self.item()
        self.backend.before_submit = lambda: (_ for _ in ()).throw(RuntimeError("simulated crash"))
        with self.assertRaises(RuntimeError):
            self.daemon.tick()
        self.assertEqual(json.loads(self.daemon.ledger_path.read_text())[0]["status"], "submitting")
        self.backend.before_submit = None
        self.restart()
        self.assertEqual(self.daemon.records[0]["status"], "submission-unknown")
        self.assertEqual(len(self.backend.submits()), 1)

    def test_unparseable_submit_response_is_unknown(self):
        self.item()
        self.backend.next_result = b"not-json"
        self.daemon.tick()
        self.assertEqual(self.daemon.records[0]["status"], "submission-unknown")
        self.restart()
        self.assertEqual(len(self.backend.submits()), 1)

    def test_approval_binds_candidate_and_trace(self):
        for name, target in (("01-candidate", "submission.json"), ("02-trace", "stub-trace.jsonl")):
            item = self.item(name)
            with (item / target).open("a") as f:
                f.write(" ")
        self.daemon.tick()
        self.assertEqual(self.backend.submits(), [])
        self.assertTrue(all(r["status"] == "awaiting-approval" for r in self.daemon.records))

    def test_approval_revoked_during_precheck(self):
        self.item()
        self.backend.after_dry = lambda: self.approvals.clear()
        self.daemon.tick()
        self.assertEqual(self.backend.submits(), [])
        self.assertEqual(self.daemon.records[0]["status"], "awaiting-approval")

    def test_changed_approval_during_precheck(self):
        self.item()
        self.backend.after_dry = lambda: self.approvals.update({"01-test": self.approvals["01-test"].replace("MOCK USER", "OTHER USER")})
        self.daemon.tick()
        self.assertEqual(self.backend.submits(), [])

    def test_stop_created_during_precheck_prevents_submit(self):
        self.item()
        self.backend.after_dry = lambda: (self.daemon.queue / "STOP").touch()
        with self.assertRaises(sd.Paused):
            self.daemon.tick()
        self.assertEqual(self.backend.submits(), [])
        self.assertIsNone(self.daemon.records[0]["submitted_at"])

    def test_api_refreshed_after_precheck(self):
        self.item()
        self.backend.after_dry = lambda: self.backend.rows.append(row("manual"))
        self.daemon.tick()
        self.assertEqual(self.backend.submits(), [])

    def test_two_daemons_cannot_enter_same_tick(self):
        self.item()
        competitor = sd.Daemon(self.root, self.backend)
        def conflict():
            with self.assertRaisesRegex(sd.SafeError, "daemon-already-running"):
                competitor.tick()
        self.backend.after_dry = conflict
        self.daemon.tick()
        self.assertEqual(len(self.backend.submits()), 1)

    def test_bad_ledger_and_duplicate_ids_fail_closed(self):
        self.item()
        self.daemon.ledger_path.parent.mkdir()
        self.daemon.ledger_path.write_text("{broken")
        with self.assertRaisesRegex(sd.SafeError, "ledger-invalid"):
            self.daemon.tick()
        self.assertEqual(self.backend.calls, [])

    def test_status_sources_must_agree_before_finishing(self):
        self.item()
        self.daemon.tick()
        self.finish()
        self.backend.status_override = row("1001")
        self.daemon.tick()
        self.assertEqual(self.daemon.records[0]["status"], "running")
        self.assertIsNone(self.daemon.records[0]["final_scorecard"])

    def test_failed_worker_still_evaluating_keeps_slot(self):
        self.item()
        self.backend.rows = [dict(row("manual", "scoring"), execStatus="failed", scoringState={"state": "evaluating"})]
        self.daemon.tick()
        self.assertEqual(self.backend.submits(), [])

    def test_pending_review_failure_is_terminal_and_null_scores_are_honest(self):
        self.item()
        self.daemon.tick()
        self.backend.rows[0].update(status="pending_review", execStatus="failed", scoringState={"state": "failed"})
        self.daemon.tick()
        self.assertEqual(self.daemon.records[0]["status"], "failed")
        self.assertIsNone(self.daemon.records[0]["final_scorecard"]["stress"]["n_at_slo"])

    def test_final_without_scorecard_waits(self):
        self.item()
        self.daemon.tick()
        self.backend.rows[0]["status"] = "scored"
        self.daemon.tick()
        self.assertNotIn(self.daemon.records[0]["status"], sd.FINAL)

    def test_full_pagination_includes_inflight_on_second_page(self):
        self.item()
        self.backend.rows = [row(i, "scored", author="other") for i in range(100)] + [row("manual")]
        self.daemon.tick()
        self.assertEqual(self.backend.submits(), [])
        self.assertTrue(any("page=2" in value for kind, value in self.backend.calls if kind == "api"))

    def test_broken_api_owner_or_timestamp_closes_gate(self):
        self.item()
        self.backend.rows = [{"id": "a", "status": "scored"}]
        self.daemon.tick()
        self.assertEqual(self.backend.submits(), [])
        self.backend.rows = [{"id": "a", "status": "scored", "authorId": "test-user"}]
        self.daemon.tick()
        self.assertEqual(self.backend.submits(), [])

    def test_queue_helper_copies_without_approval_and_numbers_uniquely(self):
        source = self.root / "candidate.json"
        source.write_text(json.dumps(CANDIDATE))
        trace = self.root / "trace.jsonl"
        trace.write_bytes(TRACE)
        first = sd.create_queue_item(self.root, source, "candidate", trace)
        second = sd.create_queue_item(self.root, source, "candidate", trace)
        self.assertEqual(first.name, "01-candidate")
        self.assertEqual(second.name, "02-candidate")
        self.assertEqual({p.name for p in first.iterdir()}, {"submission.json", "stub-trace.jsonl", "notes.md"})
        source.write_text("changed")
        self.assertEqual(json.loads((first / "submission.json").read_text()), CANDIDATE)

    def test_helper_rejects_path_traversal_and_extra_fields(self):
        source = self.root / "candidate.json"
        source.write_text(json.dumps(dict(CANDIDATE, bad=True)))
        trace = self.root / "trace.jsonl"
        trace.write_bytes(TRACE)
        with self.assertRaises(sd.SafeError):
            sd.create_queue_item(self.root, source, "../outside", trace)
        with self.assertRaises(sd.SafeError):
            sd.create_queue_item(self.root, source, "valid", trace)

    def test_trace_pointer_is_snapshotted(self):
        item = self.item()
        target = self.root / "trace.jsonl"
        target.write_bytes(TRACE)
        (item / "stub-trace.jsonl").unlink()
        (item / "stub-trace.jsonl").symlink_to(target)
        self.daemon.tick()
        self.assertEqual(len(self.backend.submits()), 1)

    def test_no_secret_in_ledger_events_or_summary(self):
        secret = "synthetic-test-secret-not-a-real-credential"
        self.item()
        self.daemon.tick()
        self.finish()
        self.backend.rows[0]["scorecard"]["scorewheel_raw_result"]["stress"]["reason"] = "Bearer " + secret
        with patch.dict(os.environ, {"PLAYGROUND_TOKEN": secret}):
            self.daemon.tick()
        for path in (self.daemon.ledger_path, self.daemon.events, self.daemon.summary):
            self.assertNotIn(secret, path.read_text())


    def test_ledger_daily_reservations_survive_api_missing_terminal_rows(self):
        self.item()
        self.daemon.tick()
        self.finish()
        self.item("02-test")
        self.daemon.tick()
        self.finish()
        self.daemon.tick()
        self.backend.rows.clear()
        self.item("03-test")
        self.daemon.tick()
        self.assertEqual(len(self.backend.submits()), 2)
        self.assertEqual(self.daemon.records[-1]["status_history"][-1]["detail"], "daily-limit")

    def test_failed_precheck_does_not_consume_quota_for_next_item(self):
        self.item()
        self.backend.fail_check = True
        self.daemon.tick()
        self.backend.fail_check = False
        self.item("02-test")
        self.daemon.tick()
        self.assertEqual(len(self.backend.submits()), 1)
        self.assertIsNone(self.daemon.records[0]["submitted_at"])

    def test_account_change_does_not_free_existing_ledger_slot(self):
        self.item()
        self.daemon.tick()
        self.item("02-test")
        with patch.object(self.backend, "owner_ids", return_value={"different-user"}):
            self.daemon.tick()
        self.assertEqual(len(self.backend.submits()), 1)
        self.assertEqual(self.daemon.records[1]["status_history"][-1]["detail"], "account-changed")

    def test_unknown_api_status_blocks_slot(self):
        self.item()
        self.backend.rows = [row("manual", "new-future-state")]
        self.daemon.tick()
        self.assertEqual(self.backend.submits(), [])

    def test_status_id_mismatch_keeps_local_attempt_active(self):
        self.item()
        self.daemon.tick()
        self.finish()
        self.backend.status_override = dict(self.backend.rows[0], id="wrong-id")
        self.daemon.tick()
        self.assertEqual(self.daemon.records[0]["status"], "submitted")
        self.assertIsNone(self.daemon.records[0]["final_scorecard"])

    def test_corrupt_ledger_cannot_hide_attempt_in_ready_state(self):
        self.item()
        self.daemon.tick()
        records = json.loads(self.daemon.ledger_path.read_text())
        records[0].update(attempt_id=None, status="ready")
        sd.atomic_json(self.daemon.ledger_path, records)
        with self.assertRaisesRegex(sd.SafeError, "ledger-invalid"):
            self.daemon.tick()
        self.assertEqual(len(self.backend.submits()), 1)


class ParsingAndBackendTests(unittest.TestCase):
    def test_strict_approval_schema_and_hashes(self):
        fields = {"approved_by": "MOCK USER", "approved_at": sd.timestamp(WHEN), "what": "01-demo",
                  "submission_sha256": sd.sha(b"a"), "trace_sha256": sd.sha(b"b")}
        text = "\n".join(k + ": " + v for k, v in fields.items())
        self.assertEqual(sd.parse_approval(text, "01-demo", b"a", b"b"), fields)
        for invalid in ("approved", text.replace("01-demo", "02-demo"), text + "\napproved_by: duplicate",
                        text.replace(sd.timestamp(WHEN), "2026-09-22")):
            with self.assertRaises(sd.SafeError):
                sd.parse_approval(invalid, "01-demo", b"a", b"b")

    def test_real_scorecard_fixtures_from_existing_data(self):
        rows = json.loads((sd.ROOT / "data/all_att.json").read_text())
        count = 0
        for a in rows:
            raw = (a.get("scorecard") or {}).get("scorewheel_raw_result") or {}
            if not isinstance(raw.get("stress"), dict):
                continue
            parsed = sd.scorecard(a)
            self.assertEqual(parsed["stress"], {k: raw["stress"].get(k) for k in sd.STRESS_FIELDS})
            self.assertEqual(parsed["aime26"]["points"], (raw.get("aime26") or {}).get("points"))
            count += 1
        self.assertGreater(count, 100)

    def test_scorecard_fallback_converts_fraction_to_points(self):
        card = sd.scorecard({"scorecard": {"scorewheel_datasets": {"aime26": .977, "gpqa-diamond": .968},
                                         "scorewheel_gate_passed": False,
                                         "scorewheel_stress": {"evaluation_status": "FAIL", "reason": "gate"}}})
        self.assertAlmostEqual(card["aime26"]["points"], 97.7)
        self.assertFalse(card["gate_passed"])

    def test_attempt_ids_and_list_envelopes(self):
        self.assertEqual(sd.attempt_id({"attempt_id": 12}), "12")
        self.assertEqual(sd.attempt_id({"attempt": {"id": "abc-12"}}), "abc-12")
        for bad in ({}, {"id": "../a"}, {"id": True}):
            with self.assertRaises(sd.SafeError):
                sd.attempt_id(bad)
        for payload in ([{"id": 1}], {"attempts": [{"id": 1}]}, {"data": {"items": [{"id": 1}]}}):
            self.assertEqual(sd.rows_from_payload(payload), [{"id": 1}])
        with self.assertRaises(sd.SafeError):
            sd.rows_from_payload({"error": "denied"})

    def test_duplicate_pagination_fails_closed(self):
        backend = sd.Backend(sd.ROOT)
        with patch.object(backend, "get_json", return_value=[row(i) for i in range(100)]):
            with self.assertRaisesRegex(sd.SafeError, "api-pagination-unstable"):
                backend.attempts()

    def test_api_short_pages_with_total_are_not_truncated(self):
        backend = sd.Backend(sd.ROOT)
        pages = [{"attempts": [row(1)], "total": 2}, {"attempts": [row(2)], "total": 2}]
        with patch.object(backend, "get_json", side_effect=pages):
            self.assertEqual(len(backend.attempts()), 2)
        with patch.object(backend, "get_json", return_value={"attempts": [], "total": 1}):
            with self.assertRaisesRegex(sd.SafeError, "api-pagination-incomplete"):
                backend.attempts()

    def test_failed_cli_does_not_expose_output(self):
        secret = "synthetic-output-that-must-stay-private"
        backend = sd.Backend(sd.ROOT)
        with self.assertRaisesRegex(sd.SafeError, "command-nonzero") as caught:
            backend.command([sys.executable, "-c", "import sys; print(sys.argv[1]); sys.exit(1)", secret])
        self.assertNotIn(secret, str(caught.exception))

    def test_command_timeout_is_safe_and_process_is_reaped(self):
        backend = sd.Backend(sd.ROOT, timeout=.03)
        with self.assertRaisesRegex(sd.SafeError, "command-failed-or-timeout"):
            backend.command([sys.executable, "-c", "import time; time.sleep(30)"])

    def test_read_approval_rejects_missing_and_symlink_without_writing_any(self):
        with tempfile.TemporaryDirectory() as tmp:
            item = Path(tmp) / "01-test"
            with self.assertRaisesRegex(sd.SafeError, "approval-required"):
                sd.read_approval(item, b"", b"")
            with patch.object(Path, "is_file", return_value=True), patch.object(Path, "is_symlink", return_value=True):
                with self.assertRaisesRegex(sd.SafeError, "approval-required"):
                    sd.read_approval(item, b"", b"")

    def test_limits_cannot_raise_daily_ceiling(self):
        for daily in (0, 3):
            with self.assertRaises(sd.SafeError):
                sd.Daemon(sd.ROOT, None, daily_limit=daily)
        with self.assertRaises(sd.SafeError):
            sd.Daemon(sd.ROOT, None, max_in_flight=0)

    def test_real_validator_accepts_digest_and_rejects_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate, trace = root / "submission.json", root / "trace.jsonl"
            candidate.write_text(json.dumps(CANDIDATE))
            trace.write_bytes(TRACE)
            argv = [sys.executable, "-B", str(sd.ROOT / "scripts/check_submission.py"),
                    str(candidate), "--final", "--trace", str(trace)]
            result = subprocess.run(argv, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout.decode())
            candidate.write_text(json.dumps(dict(CANDIDATE, image="REPLACE_IMAGE")))
            self.assertNotEqual(subprocess.run(argv, capture_output=True).returncode, 0)

    def test_cli_uses_source_env_and_proxy_with_safe_argv(self):
        with tempfile.TemporaryDirectory(prefix="cli path ") as tmp:
            root = Path(tmp)
            (root / "env.sh").write_text('export MOCK_ENV_MARKER=sourced\necho SHOULD_NOT_LOG\n')
            fake = root / "playground"
            fake.write_text('#!/usr/bin/env python3\nimport os,sys,json\nprint(json.dumps({"argv":sys.argv[1:],'
                            '"marker":os.environ.get("MOCK_ENV_MARKER"),"proxy":os.environ.get("NODE_USE_ENV_PROXY")}))\n')
            fake.chmod(0o755)
            with patch.dict(os.environ, {"PATH": str(root) + os.pathsep + os.environ["PATH"]}):
                output = sd.Backend(root).cli(["status", "--attempt-id", "literal $(touch NOPE)"])
            result = json.loads(output)
            self.assertEqual(result["marker"], "sourced")
            self.assertEqual(result["proxy"], "1")
            self.assertEqual(result["argv"][-1], "literal $(touch NOPE)")
            self.assertFalse((root / "NOPE").exists())

    def test_credentials_only_enter_environment_and_api_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            path = home / ".config/playground/credentials.env"
            path.parent.mkdir(parents=True)
            path.write_text('PLAYGROUND_TOKEN="synthetic-unit-value"\n')
            with patch.object(Path, "home", return_value=home), patch.dict(os.environ, {}, clear=False):
                out = io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                    sd.credential_env()
                self.assertEqual(out.getvalue(), "")
                backend = sd.Backend(home)
                response = contextlib.nullcontext(io.BytesIO(b'{"id":"test"}'))
                with patch("urllib.request.build_opener") as opener:
                    opener.return_value.open.return_value = response
                    self.assertEqual(backend.get_json("/auth/me"), {"id": "test"})
                    request = opener.return_value.open.call_args.args[0]
                    self.assertEqual(request.get_header("Authorization"), "Bearer synthetic-unit-value")
                    self.assertNotIn("synthetic-unit-value", request.full_url)

    def test_mock_dry_run_needs_no_credentials_and_writes_no_production_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, "-B", str(sd.ROOT / "scripts/submit_daemon.py"),
                                     "--root", tmp, "--dry-run"], capture_output=True,
                                    env=dict(os.environ, HOME=tmp))
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            self.assertIn(b"MOCK ONLY", result.stdout)
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
