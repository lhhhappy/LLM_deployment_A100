#!/usr/bin/env python3
"""T24 CPU-only tests. Fake Trisol/runner and temporary data; no APPROVED files."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import trisol_test_daemon as mod


class Clock:
    def __init__(self):
        self.value = 1790035200.0
    def now(self):
        return self.value
    def sleep(self, seconds):
        self.value += seconds


class MockCommands:
    def __init__(self, root, clock):
        self.root, self.clock = root, clock
        self.calls, self.rows = [], []
        self.waits = 0
        self.run_rc, self.collect_rc = 0, 0
        self.mirror_rc = 0
        self.deleting = False
        self.delete_stuck = False
        self.fail_create = False
        self.timeout_runner = False
        self.p0 = {"IF-01": "pass", "PF-01": "pass"}
        self.manifest = True
        self.exact = True
        self.passed = True
        self.hook = None
        self.spec = None
        self.cli_ready = True
    def run(self, argv, timeout, interrupted=lambda: False):
        self.calls.append(list(argv))
        if self.hook:
            self.hook(argv)
        if "--help" in argv:
            return (0, "--service-id --profiles --matrix --ladder-mode --levels --hint --budget-minutes --run-id --out --collect-only --max-n --max-levels") if self.cli_ready else (0, "--service --minutes")
        if "trisol" in argv:
            action = argv[argv.index("inference") + 1]
            if action == "list":
                assert "--scope" in argv and argv[argv.index("--scope")+1] == "mine"
                assert "--all" in argv
                if self.rows and not self.deleting:
                    if self.waits:
                        self.rows[0]["phase"] = "WaitingForAdmission"
                        self.waits -= 1
                    else:
                        self.rows[0]["phase"] = "Running"
                return 0, json.dumps({"scope": "mine", "items": self.rows, "total": len(self.rows)})
            if action == "create":
                assert not self.rows, "a second service was created"
                assert argv[argv.index("--gpu-count")+1] == "8"
                assert argv[argv.index("--replicas")+1] == "1"
                assert argv[argv.index("--command")+1] == "python3,-m,http.server,8000"
                self.rows = [{"id": "mock-1", "name": argv[argv.index("--name")+1], "phase": "WaitingForAdmission"}]
                if self.fail_create:
                    raise mod.Deadline("create-response-lost")
                return 0, json.dumps(self.rows[0])
            if action == "get":
                assert "--spec" in argv
                return 0, json.dumps(self.spec)
            if action == "delete":
                assert "--yes" in argv
                self.deleting = True
                self.rows = ([{**self.rows[0], "phase": "Deleting"}] if self.delete_stuck else [])
                return 0, '{"status":"deleting"}'
            raise AssertionError("unsupported bohr action " + action)
        if "--service-id" in argv:
            if "--collect-only" in argv:
                return self.collect_rc, ""
            if self.timeout_runner:
                self.clock.sleep(timeout)
                raise mod.Deadline("mock-runner-deadline")
            self.clock.sleep(20)
            out = Path(argv[argv.index("--out")+1])
            profile = Path(argv[argv.index("--profiles")+1])
            (out / "raw.jsonl").write_text('{}\n')
            mod.atomic_json(out / "run.json", {"wall_s": 20, "config": {"N": 18}})
            if self.manifest:
                mod.atomic_json(out / "session_result.json", {"run_id": out.name, "results": [
                    {"profile_sha256": mod.sha(profile), "matrix": "baseline", "N": 18,
                     "raw": "raw.jsonl", "run": "run.json", "p0": self.p0, "candidate_exact": self.exact}]})
            return self.run_rc, "DO NOT LOG secret child output"
        if any(str(arg).endswith("score_formal.py") for arg in argv):
            mod.atomic_json(Path(argv[argv.index("--out")+1]), {"estimated": {"passed": self.passed}, "tpot": {"tpot_mean": .02}})
            return 0, ""
        if any(str(arg).endswith("sim_closed_loop.py") for arg in argv):
            out = Path(argv[argv.index("--out-dir") + 1])
            out.mkdir(parents=True, exist_ok=True)
            mod.atomic_json(out / "sweep.json", {"ladders": []})
            return 0, ""
        if argv[0] in {"ssh", "rsync"}:
            return self.mirror_rc, ""
        raise AssertionError(argv)


class DaemonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="t24-mock-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "submission").mkdir()
        self.candidate = self.root / "submission/candidate-b0.json"
        mod.atomic_json(self.candidate, {"image": "registry.example/test@sha256:" + "a"*64,
                                      "command": ["python3", "-m", "sglang.launch_server"],
                                      "env": {"SGLANG_OPT_USE_TOPK_V2": "0"}, "model_name": "default"})
        (self.root / "submission/stub-trace.jsonl").write_text('{"type":"session_start"}\n')
        (self.root / "tests").mkdir()
        (self.root / "tests/TEST_PLAN.md").write_text('| IF-01 | P0 | test | todo |\n| PF-01 | P0 | test | todo |\n')
        self.spec = {"image_ref": json.loads(self.candidate.read_text())["image"], "profiles": ["submission/candidate-b0.json"],
                     "matrix": ["baseline"], "ladder": {"mode": "levels", "levels": [6, 10, 18]},
                     "time_budget_minutes": 30, "notes": "mock"}
        self.cfg = dict(mod.DEFAULTS, gpu_product_id="mock-gpu", poll_seconds=30, command_timeout_seconds=10,
                        cleanup_reserve_seconds=30, delete_reserve_seconds=10, keep_alive=False)
        self.clock = Clock()
        self.backend = MockCommands(self.root, self.clock)
        self.daemon = mod.Daemon(self.root, self.cfg, self.backend, self.clock.now, self.clock.sleep)
        self.approved = set()
        self.mock = patch.object(mod, "approval", side_effect=lambda item: item.name in self.approved)
        self.mock.start()
        self.addCleanup(self.mock.stop)
        self.addCleanup(lambda: self.assertEqual(list(self.root.rglob("APPROVED")), []))
    def item(self, name="01-first", approved=True, **changes):
        item = self.root / "tests/queue" / name
        item.mkdir(parents=True)
        mod.atomic_json(item / "spec.json", {**self.spec, **changes})
        if approved:
            self.approved.add(name)
        return item
    def actions(self):
        return [a[a.index("inference")+1] for a in self.backend.calls if "trisol" in a]
    def record(self, name="01-first"):
        return self.daemon.state["items"][name]
    def events(self):
        return [json.loads(line) for line in self.daemon.events.read_text().splitlines()]
    def run_one(self, **changes):
        self.item(**changes)
        self.assertTrue(self.daemon.tick())
        return self.record()

    def test_unapproved_queue_is_completely_inert(self):
        self.item(approved=False)
        self.assertFalse(self.daemon.tick())
        self.assertEqual(self.backend.calls, [])
    def test_global_stop_is_inert(self):
        self.item()
        (self.daemon.queue / "STOP").touch()
        self.assertFalse(self.daemon.tick())
        self.assertEqual(self.backend.calls, [])
    def test_item_stop_skips_to_next_approved(self):
        self.item()
        (self.daemon.queue / "01-first/STOP").touch()
        self.item("02-second")
        self.assertTrue(self.daemon.tick())
        self.assertNotIn("01-first", self.daemon.state["items"])
        self.assertEqual(self.record("02-second")["status"], "done")
    def test_fifo_two_sessions_and_idempotent_result(self):
        self.item("02-second")
        self.item()
        self.assertTrue(self.daemon.tick())
        self.backend.deleting = False
        self.assertTrue(self.daemon.tick())
        self.assertFalse(self.daemon.tick())
        mutations = [x for x in self.actions() if x in {"create", "delete"}]
        self.assertEqual(mutations, ["create", "delete", "create", "delete"])
        result = [x for x in self.events() if x["event"] == "RESULT"]
        self.assertEqual([x["item"] for x in result], ["01-first", "02-second"])
        self.daemon.publish(self.daemon.queue / "01-first", self.record())
        self.assertEqual(sum(x["event"] == "RESULT" for x in self.events()), 2)
    def test_runner_collect_score_mirror_result_delete_order(self):
        self.run_one()
        calls = self.backend.calls
        collect = next(i for i,c in enumerate(calls) if "--collect-only" in c)
        score = next(i for i,c in enumerate(calls) if any(x.endswith("score_formal.py") for x in c))
        mirror = next(i for i,c in enumerate(calls) if c[0] == "rsync")
        delete = next(i for i,c in enumerate(calls) if "delete" in c)
        self.assertLess(collect, score)
        self.assertLess(score, mirror)
        self.assertLess(mirror, delete)
        kinds = [x["event"] for x in self.events()]
        self.assertLess(kinds.index("RESULT"), kinds.index("DELETE_INTENT"))
        self.assertIn("PF-01/PF-02", (self.root / "notes/experiments.md").read_text())
    def test_hours_of_admission_wait_are_not_charged(self):
        self.backend.waits = 241
        r = self.run_one()
        self.assertGreaterEqual(r["queue_wait_s"], 7200)
        self.assertLess(r["deleted_at"] - r["charged_from"], 70)
        self.assertEqual(r["status"], "done")
    def test_admission_timeout_deletes(self):
        self.backend.waits = 100
        r = self.run_one(max_admission_wait_minutes=1)
        self.assertEqual(r["status"], "failed")
        self.assertIn("delete", self.actions())
    def test_runner_failure_still_collects_and_deletes(self):
        self.backend.run_rc = 5
        r = self.run_one()
        self.assertEqual(r["status"], "failed")
        self.assertTrue(r["collected"])
        self.assertIn("delete", self.actions())
    def test_hard_deadline_collects_then_deletes(self):
        self.backend.timeout_runner = True
        r = self.run_one(time_budget_minutes=2)
        self.assertLessEqual(r["deleted_at"], r["hard_deadline"])
        self.assertEqual(r["status"], "failed")
        self.assertIn("delete", self.actions())
    def test_collect_failure_does_not_leak_service(self):
        self.backend.collect_rc = 2
        self.assertEqual(self.run_one()["status"], "failed")
        self.assertIn("delete", self.actions())
    def test_mirror_failure_does_not_leak_service(self):
        self.backend.mirror_rc = 2
        self.assertEqual(self.run_one()["status"], "failed")
        self.assertIn("delete", self.actions())
    def test_missing_manifest_fails_without_inventing_pass(self):
        self.backend.manifest = False
        r = self.run_one()
        self.assertEqual(r["status"], "failed")
        self.assertIsNone(self.daemon.result(r)["best"])
        self.assertIn("delete", self.actions())
    def test_other_owned_service_blocks_create(self):
        self.item()
        self.backend.rows = [{"id": "other", "name": "manual-pod", "phase": "Running"}]
        self.assertFalse(self.daemon.tick())
        self.assertNotIn("create", self.actions())
        self.assertNotIn("delete", self.actions())
    def test_adopt_queued_service_never_creates(self):
        self.item(service_name="lh-arena-sess-a")
        self.backend.rows = [{"id": "queued", "name": "lh-arena-sess-a", "phase": "WaitingForAdmission"}]
        self.backend.waits = 2
        self.backend.spec = {"gpu_count": 8, "replicas": 1, "image_ref": self.spec["image_ref"],
                             "command": ["python3", "-m", "http.server", "8000"]}
        self.assertTrue(self.daemon.tick())
        self.assertNotIn("create", self.actions())
        self.assertIn("delete", self.actions())
        self.assertEqual(self.record()["status"], "done")
    def test_adopt_running_charges_existing_time(self):
        self.item(service_name="lh-arena-sess-a")
        self.backend.rows = [{"id": "queued", "name": "lh-arena-sess-a", "phase": "Running",
                              "allocated_at": mod.stamp(self.clock.now() - 60)}]
        self.backend.spec = {"gpu_count": 8, "image_ref": self.spec["image_ref"], "command_line": "python3 -m http.server 8000"}
        self.assertTrue(self.daemon.tick())
        r = self.record()
        self.assertGreaterEqual(r["deleted_at"] - r["charged_from"], 80)
    def test_wrong_adopted_image_never_executes_or_deletes(self):
        self.item(service_name="lh-arena-sess-a")
        self.backend.rows = [{"id": "queued", "name": "lh-arena-sess-a", "phase": "WaitingForAdmission"}]
        self.backend.spec = {"gpu_count": 8, "image_ref": "wrong", "command_line": "python3 -m http.server 8000"}
        self.assertFalse(self.daemon.tick())
        self.assertNotIn("delete", self.actions())
    def test_missing_gpu_product_blocks_before_create(self):
        self.cfg["gpu_product_id"] = None
        self.item()
        self.assertFalse(self.daemon.tick())
        self.assertNotIn("create", self.actions())
    def test_runner_contract_blocks_before_inventory_or_create(self):
        self.backend.cli_ready = False
        self.item()
        self.assertFalse(self.daemon.tick())
        self.assertEqual(self.actions(), [])
    def test_daily_budget_exhaustion_blocks(self):
        self.cfg["daily_usage_adjustments"] = {mod.stamp(self.clock.now())[:10]: 12}
        self.item()
        self.assertTrue(self.daemon.tick())
        self.assertIn("create", self.actions())
    def test_budget_is_gpu_hours_not_machine_hours(self):
        r = self.run_one(time_budget_minutes=200)
        self.assertEqual(r["hard_deadline"] - r["charged_from"], 200 * 60)
    def test_daily_utc_split(self):
        self.assertEqual(mod.gpu_hours(0, 90000, "1970-01-01"), 192)
        self.assertEqual(mod.gpu_hours(0, 90000, "1970-01-02"), 8)
    def test_delete_ack_not_completion_blocks_next(self):
        self.backend.delete_stuck = True
        self.run_one()
        self.assertEqual(self.record()["status"], "cleanup")
        self.item("02-second")
        self.daemon.tick()
        self.assertEqual(self.actions().count("create"), 1)
        self.backend.rows = []
        self.backend.delete_stuck = False
        self.daemon.tick()
        self.assertEqual(self.record()["status"], "done")
    def test_restart_recovery_deletes_before_any_new_test(self):
        r = self.run_one()
        r.pop("deleted_at")
        r["status"] = "running"
        self.daemon.save()
        self.backend.rows = [{"id": r["service_id"], "name": r["service_name"], "phase": "Running"}]
        self.item("02-second")
        self.daemon = mod.Daemon(self.root, self.cfg, self.backend, self.clock.now, self.clock.sleep)
        self.daemon.tick()
        self.assertEqual(self.actions().count("create"), 1)
        self.assertNotIn("02-second", self.daemon.state["items"])
    def test_stop_during_run_still_reclaims(self):
        def hook(argv):
            if "--service-id" in argv and "--collect-only" not in argv:
                self.daemon.stopping = True
                raise mod.Deadline("stopped")
        self.backend.hook = hook
        self.assertEqual(self.run_one()["status"], "cancelled")
        self.assertIn("delete", self.actions())
    def test_ambiguous_create_response_reconciles_owned_name(self):
        self.backend.fail_create = True
        self.assertEqual(self.run_one()["status"], "failed")
        self.assertEqual(self.actions().count("create"), 1)
        self.assertIn("delete", self.actions())
    def test_approval_revoked_during_admission_cleans(self):
        self.backend.waits = 10
        def hook(argv):
            if "create" in argv:
                self.approved.clear()
        self.backend.hook = hook
        self.assertEqual(self.run_one()["status"], "failed")
        self.assertIn("delete", self.actions())
    def test_profile_mutation_after_enqueue_rejected(self):
        item = self.item()
        mod.atomic_json(item / "profiles.sha256.json", {"submission/candidate-b0.json": "0"*64})
        self.assertFalse(self.daemon.tick())
        self.assertEqual(self.backend.calls, [])
    def test_promotion_requires_all_p0(self):
        self.backend.p0 = {"IF-01": "pass"}
        self.run_one(promote={"min_n": 18})
        self.assertFalse((self.root / "submission/queue").exists())
    def test_promotion_uses_W9_queue_writer_without_approval(self):
        r = self.run_one(promote={"min_n": 18})
        self.assertIn("promoted", r)
        p = self.root / r["promoted"]
        self.assertEqual((p / "submission.json").read_bytes(), self.candidate.read_bytes())
        self.assertTrue((p / "stub-trace.jsonl").exists())
        self.assertFalse((p / "APPROVED").exists())
        self.daemon.promote(self.daemon.queue / "01-first", r)
        self.assertEqual(len(list((self.root / "submission/queue").glob("[0-9]*-*"))), 1)
    def test_promotion_rejects_threshold_and_modified_command(self):
        self.backend.exact = False
        self.run_one(promote={"min_n": 14})
        self.assertFalse((self.root / "submission/queue").exists())
    def test_dry_run_no_subprocess_writes_or_credentials(self):
        item = self.item(approved=False)
        before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        with patch.object(subprocess, "Popen", side_effect=AssertionError("no subprocess")), redirect_stdout(io.StringIO()) as out:
            self.assertEqual(mod.dry_run(self.root, self.cfg, item / "spec.json"), 0)
        self.assertIn('"dry_run": true', out.getvalue())
        self.assertEqual(before, {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()})
    def test_enqueue_snapshots_hashes_but_no_approval(self):
        source = self.root / "spec.json"
        mod.atomic_json(source, self.spec)
        item = mod.enqueue(self.root, source, "fresh")
        self.assertEqual(item.name, "01-fresh")
        self.assertFalse((item / "APPROVED").exists())
        self.assertEqual(mod.read_json(item / "profiles.sha256.json"), {"submission/candidate-b0.json": mod.sha(self.candidate)})
    def test_list_requires_our_scope_and_all_pages(self):
        for payload in ({"scope": "team", "items": []}, {"scope": "mine", "items": [], "total": 2},
                        {"scope": "mine", "items": [], "has_more": True}):
            with self.assertRaises(mod.GuardError):
                mod.services(payload)
    def test_invalid_ladder_and_paths_rejected(self):
        item = self.item(ladder={"mode": "levels", "levels": [7]})
        self.assertFalse(self.daemon.tick())
        with self.assertRaises(mod.GuardError):
            mod.below(self.root, "../escape")
    def test_singleton_lock(self):
        with mod.exclusive(self.root / "daemon.lock"):
            with self.assertRaises(mod.GuardError):
                with mod.exclusive(self.root / "daemon.lock"):
                    pass
    def test_watchdog_does_not_schedule_approved_items(self):
        self.item()
        # Dead parent and no owned unfinished state -> no subprocess, no queue execution.
        with patch.object(mod, "process_token", return_value=None), patch.object(subprocess, "Popen", side_effect=AssertionError("unexpected child")):
            self.assertEqual(mod.watchdog(self.root, self.cfg, "999999:1"), 0)
    def test_process_token_matches_self(self):
        self.assertTrue(mod.process_token(os.getpid()).startswith(str(os.getpid()) + ":"))

    def test_watchdog_reclaims_orphan_runner_and_service(self):
        r = self.run_one()
        r.pop("deleted_at")
        r["status"] = "running"
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        self.addCleanup(lambda: child.poll() is None and child.kill())
        r["runner_process"] = mod.process_token(child.pid)
        self.daemon.save()
        self.backend.rows = [{"id": r["service_id"], "name": r["service_name"], "phase": "Running"}]
        with patch.object(mod, "Commands", return_value=self.backend), patch.object(mod.time, "sleep", return_value=None):
            self.assertEqual(mod.watchdog(self.root, self.cfg, "999999:1"), 0)
        child.wait(timeout=3)
        self.assertLess(child.returncode, 0)
        self.assertFalse(self.backend.rows)
        self.assertEqual(self.actions().count("create"), 1)

    def test_publish_disk_failure_still_deletes(self):
        self.item()
        with patch.object(self.daemon, "publish", side_effect=OSError("disk failure")):
            self.daemon.tick()
        self.assertIn("delete", self.actions())
        self.assertFalse(self.backend.rows)

    def test_stale_service_identity_never_deletes_replacement(self):
        r = self.run_one()
        r.pop("deleted_at")
        r["status"] = "cleanup"
        before = self.actions().count("delete")
        self.backend.rows = [{"id": "replacement", "name": r["service_name"], "phase": "Running"}]
        self.daemon.delete_owned(self.daemon.queue / "01-first", r)
        self.assertEqual(self.actions().count("delete"), before)
        self.assertNotIn("deleted_at", r)

    def test_rung_mismatch_is_not_scored_as_claimed_N(self):
        r = self.run_one()
        out = self.root / "runs/01-first"
        mod.atomic_json(out / "run.json", {"config": {"N": 6}})
        with self.assertRaisesRegex(mod.GuardError, "rung-metadata"):
            self.daemon.score(r, out)

    def test_empty_p0_is_never_vacuous_pass(self):
        (self.root / "tests/TEST_PLAN.md").write_text("No cases yet")
        r = self.run_one(promote={"min_n": 18})
        self.assertNotIn("promoted", r)

    def test_failed_scored_gate_is_not_promoted(self):
        self.backend.passed = False
        r = self.run_one(promote={"min_n": 2})
        self.assertNotIn("promoted", r)
        self.assertIsNone(self.daemon.result(r)["best"])

    def test_no_strategy_in_default_public_service_name(self):
        item = self.item(name="01-spf-d1-snapshot-comparison")
        spec, _ = mod.load_spec(self.root, item)
        self.assertRegex(spec["service_name"], r"^lh-t24-[0-9a-f]{12}$")

    def test_numeric_queue_order_after_99(self):
        self.item("100-later")
        self.item("99-earlier")
        self.assertTrue(self.daemon.tick())
        self.assertIn("99-earlier", self.daemon.state["items"])
        self.assertNotIn("100-later", self.daemon.state["items"])

    def test_admission_collect_identity_keeps_budget_honest(self):
        self.backend.fail_create = True
        r = self.run_one()
        self.assertIsNotNone(r["charged_from"])

    def test_lost_create_without_visible_service_is_quarantined(self):
        r = self.run_one()
        r.pop("deleted_at")
        r["service_id"] = None
        r["create_acknowledged"] = False
        r["status"] = "cleanup"
        self.daemon.delete_owned(self.daemon.queue / "01-first", r)
        self.assertNotIn("deleted_at", r)
        self.assertEqual(r["status"], "cleanup")

    def test_adopted_service_over_budget_is_reclaimed(self):
        self.cfg["daily_usage_adjustments"] = {mod.stamp(self.clock.now())[:10]: 12}
        self.item(service_name="lh-arena-sess-a")
        self.backend.rows = [{"id": "queued", "name": "lh-arena-sess-a", "phase": "Running",
                              "allocated_at": mod.stamp(self.clock.now() - 60)}]
        self.backend.spec = {"gpu_count": 8, "image_ref": self.spec["image_ref"], "command_line": "python3 -m http.server 8000"}
        self.assertTrue(self.daemon.tick())
        self.assertNotIn("create", self.actions())
        self.assertIn("delete", self.actions())
        self.assertEqual(self.record()["status"], "done")

    def test_keep_alive_reuses_one_service_and_holds_after_queue(self):
        self.daemon.cfg["keep_alive"] = True
        self.backend.spec = {"gpu_count": 8, "replicas": 1, "image_ref": self.spec["image_ref"],
                             "command": ["python3", "-m", "http.server", "8000"]}
        self.item("01-first")
        self.assertTrue(self.daemon.tick())
        self.item("02-second")
        self.assertTrue(self.daemon.tick())
        mutations = [x for x in self.actions() if x in {"create", "delete"}]
        self.assertEqual(mutations, ["create"])
        self.assertTrue(any(x["event"] == "SERVICE_HELD" for x in self.events()))

    def test_unlimited_default_ignores_daily_usage(self):
        self.assertEqual(mod.DEFAULTS["daily_gpu_hours"], 0)
        self.cfg["daily_usage_adjustments"] = {mod.stamp(self.clock.now())[:10]: 999}
        self.assertTrue(self.run_one(time_budget_minutes=200))

    def test_queue_low_is_single_event(self):
        self.daemon.cfg["target_queue_depth"] = 5
        self.item()
        self.daemon.tick()
        self.daemon.tick()
        self.assertEqual(sum(x["event"] == "QUEUE_LOW" for x in self.events()), 1)

    def test_idle_hold_releases_retained_service(self):
        self.daemon.cfg.update(keep_alive=True, idle_hold_seconds=10)
        self.run_one()
        self.clock.value += 11
        self.daemon.tick()
        self.assertIn("delete", self.actions())


    def test_real_runner_help_and_mock_end_to_end_all_modes(self):
        from session_a import test_session_a as fixture
        runner = fixture.runner
        real_runner = ["bash", str(mod.ROOT / "scripts/session_a/run_session_a.sh")]
        self.cfg.update(runner=real_runner, daily_gpu_hours=0)
        self.daemon.cfg.update(self.cfg)
        original = self.backend.run
        controllers = []
        def controller(args):
            ctl = fixture.MockPodController(args, self.root / "mock-pod")
            controllers.append(ctl)
            return ctl
        def commands(argv, timeout, interrupted=lambda: False):
            if argv[:len(real_runner)] == real_runner:
                self.backend.calls.append(list(argv))
                if "--help" in argv:
                    # This is the real local wrapper/help, never a fake flag string.
                    return mod.Commands(self.root).run(argv, timeout)
                with patch.object(runner, "Controller", side_effect=controller), redirect_stdout(io.StringIO()), \
                     patch.object(subprocess, "run", side_effect=AssertionError("unmocked subprocess")):
                    return runner.main(argv[len(real_runner):]), ""
            return original(argv, timeout, interrupted)
        self.backend.run = commands
        specs = [({"mode": "levels", "levels": [6, 10, 14, 18], "max_n": 22, "max_levels": 4}, ["baseline", "spf_d1v12", "hrrn"]),
                 ({"mode": "fast", "hint": 14, "max_n": 22, "max_levels": 3}, ["d1v12", "spf"]),
                 ({"mode": "official-climb", "max_n": 22}, ["d1"])]
        for index, (ladder, matrix) in enumerate(specs):
            name = f"{index+1:02d}-contract"
            self.item(name, ladder=ladder, matrix=matrix, time_budget_minutes=1000)
            self.backend.deleting = False
            self.assertTrue(self.daemon.tick())
            r = self.record(name)
            self.assertEqual(r["outcome"], "done", r.get("reason"))
            self.assertTrue(r["collected"])
            self.assertTrue(r["scored"])
            self.assertEqual(set(x["matrix"] for x in r["scores"]), set(matrix))
            self.assertTrue(all(x["candidate_exact"] is False for x in r["scores"]))
            run_ctl, collect_ctl = controllers[-2:]
            self.assertEqual(run_ctl.args.service, r["service_id"])
            self.assertEqual(run_ctl.args.ladder_mode, ladder["mode"])
            self.assertEqual(run_ctl.out, self.root / "runs" / name)
            self.assertEqual(run_ctl.args.profiles, [Path(p) for p in r["snapshot_profiles"]])
            self.assertFalse(any(c[0] in ("launch", "stop") for c in collect_ctl.calls))
            self.assertFalse(list((self.root / "submission/queue").glob("*/submission.json")))

    def test_five_queue_specs_runner_argv_round_trip_and_recovery_zero(self):
        from session_a import test_session_a as fixture
        real_runner = ["bash", str(mod.ROOT / "scripts/session_a/run_session_a.sh")]
        self.daemon.cfg["runner"] = real_runner
        # Only --help uses a subprocess; no bohr, service, SSH or GPU calls.
        self.daemon.commands = mod.Commands(self.root)
        queue = sorted((mod.ROOT / "tests/queue").glob("[0-9]*-*/spec.json"))
        self.assertGreaterEqual(len(queue), 5)  # real queue (T33: 40 items)
        for path in queue:
            spec = mod.read_json(path)
            self.daemon.runner_preflight(spec)
            record = {"slug": path.parent.name, "service_id": "mock-id", "work_deadline": self.clock.now(),
                      "snapshot_profiles": [str(mod.ROOT / p) for p in spec["profiles"]]}
            for collect in (False, True):
                record["work_deadline"] = self.clock.now() + (0 if collect else spec["time_budget_minutes"] * 60)
                argv = self.daemon.runner_command(spec, record, collect=collect)
                args = fixture.runner.parser().parse_args(argv[len(real_runner):])
                self.assertEqual(args.matrix, ",".join(spec["matrix"]))
                self.assertEqual(args.ladder_mode, spec["ladder"]["mode"])
                self.assertEqual(args.service, "mock-id")
                self.assertEqual(args.collect_only, collect)
                self.assertEqual(args.minutes, 0 if collect else spec["time_budget_minutes"])
                for key in ("levels", "hint", "max_n", "max_levels"):
                    if key in spec["ladder"]:
                        self.assertEqual(getattr(args, key), spec["ladder"][key])
                output = io.StringIO()
                with redirect_stdout(output), patch.object(subprocess, "run", side_effect=AssertionError("dry-run subprocess")):
                    self.assertEqual(fixture.runner.main([*argv[len(real_runner):], "--dry-run"]), 0)
                self.assertIn("COLLECT ONLY" if collect else "matrix=" + ",".join(spec["matrix"]), output.getvalue())


class RealProcessTests(unittest.TestCase):
    def test_mock_bohr_subprocess_argv_and_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mock = root / "fake_bohr.py"
            mock.write_text('import json,sys\nassert sys.argv[1:4]==["trisol","inference","list"]\nassert "--all" in sys.argv\nprint(json.dumps({"scope":"mine","items":[]}))\n')
            d = mod.Daemon(root, dict(mod.DEFAULTS, bohr=[sys.executable, str(mock)]))
            self.assertEqual(d.inventory(), [])
    def test_mock_runner_timeout_kills_process_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cmd = mod.Commands(root)
            began = time.monotonic()
            with self.assertRaises(mod.Deadline):
                cmd.run([sys.executable, "-c", "import time; time.sleep(30)"], .15)
            self.assertLess(time.monotonic() - began, 3)
    def test_real_scorer_on_synthetic_raw_is_estimated(self):
        from test_ladder_search import synthetic_files
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, run = synthetic_files(root, n=18)
            out = root / "score.json"
            rc, _ = mod.Commands(mod.ROOT).run([sys.executable, "-B", str(mod.ROOT / "scripts/score_formal.py"),
                                               "--raw", str(raw), "--run", str(run), "--out", str(out)], 15)
            self.assertEqual(rc, 0)
            score = mod.read_json(out)
            self.assertEqual(score["label"], "estimated")
            self.assertIn("passed", score["estimated"])
            self.assertAlmostEqual(score["tpot"]["tpot_mean"], .02)


class RealSpecAdoptionTests(unittest.TestCase):
    """T33: shape of real `bohr trisol inference get --spec --output json` (2026-09-22)."""
    IMAGE = "registry.dp.tech/dptech/dp/native/prod-3732438/20675/arena-sglang-glm53:260918"

    def real(self, command):
        return {"runtime": {"command": command, "image": "", "image_ref": self.IMAGE},
                "resources": {"gpu_count": 8, "gpu_model": "A100-SXM4-80GB", "replicas": 1},
                "gateway": {}, "public_access": False, "auth_mode": ""}

    def test_nested_spec_with_bind_is_adoptable(self):
        spec = {"image_ref": self.IMAGE}
        for command in (["python3", "-m", "http.server", "8000", "--bind", "0.0.0.0"],
                        ["python3", "-m", "http.server", "8000"]):
            mod.validate_adoption(spec, self.real(command), dict(mod.DEFAULTS))

    def test_real_list_row_phases(self):
        queued = {"status": "deploying", "stage": "launching", "substage": "queueing",
                  "reason": "WaitingForAdmission", "actual_replicas": 0}
        self.assertEqual(mod.phase(queued), "waitingforadmission")
        self.assertFalse(mod.allocated(queued))
        admitted = {"status": "deploying", "stage": "launching", "substage": "starting",
                    "reason": "ReadinessPending", "actual_replicas": 0}
        self.assertEqual(mod.phase(admitted), "starting")
        self.assertTrue(mod.allocated(admitted))
        self.assertEqual(mod.phase({"status": "running", "stage": "running", "reason": "-"}), "running")

    def test_nested_spec_other_command_rejected(self):
        with self.assertRaises(mod.GuardError):
            mod.validate_adoption({"image_ref": self.IMAGE},
                                  self.real(["python3", "-m", "sglang.launch_server"]), dict(mod.DEFAULTS))


if __name__ == "__main__":
    unittest.main()
