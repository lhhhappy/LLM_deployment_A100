"""CPU safety tests for local verification before deleting completed Pod copies."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pod = load("pod_archive", "scripts/pod/archive_run.py")
local = load("local_archive", "scripts/analysis/archive_completed_runs.py")


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.ax, self.dest = self.base / "ax", self.base / "local"
        for state in ("pending", "running", "done", "failed", "cancelled"):
            (self.ax / "queue" / state).mkdir(parents=True)
        self.run = self.make_run("071-test")

    def make_run(self, name):
        root = self.ax / "runs" / name
        (root / "N30").mkdir(parents=True)
        (root / "exit_code").write_text("0\n")
        (root / "server.log").write_text("[TP0] source log\n")
        (root / "N30/raw.jsonl").write_text('{"req_id":"x","output_tokens":10}\n')
        (self.ax / "queue/done" / (name + ".sh")).write_text("job")
        os.utime(root / "exit_code", (1, 1))
        return root

    def call(self, action, name, *args):
        if action == "manifest":
            return pod.manifest(self.ax, name, opened=set())
        if action == "read":
            return pod.read_chunk(self.ax, name, *args)
        if action == "cleanup":
            # The cleanup callback is reached only after a durable verified receipt.
            sha, location = args
            receipt = json.loads((Path(location) / "receipt.json").read_text())
            self.assertTrue(receipt["verified_local"])
            self.assertEqual(receipt["manifest_sha256"], sha)
            return pod.cleanup(self.ax, name, *args, opened=set())
        self.fail(action)

    def test_full_archive_verified_before_cleanup_and_idempotent_receipt(self):
        result = local.archive_one(self.run.name, self.dest, cleanup=True, call=self.call)
        self.assertFalse(self.run.exists())
        target = Path(result["archive_path"])
        self.assertTrue((target / "files/N30/raw.jsonl").is_file())
        self.assertTrue(result["source_cleaned"])
        again = pod.cleanup(self.ax, self.run.name, result["manifest_sha256"], str(target), opened=set())
        self.assertTrue(again["cleaned"])
        self.assertTrue((self.ax / "queue/done" / (self.run.name + ".sh")).exists())

    def test_default_archives_without_deleting(self):
        result = local.archive_one(self.run.name, self.dest, call=self.call)
        self.assertTrue(self.run.exists())
        self.assertFalse(result["source_cleaned"])

    def test_corrupt_or_truncated_transport_preserves_source(self):
        def corrupt(action, name, *args):
            reply = self.call(action, name, *args)
            return {"data": "eA=="} if action == "read" else reply
        with self.assertRaisesRegex(ValueError, "truncated"):
            local.archive_one(self.run.name, self.dest, cleanup=True, call=corrupt)
        self.assertTrue((self.run / "N30/raw.jsonl").is_file())
        self.assertFalse(list(self.dest.rglob("receipt.json")))

    def test_new_file_after_local_verification_prevents_cleanup(self):
        def changed(action, name, *args):
            if action == "cleanup":
                (self.run / "late.log").write_text("not archived")
            return self.call(action, name, *args)
        with self.assertRaisesRegex(ValueError, "changed"):
            local.archive_one(self.run.name, self.dest, cleanup=True, call=changed)
        self.assertTrue((self.run / "late.log").exists())

    def test_running_current_log_and_open_descriptors_protected(self):
        done = self.ax / "queue/done" / (self.run.name + ".sh")
        done.rename(self.ax / "queue/running" / done.name)
        with self.assertRaisesRegex(ValueError, "terminal"):
            pod.manifest(self.ax, self.run.name, opened=set())
        (self.ax / "queue/running" / done.name).rename(done)
        with self.assertRaisesRegex(ValueError, "open file"):
            pod.manifest(self.ax, self.run.name, opened={self.run / "server.log"})
        (self.ax / "engine_log_path").write_text(str(self.run / "server.log"))
        with self.assertRaisesRegex(ValueError, "current engine"):
            pod.manifest(self.ax, self.run.name, opened=set())

    def test_relative_alias_archived_before_owner(self):
        other = self.make_run("072-alias")
        (other / "server.log").unlink()
        (other / "server.log").symlink_to("../071-test/server.log")
        with self.assertRaisesRegex(ValueError, "references"):
            pod.manifest(self.ax, self.run.name, opened=set())
        result = local.archive_one(other.name, self.dest, cleanup=True, call=self.call)
        archived = Path(result["archive_path"]) / "files/server.log"
        self.assertFalse(archived.is_symlink())
        self.assertEqual(archived.read_bytes(), (self.run / "server.log").read_bytes())
        self.assertFalse(other.exists())
        self.assertTrue(self.run.exists())
        local.archive_one(self.run.name, self.dest, cleanup=True, call=self.call)
        self.assertFalse(self.run.exists())

    def test_internal_relative_alias_survives_detach_validation(self):
        (self.run / "N30/server.log").symlink_to("../server.log")
        result = local.archive_one(self.run.name, self.dest, cleanup=True, call=self.call)
        target = Path(result["archive_path"]) / "files"
        self.assertEqual((target / "server.log").read_bytes(), (target / "N30/server.log").read_bytes())

    def test_external_symlink_and_directory_symlink_refused(self):
        (self.run / "outside").symlink_to(self.base / "external")
        (self.base / "external").write_text("keep")
        with self.assertRaisesRegex(ValueError, "outside runs"):
            pod.manifest(self.ax, self.run.name, opened=set())
        (self.run / "outside").unlink()
        (self.run / "dirlink").symlink_to(self.run / "N30", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "directory symlink"):
            pod.manifest(self.ax, self.run.name, opened=set())

    def test_detach_validation_failure_restores_source(self):
        meta = pod.manifest(self.ax, self.run.name, opened=set())
        original = pod.inventory
        def changed(root, original_root=None):
            if original_root is not None:
                raise ValueError("concurrent change")
            return original(root)
        with patch.object(pod, "inventory", changed), self.assertRaisesRegex(ValueError, "concurrent"):
            pod.cleanup(self.ax, self.run.name, meta["manifest_sha256"], "verified-local", opened=set())
        self.assertTrue((self.run / "N30/raw.jsonl").exists())

    def test_grace_period_and_path_traversal(self):
        os.utime(self.run / "exit_code", None)
        with self.assertRaisesRegex(ValueError, "grace"):
            pod.manifest(self.ax, self.run.name, opened=set())
        with self.assertRaisesRegex(ValueError, "invalid run"):
            pod.root_for(self.ax, "../outside")
        with self.assertRaisesRegex(ValueError, "invalid chunk"):
            pod.read_chunk(self.ax, self.run.name, "../../outside", 0, 10)


if __name__ == "__main__":
    unittest.main()
