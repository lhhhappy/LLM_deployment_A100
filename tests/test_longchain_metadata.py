"""Exercise the actual derivative writer and checker, including destructive CLI regressions."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import unittest

import test_longchain_check
from scripts.longchain.longchain_check import check_dataset
from scripts.longchain.longchain_metadata import checked_output, digest, publish, read_rows, write_rows
from scripts.longchain.repair_metadata import apply_gaps, execute
from scripts.longchain.finalize_v5g import tail_only


class MetadataRepairTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_longchain_check.LongchainCheckTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.fixture.small_generated_base()
        self.parent = Path(self.fixture.root)
        self.output = self.parent.parent / (self.parent.name + "-derived")
        self.addCleanup(self.cleanup_output)
        self.manifest = json.loads((self.parent / "manifest.json").read_text())
        self.manifest["set"] = "fixture"
        cohort = json.loads((self.parent / "cohort.json").read_text())
        cohort["set"] = "fixture"
        (self.parent / "cohort.json").write_text(json.dumps(cohort))
        self.manifest["artifacts"]["cohort.json"] = digest(self.parent / "cohort.json")
        (self.parent / "manifest.json").write_text(json.dumps(self.manifest))
        self.rows = read_rows(self.parent / "requests.jsonl")

    def cleanup_output(self):
        import shutil
        if self.output.exists():
            shutil.rmtree(self.output)

    def publish(self, rows=None, **kwargs):
        return publish(self.parent, self.output, self.rows if rows is None else rows,
                       operation={"kind": "test"}, name="derived", **kwargs)

    def test_real_checker_accepts_output_edit_with_unchanged_body_and_cohort(self):
        source_hash = digest(self.parent / "requests.jsonl")
        self.rows[1]["max_output_i"] = 6
        manifest = self.publish()
        checked = check_dataset(str(self.output), ".")
        self.assertEqual(checked["status"], "STRUCTURAL_OK", checked["errors"])
        self.assertEqual(digest(self.parent / "requests.jsonl"), source_hash)
        self.assertTrue((self.output / "bodies/part.jsonl.gz").is_symlink())
        self.assertEqual(digest(self.parent / "bodies/part.jsonl.gz"), digest(self.output / "bodies/part.jsonl.gz"))
        self.assertEqual(read_rows(self.output / "chains.jsonl")[0]["max_output_i_sum"], 10)
        self.assertEqual(manifest["metadata_patches"][-1]["fields"], {"max_output_i": 1})
        self.assertIn("metadata_patch", read_rows(self.output / "provenance.jsonl")[1]["output_budget_origin"])

    def test_same_path_symlink_and_existing_output_rejected_before_mutation(self):
        before = digest(self.parent / "requests.jsonl")
        self.output.symlink_to(self.parent, target_is_directory=True)
        try:
            for target in (self.parent, self.output, self.parent / "child"):
                with self.assertRaises(ValueError):
                    checked_output(self.parent, target)
        finally:
            self.output.unlink()
        self.assertEqual(digest(self.parent / "requests.jsonl"), before)

    def test_actual_rebudget_and_regap_cli_cannot_overwrite_input(self):
        scripts = Path(__file__).resolve().parents[1] / "scripts/longchain"
        before = digest(self.parent / "requests.jsonl")
        for tool, flags in (("rebudget", ["--parent", str(self.parent), "--public", str(self.parent)]),
                            ("regap", ["--src", str(self.parent), "--organizer", str(self.parent / "chains.jsonl"), "--set", "bad"])):
            result = subprocess.run([sys.executable, "-B", str(scripts / (tool + ".py")),
                                     *flags, "--out", str(self.parent)], capture_output=True, text=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("output exists", result.stderr)
            self.assertEqual(digest(self.parent / "requests.jsonl"), before)

    def test_regap_without_imputed_gaps_writes_valid_manifest(self):
        organizer = self.parent / "organizer.jsonl"
        write_rows(organizer, [{"chain_id": "c", "first_dispatch_offset_ms": 0, "last_end_offset_ms": 100000}])
        script = Path(__file__).resolve().parents[1] / "scripts/longchain/regap.py"
        result = subprocess.run([sys.executable, "-B", str(script), "--src", str(self.parent),
            "--organizer", str(organizer), "--out", str(self.output), "--set", "unchanged-gaps"],
            capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(read_rows(self.output / "requests.jsonl"), self.rows)
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertIsNone(manifest["metadata_patches"][-1]["operation"]["factor"]["median"])
        checked = check_dataset(str(self.output), ".")
        self.assertEqual(checked["status"], "STRUCTURAL_OK", checked["errors"])

    def test_missing_bodies_explicitly_incomplete_and_rejected_by_checker(self):
        (self.parent / "bodies/part.jsonl.gz").unlink()
        with self.assertRaisesRegex(ValueError, "missing parent artifact"):
            self.publish()
        self.assertFalse(self.output.exists())
        manifest = self.publish(metadata_only=True)
        self.assertEqual(manifest["status"], "METADATA_ONLY_INCOMPLETE")
        self.assertEqual(manifest["missing_artifacts"], ["bodies/part.jsonl.gz"])
        self.assertEqual(check_dataset(str(self.output), ".")["status"], "INVALID")

    def test_stale_parent_hash_is_not_relabelled_as_valid_child(self):
        (self.parent / "requests.jsonl").write_text((self.parent / "requests.jsonl").read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.publish()
        self.assertFalse(self.output.exists())

    def test_cannot_relabel_token_counts_or_gate_phase(self):
        for field, value in (("glm_tokens", 999), ("uncached_expected", 0), ("phase", "context_reset")):
            edited = deepcopy(self.rows)
            edited[1][field] = value
            with self.assertRaisesRegex(ValueError, "cannot edit"):
                self.publish(edited)
        self.assertFalse(self.output.exists())

    def test_gap_patch_preserves_order_and_records_origin(self):
        before_order = [r["dispatch_offset_ms"] for r in self.rows]
        apply_gaps(self.rows, [{"chain_id": "c", "idx_in_chain": 1, "gap_ms": 91668}], "raw-end-to-start-sensitivity")
        self.publish()
        checked = check_dataset(str(self.output), ".")
        self.assertEqual(checked["status"], "STRUCTURAL_OK", checked["errors"])
        result = read_rows(self.output / "requests.jsonl")
        self.assertEqual([r["dispatch_offset_ms"] for r in result], before_order)
        p = read_rows(self.output / "provenance.jsonl")[1]
        self.assertFalse(p["gap_decomposition_known"])
        self.assertEqual(p["replay_gap_ms"], 91668)

    def test_bad_gap_patch_rejected_atomically(self):
        cases = [
            ([{"req_id": "p:canon:a", "gap_ms": 10}], "capped-replay"),
            ([{"req_id": "unknown", "gap_ms": 10}], "capped-replay"),
            ([{"req_id": "p:canon:b", "gap_ms": 400000}], "capped-replay"),
            ([{"req_id": "p:canon:b", "gap_ms": -1}], "raw-end-to-start-sensitivity"),
            ([{"req_id": "p:canon:b", "gap_ms": 10}] * 2, "capped-replay"),
            ([{"req_id": "p:canon:b", "gap_ms": 10}], None),
        ]
        for patches, basis in cases:
            original = deepcopy(self.rows)
            with self.assertRaises(ValueError):
                apply_gaps(self.rows, patches, basis)
            self.assertEqual(self.rows, original)

    def test_cohort_cannot_drop_requests_or_change_within_chain_order(self):
        cohort = json.loads((self.parent / "cohort.json").read_text())
        cohort["chains"][0]["req_ids"].reverse()
        path = self.parent / "bad-cohort.json"
        path.write_text(json.dumps(cohort))
        with self.assertRaisesRegex(ValueError, "cohort replacement"):
            self.publish(cohort_path=path)

    def test_import_existing_v5_budgets_only_and_refresh_manifest(self):
        donor = deepcopy(self.rows)
        donor[1]["max_output_i"] = 6
        path = self.parent / "output-donor.jsonl"
        write_rows(path, donor)
        args = argparse.Namespace(parent=self.parent, out=self.output, set="repaired",
            outputs_from=path, cohort_from=None, gap_patch=None, gap_basis=None, metadata_only=False)
        manifest = execute(args)
        self.assertEqual(manifest["set"], "repaired")
        self.assertIn("artifacts", manifest)
        self.assertEqual(read_rows(self.output / "requests.jsonl")[1]["max_output_i"], 6)

    def test_tail_arm_preserves_ordinary_waits_originals_outputs_and_heads(self):
        base = [dict(pack="p", view="canon", logical_call_id=str(i), split="synthetic",
                     replay_gap_ms=i*1000, max_output_i=100+i) for i in range(21)]
        proposal = deepcopy(base)
        for row in proposal:
            row["replay_gap_ms"] = 200000
            row["gap_regap_factor"] = 10
        result, receipt = tail_only(base, proposal, {"p:canon:0"})
        self.assertEqual(result[:20], base[:20])
        self.assertEqual(result[20]["replay_gap_ms"], 200000)
        self.assertEqual([r["max_output_i"] for r in result], [r["max_output_i"] for r in base])
        self.assertEqual(receipt["threshold_ms"], 19000)
        base[20]["split"] = "original"
        self.assertEqual(tail_only(base, proposal, {"p:canon:0"})[0], base)


if __name__ == "__main__":
    unittest.main()
