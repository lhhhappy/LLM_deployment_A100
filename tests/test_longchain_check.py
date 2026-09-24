import gzip
import hashlib
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from scripts.analysis.longchain_check import (check_dataset, _chain_summary_errors, _workload_ledger,
                                             _canonical_digest, _actual_append_edge, _new_tool_block_errors)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


class LongchainCheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        os.makedirs(os.path.join(self.root, "bodies"))
        self.r1 = {"pack": "p", "view": "canon", "logical_call_id": "a", "chain_id": "c",
                   "dispatch_offset_ms": 10, "phase": "session_start", "glm_tokens": 8,
                   "glm_lcp_with_prev": 0, "uncached_expected": 8, "in_serving_load": True,
                   "max_output_i": 4, "replay_gap_ms": 100}
        self.r2 = dict(self.r1, logical_call_id="b", dispatch_offset_ms=20,
                       phase="intra", glm_lcp_with_prev=3, uncached_expected=5)
        write_jsonl(os.path.join(self.root, "requests.jsonl"), [self.r1, self.r2])
        write_jsonl(os.path.join(self.root, "chains.jsonl"), [{"view": "canon", "chain_id": "c", "n_requests": 2}])
        self.write_bodies([
            {"req_id": "p:canon:a", "system": "s", "tools": [], "messages": [{"role": "user", "content": "one"}]},
            {"req_id": "p:canon:b", "system": "s", "tools": [], "messages": [{"role": "user", "content": "two"}]},
        ])

    def tearDown(self):
        self.tmp.cleanup()

    def write_bodies(self, bodies):
        with gzip.open(os.path.join(self.root, "bodies", "part.jsonl.gz"), "wt", encoding="utf-8") as fh:
            for body in bodies:
                fh.write(json.dumps(body) + "\n")

    def freeze_small_generated_fixture(self, *, kind="synthetic", include_provenance=True):
        """Complete hashed fixture for structural regression, not GLM acceptance."""
        with open(os.path.join(self.root, "requests.jsonl")) as f:
            rows = [json.loads(line) for line in f]
        with gzip.open(os.path.join(self.root, "bodies", "part.jsonl.gz"), "rt") as f:
            bodies = [json.loads(line) for line in f]
        append = int(_actual_append_edge(*bodies))
        chain = {"view": "canon", "chain_id": "c", "n_requests": 2,
                 "first_dispatch_offset_ms": rows[0]["dispatch_offset_ms"], "last_end_offset_ms": None,
                 "sum_glm_tokens": sum(r["glm_tokens"] for r in rows),
                 "sum_uncached_expected": sum(r["uncached_expected"] for r in rows),
                 "max_output_i_sum": sum(r["max_output_i"] for r in rows),
                 "phases": {"session_start": 1, "intra": 1},
                 "total_edges": 1, "append_only_edges": append, "append_only_frac": float(append),
                 "source_chain_targets": {"n_requests": 2, "sum_uncached_expected": 100,
                                          "sum_glm_tokens": 100, "max_output_i_sum": 8}}
        write_jsonl(os.path.join(self.root, "chains.jsonl"), [chain])
        provenance = [{"req_id": b["req_id"], "kind": "original" if i == 0 else kind,
                       "body_sha256": _canonical_digest(b), "added_tokens": 5}
                      for i, b in enumerate(bodies)]
        pp = os.path.join(self.root, "provenance.jsonl")
        if include_provenance:
            write_jsonl(pp, provenance)
        elif os.path.exists(pp):
            os.remove(pp)
        chains = [{"chain_id": "c", "req_ids": [b["req_id"] for b in bodies]}]
        sha = hashlib.sha256(json.dumps(chains, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
        with open(os.path.join(self.root, "cohort.json"), "w") as f:
            json.dump({"n_chains": 1, "n_requests": 2, "chains": chains, "cohort_sha256": sha}, f)
        artifacts = {}
        for dp, _, names in os.walk(self.root):
            for name in names:
                if name == "manifest.json":
                    continue
                path = os.path.join(dp, name)
                with open(path, "rb") as f:
                    artifacts[os.path.relpath(path, self.root)] = hashlib.sha256(f.read()).hexdigest()
        with open(os.path.join(self.root, "manifest.json"), "w") as f:
            json.dump({"generator": "regression-fixture", "max_context_tokens": 10000,
                       "artifacts": artifacts}, f)

    def small_generated_base(self):
        for r in (self.r1, self.r2):
            r.update(session_id="receiving-session", body_ref="bodies/part.jsonl.gz")
        write_jsonl(os.path.join(self.root, "requests.jsonl"), [self.r1, self.r2])
        self.write_bodies([
            {"req_id": "p:canon:a", "system": "s", "tools": [],
             "messages": [{"role": "user", "content": "one"}]},
            {"req_id": "p:canon:b", "system": "s", "tools": [],
             "messages": [{"role": "user", "content": "one"}, {"role": "assistant", "content": "done"}]},
        ])
        self.freeze_small_generated_fixture()
        baseline = check_dataset(self.root, ".")
        self.assertEqual(baseline["status"], "STRUCTURAL_OK", baseline["errors"])

    def test_generated_provenance_file_is_required_even_with_fresh_hashes(self):
        self.small_generated_base()
        self.freeze_small_generated_fixture(include_provenance=False)
        report = check_dataset(self.root, ".")
        self.assertEqual(report["status"], "INVALID")
        self.assertIn("generated dataset is missing provenance.jsonl", report["errors"])

    def test_generated_unknown_provenance_kind_is_rejected(self):
        self.small_generated_base()
        self.freeze_small_generated_fixture(kind="synthethic")
        report = check_dataset(self.root, ".")
        self.assertEqual(report["status"], "INVALID")
        self.assertTrue(any("unknown generated provenance kind" in e for e in report["errors"]))

    def test_synthetic_system_or_tools_change_cannot_hide_in_chain_summary(self):
        for field, value in (("system", "changed"), ("tools", [{"type": "function", "function": {"name": "other"}}])):
            with self.subTest(field=field):
                self.small_generated_base()
                with gzip.open(os.path.join(self.root, "bodies", "part.jsonl.gz"), "rt") as f:
                    bodies = [json.loads(line) for line in f]
                bodies[1][field] = value
                self.write_bodies(bodies)
                self.freeze_small_generated_fixture()
                report = check_dataset(self.root, ".")
                self.assertEqual(report["status"], "INVALID")
                self.assertTrue(any("changes system/tools" in e for e in report["errors"]))

    def test_generated_missing_session_or_changed_session_is_rejected(self):
        for changed in (False, True):
            with self.subTest(changed=changed):
                self.small_generated_base()
                if changed:
                    self.r2["session_id"] = "other-session"
                else:
                    self.r2.pop("session_id")
                write_jsonl(os.path.join(self.root, "requests.jsonl"), [self.r1, self.r2])
                self.freeze_small_generated_fixture()
                report = check_dataset(self.root, ".")
                self.assertEqual(report["status"], "INVALID")
                self.assertTrue(any("session_id" in e for e in report["errors"]))

    def test_new_tool_ids_cannot_reuse_prior_history_or_earlier_new_group(self):
        group = [{"role": "assistant", "tool_calls": [{"id": "call-1"}]},
                 {"role": "tool", "tool_call_id": "call-1", "content": "result"}]
        messages = group + group
        for start in (0, 2):
            with self.subTest(start=start):
                errors, _ = _new_tool_block_errors(messages, start, "r")
                self.assertTrue(any("reuses an existing history ID" in e or "repeat across the new suffix" in e
                                    for e in errors), errors)
        errors, _ = _new_tool_block_errors(group, 0, "r")
        self.assertEqual(errors, [])

    def test_repeated_adaptation_checks_ancestor_receipt_not_parent_current_body(self):
        self.small_generated_base()
        with tempfile.TemporaryDirectory() as parent:
            original_labels = {"glm_tokens": 7, "glm_lcp_with_prev": 1, "uncached_expected": 6}
            write_jsonl(os.path.join(parent, "requests.jsonl"), [self.r1])
            write_jsonl(os.path.join(parent, "provenance.jsonl"), [
                {"req_id": "p:canon:a", "kind": "adapted_original", "body_sha256": "b" * 64,
                 "source_body_sha256": "a" * 64, "source_frozen_labels": original_labels}])
            prov_path = os.path.join(self.root, "provenance.jsonl")
            with open(prov_path) as f:
                provenance = [json.loads(line) for line in f]
            provenance[0].update(kind="adapted_original", source_req_id="p:canon:a",
                                 source_body_sha256="a" * 64, source_frozen_labels=original_labels)
            write_jsonl(prov_path, provenance)
            manifest_path = os.path.join(self.root, "manifest.json")
            with open(manifest_path) as f:
                manifest = json.load(f)
            manifest["parent_root"] = parent
            with open(prov_path, "rb") as f:
                manifest["artifacts"]["provenance.jsonl"] = hashlib.sha256(f.read()).hexdigest()
            with open(manifest_path, "w") as f:
                json.dump(manifest, f)
            report = check_dataset(self.root, ".")
            self.assertEqual(report["status"], "STRUCTURAL_OK", report["errors"])
            self.assertEqual(report["source_hash_checks"]["verified_against_parent_provenance"], 1)

    def test_missing_body_is_invalid_even_without_tokenizer(self):
        self.write_bodies([{ "req_id": "p:canon:a", "system": "s", "messages": [] }])
        report = check_dataset(self.root, ".")
        self.assertEqual(report["status"], "INVALID")
        self.assertTrue(any("has no body" in e for e in report["errors"]))

    def test_cohort_duplicate_or_unknown_id_is_invalid(self):
        report = check_dataset(self.root, ".", cohort=["p:canon:a", "p:canon:a", "p:canon:ghost"])
        self.assertEqual(report["status"], "INVALID")
        self.assertTrue(any("duplicate request IDs" in e for e in report["errors"]))
        self.assertTrue(any("absent from serving" in e for e in report["errors"]))

    def test_explicit_bad_tool_pair_is_invalid(self):
        self.write_bodies([
            {"req_id": "p:canon:a", "system": "s", "messages": [{"role": "assistant", "tool_calls": [{"id": "known"}]}]},
            {"req_id": "p:canon:b", "system": "s", "messages": [{"role": "tool", "tool_call_id": "missing", "content": "x"}]},
        ])
        report = check_dataset(self.root, ".")
        self.assertEqual(report["status"], "INVALID")
        self.assertTrue(any("unknown call id" in e for e in report["errors"]))

    def test_structural_report_is_not_pass(self):
        report = check_dataset(self.root, ".")
        self.assertEqual(report["status"], "STRUCTURAL_OK")
        self.assertFalse(report["complete_pass"])
        self.assertTrue(any("not a complete PASS" in w for w in report["warnings"]))

    def test_original_source_head_tags_and_longer_source_chain_are_diagnostic_only(self):
        # Source metadata points to predecessors outside this visible replay slice.
        self.r1["glm_lcp_with_prev"] = 6
        self.r1["uncached_expected"] = 2
        self.r2["glm_lcp_with_prev"] = 7
        self.r2["uncached_expected"] = 1
        write_jsonl(os.path.join(self.root, "requests.jsonl"), [self.r1, self.r2])
        write_jsonl(os.path.join(self.root, "chains.jsonl"), [{"view": "canon", "chain_id": "c", "n_requests": 20}])
        self.write_bodies([
            {"req_id": "p:canon:a", "system": "", "messages": [{"role": "user", "content": "abcdefgh"}]},
            {"req_id": "p:canon:b", "system": "", "messages": [{"role": "user", "content": "abcXYZ12"}]},
        ])

        class FakeRenderer:
            def __init__(self, _tok_dir):
                self.tokenizer = self
            def render(self, body):
                return body["messages"][0]["content"]
            def encode(self, text, add_special_tokens=False):
                return [ord(c) for c in text]

        with patch.dict("sys.modules", {"s1_common": type("M", (), {"Renderer": FakeRenderer})}):
            report = check_dataset(self.root, ".", tok_dir="fake")
        self.assertEqual(report["status"], "VALID")
        self.assertEqual(report["per_request"]["p:canon:a"]["lcp_tokens"], 0)
        self.assertEqual(report["per_request"]["p:canon:b"]["lcp_tokens"], 3)
        self.assertTrue(any("source n_requests describes" in w for w in report["warnings"]))

    def test_empty_serving_set_is_invalid(self):
        self.r1["in_serving_load"] = False
        self.r2["in_serving_load"] = False
        write_jsonl(os.path.join(self.root, "requests.jsonl"), [self.r1, self.r2])
        report = check_dataset(self.root, ".")
        self.assertEqual(report["status"], "INVALID")
        self.assertTrue(any("empty serving/cohort" in e for e in report["errors"]))

    def test_original_provenance_is_checked_without_overwriting_source_labels(self):
        self.write_bodies([
            {"req_id": "p:canon:a", "system": "", "messages": [{"role": "user", "content": "abcdefgh"}]},
            {"req_id": "p:canon:b", "system": "", "messages": [{"role": "user", "content": "abcXYZ12"}]},
        ])
        self.r1["glm_lcp_with_prev"] = 6
        self.r1["uncached_expected"] = 2
        self.r2["glm_lcp_with_prev"] = 7
        self.r2["uncached_expected"] = 1
        write_jsonl(os.path.join(self.root, "requests.jsonl"), [self.r1, self.r2])
        write_jsonl(os.path.join(self.root, "provenance.jsonl"), [
            {"req_id": "p:canon:a", "kind": "original", "prompt_tokens": 8, "lcp_tokens": 0,
             "added_tokens": 8, "removed_tokens": 0, "previous_req_id": None},
            {"req_id": "p:canon:b", "kind": "original", "prompt_tokens": 8, "lcp_tokens": 3,
             "added_tokens": 5, "removed_tokens": 5, "previous_req_id": "p:canon:a"},
        ])
        class FakeRenderer:
            def __init__(self, _tok_dir): self.tokenizer = self
            def render(self, body): return body["messages"][0]["content"]
            def encode(self, text, add_special_tokens=False): return [ord(c) for c in text]
        with patch.dict("sys.modules", {"s1_common": type("M", (), {"Renderer": FakeRenderer})}):
            report = check_dataset(self.root, ".", tok_dir="fake")
        self.assertEqual(report["status"], "VALID")
        self.assertEqual(report["distribution"]["rendered_lcp_tokens"]["n"], 2)
        self.assertEqual(report["actual_token_accounting"]["new_tokens_after_visible_lcp"], 13)
        self.assertAlmostEqual(report["actual_token_accounting"]["added_fraction_of_prompt_tokens"], 13 / 16)
        self.assertEqual(self.r1["glm_lcp_with_prev"], 6)

    def test_adapted_original_frozen_labels_are_checked_like_synthetic(self):
        self.write_bodies([
            {"req_id": "p:canon:a", "system": "", "messages": [{"role": "user", "content": "abcdefgh"}]},
            {"req_id": "p:canon:b", "system": "", "messages": [{"role": "user", "content": "abcXYZ12"}]},
        ])
        self.r1.update(glm_tokens=8, glm_lcp_with_prev=0, uncached_expected=8)
        self.r2.update(glm_tokens=8, glm_lcp_with_prev=4, uncached_expected=4)
        write_jsonl(os.path.join(self.root, "requests.jsonl"), [self.r1, self.r2])
        write_jsonl(os.path.join(self.root, "provenance.jsonl"), [
            {"req_id": "p:canon:a", "kind": "adapted_original", "prompt_tokens": 8,
             "lcp_tokens": 0, "added_tokens": 8, "removed_tokens": 0, "previous_req_id": None,
             "source_frozen_labels": {"glm_tokens": 8, "glm_lcp_with_prev": 6, "uncached_expected": 2}},
            {"req_id": "p:canon:b", "kind": "adapted_original", "prompt_tokens": 8,
             "lcp_tokens": 3, "added_tokens": 5, "removed_tokens": 5, "previous_req_id": "p:canon:a",
             "source_frozen_labels": {"glm_tokens": 8, "glm_lcp_with_prev": 7, "uncached_expected": 1}},
        ])
        class FakeRenderer:
            def __init__(self, _tok_dir): self.tokenizer = self
            def render(self, body): return body["messages"][0]["content"]
            def encode(self, text, add_special_tokens=False): return [ord(c) for c in text]
        with patch.dict("sys.modules", {"s1_common": type("M", (), {"Renderer": FakeRenderer})}):
            report = check_dataset(self.root, ".", tok_dir="fake")
        self.assertEqual(report["status"], "INVALID")
        self.assertTrue(any("adapted_original glm_lcp_with_prev=4 rendered=3" in e
                            for e in report["errors"]))
        self.r2.update(glm_lcp_with_prev=3, uncached_expected=5)
        write_jsonl(os.path.join(self.root, "requests.jsonl"), [self.r1, self.r2])
        with patch.dict("sys.modules", {"s1_common": type("M", (), {"Renderer": FakeRenderer})}):
            valid = check_dataset(self.root, ".", tok_dir="fake")
        self.assertEqual(valid["status"], "VALID")
        self.r2["uncached_expected"] = None
        write_jsonl(os.path.join(self.root, "requests.jsonl"), [self.r1, self.r2])
        with patch.dict("sys.modules", {"s1_common": type("M", (), {"Renderer": FakeRenderer})}):
            missing_label = check_dataset(self.root, ".", tok_dir="fake")
        self.assertEqual(missing_label["status"], "INVALID")
        self.assertTrue(any("uncached_expected must be an integer" in e for e in missing_label["errors"]))

    def test_provenance_missing_token_count_is_invalid(self):
        self.r1.update(glm_tokens=8, glm_lcp_with_prev=0, uncached_expected=8)
        self.r2.update(glm_tokens=8, glm_lcp_with_prev=3, uncached_expected=5)
        write_jsonl(os.path.join(self.root, "requests.jsonl"), [self.r1, self.r2])
        self.write_bodies([
            {"req_id": "p:canon:a", "system": "", "messages": [{"role": "user", "content": "abcdefgh"}]},
            {"req_id": "p:canon:b", "system": "", "messages": [{"role": "user", "content": "abcXYZ12"}]},
        ])
        p1 = {"req_id": "p:canon:a", "kind": "original", "prompt_tokens": 8,
              "lcp_tokens": 0, "added_tokens": 8, "removed_tokens": 0, "previous_req_id": None}
        p2 = {"req_id": "p:canon:b", "kind": "original", "prompt_tokens": 8,
              "added_tokens": 5, "removed_tokens": 5, "previous_req_id": "p:canon:a"}
        write_jsonl(os.path.join(self.root, "provenance.jsonl"), [p1, p2])
        class FakeRenderer:
            def __init__(self, _tok_dir): self.tokenizer = self
            def render(self, body): return body["messages"][0]["content"]
            def encode(self, text, add_special_tokens=False): return [ord(c) for c in text]
        with patch.dict("sys.modules", {"s1_common": type("M", (), {"Renderer": FakeRenderer})}):
            report = check_dataset(self.root, ".", tok_dir="fake")
        self.assertEqual(report["status"], "INVALID")
        self.assertTrue(any("provenance lcp_tokens must be an integer" in e for e in report["errors"]))

    def test_explicit_tokenizer_request_with_render_dependency_failure_is_invalid(self):
        class BrokenRenderer:
            def __init__(self, _tok_dir): self.tokenizer = self
            def render(self, _body): raise ImportError("apply_chat_template requires jinja2")
            def encode(self, text, add_special_tokens=False): return []
        with patch.dict("sys.modules", {"s1_common": type("M", (), {"Renderer": BrokenRenderer})}):
            report = check_dataset(self.root, ".", tok_dir="fake")
        self.assertEqual(report["status"], "INVALID")
        self.assertTrue(any("explicitly requested" in e for e in report["errors"]))

    def _synthetic_tool_fixture(self, new_suffix):
        self.r1["glm_tokens"] = 1
        self.r2["glm_tokens"] = 1
        write_jsonl(os.path.join(self.root, "requests.jsonl"), [self.r1, self.r2])
        write_jsonl(os.path.join(self.root, "provenance.jsonl"), [
            {"req_id": "p:canon:a", "kind": "original"},
            {"req_id": "p:canon:b", "kind": "synthetic", "donor_fingerprint": "fp-1",
             "donor_mode": "observed_transition", "donor_same_family": True},
        ])
        self.write_bodies([
            {"req_id": "p:canon:a", "system": "", "messages": [{"role": "user", "content": "seed"}]},
            {"req_id": "p:canon:b", "system": "", "messages": [
                {"role": "user", "content": "seed"}, *new_suffix]},
        ])

    def test_synthetic_new_tool_block_missing_result_is_invalid(self):
        self._synthetic_tool_fixture([
            {"role": "assistant", "tool_calls": [{"id": "new-call", "function": {"name": "x"}}]},
        ])
        report = check_dataset(self.root, ".")
        self.assertEqual(report["status"], "INVALID")
        self.assertTrue(any("calls but 0 results" in e for e in report["errors"]))

    def test_synthetic_count_matched_implicit_tool_result_is_warning(self):
        self._synthetic_tool_fixture([
            {"role": "assistant", "tool_calls": [{"id": "new-call", "function": {"name": "x"}}]},
            {"role": "tool", "content": "result without ID"},
        ])
        report = check_dataset(self.root, ".")
        self.assertEqual(report["status"], "STRUCTURAL_OK")
        self.assertEqual(report["counts"]["synthetic_tool_blocks_implicit_ids"], 1)
        self.assertEqual(report["donor_reuse"]["unique_fingerprints"], 1)
        self.assertEqual(report["donor_reuse"]["same_family_uses"], 1)
        self.assertTrue(any("implicit IDs" in w for w in report["warnings"]))

    def test_manifest_missing_sum_is_reported_unavailable_not_zero(self):
        with open(os.path.join(self.root, "manifest.json"), "w", encoding="utf-8") as fh:
            json.dump({"chain_summaries": [{"source_prompt_sum": 10, "prompt_sum": 11}]}, fh)
        report = check_dataset(self.root, ".")
        self.assertEqual(report["manifest_checks"]["source_prompt_sum"], 10)
        self.assertIsNone(report["manifest_checks"]["source_output_sum"])
        self.assertTrue(any("source_output_sum" in w and "omitted" in w for w in report["warnings"]))

    def test_generated_body_ref_must_match_actual_containing_shard(self):
        with open(os.path.join(self.root, "manifest.json"), "w", encoding="utf-8") as fh:
            json.dump({"generator": "test"}, fh)
        self.r1["body_ref"] = "bodies/part.jsonl.gz"
        self.r2["body_ref"] = "bodies/wrong.jsonl.gz"
        write_jsonl(os.path.join(self.root, "requests.jsonl"), [self.r1, self.r2])
        report = check_dataset(self.root, ".")
        self.assertEqual(report["status"], "INVALID")
        self.assertTrue(any("body_ref='bodies/wrong.jsonl.gz'" in e for e in report["errors"]))
        self.r2.pop("body_ref")
        write_jsonl(os.path.join(self.root, "requests.jsonl"), [self.r1, self.r2])
        report = check_dataset(self.root, ".")
        self.assertEqual(report["status"], "INVALID")
        self.assertTrue(any("body_ref=None" in e for e in report["errors"]))

    def test_generated_budget_gap_and_context_are_strict(self):
        with open(os.path.join(self.root, "manifest.json"), "w", encoding="utf-8") as fh:
            json.dump({"generator": "test", "max_context_tokens": 10000}, fh)
        self.r2["max_output_i"] = 9999
        self.r2["replay_gap_ms"] = -1000
        self.r2["gap_valid"] = True
        write_jsonl(os.path.join(self.root, "requests.jsonl"), [self.r1, self.r2])
        report = check_dataset(self.root, ".")
        self.assertEqual(report["status"], "INVALID")
        self.assertTrue(any("prompt plus output budget exceeds max_context_tokens" in e for e in report["errors"]))
        self.assertTrue(any("finite non-negative" in e for e in report["errors"]))

    def test_generated_cohort_hash_and_artifact_hash_are_verified(self):
        cohort = {"n_chains": 1, "n_requests": 2,
                  "chains": [{"chain_id": "c", "req_ids": ["p:canon:a", "p:canon:b"]}],
                  "cohort_sha256": "stale"}
        with open(os.path.join(self.root, "cohort.json"), "w", encoding="utf-8") as fh:
            json.dump(cohort, fh)
        with open(os.path.join(self.root, "manifest.json"), "w", encoding="utf-8") as fh:
            json.dump({"generator": "test", "max_context_tokens": 10000,
                       "artifacts": {"requests.jsonl": "0"}}, fh)
        report = check_dataset(self.root, ".")
        self.assertTrue(any("cohort_sha256 does not match" in e for e in report["errors"]))
        self.assertTrue(any("artifact SHA256 mismatch: requests.jsonl" in e for e in report["errors"]))

    def test_synthetic_new_user_is_turn_start_even_with_high_token_lcp(self):
        seed = "q" * 100
        self.r1.update(glm_tokens=100, glm_lcp_with_prev=0, uncached_expected=100,
                       phase="session_start")
        self.r2.update(glm_tokens=101, glm_lcp_with_prev=100, uncached_expected=1,
                       phase="turn_start")
        write_jsonl(os.path.join(self.root, "requests.jsonl"), [self.r1, self.r2])
        write_jsonl(os.path.join(self.root, "provenance.jsonl"), [
            {"req_id": "p:canon:a", "kind": "original", "prompt_tokens": 100,
             "lcp_tokens": 0, "added_tokens": 100, "removed_tokens": 0, "previous_req_id": None},
            {"req_id": "p:canon:b", "kind": "synthetic", "prompt_tokens": 101,
             "lcp_tokens": 100, "added_tokens": 1, "removed_tokens": 0, "previous_req_id": "p:canon:a"},
        ])
        self.write_bodies([
            {"req_id": "p:canon:a", "system": "", "messages": [{"role": "user", "content": seed}]},
            {"req_id": "p:canon:b", "system": "", "messages": [
                {"role": "user", "content": seed}, {"role": "user", "content": "?"}]},
        ])
        class FakeRenderer:
            def __init__(self, _tok_dir): self.tokenizer = self
            def render(self, body): return "".join(m.get("content", "") for m in body["messages"])
            def encode(self, text, add_special_tokens=False): return [ord(c) for c in text]
        with patch.dict("sys.modules", {"s1_common": type("M", (), {"Renderer": FakeRenderer})}):
            report = check_dataset(self.root, ".", tok_dir="fake")
        self.assertEqual(report["status"], "VALID")
        self.assertEqual(report["per_request"]["p:canon:b"]["lcp_tokens"], 100)
        self.r2["phase"] = "intra"
        write_jsonl(os.path.join(self.root, "requests.jsonl"), [self.r1, self.r2])
        with patch.dict("sys.modules", {"s1_common": type("M", (), {"Renderer": FakeRenderer})}):
            wrong = check_dataset(self.root, ".", tok_dir="fake")
        self.assertEqual(wrong["status"], "INVALID")
        self.assertTrue(any("message events require 'turn_start'" in e for e in wrong["errors"]))

    def test_synthetic_context_reset_without_new_user_is_invalid(self):
        self.r2["phase"] = "context_reset"
        write_jsonl(os.path.join(self.root, "requests.jsonl"), [self.r1, self.r2])
        write_jsonl(os.path.join(self.root, "provenance.jsonl"), [
            {"req_id": "p:canon:a", "kind": "original"},
            {"req_id": "p:canon:b", "kind": "synthetic"},
        ])
        self.write_bodies([
            {"req_id": "p:canon:a", "system": "", "messages": [{"role": "user", "content": "seed"}]},
            {"req_id": "p:canon:b", "system": "", "messages": [
                {"role": "user", "content": "seed"}, {"role": "assistant", "content": "continuing"}]},
        ])
        report = check_dataset(self.root, ".")
        self.assertEqual(report["status"], "INVALID")
        self.assertTrue(any("message events require 'intra'" in e for e in report["errors"]))

    def test_generated_chain_summary_recomputed_from_visible_rows_and_bodies(self):
        rows = {
            "p:canon:a": {"chain_id": "c", "glm_tokens": 8, "uncached_expected": 8,
                           "max_output_i": 4, "phase": "session_start", "dispatch_offset_ms": 10,
                           "end_offset_ms": 11},
            "p:canon:b": {"chain_id": "c", "glm_tokens": 9, "uncached_expected": 2,
                           "max_output_i": 5, "phase": "intra", "dispatch_offset_ms": 20,
                           "end_offset_ms": 21},
        }
        bodies = {
            "p:canon:a": {"system": "s", "tools": [], "messages": [{"role": "user", "content": "q"}]},
            "p:canon:b": {"system": "s", "tools": [], "messages": [
                {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]},
        }
        chain = {"chain_id": "c", "n_requests": 2, "first_dispatch_offset_ms": 10,
                 "last_end_offset_ms": 21, "sum_glm_tokens": 17, "sum_uncached_expected": 10,
                 "max_output_i_sum": 9, "phases": {"session_start": 1, "intra": 1},
                 "total_edges": 1, "append_only_edges": 0, "append_only_frac": 0.0}
        errors = _chain_summary_errors({"c": chain}, {"c": ["p:canon:a", "p:canon:b"]}, rows, bodies)
        self.assertTrue(any("append_only_edges=0, recomputed=1" in e for e in errors))
        self.assertTrue(any("append_only_frac=0.0, recomputed=1.0" in e for e in errors))

    def test_source_hash_checks_use_parent_provenance_or_report_skipped(self):
        parent = os.path.join(self.root, "parent")
        os.makedirs(parent)
        source_id = "p:canon:a"
        with open(os.path.join(parent, "provenance.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"req_id": source_id, "body_sha256": "a" * 64}) + "\n")
        with open(os.path.join(parent, "requests.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(dict(self.r1, glm_tokens=8, glm_lcp_with_prev=0, uncached_expected=8)) + "\n")
        with open(os.path.join(self.root, "manifest.json"), "w", encoding="utf-8") as fh:
            json.dump({"generator": "test", "parent_root": parent}, fh)
        write_jsonl(os.path.join(self.root, "provenance.jsonl"), [
            {"req_id": source_id, "kind": "adapted_original", "source_req_id": source_id,
             "body_sha256": "b" * 64, "source_body_sha256": "c" * 64,
             "source_frozen_labels": {"glm_tokens": 8, "glm_lcp_with_prev": 0, "uncached_expected": 8}},
        ])
        report = check_dataset(self.root, ".")
        self.assertEqual(report["source_hash_checks"]["verified_against_parent_provenance"], 0)
        self.assertTrue(any("source_body_sha256 differs" in e for e in report["errors"]))

    def test_workload_ledger_separates_frozen_synthetic_and_visible_lcp_work(self):
        grouped = {"c": ["a", "b", "d"]}
        rows = {"a": {"uncached_expected": 20}, "b": {"uncached_expected": 30},
                "d": {"uncached_expected": 10}}
        provenance = {"a": {"kind": "original"},
                      "b": {"kind": "adapted_original", "source_frozen_labels": {"uncached_expected": 30}},
                      "d": {"kind": "synthetic", "added_tokens": 40}}
        per = {"a": {"added_tokens": 10}, "b": {"added_tokens": 11}, "d": {"added_tokens": 12}}
        chain = {"source_chain_targets": {"n_requests": 3, "sum_uncached_expected": 100,
                                          "sum_glm_tokens": 500, "max_output_i_sum": 20},
                 "sum_glm_tokens": 510, "max_output_i_sum": 22}
        ledger, errors = _workload_ledger({"c": chain}, grouped, rows, provenance, per)
        self.assertEqual(errors, [])
        row = ledger["rows"][0]
        self.assertEqual(row["missing_source_frozen_uncached"], 50)
        self.assertEqual(row["generated_synthetic_lcp_added"], 40)
        self.assertEqual(row["missing_budget_delta"], -10)
        self.assertEqual(row["generated_frozen_uncached"], 90)
        self.assertEqual(row["generated_cohort_visible_lcp_added"], 33)


if __name__ == "__main__":
    unittest.main()
