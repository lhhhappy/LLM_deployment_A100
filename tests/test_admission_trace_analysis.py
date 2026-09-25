import importlib.util
import json
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location(
    "admission_analysis", Path(__file__).resolve().parents[1] / "scripts/analysis/admission_trace.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def record(event, rid="a", counts=None):
    row = {"rid": rid, "observed_wait_s": 1.0, "decisions": counts or {"partial_host_restore": 2}}
    obj = {"event": "admit", **row} if event == "admit" else {
        "event": "waiting", "queue_size": 1, "sample": [row],
    }
    return module.MARKER + json.dumps(obj)


class AdmissionAnalysisTests(unittest.TestCase):
    def test_cumulative_snapshots_not_double_counted(self):
        result = module.analyze([record("waiting"), record("waiting"), record("admit")])
        self.assertEqual(result["admitted_decision_counts"], {"partial_host_restore": 2})
        self.assertIsNone(result["rows"][0]["last_pending_decisions"])

    def test_retraction_episodes_and_pending_kept_separate(self):
        result = module.analyze([record("admit"), record("admit"), record("waiting")])
        row = result["rows"][0]
        self.assertEqual(row["admission_episodes"], 2)
        self.assertEqual(row["admitted_decisions"], {"partial_host_restore": 4})
        self.assertEqual(row["last_pending_decisions"], {"partial_host_restore": 2})

    def test_missing_diagnostics_is_unknown(self):
        result = module.analyze([record("admit")], [{"req_id": "b", "ttft_s": 4}])
        row = next(r for r in result["rows"] if r["req_id"] == "b")
        self.assertFalse(row["diagnostic_present"])
        self.assertIsNone(row["admitted_decisions"])

    def test_invalid_or_missing_input_fails_closed(self):
        for lines in ([], [record("admit", counts={"kv_budget": -1})],
                      [record("admit", counts={"kv_budget": True})],
                      [module.MARKER + "{broken"]):
            with self.assertRaises(ValueError):
                module.analyze(lines)
        with self.assertRaises(ValueError):
            module.analyze([record("admit")], [{"req_id": "a"}, {"req_id": "a"}])

    def test_byte_cap_is_explicit_partial_observation(self):
        result = module.analyze([module.MARKER + json.dumps({"event": "budget_exhausted", "max_bytes": 512})])
        self.assertTrue(result["budget_exhausted"])
        self.assertEqual(result["admission_events"], 0)
        self.assertEqual(result["rows"], [])


if __name__ == "__main__":
    unittest.main()
