"""CPU regressions for profiler accounting; no engine or GPU required."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "prof_ledger", Path(__file__).resolve().parents[1] / "scripts/pod/verify/prof_ledger.py"
)
ledger = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ledger)


def event(cat, name, start, duration, external_id, stream=1):
    return dict(cat=cat, name=name, ts=start, dur=duration, ph="X", pid=0,
                tid=stream, args={"External id": external_id})


class ProfLedgerTest(unittest.TestCase):
    def test_multistream_graph_counts_one_forward(self):
        extend, decode = "step[EXTEND bs=1 toks=1024]", "step[DECODE bs=1]"
        events = [event("gpu_user_annotation", extend, 0, 100, 10),
                  event("gpu_user_annotation", decode, 110, 20, 20),
                  event("gpu_user_annotation", decode, 112, 5, 20, stream=2),
                  event("gpu_user_annotation", decode, 120, 5, 20, stream=3),
                  event("user_annotation", decode, 109, 4, 20),
                  event("kernel", "marlin_moe", 10, 30, 30),
                  event("kernel", "mhc_post", 20, 30, 31, stream=2),
                  event("kernel", "marlin_moe", 110, 20, 32)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.json"
            path.write_text(json.dumps({"traceEvents": events}))
            result = ledger.analyze(str(path))
        self.assertNotIn("error", result)
        self.assertEqual(result["logical_step_count"], 2)
        self.assertEqual(result["gpu_annotation_count"], 4)
        self.assertEqual(result["classes"]["decode"]["n"], 1)
        self.assertAlmostEqual(result["classes"]["decode"]["mean_gpu_span_ms"], .020)
        self.assertAlmostEqual(result["classes"]["decode"]["mean_cpu_span_ms"], .004)
        self.assertAlmostEqual(result["classes"]["extend<2k"]["mean_kernel_busy_ms"], .040)
        self.assertAlmostEqual(result["timeline"]["outside_steps"], 10 / 130)
        self.assertAlmostEqual(sum(result["timeline"].values()), 1)

    def test_repeated_names_are_distinct_steps(self):
        events = [event("gpu_user_annotation", "step[DECODE bs=1]", 0, 10, 10),
                  event("gpu_user_annotation", "step[DECODE bs=1]", 20, 10, 20)]
        self.assertEqual(len(ledger.logical_gpu_steps(events)), 2)

    def test_ambiguous_overlap_is_rejected(self):
        events = [event("gpu_user_annotation", "step[DECODE bs=1]", 0, 10, 10),
                  event("gpu_user_annotation", "step[DECODE bs=1]", 5, 10, 20)]
        with self.assertRaisesRegex(ValueError, "overlap"):
            ledger.logical_gpu_steps(events)

    def test_component_names_precede_framework_and_gemm(self):
        self.assertEqual(ledger.cat_of("mhc_pre_gemm_sqrsum"), "mhc_norm")
        self.assertEqual(ledger.cat_of("tilelang_chunk_kda"), "kda")
        self.assertEqual(ledger.cat_of("tilelang_main_kernel"), "other")
        self.assertEqual(ledger.cat_of("marlin_moe"), "moe")

    def test_extend_gap_excludes_a_kernel_outside_both_steps(self):
        extend = "step[EXTEND bs=1 toks=8192]"
        steps = [(0, 10_000, extend), (20_000, 30_000, extend)]
        kernels = [(12_000, 14_000, "copy_kernel")]
        gap = ledger.consecutive_extend_gaps(steps, kernels)
        self.assertEqual(gap["n"], 1)
        self.assertEqual(gap["mean_annotated_gap_ms"], 10.0)
        self.assertEqual(gap["mean_no_kernel_gap_ms"], 8.0)


if __name__ == "__main__":
    unittest.main()
