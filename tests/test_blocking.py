"""Regression tests for truthful window accounting, malformed data and model bounds."""
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("blocking", ROOT / "scripts/analysis/blocking.py")
blocking = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(blocking)
COMMON = blocking.load_common(ROOT / "s1-dev/harness")


def row(rid="a", idx=0, recv=100, start=102, first=104):
    return dict(req_id=rid, chain_id="chain", idx_in_chain=idx,
                phase="chain_start" if idx == 0 else "intra",
                edge_type="start" if idx == 0 else "append-only",
                uncached_expected=100, max_output_i=10, prompt_tokens=100,
                cached_tokens=0, output_tokens=10, error=None,
                t_recv_s=recv, t_exec_start_s=start, t_first_token_s=first,
                ttft_s=first-recv, tpot_s=.2,
                client_dispatch_at_s=recv, client_first_token_at_s=first,
                client_finish_at_s=first+1.8)


def expected(rows):
    result = {}
    for item in rows:
        source = dict(item)
        source["_idx_in_chain"] = item["idx_in_chain"]
        source["glm_tokens"] = item["prompt_tokens"]
        result[item["req_id"]] = source
    return result


class BlockingTest(unittest.TestCase):
    def test_equal_sized_wrong_cohort_rejected(self):
        real = [row()]
        blocking.validate_rows(real, expected(real))
        wrong = [dict(real[0], req_id="foreign")]
        with self.assertRaisesRegex(ValueError, "cohort mismatch"):
            blocking.validate_rows(wrong, expected(real))

    def test_duplicate_and_frozen_bucket_mismatch_rejected(self):
        real = [row()]
        with self.assertRaisesRegex(ValueError, "duplicates"):
            blocking.validate_rows(real * 2, expected(real))
        wrong = [dict(real[0], phase="intra")]
        with self.assertRaisesRegex(ValueError, "frozen phase"):
            blocking.validate_rows(wrong, expected(real))

    def test_bad_timing_nan_and_cache_are_rejected(self):
        real = [row()]
        for changes in ({"t_exec_start_s": 99}, {"t_recv_s": float("nan")},
                        {"cached_tokens": 101}, {"client_finish_at_s": 90},
                        {"tpot_s": float("inf")}, {"ttft_s": 99}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                blocking.validate_rows([dict(real[0], **changes)], expected(real))

    def test_overlap_union_is_not_sum_and_excludes_self(self):
        intervals = [(0, (0, 100)), (1, (1, 6)), (2, (4, 9))]
        union, share = blocking.attribute((0, 10), intervals, exclude=0)
        self.assertEqual(union, 8)
        self.assertEqual(share, {1: 4, 2: 4})
        self.assertEqual(sum(share.values()), union)
        self.assertEqual(blocking.attribute((10, 10), intervals, 0), (0, {}))

    def test_cost_overestimate_is_preserved_and_missing_lcp_unknown(self):
        items, summary = blocking.analyze([row()], COMMON, fixed_s=10)
        self.assertEqual(items[0]["exec_window_s"], 2)
        self.assertGreater(items[0]["prefill_cost_est_s"], 10)
        self.assertLess(items[0]["model_residual_s"], -8)
        self.assertIsNone(items[0]["cache_gap_tokens"])
        self.assertIsNone(items[0]["cache_gap_variable_cost_est_s"])
        self.assertEqual(summary["model_exceeds_exec_window_n"], 1)
        known, _ = blocking.analyze([row()], COMMON, lcp={"a": 0})
        self.assertEqual(known[0]["cache_gap_tokens"], 0)

    def test_cost_scenario_does_not_change_observed_overlap(self):
        rows = [row(), row("b", recv=99, start=101, first=107)]
        left, _ = blocking.analyze(rows, COMMON, chunk=1, fixed_s=10, server_minus_client_s=0)
        right, _ = blocking.analyze(rows, COMMON, chunk=16384, fixed_s=0, server_minus_client_s=0)
        for key in ("queue_s", "exec_window_s", "queue_prefill_window_overlap_s",
                    "decode_prefill_window_overlap_s", "ttft_over", "tpot_over"):
            self.assertEqual(left[0][key], right[0][key])

    def test_decode_overlap_requires_explicit_clock_and_measured_duration(self):
        rows = [row(), row("b", recv=99, start=103, first=108)]
        # Client clock is 1000 seconds behind the server.
        for item in rows:
            for key in ("client_dispatch_at_s", "client_first_token_at_s", "client_finish_at_s"):
                item[key] -= 1000
        absent, _ = blocking.analyze(rows, COMMON)
        self.assertIsNone(absent[0]["decode_prefill_window_overlap_s"])
        shifted, _ = blocking.analyze(rows, COMMON, server_minus_client_s=1000)
        self.assertAlmostEqual(shifted[0]["decode_overlap_fraction"], 1)
        # TPOT fields determine threshold membership, not the overlap denominator.
        changed = copy.deepcopy(rows)
        changed[0]["tpot_s"] = 100
        result, _ = blocking.analyze(changed, COMMON, server_minus_client_s=1000)
        self.assertAlmostEqual(result[0]["decode_overlap_fraction"], 1)
        self.assertEqual(result[0]["decode_overlap_candidates"][0]["req_id"], "b")

    def test_lcp_pairs_validate_predecessor_and_missing_is_not_zero(self):
        rows = [row(), row("b", idx=1, recv=110, start=111, first=112)]
        pair = dict(req_id="b", chain_id="chain", idx=1, prompt=100, previous_prompt=100, true_lcp=64)
        self.assertEqual(blocking.load_lcp([pair], rows), {"b": 64})
        self.assertEqual(blocking.load_lcp(None, rows), {})
        for changes in ({"previous_prompt": 90}, {"true_lcp": 101}, {"req_id": "a"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                blocking.load_lcp([dict(pair, **changes)], rows)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            blocking.load_lcp([pair, pair], rows)


if __name__ == "__main__":
    unittest.main()
