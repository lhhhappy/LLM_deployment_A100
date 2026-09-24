#!/usr/bin/env python3
"""scripts/analysis/compare_runs.py: same-source invariance and fail-closed counterexamples on real 047 data (CPU).

Needs evidence/L047-official_a_n22/N22 (raw, run, level_verdict, server.log) and evidence/T56/pairs_rendered.json.
Run: python3 -m unittest discover -s tests -p test_compare_runs.py
"""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "scripts/analysis/compare_runs.py"
SRC = ROOT / "evidence/L047-official_a_n22/N22"


def run(base, cand, *extra):
    p = subprocess.run([sys.executable, str(TOOL), str(base), str(cand), *extra], capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


class CompareRuns(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.raw_name = next(SRC.glob("raw_*.jsonl")).name
        self.run_name = next(SRC.glob("run_*.json")).name

    def copy(self, name):
        d = self.tmp / name
        d.mkdir()
        for f in (self.raw_name, self.run_name, "level_verdict.json", "server.log"):
            shutil.copy(SRC / f, d / f)
        return d

    def rows(self, d):
        return [json.loads(l) for l in (d / self.raw_name).read_text().splitlines() if l.strip()]

    def write(self, d, rows):
        (d / self.raw_name).write_text("".join(json.dumps(r) + "\n" for r in rows))

    def test_self_compare_is_invariant(self):
        rc, out = run(SRC, SRC)
        self.assertEqual(rc, 0, out)
        for gate in ("fast_intra", "overall_intra", "turn_start", "chain_start"):
            line = next(l for l in out.splitlines() if l.strip().startswith(gate + " ("))
            self.assertIn("fixed 0 {}", line)
            self.assertIn("new 0 {}", line)
        self.assertIn("metadata-checked 411, rejected 0, no pair (unknown) 311 of 722", out)
        self.assertIn("before/after the window base 919 cand 919", out)
        # the last measured second (partly after the last first token) is an edge second, not interior
        self.assertIn("in boundary seconds base 1 cand 1", out)

    def test_missing_row_on_both_sides_is_invalid(self):
        b, c = self.copy("b"), self.copy("c")
        for d in (b, c):
            self.write(d, self.rows(d)[1:])
        rc, out = run(b, c)
        self.assertNotEqual(rc, 0)
        self.assertIn("INVALID", out)
        self.assertIn("1 missing", out)

    def test_duplicate_row_is_invalid(self):
        c = self.copy("c")
        rows = self.rows(c)
        self.write(c, rows + rows[:1])
        rc, out = run(SRC, c)
        self.assertIn("INVALID: duplicate", out)

    def test_invalid_verdict_is_invalid(self):
        c = self.copy("c")
        v = json.loads((c / "level_verdict.json").read_text())
        v["status"] = "INVALID"
        (c / "level_verdict.json").write_text(json.dumps(v))
        rc, out = run(SRC, c)
        self.assertIn("not VALID", out)

    def test_other_workload_is_invalid(self):
        c = self.copy("c")
        r = json.loads((c / self.run_name).read_text())
        r["config"]["workload_hash"] = "0000000000000000"
        (c / self.run_name).write_text(json.dumps(r))
        rc, out = run(SRC, c)
        self.assertIn("workload identity differs", out)

    def test_request_metadata_must_match(self):
        c = self.copy("c")
        rows = self.rows(c)
        rows[5]["max_output_i"] += 1
        self.write(c, rows)
        rc, out = run(SRC, c)
        self.assertIn("differ in replay metadata", out)

    def test_gate_is_judged_at_raw_precision(self):
        c = self.copy("c")
        rows = self.rows(c)
        sys.path.insert(0, str(ROOT / "s1-dev/harness"))
        from s1_common import in_ttft_gate
        i = next(k for k, r in enumerate(rows) if in_ttft_gate(r, "fast_intra") and r["ttft_s"] < 3)
        rows[i]["ttft_s"] = 3.0004  # rounds to 3.000 but is over the 3 s limit
        self.write(c, rows)
        rc, out = run(SRC, c)
        line = next(l for l in out.splitlines() if l.strip().startswith("fast_intra ("))
        self.assertIn("new 1 ", line)

    def test_tpot_is_judged_at_raw_precision(self):
        c = self.copy("c")
        rows = self.rows(c)
        base_over = sum(r["tpot_s"] > 0.10 for r in rows if r.get("tpot_s") is not None)
        i = next(k for k, r in enumerate(rows) if r.get("tpot_s") is not None and r["tpot_s"] < 0.10)
        rows[i]["tpot_s"] = 0.10004
        self.write(c, rows)
        rc, out = run(SRC, c)
        self.assertIn(f"tpot per request > 0.10: {base_over} -> {base_over + 1}", out)

    def test_pair_with_wrong_predecessor_is_rejected(self):
        b, c = self.copy("b"), self.copy("c")
        pairs = json.loads((ROOT / "evidence/T56/pairs_rendered.json").read_text())
        pairs[0]["previous_prompt"] += 1
        pf = self.tmp / "pairs.json"
        pf.write_text(json.dumps(pairs))
        rc, out = run(b, c, "--pairs", str(pf))
        self.assertIn("metadata-checked 410, rejected 1", out)


if __name__ == "__main__":
    unittest.main()
