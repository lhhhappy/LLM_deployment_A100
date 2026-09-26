#!/usr/bin/env python3
"""Paired local CPU measurements of the review's family/metadata optimizations.

These measure production Python helpers with synthetic requests/shape objects.
They do not estimate GPU, end-to-end TPOT, or official SLO improvements.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from test_ax_admission_scheduler import cold
from test_sched_protect_chain import make_scheduler, tree_dir
from test_s1s2_review_optimizations import ENV, humming_metadata_fixture


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    baseline = tree_dir("84dcca0e")
    candidate = ROOT / "engine/sglang"
    result = {"scope": "local CPU helpers; synthetic requests/shape objects, no GPU or SLO claim",
              "baseline": "84dcca0ed84f949cf44acd0a5d2427b171d317a2",
              "family": [], "humming_metadata": {}}
    clean_env = {k: v for k, v in os.environ.items() if not k.startswith("SGLANG_AX_")}
    clean_env.update(ENV, SGLANG_AX_DEADLINE_FAMILY="1")
    with patch.dict(os.environ, clean_env, clear=True):
        for count in (8, 30, 64):
            observations = {"base": [], "candidate": []}
            for iteration in range(5):
                arms = [("base", baseline), ("candidate", candidate)]
                if iteration % 2:
                    arms.reverse()
                outcomes = {}
                for name, source in arms:
                    reqs = [cold(f"r{i:02d}", 250000) for i in range(count)]
                    for i, req in enumerate(reqs):
                        req.origin_input_ids[-1000:] = [i + 2] * 1000
                    sched, _ = make_scheduler(root=source, waiting=reqs)
                    deadline, _ = sched._ax_admission_cfgs()
                    start = time.perf_counter_ns()
                    outcomes[name] = sched._ax_family_plan(sched._ax_family_cfg, deadline)
                    first = (time.perf_counter_ns() - start) / 1e6
                    start = time.perf_counter_ns()
                    for _ in range(20):
                        sched._ax_family_plan(sched._ax_family_cfg, deadline)
                    warm = (time.perf_counter_ns() - start) / 20e6
                    observations[name].append({"first_ms": first, "cached_ms": warm})
                assert outcomes["base"] == outcomes["candidate"]
            summary = {name: {metric: statistics.median(v[metric] for v in values)
                              for metric in ("first_ms", "cached_ms")}
                       for name, values in observations.items()}
            result["family"].append({"requests": count, "tokens_each": 250000,
                                     "shared_tokens": 249000, "samples": observations,
                                     "median": summary})
    base, cached, tensor, gemm, _ = humming_metadata_fixture()
    x, ids = tensor(8192, 4096), tensor(8192, 9)
    baseline_value = base.get_buffer_metas(x, ids, gemm.INDEXED)
    assert cached.get_buffer_metas(x, ids, gemm.INDEXED) == baseline_value
    samples = {"base": [], "candidate": []}
    for iteration in range(7):
        arms = [("base", base), ("candidate", cached)]
        if iteration % 2:
            arms.reverse()
        for name, runner in arms:
            start = time.perf_counter_ns()
            for _ in range(20000):
                # Base prepare_buffers calls get_buffer_metas twice per forward.
                runner.get_buffer_metas(x, ids, gemm.INDEXED)
                runner.get_buffer_metas(x, ids, gemm.INDEXED)
            samples[name].append((time.perf_counter_ns() - start) / 20000e3)
    result["humming_metadata"] = {"unit": "microseconds per pair of metadata calls",
                                   "samples": samples,
                                   "median": {k: statistics.median(v) for k, v in samples.items()}}
    result["source_sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [Path(__file__), ROOT / "tests/test_s1s2_review_optimizations.py",
                     candidate / "srt/managers/ax_deadline.py",
                     candidate / "srt/managers/scheduler.py",
                     candidate / "srt/layers/quantization/fp8_humming_moe.py"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"family": [v["median"] for v in result["family"]],
                      "humming_metadata": result["humming_metadata"]["median"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
