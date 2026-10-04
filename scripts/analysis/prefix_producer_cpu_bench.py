#!/usr/bin/env python3
"""Bounded CPU planning cost, synthetic native array('q') inputs; no GPU/SLO claim."""
import argparse
from array import array
import gc
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests'))
from test_prefix_producer import ENV, request, scheduler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = []
    for count in (34, 64):
        samples = []
        for _ in range(3):
            with patch.dict(os.environ, ENV):
                reqs = [request(str(i), shared=200000, tail=1024) for i in range(count)]
                for req in reqs:
                    req.origin_input_ids = array('q', req.origin_input_ids)
                s, _, _ = scheduler(reqs)
                s.policy.calc_priority(reqs, s.running_batch)
                deadline, _ = s._ax_admission_cfgs()
                tracker = s._ax_prefix_tracker
                inputs = (reqs, None, [], set(), {}, time.perf_counter(), lambda r: 0,
                          deadline, 8192, 256, 4096, lambda r: True)
                start = time.perf_counter()
                tracker.prepare(*inputs)
                cold = time.perf_counter() - start
                start = time.perf_counter()
                for _ in range(30):
                    tracker.prepare(*inputs)
                warm = (time.perf_counter() - start) / 30
                samples.append(dict(first_ms=cold * 1000, steady_ms=warm * 1000,
                    prompt_bytes=sum(len(r.raw) for r in tracker.records.values()),
                    pairs=len(tracker.pairs)))
            del s, tracker, reqs
            gc.collect()
        results.append(dict(candidates=count, samples=samples,
            first_median_ms=statistics.median(x['first_ms'] for x in samples),
            steady_median_ms=statistics.median(x['steady_ms'] for x in samples)))
    result = dict(validity='CPU_DIAGNOSTIC', python=platform.python_version(),
                  tokens_per_request=201024, representation="array('q')", results=results)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
