#!/usr/bin/env python3
"""Local CPU collector timings. Not an engine/TP8 overhead measurement.

Actual AxAdmissionTrace source; small request fixtures and ordinary FileHandler.
No torch, Pod access, GPU calls, prompts, or permanent benchmark logs.
"""
import argparse
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import platform
import statistics
import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'engine/sglang/srt/managers/ax_admission_trace.py'
spec = importlib.util.spec_from_file_location('collector', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Req:
    def __init__(self, index):
        self.rid = 'scimaster:canon:lc_diagnostic_fixture:llm:' + str(index).zfill(4)
        self.time_stats = SimpleNamespace(wait_queue_entry_time=time.perf_counter())
        self.prefix_indices = range(32768)
        self.host_hit_length = 8192
        self.mamba_host_hit_length = 8192
        self.seqlen = 40960 + 801

    def needs_host_load_back(self):
        return True


def measure(fn, count, rounds=7):
    samples = []
    for _ in range(rounds):
        wall, cpu = time.perf_counter_ns(), time.thread_time_ns()
        for _ in range(count):
            fn()
        samples.append(dict(wall_ns_per_call=(time.perf_counter_ns()-wall)/count,
                            cpu_ns_per_call=(time.thread_time_ns()-cpu)/count))
    return dict(calls_per_round=count, rounds=rounds,
                wall_us_median=statistics.median(s['wall_ns_per_call'] for s in samples)/1000,
                wall_us_round_max=max(s['wall_ns_per_call'] for s in samples)/1000,
                cpu_us_median=statistics.median(s['cpu_ns_per_call'] for s in samples)/1000,
                samples=samples)


def run(out):
    logger = logging.Logger('admission-bench', level=logging.INFO)
    results = {}
    with tempfile.TemporaryDirectory(prefix='admission-bench-') as directory:
        path = Path(directory) / 'trace.log'
        handler = logging.FileHandler(path, encoding='utf-8')
        handler.setFormatter(logging.Formatter('[%(asctime)s TP0] %(message)s'))
        logger.addHandler(handler)
        try:
            def noop():
                pass
            results['empty_python_call'] = measure(noop, 10000)
            for n in (1, 8, 30, 38):
                requests = [Req(i) for i in range(n)]
                trace = module.AxAdmissionTrace(logger)
                trace.record_many(requests, 'decode_cadence')
                trace.snapshot(requests)  # one initial snapshot, then rate-limited
                def record():
                    trace.record_many(requests, 'decode_cadence')
                    trace.snapshot(requests)
                results[f'hot_record_and_gated_snapshot_n{n}'] = measure(record, 5000)
                assert not trace.exhausted

            requests = [Req(i) for i in range(38)]
            trace = module.AxAdmissionTrace(logger)
            context = {'partial_rid': 'scimaster:canon:long_request:llm:0001'}
            def rejected_attempt():
                req = requests[0]
                trace.begin_attempt(req)
                trace.record(req, 'partial_host_restore', context)
                trace.rejected(req, 'adder_other')
            rejected_attempt()
            results['hot_one_rejection'] = measure(rejected_attempt, 10000)

            trace = module.AxAdmissionTrace(logger)
            def admission():
                req = requests[0]
                trace.record(req, 'partial_host_restore', context)
                trace.record(req, 'decode_cadence')
                trace.admitted(req)
            results['admission_json_and_file_flush'] = measure(admission, 300)
            assert not trace.exhausted, 'admission sample hit budget; timing invalid'
            results['charged_bytes_per_admission'] = trace.bytes_emitted/(300*7)

            trace = module.AxAdmissionTrace(logger, snapshot_interval=0)
            for req in requests:
                trace.record(req, 'partial_host_restore', context)
                trace.record(req, 'decode_cadence')
            results['snapshot_32_of_38_json_and_file_flush'] = measure(lambda:trace.snapshot(requests), 12)
            assert not trace.exhausted, 'snapshot sample hit budget; timing invalid'

            trace.exhausted = True
            results['exhausted_record_and_snapshot_n38'] = measure(
                lambda:(trace.record_many(requests, 'decode_cadence'),trace.snapshot(requests)), 10000)
            handler.flush()
            log_bytes = path.stat().st_size
            assert log_bytes < 8*1024*1024, 'local temporary log budget exceeded'
        finally:
            handler.close()
            logger.removeHandler(handler)
    model = next((line.partition(':')[2].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                  if line.startswith('model name')), 'unknown')
    receipt = dict(scope='local CPU collector microbenchmark; not TP8 or end-to-end overhead',
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        measured_at=time.time(), python=platform.python_version(), cpu=model,
        affinity_cpus=len(os.sched_getaffinity(0)), results=results, temporary_log_bytes=log_bytes,
        caveats=['No real request/cache/pool code, TP collective, GPU or Python scheduler integration.',
                 'FileHandler writes to this host temporary filesystem; Pod RAM log latency may differ.',
                 'Median/max describe repeated loop averages, not individual-call tail latency.',
                 'TP0 overhead can delay other ranks; these timings do not measure that amplification.',
                 'Byte budget is per process; exceeding it intentionally ends diagnostic coverage.'])
    body=json.dumps(receipt,indent=2)+'\n'
    assert len(body.encode())<32768
    out.write_text(body)
    print(json.dumps({k:v for k,v in results.items() if not isinstance(v,dict)}))
    for key,result in results.items():
        if isinstance(result,dict):
            print(f'{key}: median {result["wall_us_median"]:.3f} us; round max {result["wall_us_round_max"]:.3f} us')
    print(f'temporary log {log_bytes} bytes removed; result {out}')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path(__file__).with_name('collector-cost.json'))
    run(parser.parse_args().out)
