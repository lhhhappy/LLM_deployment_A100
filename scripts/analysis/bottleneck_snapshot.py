#!/usr/bin/env python3
"""One-line bottleneck reading of a running pod job from its sampled server metrics (read-only, via pread tail).

Usage: bottleneck_snapshot.py <job-name> [--lines N]
Reads the last N samples of /tmp/ax/runs/<job>/N<level>/metrics.jsonl (level parsed from the job name, e.g. _n30_),
each a JSON line of /metrics gauges written by build/verify_kit/metrics_sampler.py, and prints:
  KV pool used/evictable share, KDA slots used, running and queued requests, decode throughput, MTP acceptance,
  host tier fill, load-back and eviction deltas over the sampled span, retractions,
followed by a reading of what binds now:
  KV wall      used >= 90% of the pool, little evictable, and requests queued: admission waits for KV (capacity);
  lane-bound   requests queued while KV is available: the single chunked-prefill lane / ordering decides the waits;
  cap-bound    running at the --max-running-requests cap with a queue: the concurrency cap, not memory;
  decode-only  no queue: whatever misses appear come from execution time, not from waiting.
The thresholds only label the snapshot; the numbers are printed so the reader can disagree. Not a verdict.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def pread_tail(path, n):
    out = subprocess.run(['bash', str(ROOT / 'scripts/pod/pread'), 'tail', path, str(n)], capture_output=True, text=True, timeout=120).stdout
    rows = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith('{'):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def share(used, avail, evict):
    pool = (used or 0) + (avail or 0) + (evict or 0)
    return (used or 0) / pool if pool else 0.0, (evict or 0) / pool if pool else 0.0, pool


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('job')
    ap.add_argument('--lines', type=int, default=12)
    ap.add_argument('--max-running', type=int, default=None, help='the job\'s --max-running-requests (default: parse 48 if the name has _48_, else 32)')
    a = ap.parse_args()
    m = re.search(r'_n(\d+)_', a.job)
    level = m.group(1) if m else '30'
    cap = a.max_running or (48 if '_48_' in a.job or 'n34' in a.job else 32)
    rows = pread_tail(f'/tmp/ax/runs/{a.job}/N{level}/metrics.jsonl', a.lines)
    if not rows:
        print(f'{a.job}: no metrics samples yet')
        return
    last, first = rows[-1], rows[0]
    kv_used, kv_evict, pool = share(last.get('kv_used_tokens'), last.get('kv_available_tokens'), last.get('kv_evictable_tokens'))
    kda_used, _, kda_pool = share(last.get('mamba_used_tokens'), last.get('mamba_available_tokens'), last.get('mamba_evictable_tokens'))
    running, queued = int(last.get('num_running_reqs') or 0), int(last.get('num_queue_reqs') or 0)
    span = max(1.0, (last.get('t', 0) - first.get('t', 0)))
    lb = (last.get('load_back_tokens_total') or 0) - (first.get('load_back_tokens_total') or 0)
    ev = (last.get('evicted_tokens_total') or 0) - (first.get('evicted_tokens_total') or 0)
    host = last.get('hicache_host_used_tokens') or 0
    host_tot = last.get('hicache_host_total_tokens') or 0
    avg_running = sum(int(r.get('num_running_reqs') or 0) for r in rows) / len(rows)
    avg_queued = sum(int(r.get('num_queue_reqs') or 0) for r in rows) / len(rows)
    max_kv = max(share(r.get('kv_used_tokens'), r.get('kv_available_tokens'), r.get('kv_evictable_tokens'))[0] for r in rows)
    if avg_queued >= 0.5 and max_kv >= 0.90 and kv_evict <= 0.05:
        reading = 'KV wall (admission waits for pool space)'
    elif avg_queued >= 0.5 and running >= cap:
        reading = f'cap-bound (running at the {cap} cap with a queue)'
    elif avg_queued >= 0.5:
        reading = 'lane-bound (queue with KV available: prefill lane / ordering decides the waits)'
    else:
        reading = 'decode-only (no queue in the sampled span)'
    print(f'{a.job}: last {len(rows)} samples over {span:.0f}s | KV used {kv_used:.0%} (peak {max_kv:.0%}), evictable {kv_evict:.0%} of {pool/1e6:.2f}M | '
          f'KDA slots {kda_used:.0%} of {kda_pool:.0f} | running {running} (avg {avg_running:.1f}), queued {queued} (avg {avg_queued:.1f}) | '
          f'decode {last.get("gen_throughput", 0):.0f} tok/s, MTP accept {last.get("spec_accept_length", 0):.2f} | host {host/1e6:.2f}/{host_tot/1e6:.2f}M | '
          f'load-back +{lb/1e3:.0f}k tok, evicted +{ev/1e3:.0f}k tok, retracted {int(last.get("num_retracted_reqs") or 0)} | reading: {reading}')


if __name__ == '__main__':
    main()
