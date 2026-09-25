#!/usr/bin/env python3
"""Original preflight/warmup/strict flush, followed by a timed diagnostic replay.

Never replaces complete-data validation with a partial-cohort PASS. The runner
exits naturally after admission closes and all outstanding requests have ended.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

from run_dev_checked import strict_flush
from timed_loadgen import write_json


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runner', type=Path, required=True)
    ap.add_argument('--seconds', type=float, default=4200)
    ap.add_argument('--warmup-profile', choices=['original', 'rep16-v1'], default='original')
    args, rest = ap.parse_known_args(argv)
    if rest[:1] == ['--']: rest = rest[1:]
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--n', type=int, required=True)
    opts, _ = p.parse_known_args(rest)
    opts.out.mkdir(parents=True, exist_ok=True)
    evidence = dict(schema_version=1, runner_started_s=time.time(), n=opts.n,
                    flush_success=False, runner_rc=None, scope='fixed_duration_diagnostic',
                    warmup_profile=args.warmup_profile)
    receipt_path = opts.out/'flush_evidence.json'
    write_json(receipt_path, evidence)
    spec = importlib.util.spec_from_file_location('timed_original_runner', args.runner)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_run = module._run
    def run(command, log, env):
        if args.warmup_profile == 'rep16-v1' and Path(command[1]).name == 's1_loadgen.py' and '--warmup' in command:
            command = [command[0], str(Path(__file__).with_name('short_warmup_loadgen.py')),
                       '--loadgen', command[1], '--', *command[2:]]
        if Path(command[1]).name == 's1_loadgen.py' and '--instance-id' in command:
            instance = command[command.index('--instance-id')+1]
            if instance.endswith('-measure'):
                command = [command[0], str(Path(__file__).with_name('timed_loadgen.py')),
                           '--loadgen', command[1], '--seconds', str(args.seconds), '--', *command[2:]]
        return original_run(command, log, env)
    module._run = run
    def flush(url):
        if args.warmup_profile == 'rep16-v1':
            warm = json.loads((opts.out/'short_warmup_receipt.json').read_text())
            if warm.get('status') != 'COMPLETE' or warm.get('started_s', 0) < evidence['runner_started_s']:
                raise ValueError('missing successful short warmup from this invocation')
            evidence['short_warmup'] = warm
        return strict_flush(url, evidence, receipt_path)
    module.flush_kv = flush
    rc = 2
    try:
        rc = int(module.main(rest) or 0)
    finally:
        evidence.update(runner_rc=rc, runner_finished_s=time.time())
        summary = opts.out/'summary.json'
        if summary.exists():
            s = json.loads(summary.read_text())
            evidence.update({k: Path(s.get(k) or '').name for k in ('raw', 'run')})
            s.update(scope='fixed_duration_diagnostic', full_cohort_complete=False,
                     warmup_profile=args.warmup_profile)
            # The original report remains for audit; its partial-cohort flags
            # must not be displayed as the overall experiment verdict.
            s.pop('ALL_PASS', None)
            s.pop('evaluation_status', None)
            write_json(summary, s)
        write_json(receipt_path, evidence)
    return rc


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    sys.exit(main())
