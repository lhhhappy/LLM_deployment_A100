#!/usr/bin/env python3
"""Diagnostic admission window around the untouched organizer loadgen.

Only drive's pre-dispatch check changes. Original rendering, gaps, chain slots,
engine calls, token budgets, timestamps and raw records are preserved. A deadline
stops new admissions and gap waits, then joins every outstanding request normally.
The dispatch ledger is the census; completed raw alone is not one.
"""
import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys
import threading
import time


def write_json(path, data):
    tmp = path.with_suffix(path.suffix+'.part')
    tmp.write_text(json.dumps(data, indent=2)+'\n')
    tmp.replace(path)


class AdmissionWindow:
    def __init__(self, seconds, out, clock=time, timer_factory=threading.Timer):
        if not math.isfinite(seconds) or seconds <= 0: raise ValueError('invalid duration')
        self.seconds, self.out, self.clock = seconds, Path(out), clock
        self.timer_factory = timer_factory
        self.lock = threading.Lock()
        self.stop = None
        self.timer = None
        self.started = self.deadline = self.epoch = None
        self.admitted, self.finished = {}, set()
        self.closed = False
        self.out.mkdir(parents=True, exist_ok=True)
        self.ledger = (self.out/'dispatch_ledger.jsonl').open('x')

    def event(self, data):
        self.ledger.write(json.dumps(data, separators=(',', ':'))+'\n')
        self.ledger.flush()

    def bind(self, stop):
        with self.lock:
            if self.stop is not None and self.stop is not stop:
                raise ValueError('workers do not share the same stop event')
            self.stop = stop

    def close(self):
        with self.lock:
            if not self.closed:
                self.closed = True
                if self.stop is not None: self.stop.set()
                self.event(dict(event='admission_closed', at_s=self.clock.time()))

    def admit(self, ctx):
        with self.lock:
            now = self.clock.monotonic()
            if self.closed or (self.deadline is not None and now >= self.deadline):
                if self.stop is not None: self.stop.set()
                return None
            if self.started is None:
                self.started, self.deadline = now, now+self.seconds
                self.epoch = self.clock.time()
                self.event(dict(event='window_started', at_s=self.epoch, duration_s=self.seconds,
                                deadline_s=self.epoch+self.seconds))
                self.timer = self.timer_factory(self.seconds, self.close)
                self.timer.daemon = True
                self.timer.start()
            rid = ctx['req_id']
            if rid in self.admitted: raise ValueError('duplicate admission: '+str(rid))
            ts = self.clock.time()
            self.admitted[rid] = ts
            self.event(dict(event='dispatch', req_id=rid, client_dispatch_at_s=ts))
            return ts

    def record(self, row):
        with self.lock:
            rid = row['req_id']
            if rid not in self.admitted:
                # Rendering/data failures must remain visible and invalidate the window.
                self.event(dict(event='unadmitted_failure', req_id=rid, error=row.get('error')))
                return
            if rid in self.finished: raise ValueError('duplicate completion: '+str(rid))
            if row['client_dispatch_at_s'] != self.admitted[rid]:
                raise ValueError('dispatch time changed')
            self.finished.add(rid)
            self.event(dict(event='completed', req_id=rid, client_finish_at_s=row['client_finish_at_s']))

    def finish(self, rc, rows, raw_name):
        if self.timer: self.timer.cancel()
        self.close()
        self.ledger.close()
        ids = [r['req_id'] for r in rows]
        valid = (rc == 0 and bool(ids) and len(ids) == len(set(ids))
                 and set(ids) == set(self.admitted) == self.finished)
        receipt = dict(scope='fixed_duration_diagnostic', status='DRAINED' if valid else 'INVALID',
                       full_cohort_complete=False, raw=raw_name, duration_s=self.seconds,
                       first_dispatch_at_s=self.epoch,
                       admission_deadline_s=self.epoch+self.seconds if self.epoch is not None else None,
                       drained_at_s=self.clock.time(), n_dispatched=len(self.admitted), n_completed=len(rows),
                       outstanding=sorted(set(self.admitted)-self.finished), runner_rc=rc,
                       cutoff_reached=self.deadline is not None and self.clock.monotonic() >= self.deadline,
                       ledger_sha256=hashlib.sha256((self.out/'dispatch_ledger.jsonl').read_bytes()).hexdigest())
        write_json(self.out/'timed_window.json', receipt)
        return receipt


class ChainStartPacer:
    """Delay only a chain's first turn; the original driver handles all later gaps."""
    def __init__(self, interval, chains, out, clock=time):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError('invalid chain start interval')
        ids = [c['chain_id'] for c in chains]
        if len(ids) != len(set(ids)):
            raise ValueError('duplicate chain id in cohort')
        self.interval, self.order, self.clock = interval, {cid: i for i, cid in enumerate(ids)}, clock
        self.lock = threading.Lock()
        self.origin = None
        self.ledger = (Path(out)/'chain_start_ledger.jsonl').open('x')

    def wait(self, chain, stop):
        cid = chain['chain_id']
        index = self.order[cid]
        with self.lock:
            if self.origin is None:
                self.origin = self.clock.monotonic()
            due = self.origin + index*self.interval
        if stop.wait(max(0, due-self.clock.monotonic())):
            return False
        with self.lock:
            self.ledger.write(json.dumps(dict(chain_id=cid, cohort_index=index,
                scheduled_after_s=index*self.interval,
                released_after_s=self.clock.monotonic()-self.origin,
                released_at_s=self.clock.time()), separators=(',', ':'))+'\n')
            self.ledger.flush()
        return True

    def close(self):
        self.ledger.close()


def install(module, controller):
    # Strictly locate the original request dispatch timestamp assignment. Fail
    # rather than silently instrument the wrong upstream code after an update.
    tree = ast.parse(inspect.getsource(module.drive))
    count = 0
    class Insert(ast.NodeTransformer):
        def visit_Assign(self, node):
            nonlocal count
            if ast.unparse(node) == 't0 = time.time()':
                count += 1
                return ast.parse('t0 = _timed_window.admit(ctx)\nif t0 is None:\n    return').body
            return node
    tree = ast.fix_missing_locations(Insert().visit(tree))
    if count != 1: raise ValueError('organizer drive dispatch hook changed')
    module.__dict__['_timed_window'] = controller
    exec(compile(tree, str(module.__file__)+' [timed dispatch hook]', 'exec'), module.__dict__)
    original_drive = module.drive
    signature = inspect.signature(original_drive)
    def drive(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        controller.bind(bound.arguments['stop'])
        return original_drive(*args, **kwargs)
    module.drive = drive
    original_record = module.raw_record
    def record(rec):
        row = original_record(rec)
        controller.record(row)
        return row
    module.raw_record = record


def install_stagger(module, pacer, original_signature):
    original_drive = module.drive
    def drive(*args, **kwargs):
        bound = original_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if pacer.wait(bound.arguments['chain'], bound.arguments['stop']):
            return original_drive(*args, **kwargs)
    module.drive = drive


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--loadgen', type=Path, required=True)
    ap.add_argument('--seconds', type=float, required=True)
    ap.add_argument('--chain-start-interval-s', type=float, default=0)
    args, rest = ap.parse_known_args(argv)
    if rest[:1] == ['--']: rest = rest[1:]
    opts = argparse.ArgumentParser(add_help=False)
    opts.add_argument('--out-dir', type=Path, required=True)
    opts.add_argument('--cohort', type=Path)
    parsed, _ = opts.parse_known_args(rest)
    if '--warmup' in rest or '--no-gap' in rest or '--max-chains' in rest:
        raise ValueError('timed measurement must preserve the full frozen workload')
    spec = importlib.util.spec_from_file_location('timed_original_loadgen', args.loadgen)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(args.loadgen.parent))
    spec.loader.exec_module(module)
    controller = AdmissionWindow(args.seconds, parsed.out_dir)
    original_signature = inspect.signature(module.drive)
    install(module, controller)
    pacer = None
    if args.chain_start_interval_s:
        if parsed.cohort is None:
            raise ValueError('chain start pacing requires --cohort')
        chains = json.loads(parsed.cohort.read_text())['chains']
        pacer = ChainStartPacer(args.chain_start_interval_s, chains, parsed.out_dir)
        install_stagger(module, pacer, original_signature)
    before = set(parsed.out_dir.glob('raw_*.jsonl'))
    sys.argv = [str(args.loadgen), *rest]
    rc = 2
    try:
        rc = int(module.main() or 0)
    finally:
        if pacer: pacer.close()
        paths = set(parsed.out_dir.glob('raw_*.jsonl'))-before
        if len(paths) == 1:
            path = paths.pop()
            rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
            receipt = controller.finish(rc, rows, path.name)
        else:
            receipt = controller.finish(2, [], None)
        print('TIMED_WINDOW '+json.dumps(receipt), flush=True)
    return 0 if receipt['status'] == 'DRAINED' else 2


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    sys.exit(main())
