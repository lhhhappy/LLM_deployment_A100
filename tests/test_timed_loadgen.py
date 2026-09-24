import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
import hashlib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts/pod/verify'))
import timed_loadgen as timed
import timed_score


class Clock:
    now=100
    def time(self): return self.now
    def monotonic(self): return self.now


class Timer:
    def __init__(self, seconds, callback): self.callback=callback
    def start(self): pass
    def cancel(self): pass


class TimedReplay(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.out=Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def test_deadline_boundary_and_late_completion(self):
        clock=Clock()
        c=timed.AdmissionWindow(10, self.out, clock, Timer)
        c.bind(threading.Event())
        self.assertEqual(c.admit({'req_id':'slow'}), 100)
        clock.now=110
        self.assertIsNone(c.admit({'req_id':'late'}))
        self.assertTrue(c.stop.is_set())
        clock.now=140
        row=dict(req_id='slow', client_dispatch_at_s=100, client_finish_at_s=140)
        c.record(row)
        receipt=c.finish(0, [row], 'raw.jsonl')
        self.assertEqual(receipt['status'], 'DRAINED')
        self.assertEqual(receipt['drained_at_s'], 140)
        self.assertEqual(receipt['admission_deadline_s'], 110)
        self.assertFalse(receipt['full_cohort_complete'])

    def test_missing_slow_request_invalidates(self):
        c=timed.AdmissionWindow(10, self.out, Clock(), Timer)
        c.bind(threading.Event()); c.admit({'req_id':'slow'})
        receipt=c.finish(0, [], 'raw.jsonl')
        self.assertEqual(receipt['status'], 'INVALID')
        self.assertEqual(receipt['outstanding'], ['slow'])

    def test_real_harness_drains_inflight_and_interrupts_gap(self):
        spec=importlib.util.spec_from_file_location('test_timed_original', ROOT/'s1-dev/harness/s1_loadgen.py')
        mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        c=timed.AdmissionWindow(.08, self.out)
        timed.install(mod, c)
        sent=[]
        def engine(ctx, prompt, budget):
            sent.append(ctx['req_id'])
            time.sleep(.15)
            return dict(ttft_s=.01, prompt_tokens=1, cached_tokens=0, output_tokens=budget,
                        ttft_client_s=.01, tpot_s=.01, error=None)
        mod.call_engine=engine
        class Renderer:
            def render(self, body): return body
        def row(rid, gap=0):
            return dict(_req_id=rid, session_id='s', chain_id='c', pack='x', view='x',
                        logical_call_id=rid, max_output_i=10, replay_gap_ms=gap)
        rows={'a':row('a'), 'b':row('b'), 'gap':row('gap', 10000)}
        results=[]; lock=threading.Lock(); stop=threading.Event()
        with (self.out/'raw.jsonl').open('w') as fh:
            def drive(ids):
                mod.drive({'req_ids':ids}, rows, Renderer(), {r:r for r in rows},
                          'model','ns', results,lock,stop,raw_fh=fh)
            ts=[threading.Thread(target=drive,args=(ids,)) for ids in (['a','b'],['gap'])]
            start=time.monotonic()
            for t in ts:t.start()
            for t in ts:t.join(timeout=1)
            self.assertTrue(all(not t.is_alive() for t in ts))
            self.assertGreaterEqual(time.monotonic()-start, .15)
        raw=[json.loads(s) for s in (self.out/'raw.jsonl').read_text().splitlines()]
        self.assertEqual(sent, ['a'])
        self.assertEqual(len(raw), 1)
        self.assertEqual(raw[0]['output_tokens'], 10)
        self.assertEqual(c.finish(0,raw,'raw.jsonl')['status'], 'DRAINED')

    def test_frozen_prefix_and_dispatch_census_validation_on_real_raw(self):
        data=ROOT/'data/s1-dev-longchain-lite'
        evidence=ROOT/'evidence/L058-official_a_longchain_lite_n14/N14'
        cohort=json.loads((data/'cohort.json').read_text())
        summary=json.loads((evidence/'summary.json').read_text())
        rawname=Path(summary['raw']).name
        allrows={r['req_id']:r for r in (json.loads(s) for s in (evidence/rawname).read_text().splitlines())}
        wanted=cohort['chains'][0]['req_ids'][:2]
        rows=[allrows[r] for r in wanted]
        start=min(r['client_dispatch_at_s'] for r in rows)
        ledger=[]
        for r in rows:
            ledger.extend([dict(event='dispatch',req_id=r['req_id'],client_dispatch_at_s=r['client_dispatch_at_s']),
                           dict(event='completed',req_id=r['req_id'],client_finish_at_s=r['client_finish_at_s'])])
        b=('\n'.join(json.dumps(e) for e in ledger)+'\n').encode()
        (self.out/'dispatch_ledger.jsonl').write_bytes(b)
        (self.out/rawname).write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
        (self.out/'timed_window.json').write_text(json.dumps(dict(status='DRAINED',scope='fixed_duration_diagnostic',
            raw=rawname,n_dispatched=len(rows),n_completed=len(rows),first_dispatch_at_s=start,
            admission_deadline_s=start+4200,duration_s=4200,ledger_sha256=hashlib.sha256(b).hexdigest())))
        (self.out/'summary.json').write_text(json.dumps(summary))
        for f in ['flush_evidence.json',Path(summary['run']).name]:
            (self.out/f).write_bytes((evidence/f).read_bytes())
        timed_score.validate(self.out,data,ROOT/'s1-dev/harness')
        # A slow request omitted from completed raw must invalidate even if the
        # other rows and a terminal queue flag look healthy.
        (self.out/rawname).write_text(json.dumps(rows[0])+'\n')
        with self.assertRaises(AssertionError):timed_score.validate(self.out,data,ROOT/'s1-dev/harness')
        rows[0]['uncached_expected']+=1
        (self.out/rawname).write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
        with self.assertRaises(AssertionError):timed_score.validate(self.out,data,ROOT/'s1-dev/harness')


if __name__ == '__main__':unittest.main()
