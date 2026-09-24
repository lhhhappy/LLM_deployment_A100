import base64
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import subprocess
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts/analysis'))
import window_gates as gates
import window_watch as watch
import window_notify as bridge


class Windows(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def rows(self):
        # Slow unfinished requests are deliberately absent from completed raw.
        return [dict(req_id=str(i), client_dispatch_at_s=t, client_finish_at_s=t+1,
                     wall_s=1, phase='chain_start', output_tokens=10, ttft_s=1, tpot_s=.01)
                for i, t in enumerate([100, 3700])]

    def test_old_window_cannot_close_from_completed_max_wall(self):
        run = gates.windows_for(self.rows(), 25, True, gates.score_formal.load_harness())
        self.assertTrue(all(w['open'] for w in run['windows']))
        self.assertIn('unfinished requests absent', gates.summary(run, 'live'))

    def test_invalid_window_width(self):
        for width in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError): gates.windows_for(self.rows(), width, True, None)

    def test_health_alerts_distinguish_warmup_from_no_measurement_progress(self):
        meta = dict(job_state='running', health=dict(phase='warmup', server_log_age_s=600))
        self.assertEqual(watch.health_alerts(meta), [])
        meta['health'].update(phase='measurement', raw_age_s=301)
        self.assertEqual(len(watch.health_alerts(meta)), 2)
        meta['health'].update(raw_age_s=1, server_log_age_s=1, recent_error_lines=['CUDA out of memory'])
        self.assertEqual(len(watch.health_alerts(meta)), 1)

    def test_progress_before_first_checkpoint_is_not_reported_as_stall(self):
        root = self.out/'runs/job'; level = root/'N30'; level.mkdir(parents=True)
        running = self.out/'queue/running'; running.mkdir(parents=True)
        (running/'job.sh').touch()
        now = time.time()
        (level/'flush_evidence.json').write_text(json.dumps(dict(
            runner_started_s=now-400, flush_success=True, flush_finished_s=now-310)))
        # Preflight is excluded by its dispatch timestamp even if its mtime is new.
        (level/'raw_preflight.jsonl').write_text(json.dumps(dict(client_dispatch_at_s=now-350))+'\n')
        (level/'raw_measure.jsonl').write_text(json.dumps(dict(client_dispatch_at_s=now-10))+'\n')
        code = watch.SNAPSHOT_CODE.replace('/tmp/ax', str(self.out))
        output = subprocess.check_output([sys.executable, '-c', code, 'job', '0'], text=True)
        meta = json.loads(watch.marked(output, 'WINDOW_META '))
        self.assertLess(meta['health']['raw_age_s'], 10)
        self.assertEqual(watch.health_alerts(meta), [])
        self.assertEqual(meta['health']['first_observed_dispatch_s'], now-10)

    def test_first_fifteen_then_thirty_minutes_is_anchored_to_measurement(self):
        first = watch.next_check(100, 900, 1800)
        second = watch.next_check(100, 900, 1800, first)
        third = watch.next_check(100, 900, 1800, second)
        self.assertEqual((first, second, third), (1000, 2800, 4600))

    def test_compact_poll_ignores_ticks_but_detects_alert_staleness_and_recovery(self):
        state = dict(job_state='running', health='up', heartbeat=1000, last_report='open')
        health = dict(phase='measurement', completed_rows=12, alerts=[])
        view, original = watch.compact_status(state, health, 1010)
        health['completed_rows'] = 13
        self.assertEqual(watch.compact_status(state, health, 1060)[1], original)
        self.assertEqual(view['completed'], 12)
        self.assertNotEqual(watch.compact_status(state, health, 1200)[1], original)
        health['alerts'] = ['server error']
        self.assertNotEqual(watch.compact_status(state, health, 1010)[1], original)
        health['alerts'] = []
        state.update(health='retrying', error='transport unavailable')
        self.assertNotEqual(watch.compact_status(state, health, 1010)[1], original)
        state.update(health='up')
        self.assertEqual(watch.compact_status(state, health, 1010)[1], original)
        state['last_scheduled_deadline'] = 900
        self.assertNotEqual(watch.compact_status(state, health, 1010)[1], original)

    def test_bridge_only_wakes_for_diagnostic_or_health_changes(self):
        s = dict(job_state='running', health='up', heartbeat=1000,
                 last_report='open window', last_scheduled_deadline=900)
        h = dict(phase='measurement', completed_rows=400, alerts=[])
        original = bridge.event_key(s, h, 1010)
        h['completed_rows'] = 420
        s['heartbeat'] = 1060
        self.assertEqual(bridge.event_key(s, h, 1070), original)
        s['last_scheduled_deadline'] = 2700
        self.assertNotEqual(bridge.event_key(s, h, 1070), original)
        text = bridge.message('job', original, s, h, 1070)
        self.assertLessEqual(len(text), 1600)
        self.assertIn('完成=420', text)

    def test_only_live_unterminated_last_fragment_ignored(self):
        p=self.out/'raw'
        p.write_text('{"req_id":"a"}\n{"req_id":')
        self.assertEqual(len(gates.load_raw(p, allow_partial=True)), 1)
        with self.assertRaises(ValueError): gates.load_raw(p)
        p.write_text('{"req_id":"a"}\nbad\n')
        with self.assertRaises(ValueError): gates.load_raw(p, allow_partial=True)

    def test_duplicate_request_rejected(self):
        p=self.out/'raw'; p.write_text('{"req_id":"a"}\n'*2)
        with self.assertRaises(ValueError): gates.load_raw(p)

    def meta(self):
        return dict(job_state='failed', raw='raw_x.jsonl', n=30, score={},
                    verdict=dict(status='VALID', rows=2, raw='raw_x.jsonl'),
                    summary=dict(raw='/tmp/ax/runs/x/N30/raw_x.jsonl', n=30))

    def test_failed_is_complete_only_with_matching_receipt(self):
        m=self.meta()
        self.assertTrue(watch.is_complete(m, self.rows()))
        for key, value in [('rows', 3), ('raw', 'other.jsonl'), ('status', 'INVALID')]:
            m=self.meta(); m['verdict'][key]=value
            self.assertFalse(watch.is_complete(m, self.rows()))
        m=self.meta(); m['job_state']='running'
        self.assertFalse(watch.is_complete(m, self.rows()))
        m=self.meta(); m['summary']['n']=14
        self.assertFalse(watch.is_complete(m, self.rows()))

    def test_hashed_chunk_download_and_truncation(self):
        data=('\n'.join(json.dumps(r) for r in self.rows())+'\n').encode()
        z=gzip.compress(data)
        m=dict(size=len(z), sha256=hashlib.sha256(z).hexdigest(), archive='snapshot.gz')
        def call(_code, _path, start, length):
            return 'WINDOW_DATA '+base64.b64encode(z[start:start+length]).decode()+'\nexit_code: 0\n'
        self.assertEqual(watch.download(m, self.out, call), self.rows())
        with self.assertRaises(ValueError):
            watch.download(m, self.out, lambda *a:'WINDOW_DATA '+base64.b64encode(z[:-2]).decode())
        m['sha256']='0'*64
        with self.assertRaises(ValueError): watch.download(m, self.out, call)
        self.assertEqual((self.out/'raw.jsonl').read_bytes(), data)


if __name__ == '__main__': unittest.main()
