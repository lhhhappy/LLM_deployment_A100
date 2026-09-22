#!/usr/bin/env python3
"""T26 comparator checks: reject partial/error data, immutable ID and prompt join."""
import json
from pathlib import Path
import tempfile
import unittest
from compare_e2b import compare, rows, quantile, trace_summary
from run_e2b_batch import flush_success


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.r = dict(req_id='a', chain_id='c', idx_in_chain=1, phase='intra', uncached_expected=2,
            prompt_sha256='sha', rendered_prompt_tokens=1000, prompt_tokens=1000,
            output_tokens=4, cached_tokens=800, error=None,
            predictions={'role_conservative': {'cached_tokens': 900}})

    def write(self, directory, r):
        path = self.root/directory/'requests.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(r)+'\n')
        return path

    def setup_groups(self):
        for v in ('stock', 'on'):
            self.write('old/'+v+'_reminder_heavy', self.r)
        self.write('control_reminder_heavy', self.r)
        self.write('candidate_reminder_heavy', dict(self.r, cached_tokens=900))

    def test_comparison(self):
        self.setup_groups()
        s = compare(self.root, self.root/'old', 'reminder_heavy')['summary']
        self.assertEqual(s['pairs']['candidate_vs_control'], dict(better=1, equal=0, worse=0))
        self.assertEqual(s['fast_uncached_p95']['candidate'], 100)
        self.assertEqual(s['candidate_within_64_fraction'], 1)

    def test_drift_rejected(self):
        self.setup_groups()
        self.write('candidate_reminder_heavy', dict(self.r, prompt_sha256='changed'))
        with self.assertRaises(AssertionError):
            compare(self.root, self.root/'old', 'reminder_heavy')

    def test_errors_duplicates_and_bad_counts_rejected(self):
        for bad in (dict(error='oops'), dict(output_tokens=3), dict(cached_tokens=1001),
                    dict(prompt_tokens=999)):
            with self.subTest(bad=bad):
                with self.assertRaises(AssertionError):
                    rows(self.write('bad', dict(self.r, **bad)))
        path = self.write('bad', self.r)
        path.write_text(path.read_text()*2)
        with self.assertRaises(AssertionError):
            rows(path)

    def test_exact_harness_quantile(self):
        self.assertEqual(quantile(list(range(20))), 19)
        self.assertIsNone(quantile([]))

    def test_pool_leak_not_passing(self):
        path = self.root/'trace.jsonl'
        events = [dict(kind='startup', pools={'kv': 64}, truncation_align=None),
                  dict(kind='flush', success=True, after={'kv': 63})]
        path.write_text(''.join(json.dumps(e)+'\n' for e in events))
        self.assertFalse(trace_summary(path)['all_flushes_restore_pools'])

    def test_json_flush_optional_message(self):
        self.assertTrue(flush_success({'success': True, 'message': 'Cache flushed.'}))
        self.assertTrue(flush_success({'success': True}))
        for bad in ({'success': False}, {'success': 'true'}, {'success': 1}, {}, None, 'Cache flushed.'):
            self.assertFalse(flush_success(bad))


if __name__ == '__main__':
    unittest.main()
