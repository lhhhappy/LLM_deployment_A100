#!/usr/bin/env python3
"""T26: actual 004 PrefillAdder methods, no CUDA; one known alignment defect.

The expectedFailure is a reproduced defect, NOT a satisfied alignment gate.
"""
from types import SimpleNamespace as NS
import unittest

from test_spf_scheduling import ROOT, REL, load_source, request, adder


class FinalChunkTests(unittest.TestCase):
    def setUp(self):
        self.ns = load_source(ROOT / 'build/d4/b' / REL)
        self.ns['_role_boundary_token_ids'] = lambda: frozenset({99})
        self.ns['get_schedule'] = lambda: NS(schedule_policy='fcfs')
        self.a = adder(self.ns, budget=8192, mamba=True)

    def test_split_same_request_then_finish(self):
        r = request('tail', 1024, cached=8192, boundary=8500)
        self.assertIs(self.a.add_chunked_req(r), r)
        self.assertEqual(r.extend_range.length, 256)
        self.assertIsNone(self.a.new_chunked_req)
        self.assertEqual(self.a.rem_chunk_tokens, 7936)
        self.assertEqual(r.sampling_params.max_new_tokens, 4)
        r.prefix_indices = [0] * 8448
        b = adder(self.ns, budget=8192, mamba=True)
        self.assertIsNone(b.add_chunked_req(r))
        self.assertEqual((r.extend_range.start, r.extend_range.end), (8448, 9216))
        self.assertEqual(self.ns['ROLE_BOUNDARY_STATS']['tail_taken'], 1)

    def test_nonfinal_is_unchanged(self):
        r = request('long', 9000, cached=8192, boundary=8500)
        self.assertIs(self.a.add_chunked_req(r), r)
        self.assertEqual(r.extend_range.length, 8192)
        self.assertEqual(dict(self.ns['ROLE_BOUNDARY_STATS']), {})

    def test_feature_off_exact_control(self):
        self.ns['_role_boundary_token_ids'] = lambda: frozenset()
        control = load_source(ROOT / 'build/d2/b' / REL)
        control['get_schedule'] = lambda: NS(schedule_policy='fcfs')
        for size in (63, 64, 1024, 8192, 8193, 17000):
            r, s = request('r', size, cached=8192), request('s', size, cached=8192)
            a, b = adder(self.ns, mamba=True), adder(control, mamba=True)
            self.assertEqual(a.add_chunked_req(r) is None, b.add_chunked_req(s) is None)
            self.assertEqual(vars(r.extend_range), vars(s.extend_range))
            self.assertEqual(a.rem_chunk_tokens, b.rem_chunk_tokens)
            self.assertEqual(a.rem_total_tokens, b.rem_total_tokens)

    def test_branch_has_priority(self):
        r = request('branch', 1024, cached=8192, boundary=8500)
        r.mamba_branching_seqlen = 8384
        self.assertIsNone(self.a.add_chunked_req(r))
        self.assertEqual(self.ns['ROLE_BOUNDARY_STATS']['tail_skipped_branch_conflict'], 1)

    def test_no_boundary_short_and_unaligned(self):
        for cached, work, pos, counter in ((8192, 1024, None, 'no_boundary'),
                                           (8192, 1024, 8220, 'short'),
                                           (8192, 1030, 9220, 'short'),
                                           (8193, 1024, 8500, 'unaligned_prefix')):
            with self.subTest(counter=counter, pos=pos):
                a = adder(self.ns, mamba=True)
                r = request('skip', work, cached=cached, boundary=pos)
                before = self.ns['ROLE_BOUNDARY_STATS']['tail_skipped_' + counter]
                self.assertIsNone(a.add_chunked_req(r))
                self.assertEqual(self.ns['ROLE_BOUNDARY_STATS']['tail_skipped_' + counter], before+1)

    def test_last_boundary_and_scan_window(self):
        r = request('last', 1024, cached=8192, boundary=8400)
        r.full_untruncated_fill_ids[8700] = 99
        self.assertIs(self.a.add_chunked_req(r), r)
        self.assertEqual(r.extend_range.length, 448)
        a = adder(self.ns, budget=65536, mamba=True)
        old = request('old', 40000, cached=8192, boundary=8500)
        self.assertIsNone(a.add_chunked_req(old))
        self.assertEqual(self.ns['ROLE_BOUNDARY_STATS']['tail_skipped_no_boundary'], 1)

    def test_full_waiter_allowed_second_partial_rejected(self):
        r = request('active', 1024, cached=8192, boundary=8500)
        self.assertIs(self.a.add_chunked_req(r), r)
        full = request('full', 1000, boundary=350)
        self.a.add_one_req(full, True, None)
        self.assertEqual(full.extend_range.length, 1000)
        partial = request('partial', 9000)
        self.assertEqual(self.a.add_one_req(partial, True, None).name, 'OTHER')
        partial.set_extend_range.assert_not_called()
        self.assertEqual(self.a.can_run_list, [r, full])

    def test_host_miss_reselect_rejects_second_partial(self):
        r = request('active', 1024, cached=8192, boundary=8500)
        self.a.add_chunked_req(r)
        miss = request('miss', 9000, host=8000)
        self.a.tree_cache.init_load_back.return_value = ([], miss.last_node)
        self.assertEqual(self.a.add_one_req(miss, True, None).name, 'OTHER')
        self.a.tree_cache.init_load_back.assert_called_once()
        miss.set_extend_range.assert_not_called()

    def test_larger_page_grid(self):
        a = adder(self.ns, page=256, mamba=True)
        r = request('page', 2048, cached=8192, boundary=9000)
        a.add_chunked_req(r)
        self.assertEqual(r.extend_range.length, 768)

    def test_alignment_works_if_manually_injected(self):
        self.a._arena_truncation_align = 512
        r = request('align', 2048, cached=8192, boundary=9000)
        self.a.add_chunked_req(r)
        self.assertEqual(r.extend_range.length, 512)

    @unittest.expectedFailure
    def test_real_fresh_adder_does_not_receive_scheduler_alignment(self):
        # Scheduler creates a fresh adder, calls add_chunked_req first, and only
        # passes truncation_align_size=512 to later add_one_req calls. Actual
        # 004 chooses 768, violating the required 512 grid for next chunk start.
        self.assertFalse(hasattr(self.a, '_arena_truncation_align'))
        r = request('align', 2048, cached=8192, boundary=9000)
        self.a.add_chunked_req(r)
        self.assertEqual(r.extend_range.length % 512, 0)


if __name__ == '__main__':
    unittest.main()
