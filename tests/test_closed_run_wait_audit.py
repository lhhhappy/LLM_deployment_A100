"""Censoring and overlap semantics; fake timing only, no schedule simulation."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/analysis'))
from closed_run_wait_audit import census, union_seconds


class AuditTests(unittest.TestCase):
    def test_waiting_and_decoding_requests_both_missing_from_completed_raw(self):
        h=SimpleNamespace(TTFT_GATE_SPECS=[('chain','chain_start',30)],in_ttft_gate=lambda r,g:True)
        def row(rid,received,admitted,first,finished):
            return dict(req_id=rid,t_recv_s=received,t_exec_start_s=admitted,t_first_token_s=first,
                        client_dispatch_at_s=received,client_first_token_at_s=first,client_finish_at_s=finished)
        rows=[row('waiting',0,45,50,60),row('decoding',0,1,35,55),
              row('partial',20,22,50,60),row('future',41,42,43,44),row('done',0,1,2,3)]
        c,pending=census(rows,40,h)
        self.assertEqual(c['arrived'],4)
        self.assertEqual({r['req_id'] for r in pending},{'waiting','partial'})
        self.assertEqual(c['before_first_admission'],1)
        self.assertEqual(c['after_first_admission'],1)
        self.assertEqual(c['client_unfinished'],3)
        self.assertEqual(c['gates']['chain_start']['overdue_pending'],1)
        self.assertEqual(c['gates']['chain_start']['observed_bad_but_not_completed'],2)

    def test_first_token_at_boundary_is_not_pending_and_threshold_is_strict(self):
        h=SimpleNamespace(TTFT_GATE_SPECS=[('chain','chain_start',30)],in_ttft_gate=lambda r,g:True)
        r=dict(req_id='r',t_recv_s=10,t_exec_start_s=20,t_first_token_s=40,
               client_dispatch_at_s=10,client_first_token_at_s=40,client_finish_at_s=50)
        c,_=census([r],40,h)
        self.assertEqual(c['server_first_pending'],0)
        self.assertEqual(c['gates']['chain_start']['observed_bad_among_all_arrived'],0)

    def test_simultaneous_lifecycles_not_double_counted(self):
        self.assertEqual(union_seconds([(1,4),(2,3),(3,7),(9,10)]),7)
        self.assertEqual(union_seconds([]),0)


if __name__=='__main__':
    unittest.main()
