import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/analysis'))
from compare_window import compare
import window_gates


class WindowPair(unittest.TestCase):
    def setUp(self):
        self.scorer=window_gates.score_formal.load_harness()
        self.row=dict(req_id='x',chain_id='c',phase='intra',idx_in_chain=1,edge_type='continue',
                      glm_tokens=100,uncached_expected=64,max_output_i=5,effective_replay_gap_ms=0,
                      prompt_tokens=100,cached_tokens=80,output_tokens=5,ttft_s=2.,tpot_s=.01,
                      t_recv_s=1.,t_exec_start_s=2.,t_first_token_s=3.,error=None)

    def test_overlapping_gates_and_cache_does_not_reclassify(self):
        cand={**self.row,'ttft_s':6.,'t_first_token_s':7.,'cached_tokens':0}
        summary,_=compare([self.row],[cand],self.scorer)
        self.assertEqual(summary['gate_changes']['fast_intra']['new'],1)
        self.assertEqual(summary['gate_changes']['overall_intra']['new'],1)
        self.assertEqual(summary['unique_candidate_ttft_bad'],1)

    def test_threshold_uses_raw_precision(self):
        base={**self.row,'ttft_s':3.00001}
        summary,_=compare([base],[{**self.row,'ttft_s':2.99999}],self.scorer)
        self.assertEqual(summary['gate_changes']['fast_intra']['repaired'],1)

    def test_missing_timing_stays_unknown(self):
        cand=copy.copy(self.row);cand.pop('t_exec_start_s')
        summary,details=compare([self.row],[cand],self.scorer)
        self.assertEqual(summary['unknown_candidate_timing'],1)
        self.assertIsNone(details[0]['delta_recv_to_exec_s'])

    def test_missing_ttft_cannot_be_counted_as_repaired(self):
        with self.assertRaisesRegex(AssertionError,'TTFT'):
            compare([{**self.row,'ttft_s':5.}],[{**self.row,'ttft_s':None}],self.scorer)

    def test_changed_metadata_rejected(self):
        with self.assertRaisesRegex(AssertionError,'metadata'):
            compare([self.row],[{**self.row,'uncached_expected':40000}],self.scorer)

    def test_reduced_output_rejected(self):
        with self.assertRaisesRegex(AssertionError,'output contract'):
            compare([self.row],[{**self.row,'output_tokens':3}],self.scorer)


if __name__=='__main__':
    unittest.main()
