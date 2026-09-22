#!/usr/bin/env python3
"""CPU policy, source wiring and exact/legacy stats semantics tests for T48."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from extract_spec_stats_160 import parse
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'build/p160/candidate/sglang/srt/arg_groups/ax_mtp_sm80.py'
spec=importlib.util.spec_from_file_location('policy160',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class TestPolicy(unittest.TestCase):
    def cfg(self,**kw):return NS(**(dict(speculative_algorithm='EAGLE',speculative_eagle_topk=1,speculative_draft_model_path='/mnt/models')|kw))
    def test_policy(self):
        f,e=m.policy(self.cfg(),'Glm5NextForConditionalGeneration',80)
        self.assertEqual(f['dsa_decode_backend'],'tilelang');self.assertEqual(f['linear_attn_verify_backend'],'triton')
        self.assertEqual(e['SGLANG_AX_KDA_DUAL_SNAPSHOT'],'0');self.assertEqual(e['SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS'],'')
        self.assertNotIn('max_running_requests',f);self.assertNotIn('speculative_accept_threshold_single',f)
    def test_other_arch(self):self.assertEqual(m.policy(self.cfg(),'Other',80),({},{}))
    def test_other_sm(self):self.assertEqual(m.policy(self.cfg(),'Glm5NextForConditionalGeneration',90),({},{}))
    def test_off(self):self.assertEqual(m.policy(self.cfg(speculative_algorithm=None),'Glm5NextForConditionalGeneration',80),({},{}))
    def test_topk(self):
        with self.assertRaises(ValueError):m.policy(self.cfg(speculative_eagle_topk=2),'Glm5NextForConditionalGeneration',80)
    def test_path(self):
        with self.assertRaises(ValueError):m.policy(self.cfg(speculative_draft_model_path=None),'Glm5NextForConditionalGeneration',80)
    def test_wiring(self):
        s=(p.parent/'speculative_hook.py').read_text();self.assertLess(s.index('_handle_eagle_family.auto_params'),s.index('configure(server_args, model_arch)'))
        self.assertIn('from sglang.srt.arg_groups.overrides import declare_resolution',p.read_text())
    def test_weighted(self):
        s=['[TP0] Decode batch, spec tokens: 12, spec rounds: 6, accept len: 2.00, accept rate: 0.33,',
           '[TP1] Decode batch, spec tokens: 12, spec rounds: 6, accept len: 2.00, accept rate: 0.33,',
           'Decode batch, spec tokens: 2, spec rounds: 2, accept len: 1.00, accept rate: 0.00,']
        d=parse(s);self.assertEqual(d['weighted_tokens_per_request_step'],1.75);self.assertEqual(d['weighted_draft_accept_rate'],.25)
        self.assertEqual(parse(s,start_line=3)['request_rounds'],2)
    def test_legacy(self):
        d=parse(['Decode batch, accept len: 1.66, accept rate: 0.22,']);self.assertEqual(d['window_count'],1);self.assertIsNone(d['weighted_tokens_per_request_step'])
    def test_empty(self):self.assertEqual(parse([])['window_count'],0)

if __name__=='__main__':unittest.main(verbosity=2)
