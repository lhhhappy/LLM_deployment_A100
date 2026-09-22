"""T13 CPU reproducer: execute actual v1.1 add_one_req/commit with fake host IO.

Passing the host-miss reproducer CONFIRMS a known unsafe path; it is not a
correctness pass for HiCache. No source edits, HTTP calls, torch import or GPU.
"""
import ast
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum, auto
import math
from pathlib import Path
from types import SimpleNamespace, MethodType
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load_adder():
    tree = ast.parse((ROOT / 'build/d1/b/python/sglang/srt/managers/schedule_policy.py').read_text())
    selected = [n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name in ('_PrefillAdmission', 'AddReqResult')]
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PrefillAdder')
    selected += [n for n in cls.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('add_one_req', '_maybe_role_boundary_split', '_commit_prefill_admission')]
    ns = dict(dataclass=dataclass, Enum=Enum, auto=auto, math=math,
              CLIP_MAX_NEW_TOKENS=4096, _role_boundary_token_ids=lambda: {99},
              ROLE_BOUNDARY_STATS=Counter(), _ROLE_BOUNDARY_SCAN_WINDOW=32768,
              mamba_checkpoint_grid=lambda page: 64,
              InitLoadBackParams=lambda **kw: SimpleNamespace(**kw),
              torch=SimpleNamespace(cat=lambda values: sum(values, [])))
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + selected, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), '<actual-D1-v1.1-methods>', 'exec'), ns)
    return ns


class AdmissionReview(unittest.TestCase):
    def setup_case(self, shapes, host=False):
        ns = load_adder()
        req = SimpleNamespace(sampling_params=SimpleNamespace(ignore_eos=False, max_new_tokens=4),
                              output_ids=[], full_untruncated_fill_ids=[1]*256,
                              prefix_indices=[], host_hit_length=128 if host else 0,
                              swa_host_hit_length=0, last_node=object(), best_match_node=object(),
                              kv=SimpleNamespace(), mamba_branching_seqlen=None,
                              retracted_stain=False, needs_host_load_back=lambda: host)
        req.set_extend_range = lambda start, end: setattr(req, 'extend_range', (start, end))
        old = object()
        choices = iter(ns['_PrefillAdmission'](*s) for s in shapes)
        adder = SimpleNamespace(prefill_max_requests=None, can_run_list=[old],
            tree_cache=SimpleNamespace(disable=False, page_size=64, supports_mamba=lambda: True,
                init_load_back=lambda params: ([], req.last_node)),
            page_size=64, rem_total_tokens=4096, rem_chunk_tokens=128,
            dllm_config=None, new_chunked_req=old, prefill_delayer_single_pass=None,
            exact_chunk_fill=False, _mamba_gap_budget_for_req=lambda r: 0,
            _lock_node=lambda node: nullcontext(), _select_prefill_admission=lambda *a, **k: next(choices),
            _req_inc_lock_ref=lambda r: None, _update_prefill_budget=lambda *a, **k: None,
            _account_prefill_cache_admission=lambda *a: None,
            budget_state=lambda: ns['AddReqResult'].CONTINUE)
        for name in ('add_one_req', '_maybe_role_boundary_split', '_commit_prefill_admission'):
            setattr(adder, name, MethodType(ns[name], adder))
        return ns, adder, req, old

    def test_initial_partial_is_rejected_after_existing_partial(self):
        ns, adder, req, old = self.setup_case([(0,128,0,True)])
        self.assertEqual(adder.add_one_req(req, False, None), ns['AddReqResult'].OTHER)
        self.assertIs(adder.new_chunked_req, old)
        self.assertEqual(adder.can_run_list, [old])

    def test_full_request_can_follow_existing_partial(self):
        ns, adder, req, old = self.setup_case([(0,256,4,False)])
        self.assertEqual(adder.add_one_req(req, False, None), ns['AddReqResult'].CONTINUE)
        self.assertIs(adder.new_chunked_req, old)
        self.assertEqual(adder.can_run_list, [old,req])

    def test_reproduces_host_miss_second_partial_overwrite(self):
        ns, adder, req, old = self.setup_case([(128,128,4,False),(0,128,0,True)], host=True)
        adder.add_one_req(req, False, None)
        # This is the defect witness: the initial full shape passed the guard,
        # then host miss changed it into a partial without a second guard.
        self.assertIs(adder.new_chunked_req, req)
        self.assertEqual(adder.can_run_list, [old,req])
        self.assertEqual(req.extend_range, (0,128))


if __name__ == '__main__':
    unittest.main()
