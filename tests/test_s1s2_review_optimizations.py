"""Production CPU decision/metadata regressions for the S1/S2 optimization review.

Run with PYTHONPATH=tests python3 -B -m unittest test_s1s2_review_optimizations.
Scheduler fixture uses real method ASTs and fake Req/pools, not GPU forwards.
"""
import ast
import bisect
from enum import Enum
import importlib.util
import os
from pathlib import Path
import random
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch
import weakref

from test_ax_admission_scheduler import scheduler, cold, continuation
from test_sched_protect_chain import step

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine/sglang"
ENV = {"SGLANG_AX_SCHED_PROTECT": "1", "SGLANG_AX_SCHED_COLD_CAP": "6144",
       "SGLANG_AX_SCHED_SHORT_TOKENS": "8192", "SGLANG_AX_DEADLINE_TIERS": "1",
       "SGLANG_AX_PACE_TPOT": "0", "SGLANG_AX_BACKLOG_RELIEF": "0",
       "SGLANG_AX_SCHED_COLD_CAP_MAX": "0", "SGLANG_AX_DEADLINE_FAMILY": "0"}


def extract(path, cls, name, ns):
    tree = ast.parse(path.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    node = next(n for n in owner.body if getattr(n, "name", "") == name)
    node.decorator_list = []
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node]
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), "exec"), ns)
    return ns[name]


def deadline_module():
    spec = importlib.util.spec_from_file_location("audit_deadline", ENGINE / "srt/managers/ax_deadline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Admission(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, ENV)
        self.env.start()
        self.addCleanup(self.env.stop)

    def make(self, waiters, available=10**7, slots=64):
        return scheduler(chunk=continuation("cont", 65536, 131072), waiting=waiters,
                         budget=8192, available=available, slots=slots)[0]

    def test_output_and_page_reserve_do_not_waste_the_round(self):
        s = self.make([cold("head", 1024)], available=1250)
        self.assertEqual(step(s)["reqs"], [("cont", 65536, 66560)])
        self.assertEqual(s._ax_admission_stats["parks"], 0)

    def test_host_restore_charges_resident_prefix(self):
        head = cold("host", 1024, cached=65536)
        head.prefix_indices, head.host_hit_length = [], 65536
        s = self.make([head], available=8192)
        self.assertEqual(step(s)["reqs"][0][0], "cont")
        self.assertEqual(s._ax_admission_stats["parks"], 0)

    def test_warm_7k_uses_full_round_without_second_partial(self):
        s = self.make([cold("warm", 7000, cached=65536, waited=1)])
        self.assertEqual(step(s)["reqs"], [("warm", 65536, 72536)])
        self.assertEqual(s.chunked_req.rid, "cont")
        self.assertEqual(s.chunked_req.inflight_middle_chunks, 0)

    def test_starved_warm_gets_service_even_after_its_deadline(self):
        with patch.dict(os.environ, {"SGLANG_AX_DEADLINE_MAX_WAIT_WARM_S": "10"}):
            s = self.make([cold("warm", 7000, cached=65536, waited=11)])
            self.assertEqual(step(s)["reqs"][0][0], "warm")

    def test_no_free_request_slot_keeps_owner_running(self):
        s = self.make([cold("warm", 1024, cached=65536)], slots=0)
        self.assertEqual(step(s)["reqs"][0][0], "cont")
        self.assertEqual(s._ax_admission_stats["parks"], 0)

    def test_scan_selects_runnable_head_behind_unfit_host_hit(self):
        head = cold("host", 1024, cached=65536)
        head.prefix_indices, head.host_hit_length = [], 65536
        s = self.make([head, cold("fit", 2048)], available=8192)
        self.assertEqual(step(s)["reqs"][0][0], "fit")

    def test_resource_change_after_preview_falls_back_in_same_pass(self):
        s = self.make([cold("head", 1024)])
        plan = s._ax_admission_plan
        def changed(*args, **kwargs):
            result = plan(*args, **kwargs)
            self.assertTrue(result[1])
            s.token_to_kv_pool_allocator.available_size.return_value = 1250
            return result
        s._ax_admission_plan = changed
        self.assertEqual(step(s)["reqs"], [("cont", 65536, 66560)])
        self.assertEqual(s._ax_admission_stats["parks"], 0)
        self.assertEqual(s.chunked_req._ax_parked_rounds, 0)
        self.assertFalse(s.running_batch.batch_is_full)

    def test_fallback_with_no_kv_defers_safely(self):
        s = self.make([cold("head", 1024)])
        plan = s._ax_admission_plan
        def changed(*args, **kwargs):
            result = plan(*args, **kwargs)
            s.token_to_kv_pool_allocator.available_size.return_value = 0
            return result
        s._ax_admission_plan = changed
        trace = step(s)
        self.assertEqual(trace["mode"], "idle")
        self.assertEqual(s.chunked_req.rid, "cont")
        self.assertEqual(s._ax_admission_stats["parks"], 0)

    def test_rejected_waiter_releases_cow_state_before_owner_resumes(self):
        head = cold("head", 1024)
        s = self.make([head])
        s.tree_cache.allocate_on_match = True
        plan = s._ax_admission_plan
        def changed(*args, **kwargs):
            result = plan(*args, **kwargs)
            self.assertTrue(result[1])
            s.token_to_kv_pool_allocator.available_size.return_value = 1250
            return result
        s._ax_admission_plan = changed
        self.assertEqual(step(s)["reqs"][0][0], "cont")
        s.tree_cache.req_to_token_pool.mamba_allocator.free.assert_called_once()
        self.assertIsNone(head.kv.mamba_pool_idx)
        self.assertIsNone(head.kv.mamba_cow_src_index)
        self.assertFalse(head.kv.mamba_needs_clear)

    def test_two_ranks_apply_same_feasible_waiter_and_park(self):
        payload, outputs = [], []
        for rank, waited in enumerate((1, 121)):
            host = cold("host", 1024, cached=65536)
            host.prefix_indices, host.host_hit_length = [], 65536
            s = self.make([host, cold("warm", 7000, cached=65536, waited=waited)], available=8192)
            def decide(compute, rank=rank):
                if rank == 0:
                    payload.append(compute())
                return payload[0]
            s._ax_rank0_decide = decide
            outputs.append(step(s)["reqs"])
        self.assertTrue(payload[0][1])
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0][0][0], "warm")

    def test_waiter_larger_than_round_never_creates_second_partial(self):
        s = self.make([cold("large", 9000, cached=65536, waited=121)])
        self.assertEqual(step(s)["reqs"][0][0], "cont")


class Families(unittest.TestCase):
    def setUp(self):
        self.ax = deadline_module()

    def test_namespace_separates_cache_salt_and_extra_key(self):
        for attr in ("cache_salt", "extra_key"):
            a, b = cold("a", 12000), cold("b", 12000)
            setattr(a, attr, "one")
            setattr(b, attr, "two")
            work, held, groups = self.ax.family_plan([a, b], lambda a, b: 12000,
                                                     self.ax.FamilyConfig())
            self.assertEqual((work, held, groups), ({}, set(), []))

    def test_sorted_adjacent_lcp_matches_naive_all_pairs(self):
        rng = random.Random(8128)
        for _ in range(30):
            reqs = []
            hashes = {}
            for i in range(30):
                req = cold(str(i), 0)
                req.cache_salt = str(i % 3)
                reqs.append(req)
                hashes[id(req)] = tuple(rng.randrange(4) for _ in range(rng.randrange(0, 25)))
            cache = {}
            self.ax.fill_shared_prefix_cache(reqs, lambda r: hashes[id(r)], cache, 256)
            for i, a in enumerate(reqs):
                for b in reqs[i + 1:]:
                    key = self.ax.family_pair_key(a, b)
                    if a.cache_salt != b.cache_salt:
                        self.assertNotIn(key, cache)
                    else:
                        expected = self.ax.shared_prefix_blocks(hashes[id(a)], hashes[id(b)]) * 256
                        self.assertEqual(cache[key], expected)
            self.ax.fill_shared_prefix_cache(reqs, lambda r: self.fail("cached pairs rehashed"), cache, 256)

    def test_reused_rid_and_reordered_members_have_correct_lcp(self):
        with patch.dict(os.environ, {**ENV, "SGLANG_AX_DEADLINE_FAMILY": "1"}):
            a, b = cold("a", 12000), cold("b", 12000)
            s, ns = scheduler(waiting=[a, b])
            # Model address recycling after an old Req has been collected.
            # Membership must not rely on Python's numeric object id.
            ns["id"] = lambda _: 1
            deadline, _ = s._ax_admission_cfgs()
            self.assertEqual(s._ax_family_plan(s._ax_family_cfg, deadline)[1], {"b"})
            s.waiting_queue.reverse()
            self.assertEqual(s._ax_family_plan(s._ax_family_cfg, deadline)[1], {"b"})
            replacement = cold("b", 12000)
            replacement.origin_input_ids = [2] * 12000
            s.waiting_queue = [a, replacement]
            self.assertEqual(s._ax_family_plan(s._ax_family_cfg, deadline), ({}, set()))

    def test_namespace_changes_and_flush_invalidate_pair_cache(self):
        with patch.dict(os.environ, {**ENV, "SGLANG_AX_DEADLINE_FAMILY": "1"}):
            a, b = cold("a", 12000), cold("b", 12000)
            s, _ = scheduler(waiting=[a, b])
            deadline, _ = s._ax_admission_cfgs()
            self.assertEqual(s._ax_family_plan(s._ax_family_cfg, deadline)[1], {"b"})
            b.cache_salt = "isolated"
            self.assertEqual(s._ax_family_plan(s._ax_family_cfg, deadline), ({}, set()))
            self.assertEqual(s._ax_family_shared, {})
            b.cache_salt = None
            self.assertEqual(s._ax_family_plan(s._ax_family_cfg, deadline)[1], {"b"})
            s._ax_flush_admission_state()
            self.assertEqual(s._ax_family_shared, {})
            self.assertFalse(s._ax_family_members)
            self.assertEqual(s._ax_family_plan(s._ax_family_cfg, deadline)[1], {"b"})


class GraphLayout(unittest.TestCase):
    def setUp(self):
        path = ENGINE / "srt/model_executor/runner/prefill_cuda_graph_runner.py"
        ns = {"is_cp_v2_active": lambda _: True, "_MAX_PREFILL_CUDA_GRAPH_PADDING_FACTOR": 2}
        self.can_run = extract(path, "PrefillCudaGraphRunner", "can_run_graph", ns)
        select = extract(path, "PrefillCudaGraphRunner", "_select_replay_bucket", ns)
        self.runner = NS(_has_inactive_dp_rank=lambda _: False, can_replay_locally=lambda **kw: True,
                         enable_lora=False, _uses_eager_prefill_tail=lambda: True,
                         capture_num_tokens=[1024, 2048, 4096],
                         _ax170_capture_input_scattered={4096: True, 2048: True, 1024: False})
        self.runner._pad_to_bucket = lambda n, sizes: sizes[bisect.bisect_left(sizes, n)]
        self.runner._select_replay_bucket = lambda b: select(self.runner, b)
        module = ModuleType("sglang.srt.layers.communicator")
        module.get_attn_tp_context = lambda: NS(use_input_scattered=lambda b: len(b.input_ids) >= 1025)
        self.module = patch.dict(sys.modules, {module.__name__: module})
        self.module.start()
        self.addCleanup(self.module.stop)

    def batch(self, n):
        return NS(global_num_tokens_cpu=None, batch_size=1, input_ids=range(n),
                  input_embeds=None, replace_embeds=None, extend_prefix_lens_cpu=[0],
                  extend_seq_lens_cpu=[n], forward_mode=NS(is_target_verify=lambda: False),
                  capture_hidden_mode=None, return_logprob=False)

    def test_both_scatter_layouts_replay_their_buckets(self):
        for n in (1000, 1024, 1025, 2048, 3000, 4096):
            self.assertTrue(self.can_run(self.runner, self.batch(n)), n)

    def test_unknown_or_mismatched_layout_remains_eager(self):
        del self.runner._ax170_capture_input_scattered[4096]
        self.assertFalse(self.can_run(self.runner, self.batch(4096)))
        self.runner._ax170_capture_input_scattered[2048] = False
        self.assertFalse(self.can_run(self.runner, self.batch(2048)))

    def test_cp_uses_actual_selected_bucket_and_handles_no_fit(self):
        self.runner.enable_cp_v2_bcg_capture = True
        self.runner.prefill_cp_bcg_input = NS(select_replay_bucket_for_batch=lambda **kw: 4096)
        self.assertTrue(self.can_run(self.runner, self.batch(2048)))
        self.runner.prefill_cp_bcg_input.select_replay_bucket_for_batch = lambda **kw: None
        self.assertFalse(self.can_run(self.runner, self.batch(2048)))


def humming_metadata_fixture():
    """Load actual base/candidate metadata methods, substituting dtype constants only."""
    class DType:
        def __init__(self, bits):
            self.num_bits = bits
            self.itemsize = bits // 8
    types = NS(float16=DType(16), bfloat16=DType(16), float8e4m3=DType(8),
               int8=DType(8), int4=DType(4))
    torch_types = NS(float16=types.float16, bfloat16=types.bfloat16,
                     float8_e4m3fn=types.float8e4m3, int8=types.int8, uint8=DType(8))
    class GemmType(Enum):
        INDEXED = "indexed"
        GROUPED_CONTIGUOUS = "grouped_contiguous"
        GROUPED_MASKED = "grouped_masked"
    class TensorShape:
        def __init__(self, *shape):
            self.shape = shape
            self.ndim = len(shape)
        def size(self, i):
            return self.shape[i]
    ns = {"dtypes": types, "torch": torch_types, "HummingGemmType": GemmType}
    fn = extract(ENGINE / "srt/layers/moe/moe_runner/humming.py", "HummingRunnerCore", "get_buffer_metas", ns)
    base = type("HummingRunnerCore", (), {"get_buffer_metas": fn})
    ns["HummingRunnerCore"] = base
    path = ENGINE / "srt/layers/quantization/fp8_humming_moe.py"
    cls = next(n for n in ast.parse(path.read_text()).body if getattr(n, "name", "") == "_AxFp8HummingRunnerCore")
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), "exec"), ns)
    candidate = ns["_AxFp8HummingRunnerCore"]
    def make(typ):
        obj = typ()
        obj.num_experts = 289
        obj.layer = NS(hidden_size=4096, intermediate_size_per_partition=256,
                       humming_metas={"w13": NS(a_dtype=types.bfloat16, c_dtype=types.bfloat16)})
        return obj
    return make(base), make(candidate), TensorShape, GemmType, types


class HummingMetadata(unittest.TestCase):
    def test_all_runner_shapes_equal_uncached_metadata(self):
        base, cached, tensor, gemm, _ = humming_metadata_fixture()
        for kind in gemm:
            for m in (0, 1, 33, 1025, 8192):
                x = tensor(289, m, 4096) if kind == gemm.GROUPED_MASKED else tensor(m, 4096)
                topk = tensor(m, 9)
                expected = base.get_buffer_metas(x, topk, kind)
                self.assertEqual(cached.get_buffer_metas(x, topk, kind), expected)
                self.assertEqual(cached.get_buffer_metas(x, topk, kind), expected)

    def test_shape_dtype_and_layer_geometry_invalidate_cache(self):
        base, cached, tensor, gemm, types = humming_metadata_fixture()
        for dtype in (types.bfloat16, types.float16, types.float8e4m3, types.int4):
            for n in (256, 512):
                for obj in (base, cached):
                    obj.layer.intermediate_size_per_partition = n
                    obj.layer.humming_metas["w13"].a_dtype = dtype
                for topk in (8, 9):
                    args = (tensor(128, 4096), tensor(128, topk), gemm.INDEXED)
                    self.assertEqual(cached.get_buffer_metas(*args), base.get_buffer_metas(*args))

    def test_cache_is_bounded_and_keeps_no_tensor_references(self):
        _, cached, tensor, gemm, _ = humming_metadata_fixture()
        for m in range(100):
            cached.get_buffer_metas(tensor(m, 4096), tensor(m, 9), gemm.INDEXED)
        self.assertEqual(len(cached._ax_buffer_metas), 64)
        def check(obj):
            self.assertNotIsInstance(obj, tensor)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    check(k)
                    check(v)
            elif isinstance(obj, (tuple, list)):
                for value in obj:
                    check(value)
        check(cached._ax_buffer_metas)


class HostPoolLifetime(unittest.TestCase):
    def test_destroy_releases_views_and_unregisters_once(self):
        calls = []
        class Base:
            def destroy(self):
                self._destroyed = True
        class Buffer:
            pass
        path = ENGINE / "srt/mem_cache/pool_host/dsa.py"
        source = ast.parse(path.read_text())
        owner = next(n for n in source.body if getattr(n, "name", "") == "DSAIndexerPoolHost")
        destroy = next(n for n in owner.body if getattr(n, "name", "") == "destroy")
        cls = ast.ClassDef(name="Pool", bases=[ast.Name(id="Base", ctx=ast.Load())],
                           keywords=[], body=[destroy], decorator_list=[])
        ns = {"Base": Base, "_is_cuda": True, "_is_hip": False,
              "_cuda_host_unregister": lambda _: calls.append("unregister")}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), "exec"), ns)
        pool = ns["Pool"]()
        buffer = Buffer()
        reference = weakref.ref(buffer)
        pool.pin_memory = True
        pool.index_k_with_scale_buffer = buffer
        pool.index_k_data_refs = [buffer]  # Same ownership effect as a tensor view.
        pool.staging_buffer = Buffer()
        del buffer
        pool.destroy()
        self.assertIsNone(reference())
        self.assertIsNone(pool.staging_buffer)
        pool.destroy()
        self.assertEqual(calls, ["unregister"])


class IndexerOptOut(unittest.TestCase):
    def shim(self):
        path = ENGINE / "srt/layers/attention/dsa/sm80_deep_gemm.py"
        nodes = [n for n in ast.parse(path.read_text()).body
                 if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        ns = {"os": os, "_NEED": None, "_INSTALLED": False,
              "torch": NS(cuda=NS(is_available=lambda: True, get_device_capability=lambda: (8, 0))),
              "_OVERRIDES": {"fp8_mqa_logits": lambda: "shim", "missing_entry": lambda: "fallback"}}
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), ns)
        return ns

    def test_explicit_off_leaves_base_module_and_missing_api_unchanged(self):
        base = ModuleType("deep_gemm")
        base.fp8_mqa_logits = lambda: "base"
        before = dict(vars(base))
        with patch.dict(os.environ, {"SGLANG_AX_SM80_INDEXER": "0"}):
            wrapper = self.shim()["maybe_wrap"](base)
            self.assertEqual(vars(base), before)
            self.assertEqual(wrapper.fp8_mqa_logits(), "base")
            with self.assertRaises(AttributeError):
                wrapper.missing_entry()

    def test_on_installs_lazy_sm80_dispatch_for_both_import_styles(self):
        base = ModuleType("deep_gemm")
        base.fp8_mqa_logits = lambda: "base"
        with patch.dict(os.environ, {"SGLANG_AX_SM80_INDEXER": "1"}):
            wrapper = self.shim()["maybe_wrap"](base)
            self.assertEqual(wrapper.fp8_mqa_logits(), "shim")
            self.assertEqual(base.fp8_mqa_logits(), "shim")
            self.assertEqual(base.missing_entry(), "fallback")

    def test_off_preserves_base_import_error(self):
        error = ImportError("base deep_gemm unavailable")
        with patch.dict(os.environ, {"SGLANG_AX_SM80_INDEXER": "0"}):
            wrapper = self.shim()["maybe_wrap"](error)
            with self.assertRaises(ImportError) as caught:
                wrapper.fp8_mqa_logits()
            self.assertIs(caught.exception, error)


if __name__ == "__main__":
    unittest.main()
