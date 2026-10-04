"""CPU policy contracts; GPU attention/collective numerics live in the probes."""
import importlib.util
import ast
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace as NS

SOURCE = Path(__file__).resolve().parents[2] / "engine/sglang/srt/layers/dcp/local_extend.py"
spec = importlib.util.spec_from_file_location("local_extend_policy", SOURCE)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class PolicyTests(unittest.TestCase):
    def test_startup_envelope_and_disabled_identity(self):
        ps = NS(attn_dcp_size=2, attn_cp_size=1, attn_tp_size=8,
                dcp_comm_backend="ag_rs", dcp_replicate_q_proj=False)
        sa = NS(dp_size=1, pp_size=1, enable_hisparse=False,
                dsa_prefill_backend="tilelang", dsa_decode_backend="tilelang")
        mc = NS(num_attention_heads=64, kv_lora_rank=512, qk_rope_head_dim=0,
                hf_config=NS(architectures=["Glm5NextForConditionalGeneration"], index_topk=2048))
        mr = NS(server_args=sa, model_config=mc, kv_cache_dtype="bf16", device="cuda", page_size=64)
        enabled = [False]
        stubs = {
            "sglang.srt.runtime_context": NS(get_parallel=lambda: ps),
            "sglang.srt.utils": NS(get_bool_env_var=lambda name: enabled[0] if name == "SGLANG_AX_DCP_LOCAL_EXTEND" else False),
            "torch": NS(bfloat16="bf16", cuda=NS(get_device_capability=lambda: (8, 0))),
            "sglang.srt.configs.model_config": NS(get_dsa_index_kpool=lambda config: getattr(config,"index_kpool",4)),
        }
        with patch.dict(sys.modules, stubs), patch.dict(module.os.environ, {}, clear=True):
            self.assertIsNone(module.LocalExtendPolicy.from_runner(None))
            enabled[0] = True
            self.assertEqual(module.LocalExtendPolicy.from_runner(mr).heads, 8)
            for obj, attr, value in ((ps, "attn_dcp_size", 1), (ps, "attn_cp_size", 2),
                                     (sa, "dp_size", 2), (sa, "pp_size", 2),
                                     (sa, "enable_hisparse", True),
                                     (sa, "dsa_prefill_backend", "flashmla"),
                                     (ps, "dcp_replicate_q_proj", True),
                                     (ps, "dcp_comm_backend", "a2a"),
                                     (mc, "qk_rope_head_dim", 64),
                                     (mr, "kv_cache_dtype", "fp8"), (mr, "device", "cpu"),
                                     (mr, "page_size", 63)):
                with self.subTest(attr=attr, value=value):
                    old = getattr(obj, attr)
                    setattr(obj, attr, value)
                    with self.assertRaises(ValueError):
                        module.LocalExtendPolicy.from_runner(mr)
                    setattr(obj, attr, old)
            for limit in ("0", "1025", "bad"):
                module.os.environ["SGLANG_AX_DCP_LOCAL_EXTEND_MAX_TOKENS"] = limit
                with self.assertRaises(ValueError):
                    module.LocalExtendPolicy.from_runner(mr)
            module.os.environ.pop("SGLANG_AX_DCP_LOCAL_EXTEND_MAX_TOKENS")
            mc.hf_config.index_kpool=1
            with self.assertRaisesRegex(ValueError, "KPool=4"):
                module.LocalExtendPolicy.from_runner(mr)
            mc.hf_config.index_kpool=4
            module.os.environ["SGLANG_AX_DCP_LOCAL_EXTEND_LARGE_MAX"] = "8192"
            self.assertTrue(module.LocalExtendPolicy.from_runner(mr).select(0,8192))
            for obj,attr,value in ((mc,"num_attention_heads",32),(mc,"kv_lora_rank",256)):
                old=getattr(obj,attr);setattr(obj,attr,value)
                with self.assertRaises(ValueError):
                    module.LocalExtendPolicy.from_runner(mr)
                setattr(obj,attr,old)
            mc.hf_config.index_kpool=1
            with self.assertRaises(ValueError):
                module.LocalExtendPolicy.from_runner(mr)
            mc.hf_config.index_kpool=4
            mc.hf_config.index_topk=4096
            with self.assertRaises(ValueError):
                module.LocalExtendPolicy.from_runner(mr)

    def test_long_prefix_short_tail_and_padding(self):
        policy = module.LocalExtendPolicy(32, 2, 512, max_tokens=128)
        self.assertTrue(policy.select(14336, 26))
        self.assertTrue(policy.select(14336, 28))  # TP-padded ragged batch
        self.assertFalse(policy.select(14336, 128))  # Q/merge traffic dominates
        self.assertFalse(policy.select(0, 1))
        self.assertFalse(policy.select(4095, 1))
        self.assertFalse(policy.select(65536, 0))
        self.assertFalse(policy.select(65536, 129))  # explicit token budget

    def test_tp8_shape_is_not_tp2_shape(self):
        self.assertFalse(module.LocalExtendPolicy(32, 2, 512).select(16384, 128))
        self.assertTrue(module.LocalExtendPolicy(8, 2, 512).select(16384, 128))

    def test_larger_tail_requires_sufficient_prefix(self):
        p = module.LocalExtendPolicy(8, 2, 512)
        self.assertTrue(p.select(32768, 512))
        self.assertFalse(p.select(32768, 513))
        self.assertFalse(p.select(8192, 512))

    def test_large_prefill_requires_explicit_memory_tradeoff(self):
        default = module.LocalExtendPolicy(8, 2, 512)
        self.assertFalse(default.select(0, 2048))
        self.assertFalse(default.select(262144, 8192))
        bounded = module.LocalExtendPolicy(8, 2, 512, large_max_tokens=2048)
        self.assertTrue(bounded.select(0, 2048))
        self.assertFalse(bounded.select(0, 2049))
        experimental = module.LocalExtendPolicy(8, 2, 512, large_max_tokens=8192)
        self.assertTrue(experimental.select(0, 8192))
        self.assertFalse(experimental.select(0, 8193))

    def test_reused_speculative_batch_does_not_inherit_extend_route(self):
        for ordinary, selected, expected in [(True, True, True), (True, False, False),
                                               (False, True, False), (False, False, False)]:
            fb = NS(forward_mode=NS(is_context_parallel_extend=lambda: ordinary),
                    attn_dcp_metadata=NS(dcp_local_extend=selected))
            self.assertEqual(module.uses_local_extend(fb), expected)
        fb.attn_dcp_metadata = None
        self.assertFalse(module.uses_local_extend(fb))

    def test_mechanism_reports_initialized_policy_not_environment(self):
        runner = NS(eager_runner=NS(dcp_local_extend_policy=None))
        with patch.dict(module.os.environ, {"SGLANG_AX_DCP_LOCAL_EXTEND": "1"}):
            self.assertIn("dcp_local=off", module.local_extend_mechanism_tokens(runner))
        runner.eager_runner.dcp_local_extend_policy = module.LocalExtendPolicy(8, 2, 512, large_max_tokens=8192)
        with patch.dict(module.os.environ, {"SGLANG_AX_DCP_LOCAL_EXTEND": "0"}):
            self.assertEqual(module.local_extend_mechanism_tokens(runner),
                             "dcp_local=on dcp_local_max=512 dcp_local_large=8192")

    def test_route_counts_are_cumulative_role_separated_and_bounded(self):
        stats = module.LocalExtendStats("target", 3)
        with patch.object(module.time, "monotonic", return_value=100), patch.object(module.logger, "info") as log:
            stats.record(local=False, query_tokens=8192, prefix_tokens=0)
            stats.record(local=True, query_tokens=27, prefix_tokens=14336)
            stats.record(local=True, query_tokens=27, prefix_tokens=14336)
            stats.record(local=True, query_tokens=8192, prefix_tokens=4096)
            self.assertEqual(log.call_count, 3)
            snapshot = stats.snapshot()
            self.assertEqual(snapshot["routes"]["local_short"]["batches"], 2)
            self.assertEqual(snapshot["routes"]["local_short"]["query_tokens"], 54)
            self.assertEqual(snapshot["routes"]["local_short"]["prefix_tokens"], 28672)
            self.assertEqual(snapshot["routes"]["gather_kv"]["batches"], 1)
            snapshot["routes"]["local_short"]["batches"] = -1
            self.assertEqual(stats.routes["local_short"]["batches"], 2)
            self.assertEqual(module.LocalExtendStats("draft", 3).routes["local_short"]["batches"], 0)
        with patch.object(module.time, "monotonic", return_value=131), patch.object(module.logger, "info") as log:
            stats.record(local=False, query_tokens=4096, prefix_tokens=0)
            self.assertEqual(log.call_count, 1)

    def test_scheduler_mechanism_line_includes_runtime_tokens(self):
        path = SOURCE.parents[2] / "managers/scheduler.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_ax_mechanism_report")
        ns = {"os": module.os, "sys": sys,
              "get_spec": lambda: NS(speculative_algorithm="NEXTN"),
              "get_parallel": lambda: NS(dcp_size=2)}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), ns)
        runner = NS(eager_runner=NS(dcp_local_extend_policy=module.LocalExtendPolicy(8, 2, 512)))
        scheduler = NS(tp_worker=NS(model_runner=runner), enable_hierarchical_cache=True,
                       enable_hicache_storage=False, schedule_policy="lpm",
                       _ax_sched_protect_blocker=lambda: None, _ax_admission_cfgs=lambda: (True, True),
                       _ax_humming_report=lambda: "on:8_layers", _ax_scatter_report=lambda: "off",
                       _ax_demand_cap_max=lambda: None)
        stubs = {"sglang.srt.layers.dcp.local_extend": module,
                 "sglang.srt.managers.schedule_policy": NS(_ax_srpt_aging=lambda: None,
                                                           _role_boundary_token_ids=lambda: [])}
        with patch.dict(sys.modules, stubs):
            line = ns[method.name](scheduler)
        self.assertIn(" dcp_local=on dcp_local_max=512 dcp_local_large=0 |", line)

    def test_route_audit_rejects_enabled_but_unobserved_rank(self):
        path = SOURCE.parents[5] / "scripts/analysis/dcp_route_audit.py"
        spec = importlib.util.spec_from_file_location("route_audit", path)
        audit_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(audit_module)
        records = []
        with patch.object(module.logger, "info"):
            for rank in (0, 1):
                stats = module.LocalExtendStats("target", rank)
                stats.record(local=False, query_tokens=8192, prefix_tokens=0)
                if rank == 0:
                    stats.record(local=True, query_tokens=27, prefix_tokens=14336)
                records.append(stats.snapshot())
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "server.log"
            log.write_text("[ax] mechanisms: dcp_local=on\n" + "\n".join(
                "[ax-dcp-local] " + json.dumps(r) for r in records))
            result = audit_module.audit(log, tp_size=2, required=["local_short"])
            self.assertFalse(result["passed"])
            self.assertEqual(result["missing"], [(1, "local_short")])
            # Prior warmup activity cannot prove activity in the measured window.
            snapshots = []
            for t in (90, 110, 150):
                for rank in (0, 1):
                    r = json.loads(json.dumps(records[0]))
                    r.update(rank=rank, observed_at_s=t)
                    snapshots.append(r)
            log.write_text("\n".join("[ax-dcp-local] " + json.dumps(r) for r in snapshots))
            self.assertFalse(audit_module.audit(log, tp_size=2, required=["local_short"], since=100, until=160)["passed"])
            for r in snapshots[-2:]:
                r["routes"]["local_short"]["batches"] += 1
                r["routes"]["local_short"]["query_tokens"] += 27
                r["routes"]["local_short"]["prefix_tokens"] += 14336
            log.write_text("\n".join("[ax-dcp-local] " + json.dumps(r) for r in snapshots))
            self.assertTrue(audit_module.audit(log, tp_size=2, required=["local_short"], since=100, until=160)["passed"])


if __name__ == "__main__":
    unittest.main()
