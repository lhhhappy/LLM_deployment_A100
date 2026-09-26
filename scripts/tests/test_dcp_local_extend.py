"""CPU policy contracts; GPU attention/collective numerics live in the probes."""
import importlib.util
import sys
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
        mr = NS(server_args=sa, model_config=mc, kv_cache_dtype="bf16", device="cuda")
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
                                     (mr, "kv_cache_dtype", "fp8"), (mr, "device", "cpu")):
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


if __name__ == "__main__":
    unittest.main()
