"""Execute the real 118 dispatch and forward_extend methods without a GPU.

The numerical attention kernels are replaced with recorders. This checks routing,
not their arithmetic; tests/gpu/test_dsa_sparse_118.py covers that separately.
"""
import ast
import enum
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "engine/sglang/srt/layers/attention/dsa_backend.py"


class Mode(enum.Enum):
    EXTEND = 1
    MIXED = 2
    TARGET_VERIFY = 3
    DRAFT_EXTEND_V2 = 4

    def is_target_verify(self):
        return self == Mode.TARGET_VERIFY

    def is_draft_extend_v2(self):
        return self == Mode.DRAFT_EXTEND_V2


class Tensor:
    def __init__(self, shape, label="tensor"):
        self.shape, self.label = shape, label

    def view(self, *shape):
        return Tensor(tuple(self.shape[0] if n == -1 else n for n in shape), self.label)

    def nan_to_num_(self, **kw):
        return self

    def unsqueeze(self, dim):
        return Tensor((*self.shape[:dim], 1, *self.shape[dim:]), self.label)

    def squeeze(self, dim):
        return Tensor(self.shape[:dim] + self.shape[dim + 1:], self.label)


def methods(namespace):
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "DeepseekSparseAttnBackend")
    wanted = {"forward_extend", "_ax118_init", "_forward_tilelang"}
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    nodes += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "ax118_state"]
    source = "from __future__ import annotations\n" + ast.unparse(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(source, str(SOURCE), "exec"), namespace)
    return namespace


class Routing(unittest.TestCase):
    def test_startup_guards_and_prefill_warmup_contract(self):
        cfg = dict(dsa_prefill_impl="tilelang", dsa_decode_impl="tilelang",
                   kv_cache_dtype="bf16", qk_rope_head_dim=0, hisparse_coordinator=None,
                   num_q_heads=8, kv_lora_rank=512, dsa_index_topk=2048,
                   dsa_index_kpool=4, device="cuda", device_sm_major=8)
        for whole, prefill, dcp, overrides, expect in (
            (False, False, True, {"kv_cache_dtype": "fp8"}, "off"),
            (False, True, True, {}, "on:prefill"),
            (False, True, False, {}, "on:prefill"),
            (True, False, False, {}, "on"),
            (True, False, True, {}, "error"),
            (True, True, False, {}, "error"),
            (False, True, True, {"dsa_prefill_impl": "fa3"}, "off"),
            (False, True, True, {"kv_cache_dtype": "fp8"}, "error"),
            (False, True, True, {"hisparse_coordinator": object()}, "error"),
            (False, True, True, {"qk_rope_head_dim": 64}, "error"),
        ):
            with self.subTest(whole=whole, prefill=prefill, dcp=dcp, expect=expect):
                warmed = []
                ns = methods({
                    "_AX_DSA_SPARSE_TRITON": whole, "_AX_DSA_SPARSE_TRITON_PREFILL": prefill,
                    "_ax118_engaged": False, "is_cuda": lambda: True,
                    "torch": NS(bfloat16="bf16", device=lambda x: x),
                    "get_parallel": lambda: NS(dcp_enabled=dcp),
                    "get_exec": lambda: NS(deterministic=NS(enable_deterministic_inference=False)),
                })
                stubs = {"sglang.srt.layers.attention.dsa.sparse_attention_triton":
                         NS(warmup_sparse_attention_fwd=lambda **kw: warmed.append(kw))}
                with patch.dict(sys.modules, stubs):
                    if expect == "error":
                        with self.assertRaises(ValueError):
                            ns["_ax118_init"](NS(**dict(cfg, **overrides)))
                        self.assertFalse(warmed)
                        continue
                    ns["_ax118_init"](NS(**dict(cfg, **overrides)))
                self.assertTrue(ns["ax118_state"]().startswith(expect))
                if expect.startswith("on"):
                    self.assertEqual(warmed, [dict(num_heads=8, dim=512, width=2051, device="cuda")])
                else:
                    self.assertFalse(warmed)

    def test_real_extend_call_site(self):
        # Full-KV prefill, speculative partials and local extend all share
        # forward_extend. Test that the full-KV call alone supplies the opt-in.
        for dcp in (False, True):
            for mode in Mode:
                for local in ((False, True) if dcp and mode == Mode.EXTEND else (False,)):
                    with self.subTest(dcp=dcp, mode=mode, local=local):
                        records = []
                        gathered = Tensor((65536, 1, 512), "gathered")
                        sharded = Tensor((40000, 1, 512), "pool")
                        table = Tensor((33, 2051), "virtual_indices")
                        ns = methods({
                            "torch": NS(nan_to_num=lambda x, **kw: x), "math": __import__("math"),
                            "ForwardMode": Mode, "TopkTransformMethod": NS(PAGED=1),
                            "get_parallel": lambda: NS(dcp_enabled=dcp, attn_dcp_size=2, attn_dcp_rank=0),
                            "_DSA_TRITON_PREFILL": False,
                            "concat_mla_absorb_q_general": lambda q, r: q,
                            "_ax116_dcp_extend_rows": lambda *a: (gathered, table),
                            "_should_return_dsa_dcp_lse": lambda **kw: dcp and (
                                mode in (Mode.TARGET_VERIFY, Mode.DRAFT_EXTEND_V2) or local),
                            "_ax116_dcp_local_indices": lambda x, **kw: x,
                        })

                        def record(**kw):
                            records.append(kw)
                            out = Tensor((1, 33, 8, 512))
                            return (out, "lse") if kw["return_lse"] else out

                        backend = NS(
                            dsa_prefill_impl="tilelang", dsa_decode_impl="tilelang",
                            _resolve_kpool_tail_backend=lambda t, b: b,
                            _check_kpool_tail_backend=lambda *a: None,
                            use_mha=False, forward_metadata=NS(), use_fused_topk=True,
                            token_to_kv_pool=NS(get_key_buffer=lambda layer: sharded),
                            get_topk_transform_method=lambda mode: 1,
                            _pad_topk_indices=lambda indices, rows: indices,
                            _get_fused_topk_page_table=lambda t: table,
                            hisparse_coordinator=None, dsa_index_kpool=4, dcp_topk_column_stride=1,
                            _forward_tilelang=record,
                        )
                        layer = NS(is_cross_attention=False, layer_id=0,
                                   tp_q_head_num=8, v_head_dim=512, head_dim=512, scaling=512**-.5)
                        fb = NS(forward_mode=mode, attn_dcp_metadata=NS(dcp_kv_buffer=gathered))
                        stubs = {
                            "sglang.srt.layers.dcp.local_extend": NS(uses_local_extend=lambda fb: local),
                            "sglang.kernels.ops.attention.dcp_local_indices": NS(local_dcp_indices=lambda x, **kw: x),
                        }
                        with patch.dict(sys.modules, stubs):
                            ns["forward_extend"](
                                backend, Tensor((33, 8, 512)), None, None, layer, fb,
                                q_rope=Tensor((33, 8, 0)), topk_indices=table)
                        self.assertEqual(len(records), 1)
                        call = records[0]
                        self.assertEqual(call.get("is_prefill", False), mode == Mode.EXTEND and not local)
                        if dcp and mode in (Mode.EXTEND, Mode.MIXED) and not local:
                            self.assertIs(call["kv_cache"], gathered)
                        else:
                            self.assertIs(call["kv_cache"], sharded)

    def test_wrapper_uses_explicit_route_and_never_partial_lse(self):
        for enabled in (False, True):
            for prefill in (False, True):
                for lse in (False, True):
                    calls = []

                    def recorder(name):
                        def run(**kw):
                            calls.append(name)
                            out = Tensor((1, 5, 8, 512))
                            return (out, Tensor((1, 5, 8))) if kw.get("return_lse", False) else out
                        return run

                    ns = methods({"_AX_DSA_SPARSE_TRITON": False,
                                  "_AX_DSA_SPARSE_TRITON_PREFILL": enabled})
                    stubs = {
                        "sglang.srt.layers.attention.dsa.sparse_attention_triton": NS(sparse_attention_fwd=recorder("triton")),
                        "sglang.kernels.ops.attention.dsa.tilelang_kernel": NS(tilelang_sparse_fwd=recorder("tilelang")),
                    }
                    # Width 2048 avoids testing unrelated padding mechanics.
                    with patch.dict(sys.modules, stubs):
                        ns["_forward_tilelang"](NS(_ax118_prefill_seen=True), Tensor((5, 8, 512)),
                                                 Tensor((64, 1, 512)), 512, Tensor((5, 2048)), .04,
                                                 return_lse=lse, is_prefill=prefill)
                    self.assertEqual(calls, ["triton" if enabled and prefill and not lse else "tilelang"])


if __name__ == "__main__":
    unittest.main()
