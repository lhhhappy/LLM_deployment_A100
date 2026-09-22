"""CPU regressions execute actual extracted SGLang methods, with fake workers.

These tests do not import CUDA dependencies or claim distributed GPU coverage.
"""
import ast
import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock

from contract_probe import events
from prepare_patches import CONTROL_PATH, OLD_CONTROL, NEW_CONTROL, replace_once
from make_reduced_checkpoint import reduce_config, select_key

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class FlushResult:
    success: bool
    message: str = ""


def load_flush(source):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TokenizerControlMixin")
    method = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "flush_cache")
    method.returns = None
    for arg in method.args.args:
        arg.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {"FlushCacheReqOutput": FlushResult,
                 "FlushCacheReqInput": lambda **kw: SimpleNamespace(**kw)}
    exec(compile(module, "<actual-sglang-flush-method>", "exec"), namespace)
    return namespace["flush_cache"]


def invoke(flush, replies):
    manager = SimpleNamespace(auto_create_handle_loop=Mock(),
        flush_cache_communicator=AsyncMock(return_value=replies), mm_processor=Mock())
    result = asyncio.run(flush(manager, timeout_s=0))
    return result, manager


class FlushRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (ROOT / "src/sglang" / CONTROL_PATH).read_text()
        cls.original = staticmethod(load_flush(source))
        cls.patched = staticmethod(load_flush(replace_once(source, OLD_CONTROL, NEW_CONTROL)))

    def test_reproduces_first_reply_masking_later_failure(self):
        result, _ = invoke(self.original, [FlushResult(True), FlushResult(False, "worker busy")])
        self.assertTrue(result.success, "upstream changed; reassess the candidate patch")

    def test_patch_rejects_failure_in_either_response_order(self):
        for replies in ([FlushResult(True), FlushResult(False, "busy")],
                        [FlushResult(False, "busy"), FlushResult(True)]):
            result, manager = invoke(self.patched, replies)
            self.assertFalse(result.success)
            self.assertIn("busy", result.message)
            manager.mm_processor.clear_preprocess_cache.assert_not_called()

    def test_patch_requires_all_acknowledgements_successful(self):
        result, manager = invoke(self.patched, [FlushResult(True) for _ in range(8)])
        self.assertTrue(result.success)
        manager.mm_processor.clear_preprocess_cache.assert_called_once()

    def test_empty_reply_set_is_failure(self):
        result, _ = invoke(self.patched, [])
        self.assertFalse(result.success)


class ProbeParserTests(unittest.TestCase):
    def test_sse_comments_multiline_and_done(self):
        stream = [b":keepalive\r\n", b"\r\n", b'data: {"meta_info":\n',
                  b'data: {"completion_tokens":3}}\n', b"\n", b"data: [DONE]\n", b"\n"]
        data = list(events(stream))
        self.assertEqual(json.loads(data[0])["meta_info"]["completion_tokens"], 3)
        self.assertEqual(data[1], "[DONE]")


class ReducedFixtureTests(unittest.TestCase):
    def test_real_config_keeps_kernel_dimensions_and_removes_mtp(self):
        original = json.loads((ROOT / "s1-dev/glm_tok/config.json").read_text())
        reduced = reduce_config(original, 5)
        self.assertEqual(original["text_config"]["num_hidden_layers"], 45)
        self.assertEqual(reduced["text_config"]["num_nextn_predict_layers"], 0)
        self.assertEqual(reduced["text_config"]["linear_attn_config"]["full_attn_layers"], [3])
        self.assertEqual(reduced["text_config"]["index_head_dim"], 128)
        self.assertEqual(reduced["text_config"]["n_routed_experts"], 288)

    def test_checkpoint_keys_preserve_vision_and_drop_removed_layers(self):
        self.assertTrue(select_key("model.language_model.layers.3.self_attn.wq.weight", 5))
        self.assertTrue(select_key("model.visual.blocks.23.attn.qkv.weight", 5))
        self.assertFalse(select_key("model.language_model.layers.45.eh_proj.weight", 5))
        self.assertFalse(select_key("model.layers.5.self_attn.weight", 5))


if __name__ == "__main__":
    unittest.main()
