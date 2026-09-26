#!/usr/bin/env python3
"""CPU regressions for DCP/NextN argument resolution, using production AST.

Only runtime configuration accessors are substituted; the compatibility guard
and default-parameter chooser are loaded verbatim from the selected source tree.
No Torch/model load is necessary for this startup-order contract.
"""
from __future__ import annotations

import ast
import logging
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[2]
ARGS = ROOT / "engine/sglang/srt/arg_groups"


def source_function(path, name, namespace):
    tree = ast.parse(path.read_text(), filename=str(path))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


class TestDCPNextNResolution(unittest.TestCase):
    def setUp(self):
        self.hf = NS(architectures=["Glm5NextForConditionalGeneration"],
                     text_config=NS(num_nextn_predict_layers=1))
        self.mla = True
        self.cfg = NS(dcp_size=2, enable_hierarchical_cache=True,
                      hicache_storage_backend=None, speculative_algorithm="NEXTN",
                      speculative_eagle_topk=None, speculative_num_steps=None,
                      speculative_num_draft_tokens=None,
                      speculative_draft_model_path="/model", model_path="/model",
                      enable_lmcache=False, enable_hisparse=False)
        ns = {"Any": object, "ServerArgs": object, "logger": logging.getLogger(__name__),
              "resolving_view": lambda args: args,
              "model_config_of": lambda args: NS(hf_config=self.hf),
              "use_mla_backend": lambda args: self.mla}
        self.check = source_function(ARGS / "hicache_hook.py",
                                     "resolve_hicache_dcp_compatibility", ns)
        self.defaults = source_function(ARGS / "speculative_hook.py",
                                       "_auto_choose_speculative_params", ns)

    def test_default_topk_at_actual_pipeline_order(self):
        pipeline = ast.parse((ARGS / "pipeline.py").read_text())
        calls = {n.func.id: n.lineno for n in ast.walk(pipeline)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id in {"handle_hicache", "handle_speculative_decoding"}}
        self.assertLess(calls["handle_hicache"], calls["handle_speculative_decoding"])
        for algorithm in ("NEXTN", "EAGLE"):
            for width in (2, 4, 8):
                with self.subTest(algorithm=algorithm, width=width):
                    self.cfg.speculative_algorithm = algorithm
                    self.cfg.dcp_size = width
                    self.assertEqual(self.defaults(self.cfg, self.hf.architectures[0]), (3, 1, 4))
                    self.check(self.cfg)
                    self.assertIsNone(self.cfg.speculative_eagle_topk, "guard must not mutate resolution")

    def test_explicit_chain(self):
        self.cfg.speculative_eagle_topk = 1
        self.check(self.cfg)

    def test_tree_drafting_rejected(self):
        for topk in (0, 2, 4):
            with self.subTest(topk=topk):
                self.cfg.speculative_eagle_topk = topk
                with self.assertRaises(NotImplementedError):
                    self.check(self.cfg)

    def test_unvalidated_models_and_algorithms_stay_rejected(self):
        for attr, value in (("speculative_draft_model_path", "/other"),
                            ("speculative_algorithm", "STANDALONE"),
                            ("speculative_algorithm", "EAGLE3")):
            with self.subTest(attr=attr, value=value):
                saved = getattr(self.cfg, attr)
                setattr(self.cfg, attr, value)
                with self.assertRaises(NotImplementedError):
                    self.check(self.cfg)
                setattr(self.cfg, attr, saved)
        self.hf.text_config.num_nextn_predict_layers = 2
        with self.assertRaises(NotImplementedError):
            self.check(self.cfg)
        self.hf.text_config.num_nextn_predict_layers = 1
        self.hf.architectures = ["LlamaForCausalLM"]
        with self.assertRaises(NotImplementedError):
            self.check(self.cfg)

    def test_other_layout_guards_stay_active(self):
        self.cfg.speculative_eagle_topk = 1
        for attr, value in (("hicache_storage_backend", "file"),
                            ("enable_lmcache", True), ("enable_hisparse", True)):
            with self.subTest(attr=attr):
                saved = getattr(self.cfg, attr)
                setattr(self.cfg, attr, value)
                with self.assertRaises(NotImplementedError):
                    self.check(self.cfg)
                setattr(self.cfg, attr, saved)
        self.mla = False
        with self.assertRaises(NotImplementedError):
            self.check(self.cfg)

    def test_no_dcp_or_no_hicache_keeps_existing_behavior(self):
        self.cfg.speculative_eagle_topk = 4
        self.cfg.dcp_size = 1
        self.check(self.cfg)
        self.cfg.dcp_size = 2
        self.cfg.enable_hierarchical_cache = False
        self.check(self.cfg)

    def test_none_and_dspark_keep_existing_behavior(self):
        for algorithm in (None, "DSPARK"):
            self.cfg.speculative_algorithm = algorithm
            self.check(self.cfg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
