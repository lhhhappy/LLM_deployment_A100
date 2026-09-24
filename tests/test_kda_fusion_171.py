"""CPU eligibility checks against the real checkpoint config and source helper."""
import ast
import json
import os
from pathlib import Path
import sys
from types import MappingProxyType, SimpleNamespace
from typing import List, Mapping
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "build/base_exact/sglang"
sys.path.insert(0, str(ROOT / "scripts/engine"))
from tree import tree_dir  # noqa: E402


def definitions(path, names, ns):
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if getattr(n, "name", None) in names
                or isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
                and n.target.id in names]
    assert len(selected) == len(names), (path, names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), ns)


class EligibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model = tree_dir("mech:171") / "srt/models/glm5_next.py"
        ns = dict(List=List, Mapping=Mapping, MappingProxyType=MappingProxyType)
        definitions(BASE / "srt/layers/quantization/utils.py",
                    {"_module_path_match", "_FALLBACK_FUSED_SHARDS", "is_layer_skipped"}, ns)
        ns["get_bool_env_var"] = lambda name, default: os.getenv(name, default).lower() in ("1", "true")
        definitions(model, {"_ax171_can_fuse_kda_projections"}, ns)
        cls.can_fuse = staticmethod(ns["_ax171_can_fuse_kda_projections"])
        cls.config = json.loads((ROOT / "s1-dev/glm_tok/config.json").read_text())


    def qc(self, ignored=None, name="fp8"):
        return SimpleNamespace(get_name=lambda: name, packed_modules_mapping={},
                               ignored_layers=ignored if ignored is not None else
                               self.config["quantization_config"]["modules_to_not_convert"])

    def test_real_checkpoint_all_34_layers_opt_in(self):
        layers = self.config["text_config"]["linear_attn_config"]["kda_layers"]
        self.assertEqual(len(layers), 34)
        with patch.dict(os.environ, SGLANG_AX_KDA_FUSE_PROJ="1"):
            for layer in layers:
                self.assertTrue(self.can_fuse(self.qc(), f"model.layers.{layer}.self_attn", 8, 8))

    def test_default_off_and_unquantized_base_unchanged(self):
        with patch.dict(os.environ, SGLANG_AX_KDA_FUSE_PROJ="0"):
            self.assertFalse(self.can_fuse(self.qc(), "model.layers.0.self_attn", 8, 8))
            self.assertTrue(self.can_fuse(None, "model.layers.0.self_attn", 8, 8))
        self.assertFalse(self.can_fuse(None, "", 2, 8))

    def test_each_original_projection_must_be_unquantized(self):
        prefix = "model.layers.0.self_attn"
        names = ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj", "f_b_proj", "g_b_proj")
        ignored = [f"{prefix}.{name}" for name in names]
        with patch.dict(os.environ, SGLANG_AX_KDA_FUSE_PROJ="1"):
            for missing in names:
                reduced = [x for x in ignored if x != f"{prefix}.{missing}"]
                if missing in ("q_proj", "k_proj", "v_proj"):
                    with self.assertRaises(ValueError):
                        self.can_fuse(self.qc(reduced), prefix, 8, 8)
                else:
                    self.assertFalse(self.can_fuse(self.qc(reduced), prefix, 8, 8))
            self.assertFalse(self.can_fuse(self.qc([prefix + ".fused_qkvbfg_a_proj"]), prefix, 8, 8))
            self.assertFalse(self.can_fuse(self.qc(ignored, "awq"), prefix, 8, 8))
            self.assertFalse(self.can_fuse(self.qc(ignored), prefix, 2, 8))


if __name__ == "__main__":
    unittest.main()
