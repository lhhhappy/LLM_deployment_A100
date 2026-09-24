"""CPU checks that the 052 installer only touches the intended copied model."""

import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "scripts/pod/verify/install_numtrace.py"
BASE_MODEL = ROOT / "build/base_exact/sglang/srt/models/glm5_next.py"
spec = importlib.util.spec_from_file_location("install_numtrace", INSTALLER)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class InstallNumtraceTest(unittest.TestCase):
    def test_inserts_all_stage_hooks_and_rejects_second_install(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "sglang"
            models = package / "srt/models"
            models.mkdir(parents=True)
            shutil.copyfile(BASE_MODEL, models / "glm5_next.py")
            module.install(package)
            source = (models / "glm5_next.py").read_text()
            self.assertEqual(source.count(module.MARKER), 8)
            for stage in ("attn_input", "attn_output", "mlp_input", "mlp_output", "layer_exit"):
                self.assertEqual(source.count(f"'{stage}'"), 1)
            self.assertTrue((models / "ax_numtrace.py").is_file())
            with self.assertRaisesRegex(RuntimeError, "already installed"):
                module.install(package)

    def test_rejects_changed_anchor_without_writing_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "sglang"
            models = package / "srt/models"
            models.mkdir(parents=True)
            source = BASE_MODEL.read_text().replace("        hidden_states = self.self_attn(\n",
                                                    "        hidden_states = fake_attn(\n", 1)
            (models / "glm5_next.py").write_text(source)
            with self.assertRaisesRegex(ValueError, "anchor count=0"):
                module.install(package)
            self.assertFalse((models / "ax_numtrace.py").exists())
            self.assertEqual((models / "glm5_next.py").read_text(), source)


if __name__ == "__main__":
    unittest.main()
