"""Exercise the real base JIT dependency filter without importing CUDA libraries."""
import ast
from collections import namedtuple
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
source = ROOT/'engine/sglang/kernels/jit/utils/compile/cache.py'
function = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == '_to_entries')
scope = dict(_DepEntry=namedtuple('Dep', 'root relpath digest'),
             _normalize_path=lambda p: ('abs', str(p)),
             _file_digest=lambda p: hashlib.sha256(p.read_bytes()).hexdigest())
exec(compile('from __future__ import annotations\n'+ast.unparse(function), str(source), 'exec'), scope)
entries = scope['_to_entries']


class CachePaths(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.real = self.root/'ram/ax'
        self.real.mkdir(parents=True)
        self.link = self.root/'ax'
        self.link.symlink_to(self.real, target_is_directory=True)
        self.header = self.real/'source.h'
        self.header.write_text('unchanged kernel header')

    def staged(self, root):
        stage = root/'cache/sglang/jit/.staging-test'
        stage.mkdir(parents=True)
        (stage/'cuda.cu').write_text('generated wrapper')
        return stage

    def test_base_bug_records_vanishing_wrapper_through_symlink(self):
        stage = self.staged(self.link)
        deps = entries(dependencies=[Path('cuda.cu'), self.header], build_dir=stage)
        self.assertEqual(len(deps), 2)
        stage.rename(stage.with_name('deps-published'))
        self.assertEqual([Path(d.relpath).name for d in deps if not Path(d.relpath).is_file()], ['cuda.cu'])

    def test_canonical_path_keeps_only_real_dependency_after_publish(self):
        stage = self.staged(self.real)
        deps = entries(dependencies=[Path('cuda.cu'), self.header], build_dir=stage)
        self.assertEqual([d.relpath for d in deps], [str(self.header)])
        stage.rename(stage.with_name('deps-published'))
        self.assertTrue(all(Path(d.relpath).is_file() for d in deps))
        self.assertEqual(deps[0].digest, hashlib.sha256(self.header.read_bytes()).hexdigest())

    def test_storage_exports_canonical_paths_for_existing_link(self):
        script = (ROOT/'scripts/pod/storage_env.sh').read_text()
        exports = script[script.index('  _ax_runtime_path='):].rsplit('\nfi', 1)[0]
        out = subprocess.check_output(['bash', '-eu', '-c', exports],
                                      env={**os.environ, 'AX': str(self.link)}, text=True)
        values = dict(line.split('=', 1) for line in out.splitlines() if '=' in line)
        self.assertEqual(values['SGLANG_JIT_CACHE_DIR'], str(self.real/'cache/sglang/jit'))
        self.assertEqual(values['TMPDIR'], str(self.real/'tmp'))
        self.assertTrue(all(not value.startswith(str(self.link)+'/') for value in values.values()))


if __name__ == '__main__':
    unittest.main()
