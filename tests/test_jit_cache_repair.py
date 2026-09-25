import ast
from collections import namedtuple
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('repair', ROOT/'scripts/pod/repair_jit_cache.py')
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


class Repair(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.scope = self.root/'sgl_kernel_jit_test/build-0123456789abcdef'
        self.scope.mkdir(parents=True)
        self.header = self.root/'header.h'
        self.header.write_text('real header')
        staging = self.scope/('.staging-'+'a'*32)
        staging.mkdir()
        (staging/'cuda.cu').write_text('wrapper')
        (staging/'sgl_kernel_jit_test.so').write_bytes(b'exact compiled binary')
        self.entries = [['abs',str(staging/'cuda.cu'),repair.digest(staging/'cuda.cu')],
                        ['sys','header.h',repair.digest(self.header)]]
        (staging/'sgl_deps.json').write_text(json.dumps(self.entries))
        self.leaf = self.scope/('deps-'+repair.deps_key(self.entries))
        staging.rename(self.leaf)
        self.anchors = {'sys':self.root}

    def test_published_binary_unchanged_and_real_loader_accepts_leaf(self):
        row = repair.repair_leaf(self.leaf,self.anchors,apply=True)
        target = Path(row['target'])
        self.assertEqual((target/'sgl_kernel_jit_test.so').read_bytes(),b'exact compiled binary')
        self.assertEqual(json.loads((self.leaf/'sgl_deps.json').read_text()), self.entries)
        # Exercise the actual frozen loader lookup, without importing CUDA.
        source = ROOT/'engine/sglang/kernels/jit/utils/compile/cache.py'
        names = {'_hash_parts','_deps_key','_refresh','find_prebuilt'}
        nodes = [n for n in ast.parse(source.read_text()).body
                 if isinstance(n,ast.FunctionDef) and n.name in names]
        Dep = namedtuple('Dep','root relpath digest')
        env = dict(_DepEntry=Dep, _KEY_HEX_LEN=16, _DEPS_KEY_PREFIX='deps-',
                   _MAX_LEAVES_SCANNED=100, hashlib=repair.hashlib, os=repair.os,
                   _resolve_path=lambda root,relpath: Path(relpath) if root=='abs' else self.anchors[root]/relpath,
                   _file_digest=lambda p:repair.digest(p) if p.is_file() else None,
                   _read_deps=lambda p:[Dep(*e) for e in json.loads((p/'sgl_deps.json').read_text())])
        exec(compile('from __future__ import annotations\n'+'\n'.join(ast.unparse(n) for n in nodes),str(source),'exec'),env)
        self.assertEqual(env['find_prebuilt'](scope=self.scope,module_name='sgl_kernel_jit_test'),target/'sgl_kernel_jit_test.so')

    def test_changed_header_rejected(self):
        self.header.write_text('changed')
        with self.assertRaisesRegex(AssertionError,'dependency changed'):
            repair.repair_leaf(self.leaf,self.anchors,apply=True)
        self.assertEqual(len(list(self.scope.glob('deps-*'))),1)

    def test_changed_wrapper_rejected(self):
        (self.leaf/'cuda.cu').write_text('changed')
        with self.assertRaisesRegex(AssertionError,'wrapper changed'):
            repair.repair_leaf(self.leaf,self.anchors,apply=True)

    def test_other_missing_dependency_not_dropped(self):
        self.header.unlink()
        with self.assertRaises(OSError):
            repair.repair_leaf(self.leaf,self.anchors,apply=True)

    def test_tampered_manifest_rejected(self):
        (self.leaf/'sgl_deps.json').write_text(json.dumps(self.entries[1:]))
        with self.assertRaisesRegex(AssertionError,'manifest hash mismatch'):
            repair.repair_leaf(self.leaf,self.anchors,apply=True)

    def test_dry_run_does_not_write_and_repeated_apply_is_idempotent(self):
        row=repair.repair_leaf(self.leaf,self.anchors)
        self.assertFalse(Path(row['target']).exists())
        repair.repair_leaf(self.leaf,self.anchors,apply=True)
        self.assertEqual(repair.repair_leaf(self.leaf,self.anchors,apply=True)['status'],'already_published')


if __name__=='__main__':
    unittest.main()
