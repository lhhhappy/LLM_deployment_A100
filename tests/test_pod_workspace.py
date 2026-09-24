import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('workspace', ROOT/'scripts/pod/prepare_workspace.py')
workspace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workspace)


class Workspace(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base/'memory/runtime'
        self.root.parent.mkdir()
        self.ax = self.base/'tmp/ax'
        self.mounts = f'1 0 0:1 / / rw - overlay overlay rw\n2 1 0:2 / {self.root.parent} rw,noexec - tmpfs shm rw\n'

    def run_prepare(self, **kw):
        return workspace.prepare(self.root, self.ax, mountinfo=self.mounts,
                                 headroom=1024, minimum=1, **kw)

    def test_fresh_layout_and_repeat_preserve_evidence(self):
        result = self.run_prepare(apply=True)
        self.assertEqual(self.ax.resolve(), self.root/'ax')
        (self.ax/'raw.jsonl').write_text('evidence')
        self.assertTrue(self.run_prepare(apply=True)['already_linked'])
        self.assertEqual((self.ax/'raw.jsonl').read_text(), 'evidence')
        self.assertFalse(result['jit_caches_relocated'])

    def test_empty_probe_directories_preserved(self):
        (self.ax/'codex').mkdir(parents=True)
        result = self.run_prepare(apply=True)
        self.assertTrue((Path(result['empty_directory_backup'])/'codex').is_dir())

    def test_nonempty_workspace_refuses_without_mutation(self):
        self.ax.mkdir(parents=True)
        (self.ax/'raw').write_text('keep')
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            self.run_prepare(apply=True)
        self.assertFalse(self.ax.is_symlink())
        self.assertEqual((self.ax/'raw').read_text(), 'keep')

    def test_foreign_link_refuses(self):
        self.ax.parent.mkdir()
        self.ax.symlink_to(self.base/'other')
        with self.assertRaisesRegex(ValueError, 'foreign'):
            self.run_prepare(apply=True)

    def test_non_tmpfs_or_low_memory_refuses(self):
        with self.assertRaisesRegex(ValueError, 'not on tmpfs'):
            workspace.prepare(self.base/'disk', self.ax, mountinfo=self.mounts, headroom=1024, minimum=1)
        with self.assertRaisesRegex(ValueError, 'headroom'):
            workspace.prepare(self.root, self.ax, mountinfo=self.mounts, headroom=0, minimum=1)
        self.assertFalse(self.ax.exists())

    def test_target_link_cannot_redirect_back_to_disk(self):
        self.root.mkdir()
        disk = self.base/'disk'
        disk.mkdir()
        (self.root/'ax').symlink_to(disk)
        with self.assertRaisesRegex(ValueError, 'foreign target'):
            self.run_prepare(apply=True)
        self.assertFalse(self.ax.exists())
        self.assertEqual(list(disk.iterdir()), [])

    def test_target_old_queue_cannot_be_attached(self):
        target = self.root/'ax'
        (target/'queue/pending').mkdir(parents=True)
        (target/'queue/pending/old.sh').write_text('old job')
        with self.assertRaisesRegex(ValueError, 'nonempty target'):
            self.run_prepare(apply=True)
        self.assertFalse(self.ax.exists())
        self.assertEqual((target/'queue/pending/old.sh').read_text(), 'old job')

    def test_dry_run_does_not_create_workspace(self):
        self.assertFalse(self.run_prepare()['applied'])
        self.assertFalse(self.root.exists())
        self.assertFalse(self.ax.exists())


if __name__ == '__main__':
    unittest.main()
