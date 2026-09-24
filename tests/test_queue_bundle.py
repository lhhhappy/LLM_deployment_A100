import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('queue_bundle', ROOT/'scripts/pod/queue_bundle.py')
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


class DeployRuntime(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.ax, self.stage = self.base/'ax', self.base/'stage'
        self.commit = 'a'*40
        for p in [self.ax/'queue/running', self.ax/'queue/pending', self.stage/'engine',
                  self.stage/'jobs', self.stage/'bin/scripts/pod', self.ax/'bin/scripts/pod']:
            p.mkdir(parents=True, exist_ok=True)
        (self.ax/'queue/PAUSE').touch()
        (self.ax/'bin/scripts/pod/lib.sh').write_text('old runtime')
        (self.stage/'bin/scripts/pod/lib.sh').write_text('commit runtime')
        (self.stage/'engine'/ (self.commit+'.diff')).write_text('source diff')
        (self.stage/'jobs/new.sh').write_text('G_COMMIT='+self.commit)
        (self.stage/'bundle.json').write_text(json.dumps(dict(workflow='engine_commit_v1',
            jobs=[dict(name='new.sh',source='jobs/new.sh',commit=self.commit)],
            runtime=['bin/scripts/pod/lib.sh'])))

    def test_deploy_runtime_and_source_before_job_and_allow_pending_retry(self):
        q.publish(self.stage, self.ax)
        q.publish(self.stage, self.ax)
        self.assertEqual((self.ax/'bin/scripts/pod/lib.sh').read_text(), 'commit runtime')
        self.assertTrue((self.ax/'engine'/(self.commit+'.diff')).exists())
        self.assertTrue((self.ax/'queue/pending/new.sh').exists())
        self.assertTrue((self.ax/'runs/new/deployment-receipt.json').exists())
        self.assertTrue((self.ax/'queue/PAUSE').exists())

    def test_running_job_refuses_runtime_change(self):
        (self.ax/'queue/running/active.sh').touch()
        with self.assertRaises(AssertionError): q.publish(self.stage, self.ax)
        self.assertEqual((self.ax/'bin/scripts/pod/lib.sh').read_text(),'old runtime')
        self.assertFalse((self.ax/'queue/pending/new.sh').exists())

    def test_missing_diff_leaves_runtime_untouched(self):
        (self.stage/'engine'/(self.commit+'.diff')).unlink()
        with self.assertRaises(AssertionError): q.publish(self.stage, self.ax)
        self.assertEqual((self.ax/'bin/scripts/pod/lib.sh').read_text(),'old runtime')

    def test_existing_results_cannot_be_overwritten(self):
        (self.ax/'runs/new').mkdir(parents=True)
        with self.assertRaises(AssertionError): q.publish(self.stage, self.ax)
        self.assertEqual((self.ax/'bin/scripts/pod/lib.sh').read_text(),'old runtime')


if __name__ == '__main__': unittest.main()
