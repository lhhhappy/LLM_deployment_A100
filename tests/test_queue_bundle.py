import importlib.util
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Optional
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


class EffectiveMechanisms(unittest.TestCase):
    def test_nextn_alias_and_both_jobs_against_observed_log(self):
        source = ROOT/'engine/sglang/srt/arg_groups/speculative_hook.py'
        node = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == '_resolve_speculative_algorithm_alias')
        namespace = {'Optional': Optional}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
        self.assertEqual(namespace[node.name]('NEXTN', None), 'EAGLE')
        # Observed on TP0 in 061r at 13:15:33: NEXTN was resolved to EAGLE.
        line = ('[ax] mechanisms: 101=off:role_ids_unset 120=on 122=on '
                '123=off:SGLANG_AX_SRPT_AGING_unset 140=off 180=off:no_hierarchical_cache '
                '| spec=EAGLE dcp=1 | requested: SGLANG_AX_SM80_INDEXER=1 '
                'SGLANG_AX_SM80_FP8_MOE_MARLIN=1 SGLANG_AX_INDEXER_ROW_SHARD=1 '
                'SGLANG_AX_KDA_FUSE_PROJ=0 SGLANG_AX_MOE_FUSE_SWIGLU=0')
        template = (ROOT/'scripts/pod/jobs/dev_ladder_template.sh').read_text()
        check = template[template.index('[ -n "${G_EXPECT:-}" ]'):template.index('\ngrep -h "KV Cache')]
        with tempfile.TemporaryDirectory() as d:
            for suffix, host, larger_mem in [('', False, False), ('_mem087', False, True),
                                             ('_180_mem087', True, True)]:
                job = (ROOT/f'scripts/pod/jobs/official_a{suffix}_full_n30_70m.sh').read_text()
                expected = re.search(r'^G_EXPECT="([^"]+)"', job, re.M)[1]
                self.assertIn('--speculative-algorithm NEXTN', job)
                self.assertIn('SGLANG_AX_PACE_TPOT=0 ', job)
                self.assertEqual('--mem-fraction-static 0.87' in job, larger_mem)
                # Alias comes from the observed log; mechanism states below
                # model the newly requested jobs, not an observed GPU result.
                observed = line.replace('122=on', '122=off:SGLANG_AX_PACE_TPOT_unset')
                if host: observed = observed.replace('180=off:no_hierarchical_cache', '180=on')
                for log, rc in [(observed, 0), (observed.replace('122=off:SGLANG_AX_PACE_TPOT_unset', '122=on'), 2),
                                (observed.replace('spec=EAGLE', 'spec=-'), 2)]:
                    Path(d, 'server.log').write_text(log+'\n')
                    result = subprocess.run(['bash', '-c', check], text=True, capture_output=True,
                                            env={**os.environ, 'RUN_DIR':d, 'G_EXPECT':expected})
                    self.assertEqual(result.returncode, rc, result.stdout+result.stderr)


if __name__ == '__main__': unittest.main()
