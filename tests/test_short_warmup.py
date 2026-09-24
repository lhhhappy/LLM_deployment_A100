"""CPU evidence for deterministic warmup selection and unchanged measured replay."""
import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts/pod/verify'))
sys.path.insert(0, str(ROOT/'s1-dev/harness'))
import short_warmup_loadgen as short
import run_dev_checked as checked
from s1_common import load_index


def original_runner():
    spec = importlib.util.spec_from_file_location('short_test_runner', ROOT/'s1-dev/run_dev.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ShortWarmupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, _, _ = load_index(str(ROOT/'data/s1-dev-longchain'))
        cls.chains = json.loads((ROOT/'data/s1-dev-longchain/cohort.json').read_text())['chains']
        tree = ast.parse((ROOT/'s1-dev/harness/s1_loadgen.py').read_text())
        ns = {}
        names = ('_tier', 'warmup_shape_key', 'select_warmup_chains')
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                                     and n.name in names], type_ignores=[]), 'original_selector', 'exec'), ns)
        cls.shape_key = staticmethod(ns['warmup_shape_key'])
        cls.original_select = staticmethod(ns['select_warmup_chains'])

    def test_frozen_dataset_selection_is_small_adjacent_and_unchanged(self):
        before = json.dumps(self.chains, sort_keys=True)
        pairs, plan = short.plan_for(self.chains, self.rows, self.shape_key)
        self.assertEqual((plan['n_pairs'], plan['n_requests']), (8, 16))
        self.assertEqual(len({c['chain_id'] for c in pairs}), 8)
        self.assertEqual(plan, short.plan_for(list(reversed(self.chains)), self.rows, self.shape_key)[1])
        self.assertEqual(before, json.dumps(self.chains, sort_keys=True))
        original = {c['chain_id']: c for c in self.chains}
        for pair in pairs:
            ids = original[pair['chain_id']]['req_ids']
            i = ids.index(pair['req_ids'][0])
            self.assertEqual(pair['req_ids'], ids[i:i+2])
        old, shapes, missing = self.original_select(self.chains, self.rows, minimum=16)
        self.assertEqual((len(old), sum(len(c['req_ids']) for c in old), shapes, missing),
                         (59, 1326, 182, 0))
        self.assertGreater(plan['missing_shape_buckets'], 0)

    def test_census_failures_cannot_be_called_warm(self):
        _, plan = short.plan_for(self.chains, self.rows, self.shape_key)
        rows = [dict(req_id=rid, output_tokens=budget, error=None, error_class=None)
                for c in plan['pairs'] for rid, budget in zip(c['req_ids'], c['output_budgets'])]
        short.validate_records(plan, rows)
        for bad in (rows[:-1], rows+[rows[0]], [dict(r, error='failed') for r in rows],
                    [dict(r, output_tokens=0) for r in rows],
                    [dict(r, output_tokens=1) for r in rows],
                    [dict(r, output_tokens=True) for r in rows]):
            with self.assertRaises(ValueError): short.validate_records(plan, bad)

    def test_only_warmup_command_is_replaced_and_flush_precedes_measurement(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict('os.environ', {'S1_HARNESS_DIR': str(ROOT/'s1-dev/harness')}):
            out = Path(temp); original = original_runner(); calls = []
            def run(command, log, env):
                calls.append((Path(log).name, command))
                if Path(log).name == 'warmup.log':
                    (out/'short_warmup_receipt.json').write_text(json.dumps(
                        dict(status='COMPLETE', profile='rep16-v1', started_s=time.time())))
                if Path(log).name == 'loadgen.log':
                    (out/'raw_measure.jsonl').write_text('{}\n')
                    (out/'run_measure.json').write_text('{}\n')
                return 0
            original._run = run
            spec = Mock(); spec.loader.exec_module = Mock()
            args = ['--out', str(out), '--n', '30', '--root', str(ROOT/'data/s1-dev-longchain'),
                    '--cohort', str(ROOT/'data/s1-dev-longchain/cohort.json'), '--base-url', 'http://test.invalid']
            def flush(url, evidence, receipt): calls.append(('flush', [])); return True
            with patch.object(checked.importlib.util, 'spec_from_file_location', return_value=spec), \
                 patch.object(checked.importlib.util, 'module_from_spec', return_value=original), \
                 patch.object(checked, 'strict_flush', side_effect=flush), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(checked.run(ROOT/'s1-dev/run_dev.py', args, 'rep16-v1'), 0)
            self.assertEqual([name for name, _ in calls],
                             ['preflight.log', 'warmup.log', 'flush', 'loadgen.log', 'score.log'])
            commands = dict(calls)
            self.assertEqual(Path(commands['warmup.log'][1]).name, 'short_warmup_loadgen.py')
            self.assertEqual(Path(commands['loadgen.log'][1]).name, 's1_loadgen.py')
            for flag in ('--max-chains', '--no-gap', '--warmup', '--skip-warmup'):
                self.assertNotIn(flag, commands['loadgen.log'])
            self.assertEqual(commands['loadgen.log'][commands['loadgen.log'].index('--n')+1], '30')

    def test_failed_warmup_stops_before_flush_and_measurement(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict('os.environ', {'S1_HARNESS_DIR': str(ROOT/'s1-dev/harness')}):
            original = original_runner(); calls = []
            def run(command, log, env):
                calls.append(Path(log).name)
                return 2 if Path(log).name == 'warmup.log' else 0
            original._run = run
            spec = Mock(); spec.loader.exec_module = Mock()
            args = ['--out', temp, '--n', '30', '--root', str(ROOT/'data/s1-dev-longchain'),
                    '--cohort', str(ROOT/'data/s1-dev-longchain/cohort.json'),
                    '--base-url', 'http://test.invalid']
            with patch.object(checked.importlib.util, 'spec_from_file_location', return_value=spec), \
                 patch.object(checked.importlib.util, 'module_from_spec', return_value=original), \
                 patch.object(checked, 'strict_flush') as flush, contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(checked.run(ROOT/'s1-dev/run_dev.py', args, 'rep16-v1'), 2)
                flush.assert_not_called()
            self.assertEqual(calls, ['preflight.log', 'warmup.log'])

    def test_original_loadgen_runs_exact_selected_requests_with_original_budgets(self):
        spec = importlib.util.spec_from_file_location('test_original_loadgen', ROOT/'s1-dev/harness/s1_loadgen.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        calls = []
        module.ensure_bodies = lambda root, want, cache, use_cache: {rid: rid for rid in want}
        module.Renderer = lambda tok: type('Renderer', (), {'render': lambda self, body: body})()
        def engine(ctx, prompt, budget):
            calls.append((ctx['req_id'], prompt, budget))
            return dict(output_tokens=budget, prompt_tokens=self.rows[ctx['req_id']]['glm_tokens'],
                        cached_tokens=0, ttft_s=.1, tpot_s=.01)
        module.call_engine = engine
        fake_spec = Mock(); fake_spec.loader.exec_module = Mock()
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(short.importlib.util, 'spec_from_file_location', return_value=fake_spec), \
             patch.object(short.importlib.util, 'module_from_spec', return_value=module), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            rc = short.main(['--loadgen', str(ROOT/'s1-dev/harness/s1_loadgen.py'), '--',
                             '--root', str(ROOT/'data/s1-dev-longchain'),
                             '--cohort-file', str(ROOT/'data/s1-dev-longchain/cohort.json'),
                             '--set', 's1-dev-longchain', '--out-dir', temp, '--n', '30', '--warmup'])
            receipt = json.loads((Path(temp)/'short_warmup_receipt.json').read_text())
        self.assertEqual(rc, 0)
        self.assertEqual(receipt['status'], 'COMPLETE')
        self.assertEqual(len(calls), 16)
        for rid, prompt, budget in calls:
            self.assertEqual(prompt, rid)
            self.assertEqual(budget, self.rows[rid]['max_output_i'])


if __name__ == '__main__': unittest.main()
