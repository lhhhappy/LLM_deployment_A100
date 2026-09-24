#!/usr/bin/env python3
"""CPU-only integration counterexamples; never calls an engine or changes harness."""
import contextlib
import gzip
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 's1-dev/harness'))
import s1_common as common
import s1_loadgen as loadgen


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(x) + '\n' for x in rows))


def main():
    with tempfile.TemporaryDirectory(prefix='review-harness-') as folder:
        root = Path(folder)
        cached = {'req_id': 'same-id', 'messages': [{'role': 'user', 'content': 'OLD'}]}
        fresh = {'req_id': 'same-id', 'messages': [{'role': 'user', 'content': 'NEW'}]}
        (root / 'bodies').mkdir()
        with gzip.open(root / 'bodies/data.jsonl.gz', 'wt') as f:
            f.write(json.dumps(fresh) + '\n')
        cache = root / 'cache.jsonl'
        write_jsonl(cache, [cached])
        stale = common.ensure_bodies(str(root), ['same-id'], str(cache))
        assert stale['same-id'] == cached and stale['same-id'] != fresh

        # Equal row count also accepts a cache for entirely different IDs.
        unrelated = common.ensure_bodies(str(root), ['new-id'], str(cache))
        assert 'new-id' not in unrelated

        # Exercise actual loadgen.main self-check branch with one missing body.
        # Renderer stub only avoids loading GLM assets: there are zero bodies to render.
        class MustNotRender:
            def __init__(self, *_):
                pass
            def render(self, *_):
                raise AssertionError('unexpected body')
        dataset = root / 'empty-bodies'
        row = {'pack': 'bio', 'view': 'canon', 'logical_call_id': 'q',
               'in_serving_load': True, 'chain_id': 'c', 'phase': 'session_start',
               'glm_tokens': 123, 'max_output_i': 10, 'uncached_expected': 123}
        write_jsonl(dataset / 'requests.jsonl', [row])
        write_jsonl(dataset / 'chains.jsonl', [{'view': 'canon', 'chain_id': 'c'}])
        write_jsonl(dataset / 'samples/review.jsonl', [{'chain_id': 'c'}])
        old_renderer, old_argv = loadgen.Renderer, sys.argv
        loadgen.Renderer = MustNotRender
        sys.argv = ['s1_loadgen.py', '--root', str(dataset), '--set', 'review',
                    '--out-dir', str(root / 'out'), '--self-check', '--no-body-cache']
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
                rc = loadgen.main()
        finally:
            loadgen.Renderer, sys.argv = old_renderer, old_argv
        assert rc == 0 and 'SELF_CHECK: PASS' in output.getvalue()
        print(json.dumps({'stale_body_cache_returned_old_text': True,
                          'equal_count_cache_with_wrong_ids_accepted': True,
                          'missing_body_selfcheck_returncode': rc,
                          'missing_body_selfcheck_stdout': output.getvalue()}, indent=2))


if __name__ == '__main__':
    main()
