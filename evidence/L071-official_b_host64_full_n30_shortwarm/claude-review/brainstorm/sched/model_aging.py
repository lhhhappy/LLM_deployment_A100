#!/usr/bin/env python3
"""Offline lane model (推算): chain_start over-count vs the aging rate of engine 123's score
(remaining - aging * waited). Same assumptions as lane_model.py; short-bypass lane rule."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lane_model as M, sched_wait as SW

out = {}
orig = M.simulate
for run in SW.RUNS:
    R, t0 = SW.load(run)
    cap = M.capacity_trace(SW.load_log(run), min(r['recv'] for r in R.values()))
    out[run] = {}
    for aging in (0, 100, 300, 1000, 2000):
        src = Path(HERE / 'lane_model.py').read_text()
        # monkeypatch: reuse SRPT_aging2000 branch with a different constant
        M_policy = 'SRPT_aging2000'
        import types
        code = src.replace('2000.0 * (t - r[\'recv\'])', '%r * (t - r[\'recv\'])' % float(aging))
        mod = types.ModuleType('lm_%d' % aging); mod.__file__ = str(HERE / 'lane_model.py')
        exec(compile(code, 'lane_model_aging', 'exec'), mod.__dict__)
        tt = mod.simulate(R, cap, M_policy, bypass_short=True)
        cs = [tt[rid] for rid, r in R.items() if 'chain_start' in r['gates']]
        out[run][aging] = dict(chain_over=sum(v > 30 for v in cs), chain_max=round(max(cs), 1))
    print(run, out[run])
(HERE / 'model-aging.json').write_text(json.dumps({'label': 'OFFLINE MODEL 推算', **out}, indent=1))
