#!/usr/bin/env python3
"""CPU review probes using the candidate's real scheduler/adder methods.

Cache publication, memory pools and GPU forwards use the existing test fakes.
This proves control flow and budget decisions, not latency or TP8 performance.
No request identities or workload contents from the benchmark are used here.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests'))
from test_prefix_producer import request, scheduler, step


def job_value(text, key):
    lines = [line for line in text.splitlines() if line.startswith(key + '=')]
    if len(lines) != 1:
        raise ValueError(f'Expected one literal assignment: {key}')
    return shlex.split(lines[0].split('=', 1)[1])[0]


def run_budget_case(env, cap, short, new_head=False, demand_max=None, with_warm=True):
    clean = {k: v for k, v in os.environ.items() if not k.startswith('SGLANG_')}
    config = dict(env, SGLANG_AX_SCHED_COLD_CAP=str(cap),
                  SGLANG_AX_BACKLOG_COLD_CAP=str(demand_max or cap),
                  SGLANG_AX_SCHED_SHORT_TOKENS=str(short))
    if demand_max is not None:
        config['SGLANG_AX_SCHED_COLD_CAP_MAX'] = str(demand_max)
    with patch.dict(os.environ, dict(clean, **config), clear=True):
        cold = request('cold', shared=80000 if not new_head else 20000, tail=0,
                       tag=7, cached=32768 if not new_head else 0, waited=2)
        warm = request('warm', shared=32768, tail=512, tag=9, waited=.1)
        waiting = ([cold] if new_head else []) + ([warm] if with_warm else [])
        s, ns, cache = scheduler(waiting,
                                 None if new_head else cold, budget=16384, interval=2)
        if with_warm:
            cache.publish(warm, 32768)
        if not new_head:
            cache.publish(cold, 32768)
            cache.match(cold)
        plans = []
        def decide(fn):
            value = fn()
            if isinstance(value, dict) and 'version' in value:
                plans.append(value)
            return value
        s._ax_rank0_decide = decide
        result = step(s)
        return dict(cold_cap=cap, short_threshold=short, new_head=new_head,
                    demand_max=demand_max, with_warm=with_warm,
                    warm_actual_tail=512, kv_available=10_000_000,
                    plan=plans[-1], trace=result,
                    rows=s._ax_prefix_tracker.rows,
                    effective_deadline=vars(s._ax_admission_cfgs()[0]),
                    effective_risk=vars(s._ax_chain_risk_cfg()))


def risk_cases(env):
    clean = {k: v for k, v in os.environ.items() if not k.startswith('SGLANG_')}
    with patch.dict(os.environ, dict(clean, **env), clear=True):
        s, ns, _ = scheduler([], budget=16384, interval=2)
        ax = ns['ax_deadline']
        dl = ax.deadline_config()
        actual = ax.chain_risk_config(2)
        from dataclasses import replace
        corrected = replace(actual, chunk=16384)
        result = []
        for waited in (4.0, 20.4):
            r = request('cold', shared=250000, tail=0, tag=7)
            row = dict(remaining=200000, waited_s=waited)
            for name, cfg in [('configured_8k', actual), ('model_16k', corrected)]:
                row[name] = dict(projected_s=waited+dl.arrival_offset_s+ax.service_s(200000, cfg.chunk, dl),
                                 interval=ax.chain_risk_interval(r, 200000, waited, dl, cfg))
            result.append(row)
        return result


def rank_risk_cases(env):
    """Run real arming/defer code twice, with rank-local clocks/state.

    This does not emulate a distributed backend or claim a measured TP8 hang.
    It proves the two schedulers can choose different collective-bearing paths.
    """
    clean = {k: v for k, v in os.environ.items() if not k.startswith('SGLANG_')}
    outputs = []
    cases = [('clock_boundary', 249152, 32768, (3.90, 3.92)),
             ('frozen_class_not_broadcast', 160000, 99840, (24.0, 24.0))]
    with patch.dict(os.environ, dict(clean, **env), clear=True):
        for name, prompt, prefix, waits in cases:
            ranks = []
            for rank, waited in enumerate(waits):
                s, ns, _ = scheduler([], budget=16384, interval=2)
                r = request('cold', shared=prompt, tail=0, tag=7)
                dl = s._ax_admission_cfgs()[0]
                if name == 'frozen_class_not_broadcast' and rank == 0:
                    # Only rank 0 runs admission ordering at the initial miss.
                    ns['ax_deadline'].deadline_cold(r, dl)
                r.prefix_indices = [0] * prefix
                r.num_matched_prefix_tokens = prefix
                r.set_extend_range(prefix, prefix + 16384)
                r.time_stats.scheduler_recv_time = 100.0 - waited
                old_time = ns['time']
                calls = []
                def decide(fn):
                    calls.append('rank0_decide')
                    return fn()
                s._ax_rank0_decide = decide
                ns['time'] = NS(perf_counter=lambda: 100.0)
                try:
                    s._arm_prefill_decode_interval(ns['ScheduleBatch']([r]))
                    interval = s._prefill_decode_interval_remaining
                    defer = [s._should_defer_prefill(), s._should_defer_prefill()]
                finally:
                    ns['time'] = old_time
                ranks.append(dict(rank=rank, waited_s=waited, interval=interval,
                                  frozen_cold=r._ax_deadline_cold,
                                  defer_next_two=defer, rank0_decisions=calls))
            assert ranks[0]['interval'] != ranks[1]['interval']
            assert all(not r['rank0_decisions'] for r in ranks)
            outputs.append(dict(case=name, prompt=prompt, device_prefix=prefix,
                                current_chunk=16384, ranks=ranks))
    return outputs


def cold_rescue_cases(env):
    """Show the existing complete-waiter-only parking boundary, not a new crash."""
    clean = {k: v for k, v in os.environ.items() if not k.startswith('SGLANG_')}
    results = []
    with patch.dict(os.environ, dict(clean, **env), clear=True):
        for rescue_tokens in (12000, 20000):
            late = request('late', shared=120000, tail=0, tag=7, cached=32768, waited=31)
            head = request('new', shared=rescue_tokens, tail=0, tag=9, waited=.1)
            s, ns, cache = scheduler([head], late, budget=16384, interval=2)
            cache.publish(late, 32768)
            cache.match(late)
            plans = []
            def decide(fn):
                value = fn()
                if isinstance(value, dict) and 'version' in value:
                    plans.append(value)
                return value
            s._ax_rank0_decide = decide
            trace = step(s)
            results.append(dict(new_cold_tokens=rescue_tokens, late_waited_s=31,
                                late_remaining=87232, plan=plans[-1], trace=trace))
    assert results[0]['plan']['park'] and results[0]['trace']['reqs'][0][0] == 'new'
    assert not results[1]['plan']['park'] and results[1]['trace']['reqs'][0][0] == 'late'
    return results


def source_graph():
    selected = {
        'scheduler.py': {'_get_new_batch_prefill_raw', '_ax_admission_plan', '_ax_prefix_plan',
                         '_ax_prefix_admit_ready', '_ax_short_hit_reserve',
                         '_ax_chain_risk_interval', '_ax_mechanism_report',
                         '_arm_prefill_decode_interval', '_should_defer_prefill',
                         'get_next_batch_to_run'},
        'schedule_policy.py': {'_ax_short_hit', 'add_chunked_req', 'add_one_req',
                               '_update_prefill_budget', 'ax_complete_waiter_budget'},
        'ax_prefix_producer.py': {'prepare', 'should_park'},
        'ax_deadline.py': {'tier_order', 'service_s', 'slack_s', 'deadline_cold',
                          'chain_risk_config', 'chain_risk_interval'},
    }
    result = []
    for filename, names in selected.items():
        path = ROOT / 'engine/sglang/srt/managers' / filename
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.FunctionDef) and node.name in names:
                result.append(dict(file=str(path.relative_to(ROOT)), function=node.name,
                                   line=node.lineno, end_line=node.end_lineno,
                                   calls=sorted({ast.unparse(n.func) for n in ast.walk(node)
                                                 if isinstance(n, ast.Call)})))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    text = args.job.read_text()
    env = dict(item.split('=', 1) for item in shlex.split(job_value(text, 'G_ENV')))
    cases = [run_budget_case(env, 16384, 2048), run_budget_case(env, 16384, 8192),
             run_budget_case(env, 14336, 8192), run_budget_case(env, 16384, 8192, True)]
    for c in cases[:2]:
        assert not c['plan']['park'] and c['plan']['ordinary_reserve'] == 0
        assert [r[0] for r in c['trace']['reqs']] == ['cold']
        assert c['trace']['budget'][:2] == [0, 0]
    assert [r[0] for r in cases[2]['trace']['reqs']] == ['cold', 'warm']
    assert cases[3]['plan']['order'] == ['cold', 'warm']
    assert [r[0] for r in cases[3]['trace']['reqs']] == ['cold']
    reuse126 = [run_budget_case(env, 12288, 4096, demand_max=16384),
                run_budget_case(env, 12288, 4096, demand_max=16384, with_warm=False)]
    assert [r[0] for r in reuse126[0]['trace']['reqs']] == ['cold', 'warm']
    assert reuse126[0]['plan']['ordinary_reserve'] == 512
    assert reuse126[1]['trace']['reqs'] == [('cold', 32768, 49152)]
    output = dict(scope='CPU control-flow reproductions; no TP8 latency prediction',
                  source_commit=job_value(text, 'G_COMMIT'), job=str(args.job.resolve()),
                  job_args=job_value(text, 'G_ARGS'), job_env=env,
                  cases=cases, reuse126=reuse126, risk_model=risk_cases(env),
                  rank_risk=rank_risk_cases(env), cold_rescue=cold_rescue_cases(env),
                  source_graph=source_graph())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(budget_cases=[dict(cap=c['cold_cap'], short=c['short_threshold'],
        new_head=c['new_head'], admitted=c['trace']['reqs'], park=c['plan']['park'],
        ordinary_reserve=c['plan']['ordinary_reserve']) for c in cases],
        reuse126=[dict(with_warm=c['with_warm'], reserved=c['plan']['ordinary_reserve'],
                       admitted=c['trace']['reqs']) for c in reuse126],
        risk_model=output['risk_model'], rank_risk=output['rank_risk'],
        cold_rescue=[dict(new_cold=c['new_cold_tokens'], park=c['plan']['park'],
                         admitted=c['trace']['reqs']) for c in output['cold_rescue']]), indent=2))


if __name__ == '__main__':
    main()
