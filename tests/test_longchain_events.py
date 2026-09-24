"""Independent v2 generation review: history contracts and pressure fallback."""
import copy
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/longchain'))
import longchain as lc
import longchain_events as events


def compiler_stub():
    compiler = events.EventCompiler.__new__(events.EventCompiler)
    compiler.rng = random.Random(7)
    compiler.global_usage = Counter()
    compiler.growths = [10]
    compiler.donors = [lc.Donor('donor', 'other-chain', 'p', 'f',
        [{'role': 'assistant', 'content': 'continue'}], False, 'historical_block',
        'intra', 10, 0, 'fixture', None, None, 10, 'fingerprint', frozenset(), {})]
    compiler.queries = [{'message': {'role': 'user', 'content': 'Another question'},
        'source_req_id': 'other', 'source_session_id': 'other-session',
        'source_chain_id': 'other-chain', 'pack': 'p', 'fingerprint': 'query'}]
    compiler.answers = [{'message': {'role': 'assistant', 'content': 'Prior result'},
        'source_req_id': 'other', 'source_session_id': 'other-session',
        'source_chain_id': 'other-chain', 'pack': 'p', 'fingerprint': 'answer'}]
    return compiler


def event_item(kind):
    return {'kind': kind, 'position_origin': 'fixture', 'template_boundary': False,
            'reference': {'gap_s': 0, 'prompt_delta_source': 10}}


@pytest.mark.parametrize('kind', ['intra', 'turn_start'])
def test_rebuild_summary_survives_next_nonreset_event(kind):
    original = {'system': '', 'tools': [], 'messages': [
        {'role': 'user', 'content': 'Original task'},
        {'role': 'assistant', 'content': 'important prior context ' * 80}]}
    frozen = copy.deepcopy(original)
    rebuilt, receipt = events.rebuild_history(original, .05)
    assert receipt['tail_start'] == len(original['messages'])
    target = {'pack': 'p', 'chain_id': 'c', 'session_id': 'receiver', 'sys_tools_hash': 'f'}
    result, _, _ = compiler_stub().event(event_item(kind), target, rebuilt, 'test', 3, Counter(), 100000)
    assert result['messages'][:len(rebuilt['messages'])] == rebuilt['messages']
    assert original == frozen


def test_rebuild_preserves_recent_tool_group_and_input_immutability():
    messages = [{'role': 'user', 'content': 'Task'},
                {'role': 'assistant', 'content': 'old intermediate history ' * 200},
                {'role': 'assistant', 'content': '', 'tool_calls': [
                    {'id': 'call', 'function': {'name': 'Read', 'arguments': '{}'}}]},
                {'role': 'tool', 'tool_call_id': 'call', 'content': 'recent result'}]
    body = {'system': 'shared', 'tools': [{'name': 'Read'}], 'messages': messages}
    snapshot = copy.deepcopy(body)
    rebuilt, receipt = events.rebuild_history(body, .4)
    assert rebuilt['messages'][-2:] == messages[-2:]
    assert lc.tool_pairing(rebuilt['messages']) != 'incomplete'
    assert receipt['tail_start'] == 2
    rebuilt['messages'][-1]['content'] = 'changed test output'
    assert body == snapshot


@pytest.mark.parametrize('history_chars, expected_adjustment', [(500, None), (180, 120), (70, 32), (35, 1)])
def test_build_handles_pressure_and_short_history_rebuild(tmp_path, monkeypatch, history_chars, expected_adjustment):
    """Pressure and short-history summaries use one explicit accepted reset."""
    import argparse
    import gzip
    sys.path.insert(0, str(ROOT / 's1-dev/harness'))
    import s1_common as common

    class CharacterRenderer:
        def __init__(self, *_): self.tokenizer = self
        def render(self, body): return ''.join(m.get('content', '') for m in body['messages'])
        def encode(self, text, **_): return list(map(ord, text))
        def n_tokens(self, text): return len(text)

    monkeypatch.setattr(common, 'Renderer', CharacterRenderer)
    monkeypatch.setattr(lc, 'implementation_receipt', lambda *_: {'fixture': True})
    source = tmp_path / 'source'
    (source / 'bodies').mkdir(parents=True)
    tok = tmp_path / 'tok'; tok.mkdir()
    body0 = {'req_id': 'p:canon:a', 'system': '', 'tools': [],
             'messages': [{'role': 'user', 'content': 'Q'}]}
    body1 = copy.deepcopy(body0)
    body1['req_id'] = 'p:canon:b'
    body1['messages'].append({'role': 'assistant', 'content': 'x' * history_chars})
    if history_chars == 180:
        body1['messages'].append({'role': 'assistant', 'content': 'recent tail ' * 7})
    visible_chars = len(CharacterRenderer().render(body1))
    row0 = {'pack': 'p', 'view': 'canon', 'session_id': 's', 'chain_id': 'c', 'chain_index': 0,
            'logical_call_id': 'a', 'in_serving_load': True, 'phase': 'session_start',
            'dispatch_offset_ms': 0, 'end_offset_ms': 1, 'glm_tokens': 1,
            'glm_lcp_with_prev': 0, 'uncached_expected': 1, 'max_output_i': 10,
            'gap_valid': True, 'replay_gap_ms': 0, 'sys_tools_hash': 'f', 'body_ref': 'bodies/a.jsonl.gz'}
    row1 = dict(row0, logical_call_id='b', dispatch_offset_ms=2, end_offset_ms=3,
                phase='intra', glm_tokens=visible_chars, glm_lcp_with_prev=1, uncached_expected=visible_chars - 1)
    phases = {'session_start': 1, 'intra': 2} if expected_adjustment is None else {'session_start': 1, 'intra': 1, 'context_reset': 1}
    context_limit = 650 if expected_adjustment is None else 1000
    chain = {'view': 'canon', 'pack': 'p', 'chain_id': 'c', 'session_id': 's', 'chain_index': 0,
             'n_requests': 3, 'sum_glm_tokens': 2 * history_chars + 3, 'sum_uncached_expected': 2 * history_chars + 1,
             'max_output_i_sum': 120, 'phases': phases, 'sys_tools_hash': 'f'}
    def lines(path, values):
        path.write_text(''.join(json.dumps(v) + '\n' for v in values))
    lines(source / 'requests.jsonl', [row0, row1]); lines(source / 'chains.jsonl', [chain])
    with gzip.open(source / 'bodies/a.jsonl.gz', 'wt') as f:
        for body in (body0, body1): f.write(json.dumps(body) + '\n')
    observation = {'session_id': 'observed', 'run_id': 'r', 'previous_span': 'x', 'current_span': 'y',
       'same_system_hash': True, 'same_tools_hash': True, 'model_changed': False,
       'gap_s': 1, 'prompt_delta_source': 10, 'weight': 1, 'next_output_tokens_source': 10,
       'event_observation': 'reported_compression', 'before_prompt_tokens_source': 1000,
       'after_prompt_tokens_source': 450 if history_chars == 180 else 100}
    profile = tmp_path / 'profile.jsonl'; lines(profile, [observation])
    out = tmp_path / 'out'
    event_attempts = []
    rendered_attempts = []
    original_event = events.EventCompiler.event
    def tracked_event(self, item, *args, **kwargs):
        event_attempts.append(copy.deepcopy(item))
        result = original_event(self, item, *args, **kwargs)
        rendered_attempts.append((len(CharacterRenderer().render(result[0])), copy.deepcopy(result[1])))
        return result
    monkeypatch.setattr(events.EventCompiler, 'event', tracked_event)
    commits = []
    original_commit = events.EventCompiler.commit
    def tracked_commit(self, receipt, usage):
        commits.append(receipt['event_kind'])
        return original_commit(self, receipt, usage)
    monkeypatch.setattr(events.EventCompiler, 'commit', tracked_commit)
    lc.build(argparse.Namespace(source_root=str(source), out=str(out), harness_dir=str(ROOT / 's1-dev/harness'),
        tok_dir=str(tok), chains=1, seed=3, set='probe', max_context_tokens=context_limit, behavior_profile=str(profile)))
    rows = list(lc.read_jsonl(out / 'requests.jsonl'))
    assert len(rows) == 3
    assert rows[-1]['phase'] == 'context_reset'
    assert rows[-1]['glm_tokens'] + rows[-1]['max_output_i'] <= context_limit
    assert rows[-1]['glm_tokens'] < row1['glm_tokens']
    assert rows[-1]['max_output_i'] == 100
    assert rows[0]['max_output_i'] == rows[1]['max_output_i'] == 10
    provenance = list(lc.read_jsonl(out / 'provenance.jsonl'))
    if expected_adjustment is None:
        assert provenance[-1]['displaced_planned_kind'] == 'intra'
    else:
        assert rendered_attempts[0][0] > row1['glm_tokens']
        if history_chars == 180:
            assert rendered_attempts[0][1]['rebuild']['tail_start'] == 2
            assert provenance[-1]['rebuild']['tail_start'] == len(body1['messages'])
        assert 'displaced_planned_kind' not in provenance[-1]
        assert provenance[-1]['event_kind'] == 'context_reset'
        assert provenance[-1]['rebuild_render_adjustment'] == {
            'keep_fraction': 0, 'excerpt_chars': expected_adjustment,
            'excerpt_count': 4 if expected_adjustment == 120 else 1}
        expected_limits = [120, 32, 1][: [120, 32, 1].index(expected_adjustment) + 1]
        assert [a['rebuild_limits']['excerpt_chars'] for a in event_attempts if 'rebuild_limits' in a] == expected_limits
        assert all(a['kind'] == 'context_reset' for a in event_attempts)
        assert all(a['reference'] == event_attempts[0]['reference'] for a in event_attempts)
    generated_bodies = list(lc.read_jsonl(out / 'bodies/probe.jsonl.gz'))
    assert generated_bodies[:2] == [body0, body1]
    assert generated_bodies[-1]['messages'][0] == body0['messages'][0]
    assert commits == ['context_reset']
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['chain_summaries'][0]['unique_donor_blocks'] == 0


def test_plan_counts_missing_events_once_across_long_template_stitches():
    compiler = compiler_stub()
    ref = {'weight': 1, 'event_observation': 'continuation_no_observed_rebuild',
           'next_output_tokens_source': 10}
    reset = dict(ref, event_observation='reported_compression')
    compiler.segments = [[ref, reset, ref]]
    compiler.compressions = [reset]
    compiler.outputs = [10]
    target = {'n_requests': 100, 'phases': {'session_start': 1, 'intra': 90, 'turn_start': 6, 'context_reset': 3}}
    original = [{'phase': 'session_start'}, {'phase': 'intra'}, {'phase': 'turn_start'}, {'phase': 'context_reset'}]
    plan = compiler.plan(target, original, compiler.donors)
    assert len(plan) == 96
    assert Counter(p['kind'] for p in plan) == {'intra': 89, 'turn_start': 5, 'context_reset': 2}
    assert all(p['output_weight'] > 0 for p in plan)
    assert any(p['template_boundary'] for p in plan[1:])


def test_cross_session_query_keeps_receiver_history_and_records_source():
    compiler = compiler_stub()
    compiler.queries.insert(0, dict(compiler.queries[0], source_session_id='receiver', fingerprint='exclude-self'))
    body = {'system': 'receiver-system', 'tools': [], 'messages': [{'role': 'user', 'content': 'Receiver task'}]}
    original = copy.deepcopy(body)
    target = {'pack': 'p', 'chain_id': 'c', 'session_id': 'receiver', 'sys_tools_hash': 'f'}
    result, receipt, _ = compiler.event(event_item('turn_start'), target, body, 'test', 2, Counter(), 100000)
    assert result['system'] == 'receiver-system'
    assert result['messages'][:1] == body['messages']
    assert result['messages'][-1]['role'] == 'user'
    assert receipt['query_material']['source_session_id'] == 'other-session'
    assert body == original


@pytest.mark.parametrize('kind', ['intra', 'turn_start'])
def test_rejected_candidate_does_not_consume_material_and_acceptance_counts_once(kind):
    compiler = compiler_stub()
    target = {'pack': 'p', 'chain_id': 'c', 'session_id': 'receiver', 'sys_tools_hash': 'f'}
    body = {'system': '', 'tools': [], 'messages': [{'role': 'user', 'content': 'Task'}]}
    usage = Counter()
    # Constructing then discarding a render candidate is not a material use.
    compiler.event(event_item(kind), target, body, 'discarded', 1, usage, 100000)
    assert compiler.global_usage == Counter()
    assert usage == Counter()
    _, accepted, _ = compiler.event(event_item(kind), target, body, 'accepted', 2, usage, 100000)
    compiler.commit(accepted, usage)
    if kind == 'intra':
        assert compiler.global_usage == {'fingerprint': 1}
        assert usage == {'fingerprint': 1}
        assert accepted['donor_reuse_in_chain'] == 0
    else:
        assert compiler.global_usage == {'query': 1, 'answer': 1}
        assert usage == Counter()
    # Replacing a discarded candidate by an accepted reset does not add usage.
    compiler.commit({'event_kind': 'context_reset'}, usage)
    assert sum(compiler.global_usage.values()) == (1 if kind == 'intra' else 2)


@pytest.mark.parametrize('kind', ['intra', 'turn_start'])
def test_rebuild_summary_with_quoted_reminder_is_persistent(kind):
    """An extract can quote old reminder text without becoming transient itself."""
    compiler = compiler_stub()
    body = {'system': '', 'tools': [], 'messages': [
        {'role': 'user', 'content': 'Task'},
        {'role': 'assistant', 'content': 'older result ' * 100},
        {'role': 'user', 'content': '<system-reminder>Original controller note</system-reminder>'},
        {'role': 'assistant', 'content': 'later result ' * 100}]}
    rebuilt, receipt = events.rebuild_history(body, .05)
    assert receipt['tail_start'] == len(body['messages'])
    assert 'system-reminder' in rebuilt['messages'][-1]['content']
    target = {'pack': 'p', 'chain_id': 'c', 'session_id': 'receiver', 'sys_tools_hash': 'f'}
    result, _, _ = compiler.event(event_item(kind), target, rebuilt, 'test', 3, Counter(), 100000)
    assert result['messages'][:len(rebuilt['messages'])] == rebuilt['messages']
