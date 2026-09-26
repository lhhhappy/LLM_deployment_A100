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


TEMPLATE = {'pack': 'p', 'logical_call_id': 'public', 'phase': 'intra', 'uncached_expected': 10,
            'glm_tokens': 100, 'replay_gap_ms': 0, 'max_output_i': 10}


def compiler_stub():
    compiler = events.EventCompiler.__new__(events.EventCompiler)
    compiler.rng = random.Random(7)
    compiler.global_usage = Counter()
    compiler.templates = {(kind, rewrite): [dict(TEMPLATE, phase=kind)]
                          for kind in ('intra', 'turn_start', 'context_reset') for rewrite in (False, True)}
    compiler.tokens_per_char = 1.0
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
    return {'kind': kind, 'position_origin': 'fixture', 'template': dict(TEMPLATE, phase=kind)}


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


@pytest.mark.parametrize('history_chars', [500, 180, 70, 35])
def test_build_handles_pressure_and_short_history_rebuild(tmp_path, monkeypatch, history_chars):
    """Context pressure forces one explicit rebuild; a short history tightens its summary until it shrinks."""
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
            'gap_valid': True, 'replay_gap_ms': 0, 'sys_tools_hash': 'f', 'body_ref': 'bodies/a.jsonl.gz',
            'edge_type': 'chain-head'}
    row1 = dict(row0, logical_call_id='b', dispatch_offset_ms=2, end_offset_ms=3, edge_type='append-only',
                phase='intra', glm_tokens=visible_chars, glm_lcp_with_prev=1, uncached_expected=visible_chars - 1)
    phases = {'session_start': 1, 'intra': 2}
    # Room for the public prompt and its output, not for another continuation of the same size.
    context_limit = visible_chars + 100 + 10
    chain = {'view': 'canon', 'pack': 'p', 'chain_id': 'c', 'session_id': 's', 'chain_index': 0,
             'n_requests': 3, 'sum_glm_tokens': 2 * history_chars + 3, 'sum_uncached_expected': 2 * history_chars + 1,
             'max_output_i_sum': 120, 'phases': phases, 'sys_tools_hash': 'f'}
    def lines(path, values):
        path.write_text(''.join(json.dumps(v) + '\n' for v in values))
    lines(source / 'requests.jsonl', [row0, row1]); lines(source / 'chains.jsonl', [chain])
    with gzip.open(source / 'bodies/a.jsonl.gz', 'wt') as f:
        for body in (body0, body1): f.write(json.dumps(body) + '\n')
    out = tmp_path / 'out'
    corpus = tmp_path / 'corpus'; corpus.mkdir()
    (corpus / 'notes.md').write_text('corpus line\n' * 100)
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
        tok_dir=str(tok), chains=1, seed=3, set='probe', max_context_tokens=context_limit,
        read_corpus=str(corpus)))
    rows = list(lc.read_jsonl(out / 'requests.jsonl'))
    assert len(rows) == 3
    assert rows[-1]['phase'] == 'context_reset'
    assert rows[-1]['glm_tokens'] + rows[-1]['max_output_i'] <= context_limit
    assert rows[-1]['glm_tokens'] < row1['glm_tokens']
    assert rows[-1]['max_output_i'] == 100
    assert rows[0]['max_output_i'] == rows[1]['max_output_i'] == 10
    provenance = list(lc.read_jsonl(out / 'provenance.jsonl'))
    assert provenance[-1]['displaced_planned_kind'] == 'intra'
    assert provenance[-1]['event_kind'] == 'context_reset' and provenance[-1]['rebuild']
    assert rows[-1]['edge_type'] == 'compact-rebuild'
    # Summary tightening goes 120 -> 32 -> 1 characters and stops at the first rebuild that shrinks.
    tried = [a['rebuild_limits']['excerpt_chars'] for a in event_attempts if 'rebuild_limits' in a]
    assert tried == [120, 32, 1][:len(tried)]
    adjustment = provenance[-1]['rebuild_render_adjustment']
    assert (adjustment or {}).get('excerpt_chars') == (tried[-1] if tried else None)
    if history_chars == 35:
        assert tried, 'a 35-character history cannot hold the default 240-character summary'
    rebuilds = [a for a in event_attempts if a.get('rebuild')]
    assert rebuilds and all(a['kind'] == 'context_reset' for a in rebuilds)
    assert all(a['template'] == rebuilds[0]['template'] for a in rebuilds)
    generated_bodies = list(lc.read_jsonl(out / 'bodies/probe.jsonl.gz'))
    assert generated_bodies[:2] == [body0, body1]
    assert generated_bodies[-1]['messages'][0] == body0['messages'][0]
    assert commits == ['context_reset']
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['chain_summaries'][0]['unique_donor_blocks'] == 0


def test_plan_counts_missing_events_once_and_copies_public_load():
    compiler = compiler_stub()
    compiler.templates['turn_start', False] = compiler.templates['turn_start', True] = [
        dict(TEMPLATE, phase='turn_start', max_output_i=1000)]
    target = {'n_requests': 100, 'phases': {'session_start': 1, 'intra': 90, 'turn_start': 6, 'context_reset': 3}}
    original = [{'phase': 'session_start'}, {'phase': 'intra'}, {'phase': 'turn_start'}, {'phase': 'context_reset'}]
    plan = compiler.plan(target, original, compiler.donors)
    assert len(plan) == 96
    assert Counter(p['kind'] for p in plan) == {'intra': 89, 'turn_start': 5, 'context_reset': 2}
    # Each step carries the load template of its own kind; output weights come from the template.
    assert all(p['template']['phase'] == p['kind'] for p in plan if p['kind'] != 'context_reset')
    assert {p['output_weight'] for p in plan if p['kind'] == 'turn_start'} == {1000}
    assert plan[0]['kind'] != 'context_reset' and plan[-1]['kind'] != 'context_reset'


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


class CharTokenizer:
    """One token per character, so token budgets in these tests are exact character counts."""
    def encode(self, text, **_): return list(text)


def narrated_call(call_id, narration='Let me check that.'):
    return [{'role': 'assistant', 'content': narration, 'tool_calls': [
                {'id': call_id, 'type': 'function', 'function': {'name': 'Bash', 'arguments': {'cmd': 'ls'}}}]},
            {'role': 'tool', 'tool_call_id': call_id, 'content': '{"ok": true, "stdout": "' + 'x' * 300 + '"}'}]


# The public Read schema: pagination is required but nullable.
READ_SCHEMA = {'required': ['source', 'target', 'offset', 'limit'],
               'properties': {'source': {'type': 'string'}, 'target': {'type': 'string'},
                              'offset': {'anyOf': [{'type': 'integer'}, {'type': 'null'}]},
                              'limit': {'anyOf': [{'type': 'integer'}, {'type': 'null'}]}}}


def turn_history():
    return [{'role': 'user', 'content': 'First task'}, *narrated_call('a'),
            {'role': 'assistant', 'content': 'First answer'},
            {'role': 'user', 'content': 'Second task'}, *narrated_call('b'), *narrated_call('c'),
            {'role': 'user', 'content': '<system-reminder>runtime note</system-reminder>'}]


def test_turn_start_closes_only_the_finished_turn_and_checker_rederives_it():
    from scripts.longchain.longchain_check import _rewrite_start
    compiler = compiler_stub()
    body = {'system': '', 'tools': [], 'messages': turn_history()}
    target = {'pack': 'p', 'chain_id': 'c', 'session_id': 'receiver', 'sys_tools_hash': 'f'}
    item = event_item('turn_start')
    item['template']['trailing_reminder'] = {'role': 'user', 'content': '<system-reminder>template</system-reminder>'}
    result, receipt, _ = compiler.event(item, target, body, 'test', 2, Counter(), 100000)
    messages = result['messages']
    # The earlier turn keeps its narration; the finished turn's two tool-call messages lose theirs.
    assert messages[1]['content'] == 'Let me check that.'
    assert [messages[i]['content'] for i in (5, 7)] == ['', '']
    assert receipt['turn_close']['stripped_indices'] == [5, 7]
    # The template ends with a runtime reminder, so the session's reminder moves behind the new query.
    assert [m['role'] for m in messages[-3:]] == ['assistant', 'user', 'user']
    assert messages[-1] == body['messages'][-1]
    start, emptied, divergence = _rewrite_start(body['messages'], messages, True)
    assert (start, emptied, divergence) == (9, [5, 7], None)
    # Emptying an earlier turn's narration is not the agent's rewrite.
    wrong = copy.deepcopy(messages)
    wrong[1]['content'] = ''
    assert _rewrite_start(body['messages'], wrong, True) == (None, None, None)


def test_turn_start_diverges_with_one_read_to_reach_its_size(tmp_path):
    from scripts.longchain.longchain_check import _rewrite_start
    (tmp_path / 'notes.md').write_text('corpus line\n' * 2000)
    compiler = compiler_stub()
    compiler.read_corpus = events.ReadCorpus({}, tmp_path, CharTokenizer(), seed=1)
    compiler.tokens_per_char = 1.0
    history = turn_history()
    closed, stripped = events.close_turn(history[:-1])
    block = [{'role': 'assistant', 'content': 'Second answer'}, {'role': 'user', 'content': 'Third task'}, history[-1]]
    tail = compiler.approximate_tokens(block)
    suffix_b = compiler.approximate_tokens(closed[5:])
    starts = [1, 5, 7]  # tool-call messages after the first human message
    closed_from = stripped[0]  # closing the turn already recomputes from here
    def estimated(i):
        suffix = compiler.approximate_tokens(closed[min(i, closed_from):])
        return suffix + tail + min(events.READ_DIVERGE_MAX, max(events.READ_MIN_TOKENS, target - tail - suffix))
    # The divergence lands where the estimated recompute is closest to the target (later on ties).
    target = tail + suffix_b + 1500
    reissued, receipt = compiler.diverge_to_target(closed, stripped, block, target, READ_SCHEMA, 'p_t', force=False)
    d = receipt['diverged_at']
    assert d == min(reversed(starts), key=lambda i: abs(estimated(i) - target)) == 1
    assert events.READ_MIN_TOKENS <= receipt['read_lengthening']['read_target_tokens'] <= events.READ_DIVERGE_MAX
    # A target beyond the whole history does not grow an unbounded Read into the context.
    _, beyond = compiler.diverge_to_target(closed, stripped, block, 10 * target, READ_SCHEMA, 'p_x', force=False)
    assert beyond['diverged_at'] == 1 and beyond['read_lengthening']['read_target_tokens'] == events.READ_DIVERGE_MAX
    # Only the diverging message gains a Read call, whose result follows that message's own result.
    assert reissued[:d] == closed[:d]
    assert [c['function']['name'] for c in reissued[d]['tool_calls']] == ['Bash', 'Read']
    assert reissued[d + 1] == closed[d + 1]
    assert reissued[d + 2]['tool_call_id'] == reissued[d]['tool_calls'][1]['id']
    assert reissued[d + 3:] == closed[d + 2:]
    after = reissued + block
    assert _rewrite_start(history, after, True) == (len(reissued), [5, 7], d)
    # Within a turn narration stays: the same body is not a continuation's rewrite.
    assert _rewrite_start(history, after, False) == (None, None, None)
    # The Read result must directly follow its message's own results.
    misplaced = after[:d + 1] + [after[d + 3], after[d + 2]] + after[d + 4:]
    assert _rewrite_start(history, misplaced, True) == (None, None, None)
    # Any other change in the diverged history is rejected.
    edited = copy.deepcopy(after)
    edited[d + 3]['content'] = 'changed'
    assert _rewrite_start(history, edited, True) == (None, None, None)


def test_read_arguments_follow_the_session_schema():
    # Pagination is sent as null only where the schema allows null.
    assert events.read_arguments(READ_SCHEMA, 'f.md') == {'source': 'path', 'target': 'f.md', 'offset': None, 'limit': None}
    path_schema = {'required': ['path'], 'properties': {'path': {'type': 'string'}, 'offset': {'type': 'number'},
                                                        'limit': {'type': 'number'}}}
    assert events.read_arguments(path_schema, 'f.md') == {'path': 'f.md'}
    assert events.read_arguments({'properties': {}}, 'f.md') is None


def test_plan_rewrites_follow_source_count_without_the_head_edge():
    compiler = compiler_stub()
    # Mid-session chain: total_edges counts the head's incoming edge (a system/tools change here).
    target = {'n_requests': 12, 'total_edges': 12, 'append_only_edges': 7,
              'phases': {'intra': 10, 'turn_start': 2}}
    original = [{'phase': 'intra', 'edge_type': 'system-tools-changed'},
                {'phase': 'turn_start', 'edge_type': 'unexplained-break'}]
    plan = compiler.plan(target, original, compiler.donors)
    # 5 rewrites in the chain, minus the head edge and the public turn start's break: 3 to synthesize,
    # the synthesized turn start first; the reset-free plan keeps every other step an append.
    assert sum(p['rewrite'] for p in plan) == 3
    assert [p['rewrite'] for p in plan if p['kind'] == 'turn_start'] == [True]


def test_read_corpus_reads_forward_without_repeating(tmp_path):
    (tmp_path / 'a.md').write_text('alpha\n' * 200)
    (tmp_path / 'b.md').write_text('beta\n' * 200)
    corpus = events.ReadCorpus({}, tmp_path, CharTokenizer(), seed=1)
    ranges = []
    for tokens in (300, 700, 600):
        text, used = corpus.take(tokens)
        # Newline escapes render as two characters, so escaped length is the token count here.
        assert abs(len(json.dumps(text)[1:-1]) - tokens) <= 20
        ranges += [(u['source'], u['start'], u['end']) for u in used]
    seen = {}
    for source, start, end in ranges:
        assert seen.get(source, 0) == start  # each file continues exactly where it stopped
        seen[source] = end
    assert corpus.wraps == 0
