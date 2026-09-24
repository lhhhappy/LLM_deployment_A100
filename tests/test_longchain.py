"""Tests of irreversible data mistakes, not a mirror of synthetic statistics."""
import importlib.util
from pathlib import Path
import sys

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/longchain/longchain.py"
spec = importlib.util.spec_from_file_location("longchain_build", PATH)
lc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = lc
spec.loader.exec_module(lc)


def test_reminder_replace_does_not_authorize_old_history_rewrite():
    user = {"role": "user", "content": "real user question"}
    reminder = {"role": "user", "content": "<system-reminder>old</system-reminder>"}
    assistant = {"role": "assistant", "content": "next"}
    assert lc.transition_messages([user, reminder], [user, assistant]) == ([assistant], True)
    assert lc.transition_messages([user], [assistant]) is None
    assert lc.transition_messages([user, reminder], [dict(user, content="changed"), assistant]) is None


def test_new_call_ids_and_embedded_references_move_together_without_mutating_source():
    messages = [{"role": "assistant", "tool_calls": [{"id": "old-call", "function": {"name": "Read", "arguments": {"path": "old-call.json"}}}]},
                {"role": "tool", "tool_call_id": "old-call", "content": "see old-call.json"}]
    actual = lc.rename_new_calls(messages, "new")
    assert actual[0]["tool_calls"][0]["id"] == "new_0"
    assert actual[1]["tool_call_id"] == "new_0"
    assert actual[1]["content"] == "see new_0.json"
    assert messages[1]["tool_call_id"] == "old-call"


def test_donor_filter_rejects_missing_results_but_understands_bundled_outputs():
    call = {"role": "assistant", "tool_calls": [{"id": "a"}, {"id": "b"}]}
    partial = [call, {"role": "tool", "tool_call_id": "a", "content": "A"}]
    assert lc.tool_pairing(partial) == "incomplete"
    complete = [call, {"role": "tool", "content": [{"tool_call_id": "b", "output": "B"}, {"tool_call_id": "a", "output": "A"}]}]
    assert lc.tool_pairing(complete) == "explicit_or_no_tools"
    assert lc.tool_pairing([call, {"role": "tool", "content": "A"}, {"role": "tool", "content": "B"}]) == "implicit_order"
    assert lc.tool_pairing([{"role": "tool", "content": "orphan"}]) == "incomplete"


def test_short_and_alternative_call_ids_preserve_roles_tools_and_prose():
    messages = [{"role": "assistant", "tool_calls": [
        {"tool_call_id": "a", "type": "function", "function": {"name": "data", "arguments": "a.json"}}]},
        {"role": "tool", "tool_call_id": "a", "content": "data from a.json"}]
    actual = lc.rename_new_calls(messages, "new")
    assert actual[0]["role"] == "assistant"
    assert actual[0]["tool_calls"][0]["function"] == {"name": "data", "arguments": "new_0.json"}
    assert actual[0]["tool_calls"][0]["tool_call_id"] == actual[1]["tool_call_id"] == "new_0"
    assert actual[1]["content"] == "data from new_0.json"
    assert messages[0]["tool_calls"][0]["tool_call_id"] == "a"


def test_call_id_rewrite_is_single_pass_and_rejects_duplicate_definitions():
    block = [{"role": "assistant", "tool_calls": [{"id": "old"}, {"id": "new_0"}]},
             {"role": "tool", "content": [{"tool_call_id": "old", "output": "old.json"},
                                            {"tool_call_id": "new_0", "output": "new_0.json"}]}]
    result = lc.rename_new_calls(block, "new")
    assert [c["id"] for c in result[0]["tool_calls"]] == ["new_0", "new_1"]
    assert [x["output"] for x in result[1]["content"]] == ["new_0.json", "new_1.json"]
    with pytest.raises(ValueError, match="duplicate"):
        lc.rename_new_calls([{"tool_calls": [{"id": "x"}, {"id": "x"}]}], "new")


def test_polish_rejects_tampered_or_unlisted_parent_file(tmp_path):
    path = tmp_path / "requests.jsonl"
    path.write_text('{"max_output_i":500}\n')
    manifest = {"artifacts": {path.name: lc.file_digest(path)}}
    lc.verify_parent_artifacts(tmp_path, manifest)
    path.write_text('{"max_output_i":2}\n')
    with pytest.raises(ValueError, match="hash/path mismatch"):
        lc.verify_parent_artifacts(tmp_path, manifest)
    manifest["artifacts"][path.name] = lc.file_digest(path)
    (tmp_path / "unlisted.json").write_text('{}')
    with pytest.raises(ValueError, match="inventory"):
        lc.verify_parent_artifacts(tmp_path, manifest)


def test_call_id_refs_in_argument_fields_named_like_metadata_are_rewritten():
    block = [{"role": "assistant", "tool_calls": [{"id": "old", "function": {
        "name": "old", "arguments": {"name": "old", "type": "old", "role": "old"}}}]}]
    result = lc.rename_new_calls(block, "new")[0]
    assert result["role"] == "assistant"
    assert result["tool_calls"][0]["function"]["name"] == "old"
    assert set(result["tool_calls"][0]["function"]["arguments"].values()) == {"new_0"}


def test_synthetic_order_respects_last_observed_end_without_guessing_later_latency():
    assert lc.continuation_order_offset({"dispatch_offset_ms": 100, "end_offset_ms": 1000}, 50) == 1050
    assert lc.continuation_order_offset({"dispatch_offset_ms": 1050, "end_offset_ms": None}, 50) == 1100
    assert lc.continuation_order_offset({"dispatch_offset_ms": 0, "end_offset_ms": None}, 0) == 1


def test_call_id_does_not_rewrite_structured_content_part_type():
    block = [{"role": "assistant", "tool_calls": [{"id": "text", "function": {"name": "Read"}}]},
             {"role": "tool", "tool_call_id": "text", "content": [{"type": "text", "text": "text.json"}]}]
    result = lc.rename_new_calls(block, "new")
    assert result[1]["content"][0] == {"type": "text", "text": "new_0.json"}


def test_original_output_budgets_never_shrunk_to_fit_a_target():
    originals = [{"max_output_i": 800}]
    budgets, residual = lc.output_budgets(1800, originals, [1, 3])
    assert sum(budgets) == 1000 and budgets[1] > budgets[0] and residual == 0
    assert originals == [{"max_output_i": 800}]
    with pytest.raises(ValueError):
        lc.output_budgets(801, originals, [1])


def test_chain_selection_retains_full_lengths_and_is_reproducible():
    chains = {str(i): {"pack": "biomaster", "n_requests": n, "sum_glm_tokens": n*10000,
                       "max_output_i_sum": n*500, "sys_tools_hash": f"family{i//4}"}
              for i, n in enumerate([1]*8+[8]*8+[60]*8+[200]*8)}
    selected = lc.choose_chains(chains, 16, 42)
    assert selected == lc.choose_chains(chains, 16, 42)
    assert len(set(selected)) == 16
    assert sorted(chains[c]["n_requests"] for c in selected) == [1]*4+[8]*4+[60]*4+[200]*4


def test_token_lcp_handles_divergence_and_exact_prefix():
    assert lc.lcp([1,2,3], [1,2,4,5]) == 2
    assert lc.lcp([1,2], [1,2,3]) == 2
    assert lc.lcp([], [1]) == 0


def test_phase_follows_the_constructed_event_not_a_donor_reset_label():
    tool_continuation = [{"role": "assistant", "content": "continue"},
                         {"role": "tool", "content": "result"},
                         {"role": "user", "content": "<system-reminder>next</system-reminder>"}]
    assert lc.continuation_phase(tool_continuation) == "intra"
    # A new user question is a turn even if almost the entire prefix is shared.
    assert lc.continuation_phase(tool_continuation + [{"role": "user", "content": "Why?"}]) == "turn_start"


def test_chain_aggregates_describe_output_instead_of_inheriting_source_counts():
    source = {"append_only_edges": 99, "append_only_frac": 0.7, "first_dispatch_offset_ms": 1}
    row = {"chain_id": "c", "pack": "p", "view": "canon", "session_id": "s", "chain_index": 0,
           "glm_tokens": 100, "uncached_expected": 20, "max_output_i": 50, "phase": "intra",
           "dispatch_offset_ms": 42, "end_offset_ms": 45}
    result = lc.chain_record(source, [row, dict(row, phase="turn_start", end_offset_ms=None)], 1)
    assert result["append_only_edges"] == result["total_edges"] == 1
    assert result["append_only_frac"] == 1
    assert result["first_dispatch_offset_ms"] == 42 and result["last_end_offset_ms"] is None
    assert result["phases"] == {"intra": 1, "turn_start": 1}
    assert result["sum_glm_tokens"] == 200 and result["max_output_i_sum"] == 100
    assert result["source_chain_targets"] == source


def test_polish_propagates_only_while_the_exact_message_survives():
    old = {"role": "user", "content": "<system-reminder>check files</system-reminder>"}
    edit = {"message_index": 1, "message_sha256": lc.digest(old), "replacement_content": "<system-reminder>check the saved table</system-reminder>", "edit_id": "e1"}
    source = {"messages": [{"role": "user", "content": "question"}, old]}
    changed, active = lc.apply_active_edits(source, {}, [edit])
    later = {"messages": source["messages"] + [{"role": "assistant", "content": "answer"}]}
    propagated, active = lc.apply_active_edits(later, active, [])
    assert changed["messages"][1] == propagated["messages"][1]
    replaced = {"messages": [source["messages"][0], {"role": "assistant", "content": "different turn"}]}
    unchanged, active = lc.apply_active_edits(replaced, active, [])
    assert not active and unchanged == replaced
    assert source["messages"][1] == old


def test_polish_cannot_rewrite_retained_original_history_or_tool_results():
    rows = {"p:canon:0": {"pack": "p", "view": "canon", "chain_id": "c", "logical_call_id": "0", "dispatch_offset_ms": 0},
            "p:canon:1": {"pack": "p", "view": "canon", "chain_id": "c", "logical_call_id": "1", "dispatch_offset_ms": 1}}
    user = {"role": "user", "content": "old question"}
    bodies = {"p:canon:0": {"messages": [user]}, "p:canon:1": {"messages": [user, {"role": "tool", "content": "frozen result"}]}}
    provenance = {"p:canon:0": {"kind": "original"}, "p:canon:1": {"kind": "synthetic"}}
    grouped = {"c": list(rows.values())}
    for rid, index, content in [("p:canon:0",0,"old question"),("p:canon:1",0,"old question"),("p:canon:1",1,"frozen result")]:
        with pytest.raises(ValueError):
            lc.prepare_edits([{"req_id": rid, "message_index": index, "expected_content": content,
                               "replacement_content": "modified"}], rows, bodies, grouped, provenance)
    edit = {"scope": "chain_seed_query", "chain_id": "c", "req_id": "p:canon:0", "message_index": 0,
            "expected_content": "old question", "replacement_content": "new synthetic task"}
    assert lc.prepare_edits([edit], rows, bodies, grouped, provenance)["p:canon:0"][0]["scope"] == "chain_seed_query"


def test_polish_does_not_transfer_an_empty_assistant_edit_to_another_call():
    old = {"role": "assistant", "content": "", "tool_calls": [{"id": "one"}]}
    other = {"role": "assistant", "content": "", "tool_calls": [{"id": "two"}]}
    edit = {"message_index": 0, "message_sha256": lc.digest(old), "replacement_content": "explain one", "edit_id": "e"}
    _, active = lc.apply_active_edits({"messages": [old]}, {}, [edit])
    result, active = lc.apply_active_edits({"messages": [other]}, active, [])
    assert not active and result["messages"][0]["content"] == ""
