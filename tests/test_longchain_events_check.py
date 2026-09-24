"""Independent negative receipt fixtures; no engine/token-length acceptance."""
import copy
import json
import unittest

from scripts.analysis.longchain_check import (
    _canonical_digest, _new_message_suffix, _new_tool_block_errors, _rebuild_errors,
)


def group(call_id):
    return [
        {"role": "assistant", "content": "checking " * 200,
         "tool_calls": [{"id": call_id, "type": "function",
                         "function": {"name": "calculate", "arguments": {}}}]},
        {"role": "tool", "tool_call_id": call_id, "content": "result " * 200},
    ]


def replacement(before, left, right):
    """Construct an explicit method-v1 receipt independently of the compiler."""
    removed = before["messages"][left:right]
    excerpts = []
    for i, message in enumerate(removed):
        content = message.get("content")
        if isinstance(content, str) and content and message.get("role") != "tool":
            excerpts.append({"removed_index": i, "text": content[:240]})
        if len(excerpts) == 8:
            break
    if not excerpts:
        excerpts = [{"removed_index": 0, "text": json.dumps(removed[0], ensure_ascii=False,
                                                            sort_keys=True, separators=(",", ":"))[:240]}]
    summary = {"role": "user", "content": "历史上下文摘录（中间记录已归档）：\n" +
               "\n".join(e["text"] for e in excerpts) + "\n继续当前任务。"}
    receipt = {"prefix_end": left, "tail_start": right, "removed_messages": len(removed),
               "removed_sha256": _canonical_digest(removed), "summary_message": summary,
               "summary_excerpts": excerpts, "retained_tail_sha256": _canonical_digest(before["messages"][right:]),
               "method": "receiving_history_extractive_summary_v1"}
    after = copy.deepcopy(before)
    after["messages"] = copy.deepcopy(before["messages"][:left]) + [summary] + copy.deepcopy(before["messages"][right:])
    return after, {"kind": "synthetic", "event_kind": "context_reset", "rebuild": receipt}


class LongchainEventsCheckTests(unittest.TestCase):
    def setUp(self):
        self.before = {"system": "fixed", "tools": [], "messages": [
            {"role": "user", "content": "Task input " * 50},
            {"role": "assistant", "content": "analysis " * 1000},
            *group("old-1"),
            {"role": "assistant", "content": "detail " * 1000},
            *group("old-2"),
        ]}
        self.row = {"phase": "context_reset"}
        self.after, self.prov = replacement(self.before, 1, 5)

    def check(self, after=None, prov=None, before=None, row=None):
        return _rebuild_errors(before if before is not None else self.before,
                               after if after is not None else self.after,
                               row if row is not None else self.row,
                               prov if prov is not None else self.prov, "fixture:reset")

    def test_valid_replace_middle_and_complete_tail(self):
        self.assertEqual(self.check(), [])
        self.assertIsNone(_new_message_suffix(self.before["messages"], self.after["messages"]))

    def test_noop_reset_and_wrong_phase_rejected(self):
        self.assertTrue(self.check(after=self.before))
        self.assertTrue(self.check(row={"phase": "intra"}))
        self.assertTrue(self.check(before={}))

    def test_receipt_hash_and_tail_mismatch_rejected(self):
        for field in ("removed_sha256", "retained_tail_sha256"):
            with self.subTest(field=field):
                p = copy.deepcopy(self.prov)
                p["rebuild"][field] = "0" * 64
                self.assertTrue(self.check(prov=p))
        changed = copy.deepcopy(self.after)
        changed["messages"][-1]["content"] = "tail silently changed"
        self.assertTrue(self.check(after=changed))

    def test_tail_cannot_begin_with_orphan_result(self):
        after, prov = replacement(self.before, 1, 6)
        self.assertTrue(self.check(after=after, prov=prov))

    def test_prefix_cannot_end_with_unclosed_call(self):
        # Index 2 assistant call remains; its result at index 3 is removed.
        after, prov = replacement(self.before, 3, 5)
        self.assertTrue(self.check(after=after, prov=prov))

    def test_original_task_prefix_cannot_be_removed(self):
        after, prov = replacement(self.before, 0, 5)
        self.assertTrue(self.check(after=after, prov=prov))

    def test_summary_cannot_inject_text_absent_from_extract_receipt(self):
        changed = copy.deepcopy(self.after)
        p = copy.deepcopy(self.prov)
        p["rebuild"]["summary_message"]["content"] += "\nUNACCOUNTED BLOCK " * 30
        changed["messages"][1] = p["rebuild"]["summary_message"]
        self.assertTrue(self.check(after=changed, prov=p))

    def test_duplicate_summary_excerpt_is_rejected(self):
        p = copy.deepcopy(self.prov)
        p["rebuild"]["summary_excerpts"].append(copy.deepcopy(p["rebuild"]["summary_excerpts"][0]))
        self.assertTrue(self.check(prov=p))

    def test_invalid_excerpt_object_returns_errors_instead_of_crashing(self):
        p = copy.deepcopy(self.prov)
        p["rebuild"]["summary_excerpts"] = [None]
        self.assertTrue(self.check(prov=p))

    def test_malformed_receipts_return_errors(self):
        mutations = [
            ("summary_message", None), ("summary_message", "not an object"),
            ("summary_excerpts", None), ("summary_excerpts", []),
            ("summary_excerpts", [{"removed_index": True, "text": "analysis"}]),
            ("summary_excerpts", [{"removed_index": 0, "text": ""}]),
            ("summary_excerpts", [{"removed_index": 0, "text": "analysis " * 50}]),
            ("summary_excerpts", [{"removed_index": 999, "text": "analysis"}]),
            ("summary_excerpts", [{"removed_index": 0, "text": "absent"}]),
            ("removed_messages", True), ("prefix_end", True), ("tail_start", 999),
            ("method", "unimplemented"),
        ]
        for field, value in mutations:
            with self.subTest(field=field, value=value):
                p = copy.deepcopy(self.prov)
                p["rebuild"][field] = value
                self.assertTrue(self.check(prov=p))

    def test_excerpt_order_and_max_count_are_strict(self):
        p = copy.deepcopy(self.prov)
        p["rebuild"]["summary_excerpts"].reverse()
        self.assertTrue(self.check(prov=p))
        p["rebuild"]["summary_excerpts"] *= 9
        self.assertTrue(self.check(prov=p))

    def test_summary_cannot_introduce_unpaired_call(self):
        changed = copy.deepcopy(self.after)
        p = copy.deepcopy(self.prov)
        p["rebuild"]["summary_message"]["tool_calls"] = group("hidden-new-call")[0]["tool_calls"]
        changed["messages"][1] = p["rebuild"]["summary_message"]
        self.assertTrue(self.check(after=changed, prov=p))

    def test_after_reset_append_uses_new_history_and_does_not_restore_old_snapshot(self):
        messages = self.after["messages"] + group("new-3")
        start = _new_message_suffix(self.after["messages"], messages)
        self.assertEqual(start, len(self.after["messages"]))
        self.assertEqual(_new_tool_block_errors(messages, start, "regrowth")[0], [])
        restored = self.before["messages"] + group("new-3")
        self.assertIsNone(_new_message_suffix(self.after["messages"], restored))

    def test_persistent_tail_summary_is_not_a_replaceable_reminder(self):
        before = copy.deepcopy(self.before)
        before["messages"][1]["content"] = "<system-reminder>" + before["messages"][1]["content"]
        after, prov = replacement(before, 1, len(before["messages"]))
        self.assertEqual(self.check(before=before, after=after, prov=prov), [])
        self.assertIn("system-reminder", after["messages"][-1]["content"])
        dropped = after["messages"][:-1] + group("new-3")
        self.assertIsNone(_new_message_suffix(after["messages"], dropped))
        preserved = after["messages"] + group("new-3")
        self.assertEqual(_new_message_suffix(after["messages"], preserved), len(after["messages"]))

    def test_adaptive_shorter_extracts_preserve_same_receipt_contract(self):
        for length, count in ((120, 4), (32, 1), (1, 1)):
            with self.subTest(length=length, count=count):
                after, prov = replacement(self.before, 1, len(self.before["messages"]))
                rec = prov["rebuild"]
                rec["summary_excerpts"] = rec["summary_excerpts"][:count]
                for ex in rec["summary_excerpts"]:
                    ex["text"] = ex["text"][:length]
                rec["summary_message"]["content"] = "历史上下文摘录（中间记录已归档）：\n" + "\n".join(
                    e["text"] for e in rec["summary_excerpts"]) + "\n继续当前任务。"
                after["messages"][1] = rec["summary_message"]
                self.assertEqual(self.check(after=after, prov=prov), [])

    def test_duplicate_summary_message_in_body_is_rejected(self):
        changed = copy.deepcopy(self.after)
        changed["messages"].insert(2, copy.deepcopy(changed["messages"][1]))
        self.assertTrue(self.check(after=changed))


if __name__ == "__main__":
    unittest.main()
