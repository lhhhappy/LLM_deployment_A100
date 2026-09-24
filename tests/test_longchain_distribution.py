from array import array
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from scripts.analysis.longchain_distribution import blocks, head_comparison, inspect, normalized_block


def tool_group(call_id):
    return [{"role": "assistant", "tool_calls": [{"id": call_id, "function": {"name": "read", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": call_id, "content": "result"}]


class DistributionTests(unittest.TestCase):
    def test_short_id_preserves_words_and_structural_names_only(self):
        original = [{"role": "assistant", "tool_calls": [{"id": "a", "type": "a", "function": {
            "name": "a", "arguments": {"role": "a", "name": "a", "note": "data a call-a a-long"}}}]},
            {"role": "tool", "tool_call_id": "a", "content": "data a"}]
        normalized = normalized_block(original)
        call = normalized[0]["tool_calls"][0]
        self.assertEqual(normalized[0]["role"], "assistant")
        self.assertEqual(call["type"], "a")
        self.assertEqual(call["function"]["name"], "a")
        self.assertEqual(call["function"]["arguments"]["role"], "__call_0__")
        self.assertEqual(call["function"]["arguments"]["name"], "__call_0__")
        self.assertEqual(call["function"]["arguments"]["note"], "data __call_0__ call-a a-long")
        self.assertEqual(normalized[1]["content"], "data __call_0__")
        self.assertEqual(original[1]["content"], "data a")

    def test_renaming_call_ids_does_not_hide_material_clone(self):
        a, b = blocks(tool_group("call_A"))[0], blocks(tool_group("call_B"))[0]
        self.assertNotEqual(a[0], b[0])
        self.assertEqual(a[1], b[1])
        changed = tool_group("call_B")
        changed[1]["content"] = "different result"
        self.assertNotEqual(a[1], blocks(changed)[0][1])

    def test_inherited_source_branch_prefix_is_not_new_clone(self):
        a = dict(tokens=array("I", [1, 2, 3, 4]), preamble_len=2, source_req_id="a", req_id="a", session_id="s", pack="p")
        b = dict(tokens=array("I", [1, 2, 3, 5]), preamble_len=2, source_req_id="b", req_id="b", session_id="s", pack="p")
        result = head_comparison({"a": a, "b": b}, {"a": a, "b": b})
        self.assertEqual(result["pairs_with_added_shared_history"], 0)
        self.assertEqual(result["pairs_from_same_source_session"], 1)
        cloned = dict(b, tokens=a["tokens"])
        result = head_comparison({"a": a, "b": cloned}, {"a": a, "b": b})
        self.assertEqual(result["pairs_with_added_shared_history"], 1)

    def test_snapshot_carry_does_not_recount_tool_group_introduction(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/"data"
            (root/"bodies").mkdir(parents=True)
            rows, bodies, chains = [], [], []
            for chain in ("a", "b"):
                chain_bodies = [[], tool_group("call_"+chain), tool_group("call_"+chain)+[{"role": "user", "content": "next "+chain}]]
                for i, history in enumerate(chain_bodies):
                    logical = chain+str(i)
                    rows.append(dict(pack="p", view="canon", logical_call_id=logical, session_id=chain, chain_id=chain,
                                     dispatch_offset_ms=i, phase="intra", in_serving_load=True, glm_tokens=10+i,
                                     uncached_expected=1, max_output_i=2, replay_gap_ms=0))
                    bodies.append(dict(req_id="p:canon:"+logical, system="same", tools=[], messages=history))
                chains.append(dict(view="canon", chain_id=chain, n_requests=3, phases={"intra": 3}))
            for name, values in (("requests", rows), ("chains", chains)):
                (root/(name+".jsonl")).write_text("".join(json.dumps(v)+"\n" for v in values))
            with gzip.open(root/"bodies/all.jsonl.gz", "wt") as f:
                for b in reversed(bodies):  # Body shard order must not affect transition accounting.
                    f.write(json.dumps(b)+"\n")
            report, _ = inspect(root, None, Path(temp)/"index.sqlite")
            repetition = report["repetition"]
            self.assertEqual(repetition["introduced_block_occurrences"], 4)  # 2 tools + 2 users, not 4 tools
            normalized = repetition["introduced_blocks_call_ids_normalized"]
            self.assertEqual(normalized["cross_session_groups"], 1)
            self.assertEqual(normalized["cross_session_occurrences"], 2)
            self.assertEqual(repetition["introduced_blocks_exact"]["cross_session_groups"], 0)
            self.assertEqual(repetition["complete_prompt_excluding_req_id"]["cross_session_groups"], 1)


if __name__ == "__main__":
    unittest.main()
