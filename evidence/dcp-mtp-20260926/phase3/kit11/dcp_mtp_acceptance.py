"""Diagnostic proposals that exercise real MTP acceptance lengths 1/2/3/4.

Only used by dcp_mtp_probe, never imported by the engine. Proposals come from a
previous natural, greedy reference run. Target logits, the real verifier, KV
writes and KDA commit are untouched. Draft extend rewrites the accepted draft
window using the target's verified tokens and hidden states. This is coverage
of cache/state transitions, not a natural acceptance or performance result.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def install():
    import torch
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.speculative import eagle_worker_common as common
    from sglang.srt.speculative import eagle_worker_v2 as worker

    oracle_path = Path(os.environ["AX_MTP_ACCEPT_ORACLE"])
    summary = json.loads((oracle_path.parent / "summary.json").read_text())
    assert summary["dcp"] == 1 and not summary.get("controlled_proposals", False)
    oracle_bytes = oracle_path.read_bytes()
    oracle = {row["case"]: row for row in json.loads(oracle_bytes)}
    trace = Path(os.environ["AX_MTP_TRACE_DIR"])
    original_build, original_sample = worker.build_eagle_verify_input, common.eagle_sample
    rounds, pending = {}, []

    def build(batch, draft_input, parent_list, top_scores_index, draft_tokens,
              draft_probs, **kwargs):
        if not batch.forward_mode.is_idle() and (trace / "RECORDING").exists():
            assert kwargs["topk"] == 1 and kwargs["num_draft_tokens"] == 4
            assert get_parallel().tp_size == summary.get("tp", 2)
            assert tuple(draft_tokens.shape) == (1, 3), draft_tokens.shape
            label = (trace / "CURRENT_CASE").read_text()
            row = oracle[label]
            gold = row["result"]["output_ids"]
            offset = int(batch.seq_lens.item()) - row["prompt_length"]
            step = rounds.get(label, 0)
            wanted = 1 + step % 4
            assert 0 <= offset, (label, offset)
            # Overlap can launch another round while the last requested token
            # is still being sent to the client. The saved reference ends at
            # max_new_tokens; let that speculative tail run naturally and mark
            # it unchecked. Never invent an oracle past its retained history.
            if offset + wanted >= len(gold):
                assert offset >= len(gold) - 1, (label, offset, wanted)
                pending.append({"case": label, "round": step, "offset": offset,
                                "checked": False, "reason": "past_reference_output_window"})
                return original_build(batch, draft_input, parent_list, top_scores_index,
                                      draft_tokens, draft_probs, **kwargs)
            assert draft_input.bonus_tokens.tolist() == [gold[offset]], (label, offset)
            vocab = kwargs["target_worker"].model_runner.model_config.vocab_size
            proposals = [gold[offset + i + 1] if offset + i + 1 < len(gold)
                         else (gold[-1] + 1) % vocab for i in range(3)]
            if wanted < 4:
                proposals[wanted - 1] = (proposals[wanted - 1] + 1) % vocab
            # Clone the proposals: the graph owns its output buffers. The
            # normal tree builder and its stream handoff still run below.
            draft_tokens = torch.tensor([proposals], device=draft_tokens.device,
                                        dtype=draft_tokens.dtype)
            pending.append({"case": label, "round": step, "offset": offset,
                            "checked": True,
                            "wanted_accept_length": wanted,
                            "proposals": proposals,
                            "expected_tokens": gold[offset + 1:offset + wanted + 1],
                            "oracle_sha256": hashlib.sha256(oracle_bytes).hexdigest()})
            rounds[label] = step + 1
        return original_build(batch, draft_input, parent_list, top_scores_index,
                              draft_tokens, draft_probs, **kwargs)

    def sample(verify_input, batch, logits_output, grammar_mask):
        result = original_sample(verify_input, batch, logits_output, grammar_mask)
        if not batch.forward_mode.is_idle() and (trace / "RECORDING").exists():
            record = pending.pop(0)
            predict, accept_lens, accept_index = result
            actual = accept_lens.tolist()
            tokens = predict[accept_index[0, :actual[0]].long()].tolist()
            parallel = get_parallel()
            locs = batch.out_cache_loc.tolist()
            record.update(actual_accept_lengths=actual, actual_tokens=tokens,
                          verify_virtual_locs=locs,
                          verify_owners=[x % parallel.attn_dcp_size for x in locs],
                          rank=parallel.tp_rank, dcp_size=parallel.attn_dcp_size)
            record["passed"] = (
                actual == [record["wanted_accept_length"]]
                and tokens == record["expected_tokens"]
            ) if record["checked"] else None
            with (trace.parent / f"acceptance_rank{parallel.tp_rank}.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            if record["checked"] and not record["passed"]:
                raise RuntimeError(f"Controlled proposal failed real verification: {record}")
        return result

    worker.build_eagle_verify_input = build
    common.eagle_sample = sample


def verdict(output: Path, responses, oracle_path: Path, tp_size: int):
    oracle = {r["case"]: r for r in json.loads(oracle_path.read_text())}
    checks = []
    for response in responses:
        case = response["case"]
        checks.append({"case": case, "same_greedy_tokens":
                       response["result"]["output_ids"] == oracle[case]["result"]["output_ids"]})
    ranks = {}
    for rank in range(tp_size):
        path = output / f"acceptance_rank{rank}.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        cases = {}
        for case in (r["case"] for r in responses):
            actual = {r["actual_accept_lengths"][0] for r in records
                      if r["case"] == case and r["checked"]}
            cases[case] = {"observed": sorted(actual), "passed": actual == {1, 2, 3, 4}}
        ranks[str(rank)] = {"records": len(records), "cases": cases,
                            "unchecked_tail_rounds": sum(not r["checked"] for r in records),
                            "passed": all(r["passed"] for r in records if r["checked"])
                            and all(c["passed"] for c in cases.values())}
    result = {"validity": "CONTROLLED_PROPOSALS_REAL_VERIFIER",
              "responses": checks, "ranks": ranks,
              "passed": all(c["same_greedy_tokens"] for c in checks)
                        and all(r["passed"] for r in ranks.values()),
              "limitations": "Proposals injected from a natural reference; not an acceptance-rate or speed result."}
    (output / "acceptance_verdict.json").write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise RuntimeError("Acceptance lengths or greedy token histories were not validated")
    return result
