#!/usr/bin/env python3
"""Independently audit frozen TP2 acceptance/restore diagnostics on CPU.

ROOT contains the original mtp_tp2_*13 run directories. No model is loaded.
Torch files must come from the trusted project diagnostic producer.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def audit(root):
    hashes = {}

    def read(path):
        data = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(data).hexdigest()
        return data

    def js(path):
        return json.loads(read(path))

    def trace(path):
        read(path)
        return torch.load(path, map_location="cpu", weights_only=False)

    oracle_path = root / "mtp_tp2_ref13/responses.json"
    oracle = {r["case"]: r for r in js(oracle_path)}
    acceptance = {}
    for name in ("accept_ref", "accept_dcp", "compact"):
        run = root / f"mtp_tp2_{name}13"
        for response in js(run / "responses.json"):
            gold = oracle[response["case"]]
            assert response["prompt_sha256"] == gold["prompt_sha256"]
            assert response["result"]["output_ids"] == gold["result"]["output_ids"]
        for rank in range(2):
            events = [json.loads(line) for line in read(run / f"acceptance_rank{rank}.jsonl").splitlines()]
            cases, owners, unchecked, crossing = {}, set(), 0, {256: 0, 512: 0}
            for event in events:
                assert event["rank"] == rank
                locs, width = event["verify_virtual_locs"], event["dcp_size"]
                assert event["verify_owners"] == [loc % width for loc in locs]
                if not event["checked"]:
                    unchecked += 1
                    continue
                case, offset, wanted = event["case"], event["offset"], event["wanted_accept_length"]
                gold = oracle[case]["result"]["output_ids"]
                assert event["oracle_sha256"] == hashes[str(oracle_path)]
                assert event["actual_accept_lengths"] == [wanted]
                assert event["actual_tokens"] == event["expected_tokens"] == gold[offset + 1:offset + wanted + 1]
                assert event["proposals"][:wanted - 1] == gold[offset + 1:offset + wanted]
                if wanted < 4:
                    assert event["proposals"][wanted - 1] != gold[offset + wanted]
                cases.setdefault(case, set()).add(wanted)
                owners.update(event["verify_owners"])
                for boundary in crossing:
                    crossing[boundary] += len({loc // boundary for loc in locs}) > 1
            assert set(cases) == set(oracle) and all(v == {1, 2, 3, 4} for v in cases.values())
            acceptance[f"{name}:rank{rank}"] = {
                "checked": len(events) - unchecked, "unchecked_tail": unchecked,
                "cases": {k: sorted(v) for k, v in cases.items()},
                "owners": sorted(owners), "windows_crossing_virtual_boundary": crossing,
            }

    def selected(record):
        values = record["attn"].float()
        if record["mode"] != "DRAFT_EXTEND_V2":
            return values
        counts, width = record["accepted_tokens"], record["draft_window_width"]
        full = (torch.arange(width)[None, :] < counts[:, None] + record["num_front_tokens"]).reshape(-1)
        mask = full[record["rows"]]
        assert torch.equal(mask, record["valid_rows"])
        return values[mask]

    restore = {}
    for name in ("host_ref", "host_dcp"):
        run = root / f"mtp_tp2_{name}13"
        responses = {r["case"]: r["result"] for r in js(run / "responses.json")}
        warm, host = (responses[k] for k in ("device_repeat", "host_restore"))
        assert warm["output_ids"] == host["output_ids"]
        assert warm["meta_info"]["cached_tokens_details"] == {"device": 4096, "host": 0}
        assert host["meta_info"]["cached_tokens_details"] == {"device": 0, "host": 4096}
        flush = js(run / "flush.json")
        assert flush["flush_result"]["success"] and flush["after_flush_meta"]["cached_tokens"] == 0
        churn = js(run / "churn.json")
        assert len(churn) == 32 and sum(r["tokens"] for r in churn) == 131072
        records = {"device_repeat": [], "host_restore": []}
        for path in sorted((run / "trace").glob("*.pt")):
            record = trace(path)
            if record["case"] in records:
                records[record["case"]].append((path.name, record))
        assert len(records["device_repeat"]) == len(records["host_restore"]) > 0
        checks = []
        for (ref_name, ref), (test_name, test) in zip(*records.values()):
            assert all(test[k] == ref[k] for k in ("role", "rank", "mode", "layer", "shape", "rows"))
            assert all(torch.equal(test[k], ref[k]) for k in ("positions", "seq_lens"))
            x, y = selected(test), selected(ref)
            assert x.shape == y.shape and torch.isfinite(x).all() and torch.isfinite(y).all()
            assert y.abs().max() > 1e-6
            rel = float(((x - y).abs().flatten(1).amax(1) / y.abs().flatten(1).amax(1).clamp_min(1e-30)).max())
            assert rel <= .01
            checks.append({"test_file": test_name, "ref_file": ref_name, "max_row_relative_linf": rel})
        restore[name] = {"checked_pairs": len(checks), "max_row_relative_linf": max(r["max_row_relative_linf"] for r in checks), "records": checks}
    return {"passed": True, "validity": "CPU_AUDIT_OF_TP2_DIAGNOSTICS", "acceptance": acceptance, "restore": restore, "source_sha256": hashes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"], "acceptance": result["acceptance"], "restore": {k: {a: b for a, b in v.items() if a != "records"} for k, v in result["restore"].items()}}, indent=2))
