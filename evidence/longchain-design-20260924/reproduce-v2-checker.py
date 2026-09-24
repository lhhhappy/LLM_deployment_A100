#!/usr/bin/env python3
"""Real original-Renderer fixtures for rebuild checker failures; no engine/network."""
import copy
import collections
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tests"), str(REPO / "s1-dev/harness")]
from s1_common import Renderer
from scripts.longchain.longchain_check import check_dataset, _canonical_digest
from test_longchain_events_check import LongchainEventsCheckTests, replacement, group


def write_jl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def run_fixture(renderer, case):
    sample = LongchainEventsCheckTests()
    sample.setUp()
    before, after, p = sample.before, sample.after, sample.prov
    if case == "prefix_unclosed_call":
        after, p = replacement(before, 3, 5)
    elif case == "initial_task_removed":
        after, p = replacement(before, 0, 5)
    elif case == "unaccounted_summary_text":
        p["rebuild"]["summary_message"]["content"] += "\nUNACCOUNTED BLOCK " * 30
        after["messages"][1] = p["rebuild"]["summary_message"]
    elif case == "summary_unpaired_call":
        p["rebuild"]["summary_message"]["tool_calls"] = group("hidden-new-call")[0]["tool_calls"]
        after["messages"][1] = p["rebuild"]["summary_message"]
    elif case == "duplicate_excerpt":
        p["rebuild"]["summary_excerpts"].append(copy.deepcopy(p["rebuild"]["summary_excerpts"][0]))
    elif case == "nonobject_excerpt":
        p["rebuild"]["summary_excerpts"] = [None]
    with tempfile.TemporaryDirectory(prefix="longchain-rebuild-review-") as td:
        root = Path(td)
        (root / "bodies").mkdir()
        bodies, rows, provenance = [before, after], [], []
        previous = []
        for i, body in enumerate(bodies):
            body["req_id"] = f"p:canon:{i}"
            tokens = renderer.tokenizer.encode(renderer.render(body), add_special_tokens=False)
            lcp = 0
            for a,b in zip(previous, tokens):
                if a != b:
                    break
                lcp += 1
            rows.append({"pack": "p", "view": "canon", "logical_call_id": str(i),
                         "session_id": "receiving-session", "chain_id": "c", "dispatch_offset_ms": i*1000,
                         "end_offset_ms": i*1000+1, "phase": "context_reset" if i else "session_start",
                         "glm_tokens": len(tokens), "glm_lcp_with_prev": lcp, "uncached_expected": len(tokens)-lcp,
                         "max_output_i": 16, "in_serving_load": True, "body_ref": "bodies/part.jsonl.gz",
                         "replay_gap_ms": i*100, "gap_valid": True})
            receipt = copy.deepcopy(p) if i else {"kind": "original"}
            receipt.update(req_id=body["req_id"], body_sha256=_canonical_digest(body),
                           prompt_tokens=len(tokens), lcp_tokens=lcp, added_tokens=len(tokens)-lcp,
                           removed_tokens=len(previous)-lcp, previous_req_id="p:canon:0" if i else None)
            provenance.append(receipt)
            previous = tokens
        chain = {"view": "canon", "chain_id": "c", "n_requests": 2, "first_dispatch_offset_ms": 0,
                 "last_end_offset_ms": 1001, "sum_glm_tokens": sum(r["glm_tokens"] for r in rows),
                 "sum_uncached_expected": sum(r["uncached_expected"] for r in rows), "max_output_i_sum": 32,
                 "phases": {"session_start": 1, "context_reset": 1}, "total_edges": 1,
                 "append_only_edges": 0, "append_only_frac": 0.0,
                 "source_chain_targets": {"n_requests": 2, "sum_uncached_expected": 999999,
                                          "sum_glm_tokens": 999999, "max_output_i_sum": 32}}
        write_jl(root / "requests.jsonl", rows)
        write_jl(root / "chains.jsonl", [chain])
        write_jl(root / "provenance.jsonl", provenance)
        with gzip.open(root / "bodies/part.jsonl.gz", "wt") as f:
            for body in bodies:
                f.write(json.dumps(body) + "\n")
        chains = [{"chain_id": "c", "req_ids": [b["req_id"] for b in bodies]}]
        sha = hashlib.sha256(json.dumps(chains, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
        (root / "cohort.json").write_text(json.dumps({"n_chains": 1, "n_requests": 2, "chains": chains, "cohort_sha256": sha}))
        artifacts = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in root.rglob("*") if path.is_file()}
        (root / "manifest.json").write_text(json.dumps({"generator": "review-events-v2", "artifacts": artifacts,
                                                        "max_context_tokens": 1048576}))
        try:
            report = check_dataset(str(root), str(REPO / "s1-dev/harness"), str(REPO / "s1-dev/glm_tok"))
            return {"case": case, "status": report["status"], "errors": report["errors"],
                    "rendered_prompts": report["counts"]["rendered_prompts"],
                    "prompt_tokens": [r["glm_tokens"] for r in rows]}
        except Exception as exc:
            return {"case": case, "status": "CRASH", "exception": repr(exc),
                    "prompt_tokens": [r["glm_tokens"] for r in rows]}


if __name__ == "__main__":
    renderer = Renderer(str(REPO / "s1-dev/glm_tok"))
    cases = ["baseline", "prefix_unclosed_call", "initial_task_removed", "unaccounted_summary_text",
             "summary_unpaired_call", "duplicate_excerpt", "nonobject_excerpt"]
    print(json.dumps({"versions": {n: importlib.metadata.version(n) for n in ("transformers", "tokenizers", "jinja2")},
                      "checker_sha256": hashlib.sha256((REPO / "scripts/longchain/longchain_check.py").read_bytes()).hexdigest(),
                      "fixtures": [run_fixture(renderer, c) for c in cases]}, ensure_ascii=False, indent=2))
