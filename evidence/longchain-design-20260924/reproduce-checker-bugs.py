#!/usr/bin/env python3
"""Small negative fixtures. --fake-renderer isolates checker control flow."""
import argparse
import collections
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "s1-dev/harness"))
from s1_common import Renderer
from scripts.analysis.longchain_check import check_dataset, _actual_append_edge


class CharacterRenderer:
    """Test double, NOT a token-length or compatibility acceptance."""
    def __init__(self, _path):
        self.tokenizer = self
    def render(self, body):
        return json.dumps({k: body.get(k) for k in ("system", "tools", "messages")},
                          ensure_ascii=False, sort_keys=True)
    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))


def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def jl(path, rows):
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))


def fixture(renderer, case):
    with tempfile.TemporaryDirectory(prefix="longchain-checker-review-") as td:
        root = Path(td)
        (root / "bodies").mkdir()
        m1 = [{"role": "user", "content": "Review this calculation."}]
        toolgroup = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "calculate", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "2"},
        ]
        if case == "duplicate_call_across_blocks":
            m1 += copy.deepcopy(toolgroup)
        m2 = copy.deepcopy(m1) + (copy.deepcopy(toolgroup) if case == "duplicate_call_across_blocks"
                                  else [{"role": "assistant", "content": "The calculation is ready."}])
        bodies = [{"req_id": "p:canon:a", "system": "Review carefully.", "tools": [], "messages": m1},
                  {"req_id": "p:canon:b", "system": "Review carefully.", "tools": [], "messages": m2}]
        if case == "silent_system_change":
            bodies[1]["system"] = "An entirely different system instruction."
        if case in ("unknown_kind", "missing_provenance"):
            bodies[1]["messages"] = [{"role": "user", "content": "Unmarked history replacement."}]
        toks = [renderer.tokenizer.encode(renderer.render(b), add_special_tokens=False) for b in bodies]
        prefix = 0
        for a, b in zip(*toks):
            if a != b:
                break
            prefix += 1
        rows, prov = [], []
        for i, (b, tokens) in enumerate(zip(bodies, toks)):
            n = len(tokens)
            lcp = prefix if i else 0
            rows.append({"pack": "p", "view": "canon", "logical_call_id": "ab"[i],
                         "session_id": "receiving_session", "chain_id": "c", "dispatch_offset_ms": i * 1000,
                         "end_offset_ms": i * 1000 + 10, "phase": "intra" if i else "session_start",
                         "glm_tokens": n, "glm_lcp_with_prev": lcp, "uncached_expected": n - lcp,
                         "max_output_i": 4, "in_serving_load": True, "body_ref": "bodies/part.jsonl.gz",
                         "replay_gap_ms": 0 if i == 0 else 100, "gap_valid": True})
            prov.append({"req_id": b["req_id"], "kind": "synthetic" if i else "original",
                         "body_sha256": digest(b), "prompt_tokens": n, "lcp_tokens": lcp,
                         "added_tokens": n-lcp, "removed_tokens": len(toks[0])-lcp if i else 0,
                         "previous_req_id": "p:canon:a" if i else None})
        if case == "unknown_kind":
            prov[1]["kind"] = "synthethic"
            rows[1]["glm_lcp_with_prev"] = 0
            rows[1]["uncached_expected"] = 0
        if case == "missing_session_id":
            del rows[1]["session_id"]
        if case == "session_changes_mid_chain":
            rows[1]["session_id"] = "other_session"
        if case == "invalid_gap_decomposition":
            rows[1]["replay_gap_ms"] = 999999999
        if case == "nonexistent_original_source":
            prov[0]["source_req_id"] = "p:canon:does_not_exist"
            prov[0]["source_body_sha256"] = "a" * 64
        sums = {"sum_glm_tokens": sum(r["glm_tokens"] for r in rows),
                "sum_uncached_expected": sum(r["uncached_expected"] for r in rows),
                "max_output_i_sum": sum(r["max_output_i"] for r in rows)}
        append = int(_actual_append_edge(*bodies))
        chain = {"view": "canon", "chain_id": "c", "n_requests": 2, **sums,
                 "first_dispatch_offset_ms": 0, "last_end_offset_ms": 1010,
                 "phases": dict(collections.Counter(r["phase"] for r in rows)), "total_edges": 1,
                 "append_only_edges": append, "append_only_frac": float(append),
                 "source_chain_targets": {"n_requests": 2, "sum_uncached_expected": 999,
                                          "sum_glm_tokens": 999, "max_output_i_sum": 8}}
        jl(root / "requests.jsonl", rows)
        jl(root / "chains.jsonl", [chain])
        if case != "missing_provenance":
            jl(root / "provenance.jsonl", prov)
        with gzip.open(root / "bodies/part.jsonl.gz", "wt") as f:
            for b in bodies:
                f.write(json.dumps(b) + "\n")
        chains = [{"chain_id": "c", "req_ids": [b["req_id"] for b in bodies]}]
        cohort_sha = hashlib.sha256(json.dumps(chains, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
        (root / "cohort.json").write_text(json.dumps({"n_chains": 1, "n_requests": 2,
                                                      "chains": chains, "cohort_sha256": cohort_sha}))
        artifacts = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in root.rglob("*") if p.is_file()}
        (root / "manifest.json").write_text(json.dumps({"generator": "negative-fixture", "artifacts": artifacts,
                                                        "max_context_tokens": 1048576}))
        report = check_dataset(str(root), str(REPO / "s1-dev/harness"), str(REPO / "s1-dev/glm_tok"))
        return {"case": case, "status": report["status"], "errors": report["errors"],
                "rendered_prompts": report["counts"]["rendered_prompts"],
                "verified_artifacts": report["artifact_checks"]["verified"],
                "note": "Fresh complete manifest/cohort; no network/engine. Renderer type: " + type(renderer).__name__}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake-renderer", action="store_true")
    args = ap.parse_args()
    if args.fake_renderer:
        import s1_common
        Renderer = CharacterRenderer
        s1_common.Renderer = CharacterRenderer
    renderer = Renderer(str(REPO / "s1-dev/glm_tok"))
    cases = ["baseline", "missing_provenance", "unknown_kind", "duplicate_call_across_blocks",
             "silent_system_change", "missing_session_id", "session_changes_mid_chain",
             "invalid_gap_decomposition", "nonexistent_original_source"]
    print(json.dumps([fixture(renderer, case) for case in cases], ensure_ascii=False, indent=2))
