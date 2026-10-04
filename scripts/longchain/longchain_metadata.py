"""Publish metadata-only workload edits while preserving bodies and request order.

The output is always a new dataset. Bodies are linked, never regenerated. A
metadata snapshot can be prepared without its bodies, but stays explicitly
incomplete and cannot pass the existing replay guard.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

REPO = Path(__file__).resolve().parents[2]
EDITABLE = {"max_output_i", "replay_gap_ms", "gap_valid", "gap_imputed",
            "net_think_ms", "tool_union_ms", "gap_regap_factor"}


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_rows(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"),
                                           allow_nan=False) + "\n" for row in rows))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def req_id(row):
    return f"{row['pack']}:{row['view']}:{row['logical_call_id']}"


def checked_output(parent, out):
    parent, out_arg = Path(parent).resolve(), Path(out)
    if out_arg.exists() or out_arg.is_symlink():
        raise ValueError("output exists; choose a fresh directory, never overwrite a frozen input")
    out = out_arg.resolve()
    protected = [REPO / p for p in ("s1-dev", "llm-challenge-arena-v1", "build/base_exact", "refs")]
    if (parent == out or parent in out.parents or out in parent.parents
            or any(out == p or p in out.parents for p in protected)):
        raise ValueError("output overlaps an input or protected source tree")
    return parent, out


def _safe_relative(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe artifact path: {value}")
    return path


def publish(parent, out, edited, *, operation, name=None, cohort_path=None, metadata_only=False):
    parent, out = checked_output(parent, out)
    manifest = json.loads((parent / "manifest.json").read_text())
    original = read_rows(parent / "requests.jsonl")
    chains = read_rows(parent / "chains.jsonl")
    provenance = read_rows(parent / "provenance.jsonl")
    cohort = json.loads((parent / "cohort.json").read_text())
    name = name or manifest.get("set")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("a simple dataset --set name is required")
    if len({req_id(r) for r in original}) != len(original):
        raise ValueError("duplicate parent request ID")
    if [req_id(r) for r in edited] != [req_id(r) for r in original]:
        raise ValueError("metadata edits must preserve every request and its file order")
    if set(p["req_id"] for p in provenance) != {req_id(r) for r in original} or len(provenance) != len(original):
        raise ValueError("parent provenance coverage differs from requests")
    max_context = manifest.get("max_context_tokens")
    if not isinstance(max_context, int) or isinstance(max_context, bool) or max_context <= 0:
        raise ValueError("parent lacks max_context_tokens; derive from its complete original manifest")
    changes = {}
    for before, after in zip(original, edited):
        fields = {k: {"before": before.get(k), "after": after.get(k)}
                  for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
        if fields.keys() - EDITABLE:
            raise ValueError(f"cannot edit body/phase/token/identity fields: {req_id(before)}")
        budget, gap = after.get("max_output_i"), after.get("replay_gap_ms")
        if (not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0
                or after["glm_tokens"] + budget > max_context):
            raise ValueError(f"invalid output budget/context: {req_id(after)}")
        if gap is not None and (isinstance(gap, bool) or not isinstance(gap, (int, float))
                                or not math.isfinite(gap) or gap < 0):
            raise ValueError(f"invalid replay gap: {req_id(after)}")
        if gap is None and after.get("gap_valid") is True:
            raise ValueError(f"valid gap cannot be null: {req_id(after)}")
        if fields:
            changes[req_id(before)] = fields
    by_chain = defaultdict(list)
    for row in edited:
        by_chain[row["chain_id"]].append(row)
    if {c["chain_id"] for c in chains} != set(by_chain) or len(chains) != len(by_chain):
        raise ValueError("chain metadata coverage mismatch")
    if cohort_path:
        replacement = json.loads(Path(cohort_path).read_text())
        old = {c["chain_id"]: c["req_ids"] for c in cohort["chains"]}
        new = {c["chain_id"]: c["req_ids"] for c in replacement["chains"]}
        if old != new or len(replacement["chains"]) != len(old):
            raise ValueError("cohort replacement may reorder chains only; request membership/order differs")
        cohort = replacement
    wanted = [rid for c in cohort["chains"] for rid in c["req_ids"]]
    if len(wanted) != len(original) or set(wanted) != {req_id(r) for r in original}:
        raise ValueError("cohort/request coverage mismatch")
    for c in cohort["chains"]:
        ordered = sorted(by_chain[c["chain_id"]],
                         key=lambda r: (r.get("dispatch_offset_ms") or 0, r["logical_call_id"]))
        if c["req_ids"] != [req_id(r) for r in ordered]:
            raise ValueError("cohort differs from harness within-chain order")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("derive from a parent with a complete artifacts map")
    core = {"requests.jsonl", "chains.jsonl", "provenance.jsonl", "cohort.json"}
    if not core <= artifacts.keys():
        raise ValueError("parent manifest omits core metadata hashes")
    body_refs = {r["body_ref"] for r in original}
    if not body_refs <= artifacts.keys():
        raise ValueError("parent manifest omits body hashes")
    missing = []
    # One verification pass; no re-rendering or copying of body bytes.
    for relative, expected in artifacts.items():
        path = parent / _safe_relative(relative)
        if not path.is_file():
            if relative in core or not metadata_only:
                raise ValueError(f"missing parent artifact: {relative}; use --metadata-only only for an incomplete snapshot")
            missing.append(relative)
        elif digest(path) != expected:
            raise ValueError(f"parent artifact hash mismatch: {relative}")
    parent_receipt = {"root": str(parent), "manifest_sha256": digest(parent / "manifest.json"),
                      "requests_sha256": artifacts["requests.jsonl"],
                      "cohort_sha256": artifacts["cohort.json"]}
    receipt = dict(operation=operation, parent=parent_receipt, changed_requests=len(changes),
                   fields=dict(Counter(k for change in changes.values() for k in change)),
                   bodies="unchanged; existing shards linked", token_labels="unchanged",
                   cohort_source=str(Path(cohort_path).resolve()) if cohort_path else str(parent / "cohort.json"),
                   cohort_source_sha256=digest(cohort_path or parent / "cohort.json"),
                   implementation_sha256=digest(__file__))
    prov_by_id = {p["req_id"]: p for p in provenance}
    edited_by_id = {req_id(r): r for r in edited}
    for rid, fields in changes.items():
        p = prov_by_id[rid]
        p.setdefault("metadata_patches", []).append({"operation": operation["kind"],
            "parent_requests_sha256": parent_receipt["requests_sha256"], "fields": fields})
        if "max_output_i" in fields:
            p["output_budget_origin"] = "metadata_patch:" + operation["kind"]
        if set(fields) & {"replay_gap_ms", "net_think_ms", "tool_union_ms"}:
            p["gap_origin"] = "metadata_patch:" + operation["kind"]
            p["replay_gap_ms"] = edited_by_id[rid]["replay_gap_ms"]
            p["gap_decomposition_known"] = all(edited_by_id[rid].get(k) is not None
                                               for k in ("net_think_ms", "tool_union_ms"))
    for chain in chains:
        chain["max_output_i_sum"] = sum(r["max_output_i"] for r in by_chain[chain["chain_id"]])
    sums = {c["chain_id"]: c["max_output_i_sum"] for c in chains}
    child = deepcopy(manifest)
    for summary in child.get("chain_summaries", []):
        summary["output_sum"] = sums[summary["chain_id"]]
    child.update(generator="longchain-metadata-v1", set=name, status="BUILT_UNVALIDATED",
                 formal_score=None, actual_output_sum=sum(sums.values()))
    child.setdefault("metadata_patches", []).append(receipt)
    child["quality_status"] = "DIAGNOSTIC_CANDIDATE_NOT_REPRESENTATIVE"
    cohort["set"] = name
    cohort["cohort_sha256"] = hashlib.sha256(json.dumps(cohort["chains"], ensure_ascii=False,
                                                       sort_keys=True).encode()).hexdigest()[:16]
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=out.name + ".tmp-", dir=out.parent) as temporary:
        stage = Path(temporary)
        write_rows(stage / "requests.jsonl", edited)
        write_rows(stage / "chains.jsonl", chains)
        write_rows(stage / "provenance.jsonl", provenance)
        (stage / "cohort.json").write_text(json.dumps(cohort, ensure_ascii=False, indent=1) + "\n")
        (stage / "samples").mkdir()
        write_rows(stage / "samples" / (name + ".jsonl"),
                   [{"chain_id": c["chain_id"], "set_role": "inherited-frozen-cohort"} for c in cohort["chains"]])
        absent = {}
        for relative in artifacts:
            if relative in core or relative.startswith("samples/"):
                continue
            target = stage / _safe_relative(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            source = parent / relative
            if relative in missing:
                absent[relative] = artifacts[relative]
            elif relative in body_refs:
                # Relative file symlinks remain correct after this sibling staging dir is published.
                target.symlink_to(os.path.relpath(source.resolve(), out / Path(relative).parent))
            else:
                target.write_bytes(source.read_bytes())
        hashes = {str(p.relative_to(stage)): (artifacts[str(p.relative_to(stage))]
                  if str(p.relative_to(stage)) in body_refs else digest(p))
                  for p in stage.rglob("*") if p.is_file()}
        # Body hashes were verified already; symlinks are not rehashed in a second pass.
        child["artifacts"] = {**hashes, **absent}
        child["missing_artifacts"] = sorted(absent)
        if absent:
            child["status"] = "METADATA_ONLY_INCOMPLETE"
        (stage / "manifest.json").write_text(json.dumps(child, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        # Reserve without overwriting an existing output, then publish the manifest last.
        out.mkdir(exist_ok=False)
        for entry in stage.iterdir():
            if entry.name != "manifest.json":
                entry.rename(out / entry.name)
        (stage / "manifest.json").rename(out / "manifest.json")
    return child
