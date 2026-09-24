#!/usr/bin/env python3
"""Independent structural and optional Renderer check for S1 replay datasets."""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import json
import math
import os
import statistics
import sys


def _jsonl(path):
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except Exception as exc:
                    raise ValueError(f"{path}:{n}: invalid JSON: {exc}") from exc


def _req_id(row):
    return "%s:%s:%s" % (row.get("pack"), row.get("view"), row.get("logical_call_id"))


def _q(values, p):
    a = sorted(x for x in values if isinstance(x, (int, float)))
    return a[min(len(a)-1, int(p * len(a)))] if a else None


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _canonical_digest(value):
    blob = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _cohort_ids(cohort):
    if isinstance(cohort, (str, os.PathLike)):
        with open(cohort, encoding="utf-8") as fh:
            cohort = json.load(fh)
    if cohort is None:
        return None
    # Also accept a plain list of req IDs for small test fixtures / callers.
    if isinstance(cohort, list):
        if not cohort or isinstance(cohort[0], str):
            return list(cohort)
        chains = cohort
    else:
        chains = cohort.get("chains", [])
    return [rid for chain in chains for rid in chain.get("req_ids", [])]


def _tool_errors(body, rid):
    """Check only associations represented by this source body; no schema assumptions."""
    errors = []
    messages = body.get("messages") or []
    calls, results = [], []
    for mi, msg in enumerate(messages):
        if not isinstance(msg, dict):
            errors.append(f"{rid}: messages[{mi}] is not an object")
            continue
        for call in msg.get("tool_calls") or []:
            if isinstance(call, dict) and call.get("id") is not None:
                calls.append(str(call["id"]))
        # Different source exporters use different association keys. Validate a
        # result only when the body actually carries an explicit association.
        if msg.get("role") == "tool":
            assoc = msg.get("tool_call_id", msg.get("call_id", msg.get("tool_use_id")))
            if assoc is not None:
                results.append(str(assoc))
    # Historical bodies repeat earlier assistant/tool messages on every turn;
    # reused call IDs therefore are not inherently duplicates in this format.
    callset = set(calls)
    errors.extend(f"{rid}: tool result references unknown call id {x}" for x in results if x not in callset)
    return errors


def _new_message_suffix(previous, current):
    """Return the first new message, allowing the harness reminder replacement."""
    before, after = previous or [], current or []
    i = 0
    while i < min(len(before), len(after)) and before[i] == after[i]:
        i += 1
    if i == len(before):
        return i
    if (i == len(before) - 1 and isinstance(before[i], dict) and before[i].get("role") == "user"
            and "system-reminder" in str(before[i].get("content", ""))):
        return i
    return None


def _new_tool_block_errors(messages, start, rid):
    """Validate only calls/results introduced in one newly appended message suffix."""
    errors, implicit = [], 0
    suffix = messages[start:]
    def call_ids(seq):
        return [str(c.get("id") or c.get("tool_call_id")) for m in seq if isinstance(m, dict)
                for c in (m.get("tool_calls") or []) if isinstance(c, dict)
                and (c.get("id") or c.get("tool_call_id"))]
    old_ids, new_ids = set(call_ids(messages[:start])), call_ids(suffix)
    if len(new_ids) != len(set(new_ids)):
        errors.append(f"{rid}: introduced tool call IDs repeat across the new suffix")
    if old_ids & set(new_ids):
        errors.append(f"{rid}: introduced tool call ID reuses an existing history ID")
    i = 0
    while i < len(suffix):
        msg = suffix[i]
        if not isinstance(msg, dict):
            i += 1
            continue
        calls = msg.get("tool_calls") or []
        if calls:
            result_entries, result_messages = [], []
            j = i + 1
            while j < len(suffix) and isinstance(suffix[j], dict) and suffix[j].get("role") == "tool":
                result_messages.append(suffix[j])
                content = suffix[j].get("content")
                entries = (content if isinstance(content, list) and content
                           and all(isinstance(e, dict) and "output" in e for e in content)
                           else [suffix[j]])
                result_entries.extend(entries)
                j += 1
            if len(result_entries) != len(calls):
                errors.append(f"{rid}: introduced tool block has {len(calls)} calls but {len(result_entries)} results")
            call_ids = [(c.get("id") or c.get("tool_call_id")) if isinstance(c, dict) else None for c in calls]
            result_ids = [(e.get("tool_call_id") or e.get("id")) if isinstance(e, dict) else None for e in result_entries]
            known_calls = [str(x) for x in call_ids if x is not None]
            explicit_results = [str(x) for x in result_ids if x is not None]
            if len(known_calls) != len(set(known_calls)):
                errors.append(f"{rid}: introduced tool block has duplicate call IDs")
            if len(explicit_results) != len(set(explicit_results)):
                errors.append(f"{rid}: introduced tool block has duplicate result IDs")
            if any(x not in known_calls for x in explicit_results):
                errors.append(f"{rid}: introduced tool result ID does not match a call ID")
            if all(x is not None for x in call_ids + result_ids):
                if collections.Counter(map(str, call_ids)) != collections.Counter(map(str, result_ids)):
                    errors.append(f"{rid}: introduced tool call/result ID sets differ")
            elif len(result_entries) == len(calls):
                implicit += 1
            i = j
            continue
        if msg.get("role") == "tool":
            errors.append(f"{rid}: introduced tool result has no preceding call in its new suffix")
        i += 1
    return errors, implicit


def _actual_append_edge(before_body, after_body):
    if before_body.get("system") != after_body.get("system") or before_body.get("tools") != after_body.get("tools"):
        return False
    start = _new_message_suffix(before_body.get("messages", []), after_body.get("messages", []))
    if start is None:
        return False
    return any(isinstance(m, dict) and m.get("role") == "assistant"
               for m in after_body.get("messages", [])[start:])


def _chain_summary_errors(chain_meta, grouped, req_rows, bodies):
    errors = []
    if set(chain_meta) != set(grouped):
        errors.append(f"generated chains.jsonl chain ID set mismatch: missing={len(set(grouped)-set(chain_meta))}, extra={len(set(chain_meta)-set(grouped))}")
    for cid, ids in grouped.items():
        ch = chain_meta.get(cid)
        if ch is None:
            continue
        rs = [req_rows[rid] for rid in ids]
        checks = {
            "n_requests": len(rs),
            "first_dispatch_offset_ms": rs[0].get("dispatch_offset_ms"),
            "last_end_offset_ms": rs[-1].get("end_offset_ms"),
            "sum_glm_tokens": sum(r.get("glm_tokens", 0) for r in rs),
            "sum_uncached_expected": sum(r.get("uncached_expected", 0) for r in rs),
            "max_output_i_sum": sum(r.get("max_output_i", 0) for r in rs),
            "phases": dict(collections.Counter(r.get("phase") for r in rs)),
        }
        total_edges = max(0, len(rs) - 1)
        append_edges = sum(_actual_append_edge(bodies[before], bodies[after])
                           for before, after in zip(ids, ids[1:]) if before in bodies and after in bodies)
        checks["total_edges"] = total_edges
        checks["append_only_edges"] = append_edges
        checks["append_only_frac"] = append_edges / total_edges if total_edges else None
        for field, expected in checks.items():
            actual = ch.get(field)
            if field == "append_only_frac" and actual is not None and expected is not None:
                matched = isinstance(actual, (int, float)) and not isinstance(actual, bool) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
            else:
                matched = actual == expected
            if not matched:
                errors.append(f"chain {cid}: chains.jsonl {field}={actual!r}, recomputed={expected!r}")
    return errors


def _workload_ledger(chain_meta, grouped, req_rows, provenance, per_request):
    """Reconcile source-chain frozen budget with visible source and generated work."""
    rows, errors = [], []
    fields = ("source_requests", "visible_original_requests", "synthetic_requests",
              "source_frozen_uncached", "visible_original_source_frozen_uncached",
              "visible_original_current_frozen_uncached", "adapted_original_frozen_delta",
              "missing_source_frozen_uncached", "generated_synthetic_lcp_added",
              "missing_budget_delta", "generated_frozen_uncached",
              "generated_cohort_visible_lcp_added", "source_prompt_sum", "generated_prompt_sum",
              "source_output_sum", "generated_output_sum")
    for cid, ids in sorted(grouped.items()):
        chain = chain_meta.get(cid, {})
        source = chain.get("source_chain_targets")
        if not isinstance(source, dict):
            errors.append(f"chain {cid}: missing nested source_chain_targets for workload ledger")
            source = {}
        source_budget = source.get("sum_uncached_expected")
        source_requests = source.get("n_requests")
        visible_ids = [rid for rid in ids if (provenance.get(rid) or {}).get("kind") in ("original", "adapted_original")]
        synthetic_ids = [rid for rid in ids if (provenance.get(rid) or {}).get("kind") == "synthetic"]
        visible_source_budget = 0
        visible_current_budget = 0
        adapted_delta = 0
        for rid in visible_ids:
            p, row = provenance.get(rid) or {}, req_rows[rid]
            frozen = (p.get("source_frozen_labels") or {}).get("uncached_expected") if p.get("kind") == "adapted_original" else row.get("uncached_expected")
            if not _is_int(frozen):
                errors.append(f"{rid}: missing integer source frozen uncached_expected for workload ledger")
            else:
                visible_source_budget += frozen
            current = row.get("uncached_expected")
            if not _is_int(current):
                errors.append(f"{rid}: missing integer current uncached_expected for workload ledger")
            else:
                visible_current_budget += current
                if p.get("kind") == "adapted_original" and _is_int(frozen):
                    adapted_delta += current - frozen
        synthetic_added = 0
        for rid in synthetic_ids:
            value = (provenance.get(rid) or {}).get("added_tokens")
            if not _is_int(value):
                errors.append(f"{rid}: missing integer synthetic added_tokens for workload ledger")
            else:
                synthetic_added += value
        visible_lcp_added_values = [
            per_request[rid].get("added_tokens") for rid in ids
            if _is_int(per_request.get(rid, {}).get("added_tokens"))]
        visible_lcp_added = (sum(visible_lcp_added_values)
                             if len(visible_lcp_added_values) == len(ids) else None)
        if not _is_int(source_budget) or not _is_int(source_requests):
            errors.append(f"chain {cid}: source target lacks integer request count/uncached budget")
            missing = None
        else:
            missing = source_budget - visible_source_budget
            if missing < 0:
                errors.append(f"chain {cid}: visible original frozen budget exceeds source target by {-missing}")
        row = {
            "chain_id": cid, "source_requests": source_requests,
            "visible_original_requests": len(visible_ids), "synthetic_requests": len(synthetic_ids),
            "source_frozen_uncached": source_budget,
            "visible_original_source_frozen_uncached": visible_source_budget,
            "visible_original_current_frozen_uncached": visible_current_budget,
            "adapted_original_frozen_delta": adapted_delta,
            "missing_source_frozen_uncached": missing,
            "generated_synthetic_lcp_added": synthetic_added,
            "missing_budget_delta": (synthetic_added - missing) if missing is not None else None,
            "generated_frozen_uncached": visible_current_budget + synthetic_added,
            "generated_cohort_visible_lcp_added": visible_lcp_added,
            "source_prompt_sum": source.get("sum_glm_tokens"),
            "generated_prompt_sum": chain.get("sum_glm_tokens"),
            "source_output_sum": source.get("max_output_i_sum"),
            "generated_output_sum": chain.get("max_output_i_sum"),
        }
        rows.append(row)
    totals = {}
    for field in fields:
        vals = [r.get(field) for r in rows]
        totals[field] = sum(vals) if vals and all(_is_int(v) for v in vals) else None
    return {"rows": rows, "totals": totals,
            "note": "Token-work accounting from frozen provenance and rendered visible LCP; not GPU measurements or cache-hit claims."}, errors


def check_dataset(root, harness_dir, tok_dir=None, cohort=None):
    """Return an auditable report. Structural-only reports can never be PASS."""
    errors, warnings = [], []
    manifest_path = os.path.join(root, "manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, encoding="utf-8") as fh:
                manifest = json.load(fh)
        except Exception as exc:
            errors.append(f"cannot read manifest.json: {exc}")
    generated = bool(manifest.get("generator"))
    artifact_validation = {"listed": 0, "verified": 0, "missing_entries": [], "mismatches": []}
    if generated:
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict) or not artifacts:
            errors.append("generated manifest must contain a non-empty artifacts hash map")
            artifacts = {}
        artifact_validation["listed"] = len(artifacts)
        normalized_artifacts = {str(k).replace(os.sep, "/"): v for k, v in artifacts.items()}
        expected_files = set()
        for dp, _, names in os.walk(root):
            for name in names:
                path = os.path.join(dp, name)
                rel = os.path.relpath(path, root).replace(os.sep, "/")
                if rel != "manifest.json":
                    expected_files.add(rel)
        artifact_validation["missing_entries"] = sorted(expected_files - set(normalized_artifacts))
        if artifact_validation["missing_entries"]:
            errors.append("manifest artifacts omit dataset files: " + ", ".join(artifact_validation["missing_entries"]))
        for rel, expected_hash in normalized_artifacts.items():
            path = os.path.join(root, rel)
            if not os.path.isfile(path):
                artifact_validation["mismatches"].append(rel)
                errors.append(f"manifest artifact file is missing: {rel}")
                continue
            actual_hash = _file_digest(path)
            if not isinstance(expected_hash, str) or actual_hash != expected_hash:
                artifact_validation["mismatches"].append(rel)
                errors.append(f"manifest artifact SHA256 mismatch: {rel}")
            else:
                artifact_validation["verified"] += 1
    req_rows, req_dupes = {}, []
    for row in _jsonl(os.path.join(root, "requests.jsonl")):
        if row.get("view") != "canon":
            continue
        rid = _req_id(row)
        if rid in req_rows:
            req_dupes.append(rid)
        else:
            row["_req_id"] = rid
            req_rows[rid] = row
    errors.extend(f"duplicate request id: {x}" for x in req_dupes)
    serving = {rid for rid, row in req_rows.items() if row.get("in_serving_load")}
    # Match s1_common.load_index: dispatch then logical call ID.
    by_chain = collections.defaultdict(list)
    chain_meta = {}
    for c in _jsonl(os.path.join(root, "chains.jsonl")):
        if c.get("view") == "canon":
            if c.get("chain_id") in chain_meta:
                errors.append("duplicate chain record: " + str(c.get("chain_id")))
            chain_meta[c.get("chain_id")] = c
    for rid in serving:
        by_chain[req_rows[rid].get("chain_id")].append(rid)
    source_length_mismatches = 0
    for cid, ids in by_chain.items():
        ids.sort(key=lambda x: (req_rows[x].get("dispatch_offset_ms") or 0,
                                req_rows[x].get("logical_call_id") or ""))
        for idx, rid in enumerate(ids):
            req_rows[rid]["_idx_in_chain"] = idx
        declared = chain_meta.get(cid, {}).get("n_requests")
        if declared is not None and int(declared) != len(ids):
            source_length_mismatches += 1
    if source_length_mismatches:
        msg = (f"{source_length_mismatches} chains.jsonl n_requests values differ from visible serving rows; "
               "source n_requests describes the source chain target")
        (errors if generated else warnings).append(msg)

    cohort_obj = cohort
    if generated and cohort_obj is None:
        frozen_path = os.path.join(root, "cohort.json")
        if os.path.isfile(frozen_path):
            cohort_obj = frozen_path
        else:
            errors.append("generated dataset is missing frozen cohort.json")
    if isinstance(cohort, (str, os.PathLike)):
        with open(cohort, encoding="utf-8") as fh:
            cohort_obj = json.load(fh)
    cohort_ids = _cohort_ids(cohort_obj)
    target_ids = serving if cohort_ids is None else set(cohort_ids)
    if generated and isinstance(cohort_obj, (str, os.PathLike)):
        with open(cohort_obj, encoding="utf-8") as fh:
            cohort_obj = json.load(fh)
    if cohort_ids is not None:
        if len(cohort_ids) != len(set(cohort_ids)):
            errors.append("cohort contains duplicate request IDs")
        missing = set(cohort_ids) - serving
        errors.extend(f"cohort ID is absent from serving requests: {x}" for x in sorted(missing))
        # Specialized roots commonly contain precisely this cohort. For a broader
        # root, membership is intentionally a subset and is reported explicitly.
        if target_ids == serving:
            pass
        if isinstance(cohort_obj, dict):
            for ch in cohort_obj.get("chains", []):
                selected = ch.get("req_ids", [])
                expected = sorted((rid for rid in serving
                                   if req_rows[rid].get("chain_id") == ch.get("chain_id")),
                                  key=lambda x: (req_rows[x].get("dispatch_offset_ms") or 0,
                                                 req_rows[x].get("logical_call_id") or ""))
                if selected != expected:
                    errors.append(f"cohort chain order/content mismatch for {ch.get('chain_id')}")
            if cohort_obj.get("n_requests") is not None and int(cohort_obj["n_requests"]) != len(cohort_ids):
                errors.append("cohort n_requests does not match flattened req_ids")
            if cohort_obj.get("n_chains") is not None and int(cohort_obj["n_chains"]) != len(cohort_obj.get("chains", [])):
                errors.append("cohort n_chains does not match chains list")
            if generated:
                expected_sha = hashlib.sha256(json.dumps(cohort_obj.get("chains"), ensure_ascii=False,
                                                          sort_keys=True).encode("utf-8")).hexdigest()[:16]
                if cohort_obj.get("cohort_sha256") != expected_sha:
                    errors.append("generated cohort_sha256 does not match frozen ordered chains")
    if generated and cohort_ids is not None and set(cohort_ids) != serving:
        errors.append(f"generated root serving/cohort mismatch: serving={len(serving)}, cohort={len(set(cohort_ids))}")
    absent = target_ids - set(req_rows)
    errors.extend(f"expected request ID absent from requests: {x}" for x in sorted(absent))
    if generated:
        max_context = manifest.get("max_context_tokens")
        if not _is_int(max_context) or max_context <= 0:
            errors.append("generated manifest max_context_tokens must be a positive integer")
        for rid in sorted(target_ids & set(req_rows)):
            row = req_rows[rid]
            for key in ("pack", "session_id", "chain_id", "logical_call_id"):
                if not isinstance(row.get(key), str) or not row[key]:
                    errors.append(f"{rid}: generated {key} must be a nonempty string")
            if row.get("phase") not in ("session_start", "turn_start", "context_reset", "intra"):
                errors.append(f"{rid}: unknown generated phase {row.get('phase')!r}")
            budget = row.get("max_output_i")
            if not _is_int(budget) or budget <= 0:
                errors.append(f"{rid}: generated max_output_i must be a positive integer")
            gap = row.get("replay_gap_ms")
            gap_valid = row.get("gap_valid")
            if gap is None:
                if gap_valid is True:
                    errors.append(f"{rid}: replay_gap_ms is null while gap_valid is true")
            elif (isinstance(gap, bool) or not isinstance(gap, (int, float))
                  or not math.isfinite(gap) or gap < 0):
                errors.append(f"{rid}: replay_gap_ms must be null or a finite non-negative number")
            prompt = row.get("glm_tokens")
            if not _is_int(prompt) or prompt < 0:
                errors.append(f"{rid}: generated glm_tokens must be a non-negative integer")
            elif _is_int(budget) and _is_int(max_context) and prompt + budget > max_context:
                errors.append(f"{rid}: prompt plus output budget exceeds max_context_tokens")
        for cid, ids in by_chain.items():
            sessions = {req_rows[rid].get("session_id") for rid in ids
                        if isinstance(req_rows[rid].get("session_id"), str)}
            if len(sessions) != 1:
                errors.append(f"chain {cid}: generated session_id changes or is absent within chain")
            declared_session = chain_meta.get(cid, {}).get("session_id")
            if declared_session is not None and sessions != {declared_session}:
                errors.append(f"chain {cid}: chain metadata session_id differs from its requests")

    body_ids, body_dupes, bodies, body_paths = set(), [], {}, {}
    body_dir = os.path.join(root, "bodies")
    for dp, _, names in os.walk(body_dir):
        for name in sorted(names):
            if not name.endswith(".jsonl.gz"):
                continue
            path = os.path.join(dp, name)
            try:
                with gzip.open(path, "rt", encoding="utf-8") as fh:
                    for n, line in enumerate(fh, 1):
                        if not line.strip():
                            continue
                        try:
                            body = json.loads(line)
                        except Exception as exc:
                            errors.append(f"{path}:{n}: invalid body JSON: {exc}")
                            continue
                        rid = body.get("req_id")
                        if not isinstance(rid, str):
                            errors.append(f"{path}:{n}: body missing string req_id")
                            continue
                        if rid in body_ids:
                            body_dupes.append(rid)
                        else:
                            body_ids.add(rid)
                            body_paths[rid] = os.path.relpath(path, root).replace(os.sep, "/")
                            if rid in target_ids:
                                bodies[rid] = body
            except OSError as exc:
                errors.append(f"cannot read body shard {path}: {exc}")
    errors.extend(f"duplicate body id: {x}" for x in sorted(set(body_dupes)))
    errors.extend(f"serving/cohort request has no body: {x}" for x in sorted(target_ids - body_ids))
    if generated:
        for rid in sorted(target_ids & body_ids & set(req_rows)):
            declared_ref = req_rows[rid].get("body_ref")
            actual_ref = body_paths.get(rid)
            if declared_ref != actual_ref:
                errors.append(f"{rid}: body_ref={declared_ref!r} does not match containing shard {actual_ref!r}")
    if not target_ids:
        errors.append("empty serving/cohort dataset")
    extra_bodies = body_ids - set(req_rows)
    if extra_bodies:
        warnings.append(f"{len(extra_bodies)} body IDs have no canon request row")

    # Validate tool call associations inside the bodies without requiring a
    # tool_call_id field that the source format may not provide.
    for rid, body in bodies.items():
        errors.extend(_tool_errors(body, rid))

    grouped = collections.defaultdict(list)
    for rid in target_ids & set(req_rows):
        grouped[req_rows[rid].get("chain_id")].append(rid)
    for cid, ids in grouped.items():
        ids.sort(key=lambda x: req_rows[x]["_idx_in_chain"])
    if generated:
        errors.extend(_chain_summary_errors(chain_meta, grouped, req_rows, bodies))

    provenance = {}
    prov_path = os.path.join(root, "provenance.jsonl")
    if os.path.exists(prov_path):
        try:
            for row in _jsonl(prov_path):
                rid = row.get("req_id")
                if rid in provenance:
                    errors.append(f"duplicate provenance req_id: {rid}")
                provenance[rid] = row
            if set(provenance) != target_ids:
                errors.append(f"provenance ID set mismatch: missing={len(target_ids-set(provenance))}, extra={len(set(provenance)-target_ids)}")
        except Exception as exc:
            errors.append(f"cannot validate provenance.jsonl: {exc}")
    elif generated:
        errors.append("generated dataset is missing provenance.jsonl")
    if generated:
        for rid in sorted(target_ids):
            if provenance.get(rid, {}).get("kind") not in ("original", "adapted_original", "synthetic"):
                errors.append(f"{rid}: missing or unknown generated provenance kind")
        for rid in sorted(target_ids & body_ids & set(provenance)):
            actual_sha = _canonical_digest(bodies[rid])
            declared_sha = provenance[rid].get("body_sha256")
            if not isinstance(declared_sha, str) or declared_sha != actual_sha:
                errors.append(f"{rid}: provenance body_sha256 does not match canonical body JSON")
            if provenance[rid].get("kind") == "adapted_original":
                source_sha = provenance[rid].get("source_body_sha256")
                if (not isinstance(source_sha, str) or len(source_sha) != 64
                        or any(c not in "0123456789abcdef" for c in source_sha)):
                    errors.append(f"{rid}: adapted_original source_body_sha256 is missing or malformed")
    source_hash_checks = {"adapted_original_rows": 0, "verified_against_parent_provenance": 0,
                          "skipped_reason": None}
    adapted = [(rid, p) for rid, p in provenance.items() if p.get("kind") == "adapted_original"]
    source_hash_checks["adapted_original_rows"] = len(adapted)
    if generated and adapted:
        parent_root = manifest.get("parent_root")
        parent_prov_path = os.path.join(parent_root, "provenance.jsonl") if parent_root else None
        parent_req_path = os.path.join(parent_root, "requests.jsonl") if parent_root else None
        if not parent_prov_path or not os.path.isfile(parent_prov_path) or not os.path.isfile(parent_req_path):
            source_hash_checks["skipped_reason"] = "parent provenance/requests unavailable; source_body_sha256 shape checked only"
            warnings.append("adapted_original source hashes were not compared with parent data: " + source_hash_checks["skipped_reason"])
        else:
            wanted_sources = {p.get("source_req_id") for _, p in adapted}
            parent_prov = {x.get("req_id"): x for x in _jsonl(parent_prov_path)
                           if x.get("req_id") in wanted_sources}
            parent_rows = {_req_id(x): x for x in _jsonl(parent_req_path)
                           if _req_id(x) in wanted_sources}
            for rid, p in adapted:
                source_id = p.get("source_req_id")
                pp, pr = parent_prov.get(source_id), parent_rows.get(source_id)
                if pp is None or pr is None:
                    errors.append(f"{rid}: adapted_original source request absent from parent data: {source_id}")
                    continue
                parent_adapted = pp.get("kind") == "adapted_original"
                expected_source_hash = pp.get("source_body_sha256") if parent_adapted else pp.get("body_sha256")
                if p.get("source_body_sha256") != expected_source_hash:
                    errors.append(f"{rid}: source_body_sha256 differs from parent's original-source hash")
                frozen = p.get("source_frozen_labels")
                expected_frozen = (pp.get("source_frozen_labels") if parent_adapted else
                                   {k: pr.get(k) for k in ("glm_tokens", "glm_lcp_with_prev", "uncached_expected")})
                if frozen != expected_frozen:
                    errors.append(f"{rid}: source_frozen_labels differ from parent's original-source labels")
                if p.get("source_body_sha256") == expected_source_hash and frozen == expected_frozen:
                    source_hash_checks["verified_against_parent_provenance"] += 1
            source_hash_checks["verification_basis"] = "original-source hash/labels compared with parent provenance for adapted parents, or parent body/request rows for unadapted parents"
    prompt_tokens, output_tokens, gaps, chain_lengths, per_request = [], [], [], [], {}
    token_lcps, lcp_by_chain = [], {}
    implicit_tool_blocks = 0
    renderer_unavailable = collections.Counter()
    renderer = None
    if tok_dir:
        try:
            sys.path.insert(0, os.path.abspath(harness_dir))
            from s1_common import Renderer
            renderer = Renderer(tok_dir)
        except Exception as exc:
            errors.append(f"cannot initialize original Renderer: {exc!r}")

    for cid, ids in grouped.items():
        chain_lengths.append(len(ids))
        prev_tokens = None
        prev_rid = None
        prev_body = None
        for rid in ids:
            row = req_rows[rid]
            rec = {"req_id": rid, "chain_id": cid, "idx_in_chain": row.get("_idx_in_chain"),
                   "phase": row.get("phase"), "prompt_tokens": row.get("glm_tokens"),
                   "output_budget": row.get("max_output_i"), "gap_ms": row.get("replay_gap_ms"),
                   "pack": row.get("pack"), "family": row.get("sys_tools_hash"),
                   "round": (row.get("_idx_in_chain") + 1) if isinstance(row.get("_idx_in_chain"), int) else None,
                   "kind": (provenance.get(rid) or {}).get("kind", "unclassified"), "lcp_tokens": None}
            if isinstance(row.get("glm_tokens"), (int, float)):
                prompt_tokens.append(row["glm_tokens"])
            if isinstance(row.get("max_output_i"), (int, float)):
                output_tokens.append(row["max_output_i"])
            if isinstance(row.get("replay_gap_ms"), (int, float)):
                gaps.append(row["replay_gap_ms"])
            p = provenance.get(rid)
            is_synthetic = bool(p and p.get("kind") == "synthetic")
            has_regenerated_labels = bool(p and p.get("kind") in ("synthetic", "adapted_original"))
            if is_synthetic and rid in bodies:
                if prev_body is not None and (prev_body.get("system") != bodies[rid].get("system")
                                             or prev_body.get("tools") != bodies[rid].get("tools")):
                    errors.append(f"{rid}: synthetic continuation changes system/tools without an implemented event")
                start = _new_message_suffix((prev_body or {}).get("messages", []),
                                            bodies[rid].get("messages", []))
                if start is None:
                    errors.append(f"{rid}: synthetic body is not an append/reminder-replacement of predecessor")
                else:
                    new_messages = bodies[rid].get("messages", [])[start:]
                    has_new_user_turn = any(
                        isinstance(msg, dict) and msg.get("role") == "user"
                        and "system-reminder" not in str(msg.get("content", ""))
                        for msg in new_messages)
                    expected_phase = "turn_start" if has_new_user_turn else "intra"
                    if row.get("phase") != expected_phase:
                        errors.append(f"{rid}: synthetic phase={row.get('phase')!r}, but appended message events require {expected_phase!r}")
                    block_errors, implicit = _new_tool_block_errors(
                        bodies[rid].get("messages", []), start, rid)
                    errors.extend(block_errors)
                    implicit_tool_blocks += implicit
            if renderer and rid in bodies:
                try:
                    text = renderer.render(bodies[rid])
                    ids_tok = renderer.tokenizer.encode(text, add_special_tokens=False)
                    n = len(ids_tok)
                    rec["rendered_prompt_tokens"] = n
                    if not _is_int(row.get("glm_tokens")):
                        errors.append(f"{rid}: glm_tokens must be an integer when Renderer validation is enabled")
                    elif row["glm_tokens"] != n:
                        errors.append(f"{rid}: glm_tokens={row['glm_tokens']} Renderer={n}")
                    if prev_tokens is not None:
                        lcp = 0
                        for a, b in zip(prev_tokens, ids_tok):
                            if a != b:
                                break
                            lcp += 1
                        rec["lcp_tokens"] = lcp
                    else:
                        lcp = 0
                        rec["lcp_tokens"] = 0
                    rec["added_tokens"] = n - lcp
                    rec["added_fraction_of_prompt"] = (n - lcp) / n if n else None
                    if p:
                        provenance_actual = (("prompt_tokens", n), ("lcp_tokens", lcp),
                                             ("added_tokens", n-lcp),
                                             ("removed_tokens", (len(prev_tokens)-lcp) if prev_tokens is not None else 0))
                        for key, actual in provenance_actual:
                            if not _is_int(p.get(key)):
                                errors.append(f"{rid}: provenance {key} must be an integer")
                            elif p[key] != actual:
                                errors.append(f"{rid}: provenance {key}={p[key]} rendered={actual}")
                        if "previous_req_id" not in p or p.get("previous_req_id") != prev_rid:
                            errors.append(f"{rid}: provenance previous_req_id does not match harness chain predecessor")
                    if has_regenerated_labels:
                        for key, actual in (("glm_lcp_with_prev", lcp), ("uncached_expected", n-lcp)):
                            value = row.get(key)
                            if not _is_int(value):
                                errors.append(f"{rid}: {p['kind']} {key} must be an integer")
                            elif value != actual:
                                errors.append(f"{rid}: {p['kind']} {key}={value} rendered={actual}")
                    if rec.get("lcp_tokens") is not None:
                        token_lcps.append(rec["lcp_tokens"])
                    prev_tokens = ids_tok
                    prev_rid = rid
                except Exception as exc:
                    if isinstance(exc, ImportError) and "jinja2" in str(exc).lower():
                        renderer_unavailable[str(exc)] += 1
                    else:
                        errors.append(f"{rid}: Renderer failed: {exc!r}")
            rec["uncached_expected"] = row.get("uncached_expected")
            per_request[rid] = rec
            if rid in bodies:
                prev_body = bodies[rid]

    workload_ledger = None
    if generated:
        workload_ledger, ledger_errors = _workload_ledger(
            chain_meta, grouped, req_rows, provenance, per_request)
        errors.extend(ledger_errors)

    # Original-source LCP labels refer to their source predecessor. Preserve
    # those labels and report the cohort-visible rendered LCP independently.
    if renderer_unavailable:
        warnings.append(f"original Renderer could not render {sum(renderer_unavailable.values())} prompts: "
                        + next(iter(renderer_unavailable)))
    render_complete = bool(renderer) and not renderer_unavailable and not any(
        "Renderer failed:" in e for e in errors)
    status = "VALID" if render_complete and not errors else "STRUCTURAL_OK"
    if not tok_dir:
        warnings.append("full Renderer/tokenizer validation unavailable; structural checks only, not a complete PASS")
    elif not render_complete:
        errors.append("tokenizer was explicitly requested, but full Renderer validation did not complete")
    if cohort_ids is None:
        warnings.append("no frozen cohort supplied; checking the complete serving set")
    if cohort is not None:
        expected_chains = {x.get("chain_id") for x in (cohort_obj.get("chains", []) if isinstance(cohort_obj, dict) else [])}
        actual_chains = set(grouped)
        if expected_chains and expected_chains != actual_chains:
            warnings.append("cohort chain set differs from chains represented by selected request IDs")
    if implicit_tool_blocks:
        warnings.append(f"{implicit_tool_blocks} introduced tool blocks have count-matched but implicit IDs")

    def dist(vals):
        valid = [x for x in vals if isinstance(x, (int, float))]
        return {"n": len(valid), "min": min(valid) if valid else None,
                "p50": _q(valid, .5), "p90": _q(valid, .9), "p95": _q(valid, .95),
                "max": max(valid) if valid else None, "mean": statistics.mean(valid) if valid else None}
    def bucket(v, edges):
        for e in edges:
            if v <= e:
                return f"<={e}"
        return f">{edges[-1]}"
    joint = collections.Counter()
    phase_round = collections.Counter()
    grouped_stats = {key: collections.defaultdict(list) for key in ("pack", "family", "phase", "kind")}
    for rec in per_request.values():
        phase_round[(str(rec["phase"]), bucket(rec.get("round") or 0, [1, 2, 4, 8, 16, 32, 64]))] += 1
        joint[(str(rec["phase"]), bucket(rec.get("round") or 0, [1, 2, 4, 8, 16, 32, 64]),
               bucket(rec.get("prompt_tokens") or 0, [4096, 16384, 65536]),
               bucket(rec.get("output_budget") or 0, [512, 2048, 8192]),
               bucket(rec.get("gap_ms") or 0, [1000, 10000, 60000]))] += 1
        for dimension in grouped_stats:
            grouped_stats[dimension][str(rec.get(dimension) or "?")].append(rec)
    group_report = {}
    for dimension, groups in grouped_stats.items():
        group_report[dimension] = {}
        for key, recs in sorted(groups.items()):
            group_report[dimension][key] = {
                "n": len(recs),
                "prompt_tokens": dist([x.get("prompt_tokens") for x in recs]),
                "lcp_tokens": dist([x.get("lcp_tokens") for x in recs]),
                "added_tokens": dist([x.get("added_tokens") for x in recs]),
                "added_fraction_of_prompt": dist([x.get("added_fraction_of_prompt") for x in recs]),
                "uncached_expected": dist([x.get("uncached_expected") for x in recs]),
                "output_budget": dist([x.get("output_budget") for x in recs]),
                "gap_ms": dist([x.get("gap_ms") for x in recs]),
            }
    kind_comparison = {}
    for kind in sorted({x.get("kind") for x in per_request.values()}):
        recs = [x for x in per_request.values() if x.get("kind") == kind]
        kind_comparison[kind] = {"n": len(recs),
                                 "prompt_tokens": dist([x.get("prompt_tokens") for x in recs]),
                                 "output_budget": dist([x.get("output_budget") for x in recs]),
                                 "gap_ms": dist([x.get("gap_ms") for x in recs]),
                                 "lcp_tokens": dist([x.get("lcp_tokens") for x in recs]),
                                 "added_tokens": dist([x.get("added_tokens") for x in recs]),
                                 "added_fraction_of_prompt": dist([x.get("added_fraction_of_prompt") for x in recs])}
    donor_rows = [p for p in provenance.values() if p.get("kind") == "synthetic" and p.get("donor_fingerprint")]
    donor_freq = collections.Counter(p["donor_fingerprint"] for p in donor_rows)
    donor_chain_freq = collections.Counter((p.get("chain_id") or p.get("source_chain_id"), p["donor_fingerprint"])
                                           for p in donor_rows)
    donor_reuse = {
        "synthetic_requests_with_fingerprint": len(donor_rows),
        "unique_fingerprints": len(donor_freq),
        "fingerprints_used_more_than_once": sum(n > 1 for n in donor_freq.values()),
        "extra_uses_after_first_global": sum(max(0, n - 1) for n in donor_freq.values()),
        "max_uses_of_one_fingerprint_global": max(donor_freq.values(), default=0),
        "max_uses_of_one_fingerprint_within_chain": max(donor_chain_freq.values(), default=0),
        "same_family_uses": sum(p.get("donor_same_family") is True for p in donor_rows),
        "cross_family_uses": sum(p.get("donor_same_family") is False for p in donor_rows),
        "donor_mode_counts": dict(collections.Counter(str(p.get("donor_mode") or "?") for p in donor_rows)),
        "interpretation": "Donor reuse counts are provenance counts only; they do not imply GPU cache hits or cross-chain prefix sharing.",
    }
    rendered_recs = [x for x in per_request.values() if isinstance(x.get("rendered_prompt_tokens"), int)
                     and isinstance(x.get("lcp_tokens"), int)]
    rendered_prompt_sum = sum(x["rendered_prompt_tokens"] for x in rendered_recs)
    rendered_added_sum = sum(x["rendered_prompt_tokens"] - x["lcp_tokens"] for x in rendered_recs)
    actual_token_accounting = {
        "requests": len(rendered_recs), "rendered_prompt_tokens": rendered_prompt_sum,
        "new_tokens_after_visible_lcp": rendered_added_sum,
        "added_fraction_of_prompt_tokens": rendered_added_sum / rendered_prompt_sum if rendered_prompt_sum else None,
        "lcp_share_of_prompt_tokens": (rendered_prompt_sum-rendered_added_sum) / rendered_prompt_sum if rendered_prompt_sum else None,
        "interpretation": "Rendered prompt/LCP accounting for this visible replay; not a prediction of physical KV reuse or GPU cache hits.",
    }
    manifest_chains = manifest.get("chain_summaries", [])
    def manifest_sum(field):
        vals = [c.get(field) for c in manifest_chains]
        if not vals:
            return None
        if any(not isinstance(v, (int, float)) for v in vals):
            warnings.append(f"manifest chain summaries lack complete numeric {field}; aggregate omitted")
            return None
        return sum(vals)
    manifest_checks = {
        "source_aggregate": manifest.get("selected_source_aggregate", manifest.get("source_aggregate")),
        "source_output_aggregate_conflict_chains": sum(bool(c.get("source_output_aggregate_conflict")) for c in manifest_chains),
        "visible_start_after_source_start_chains": sum(bool(c.get("visible_start_after_source_start")) for c in manifest_chains),
        "manifest_chain_summaries": len(manifest_chains),
        "source_prompt_sum": manifest_sum("source_prompt_sum"),
        "generated_prompt_sum": manifest_sum("prompt_sum"),
        "source_output_sum": manifest_sum("source_output_sum"),
        "generated_output_sum": manifest_sum("output_sum"),
    }
    for key in ("source_prompt_sum", "generated_prompt_sum", "source_output_sum", "generated_output_sum"):
        manifest_checks[key.replace("_sum", "_mean_per_chain")] = (
            manifest_checks[key] / len(manifest_chains)
            if manifest_chains and manifest_checks[key] is not None else None)
    source_phase_counts = collections.Counter()
    generated_phase_counts = collections.Counter()
    for c in chain_meta.values():
        source_target = c.get("source_chain_targets") or {}
        source_phase_counts.update(source_target.get("phases") or {})
        generated_phase_counts.update(c.get("phases") or {})
    manifest_checks["source_vs_generated_phase_counts"] = [
        {"phase": phase, "source_target_requests": source_phase_counts.get(phase, 0),
         "generated_actual_requests": generated_phase_counts.get(phase, 0),
         "delta_requests": generated_phase_counts.get(phase, 0) - source_phase_counts.get(phase, 0)}
        for phase in sorted(set(source_phase_counts) | set(generated_phase_counts))]
    report = {
        "status": status, "complete_pass": False, "errors": errors, "warnings": warnings,
        "counts": {"canon_request_ids": len(req_rows), "serving_request_ids": len(serving),
                   "expected_ids": len(target_ids), "body_ids": len(body_ids),
                   "matched_bodies": len(target_ids & body_ids), "chains": len(grouped),
                   "source_chain_length_mismatches": source_length_mismatches,
                   "rendered_prompts": sum("rendered_prompt_tokens" in x for x in per_request.values()),
                   "synthetic_tool_blocks_implicit_ids": implicit_tool_blocks},
        "distribution": {"chain_length": dist(chain_lengths), "prompt_tokens": dist(prompt_tokens),
                         "output_budget": dist(output_tokens), "gap_ms": dist(gaps),
                         "rendered_lcp_tokens": dist(token_lcps),
                         "rendered_added_fraction_of_prompt_tokens": actual_token_accounting["added_fraction_of_prompt_tokens"],
                         "by_group": group_report,
                         "source_vs_generated_kind": kind_comparison,
                         "joint_phase_round_prompt_output_gap": [
                             {"phase": k[0], "round_bucket": k[1], "prompt_bucket": k[2],
                              "output_bucket": k[3], "gap_bucket": k[4], "n": n}
                             for k,n in sorted(joint.items())],
                         "phase_round": [{"phase": k[0], "round_bucket": k[1], "n": n}
                                         for k,n in sorted(phase_round.items())],
                         "joint_phase_prompt_output_gap": [
                             {"phase": k[0], "round_bucket": k[1], "prompt_bucket": k[2],
                              "output_bucket": k[3], "gap_bucket": k[4], "n": n}
                             for k,n in sorted(joint.items())]},
        "per_request": per_request,
        "manifest_checks": manifest_checks,
        "artifact_checks": artifact_validation,
        "source_hash_checks": source_hash_checks,
        "workload_ledger": workload_ledger,
        "donor_reuse": donor_reuse,
        "actual_token_accounting": actual_token_accounting,
        "calibration_note": "Validates this dataset against its own metadata; does not establish hidden-set calibration.",
    }
    if provenance:
        report["provenance"] = {"rows": len(provenance), "path": "provenance.jsonl"}
    if errors:
        report["status"] = "INVALID"
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--harness-dir", required=True)
    ap.add_argument("--tok-dir")
    ap.add_argument("--cohort", help="frozen cohort JSON")
    ap.add_argument("--out-json")
    ap.add_argument("--out-text")
    ap.add_argument("--out-ledger-csv")
    args = ap.parse_args(argv)
    report = check_dataset(args.root, args.harness_dir, args.tok_dir, args.cohort)
    blob = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as fh:
            fh.write(blob + "\n")
    text = (f"{report['status']}: {report['counts']['expected_ids']} expected requests, "
            f"{report['counts']['matched_bodies']} bodies, {len(report['errors'])} errors\n")
    text += "\n".join("ERROR: " + x for x in report["errors"][:100])
    text += "\n".join("WARNING: " + x for x in report["warnings"])
    if args.out_text:
        with open(args.out_text, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    if args.out_ledger_csv:
        ledger = report.get("workload_ledger") or {"rows": []}
        fields = ["chain_id", "source_requests", "visible_original_requests", "synthetic_requests",
                  "source_frozen_uncached", "visible_original_source_frozen_uncached",
                  "visible_original_current_frozen_uncached", "adapted_original_frozen_delta",
                  "missing_source_frozen_uncached", "generated_synthetic_lcp_added",
                  "missing_budget_delta", "generated_frozen_uncached",
                  "generated_cohort_visible_lcp_added", "source_prompt_sum", "generated_prompt_sum",
                  "source_output_sum", "generated_output_sum"]
        with open(args.out_ledger_csv, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(ledger.get("rows", []))
    print(text)
    return 0 if report["status"] in ("STRUCTURAL_OK", "VALID") else 2


if __name__ == "__main__":
    raise SystemExit(main())
