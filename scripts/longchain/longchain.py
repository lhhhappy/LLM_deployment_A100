#!/usr/bin/env python3
"""Build frozen long-chain mechanism workloads from public S1 histories.

The original replay and harness are read-only. Missing turns are synthetic;
source aggregate constraints do not reconstruct a hidden trace. No model/API or
live service is called. This CLI provides build, polish, and check.
"""
from __future__ import annotations

import argparse
import collections
import copy
import dataclasses
import gzip
import hashlib
import importlib
import importlib.metadata
import io
import json
import math
import os
import platform
from pathlib import Path
import random
import re
import resource
import sys

REPO = Path(__file__).resolve().parents[2]
VERSION = "source-event-longchain-v3"
VERSION_TOPUP = "source-event-longchain-v4"  # v3 rules + rewrite top-up (--rewrite-topup)
# Lower bound of the factor that scales continuation growth down to the chain's frozen prompt total.
GROWTH_SCALE_MIN = 1 / 16


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except Exception as exc:
                    raise ValueError(f"{path}:{number}: invalid JSON") from exc


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def implementation_receipt(harness_dir):
    """Record the code and CPU rendering environment used for this artifact."""
    harness_dir = Path(harness_dir).resolve()
    return {"generator_sha256": file_digest(Path(__file__)),
            "python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name)
                         for name in ("transformers", "tokenizers", "jinja2")},
            "harness_files": {name: file_digest(harness_dir / name)
                              for name in ("s1_common.py", "s1_loadgen.py")}}


def req_id(row):
    return f"{row['pack']}:{row['view']}:{row['logical_call_id']}"


def order(row):
    return row.get("dispatch_offset_ms") or 0, row.get("logical_call_id") or ""


def load_source(root):
    rows, chains, bodies = {}, {}, {}
    for row in read_jsonl(root / "requests.jsonl"):
        if row.get("view") != "canon" or not row.get("in_serving_load"):
            continue
        rid = req_id(row)
        if rid in rows:
            raise ValueError(f"duplicate source request {rid}")
        rows[rid] = row
    for c in read_jsonl(root / "chains.jsonl"):
        if c.get("view") == "canon":
            if c["chain_id"] in chains:
                raise ValueError("duplicate source chain")
            chains[c["chain_id"]] = c
    for path in sorted((root / "bodies").rglob("*.jsonl.gz")):
        for b in read_jsonl(path):
            rid = b.get("req_id")
            if rid not in rows:
                continue
            if rid in bodies:
                raise ValueError(f"duplicate source body {rid}")
            bodies[rid] = b
    if not rows or set(rows) != set(bodies):
        raise ValueError(f"source body coverage: {len(bodies)}/{len(rows)}")
    grouped = collections.defaultdict(list)
    for row in rows.values():
        grouped[row["chain_id"]].append(row)
    for cid, rs in grouped.items():
        rs.sort(key=order)
        if cid not in chains or chains[cid]["n_requests"] < len(rs):
            raise ValueError(f"invalid source chain target {cid}")
    return rows, {c: chains[c] for c in grouped}, bodies, dict(grouped)


def length_bin(n):
    return sum(n > edge for edge in (1, 4, 9, 19, 49, 99))


def apportion(total, weights):
    """Deterministic largest-remainder allocation, without losing the total."""
    if total < 0 or not weights or any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("invalid apportionment")
    scale = total / sum(weights)
    exact = [w * scale for w in weights]
    result = [math.floor(x) for x in exact]
    for i in sorted(range(len(weights)), key=lambda i: (-(exact[i] - result[i]), i))[:total - sum(result)]:
        result[i] += 1
    return result


def aggregate(chains):
    n = sum(c["n_requests"] for c in chains)
    counts = collections.Counter(c.get("sys_tools_hash") for c in chains)
    return {"chains": len(chains), "requests": n,
            "mean_length": n / max(1, len(chains)),
            "mean_prompt": sum(c["sum_glm_tokens"] for c in chains) / max(1, n),
            "mean_output": sum(c["max_output_i_sum"] for c in chains) / max(1, n),
            "family_pair_probability": sum(v * (v - 1) for v in counts.values()) / max(1, len(chains) * (len(chains) - 1)),
            "family_count": len(counts)}


def choose_chains(chains, count, seed):
    """Stratify pack/length, then retain a draw with similar aggregate/family mix.

    Selection sees source metadata only, never candidate engine performance.
    Family collision probability is used instead of demanding that every small
    cohort contain the single largest source family or the 240-turn maximum.
    """
    if not 1 <= count <= len(chains):
        raise ValueError(f"chain count must be 1..{len(chains)}")
    strata = collections.defaultdict(list)
    for cid in sorted(chains):
        c = chains[cid]
        strata[(c["pack"], length_bin(c["n_requests"]))].append(cid)
    keys = sorted(strata)
    quotas = apportion(count, [len(strata[k]) for k in keys])
    target = aggregate(list(chains.values()))
    rng = random.Random(seed)
    best, best_score = None, float("inf")
    for _ in range(128):
        selected = [cid for k, q in zip(keys, quotas) for cid in rng.sample(strata[k], q)]
        current = aggregate([chains[c] for c in selected])
        score = sum(abs(math.log(max(current[k], 1e-9) / max(target[k], 1e-9)))
                    for k in ("mean_length", "mean_prompt", "mean_output"))
        score += abs(current["family_pair_probability"] - target["family_pair_probability"]) * 8
        if score < best_score:
            best, best_score = sorted(selected), score
    return best


def reminder(message):
    return message.get("role") == "user" and "system-reminder" in str(message.get("content", ""))


def transition_messages(before, after):
    """Only append or replace the *last reminder*; never rewrite old history."""
    i = 0
    while i < min(len(before), len(after)) and before[i] == after[i]:
        i += 1
    replace = i == len(before) - 1 and reminder(before[-1])
    if i != len(before) and not replace:
        return None
    suffix = after[i:]
    if not suffix or not any(m.get("role") == "assistant" for m in suffix):
        return None
    return suffix, replace


def continuation_phase(messages):
    """Label the event we generated, not an unobserved donor-source event.

    This generator only appends messages or replaces a terminal reminder. It
    does not implement a context rebuild. A new human message starts a turn;
    tools and their reminders continue the current turn, regardless of LCP.
    """
    return "turn_start" if any(m.get("role") == "user" and not reminder(m)
                                for m in messages) else "intra"


def observed_append_edges(rows, bodies):
    """Count the cohort-visible message transitions, not source edge labels."""
    count = 0
    for before, after in zip(rows, rows[1:]):
        a, b = bodies[req_id(before)], bodies[req_id(after)]
        if (a.get("system") == b.get("system") and a.get("tools") == b.get("tools")
                and transition_messages(a["messages"], b["messages"]) is not None):
            count += 1
    return count


def chain_record(source, rows, append_edges):
    """Keep source targets nested; every top-level aggregate describes output."""
    total_edges = len(rows) - 1
    if not rows or not 0 <= append_edges <= total_edges:
        raise ValueError("invalid generated chain edge accounting")
    head = rows[0]
    return {**{k: head[k] for k in ("chain_id", "pack", "view", "session_id", "chain_index")},
            "split": "synthetic", "n_requests": len(rows),
            "first_dispatch_offset_ms": head.get("dispatch_offset_ms"),
            "last_end_offset_ms": rows[-1].get("end_offset_ms"),
            "sum_glm_tokens": sum(r["glm_tokens"] for r in rows),
            "sum_uncached_expected": sum(r["uncached_expected"] for r in rows),
            "max_output_i_sum": sum(r["max_output_i"] for r in rows),
            "phases": dict(collections.Counter(r["phase"] for r in rows)),
            "sys_tools_hash": head.get("sys_tools_hash"),
            "total_edges": total_edges, "append_only_edges": append_edges,
            "append_only_frac": append_edges / total_edges if total_edges else None,
            "append_definition": "cohort message append or terminal reminder replacement, unchanged system/tools",
            "truncated": any(bool(r.get("truncated")) for r in rows),
            "source_chain_targets": source.get("source_chain_targets", source)}


def tool_names(messages):
    return frozenset((t.get("function") or t).get("name")
                     for m in messages for t in (m.get("tool_calls") or []))


def available_tools(body):
    return frozenset((t.get("function") or t).get("name") for t in (body.get("tools") or []))


def tool_schemas(body):
    result = {}
    for raw in body.get("tools") or []:
        tool = raw.get("function") or raw
        result[tool.get("name")] = tool.get("parameters", tool.get("input_schema"))
    return result


def tool_pairing(messages):
    """Classify complete blocks without inventing IDs absent from source data."""
    implicit = False
    covered_tools = set()
    for i, message in enumerate(messages):
        calls = message.get("tool_calls") or []
        if not calls:
            continue
        result_ids = []
        j = i + 1
        while j < len(messages) and messages[j].get("role") == "tool":
            covered_tools.add(j)
            result = messages[j]
            content = result.get("content")
            entries = content if isinstance(content, list) and content and all(isinstance(e, dict) and "output" in e for e in content) else [result]
            result_ids.extend(e.get("tool_call_id") or e.get("id") for e in entries)
            j += 1
        call_ids = [c.get("id") or c.get("tool_call_id") for c in calls]
        if len(result_ids) != len(calls):
            return "incomplete"
        if len(set(x for x in call_ids if x)) != len([x for x in call_ids if x]):
            return "incomplete"
        if all(call_ids) and all(result_ids):
            if collections.Counter(call_ids) != collections.Counter(result_ids):
                return "incomplete"
        else:
            explicit = [x for x in result_ids if x]
            if len(set(explicit)) != len(explicit) or any(x not in call_ids for x in explicit):
                return "incomplete"
            implicit = True
    if any(m.get("role") == "tool" and i not in covered_tools for i, m in enumerate(messages)):
        return "incomplete"
    return "implicit_order" if implicit else "explicit_or_no_tools"


@dataclasses.dataclass
class Donor:
    source_req_id: str
    chain_id: str
    pack: str
    family: str
    messages: list
    replace_reminder: bool
    kind: str
    phase: str
    weight: int
    gap: int
    gap_origin: str
    net_think_ms: int | None
    tool_union_ms: int | None
    increment: int
    fingerprint: str
    names: frozenset
    schemas: dict


def extract_donors(grouped, bodies, renderer):
    donors, rejected, histories = [], collections.Counter(), []
    for cid in sorted(grouped):
        rs = grouped[cid]
        for prev, row in zip(rs, rs[1:]):
            bp, bn = bodies[req_id(prev)], bodies[req_id(row)]
            if bp.get("system") != bn.get("system") or bp.get("tools") != bn.get("tools"):
                rejected["changed_system_or_tools"] += 1
                continue
            delta = transition_messages(bp["messages"], bn["messages"])
            if delta is None:
                rejected["not_append_or_last_reminder"] += 1
                continue
            messages, replace = delta
            if tool_pairing(messages) == "incomplete":
                rejected["incomplete_tool_pair"] += 1
                continue
            if not row.get("gap_valid") or row.get("replay_gap_ms") is None:
                rejected["unobserved_gap"] += 1
                continue
            inc = row["glm_tokens"] - prev["glm_tokens"]
            if inc <= 0:
                rejected["nonpositive_growth"] += 1
                continue
            donors.append(Donor(req_id(row), cid, row["pack"], row.get("sys_tools_hash"),
                                messages, replace, "observed_transition", row["phase"],
                                int(row["max_output_i"]), int(row["replay_gap_ms"]),
                                "observed_adjacent_transition", row.get("net_think_ms"), row.get("tool_union_ms"),
                                inc, digest(messages), tool_names(messages), tool_schemas(bn)))
        # Last public history includes earlier blocks. Keep whole assistant +
        # contiguous results + optional reminder, rather than transplanting a
        # complete prompt. These blocks do not have measured future gap/output.
        row = rs[-1]
        messages = bodies[req_id(row)]["messages"]
        for i, m in enumerate(messages):
            if m.get("role") != "assistant" or not m.get("tool_calls"):
                continue
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                j += 1
            if j == i + 1:
                continue
            if j < len(messages) and reminder(messages[j]):
                j += 1
            if tool_pairing(messages[i:j]) == "incomplete":
                rejected["incomplete_historical_tool_pair"] += 1
            else:
                histories.append((row, messages[i:j]))
    if not donors:
        raise ValueError("no actual adjacent transitions available")
    seen = {(d.chain_id, d.fingerprint) for d in donors}
    for row, messages in histories:
        fp = digest(messages)
        key = row["chain_id"], fp
        if key in seen:
            continue
        seen.add(key)
        # A cheap isolated rendering only estimates the selection cost; actual
        # generated prompt tokens and LCP always come from a full rendering.
        inc = renderer.n_tokens(renderer.render({"messages": messages, "tools": [], "system": ""}))
        pool = [d for d in donors if d.kind == "observed_transition" and d.pack == row["pack"]]
        if not pool:
            continue
        paired = min(pool, key=lambda d: (0 if d.chain_id == row["chain_id"] else 1,
                                          0 if d.family == row.get("sys_tools_hash") else 1,
                                          abs(math.log((d.increment + 128) / (inc + 128))), d.source_req_id))
        donors.append(Donor(req_id(row), row["chain_id"], row["pack"], row.get("sys_tools_hash"),
                            messages, False, "historical_block", "intra", paired.weight, paired.gap,
                            f"estimated_from_transition:{paired.source_req_id}", paired.net_think_ms,
                            paired.tool_union_ms, max(1, inc), fp, tool_names(messages), tool_schemas(bodies[req_id(row)])))
    return donors, dict(rejected)


def rename_new_calls(messages, prefix):
    """Rename only the copied block, including references embedded in results."""
    calls = [t for m in messages for t in (m.get("tool_calls") or [])]
    old_ids = [str(t.get("id") or t.get("tool_call_id")) for t in calls
               if t.get("id") or t.get("tool_call_id")]
    if len(old_ids) != len(set(old_ids)):
        raise ValueError("copied block contains duplicate tool call IDs")
    mapping = {str(t.get("id") or t.get("tool_call_id")): f"{prefix}_{i}"
               for i, t in enumerate(calls) if t.get("id") or t.get("tool_call_id")}
    # One substitution pass prevents newly generated IDs being rewritten by a
    # second old ID. Identifier boundaries protect ordinary words from short
    # IDs (e.g. 'a' must not rewrite 'assistant' or 'data').
    pattern = re.compile(r"(?<![\w-])(?:" + "|".join(re.escape(x) for x in
                         sorted(mapping, key=len, reverse=True)) + r")(?![\w-])") if mapping else None
    def visit(value, path=()):
        if isinstance(value, str):
            if path in (("role",), ("tool_calls", "name"),
                        ("tool_calls", "type"), ("tool_calls", "function", "name")):
                return value
            if path == ("content", "type") and value in ("text", "image_url", "input_text", "input_image"):
                return value
            if path and path[-1] in ("id", "tool_call_id", "call_id", "tool_use_id"):
                return mapping.get(value, value)
            return pattern.sub(lambda m: mapping[m.group(0)], value) if pattern else value
        if isinstance(value, list):
            return [visit(v, path) for v in value]
        if isinstance(value, dict):
            return {k: visit(v, path + (k,)) for k, v in value.items()}
        return value
    return visit(messages)


def continuation_order_offset(previous, gap_ms):
    """Monotone synthetic ordering, never an observed service timestamp.

    Honor the last observed end when available. Subsequent synthetic ends are
    unknown; real pacing remains the separate replay_gap_ms used by the harness.
    """
    return max(previous.get("dispatch_offset_ms") or 0,
               previous.get("end_offset_ms") or 0) + max(1, gap_ms)


def verify_parent_artifacts(root, manifest):
    """Do not let a new polish manifest launder modified frozen input files."""
    root = Path(root).resolve()
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("polish parent lacks artifact hashes")
    files = {str(p.relative_to(root)) for p in root.rglob("*")
             if p.is_file() and p != root / "manifest.json"}
    if files != set(artifacts):
        raise ValueError("polish parent artifact inventory differs from manifest")
    for rel, expected in artifacts.items():
        path = (root / rel).resolve()
        if root not in path.parents or not path.is_file() or file_digest(path) != expected:
            raise ValueError(f"polish parent artifact hash/path mismatch: {rel}")


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def output_budgets(target_total, original_rows, donor_weights):
    """Preserve original budgets; allocate missing total using paired weights."""
    remaining = target_total - sum(r["max_output_i"] for r in original_rows)
    if not donor_weights:
        return [], remaining
    if remaining < 2 * len(donor_weights):
        raise ValueError("source output aggregate cannot cover missing turns without changing originals")
    shares = apportion(remaining - 2 * len(donor_weights), donor_weights)
    return [n + 2 for n in shares], 0


def json_dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_line(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def build(args):
    from longchain_events import EventCompiler, NoMaterialFits
    root, dest = Path(args.source_root).resolve(), Path(args.out).resolve()
    protected = [REPO / p for p in ("s1-dev", "llm-challenge-arena-v1", "build/base_exact", "refs")]
    if dest == root or root in dest.parents or any(dest == p or p in dest.parents for p in protected):
        raise ValueError("output must be separate from the read-only source trees")
    if dest.exists():
        raise ValueError("output already exists; choose a fresh directory (no overwrite)")
    if not args.set or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.set):
        raise ValueError("set must be a simple dataset name")
    sys.path.insert(0, str(Path(args.harness_dir).resolve()))
    common = importlib.import_module("s1_common")
    renderer = common.Renderer(str(Path(args.tok_dir).resolve()))
    rows, chains, bodies, grouped = load_source(root)
    chosen = choose_chains(chains, args.chains, args.seed)
    donors, rejected = extract_donors(grouped, bodies, renderer)
    compiler = EventCompiler(args, grouped, bodies, donors, renderer.tokenizer)
    print(f"SOURCE requests={len(rows)} chains={len(chains)} donors={len(donors)} selected_chains={len(chosen)}", flush=True)
    dest.mkdir(parents=True)
    (dest / "bodies").mkdir()
    (dest / "samples").mkdir()
    manifest = {"generator": VERSION_TOPUP if args.rewrite_topup else VERSION, "status": "BUILDING", "set": args.set, "seed": args.seed,
                "quality_status": "DIAGNOSTIC_CANDIDATE_NOT_REPRESENTATIVE",
                "implementation": implementation_receipt(args.harness_dir),
                "source_root": str(root), "selected_chain_ids": chosen,
                "source_aggregate": aggregate(list(chains.values())),
                "selected_source_aggregate": aggregate([chains[c] for c in chosen]),
                "donors": dict(collections.Counter(d.kind for d in donors)), "rejected_transitions": rejected,
                "tokenizer": {"path": str(Path(args.tok_dir).resolve()),
                              "files": {p.name: file_digest(p) for p in sorted(Path(args.tok_dir).iterdir()) if p.is_file()}},
                "source_files": {str(p.relative_to(root)): file_digest(p) for p in
                                 [root / "requests.jsonl", root / "chains.jsonl", *sorted((root / "bodies").rglob("*.jsonl.gz"))]},
                "implementation_event_compiler_sha256": file_digest(Path(__file__).with_name("longchain_events.py")),
                "read_corpus": compiler.read_corpus.receipt,
                "load_templates": {"origin": "public requests with a valid replay gap: non-head intra/turn_start, all context_reset",
                                   "n": {f"{kind}{'/rewrite' if rewrite else ''}": len(v)
                                         for (kind, rewrite), v in sorted(compiler.templates.items())}},
                "assumptions": ["Public content plus explicit synthetic events, not recovered hidden requests.",
                                "Public bodies/budgets/gaps/labels retained; session IDs scoped to generated receiving chains.",
                                "Source phase counts constrain event frequencies; event positions are unobserved and drawn at random.",
                                "Each synthesized step copies one public request of its kind and rewrite class (frozen uncached_expected, replay gap, output budget, trailing runtime reminder); output budgets are apportioned to the source chain's output total.",
                                "Public templates come from the first steps of chains; continuation growth is scaled down (never up, not below 1/16) when the source chain's frozen prompt total for the synthesized part would be exceeded. The new-token total is reported, not fitted.",
                                "A continuation whose donor block is shorter than its size gets one extra parallel Read call; its text is corpus filler read front to back, so only its token count is meaningful.",
                                "A turn start closes the finished turn as the public agent does: tool-call narration of that turn becomes empty and the runtime reminder follows the new query.",
                                "History rewrites per chain follow the source count (total_edges - append_only_edges, head edge excluded). A rewrite diverges with one extra Read call (at most 1024 tokens) in an earlier tool-call message, so the suffix is recomputed while the context barely grows; unplanned rewrites cancel later planned ones.",
                                "Planned context resets are appends sized like public context_reset requests (all public ones are appends); only context pressure rebuilds and shrinks the prompt.",
                                "Replay gaps are copied from public requests, which follow the organizer's capped tool-plus-thinking rule; the decomposition is not copied.",
                                "Synthetic dispatch offsets only order requests; original harness executes completion-relative gaps and chain gap cap.",
                                "Rebuilds use explicit receiving-history extracts and retain whole recent groups; no engine-side truncation.",
                                "No cache salt, per-session flush, N-dependent timing, tool execution or generated-output feedback.",
                                "Prompt aggregates are diagnostics, not forcibly fitted; context pressure may add explicit rebuild events."],
                "max_context_tokens": args.max_context_tokens,
                "formal_score": None}
    json_dump(dest / "manifest.json", manifest)
    summaries, used = [], collections.Counter()
    rng = random.Random(args.seed)
    # Low gzip level is intentional: this is a reproducible artifact, not a CPU
    # compression benchmark. mtime=0 avoids timestamps changing identical builds.
    shard_path = dest / "bodies" / f"{args.set}.jsonl.gz"
    with open(dest / "requests.jsonl", "w") as rf, open(dest / "provenance.jsonl", "w") as pf, \
         open(dest / "chains.jsonl", "w") as cf, open(dest / "samples" / f"{args.set}.jsonl", "w") as sf, \
         open(dest / "event-plans.jsonl", "w") as ef, \
         open(shard_path, "wb") as raw_shard, gzip.GzipFile(fileobj=raw_shard, mode="wb", compresslevel=1, mtime=0) as gz, \
         io.TextIOWrapper(gz, encoding="utf-8") as bf:
        for ci, cid in enumerate(chosen):
            original = grouped[cid]
            target = chains[cid]
            generated_session = f"lc_{args.seed}_{ci:03d}"
            chain_rows, chain_prov = [], []
            prev_tokens, previous = [], None
            for row in original:
                rid = req_id(row)
                b = bodies[rid]
                ids = renderer.tokenizer.encode(renderer.render(b), add_special_tokens=False)
                if len(ids) != row["glm_tokens"]:
                    raise ValueError(f"source Renderer mismatch {rid}: {len(ids)} != {row['glm_tokens']}")
                common_tokens = lcp(prev_tokens, ids) if previous else 0
                prov = {"req_id": rid, "source_req_id": rid, "source_chain_id": cid,
                        "source_session_id": row["session_id"], "receiving_session_id": generated_session,
                        "kind": "original", "donor_req_id": None, "donor_chain_id": None, "donor_mode": None,
                        "previous_req_id": previous, "prompt_tokens": len(ids), "lcp_tokens": common_tokens,
                        "added_tokens": len(ids) - common_tokens, "removed_tokens": len(prev_tokens) - common_tokens,
                        "target_chain_requests": target["n_requests"], "output_budget_origin": "original",
                        "gap_origin": "original", "body_sha256": digest(b),
                        "source_body_sha256": digest(b), "source_body_ref": row.get("body_ref")}
                output_row = copy.deepcopy(row)
                output_row["session_id"] = generated_session
                output_row["body_ref"] = f"bodies/{args.set}.jsonl.gz"
                chain_rows.append(output_row)
                chain_prov.append(prov)
                write_line(bf, b)
                prev_tokens, previous = ids, rid
            current_body = copy.deepcopy(bodies[req_id(original[-1])])
            pool = compiler.pool(target, current_body)
            if len(original) < target["n_requests"] and not pool:
                raise ValueError(f"no compatible whole-block donor for {cid}")
            plan = compiler.plan(target, original, pool)
            weights = [item["output_weight"] for item in plan]
            aggregate_conflict = bool(weights and target["max_output_i_sum"] - sum(r["max_output_i"] for r in original) < 2*len(weights))
            if aggregate_conflict:
                budgets = [max(2,w) for w in weights]
                budget_residual = target["max_output_i_sum"] - sum(r["max_output_i"] for r in original) - sum(budgets)
            else:
                budgets, budget_residual = output_budgets(target["max_output_i_sum"], original, weights)
            write_line(ef, {"chain_id": cid, "source_session_id": target["session_id"],
                           "receiving_session_id": generated_session, "events": plan, "output_budgets": budgets})
            usage = collections.Counter()
            prefix = generated_session
            total_prompt = sum(r["glm_tokens"] for r in chain_rows)
            generated_append_edges = 0
            # Each step's size is its public template's. Public templates come from the first steps of
            # chains; a growth persists in every later prompt, so when the chain's frozen prompt total
            # (synthesized part) would be exceeded, continuation growth is scaled down (never up). The
            # new-token total is reported as a check, not fitted.
            new_token_budget = max(0, target["sum_uncached_expected"] - sum(r["uncached_expected"] for r in original))
            prompt_budget = max(0, target["sum_glm_tokens"] - sum(r["glm_tokens"] for r in original))
            shapes = [compiler.step_shape(item) if item["kind"] in ("intra", "context_reset") else 0 for item in plan]
            # The realized number of history rewrites follows the plan: a rewrite nobody planned (a turn
            # close with narration, a pressure rebuild) cancels the last planned continuation rewrite ahead.
            target_rewrites, rewrites = sum(item["rewrite"] for item in plan), 0
            new_tokens = prompt_tokens = 0
            topups = 0
            for offset, (item, budget) in enumerate(zip(plan, budgets)):
                step = len(original) + offset
                before_body = current_body
                room = args.max_context_tokens - len(prev_tokens) - budget - 256
                logical_id = f"{prefix}:llm:{step:04d}"
                rid = f"{target['pack']}:canon:{logical_id}"
                if shapes[offset]:
                    # Prompt tokens left above carrying the current context to the end, shared by the
                    # remaining growth steps in proportion to how many later prompts carry each one.
                    rest = range(offset, len(plan))
                    headroom = prompt_budget - prompt_tokens - len(rest) * len(prev_tokens)
                    cost = sum(shapes[j] * (len(plan) - j) for j in rest)
                    scale = min(1.0, max(GROWTH_SCALE_MIN, headroom / cost)) if cost else 1.0
                    item = {**item, "growth_scale": scale, "step_target": max(1, round(shapes[offset] * scale))}
                if args.rewrite_topup and item.get("rewrite") and item["kind"] in ("intra", "turn_start"):
                    # v4: share the chain's remaining new-token deficit over the planned rewrites still ahead
                    # (this one included). A rewrite recomputes a suffix of the history, so it adds new tokens
                    # without growing the prompt; diverge_to_target caps the recompute at the whole history.
                    ahead = [j for j in range(offset, len(plan)) if plan[j].get("rewrite") and plan[j]["kind"] in ("intra", "turn_start")]
                    # What the steps still ahead will deliver on their own: growth steps after the same
                    # prompt-budget scaling the loop applies (v3's shortfall comes mostly from that
                    # scaling), rewriting turn starts their template's new tokens, other turn starts 0.
                    rest = range(offset, len(plan))
                    headroom = prompt_budget - prompt_tokens - len(rest) * len(prev_tokens)
                    cost = sum(shapes[j] * (len(plan) - j) for j in rest)
                    proj_scale = min(1.0, max(GROWTH_SCALE_MIN, headroom / cost)) if cost else 1.0
                    planned_rest = sum(shapes[j] * proj_scale if shapes[j] else
                                       (compiler.step_shape(plan[j]) if plan[j].get("rewrite") and plan[j]["kind"] == "turn_start" else 0)
                                       for j in rest)
                    deficit = new_token_budget - new_tokens - planned_rest
                    if deficit > 0 and ahead:
                        # Only the divergence target grows (earlier divergence = larger recomputed suffix); the
                        # appended block keeps its template/growth-scaled size so prompt totals stay matched.
                        item = {**item, "divergence_topup": int(deficit / len(ahead))}
                        topups += 1
                try:
                    current_body, event_receipt, donor = compiler.event(item, target, before_body, prefix, step, usage, room)
                    current_body["req_id"] = rid
                    ids = renderer.tokenizer.encode(renderer.render(current_body), add_special_tokens=False)
                    pressure = len(ids) + budget > args.max_context_tokens
                except NoMaterialFits:
                    pressure = True
                active_item = item
                if pressure:
                    # Whole-event rebuild, never truncate a request sent to the engine.
                    forced = {**item, "kind": "context_reset", "position_origin": "explicit_context_pressure",
                              "rebuild": True, "rewrite": False}
                    forced["template"] = compiler.template("context_reset")
                    active_item = forced
                    current_body, event_receipt, donor = compiler.event(forced, target, before_body, prefix, step, usage, room)
                    event_receipt["displaced_planned_kind"] = item["kind"]
                    current_body["req_id"] = rid
                    ids = renderer.tokenizer.encode(renderer.render(current_body), add_special_tokens=False)
                if event_receipt.get("rebuild"):
                    # An extract can be longer than a small removed middle.
                    # Recompile the same explicit rebuild with less retained
                    # material, recording the adjustment, never relabel it.
                    for excerpt_chars, excerpt_count in ((120, 4), (32, 1), (1, 1)):
                        if len(ids) < len(prev_tokens) and len(ids) + budget <= args.max_context_tokens:
                            break
                        adjusted = {**active_item, "kind": "context_reset", "rebuild_limits": {
                            "keep_fraction": 0, "excerpt_chars": excerpt_chars, "excerpt_count": excerpt_count}}
                        current_body, event_receipt, donor = compiler.event(adjusted, target, before_body, prefix, step, usage, room)
                        if pressure:
                            event_receipt["displaced_planned_kind"] = item["kind"]
                        current_body["req_id"] = rid
                        ids = renderer.tokenizer.encode(renderer.render(current_body), add_special_tokens=False)
                common_tokens = lcp(prev_tokens, ids)
                if len(ids) + budget > args.max_context_tokens:
                    raise ValueError(f"context limit exceeded for {rid}; no truncation performed")
                phase = event_receipt["event_kind"]
                gap = event_receipt["replay_gap_ms"]
                new_tokens += len(ids) - common_tokens
                prompt_tokens += len(ids)
                is_append = transition_messages(before_body["messages"], current_body["messages"]) is not None
                generated_append_edges += is_append
                if event_receipt.get("rebuild") and len(ids) >= len(prev_tokens):
                    raise ValueError(f"rebuild must reduce actual GLM prompt tokens: {rid}")
                if not is_append:
                    rewrites += 1
                    if not active_item.get("rewrite"):
                        planned = [j for j in range(offset + 1, len(plan)) if plan[j]["rewrite"] and plan[j]["kind"] == "intra"]
                        if planned:
                            plan[planned[-1]]["rewrite"] = False
                compiler.commit(event_receipt, usage)
                synthetic = {"pack": target["pack"], "view": "canon", "session_id": generated_session,
                             "chain_id": cid, "chain_index": target.get("chain_index", 0), "logical_call_id": logical_id,
                             "split": "synthetic", "phase": phase, "protocol": VERSION,
                             "dispatch_offset_ms": continuation_order_offset(chain_rows[-1], gap),
                             "first_output_offset_ms": None, "end_offset_ms": None,
                             "glm_tokens": len(ids), "glm_lcp_with_prev": common_tokens,
                             "uncached_expected": len(ids) - common_tokens,
                             # Edge labels use the organizer's vocabulary so rewrite counts compare directly.
                             "edge_type": "append-only" if is_append else "compact-rebuild" if event_receipt.get("rebuild")
                                          else "unexplained-break",
                             "edge_subtype": phase,
                             "break_reason": None if is_append else "explicit_context_rebuild" if event_receipt.get("rebuild")
                                             else "divergence" if (event_receipt.get("divergence") or {}).get("diverged_at") is not None
                                             else "finished_turn_narration_removed",
                             "sys_tools_hash": original[-1].get("sys_tools_hash"),
                             "prefix_family_id": original[-1].get("prefix_family_id"),
                             "max_output_i": budget, "replay_gap_ms": gap, "gap_valid": True,
                             "gap_imputed": True, "gap_invalid_reason": None,
                             "net_think_ms": None, "tool_union_ms": None,
                             "body_ref": f"bodies/{args.set}.jsonl.gz", "in_serving_load": True, "in_main": True,
                             "sampling_weight": 1, "weighted": 1, "excluded_reason": None,
                             "over_context": False, "truncated": False, "over_limit_after_trunc": False,
                             "long_output_excluded": False, "in_truncated_chain": False}
                prov = {"req_id": rid, "source_req_id": req_id(original[-1]), "source_chain_id": cid,
                        "source_session_id": target["session_id"], "receiving_session_id": generated_session,
                        "dispatch_offset_origin": "synthetic_order_only_not_observed_time",
                        "kind": "synthetic", "previous_req_id": previous,
                        "prompt_tokens": len(ids), "lcp_tokens": common_tokens,
                        "added_tokens": len(ids) - common_tokens, "removed_tokens": len(prev_tokens) - common_tokens,
                        "target_chain_requests": target["n_requests"], "body_sha256": digest(current_body),
                        **event_receipt}
                if aggregate_conflict:
                    prov["output_budget_origin"] = "rank_transferred_public_weight_due_source_aggregate_conflict"
                used[event_receipt["donor_mode"]] += 1
                chain_rows.append(synthetic)
                chain_prov.append(prov)
                write_line(bf, current_body)
                total_prompt += len(ids)
                prev_tokens, previous = ids, rid
            for row, prov in zip(chain_rows, chain_prov):
                if row["glm_tokens"] + row["max_output_i"] > args.max_context_tokens:
                    raise ValueError(f"prompt+output exceeds context: {req_id(row)}")
                write_line(rf, row)
                write_line(pf, prov)
            phase_counts = dict(collections.Counter(r["phase"] for r in chain_rows))
            out_chain = chain_record(target, chain_rows,
                                     observed_append_edges(original, bodies) + generated_append_edges)
            write_line(cf, out_chain)
            write_line(sf, {"chain_id": cid, "set_role": "source-length-stratified"})
            summary = {"chain_id": cid, "requests": len(chain_rows), "original_requests": len(original),
                       "prompt_sum": total_prompt, "source_prompt_sum": target["sum_glm_tokens"],
                       "prompt_sum_relative_error": total_prompt / target["sum_glm_tokens"] - 1,
                       # Targets are the source chain's totals minus its public requests: the synthesized part.
                       "synthesized_new_tokens": new_tokens, "target_new_tokens": new_token_budget,
                       "synthesized_prompt_tokens": prompt_tokens, "target_prompt_tokens": prompt_budget,
                       "synthesized_rewrites": rewrites, "target_rewrites": target_rewrites, "rewrite_topups": topups,
                       "new_tokens_relative_error": new_tokens / new_token_budget - 1 if new_token_budget else None,
                       "output_sum": out_chain["max_output_i_sum"], "source_output_sum": target["max_output_i_sum"],
                       "source_output_residual": budget_residual, "phases": phase_counts, "source_phases": target["phases"],
                       "source_output_aggregate_conflict": aggregate_conflict,
                       "visible_start_after_source_start": order(original[0])[0] > (target.get("first_dispatch_offset_ms") or 0),
                       "unique_donor_blocks": len(usage), "max_donor_reuse": max(usage.values(), default=0)}
            summaries.append(summary)
            print(f"CHAIN {ci+1}/{len(chosen)} requests={len(chain_rows)} prompt_error={summary['prompt_sum_relative_error']:+.1%} "
                  f"new_tokens={new_tokens}/{new_token_budget} rewrites={rewrites}/{target_rewrites} topups={topups} max_reuse={summary['max_donor_reuse']} "
                  f"peak_rss_mb={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024}", flush=True)
    new_index, _, _ = common.load_index(str(dest))
    cohort = common.freeze_cohort(str(dest), args.set, new_index, args.seed, str(dest / "cohort.json"))
    manifest.update(status="BUILT_UNVALIDATED", n_chains=cohort["n_chains"], n_requests=cohort["n_requests"],
                    donor_usage=dict(used), chain_summaries=summaries,
                    original_requests=sum(s["original_requests"] for s in summaries),
                    actual_prompt_sum=sum(s["prompt_sum"] for s in summaries),
                    actual_output_sum=sum(s["output_sum"] for s in summaries),
                    synthesized_new_tokens=sum(s["synthesized_new_tokens"] for s in summaries),
                    target_new_tokens=sum(s["target_new_tokens"] for s in summaries),
                    synthesized_rewrites=sum(s["synthesized_rewrites"] for s in summaries),
                    target_rewrites=sum(s["target_rewrites"] for s in summaries),
                    read_corpus_position={"piece": compiler.read_corpus.index, "offset": compiler.read_corpus.offset,
                                          "wraps": compiler.read_corpus.wraps},
                    artifacts={str(p.relative_to(dest)): file_digest(p) for p in
                               [dest / "requests.jsonl", dest / "chains.jsonl", dest / "provenance.jsonl", shard_path,
                                dest / "cohort.json", dest / "event-plans.jsonl", dest / "samples" / f"{args.set}.jsonl"]})
    json_dump(dest / "manifest.json", manifest)
    print(f"BUILT {dest} chains={cohort['n_chains']} requests={cohort['n_requests']} status=BUILT_UNVALIDATED", flush=True)
    return 0


def prepare_edits(edits, rows, bodies, grouped, provenance):
    """Validate human/Luna edits against unmodified generated introductions."""
    by_req, seen = collections.defaultdict(list), set()
    predecessors, heads = {}, {}
    for rs in grouped.values():
        heads[rs[0]["chain_id"]] = req_id(rs[0])
        for previous, current in zip(rs, rs[1:]):
            predecessors[req_id(current)] = req_id(previous)
    for number, edit in enumerate(edits):
        rid, index = edit.get("req_id"), edit.get("message_index")
        seed_query = edit.get("scope") == "chain_seed_query"
        if rid not in rows:
            raise ValueError(f"edit {number}: unknown request")
        if seed_query:
            if rid != heads[rows[rid]["chain_id"]] or edit.get("chain_id") != rows[rid]["chain_id"]:
                raise ValueError(f"edit {number}: query redesign must target the selected chain head")
        elif provenance.get(rid, {}).get("kind") != "synthetic":
            raise ValueError(f"edit {number}: only synthetic introductions can be edited")
        if not isinstance(index, int) or isinstance(index, bool) or (rid, index) in seen:
            raise ValueError(f"edit {number}: invalid/duplicate message index")
        seen.add((rid, index))
        messages = bodies[rid]["messages"]
        if not 0 <= index < len(messages):
            raise ValueError(f"edit {number}: message index out of range")
        message = messages[index]
        before = bodies[predecessors[rid]]["messages"] if rid in predecessors else []
        common = 0
        while common < min(len(messages), len(before)) and messages[common] == before[common]:
            common += 1
        if index < common and not seed_query:
            raise ValueError(f"edit {number}: cannot rewrite an already-present history message")
        old, new = edit.get("expected_content"), edit.get("replacement_content")
        if not isinstance(old, str) or not isinstance(new, str) or message.get("content") != old:
            raise ValueError(f"edit {number}: stale content or non-text replacement")
        if old == new:
            raise ValueError(f"edit {number}: no-op")
        if seed_query and (message.get("role") != "user" or reminder(message)):
            raise ValueError(f"edit {number}: seed query must be an existing human user message")
        if not seed_query and message.get("role") != "assistant" and not reminder(message):
            raise ValueError(f"edit {number}: only new assistant narrative or system reminder text is editable")
        if reminder(message) and "system-reminder" not in new:
            raise ValueError(f"edit {number}: reminder identity must be retained")
        if any(mark in new for mark in ("<|user|>", "<|assistant|>", "<|observation|>", "<|system|>")):
            raise ValueError(f"edit {number}: role-token injection is not a narrative edit")
        normalized = {**edit, "edit_id": f"edit-{number:04d}", "message_sha256": digest(message)}
        by_req[rid].append(normalized)
    return dict(by_req)


def apply_active_edits(body, active, introductions):
    """Propagate edits while that exact message survives in subsequent snapshots.

    Comparison always uses the original snapshot. A replaced tail naturally
    expires an edit; identical empty assistant content at a reused index is not
    enough to transfer it to a different tool call.
    """
    messages = body["messages"]
    active = {i: e for i, e in active.items()
              if i < len(messages) and digest(messages[i]) == e["message_sha256"]}
    for edit in introductions:
        active[edit["message_index"]] = edit
    result = copy.deepcopy(body)
    for i, edit in active.items():
        if digest(messages[i]) != edit["message_sha256"]:
            raise ValueError("edit introduction changed before application")
        result["messages"][i]["content"] = edit["replacement_content"]
    return result, active


def polish(args):
    """Create a new frozen dataset after reviewed, consistently propagated edits."""
    root, dest = Path(args.root).resolve(), Path(args.out).resolve()
    protected = [REPO / p for p in ("s1-dev", "llm-challenge-arena-v1", "build/base_exact", "refs")]
    if dest.exists() or dest == root or root in dest.parents or any(dest == p or p in dest.parents for p in protected):
        raise ValueError("polish output must be a fresh directory outside protected/source trees")
    source_manifest = json.loads((root / "manifest.json").read_text())
    if source_manifest.get("generator") != "source-aggregate-longchain-v1" or source_manifest.get("status") not in ("BUILT_UNVALIDATED", "VALIDATED"):
        raise ValueError("polish supports completed legacy v1 builds; for event v2 edit material inputs before building")
    verify_parent_artifacts(root, source_manifest)
    rows, chains, bodies, grouped = load_source(root)
    provenance = {r["req_id"]: r for r in read_jsonl(root / "provenance.jsonl")}
    if set(provenance) != set(rows):
        raise ValueError("source provenance coverage mismatch")
    edits = list(read_jsonl(args.edits))
    if not edits:
        raise ValueError("no edits supplied")
    by_req = prepare_edits(edits, rows, bodies, grouped, provenance)
    sys.path.insert(0, str(Path(args.harness_dir).resolve()))
    common = importlib.import_module("s1_common")
    renderer = common.Renderer(str(Path(args.tok_dir).resolve()))
    name = args.set or source_manifest["set"] + "-polished"
    if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("invalid set name")
    dest.mkdir(parents=True)
    (dest / "bodies").mkdir()
    (dest / "samples").mkdir()
    manifest = copy.deepcopy(source_manifest)
    manifest["parent_metadata_repairs"] = manifest.pop("known_metadata_defects", [])
    manifest.update(status="BUILDING", set=name, parent_root=str(root),
                    implementation=implementation_receipt(args.harness_dir),
                    parent_manifest_sha256=file_digest(root / "manifest.json"),
                    polish={"edits_file": str(Path(args.edits).resolve()), "edits_sha256": file_digest(args.edits),
                            "edit_count": len(edits), "rules": "explicit chain seed query redesign or new narrative/reminders; propagate surviving messages; re-render every prompt"})
    json_dump(dest / "manifest.json", manifest)
    json_dump(dest / "applied-edits.json", edits)
    shard_path = dest / "bodies" / f"{name}.jsonl.gz"
    summaries, changed, kind_counts = [], 0, collections.Counter()
    with open(dest / "requests.jsonl", "w") as rf, open(dest / "provenance.jsonl", "w") as pf, \
         open(dest / "chains.jsonl", "w") as cf, open(dest / "samples" / f"{name}.jsonl", "w") as sf, \
         open(shard_path, "wb") as raw_shard, gzip.GzipFile(fileobj=raw_shard, mode="wb", compresslevel=1, mtime=0) as gz, \
         io.TextIOWrapper(gz, encoding="utf-8") as bf:
        for ci, (cid, rs) in enumerate(grouped.items()):
            active, prev_tokens, previous = {}, [], None
            rendered_rows = []
            for old in rs:
                rid = req_id(old)
                b, active = apply_active_edits(bodies[rid], active, by_req.get(rid, []))
                ids = renderer.tokenizer.encode(renderer.render(b), add_special_tokens=False)
                common_tokens = lcp(prev_tokens, ids) if previous else 0
                row, prov = copy.deepcopy(old), copy.deepcopy(provenance[rid])
                # Every row refers to this artifact, including unchanged public
                # bodies. Original path/hash remain provenance, never stale refs.
                row["body_ref"] = f"bodies/{name}.jsonl.gz"
                if prov["kind"] == "original":
                    prov.setdefault("source_body_ref", old.get("body_ref"))
                    prov.setdefault("source_body_sha256", prov["body_sha256"])
                if prov["kind"] == "synthetic":
                    delta = transition_messages(bodies[previous]["messages"], bodies[rid]["messages"]) if previous else None
                    if delta is None:
                        raise ValueError(f"synthetic source is not an implemented continuation: {rid}")
                    prov.setdefault("donor_source_phase", old["phase"])
                    prov["phase_origin"] = "generated_message_transition"
                    row["phase"] = continuation_phase(delta[0])
                if prov["kind"] == "original" and active:
                    if any(e.get("scope") != "chain_seed_query" for e in active.values()):
                        raise ValueError(f"non-query edit reached an original snapshot {rid}")
                    prov.update(kind="adapted_original", source_body_sha256=provenance[rid]["body_sha256"],
                                source_frozen_labels={k: old.get(k) for k in ("glm_tokens", "glm_lcp_with_prev", "uncached_expected")})
                if prov["kind"] == "original":
                    if b != bodies[rid] or len(ids) != old["glm_tokens"]:
                        raise ValueError(f"polish changed original request {rid}")
                else:
                    row.update(glm_tokens=len(ids), glm_lcp_with_prev=common_tokens,
                               uncached_expected=len(ids) - common_tokens,
                               body_ref=f"bodies/{name}.jsonl.gz")
                kind_counts[prov["kind"]] += 1
                if len(ids) + row["max_output_i"] > source_manifest["max_context_tokens"]:
                    raise ValueError(f"polish exceeds context without truncation: {rid}")
                prov.update(prompt_tokens=len(ids), lcp_tokens=common_tokens,
                            added_tokens=len(ids) - common_tokens, removed_tokens=len(prev_tokens) - common_tokens,
                            previous_req_id=previous, body_sha256=digest(b),
                            polish_edit_ids=[e["edit_id"] for e in active.values()])
                changed += bool(active)
                for fh, value in ((rf, row), (pf, prov), (bf, b)):
                    write_line(fh, value)
                rendered_rows.append(row)
                prev_tokens, previous = ids, rid
            out_chain = chain_record(chains[cid], rendered_rows, observed_append_edges(rs, bodies))
            write_line(cf, out_chain)
            write_line(sf, {"chain_id": cid, "set_role": "source-length-stratified"})
            original_summary = next(s for s in manifest["chain_summaries"] if s["chain_id"] == cid)
            summary = {**original_summary, "prompt_sum": out_chain["sum_glm_tokens"], "phases": out_chain["phases"],
                       "prompt_sum_relative_error": out_chain["sum_glm_tokens"] / original_summary["source_prompt_sum"] - 1}
            summaries.append(summary)
            print(f"POLISH {ci+1}/{len(grouped)} {cid}", flush=True)
    new_index, _, _ = common.load_index(str(dest))
    cohort = common.freeze_cohort(str(dest), name, new_index, source_manifest["seed"], str(dest / "cohort.json"))
    manifest.update(status="BUILT_UNVALIDATED", chain_summaries=summaries,
                    actual_prompt_sum=sum(s["prompt_sum"] for s in summaries),
                    n_chains=cohort["n_chains"], n_requests=cohort["n_requests"],
                    artifacts={str(p.relative_to(dest)): file_digest(p) for p in
                               [dest / "requests.jsonl", dest / "chains.jsonl", dest / "provenance.jsonl", shard_path,
                                dest / "cohort.json", dest / "applied-edits.json", dest / "samples" / f"{name}.jsonl"]})
    manifest["polish"]["snapshots_with_edits"] = changed
    manifest["polish"]["request_kinds"] = dict(kind_counts)
    manifest["assumptions"] = [a for a in manifest["assumptions"]
                               if a not in ("Original requests/bodies/budgets/gaps/frozen labels are retained.",
                                            "Observed donor phases are retained; unseen reset/tool-change events are not invented to match totals.")]
    phase_rule = "Synthetic phase describes the generated message transition: a new human user message starts a turn, otherwise intra; no context rebuild is synthesized. Donor-source phases are provenance only."
    if phase_rule not in manifest["assumptions"]:
        manifest["assumptions"].append(phase_rule)
    manifest["assumptions"].append("Original output budgets and gaps are retained. Reviewed chain_seed_query edits propagate to surviving messages; affected public snapshots are adapted_original, with source body hashes/frozen labels retained separately. Unedited originals remain unchanged.")
    json_dump(dest / "manifest.json", manifest)
    print(f"POLISHED {dest} edits={len(edits)} affected_snapshots={changed}", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    bp = sub.add_parser("build", help="build a new frozen, auditable dataset; never overwrite")
    bp.add_argument("--source-root", default=str(REPO / "s1-dev/data/dev-combined-v1"))
    bp.add_argument("--harness-dir", default=str(REPO / "s1-dev/harness"))
    bp.add_argument("--tok-dir", default=str(REPO / "s1-dev/glm_tok"))
    bp.add_argument("--out", required=True)
    bp.add_argument("--set", default="s1-dev-longchain")
    bp.add_argument("--chains", type=int, default=311)
    bp.add_argument("--seed", type=int, default=20260924)
    bp.add_argument("--max-context-tokens", type=int, default=524288)
    bp.add_argument("--rewrite-topup", action="store_true",
                    help="v4: raise planned rewrites' recompute targets (earlier divergence, never past the first tool call) "
                         "so each chain's synthesized new tokens reach the source's sum_uncached_expected; v3 never scales up")
    bp.add_argument("--read-corpus", default=str(REPO / "build/base_exact"),
                    help="frozen source tree whose text files follow the public tool-result text as Read filler")
    pp = sub.add_parser("polish", help="apply reviewed edits consistently, re-render, and freeze a new dataset")
    pp.add_argument("--root", required=True)
    pp.add_argument("--edits", required=True)
    pp.add_argument("--out", required=True)
    pp.add_argument("--set")
    pp.add_argument("--harness-dir", default=str(REPO / "s1-dev/harness"))
    pp.add_argument("--tok-dir", default=str(REPO / "s1-dev/glm_tok"))
    cp = sub.add_parser("check", help="independent structural and optional full-render validation")
    cp.add_argument("--root", required=True)
    cp.add_argument("--harness-dir", default=str(REPO / "s1-dev/harness"))
    cp.add_argument("--tok-dir")
    cp.add_argument("--cohort")
    cp.add_argument("--out-json")
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            return build(args)
        if args.command == "polish":
            return polish(args)
        from longchain_check import check_dataset
        cohort = args.cohort or str(Path(args.root) / "cohort.json")
        report = check_dataset(args.root, args.harness_dir, args.tok_dir,
                               cohort if Path(cohort).exists() else None)
        if args.out_json:
            json_dump(Path(args.out_json), report)
        print(json.dumps({k: v for k, v in report.items() if k in ("status", "errors", "warnings", "counts")}, ensure_ascii=False, indent=2))
        return 2 if report["status"] == "INVALID" else 0
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
