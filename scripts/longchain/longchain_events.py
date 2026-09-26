"""Offline event plans and immutable message transitions for longchain.build.

Every synthesized step copies the load of one public request of the same kind (new tokens, replay
gap, output budget) from the organizer's frozen metadata; text comes from public material and filler.
Positions of events within a chain are not observed and are drawn at random.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random

import longchain as lc

SUMMARY_PREFIX = "历史上下文摘录（中间记录已归档）：\n"
# Text files of the read corpus (source code and docs), after the public tool-result text.
READ_CORPUS_SUFFIXES = (".py", ".md", ".rst", ".txt", ".cu", ".cuh", ".cc", ".cpp", ".c", ".h", ".hpp")
# A continuation is lengthened when its donor falls short of the sampled step size by more than this.
READ_MIN_TOKENS, READ_TOLERANCE = 128, 0.1
# Rendered tokens of a Read call and its JSON result around the text itself (measured ~40-60).
READ_WRAPPER_TOKENS = 64


class NoMaterialFits(ValueError):
    """A whole continuation requires an explicit context rebuild first."""


def text_content(message):
    value = message.get("content")
    return value if isinstance(value, str) else ""


def material_bank(grouped, bodies):
    """Query seeds and turn endings, de-duplicated across all source snapshots."""
    queries, answers = {}, {}
    for cid, rows in grouped.items():
        for row in rows:
            body = bodies[lc.req_id(row)]
            for message in body["messages"]:
                text = text_content(message)
                if not text.strip():
                    continue
                record = {"message": deepcopy(message), "source_req_id": lc.req_id(row),
                          "source_session_id": row["session_id"], "source_chain_id": cid,
                          "pack": row["pack"], "fingerprint": lc.digest(message)}
                if message.get("role") == "user" and not lc.reminder(message):
                    # Exclude exporter/controller summaries from the user actor.
                    lower = text.lstrip().lower()
                    if not any(x in lower[:500] for x in
                               ("this session is being continued", "context summary", "context has been compacted",
                                "conversation summary", "<system>", "<|user|>", "<|system|>")):
                        queries.setdefault((row["pack"], record["fingerprint"]), record)
                        break  # one earliest task seed per snapshot, not every control injection
            for message in body["messages"]:
                if message.get("role") == "assistant" and not message.get("tool_calls") and text_content(message).strip():
                    record = {"message": deepcopy(message), "source_req_id": lc.req_id(row),
                              "source_session_id": row["session_id"], "source_chain_id": cid,
                              "pack": row["pack"], "fingerprint": lc.digest(message)}
                    answers.setdefault((row["pack"], record["fingerprint"]), record)
    return list(queries.values()), list(answers.values())


def atomic_ranges(messages):
    """Whole messages or complete call/result groups; incomplete groups stay marked."""
    result = []
    i = 0
    while i < len(messages):
        j = i + 1
        if messages[i].get("tool_calls"):
            while j < len(messages) and messages[j].get("role") == "tool":
                j += 1
        valid = lc.tool_pairing(messages[i:j]) != "incomplete"
        result.append((i, j, valid))
        i = j
    return result


def rebuild_history(body, keep_fraction, excerpt_chars=240, excerpt_count=8):
    """Explicit extractive rebuild, preserving a task prefix and complete tail.

    Summary snippets derive exclusively from the receiving history. No source
    snapshot is imported wholesale, and old history cannot silently grow back.
    """
    messages = body["messages"]
    ranges = atomic_ranges(messages)
    prefix_end = next((i for i, _, _ in ranges if messages[i].get("role") == "assistant"), 0)
    # Keep the original opening task; the new summary replaces middle history.
    if prefix_end == 0 and messages and messages[0].get("role") == "user":
        prefix_end = 1
    available = [(i, j, ok) for i, j, ok in ranges if i >= prefix_end]
    if not available:
        raise ValueError("history has no replaceable complete message range")
    total_chars = sum(len(lc.canonical(m)) for m in messages)
    tail_limit = total_chars * min(.45, max(0, keep_fraction))
    tail_start, tail_chars = len(messages), 0
    for i, j, valid in reversed(available):
        size = sum(len(lc.canonical(m)) for m in messages[i:j])
        if not valid or tail_chars + size > tail_limit or i <= prefix_end:
            break
        tail_start, tail_chars = i, tail_chars + size
    removed = messages[prefix_end:tail_start]
    if not removed:
        raise ValueError("no-op context rebuild")
    excerpts = []
    # Short, ordered extracts, not a fictitious semantically complete summary.
    for index, message in enumerate(removed):
        value = text_content(message)
        if value and message.get("role") != "tool":
            excerpts.append((index, value[:excerpt_chars]))
        if len(excerpts) == excerpt_count:
            break
    if not excerpts:
        excerpts = [(0, lc.canonical(removed[0])[:excerpt_chars])]
    summary = {"role": "user", "content": SUMMARY_PREFIX +
               "\n".join(text for _, text in excerpts) + "\n继续当前任务。"}
    new_messages = deepcopy(messages[:prefix_end]) + [summary] + deepcopy(messages[tail_start:])
    if new_messages == messages or lc.transition_messages(messages, new_messages) is not None:
        raise ValueError("rebuild did not replace history")
    receipt = {"prefix_end": prefix_end, "tail_start": tail_start,
               "removed_messages": len(removed), "removed_sha256": lc.digest(removed),
               "summary_message": summary, "summary_excerpts": [{"removed_index": i, "text": t} for i,t in excerpts],
               "retained_tail_sha256": lc.digest(messages[tail_start:]),
               "method": "receiving_history_extractive_summary_v1"}
    return {**body, "messages": new_messages}, receipt


def close_turn(messages):
    """The agent's rewrite of a finished turn when the next human message arrives.

    Measured on every adjacent pair of the public bodies with unchanged system/tools: at a turn
    start each tool-call message of the finished turn loses its narration text (content -> ""),
    11 of 11 times the turn had narration; within a turn the narration is kept, 231 of 231.
    Earlier turns were closed already, so only messages after the last human message change.
    Returns the closed messages and the indices whose content was removed.
    """
    last_human = max((i for i, m in enumerate(messages)
                      if m.get("role") == "user" and not lc.reminder(m)), default=-1)
    closed, stripped = list(messages), []
    for i in range(last_human + 1, len(messages)):
        message = messages[i]
        if message.get("role") == "assistant" and message.get("tool_calls") and message.get("content"):
            if not isinstance(message["content"], str):
                raise ValueError("tool-call narration must be text to close a turn")
            closed[i] = {**message, "content": ""}
            stripped.append(i)
    return closed, stripped


def _long_strings(value, minimum=400):
    if isinstance(value, str):
        if len(value) >= minimum:
            yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _long_strings(item, minimum)
    elif isinstance(value, list):
        for item in value:
            yield from _long_strings(item, minimum)


def read_arguments(schema, path):
    """Arguments for a Read call under the receiving session's own Read schema, or None."""
    properties = set((schema or {}).get("properties") or {})
    if {"source", "target"} <= properties:
        arguments = {"source": "path", "target": path}
    elif "path" in properties:
        arguments = {"path": path}
    else:
        return None
    for name in ("offset", "limit"):
        if name in properties:
            arguments[name] = None
    if not set((schema or {}).get("required") or ()) <= set(arguments):
        return None
    return arguments


class ReadCorpus:
    """Text for Read results that bring a continuation up to its sampled length.

    Only the token count of this text matters to the workload. It is read front to back so that
    no text repeats until the corpus is exhausted: first the long strings of the public tool
    results (web pages, papers, command output), then the text files of a frozen source tree.
    """

    def __init__(self, bodies, file_root, tokenizer, seed):
        self.tokenizer = tokenizer
        pieces = {}
        for rid in sorted(bodies):
            for message in bodies[rid]["messages"]:
                if message.get("role") != "tool" or not isinstance(message.get("content"), str):
                    continue
                try:
                    value = json.loads(message["content"])
                except ValueError:
                    continue
                for text in _long_strings(value):
                    pieces.setdefault(hashlib.sha256(text.encode()).hexdigest(),
                                      {"kind": "public_tool_result", "source": rid, "text": text})
        public = [pieces[k] for k in sorted(pieces)]
        random.Random(seed).shuffle(public)
        root = Path(file_root).resolve()
        paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in READ_CORPUS_SUFFIXES)
        files = []
        for path in paths:
            text = path.read_text(encoding="utf-8", errors="replace")
            if len(text) >= 400:
                files.append({"kind": "file", "source": str(path.relative_to(root)), "text": text})
        random.Random(seed + 1).shuffle(files)
        self.pieces = public + files
        for piece in self.pieces:
            piece["sha256"] = hashlib.sha256(piece["text"].encode()).hexdigest()
        if not self.pieces:
            raise ValueError("empty Read corpus")
        self.receipt = {"root": str(root), "public_pieces": len(public), "files": len(files),
                        "chars": sum(len(p["text"]) for p in self.pieces),
                        "sha256": lc.digest([[p["kind"], p["source"], p["sha256"]] for p in self.pieces])}
        self.index, self.offset, self.wraps = 0, 0, 0

    def _tokens_per_char(self, piece):
        """Rendered tokens per character; the text sits JSON-escaped inside the tool result."""
        if "ratio" not in piece:
            escaped = json.dumps(piece["text"], ensure_ascii=False)[1:-1]
            piece["ratio"] = len(self.tokenizer.encode(escaped, add_special_tokens=False)) / len(piece["text"])
        return piece["ratio"]

    def take(self, tokens):
        """About `tokens` rendered tokens of consecutive corpus text, with the ranges used."""
        parts, used, need = [], [], tokens
        while need > 0:
            piece = self.pieces[self.index]
            ratio = self._tokens_per_char(piece)
            start = self.offset
            end = len(piece["text"])
            if (end - start) * ratio > need:
                end = start + max(1, int(need / ratio))
                cut = piece["text"].rfind("\n", start, end)
                end = cut + 1 if cut > start + (end - start) // 2 else end
            parts.append(piece["text"][start:end])
            used.append({"kind": piece["kind"], "source": piece["source"], "sha256": piece["sha256"],
                         "start": start, "end": end})
            need -= round((end - start) * ratio)
            self.offset = end
            if end >= len(piece["text"]):
                self.index, self.offset = self.index + 1, 0
                if self.index == len(self.pieces):
                    self.index, self.wraps = 0, self.wraps + 1
        return "\n\n".join(parts), used

    def lengthen(self, block, tokens, schema, call_id):
        """Add one parallel Read call, and its result of about `tokens` tokens, to the first tool-call
        message of a continuation. Returns the new block and a receipt, or (block, reason) unchanged."""
        index = next((i for i, m in enumerate(block) if m.get("role") == "assistant" and m.get("tool_calls")), None)
        if index is None:
            return block, {"read_skipped": "continuation has no tool-call message"}
        if read_arguments(schema, "probe") is None:
            return block, {"read_skipped": "receiving session has no usable Read schema"}
        text, used = self.take(max(1, tokens - READ_WRAPPER_TOKENS))
        path = used[0]["source"] if used[0]["kind"] == "file" else f"notes/{used[0]['sha256'][:16]}.md"
        arguments = read_arguments(schema, path)
        end = index + 1
        while end < len(block) and block[end].get("role") == "tool":
            end += 1
        call = {"id": call_id, "type": "function", "function": {"name": "Read", "arguments": arguments}}
        payload = {"content": text, "ok": True, "path": path, "totalLines": text.count("\n") + 1,
                   "truncated": False, "type": "text"}
        result = {"role": "tool", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
                  "tool_call_id": call_id}
        caller = {**block[index], "tool_calls": list(block[index]["tool_calls"]) + [call]}
        lengthened = block[:index] + [caller] + block[index + 1:end] + [result] + block[end:]
        return lengthened, {"read_target_tokens": tokens, "read_text_chars": len(text), "read_pieces": used}


class EventCompiler:
    def __init__(self, args, grouped, bodies, donors, tokenizer):
        self.seed = args.seed
        self.rng = random.Random(args.seed)
        self.grouped, self.bodies, self.donors = grouped, bodies, donors
        # Templates: public requests that are not chain heads (a head carries its whole visible history),
        # except context resets, which are too few otherwise. Each synthesized step copies one template
        # of its kind: frozen new tokens (uncached_expected), replay gap and output budget.
        self.templates = defaultdict(list)
        for rows in grouped.values():
            for position, row in enumerate(rows):
                if (row["phase"] in ("intra", "turn_start") and position > 0 or row["phase"] == "context_reset") \
                        and row.get("gap_valid") and row.get("replay_gap_ms") is not None:
                    self.templates[row["phase"]].append(
                        {k: row[k] for k in ("pack", "logical_call_id", "phase", "uncached_expected",
                                             "glm_tokens", "replay_gap_ms", "max_output_i")})
        if not self.templates["intra"]:
            raise ValueError("source has no within-turn continuation to copy load from")
        self.read_corpus = ReadCorpus(bodies, args.read_corpus, tokenizer, args.seed)
        # Rendered GLM tokens per character of canonical message JSON, measured on the public
        # append-only transitions (message growth only; system and tools are excluded).
        tokens = chars = 0
        for rows in grouped.values():
            for before, after in zip(rows, rows[1:]):
                a, b = bodies[lc.req_id(before)], bodies[lc.req_id(after)]
                if after.get("edge_type") == "append-only" and a.get("system") == b.get("system") and a.get("tools") == b.get("tools"):
                    tokens += after["glm_tokens"] - before["glm_tokens"]
                    chars += sum(len(lc.canonical(m)) for m in b["messages"]) - sum(len(lc.canonical(m)) for m in a["messages"])
        self.tokens_per_char = tokens / chars if chars > 0 else 0.3
        self.queries, self.answers = material_bank(grouped, bodies)
        self.global_usage = Counter()

    def template(self, kind):
        """A random template of `kind`; kinds without public examples borrow a continuation's load."""
        return self.rng.choice(self.templates.get(kind) or self.templates["intra"])

    def plan(self, target, original, pool):
        """Kinds, load templates and history rewrites of the missing requests.

        Phase counts and the number of history rewrites follow the source chain's metadata (a rewrite
        is an edge that is not append-only: total_edges - append_only_edges, where total_edges also
        counts the head's incoming edge for chains that start mid-session). Rebuilds always rewrite,
        then turn starts, then random continuations. Positions are random (resets avoid the first
        and last step when there is room).
        """
        count = target["n_requests"] - len(original)
        if not count:
            return []
        rng = self.rng
        seen_phases = Counter(r["phase"] for r in original)
        phases = target.get("phases", {})
        reset_n = min(count, max(0, phases.get("context_reset", 0) - seen_phases["context_reset"]))
        candidates = list(range(count))
        rng.shuffle(candidates)
        candidates.sort(key=lambda i: i in (0, count - 1))
        reset_at = set(candidates[:reset_n])
        turn_candidates = [i for i in range(count) if i not in reset_at]
        rng.shuffle(turn_candidates)
        turn_at = set(turn_candidates[:max(0, phases.get("turn_start", 0) - seen_phases["turn_start"])])
        counted = original if target.get("total_edges") == target["n_requests"] else original[1:]
        rewrites = (target.get("total_edges", 0) - target.get("append_only_edges", 0)
                    - sum(r.get("edge_type") != "append-only" for r in counted)) - len(reset_at)
        turn_order = sorted(turn_at)
        rng.shuffle(turn_order)
        rewriting = set(turn_order[:max(0, rewrites)])
        intra_order = [i for i in range(count) if i not in reset_at and i not in turn_at]
        rng.shuffle(intra_order)
        rewriting |= set(intra_order[:max(0, rewrites - len(turn_order))])
        plan = []
        for i in range(count):
            kind = "context_reset" if i in reset_at else "turn_start" if i in turn_at else "intra"
            template = self.template(kind)
            plan.append({"kind": kind, "template": template, "rewrite": kind == "context_reset" or i in rewriting,
                         "position_origin": "source_phase_counts_random_positions",
                         "output_weight": max(2, template["max_output_i"])})
        return plan

    def step_shape(self, item):
        """New tokens of a continuation or rewriting turn start before the chain's budget scaling: its
        template's. Other steps take what their event produces."""
        sized = item["kind"] == "intra" or item["kind"] == "turn_start" and item.get("rewrite")
        return item["template"]["uncached_expected"] if sized else 0

    def step_ceiling(self, item):
        """Largest public size of the event's kind; scaled targets do not exceed it."""
        return max(t["uncached_expected"] for t in self.templates.get(item["kind"]) or self.templates["intra"])

    def approximate_tokens(self, messages):
        """Rendered tokens estimated from canonical JSON length; only used to choose sizes."""
        return round(sum(len(lc.canonical(m)) for m in messages) * self.tokens_per_char)

    def diverge_to_target(self, history, stripped, block, target_tokens, schema, prefix):
        """Make a rewriting step recompute about `target_tokens` in total with its new block.

        In the public bodies a rewrite continues a different trajectory than the previous request
        (5 of the 20 turn starts diverge right after the session's first human message), so
        everything after the divergence is recomputed while the context barely grows. Here the
        divergence is one extra Read call in an earlier tool-call message: the latest one whose
        suffix, with the Read, reaches the target. Call IDs are not rendered, so renaming them would
        not diverge. Token counts here are approximations; the build measures the real ones.
        """
        first_human = next((i for i, m in enumerate(history) if m.get("role") == "user" and not lc.reminder(m)), -1)
        starts = [i for i in range(first_human + 1, len(history)) if history[i].get("tool_calls")]
        tail = self.approximate_tokens(block)
        suffix = [0] * (len(history) + 1)  # suffix[i]: estimated tokens of history[i:]
        for i in reversed(range(len(history))):
            suffix[i] = suffix[i + 1] + self.approximate_tokens([history[i]])
        natural = tail + (suffix[stripped[0]] if stripped else 0)
        if natural >= target_tokens or not starts:
            return history, {"natural_tokens": natural, "diverged_at": None,
                             "reason": "turn close reaches the target" if starts else "history has no tool calls"}
        start = next((i for i in reversed(starts) if suffix[i] + tail + READ_MIN_TOKENS >= target_tokens), starts[0])
        read_tokens = max(READ_MIN_TOKENS, target_tokens - tail - suffix[start])
        head, lengthening = self.read_corpus.lengthen(history[start:], read_tokens, schema, f"{prefix}_read")
        if "read_skipped" in lengthening:
            return history, {"natural_tokens": natural, "diverged_at": None, "reason": lengthening["read_skipped"]}
        return history[:start] + head, {"natural_tokens": natural, "diverged_at": start, "read_lengthening": lengthening}

    def pool(self, target, body):
        allowed, schemas = lc.available_tools(body), lc.tool_schemas(body)
        result = [d for d in self.donors if d.pack == target["pack"] and d.names <= allowed
                  and all(d.schemas.get(name) == schemas.get(name) for name in d.names)]
        return result

    def event(self, item, target, current, prefix, step, usage, room):
        template = item["template"]
        # The template's replay gap already follows the organizer's rule (capped tool union plus
        # capped thinking); its decomposition is not copied into tool_union_ms/net_think_ms.
        receipt = {"event_kind": item["kind"], "event_position_origin": item["position_origin"],
                   "load_template": template, "gap_origin": "public_template_replay_gap",
                   "gap_decomposition_known": False, "replay_gap_ms": template["replay_gap_ms"],
                   "phase_origin": "explicit_generated_event",
                   "output_budget_origin": "source_chain_output_total_apportioned_by_template_outputs"}
        kind = item["kind"]
        if kind == "context_reset":
            # Keep a tail so the rebuilt prompt is near the template's post-reset prompt size.
            fraction = min(.45, template["glm_tokens"] / max(1, self.approximate_tokens(current["messages"])))
            limits = item.get("rebuild_limits", {})
            result, rebuild = rebuild_history(current, limits.get("keep_fraction", fraction),
                limits.get("excerpt_chars", 240), limits.get("excerpt_count", 8))
            receipt["rebuild"] = rebuild
            receipt["rebuild_render_adjustment"] = limits or None
            receipt["donor_mode"] = "receiving_history_rebuild"
            return result, receipt, None
        if kind == "turn_start":
            queries = [q for q in self.queries if q["pack"] == target["pack"]
                       and q["source_session_id"] != target["session_id"]]
            if not queries:
                raise ValueError("no cross-session query material")
            queries.sort(key=lambda q: (self.global_usage[q["fingerprint"]], q["fingerprint"]))
            query = self.rng.choice(queries[:min(24, len(queries))])
            endings = [a for a in self.answers if a["pack"] == target["pack"]]
            endings.sort(key=lambda a: (a["source_chain_id"] != target["chain_id"], self.global_usage[a["fingerprint"]]))
            if not endings:
                raise ValueError("no assistant material to close user turn")
            answer = self.rng.choice(endings[:min(12, len(endings))])
            question = {"role": "user", "content": "接下来处理另一个子任务。以下问题中未提供的材料请列为待补充，不要假定此前已经完成：\n" + query["message"]["content"]}
            block = [deepcopy(answer["message"]), question]
            receipt.update(donor_mode="cross_session_query", donor_req_id=query["source_req_id"],
                           donor_chain_id=query["source_chain_id"], donor_fingerprint=query["fingerprint"],
                           query_material={k:v for k,v in query.items() if k != "message"},
                           answer_material={k:v for k,v in answer.items() if k != "message"},
                           query_adaptation="new_subtask_with_explicit_missing_material_boundary")
            messages = current["messages"]
            replaced = bool(messages and lc.reminder(messages[-1]) and not
                            text_content(messages[-1]).startswith(SUMMARY_PREFIX))
            # The finished turn is closed as the agent does it, so everything from its first narrated
            # tool call onward is new to the prefix cache; the runtime reminder follows the new query.
            history, stripped = close_turn(messages[:-1] if replaced else messages)
            if replaced:
                block.append(deepcopy(messages[-1]))
            receipt.update(replaced_last_reminder=replaced, reminder_follows_query=replaced,
                           turn_close={"rule": "finished_turn_tool_call_narration_removed_v1",
                                       "stripped_indices": stripped,
                                       "stripped_sha256": lc.digest([messages[i] for i in stripped])})
            if item.get("rewrite") and item.get("step_target"):
                history, receipt["divergence"] = self.diverge_to_target(
                    history, stripped, block, item["step_target"], lc.tool_schemas(current).get("Read")
                    if "Read" in lc.available_tools(current) else None, f"{prefix}_{step:04d}_t")
            receipt.update(step_scale=item.get("step_scale"), step_target_tokens=item.get("step_target"))
            return {**current, "messages": history + block}, receipt, None
        pool = self.pool(target, current)
        # Size: the template's new tokens, scaled
        # by the build towards the chain's source new-token total. Donors provide structure; a Read
        # result makes up a donor's shortfall. Source-provider tokens never become GLM labels.
        target_tokens = item.get("step_target") or self.step_shape(item)
        desired = min(item.get("growth_target") or target_tokens, max(1, room - READ_WRAPPER_TOKENS))
        def score(d):
            return abs(math.log((d.increment+128)/(desired+128))) + .6*usage[d.fingerprint] + .12*self.global_usage[d.fingerprint]
        fitting = [d for d in pool if d.increment < room]
        if not fitting:
            raise NoMaterialFits("no complete compatible material fits context budget; explicit rebuild required")
        # A Read makes up a shortfall but nothing shrinks a block, so blocks above the size are a last resort.
        fitting = [d for d in fitting if d.increment <= desired * (1 + READ_TOLERANCE)] or \
                  [min(fitting, key=lambda d: (d.increment, d.fingerprint))]
        ranked = sorted(fitting, key=lambda d: (score(d), d.fingerprint))[:16]
        donor = self.rng.choices(ranked, weights=[math.exp(-(score(d)-score(ranked[0]))/.35) for d in ranked], k=1)[0]
        # A tool continuation must not smuggle a second human-user event.
        block = [m for m in donor.messages if m.get("role") != "user" or lc.reminder(m)]
        lengthening = None
        if desired - donor.increment > max(READ_MIN_TOKENS, READ_TOLERANCE * desired):
            schema = lc.tool_schemas(current).get("Read") if "Read" in lc.available_tools(current) else None
            block, lengthening = self.read_corpus.lengthen(block, desired - donor.increment, schema, "read_lengthening")
        block = lc.rename_new_calls(block, f"{prefix}_{step:04d}")
        messages = current["messages"]
        replaced = bool(messages and lc.reminder(messages[-1]) and not
                        text_content(messages[-1]).startswith(SUMMARY_PREFIX))
        history = messages[:-1] if replaced else messages
        if item.get("rewrite"):
            schema = lc.tool_schemas(current).get("Read") if "Read" in lc.available_tools(current) else None
            history, receipt["divergence"] = self.diverge_to_target(
                history, [], block, max(target_tokens, desired), schema, f"{prefix}_{step:04d}_d")
        receipt.update(donor_req_id=donor.source_req_id, donor_chain_id=donor.chain_id,
                       donor_mode=donor.kind, donor_fingerprint=donor.fingerprint,
                       donor_source_phase=donor.phase, donor_tool_pairing=lc.tool_pairing(block),
                       donor_same_family=donor.family == target.get("sys_tools_hash"),
                       donor_reuse_in_chain=usage[donor.fingerprint],
                       step_target_tokens=target_tokens, growth_target_tokens=desired, step_scale=item.get("step_scale"),
                       read_lengthening=lengthening, replaced_last_reminder=replaced)
        return {**current, "messages": history + block}, receipt, donor

    def commit(self, receipt, usage):
        """Only accepted events consume material; rejected trials are not usage."""
        if receipt["event_kind"] == "turn_start":
            for field in ("query_material", "answer_material"):
                self.global_usage[receipt[field]["fingerprint"]] += 1
        elif receipt["event_kind"] == "intra":
            fingerprint = receipt["donor_fingerprint"]
            self.global_usage[fingerprint] += 1
            usage[fingerprint] += 1
