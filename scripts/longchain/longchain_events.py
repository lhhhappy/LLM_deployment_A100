"""Offline event plans and immutable message transitions for longchain.build.

Phoenix supplies structural ranks/sequences, never text or GLM token labels.
All unknown time decompositions and transplanted relationships are estimates.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from copy import deepcopy
import math
from pathlib import Path
import random

import longchain as lc


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


def quantile(values, fraction):
    return values[min(len(values)-1, max(0, int(fraction * (len(values)-1))))]


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
    summary = {"role": "user", "content": "历史上下文摘录（中间记录已归档）：\n" +
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


class EventCompiler:
    def __init__(self, args, grouped, bodies, donors):
        self.seed = args.seed
        self.rng = random.Random(args.seed)
        self.grouped, self.bodies, self.donors = grouped, bodies, donors
        self.queries, self.answers = material_bank(grouped, bodies)
        self.global_usage = Counter()
        self.profile_path = Path(args.behavior_profile)
        events = list(lc.read_jsonl(self.profile_path))
        # Unchanged model/system/tools only: changed formats are not transplanted.
        self.events = [e for e in events if e.get("same_system_hash") is True
                       and e.get("same_tools_hash") is True and not e.get("model_changed")
                       and isinstance(e.get("gap_s"), (int, float)) and e["gap_s"] >= 0
                       and isinstance(e.get("prompt_delta_source"), (int,float))]
        if not self.events:
            raise ValueError("no usable frozen structural observations")
        by_run = defaultdict(list)
        for e in self.events:
            by_run[(e["session_id"], e["run_id"])].append(e)
        # Keep only directly linked spans in each segment. Missing spans do not
        # become invented continuous evidence.
        self.segments = []
        for run in by_run.values():
            segment = []
            for e in run:
                if segment and segment[-1]["current_span"] != e["previous_span"]:
                    self.segments.append(segment)
                    segment = []
                segment.append(e)
            if segment:
                self.segments.append(segment)
        self.growths = sorted(e["prompt_delta_source"] for e in self.events if e["prompt_delta_source"] > 0)
        self.outputs = sorted(e["next_output_tokens_source"] for e in self.events
                              if isinstance(e.get("next_output_tokens_source"), (int,float)))
        self.compressions = [e for e in self.events if e["event_observation"] == "reported_compression"]
        if not self.compressions:
            raise ValueError("profile lacks observed compression events")
        self.last_history_used = Counter()

    def plan(self, target, original, pool):
        count = target["n_requests"] - len(original)
        if not count:
            return []
        rng = self.rng
        skeleton = []
        while len(skeleton) < count:
            remaining = count - len(skeleton)
            close = sorted(self.segments, key=lambda s: abs(math.log(len(s)/remaining)))[:128]
            segment = rng.choices(close, weights=[s[0]["weight"] for s in close], k=1)[0]
            crop = segment[:remaining]
            skeleton.extend({"reference": e, "template_boundary": i == 0} for i,e in enumerate(crop))
        seen_phases = Counter(r["phase"] for r in original)
        desired_reset = max(0, target.get("phases", {}).get("context_reset", 0) - seen_phases["context_reset"])
        desired_turn = max(0, target.get("phases", {}).get("turn_start", 0) - seen_phases["turn_start"])
        # Aggregate counts constrain frequency; positions are explicitly inferred
        # using contiguous observed templates, never claimed as recovered source.
        reset_n = min(desired_reset, count)
        candidates = list(range(count))
        rng.shuffle(candidates)
        candidates.sort(key=lambda i: (skeleton[i]["reference"]["event_observation"] != "reported_compression",
                                        i == count-1, i == 0))
        reset_at = set(candidates[:reset_n])
        turn_candidates = [i for i in range(count) if i not in reset_at]
        rng.shuffle(turn_candidates)
        turn_candidates.sort(key=lambda i: not skeleton[i]["template_boundary"])
        turn_at = set(turn_candidates[:min(desired_turn, len(turn_candidates))])
        output_weights = sorted(d.weight for d in pool)
        for i, item in enumerate(skeleton):
            kind = "context_reset" if i in reset_at else "turn_start" if i in turn_at else "intra"
            reference = item["reference"]
            if kind == "context_reset" and reference["event_observation"] != "reported_compression":
                reference = rng.choices(self.compressions, weights=[e["weight"] for e in self.compressions], k=1)[0]
                item["replaced_reference_for_event"] = True
            item.update(kind=kind, reference=reference, position_origin="source_phase_counts_with_observed_template_preferences")
            out_rank = bisect_right(self.outputs, reference.get("next_output_tokens_source") or 0)/len(self.outputs)
            item["output_weight"] = max(2, quantile(output_weights, out_rank))
        return skeleton

    def pool(self, target, body):
        allowed, schemas = lc.available_tools(body), lc.tool_schemas(body)
        result = [d for d in self.donors if d.pack == target["pack"] and d.names <= allowed
                  and all(d.schemas.get(name) == schemas.get(name) for name in d.names)]
        return result

    def event(self, item, target, current, prefix, step, usage, room):
        reference = item["reference"]
        # Capped end-to-start gap is an explicit proxy. Unknown tool-vs-user
        # decomposition is NOT fabricated as tool_union_ms/net_think_ms.
        gap = round(min(reference["gap_s"], 300.0)*1000)
        receipt = {"event_kind": item["kind"], "event_position_origin": item["position_origin"],
                   "behavior_reference": {k: reference.get(k) for k in
                      ("session_id", "run_id", "previous_span", "current_span", "event_observation", "gap_s",
                       "before_prompt_tokens_source", "after_prompt_tokens_source", "next_output_tokens_source")},
                   "template_boundary": item["template_boundary"],
                   "gap_origin": "estimated_capped_phoenix_end_to_start_proxy", "raw_observed_gap_ms": reference["gap_s"]*1000,
                   "gap_decomposition_known": False, "replay_gap_ms": gap,
                   "phase_origin": "explicit_generated_event", "output_budget_origin": "source_aggregate_scaled_rank_transferred_output"}
        kind = item["kind"]
        if kind == "context_reset":
            before = reference.get("before_prompt_tokens_source") or 1
            fraction = (reference.get("after_prompt_tokens_source") or before*.25)/before
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
                            text_content(messages[-1]).startswith("历史上下文摘录（中间记录已归档）：\n"))
            receipt["replaced_last_reminder"] = replaced
            return {**current, "messages": (messages[:-1] if replaced else messages) + block}, receipt, None
        pool = self.pool(target, current)
        # Positive observed growth supplies a rank, mapped to public material
        # increments; source-provider tokens never become GLM labels.
        delta = max(1, reference["prompt_delta_source"])
        rank = bisect_right(self.growths, delta)/len(self.growths)
        sizes = sorted(d.increment for d in pool)
        desired = quantile(sizes, rank)
        def score(d):
            return abs(math.log((d.increment+128)/(desired+128))) + .6*usage[d.fingerprint] + .12*self.global_usage[d.fingerprint]
        fitting = [d for d in pool if d.increment < room]
        if not fitting:
            raise NoMaterialFits("no complete compatible material fits context budget; explicit rebuild required")
        ranked = sorted(fitting, key=lambda d: (score(d), d.fingerprint))[:16]
        donor = self.rng.choices(ranked, weights=[math.exp(-(score(d)-score(ranked[0]))/.35) for d in ranked], k=1)[0]
        # A tool continuation must not smuggle a second human-user event.
        block = [m for m in donor.messages if m.get("role") != "user" or lc.reminder(m)]
        block = lc.rename_new_calls(block, f"{prefix}_{step:04d}")
        messages = current["messages"]
        replaced = bool(messages and lc.reminder(messages[-1]) and not
                        text_content(messages[-1]).startswith("历史上下文摘录（中间记录已归档）：\n"))
        receipt.update(donor_req_id=donor.source_req_id, donor_chain_id=donor.chain_id,
                       donor_mode=donor.kind, donor_fingerprint=donor.fingerprint,
                       donor_source_phase=donor.phase, donor_tool_pairing=lc.tool_pairing(block),
                       donor_same_family=donor.family == target.get("sys_tools_hash"),
                       donor_reuse_in_chain=usage[donor.fingerprint],
                       selection_target_growth=desired, replaced_last_reminder=replaced)
        return {**current, "messages": (messages[:-1] if replaced else messages) + block}, receipt, donor

    def commit(self, receipt, usage):
        """Only accepted events consume material; rejected trials are not usage."""
        if receipt["event_kind"] == "turn_start":
            for field in ("query_material", "answer_material"):
                self.global_usage[receipt[field]["fingerprint"]] += 1
        elif receipt["event_kind"] == "intra":
            fingerprint = receipt["donor_fingerprint"]
            self.global_usage[fingerprint] += 1
            usage[fingerprint] += 1
