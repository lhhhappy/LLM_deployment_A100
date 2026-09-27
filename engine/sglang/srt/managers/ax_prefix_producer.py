"""128: bounded producer/dependent ledger; evaluated only on request-plane rank 0.

All inputs are native consumer matches and live requests. No harness labels,
GPU operations, cache allocations or persistent Req/tree references. The plan
contains only serializable CPU data; the native PrefillAdder remains authoritative.
"""

import json
import math
import os
from array import array
from collections import Counter
from dataclasses import dataclass

from sglang.srt.managers import ax_deadline
from sglang.srt.mem_cache import ax_prefix_readiness as readiness


@dataclass(frozen=True)
class Config:
    max_candidates: int = 64
    min_shared: int = 4096
    max_hold_s: float = 8.0
    stall_s: float = 2.0
    min_progress: int = 1024
    max_ready: int = 4
    trace_s: float = 120.0
    trace_rounds: int = 2048


def config():
    if os.environ.get("SGLANG_AX_PREFIX_PRODUCER", "0") != "1":
        return None  # off does not parse tuning knobs
    defaults = Config()
    values = {}
    for name in defaults.__dataclass_fields__:
        default = getattr(defaults, name)
        values[name] = type(default)(os.environ.get(
            "SGLANG_AX_PREFIX_" + name.upper(), str(default)))
    cfg = Config(**values)
    if not (1 <= cfg.max_candidates <= 64 and cfg.min_shared > 0
            and 0 < cfg.stall_s <= cfg.max_hold_s and cfg.min_progress > 0
            and 1 <= cfg.max_ready <= 8 and 0 <= cfg.trace_s <= 3600
            and 0 <= cfg.trace_rounds <= 10000
            and all(math.isfinite(v) for v in values.values())):
        raise ValueError(f"[ax-prefix] invalid configuration: {cfg}")
    return cfg


@dataclass
class Record:
    identity: tuple
    generation: int
    due: float
    cold: bool
    received: float
    raw: bytes = b""
    hashes: tuple = ()
    participant: bool = False
    fallback: bool = False
    first_admitted: float = None
    first_finished: float = None


@dataclass
class Dependency:
    producer: str
    producer_generation: int
    target: int
    created: float
    progress: int
    progress_at: float
    state: str = "WAIT_PREFIX"
    reason: str = "producer_pending"


class Tracker:
    VERSION = 1
    BLOCK_BYTES = 256 * 8

    def __init__(self, cfg):
        self.cfg = cfg
        self.epoch = 0
        self.reset()

    def reset(self):
        self.epoch += 1
        self.sequence = 0
        self.generation = 0
        self.records = {}
        self.dependencies = {}
        self.pairs = {}
        self.pair_signature = ()
        self.stats = Counter()
        self.trace_start = None
        self.trace_closed = False
        self.log_at = 0.0
        self.events = []

    @staticmethod
    def identity(req):
        return (id(req), id(req.origin_input_ids), len(req.origin_input_ids),
                getattr(req, "retraction_count", 0), readiness.domain(req))

    def _record(self, req, now, waited, deadline):
        rec = self.records.get(req.rid)
        identity = self.identity(req)
        if rec is None or rec.identity != identity:
            # A retract is a new cache generation, but not a new deadline.
            same_request = rec is not None and rec.identity[0] == id(req)
            due = rec.due if same_request else now - waited(req) + ax_deadline.budget_s(req, deadline)
            cold = rec.cold if same_request else ax_deadline.deadline_cold(req, deadline)
            self.generation += 1
            received = rec.received if same_request else now - waited(req)
            rec = Record(identity, self.generation, due, cold, received,
                         participant=bool(same_request and rec.participant),
                         fallback=bool(same_request))
            self.records[req.rid] = rec
            self.dependencies.pop(req.rid, None)
        return rec

    def _shared(self, a, b):
        ra, rb = self.records[a.rid], self.records[b.rid]
        key = tuple(sorted((ra.generation, rb.generation)))
        if key in self.pairs:
            return self.pairs[key]
        if readiness.domain(a) != readiness.domain(b):
            self.pairs[key] = 0
            return 0
        for req, rec in ((a, ra), (b, rb)):
            if not rec.raw:
                rec.raw = array("q", req.origin_input_ids).tobytes()
                rec.hashes = tuple(hash(rec.raw[i:i + self.BLOCK_BYTES])
                                   for i in range(0, len(rec.raw), self.BLOCK_BYTES))
        n = 0
        for ha, hb in zip(ra.hashes, rb.hashes):
            if ha != hb:
                break
            n += self.BLOCK_BYTES
        n = min(n, len(ra.raw), len(rb.raw))
        # Hash equality is only a filter. Verify exact bytes once per pair;
        # collisions fall back to token comparison and cannot invent a family.
        if ra.raw[:n] != rb.raw[:n]:
            n = 0
        tokens = n // 8
        while (tokens < min(len(a.origin_input_ids), len(b.origin_input_ids))
               and a.origin_input_ids[tokens] == b.origin_input_ids[tokens]):
            tokens += 1
        self.pairs[key] = tokens
        return tokens

    def _prime_shared(self, reqs):
        """Exact all-pair LCPs from adjacent lexical neighbors, once per cohort.

        For sorted strings, LCP(i,j) is the minimum adjacent LCP on [i,j].
        Thus long token spans need exact verification O(n), not O(n²). This
        computes pair lengths only; it does NOT merge transitive families.
        """
        signature = tuple(sorted(self.records[r.rid].generation for r in reqs))
        if signature == self.pair_signature:
            return
        self.pair_signature = signature
        domains = []
        for req in reqs:
            domain = readiness.domain(req)
            group = next((items for key, items in domains if key == domain), None)
            if group is None:
                group = []
                domains.append((domain, group))
            group.append(req)
            rec = self.records[req.rid]
            if not rec.raw:
                rec.raw = array("q", req.origin_input_ids).tobytes()
                rec.hashes = tuple(hash(rec.raw[i:i + self.BLOCK_BYTES])
                                   for i in range(0, len(rec.raw), self.BLOCK_BYTES))
        for _, reqs in domains:
            ordered = sorted(reqs, key=lambda r: self.records[r.rid].raw)
            adjacent = [self._shared(a, b) for a, b in zip(ordered, ordered[1:])]
            for i, left in enumerate(ordered):
                shared = len(left.origin_input_ids)
                for j in range(i + 1, len(ordered)):
                    shared = min(shared, adjacent[j - 1])
                    key = tuple(sorted((self.records[left.rid].generation,
                                        self.records[ordered[j].rid].generation)))
                    self.pairs[key] = shared

    def _fallback(self, rid, reason):
        dep = self.dependencies[rid]
        if dep.state != "FALLBACK":
            self.stats["fallback_" + reason] += 1
        dep.state, dep.reason = "FALLBACK", reason
        self.records[rid].fallback = True

    def prepare(self, waiting, continuation, running, held, held_by, now, waited,
                deadline, chunk, grid, short, can_produce, held_depth=None):
        """Phase one: lifetimes, exact relationships, readiness and stable order."""
        self.sequence += 1
        self._now = now
        self.events = []
        if self.trace_start is None:
            self.trace_start = now
        live = {r.rid: r for r in [*waiting, *running, *([continuation] if continuation else [])]}
        wait_ids = {r.rid for r in waiting}
        tracked_ids = set(self.records) & set(live)
        candidates = [r for r in waiting if readiness.eligible(r)]
        bounded = len(candidates) <= self.cfg.max_candidates
        candidates = candidates if bounded else []
        tracked_ids.update(r.rid for r in candidates)
        if continuation is not None and readiness.eligible(continuation):
            tracked_ids.add(continuation.rid)
        for rid in tracked_ids:
            self._record(live[rid], now, waited, deadline)
            if live[rid].output_ids:
                # Decoding producers no longer participate in discovery.
                self.records[rid].raw, self.records[rid].hashes = b"", ()
        held_depth = held_depth or {}

        def can_lift(producer, members):
            if producer not in held:
                return True
            owner = held_by.get(producer)
            if owner in {r for r, _ in members}:
                return True
            # Native LPM may hold a deep family behind an unrelated request
            # sharing only a system prefix. A verified producer can publish
            # that entire native-held span itself; keeping this shallow hold
            # would defeat the deep family. Never override an unexplained edge.
            depth = held_depth.get(producer, 0)
            return bool(depth > 0 and owner in wait_ids and owner in self.records
                        and readiness.eligible(live[owner])
                        and max(g for _, g in members) >= depth
                        and self._shared(live[producer], live[owner]) >= depth)
        for rid in list(self.records):
            if rid not in live:
                del self.records[rid]
        gens = {r.generation for r in self.records.values()}
        self.pairs = {k: v for k, v in self.pairs.items() if all(g in gens for g in k)}
        self._prime_shared([*candidates, *([continuation] if continuation is not None
            and readiness.eligible(continuation) and continuation.rid not in wait_ids else [])])
        for rid in list(self.dependencies):
            if rid not in wait_ids:
                del self.dependencies[rid]
                continue
            dep = self.dependencies[rid]
            req = live[rid]
            match = readiness.view(req)
            rec = self.records[rid]
            # A completed/cancelled producer need not stay alive to consume an
            # already-published checkpoint. A fresh consumer match is decisive.
            if match is not None and match.reason(req.seqlen, short) == "ready":
                if dep.state != "READY":
                    self.stats["ready"] += 1
                    self.events.append(dict(event="cache_ready", rid=rid, producer=dep.producer,
                                            t=now, device=match.device))
                dep.state, dep.reason = "READY", "consumer_checkpoint"
                continue
            producer = live.get(dep.producer)
            parent = self.records.get(dep.producer)
            if parent is None or parent.generation != dep.producer_generation or producer.finished():
                self._fallback(rid, "producer_gone")
                continue
            if dep.state == "FALLBACK":
                continue
            progress = len(producer.prefix_indices)
            if progress > dep.progress:
                dep.progress, dep.progress_at = progress, now
            need = max(0, dep.target - progress) + req.seqlen - dep.target
            if now - dep.created >= self.cfg.max_hold_s:
                self._fallback(rid, "hold_timeout")
            elif producer is continuation and now - dep.progress_at >= self.cfg.stall_s:
                self._fallback(rid, "producer_stalled")
            elif self.starved(req, waited(req), deadline):
                self._fallback(rid, "starvation")
            elif rec.due - now - deadline.arrival_offset_s < ax_deadline.service_s(need, chunk, deadline):
                self._fallback(rid, "deadline")
            else:
                dep.state = "WAIT_PREFIX"
                dep.reason = match.reason(req.seqlen, short) if match else "no_native_match"

        # Existing groups survive waiting -> partial -> running. Discovery uses
        # direct producer-to-rider prefixes, never a transitive union of LCPs.
        used = set(self.dependencies)
        used.update(d.producer for d in self.dependencies.values() if d.state != "FALLBACK")
        producers = ([continuation] if continuation is not None and readiness.eligible(continuation) else [])
        producers += [r for r in candidates if r.rid not in used and not self.records[r.rid].fallback]
        positions = {r.rid: i for i, r in enumerate(waiting)}
        offers = []
        for producer in producers:
            if producer.rid in self.dependencies or not can_produce(producer):
                continue
            members = []
            progress = len(producer.prefix_indices)
            for req in candidates:
                if req is producer or req.rid in used or self.records[req.rid].fallback:
                    continue
                match = readiness.view(req)
                if match is None:
                    continue
                shared = self._shared(producer, req) // grid * grid
                tail = req.seqlen - shared
                if shared - match.device < self.cfg.min_shared or not 0 < tail <= short:
                    continue
                cost = ax_deadline.service_s(max(0, shared - progress) + tail, chunk, deadline)
                if self.records[req.rid].due - now - deadline.arrival_offset_s >= cost:
                    members.append((req, shared))
            if not members:
                continue
            # Count only tails whose cumulative, conservative service estimate
            # fits their own deadline, not every individually cheap rider.
            publish_s = ax_deadline.service_s(max(0, max(g for _, g in members) - progress), chunk, deadline)
            elapsed, feasible = publish_s, []
            for req, target in sorted(members, key=lambda item: (self.records[item[0].rid].due,
                                                                item[0].seqlen - item[1], item[0].rid)):
                cost_s = ax_deadline.service_s(req.seqlen - target, chunk, deadline)
                if self.records[req.rid].due - now - deadline.arrival_offset_s >= elapsed + cost_s:
                    feasible.append((req, target))
                    elapsed += cost_s
            members = feasible
            if not members:
                continue
            if not can_lift(producer.rid, [(r.rid, g) for r, g in members]):
                continue  # cannot explain this native hold; do not clear it
            target = max(g for _, g in members)
            publish = max(0, target - progress)
            cost = publish + sum(r.seqlen - g for r, g in members)
            # Account for a producer tail only when counting it as a completion.
            own_tail = producer.seqlen - target
            benefit = len(members)
            if own_tail <= short:
                cost += own_tail
                benefit += 1
            # Equal group costs prefer the producer with the shorter own tail,
            # then queue order. Random client RIDs must not choose the winner.
            key = (producer is not continuation, cost / benefit,
                   producer.seqlen - progress, positions.get(producer.rid, -1))
            offers.append((key, producer, members))
        for _, producer, members in sorted(offers, key=lambda x: x[0]):
            if producer.rid in used and producer is not continuation:
                continue
            members = [(r, g) for r, g in members if r.rid not in used]
            if not members:
                continue
            # Recheck held ownership after overlapping offers were removed.
            if not can_lift(producer.rid, [(r.rid, g) for r, g in members]):
                continue
            self.records[producer.rid].participant = True
            used.add(producer.rid)
            for req, target in members:
                self.records[req.rid].participant = True
                self.dependencies[req.rid] = Dependency(
                    producer.rid, self.records[producer.rid].generation, target,
                    now, len(producer.prefix_indices), now)
                used.add(req.rid)
                self.stats["dependency"] += 1
                self.events.append(dict(event="dependency", rid=req.rid, producer=producer.rid,
                                        target=target, t=now))

        effective_held = set(held)
        wait_prefix = set()
        self.work = {}
        group_slack = {}
        groups = {}
        for rid, dep in self.dependencies.items():
            if dep.state == "WAIT_PREFIX":
                wait_prefix.add(rid)
                groups.setdefault(dep.producer, []).append((rid, dep.target))
            elif dep.state in ("READY", "FALLBACK"):
                # The verified relationship supersedes a stale in-batch hold.
                effective_held.discard(rid)
        for producer, members in groups.items():
            if producer in live and can_lift(producer, members):
                effective_held.discard(producer)
            p = live.get(producer)
            if p is not None:
                target = max(g for _, g in members)
                cost = max(0, target - len(p.prefix_indices)) + sum(live[r].seqlen - g for r, g in members)
                benefit = len(members)
                if p.seqlen - target <= short:
                    cost += p.seqlen - target
                    benefit += 1
                self.work[producer] = max(1, cost / benefit)
                group_slack[producer] = max(self.records[r].due - now - deadline.arrival_offset_s
                    - ax_deadline.service_s(max(0, g - len(p.prefix_indices)) + live[r].seqlen - g,
                                            chunk, deadline) for r, g in members)
        self.effective_held, self.wait_prefix = effective_held, wait_prefix
        self.bounded = bounded

        def key(item):
            i, req = item
            rec = self.records.get(req.rid)
            if self.starved(req, waited(req), deadline) and rec and rec.participant:
                return 0, 0, -waited(req), i
            if req.rid in effective_held or req.rid in wait_prefix:
                return 3, 0, 0, i
            if ax_deadline.is_starved(req, waited(req), deadline):
                return 0, 0, -waited(req), i
            left = ax_deadline.remaining_tokens(req)
            slack = (rec.due - now - deadline.arrival_offset_s - ax_deadline.service_s(left, chunk, deadline)
                     if rec and rec.participant else ax_deadline.slack_s(req, left, waited(req), chunk, deadline))
            slack = max(slack, group_slack.get(req.rid, slack))
            # 132 chain_first (same rule as ax_deadline.tier_order): rescuable cold before rescuable warm.
            group = 0 if (not deadline.chain_first or slack < 0 or ax_deadline.deadline_cold(req, deadline)) else 1
            return (2 if slack < 0 else 1), group, self.work.get(req.rid, left), i

        ranked = [r for _, r in sorted(enumerate(waiting), key=key)]
        self.rows = []
        for req in ranked[:self.cfg.max_candidates]:
            rec, dep = self.records.get(req.rid), self.dependencies.get(req.rid)
            match = readiness.view(req)
            self.rows.append(dict(rid=req.rid, gen=rec.generation if rec else None,
                producer=dep.producer if dep else None, target=dep.target if dep else 0,
                state=dep.state if dep else ("PRODUCER_WAIT" if req.rid in groups else "ORDINARY"),
                reason=dep.reason if dep else "ordinary",
                device=match.device if match else -1, full=match.full if match else -1,
                host=match.host if match else -1, left=req.seqlen - (match.device if match else 0),
                held=req.rid in held, held_by=held_by.get(req.rid),
                held_depth=held_depth.get(req.rid, 0),
                effective_held=req.rid in effective_held, work=self.work.get(req.rid),
                received=rec.received if rec else None,
                due_in=round(rec.due - now, 6) if rec else None))
            hold_check = getattr(req, "_ax_lpm_hold_check", None)
            if hold_check is not None:
                self.rows[-1]["lpm_hold"] = hold_check
                self.stats["lpm_hold_" + hold_check["decision"]] += 1
        return ranked, effective_held, wait_prefix

    def starved(self, req, waited, deadline):
        rec = self.records.get(req.rid)
        cold = rec.cold if rec and rec.participant else ax_deadline.deadline_cold(req, deadline)
        return waited > (deadline.max_wait_s if cold else deadline.max_wait_warm_s)

    def should_park(self, continuation, left, continuation_waited, head, head_waited,
                    budget, kv_room, rounds, parked_s, deadline):
        if (deadline.park_max_rounds == 0 or rounds >= deadline.park_max_rounds
                or (rounds and parked_s >= deadline.park_max_s)):
            return False
        work = ax_deadline.remaining_tokens(head)
        if work > min(budget, kv_room):
            return False
        def slack(req, tokens, waited):
            rec = self.records.get(req.rid)
            if rec and rec.participant:
                # prepare() used the same rank-0 clock; due_in includes waiting.
                return rec.due - self._now - deadline.arrival_offset_s - ax_deadline.service_s(tokens, budget, deadline)
            return ax_deadline.slack_s(req, tokens, waited, budget, deadline)
        starved = self.starved(head, head_waited, deadline)
        return ((starved or slack(head, work, head_waited) >= 0)
                and (starved or left > deadline.park_min_remaining
                     or slack(continuation, left, continuation_waited) < 0))

    def ready(self, rid):
        dep = self.dependencies.get(rid)
        return dep is not None and dep.state == "READY"

    def note_prefill_finished(self, req, logger):
        """Result path, including the last batch when no later plan is needed.

        Only rank 0 owns records. Uses the engine's existing completion stamp;
        no CUDA synchronization or additional timestamp on the model stream.
        """
        rec = self.records.get(req.rid)
        if rec is None or not rec.participant or rec.first_finished is not None:
            return
        now = req.time_stats.prefill_finished_time
        if rec.first_admitted is None or now < rec.first_admitted:
            return  # an old completion stamp after a retract is not this prefill
        rec.first_finished = now
        self.stats["prefill_finished"] += 1
        if (self.trace_start is not None and self.sequence <= self.cfg.trace_rounds
                and now - self.trace_start <= self.cfg.trace_s):
            logger.info("[ax-prefix-finish] %s", json.dumps(dict(epoch=self.epoch, rid=req.rid,
                gen=rec.generation, t=now, first_admitted=rec.first_admitted,
                received=rec.received, due=rec.due), separators=(",", ":")))

    def log(self, logger, now, plan, results, admitted):
        self.stats["plans"] += 1
        self.stats["reserved"] += len(plan["ready"])
        for rid, reason in results:
            self.stats["admission_" + reason] += 1
        for rid in admitted:
            rec = self.records.get(rid)
            if rec and rec.participant and rec.first_admitted is None:
                rec.first_admitted = now
            if rid in self.dependencies:
                self.stats["dependent_admitted"] += 1
                self.dependencies[rid].state = "ADMITTED"
            if rid in self.work:
                self.stats["producer_admitted"] += 1
        if self.trace_start is None:
            self.trace_start = now
        trace = (self.sequence <= self.cfg.trace_rounds and now - self.trace_start <= self.cfg.trace_s)
        if trace:
            outcomes = dict(results)
            stop = next((reason for _, reason in results if reason.startswith("queue_stop_")), None)
            for row in self.rows:
                row["result"] = outcomes.get(row["rid"], "not_reached:" + stop if stop else "not_attempted")
            logger.info("[ax-prefix-decision] %s", json.dumps(dict(
                epoch=self.epoch, sequence=self.sequence, t=round(now, 6),
                plan=plan, candidates=self.rows, discovery_bounded=self.bounded,
                events=self.events, result=results, admitted=admitted), separators=(",", ":")))
        elif not self.trace_closed:
            self.trace_closed = True
            logger.info("[ax-prefix-trace-end] epoch=%d sequence=%d elapsed=%.3f limits=%ss/%s_rounds",
                        self.epoch, self.sequence, now - self.trace_start, self.cfg.trace_s, self.cfg.trace_rounds)
        if now - self.log_at >= 30:
            self.log_at = now
            logger.info("[ax-prefix-stats] epoch=%d %s", self.epoch, json.dumps(self.stats, sort_keys=True))
