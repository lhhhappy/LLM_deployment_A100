"""124: one live continuation and one parked continuation, opt-in.

Parking retains the original Req, request row, KV lock and live Mamba state.
The radix checkpoint may lag the computed prefix; it is never a resume point.
Only rank 0 evaluates ages/costs. All ownership transitions apply its broadcast.
"""

import json
import logging
import time
from dataclasses import dataclass

from sglang.srt.managers import ax_deadline as dl

logger = logging.getLogger(__name__)


def received(req):
    ts = req.time_stats
    return (getattr(ts, "scheduler_recv_time", 0.0)
            or getattr(ts, "wait_queue_entry_time", 0.0))


def age(req, now):
    start = received(req)
    return max(0.0, now - start) if start else 0.0


def remaining(req):
    return max(0, req.seqlen - len(req.prefix_indices))


def age_limit(req, cfg):
    # 60 s margin to the harness's 1200 s no-event limit. This is a server
    # age proxy, not a claim to know the client's dispatch time or its bucket.
    return min(cfg.max_wait_s if dl.deadline_cold(req, cfg)
               else cfg.max_wait_warm_s, 1140.0)


def rescue_decision(owner, head, now, chunk, cfg):
    """Pure request-state/cost decision; None means no multi-round rescue.

    A one-round waiter remains the existing 124 parking path. Require a cold
    waiter, actual (not family-discounted) work, and enough age allowance to
    finish both requests. No IDs, cohort positions or harness labels are used.
    """
    work_a, work_b = remaining(owner), remaining(head)
    age_a, age_b = age(owner, now), age(head, now)
    slack_a = dl.slack_s(owner, work_a, age_a, chunk, cfg)
    # Under overlap the prior chunk may still be on the GPU when this plan is
    # made. Charge its entire modeled cost, plus one fixed dispatch overhead,
    # instead of treating the fence/ownership transfer as free.
    handoff_s = cfg.fixed_s * cfg.load_factor
    if owner.inflight_middle_chunks > 0:
        handoff_s += dl.service_s(owner.extend_range.length, chunk, cfg)
    slack_b = dl.slack_s(head, work_b, age_b, chunk, cfg) - handoff_s
    if (not received(owner) or not received(head) or work_b <= chunk or work_a <= chunk
            or not dl.deadline_cold(head, cfg) or slack_a >= 0 or slack_b < 0):
        return None
    if (age_a + cfg.arrival_offset_s + dl.service_s(work_a + work_b, chunk, cfg)
            + handoff_s + 1.0 >= age_limit(owner, cfg)):
        return None
    # Finite service lease: 50% model-error allowance, bounded by B's actual
    # deadline. This is a scheduling guard, not a measured speed guarantee.
    lease_s = min(1.5 * dl.service_s(work_b, chunk, cfg) + handoff_s,
                  dl.budget_s(head, cfg) - age_b - cfg.arrival_offset_s)
    return dict(a=owner.rid, b=head.rid, a_remaining=work_a, b_remaining=work_b,
                a_age=age_a, b_age=age_b, a_slack=slack_a, b_slack=slack_b,
                handoff_s=handoff_s, lease_s=lease_s, lease_end=now + lease_s,
                chunk=chunk, reason="owner_hopeless_waiter_rescuable")


@dataclass
class Parked:
    req: object
    computed: int
    row: object
    mamba: object
    tracking: object

    @classmethod
    def capture(cls, req):
        return cls(req, len(req.prefix_indices), req.kv.req_pool_idx,
                   req.kv.mamba_pool_idx,
                   getattr(req.kv, "mamba_ping_pong_track_buffer", None))

    def check(self):
        if (len(self.req.prefix_indices) != self.computed
                or self.req.kv.req_pool_idx != self.row
                or self.req.kv.mamba_pool_idx is not self.mamba
                or getattr(self.req.kv, "mamba_ping_pong_track_buffer", None) is not self.tracking):
            raise RuntimeError("[ax-124m] parked request changed live KV ownership")


def parked_reqs(scheduler):
    state = getattr(scheduler, "_ax_multi_park", None)
    return state.reqs() if state is not None else ()


class State:
    """No allocator ownership of its own. At most one parked request."""

    def __init__(self):
        self.parked = []
        self.active = None  # also tracks the final chunk until its callback
        self.protect = False  # a starvation restoration runs to completion
        self.pending_abort = {}
        self.watch = {}
        self.sequence = 0
        self.lease_end = 0.0
        self.chunk = 1

    def reqs(self):
        return tuple(p.req for p in self.parked)

    def emit(self, s, event, **fields):
        s._ax_rank0_decide(lambda: logger.info(
            "[ax-124m] %s", json.dumps(dict(event=event, seq=self.sequence, t=time.perf_counter(),
                                          **fields), separators=(",", ":"))))

    def reset(self):
        assert not self.parked and not self.pending_abort
        self.active = None
        self.protect = False
        self.watch.clear()
        self.sequence = 0
        self.lease_end = 0.0

    def observe(self, s):
        """Called after result processing, including late middle callbacks."""
        done = []
        for rid, req in self.watch.items():
            if req.output_ids or req.finished():
                def record(req=req):
                    first = getattr(req.time_stats, "prefill_finished_time", 0.0)
                    now = first if req.output_ids and first else time.perf_counter()
                    return dict(rid=req.rid, first_token=bool(req.output_ids),
                                first_token_at=now if req.output_ids else None, ttft_server_s=age(req, now),
                                coverage_risk=age(req, now) >= 1200.0)
                fields = s._ax_rank0_decide(record)
                self.emit(s, "first_token" if req.output_ids else "terminal", **fields)
                done.append(rid)
        for rid in done:
            del self.watch[rid]

    def prepare_step(self, s):
        """After stashing the latest chunk, before last_batch is merged."""
        if not self.parked:
            return
        p = self.parked[0]
        p.check()
        cfg = s._ax_admission_cfgs()[0]

        def decide():
            if self.active is None or self.active.output_ids or self.active.finished():
                return "completed"
            # A final chunk has been launched: let its per-request callback
            # retire before assigning the next continuation.
            if s.chunked_req is None or self.protect:
                return None
            chunk = self.chunk
            now = time.perf_counter()
            if (age(p.req, now) + cfg.arrival_offset_s
                    + dl.service_s(remaining(p.req), chunk, cfg) + 1.0
                    >= age_limit(p.req, cfg)):
                return "absolute_age"
            if now >= self.lease_end:
                return "service_lease"
            return None

        reason = s._ax_rank0_decide(decide)
        if reason is None:
            return
        # Rare ownership transition only: no per-chunk GPU synchronization.
        # Pending CPU callbacks keep their per-Req inflight counters intact.
        s.forward_stream.synchronize()
        old = s.chunked_req
        if reason == "completed" and old is not None:
            raise RuntimeError("[ax-124m] completed rescuer still owns a partial")
        self.parked.clear()
        if old is not None:
            self.parked.append(Parked.capture(old))
        s.chunked_req = p.req
        self.active = p.req if old is not None else None
        self.protect = old is not None
        s.running_batch.batch_is_full = False
        self.emit(s, "resume", rid=p.req.rid, reason=reason,
                  computed=p.computed, park=old.rid if old is not None else None)

    def plan(self, s, adder, running_batch, chunk, prefix_plan):
        """Rank-0 preview; native admission and its rollback remain authoritative."""
        a = s.chunked_req
        if self.parked or a is None or a.output_ids or a.finished():
            return None
        if a.beam_group is not None or getattr(a, "session", None):
            return None
        if (running_batch.batch_is_full
                or s.get_num_allocatable_reqs(len(running_batch.reqs),
                                             running_batch=running_batch) <= 0
                or len(running_batch.reqs) + 2 > s.max_running_requests):
            # Reserve the parked row in the logical running limit too.
            return None
        held = set(getattr(s.policy, "ax_held", ()))
        if prefix_plan is not None:
            held.update(prefix_plan["wait_prefix"])
        cfg = s._ax_admission_cfgs()[0]
        now = time.perf_counter()
        for b in s.waiting_queue[:64]:
            if (b.rid in held or b.needs_host_load_back() or b.beam_group is not None
                    or getattr(b, "session", None)):
                continue
            if adder._request_total_tokens(b, remaining(b)) >= adder.rem_total_tokens:
                continue
            decision = rescue_decision(a, b, now, chunk, cfg)
            if decision is not None:
                decision.update(sequence=self.sequence + 1, a_computed=len(a.prefix_indices),
                                b_computed=len(b.prefix_indices),
                                a_inflight=a.inflight_middle_chunks)
                return decision
        return None

    def begin(self, s, decision):
        if decision is None:
            return None
        matches = [r for r in s.waiting_queue if r.rid == decision["b"]]
        valid = (not self.parked and s.chunked_req is not None
                 and s.chunked_req.rid == decision["a"] and len(matches) == 1
                 and decision["sequence"] == self.sequence + 1
                 and len(s.chunked_req.prefix_indices) == decision["a_computed"]
                 and s.chunked_req.inflight_middle_chunks == decision["a_inflight"]
                 and len(matches[0].prefix_indices) == decision["b_computed"])
        agreed = s._ax_prefix_consensus(valid)
        if not agreed or not valid:
            self.emit(s, "decline", a=decision["a"], b=decision["b"], reason="rank_precondition")
            return None
        a, b = s.chunked_req, matches[0]
        s.forward_stream.synchronize()
        self.parked.append(Parked.capture(a))
        s.chunked_req = None
        # Admit only B in this transaction. A failed rematch/allocation then
        # leaves an unspent adder and resumes A in this same scheduling pass.
        return (a, b, decision)

    def finish(self, s, adder, transaction, running_batch, batch_was_full):
        a, b, decision = transaction
        admitted = b in adder.can_run_list
        # Detection, not silent recovery: allocations must agree before any
        # rank launches a forward with a different partial.
        if not s._ax_prefix_consensus((admitted, b.extend_range.start, b.extend_range.end,
                                       len(b.prefix_indices), a.inflight_middle_chunks,
                                       adder.new_chunked_req.rid if adder.new_chunked_req else None)):
            raise RuntimeError("[ax-124m] native admission differs across ranks")
        if not admitted:
            self.parked.pop().check()
            a.init_next_round_input()  # no cache rematch / no KDA COW
            running_batch.batch_is_full = batch_was_full
            s.chunked_req = adder.add_chunked_req(a)
            self.emit(s, "rollback", a=a.rid, b=b.rid)
            return
        self.active = b
        self.protect = False
        self.lease_end = decision["lease_end"]
        self.chunk = decision["chunk"]
        self.sequence += 1
        self.watch[a.rid], self.watch[b.rid] = a, b
        self.emit(s, "yield", **decision, computed=len(a.prefix_indices),
                  checkpoint=a.kv.cache_protected_len,
                  allocated=a.kv.kv_allocated_len, inflight=a.inflight_middle_chunks)

    def forget(self, s, req, reason):
        self.parked[:] = [p for p in self.parked if p.req is not req]
        self.pending_abort.pop(req.rid, None)
        if self.active is req:
            self.active = None
        if req.rid in self.watch:
            self.emit(s, "terminal", rid=req.rid, reason=reason)
            del self.watch[req.rid]
        if not self.parked:
            self.active = None
            self.protect = False
