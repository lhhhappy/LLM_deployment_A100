"""Opt-in 120 diagnostics; decision counts and following wall intervals.

The scheduler instantiates this on TP0 only. State lives on waiting requests, so
aborted requests do not leak through a collector-owned map. There is no CUDA,
collective, event synchronization, or request/prompt content inspection here.
"""

import json
import time


class AxAdmissionTrace:
    def __init__(self, logger, clock=time.perf_counter, snapshot_interval=30.0,
                 max_bytes=8 * 1024 * 1024, to_epoch=None):
        if not 512 <= max_bytes <= 8 * 1024 * 1024:
            raise ValueError("admission log budget must be between 512 bytes and 8 MiB")
        self.logger = logger
        self.clock = clock
        # Production passes the SAME converter as the scoring timestamps.
        # The stdlib fallback supports standalone CPU fixtures.
        offset = time.time() - time.perf_counter()
        self.to_epoch = to_epoch or (lambda value: value + offset)
        self.snapshot_interval = snapshot_interval
        self.next_snapshot = 0.0
        self.max_bytes = max_bytes
        self.bytes_emitted = 0
        self.exhausted = False
        self.step_seq = 0
        self.step_open = False
        self.last_snapshot_nonempty = False
        self.limit_notice = json.dumps({"schema": 3, "event": "budget_exhausted", "max_bytes": max_bytes})

    @staticmethod
    def _charge(payload):
        # Includes a conservative allowance for the ordinary server log prefix.
        return len(payload.encode("utf-8")) + 128

    def _emit(self, row):
        if self.exhausted:
            return
        payload = json.dumps({"schema": 3, **row}, separators=(",", ":"))
        if self.bytes_emitted + self._charge(payload) + self._charge(self.limit_notice) > self.max_bytes:
            payload = self.limit_notice
            self.exhausted = True
        self.bytes_emitted += self._charge(payload)
        self.logger.info("[ax-admission] %s", payload)

    def _state(self, req):
        state = getattr(req, "_ax_admission_trace_state", None)
        entered = getattr(req.time_stats, "wait_queue_entry_time", None)
        if state is None or state["queue_entry"] != entered:
            state = {"first_observed": self.clock(), "decisions": {},
                     "examples": {}, "attempt_reason": None, "queue_entry": entered,
                     "reason_step": None, "pending_interval": None,
                     "wall_by_reason": {}, "wall_by_batch": {}, "intervals": 0}
            req._ax_admission_trace_state = state
        return state

    def begin_step(self, waiting):
        """Close only the preceding instrumented step, on still-waiting objects.

        No collector-owned request references survive a call. Removed/aborted
        requests are not silently reported as still waiting. Selection CPU time
        and missing hooks remain uncovered; snapshots never extrapolate it.
        """
        if self.exhausted:
            return
        now = self.clock()
        self.step_seq += 1
        self.step_open = True
        for req in waiting:
            state = self._state(req)
            pending = state["pending_interval"]
            state["pending_interval"] = None
            if pending is None:
                continue
            seq, started, reason, batch = pending
            if seq != self.step_seq - 1 or now < started:
                continue
            elapsed = now - started
            for key, label in (("wall_by_reason", reason), ("wall_by_batch", batch)):
                counts = state[key]
                counts[label] = counts.get(label, 0.0) + elapsed
            state["intervals"] += 1

    def end_step(self, waiting, batch_kind):
        """Label the following wall interval by this decision, not by its cause.

        Overlap execution and asynchronous restoration may change readiness
        during this interval. This is neither GPU service time nor proof that
        the observed rejection condition persisted for the entire interval.
        """
        if self.exhausted or not self.step_open:
            return
        now = self.clock()
        self.step_open = False
        for req in waiting:
            state = self._state(req)
            reason = (state["attempt_reason"]
                      if state["reason_step"] == self.step_seq else None)
            state["pending_interval"] = (
                self.step_seq, max(now, state["first_observed"]),
                reason or "unobserved_path", batch_kind
            )

    def begin_attempt(self, req):
        if self.exhausted:
            return
        self._state(req)["attempt_reason"] = None

    def record(self, req, reason, context=None):
        if self.exhausted:
            return
        state = self._state(req)
        counts = state["decisions"]
        counts[reason] = counts.get(reason, 0) + 1
        if context is not None and reason not in state["examples"]:
            # One small example per fixed-vocabulary reason, not one per step.
            state["examples"][reason] = dict(context)
        state["attempt_reason"] = reason
        state["reason_step"] = self.step_seq if self.step_open else None

    def record_many(self, reqs, reason, context=None):
        if self.exhausted:
            return
        for req in reqs:
            self.record(req, reason, context)

    def rejected(self, req, fallback):
        if self.exhausted:
            return
        # A more precise PrefillAdder branch wins over its generic result.
        if self._state(req)["attempt_reason"] is None:
            self.record(req, fallback)

    def _row(self, req, now):
        state = self._state(req)
        entered = getattr(req.time_stats, "wait_queue_entry_time", 0.0)
        observed = max(0.0, now - state["first_observed"])
        covered = sum(state["wall_by_reason"].values())
        return {
            "rid": req.rid,
            "observed_at_s": self.to_epoch(now),
            "queue_entry_at_s": self.to_epoch(entered) if entered else None,
            "observed_wait_s": observed,
            "decisions": dict(state["decisions"]),
            "examples": dict(state["examples"]),
            "device_prefix_tokens": len(req.prefix_indices),
            "host_hit_tokens": req.host_hit_length,
            "mamba_host_hit": getattr(req, "mamba_host_hit_length", 0),
            "needs_host_load_back": req.needs_host_load_back(),
            "unmatched_tokens_at_observation": req.seqlen - len(req.prefix_indices),
            "decision_intervals": {
                "seconds_by_reason": dict(state["wall_by_reason"]),
                "seconds_by_batch": dict(state["wall_by_batch"]),
                "count": state["intervals"],
                "uncovered_s": max(0.0, observed - covered),
            },
        }

    def admitted(self, req):
        if not self.exhausted:
            row = self._row(req, self.clock())
            self._emit({"event": "admit", **row})
        req._ax_admission_trace_state = None

    def snapshot(self, waiting):
        if self.exhausted:
            return
        now = self.clock()
        if now < self.next_snapshot or (not waiting and not self.last_snapshot_nonempty):
            return
        self.next_snapshot = now + self.snapshot_interval
        self.last_snapshot_nonempty = bool(waiting)
        # Bounded log volume even if a different workload grows an unbounded queue.
        rows = [self._row(req, now) for req in waiting[:32]]
        self._emit({
            "event": "waiting", "queue_size": len(waiting), "sample": rows,
            "observed_at_s": self.to_epoch(now),
            "sample_order": "current_queue", "truncated": len(waiting) > len(rows),
        })
