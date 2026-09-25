"""Opt-in 120 diagnostics. Counts observed decisions, never attributes GPU time.

The scheduler instantiates this on TP0 only. State lives on waiting requests, so
aborted requests do not leak through a collector-owned map. There is no CUDA,
collective, event synchronization, or request/prompt content inspection here.
"""

import json
import time


class AxAdmissionTrace:
    def __init__(self, logger, clock=time.monotonic, snapshot_interval=30.0, max_bytes=8 * 1024 * 1024):
        if not 512 <= max_bytes <= 8 * 1024 * 1024:
            raise ValueError("admission log budget must be between 512 bytes and 8 MiB")
        self.logger = logger
        self.clock = clock
        self.snapshot_interval = snapshot_interval
        self.next_snapshot = 0.0
        self.max_bytes = max_bytes
        self.bytes_emitted = 0
        self.exhausted = False
        self.limit_notice = json.dumps({"event": "budget_exhausted", "max_bytes": max_bytes})

    @staticmethod
    def _charge(payload):
        # Includes a conservative allowance for the ordinary server log prefix.
        return len(payload.encode("utf-8")) + 128

    def _emit(self, row):
        if self.exhausted:
            return
        payload = json.dumps(row, separators=(",", ":"))
        if self.bytes_emitted + self._charge(payload) + self._charge(self.limit_notice) > self.max_bytes:
            payload = self.limit_notice
            self.exhausted = True
        self.bytes_emitted += self._charge(payload)
        self.logger.info("[ax-admission] %s", payload)

    def _state(self, req):
        state = getattr(req, "_ax_admission_trace_state", None)
        if state is None:
            state = {"first_observed": self.clock(), "decisions": {}, "attempt_reason": None}
            req._ax_admission_trace_state = state
        return state

    def begin_attempt(self, req):
        if self.exhausted:
            return
        self._state(req)["attempt_reason"] = None

    def record(self, req, reason):
        if self.exhausted:
            return
        state = self._state(req)
        counts = state["decisions"]
        counts[reason] = counts.get(reason, 0) + 1
        state["attempt_reason"] = reason

    def record_many(self, reqs, reason):
        if self.exhausted:
            return
        for req in reqs:
            self.record(req, reason)

    def rejected(self, req, fallback):
        if self.exhausted:
            return
        # A more precise PrefillAdder branch wins over its generic result.
        if self._state(req)["attempt_reason"] is None:
            self.record(req, fallback)

    def _row(self, req, now):
        state = self._state(req)
        return {
            "rid": req.rid,
            "observed_wait_s": max(0.0, now - state["first_observed"]),
            "decisions": dict(state["decisions"]),
            "device_prefix_tokens": len(req.prefix_indices),
            "host_hit_tokens": req.host_hit_length,
            "mamba_host_hit": getattr(req, "mamba_host_hit_length", 0),
            "needs_host_load_back": req.needs_host_load_back(),
            "unmatched_tokens_at_observation": req.seqlen - len(req.prefix_indices),
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
        if now < self.next_snapshot or not waiting:
            return
        self.next_snapshot = now + self.snapshot_interval
        # Bounded log volume even if a different workload grows an unbounded queue.
        rows = [self._row(req, now) for req in waiting[:32]]
        self._emit({
            "event": "waiting", "queue_size": len(waiting), "sample": rows,
            "sample_order": "current_queue", "truncated": len(waiting) > len(rows),
        })
