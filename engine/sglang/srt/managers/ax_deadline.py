"""[ax] 124/125 decisions: deadline-tiered cold admission, chunk-level parking, backlog decode relief.

Pure functions of request state, elapsed seconds and budgets. The scheduler evaluates them on TP rank 0
only and broadcasts the outcome, because arrival stamps, clocks and throughput differ between ranks and
a 1 ms difference is enough to flip an order (evidence/opening-q123-20260925). Nothing here reads a
clock or the environment after the config is built.

124 orders the waiting queue in tiers: requests that can still meet their TTFT budget (by the service
estimate below) first, shortest remaining work first; then requests that cannot, also shortest first;
requests LPM holds back for in-batch prefix sharing stay last. It may also park the running continuation
for one round so a rescuable waiter that fits entirely in the round's budget runs instead.
125 lowers the decode rounds armed after each prefill while the cold backlog would take long to clear,
under a guard on how many decoding requests have run above the TPOT gate.
"""

import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Set


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default) or default


@dataclass(frozen=True)
class DeadlineConfig:
    # TTFT budget from what the server can see (a heuristic, not the harness buckets, which the server
    # does not know): a hit below cold_hit_ratio of the prompt counts as a chain start (30 s); otherwise
    # at most fast_tokens uncached as a fast mid-chain request (3 s), else overall mid-chain (5 s).
    cold_budget_s: float = 30.0
    warm_budget_s: float = 5.0
    fast_budget_s: float = 3.0
    fast_tokens: int = 4096
    cold_hit_ratio: float = 0.5
    # Service estimate load * (chunks * fixed + tokens * per_token); defaults measured on run 071 with
    # 8192-token chunks (0.12-0.15 s per chunk, 66-70 us/token, loaded/pure execution p50 1.27). Other
    # chunk sizes or decode cadences need their own values.
    fixed_s: float = 0.13
    per_token_s: float = 68e-6
    load_factor: float = 1.27
    # Receive-to-dispatch time the scheduler does not see (071: t_admit - t_recv about 0.36-0.39 s).
    arrival_offset_s: float = 0.4
    # Starvation bound: a request waiting longer than this goes first whatever its tier (oldest first).
    max_wait_s: float = 120.0
    # Parking: only a continuation with more remaining work than this yields, for at most this many
    # rounds and seconds in a row; 0 rounds disables parking.
    park_min_remaining: int = 65536
    park_max_rounds: int = 8
    park_max_s: float = 2.0


def deadline_config() -> Optional[DeadlineConfig]:
    """SGLANG_AX_DEADLINE_TIERS=1 enables 124; unset or 0 keeps the configured policy."""
    if _env("SGLANG_AX_DEADLINE_TIERS", "0") != "1":
        return None
    cfg = DeadlineConfig(
        cold_budget_s=float(_env("SGLANG_AX_DEADLINE_COLD_S", "30")),
        warm_budget_s=float(_env("SGLANG_AX_DEADLINE_WARM_S", "5")),
        fast_budget_s=float(_env("SGLANG_AX_DEADLINE_FAST_S", "3")),
        fast_tokens=int(_env("SGLANG_AX_DEADLINE_FAST_TOKENS", "4096")),
        cold_hit_ratio=float(_env("SGLANG_AX_DEADLINE_COLD_HIT_RATIO", "0.5")),
        fixed_s=float(_env("SGLANG_AX_DEADLINE_FIXED_S", "0.13")),
        per_token_s=float(_env("SGLANG_AX_DEADLINE_PER_TOKEN_S", "0.000068")),
        load_factor=float(_env("SGLANG_AX_DEADLINE_LOAD", "1.27")),
        arrival_offset_s=float(_env("SGLANG_AX_DEADLINE_ARRIVAL_OFFSET_S", "0.4")),
        max_wait_s=float(_env("SGLANG_AX_DEADLINE_MAX_WAIT_S", "120")),
        park_min_remaining=int(_env("SGLANG_AX_PARK_MIN_REMAINING", "65536")),
        park_max_rounds=int(_env("SGLANG_AX_PARK_MAX_ROUNDS", "8")),
        park_max_s=float(_env("SGLANG_AX_PARK_MAX_S", "2")),
    )
    if not (cfg.cold_budget_s > 0 and cfg.warm_budget_s > 0 and cfg.fast_budget_s > 0
            and cfg.fast_tokens >= 0 and 0 <= cfg.cold_hit_ratio <= 1
            and cfg.per_token_s > 0 and cfg.fixed_s >= 0 and cfg.load_factor > 0
            and cfg.arrival_offset_s >= 0 and cfg.max_wait_s > 0 and cfg.park_min_remaining >= 0
            and cfg.park_max_rounds >= 0 and cfg.park_max_s >= 0):
        raise ValueError(f"[ax] 124: invalid deadline config {cfg}")
    return cfg


def is_cold(req, hit_ratio: float = 0.5) -> bool:
    """A start of a chain or turn, as far as the server can tell: most of the prompt is uncached."""
    return req.num_matched_prefix_tokens < hit_ratio * len(req.origin_input_ids)


def remaining_tokens(req) -> int:
    """Prefill work left, as 123 counts it: prompt + output so far - matched prefix."""
    return len(req.origin_input_ids) + len(req.output_ids) - req.num_matched_prefix_tokens


def budget_s(req, cfg: DeadlineConfig) -> float:
    if is_cold(req, cfg.cold_hit_ratio):
        return cfg.cold_budget_s
    return cfg.fast_budget_s if remaining_tokens(req) <= cfg.fast_tokens else cfg.warm_budget_s


def service_s(tokens: int, chunk: int, cfg: DeadlineConfig) -> float:
    chunks = -(-tokens // chunk) if tokens > 0 else 0
    return cfg.load_factor * (chunks * cfg.fixed_s + tokens * cfg.per_token_s)


def slack_s(req, waited_s: float, chunk: int, cfg: DeadlineConfig) -> float:
    """Seconds left if served alone from now; ignores the queue ahead, so it is optimistic."""
    return (budget_s(req, cfg) - (waited_s + cfg.arrival_offset_s)
            - service_s(remaining_tokens(req), chunk, cfg))


def tier_order(reqs: Sequence, waited_s: Callable[[object], float], held: Set[str], chunk: int,
               cfg: DeadlineConfig) -> List:
    """Starved (oldest first), rescuable and hopeless (each by remaining work), held-back; stable.

    Recomputed every round, so a request misjudged as hopeless is promoted as soon as it is not.
    """

    def key(item):
        index, req = item
        if req.rid in held:
            return (3, 0.0, index)
        waited = waited_s(req)
        if waited > cfg.max_wait_s:
            return (0, -waited, index)
        hopeless = slack_s(req, waited, chunk, cfg) < 0
        return (2 if hopeless else 1, remaining_tokens(req), index)

    return [req for _, req in sorted(enumerate(reqs), key=key)]


def should_park(continuation, continuation_waited_s: float, head, head_waited_s: float,
                round_budget: int, kv_room: int, parked_rounds: int, parked_s: float,
                cfg: DeadlineConfig) -> bool:
    """Yield this round's budget to a rescuable waiter that finishes its prefill within the round.

    The waiter must fit entirely in both the round budget and the KV room (a truncated one would be a
    second partial, which protection refuses, so the round would be wasted); the continuation must be
    long or already hopeless, so short ones are never delayed. Reads nothing but its arguments.
    """
    if head is None or cfg.park_max_rounds == 0 or parked_rounds >= cfg.park_max_rounds:
        return False
    if parked_rounds and parked_s >= cfg.park_max_s:
        return False
    head_work = remaining_tokens(head)
    if head_work > min(round_budget, kv_room) or slack_s(head, head_waited_s, round_budget, cfg) < 0:
        return False
    long_left = remaining_tokens(continuation) > cfg.park_min_remaining
    hopeless = slack_s(continuation, continuation_waited_s, round_budget, cfg) < 0
    return long_left or hopeless


@dataclass(frozen=True)
class BacklogConfig:
    # Enter relief when the cold backlog needs more than high_s at the recent prefill rate, leave
    # below low_s; while relieved, arm relaxed_interval decode rounds after each prefill.
    high_s: float = 30.0
    low_s: float = 10.0
    relaxed_interval: int = 1
    # Guard: stop relieving once max_slow decoding requests, or more than max_slow_ratio of those seen
    # (after min_seen), have run above the TPOT gate. Counted since the last /flush_cache, which the
    # platform calls before every level; the server does not know the level otherwise.
    max_slow: int = 40
    max_slow_ratio: float = 0.03
    min_seen: int = 200
    gate: float = 0.10
    # Smoothing of the prefill rate (weight of the newest sample).
    rate_weight: float = 0.2


def backlog_config(configured_interval: int) -> Optional[BacklogConfig]:
    """SGLANG_AX_BACKLOG_RELIEF=1 enables 125; it needs a fixed decode interval to relax."""
    if _env("SGLANG_AX_BACKLOG_RELIEF", "0") != "1":
        return None
    cfg = BacklogConfig(
        high_s=float(_env("SGLANG_AX_BACKLOG_HIGH_S", "30")),
        low_s=float(_env("SGLANG_AX_BACKLOG_LOW_S", "10")),
        relaxed_interval=int(_env("SGLANG_AX_BACKLOG_INTERVAL", "1")),
        max_slow=int(_env("SGLANG_AX_BACKLOG_MAX_SLOW", "40")),
        max_slow_ratio=float(_env("SGLANG_AX_BACKLOG_MAX_SLOW_RATIO", "0.03")),
        min_seen=int(_env("SGLANG_AX_BACKLOG_MIN_SEEN", "200")),
        gate=float(_env("SGLANG_AX_BACKLOG_GATE", "0.10")),
        rate_weight=float(_env("SGLANG_AX_BACKLOG_RATE_WEIGHT", "0.2")),
    )
    if not (0 <= cfg.low_s < cfg.high_s and 0 <= cfg.relaxed_interval < configured_interval
            and cfg.max_slow >= 0 and 0 <= cfg.max_slow_ratio <= 1 and cfg.min_seen >= 0
            and cfg.gate > 0 and 0 < cfg.rate_weight <= 1):
        raise ValueError(f"[ax] 125: invalid backlog config {cfg} for --prefill-decode-interval "
                         f"{configured_interval} (relief needs a fixed interval above the relaxed one)")
    return cfg


@dataclass
class BacklogState:
    """Rank-0 state of 125; the scheduler broadcasts only the relief decision."""

    cfg: BacklogConfig
    rate: float = 0.0
    relieved: bool = False
    slow: Set[str] = field(default_factory=set)
    seen: Set[str] = field(default_factory=set)

    def reset(self) -> None:
        """At /flush_cache: a new level starts; the prefill rate is kept."""
        self.relieved = False
        self.slow.clear()
        self.seen.clear()

    def note_prefill(self, tokens: int, seconds: float) -> None:
        if tokens <= 0 or seconds <= 0:
            return
        sample = tokens / seconds
        w = self.cfg.rate_weight
        self.rate = sample if self.rate == 0 else (1 - w) * self.rate + w * sample

    def note_tpot(self, rid: str, tpot_s: float) -> None:
        """Mean seconds per token so far of a decoding request; once above the gate it stays counted.

        Conservative: a request slow early may still finish under the gate, but it is counted anyway.
        """
        self.seen.add(rid)
        if tpot_s > self.cfg.gate:
            self.slow.add(rid)

    def guard_tripped(self) -> bool:
        if len(self.slow) >= self.cfg.max_slow:
            return True
        return len(self.seen) >= self.cfg.min_seen and len(self.slow) > self.cfg.max_slow_ratio * len(self.seen)

    def decide(self, backlog_tokens: int) -> bool:
        """Hysteresis on backlog seconds; a tripped guard keeps relief off until the next reset."""
        if self.guard_tripped() or self.rate <= 0:
            self.relieved = False
            return False
        seconds = backlog_tokens / self.rate
        if self.relieved and seconds < self.cfg.low_s:
            self.relieved = False
        elif not self.relieved and seconds > self.cfg.high_s:
            self.relieved = True
        return self.relieved
