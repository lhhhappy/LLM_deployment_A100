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
under a guard on how many finished requests exceeded the TPOT gate.
"""

import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Set


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default) or default


@dataclass(frozen=True)
class DeadlineConfig:
    # TTFT budget by what the server can see: a hit below cold_hit_ratio of the prompt counts as a
    # chain/turn start (30 s gate), anything else as a mid-chain request (5 s overall gate).
    cold_budget_s: float = 30.0
    warm_budget_s: float = 5.0
    cold_hit_ratio: float = 0.5
    # Service estimate load * (fixed + tokens * per_token); defaults measured on run 071 (8192-token
    # chunks: 0.12-0.15 s fixed, 66-70 us/token, loaded/pure execution p50 1.27).
    fixed_s: float = 0.13
    per_token_s: float = 68e-6
    load_factor: float = 1.27
    # Receive-to-dispatch time the scheduler does not see (071: t_admit - t_recv about 0.36-0.39 s).
    arrival_offset_s: float = 0.4
    # Parking: only a continuation with more remaining work than this yields, at most this many rounds
    # in a row; 0 rounds disables parking.
    park_min_remaining: int = 65536
    park_max_rounds: int = 8


def deadline_config() -> Optional[DeadlineConfig]:
    """SGLANG_AX_DEADLINE_TIERS=1 enables 124; unset or 0 keeps the configured policy."""
    if _env("SGLANG_AX_DEADLINE_TIERS", "0") != "1":
        return None
    cfg = DeadlineConfig(
        cold_budget_s=float(_env("SGLANG_AX_DEADLINE_COLD_S", "30")),
        warm_budget_s=float(_env("SGLANG_AX_DEADLINE_WARM_S", "5")),
        cold_hit_ratio=float(_env("SGLANG_AX_DEADLINE_COLD_HIT_RATIO", "0.5")),
        fixed_s=float(_env("SGLANG_AX_DEADLINE_FIXED_S", "0.13")),
        per_token_s=float(_env("SGLANG_AX_DEADLINE_PER_TOKEN_S", "0.000068")),
        load_factor=float(_env("SGLANG_AX_DEADLINE_LOAD", "1.27")),
        arrival_offset_s=float(_env("SGLANG_AX_DEADLINE_ARRIVAL_OFFSET_S", "0.4")),
        park_min_remaining=int(_env("SGLANG_AX_PARK_MIN_REMAINING", "65536")),
        park_max_rounds=int(_env("SGLANG_AX_PARK_MAX_ROUNDS", "8")),
    )
    if not (cfg.cold_budget_s > 0 and cfg.warm_budget_s > 0 and 0 <= cfg.cold_hit_ratio <= 1
            and cfg.per_token_s > 0 and cfg.fixed_s >= 0 and cfg.load_factor > 0
            and cfg.arrival_offset_s >= 0 and cfg.park_min_remaining >= 0 and cfg.park_max_rounds >= 0):
        raise ValueError(f"[ax] 124: invalid deadline config {cfg}")
    return cfg


def remaining_tokens(req) -> int:
    """Prefill work left, as 123 counts it: prompt + output so far - matched prefix."""
    return len(req.origin_input_ids) + len(req.output_ids) - req.num_matched_prefix_tokens


def budget_s(req, cfg: DeadlineConfig) -> float:
    prompt = len(req.origin_input_ids)
    cold = req.num_matched_prefix_tokens < cfg.cold_hit_ratio * prompt
    return cfg.cold_budget_s if cold else cfg.warm_budget_s


def service_s(tokens: int, cfg: DeadlineConfig) -> float:
    return cfg.load_factor * (cfg.fixed_s + tokens * cfg.per_token_s)


def slack_s(req, waited_s: float, cfg: DeadlineConfig) -> float:
    """Seconds left if served alone from now; ignores the queue ahead, so it is optimistic."""
    return budget_s(req, cfg) - (waited_s + cfg.arrival_offset_s) - service_s(remaining_tokens(req), cfg)


def tier_order(reqs: Sequence, waited_s: Callable[[object], float], held: Set[str],
               cfg: DeadlineConfig) -> List:
    """Rescuable by remaining work, then hopeless by remaining work, then held-back; stable."""

    def key(item):
        index, req = item
        if req.rid in held:
            return (2, 0, index)
        hopeless = slack_s(req, waited_s(req), cfg) < 0
        return (1 if hopeless else 0, remaining_tokens(req), index)

    return [req for _, req in sorted(enumerate(reqs), key=key)]


def should_park(continuation, continuation_waited_s: float, head, head_waited_s: float,
                round_budget: int, parked_rounds: int, cfg: DeadlineConfig) -> bool:
    """Yield this round's budget to a rescuable waiter that finishes its prefill within the round.

    The waiter must fit entirely (it would otherwise become a second partial, which protection
    forbids); the continuation must be long or already hopeless, so short ones are never delayed.
    """
    if head is None or cfg.park_max_rounds == 0 or parked_rounds >= cfg.park_max_rounds:
        return False
    head_work = remaining_tokens(head)
    if head_work > round_budget or slack_s(head, head_waited_s, cfg) < 0:
        return False
    long_left = remaining_tokens(continuation) > cfg.park_min_remaining
    hopeless = slack_s(continuation, continuation_waited_s, cfg) < 0
    return long_left or hopeless


@dataclass(frozen=True)
class BacklogConfig:
    # Enter relief when the cold backlog needs more than high_s at the recent prefill rate, leave
    # below low_s; while relieved, arm relaxed_interval decode rounds after each prefill.
    high_s: float = 30.0
    low_s: float = 10.0
    relaxed_interval: int = 1
    # Guard: stop relieving once this many finished requests had a server-side TPOT above gate.
    max_slow: int = 40
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
        gate=float(_env("SGLANG_AX_BACKLOG_GATE", "0.10")),
        rate_weight=float(_env("SGLANG_AX_BACKLOG_RATE_WEIGHT", "0.2")),
    )
    if not (0 <= cfg.low_s < cfg.high_s and 0 <= cfg.relaxed_interval < configured_interval
            and cfg.max_slow >= 0 and cfg.gate > 0 and 0 < cfg.rate_weight <= 1):
        raise ValueError(f"[ax] 125: invalid backlog config {cfg} for --prefill-decode-interval "
                         f"{configured_interval} (relief needs a fixed interval above the relaxed one)")
    return cfg


@dataclass
class BacklogState:
    """Rank-0 state of 125; the scheduler broadcasts only `relieved`."""

    cfg: BacklogConfig
    rate: float = 0.0
    relieved: bool = False
    slow: int = 0
    finished: int = 0

    def note_prefill(self, tokens: int, seconds: float) -> None:
        if tokens <= 0 or seconds <= 0:
            return
        sample = tokens / seconds
        w = self.cfg.rate_weight
        self.rate = sample if self.rate == 0 else (1 - w) * self.rate + w * sample

    def note_finished(self, tpot_s: Optional[float]) -> None:
        self.finished += 1
        if tpot_s is not None and tpot_s > self.cfg.gate:
            self.slow += 1

    def decide(self, backlog_tokens: int) -> bool:
        """Hysteresis on backlog seconds; the guard switches relief off for good once tripped."""
        if self.slow >= self.cfg.max_slow or self.rate <= 0:
            self.relieved = False
            return False
        seconds = backlog_tokens / self.rate
        if self.relieved and seconds < self.cfg.low_s:
            self.relieved = False
        elif not self.relieved and seconds > self.cfg.high_s:
            self.relieved = True
        return self.relieved
