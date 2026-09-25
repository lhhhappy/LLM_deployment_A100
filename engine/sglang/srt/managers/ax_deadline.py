"""[ax] 124 decisions: deadline-tiered cold admission and chunk-level parking.

Pure functions of request state, elapsed seconds and budgets. The scheduler evaluates them on TP rank 0
only and broadcasts the outcome, because arrival stamps, clocks and throughput differ between ranks and
a 1 ms difference is enough to flip an order (evidence/opening-q123-20260925). Nothing here reads a
clock or the environment after the config is built.

124 orders the waiting queue in tiers: requests that can still meet their TTFT budget (by the service
estimate below) first, shortest remaining work first; then requests that cannot, also shortest first;
requests LPM holds back for in-batch prefix sharing stay last. It may also park the running continuation
for one round so a rescuable waiter that fits entirely in the round's budget runs instead.
"""

import os
from dataclasses import dataclass
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
