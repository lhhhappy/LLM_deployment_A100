"""[ax] 124/125 decisions: deadline-tiered cold admission, chunk-level parking, backlog decode relief.

Pure functions of request state, elapsed seconds and budgets. The scheduler evaluates them on TP rank 0
only and broadcasts the outcome, because arrival stamps, clocks and throughput differ between ranks and
a 1 ms difference is enough to flip an order (evidence/opening-q123-20260925). Nothing here reads a
clock or the environment after the config is built.

124 orders the waiting queue in tiers: requests that can still meet their TTFT budget (by the service
estimate below) first, shortest remaining work first; then requests that cannot, also shortest first;
requests LPM holds back for in-batch prefix sharing stay last. It may also park the running continuation
for one round so a rescuable waiter that fits entirely in the round's budget runs instead.
125 is an opening mode: while the cold backlog would take long to clear (after /flush_cache every level
starts with all users sending cold chain starts at once), it raises the cold chunk cap and/or lowers the
decode rounds armed after each prefill, under a guard on how many decoding requests ran slow since the
last flush.
"""

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple


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
    # 128: rank a request that leads a prefix-sharing family by the family's work per request.
    family: bool = False


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
        family=_env("SGLANG_AX_DEADLINE_FAMILY", "0") == "1",
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
    """Prefill work left of a waiting request, as 123 counts it: prompt + output so far - matched prefix.

    The match is refreshed for waiting requests only; for the running continuation this is still its work
    at admission, so should_park takes the continuation's progress as an argument.
    """
    return len(req.origin_input_ids) + len(req.output_ids) - req.num_matched_prefix_tokens


def budget_s(req, cfg: DeadlineConfig) -> float:
    if is_cold(req, cfg.cold_hit_ratio):
        return cfg.cold_budget_s
    return cfg.fast_budget_s if remaining_tokens(req) <= cfg.fast_tokens else cfg.warm_budget_s


def service_s(tokens: int, chunk: int, cfg: DeadlineConfig) -> float:
    chunks = -(-tokens // chunk) if tokens > 0 else 0
    return cfg.load_factor * (chunks * cfg.fixed_s + tokens * cfg.per_token_s)


def slack_s(req, work: int, waited_s: float, chunk: int, cfg: DeadlineConfig) -> float:
    """Seconds left if its `work` tokens are served alone from now; ignores the queue ahead, so optimistic."""
    return budget_s(req, cfg) - (waited_s + cfg.arrival_offset_s) - service_s(work, chunk, cfg)


def _prefix(req, n: int) -> List[int]:
    ids = req.origin_input_ids
    return ids[:n] if n <= len(ids) else (ids + req.output_ids)[:n]


def _namespace(req) -> Tuple:
    # LPM keys its in-batch tree by (tokens, extra_key, cache_salt); requests differing in either never share.
    return getattr(req, "extra_key", None), getattr(req, "cache_salt", None)


def link_families(reqs: Sequence, shared: Mapping[str, int],
                  cache: Optional[Dict[str, Tuple[str, int]]] = None) -> Dict[str, Tuple[str, int]]:
    """128: map each held-back request to the waiting request whose prompt it shares.

    `shared` holds, for the requests LPM held back for in-batch prefix sharing, how many leading tokens
    they share with a request queued before them. Their leader is the first request not held back whose
    prompt starts with the same tokens in the same cache namespace (extra_key and cache_salt, as LPM's
    RadixKey); any such request computes the shared prefix. Returns
    {held rid: (leader rid, shared tokens)}; held requests without a leader in `reqs` are left out.
    Prompts do not change while waiting, so a link found earlier is reused from `cache` (updated in
    place) while its leader still waits unheld and the shared length is the same.
    """
    leaders = [r for r in reqs if r.rid not in shared]
    leader_ids = {r.rid for r in leaders}
    links = {}
    for r in reqs:
        n = shared.get(r.rid)
        if not n:
            continue
        known = cache.get(r.rid) if cache is not None else None
        if known is not None and known[1] == n and known[0] in leader_ids:
            links[r.rid] = known
            continue
        prefix = _prefix(r, n)
        for leader in leaders:
            if (_namespace(leader) == _namespace(r)
                    and len(leader.origin_input_ids) + len(leader.output_ids) >= n and _prefix(leader, n) == prefix):
                links[r.rid] = (leader.rid, n)
                break
    if cache is not None:
        cache.clear()
        cache.update(links)
    return links


def family_unit_work(reqs: Sequence, links: Mapping[str, Tuple[str, int]]) -> Dict[str, float]:
    """128: prefill work per request of each family: the leader's remaining work plus what each follower
    has left once the shared prefix is cached, divided by the family's size. Leaders without followers are
    left out (their own remaining work applies)."""
    by_rid = {r.rid: r for r in reqs}
    total: Dict[str, float] = {}
    count: Dict[str, int] = {}
    for rid, (leader, n) in links.items():
        if leader not in by_rid or rid not in by_rid:
            continue
        if leader not in total:
            total[leader], count[leader] = float(remaining_tokens(by_rid[leader])), 1
        total[leader] += max(0, remaining_tokens(by_rid[rid]) - n)
        count[leader] += 1
    return {leader: total[leader] / count[leader] for leader in total}


def tier_order(reqs: Sequence, waited_s: Callable[[object], float], held: Set[str], chunk: int,
               cfg: DeadlineConfig, unit_work: Optional[Mapping[str, float]] = None) -> List:
    """Starved (oldest first), rescuable and hopeless (each by remaining work), held-back; stable.

    Recomputed every round, so a request misjudged as hopeless is promoted as soon as it is not.
    With 128, a request leading a prefix-sharing family ranks by `unit_work` (the family's work per
    request) instead of its own work; whether it can still make its budget stays its own.
    """

    def key(item):
        index, req = item
        if req.rid in held:
            return (3, 0.0, index)
        waited = waited_s(req)
        if waited > cfg.max_wait_s:
            return (0, -waited, index)
        work = remaining_tokens(req)
        hopeless = slack_s(req, work, waited, chunk, cfg) < 0
        rank = unit_work.get(req.rid, work) if unit_work else work
        return (2 if hopeless else 1, rank, index)

    return [req for _, req in sorted(enumerate(reqs), key=key)]


def should_park(continuation, continuation_left: int, continuation_waited_s: float, head,
                head_waited_s: float, round_budget: int, kv_room: int, parked_rounds: int, parked_s: float,
                cfg: DeadlineConfig) -> bool:
    """Yield this round's budget to a rescuable waiter that finishes its prefill within the round.

    The waiter must fit entirely in both the round budget and the KV room (a truncated one would be a
    second partial, which protection refuses, so the round would be wasted); the continuation must be
    long or already hopeless, so short ones are never delayed. `continuation_left` is the continuation's
    prefill still to run; its budget stays the one it was admitted with. Reads nothing but its arguments.
    """
    if head is None or cfg.park_max_rounds == 0 or parked_rounds >= cfg.park_max_rounds:
        return False
    if parked_rounds and parked_s >= cfg.park_max_s:
        return False
    head_work = remaining_tokens(head)
    if head_work > min(round_budget, kv_room) or slack_s(head, head_work, head_waited_s, round_budget, cfg) < 0:
        return False
    long_left = continuation_left > cfg.park_min_remaining
    hopeless = slack_s(continuation, continuation_left, continuation_waited_s, round_budget, cfg) < 0
    return long_left or hopeless


@dataclass(frozen=True)
class BacklogConfig:
    # Enter relief when the cold backlog needs more than high_s at the recent prefill rate, leave
    # below low_s. While relieved: the cold chunk cap is cold_cap (0 keeps 120's cap) and
    # relaxed_interval decode rounds are armed after each prefill (the configured interval keeps it).
    high_s: float = 30.0
    low_s: float = 10.0
    relaxed_interval: int = 1
    cold_cap: int = 0
    # Guard: stop relieving once max_slow decoding requests, or more than max_slow_ratio of those seen
    # (after min_seen; 1.0 disables the ratio), have run above `gate` seconds per token. Counted since
    # the last /flush_cache, which the platform calls before every level; the server does not know the
    # level otherwise. The count covers requests still decoding, by their TPOT so far; one that is under
    # the gate now can still end above it, so `gate` defaults below the 0.10 of the harness.
    max_slow: int = 40
    max_slow_ratio: float = 1.0
    min_seen: int = 200
    gate: float = 0.09
    # Smoothing of the prefill rate (weight of the newest sample).
    rate_weight: float = 0.2


def backlog_config(configured_interval: int) -> Optional[BacklogConfig]:
    """SGLANG_AX_BACKLOG_RELIEF=1 enables 125; it must change the cold cap, the interval or both."""
    if _env("SGLANG_AX_BACKLOG_RELIEF", "0") != "1":
        return None
    cfg = BacklogConfig(
        high_s=float(_env("SGLANG_AX_BACKLOG_HIGH_S", "30")),
        low_s=float(_env("SGLANG_AX_BACKLOG_LOW_S", "10")),
        relaxed_interval=int(_env("SGLANG_AX_BACKLOG_INTERVAL", str(configured_interval))),
        cold_cap=int(_env("SGLANG_AX_BACKLOG_COLD_CAP", "0")),
        max_slow=int(_env("SGLANG_AX_BACKLOG_MAX_SLOW", "40")),
        max_slow_ratio=float(_env("SGLANG_AX_BACKLOG_MAX_SLOW_RATIO", "1.0")),
        min_seen=int(_env("SGLANG_AX_BACKLOG_MIN_SEEN", "200")),
        gate=float(_env("SGLANG_AX_BACKLOG_GATE", "0.09")),
        rate_weight=float(_env("SGLANG_AX_BACKLOG_RATE_WEIGHT", "0.2")),
    )
    changes = cfg.cold_cap > 0 or cfg.relaxed_interval < configured_interval
    if not (changes and 0 <= cfg.low_s < cfg.high_s and 0 <= cfg.relaxed_interval <= configured_interval
            and cfg.cold_cap >= 0 and cfg.max_slow >= 0 and 0 <= cfg.max_slow_ratio <= 1
            and cfg.min_seen >= 0 and cfg.gate > 0 and 0 < cfg.rate_weight <= 1):
        raise ValueError(f"[ax] 125: invalid backlog config {cfg} for --prefill-decode-interval "
                         f"{configured_interval} (relief must set SGLANG_AX_BACKLOG_COLD_CAP or an "
                         f"SGLANG_AX_BACKLOG_INTERVAL below the configured interval)")
    return cfg


@dataclass
class BacklogState:
    """Rank-0 state of 125; the scheduler broadcasts only the relief decision."""

    cfg: BacklogConfig
    rate: float = 0.0
    relieved: bool = False
    tripped: bool = False
    slow: Set[str] = field(default_factory=set)
    seen: Set[str] = field(default_factory=set)
    last_batch: Optional[Tuple[int, float]] = None  # (tokens, time) of the last prefill batch scheduled

    def reset(self) -> None:
        """At /flush_cache: a new level starts; only the prefill rate is kept."""
        self.relieved = False
        self.tripped = False
        self.slow.clear()
        self.seen.clear()
        self.last_batch = None

    def note_batch(self, tokens: int, now: float) -> None:
        """A prefill batch of `tokens` is scheduled at `now` (seconds, any monotonic clock).

        The time since the previous prefill batch was scheduled (its execution and the decode rounds after
        it) is charged to that batch's tokens, the ones computed in it.
        """
        if self.last_batch is not None:
            done, since = self.last_batch
            if done > 0 and now > since:
                sample = done / (now - since)
                w = self.cfg.rate_weight
                self.rate = sample if self.rate == 0 else (1 - w) * self.rate + w * sample
        self.last_batch = (tokens, now)

    def note_tpot(self, rid: str, tpot_s: float) -> None:
        """Mean seconds per token so far of a decoding request; once above the gate it stays counted.

        Conservative: a request slow early may still finish under the gate, but it is counted anyway.
        """
        self.seen.add(rid)
        if tpot_s > self.cfg.gate:
            self.slow.add(rid)

    def decide(self, backlog_tokens: int) -> bool:
        """Hysteresis on backlog seconds; a tripped guard keeps relief off until the next reset.

        The guard latches, so a slow ratio that falls back under the limit as more requests are seen does
        not re-enable relief.
        """
        slow, seen = len(self.slow), len(self.seen)
        self.tripped = self.tripped or slow >= self.cfg.max_slow or (
            seen >= self.cfg.min_seen and slow > self.cfg.max_slow_ratio * seen)
        if self.tripped or self.rate <= 0:
            self.relieved = False
            return False
        seconds = backlog_tokens / self.rate
        if self.relieved and seconds < self.cfg.low_s:
            self.relieved = False
        elif not self.relieved and seconds > self.cfg.high_s:
            self.relieved = True
        return self.relieved
