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
from array import array
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple


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
    # Same bound for warm requests (most of the prompt cached: turn starts and mid-chain requests, gates 3/5/15 s).
    # Run 112: warm turn starts judged hopeless after their 5 s budget waited the full 120 s behind rescuable cold
    # heads (turn misses 12 -> 24). A lower bound frees them in time without letting them outrank cold heads while
    # they can still make it. Defaults to max_wait_s (no change unless set).
    max_wait_warm_s: float = 120.0
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
        max_wait_warm_s=float(_env("SGLANG_AX_DEADLINE_MAX_WAIT_WARM_S", _env("SGLANG_AX_DEADLINE_MAX_WAIT_S", "120"))),
        park_min_remaining=int(_env("SGLANG_AX_PARK_MIN_REMAINING", "65536")),
        park_max_rounds=int(_env("SGLANG_AX_PARK_MAX_ROUNDS", "8")),
        park_max_s=float(_env("SGLANG_AX_PARK_MAX_S", "2")),
    )
    if not (cfg.cold_budget_s > 0 and cfg.warm_budget_s > 0 and cfg.fast_budget_s > 0
            and cfg.fast_tokens >= 0 and 0 <= cfg.cold_hit_ratio <= 1
            and cfg.per_token_s > 0 and cfg.fixed_s >= 0 and cfg.load_factor > 0
            and cfg.arrival_offset_s >= 0 and cfg.max_wait_s > 0 and cfg.max_wait_warm_s > 0 and cfg.park_min_remaining >= 0
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


def tier_order(reqs: Sequence, waited_s: Callable[[object], float], held: Set[str], chunk: int,
               cfg: DeadlineConfig, work: Optional[Dict[str, int]] = None,
               family_held: Optional[Set[str]] = None) -> List:
    """Starved (oldest first), rescuable and hopeless (each by remaining work), held-back; stable.

    Recomputed every round, so a request misjudged as hopeless is promoted as soon as it is not. `held` are LPM's
    in-batch prefix-sharing holdbacks, last as under 124 whatever they waited. `work` and `family_held` belong to
    128: `work` overrides the ranking work of a family leader (the rescuable/hopeless judgement keeps its real
    remaining); a rider in `family_held` is last while its leader waits, except that the starvation bound still
    frees it, so a leader that never runs cannot keep its riders last for ever. With 128 off both are empty and
    the order equals 124's.
    """
    family_held = family_held or set()

    def key(item):
        index, req = item
        if req.rid in held:
            return (3, 0.0, index)
        waited = waited_s(req)
        if waited > (cfg.max_wait_s if is_cold(req, cfg.cold_hit_ratio) else cfg.max_wait_warm_s):
            return (0, -waited, index)
        if req.rid in family_held:
            return (3, 0.0, index)
        remaining = remaining_tokens(req)
        hopeless = slack_s(req, remaining, waited, chunk, cfg) < 0
        return (2 if hopeless else 1, work.get(req.rid, remaining) if work else remaining, index)

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


# --------------------------------------------------------------------------------------------- 128
@dataclass(frozen=True)
class FamilyConfig:
    """128: waiting cold requests that share an uncached prefix form a family; the leader ranks by work per rider.

    Run 104/109 (v3 N26 opening): four chain starts of 35-37k tokens shared 32k; their leader ranked by its own
    36k behind smaller heads and finished at 29.7 s, so all four siblings (3k of their own work each) missed 30 s.
    """
    block: int = 256        # hash block in tokens (the checkpoint grid: a shared prefix is reusable per whole block)
    link_min: int = 4096    # uncached shared prefix that links two requests (tokens)
    rider_max: int = 6144   # a member finishes as a short hit after the leader when its own tail is at most this
    max_candidates: int = 64  # more waiting cold requests than this: skip the pairwise scan this round (CPU bound)


def family_config() -> Optional[FamilyConfig]:
    """SGLANG_AX_DEADLINE_FAMILY=1 enables 128 inside 124; unset or 0 keeps 124's per-request order."""
    if _env("SGLANG_AX_DEADLINE_FAMILY", "0") != "1":
        return None
    cfg = FamilyConfig(block=int(_env("SGLANG_AX_FAMILY_BLOCK", "256")),
                       link_min=int(_env("SGLANG_AX_FAMILY_LINK_MIN", "4096")),
                       rider_max=int(_env("SGLANG_AX_FAMILY_RIDER_MAX", "6144")),
                       max_candidates=int(_env("SGLANG_AX_FAMILY_MAX_CANDIDATES", "64")))
    if cfg.block <= 0 or cfg.link_min <= 0 or cfg.rider_max <= 0 or cfg.max_candidates <= 0:
        raise ValueError(f"[ax] 128: invalid family config {cfg}")
    return cfg


def block_hashes(ids: Sequence[int], block: int) -> Tuple[int, ...]:
    """One hash per complete block of `block` prompt tokens; a tail shorter than a block is ignored."""
    arr = ids if isinstance(ids, array) else array("q", ids)
    return tuple(hash(arr[i:i + block].tobytes()) for i in range(0, len(arr) - block + 1, block))


def shared_prefix_blocks(a: Sequence[int], b: Sequence[int]) -> int:
    """Number of leading equal blocks of two hash sequences."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def family_plan(reqs: Sequence, shared_tokens: Callable[[object, object], int], cfg: FamilyConfig):
    """Families among waiting cold requests and the ranking work of their leaders.

    `shared_tokens(a, b)` is the shared prompt prefix of two requests in tokens (block hashes, cached by the caller).
    Two requests are linked when that prefix minus the longer cache match of the two is at least link_min: the part
    the first of them computes and the other reuses. The leader of a family is the member with the least remaining
    work (ties by rid); a rider is another member whose remaining after the leader's prefix is at most rider_max,
    so it finishes as a short hit beside the next cold chunk; a member with a longer tail neither holds nor counts.
    Returns (work: leader rid -> remaining // (1 + riders), held: rider rids, families: [(leader, riders, others)]).
    Riders are held only while their leader waits (this plan sees waiting requests only): once the leader runs, its
    chunks enter the tree and the riders' matches grow until they are short hits.
    """
    n = len(reqs)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            matched = max(reqs[i].num_matched_prefix_tokens, reqs[j].num_matched_prefix_tokens)
            if shared_tokens(reqs[i], reqs[j]) - matched >= cfg.link_min:
                parent[find(i)] = find(j)
    groups: Dict[int, list] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(reqs[i])
    work: Dict[str, int] = {}
    held: Set[str] = set()
    families = []
    for members in groups.values():
        if len(members) < 2:
            continue
        leader = min(members, key=lambda r: (remaining_tokens(r), r.rid))
        riders, others = [], []
        for m in members:
            if m is leader:
                continue
            tail = (len(m.origin_input_ids) + len(m.output_ids)
                    - max(m.num_matched_prefix_tokens, shared_tokens(leader, m)))
            (riders if tail <= cfg.rider_max else others).append(m)
        if riders:
            work[leader.rid] = max(1, remaining_tokens(leader) // (1 + len(riders)))
            held.update(m.rid for m in riders)
        families.append((leader.rid, [m.rid for m in riders], [m.rid for m in others]))
    return work, held, families
