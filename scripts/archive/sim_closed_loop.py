#!/usr/bin/env python3
"""CPU-only closed-loop contest replay model; every result is a MODEL OUTPUT.

No engine, tokenizer, GPU, network, or submission calls. Read the frozen dev
harness, never write there (including bytecode). See R9_offline_simulator.md.

Example sensitivity sweep (per-request records are written unless --summary-only):
  python3 scripts/sim_closed_loop.py --prefill-rates 20000,40000,80000 \
      --decode-ms 30,50,80 --mix both --out-dir evidence/T15_simulator

Optional cache JSONL: one row per ORIGINAL req_id with integer `stock` and
`role_conservative` uncached token counts. This accepts exported F24 policy
results or measured cache counts converted to this schema. No silent fallback.
Optional output JSONL: one row per ORIGINAL req_id with `output_tokens`.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import heapq
import itertools
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HARNESS = REPO / "s1-dev/harness"
_bytecode = sys.dont_write_bytecode
sys.dont_write_bytecode = True
sys.path.insert(0, str(HARNESS))
try:
    from s1_common import (GATES, TTFT_GATE_SPECS, in_ttft_gate, load_index,
                           phase_gate, q)
    from s1_loadgen import build_gap_plan
    from s1_score import evaluate
finally:
    sys.dont_write_bytecode = _bytecode

LABEL = "MODEL OUTPUT — uncalibrated offline simulation; not an official capacity"
WHAT_IF = "WHAT-IF ONLY — synthetic formal composition, not a formal prediction"
POLICIES = ("stock", "role_conservative")
SLO_SCHEDULERS = ("edf", "least-slack", "edf-chain-weighted")
SCHEDULERS = ("fcfs", "spf", "spf-upstream", "hrrn", "lpm", *SLO_SCHEDULERS)
TPOT_LIMIT = .10
F24_FAST_FACTOR = 7238 / 3144  # F24 stock / frozen fast p95 (F13).
F24_SLOW_FACTOR = 15744 / 14033  # F24 overall / frozen overall p95 (F13).


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                              allow_nan=False) + "\n")
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}: {value!r}")
    return value


@dataclass
class Workload:
    rows: dict
    chains: list
    gaps: dict
    metadata: dict


def load_workload(root, cohort_path, cap_s=3600.0, pacing="auto"):
    """Use harness filtering, index assignment and exact stored cohort ordering."""
    root, cohort_path = Path(root), Path(cohort_path)
    if not math.isfinite(cap_s) or cap_s < 0:
        raise ValueError("gap cap must be finite and nonnegative")
    if pacing == "auto":
        pacing = "chain-total-gap-scaled-v1" if cap_s > 0 else "replay_gap"
    if pacing not in ("replay_gap", "chain-total-gap-scaled-v1"):
        raise ValueError("unsupported pacing")
    if pacing == "chain-total-gap-scaled-v1" and cap_s <= 0:
        raise ValueError("scaled pacing needs a positive cap")
    cap_ms = int(round(cap_s * 1000)) if pacing == "chain-total-gap-scaled-v1" else 0
    all_rows, chain_index, _ = load_index(str(root))
    cohort = json.loads(cohort_path.read_text())
    chains, rows, seen_chains = [], {}, set()
    for ch in cohort["chains"]:
        cid, ids = ch["chain_id"], ch["req_ids"]
        if cid in seen_chains or cid not in chain_index or not ids:
            raise ValueError(f"duplicate, missing or empty chain: {cid}")
        seen_chains.add(cid)
        last_index = -1
        for rid in ids:
            if rid in rows or rid not in all_rows or all_rows[rid]["chain_id"] != cid:
                raise ValueError(f"duplicate, unavailable or wrong-chain request: {rid}")
            row = dict(all_rows[rid])
            if row["_idx_in_chain"] <= last_index:
                raise ValueError(f"cohort is not in chain replay order: {cid}")
            last_index = row["_idx_in_chain"]
            integer(row["glm_tokens"], "glm_tokens", 1)
            integer(row["uncached_expected"], "uncached_expected")
            if row["uncached_expected"] > row["glm_tokens"]:
                raise ValueError(f"uncached exceeds prompt: {rid}")
            row["source_req_id"] = rid
            rows[rid] = row
        chains.append({"chain_id": cid, "req_ids": list(ids)})
    if not chains:
        raise ValueError("empty cohort")
    gaps, stats = build_gap_plan(chains, rows, cap_ms)
    if any(g < 0 for gs in gaps.values() for g in gs):
        raise ValueError("harness gap plan contains a negative gap")
    files = [root / "requests.jsonl", root / "chains.jsonl", cohort_path,
             HARNESS / "s1_common.py", HARNESS / "s1_loadgen.py", HARNESS / "s1_score.py"]
    return Workload(rows, chains, gaps, {
        "label": LABEL, "mix": "dev", "pacing": pacing, "gap_stats": stats,
        "cohort_sha256": cohort.get("cohort_sha256"),
        "input_sha256": {str(p): sha256(p) for p in files},
        "n_chains": len(chains), "n_requests": len(rows),
        "composition": composition(rows.values()),
        "formal_reference": cohort.get("composition_alignment_measured", {}).get(
            "formal_set_reference", {}),
        "note": "indices come from load_index, not re-enumerated cohort subsets",
    })


def category(row):
    return {30.0: "chain_start", 15.0: "turn_start", 3.0: "intra"}[phase_gate(row)]


def composition(rows):
    return dict(collections.Counter(category(r) for r in rows))


def formal_mix(workload, seed=20260922, counts=None):
    """Exact composition; stratified balanced resampling, synthetic longer chains.

    Cycle through shuffled dev templates in each gate, then shuffle the sampled
    continuations and distribute round-robin. Cache work is inherited from the
    template, NOT recomputed from fictitious adjacent prompts. This is a load
    composition experiment; neither a hidden workload reconstruction nor a KV
    residency model. Templates may repeat, but replay instance IDs are unique.
    """
    if counts is None:
        ref = workload.metadata["formal_reference"]
        counts = {k: ref[k]["n"] for k in ("chain_start", "intra", "turn_start")}
    for k in ("chain_start", "intra", "turn_start"):
        integer(counts[k], k, 1 if k == "chain_start" else 0)
    rng = random.Random(seed)
    pools = {k: [r for r in workload.rows.values() if category(r) == k] for k in counts}

    def sample(k):
        if counts[k] and not pools[k]:
            raise ValueError(f"cannot synthesize absent stratum: {k}")
        out = []
        while len(out) < counts[k]:
            cycle = list(pools[k])
            rng.shuffle(cycle)
            out.extend(cycle[:counts[k] - len(out)])
        return out

    heads = sample("chain_start")
    continuations = sample("intra") + sample("turn_start")
    rng.shuffle(continuations)
    sequences = [[r] for r in heads]
    for i, row in enumerate(continuations):
        sequences[i % len(sequences)].append(row)
    rows, chains = {}, []
    for c, seq in enumerate(sequences):
        cid = f"whatif-chain-{c:04d}"
        ids = []
        for i, source in enumerate(seq):
            rid = f"{cid}:request-{i:04d}"
            row = dict(source, _req_id=rid, chain_id=cid, _idx_in_chain=i,
                       source_chain_id=source["chain_id"], sampling_weight=1.0)
            rows[rid] = row
            ids.append(rid)
        chains.append({"chain_id": cid, "req_ids": ids})
    gaps, stats = build_gap_plan(chains, rows,
                                 workload.metadata["gap_stats"]["chain_gap_cap_ms"])
    meta = dict(workload.metadata, label=WHAT_IF, mix="formal-mix", seed=seed,
                composition=composition(rows.values()), n_chains=len(chains),
                n_requests=len(rows), gap_stats=stats,
                note="Synthetic template chains; inherited gaps/cache work; no semantic cache path or eviction model")
    return Workload(rows, chains, gaps, meta)


def read_profiles(path, fields):
    profiles = {}
    if path:
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                rid = row["req_id"]
                if rid in profiles:
                    raise ValueError(f"duplicate profile: {rid}")
                for field in fields:
                    integer(row[field], field, 1 if field == "output_tokens" else 0)
                profiles[rid] = row
    return profiles


def prepare_profiles(workload, cache_profiles=None, output_profiles=None,
                     output_mode="budget", output_scale=1.0,
                     stock_fast_factor=F24_FAST_FACTOR, stock_slow_factor=F24_SLOW_FACTOR):
    """Default cache counts are explicit frozen-field proxies, not F24 row outputs.

    Stock inflation is applied within frozen fast and non-fast intra strata only,
    clipped to prompt length. Cold/turn-start work equals frozen expected. The
    F24 aggregate percentiles cannot identify per-request misses or correlations.
    """
    if output_mode not in ("budget", "source"):
        raise ValueError("unknown output mode")
    if not math.isfinite(output_scale) or output_scale <= 0:
        raise ValueError("output scale must be positive")
    if any(not math.isfinite(f) or f < 1 for f in (stock_fast_factor, stock_slow_factor)):
        raise ValueError("stock inflation factors must be finite and >= 1")
    profiles, fallback = {}, 0
    for rid, row in workload.rows.items():
        source = row.get("source_req_id", rid)
        expected, prompt = row["uncached_expected"], row["glm_tokens"]
        if cache_profiles is not None:
            if source not in cache_profiles:
                raise ValueError(f"missing cache profile: {source}")
            counts = {p: integer(cache_profiles[source][p], p) for p in POLICIES}
            if any(u > prompt for u in counts.values()):
                raise ValueError(f"cache profile exceeds prompt: {source}")
        else:
            factor = (stock_fast_factor if in_ttft_gate(row, "fast_intra") else
                      stock_slow_factor if in_ttft_gate(row, "overall_intra") else 1.0)
            # Avoid a floating-point 15744.000000000002 becoming 15745.
            counts = {"stock": min(prompt, math.ceil(expected * factor - 1e-9)),
                      "role_conservative": expected}
        budget = integer(int(row.get("max_output_i") or 512), "max_output_i", 1)
        if output_profiles is not None:
            if source not in output_profiles:
                raise ValueError(f"missing output profile: {source}")
            output = integer(output_profiles[source]["output_tokens"], "output_tokens", 1)
            if output > budget:
                raise ValueError(f"output profile exceeds budget: {source}")
        elif output_mode == "source" and row.get("source_completion_tokens"):
            output = min(budget, integer(row["source_completion_tokens"], "source output", 1))
        else:
            output = budget
            fallback += int(output_mode == "source")
        output = min(budget, max(1, math.ceil(output * output_scale)))
        profiles[rid] = dict(counts, output_tokens=output)
    meta = {
        "cache_model": "per-request supplied counts" if cache_profiles is not None else
            "frozen proxy: D1=uncached_expected; F24-tail-ratio stock inflation (not exact F24 replay)",
        "stock_fast_factor": stock_fast_factor, "stock_nonfast_intra_factor": stock_slow_factor,
        "output_model": "per-request supplied counts" if output_profiles is not None else output_mode,
        "output_scale": output_scale, "source_output_budget_fallbacks": fallback,
        "uncached_p95": {p: {sel: q([profiles[rid][p] for rid, r in workload.rows.items()
                                            if in_ttft_gate(r, sel)], .95)
                              for _, sel, _ in TTFT_GATE_SPECS} for p in POLICIES},
        "sum_output_tokens": sum(p["output_tokens"] for p in profiles.values()),
    }
    return profiles, meta


@dataclass(frozen=True)
class Engine:
    prefill_tps: float = 40000.0
    chunk_tokens: int = 8192
    page_tokens: int = 64
    scheduler: str = "fcfs"
    prefill_length_alpha: float = 0.0
    prefill_length_reference: float = 32768.0
    decode_ms: float = 50.0
    decode_batch_slope: float = .02
    decode_speedup: float = 1.0  # effective compute speedup; not a speculative acceptance model.
    decode_curve: tuple = ()  # (batch_size, ms), piecewise linear; clamp endpoints.
    forward_overhead_ms: float = 2.0
    frontend_ms: float = 0.0
    prefill_decode_interval: int = 0
    max_running: int = 0  # zero = logical N; excludes slot workers sleeping in gaps.
    d1_extra_forward_equivalents: float = 0.0  # overhead-only sensitivity, not boundary placement.

    chain_start_weight: float = 2.0  # used only by edf-chain-weighted; targets stay 30s.

    def __post_init__(self):
        for key in ("prefill_tps", "prefill_length_reference", "decode_ms", "decode_speedup", "chain_start_weight"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive and finite")
        for key in ("prefill_length_alpha", "decode_batch_slope", "forward_overhead_ms",
                    "frontend_ms", "d1_extra_forward_equivalents"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be nonnegative and finite")
        for key in ("chunk_tokens", "page_tokens"):
            integer(getattr(self, key), key, 1)
        for key in ("prefill_decode_interval", "max_running"):
            integer(getattr(self, key), key)
        if self.chunk_tokens % self.page_tokens or self.scheduler not in SCHEDULERS:
            raise ValueError(f"chunk must be page-aligned and scheduler one of {SCHEDULERS}")
        if self.chain_start_weight < 1:
            raise ValueError("chain_start_weight must be >= 1")
        last = 0
        for b, ms in self.decode_curve:
            integer(b, "decode curve batch", 1)
            if b <= last or not math.isfinite(ms) or ms <= 0:
                raise ValueError("decode curve needs increasing batches and positive finite times")
            last = b

    def prefill_seconds(self, tokens, prompt):
        penalty = max(1.0, prompt / self.prefill_length_reference) ** self.prefill_length_alpha
        return tokens * penalty / self.prefill_tps

    def decode_seconds(self, batch):
        if self.decode_curve:
            ms = self.decode_curve[-1][1]
            if batch <= self.decode_curve[0][0]:
                ms = self.decode_curve[0][1]
            else:
                for (a, x), (b, y) in zip(self.decode_curve, self.decode_curve[1:]):
                    if batch <= b:
                        ms = x + (y - x) * (batch - a) / (b - a)
                        break
        else:
            ms = self.decode_ms * (1 + self.decode_batch_slope * (batch - 1))
        return (ms / self.decode_speedup + self.forward_overhead_ms) / 1000


@dataclass(eq=False)
class Request:
    rid: str
    worker: int
    chain: int
    index: int
    dispatch: float
    ready: float
    order: int
    remaining: int
    output_left: int
    uncached: int
    output: int
    start: float | None = None
    first: float | None = None
    prefill_forwards: int = 0
    matched: int = 0
    arrival_processed_tokens: int = 0


def inferred_slo_limit(req):
    """Server-visible session ordinal + observed cache miss; NO frozen phase/label.

    There is no reliable turn-start signal in these two inputs. Treat all known
    continuations conservatively as intra. Unknown/rebuilt sessions are a live
    classification limitation; simulated chains provide reliable session IDs.
    """
    return 30.0 if req.index == 0 else (3.0 if req.uncached <= 4096 else 5.0)


def slo_priority(req, engine, now=0.0):
    limit = inferred_slo_limit(req)
    weight = engine.chain_start_weight if (
        engine.scheduler == "edf-chain-weighted" and req.index == 0) else 1.0
    deadline = req.dispatch + limit * weight
    service = 0.0
    if engine.scheduler == "least-slack":
        service = (engine.prefill_seconds(req.remaining, req.matched + req.uncached)
                   + engine.forward_overhead_ms / 1000)
    return (deadline - now - service, req.dispatch, req.order)


def select_prefill(waiting, partial, decode_count, engine, max_running, processed_tokens=0,
                   now=0.0):
    """Return [(request, real work tokens)]; at most one partial after a forward.

    FCFS gives continuation priority. SPF reserves page-rounded *complete*
    shorter waiters while guaranteeing the continuation at least one page.
    Legacy spf admits waiters before continuation and factors in request slots.
    spf-upstream models #40024 reservation then continuation then admission.
    Neither models live cache matching, host misses, or KV allocator capacity.
    Mutates waiting and remaining, only at a non-preemptible forward boundary.
    """
    budget = engine.chunk_tokens
    allocations = []
    running = decode_count + int(partial is not None)
    scheduler = engine.scheduler
    if scheduler in ("hrrn", "lpm") and len(waiting) > 128:
        scheduler = "fcfs"  # v0.5.20 expensive-cache-policy fallback.
    if scheduler in ("spf", "spf-upstream"):
        ordered = sorted(waiting, key=lambda r: (r.remaining, r.order))
    elif scheduler in SLO_SCHEDULERS:
        ordered = sorted(waiting, key=lambda r: slo_priority(r, engine, now))
    elif scheduler == "hrrn":
        ordered = sorted(waiting, key=lambda r: (
            -math.inf if r.uncached == 0 else
            -max(0, processed_tokens - r.arrival_processed_tokens) / r.uncached,
            r.rid))
    elif scheduler == "lpm":
        ordered = sorted(waiting, key=lambda r: -r.matched)  # stable ties, like stock.
    else:
        ordered = sorted(waiting, key=lambda r: r.order)
    if scheduler in ("hrrn", "lpm"):
        waiting[:] = ordered  # keep stable LPM ties across scheduling rounds.

    def allocate(req, tokens):
        nonlocal budget
        charge = math.ceil(tokens / engine.page_tokens) * engine.page_tokens
        req.remaining -= tokens
        budget -= charge
        allocations.append((req, tokens))

    if scheduler == "spf-upstream" or scheduler in SLO_SCHEDULERS:
        # Match #40024's two-stage process: reserve WITHOUT looking at free
        # request slots, run the continuation first, then admit waiters in order.
        # A failed admission does not redistribute the continuation's budget.
        active = partial
        if partial is not None:
            reserved = 0
            for req in ordered:
                charge = math.ceil(req.remaining / engine.page_tokens) * engine.page_tokens
                more_urgent = (slo_priority(req, engine, now) < slo_priority(partial, engine, now)
                               if scheduler in SLO_SCHEDULERS else req.remaining < partial.remaining)
                if not more_urgent or reserved + charge > budget - engine.page_tokens:
                    break
                reserved += charge
            limit = (budget - reserved) // engine.page_tokens * engine.page_tokens
            allocate(partial, min(partial.remaining, limit))
            if not partial.remaining:
                active = None
        for req in ordered:
            if running >= max_running or budget < engine.page_tokens:
                break
            charge = math.ceil(req.remaining / engine.page_tokens) * engine.page_tokens
            if active is not None and charge > budget:
                break  # sole partial slot is held by the continuation.
            allocate(req, min(req.remaining, budget))
            waiting.remove(req)
            running += 1
            if req.remaining:
                active = req
                break
        return allocations, active

    if partial is not None:
        if scheduler == "spf":
            for req in list(ordered):
                charge = math.ceil(req.remaining / engine.page_tokens) * engine.page_tokens
                if (req.remaining >= partial.remaining or running >= max_running
                        or charge > budget - engine.page_tokens):
                    break
                allocate(req, req.remaining)
                running += 1
                waiting.remove(req)
                ordered.remove(req)
        allocate(partial, min(partial.remaining, budget))
        if partial.remaining:
            return allocations, partial
    for req in ordered:
        if running >= max_running or budget < engine.page_tokens:
            break
        allocate(req, min(req.remaining, budget))
        waiting.remove(req)
        running += 1
        if req.remaining:
            return allocations, req
    return allocations, None


def simulate(workload, profiles, n, engine, policy="stock", keep_trace=False,
             coalesce_decode=True):
    integer(n, "N", 1)
    if policy not in POLICIES:
        raise ValueError("unknown cache policy")
    max_running = engine.max_running or n
    events, waiting, decoding, records, trace = [], [], [], [], []
    counter = itertools.count()
    next_chain, now, partial, forced_decode = 0, 0.0, None, 0
    stats = collections.Counter()
    decode_batches = collections.Counter()
    peak_waiting = peak_running = 0

    def schedule(worker, ci, ri, finished):
        nonlocal next_chain
        if ci is None or ri == len(workload.chains[ci]["req_ids"]):
            if next_chain == len(workload.chains):
                return
            ci, ri = next_chain, 0
            next_chain += 1
        ch = workload.chains[ci]
        rid = ch["req_ids"][ri]
        dispatch = finished + workload.gaps[ch["chain_id"]][ri] / 1000
        ready = dispatch + engine.frontend_ms / 1000
        p = profiles[rid]
        request = Request(rid, worker, ci, ri, dispatch, ready, next(counter),
                          max(1, p[policy]), p["output_tokens"] - 1,
                          p[policy], p["output_tokens"],
                          matched=workload.rows[rid]["glm_tokens"] - p[policy])
        heapq.heappush(events, (ready, request.order, request))

    def finish(req):
        row = dict(workload.rows[req.rid])
        row["idx_in_chain"] = row.pop("_idx_in_chain")
        row.pop("_req_id", None)
        row.update(req_id=req.rid, label=workload.metadata["label"], mix=workload.metadata["mix"],
                   N=n, cache_policy=policy, worker=req.worker,
                   prompt_tokens=row["glm_tokens"], cached_tokens=row["glm_tokens"] - req.uncached,
                   uncached_actual_model=req.uncached, output_tokens=req.output,
                   ttft_s=req.first - req.dispatch,
                   tpot_s=(now - req.first) / (req.output - 1) if req.output > 1 else None,
                   wall_s=now - req.dispatch, ttft_source="model",
                   sim_dispatch_s=req.dispatch, sim_ready_s=req.ready,
                   sim_exec_start_s=req.start, sim_first_token_s=req.first, sim_finish_s=now,
                   queue_time_s=req.start - req.ready,
                   effective_replay_gap_ms=workload.gaps[row["chain_id"]][req.index],
                   prefill_forwards=req.prefill_forwards, error=None, error_class=None)
        records.append(row)
        schedule(req.worker, req.chain, req.index + 1, now)

    for worker in range(min(n, len(workload.chains))):
        schedule(worker, None, 0, 0.0)
    while events or waiting or partial is not None or decoding:
        while events and events[0][0] <= now:
            arrived = heapq.heappop(events)[2]
            arrived.arrival_processed_tokens = stats["prefill_work_tokens"]
            waiting.append(arrived)
        peak_waiting = max(peak_waiting, len(waiting))
        can_prefill = partial is not None or (waiting and len(decoding) < max_running)
        if can_prefill and not (forced_decode and decoding):
            batch, partial = select_prefill(waiting, partial, len(decoding), engine, max_running,
                                            stats["prefill_work_tokens"], now=now)
            if not batch:
                raise RuntimeError("prefill admission made no progress")
            duration = engine.forward_overhead_ms / 1000
            for req, tokens in batch:
                if req.start is None:
                    req.start = now
                req.prefill_forwards += 1
                duration += engine.prefill_seconds(tokens, workload.rows[req.rid]["glm_tokens"])
                if (policy == "role_conservative" and req.remaining == 0
                        and req.uncached < workload.rows[req.rid]["glm_tokens"]):
                    duration += engine.d1_extra_forward_equivalents * engine.forward_overhead_ms / 1000
            if keep_trace:
                trace.append({"kind": "prefill", "start_s": now, "end_s": now + duration,
                              "requests": [r.rid for r, _ in batch],
                              "tokens": [t for _, t in batch], "running_decode": len(decoding),
                              "partial_after": partial.rid if partial else None})
            peak_running = max(peak_running, len(decoding) + len(batch))
            now += duration
            stats["prefill_seconds"] += duration
            stats["prefill_forwards"] += 1
            stats["prefill_work_tokens"] += sum(t for _, t in batch)
            # Complete tokens become visible only at forward end; same-time events
            # break ties by their deterministic creation order / worker assignment.
            for req, _ in batch:
                if req.remaining == 0:
                    req.first = now
                    if req.output_left:
                        decoding.append(req)
                    else:
                        finish(req)
            forced_decode = engine.prefill_decode_interval
        elif decoding:
            step = engine.decode_seconds(len(decoding))
            rounds = min(r.output_left for r in decoding) if coalesce_decode else 1
            if can_prefill:
                rounds = min(rounds, max(1, forced_decode))
            if events:
                # The next arrival can interrupt only AFTER its enclosing forward.
                rounds = min(rounds, max(1, math.ceil((events[0][0] - now) / step)))
            duration = rounds * step
            if keep_trace:
                trace.append({"kind": "decode", "start_s": now, "end_s": now + duration,
                              "rounds": rounds, "batch": len(decoding), "step_s": step})
            stats["decode_forwards"] += rounds
            stats["decode_seconds"] += duration
            stats["decode_tokens"] += rounds * len(decoding)
            decode_batches[len(decoding)] += rounds
            now += duration
            forced_decode = max(0, forced_decode - rounds)
            completed = []
            for req in decoding:
                req.output_left -= rounds
                if req.output_left == 0:
                    completed.append(req)
            for req in completed:
                decoding.remove(req)
                finish(req)
        elif events:
            stats["idle_seconds"] += events[0][0] - now
            now = events[0][0]
        else:
            raise RuntimeError("engine stalled")
    if len(records) != len(workload.rows):
        raise RuntimeError("incomplete replay")
    report = score_records(records, now)
    report.update(label=workload.metadata["label"], N=n, cache_policy=policy,
                  engine=asdict(engine), mix=workload.metadata["mix"],
                  wall_s=now, n_requests=len(records), n_chains=len(workload.chains),
                  engine_stats=dict(stats, peak_waiting=peak_waiting, peak_running=peak_running,
                                    decode_batch_forward_counts=dict(decode_batches)))
    return {"summary": report, "requests": records, "trace": trace}


def score_records(records, wall):
    """Call the unmodified dev scorer; TPOT extension is explicitly separate."""
    dev = evaluate(records, wall, {"lane": "dev", "strict_ttft_basis": False,
                                   "set": "OFFLINE_MODEL", "N": records[0].get("N") if records else None})
    tpots = [r["tpot_s"] for r in records if not r.get("error") and r.get("tpot_s") is not None]
    tpot = q(tpots, .95)
    tpot_pass = tpot is not None and tpot <= TPOT_LIMIT
    gates = {k: dev["gates"][k] for k in GATES}
    details = dev["ttft_gate_detail"]
    ratios = {k: d["p95"] / d["limit_s"] for k, d in details.items() if d["p95"] is not None}
    if tpot is not None:
        ratios["tpot_p95<=0.10"] = tpot / TPOT_LIMIT
    failed = [k for k, passed in gates.items() if not passed]
    if not tpot_pass:
        failed.append("tpot_p95<=0.10")
    return {"dev_gates": gates, "dev_all_pass": dev["SERVICE_GATES_PASS"],
            "ttft_gate_detail": details, "tpot_p95_s": tpot,
            "tpot_mean_s": sum(tpots) / len(tpots) if tpots else None, "tpot_p95_le_0_10": tpot_pass,
            "model_pass_dev_plus_tpot": dev["SERVICE_GATES_PASS"] and tpot_pass,
            "failed_gates": failed, "gate_utilization": ratios,
            "tightest_gate": max(ratios, key=ratios.get) if ratios else None,
            "timing_basis": "simulated server TTFT; no measured timing or correctness contract",
            "error_model": "all requests succeed; correctness/OOM/infra failures are not modeled"}


def ladder_summary(runs):
    ordered = sorted(runs, key=lambda r: r["N"])
    passing = [r["N"] for r in ordered if r["model_pass_dev_plus_tpot"]]
    first = next((r for r in ordered if not r["model_pass_dev_plus_tpot"]), None)
    prefix = []
    for r in ordered:
        if not r["model_pass_dev_plus_tpot"]:
            break
        prefix.append(r["N"])
    return {"tested_levels": [r["N"] for r in ordered], "passing_levels": passing,
            "max_passing_tested_N": max(passing, default=None),
            "contiguous_pass_through_N": max(prefix, default=None),
            "first_failing_N": first["N"] if first else None,
            "first_binding_gates": first["failed_gates"] if first else [],
            "first_failure_by_gate": {k: next((r["N"] for r in ordered
                                               if k in r["failed_gates"]), None)
                                      for k in (*GATES, "tpot_p95<=0.10")},
            "pass_after_fail": bool(first and any(n > first["N"] for n in passing)),
            "right_censored": first is None,
            "note": "All requested levels evaluated; closed-loop p95 need not be monotone. No extrapolation."}


def csv_values(value, convert):
    values = [convert(v.strip()) for v in value.split(",") if v.strip()]
    if not values:
        raise ValueError("empty parameter list")
    return list(dict.fromkeys(values))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO / "s1-dev/data/dev-combined-v1")
    ap.add_argument("--cohort-file", type=Path,
                    default=HARNESS / "g0a/samples_v3/cohort_dev-combined-v1.json")
    ap.add_argument("--out-dir", type=Path, default=REPO / "evidence/T15_simulator")
    ap.add_argument("--levels", default="2,6,10,14,18,22,26,30")
    ap.add_argument("--prefill-rates", default="40000", help="aggregate engine uncached tok/s")
    ap.add_argument("--decode-ms", default="50", help="batch-one decode times, excluding fixed overhead")
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--schedulers", default="fcfs", help="fcfs,spf,spf-upstream,hrrn,lpm,edf,least-slack,edf-chain-weighted")
    ap.add_argument("--chain-start-weight", type=float, default=2.0,
                    help="effective deadline multiplier for edf-chain-weighted only; no gate change")
    ap.add_argument("--chunk-tokens", type=int, default=8192)
    ap.add_argument("--page-tokens", type=int, default=64)
    ap.add_argument("--prefill-length-alpha", type=float, default=0.0)
    ap.add_argument("--decode-batch-slope", type=float, default=.02)
    ap.add_argument("--decode-speedup", type=float, default=1.0,
                    help="effective decode compute speedup; overhead unchanged; no MTP acceptance model")
    ap.add_argument("--decode-curve", default="", help="batch:ms,batch:ms; overrides --decode-ms")
    ap.add_argument("--forward-overhead-ms", type=float, default=2.0)
    ap.add_argument("--frontend-ms", type=float, default=0.0)
    ap.add_argument("--prefill-decode-interval", type=int, default=0)
    ap.add_argument("--max-running", type=int, default=0)
    ap.add_argument("--d1-extra-forward-equivalents", type=float, default=0.0)
    ap.add_argument("--chain-gap-cap-s", type=float, default=3600.0)
    ap.add_argument("--pacing", choices=("auto", "replay_gap", "chain-total-gap-scaled-v1"), default="auto")
    ap.add_argument("--cache-input", type=Path)
    ap.add_argument("--output-input", type=Path)
    ap.add_argument("--output-mode", choices=("budget", "source"), default="budget")
    ap.add_argument("--output-scale", type=float, default=1.0)
    ap.add_argument("--stock-fast-factor", type=float, default=F24_FAST_FACTOR)
    ap.add_argument("--stock-slow-factor", type=float, default=F24_SLOW_FACTOR)
    ap.add_argument("--mix", choices=("dev", "formal-mix", "both"), default="dev")
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--summary-only", action="store_true", help="skip per-request JSONL for large sweeps")
    ap.add_argument("--trace", action="store_true", help="write compressed forward traces")
    ap.add_argument("--envelope-fit", type=Path,
                    help="fit formal-mix against public passing-row medians in all_att.json (prior only)")
    ap.add_argument("--fit-samples", type=int, default=128,
                    help="deterministic broad random candidates; zero allows supplied candidates only")
    ap.add_argument("--fit-candidates", type=Path, help="JSON list of explicit Engine parameter overrides")
    ap.add_argument("--fit-scheduler", choices=("fcfs", "spf"), default="fcfs",
                    help="assumed scheduler of unknown public deployments; compare both after fitting")
    ap.add_argument("--fit-band-factor", type=float, default=2.0,
                    help="symmetric multiplicative diagnostic band, not a confidence interval")
    ap.add_argument("--fit-compare-top", type=int, default=4,
                    help="compare full stock/D1 x FCFS/SPF ladders for top candidates per selection class")
    args = ap.parse_args(argv)
    try:
        # Defend the read-only repository inputs even when --out-dir is explicit.
        output = args.out_dir.resolve()
        for forbidden in (REPO / "s1-dev", REPO / "src/sglang", REPO / "llm-challenge-arena-v1"):
            if output == forbidden or forbidden in output.parents:
                raise ValueError(f"output directory is read-only: {output}")
        levels = sorted(csv_values(args.levels, int))
        for n in levels:
            integer(n, "N", 1)
        rates, decode_ms = csv_values(args.prefill_rates, float), csv_values(args.decode_ms, float)
        policies, schedulers = csv_values(args.policies, str), csv_values(args.schedulers, str)
        if set(policies) - set(POLICIES):
            raise ValueError("unknown policy")
        curve = tuple((int(b), float(ms)) for b, ms in
                      (v.split(":") for v in args.decode_curve.split(","))) if args.decode_curve else ()
        if curve and len(decode_ms) > 1:
            raise ValueError("decode curve overrides base time; cannot also sweep --decode-ms")
        dev = load_workload(args.root, args.cohort_file, args.chain_gap_cap_s, args.pacing)
        workloads = ([dev] if args.mix in ("dev", "both") else [])
        if args.mix in ("formal-mix", "both"):
            workloads.append(formal_mix(dev, args.seed))
        cache = read_profiles(args.cache_input, POLICIES) if args.cache_input else None
        outputs = read_profiles(args.output_input, ("output_tokens",)) if args.output_input else None
        if args.envelope_fit:
            if args.mix != "formal-mix":
                raise ValueError("--envelope-fit requires --mix formal-mix")
            if curve or len(rates) != 1 or len(decode_ms) != 1:
                raise ValueError("envelope fit uses its own candidate list; no decode curve or rate/time grid")
            from sim_envelope_fit import run_fit
            return run_fit(args, dev, cache, outputs)
        manifest = {"label": LABEL, "arguments": {k: str(v) if isinstance(v, Path) else v
                                                 for k, v in vars(args).items()},
                    "simulator_sha256": sha256(__file__), "workloads": [], "ladders": [],
                    "profile_files": {str(p): sha256(p) for p in (args.cache_input, args.output_input) if p}}
        output.mkdir(parents=True, exist_ok=True)
        for workload in workloads:
            profiles, profile_meta = prepare_profiles(workload, cache, outputs, args.output_mode,
                args.output_scale, args.stock_fast_factor, args.stock_slow_factor)
            manifest["workloads"].append(dict(workload.metadata, profile_model=profile_meta))
            for rate, ms, scheduler, policy in itertools.product(rates, decode_ms, schedulers, policies):
                engine = Engine(prefill_tps=rate, chunk_tokens=args.chunk_tokens, page_tokens=args.page_tokens,
                    scheduler=scheduler, chain_start_weight=args.chain_start_weight, prefill_length_alpha=args.prefill_length_alpha,
                    decode_ms=ms, decode_batch_slope=args.decode_batch_slope, decode_curve=curve,
                    decode_speedup=args.decode_speedup,
                    forward_overhead_ms=args.forward_overhead_ms, frontend_ms=args.frontend_ms,
                    prefill_decode_interval=args.prefill_decode_interval, max_running=args.max_running,
                    d1_extra_forward_equivalents=args.d1_extra_forward_equivalents)
                group = f"{workload.metadata['mix']}_{scheduler}_{policy}_p{rate:g}_d{ms:g}"
                runs = []
                for n in levels:
                    result = simulate(workload, profiles, n, engine, policy, args.trace)
                    runs.append(result["summary"])
                    name = f"{group}_N{n}"
                    if not args.summary_only:
                        raw = output / f"{name}.requests.jsonl"
                        with raw.open("w") as fh:
                            for r in result["requests"]:
                                fh.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
                        runs[-1]["requests_file"] = raw.name
                    if args.trace:
                        with (output / f"{name}.trace.jsonl").open("w") as fh:
                            for event in result["trace"]:
                                fh.write(json.dumps(event, allow_nan=False) + "\n")
                ladder = dict(ladder_summary(runs), label=workload.metadata["label"],
                              group=group, mix=workload.metadata["mix"], cache_policy=policy,
                              engine=asdict(engine), runs=runs)
                manifest["ladders"].append(ladder)
                write_json(output / "sweep.json", manifest)
                print(f"MODEL OUTPUT {group}: passing={ladder['passing_levels']}; "
                      f"first fail={ladder['first_failing_N']} {ladder['first_binding_gates']}", flush=True)
        write_json(output / "sweep.json", manifest)
    except (ValueError, KeyError, OSError) as exc:
        ap.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
