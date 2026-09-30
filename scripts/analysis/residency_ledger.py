#!/usr/bin/env python3
"""Closed-loop residency ledger for a frozen replay set (CPU only; every output is a MODEL ESTIMATE).

Question: with N harness slots replaying this set, how much chain context has to be held on the
cards at once, when does that exceed the KV pool, and how many tokens are re-prefilled because
idle chain histories were evicted.

Replay order (matches s1_loadgen.py): chains in cohort.json order, popped by N workers; requests
inside a chain in the cohort's req_ids order (harness: dispatch_offset_ms, logical_call_id); the
effective gap before each request follows chain-total-gap-scaled-v1 (cap per chain).

Model:
- Prefill: one FIFO server at --prefill-rate tokens/s plus --prefill-fixed s per request. No chunk
  interleaving and no decode interference. Decode takes max_output_i * --tpot s after prefill.
- Cache: a chain's context = its last prompt + its output, in pages. Contexts of chains whose
  request is in flight are locked. The history a chain head inherits from an earlier chain of the
  same session (frozen glm_tokens - uncached_expected) is counted once per session. Family-level
  sharing of system/tools across sessions is NOT modeled, so held/locked are upper bounds.
- Admission: a request is admitted only when its new pages fit after evicting idle contexts
  (LRU, whole chain at a time; real radix eviction is per page). Otherwise it waits and is retried
  whenever a request finishes; that wait counts in its modeled TTFT.
- Hit: a follow-up whose chain context is still resident hits glm_tokens - uncached_expected; an
  evicted chain hits 0 and the lost history is counted as refill. Chain heads take their frozen
  expectation as a hit (--head-hit expected) or 0 (--head-hit none).

Reported per N (steady window = until the chain queue empties, time-weighted where stated):
  held: contexts of every chain currently assigned to a slot (in flight or waiting a gap);
  locked: contexts of in-flight requests only; over%: share of steady TIME with held > pool.
"""
import argparse
import collections
import heapq
import json
import os


def load_jsonl(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def pages(tokens, page):
    return -(-int(tokens) // page)


def effective_gaps(chain_rows, cap_s):
    orig = [int(r.get("replay_gap_ms") or 0) for r in chain_rows]
    total = sum(orig)
    cap_ms = int(cap_s * 1000)
    if cap_ms <= 0 or total <= cap_ms:
        return [g / 1000.0 for g in orig]
    alpha = cap_ms / float(total)
    eff = [round(g * alpha) for g in orig]
    nz = [i for i, g in enumerate(orig) if g > 0]
    if nz:
        eff[nz[-1]] += cap_ms - sum(eff)
    return [g / 1000.0 for g in eff]


def load_replay(root, cohort=None):
    """Rows keyed like the harness, chains in cohort order with cohort req_ids."""
    rows = {}
    for r in load_jsonl(os.path.join(root, "requests.jsonl")):
        if r.get("view") != "canon" or not r.get("in_serving_load", True):
            continue
        rows["%s:%s:%s" % (r["pack"], r["view"], r["logical_call_id"])] = r
    coh = json.load(open(cohort or os.path.join(root, "cohort.json")))
    chains = []
    for c in coh["chains"]:
        rids = [q for q in c["req_ids"] if q in rows]
        if rids:
            chains.append({"chain_id": c["chain_id"], "req_ids": rids,
                           "session_id": rows[rids[0]].get("session_id") or c["chain_id"]})
    return rows, chains, coh.get("cohort_sha256")


def percentile(values, q):
    if not values:
        return None
    v = sorted(values)
    return v[min(len(v) - 1, int(len(v) * q))]


def simulate(chains, rows, n_slots, pool_tokens, page, rate, fixed, tpot, cap_s, head_hit):
    pool_pages = pool_tokens // page
    queue = collections.deque(chains)
    events = []
    seq = [0]
    slot = {}          # slot -> {"chain","rows","gaps","i","phase","arrive"}  phase: gap|pending|prefill|decode
    resident = {}      # chain_id -> {"pages","last","locked","sid"}
    shared = {}        # session_id -> pages counted once
    head_shared = {ch["chain_id"]: pages(int(rows[ch["req_ids"][0]]["glm_tokens"]) -
                                         int(rows[ch["req_ids"][0]].get("uncached_expected") or 0), page)
                   for ch in chains}
    chain_sid = {ch["chain_id"]: ch["session_id"] for ch in chains}
    pending = collections.deque()  # slots waiting for pool space, FIFO
    server_free_at = [0.0]
    stats = collections.Counter()
    ttft = collections.defaultdict(list)
    steady_until = [None]
    samples = []  # (time, held_pages, locked_pages, resident_pages, resident_chains)

    def push(t, kind, s):
        seq[0] += 1
        heapq.heappush(events, (t, seq[0], kind, s))

    def start_chain(s, t):
        if not queue:
            slot[s] = None
            if steady_until[0] is None:
                steady_until[0] = t
            return
        ch = queue.popleft()
        crow = [rows[q] for q in ch["req_ids"]]
        slot[s] = {"chain": ch["chain_id"], "rows": crow, "gaps": effective_gaps(crow, cap_s), "i": 0,
                   "phase": "gap", "arrive": None}
        push(t + slot[s]["gaps"][0], "arrive", s)

    def used_pages():
        return sum(v["pages"] for v in resident.values()) + sum(shared.values())

    def drop_chain(k):
        sid = resident[k]["sid"]
        del resident[k]
        if sid in shared and not any(v["sid"] == sid for v in resident.values()):
            del shared[sid]

    def make_room(need):
        if pool_pages - used_pages() >= need:
            return True
        for _, k in sorted((v["last"], k) for k, v in resident.items() if not v["locked"]):
            if pool_pages - used_pages() >= need:
                break
            drop_chain(k)
            stats["evictions"] += 1
        return pool_pages - used_pages() >= need

    def ctx_pages(row):
        return pages(int(row["glm_tokens"]) + int(row.get("max_output_i") or 0), page)

    def record(t):
        held, hs, locked, ls = 0, {}, 0, {}
        for st in slot.values():
            if st is None:
                continue
            c, r = st["chain"], st["rows"][st["i"]]
            priv = max(ctx_pages(r) - head_shared[c], 0)
            held += priv
            hs[chain_sid[c]] = max(hs.get(chain_sid[c], 0), head_shared[c])
            if st["phase"] in ("prefill", "decode"):
                locked += priv
                ls[chain_sid[c]] = max(ls.get(chain_sid[c], 0), head_shared[c])
        samples.append((t, held + sum(hs.values()), locked + sum(ls.values()), used_pages(), len(resident)))

    def try_admit(s, now):
        st = slot[s]
        row = st["rows"][st["i"]]
        cid, sid = st["chain"], chain_sid[st["chain"]]
        first = st["i"] == 0
        prompt = int(row["glm_tokens"])
        out = int(row.get("max_output_i") or 0)
        expect_hit = prompt - int(row.get("uncached_expected") or prompt)
        priv = max(ctx_pages(row) - head_shared[cid], 0)
        need = priv - (resident[cid]["pages"] if cid in resident else 0)
        if sid not in shared:
            need += head_shared[cid]
        need = max(need, 0)
        if not make_room(need):
            return False
        if first:
            hit = expect_hit if head_hit == "expected" else 0
        elif cid in resident:
            hit = min(expect_hit, (resident[cid]["pages"] + head_shared[cid]) * page)
        else:
            hit = 0
            if expect_hit > 0:
                stats["followups_missing_history"] += 1
                stats["refill_tokens"] += expect_hit
        new_tokens = prompt - hit
        shared[sid] = max(shared.get(sid, 0), head_shared[cid])
        resident[cid] = {"pages": priv, "last": now, "locked": True, "sid": sid}
        start = max(now, server_free_at[0])
        end = start + fixed + new_tokens / rate
        server_free_at[0] = end
        stats["busy_s"] += fixed + new_tokens / rate
        stats["prefill_tokens"] += new_tokens
        stats["ideal_prefill_tokens"] += prompt - expect_hit
        phase = "chain_start" if first else ("turn_start" if row.get("phase") == "turn_start" else "intra")
        ttft[phase].append(end - st["arrive"])
        st["phase"] = "prefill"
        push(end, "first", s)
        push(end + out * tpot, "finish", s)
        return True

    for s in range(n_slots):
        start_chain(s, 0.0)

    now = 0.0
    while events:
        now, _, kind, s = heapq.heappop(events)
        st = slot.get(s)
        if st is None:
            continue
        if kind == "arrive":
            st["arrive"] = now
            st["phase"] = "pending"
            if not try_admit(s, now):
                pending.append(s)
                stats["admission_waits"] += 1
            record(now)
        elif kind == "first":
            st["phase"] = "decode"
        elif kind == "finish":
            cid = st["chain"]
            resident[cid]["locked"] = False
            resident[cid]["last"] = now
            st["i"] += 1
            if st["i"] < len(st["rows"]):
                st["phase"] = "gap"
                push(now + st["gaps"][st["i"]], "arrive", s)
            else:
                start_chain(s, now)
            while pending:
                p = pending[0]
                if slot.get(p) and slot[p]["phase"] == "pending" and try_admit(p, now):
                    pending.popleft()
                else:
                    break
            record(now)

    wall = now
    steady = steady_until[0] if steady_until[0] is not None else wall
    win = [x for x in samples if x[0] <= steady]
    weights = []
    over_t = 0.0
    for (t0, held, locked, _, _), (t1, _, _, _, _) in zip(win, win[1:]):
        dt = t1 - t0
        weights.append((held, locked, dt))
        if held > pool_pages:
            over_t += dt

    def tw_pct(idx, q):
        tot = sum(w[2] for w in weights)
        acc = 0.0
        for v, dt in sorted((w[idx], w[2]) for w in weights):
            acc += dt
            if acc >= q * tot:
                return v
        return None

    def gate_over(k, v):
        lim = {"chain_start": 30, "turn_start": 15, "intra": 5}[k]
        return sum(1 for x in v if x > lim)

    mt = lambda pg: round(pg * page / 1e6, 2) if pg is not None else None
    return {
        "N": n_slots,
        "wall_s": round(wall),
        "steady_s": round(steady),
        "prefill_busy_frac": round(stats["busy_s"] / wall, 3) if wall else None,
        "prefill_tokens_M": round(stats["prefill_tokens"] / 1e6, 2),
        "ideal_prefill_tokens_M": round(stats["ideal_prefill_tokens"] / 1e6, 2),
        "refill_tokens_M": round(stats["refill_tokens"] / 1e6, 2),
        "followups_missing_history": stats["followups_missing_history"],
        "evictions": stats["evictions"],
        "admission_waits": stats["admission_waits"],
        "held_over_pool_time_frac": round(over_t / steady, 3) if steady else None,
        "held_p50_Mtok": mt(tw_pct(0, 0.5)) if weights else None,
        "held_p95_Mtok": mt(tw_pct(0, 0.95)) if weights else None,
        "locked_p50_Mtok": mt(tw_pct(1, 0.5)) if weights else None,
        "locked_p95_Mtok": mt(tw_pct(1, 0.95)) if weights else None,
        "locked_max_Mtok": mt(max(w[1] for w in weights)) if weights else None,
        "resident_chains_max": max(x[4] for x in win) if win else None,
        "sim_ttft_p95": {k: round(percentile(v, 0.95), 2) for k, v in ttft.items()},
        "sim_ttft_over": {k: gate_over(k, v) for k, v in ttft.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/s1-dev-longchain")
    ap.add_argument("--n", default="10,14,18,22,26")
    ap.add_argument("--pool-tokens", type=int, default=1036288, help="047 KV pool (measured)")
    ap.add_argument("--page-size", type=int, default=64)
    ap.add_argument("--prefill-rate", type=float, default=10750.0, help="047 big-chunk tok/s (measured p50)")
    ap.add_argument("--prefill-fixed", type=float, default=0.1, help="per-request fixed seconds (assumed)")
    ap.add_argument("--tpot", type=float, default=0.03, help="s/token during decode (assumed)")
    ap.add_argument("--chain-gap-cap-s", type=float, default=3600.0)
    ap.add_argument("--head-hit", choices=["expected", "none"], default="expected")
    ap.add_argument("--cohort", default=None, help="cohort.json (default: <root>/cohort.json)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    rows, chains, sha = load_replay(args.root, args.cohort)
    n_req = sum(len(ch["req_ids"]) for ch in chains)
    results = []
    for n in [int(x) for x in args.n.split(",")]:
        r = simulate(chains, rows, n, args.pool_tokens, args.page_size, args.prefill_rate,
                     args.prefill_fixed, args.tpot, args.chain_gap_cap_s, args.head_hit)
        results.append(r)
        print(json.dumps(r, ensure_ascii=False))
    print()
    print(f"root={args.root} cohort_sha={sha} requests={n_req} chains={len(chains)} pool={args.pool_tokens} "
          f"rate={args.prefill_rate} fixed={args.prefill_fixed} tpot={args.tpot} head_hit={args.head_hit}")
    print("MODEL ESTIMATES. held/locked/over% are time-weighted over the steady window (until the chain queue empties).")
    print(f"{'N':>3} {'wall':>6} {'steady':>6} {'busy':>5} {'prefill':>8} {'ideal':>6} {'refill':>7} {'miss':>5} "
          f"{'evict':>6} {'adm.wait':>8} {'over%':>6} {'held p50':>8} {'held p95':>8} {'lock p50':>8} {'lock p95':>8} {'lock max':>8}")
    for r in results:
        print(f"{r['N']:>3} {r['wall_s']:>6} {r['steady_s']:>6} {r['prefill_busy_frac']:>5} {r['prefill_tokens_M']:>7}M "
              f"{r['ideal_prefill_tokens_M']:>5}M {r['refill_tokens_M']:>6}M {r['followups_missing_history']:>5} "
              f"{r['evictions']:>6} {r['admission_waits']:>8} {100 * r['held_over_pool_time_frac']:>5.0f}% "
              f"{r['held_p50_Mtok']:>7}M {r['held_p95_Mtok']:>7}M {r['locked_p50_Mtok']:>7}M {r['locked_p95_Mtok']:>7}M "
              f"{r['locked_max_Mtok']:>7}M")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"args": vars(args), "cohort_sha256": sha, "results": results}, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
