#!/usr/bin/env python3
"""Request-stage and prefill-window overlap diagnostics (CPU only; not SLO scoring).

Example for a verified same-host dev run:
  blocking.py RAW --pairs evidence/T56/pairs_attributed.json --server-minus-client-s 0 --csv OUT.csv --json OUT.json

Measured: receive-to-first-forward, first-forward-to-first-token, and client decode
window. A prefill *window* includes gaps and other requests' work; overlap is NOT
GPU busy time or causal blocking. Concurrent candidate windows split overlap evenly
for a descriptive ranking, not causal attribution. Summing across waiting requests
produces request-seconds, not GPU seconds.

The optional cost scenario defaults to 0.11 s/assumed forward + 65 us/new token,
assuming 16384-token chunks. Actual chunks/batches/context can differ. Its signed
residual is not "interleave"; estimates are never clipped to fit measured time.
LCP pairs are optional, validated against this cohort's IDs/chain positions/lengths;
missing LCP remains unknown. They must come from the same frozen prompt bodies.

Server/client overlap requires an explicit clock offset: server = client + offset.
Use 0 only when the clocks are known to share a timebase. No offset => no decode
overlap estimate. Use level_verdict.py separately for full runner/SLO validation.
"""
import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]


def load_common(harness):
    spec = importlib.util.spec_from_file_location("blocking_s1_common", Path(harness) / "s1_common.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def number(value, label, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label}: expected finite number")
    if integer and (value < 0 or int(value) != value):
        raise ValueError(f"{label}: expected nonnegative integer")
    return value


def validate_rows(rows, expected):
    """Validate the complete frozen cohort, not merely a hard-coded row count."""
    if not rows:
        raise ValueError("empty raw")
    ids = [r.get("req_id") for r in rows]
    if any(not isinstance(rid, str) for rid in ids):
        raise ValueError("missing/non-string req_id")
    counts = collections.Counter(ids)
    missing, extra = set(expected) - set(ids), set(ids) - set(expected)
    duplicate = [rid for rid, count in counts.items() if count != 1]
    if missing or extra or duplicate:
        raise ValueError(f"cohort mismatch: missing={len(missing)} extra={len(extra)} duplicates={len(duplicate)}")
    metadata = {"idx_in_chain": "_idx_in_chain", "chain_id": "chain_id",
                "phase": "phase", "edge_type": "edge_type",
                "uncached_expected": "uncached_expected", "max_output_i": "max_output_i"}
    for r in rows:
        rid = r["req_id"]
        if r.get("error"):
            raise ValueError(f"{rid}: engine/runner error")
        number(r.get("idx_in_chain"), f"{rid}/idx_in_chain", integer=True)
        for raw_key, source_key in metadata.items():
            if r.get(raw_key) != expected[rid].get(source_key):
                raise ValueError(f"{rid}: frozen {raw_key} mismatch")
        for key in ("prompt_tokens", "cached_tokens", "output_tokens"):
            number(r.get(key), f"{rid}/{key}", integer=True)
        if not 0 <= r["cached_tokens"] <= r["prompt_tokens"]:
            raise ValueError(f"{rid}: cached_tokens outside prompt")
        if r["prompt_tokens"] != expected[rid]["glm_tokens"]:
            raise ValueError(f"{rid}: prompt length differs from frozen cohort")
        if r["output_tokens"] != expected[rid]["max_output_i"]:
            raise ValueError(f"{rid}: incomplete/frozen output length mismatch")
        for key in ("t_recv_s", "t_exec_start_s", "t_first_token_s",
                    "client_dispatch_at_s", "client_first_token_at_s", "client_finish_at_s", "ttft_s"):
            number(r.get(key), f"{rid}/{key}")
        if not r["t_recv_s"] <= r["t_exec_start_s"] <= r["t_first_token_s"]:
            raise ValueError(f"{rid}: server timestamps out of order")
        if not r["client_dispatch_at_s"] <= r["client_first_token_at_s"] <= r["client_finish_at_s"]:
            raise ValueError(f"{rid}: client timestamps out of order")
        if abs(r["ttft_s"] - (r["t_first_token_s"] - r["t_recv_s"])) > 1e-4:
            raise ValueError(f"{rid}: ttft_s disagrees with server timestamps")
        if r["output_tokens"] > 1 or r.get("tpot_s") is not None:
            if number(r.get("tpot_s"), f"{rid}/tpot_s") < 0:
                raise ValueError(f"{rid}: negative tpot_s")


def load_lcp(pairs, rows):
    if pairs is None:
        return {}
    by_id = {r["req_id"]: r for r in rows}
    by_position = {(r["chain_id"], r["idx_in_chain"]): r for r in rows}
    result = {}
    for pair in pairs:
        rid = pair.get("req_id")
        if rid in result or rid not in by_id:
            raise ValueError("LCP duplicate/foreign req_id")
        r = by_id[rid]
        previous = by_position.get((r["chain_id"], r["idx_in_chain"] - 1))
        if previous is None:
            raise ValueError(f"{rid}: LCP pair has no cohort predecessor")
        for key, wanted in (("chain_id", r["chain_id"]), ("idx", r["idx_in_chain"]),
                            ("prompt", r["prompt_tokens"]), ("previous_prompt", previous["prompt_tokens"])):
            if pair.get(key) != wanted:
                raise ValueError(f"{rid}: LCP {key} mismatch")
        lcp = number(pair.get("true_lcp"), f"{rid}/true_lcp", integer=True)
        if lcp > min(r["prompt_tokens"], previous["prompt_tokens"]):
            raise ValueError(f"{rid}: LCP exceeds prompt length")
        result[rid] = int(lcp)
    return result


def own_cost(uncached, fixed_s, per_token_s, chunk):
    # Scenario only: batch-shared overhead and real chunk counts are not observed here.
    return max(1, math.ceil(uncached / chunk)) * fixed_s + uncached * per_token_s


def kind(r):
    uncached = r["prompt_tokens"] - r["cached_tokens"]
    size = "<8k" if uncached < 8192 else ("8k-64k" if uncached < 65536 else ">=64k")
    return ("cohort_head" if r["idx_in_chain"] == 0 else "followup") + ":" + size


def attribute(window, intervals, exclude):
    """Union and equal-share descriptive allocation, in this window's seconds."""
    w0, w1 = window
    if w1 <= w0:
        return 0.0, {}
    points, candidates = {w0, w1}, []
    for j, (start, end) in intervals:
        if j != exclude and end > start and end > w0 and start < w1:
            start, end = max(start, w0), min(end, w1)
            candidates.append((j, start, end))
            points.update((start, end))
    union, share = 0.0, collections.defaultdict(float)
    points = sorted(points)
    for left, right in zip(points, points[1:]):
        live = [j for j, start, end in candidates if start <= left and end >= right]
        if live:
            union += right - left
            for j in live:
                share[j] += (right - left) / len(live)
    return union, dict(share)


def by_kind(share, rows):
    result = collections.defaultdict(float)
    for j, seconds in share.items():
        result[kind(rows[j])] += seconds
    return dict(result)


def candidates(share, rows):
    return [{"req_id": rows[j]["req_id"], "allocated_window_overlap_s": seconds}
            for j, seconds in sorted(share.items(), key=lambda x: -x[1])]


def analyze(rows, common, lcp=None, fixed_s=.11, per_token_s=65e-6,
            chunk=16384, page_size=64, server_minus_client_s=None):
    lcp = {} if lcp is None else lcp
    for label, value in (("fixed_s", fixed_s), ("per_token_s", per_token_s)):
        if number(value, label) < 0:
            raise ValueError(f"{label}: must be nonnegative")
    for label, value in (("chunk", chunk), ("page_size", page_size)):
        if number(value, label, integer=True) <= 0:
            raise ValueError(f"{label}: must be positive")
    if server_minus_client_s is not None:
        number(server_minus_client_s, "server_minus_client_s")
    intervals = [(i, (r["t_exec_start_s"], r["t_first_token_s"])) for i, r in enumerate(rows)]
    out = []
    overlap_totals = {"ttft": collections.defaultdict(float), "tpot": collections.defaultdict(float)}
    gates = {selector: limit for _, selector, limit in common.TTFT_GATE_SPECS}
    for i, r in enumerate(rows):
        recv, start, first = (r[k] for k in ("t_recv_s", "t_exec_start_s", "t_first_token_s"))
        uncached = r["prompt_tokens"] - r["cached_tokens"]
        estimate = own_cost(uncached, fixed_s, per_token_s, chunk)
        gap = max(0, lcp[r["req_id"]] // page_size * page_size - r["cached_tokens"]) if r["req_id"] in lcp else None
        queue_union, queue_share = attribute((recv, start), intervals, i)
        decode_window = r["client_finish_at_s"] - r["client_first_token_at_s"]
        decode_union, decode_share = None, {}
        if server_minus_client_s is not None:
            decode_union, decode_share = attribute(
                (r["client_first_token_at_s"] + server_minus_client_s,
                 r["client_finish_at_s"] + server_minus_client_s), intervals, i)
        selected = [g for g in gates if common.in_ttft_gate(r, g)]
        over = [g for g in selected if r["ttft_s"] > gates[g]]
        tpot_over = r.get("tpot_s") is not None and r["tpot_s"] > .10
        for label, enabled, share in (("ttft", bool(over), queue_share), ("tpot", tpot_over, decode_share)):
            if enabled:
                for j, seconds in share.items():
                    overlap_totals[label][j] += seconds
        out.append({
            "req_id": r["req_id"], "chain_id": r["chain_id"], "idx_in_chain": r["idx_in_chain"],
            "kind": kind(r), "gates": selected, "ttft_over": over, "tpot_over": tpot_over,
            "ttft_s": r["ttft_s"], "queue_s": start - recv, "exec_window_s": first - start,
            "prefill_cost_est_s": estimate, "model_residual_s": first - start - estimate,
            "model_exceeds_exec_window": estimate > first - start + 1e-6,
            "cache_gap_tokens": gap, "cache_gap_variable_cost_est_s": None if gap is None else gap * per_token_s,
            "uncached_tokens": uncached, "output_tokens": r["output_tokens"], "tpot_s": r.get("tpot_s"),
            "queue_prefill_window_overlap_s": queue_union,
            "queue_uncovered_s": max(0, start - recv - queue_union),
            "queue_overlap_by_kind_s": by_kind(queue_share, rows),
            "queue_overlap_candidates": candidates(queue_share, rows),
            "decode_window_s": decode_window, "decode_prefill_window_overlap_s": decode_union,
            "decode_overlap_fraction": decode_union / decode_window if decode_union is not None and decode_window > 0 else None,
            "decode_overlap_by_kind_s": by_kind(decode_share, rows),
            "decode_overlap_candidates": candidates(decode_share, rows),
        })
    summary = {"n": len(out), "model_exceeds_exec_window_n": sum(o["model_exceeds_exec_window"] for o in out),
               "followup_n": sum(o["idx_in_chain"] > 0 for o in out),
               "lcp_covered_followup_n": sum(o["idx_in_chain"] > 0 and o["cache_gap_tokens"] is not None for o in out),
               "gates": {}}

    def group_sum(items, key):
        counts = collections.defaultdict(float)
        for item in items:
            for label, seconds in item[key].items():
                counts[label] += seconds
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    for gate in gates:
        items = [o for o in out if gate in o["ttft_over"]]
        queue = sum(o["queue_s"] for o in items)
        exec_window = sum(o["exec_window_s"] for o in items)
        union = sum(o["queue_prefill_window_overlap_s"] for o in items)
        summary["gates"][gate] = {
            "over_n": len(items), "queue_gt_half_ttft_n": sum(o["queue_s"] > o["ttft_s"] / 2 for o in items),
            "queue_request_s": queue, "exec_window_request_s": exec_window,
            "queue_share_of_ttft": queue / (queue + exec_window) if queue + exec_window > 0 else None,
            "queue_overlap_request_s": union, "queue_overlap_coverage": union / queue if queue > 0 else None,
            "queue_overlap_by_kind_request_s": group_sum(items, "queue_overlap_by_kind_s"),
            "lcp_covered_n": sum(o["cache_gap_tokens"] is not None for o in items),
            "cache_gap_tokens_known_sum": sum(o["cache_gap_tokens"] or 0 for o in items),
            "model_exceeds_exec_window_n": sum(o["model_exceeds_exec_window"] for o in items),
        }
    tpot_items = [o for o in out if o["tpot_over"]]
    fractions = [o["decode_overlap_fraction"] for o in tpot_items if o["decode_overlap_fraction"] is not None]
    summary["tpot"] = {
        "over_n": len(tpot_items), "overlap_available_n": len(fractions),
        "decode_window_overlap_fraction_median": statistics.median(fractions) if fractions else None,
        "decode_overlap_by_kind_request_s": group_sum(tpot_items, "decode_overlap_by_kind_s"),
    }
    summary["top_overlapping_requests"] = {}
    for label, shares in overlap_totals.items():
        summary["top_overlapping_requests"][label] = [
            {"req_id": rows[j]["req_id"], "kind": kind(rows[j]),
             "uncached_tokens": rows[j]["prompt_tokens"] - rows[j]["cached_tokens"],
             "prefill_window_s": rows[j]["t_first_token_s"] - rows[j]["t_exec_start_s"],
             "allocated_overlap_request_s": seconds}
            for j, seconds in sorted(shares.items(), key=lambda x: -x[1])[:5]
        ]
    return out, summary


def print_summary(report):
    s = report["summary"]
    print(f"DIAGNOSTIC_ONLY schema=2 n={s['n']} cohort=verified; not a runner/SLO verdict")
    print("Measured phases and overlapping request windows; no causal labels or GPU busy-time claims.")
    print("Cost SCENARIO: " + json.dumps(report["cost_scenario"], sort_keys=True))
    print(f"Model above measured exec window: {s['model_exceeds_exec_window_n']}/{s['n']}; signed residual retained.")
    print(f"LCP coverage: {s['lcp_covered_followup_n']}/{s['followup_n']} followups; missing is unknown.")
    print(f"Server minus client clock offset: {report['server_minus_client_s']}; None disables decode overlap.")

    def show_groups(groups):
        total = sum(groups.values())
        return ", ".join(f"{k} {v/total:.0%}" for k, v in groups.items()) if total else "no overlap"

    for gate, value in s["gates"].items():
        print(f"-- {gate}: {value['over_n']} threshold overs; queue>half TTFT: {value['queue_gt_half_ttft_n']}")
        if value["over_n"]:
            print(f"   measured queue={value['queue_request_s']:.3f}, exec_window={value['exec_window_request_s']:.3f} request-s")
            coverage = value["queue_overlap_coverage"]
            print("   queue window overlap coverage: " + ("n/a" if coverage is None else f"{coverage:.1%}"))
            print("   within that overlap, equal-share candidate kinds: " + show_groups(value["queue_overlap_by_kind_request_s"]))
    value = s["tpot"]
    fraction = value["decode_window_overlap_fraction_median"]
    print(f"-- TPOT>0.10: {value['over_n']}; decode WINDOW overlap median: " + ("unavailable" if fraction is None else f"{fraction:.1%}"))
    print("   within that overlap, equal-share candidate kinds: " + show_groups(value["decode_overlap_by_kind_request_s"]))
    for label, candidates in s["top_overlapping_requests"].items():
        print(f"-- {label} overlapping candidates (request-seconds summed across affected requests, not GPU seconds):")
        for candidate in candidates:
            print(f"   {candidate['allocated_overlap_request_s']:.3f} request-s; window={candidate['prefill_window_s']:.3f}s "
                  f"{candidate['kind']} {candidate['req_id']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--data-root", type=Path, default=ROOT / "s1-dev/data/dev-combined-v1")
    parser.add_argument("--harness-dir", type=Path, default=ROOT / "s1-dev/harness")
    parser.add_argument("--pairs", type=Path, help="true LCP pairs from the same frozen prompt bodies; no implicit old file")
    parser.add_argument("--server-minus-client-s", type=float)
    parser.add_argument("--fixed-s", type=float, default=.11)
    parser.add_argument("--per-token-us", type=float, default=65)
    parser.add_argument("--model-chunk", type=int, default=16384, help="assumed chunk size, not observed forwards")
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--json", type=Path, dest="json_out")
    args = parser.parse_args()
    try:
        common = load_common(args.harness_dir)
        expected, _, _ = common.load_index(str(args.data_root))
        raw = args.raw.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        validate_rows(rows, expected)
        pairs = json.loads(args.pairs.read_text()) if args.pairs is not None else None
        lcp = load_lcp(pairs, rows)
        out, summary = analyze(rows, common, lcp, args.fixed_s, args.per_token_us * 1e-6,
                               args.model_chunk, args.page_size, args.server_minus_client_s)
        report = {
            "schema_version": 2, "diagnostic_only": True, "raw": str(args.raw),
            "raw_sha256": hashlib.sha256(raw).hexdigest(), "data_root": str(args.data_root),
            "pairs": str(args.pairs) if args.pairs else None,
            "server_minus_client_s": args.server_minus_client_s,
            "cost_scenario": {"fixed_s": args.fixed_s, "per_token_us": args.per_token_us,
                              "assumed_chunk": args.model_chunk, "page_size": args.page_size},
            "limits": [
                "Receive-to-first-forward includes admission and CPU preparation, not only scheduler waiting.",
                "Request prefill windows include decode and other work; overlap is not measured GPU blocking.",
                "Equal split of concurrent windows is descriptive, not causal; category percentages condition on covered overlap.",
                "Sums across affected requests are request-seconds, not device wall seconds.",
                "Cost and cache-gap seconds are model scenarios; signed residual is not measured interleaving.",
                "LCP checks IDs, positions and lengths; caller must ensure the same frozen tokenized prompts.",
                "Cohort validation does not replace full runner, capability and SLO checks.",
            ],
            "summary": summary, "requests": out,
        }
        if args.json_out:
            args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        if args.csv:
            with args.csv.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(out[0]), lineterminator="\n")
                writer.writeheader()
                for row in out:
                    writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                                     for k, v in row.items()})
        print_summary(report)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"INVALID_DIAGNOSTIC: {error}\n")


if __name__ == "__main__":
    main()
