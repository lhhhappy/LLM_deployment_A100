#!/usr/bin/env python3
"""README — offline dev scoring plus explicitly estimated arena checks (CPU only).

Usage:
  python3 scripts/score_formal.py --run-dir /path/to/level --out score_estimated.json
  python3 scripts/score_formal.py --raw raw_x.jsonl --run run_x.json --out score.json

Python >= 3.10, standard library only. Imports the unmodified public s1_score.py
with bytecode writes disabled, calls its dev evaluate() with run_dev's defaults,
and retains that entire report under `dev`. Never invokes the harness formal lane.
A directory uses summary.json to select measured files; ambiguous globs are an
error (preflight/warmup must not be pooled with measured requests).

TTFT: same buckets and order-statistic p95 as dev; one-sided 95% Clopper–Pearson
lower bound L = Beta_quantile(.05, k, n-k+1), L=0 for k=0. Numerically invert
P[Binomial(n,L) >= k] = .05 by bisection, using log-space binomial sums.
Fail the estimated TTFT check only if L > .05; equality at the time limit is
not an exceedance. The organizer has not specified its interval implementation.
Wilson/Wald comparisons are diagnostic only; they never replace the CP result
or waive any other gate. Use level_verdict.py to verify runner/flush evidence.
Method reference: https://www.stat.ethz.ch/R-manual/R-devel/library/stats/html/binom.test.html
Within-chain dependence and this dev cohort also limit population inference.

TPOT uses raw tpot_s (client first/last SSE timing divided by output_tokens-1),
unweighted per-request mean and dev's p95 convention. No statistical allowance
is applied to the 0.10 s/token gate. Missing/invalid multi-token TPOT blocks the
estimate; single-token outputs have undefined TPOT and are counted separately.
The file entrypoint also checks the full dev request index (each req_id once,
with its chain position), and checks successful responses against the frozen
prompt/output token budgets and legal cache counts before reporting PASS/FAIL.
This is replay validity, not an additional official SLO gate. Error rows remain
in the original harness error-rate denominator. Partial raw files are
invalid input; use score_records() directly for diagnostic subsets.
Cache diagnostics retain frozen uncached_expected bucket membership.

Official weighting (s1-dev/harness/s1_score.py evaluate): ONLY the overall
budget_attainment diagnostic uses sampling_weight in lane=formal; dev uses its
unweighted value. Both are reported, and its 80% tripwire is NOT a hard gate.
TTFT gate p95, coverage, error rates, timing coverage, slack/uncached quantiles,
stratified budget attainment, and TPM are NOT sampling-weighted in either lane.
Token-weighted cache hit = sum(cached)/sum(prompt), not sampling weighting.
The public scorer has no TPOT hard gate; our estimated TPOT check/mean remain
unweighted. Added weighted TTFT/TPOT/cache and per-gate budget statistics are
INTERNAL population diagnostics, never replacements for official hard gates.
No binomial confidence interval is applied to fractional sampling weights.
Weights are joined by req_id from requests.jsonl (default shipped dev index),
falling back to raw sampling_weight only for IDs absent from that index.
Missing weights contribute zero, as in the harness; coverage is reported.
Weighted quantiles use first cumulative weight STRICTLY above p*total (last
value at p=1), reproducing dev's floor(p*n) convention for equal weights.

Outputs JSON to stdout, optionally also --out (atomic write). Exit 0 means a
report was produced, including for failing gates; exit 2 means invalid inputs.
These are dev estimates, never an official capacity or verdict.
"""

from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import importlib.util
import json
import math
from statistics import NormalDist
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_HARNESS = REPO / "s1-dev" / "harness"
TPOT_GATE = "tpot_p95<=0.10"
METHOD = "one-sided 95% Clopper-Pearson exact binomial lower bound"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_harness(directory: Path = DEFAULT_HARNESS):
    """Reuse dev policy verbatim without creating files in the read-only harness."""
    directory = directory.resolve()
    common = sys.modules.get("s1_common")
    if common and Path(common.__file__).resolve().parent != directory:
        raise ValueError("a different s1_common is already imported")
    spec = importlib.util.spec_from_file_location("_t12_dev_score", directory / "s1_score.py")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def binomial_tail(n: int, k: int, p: float) -> float:
    """P(X >= k), stable even when endpoint probability masses underflow."""
    if k <= 0:
        return 1.0
    if k > n or p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    lp, lq = math.log(p), math.log1p(-p)
    term = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    term += k * lp + (n - k) * lq
    terms = [term]
    for j in range(k, n):
        term += math.log(n - j) - math.log(j + 1) + lp - lq
        terms.append(term)
    peak = max(terms)
    return min(1.0, math.exp(peak) * math.fsum(math.exp(t - peak) for t in terms))


@lru_cache(maxsize=4096)
def binomial_lower(k: int, n: int, alpha: float = .05) -> float | None:
    if n < 0 or not 0 <= k <= n or not 0 < alpha < 1:
        raise ValueError("invalid binomial counts or alpha")
    if n == 0:
        return None
    if k == 0:
        return 0.0
    if k == n:
        return alpha ** (1.0 / n)
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if binomial_tail(n, k, mid) < alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


@lru_cache(maxsize=256)
def allowed_over(n: int) -> int | None:
    """Largest accepted count for the estimated exact one-sided binomial test."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    if n == 0:
        return None
    lo, hi = 0, n + 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        # Inverting the same one-sided test avoids repeatedly finding a CI.
        if binomial_tail(n, mid, .05) >= .05:
            lo = mid
        else:
            hi = mid
    return lo


def finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def alternative_lower(k: int, n: int, method: str) -> float | None:
    """One-sided 95% Wilson/Wald diagnostics; neither replaces our exact estimate."""
    if isinstance(n, bool) or isinstance(k, bool) or not isinstance(n, int) or not isinstance(k, int) or not 0 <= k <= n:
        raise ValueError("invalid binomial counts")
    if method not in ("wilson", "wald"):
        raise ValueError("unknown diagnostic interval")
    if n == 0:
        return None
    p, z = k / n, NormalDist().inv_cdf(.95)
    if method == "wald":
        return max(0.0, p - z * math.sqrt(p * (1 - p) / n))
    return max(0.0, (p + z*z/(2*n) - z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n))


@lru_cache(maxsize=256)
def alternative_allowed(n: int, method: str) -> int | None:
    alternative_lower(0, n, method)
    if n == 0:
        return None
    lo, hi = 0, n + 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if alternative_lower(mid, n, method) <= .05:
            lo = mid
        else:
            hi = mid
    return lo


def interval_sensitivity(k: int, n: int, evaluable=True) -> dict:
    methods = {"clopper_pearson": {"rate_ci_lower": binomial_lower(k, n), "allowed_over": allowed_over(n)}}
    for method in ("wilson", "wald"):
        methods[method] = {"rate_ci_lower": alternative_lower(k, n, method),
                           "allowed_over": alternative_allowed(n, method)}
    for values in methods.values():
        values["pass_estimated"] = bool(evaluable and n > 0 and values["rate_ci_lower"] <= .05)
    return {"diagnostic_only": True, "organizer_method_unknown": True, "methods": methods,
            "method_sensitive": bool(evaluable and n > 0 and len({v["pass_estimated"] for v in methods.values()}) > 1)}


def distribution(values: list, scorer) -> dict:
    return {"n": len(values), "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": math.fsum(values) / len(values) if values else None,
            **scorer.qs(values)}


def cache_distribution(records: list[dict], scorer) -> dict:
    fields = ("prompt_tokens", "cached_tokens", "uncached_expected", "glm_tokens")
    valid = lambda x: finite_number(x) and x >= 0 and int(x) == x
    paired = [r for r in records if all(valid(r.get(k)) for k in fields[:3])
              and r["cached_tokens"] <= r["prompt_tokens"]]
    actual = [r["prompt_tokens"] - r["cached_tokens"] for r in paired]
    expected = [r["uncached_expected"] for r in paired]
    delta = [a - e for a, e in zip(actual, expected)]
    frozen = [r for r in paired if valid(r.get("glm_tokens"))
              and r["glm_tokens"] >= r["uncached_expected"]]
    return {
        "weighted": {field: weighted_distribution([
            {**r, "uncached_actual": r["prompt_tokens"] - r["cached_tokens"]}
            for r in paired], field) for field in
            ("cached_tokens", "uncached_expected", "uncached_actual")},
        "n_successful": len(records), "n_paired": len(paired),
        "n_missing_or_invalid_pair": len(records) - len(paired),
        "n_missing_by_field": {k: sum(r.get(k) is None for r in records) for k in fields},
        "cached_tokens": distribution([r["cached_tokens"] for r in paired], scorer),
        "uncached_expected": distribution(expected, scorer),
        "uncached_actual": distribution(actual, scorer),
        "extra_uncached_tokens": distribution(delta, scorer),
        "actual_to_expected_ratio": distribution([a / e for a, e in zip(actual, expected) if e], scorer),
        "n_zero_expected": sum(e == 0 for e in expected),
        "n_actual_above_expected": sum(d > 0 for d in delta),
        "n_actual_equal_expected": sum(d == 0 for d in delta),
        "n_actual_below_expected": sum(d < 0 for d in delta),
        "extra_uncached_histogram": {
            "negative": sum(d < 0 for d in delta), "zero": sum(d == 0 for d in delta),
            **{f"({lo},{hi}]": sum(lo < d <= hi for d in delta)
               for lo, hi in ((0, 64), (64, 256), (256, 1024), (1024, 4096))},
            ">4096": sum(d > 4096 for d in delta)},
        "frozen_cached_expected": distribution([r["glm_tokens"] - r["uncached_expected"]
                                                  for r in frozen], scorer),
        "cached_minus_frozen_expected": distribution([
            r["cached_tokens"] - (r["glm_tokens"] - r["uncached_expected"])
            for r in frozen], scorer),
        "prompt_minus_frozen_glm_tokens": distribution([
            r["prompt_tokens"] - r["glm_tokens"] for r in frozen], scorer),
    }


def weighted_distribution(records, field, limit=None):
    valid = [r for r in records if finite_number(r.get(field)) and r[field] >= 0]
    pairs = sorted((r[field], r.get("sampling_weight") or 0.0) for r in valid
                   if (r.get("sampling_weight") or 0.0) > 0)
    total = math.fsum(w for _, w in pairs)
    def quantile(p):
        if not total:
            return None
        # Avoid cumulative floating-point ties changing dev's order statistic.
        if all(w == pairs[0][1] for _, w in pairs):
            return pairs[min(len(pairs)-1, int(p*len(pairs)))][0]
        cumulative = 0.0
        for value, weight in pairs:
            cumulative += weight
            if cumulative > p * total:
                return value
        return pairs[-1][0]
    return {"label": "diagnostic_only", "n_valid": len(valid),
            "n_positive_weight": len(pairs), "sum_weight": total,
            "n_missing_weight": sum(r.get("sampling_weight") is None for r in valid),
            "mean": math.fsum(v * w for v, w in pairs) / total if total else None,
            "p50": quantile(.5), "p95": quantile(.95),
            "exceed_rate": (math.fsum(w for v, w in pairs if v > limit) / total
                            if total and limit is not None else None)}


def attach_weights(records, request_file=None):
    """Copy inputs; index weights take precedence, without silently inventing 1."""
    index = {}
    if request_file is not None:
        with Path(request_file).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    rid = row.get("req_id") or ":".join(
                        str(row[k]) for k in ("pack", "view", "logical_call_id"))
                    if rid in index:
                        raise ValueError("duplicate req_id in weight index")
                    index[rid] = row.get("sampling_weight")
    copied, counts = [], {"from_requests": 0, "from_raw": 0, "missing": 0, "raw_conflicts": 0}
    for original in records:
        row = dict(original)
        if row.get("req_id") in index:
            weight = index[row["req_id"]]
            counts["from_requests"] += 1
            counts["raw_conflicts"] += (row.get("sampling_weight") is not None
                                        and row["sampling_weight"] != weight)
        else:
            weight = row.get("sampling_weight")
            counts["from_raw"] += weight is not None
        if weight is not None and (not finite_number(weight) or weight < 0):
            raise ValueError("sampling_weight must be finite and nonnegative, or null")
        counts["missing"] += weight is None
        row["sampling_weight"] = weight
        copied.append(row)
    return copied, counts


def score_records(records: list[dict], run: dict, scorer=None, request_file=None) -> dict:
    scorer = scorer or load_harness()
    records, weight_sources = attach_weights(records, request_file)
    if not finite_number(run.get("wall_s")) or run["wall_s"] < 0:
        raise ValueError("run metadata must include finite, nonnegative wall_s")
    cfg = dict(run.get("config", {}))
    # Match s1_score.py's CLI defaults used by run_dev.py, without adding gates.
    cfg.update(lane="dev", strict_ttft_basis=False,
               steady_start_min=scorer.STEADY_START_MIN, steady_len_min=scorer.STEADY_LEN_MIN)
    dev = scorer.evaluate(records, run["wall_s"], cfg)
    ok = [r for r in records if not r.get("error")
          and r.get("error_class") not in (scorer.ERR_HARNESS_DATA, scorer.ERR_HARNESS_RENDER)]
    ttft = {}
    for name, selector, limit in scorer.TTFT_GATE_SPECS:
        bucket = [r for r in ok if scorer.in_ttft_gate(r, selector)]
        values = [r["ttft_s"] for r in bucket
                  if finite_number(r.get("ttft_s")) and r["ttft_s"] >= 0]
        n, over = len(values), sum(v > limit for v in values)
        lower = binomial_lower(over, n)
        evaluable = n > 0 and n == len(bucket)
        passed = evaluable and lower <= .05
        ttft[name] = {
            "label": "estimated", "method": METHOD, "limit_s": limit,
            "n": n, "n_in_bucket": len(bucket), "n_missing_or_invalid": len(bucket) - n,
            "p95": scorer.q(values, .95),
            "weighted": weighted_distribution(bucket, "ttft_s", limit),
            "pass_point": bool(evaluable and scorer.q(values, .95) <= limit),
            "over_limit": over, "exceed_rate": over / n if n else None,
            "rate_ci_lower": lower, "confidence": .95, "exceed_rate_limit": .05,
            "allowed_over": allowed_over(n), "evaluable": evaluable,
            "pass_estimated": passed,
            "interval_sensitivity": interval_sensitivity(over, n, evaluable),
        }
    eligible = [r for r in ok if finite_number(r.get("output_tokens")) and r["output_tokens"] > 1]
    undefined = [r for r in ok if r.get("output_tokens") == 1]
    tpots = [r["tpot_s"] for r in eligible
             if finite_number(r.get("tpot_s")) and r["tpot_s"] >= 0]
    missing = len(ok) - len(undefined) - len(tpots)
    p95 = scorer.q(tpots, .95)
    tpot = {"name": TPOT_GATE, "limit_s_per_token": .10,
            "n": len(tpots), "n_successful": len(ok),
            "n_single_token_undefined": len(undefined), "n_missing_or_invalid": missing,
            "tpot_mean": math.fsum(tpots) / len(tpots) if tpots else None,
            "tpot_p95": p95, "passed": bool(tpots and missing == 0 and p95 <= .10),
            "weighted": weighted_distribution(eligible, "tpot_s", .10),
            "source": "raw tpot_s; client SSE elapsed / (output_tokens - 1)",
            "quantile_method": "sorted[min(n-1, floor(0.95*n))], as in dev; no CI allowance"}
    gates = dict(dev["gates"])
    gates.update({name: detail["pass_estimated"] for name, detail in ttft.items()})
    gates[TPOT_GATE] = tpot["passed"]
    passed = all(gates.values())
    return {
        "schema_version": 1, "label": "estimated",
        "note": "Dev cohort only; organizer CI implementation unknown; binomial model assumes independent trials.",
        "dev": dev, "tpot": tpot, "ttft_estimated": ttft,
        "interval_sensitivity": {
            "diagnostic_only": True, "primary_method": "clopper_pearson", "organizer_method_unknown": True,
            "sensitive_ttft_gates": [name for name, detail in ttft.items() if detail["interval_sensitivity"]["method_sensitive"]],
            "all_gates_pass_by_method": {
                method: all(value for name, value in gates.items() if name not in ttft) and
                        all(detail["interval_sensitivity"]["methods"][method]["pass_estimated"] for detail in ttft.values())
                for method in ("clopper_pearson", "wilson", "wald")},
        },
        "sampling_weights": {**weight_sources,
            "source": str(request_file) if request_file is not None else "raw records",
            "official_formal_authority": "overall budget_attainment.weighted only; diagnostic, not a gate",
            "weighted_gate_statistics": "diagnostic_only; never used in pass/fail or binomial CI",
            "quantile_method": "first cumulative weight > p * total; equal weights match dev floor(p*n)"},
        "budget_attainment_by_gate": {
            name: gate_budget([r for r in ok if scorer.in_ttft_gate(r, selector)], scorer)
            for name, selector, _ in scorer.TTFT_GATE_SPECS},
        "estimated": {"label": "estimated", "passed": passed,
                      "evaluation_status": "estimated PASS" if passed else "estimated FAIL",
                      "gates": gates},
        "cache_reconciliation": {
            "formula": "uncached_actual = prompt_tokens - cached_tokens; extra = actual - frozen uncached_expected",
            "all_successful": cache_distribution(ok, scorer),
            "by_gate": {name: cache_distribution([r for r in ok if scorer.in_ttft_gate(r, selector)], scorer)
                        for name, selector, _ in scorer.TTFT_GATE_SPECS}},
    }


def gate_budget(records, scorer):
    judged = [r for r in records if all(finite_number(r.get(k)) for k in
              ("prompt_tokens", "cached_tokens", "ttft_s"))]
    passed = [r for r in judged if r["ttft_s"] <= scorer.BUDGET_BASE_S +
              max(0, r["prompt_tokens"] - r["cached_tokens"]) / scorer.BUDGET_RATE]
    weight = math.fsum(r.get("sampling_weight") or 0 for r in judged)
    return {"label": "diagnostic_only", "n": len(judged), "sum_weight": weight,
            "unweighted": len(passed) / len(judged) if judged else None,
            "weighted": math.fsum(r.get("sampling_weight") or 0 for r in passed) / weight
                        if weight else None}


def read_json(path: Path):
    def invalid(value):
        raise ValueError(f"non-finite JSON number: {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid)


def validate_replay_tokens(records: list[dict], index: dict) -> dict:
    """Reject shortened/misrendered successful replays; never trust raw budgets.

    A stream can end normally before its requested length without the public
    loadgen reporting an error. Such a run must not be compared as equal work.
    Failed requests keep their original error classification and rate checks.
    Counts are reported metadata, not independent proof of model execution.
    """
    checked = skipped = 0
    invalid = Counter()
    examples = {}
    for row in records:
        if row.get("error") or row.get("error_class"):
            skipped += 1
            continue
        rid = row["req_id"]
        frozen = index[rid]
        prompt = frozen.get("glm_tokens")
        # Exactly the original loadgen's default for an absent/zero budget.
        budget = frozen.get("max_output_i") or 512
        if type(prompt) is not int or prompt <= 0 or type(budget) is not int or budget <= 0:
            raise ValueError(f"invalid frozen token budget for {rid}")
        checked += 1
        failures = {
            "output_tokens": type(row.get("output_tokens")) is not int or row["output_tokens"] != budget,
            "prompt_tokens": type(row.get("prompt_tokens")) is not int or row["prompt_tokens"] != prompt,
            "cached_tokens": type(row.get("cached_tokens")) is not int
                             or not 0 <= row["cached_tokens"] <= prompt,
        }
        for field, failed in failures.items():
            if failed:
                invalid[field] += 1
                if len(examples.setdefault(field, [])) < 3:
                    examples[field].append(rid)
    if invalid:
        detail = "; ".join(f"{field} x{count} ({', '.join(examples[field])})"
                           for field, count in sorted(invalid.items()))
        raise ValueError("replay token contract mismatch: " + detail)
    return {"verified": True, "successful_checked": checked, "error_rows_skipped": skipped,
            "basis": "frozen glm_tokens/max_output_i; 0 <= cached_tokens <= prompt_tokens",
            "scope": "reported counts; errors remain subject to original harness gates"}


def resolve_inputs(raw: Path | None, run: Path | None, directory: Path | None) -> tuple[Path, Path]:
    if directory is not None:
        directory = directory.resolve()
        summary_path = directory / "summary.json"
        if summary_path.is_file():
            summary = read_json(summary_path)
            # run_dev can store relative paths; basenames are in its output directory.
            raw = directory / Path(summary["raw"]).name
            run = directory / Path(summary["run"]).name
        else:
            matches = sorted(directory.glob("raw_*.jsonl"))
            if len(matches) != 1:
                raise ValueError("run directory needs summary.json or exactly one raw_*.jsonl; choose --raw explicitly")
            raw = matches[0]
    if raw is None:
        raise ValueError("provide --raw or --run-dir")
    raw = raw.resolve()
    if run is None:
        if not raw.name.startswith("raw_"):
            raise ValueError("cannot infer run metadata: provide --run")
        run = raw.with_name("run_" + raw.name[4:]).with_suffix(".json")
    return raw, run.resolve()


def score_files(raw: Path, run: Path, harness: Path = DEFAULT_HARNESS,
                requests: Path | None = REPO / "s1-dev/data/dev-combined-v1/requests.jsonl") -> dict:
    records = []
    with raw.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"raw line {line_number} is not an object")
                if any(isinstance(v, float) and not math.isfinite(v) for v in record.values()):
                    raise ValueError(f"raw line {line_number} has non-finite values")
                records.append(record)
    scorer = load_harness(harness)
    replay_tokens = {"verified": False, "reason": "no frozen request index supplied"}
    if requests is not None:
        common = sys.modules["s1_common"]
        expected_index, _, _ = common.load_index(str(requests.parent))
        expected = {rid: row["_idx_in_chain"] for rid, row in expected_index.items()}
        counts = Counter(row.get("req_id") for row in records)
        duplicate = sum(n - 1 for n in counts.values() if n > 1)
        missing = len(set(expected) - set(counts))
        unknown = len(set(counts) - set(expected))
        wrong_index = sum(row.get("idx_in_chain") != expected[row["req_id"]]
                          for row in records if row.get("req_id") in expected)
        if duplicate or missing or unknown or wrong_index:
            raise ValueError("raw does not match full dev request index: "
                             f"duplicate={duplicate}, missing={missing}, unknown={unknown}, "
                             f"wrong_idx_in_chain={wrong_index}")
        replay_tokens = validate_replay_tokens(records, expected_index)
    report = score_records(records, read_json(run), scorer, requests)
    report["replay_tokens"] = replay_tokens
    report["paths"] = {"raw": str(raw.resolve()), "run": str(run.resolve()),
                       "dev_scorer": str(harness.resolve() / "s1_score.py")}
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--raw", type=Path)
    inputs.add_argument("--run-dir", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--requests", type=Path,
                        default=REPO / "s1-dev/data/dev-combined-v1/requests.jsonl",
                        help="frozen sampling_weight index; req_id join")
    parser.add_argument("--harness-dir", type=Path, default=DEFAULT_HARNESS)
    args = parser.parse_args(argv)
    try:
        raw, run = resolve_inputs(args.raw, args.run, args.run_dir)
        if args.out and args.out.resolve() in (raw, run, args.requests.resolve()):
            raise ValueError("output must not overwrite raw or run input")
        report = score_files(raw, run, args.harness_dir, args.requests)
        if args.out:
            write_json(args.out, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"score_formal: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
