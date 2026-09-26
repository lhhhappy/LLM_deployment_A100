#!/usr/bin/env python3
"""Reproduce the S1/S2 patch audit against an explicitly selected frozen tree.

Standard library only. Scheduler methods are production ASTs loaded by the
repository's CPU fixture; KV pools, clock observations, forwards and Req are
fakes. These are single-decision counterexamples, not CUDA correctness tests.
No engine source, running service, or original evidence is modified.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
S1 = "4f9d1f0b9c71522ad8e97413924baf513e6f4357"
S2 = "84dcca0ed84f949cf44acd0a5d2427b171d317a2"


def git(tree, *args):
    return subprocess.check_output(["git", "-C", str(tree), *args], text=True).strip()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def method(path, cls, name, ns):
    source = ast.parse(path.read_text())
    owner = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == cls)
    node = next(n for n in owner.body if getattr(n, "name", None) == name)
    node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    return ns[name]


def probes(tree):
    sys.path.insert(0, str(tree / "tests"))
    from test_ax_admission_scheduler import scheduler, cold, continuation
    from test_sched_protect_chain import step

    env = {k: v for k, v in os.environ.items() if not k.startswith("SGLANG_AX_")}
    env.update(SGLANG_AX_SCHED_PROTECT="1", SGLANG_AX_SCHED_COLD_CAP="6144",
               SGLANG_AX_SCHED_SHORT_TOKENS="8192", SGLANG_AX_DEADLINE_TIERS="1",
               SGLANG_AX_PACE_TPOT="0")
    results = {}

    def one_case(kind, on=True, waited=0, max_wait=None, full_park_budget=False):
        settings = dict(env, SGLANG_AX_DEADLINE_TIERS=str(int(on)))
        if max_wait is not None:
            settings["SGLANG_AX_DEADLINE_MAX_WAIT_WARM_S"] = str(max_wait)
        with patch.dict(os.environ, settings, clear=True):
            cont = continuation("cont", 65536, 131072)
            if kind == "warm":
                head = cold("warm_7k", 7000, waited=waited, cached=65536)
                available = 10**7
            else:
                head = cold("waiter", 1024, cached=65536 if kind == "host" else 0)
                available = 8192 if kind == "host" else 1250
                if kind == "host":
                    head.prefix_indices = []
                    head.host_hit_length = 65536
            s, ns = scheduler(chunk=cont, waiting=[head], budget=8192,
                              available=available, interval=2)
            if full_park_budget:
                plan = s._ax_admission_plan
                s._ax_admission_plan = lambda running, budget, kv: plan(running, 8192, kv)
            trace = step(s)
            trace.update(parked=getattr(cont, "_ax_parked_rounds", 0),
                         batch_full=s.running_batch.batch_is_full,
                         waiting=[r.rid for r in s.waiting_queue])
            return trace, ns, cont, head

    on, _, _, _ = one_case("kv")
    off, _, _, _ = one_case("kv", on=False)
    assert on["mode"] == "idle" and on["parked"] == 1
    assert off["mode"] == "prefill" and off["reqs"][0][0] == "cont"
    results["F1_park_without_output_reserve"] = {"124_on": on, "124_off": off}
    host, _, _, _ = one_case("host")
    assert host["mode"] == "idle" and host["parked"] == 1
    results["F1_park_without_host_load_room"] = host

    warm, ns, cont, head = one_case("warm", waited=1)
    ax = ns["ax_deadline"]
    cfg = ax.DeadlineConfig()
    by_budget = {str(b): ax.should_park(cont, 131072, 0, head, 1, b, 10**7, 0, 0, cfg)
                 for b in (6144, 8192)}
    assert by_budget == {"6144": False, "8192": True}
    assert warm["waiting"] == ["warm_7k"]
    full_budget, _, _, _ = one_case("warm", waited=1, full_park_budget=True)
    assert full_budget["reqs"][0][0] == "warm_7k" and not full_budget["waiting"]
    results["F2_cold_cap_used_for_warm_park"] = {
        "actual": warm, "parking_with_budget": by_budget,
        "full_round_budget_control": full_budget}

    starved, ns, cont, head = one_case("warm", waited=11, max_wait=10)
    ax = ns["ax_deadline"]
    cfg = ax.DeadlineConfig(max_wait_warm_s=10)
    fresh = cold("fresh", 1024)
    order = ax.tier_order([fresh, head], lambda r: 11 if r is head else 0,
                          set(), 8192, cfg)
    should_park = ax.should_park(cont, 131072, 0, head, 11, 8192, 10**7, 0, 0, cfg)
    assert order[0] is head and not should_park and starved["waiting"] == ["warm_7k"]
    results["F3_warm_wait_bound_is_priority_only"] = {
        "actual": starved, "order": [r.rid for r in order],
        "parking_even_with_full_budget": should_park}

    with patch.dict(os.environ, dict(env, SGLANG_AX_DEADLINE_FAMILY="1"), clear=True):
        a, b = cold("a", 12000), cold("b", 12000)
        a.extra_key = b.extra_key = None
        a.cache_salt, b.cache_salt = "tenant-a", "tenant-b"
        s, ns = scheduler(waiting=[a, b])
        deadline, _ = s._ax_admission_cfgs()
        work, held = s._ax_family_plan(s._ax_family_cfg, deadline)
        assert held == {"b"} and work["a"] == 6000
        results["F4_family_ignores_cache_namespace"] = {
            "work": work, "held": sorted(held),
            "cache_salts": [a.cache_salt, b.cache_salt],
            "prompt_tokens": len(a.origin_input_ids)}

    runner_path = tree / "engine/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py"
    comm_path = tree / "engine/sglang/srt/layers/communicator.py"
    # Extract both actual predicates. All unrelated graph checks are satisfied.
    comm_ast = ast.parse(comm_path.read_text())
    comm_cls = next(n.name for n in comm_ast.body if isinstance(n, ast.ClassDef)
                    and any(getattr(m, "name", "") == "use_input_scattered" for m in n.body))
    use_scatter = method(comm_path, comm_cls, "use_input_scattered", {})
    context = NS(allow_input_scattered=True, scatter_min_tokens=1025)
    context.use_input_scattered = lambda batch: use_scatter(context, batch)
    module = ModuleType("sglang.srt.layers.communicator")
    module.get_attn_tp_context = lambda: context
    run = method(runner_path, "PrefillCudaGraphRunner", "can_run_graph", {})

    class Tokens:
        def __init__(self, n):
            self.shape = (n,)
        def __len__(self):
            return self.shape[0]

    def batch(n):
        return NS(global_num_tokens_cpu=None, batch_size=1, input_ids=Tokens(n),
                  input_embeds=None, replace_embeds=None, extend_prefix_lens_cpu=[0],
                  forward_mode=NS(is_target_verify=lambda: False, is_extend=lambda: True),
                  capture_hidden_mode=None, return_logprob=False, can_run_tbo=False)

    runner = NS(_has_inactive_dp_rank=lambda _: False, can_replay_locally=lambda **kw: True,
                enable_lora=False, _uses_eager_prefill_tail=lambda: True)
    captures = [(4096, context.use_input_scattered(batch(4096))),
                (1024, context.use_input_scattered(batch(1024)))]
    # Production capture writes this single field in descending bucket order.
    runner._ax170_capture_input_scattered = captures[-1][1]
    with patch.dict(sys.modules, {module.__name__: module}):
        small, large = run(runner, batch(1024)), run(runner, batch(4096))
        runner._ax170_capture_input_scattered = captures[0][1]
        corrected_large = run(runner, batch(4096))
    assert small and not large and corrected_large
    results["F5_scatter_flag_is_not_per_graph_bucket"] = {
        "capture_order": captures, "small_replay": small, "large_replay": large,
        "large_replay_with_its_own_flag": corrected_large}
    return results


def comparisons(out, raw_root):
    """Original harness selectors, common completed IDs; never a full verdict."""
    sys.path.insert(0, str(raw_root / "scripts"))
    import score_formal
    scorer = score_formal.load_harness()
    paths = {
        "111": "evidence/L111-v3_n30_base_60m/N30/raw_s1-dev-longchain-v3_N30_1790411861.jsonl",
        "112": "evidence/L112-v3_n30_124_125x_117_60m/N30/raw_s1-dev-longchain-v3_N30_1790416026.jsonl",
        "130b": "evidence/L130b-v3_n30_A_warm15_60m/N30/raw_s1-dev-longchain-v3_N30_1790423235.jsonl",
        "130a": "evidence/L130a-v3_open_A_warm15_n30/window/raw.jsonl",
        "130c": "evidence/L130c-v3_open_A_warm15_126_n30/window/raw.jsonl",
    }
    rows, receipts = {}, {}
    for name, relative in paths.items():
        path = raw_root / relative
        data = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        by_id = {r["req_id"]: r for r in data}
        if len(data) != len(by_id):
            raise ValueError(f"duplicate request IDs in {path}")
        rows[name] = by_id
        receipts[name] = {"path": str(path), "sha256": sha(path), "rows": len(data),
                          "tpot_gt_100ms": sum(r.get("tpot_s") is not None and r["tpot_s"] > .1 for r in data)}
    pairs = {}
    for before, after in (("111", "112"), ("112", "130b"), ("130a", "130c")):
        ids = sorted(rows[before].keys() & rows[after].keys())
        a, b = rows[before], rows[after]
        gates = {}
        for name, selector, limit in scorer.TTFT_GATE_SPECS:
            selected = [rid for rid in ids if scorer.in_ttft_gate(a[rid], selector)]
            if any(scorer.in_ttft_gate(a[rid], selector) != scorer.in_ttft_gate(b[rid], selector) for rid in ids):
                raise ValueError("bucket membership changed")
            gates[selector] = {"n": len(selected), "limit_s": limit,
                               "before": sum(a[rid]["ttft_s"] > limit for rid in selected),
                               "after": sum(b[rid]["ttft_s"] > limit for rid in selected),
                               "fixed": sum(a[rid]["ttft_s"] > limit >= b[rid]["ttft_s"] for rid in selected),
                               "new": sum(b[rid]["ttft_s"] > limit >= a[rid]["ttft_s"] for rid in selected)}
        pairs[f"{before}_to_{after}"] = {
            "common_ids": len(ids), "before_only": len(a.keys() - b.keys()),
            "after_only": len(b.keys() - a.keys()), "gates": gates,
            "tpot_gt_100ms": [sum(data[rid].get("tpot_s") is not None and data[rid]["tpot_s"] > .1 for rid in ids) for data in (a, b)]}
        with (out / f"{before}_to_{after}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["req_id", "ttft_before", "ttft_after", "tpot_before", "tpot_after",
                             "queue_before", "queue_after", "cached_before", "cached_after", "selectors"])
            for rid in ids:
                writer.writerow([rid, a[rid].get("ttft_s"), b[rid].get("ttft_s"),
                                 a[rid].get("tpot_s"), b[rid].get("tpot_s"),
                                 a[rid].get("queue_time_s"), b[rid].get("queue_time_s"),
                                 a[rid].get("cached_tokens"), b[rid].get("cached_tokens"),
                                 ";".join(selector for _, selector, _ in scorer.TTFT_GATE_SPECS if scorer.in_ttft_gate(a[rid], selector))])
    log = raw_root / "evidence/L112-v3_n30_124_125x_117_60m/N30/server.log"
    relief_lines = [line for line in log.read_text().splitlines() if "[ax-125] relief " in line]
    if not relief_lines:
        raise ValueError("no relief transition lines found")
    (out / "112-relief-transitions.txt").write_text("\n".join(relief_lines) + "\n")
    result = {"scope": "common completed IDs in partial dispatch windows; no full score or causal attribution",
              "raw_root": str(raw_root),
              "harness_sha256": sha(raw_root / "s1-dev/harness/s1_score.py"),
              "common_sha256": sha(raw_root / "s1-dev/harness/s1_common.py"),
              "sources": receipts, "pairs": pairs, "112_server_log_sha256": sha(log)}
    (out / "comparisons.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


def graph_receipts(tree, out):
    queries = [
        ("callees", "Scheduler._get_new_batch_prefill_raw"),
        ("callees", "Scheduler._ax_admission_plan"),
        ("callers", "BacklogState.note_tpot"),
        ("callees", "PrefillAdder.add_one_req"),
        ("callers", "fp8_mqa_logits"),
        ("callees", "IndexerKPool._get_topk_ragged_kpool_plan"),
        ("callees", "Fp8HummingMoEMethod.apply"),
        ("callers", "PrefillCudaGraphRunner.can_run_graph"),
        ("callers", "attach_hybrid_pool_to_unified_cache"),
        ("callers", "AsyncTextTokenizer.run"),
    ]
    dest = out / "codegraph"
    dest.mkdir(exist_ok=True)
    manifest = {"commit": S2, "version": subprocess.check_output(["codegraph", "--version"], text=True).strip(),
                "caveat": "Static candidates only. Python dynamic dispatch and deep_gemm monkey patches need manual receiver/source validation.",
                "queries": []}
    for command, symbol in queries:
        args = ["codegraph", command, symbol, "-j", "-l", "80"]
        process = subprocess.run(args, cwd=tree, env=dict(os.environ, CODEGRAPH_NO_DAEMON="1"), capture_output=True, text=True)
        filename = f"{command}-{symbol}.json"
        (dest / filename).write_text(process.stdout)
        manifest["queries"].append({"command": args, "returncode": process.returncode,
                                    "stderr": process.stderr, "file": filename})
        if process.returncode:
            raise RuntimeError(process.stderr)
    (dest / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


def family_benchmark(tree, out):
    from test_ax_admission_scheduler import scheduler, cold
    results = []
    # Local CPU only; measures rank-0 helper wall time, excludes request creation.
    for count in (8, 30, 64):
        with patch.dict(os.environ, {"SGLANG_AX_DEADLINE_TIERS": "1", "SGLANG_AX_DEADLINE_FAMILY": "1"}):
            reqs = [cold(f"r{i:02d}", 250000) for i in range(count)]
            s, _ = scheduler(waiting=reqs)
            deadline, _ = s._ax_admission_cfgs()
            samples = []
            for _ in range(4):
                start = time.perf_counter()
                s._ax_family_plan(s._ax_family_cfg, deadline)
                samples.append((time.perf_counter() - start) * 1000)
            results.append({"requests": count, "prompt_tokens_each": 250000,
                            "shared_prefix": "entire prompt; identical synthetic tokens",
                            "first_ms": samples[0], "cached_ms": samples[1:]})
    (out / "family-cpu.json").write_text(json.dumps({"scope": "local CPU synthetic upper sharing case, not TP8 service timing", "samples": results}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--comparisons", action="store_true")
    parser.add_argument("--raw-root", type=Path, default=ROOT,
                        help="repository containing original replay evidence and s1-dev")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--family-benchmark", action="store_true")
    args = parser.parse_args()
    tree, out = args.tree.resolve(), args.out.resolve()
    head = git(tree, "rev-parse", "HEAD")
    if git(tree, "diff", S2, "HEAD", "--", "engine/sglang", "tests"):
        raise ValueError(f"audit expects engine/tests identical to frozen S2 {S2}")
    if git(tree, "diff", "HEAD", "--", "engine/sglang", "tests"):
        raise ValueError("frozen engine or test fixtures have local changes")
    out.mkdir(parents=True, exist_ok=True)
    commits = []
    for line in git(tree, "log", "--reverse", "--format=%H%x09%s", f"engine-base..{S2}", "--", "engine/sglang").splitlines():
        commit, title = line.split("\t", 1)
        files = git(tree, "diff-tree", "--no-commit-id", "--name-only", "-r", commit, "--", "engine/sglang").splitlines()
        commits.append({"commit": commit, "title": title, "engine_files": files})
    changed = git(tree, "diff", "--name-only", "engine-base", S2, "--", "engine/sglang").splitlines()
    inventory = {"s1": S1, "s2": S2, "base": git(tree, "rev-parse", "engine-base"),
                 "s1_is_ancestor": subprocess.run(["git", "-C", str(tree), "merge-base", "--is-ancestor", S1, S2]).returncode == 0,
                 "commit_count": len(commits), "source_file_count": len(changed),
                 "commits": commits, "source_sha256": {p: sha(tree / p) for p in changed},
                 "s2_only_diff": git(tree, "diff", S1, S2, "--", "engine/sglang", "tests")}
    (out / "inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n")
    result = {"scope": "CPU decision counterexamples; fake pools/Req/forwards, no GPU claims",
              "tree": str(tree), "engine_commit": S2, "review_head": head, "script_sha256": sha(Path(__file__)),
              "reproduced": probes(tree)}
    (out / "probes.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if args.comparisons:
        comparisons(out, args.raw_root.resolve())
    if args.graph:
        graph_receipts(tree, out)
    if args.family_benchmark:
        family_benchmark(tree, out)
    print(json.dumps({"commits": len(commits), "source_files": len(changed),
                      "reproduced_probes": list(result["reproduced"]), "out": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
