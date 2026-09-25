"""CPU-only review receipts; reads original evidence and writes <= 1 MiB locally."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "s1-dev/harness"))
spec = importlib.util.spec_from_file_location("review_score", ROOT / "scripts/score_formal.py")
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


def real_replays():
    result = []
    for name, data in (
        ("L042", "s1-dev/data/dev-combined-v1"),
        ("L067-official_b_full_n30_shortwarm/N30", "data/s1-dev-longchain"),
        ("L068-official_b_pace_off_full_n30_shortwarm/N30", "data/s1-dev-longchain"),
        ("L069-official_b_pace_off_host64_full_n30_shortwarm/N30", "data/s1-dev-longchain"),
    ):
        directory = ROOT / "evidence" / name
        selected = json.loads((directory / "summary.json").read_text())
        raw, run = [directory / Path(selected[key]).name for key in ("raw", "run")]
        report = score.score_files(raw, run, requests=ROOT / data / "requests.jsonl")
        prior = json.loads((directory / "level_verdict.json").read_text())
        failures = [key for key, value in report["estimated"]["gates"].items() if not value]
        assert set(failures) == set(prior["failed_gates"]), name
        assert report["tpot"]["tpot_mean"] == prior["tpot_mean"], name
        assert report["tpot"]["tpot_p95"] == prior["tpot_p95"], name
        result.append({"run": name, "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                       "replay_tokens": report["replay_tokens"], "failed_gates": failures,
                       "prior_score_unchanged": True})
    return result


def scheduler_probes():
    """Run actual scheduler methods with fake state, never GPU/TP performance."""
    path = ROOT / "engine/sglang/srt/managers/scheduler.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Scheduler")
    wanted = {"is_fully_idle", "get_num_allocatable_reqs"}
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                             *[node for node in cls.body if getattr(node, "name", None) in wanted]], type_ignores=[])
    scope = {"DisaggregationMode": NS(PREFILL="prefill", DECODE="decode"),
             "get_parallel": lambda: NS(pp_max_micro_batch_size=32)}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    empty = NS(is_empty=lambda: True)
    tree_cache = NS(ongoing_write_through={}, ongoing_load_back={}, enable_storage=False)
    scheduler = NS(running_batch=empty, chunked_req=None,
                   dllm_manager=NS(any_staging_reqs=lambda: False), last_batch=None,
                   enable_overlap=True, result_queue=[], _pp_microbatches_drained=lambda: True,
                   waiting_queue=[], grammar_manager=NS(grammar_queue=[]),
                   disaggregation_mode="none", enable_hisparse=False,
                   enable_hierarchical_cache=True, tree_cache=tree_cache,
                   req_to_token_pool=NS(available_size=lambda: 100),
                   beam_coordinator=NS(pending_member_rows=lambda _: 0))
    idle = scope["is_fully_idle"]
    observations = {"idle_without_transfers": idle(scheduler)}
    for field in ("ongoing_write_through", "ongoing_load_back"):
        setattr(tree_cache, field, {1: object()})
        observations["idle_with_" + field] = idle(scheduler)
        setattr(tree_cache, field, {})
    assert observations == {"idle_without_transfers": True,
                            "idle_with_ongoing_write_through": False,
                            "idle_with_ongoing_load_back": False}
    allocated = {str(n): scope["get_num_allocatable_reqs"](scheduler, n) for n in (0, 30, 31, 32)}
    assert allocated == {"0": 32, "30": 2, "31": 1, "32": 0}
    return {"scheduler_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "flush_idle_gate": observations, "free_rows_with_microbatch_cap_32": allocated,
            "scope": "real Python methods, fake resources; no DMA, TP8 or speed claim"}


if __name__ == "__main__":
    receipt = {"real_replays": real_replays(), "scheduler": scheduler_probes(),
               "scorer_sha256": hashlib.sha256((ROOT / "scripts/score_formal.py").read_bytes()).hexdigest()}
    content = json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    assert len(content.encode()) <= 1024 * 1024
    Path(__file__).with_name("summary.json").write_text(content)
    print("4 full replays retain their scores; token contracts and CPU scheduler probes passed.")
