#!/usr/bin/env python3
"""Bounded CPU source probes; no engine imports, GPU, network or Pod access.

Execute selected methods from the committed source with explicit collaborators.
This checks local state transitions, not the complete EngineCore or DMA path.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[2]
REV = "19673ef4"
PREFIX = "engine/vllm/vllm/"
sources = {}


def source(relative):
    path = PREFIX + relative
    if path not in sources:
        body = subprocess.check_output(
            ["git", "show", f"{REV}:{path}"], cwd=ROOT, text=True
        )
        sources[path] = body
    return sources[path]


def function(relative, name, class_name=None):
    module = ast.parse(source(relative))
    scope = module.body
    if class_name:
        scope = next(n for n in scope if isinstance(n, ast.ClassDef)
                     and n.name == class_name).body
    node = next(n for n in scope if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    tree = ast.Module(body=[ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    ), node], type_ignores=[])
    ast.fix_missing_locations(tree)
    scope = {}
    exec(compile(tree, PREFIX + relative, "exec"), scope)
    return scope[name]


def main():
    kda = "models/glm5next/common/kda.py"
    resolve = function(kda, "_resolve_kda_prefill_backend")
    bf16 = object()
    resolve.__globals__["torch"] = NS(bfloat16=bf16)
    backends = {}
    for major in (8, 9):
        resolve.__globals__["current_platform"] = NS(
            is_cuda=lambda: True, get_device_capability=lambda: NS(major=major)
        )
        backends[str(major)] = resolve("auto", 128, bf16, -5.0)
    assert backends == {"8": "triton", "9": "flashkda"}
    assert "FlashKDAPrefillCheckpointExporter" not in source(kda)

    offload = "distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"
    reset = function(offload, "reset_cache", "OffloadingConnectorScheduler")
    pending = function(offload, "has_pending_push_work", "OffloadingConnectorScheduler")
    calls = []
    obj = NS(
        _current_batch_load_jobs={}, _current_batch_jobs_to_flush=set(),
        _current_batch_allocated_block_ids=set(), _jobs={16: object()},
        _req_status={}, _job_counter=17, _block_id_to_pending_jobs={2: {16}},
        _chunks_being_loaded={"old"},
        manager=NS(reset_cache=lambda: calls.append("manager_reset"),
                   has_pending_work=lambda: False),
        _events_tracker=NS(reset=lambda: calls.append("events_reset")),
    )
    assert pending(obj) is True
    reset(obj)
    after_reset = {
        "queued_worker_fences": sorted(obj._current_batch_jobs_to_flush),
        "has_pending_push_work": pending(obj),
        "stale_job_threshold": obj._stale_job_threshold,
        "tracked_jobs": len(obj._jobs), "calls": calls.copy(),
    }
    assert after_reset["queued_worker_fences"] == [16]
    assert after_reset["has_pending_push_work"] is False
    try:
        reset(obj)
    except AssertionError:
        after_reset["second_reset_before_worker_step"] = "AssertionError"
    else:
        raise AssertionError("Expected pending-fence guard to reject second reset")

    tiering = "v1/kv_offload/tiering/manager.py"
    tier_reset = function(tiering, "reset_cache", "TieringOffloadingManager")
    primary, secondary = {"old": "bytes"}, {"old": "bytes"}
    order = []

    def drain():
        order.append("secondary_drain")

    def clear_primary():
        order.append("primary_reset")
        primary.clear()

    tier = NS(drain_jobs=drain, bp_detector=None, cache=secondary,
              reset_cache=secondary.clear)
    mgr = NS(
        secondary_tiers=[tier], primary_tier=NS(reset_cache=clear_primary),
        _process_finished_jobs=lambda: order.append("consume_completions"),
        _jobs={}, _pending_load_submissions={"pending": object()},
        _metrics=NS(assert_idle=lambda: None), _req_state={},
    )
    tier_reset(mgr)
    assert order == ["secondary_drain", "consume_completions", "primary_reset"]
    assert not primary and tier.cache == {"old": "bytes"}
    result = {
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", REV], cwd=ROOT, text=True).strip(),
        "method": "AST-extracted committed functions with fake collaborators",
        "limits": "No full EngineCore, actual filesystem tier or GPU/DMA execution",
        "kda_auto_backend": backends,
        "offload_reset_with_unconsumed_worker_fence": after_reset,
        "tiering_reset": {"call_order": order, "primary_entries": len(primary),
                          "fake_secondary_entries": len(secondary)},
        "source_sha256": {p: hashlib.sha256(s.encode()).hexdigest()
                          for p, s in sorted(sources.items())},
    }
    body = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    assert len(body.encode()) < 64 * 1024
    Path(__file__).with_name("summary.json").write_text(body)
    print(body, end="")


if __name__ == "__main__":
    main()
