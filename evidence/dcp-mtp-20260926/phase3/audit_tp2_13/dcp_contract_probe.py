#!/usr/bin/env python3
"""Small CPU witnesses for DCP contracts in a frozen SGLang source tree.

Run with a PyTorch environment. Execute the named, unmodified source functions;
stub only their surrounding pool/collective/config objects. This is diagnostic,
not a distributed numerical test or a service score. JSON includes source hashes.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch


def source_function(root, path, name, namespace, hashes, *, owner=None):
    source = root / path
    data = source.read_bytes()
    hashes[path] = hashlib.sha256(data).hexdigest()
    tree = ast.parse(data)
    nodes = tree.body
    if owner:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == owner).body
    fn = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[fn], type_ignores=[])
    env = {"torch": torch, "Any": object, **namespace}
    # Keep annotations deferred as in the source files; bodies are unchanged.
    exec(compile(module, str(source), "exec", flags=__import__("__future__").annotations.compiler_flag), env)
    return env[name]


def probe(root):
    hashes, cases = {}, []
    gather = source_function(
        root, "srt/layers/dcp/comm.py", "all_gather_kv_cache_for_mla_extend",
        {"all_gather_kv_cache_for_dcp": lambda k, r, *a, **kw: k}, hashes,
    )
    pool = SimpleNamespace(get_mla_kv_buffer=lambda *a: (torch.empty(0, 1, 512), None))
    for logical, padded in ((32, 32), (33, 40), (33, 34)):
        try:
            gather(pool, None, [0], torch.empty(0, dtype=torch.int64), 0,
                   torch.empty(logical, 1, 512), 512,
                   torch.ones(padded, 1, 512), torch.empty(padded, 1, 0))
            cases.append({"case": "extend_padding", "logical": logical,
                          "padded": padded, "outcome": "completed"})
        except RuntimeError as exc:
            cases.append({"case": "extend_padding", "logical": logical,
                          "padded": padded, "outcome": "runtime_error", "error": str(exc)})

    phase = source_function(
        root, "srt/models/deepseek_common/attention_forward_methods/forward_mla.py",
        "is_dcp_mla_decode_phase",
        {"get_parallel": lambda: SimpleNamespace(dcp_enabled=True)}, hashes,
    )
    lse = source_function(root, "srt/layers/attention/dsa_backend.py",
                          "_should_return_dsa_dcp_lse", {}, hashes)
    for label in ("DECODE", "TARGET_VERIFY", "DRAFT_EXTEND_V2"):
        mode = SimpleNamespace(is_decode=lambda: label == "DECODE",
                               is_target_verify=lambda: label == "TARGET_VERIFY",
                               is_draft_extend_v2=lambda: label == "DRAFT_EXTEND_V2")
        cases.append({"case": "phase_dispatch", "mode": label,
                      "q_gather_and_merge": phase(SimpleNamespace(forward_mode=mode)),
                      "dsa_returns_lse": lse(forward_mode=mode, dcp_enabled=True)})

    move = source_function(root, "srt/mem_cache/index_key_cache.py", "move",
                           {"get_parallel": lambda: SimpleNamespace(dcp_enabled=True)},
                           hashes, owner="IndexKeyCache")
    page, dim, scales = 64, 128, 4
    width = page * (dim + scales)
    original = (torch.arange(4 * width) % 251).to(torch.uint8).reshape(4, width)
    for dst, src in ((1, 2), (65, 66)):
        actual = original.clone()
        expected = original.clone()
        # IndexKeyCache stores all key bytes, followed by all scale bytes, per page.
        for offset, stride in ((0, dim), (page * dim, scales)):
            d = offset + dst % page * stride
            s = offset + src % page * stride
            expected[dst // page, d:d + stride] = original[src // page, s:s + stride]
        try:
            move(SimpleNamespace(buffer=[actual], pool=SimpleNamespace(
                page_size=page, index_head_dim=dim, quant_block_size=128)),
                torch.tensor([dst]), torch.tensor([src]))
            cases.append({"case": "indexer_token_move", "dst": dst, "src": src,
                          "outcome": "completed", "correct": torch.equal(actual, expected),
                          "different_bytes": int((actual != expected).sum())})
        except (RuntimeError, IndexError) as exc:
            cases.append({"case": "indexer_token_move", "dst": dst, "src": src,
                          "outcome": "index_error", "error": str(exc)})

    hf = SimpleNamespace(architectures=["Glm5NextForConditionalGeneration"],
                         num_nextn_predict_layers=1)
    cfg = SimpleNamespace(dcp_size=2, enable_hierarchical_cache=True,
                          hicache_storage_backend=None, enable_lmcache=False,
                          enable_hisparse=False, speculative_algorithm=None,
                          speculative_eagle_topk=1, model_path="/model",
                          speculative_draft_model_path="/model")
    guard = source_function(
        root, "srt/arg_groups/hicache_hook.py", "resolve_hicache_dcp_compatibility",
        {"resolving_view": lambda _: cfg, "use_mla_backend": lambda _: True,
         "model_config_of": lambda _: SimpleNamespace(hf_config=hf),
         "logger": SimpleNamespace(info=lambda *a: None)}, hashes,
    )
    for algorithm in (None, "EAGLE", "NEXTN", "DSPARK"):
        cfg.speculative_algorithm = algorithm
        try:
            guard(None)
            cases.append({"case": "hicache_guard", "spec": algorithm, "outcome": "accepted"})
        except NotImplementedError as exc:
            cases.append({"case": "hicache_guard", "spec": algorithm,
                          "outcome": "rejected", "error": str(exc)})
    return {"validity": "DIAGNOSTIC", "device": "cpu", "torch_version": torch.__version__,
            "source_sha256": hashes, "cases": cases,
            "limitations": "Extracted real function bodies, synthetic fixtures; no distributed forward, GPU numerical, performance, or SLO claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.source_root)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
