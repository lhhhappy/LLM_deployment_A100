#!/usr/bin/env python3
"""E2-only, TP1 raw-logit tap; never a submission entrypoint.

Runs the unchanged launch entrypoint with process-local Python wrappers. With an
armed control.json, saves the full first-output-token logits BEFORE sampling.
Also records actual D1 split depths, without prompt contents. No tensor is
modified. The CPU copy synchronizes CUDA, so this run is NOT a timing benchmark.
Top-level hooks intentionally reinstall in multiprocessing spawn children.
"""
import json
import os
from pathlib import Path
import re
import sys

import torch

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.managers.schedule_policy import PrefillAdder

TRACE = Path(os.environ["E2_TRACE_ROOT"]).resolve()
if not TRACE.is_relative_to(Path("/sjtu/linhang/arena/runs/E2_20260922")):
    raise RuntimeError("E2 trace writes must stay inside the authorized run")
TRACE.mkdir(parents=True, exist_ok=True)
_forward = LogitsProcessor.forward


def tapped_forward(self, input_ids, hidden_states, lm_head, logits_metadata,
                   *args, **kwargs):
    output = _forward(self, input_ids, hidden_states, lm_head, logits_metadata,
                      *args, **kwargs)
    if output.next_token_logits is None or not logits_metadata.forward_mode.is_extend():
        return output
    control = TRACE / "control.json"
    if not control.exists():
        return output
    spec = json.loads(control.read_text())  # writer uses atomic rename
    name = spec["name"]
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
        raise RuntimeError("unsafe trace name")
    destination = TRACE / (name + ".pt")
    if destination.exists():
        return output
    seq_lens = getattr(logits_metadata, "seq_lens_cpu", None)
    if seq_lens is None:
        seq_lens = logits_metadata.seq_lens.cpu()
    seq_lens = seq_lens.tolist()
    if seq_lens != [spec["prompt_tokens"]]:
        return output  # intermediate chunk or another request
    rids = getattr(logits_metadata, "rids", None)
    if rids is not None and rids != [spec["rid"]]:
        return output
    tensor = output.next_token_logits.detach().cpu().clone()
    if tensor.ndim != 2 or tensor.shape[0] != 1:
        raise RuntimeError("E2 numeric tap requires a serial TP1 batch")
    torch.save({"logits": tensor, "rid": spec["rid"], "batch_rids": rids,
                "seq_lens": seq_lens,
                "extend_prefix_lens": getattr(logits_metadata, "extend_prefix_lens_cpu", None),
                "extend_seq_lens": getattr(logits_metadata, "extend_seq_lens_cpu", None),
                "pid": os.getpid(), "metric": "pre-sampler full-vocabulary raw logits"},
               destination)
    return output


LogitsProcessor.forward = tapped_forward
if hasattr(PrefillAdder, "_maybe_role_boundary_split"):
    _split = PrefillAdder._maybe_role_boundary_split

    def tapped_split(self, req, admission, has_chunked_req):
        result = _split(self, req, admission, has_chunked_req)
        if result is not None:
            event = {"rid": req.rid, "prefix_len": result.prefix_len,
                     "split_depth": result.prefix_len + result.extend_len,
                     "original_end": admission.prefix_len + admission.extend_len,
                     "pid": os.getpid()}
            with (TRACE / "splits.jsonl").open("a") as f:
                f.write(json.dumps(event) + "\n")
        return result

    PrefillAdder._maybe_role_boundary_split = tapped_split


if __name__ == "__main__":
    from sglang.launch_server import (load_plugins, prepare_server_args, run_server,
                                     kill_process_tree)
    load_plugins()
    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
