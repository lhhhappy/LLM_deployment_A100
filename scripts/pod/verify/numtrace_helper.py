"""052 target-prefill tensor fingerprints; copied into a prepare_src package."""

import hashlib
import json
import math
import os
from pathlib import Path

import torch


_active = None
_sequence = 0


def _bytes(tensor):
    # Clone before D2H so later in-place residual/allreduce work cannot change
    # the value attributed to this stage. CPU copy synchronizes this stream.
    snapshot = tensor.detach().contiguous().clone()
    return snapshot.reshape(-1).view(torch.uint8).cpu().numpy().tobytes(), snapshot


def _fingerprint(tensor):
    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"numtrace stage expected tensor, got {type(tensor)}")
    data, snapshot = _bytes(tensor)
    flat = snapshot.reshape(-1)
    # Keep a record even if the tensor itself contains NaN/Inf: that may be the
    # fault being located. JSON non-finite literals are intentionally avoided.
    head = [v if math.isfinite(v) else str(v) for v in flat[:8].float().cpu().tolist()]
    tail = [v if math.isfinite(v) else str(v) for v in flat[-8:].float().cpu().tolist()]
    return {"dtype": str(tensor.dtype), "shape": list(tensor.shape),
            "stride": list(tensor.stride()), "sha256": hashlib.sha256(data).hexdigest(),
            "first8": head, "last8": tail}


def _parameter_digest(model):
    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    for name, param in model.named_parameters():
        value = param.detach()
        # A cheap sentinel for gross mutation or accidental reload. It is not
        # an exact checksum of all parameter bytes; use full hashes only after
        # the first divergent module has been found.
        if value.is_contiguous():
            flat = value.reshape(-1).view(torch.uint8)
            sample = torch.cat((flat[:64].clone(), flat[-64:].clone())).cpu().numpy().tobytes()
        else:
            # Do not materialize a non-contiguous full parameter just to sample.
            first = value[tuple(0 for _ in value.shape)]
            last = value[tuple(-1 for _ in value.shape)]
            sample = torch.stack((first.float(), last.float())).cpu().numpy().tobytes()
        digest.update(name.encode())
        digest.update(str(param.dtype).encode())
        digest.update(str(tuple(param.shape)).encode())
        try:
            version = param._version
        except RuntimeError:  # inference tensors do not expose version counters
            version = "unavailable"
        digest.update(str(version).encode())
        digest.update(sample)
        total_bytes += param.numel() * param.element_size()
        count += 1
    return {"sha256": digest.hexdigest(), "parameter_tensors": count,
            "parameter_bytes": total_bytes,
            "scope": "sample-not-full: all local parameter structure/version plus first/last bytes; not a full weight proof"}


def _rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _write(record):
    with _active["path"].open("a") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")


def begin(model, input_ids, positions, forward_batch, embedding):
    global _active, _sequence
    _active = None
    directory = os.environ.get("SGLANG_AX_NUMTRACE_DIR")
    if not directory:
        return
    # The job creates ARMED only after the engine is ready. This excludes
    # startup warmup and graph-capture calibration even if their M is 37/256.
    if not (Path(directory) / "ARMED").is_file():
        return
    if forward_batch.forward_mode.name != "EXTEND":
        return
    if torch.cuda.is_current_stream_capturing():
        return
    # general_mm_embed_routine always embeds first (also for text-only input)
    # and invokes Glm5NextModel(input_ids=None, input_embeds=...). The original
    # batch still owns the exact token IDs used to form that embedding.
    id_source = "model_argument"
    if input_ids is None:
        input_ids = forward_batch.input_ids
        id_source = "forward_batch"
    if input_ids is None:
        raise RuntimeError("armed target EXTEND has no token IDs")
    if input_ids.numel() not in (37, 256):
        return
    if embedding.shape[0] != input_ids.numel():
        raise RuntimeError("numtrace token IDs and embedding row count differ")
    _sequence += 1
    rank = _rank()
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    _active = {"path": out_dir / f"rank-{rank}.jsonl", "rank": rank, "forward": _sequence,
               "tokens": int(input_ids.numel()),
               "expected_layers": list(range(model.start_layer, model.end_layer)),
               "observed_layers": set()}
    _write({"kind": "begin", "rank": rank, "forward": _sequence,
            "tokens": _active["tokens"], "mode": forward_batch.forward_mode.name,
            "input_id_source": id_source,
            "expected_layers": _active["expected_layers"],
            "input_ids": _fingerprint(input_ids), "positions": _fingerprint(positions),
            "parameters": _parameter_digest(model)})
    capture(-1, "embedding", embedding)


def capture(layer, stage, tensor):
    if _active is None:
        return
    if stage == "mlp_input":
        _active["mlp_layer"] = layer
    record = {"kind": "stage", "rank": _active["rank"],
              "forward": _active["forward"], "tokens": _active["tokens"],
              "layer": layer, "stage": stage, "tensor": _fingerprint(tensor)}
    if stage == "layer_exit":
        _active["observed_layers"].add(layer)
    _write(record)
    dump_layer = os.environ.get("SGLANG_AX_NUMTRACE_DUMP_LAYER")
    # Full expert weights are hashed above; do not write multi-GB duplicates.
    weight_stage = stage in {"moe_w1", "moe_w2", "moe_w1_scale", "moe_w2_scale"}
    if dump_layer is not None and str(layer) == dump_layer and tensor is not None and not weight_stage:
        name = f"rank-{_active['rank']}-forward-{_active['forward']}-layer-{layer}-{stage}.pt"
        torch.save(tensor.detach().contiguous().clone().cpu(), _active["path"].parent / name)


def capture_moe(stage, tensor):
    # First MoE layer only. Weight fingerprints here cover the complete local
    # expert tensors; the model-level parameter receipt remains a sample.
    if _active is not None and _active.get("mlp_layer") == 3:
        capture(3, "moe_" + stage, tensor)


def capture_alignment(sorted_ids, expert_ids, padded_count, block_size):
    if _active is None or _active.get("mlp_layer") != 3:
        return
    count = int(padded_count.item())
    if count < 0 or count > sorted_ids.numel() or count % block_size:
        raise RuntimeError("invalid MoE alignment extent")
    capture_moe("sorted_ids", sorted_ids[:count])
    capture_moe("expert_ids", expert_ids[:count // block_size])
    capture_moe("padded_count", padded_count)


def finish(hidden_states, residual):
    global _active
    if _active is None:
        return
    capture(-1, "pre_final_norm", hidden_states)
    capture(-1, "model_residual", residual)
    missing = sorted(set(_active["expected_layers"]) - _active["observed_layers"])
    _write({"kind": "end", "rank": _active["rank"], "forward": _active["forward"],
            "tokens": _active["tokens"],
            "observed_layers": sorted(_active["observed_layers"]), "missing_layers": missing})
    _active = None
    if missing:
        raise RuntimeError(f"numtrace missed decoder layers: {missing}")
