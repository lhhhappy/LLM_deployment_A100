"""Install before server construction in a dedicated TP8 diagnostic job.

No normal-job env hook is added. A debug launcher calls install(output_dir)
before constructing the engine. It compares eager vs prepared glue on one live
batch per key, then restores every mutated input and metadata before model
replay. Every rank writes its own JSON receipt. This deliberately clones and
synchronizes; never use it in a scored/performance run.
"""
from __future__ import annotations

import dataclasses
import enum
import json
import os
from pathlib import Path
import time

import torch


def snapshot(glue, backend, limit_bytes=256 * 1024**2):
    result = {}
    used = 0

    def walk(path, value):
        nonlocal used
        if isinstance(value, torch.Tensor):
            used += value.numel() * value.element_size()
            if used > limit_bytes:
                raise RuntimeError("metadata snapshot exceeds 256MiB/rank budget")
            result[path] = value.detach().cpu().clone()
        elif dataclasses.is_dataclass(value):
            fields = {field.name for field in dataclasses.fields(value)}
            fields.update(vars(value))
            for name in sorted(fields):
                walk(f"{path}.{name}", getattr(value, name))
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                walk(f"{path}[{index}]", item)
        elif isinstance(value, dict):
            for key in sorted(value):
                walk(f"{path}.{key}", value[key])
        elif isinstance(value, enum.Enum):
            result[path] = (type(value).__name__, value.name)
        elif isinstance(value, (torch.dtype, torch.device)):
            result[path] = str(value)
        elif value is None or isinstance(value, (str, int, float, bool)):
            result[path] = value
        else:
            raise TypeError(f"snapshot cannot cover {path}: {type(value)}")

    for index, leaf in enumerate(glue._leaves(backend)):
        label = f"backend[{index}].{type(leaf).__name__}"
        walk(label + ".forward_metadata", leaf.forward_metadata)
        for name in ("use_mha", "dsa_prefill_impl"):
            if name in vars(leaf):
                walk(label + "." + name, getattr(leaf, name))
    return result, used


def compare(expected, actual):
    failures = []
    if set(expected) != set(actual):
        failures.append(dict(path="metadata structure", expected=sorted(expected), actual=sorted(actual)))
    for path in sorted(set(expected) & set(actual)):
        left, right = expected[path], actual[path]
        if isinstance(left, torch.Tensor):
            equal = (isinstance(right, torch.Tensor) and left.shape == right.shape
                     and left.dtype == right.dtype and torch.equal(left, right))
            if not equal:
                failures.append(dict(path=path, expected_shape=list(left.shape), expected_dtype=str(left.dtype),
                                     actual_shape=list(right.shape) if isinstance(right, torch.Tensor) else None))
        elif left != right:
            failures.append(dict(path=path, expected=left, actual=right))
    return failures


def install(output_dir):
    from sglang.srt.model_executor.runner.metadata_glue_graph import MetadataGlueGraph
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner
    from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_rank

    if getattr(MetadataGlueGraph, "_diagnostic_installed", False):
        return
    MetadataGlueGraph._diagnostic_installed = True
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    original_run = MetadataGlueGraph.run
    original_key = DecodeCudaGraphRunner._metadata_glue_key
    tested = set()

    def diagnostic_key(runner, backend, view):
        # Only this debug launcher exposes the model graph's independent mask.
        mask = runner.buffers.mamba_track_mask
        view._diagnostic_mamba_track_mask = None if mask is None else mask[:view.batch_size]
        return original_key(runner, backend, view)

    DecodeCudaGraphRunner._metadata_glue_key = diagnostic_key

    def checked_run(glue, backend, view, key):
        state = glue._states.get(key)
        if glue.disabled or state is None or key in tested:
            return original_run(glue, backend, view, key)
        tested.add(key)
        rank = get_tensor_model_parallel_rank()
        receipt = dict(rank=rank, key=repr(key), batch=view.batch_size, status="RUNNING", started_at=time.time(),
                       scope="metadata-only equality; synchronization invalidates performance timing",
                       prefix_mask_contract="extend_prefix_lens is extend-only and absent from the decode replay view. "
                       "The bool track mask is a separate live model-graph input, absent from metadata replay view. "
                       "Checkpoint kernels consume the mask together with metadata's physical track destinations. "
                       "This check compares metadata only; it does not prove state-checkpoint equality.")
        path = out / f"metadata-glue-rank{rank}-bs{view.batch_size}-{len(tested)}.json"
        restores = []

        def save(tensor):
            saved = tensor.clone()
            restores.append(lambda: tensor.copy_(saved))
            return saved

        def case(label):
            backend.init_forward_metadata_out_graph(view)
            expected, nbytes = snapshot(glue, backend)
            original_run(glue, backend, view, key)
            actual, _ = snapshot(glue, backend)
            failures = compare(expected, actual)
            receipt.setdefault("cases", []).append(dict(label=label, compared_fields=len(expected), cloned_bytes=nbytes,
                                                         pass_exact=not failures, mismatches=failures))
            if failures:
                raise RuntimeError(f"metadata glue equality failed: {label}: {failures[:8]}")

        try:
            if view.batch_size > 64:
                receipt["status"] = "SKIPPED_LARGE_BATCH"
                return original_run(glue, backend, view, key)
            case("live eager -> prepared glue")
            receipt["settled_memory_before_mutations"] = dict(allocated=torch.cuda.memory_allocated(glue.device),
                                                             reserved=torch.cuda.memory_reserved(glue.device))
            # Mutate each source separately so a miss is attributable.
            seq = save(view.seq_lens)
            seq_cpu = save(view.seq_lens_cpu) if view.seq_lens_cpu is not None else None
            seq_sum = view.seq_lens_sum
            restores.append(lambda: setattr(view, "seq_lens_sum", seq_sum))
            view.seq_lens.copy_((seq + 1).clamp(max=backend.full_attn_backend.req_to_token.shape[1]))
            if seq_cpu is not None:
                view.seq_lens_cpu.copy_(view.seq_lens.cpu())
            if seq_sum is not None:
                view.seq_lens_sum = int(view.seq_lens.sum().item())
            case("same-key seq_lens update")
            view.seq_lens.copy_(seq)
            if seq_cpu is not None:
                view.seq_lens_cpu.copy_(seq_cpu)
            view.seq_lens_sum = seq_sum

            slots = save(view.req_pool_indices)
            changed_slots = slots.roll(1) if slots.numel() > 1 else (slots + 1) % backend.full_attn_backend.req_to_token.shape[0]
            view.req_pool_indices.copy_(changed_slots)
            case("same-key req_pool_indices permutation")
            view.req_pool_indices.copy_(slots)

            req_ids = torch.unique(slots).to(torch.long)
            table = backend.full_attn_backend.req_to_token
            rows = table[req_ids].clone()
            restores.append(lambda: table.index_copy_(0, req_ids, rows))
            table.index_copy_(0, req_ids, rows.roll(1, dims=1))
            case("same-key live req_to_token rows update")
            table.index_copy_(0, req_ids, rows)

            mapping = backend.linear_attn_backend.req_to_token_pool.req_index_to_mamba_index_mapping
            original_mapping = mapping[req_ids].clone()
            restores.append(lambda: mapping.index_copy_(0, req_ids, original_mapping))
            changed_mapping = original_mapping.roll(1, dims=0) if original_mapping.numel() > 1 else (original_mapping + 1) % (backend.linear_attn_backend.req_to_token_pool.mamba_pool.size + 1)
            mapping.index_copy_(0, req_ids, changed_mapping)
            case("same-key live request-to-mamba mapping update")
            mapping.index_copy_(0, req_ids, original_mapping)

            if view.mamba_track_indices is not None:
                track = save(view.mamba_track_indices)
                view.mamba_track_indices.copy_(track.roll(1))
                case("same-key track destination update")
                view.mamba_track_indices.copy_(track)
            else:
                receipt["track_case"] = "not exercised: no track destination source"
            mask = view._diagnostic_mamba_track_mask
            if mask is not None:
                save(mask)
                for enabled in (False, True, False):
                    mask.fill_(enabled)
                    case(f"model bool track mask {enabled}: metadata remains independent")
            else:
                receipt["mask_case"] = "not exercised: no bool track mask source"
            receipt["status"] = "PASS_METADATA_EXACT"
        except Exception as error:
            receipt["status"] = "FAIL_METADATA_EXACT"
            receipt["error"] = repr(error)
            raise
        finally:
            for restore in reversed(restores):
                restore()
            # Restore original input's metadata even on failure; only metadata
            # prep was replayed here, never model/KV/KDA state update graphs.
            backend.init_forward_metadata_out_graph(view)
            torch.cuda.current_stream(glue.device).synchronize()
            receipt["restored_at"] = time.time()
            receipt["pid"] = os.getpid()
            path.write_text(json.dumps(receipt, indent=2) + "\n")

    MetadataGlueGraph.run = checked_run
