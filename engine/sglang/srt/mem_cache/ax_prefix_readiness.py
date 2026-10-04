"""128 producer admission: a CPU view of a *consumer's* native cache match.

The native multi-component validator has already shortened device_indices to
the last reusable KV/KDA checkpoint. full_kv_hit_length and a producer's private
prefix_indices are not substitutes. No tree nodes, tensors, locks or allocations
are retained here. Admission must refresh this view after its COW match.
"""

import os
from dataclasses import dataclass


ENABLED = os.environ.get("SGLANG_AX_PREFIX_PRODUCER", "0") == "1"


class ReservationUnavailable(Exception):
    """A speculative READY COW could not allocate; no admission was committed."""


@dataclass(frozen=True)
class PrefixReadiness:
    device: int
    full: int
    host: int
    mamba_host: int
    swa_host: int

    def reason(self, length: int, short: int) -> str:
        if self.host or self.mamba_host or self.swa_host:
            return "host_reload"
        if self.device <= 0:
            return "kv_without_checkpoint" if self.full else "cache_miss"
        tail = length - self.device
        if not 0 < tail <= short:
            return "checkpoint_tail_long" if self.full > self.device else "tail_long"
        return "ready"


def capture(req, result, max_prefix_len: int):
    """Only called after a real match (including FORCE_MISS and request limits)."""
    req._ax_prefix_match = PrefixReadiness(
        device=min(len(result.device_indices), max_prefix_len),
        full=int(result.full_kv_hit_length),
        host=int(result.host_hit_length),
        mamba_host=int(result.mamba_host_hit_length),
        swa_host=int(result.swa_host_hit_length),
    )


def view(req):
    return getattr(req, "_ax_prefix_match", None)


def domain(req):
    return getattr(req, "extra_key", None), getattr(req, "cache_salt", None)


def eligible(req):
    # These paths have additional cache identity/lifecycle contracts. They keep
    # native scheduling, never become a producer or acquire a dependency.
    return not (
        req.output_ids
        or getattr(req, "session", None)
        or getattr(req, "beam_group", None)
        or getattr(req, "input_embeds", None) is not None
        or getattr(req, "positional_embed_overrides", None) is not None
        or getattr(req, "multimodal_inputs", None) is not None
        or getattr(req, "lora_id", None)
    )


def held_owner(req, inserted, threshold):
    """Explain native LPM held using the representatives it actually inserted.

    Verify the in-batch matched span against representatives actually inserted.
    Unexplained held remains held. A shallow relation alone never makes READY.
    """
    n = max(1, threshold)
    for other in inserted:
        if (domain(req) == domain(other)
                and len(req.origin_input_ids) >= n
                and len(other.origin_input_ids) >= n
                and req.origin_input_ids[:n] == other.origin_input_ids[:n]):
            return other.rid
    return None
