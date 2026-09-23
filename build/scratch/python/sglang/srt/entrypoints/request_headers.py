"""Override object fields based on _HEADER_OVERRIDES from header values.

This mechanism allows upstream callers to leave the body opaque
(no parse/merge/re-serialize).
"""

from fastapi import HTTPException

# request header -> (target attribute, value type)
_HEADER_OVERRIDES = {
    "x-override-rid": ("rid", str),
    "x-override-bootstrap-host": ("bootstrap_host", str),
    "x-override-bootstrap-port": ("bootstrap_port", int),
    "x-override-bootstrap-room": ("bootstrap_room", int),
    "x-override-conversation-id": ("conversation_id", str),
    "x-override-routed-dp-rank": ("routed_dp_rank", int),
    "x-override-disagg-prefill-dp-rank": ("disagg_prefill_dp_rank", int),
    "x-override-priority": ("priority", int),
}


def apply_header_overrides(obj, headers) -> None:
    """Override request based on header values. Fail the request when any override has issues."""
    for header, (attr, cast) in _HEADER_OVERRIDES.items():
        value = headers.get(header)
        if value is None:
            continue
        try:
            setattr(obj, attr, cast(value))
        except ValueError as e:
            raise HTTPException(
                status_code=400, detail=f"invalid {header} header {value!r}: {e}"
            ) from e


def apply_s1_routing_key(obj, headers) -> None:
    """Use S1 affinity hints only when the body has no explicit routing key.

    Session-ID is a routing fallback, never native session_params/cache_salt.
    Starlette Headers provides case-insensitive lookups. Ignore empty hints.
    """
    if obj.routing_key is None:
        obj.routing_key = headers.get("x-s1-routing-key") or headers.get(
            "x-s1-session-id"
        ) or None
