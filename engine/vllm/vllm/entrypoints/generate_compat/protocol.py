# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request parsing and response metadata for the SGLang-style ``/generate``.

The request body follows SGLang's native ``/generate``: a pre-rendered
``text`` (or ``input_ids``), a ``sampling_params`` dict using SGLang field
names, ``stream`` and an optional ``rid``. The text is fed to the model as is:
no chat template is applied. Unknown fields are rejected instead of being
silently ignored, so a client never believes a setting took effect when it
did not.
"""

from dataclasses import dataclass
from typing import Any

from vllm.exceptions import VLLMClientError
from vllm.outputs import CompletionOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams

# SGLang's default when `max_new_tokens` is absent.
DEFAULT_MAX_NEW_TOKENS = 128

# SGLang sampling field -> SamplingParams field. Only fields whose meaning is
# the same in both engines are accepted.
_SAMPLING_FIELDS: dict[str, str] = {
    "max_new_tokens": "max_tokens",
    "min_new_tokens": "min_tokens",
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "frequency_penalty": "frequency_penalty",
    "presence_penalty": "presence_penalty",
    "repetition_penalty": "repetition_penalty",
    "stop": "stop",
    "stop_token_ids": "stop_token_ids",
    "ignore_eos": "ignore_eos",
    "skip_special_tokens": "skip_special_tokens",
    "spaces_between_special_tokens": "spaces_between_special_tokens",
    "n": "n",
    "sampling_seed": "seed",
}

_REQUEST_FIELDS = frozenset({"text", "input_ids", "sampling_params", "stream", "rid"})


class GenerateRequestError(ValueError):
    """The request body is not a valid SGLang-style generate request."""


@dataclass
class GenerateRequest:
    text: str | None
    input_ids: list[int] | None
    sampling_params: SamplingParams
    stream: bool
    rid: str | None

    def prompt(self) -> dict[str, Any]:
        if self.text is not None:
            return {"prompt": self.text}
        return {"prompt_token_ids": self.input_ids}


def parse_sampling_params(raw: object, stream: bool) -> SamplingParams:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise GenerateRequestError("sampling_params must be an object")
    unknown = sorted(set(raw) - set(_SAMPLING_FIELDS))
    if unknown:
        raise GenerateRequestError(f"unsupported sampling_params: {unknown}")
    kwargs = {_SAMPLING_FIELDS[k]: v for k, v in raw.items() if v is not None}
    kwargs.setdefault("max_tokens", DEFAULT_MAX_NEW_TOKENS)
    if kwargs.get("n", 1) != 1:
        raise GenerateRequestError("only n=1 is supported")
    kwargs["output_kind"] = (
        RequestOutputKind.DELTA if stream else RequestOutputKind.FINAL_ONLY
    )
    try:
        return SamplingParams(**kwargs)
    except (TypeError, ValueError, VLLMClientError) as e:
        raise GenerateRequestError(f"invalid sampling_params: {e}") from e


def parse_generate_request(body: object) -> GenerateRequest:
    if not isinstance(body, dict):
        raise GenerateRequestError("request body must be a JSON object")
    unknown = sorted(set(body) - _REQUEST_FIELDS)
    if unknown:
        raise GenerateRequestError(f"unsupported fields: {unknown}")

    text = body.get("text")
    input_ids = body.get("input_ids")
    if (text is None) == (input_ids is None):
        raise GenerateRequestError("exactly one of text and input_ids is required")
    if text is not None and not isinstance(text, str):
        raise GenerateRequestError("text must be a string")
    if input_ids is not None and not (
        isinstance(input_ids, list)
        and input_ids
        and all(isinstance(t, int) and not isinstance(t, bool) for t in input_ids)
    ):
        raise GenerateRequestError("input_ids must be a non-empty list of ints")

    stream = body.get("stream", False)
    if not isinstance(stream, bool):
        raise GenerateRequestError("stream must be a boolean")
    rid = body.get("rid")
    if rid is not None and not isinstance(rid, str):
        raise GenerateRequestError("rid must be a string")

    return GenerateRequest(
        text=text,
        input_ids=input_ids,
        sampling_params=parse_sampling_params(body.get("sampling_params"), stream),
        stream=stream,
        rid=rid,
    )


def finish_reason_info(
    output: CompletionOutput, num_output_tokens: int
) -> dict[str, Any] | None:
    """SGLang's ``finish_reason`` object for a completion.

    ``num_output_tokens`` is the cumulative output length; in delta mode
    ``output.token_ids`` holds only the newest tokens.
    """
    reason = output.finish_reason
    if reason is None:
        return None
    if reason == "length":
        return {"type": "length", "length": num_output_tokens}
    if reason == "stop":
        return {"type": "stop", "matched": output.stop_reason}
    return {"type": reason}


@dataclass
class ResponseTiming:
    """Server-side timestamps reported in ``meta_info`` (epoch seconds).

    ``request_received_ts`` is taken when the route handler starts, before the
    body is read. Engine-core timestamps are CLOCK_MONOTONIC; they are placed
    on the wall clock through the receive instant, which is valid because the
    API server and the engine core run on the same host. A converted value
    outside [receive, now] means that assumption broke; it is then not used.
    """

    received_wall: float
    received_mono: float
    dispatch_finish_wall: float | None = None
    first_token_wall: float | None = None

    def engine_to_wall(self, mono_ts: float, now_wall: float) -> float | None:
        if not mono_ts:
            return None
        wall = self.received_wall + (mono_ts - self.received_mono)
        if wall < self.received_wall or wall > now_wall:
            return None
        return wall
