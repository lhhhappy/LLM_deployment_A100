# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SGLang-compatible ``POST /generate`` and ``/flush_cache`` routes.

``/generate`` streams SSE events ``data: {"text": <delta>, "meta_info": {...}}``
and ends with ``data: [DONE]``. Every event carries the cumulative
``completion_tokens``, the prompt and cached token counts reported by the
engine, and the server receive / first-token timestamps, so a client can take
the first-token fields from any event with tokens and the receive time from
any event. ``/flush_cache`` resets the prefix cache including any connector
(host/offload) tier and reports failure when blocks are still held.
"""

import json
import time
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.types import ASGIApp, Receive, Scope, Send

import vllm
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.generate_compat.protocol import (
    GenerateRequestError,
    ResponseTiming,
    finish_reason_info,
    parse_generate_request,
)
from vllm.exceptions import GracefulHTTPError, VLLMClientError
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.renderers.inputs.preprocess import parse_model_prompt
from vllm.utils import random_uuid
from vllm.v1.metrics.stats import RequestStateStats

logger = init_logger(__name__)

router = APIRouter()

# ASGI scope key holding (wall, monotonic) taken when the request reached the app.
RECEIVED_AT = "generate_compat.received_at"


class ReceiveTimeMiddleware:
    """Stamp each HTTP request at ASGI entry, before the body is read.

    Installed with the routes, i.e. before vLLM adds its core middlewares
    (CORS, auth, request id, metrics), which therefore wrap outside it and
    only add their own small time. ``request_received_ts`` of ``/generate``
    is this stamp, so server TTFT includes body transfer, JSON parsing and
    tokenization.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            scope[RECEIVED_AT] = (time.time(), time.monotonic())
        await self.app(scope, receive, send)


def _engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"message": message}})


class ResponseState:
    """Accumulates one request's output and builds its ``meta_info``."""

    def __init__(self, rid: str, timing: ResponseTiming):
        self.rid = rid
        self.timing = timing
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.output_ids: list[int] = []
        self.text = ""
        self.finish_reason: dict[str, Any] | None = None
        self.forward_entry_time: float | None = None
        self.queue_time: float | None = None

    def update(self, out: RequestOutput, delta: bool) -> str:
        """Fold one engine output in and return the text it adds."""
        now = time.time()
        completion = out.outputs[0]
        self.prompt_tokens = len(out.prompt_token_ids or ())
        self.cached_tokens = out.num_cached_tokens or 0
        if delta:
            self.output_ids.extend(completion.token_ids)
            self.text += completion.text
            new_text = completion.text
        else:
            self.output_ids = list(completion.token_ids)
            new_text = completion.text[len(self.text) :]
            self.text = completion.text
        self.finish_reason = finish_reason_info(completion, len(self.output_ids))

        stats = out.metrics if isinstance(out.metrics, RequestStateStats) else None
        if self.output_ids and self.timing.first_token_wall is None:
            first = None
            if stats is not None:
                first = self.timing.engine_to_wall(stats.first_token_ts, now)
            self.timing.first_token_wall = now if first is None else first
        if self.forward_entry_time is None and stats is not None:
            self.forward_entry_time = self.timing.engine_to_wall(
                stats.scheduled_ts, now
            )
            if stats.scheduled_ts and stats.queued_ts:
                self.queue_time = stats.scheduled_ts - stats.queued_ts
        return new_text

    def meta_info(self) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "id": self.rid,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "completion_tokens": len(self.output_ids),
            "finish_reason": self.finish_reason,
            "request_received_ts": self.timing.received_wall,
            "api_server_dispatch_finish_ts": self.timing.dispatch_finish_wall,
            "prefill_finished_time": self.timing.first_token_wall,
        }
        if self.forward_entry_time is not None:
            meta["forward_entry_time"] = self.forward_entry_time
        if self.queue_time is not None:
            meta["queue_time"] = self.queue_time
        return meta


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def stream_events(
    results: AsyncGenerator[RequestOutput, None], state: ResponseState
) -> AsyncGenerator[str, None]:
    """SSE body: one event per engine output that adds tokens or finishes."""
    try:
        async for out in results:
            text = state.update(out, delta=True)
            if out.outputs[0].token_ids or out.finished:
                yield _sse({"text": text, "meta_info": state.meta_info()})
    except Exception as e:
        logger.exception("generate %s failed", state.rid)
        yield _sse({"error": {"message": f"{type(e).__name__}: {e}"}})
    yield "data: [DONE]\n\n"


@router.post("/generate")
async def generate(raw_request: Request):
    received_wall, received_mono = raw_request.scope[RECEIVED_AT]
    timing = ResponseTiming(received_wall=received_wall, received_mono=received_mono)
    try:
        request = parse_generate_request(await raw_request.json())
    except ValueError as e:  # includes GenerateRequestError and JSON errors
        return _error(400, str(e))

    engine = _engine_client(raw_request)
    if engine.errored:
        return _error(503, f"engine error: {engine.dead_error}")
    rid = request.rid or random_uuid()
    try:
        engine.check_admission(1, rid)
        prompt = parse_model_prompt(engine.model_config, request.prompt())
        (engine_input,) = await engine.renderer.render_cmpl_async([prompt])
    except GracefulHTTPError as e:
        return _error(e.http_status, e.message)
    except (VLLMClientError, ValueError) as e:
        return _error(400, str(e))
    timing.dispatch_finish_wall = time.time()

    results = engine.generate(engine_input, request.sampling_params, rid)
    state = ResponseState(rid, timing)
    if request.stream:
        return StreamingResponse(
            stream_events(results, state), media_type="text/event-stream"
        )

    try:
        async for out in results:
            state.update(out, delta=False)
    except GracefulHTTPError as e:
        return _error(e.http_status, e.message)
    except (VLLMClientError, GenerateRequestError) as e:
        return _error(400, str(e))
    return JSONResponse(
        content={
            "text": state.text,
            "output_ids": state.output_ids,
            "meta_info": state.meta_info(),
        }
    )


@router.api_route("/flush_cache", methods=["GET", "POST"])
async def flush_cache(raw_request: Request):
    """Drop every cached prefix, including connector-managed tiers.

    Code caches and CUDA graphs are untouched. The engine refuses (HTTP 400)
    instead of flushing partially while any KV block is held (running or
    waiting requests, in-flight transfers); the caller may retry when idle.
    The response is a receipt, also logged as one ``[ax] flush_cache`` line:
    ``flush_started_ts``/``flush_finished_ts`` (API-process epoch seconds)
    enclose the engine call, ``unfinished_requests_at_start`` counts requests
    still streaming at the start, and ``kv_connector`` names any configured
    connector, whose own reset semantics bound what ``success`` proves.
    """
    engine = _engine_client(raw_request)
    kv_transfer = engine.vllm_config.kv_transfer_config
    receipt: dict[str, Any] = {
        "engine": "vllm",
        "engine_version": vllm.__version__,
        "reset_connector": True,
        "kv_connector": kv_transfer.kv_connector if kv_transfer else None,
        # Requests this API process still streams (AsyncLLM output processor).
        "unfinished_requests_at_start": engine.get_num_unfinished_requests(),
        "flush_started_ts": time.time(),
    }
    success = await engine.reset_prefix_cache(
        reset_running_requests=False, reset_connector=True
    )
    receipt["flush_finished_ts"] = time.time()
    receipt["success"] = bool(success)
    if not success:
        receipt["message"] = "KV blocks are still held by requests or transfers"
    logger.info("[ax] flush_cache %s", json.dumps(receipt, sort_keys=True))
    return JSONResponse(status_code=200 if success else 400, content=receipt)


def attach_router(app: FastAPI) -> None:
    app.add_middleware(ReceiveTimeMiddleware)
    app.include_router(router)
