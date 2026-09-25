# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the SGLang-compatible ``/generate`` and ``/flush_cache``.

A fake engine client stands in for ``AsyncLLM``; the routes, request parsing,
SSE framing and ``meta_info`` accounting are the real code.
"""

import json
import time
from collections.abc import AsyncGenerator
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.generate_compat.plugin import GenerateCompatPlugin
from vllm.entrypoints.generate_compat.protocol import (
    DEFAULT_MAX_NEW_TOKENS,
    GenerateRequestError,
    parse_generate_request,
)
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.metrics.stats import RequestStateStats

PROMPT_IDS = list(range(100, 140))
HARNESS_BODY = {
    "text": "[gMASK]<sop><|system|>x<|assistant|><think>",
    "sampling_params": {"max_new_tokens": 8, "temperature": 0, "ignore_eos": True},
    "stream": True,
    "rid": "req-1",
}


class FakeRenderer:
    def __init__(self):
        self.prompts = []

    async def render_cmpl_async(self, prompts):
        self.prompts.extend(prompts)
        return [{"type": "token", "prompt_token_ids": PROMPT_IDS}]


class FakeEngine:
    """Yields scripted outputs; `steps` is a list of (token_ids, finish)."""

    def __init__(self, steps, cached=24, stats="engine"):
        # stats: "engine" = monotonic stamps taken like the engine core does,
        # "foreign" = stamps from an unrelated clock, None = stats disabled.
        self.steps = steps
        self.cached = cached
        self.stats = stats
        self.renderer = FakeRenderer()
        self.model_config = SimpleNamespace(is_encoder_decoder=False)
        self.errored = False
        self.dead_error = None
        self.generate_calls = []
        self.reset_calls = []
        self.reset_result = True
        self.fail_after = None

    def check_admission(self, n=1, request_id=None):
        return None

    async def generate(
        self, engine_input, sampling_params, request_id
    ) -> AsyncGenerator[RequestOutput, None]:
        self.generate_calls.append((engine_input, sampling_params, request_id))
        stats = None
        if self.stats == "engine":
            stats = RequestStateStats(queued_ts=time.monotonic())
            time.sleep(0.002)
            stats.scheduled_ts = time.monotonic()
        elif self.stats == "foreign":
            stats = RequestStateStats(queued_ts=1.0, scheduled_ts=2.0)
        delta = sampling_params.output_kind == RequestOutputKind.DELTA
        produced: list[int] = []
        for i, (token_ids, finish) in enumerate(self.steps):
            if self.fail_after is not None and i == self.fail_after:
                raise RuntimeError("boom")
            produced.extend(token_ids)
            ids = token_ids if delta else list(produced)
            text = "".join(f"<{t}>" for t in ids)
            if not delta and not finish:
                continue
            if stats is not None and not stats.first_token_ts:
                stats.first_token_ts = (
                    time.monotonic() if self.stats == "engine" else 3.0
                )
            yield RequestOutput(
                request_id=request_id,
                prompt=None,
                prompt_token_ids=PROMPT_IDS,
                prompt_logprobs=None,
                outputs=[
                    CompletionOutput(
                        index=0,
                        text=text,
                        token_ids=ids,
                        cumulative_logprob=None,
                        logprobs=None,
                        finish_reason="length" if finish else None,
                    )
                ],
                finished=finish,
                metrics=stats,
                num_cached_tokens=self.cached,
            )

    async def reset_prefix_cache(
        self, reset_running_requests=False, reset_connector=False
    ):
        self.reset_calls.append((reset_running_requests, reset_connector))
        return self.reset_result


def make_client(engine) -> TestClient:
    app = FastAPI()
    GenerateCompatPlugin().attach_router(app)
    app.state.engine_client = engine
    return TestClient(app)


def read_events(client, body):
    with client.stream("POST", "/generate", json=body) as resp:
        assert resp.status_code == 200
        lines = [line for line in resp.iter_lines() if line]
    assert lines[-1] == "data: [DONE]"
    return [json.loads(line[len("data: ") :]) for line in lines[:-1]]


MTP_STEPS = [([1, 2, 3], False), ([4], False), ([5, 6, 7], False), ([8], True)]


class TestParse:
    def test_harness_body(self):
        req = parse_generate_request(HARNESS_BODY)
        assert req.prompt() == {"prompt": HARNESS_BODY["text"]}
        sp = req.sampling_params
        assert sp.max_tokens == 8
        assert sp.temperature == 0
        assert sp.ignore_eos
        assert sp.output_kind == RequestOutputKind.DELTA
        assert req.rid == "req-1"

    def test_defaults_and_renames(self):
        req = parse_generate_request(
            {
                "input_ids": [1, 2],
                "sampling_params": {"min_new_tokens": 2, "sampling_seed": 7},
            }
        )
        assert req.prompt() == {"prompt_token_ids": [1, 2]}
        sp = req.sampling_params
        assert sp.max_tokens == DEFAULT_MAX_NEW_TOKENS
        assert sp.min_tokens == 2
        assert sp.seed == 7
        assert sp.output_kind == RequestOutputKind.FINAL_ONLY

    @pytest.mark.parametrize(
        "body",
        [
            {"text": "a", "input_ids": [1]},
            {"sampling_params": {}},
            {"text": "a", "model": "x"},
            {
                "text": "a",
                "sampling_params": {"max_new_tokens": 4, "json_schema": "{}"},
            },
            {"text": "a", "sampling_params": {"n": 2}},
            {"text": "a", "stream": "yes"},
            {"input_ids": [1, True]},
            {"text": "a", "sampling_params": {"max_new_tokens": -1}},
        ],
    )
    def test_rejects(self, body):
        with pytest.raises(GenerateRequestError):
            parse_generate_request(body)


class TestGenerateStream:
    def test_meta_info_per_event(self):
        engine = FakeEngine(MTP_STEPS)
        client = make_client(engine)
        before = time.time()
        events = read_events(client, HARNESS_BODY)
        after = time.time()

        # The text reaches the renderer verbatim; no template is applied.
        assert engine.renderer.prompts == [{"prompt": HARNESS_BODY["text"]}]
        _, sampling_params, request_id = engine.generate_calls[0]
        assert request_id == "req-1"
        assert sampling_params.ignore_eos and sampling_params.max_tokens == 8

        metas = [e["meta_info"] for e in events]
        assert [m["completion_tokens"] for m in metas] == [3, 4, 7, 8]
        assert "".join(e["text"] for e in events) == "".join(
            f"<{t}>" for t in range(1, 9)
        )
        for m in metas:
            assert m["prompt_tokens"] == len(PROMPT_IDS)
            assert m["cached_tokens"] == 24
            assert m["id"] == "req-1"
        assert all(m["finish_reason"] is None for m in metas[:-1])
        assert metas[-1]["finish_reason"] == {"type": "length", "length": 8}

        recv = metas[-1]["request_received_ts"]
        first = metas[0]["prefill_finished_time"]
        assert before <= recv <= first <= after
        assert {m["prefill_finished_time"] for m in metas} == {first}
        assert recv <= metas[0]["api_server_dispatch_finish_ts"]
        assert recv <= metas[0]["forward_entry_time"] <= first
        assert 0.002 <= metas[0]["queue_time"] < 1.0

    def test_engine_clock_outside_request_window_is_not_used(self):
        # A monotonic value from another clock domain would put the first
        # token before the request arrived; the route must not report it.
        engine = FakeEngine(MTP_STEPS, stats="foreign")
        before = time.time()
        metas = [e["meta_info"] for e in read_events(make_client(engine), HARNESS_BODY)]
        recv = metas[0]["request_received_ts"]
        assert before <= recv <= metas[0]["prefill_finished_time"] <= time.time()
        assert "forward_entry_time" not in metas[0]

    def test_without_engine_stats(self):
        engine = FakeEngine(MTP_STEPS, stats=None)
        metas = [e["meta_info"] for e in read_events(make_client(engine), HARNESS_BODY)]
        assert metas[0]["request_received_ts"] <= metas[0]["prefill_finished_time"]
        assert "forward_entry_time" not in metas[0]

    def test_engine_error_is_reported_in_stream(self):
        engine = FakeEngine(MTP_STEPS)
        engine.fail_after = 2
        events = read_events(make_client(engine), HARNESS_BODY)
        assert [e["meta_info"]["completion_tokens"] for e in events[:-1]] == [3, 4]
        assert "boom" in events[-1]["error"]["message"]

    def test_bad_request(self):
        client = make_client(FakeEngine(MTP_STEPS))
        resp = client.post("/generate", json={"text": "a", "extra": 1})
        assert resp.status_code == 400
        resp = client.post("/generate", content=b"{not json")
        assert resp.status_code == 400


def test_generate_non_stream():
    engine = FakeEngine(MTP_STEPS)
    body = dict(HARNESS_BODY, stream=False)
    resp = make_client(engine).post("/generate", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["output_ids"] == list(range(1, 9))
    assert data["text"] == "".join(f"<{t}>" for t in range(1, 9))
    meta = data["meta_info"]
    assert meta["completion_tokens"] == 8
    assert meta["finish_reason"] == {"type": "length", "length": 8}
    assert meta["request_received_ts"] <= meta["prefill_finished_time"]


class TestFlushCache:
    def test_success_resets_every_tier(self):
        engine = FakeEngine(MTP_STEPS)
        resp = make_client(engine).post("/flush_cache")
        assert resp.status_code == 200
        assert resp.json() == {"success": True}
        assert engine.reset_calls == [(False, True)]

    def test_refusal_is_reported(self):
        engine = FakeEngine(MTP_STEPS)
        engine.reset_result = False
        resp = make_client(engine).post("/flush_cache")
        assert resp.status_code == 400
        assert resp.json()["success"] is False


def test_plugin_contract():
    from vllm.plugins.endpoint_plugins.interface import EndpointPlugin

    plugin = GenerateCompatPlugin()
    assert isinstance(plugin, EndpointPlugin)
    assert plugin.name == "generate_compat"
    assert plugin.required_tasks == ("generate",)
