"""Startup capture of pointer-stable decode metadata; serving misses stay eager.

Preparation must use the exact replay view, with in_capture=False. This keeps
captured model inputs and metadata destinations at their existing addresses.
The runner restricts preparation to plain DSA/KDA decode without ReplaySSM,
unified memory, TBO, LoRA, or PDMux. Serving never performs a CUDA capture.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import torch

logger = logging.getLogger(__name__)


class MetadataGlueGraph:
    NUM_WARMUP = 2

    def __init__(self, device):
        self.device = device
        self.disabled = False
        self._states: Dict[Any, dict] = {}
        self._capture_stream = None

    def reset(self):
        """Runner recapture invalidates all captured metadata addresses."""
        self._states.clear()

    @staticmethod
    def _leaves(attn_backend) -> List[Any]:
        backends = [attn_backend]
        if attn_backend.attn_backend_list is not None:
            backends.extend(attn_backend.attn_backend_list)
        return backends

    def prepare(self, attn_backend, fb_view, key) -> None:
        """Capture during runner initialization/recapture, before serving."""
        if self.disabled or key in self._states:
            return
        with torch.cuda.device(self.device):
            current = torch.cuda.current_stream(self.device)
            if self._capture_stream is None:
                self._capture_stream = torch.cuda.Stream(device=self.device)
            self._capture_stream.wait_stream(current)
            graph = torch.cuda.CUDAGraph()
            try:
                # Outer context also restores current on capture_end failure.
                with torch.cuda.stream(self._capture_stream):
                    for _ in range(self.NUM_WARMUP):
                        attn_backend.init_forward_metadata_out_graph(fb_view)
                    # torch's global sync/cache cleanup is allowed at startup.
                    with torch.cuda.graph(graph, stream=self._capture_stream):
                        attn_backend.init_forward_metadata_out_graph(fb_view)
            except Exception:
                logger.warning(
                    "Metadata glue startup capture failed for key %s; "
                    "falling back to eager metadata prep permanently.",
                    key, exc_info=True,
                )
                self.disabled = True
                self._states.clear()
                current.wait_stream(self._capture_stream)
                attn_backend.init_forward_metadata_out_graph(fb_view)
                return
            current.wait_stream(self._capture_stream)
            # These host fields can be changed by an intervening eager prefill.
            meta = [
                (backend, {
                    name: getattr(backend, name)
                    for name in ("forward_metadata", "use_mha", "dsa_prefill_impl")
                    if name in vars(backend)
                })
                for backend in self._leaves(attn_backend)
            ]
            self._states[key] = {"graph": graph, "meta": meta, "replayed": False}
            logger.info("Metadata glue captured at startup for key %s", key)

    def run(self, attn_backend, fb_view, key) -> None:
        state = self._states.get(key)
        if self.disabled or state is None:
            attn_backend.init_forward_metadata_out_graph(fb_view)
            return
        for backend, attributes in state["meta"]:
            for name, value in attributes.items():
                setattr(backend, name, value)
        with torch.cuda.device(self.device):
            state["graph"].replay()
        if not state["replayed"]:
            state["replayed"] = True
            logger.info("Metadata glue first replay for key %s", key)
