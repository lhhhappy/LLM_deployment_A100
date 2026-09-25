# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`vllm.endpoint_plugins` entry point for the SGLang-compatible routes.

Load it with ``VLLM_PLUGINS=generate_compat,...``; note that setting
``VLLM_PLUGINS`` also restricts the general plugin group, so list the other
plugins to keep (e.g. the LoRA resolvers).
"""

from argparse import Namespace
from typing import TYPE_CHECKING

from fastapi import FastAPI
from starlette.datastructures import State

from vllm.ax_mechanisms import mechanisms_line
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.tasks import SupportedTask

logger = init_logger(__name__)


class GenerateCompatPlugin:
    name = "generate_compat"
    required_tasks: "tuple[SupportedTask, ...] | None" = ("generate",)

    def attach_router(self, app: FastAPI) -> None:
        from vllm.entrypoints.generate_compat.api_router import attach_router

        attach_router(app)

    async def init_state(
        self, engine_client: EngineClient | None, state: State, args: Namespace
    ) -> None:
        # The routes read `app.state.engine_client` set by the core server.
        logger.info(mechanisms_line(endpoint_plugin_loaded=True))
