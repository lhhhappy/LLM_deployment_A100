# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which of our engine changes (engine/docs/vllm/NNN-*.md) are active.

The serving process logs one ``[ax] vllm mechanisms:`` line at startup; test
jobs compare it with the configuration they expect before measuring.
"""

# The official vLLM main commit this tree is based on.
BASE_COMMIT = "a811738a6"


def mechanisms_line(endpoint_plugin_loaded: bool) -> str:
    parts = [f"base={BASE_COMMIT}", f"000={'on' if endpoint_plugin_loaded else 'off'}"]
    return "[ax] vllm mechanisms: " + " ".join(parts)
