# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which of our engine changes (engine/docs/vllm/NNN-*.md) are active.

The serving process logs one ``[ax] vllm mechanisms:`` line at startup; test
jobs compare it with the configuration they expect before measuring.
"""

# The official vLLM main commit this tree is based on.
BASE_COMMIT = "a811738a6"


def _sm80_port_state() -> str:
    """010 is gated on the device, not on a switch: it runs on SM8x only."""
    from vllm.platforms import current_platform

    if not current_platform.is_cuda():
        return "off:no-cuda"
    cap = current_platform.get_device_capability()
    if cap is None:
        return "off:no-cuda"
    return "on" if cap.major == 8 else f"off:sm{cap.major}{cap.minor}"


def _role_checkpoint_state() -> str:
    """101 is opt-in; the engine refuses to start if its preconditions fail."""
    import vllm.envs as envs

    ids = envs.VLLM_AX_MAMBA_ROLE_CHECKPOINT_TOKEN_IDS
    return "on:" + ",".join(map(str, ids)) if ids else "off"


def mechanisms_line(endpoint_plugin_loaded: bool) -> str:
    parts = [
        f"base={BASE_COMMIT}",
        f"000={'on' if endpoint_plugin_loaded else 'off'}",
        f"010={_sm80_port_state()}",
        f"101={_role_checkpoint_state()}",
    ]
    return "[ax] vllm mechanisms: " + " ".join(parts)
