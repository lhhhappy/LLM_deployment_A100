# SPDX-License-Identifier: Apache-2.0
"""The ``[ax] vllm mechanisms:`` startup line (engine vllm 000).

Names the organizer base, the engine commit applied on top of it and the state of
each mechanism, so a run can be checked against the configuration it expects.
The commit comes from ``AX_ENGINE_COMMIT`` (runtime bundle) or the file the
submission image writes.
"""

import os

BASE = "vllm-backport-v0.13.1@cde54e8e"
COMMIT_FILE = "/opt/ax/vllm_engine_commit"


def engine_commit() -> str:
    commit = os.environ.get("AX_ENGINE_COMMIT", "")
    if not commit:
        try:
            with open(COMMIT_FILE) as f:
                commit = f.read().strip()
        except OSError:
            commit = "none"
    return commit[:12]


def mechanisms_line() -> str:
    return f"[ax] vllm mechanisms: base={BASE} commit={engine_commit()}"
