# SPDX-License-Identifier: Apache-2.0
from vllm import ax_mechanisms


def test_commit_prefers_env_then_image_file(tmp_path, monkeypatch):
    commit_file = tmp_path / "vllm_engine_commit"
    commit_file.write_text("0123456789abcdef\n")
    monkeypatch.setattr(ax_mechanisms, "COMMIT_FILE", str(commit_file))

    monkeypatch.setenv("AX_ENGINE_COMMIT", "fedcba9876543210")
    assert ax_mechanisms.engine_commit() == "fedcba987654"
    monkeypatch.delenv("AX_ENGINE_COMMIT")
    assert ax_mechanisms.engine_commit() == "0123456789ab"
    commit_file.unlink()
    assert ax_mechanisms.engine_commit() == "none"


def test_line_names_base_and_commit(monkeypatch):
    monkeypatch.setenv("AX_ENGINE_COMMIT", "0123456789abcdef")
    assert ax_mechanisms.mechanisms_line() == (
        "[ax] vllm mechanisms: base=vllm-backport-v0.13.1@cde54e8e "
        "commit=0123456789ab"
    )
