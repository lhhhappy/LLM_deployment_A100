#!/usr/bin/env python3
"""Verify the archived N42 release locally; no GPU or network is required."""

import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "evidence/submission-0930-execution"


def require(condition, message):
    if not condition:
        raise SystemExit(f"FAIL: {message}")


def read_json(path):
    return json.loads(path.read_text())


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def main():
    manifest = read_json(ARCHIVE / "manifest.json")
    for relative, expected in manifest["artifacts"].items():
        path = ROOT / relative
        require(path.is_file(), f"missing artifact: {relative}")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                f"SHA256 mismatch: {relative}")
    require(git("rev-parse", "HEAD:engine/sglang") == manifest["engine_tree"],
            "HEAD engine differs from the submitted 0930a source")
    dirty = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--",
                            "engine/sglang"], check=False)
    require(dirty.returncode == 0, "uncommitted changes in engine/sglang")

    configs = {}
    for arm in ("CAP", "TPOT"):
        path = ROOT / f"submission/official-0930-{arm}.json"
        config = read_json(path)
        require(set(config) == {"image", "command", "env", "model_name"},
                f"unexpected submission fields: {arm}")
        require(config["image"] == manifest["image"], f"image mismatch: {arm}")
        require(all(isinstance(k, str) and isinstance(v, str)
                    for k, v in config["env"].items()), f"invalid environment: {arm}")
        with zipfile.ZipFile(ARCHIVE / arm / "submission.zip") as bundle:
            audit = read_json(ARCHIVE / arm / "bundle-audit.json")
            for member in audit["engine_submission_copies"]:
                require(bundle.read(member) == path.read_bytes(),
                        f"ZIP service config differs: {arm}/{member}")
        configs[arm] = config

    require(configs["CAP"]["command"] == configs["TPOT"]["command"],
            "CAP/TPOT commands differ")
    changes = {key for key in configs["CAP"]["env"].keys() | configs["TPOT"]["env"].keys()
               if configs["CAP"]["env"].get(key) != configs["TPOT"]["env"].get(key)}
    require(changes == {"SGLANG_AX_SCHED_COLD_CAP", "SGLANG_AX_BACKLOG_INTERVAL"},
            "unexpected CAP/TPOT configuration differences")

    for attempt in (47798, 47800):
        result = read_json(ROOT / f"evidence/official/attempt-{attempt}-final-20261003.json")
        require(result["id"] == attempt, f"attempt identity mismatch: {attempt}")
        require(result["execStatus"] == "completed"
                and result["scoringState"]["scoreIsFinal"] is True,
                f"attempt is not final: {attempt}")
        scorecard = result["scorecard"]
        if attempt == 47798:
            require(scorecard["scorewheel_gate_passed"] is True,
                    "47798 capability gate did not pass")
            stress = scorecard["scorewheel_stress"]
            require(stress["n_at_slo"] == 42 and stress["passed"] is True,
                    "47798 official N42 result did not pass")
        else:
            require(scorecard["scorewheel_gate_passed"] is False
                    and scorecard["scorewheel_stress"] is None,
                    "47800 alternative outcome differs from the archived result")

    print(f"PASS: {len(manifest['artifacts'])} artifact hashes; exact submitted engine tree; "
          "both ZIP payloads and final platform results")


if __name__ == "__main__":
    main()
