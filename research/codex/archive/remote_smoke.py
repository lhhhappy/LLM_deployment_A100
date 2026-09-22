#!/usr/bin/env python3
"""Bounded Qwen smoke on the existing A100 dev environment; no installations.

Creates only its own run directory, uses a free loopback port, and tears down
only its own process group. Does not validate GLM/KDA/DSA kernels.
"""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request


def main():
    root = Path(__file__).resolve().parent
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run = root / "runs" / ("qwen-contract-" + stamp)
    run.mkdir(parents=True, exist_ok=False)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [sys.executable, "-m", "sglang.launch_server", "--model-path",
        "/sjtu/public_model/Qwen3-1.7B", "--host", "127.0.0.1", "--port", str(port),
        "--served-model-name", "default", "--tp-size", "1", "--attention-backend", "triton",
        "--mem-fraction-static", "0.35", "--context-length", "4096",
        "--max-running-requests", "4", "--disable-cuda-graph", "--enable-metrics"]
    (run / "command.json").write_text(json.dumps(command, indent=2))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    with (run / "server.log").open("w") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        print(f"run={run} pid={proc.pid} port={port}", flush=True)
        try:
            deadline = time.monotonic() + 360
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f"server exited {proc.returncode}; inspect {run / 'server.log'}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                time.sleep(2)
            else:
                raise TimeoutError("server readiness exceeded 360 seconds")
            probe = subprocess.run([sys.executable, str(root / "contract_probe.py"),
                "--base-url", f"http://127.0.0.1:{port}", "--timeout", "40",
                "--out", str(run / "probe.json")], timeout=180, check=False)
            print(f"probe_exit={probe.returncode}", flush=True)
            return probe.returncode
        finally:
            # The process group was created by this script, never a name/glob kill.
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
