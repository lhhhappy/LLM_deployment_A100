#!/usr/bin/env python3
"""Independent-process A/A and A/B row-3950 diagnostics on the developer box.

Arms 01 are full attention observations made first. Arms 02--10 execute the
same model and capture only the investigated extend row (plus the existing
cold/decode observations), keeping additional evidence below about 1 GiB.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    args = p.parse_args()
    root = args.root.resolve(strict=True)
    if not str(root).startswith("/sjtu/linhang/arena/"):
        p.error("developer arena directory required")
    receipt = root / "review_repeats.jsonl"
    with receipt.open("x") as journal:
        env = dict(os.environ, DCP_DEVICES="GPU-14c26f76-e5a4-9d1f-5d33-9ff397af4933,GPU-0fd597b1-c02f-3a31-8fa7-8ecb15791ca8",
                   DCP_ATTN_MODEL=str(root / "model-h16"), AX_PREFIX="8192,6144", AX_EXT="4096,4096",
                   AX_CAPTURE_INPUTS="1", AX_CAPTURE_INDEXER="1", AX_ATTN_CAPTURE_ROW="3950",
                   AX_TIMING_REPEATS="0", AX_PAIRED_TIMING="0", SGLANG_AX_DCP_LOCAL_EXTEND_LARGE_MAX="8192")
        for i in range(2, 11):
            for arm in (("local", "base") if i % 2 == 0 else ("base", "local")):
                name = f"review_{arm}_{i:02}"
                env["SGLANG_AX_DCP_LOCAL_EXTEND"] = str(int(arm == "local"))
                env["AX_PAIRED_TIMING"] = "1" if arm == "local" and i == 10 else "0"
                start = time.time()
                with (root / f"{name}.log").open("x") as log:
                    result = subprocess.run(["bash", str(root / "run_dcp_devbox.sh"), str(root),
                                             str(root / "engine"), name, "2", "1", "0.3"],
                                            env=env, stdout=log, stderr=subprocess.STDOUT)
                record = dict(arm=name, started_at=start, ended_at=time.time(), exit_code=result.returncode,
                              sampled_extend_row=3950, paired_timing=env["AX_PAIRED_TIMING"] == "1")
                journal.write(json.dumps(record) + "\n")
                journal.flush()
                print(json.dumps(record), flush=True)
                result.check_returncode()
                for suffix in ("", ".rank1"):
                    for kind in ("", ".json", ".inputs", ".indexer"):
                        if not (root / f"{name}.pt{suffix}{kind}").is_file():
                            raise RuntimeError(f"missing {name}, {suffix}, {kind}")


if __name__ == "__main__":
    main()
