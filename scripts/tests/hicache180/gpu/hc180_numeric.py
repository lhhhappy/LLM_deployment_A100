#!/usr/bin/env python3
"""Patch 180 numeric check on the dev box (2x A100). WRITTEN, NOT YET RUN.

Question: after a prefix is evicted from the GPU and restored from host memory,
does the model compute exactly what it computes on a device hit, for
GLM-5.3-Flash's real model code (MLA latent + packed kpool=4 DSA indexer + KDA
state [+ NEXTN draft layer])?

Design (one request in flight at a time, greedy):
  model  scaled GLM-5.3-Flash (8 layers: KDA 0-2,4-6, DSA 3,7; + 1 NEXTN layer),
         dummy weights re-initialised so attention matters (sitecustomize.py).
  arms   for each branch offset d in {64,128,192,256}: X = 1024+d shared tokens,
         A = X+Y, B = X+Z (Y, Z disjoint).
           cold_A   : /flush_cache, then A          (reference, no cache)
           dev_A    : A again                       (device hit)
           dev_A2   : A again                       (run-to-run floor, expect 0)
           dev_B    : B                             (fork inside/at a 256 group)
           churn    : unrelated prompts until > 2x --max-total-tokens uncached
                      tokens went through (A/B leave the GPU, stay on host;
                      churn reuses and so overwrites their GPU pages)
           host_A   : A (must be a host load-back: checked via meta_info
                      cached_tokens > 0 AND the load-back counter/log)
           host_B   : B
  oracle (1) generated tokens identical host_A == dev_A, host_B == dev_B;
         (2) max |logprob| diff over prompt-suffix + output tokens, host vs dev,
             must be <= the dev_A vs dev_A2 floor (expected: both 0.0);
         (3) with HC180_DUMP_DIR: o_proj inputs of every DSA layer for the
             extend pass of host_A vs dev_A (same extend length), bitwise;
         (4) informational: k3-KL host vs cold (#38474 style), reported only.
  runs   MTP off and MTP on (NEXTN steps 3/topk 1/draft 4), each against
         (a) the 180 tree  -> expect PASS,
         (b) the same stack without 180 -> expect FAIL on (1)-(3)
             (upstream reports host-restore KL 0.06-0.71 without the indexer).

Usage (on the dev box, inside /sjtu/linhang/arena):
  python3 hc180_numeric.py --tree /path/to/tree --model-dir DIR [--mtp] [--out out.json]
  --make-model DIR writes the scaled config from s1-dev/glm_tok first.
"""

import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def make_model(src, dst, layers=8):
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        if f != "config.json" and not f.endswith(".safetensors"):
            p = os.path.join(src, f)
            if os.path.isfile(p):
                with open(p, "rb") as a, open(os.path.join(dst, f), "wb") as b:
                    b.write(a.read())
    cfg = json.load(open(os.path.join(src, "config.json")))
    t = cfg.get("text_config", cfg)
    t["num_hidden_layers"] = layers
    for k in ("layer_types", "mlp_layer_types", "indexer_types"):
        if k in t:
            t[k] = t[k][:layers]
    lac = t["linear_attn_config"]
    lac["kda_layers"] = [i for i in lac["kda_layers"] if i < layers]
    lac["full_attn_layers"] = [i for i in lac["full_attn_layers"] if i < layers]
    t["n_routed_experts"] = min(t.get("n_routed_experts", 32), 32)
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=1)
    print("scaled model config ->", dst, "full_attn", lac["full_attn_layers"])


def post(url, body, timeout=600):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()


def launch(args, port, dump_dir):
    env = dict(os.environ)
    env.update(
        PYTHONPATH=f"{HERE}:{args.tree}",
        HC180_WELL_SCALED="1",
        SGLANG_OPT_USE_TOPK_V2="0",
        SGLANG_OPT_DEEPGEMM_HC_PRENORM="0",
        SGLANG_AX_SM80_INDEXER="1",
        SGLANG_AX_SM80_FP8_MOE_MARLIN="1",
        SGLANG_LOG_LEVEL="info",
    )
    if dump_dir:
        env["HC180_DUMP_DIR"] = dump_dir
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", args.model_dir, "--load-format", "dummy",
        "--tp", str(args.tp), "--port", str(port),
        "--page-size", "64", "--mamba-radix-cache-strategy", "extra_buffer",
        "--kv-cache-dtype", "bfloat16",
        "--dsa-prefill-backend", "tilelang", "--dsa-decode-backend", "tilelang",
        "--max-total-tokens", str(args.max_total_tokens),
        "--max-mamba-cache-size", "48", "--max-running-requests", "4",
        "--chunked-prefill-size", "4096", "--enable-cache-report",
        "--enable-hierarchical-cache", "--hicache-size", "4",
        "--hicache-io-backend", "kernel", "--hicache-mem-layout", "page_first",
        "--hicache-write-policy", "write_through",
        "--cuda-graph-max-bs-decode", "4",
    ]
    if args.mtp:
        cmd += [
            "--speculative-algorithm", "NEXTN",
            "--speculative-draft-model-path", args.model_dir,
            "--speculative-num-steps", "3", "--speculative-eagle-topk", "1",
            "--speculative-num-draft-tokens", "4",
        ]
    log = open(os.path.join(args.work, f"server_{port}.log"), "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    base = f"http://127.0.0.1:{port}"
    for _ in range(900):
        time.sleep(2)
        if proc.poll() is not None:
            raise SystemExit(f"server exited rc={proc.returncode}; see {log.name}")
        try:
            get(base + "/health_generate", timeout=5)
            return proc, base, log.name
        except Exception:
            pass
    raise SystemExit("server did not become healthy")


def gen(base, ids, prefix_len, n_out=24):
    r = post(
        base + "/generate",
        {
            "input_ids": ids,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": n_out, "ignore_eos": True},
            "return_logprob": True,
            "logprob_start_len": prefix_len,
            "top_logprobs_num": 5,
        },
    )
    mi = r["meta_info"]
    return {
        "out": r.get("output_ids") or [t[1] for t in mi["output_token_logprobs"]],
        "in_lp": [t[0] for t in mi["input_token_logprobs"] if t[0] is not None],
        "out_lp": [t[0] for t in mi["output_token_logprobs"]],
        "top": mi.get("output_top_logprobs"),
        "cached": mi.get("cached_tokens", 0),
    }


def maxdiff(a, b):
    return max((abs(x - y) for x, y in zip(a["in_lp"] + a["out_lp"], b["in_lp"] + b["out_lp"])), default=0.0)


def k3_kl(ref, test):
    """#38474-style fixed-token k3 estimate on the generated tokens (informational)."""
    vals = []
    for p, q in zip(ref["out_lp"], test["out_lp"]):
        r = q - p
        vals.append(math.exp(r) - 1 - r)
    vals.sort()
    return vals[len(vals) // 2] if vals else 0.0


def passes(dump_dir):
    return sorted(f for f in os.listdir(dump_dir) if f.startswith("t0_p")) if dump_dir else []


def compare_dumps(dump_dir, pa, pb):
    import torch

    a, b = torch.load(os.path.join(dump_dir, pa)), torch.load(os.path.join(dump_dir, pb))
    out = {}
    for k in a:
        if k == "extend_seq_lens":
            continue
        if a[k].shape != b[k].shape:
            out[k] = f"shape {tuple(a[k].shape)} vs {tuple(b[k].shape)}"
        else:
            out[k] = float((a[k] - b[k]).abs().max())
    return out


def run(args):
    os.makedirs(args.work, exist_ok=True)
    dump_dir = os.path.join(args.work, "dumps") if args.dump else None
    proc, base, log = launch(args, args.port, dump_dir)
    rng = random.Random(0)
    vocab = (1000, 150000)
    results = {"mtp": args.mtp, "tree": args.tree, "cases": []}
    try:
        for d in (64, 128, 192, 256):
            X = [rng.randrange(*vocab) for _ in range(1024 + d)]
            A = X + [rng.randrange(*vocab) for _ in range(700)]
            B = X + [rng.randrange(*vocab) for _ in range(700)]
            post(base + "/flush_cache", {})
            case = {"d": d}
            p0 = len(passes(dump_dir))
            case["cold_A"] = gen(base, A, len(X))
            case["dev_A"] = gen(base, A, len(X))
            p_dev = passes(dump_dir)[-1] if dump_dir else None
            case["dev_A2"] = gen(base, A, len(X))
            case["dev_B"] = gen(base, B, len(X))
            churned = 0
            while churned < 2 * args.max_total_tokens:
                n = 3000
                gen(base, [rng.randrange(*vocab) for _ in range(n)], n - 1, n_out=1)
                churned += n
            case["host_A"] = gen(base, A, len(X))
            p_host = passes(dump_dir)[-1] if dump_dir else None
            case["host_B"] = gen(base, B, len(X))
            c = case
            case["verdict"] = {
                "cached_tokens": {k: c[k]["cached"] for k in ("cold_A", "dev_A", "dev_B", "host_A", "host_B")},
                "host_hit_expected": c["host_A"]["cached"] > 0 and c["host_B"]["cached"] > 0,
                "tokens_equal_A": c["host_A"]["out"] == c["dev_A"]["out"],
                "tokens_equal_B": c["host_B"]["out"] == c["dev_B"]["out"],
                "floor_devA_devA2": maxdiff(c["dev_A"], c["dev_A2"]),
                "maxdiff_hostA_devA": maxdiff(c["host_A"], c["dev_A"]),
                "maxdiff_hostB_devB": maxdiff(c["host_B"], c["dev_B"]),
                "k3_median_hostA_vs_cold": k3_kl(c["cold_A"], c["host_A"]),
            }
            if dump_dir and p_dev and p_host:
                case["verdict"]["o_proj_in_maxdiff_host_vs_dev"] = compare_dumps(dump_dir, p_dev, p_host)
            v = case["verdict"]
            ok = (
                v["host_hit_expected"]
                and v["tokens_equal_A"]
                and v["tokens_equal_B"]
                and v["maxdiff_hostA_devA"] <= v["floor_devA_devA2"]
                and v["maxdiff_hostB_devB"] <= v["floor_devA_devA2"]
            )
            v["PASS"] = bool(ok)
            print(json.dumps({"d": d, **v}), flush=True)
            results["cases"].append(case)
        results["server_log"] = log
        load_lines = subprocess.run(
            ["grep", "-c", "init_load_back success\\|load_back", log], capture_output=True, text=True
        ).stdout.strip()
        results["log_load_back_lines"] = load_lines
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
    json.dump(results, open(args.out, "w"), indent=1)
    allok = all(c["verdict"]["PASS"] for c in results["cases"])
    print("HC180_NUMERIC", "PASS" if allok else "FAIL", "mtp=" + str(args.mtp), args.out)
    return 0 if allok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", help="dir containing the sglang/ package under test")
    ap.add_argument("--model-dir")
    ap.add_argument("--make-model", nargs=2, metavar=("SRC_GLM_TOK_DIR", "DST"))
    ap.add_argument("--mtp", action="store_true")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--port", type=int, default=31800)
    ap.add_argument("--max-total-tokens", type=int, default=16384)
    ap.add_argument("--dump", action="store_true", help="capture DSA o_proj inputs")
    ap.add_argument("--work", default="./hc180_work")
    ap.add_argument("--out", default="./hc180_numeric.json")
    a = ap.parse_args()
    if a.make_model:
        make_model(*a.make_model)
        sys.exit(0)
    sys.exit(run(a))
