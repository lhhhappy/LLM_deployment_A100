#!/usr/bin/env python3
"""Execution-time ledger from torch-profiler traces of the live engine (CPU only).

  prof_ledger.py TRACE.json.gz [TRACE2 ...] [--json OUT.json]

For each trace (one TP rank, one capture window) it uses the engine's own step spans (`step[EXTEND bs= toks=]`,
`step[DECODE bs=]`, see srt/utils/profile_utils.py build_step_span_name) and reports:
  1. wall timeline of the window: GPU time inside EXTEND spans, inside DECODE spans, and outside any step
     (scheduler / host / idle between forwards);
  2. per step class (decode; extend by new tokens <2k, 2k-8k, >=8k): count, mean GPU span, kernel-busy share,
     GPU idle inside the span (host launch gaps), mean CPU span;
  3. kernel time by category (comm, MoE, dense GEMM, DSA attention/indexer, KDA, mHC/norm, other) per class.
Kernel busy = union of kernel intervals, so overlapping streams are not double counted. Profiler overhead lengthens
host time (~1.2-1.6x in R10), so host-gap numbers are upper bounds; kernel times are close to real.
"""
import argparse
import gzip
import json
import re
import sys
from collections import defaultdict

CATS = [
    ("comm", re.compile(r"nccl|allreduce|all_reduce|cross_device|allgather|all_gather|reduce_scatter|one_shot|two_shot|custom_ar", re.I)),
    ("moe", re.compile(r"moe|expert|topk_softmax|grouped|align_block", re.I)),
    ("dsa_attn", re.compile(r"tilelang|sparse|mla|indexer|flash|attn|attention|topk|logits", re.I)),
    ("kda", re.compile(r"chunk_|kda|gated_delta|fla_|delta_h|causal_conv|conv1d|gla|recurrent|l2norm", re.I)),
    ("gemm", re.compile(r"marlin|gemm|gemv|cutlass|xmma|ampere_|sm80_|cublas|matmul|splitk", re.I)),
    ("mhc_norm", re.compile(r"hc_|mhc|rmsnorm|rms_norm|layernorm|norm_kernel", re.I)),
]


def cat_of(name):
    for c, rx in CATS:
        if rx.search(name):
            return c
    return "other"


def cls_of(span):
    m = re.search(r"step\[(\w+) bs=(\d+)(?: toks=(\d+))?", span)
    if not m:
        return None
    mode, toks = m.group(1), int(m.group(3) or 0)
    if mode == "EXTEND":
        return "extend<2k" if toks < 2048 else ("extend2k-8k" if toks < 8192 else "extend>=8k")
    return mode.lower()


def union_len(iv):
    iv.sort(); tot = 0; cs = ce = None
    for s, e in iv:
        if ce is None or s > ce:
            if ce is not None:
                tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return tot + (ce - cs if ce is not None else 0)


def analyze(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        ev = json.load(f)
    ev = ev["traceEvents"] if isinstance(ev, dict) else ev
    kern, gspans, cspans = [], [], []
    for e in ev:
        if e.get("ph") != "X" or "dur" not in e:
            continue
        c, n = e.get("cat", ""), e.get("name", "")
        s, d = float(e["ts"]), float(e["dur"])
        if c == "kernel":
            kern.append((s, s + d, n))
        elif c == "gpu_user_annotation" and n.startswith("step["):
            gspans.append((s, s + d, n))
        elif c == "user_annotation" and n.startswith("step["):
            cspans.append((s, s + d, n))
    if not gspans or not kern:
        return {"trace": path, "error": f"no step spans ({len(gspans)}) or kernels ({len(kern)})"}
    gspans.sort(); kern.sort()
    w0, w1 = gspans[0][0], gspans[-1][1]
    per = defaultdict(lambda: {"n": 0, "gpu_span_us": 0.0, "busy_us": 0.0, "cpu_span_us": 0.0, "toks": 0,
                               "cat_us": defaultdict(float)})
    ki = 0
    for s, e, n in gspans:
        k = cls_of(n)
        if k is None:
            continue
        p = per[k]; p["n"] += 1; p["gpu_span_us"] += e - s
        m = re.search(r"toks=(\d+)", n); p["toks"] += int(m.group(1)) if m else 0
        while ki < len(kern) and kern[ki][1] <= s:
            ki += 1
        iv = []; j = ki
        while j < len(kern) and kern[j][0] < e:
            ks, ke, kn = kern[j]; a, b = max(ks, s), min(ke, e)
            if b > a:
                iv.append((a, b)); p["cat_us"][cat_of(kn)] += b - a
            j += 1
        p["busy_us"] += union_len(iv)
    for s, e, n in cspans:
        k = cls_of(n)
        if k:
            per[k]["cpu_span_us"] += e - s
    wall = w1 - w0
    in_ext = sum(e - s for s, e, n in gspans if "EXTEND" in n)
    in_dec = sum(e - s for s, e, n in gspans if "EXTEND" not in n)
    out = {"trace": path, "window_s": wall / 1e6,
           "timeline": {"extend_gpu_span": in_ext / wall, "decode_gpu_span": in_dec / wall,
                        "outside_steps": 1 - (in_ext + in_dec) / wall},
           "classes": {}}
    for k, p in sorted(per.items()):
        if not p["n"]:
            continue
        cat_tot = sum(p["cat_us"].values()) or 1
        out["classes"][k] = {
            "n": p["n"], "mean_new_toks": p["toks"] / p["n"],
            "mean_gpu_span_ms": p["gpu_span_us"] / p["n"] / 1e3,
            "mean_kernel_busy_ms": p["busy_us"] / p["n"] / 1e3,
            "idle_in_span_share": 1 - p["busy_us"] / p["gpu_span_us"],
            "mean_cpu_span_ms": p["cpu_span_us"] / p["n"] / 1e3,
            "us_per_new_token": (p["gpu_span_us"] / p["toks"]) if p["toks"] else None,
            "kernel_share": {c: round(v / cat_tot, 3) for c, v in sorted(p["cat_us"].items(), key=lambda x: -x[1])},
        }
    top = defaultdict(float)
    for s, e, n in kern:
        if w0 <= s <= w1:
            top[n[:90]] += e - s
    out["top_kernels_ms"] = {n: round(v / 1e3, 1) for n, v in sorted(top.items(), key=lambda x: -x[1])[:25]}
    return out


def show(r):
    if "error" in r:
        print(f"PROF {r['trace']}: {r['error']}"); return
    t = r["timeline"]
    print(f"PROF {r['trace'].split('/')[-1]} window {r['window_s']:.1f}s | extend {t['extend_gpu_span']:.1%} "
          f"decode {t['decode_gpu_span']:.1%} outside-steps {t['outside_steps']:.1%}")
    for k, c in r["classes"].items():
        per_tok = f" {c['us_per_new_token']:.0f}us/tok" if c["us_per_new_token"] else ""
        print(f"  {k:12s} n={c['n']:4d} toks~{c['mean_new_toks']:.0f} gpu {c['mean_gpu_span_ms']:.1f}ms "
              f"(busy {c['mean_kernel_busy_ms']:.1f}, idle {c['idle_in_span_share']:.0%}) cpu {c['mean_cpu_span_ms']:.1f}ms"
              f"{per_tok} | " + " ".join(f"{a}={b:.0%}" for a, b in c["kernel_share"].items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--json")
    a = ap.parse_args()
    res = [analyze(p) for p in a.traces]
    for r in res:
        show(r)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    sys.exit(0 if all("error" not in r for r in res) else 2)


if __name__ == "__main__":
    main()
