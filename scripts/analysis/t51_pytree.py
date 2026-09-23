# T51: python call-tree breakdown from a with_stack torch-profiler trace.
# For every python_function event whose name matches <regex>, aggregate its DIRECT children (by name), depth-limited.
# Usage: python t51_pytree.py <trace.json.gz> <regex> [depth=2] [min_ms=0.3]
import gzip, json, re, sys, collections
f, rx = sys.argv[1], sys.argv[2]; DEPTH = int(sys.argv[3]) if len(sys.argv) > 3 else 2; MIN = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
ev = json.load(gzip.open(f, "rt"))["traceEvents"]
PY = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("python_function",)]
byid = {e["args"]["Python id"]: e for e in PY}
kids = collections.defaultdict(list)
for e in PY: kids[e["args"].get("Python parent id")].append(e["args"]["Python id"])
# torch-profiler python tracer sometimes leaves a frame unclosed (e.g. a 350 ms "bmm" frame inside a 70 ms forward):
# clamp every frame to its parent's end, top-down from the parentless frames.
_ch = collections.defaultdict(list)
for _e in PY: _ch[_e["args"].get("Python parent id")].append(_e)
_ids = {x["args"]["Python id"] for x in PY}
_st = [(_e, 1e30) for _e in PY if _e["args"].get("Python parent id") not in _ids]
while _st:
    _e, _pend = _st.pop(); _e["dur"] = max(0.0, min(_e["dur"], _pend - _e["ts"])); _st += [(_c, _e["ts"] + _e["dur"]) for _c in _ch[_e["args"]["Python id"]]]
# cpu_op children (aten ops / kernels launches) are not linked by Python id; count them via time containment per event
OPS = sorted([e for e in ev if e.get("ph") == "X" and e.get("cat") == "cuda_runtime" and ("Launch" in e["name"] or "Memcpy" in e["name"] or "Synchronize" in e["name"])], key=lambda e: e["ts"])
import bisect; ots = [e["ts"] for e in OPS]
def nlaunch(e): return bisect.bisect_right(ots, e["ts"] + e["dur"]) - bisect.bisect_left(ots, e["ts"])
def short(n): return re.sub(r"^.*/(sglang|site-packages)/", "", n)[:100]
def show(ids, depth, ind, parent_n):
    agg = collections.defaultdict(lambda: [0, 0, 0])
    for i in ids:
        e = byid[i]; a = agg[short(e["name"])]; a[0] += e["dur"]; a[1] += 1; a[2] += nlaunch(e)
    for n, (d, c, nl) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        if d / 1e3 / parent_n < MIN: continue
        print(f"{ind}{d/1e3/parent_n:8.2f} ms/call-of-root  x{c/parent_n:<5.1f} {n}")
        if depth > 1:
            sub = [k for i in ids if short(byid[i]["name"]) == n for k in kids.get(i, [])]
            show(sub, depth - 1, ind + "    ", parent_n)
import os
roots = [e["args"]["Python id"] for e in PY if re.search(rx, e["name"])]
if os.environ.get("LONGEST"): roots = sorted(roots, key=lambda i: -byid[i]["dur"])[:int(os.environ["LONGEST"])]  # only the N longest calls
print(f"roots matching {rx!r}: {len(roots)}, total {sum(byid[r]['dur'] for r in roots)/1e3:.2f} ms")
show(roots, 1, "", 1)
print("-- children (normalized per root call) --")
show([k for r in roots for k in kids.get(r, [])], DEPTH, "  ", len(roots))
