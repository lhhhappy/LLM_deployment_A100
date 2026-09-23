# T51: exclusive (self) host time per python function / cpu_op inside a root python frame of a with_stack trace.
# Self time of a python frame = dur - sum(child python frames); cpu_ops (aten::*) and cuda runtime calls are listed separately.
# Usage: python t51_pyself.py <trace.json.gz> <root regex> [top=40]
import gzip, json, re, sys, collections
f, rx = sys.argv[1], sys.argv[2]; TOP = int(sys.argv[3]) if len(sys.argv) > 3 else 40
ev = json.load(gzip.open(f, "rt"))["traceEvents"]
PY = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "python_function"]
byid = {e["args"]["Python id"]: e for e in PY}; kids = collections.defaultdict(list)
for e in PY: kids[e["args"].get("Python parent id")].append(e["args"]["Python id"])
# torch-profiler python tracer sometimes leaves a frame unclosed (e.g. a 350 ms "bmm" frame inside a 70 ms forward):
# clamp every frame to its parent's end, top-down from the parentless frames.
_ch = collections.defaultdict(list)
for _e in PY: _ch[_e["args"].get("Python parent id")].append(_e)
_ids = {x["args"]["Python id"] for x in PY}
_st = [(_e, 1e30) for _e in PY if _e["args"].get("Python parent id") not in _ids]
while _st:
    _e, _pend = _st.pop(); _e["dur"] = max(0.0, min(_e["dur"], _pend - _e["ts"])); _st += [(_c, _e["ts"] + _e["dur"]) for _c in _ch[_e["args"]["Python id"]]]
roots = [e for e in PY if re.search(rx, e["name"])]
def short(n): return re.sub(r"^.*/(sglang|site-packages)/", "", n)[:110]
selft = collections.Counter(); cnt = collections.Counter(); seen = set()
stack = [r["args"]["Python id"] for r in roots]
while stack:
    i = stack.pop()
    if i in seen: continue
    seen.add(i); e = byid[i]; ch = kids.get(i, [])
    selft[short(e["name"])] += e["dur"] - sum(byid[c]["dur"] for c in ch); cnt[short(e["name"])] += 1; stack += ch
tot = sum(r["dur"] for r in roots)
win = [(r["ts"], r["ts"] + r["dur"]) for r in roots]
def inwin(e): return any(a <= e["ts"] < b for a, b in win)
ops = collections.Counter(); opn = collections.Counter()
for e in ev:
    if e.get("ph") == "X" and e.get("cat") in ("cpu_op", "cuda_runtime", "cuda_driver") and inwin(e):
        ops[(e["cat"], e["name"][:70])] += e["dur"]; opn[(e["cat"], e["name"][:70])] += 1
print(f"roots {rx!r}: n={len(roots)} total={tot/1e3:.2f} ms; python frames={len(seen)}")
print("-- top python self time --")
for n, v in selft.most_common(TOP): print(f"{v/1e3:8.2f} ms x{cnt[n]:<6} {n}")
GR = [("triton launcher (triton/*)", r"^triton/"), ("tilelang/tvm eager dispatch", r"^(tilelang|tvm|tvm_ffi)/"),
      ("torch.compile/dynamo wrappers", r"^torch/_dynamo|^torch/_inductor|Torch-Compiled"), ("torch python (nn.Module etc.)", r"^torch/"),
      ("builtin C calls (torch ops, launch, pybind)", r"^<built-in"), ("sglang python", r"^sglang/")]
grp = collections.Counter()
for n, v in selft.items(): grp[next((g for g, r in GR if re.search(r, n)), "other")] += v
print("-- python self time by group --")
for g, v in grp.most_common(): print(f"{v/1e3:8.2f} ms {100*v/tot:5.1f}%  {g}")
print("-- top cpu_op / runtime inclusive (note: cpu_ops nest) --")
for k, v in ops.most_common(25): print(f"{v/1e3:8.2f} ms x{opn[k]:<6} {k[0]:12s} {k[1]}")
