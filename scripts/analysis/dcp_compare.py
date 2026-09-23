# [ax] T50/116: compare two dcp_check.py outputs (e.g. DCP2 vs non-DCP reference). Prints per-stage relative L-inf
# (max|a-b| / max|b|), max abs diff and greedy-token agreement; exit 1 if any stage exceeds --tol (default 1e-2).
# Usage: python dcp_compare.py <test.pt> <ref.pt> [--tol 1e-2]
import sys
import torch

a, b = torch.load(sys.argv[1]), torch.load(sys.argv[2])
tol = float(sys.argv[sys.argv.index("--tol") + 1]) if "--tol" in sys.argv else 1e-2
print("test info", a.get("info")); print("ref  info", b.get("info"))
bad = False
for k in ("cold", "ext", "dec"):
    if k not in a or k not in b:
        print(f"{k}: missing"); bad = True; continue
    x, y = a[k], b[k]
    if x.shape != y.shape:
        print(f"{k}: shape {tuple(x.shape)} vs {tuple(y.shape)}"); bad = True; continue
    fin = bool(torch.isfinite(x).all())
    d = (x - y).abs().max().item()
    rel = d / max(y.abs().max().item(), 1e-30)
    agree = (x.argmax(-1) == y.argmax(-1)).float().mean().item()
    ok = fin and rel <= tol
    bad |= not ok
    print(f"{k}: shape={tuple(x.shape)} finite={fin} rel_linf={rel:.3e} max_abs={d:.3e} argmax_agree={agree:.3f} {'PASS' if ok else 'FAIL'}")
# sensitive oracle: per-DSA-layer o_proj input (this rank's heads) per stage
at, bt = a.get("attn", {}), b.get("attn", {})
for st in ("cold", "ext", "dec"):
    for n in sorted(bt.get(st, {})):
        y = bt[st][n]; x = at.get(st, {}).get(n)
        if x is None or x.shape != y.shape:
            print(f"attn {st} {n}: missing/shape {None if x is None else tuple(x.shape)} vs {tuple(y.shape)}"); bad = True; continue
        d = (x - y).abs().max().item(); rel = d / max(y.abs().max().item(), 1e-30)
        fin = bool(torch.isfinite(x).all()); ok = fin and rel <= tol; bad |= not ok
        print(f"attn {st} {n}: shape={tuple(x.shape)} finite={fin} rel_linf={rel:.3e} max_abs={d:.3e} ref_max={y.abs().max().item():.3e} {'PASS' if ok else 'FAIL'}")
print("RESULT", "FAIL" if bad else "PASS")
sys.exit(1 if bad else 0)
