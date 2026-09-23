# [ax] T50/116: turn devbox_dcp_check.sh's summary.txt into a verdict file (runs unattended after the matrix).
# Usage: python3 devbox_dcp_verdict.py <run_dir>   -> writes <run_dir>/SUMMARY.txt and prints it
# Required PASS (116 correct): fix_dcp, fix_dcp_hi, fix_dcp_graph, full_dcp_hi vs their non-DCP ref; fix_ref vs orig_ref (identity).
# Expected on the stock stack: orig_dcp PASS (replicated KV, low locs), orig_dcp_hi FAIL or crash (virtual locs > per-rank rows).
import re, sys

d = sys.argv[1]
txt = open(f"{d}/summary.txt").read()
cases, cur = {}, None
for line in txt.splitlines():
    m = re.match(r"== (\S+) stack=", line)
    if m:
        cur = m.group(1); cases[cur] = {"rc": None, "err": []}; continue
    if cur and line.startswith("rc="):
        cases[cur]["rc"] = line.split()[0]; cases[cur]["done"] = "AX_CHECK done" in line
    elif cur and re.search(r"illegal memory|device-side assert|Traceback|Error", line):
        cases[cur]["err"].append(line[:200])
cmps = {}
for blk in txt.split("-- compare ")[1:]:
    head = blk.splitlines()[0].strip()
    res = re.search(r"RESULT (PASS|FAIL)", blk)
    rows = [l for l in blk.splitlines() if re.match(r"(attn |cold:|ext:|dec:)", l)]
    worst = {}
    for l in rows:
        st = l.split()[1] if l.startswith("attn") else l.split(":")[0]
        r = re.search(r"rel_linf=([0-9.e+-]+)", l)
        if r:
            worst[st] = max(worst.get(st, 0.0), float(r.group(1)))
    cmps[head] = (res.group(1) if res else "MISSING", worst)
need = ["fix_dcp vs fix_ref", "fix_dcp_hi vs fix_ref", "fix_dcp_graph vs fix_ref", "full_dcp_hi vs full_ref", "fix_ref vs orig_ref"]
out = ["T50/116 dev-box verdict (TP2, DCP2, gate rel L-inf <= 1e-2 on DSA attention output + logits)", ""]
out.append("cases:")
for c, v in cases.items():
    out.append(f"  {c:14s} {v['rc']} done={v.get('done')} {'ERR: ' + v['err'][0] if v['err'] else ''}")
out.append("comparisons (worst rel L-inf per stage):")
for h, (r, w) in cmps.items():
    out.append(f"  {h:28s} {r:7s} " + " ".join(f"{k}={x:.2e}" for k, x in sorted(w.items())))
ok = all(cmps.get(h, ("MISSING",))[0] == "PASS" for h in need)
crash = [c for c, v in cases.items() if c.startswith(("fix", "full")) and (v["rc"] != "rc=0" or v["err"] or not v.get("done"))]
repro = cmps.get("orig_dcp_hi vs orig_ref", ("MISSING",))[0]
out += ["", f"orig_dcp_hi (stock stack, high virtual locs): {repro} (expected FAIL = reproduces wrong/OOB reads)",
        f"116 cases crashed/errored: {crash or 'none'}",
        f"VERDICT_116 {'PASS' if ok and not crash else 'FAIL'}"]
open(f"{d}/SUMMARY.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
