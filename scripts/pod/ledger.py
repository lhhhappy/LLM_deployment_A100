#!/usr/bin/env python3
# Run LOCALLY: pull every pod queue job (patches/args/env + key result lines) and write notes/ledger-8card.md,
# the machine-generated factual half of the 8-card experiment notebook. Reasoning lives in notes/experiments.md.
import datetime, json, os, subprocess, sys
cmd = "cd /sjtu/linhang/arena/repo && source scripts/pod/common.sh && PEXEC_TIMEOUT=200 bexec 'python3 /tmp/ax/verify_kit/ledger_dump.py'"
out = subprocess.run(["scripts/gssh", cmd], capture_output=True, text=True, env={**os.environ, "GSSH_TIMEOUT": "300"}).stdout
rows = sorted((json.loads(l) for l in out.splitlines() if l.startswith("{")), key=lambda r: r["name"])
if not rows: sys.exit("no rows (connection?) — ledger not rewritten")
md = [f"# 8 卡试验本·事实部分（自动生成 {datetime.datetime.utcnow():%Y-%m-%d %H:%M} UTC，`scripts/pod/ledger.py`）", ""]
for r in rows:
    md += [f"## {r['name']}  [{r['status']}]", f"- 补丁：{' '.join(r['patches'])}", f"- 参数：`{r['args']}`", f"- 环境：`{r['env']}`"]
    md += [f"  - `{l}`" for l in r["results"]] or ["  - （无关键结果行）"]
    md.append("")
open("notes/ledger-8card.md", "w").write("\n".join(md)); print("rows", len(rows))
