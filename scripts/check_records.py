#!/usr/bin/env python3
"""check_records.py — mechanical consistency checks for the shared records
(idea from harness-template-cn: turn verbal rules into checks).

Checks: dispatch status values; board instance table integrity (main + W* rows
present); plans/active files have required sections; no obvious secrets in
tracked notes; findings/decisions numbering has no duplicates.
Exit 1 on errors. Usage: python3 scripts/check_records.py
"""
import re, sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
errors, warns = [], []

# 1) dispatch statuses
ok = {"queued", "accepted", "in-progress", "done", "blocked", "dropped"}
for line in (root / "notes/dispatch.md").read_text().splitlines():
    m = re.match(r"\|\s*(T\d+)\s*\|", line)
    if m:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        st = cells[5] if len(cells) > 5 else ""
        if not (st in ok or st.startswith("blocked")):
            errors.append(f"dispatch {m.group(1)}: bad status {st!r}")
ids = re.findall(r"^\|\s*(T\d+)\s*\|", (root / "notes/dispatch.md").read_text(), re.M)
dups = {i for i in ids if ids.count(i) > 1}
if dups: errors.append(f"dispatch duplicate ids: {sorted(dups)}")

# 2) board instance table
board = (root / "board.md").read_text()
if "## 活跃 Codex 实例" not in board: errors.append("board: instance table header missing")
else:
    tbl = board.split("## 活跃 Codex 实例", 1)[1].split("**进行中**", 1)[0]
    if not re.search(r"^\| main", tbl, re.M): errors.append("board: main row missing")
    ws = re.findall(r"^\| (W\d+)", tbl, re.M)
    logs = sorted({p.name.split("_")[0].split(".")[0] for p in (root / "logs/codex").glob("W*.log")})
    for w in logs:
        if w not in ws: errors.append(f"board: worker {w} has a log but no row (table overwritten?)")

# 3) active plans have required sections
need = ["## 目标", "## 范围", "## 验证方式", "## 进度记录", "## 决策记录"]
for p in (root / "plans/active").glob("*.md"):
    t = p.read_text()
    for h in need:
        if h not in t: errors.append(f"plan {p.name}: missing {h}")

# 3b) test plan: every case row has a valid status
tp = root / "tests/TEST_PLAN.md"
if not tp.exists(): errors.append("tests/TEST_PLAN.md missing")
else:
    for line in tp.read_text().splitlines():
        m = re.match(r"\|\s*([A-Z0-9]+-\d+)\s*\|", line)
        if m:
            last = line.strip().strip("|").split("|")[-1].strip()
            if not re.match(r"(todo|wip|pass|fail|blocked)", last):
                errors.append(f"TEST_PLAN {m.group(1)}: bad status {last!r}")

# 4) secrets
pat = re.compile(r"(asp_[0-9a-f]{20,}|sk-[A-Za-z0-9]{20,}|BEGIN (RSA|OPENSSH) PRIVATE KEY|PLAYGROUND_TOKEN=\S+)")
for p in list((root / "notes").glob("*.md")) + list((root / "research").rglob("*.md")) + \
         list((root / "plans").rglob("*.md")) + [root / "board.md", root / "README.md", root / "rule.md"]:
    if p.exists() and pat.search(p.read_text(errors="ignore")):
        errors.append(f"possible secret in {p.relative_to(root)}")

# 5) numbering duplicates
for f, rx in (("notes/findings.md", r"^## (F\d+)\b"), ("notes/decisions.md", r"^\| (\d+) \|")):
    nums = re.findall(rx, (root / f).read_text(), re.M)
    d = {n for n in nums if nums.count(n) > 1}
    if d: warns.append(f"{f}: duplicate numbers {sorted(d)}")

for w in warns: print("WARN ", w)
for e in errors: print("ERROR", e)
print(f"=> {len(errors)} error(s), {len(warns)} warning(s)")
sys.exit(1 if errors else 0)
