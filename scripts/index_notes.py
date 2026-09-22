#!/usr/bin/env python3
"""Regenerate notes/README.md: one-line index of every finding (F#), decision (#),
experiment section, plan and report, so the long ledgers stay navigable.
Run after editing notes (check_records.py reminds you)."""
import re, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
out = ["# notes/ 索引（自动生成：`python3 scripts/index_notes.py`，请勿手改）", "",
       "`notes/` 只放**账本**；原始证据（日志、JSON、模拟输出）放在 `evidence/<任务号>/`。", "",
       "| 文件 | 用途 |", "|---|---|",
       "| decisions.md | 决策（最新在上） |", "| findings.md | 已验证事实 F# |", "| experiments.md | 实验与工具台账 |",
       "| dispatch.md | 派发任务 T# |", "| submissions.md | 正式提交台账 |", "| quality.md / tech-debt.md | 质量评分 / 技术债 |", ""]
f = (root/"notes/findings.md").read_text()
out += ["## Findings", ""] + [f"- **{m.group(1)}** {m.group(2)[:150]}" for m in re.finditer(r"^## (F\d+) — (.*)$", f, re.M)] + [""]
d = (root/"notes/decisions.md").read_text()
def _clean(t): return re.sub(r"\*\*", "", t)[:150]
out += ["## Decisions（最新在上）", ""] + ["- **#%s** %s" % (m.group(1), _clean(m.group(2))) for m in re.finditer(r"^\| (\d+) \| [^|]+ \| ([^|]+)\|", d, re.M)] + [""]
e = (root/"notes/experiments.md").read_text()
out += ["## Experiments / Tools sections", ""] + [f"- {m.group(1)}" for m in re.finditer(r"^#{2,3} (.+)$", e, re.M)] + [""]
out += ["## Plans", ""] + [f"- plans/{p.parent.name}/{p.name}" for p in sorted((root/"plans").glob("*/*.md")) if p.parent.name != "templates"] + [""]
out += ["## Reports", ""] + [f"- {p.relative_to(root)}" for p in sorted((root/"research").rglob("R*.md"))] + [""]
(root/"notes/README.md").write_text("\n".join(out) + "\n")
print("notes/README.md regenerated:", len(out), "lines")
