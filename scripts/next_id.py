#!/usr/bin/env python3
"""Print the next free id for a shared log, to avoid numbering collisions between
agents appending concurrently. Usage: next_id.py F|T|E|D   (findings/dispatch/experiments/decisions)
Take the id immediately before writing, then run scripts/check_records.py afterwards."""
import re, sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
kind = (sys.argv[1] if len(sys.argv) > 1 else "F").upper()
src = {"F": ("notes/findings.md", r"^## F(\d+)\b"), "T": ("notes/dispatch.md", r"^\| T(\d+) "),
       "E": ("notes/experiments.md", r"\bE(\d+)\b"), "D": ("notes/decisions.md", r"^\| (\d+) \|")}[kind]
nums = [int(x) for x in re.findall(src[1], (root / src[0]).read_text(), re.M)]
print(f"{kind if kind != 'D' else ''}{max(nums, default=0) + 1}")
