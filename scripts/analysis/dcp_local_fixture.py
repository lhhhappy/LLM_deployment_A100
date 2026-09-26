#!/usr/bin/env python3
"""Clone the existing diagnostic config; no real checkpoint weights are used."""
import argparse
import json
import shutil
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("source", type=Path)
p.add_argument("destination", type=Path)
p.add_argument("--heads", type=int, default=16)
a = p.parse_args()
shutil.copytree(a.source, a.destination)
c = a.destination / "config.json"
d = json.loads(c.read_text())
t = d.get("text_config", d)
t["num_attention_heads"] = a.heads
c.write_text(json.dumps(d, indent=2) + "\n")
