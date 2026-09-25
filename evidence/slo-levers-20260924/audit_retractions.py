#!/usr/bin/env python3
"""Count TP0 retraction events inside complete raw measurement windows."""
import datetime
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
out = {}
for run in ("067-official_b_full_n30_shortwarm", "068-official_b_pace_off_full_n30_shortwarm",
            "069-official_b_pace_off_host64_full_n30_shortwarm"):
    directory = ROOT / "evidence" / ("L" + run) / "N30"
    raws = list(directory.glob("raw_*.jsonl"))
    assert len(raws) == 1
    rows = [json.loads(line) for line in raws[0].read_text().splitlines() if line.strip()]
    assert len(rows) == len({row["req_id"] for row in rows}) == 5601
    assert json.loads((directory / "level_verdict.json").read_text())["status"] == "VALID"
    lo = min(row["t_recv_s"] for row in rows)
    hi = max(row["client_finish_at_s"] for row in rows)
    events = []
    for line in (directory / "server.log").read_text().splitlines():
        match = re.match(r"\[([^]]+) TP0\] KV cache pool is full\. Retract requests\. #retracted_reqs: (\d+), #new_tokens_gained: (\d+)", line)
        if not match:
            continue
        ts = datetime.datetime.strptime(match[1], "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc).timestamp()
        if lo <= ts <= hi:
            events.append({"dispatch_relative_min": (ts - lo) / 60, "retracted_requests": int(match[2]),
                           "kv_tokens_freed": int(match[3]), "line": line})
    out[run[:3]] = {"events": events, "n_events": len(events),
                   "sum_retracted_requests": sum(e["retracted_requests"] for e in events)}
result = {"scope": "TP0-only server events inside measurement; counts retractions, not unique requests or recompute tokens", "runs": out}
(Path(__file__).resolve().parent / "retraction-audit.json").write_text(json.dumps(result, indent=2) + "\n")
print({run: facts["sum_retracted_requests"] for run, facts in out.items()})
