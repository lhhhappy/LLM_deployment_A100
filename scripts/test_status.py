#!/usr/bin/env python3
"""One-glance, ledger-only status; deliberately emits fewer than 30 lines."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    state_path = ROOT / "data/trisol_tests.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"items": {}}
    items = state.get("items", {})
    active = state.get("keep_alive_service_id")
    active_name = state.get("keep_alive_service_name", "")
    running = [n for n, r in items.items() if r.get("status") in {"running", "waiting", "creating", "cleanup"}]
    done = [r for r in items.values() if r.get("status") in {"done", "failed", "cancelled", "invalid"}]
    queued = []
    q = ROOT / "tests/queue"
    if q.exists():
        for p in sorted(q.glob("[0-9]*-*")):
            if p.is_dir() and (p / "APPROVED").is_file() and p.name not in items:
                queued.append(p.name)
    phase = "running" if active and running else ("queued" if active else "none")
    print(f"service: {phase} {active_name or active or '—'} item={running[0] if running else '—'}")
    print(f"items: done={len(done)} running={len(running)} pending={len(queued)}")
    events = ROOT / "logs/trisol_test.events"
    results = {}
    sim = {}
    if events.exists():
        for line in events.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") == "RESULT": results[row.get("item")] = row
            if row.get("event") == "SIMCHECK": sim[row.get("item")] = row
    for r in done:
        item = r.get("slug")
        best = max(r.get("scores", []), key=lambda x: x.get("N", -1), default={})
        delta = sim.get(item, {}).get("p95_delta", "—")
        print(f"finished {item}: config={best.get('matrix','—')} N={best.get('N','—')} "
              f"gate={best.get('binding_gate','—')} fast/overall p95={best.get('fast_p95','—')}/"
              f"{best.get('overall_p95','—')} tpot_mean={best.get('tpot_mean','—')} SIMCHECK delta={delta}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
