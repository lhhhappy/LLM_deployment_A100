#!/usr/bin/env python3
"""Reconstruct the prefill lane from a TP0 server log: one line per prefill round, decode rounds in between.

Usage: lane_timeline.py <server.log> <t0_epoch> <a-b> [<a-b> ...]
  t0_epoch: run origin (first dispatch, from timed_window.json first_dispatch_at_s)
  a-b:      window in seconds after t0 (e.g. 0-36 or 269-296)

Each prefill round prints #new-seq / #new-token / #cached-token / #running-req / #queue-req / #pending-token from the
engine's own "Prefill batch" log line, plus how many "Decode batch" rounds ran since the previous prefill.
[ax-124/125] and [ax-125] lines inside the window are printed as they are. Log timestamps have 1 s resolution.
The log does not name the request in a round; the lane holder is inferred from sizes (cached 0, chunk = cold cap).
Calibrated on 109 (evidence/L109-v3_open_124_125x_n26): the rounds whose cached-token equals a waiting request's
cached count coincide with that request's first batch (see notes/reports/109-chain-turn-rootcause-0926.md).
"""
import datetime
import re
import sys

STAMP = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) TP0\]")
FIELDS = ["#new-seq", "#new-token", "#cached-token", "#running-req", "#queue-req", "#pending-token"]


def stamp(line):
    m = STAMP.match(line)
    if not m:
        return None
    return datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc).timestamp()


def fields(line):
    out = {}
    for k in FIELDS:
        m = re.search(re.escape(k) + r": (\d+)", line)
        if m:
            out[k] = int(m.group(1))
    return out


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    log, t0 = sys.argv[1], float(sys.argv[2])
    windows = [tuple(float(x) for x in w.split("-")) for w in sys.argv[3:]]
    rows = []
    with open(log, errors="replace") as f:
        for line in f:
            t = stamp(line)
            if t is not None:
                rows.append((t - t0, line.rstrip()))
    for a, b in windows:
        print(f"== window {a:.1f}-{b:.1f} s")
        ndec = 0
        for t, line in rows:
            if t < a or t > b:
                continue
            if "Prefill batch" in line:
                d = fields(line)
                print(
                    f"  {t:7.1f} PREFILL seq={d.get('#new-seq', 0):2d} new={d.get('#new-token', 0):5d} "
                    f"cached={d.get('#cached-token', 0):6d} run={d.get('#running-req', 0):2d} q={d.get('#queue-req', 0):2d} "
                    f"pending={d.get('#pending-token', 0):7d}  decode rounds before: {ndec}"
                )
                ndec = 0
            elif "Decode batch" in line:
                ndec += 1
            elif "[ax-125]" in line or "[ax-124/125]" in line:
                print(f"  {t:7.1f} {line.split('] ', 1)[1][:120]}")
        print(f"  (trailing decode rounds: {ndec})")


if __name__ == "__main__":
    main()
