#!/usr/bin/env python3
"""Time-windowed gate curves for one or more replay raw files (diagnostic only).

Usage:
  python3 scripts/analysis/window_gates.py RAW [RAW ...] [--label NAME ...] [--window-min 25]
      [--live] [--check-score score_formal.json] [--out-json F] [--out-html F] [--harness DIR]

Each request is placed in the window of its client dispatch time (minutes since the first
measured dispatch of its run). Per window and cumulatively it reports, with the unmodified
harness bucketing (s1_score.in_ttft_gate) and quantile (s1_score.q):
  - each TTFT gate: n, requests over the limit, the allowance for that n (scripts/score_formal
    allowed_over), p95;
  - TPOT: n, mean, p95 and requests above 0.10 s/token (requests with >1 output token);
  - errors, prompt tokens, cached tokens and the token cache-hit ratio.
The cumulative row after the last window is the whole-run statistic; --check-score asserts it
equals score_formal.json (same n, over, p95, tpot mean/p95), which calibrates the tool.

Windows are NOT verdicts: only the complete-data harness score is a verdict. The replay is
closed-loop, so a window inherits the backlog of earlier windows. With --live (a run still in
progress) a request appears only after it finishes, so windows that may still receive late
finishers are marked "open": any window ending after the earliest dispatch among requests
that could still be running is unknown; we mark every window that ends later than
(latest finish seen - longest wall seen) as open.
Standard library only; the HTML chart is inline SVG.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import score_formal  # noqa: E402

GATE_SHORT = {"fast_intra": "fast", "overall_intra": "overall",
              "turn_start": "turn", "chain_start": "chain"}


def load_raw(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A live file can end in a partially written line; anything else is an error.
                if f.read().strip():
                    raise
    return rows


def finite(v) -> bool:
    return score_formal.finite_number(v)


def stats(records: list[dict], scorer) -> dict:
    ok = [r for r in records if not r.get("error")
          and r.get("error_class") not in (scorer.ERR_HARNESS_DATA, scorer.ERR_HARNESS_RENDER)]
    out = {"n": len(records), "n_ok": len(ok), "errors": len(records) - len(ok), "gates": {}}
    for _name, selector, limit in scorer.TTFT_GATE_SPECS:
        vals = [r["ttft_s"] for r in ok if scorer.in_ttft_gate(r, selector)
                and finite(r.get("ttft_s")) and r["ttft_s"] >= 0]
        n, over = len(vals), sum(v > limit for v in vals)
        out["gates"][GATE_SHORT[selector]] = {
            "limit_s": limit, "n": n, "over": over,
            "allowed": score_formal.allowed_over(n) if n else None,
            "over_rate": over / n if n else None,
            "p95": scorer.q(vals, .95) if vals else None}
    tp = [r["tpot_s"] for r in ok if finite(r.get("output_tokens")) and r["output_tokens"] > 1
          and finite(r.get("tpot_s")) and r["tpot_s"] >= 0]
    out["tpot"] = {"n": len(tp), "mean": math.fsum(tp) / len(tp) if tp else None,
                   "p95": scorer.q(tp, .95) if tp else None, "over_0.10": sum(v > .10 for v in tp)}
    prompt = sum(r.get("prompt_tokens") or 0 for r in ok)
    cached = sum(r.get("cached_tokens") or 0 for r in ok)
    out["tokens"] = {"prompt": prompt, "cached": cached, "uncached": prompt - cached,
                     "hit_ratio": cached / prompt if prompt else None}
    return out


def windows_for(rows: list[dict], window_min: float, live: bool, scorer) -> dict:
    disp = [r["client_dispatch_at_s"] for r in rows if finite(r.get("client_dispatch_at_s"))]
    if len(disp) != len(rows):
        raise ValueError("every raw row needs client_dispatch_at_s")
    t0 = min(disp)
    width = window_min * 60
    n_win = int((max(disp) - t0) // width) + 1
    open_after = None
    if live:
        fin = [r["client_finish_at_s"] for r in rows if finite(r.get("client_finish_at_s"))]
        walls = [r["wall_s"] for r in rows if finite(r.get("wall_s"))]
        if fin and walls:
            open_after = max(fin) - max(walls) - t0
    out = []
    for i in range(n_win):
        lo, hi = t0 + i * width, t0 + (i + 1) * width
        sel = [r for r in rows if lo <= r["client_dispatch_at_s"] < hi]
        cum = [r for r in rows if r["client_dispatch_at_s"] < hi]
        is_open = bool(live and open_after is not None and (i + 1) * width > open_after)
        out.append({"index": i, "start_min": i * window_min, "end_min": (i + 1) * window_min,
                    "open": is_open, "window": stats(sel, scorer), "cumulative": stats(cum, scorer)})
    return {"t0": t0, "window_min": window_min, "live": live, "n_rows": len(rows),
            "span_min": (max(disp) - t0) / 60, "windows": out, "whole": stats(rows, scorer)}


def check_score(whole: dict, score_path: Path) -> list[str]:
    ref = json.loads(score_path.read_text(encoding="utf-8"))
    bad = []
    by_short = {GATE_SHORT[sel]: name for name, sel, _ in
                ((n, s, l) for n, s, l in score_formal.load_harness().TTFT_GATE_SPECS)}
    for short, name in by_short.items():
        r, w = ref["ttft_estimated"][name], whole["gates"][short]
        for a, b in (("n", "n"), ("over_limit", "over"), ("allowed_over", "allowed"), ("p95", "p95")):
            if r[a] != w[b]:
                bad.append(f"{short}.{b}: score={r[a]} window_tool={w[b]}")
    for a, b in (("n", "n"), ("tpot_mean", "mean"), ("tpot_p95", "p95")):
        if not math.isclose(ref["tpot"][a], whole["tpot"][b], rel_tol=1e-12, abs_tol=0):
            bad.append(f"tpot.{b}: score={ref['tpot'][a]} window_tool={whole['tpot'][b]}")
    return bad


def summary(run: dict, label: str) -> str:
    """One line: whole-run over/allowed per gate, TPOT, and the latest closed window."""
    def gates(st):
        return " ".join(f"{k} {v['over']}/{v['allowed'] if v['allowed'] is not None else '-'}"
                        for k, v in st["gates"].items())
    w = run["whole"]
    t = w["tpot"]
    line = (f"{label}: {run['n_rows']} req, {run['span_min']:.0f} min | whole {gates(w)} | "
            f"tpot {t['mean']:.4f}/{t['p95']:.4f}" if t["n"] else f"{label}: {run['n_rows']} req")
    closed = [x for x in run["windows"] if not x["open"]]
    if closed:
        x = closed[-1]
        g = x["window"]["gates"]
        p95 = " ".join(f"{k} {v['p95']:.1f}" for k, v in g.items() if v["p95"] is not None)
        line += (f" | last closed {x['start_min']:.0f}-{x['end_min']:.0f}m: {x['window']['n']} req, "
                 f"p95 {p95}, over {sum(v['over'] for v in g.values())}")
    return line


def table(run: dict, label: str) -> str:
    head = ("window(min) state   n   fast o/a  p95   overall o/a  p95   turn o/a  p95   "
            "chain o/a  p95    tpot mean/p95 >.10  hit")
    lines = [f"== {label}  ({run['n_rows']} requests, span {run['span_min']:.1f} min)", head]
    for key in ("window", "cumulative"):
        lines.append(f"-- per {key}")
        for w in run["windows"]:
            s = w[key]
            g = s["gates"]
            def cell(k):
                x = g[k]
                p = f"{x['p95']:.2f}" if x["p95"] is not None else "-"
                a = x["allowed"] if x["allowed"] is not None else "-"
                return f"{x['over']:>3}/{a:<3} {p:>6}"
            t = s["tpot"]
            tm = f"{t['mean']:.4f}/{t['p95']:.4f} {t['over_0.10']:>3}" if t["n"] else "-"
            hit = f"{s['tokens']['hit_ratio']:.3f}" if s["tokens"]["hit_ratio"] is not None else "-"
            state = "open" if w["open"] else "done"
            lines.append(f"{w['start_min']:>5.0f}-{w['end_min']:<5.0f} {state:<5} {s['n']:>4}  "
                         f"{cell('fast')}  {cell('overall')}  {cell('turn')}  {cell('chain')}  {tm}  {hit}")
    return "\n".join(lines)


COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]


def svg_panel(title: str, series: list[tuple[str, list[tuple[float, float | None]], bool]],
              limit: float | None, y_label: str, x_max: float) -> str:
    """series: (label, [(x_mid_min, y)], dashed_if_open_points_exist)."""
    W, H, L, R, T, B = 520, 230, 56, 12, 28, 34
    ys = [y for _, pts, _ in series for _, y in pts if y is not None]
    if limit is not None:
        ys.append(limit)
    y_max = max(ys) * 1.1 if ys and max(ys) > 0 else 1.0
    sx = lambda x: L + (W - L - R) * (x / x_max if x_max else 0)
    sy = lambda y: T + (H - T - B) * (1 - y / y_max)
    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{html.escape(title)}">',
             f'<text x="{L}" y="18" class="t">{html.escape(title)}</text>',
             f'<line x1="{L}" y1="{H-B}" x2="{W-R}" y2="{H-B}" class="ax"/>',
             f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H-B}" class="ax"/>']
    for k in range(5):
        yv = y_max * k / 4
        parts.append(f'<text x="{L-4}" y="{sy(yv)+4:.1f}" class="l" text-anchor="end">{yv:.3g}</text>')
        parts.append(f'<line x1="{L}" y1="{sy(yv):.1f}" x2="{W-R}" y2="{sy(yv):.1f}" class="grid"/>')
    step = 25 if x_max <= 300 else 50
    x = 0.0
    while x <= x_max + 1e-9:
        parts.append(f'<text x="{sx(x):.1f}" y="{H-B+14}" class="l" text-anchor="middle">{x:.0f}</text>')
        x += step
    parts.append(f'<text x="{(L+W-R)/2}" y="{H-4}" class="l" text-anchor="middle">minutes since first request · {html.escape(y_label)}</text>')
    if limit is not None:
        parts.append(f'<line x1="{L}" y1="{sy(limit):.1f}" x2="{W-R}" y2="{sy(limit):.1f}" class="lim"/>')
        parts.append(f'<text x="{W-R}" y="{sy(limit)-4:.1f}" class="l" text-anchor="end">limit {limit:g}</text>')
    for i, (lab, pts, _) in enumerate(series):
        c = COLORS[i % len(COLORS)]
        seg = [(sx(x), sy(y)) for x, y in pts if y is not None]
        if len(seg) > 1:
            parts.append(f'<polyline fill="none" stroke="{c}" stroke-width="2" points="'
                         + " ".join(f"{a:.1f},{b:.1f}" for a, b in seg) + '"/>')
        for a, b in seg:
            parts.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="3" fill="{c}"/>')
    parts.append("</svg>")
    return "".join(parts)


def render_html(runs: list[tuple[str, dict]], window_min: float) -> str:
    x_max = max(max(w["end_min"] for w in run["windows"]) for _, run in runs)
    def pts(run, fn):
        return [((w["start_min"] + w["end_min"]) / 2, fn(w["window"])) for w in run["windows"]]
    panels = []
    for g, lim in (("fast", 3.0), ("overall", 5.0), ("turn", 15.0), ("chain", 30.0)):
        panels.append(svg_panel(f"{g} TTFT p95 per window (s)",
                                [(lab, pts(run, lambda s, g=g: s["gates"][g]["p95"]), False) for lab, run in runs],
                                lim, "seconds", x_max))
        panels.append(svg_panel(f"{g}: share of requests over {lim:g} s",
                                [(lab, pts(run, lambda s, g=g: s["gates"][g]["over_rate"]), False) for lab, run in runs],
                                0.05, "fraction (gate tolerates about 5% with allowance)", x_max))
    panels.append(svg_panel("TPOT p95 per window (s/token)",
                            [(lab, pts(run, lambda s: s["tpot"]["p95"]), False) for lab, run in runs],
                            0.10, "s/token", x_max))
    panels.append(svg_panel("TPOT mean per window (s/token)",
                            [(lab, pts(run, lambda s: s["tpot"]["mean"]), False) for lab, run in runs],
                            None, "s/token", x_max))
    panels.append(svg_panel("Token cache-hit ratio per window",
                            [(lab, pts(run, lambda s: s["tokens"]["hit_ratio"]), False) for lab, run in runs],
                            None, "cached / prompt tokens", x_max))
    panels.append(svg_panel("Requests per window (by dispatch time)",
                            [(lab, pts(run, lambda s: s["n"]), False) for lab, run in runs],
                            None, "requests", x_max))
    legend = "".join(f'<span class="k"><i style="background:{COLORS[i % len(COLORS)]}"></i>{html.escape(lab)}'
                     f' ({run["n_rows"]} req{", live" if run["live"] else ""})</span>'
                     for i, (lab, run) in enumerate(runs))
    opens = [f"{lab}: windows from {min(w['start_min'] for w in run['windows'] if w['open']):.0f} min still open"
             for lab, run in runs if any(w["open"] for w in run["windows"])]
    tables = "".join(f"<pre>{html.escape(table(run, lab))}</pre>" for lab, run in runs)
    return f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Window gate curves</title>
<style>
:root{{--bg:#fff;--fg:#111;--mut:#666;--grid:#e5e7eb;--lim:#b91c1c}}
@media (prefers-color-scheme:dark){{:root{{--bg:#111;--fg:#eee;--mut:#aaa;--grid:#333;--lim:#f87171}}}}
body{{background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif;margin:16px}}
.g{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}}
.t{{fill:var(--fg);font-size:13px;font-weight:600}} .l{{fill:var(--mut);font-size:10px}}
.ax{{stroke:var(--mut)}} .grid{{stroke:var(--grid)}} .lim{{stroke:var(--lim);stroke-dasharray:5 4}}
.k{{margin-right:16px}} .k i{{display:inline-block;width:12px;height:12px;margin-right:4px;vertical-align:-1px}}
pre{{overflow-x:auto;font-size:11px;color:var(--fg)}} p{{color:var(--mut)}}
</style></head><body>
<h1>Gate curves, {window_min:g}-minute windows</h1>
<p>{legend}</p>
<p>Diagnostic only: windows are not verdicts; the complete-data harness score is. Each request sits in the window
of its dispatch time; the closed-loop replay carries backlog from earlier windows. {'; '.join(opens)}</p>
<div class="g">{''.join(panels)}</div>{tables}</body></html>"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw", nargs="+", type=Path)
    ap.add_argument("--label", action="append", default=[])
    ap.add_argument("--window-min", type=float, default=25.0)
    ap.add_argument("--live", action="store_true", help="run still in progress: mark windows that may change")
    ap.add_argument("--check-score", type=Path, help="score_formal.json of the first RAW; assert equality")
    ap.add_argument("--harness", type=Path, default=score_formal.DEFAULT_HARNESS)
    ap.add_argument("--summary", action="store_true", help="print one line per run instead of tables")
    ap.add_argument("--out-json", type=Path)
    ap.add_argument("--out-html", type=Path)
    a = ap.parse_args(argv)
    scorer = score_formal.load_harness(a.harness)
    labels = a.label + [p.parent.parent.name or p.name for p in a.raw[len(a.label):]]
    runs = []
    for path, lab in zip(a.raw, labels):
        rows = load_raw(path)
        if not rows:
            print(f"{path}: no rows", file=sys.stderr)
            return 2
        runs.append((lab, windows_for(rows, a.window_min, a.live, scorer)))
    for lab, run in runs:
        print(summary(run, lab) if a.summary else table(run, lab))
    rc = 0
    if a.check_score:
        bad = check_score(runs[0][1]["whole"], a.check_score)
        print("CALIBRATION", "OK: whole-run statistics equal score_formal.json" if not bad else "FAIL")
        for b in bad:
            print("  ", b)
        rc = 1 if bad else 0
    if a.out_json:
        a.out_json.write_text(json.dumps({lab: run for lab, run in runs}, indent=1), encoding="utf-8")
    if a.out_html:
        a.out_html.write_text(render_html(runs, a.window_min), encoding="utf-8")
    return rc


if __name__ == "__main__":
    sys.exit(main())
