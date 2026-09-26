#!/usr/bin/env python3
"""Kanban of the experiment plan: one row per job, its reference, and whether each gate got better or worse.

Reads notes/kanban_plan.json (the plan: job name, reference job, what changed, kind) and fills each row from the
evidence already on disk, in this order of preference:
  1. evidence/L<job>/opening/comparison.json (opening probes paired on common request IDs by the watcher);
  2. raw records of both runs (evidence/L<job>/N*/raw_*.jsonl, evidence/L<ref>/N*/raw_*.jsonl), paired on common
     IDs with the harness buckets (same code as miss_profile.py);
  3. the latest pod_progress snapshot (build/scratch/progress/<job>.log) for a running job, marked 未闭合;
  4. otherwise the queue state (pending / not started).
Arrows: ↓ fewer misses than the reference (better), ↑ more (worse), = same. Counts are misses over the common
request IDs; probes and windows are diagnostics, not level verdicts. Writes notes/kanban.md and prints it.
Usage: kanban.py [--plan notes/kanban_plan.json] [--out notes/kanban.md]
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/analysis'))
from miss_profile import LIMITS, bucket_of, load  # noqa: E402

GATES = [('chain_start', 'chain'), ('turn_start', 'turn'), ('overall_intra', 'overall'), ('fast_intra', 'fast')]


def arrow(ref, cand):
    if ref is None or cand is None:
        return ''
    return '↓' if cand < ref else ('↑' if cand > ref else '=')


def from_comparison(job):
    p = ROOT / 'evidence' / f'L{job}' / 'opening' / 'comparison.json'
    if not p.exists():
        return None
    c = json.loads(p.read_text())
    row = {'ref': c['reference_job'], 'n': c['common_completed'], 'closed': True, 'gates': {}}
    for g, short in GATES:
        v = c['gates'].get(g)
        if v:
            row['gates'][short] = (v['reference_over'], v['candidate_over'], v.get('fixed'), v.get('new'))
    row['tpot'] = (c['reference']['tpot']['over_0.10'], c['candidate']['tpot']['over_0.10'],
                   c['reference']['tpot']['mean'], c['candidate']['tpot']['mean'])
    return row


def raw_of(job):
    hits = sorted(glob.glob(str(ROOT / 'evidence' / f'L{job}' / 'N*' / 'raw_*.jsonl')), key=lambda p: Path(p).stat().st_size)
    return hits[-1] if hits else None


def from_raws(job, ref):
    a, b = raw_of(job), raw_of(ref)
    if not a or not b:
        return None
    A, B = load(a), load(b)
    ids = sorted(set(A) & set(B))
    row = {'ref': ref, 'n': len(ids), 'closed': True, 'gates': {}}
    for g, short in GATES:
        members = [k for k in ids if bucket_of(A[k]) == g or (g in ('overall_intra',) and bucket_of(A[k]) == 'fast_intra')]
        lim = LIMITS[g]
        rm = {k for k in members if B[k]['ttft_s'] > lim}
        cm = {k for k in members if A[k]['ttft_s'] > lim}
        row['gates'][short] = (len(rm), len(cm), len(rm - cm), len(cm - rm))
    def slow(R):
        return sum(1 for k in ids if (R[k].get('tpot_s') or 0) > 0.10)
    def mean(R):
        v = [R[k]['tpot_s'] for k in ids if R[k].get('tpot_s') is not None]
        return sum(v) / len(v) if v else None
    row['tpot'] = (slow(B), slow(A), mean(B), mean(A))
    return row


def from_progress(job):
    p = ROOT / 'build/scratch/progress' / f'{job}.log'
    if not p.exists():
        return None
    last = p.read_text().strip().split('\n')[-1]
    m = re.search(r'(\d+) req, (\d+) min \| observed/incomplete fast (\d+)/\d+ overall (\d+)/\d+ turn (\d+)/\d+ chain (\d+)/\d+ \| tpot ([\d.]+)/[\d.]+ \| TPOT>0.10 (\d+)/', last)
    if not m:
        m2 = re.search(r'(\d+) req, (\d+) min \| whole fast (\d+)/\d+ overall (\d+)/\d+ turn (\d+)/\d+ chain (\d+)/\d+ \| tpot ([\d.]+)/[\d.]+ \| TPOT>0.10 (\d+)/', last)
        if not m2:
            return {'running_text': last.split(' ', 1)[-1][:120]}
        m = m2
    n, minutes, fast, overall, turn, chain, tpot, slow = m.groups()
    return {'closed': False, 'minutes': int(minutes), 'n': int(n),
            'whole': {'chain': int(chain), 'turn': int(turn), 'overall': int(overall), 'fast': int(fast)},
            'tpot_mean': float(tpot), 'slow': int(slow)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plan', default=str(ROOT / 'notes/kanban_plan.json'))
    ap.add_argument('--out', default=str(ROOT / 'notes/kanban.md'))
    a = ap.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    lines = ['# 实验看板（脚本 `scripts/analysis/kanban.py` 自动生成；↓ 超时更少=更好，↑ 更多=更差；条数为同 ID 配对；探针与窗口都是诊断值，不是整档判定）', '',
             '| 任务 | 改了什么 | 对照 | 状态 | chain | turn | overall | fast | TPOT>0.10 | 一句话 |', '|---|---|---|---|---|---|---|---|---|---|']
    for item in plan:
        job, ref, what = item['job'], item.get('ref'), item['what']
        row = from_comparison(job) or (from_raws(job, ref) if ref else None)
        if row and row.get('gates'):
            cells = []
            for _, short in GATES:
                g = row['gates'].get(short)
                cells.append(f"{g[0]}→{g[1]} {arrow(g[0], g[1])}" + (f" (修{g[2]}/新{g[3]})" if g[2] is not None else '') if g else '')
            t = row['tpot']
            tp = f"{t[0]}→{t[1]} {arrow(t[0], t[1])}"
            verdict = item.get('verdict') or ''
            lines.append(f"| {job} | {what} | {row['ref']} (同 {row['n']} 条) | 已闭合 | {' | '.join(cells)} | {tp} | {verdict} |")
            continue
        prog = from_progress(job)
        if prog and prog.get('closed') is False:
            w = prog['whole']
            lines.append(f"| {job} | {what} | {ref or '—'} | 跑到第 {prog['minutes']} 分钟，{prog['n']} 条，未闭合 | {w['chain']} | {w['turn']} | {w['overall']} | {w['fast']} | {prog['slow']} (均值 {prog['tpot_mean']*1000:.0f} ms) | {item.get('verdict') or '进行中'} |")
        elif prog and prog.get('running_text'):
            lines.append(f"| {job} | {what} | {ref or '—'} | {prog['running_text']} |  |  |  |  |  | {item.get('verdict') or ''} |")
        else:
            lines.append(f"| {job} | {what} | {ref or '—'} | {item.get('status', '排队中')} |  |  |  |  |  | {item.get('verdict') or ''} |")
    text = '\n'.join(lines) + '\n'
    Path(a.out).write_text(text)
    print(text)


if __name__ == '__main__':
    main()
