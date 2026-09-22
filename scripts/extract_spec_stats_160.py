#!/usr/bin/env python3
"""Extract rank-0 spec windows. Counts permit weighted aggregation; legacy lines do not.

Slice the measured server.log segment first or use --start-line. All counts
include one target/bonus token per request-round; draft accept rate excludes it.
"""
import argparse
import json
import re
from pathlib import Path


def parse(lines, draft_tokens=4, start_line=1):
    windows=[]
    for number,line in enumerate(lines,1):
        if number<start_line or 'Decode batch' not in line:continue
        rank=re.search(r'\bTP(\d+)\b',line)
        if rank and int(rank[1])!=0:continue
        m=re.search(r'(?<!block )accept len: ([\d.]+), accept rate: ([\d.]+)',line)
        if not m:continue
        row=dict(line=number,accept_length=float(m[1]),draft_accept_rate=float(m[2]))
        c=re.search(r'spec tokens: (\d+), spec rounds: (\d+)',line)
        if c:row.update(tokens=int(c[1]),rounds=int(c[2]))
        windows.append(row)
    exact=[w for w in windows if 'rounds' in w];tokens=sum(w['tokens'] for w in exact);rounds=sum(w['rounds'] for w in exact)
    return dict(windows=windows,window_count=len(windows),exact_window_count=len(exact),
                request_rounds=rounds,accepted_tokens_including_target=tokens,
                weighted_tokens_per_request_step=tokens/rounds if rounds else None,
                weighted_draft_accept_rate=(tokens-rounds)/(rounds*(draft_tokens-1)) if rounds and draft_tokens>1 else None,
                scope='logged windows only; legacy rounded lines excluded from weighted values; no unlogged final interval')


def main():
    p=argparse.ArgumentParser();p.add_argument('log',type=Path);p.add_argument('--draft-tokens',type=int,default=4);p.add_argument('--start-line',type=int,default=1);p.add_argument('--out',type=Path);args=p.parse_args()
    out=json.dumps(parse(args.log.read_text(errors='replace').splitlines(),args.draft_tokens,args.start_line),indent=2)+'\n'
    if args.out:args.out.write_text(out)
    else:print(out,end='')
if __name__=='__main__':main()
