#!/usr/bin/env python3
"""Render selected bad requests and their actual replay predecessors on CPU.

Prompt-prefix overlap is a verified lower bound on possible token reuse, not proof
that a valid hybrid-cache checkpoint existed or was evicted. Selection is biased
toward slow fast-intra requests; never extrapolate its fraction to the full cohort.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'s1-dev/harness'))
from s1_common import Renderer, in_ttft_gate, materialize_bodies


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('raw', type=Path)
    ap.add_argument('--data-root', required=True, type=Path)
    ap.add_argument('--tok-dir', required=True, type=Path)
    ap.add_argument('--since-min', type=float, default=0)
    ap.add_argument('--limit', type=int, default=8)
    ap.add_argument('--req-id', action='append', default=[],
                    help='explicit completed request ID (repeatable); overrides slow-fast selection')
    ap.add_argument('--alignment', type=int, required=True,
                    help='effective cache-tree token alignment (067 HiCache: 256; physical page: 64)')
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    if args.limit < 1: ap.error('limit must be positive')
    if args.alignment < 1: ap.error('alignment must be positive')
    rows = [json.loads(s) for s in args.raw.read_text().splitlines() if s.strip()]
    by_pos = {(r['chain_id'], r['idx_in_chain']): r for r in rows}
    if len(by_pos) != len(rows): raise ValueError('duplicate replay position')
    t0 = min(r['client_dispatch_at_s'] for r in rows)
    if args.req_id:
        if len(set(args.req_id)) != len(args.req_id) or len(args.req_id) > args.limit:
            ap.error('explicit IDs must be unique and fit --limit')
        by_id = {r['req_id']: r for r in rows}
        if not set(args.req_id) <= by_id.keys():
            ap.error('explicit ID absent from completed raw snapshot')
        selected = [by_id[rid] for rid in args.req_id]
        if any(r.get('error') for r in selected):
            ap.error('explicit IDs must have successful responses')
    else:
        selected = sorted((r for r in rows if not r.get('error')
            and r['client_dispatch_at_s'] >= t0+args.since_min*60
            and in_ttft_gate(r, 'fast_intra') and r['ttft_s'] > 3),
            key=lambda r: (-r['ttft_s'], r['req_id']))[:args.limit]
    pairs = []
    for r in selected:
        prev = by_pos.get((r['chain_id'], r['idx_in_chain']-1))
        if prev is None or prev.get('error'): raise ValueError('missing successful predecessor')
        if prev['client_finish_at_s'] > r['client_dispatch_at_s']:
            raise ValueError('predecessor had not completed before dispatch')
        pairs.append((prev, r))
    manifest = json.loads((args.data_root/'manifest.json').read_text())
    hashes = {}
    for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
        hashes[name] = hashlib.sha256((args.tok_dir/name).read_bytes()).hexdigest()
        if hashes[name] != manifest['tokenizer']['files'][name]:
            raise ValueError('tokenizer manifest mismatch: '+name)
    wanted = {r['req_id'] for pair in pairs for r in pair}
    bodies = materialize_bodies(str(args.data_root), wanted)
    if set(bodies) != wanted: raise ValueError('missing frozen request body')
    renderer = Renderer(str(args.tok_dir))
    tokens = {rid: renderer.tokenizer.encode(renderer.render(bodies[rid]), add_special_tokens=False)
              for rid in wanted}
    results = []
    for prev, r in pairs:
        a, b = tokens[prev['req_id']], tokens[r['req_id']]
        if len(a) != prev['prompt_tokens'] or len(b) != r['prompt_tokens']:
            raise ValueError('rendered token count differs from measured request')
        lcp = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        results.append(dict(req_id=r['req_id'], predecessor=prev['req_id'],
            prompt_tokens=len(b), cached_tokens=r['cached_tokens'], true_prompt_lcp=lcp,
            lcp_aligned=lcp//args.alignment*args.alignment,
            positive_lcp_minus_cached=max(0, lcp//args.alignment*args.alignment-r['cached_tokens']),
            actual_uncached=len(b)-r['cached_tokens'], frozen_uncached=r['uncached_expected'],
            ttft_s=r['ttft_s'], recv_to_exec_s=r['t_exec_start_s']-r['t_recv_s'],
            exec_to_first_s=r['t_first_token_s']-r['t_exec_start_s'],
            predecessor_gap_s=r['client_dispatch_at_s']-prev['client_finish_at_s']))
    report = dict(raw_sha256=hashlib.sha256(args.raw.read_bytes()).hexdigest(),
        tokenizer_hashes=hashes, since_min=args.since_min, alignment_tokens=args.alignment,
        selection='explicit request IDs; diagnostic sample' if args.req_id else
                  'slowest completed fast-intra requests; biased diagnostic sample', results=results)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(pairs=len(results), rendered_lengths_match=True,
        gap_gt4096=sum(r['positive_lcp_minus_cached']>4096 for r in results), output=str(args.out))))


if __name__ == '__main__':
    main()
