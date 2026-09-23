# Why did intra requests miss the cache? For each request whose actual cached tokens are far below the frozen
# expectation, show the previous request in the same chain: its prompt size, whether it had finished before this
# request was dispatched (client timestamps), and the overlap with other in-flight requests.
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
by_chain = {}
for r in rows: by_chain.setdefault(r["chain_id"], []).append(r)
for c in by_chain.values(): c.sort(key=lambda r: r.get("idx_in_chain", 0))
lost = []
for c in by_chain.values():
    for i, r in enumerate(c):
        exp_cached = max(0, (r.get("prompt_tokens") or 0) - (r.get("uncached_expected") or 0))
        miss = exp_cached - (r.get("cached_tokens") or 0)
        if miss > 4096 and i > 0:
            p = c[i - 1]
            gap = (r.get("client_dispatch_at_s") or 0) - (p.get("client_finish_at_s") or 0)
            lost.append((miss, r, p, gap))
lost.sort(key=lambda x: -x[0])
print(f"lost>4096: {len(lost)}  (prev finished before dispatch: {sum(g >= 0 for *_, g in lost)})")
for miss, r, p, gap in lost[:15]:
    print(f"miss={miss:7d} prompt={r['prompt_tokens']:7d} cached={r['cached_tokens']:7d} | prev prompt={p.get('prompt_tokens')} prev_cached={p.get('cached_tokens')} "
          f"prev_out={p.get('output_tokens')} gap_after_prev_finish={gap:7.1f}s idx={r.get('idx_in_chain')} edge={r.get('edge_type')}")
