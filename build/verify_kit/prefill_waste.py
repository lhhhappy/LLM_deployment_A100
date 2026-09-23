# Total prefill work actually done vs the frozen expectation (uncached_expected), per phase, and the biggest
# offenders. Usage: prefill_waste.py <raw.jsonl>
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.startswith("{")]
rows = [r for r in rows if r.get("uncached_expected") is not None and r.get("cached_tokens") is not None]
tot_a = tot_e = 0; ph = {}
for r in rows:
    a = r["prompt_tokens"] - r["cached_tokens"]; e = r["uncached_expected"]
    tot_a += a; tot_e += e; p = ph.setdefault(r["phase"], [0, 0, 0]); p[0] += a; p[1] += e; p[2] += 1
print(f"rows={len(rows)} actual_prefill={tot_a/1e6:.2f}M expected={tot_e/1e6:.2f}M waste={(tot_a-tot_e)/1e6:.2f}M ({(tot_a-tot_e)/max(1,tot_a):.0%} of actual)")
for k, (a, e, n) in sorted(ph.items()): print(f"  {k:14s} n={n:4d} actual={a/1e6:6.2f}M expected={e/1e6:6.2f}M extra={(a-e)/1e6:6.2f}M")
w = sorted(rows, key=lambda r: -(r["prompt_tokens"] - r["cached_tokens"] - r["uncached_expected"]))[:8]
for r in w: print(f"  worst {r['phase']:13s} prompt={r['prompt_tokens']} cached={r['cached_tokens']} expected_uncached={r['uncached_expected']} edge={r.get('edge_type')} gap_ms={r.get('replay_gap_ms')}")
