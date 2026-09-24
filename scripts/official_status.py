#!/usr/bin/env python3
"""Summarize `playground status --attempt-id ID --json` output read from stdin (see scripts/official_status.sh)."""
import json
import sys

s = sys.stdin.read()
d = json.loads(s[s.index("{"):])
sc = d.get("scorecard") or {}
st = sc.get("scorewheel_stress") or {}
ds = sc.get("scorewheel_datasets") or {}
print(f"attempt {d['id']} created {d['createdAt']} exec={d.get('execStatus')} outcome={d.get('outcome')} score={d.get('score')}")
print(f"  capability gate={sc.get('scorewheel_gate_passed')} aime26={ds.get('aime26')} gpqa={ds.get('gpqa-diamond')}")
if st:
    print("  stress n_at_slo={n_at_slo} tpot_mean={tpot_mean:.4f} tpot_p95={tpot_p95:.4f} fast={fast_intra_p95:.2f} "
          "overall={overall_intra_p95:.2f} turn={turn_start_p95:.2f} chain={chain_start_p95:.2f} "
          "slo_att={slo_attainment:.3f} tpm_all={tpm_all:.0f}".format(**st))
else:
    print("  no stress result yet; scoringState=", d.get("scoringState"))
if d.get("scoringDetails"):
    print("  scoringDetails:", json.dumps(d["scoringDetails"], ensure_ascii=False)[:1500])
