"""Summarize recorded samples; never reinterpret the old wrapper as baseline A/B."""
import hashlib
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
rows = [json.loads(s) for s in (ROOT / 'full-audit.jsonl').read_text().splitlines()
        if s.startswith('{')]
audit = [json.loads(s) for s in (ROOT / 'arm-audit.jsonl').read_text().splitlines()
         if s.startswith('{')]
cost = []
for n in (33, 256, 1024, 4096, 8192, 16384):
    arms = []
    for arm in (0, 1):
        selected = [r for r in rows if r['kind'] == 'full_moe_cost'
                    and r['tokens'] == n and r['arm'] == arm]
        assert len(selected) == 2
        arms.append(dict(wall_ms=median(v for r in selected for v in r['wall_ms']),
                         gpu_ms=median(v for r in selected for v in r['gpu_ms']),
                         order_wall_medians=[r['wall_p50'] for r in selected],
                         extra_allocated_bytes=[r['extra_allocated'] for r in selected]))
    cost.append(dict(tokens=n, baseline=arms[0], fused=arms[1],
                     wall_reduction_fraction=1-arms[1]['wall_ms']/arms[0]['wall_ms']))
result = dict(
    scope='single A100, random weights, TP8 per-rank shapes; no collectives or model/SLO claim',
    full_audit_numeric_limit='Old audit wrapper wrote fused for both arms. Only same-input activation equality and unwrapped cost are used.',
    arm_audit=[r for r in audit if r['kind'] == 'numeric'
               and not r['case'].startswith('activation_')],
    cost=cost,
    files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(ROOT.iterdir()) if p.suffix in ('.jsonl', '.err', '.py')},
)
(ROOT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
for r in cost:
    print(r['tokens'], round(r['baseline']['wall_ms'], 4),
          round(r['fused']['wall_ms'], 4), round(100*r['wall_reduction_fraction'], 2))
