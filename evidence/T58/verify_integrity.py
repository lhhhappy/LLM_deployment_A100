"""CPU-only verification of the existing 035/N22 raw and the recomputed score."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'evidence/L035/raw_dev-combined-v1_N22_1790184432.jsonl'
RUN = ROOT / 'evidence/L035/run_dev-combined-v1_N22_1790184432.json'
COHORT = ROOT / 's1-dev/harness/g0a/samples_v3/cohort_dev-combined-v1.json'
rows = [json.loads(line) for line in RAW.read_text().splitlines() if line.strip()]
run = json.loads(RUN.read_text())
cohort = json.loads(COHORT.read_text())
score = json.loads((Path(__file__).parent / 'score_formal.json').read_text())
expected = {rid: (chain['chain_id'], i)
            for chain in cohort['chains'] for i, rid in enumerate(chain['req_ids'])}
assert len(rows) == len(expected) == cohort['n_requests'] == 722
assert len({r['req_id'] for r in rows}) == len(rows)
assert {r['req_id'] for r in rows} == set(expected)
assert all((r['chain_id'], r['idx_in_chain']) == expected[r['req_id']] for r in rows)
assert run['n_attempted'] == run['dispatched'] == len(rows)
assert run['raw_file'] == RAW.name and run['config']['N'] == 22
assert run['config']['warmup'] is False and run['config']['max_chains'] is None
assert run['config']['cache_namespace'].endswith('-measure')
assert {r['cache_namespace'] for r in rows} == {run['config']['cache_namespace']}
assert not any(r.get('error') or r.get('error_class') for r in rows)
assert all(r['output_tokens'] == r['max_output_i'] for r in rows)
assert all(r['prompt_tokens'] == r['glm_tokens'] for r in rows)
assert all(0 <= r['cached_tokens'] <= r['prompt_tokens'] for r in rows)
keys = ('ttft_s', 'tpot_s', 't_recv_s', 't_exec_start_s', 't_first_token_s',
        'client_dispatch_at_s', 'client_first_token_at_s', 'client_finish_at_s')
assert all(isinstance(r[k], (int, float)) and math.isfinite(r[k])
           for r in rows for k in keys)
assert all(r['ttft_source'] == 'server' and r['tpot_s'] >= 0 for r in rows)
assert all(r['t_recv_s'] <= r['t_exec_start_s'] <= r['t_first_token_s'] for r in rows)
assert all(r['client_dispatch_at_s'] <= r['client_first_token_at_s']
           <= r['client_finish_at_s'] for r in rows)
ttft_delta = max(abs(r['ttft_s'] - (r['t_first_token_s'] - r['t_recv_s'])) for r in rows)
tpot_delta = max(abs(r['tpot_s'] - (r['client_finish_at_s'] - r['client_first_token_at_s'])
                    / (r['output_tokens'] - 1)) for r in rows)
assert ttft_delta < 1e-5
# The raw TPOT comes from first/last SSE perf_counter samples. The saved
# epoch client_finish is recorded after call_engine returns (loadgen:169,
# 365-387), so it cannot reconstruct that duration exactly. Keep the delta
# as a diagnostic, without changing the official raw TPOT or its gate.
tpots = sorted(r['tpot_s'] for r in rows)
assert tpots[int(.95 * len(tpots))] == score['tpot']['tpot_p95']
assert math.isclose(math.fsum(tpots) / len(tpots), score['tpot']['tpot_mean'], abs_tol=1e-12)
files = [RAW, RUN, COHORT, ROOT / 'evidence/L035/summary.json',
         ROOT / 'scripts/score_formal.py', ROOT / 's1-dev/harness/s1_common.py',
         ROOT / 's1-dev/harness/s1_score.py', ROOT / 's1-dev/harness/s1_loadgen.py']
report = dict(
    passed=True, n_rows=len(rows), n_unique_ids=len(expected),
    cohort_id_chain_position_exact_match=True, measured_namespace_only=True,
    all_output_budgets_met=True, errors=0, server_timing_coverage=1.0,
    max_ttft_reconstruction_delta=ttft_delta, max_client_epoch_tpot_proxy_delta=tpot_delta,
    tpot_timing_note='Exact first/last SSE monotonic timestamps are not saved; '
                     'score the recorded tpot_s, not the epoch finish proxy.',
    tpot_over_0_10=sum(x > .10 for x in tpots),
    tpot_mean=score['tpot']['tpot_mean'], tpot_p95=score['tpot']['tpot_p95'],
    last_completed_request={k: max(rows, key=lambda r: r['client_finish_at_s'])[k]
                            for k in ('req_id', 'prompt_tokens', 'output_tokens',
                                      'client_finish_at_s', 'tpot_s')},
    hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files},
    scope='Existing raw/run only. Not a new GPU run or an official arena score; '
          'does not reconstruct unrecorded cache-clear receipts.')
(Path(__file__).parent / 'integrity.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
