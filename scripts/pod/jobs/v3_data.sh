# Shared by the v3 jobs: select the regenerated long-chain set v3 and check it by hash before any engine starts.
# v3 copies the load of public requests of the same kind (scripts/longchain/longchain.md, "v3"). Whole chains; each
# level replays the whole set, as the official stress test does (task.md: "每档整集回放一遍").
G_DATA_ROOT="$AX/data/s1-dev-longchain-v3"
G_DATA_SET=s1-dev-longchain-v3
G_COHORT="$G_DATA_ROOT/cohort.json"
python3 - "$G_DATA_ROOT" "${G_MEASURE_SECONDS:-full}" "$LADDER_UP" <<'DATA' || exit 2
import hashlib, json, os, sys
from pathlib import Path
root, seconds, level = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
c = json.loads((root/'cohort.json').read_text())
assert c['set'] == 's1-dev-longchain-v3' and c['n_chains'] == 311 and c['n_requests'] == EXPECTED_REQUESTS
assert c['cohort_sha256'] == 'EXPECTED_COHORT'
ids = [rid for chain in c['chains'] for rid in chain['req_ids']]
assert len(ids) == len(set(ids)) == EXPECTED_REQUESTS
assert hashlib.sha256((root/'requests.jsonl').read_bytes()).hexdigest() == 'EXPECTED_REQUESTS_SHA256'
shards = [os.path.join(dp, f) for dp, _, files in os.walk(root/'bodies') for f in files if f.endswith('.jsonl.gz')]
assert shards and all(os.path.isfile(p) for p in shards), 'body shards not visible to harness os.walk'
print('DATA_READY v3 chains=311 requests=%d N%s admission_seconds=%s warmup=rep16-v1' % (len(ids), level, seconds), flush=True)
DATA
