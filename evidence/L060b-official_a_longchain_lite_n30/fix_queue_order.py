"""Place our unstarted baseline after 060 under the worker's locale sort."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess

queue = Path('/tmp/ax/queue')
old = '060b-official_a_longchain_lite_n30'
new = '060z-official_a_longchain_lite_n30'
before = '060-official_a_180_hicache_lite_n30.sh'
after = '061-official_a_122_lite_n30.sh'
want = [before, new + '.sh', after]
actual = subprocess.run(['sort'], input='\n'.join(reversed(want))+'\n',
                        text=True, capture_output=True, check=True).stdout.splitlines()
assert actual == want, actual
source, target = queue/'pending'/(old+'.sh'), queue/'pending'/(new+'.sh')
receipt = json.loads((Path('/tmp/ax/runs')/old/'deployment-receipt.json').read_text())
if target.exists():
    assert hashlib.sha256(target.read_bytes()).hexdigest() == receipt['job_sha256']
    assert not source.exists()
    print('ALREADY_RENAMED', new, flush=True)
    raise SystemExit(0)
assert sorted(p.name for p in (queue/'running').glob('*.sh')) == ['059-official_a_180_hicache_lite_n14.sh']
assert source.exists() and (queue/'pending'/before).exists() and (queue/'pending'/after).exists()
assert hashlib.sha256(source.read_bytes()).hexdigest() == receipt['job_sha256']
for state in ('running', 'done', 'failed', 'cancelled'):
    assert not (queue/state/(old+'.sh')).exists()
    assert not (queue/state/(new+'.sh')).exists()
run = Path('/tmp/ax/runs')/new
run.mkdir(exist_ok=False)
receipt.update(job=new+'.sh', renamed_from=old+'.sh',
               renamed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               order_verified_by_worker_sort=actual)
(run/'deployment-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
source.rename(target)
print('RENAMED_PENDING_ONLY', old, '->', new, 'SORT_OK', ','.join(actual), flush=True)
