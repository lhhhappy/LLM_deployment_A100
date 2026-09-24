from pathlib import Path
import hashlib,json,os,datetime
name='060-official_a_180_hicache_lite_n30'
stage=Path('/tmp/ax/staging/060-hicache-n30')
r=json.loads((stage/'evidence'/('L'+name)/'config-audit.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
job=stage/'scripts/pod/jobs/official_a_180_hicache_lite_n30.sh'
assert sha(job)==r['job_sha256'],'job hash mismatch'
for f,h in r['patch_hashes'].items():assert sha(Path('/tmp/ax/patches')/f)==h,f
for f,h in r['runtime_tool_hashes'].items():assert sha(Path(f))==h,f
assert sha(Path('/tmp/ax/data/s1-dev-longchain-lite/manifest.json'))==r['manifest_sha256'],'dataset changed'
q=Path('/tmp/ax/queue')
for status in ['running','pending','done','failed','cancelled']:
 if (q/status/(name+'.sh')).exists():
  print('ALREADY_PRESENT',status,name,flush=True);raise SystemExit(0)
running=[p.name for p in (q/'running').glob('*.sh')]
assert not set(running)-{'059-official_a_180_hicache_lite_n14.sh'},running
assert not list((q/'pending').glob('*.sh')),'another job pending'
assert not (q/'PAUSE').exists(),'queue paused'
run=Path('/tmp/ax/runs')/name
run.mkdir(exist_ok=False)
r.update(status='PUBLISHING',queued_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),predecessor_running=running)
(run/'deployment-receipt.json').write_text(json.dumps(r,indent=2)+'\n')
os.link(job,q/'pending'/(name+'.sh'))
print('QUEUED',name,'after=059 same_engine warmup_reused new_flush N30 no_N14_PASS_gate',flush=True)
