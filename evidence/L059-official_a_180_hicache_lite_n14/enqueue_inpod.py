from pathlib import Path
import hashlib,json,os,shutil,datetime
name='059-official_a_180_hicache_lite_n14'
stage=Path('/tmp/ax/staging/059-hicache-3b63d9c8')
r=json.loads((stage/'evidence'/('L'+name)/'config-audit.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
job=stage/'scripts/pod/jobs/official_a_180_hicache_lite_n14.sh'
assert sha(job)==r['job_sha256'],'job hash mismatch'
patch=stage/'patches/180-hicache-glm-dsa.patch'
assert sha(patch)==r['patch_hashes']['180-hicache-glm-dsa.patch'],'180 hash mismatch'
for f,h in r['patch_hashes'].items():
 if not f.startswith('180-'):assert sha(Path('/tmp/ax/patches')/f)==h,f
for f,h in r['runtime_tool_hashes'].items():assert sha(Path(f))==h,f
assert sha(Path('/tmp/ax/data/s1-dev-longchain-lite/manifest.json'))==r['manifest_sha256'],'dataset changed'
q=Path('/tmp/ax/queue')
for status in ['running','pending','done','failed','cancelled']:
 if (q/status/(name+'.sh')).exists():
  print('ALREADY_PRESENT',status,name,flush=True);raise SystemExit(0)
assert not list((q/'running').glob('*.sh')),'another job running'
assert not list((q/'pending').glob('*.sh')),'another job pending'
assert not (q/'PAUSE').exists(),'queue paused'
dest=Path('/tmp/ax/patches/180-hicache-glm-dsa.patch')
if dest.exists():assert sha(dest)==sha(patch),'different 180 already deployed'
else:shutil.copyfile(patch,dest)
run=Path('/tmp/ax/runs')/name
run.mkdir(exist_ok=False)
r.update(status='PUBLISHING',queued_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(run/'deployment-receipt.json').write_text(json.dumps(r,indent=2)+'\n')
os.link(job,q/'pending'/(name+'.sh'))
print('QUEUED',name,'patches=14 tools_unchanged=7 data=058 N14',flush=True)
