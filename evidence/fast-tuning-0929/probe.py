import pathlib,json,statistics,hashlib
from collections import Counter
# Same phase_gate and fast selector as the frozen harness, with thresholds inlined.
def fast(r):
 idx=r.get('_idx_in_chain');idx=r.get('idx_in_chain') if idx is None else idx
 return idx!=0 and r.get('phase') not in ('turn_start','context_reset') and (r.get('uncached_expected') or 0)<=4096

def med(v):return statistics.median(v) if v else None
def describe(rows):
 bad=[r for r in rows if r['ttft_s']>3];work=lambda r:r['prompt_tokens']-r['cached_tokens']
 def timepart(r,a,b):return r[b]-r[a] if r.get(a) is not None and r.get(b) is not None else None
 wait=[timepart(r,'t_recv_s','t_exec_start_s') for r in bad];exe=[timepart(r,'t_exec_start_s','t_first_token_s') for r in bad]
 return {'n':len(rows),'bad':len(bad),'bad_ttft_median':med([r['ttft_s'] for r in bad]),'bad_recv_to_exec_median':med([v for v in wait if v is not None]),'bad_exec_to_first_median':med([v for v in exe if v is not None]),'wait_over_half':sum(v is not None and v>r['ttft_s']/2 for r,v in zip(bad,wait)),'timing_present':sum(v is not None for v in wait),'bad_actual_work_median':med([work(r) for r in bad]),'bad_work_bins':dict(Counter('<=2048' if work(r)<=2048 else '2049-4096' if work(r)<=4096 else '>4096' for r in bad)),'bad_0cache':sum(r['cached_tokens']==0 for r in bad),'bad_prompt_median':med([r['prompt_tokens'] for r in bad])}
base=pathlib.Path('/tmp/ax/runs');out={'runs':{},'paired':{}};data={}
for code in ['130eznu','130ezns','130eznr1']:
 p=next(base.glob(code+'-*'))
 for f in p.glob('N*/timed_verdict.json'):
  d=json.loads(f.read_text());raw=f.parent/d['raw'];blob=raw.read_bytes();assert hashlib.sha256(blob).hexdigest()==d['raw_sha256'];rows=[json.loads(l) for l in blob.splitlines()];assert len(rows)==len({r['req_id'] for r in rows})==d['n_completed']
  rows=[r for r in rows if fast(r)];key=code+'/'+f.parent.name;data[key]={r['req_id']:r for r in rows};out['runs'][key]=describe(rows)
  t=d['first_dispatch_at_s'];out['runs'][key]['by_10min']={str(i):describe([r for r in rows if i*600<=r['client_dispatch_at_s']-t<(i+1)*600]) for i in range(6)}
a=data['130eznu/N34']
for key,b in data.items():
 if key=='130eznu/N34':continue
 ids=a.keys()&b.keys();new=[i for i in ids if a[i]['ttft_s']<=3<b[i]['ttft_s']]
 out['paired'][key]={'common':len(ids),'reference':describe([a[i] for i in ids]),'candidate':describe([b[i] for i in ids]),'new_bad':len(new),'repaired':sum(b[i]['ttft_s']<=3<a[i]['ttft_s'] for i in ids),'new_bad_description':describe([b[i] for i in new]),'new_bad_same_cached':sum(a[i]['cached_tokens']==b[i]['cached_tokens'] for i in new)}
print(json.dumps(out))
