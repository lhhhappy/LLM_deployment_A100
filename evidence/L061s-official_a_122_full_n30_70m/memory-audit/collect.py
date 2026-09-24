
from pathlib import Path
import ast,csv,json,time,collections
from datetime import datetime,timezone
root=Path('/tmp/ax/runs/061s-official_a_122_full_n30_70m'); level=root/'N30'
log=(root/'server.log').read_text(errors='replace')
s=next(x.split('server_args=',1)[1] for x in log.splitlines() if 'server_args=' in x)
a=ast.literal_eval(s)
gpus={}
for r in csv.reader((level/'gpu_util.csv').read_text().splitlines()):
    if len(r)!=4: continue
    t=datetime.strptime(r[0].strip(),'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc).timestamp()
    g=r[1].strip(); used=int(r[3].strip().split()[0]); util=int(r[2].strip().split()[0])
    d=gpus.setdefault(g,dict(samples=0,first=t,last=t,min_mib=used,max_mib=used,peak_at=t,util_sum=0))
    d['samples']+=1; d['last']=t; d['min_mib']=min(d['min_mib'],used); d['util_sum']+=util
    if used>d['max_mib']: d['max_mib']=used; d['peak_at']=t
rows=[]
for line in (level/'metrics.jsonl').read_text().splitlines():
    try: rows.append(json.loads(line))
    except ValueError: pass
metrics={}
for key in sorted(set().union(*(r.keys() for r in rows))-{'t','samples','err'}):
    pairs=[(r['t'],r[key]) for r in rows if isinstance(r.get(key),(int,float))]
    if not pairs: continue
    metrics[key]=dict(min=min(v for t,v in pairs),max=max(v for t,v in pairs),first=pairs[0][1],last=pairs[-1][1],max_at=max(pairs,key=lambda p:p[1])[0])
pool_lines=[x for x in log.splitlines() if 'server_args=' not in x and any(k in x for k in ('Allocated','max_total_num_tokens','Load weight begin','Load weight end','Capture cuda graph end','mem_fraction_static'))]
out=dict(observed_at=time.time(),run=root.name,phase_receipt=json.loads((level/'flush_evidence.json').read_text()),gpu=gpus,metrics=metrics,metric_first=rows[0]['t'],metric_last=rows[-1]['t'],metric_samples=len(rows),config={k:a.get(k) for k in ('mem_fraction_static','mamba_full_memory_ratio','max_mamba_cache_size','max_running_requests','page_size','speculative_num_draft_tokens','mamba_scheduler_strategy')},pool_lines=pool_lines,warmup_tail=(level/'warmup.log').read_text()[-1800:],tail=log.splitlines()[-3:])
print('MEMORY_SNAPSHOT '+json.dumps(out))
