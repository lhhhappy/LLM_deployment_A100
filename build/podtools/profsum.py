import json,gzip,glob,sys,collections,re
d=open("/tmp/ax/prof_last").read().strip()
fs=sorted(glob.glob(d+"/*TP-0*")) or sorted(glob.glob(d+"/*"))
f=fs[0]; print("trace",f)
op=gzip.open if f.endswith(".gz") else open
t=json.load(op(f,"rt"))
ev=t["traceEvents"] if isinstance(t,dict) else t
k=collections.Counter(); cnt=collections.Counter(); tot=0
for e in ev:
    if e.get("ph")=="X" and e.get("cat") in ("kernel","gpu_memcpy","gpu_memset"):
        n=re.sub(r"<.*","",e["name"])[:90]; k[n]+=e.get("dur",0); cnt[n]+=1; tot+=e.get("dur",0)
print("total GPU us",tot)
for n,v in k.most_common(30): print(f"{v/1000:9.1f}ms {100*v/tot:5.1f}% x{cnt[n]:6d}  {n}")
