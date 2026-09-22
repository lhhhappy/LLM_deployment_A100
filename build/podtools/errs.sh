d=/tmp/ax/runs/$1/dev
head -60 $d/report_*.md | cut -c1-250
echo ---RAW; python3 - $d <<'PY'
import json,sys,glob,collections
f=sorted(glob.glob(sys.argv[1]+"/raw_*.jsonl"))[-1]; rows=[json.loads(l) for l in open(f)]
print(len(rows), list(rows[0].keys())[:30])
c=collections.Counter(); ex={}
for r in rows:
    k=(r.get("status"), str(r.get("error"))[:120]); c[k]+=1; ex.setdefault(k,r)
for k,v in c.most_common(8): print(v,k)
PY
echo ---LOADGEN; tail -15 $d/loadgen.log | cut -c1-250
echo ---SERVER; grep -n "Traceback\|Error\|error" /tmp/ax/engine_current.log | tail -5 | cut -c1-250
