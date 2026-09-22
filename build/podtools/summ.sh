d=/tmp/ax/runs/$1/dev
python3 - "$d" <<'PY'
import json,sys,glob
d=sys.argv[1]; s=json.load(open(d+"/summary.json"))
def walk(o,p=""):
    if isinstance(o,dict):
        for k,v in o.items(): walk(v,p+"."+k if p else k)
    elif isinstance(o,list) and len(o)<=6: print(p,"=",o)
    elif not isinstance(o,list): print(p,"=",o)
walk(s)
PY
ls $d; grep -c "compile after serving started" /tmp/ax/engine_current.log 2>/dev/null
