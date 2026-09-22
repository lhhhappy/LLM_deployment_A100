j=$(ls /tmp/ax/queue/running | sed s/.sh//); echo RUN=$j; d=/tmp/ax/runs/$j
tail -3 $d/job.log | cut -c1-200; ls $d/dev 2>/dev/null | tr '\n' ' '; echo
f=$(ls -t $d/dev/raw_*.jsonl 2>/dev/null | head -1); [ -n "$f" ] && python3 - "$f" <<'PY'
import json,sys,collections
rows=[json.loads(l) for l in open(sys.argv[1])]
c=collections.Counter((r.get("error") is None) for r in rows)
t=sorted(r["ttft_s"] for r in rows if r.get("ttft_s") is not None)
print("rows",len(rows),"ok/err",c, "ttft p50/p95/max", t[len(t)//2] if t else None, t[int(len(t)*.95)] if t else None, t[-1] if t else None)
PY
tail -2 $d/dev/loadgen.log | cut -c1-200
grep "Decode batch" /tmp/ax/engine_current.log >/dev/null; L=$(ls -t /tmp/ax/runs/*/server.log | head -1); grep "Decode batch\|Prefill batch" $L | tail -3 | cut -c1-260; grep -c "Scheduler hit an exception" $L
