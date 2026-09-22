L=$(ls -t /tmp/ax/runs/*/server.log | head -1)
python3 - "$L" <<'PY'
import re,sys,collections
prev=None; dec=collections.defaultdict(list); pre=[]
for line in open(sys.argv[1]):
    if "TP0]" not in line: continue
    m=re.search(r"Decode batch, #running-req: (\d+).*gen throughput \(token/s\): ([\d.]+)",line)
    if m:
        if prev=="D": dec[int(m.group(1))].append(float(m.group(2)))
        prev="D"; continue
    m=re.search(r"Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+).*input throughput \(token/s\): ([\d.]+)",line)
    if m: pre.append((int(m.group(2)),int(m.group(3)),float(m.group(4)))); prev="P"
for bs in sorted(dec):
    v=sorted(dec[bs]); print(f"decode bs={bs} n={len(v)} tok/s p50={v[len(v)//2]:.1f} max={v[-1]:.1f} => ms/step≈{1000*bs/v[len(v)//2]:.0f}")
big=[p for p in pre if p[0]>=2048]
print("prefills",len(pre),"big",len(big))
for p in big[-12:]: print("  new",p[0],"cached",p[1],"in tok/s",p[2])
PY
