# Runs IN THE POD: one JSON line per queue job (done/failed/cancelled): name, status, patches, args, env, key result lines.
import glob, json, os, re
KEYS = re.compile(r"^(LADDER|INTERFERENCE|FIT|CAP_SMOKE|PRECHECK|GATE|NUMCMP|ENGINE_DIED|ENGINE_TIMEOUT|COLD|CAP|prefill batches)")
for st in ("done", "failed", "cancelled"):
    for f in sorted(glob.glob(f"/tmp/ax/queue/{st}/*.sh")):
        name = os.path.basename(f)[:-3]; src = open(f, errors="ignore").read()
        g = lambda k: (re.search(rf'^{k}="?([^"\n]*)"?', src, re.M) or [None, ""])[1]
        pats = g("G_PATCHES") or " ".join(re.findall(r"\d{3}-[a-z0-9-]+\.patch", src))
        args = g("G_ARGS") or " ".join(re.findall(r"--[a-z0-9-]+(?: [^-\s\\][^\s\\]*)?", " ".join(l for l in src.splitlines() if "ensure_engine" in l or l.strip().startswith("--"))))
        env = g("G_ENV") or " ".join(sorted(set(re.findall(r"(SGLANG_AX_[A-Z_]+=[^\s\\]*|SGLANG_MAMBA_SSM_DTYPE=\S+)", src))))
        log = f"/tmp/ax/runs/{name}/job.log"
        lines = [l.rstrip()[:420] for l in open(log, errors="ignore") if KEYS.match(l)] if os.path.exists(log) else []
        print(json.dumps(dict(name=name, status=st, patches=sorted(set(p[:3] for p in pats.split())), args=args[:600], env=env[:500], results=lines[-10:])))
