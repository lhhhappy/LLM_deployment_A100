"""Publish the frozen 061 job without replacing the live engine or shared tools."""
from pathlib import Path
import datetime
import hashlib
import json
import os

name = "062-official_a_180_sched_lite_n30"
stage = Path("/tmp/ax/staging/062-180-sched-lite-n30")
audit_dir = stage / "evidence" / ("L" + name)
receipt = json.loads((audit_dir / "config-audit.json").read_text())
job = stage / "scripts/pod/jobs/official_a_180_sched_lite_n30.sh"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as body:
        for block in iter(lambda: body.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


assert sha(job) == receipt["job_sha256"], "job changed"
for filename, digest in receipt["patch_hashes"].items():
    source = (stage/"patches"/filename) if filename.startswith("180-") else (Path("/tmp/ax/patches")/filename)
    assert sha(source) == digest, filename
for filename, digest in receipt["runtime_tool_hashes"].items():
    assert sha(Path(filename)) == digest, filename
data = Path("/tmp/ax/data/s1-dev-longchain-lite")
assert sha(data / "manifest.json") == receipt["manifest_sha256"]
for filename, digest in receipt["data_artifact_hashes"].items():
    assert sha(data / filename) == digest, filename

queue = Path("/tmp/ax/queue")
for status in ("pending", "running", "done", "failed", "cancelled"):
    existing = queue / status / (name + ".sh")
    if existing.exists():
        assert sha(existing) == receipt["job_sha256"], "queue name has other content"
        print("ALREADY_PRESENT", status, name, flush=True)
        raise SystemExit(0)

predecessors = {
    "060z-official_a_longchain_lite_n30.sh",
    "061-official_a_122_lite_n30.sh",
}
active = {status: sorted(p.name for p in (queue / status).glob("*.sh"))
          for status in ("running", "pending")}
for status, jobs in active.items():
    assert set(jobs) <= predecessors, (status, jobs)
assert (queue / "failed" / "060-official_a_180_hicache_lite_n30.sh").exists(), "old 060 must be stopped first"
assert any((queue/status/"060z-official_a_longchain_lite_n30.sh").exists()
           for status in ("pending", "running", "done", "failed")), "N30 baseline missing"
# All current consumers of the old 180 have ended. Shared runtime tools are untouched.
p180 = Path("/tmp/ax/patches/180-hicache-glm-dsa.patch")
new180 = receipt["patch_hashes"][p180.name]
assert sha(p180) in ("3b63d9c88cb28520102455bc9081e5e8a7d9f9b0654bd1f1dde1f401747c8848", new180)
if sha(p180) != new180:
    (stage/"previous-180.patch").write_bytes(p180.read_bytes())
    temp = p180.with_suffix(".062-part")
    temp.write_bytes((stage/"patches"/p180.name).read_bytes())
    temp.replace(p180)
assert sha(p180) == new180
assert not (queue / "PAUSE").exists(), "queue paused"
run = Path("/tmp/ax/runs") / name
run.mkdir(exist_ok=True)
saved = run / "deployment-receipt.json"
if saved.exists():
    assert json.loads(saved.read_text())["job_sha256"] == receipt["job_sha256"]
receipt.update(status="PUBLISHING", queued_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               predecessors_at_publish=active)
saved.write_text(json.dumps(receipt, indent=2) + "\n")
os.link(job, queue / "pending" / (name + ".sh"))
print("QUEUED", name, "after=061 patches=14 tools_unchanged=7 new180=8312cbf7 data=058 N30 fresh_flush no_predecessor_PASS_gate", flush=True)
