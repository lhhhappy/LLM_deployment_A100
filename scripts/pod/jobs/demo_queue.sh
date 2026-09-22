# Queue demo (no GPU): build submitted-B code the versioned way and prove the pod imports the copy.
source $AX/bin/scripts/pod/lib.sh
prepare_src b 000-interface-compliance.patch 101-d1v12-on-base.patch || exit 1
cd $RUN_DIR
python3 - <<PY
import sglang, hashlib, pathlib
print("sglang imported from:", sglang.__file__)
for f in ("srt/entrypoints/http_server.py", "srt/managers/schedule_policy.py", "srt/managers/tokenizer_control_mixin.py"):
    p = pathlib.Path(sglang.__file__).parent / f
    t = p.read_text()
    print(f, "patched" if ("[arena D0]" in t or "[ax]" in t) else "PRISTINE", hashlib.sha256(t.encode()).hexdigest()[:12])
PY
echo DEMO_OK
