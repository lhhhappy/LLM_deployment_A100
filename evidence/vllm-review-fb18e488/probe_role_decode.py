"""CPU lifecycle probe for the 101 source snapshot; never executes model/GPU code.

Run with the existing vLLM venv, CUDA_VISIBLE_DEVICES empty, from a directory
containing the source overlay under engine/vllm and its test helpers.
"""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
OVERLAY = ROOT / "engine/vllm"

# Reuse installed dependencies and binaries; isolate the reviewed Python files.
import vllm

installed_source = Path(vllm.__file__).resolve().parent.parent
sys.path.insert(0, str(installed_source))
for name in ("vllm", "vllm.v1", "vllm.v1.core", "vllm.v1.core.sched"):
    package = importlib.import_module(name)
    package.__path__.insert(0, str(OVERLAY.joinpath(*name.split("."))))

path = OVERLAY / "tests/v1/core/prefix_cache/test_mamba_role_checkpoint.py"
spec = importlib.util.spec_from_file_location("role_fixture", path)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
from vllm.v1.request import RequestStatus


def run_case(shared, suffix, decode_tokens, unit=32):
    manager = fixture._manager(512, unit)
    stub = fixture._stub(manager, 512, unit)
    owner = fixture.make_request(
        "owner", fixture._turn(shared, suffix), unit, fixture.sha256
    )
    owner.max_tokens = decode_tokens + 1
    owner.sampling_params.max_tokens = decode_tokens + 1
    fixture._prefill(manager, stub, owner)
    owner.append_output_token_ids(7)  # token sampled by the final prefill step
    before = fixture._follow_up_hit(manager, shared, unit)
    for _ in range(decode_tokens):
        assert manager.allocate_slots(owner, 1, has_scheduled_reqs=False) is not None
        owner.num_computed_tokens += 1
        owner.append_output_token_ids(7)  # next sample; the last stays uncomputed
        _, retained = manager.take_kv_cache_block_copies()
        if retained:
            manager.block_pool.free_blocks(retained)
        manager.new_step_starts()
    after_decode = fixture._follow_up_hit(manager, shared, unit)
    assert owner.num_output_tokens == owner.max_tokens
    owner.status = RequestStatus.FINISHED_LENGTH_CAPPED
    manager.free(owner)
    after_free = fixture._follow_up_hit(manager, shared, unit)
    return {
        "shared": shared, "suffix": suffix, "prompt_tokens": owner.num_prompt_tokens,
        "block_size": 512, "hash_unit": unit, "output_budget": owner.max_tokens,
        "decode_tokens": decode_tokens, "role_boundary": owner.role_boundary,
        "role_checkpoint": owner.role_checkpoint,
        "hit_after_prefill": before, "hit_after_decode": after_decode,
        "hit_after_owner_free": after_free,
    }


cases = [run_case(2021, 538, 0), run_case(2200, 100, 200),
         run_case(2200, 100, 300), run_case(2021, 430, 300),
         run_case(2200, 100, 300, unit=64),
         run_case(2120, 430, 128, unit=64)]
result = {
    "method": "real KVCacheManager and 101 chunking; no tensor/DMA execution",
    "cases": cases,
    "source_sha256": {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(OVERLAY.rglob("*.py"))
    },
}
body = json.dumps(result, indent=2) + "\n"
assert len(body) < 64 * 1024
(ROOT / "role_decode_summary.json").write_text(body)
print(json.dumps(cases, indent=2))
