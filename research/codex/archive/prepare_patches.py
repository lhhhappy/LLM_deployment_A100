#!/usr/bin/env python3
"""Generate reviewable patches without changing the inspected SGLang checkout.

Exact source anchors fail closed on incompatible versions. The organizer image
must be inspected separately before applying these patches there.
"""
import argparse
import ast
import difflib
from pathlib import Path

CONTROL_PATH = "python/sglang/srt/managers/tokenizer_control_mixin.py"
HTTP_PATH = "python/sglang/srt/entrypoints/http_server.py"
OLD_CONTROL = '''        result = (
            await self.flush_cache_communicator(FlushCacheReqInput(timeout_s=timeout_s))
        )[0]
        if result.success and self.mm_processor is not None:
'''
NEW_CONTROL = '''        results = await self.flush_cache_communicator(
            FlushCacheReqInput(timeout_s=timeout_s)
        )
        # A DP flush is successful only when every worker has acknowledged it.
        result = FlushCacheReqOutput(
            success=bool(results) and all(r.success for r in results),
            message=" | ".join(r.message for r in results if r.message),
        )
        if result.success and self.mm_processor is not None:
'''
OLD_HTTP = '''    if ret.success:
        content = (
            "Cache flushed.\\nPlease check backend logs for more details. "
            "(When there are running or waiting requests, the operation will not be performed.)\\n"
        )
    else:
        content = ret.message or "Flush cache failed.\\n"
    return Response(
        content=content,
        status_code=200 if ret.success else HTTPStatus.BAD_REQUEST,
    )
'''
NEW_HTTP = '''    return ORJSONResponse(
        content={"success": ret.success, "message": ret.message or ""},
        status_code=200 if ret.success else HTTPStatus.BAD_REQUEST,
    )
'''
ROUTING_ANCHOR = '''    if envs.SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES.get():
        apply_header_overrides(obj, request.headers)
    if obj.stream:
'''
ROUTING_REPLACEMENT = '''    if envs.SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES.get():
        apply_header_overrides(obj, request.headers)
    # Optional deterministic affinity control for arena A/B experiments.
    # The same prompt/cache namespace is still subject to a real global flush.
    affinity = os.environ.get("ARENA_DP_AFFINITY", "off")
    if affinity not in ("off", "session", "prefix"):
        raise HTTPException(status_code=400, detail="Invalid ARENA_DP_AFFINITY")
    dp_size = get_parallel().dp_size
    if affinity != "off" and dp_size > 1 and obj.routed_dp_rank is None:
        header = "x-s1-session-id" if affinity == "session" else "x-s1-routing-key"
        key = request.headers.get(header)
        if key:
            import hashlib

            digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
            obj.routed_dp_rank = int.from_bytes(digest, "big") % dp_size
    if obj.stream:
'''


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Source anchor mismatch; review engine version before patching")
    patched = text.replace(old, new, 1)
    ast.parse(patched)
    return patched


def diff(path, old, new):
    return "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
        fromfile="a/" + path, tofile="b/" + path))


def prepare(source, out):
    control = (source / CONTROL_PATH).read_text()
    http = (source / HTTP_PATH).read_text()
    control_fixed = replace_once(control, OLD_CONTROL, NEW_CONTROL)
    http_fixed = replace_once(http, OLD_HTTP, NEW_HTTP)
    http_affinity = replace_once(http_fixed, ROUTING_ANCHOR, ROUTING_REPLACEMENT)
    out.mkdir(parents=True, exist_ok=True)
    (out / "0001-flush-json-all-dp-workers.patch").write_text(
        diff(CONTROL_PATH, control, control_fixed) + diff(HTTP_PATH, http, http_fixed))
    (out / "0002-optional-header-affinity.patch").write_text(diff(HTTP_PATH, http_fixed, http_affinity))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2] / "src/sglang")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "patches")
    args = p.parse_args()
    prepare(args.source, args.out)
    print(f"Generated patches in {args.out}; source checkout unchanged.")
