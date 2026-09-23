#!/usr/bin/env python3
"""check_submission.py — offline validator for submission.json against llm-challenge-arena-v1/task.md.

Checks (ERROR = task.md rule or platform-observed rejection; WARN = advice):
  * image and command required; env and model_name optional (no extras, e.g. base_url, api_key)
  * image, command non-empty strings; image as name:tag (no @sha256, platform rejects it), on registry.dp.tech;
    :latest warns because a mutable tag is hard to reproduce
  * command is argv, not shell: first token is not KEY=VAL, no shell operators (&& || | ; > < etc.),
    no $ expansion or backticks
  * --served-model-name consistent with model_name (advice)
  * env is an object of string -> string; no base_url/api_key anywhere; SGLang needs
    SGLANG_OPT_USE_TOPK_V2=0 on A100
  * (SGLang commands) every --flag exists in the exact base package's server_args.py
  * (optional --trace) stub trace: first line session_start, then >=1 user/assistant line
    (`playground trace validate` 0.1.39 accepts a session_start-only file, the server does not).

Usage: scripts/check_submission.py submission/candidate-01.json [--trace T.jsonl] [--final]
       [--sglang-src build/base_exact/sglang/srt] [--skip-flag-check]
Exit 0 = no errors. Pure local; no network.
"""
import argparse
import json
import re
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIELDS = {"image", "command", "env", "model_name"}
REQUIRED_FIELDS = {"image", "command"}
FORBIDDEN_KEYS = {"base_url", "api_key", "apikey", "api-key"}
SHELL_OPS = {"&&", "||", "|", ";", ">", ">>", "<", "<<", "&", "2>", "2>&1", "|&", ";;", "(", ")"}
KV_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


class Report:
    def __init__(self):
        self.errors, self.warns, self.oks = [], [], []

    def err(self, m):
        self.errors.append(m)

    def warn(self, m):
        self.warns.append(m)

    def ok(self, m):
        self.oks.append(m)


def walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from walk_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk_keys(v)


def flag_value(argv, flag):
    vals = []
    for i, t in enumerate(argv):
        if t == flag and i + 1 < len(argv):
            vals.append(argv[i + 1])
        elif t.startswith(flag + "="):
            vals.append(t.split("=", 1)[1])
    return vals


def sglang_known_flags(srt: Path):
    flags = set()
    for f in sorted((srt / "arg_groups" / "fields").glob("*.py")):
        for m in re.finditer(r"^[ \t]{4}([a-z_][a-z0-9_]*)\s*:\s*A\[", f.read_text(), re.M):
            flags.add("--" + m.group(1).replace("_", "-"))
    sa = srt / "server_args.py"
    if sa.exists():
        source = sa.read_text()
        flags.update("--" + m.replace("_", "-") for m in
                     re.findall(r"^[ \t]{4}([a-z_][a-z0-9_]*)\s*:\s*A\[", source, re.M))
        flags.update(re.findall(r"[\"'](--[a-z0-9][a-z0-9-]*)[\"']", source))
    return flags


def check_trace(path: Path, r: Report):
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    try:
        objs = [json.loads(l) for l in lines]
    except json.JSONDecodeError as e:
        return r.err(f"trace: invalid JSON line ({e})")
    if not objs or objs[0].get("type") != "session_start":
        return r.err("trace: first line must be {\"type\":\"session_start\",...}")
    roles = [o.get("role") for o in objs[1:]]
    if not any(x in ("user", "assistant") for x in roles):
        return r.err("trace: needs >=1 user/assistant line after session_start")
    if re.search(r"(sk-|token|Bearer\s)", path.read_text(), re.I):
        r.warn("trace: contains something token-like; task.md says do not put tokens in the trace")
    r.ok(f"trace: session_start + {len(objs) - 1} message line(s) ({', '.join(map(str, roles))})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("submission")
    ap.add_argument("--trace")
    ap.add_argument("--final", action="store_true", help="treat placeholder/unpinned image as error")
    ap.add_argument("--sglang-src", default=str(ROOT / "build/base_exact/sglang/srt"))
    ap.add_argument("--skip-flag-check", action="store_true", help="skip only the local SGLang flag lookup")
    a = ap.parse_args()
    r = Report()

    try:
        sub = json.loads(Path(a.submission).read_text())
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: cannot parse {a.submission}: {e}")
        return 1
    if not isinstance(sub, dict):
        print("ERROR: top level must be a JSON object")
        return 1

    keys = set(sub)
    if keys - FIELDS:
        r.err(f"unexpected fields: {sorted(keys - FIELDS)}")
    if REQUIRED_FIELDS - keys:
        r.err(f"missing required fields: {sorted(REQUIRED_FIELDS - keys)}")
    if not (keys - FIELDS or REQUIRED_FIELDS - keys):
        r.ok("submission fields match task.md schema")
    bad = sorted({k for k in walk_keys(sub) if str(k).lower() in FORBIDDEN_KEYS})
    if bad:
        r.err(f"forbidden keys present: {bad}")

    # image
    img = sub.get("image")
    if not isinstance(img, str) or not img.strip():
        r.err("image must be a non-empty string")
    else:
        sev = r.err if a.final else r.warn
        if "REPLACE" in img or "..." in img:
            sev(f"image is a placeholder: {img}")
        else:
            if not img.startswith("registry.dp.tech/"):
                sev("image is not on registry.dp.tech (Trisol only pulls LBG-registered images)")
            # Platform fact (attempts 45734/45735, 2026-09-22): the image center rejects
            # "name:tag@sha256:..." with "imageName format is not right" (deploy fails).
            # The organizer's working example uses plain "name:tag". Use a unique, never-rebuilt tag.
            if "@sha256:" in img:
                r.err("image must be name:tag; the platform rejects tag@sha256 digests (F55)")
            elif not re.search(r":[A-Za-z0-9_.-]+$", img.rsplit("/", 1)[-1]):
                sev("image has no tag")
            if img.endswith(":latest"):
                r.warn("image tag is latest; use a unique tag for reproducibility")
        r.ok("image is a non-empty string")

    # command
    cmd = sub.get("command")
    argv = []
    if not isinstance(cmd, str) or not cmd.strip():
        r.err("command must be a non-empty string")
    else:
        try:
            argv = shlex.split(cmd)
            lex = shlex.shlex(cmd, posix=True, punctuation_chars=True)
            lex.whitespace_split = True
            ptoks = list(lex)
        except ValueError as e:
            r.err(f"command does not tokenize as argv: {e}")
            ptoks = []
        if argv:
            if KV_RE.match(argv[0]):
                r.err(f"first token '{argv[0]}' is KEY=VAL; env vars go in env, command is exec'd as argv")
            else:
                r.ok(f"first token is an executable: {argv[0]}")
            ops = [t for t in ptoks if t in SHELL_OPS or re.fullmatch(r"[();<>|&]+", t)]
            if ops:
                r.err(f"shell operators in command (no shell is used): {ops}")
            if "$" in cmd or "`" in cmd:
                r.err("command contains $ or backtick; no shell expansion happens")
            if not ops and "$" not in cmd and "`" not in cmd:
                r.ok("no shell operators / expansions")
            for tok in argv[1:]:
                if KV_RE.match(tok) and not tok.startswith("-"):
                    r.warn(f"argv token looks like KEY=VAL: {tok}")
            flags = [t.split("=", 1)[0] for t in argv if t.startswith("--")]
            dups = sorted({f for f in flags if flags.count(f) > 1})
            if dups:
                r.warn(f"duplicate flags: {dups}")

    # model_name vs --served-model-name
    mn = sub.get("model_name", "default")
    if "model_name" in sub and (not isinstance(mn, str) or not mn):
        r.err("model_name must be a non-empty string when provided")
    elif argv:
        smn = flag_value(argv, "--served-model-name")
        if not smn:
            r.warn("--served-model-name missing; check that the service accepts model_name")
        elif smn[-1] != mn:
            r.warn(f"--served-model-name {smn[-1]!r} != model_name {mn!r}; check the served model id")
        else:
            r.ok(f"--served-model-name == model_name == {mn!r}")

    # env
    env = sub.get("env", {})
    if not isinstance(env, dict):
        r.err("env must be an object of string -> string")
        env = {}
    else:
        nonstr = [k for k, v in env.items() if not isinstance(k, str) or not isinstance(v, str)]
        if nonstr:
            r.err(f"env values must be strings: {nonstr}")
        else:
            r.ok(f"env: {len(env)} string->string entries")
        for k in env:
            if re.search(r"(SECRET|PASSWORD|API_KEY|ACCESS_KEY|CREDENTIAL|(^|_)TOKEN$)", k, re.I):
                r.warn(f"env key {k} looks like a credential; no secrets in the submission")

    # SGLang-specific
    is_sglang = any("sglang" in t for t in argv[:4])
    if is_sglang:
        if env.get("SGLANG_OPT_USE_TOPK_V2") != "0":
            r.err("SGLang on A100 requires env SGLANG_OPT_USE_TOPK_V2=\"0\" (task.md)")
        else:
            r.ok("SGLANG_OPT_USE_TOPK_V2=0 set")
        if flag_value(argv, "--model-path") != ["/mnt/models"]:
            r.warn("expected exactly one --model-path /mnt/models")
        host, port = flag_value(argv, "--host"), flag_value(argv, "--port")
        if host != ["0.0.0.0"] or port != ["8000"]:
            r.warn(f"expected --host 0.0.0.0 --port 8000, got host={host} port={port}")
        srt = Path(a.sglang_src)
        if a.skip_flag_check:
            r.warn("local SGLang flag lookup skipped")
        elif srt.is_dir():
            known = sglang_known_flags(srt)
            used = sorted({t.split("=", 1)[0] for t in argv if t.startswith("--")})
            unknown = [f for f in used if f not in known]
            if unknown:
                r.err(f"flags not found in {srt} arg_groups/server_args: {unknown}")
            else:
                r.ok(f"all {len(used)} flags exist in local sglang source ({len(known)} known)")
        else:
            r.warn(f"sglang source {srt} not found; flags not verified")

    if a.trace:
        check_trace(Path(a.trace), r)

    for m in r.oks:
        print(f"OK    {m}")
    for m in r.warns:
        print(f"WARN  {m}")
    for m in r.errors:
        print(f"ERROR {m}")
    print(f"=> {len(r.errors)} error(s), {len(r.warns)} warning(s)")
    return 1 if r.errors else 0


if __name__ == "__main__":
    sys.exit(main())
