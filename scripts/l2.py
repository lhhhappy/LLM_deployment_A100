#!/usr/bin/env python3
"""L2 (Trisol 8-GPU self-test) queue: one entry point. See tests/L2.md.

  l2.py configs                          list testable configs (scripts/session_a/configs.json)
  l2.py new NAME (--base CONFIG | --profile P) [--patches 000,1xx] [--policy lpm] [--d1|--no-d1]
               [--args "--flag v --bare"] [--env K=V ...] [--note TEXT]
                                         register a new config = base + one delta (a new idea)
  l2.py batch CATALOG.json               register + enqueue every task in a catalog, in order
  l2.py add CONFIG [--slug S] [--hint 14] [--levels 6,10,14] [--budget 300] [--note TEXT]
                                         enqueue + approve one item (runs automatically)
  l2.py ls                               queue and item status
  l2.py pause | resume                   queue-wide STOP file

Items are approved on enqueue: the user delegated self-test approval to Claude (2026-09-22).
Official submissions are a separate queue (submission/queue) and are never touched here.
"""
import argparse
import datetime
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "scripts/session_a/configs.json"
QUEUE = ROOT / "tests/queue"
BASE_IMAGE = "registry.dp.tech/dptech/dp/native/prod-3732438/20675/arena-sglang-glm53:260918"
SERVICE = "lh-arena-sess-a"


def load_configs():
    return json.loads(CONFIGS.read_text())


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cmd_configs(_):
    for name, c in load_configs().items():
        delta = " ".join(f"{k} {'' if v is None else v}".strip() for k, v in c.get("args", {}).items())
        delta += "".join(f" {k}={v}" for k, v in c.get("env", {}).items())
        print(f"{name:22} {','.join(c['patches']):8} {c['policy']:10} d1={'on ' if c['d1'] else 'off'} "
              f"{delta or '-':40} {c.get('note', '')}")


def parse_args_delta(text):
    """"--a 1 --flag --b x" -> {"--a": "1", "--flag": None, "--b": "x"} (argv, no shell)."""
    tokens, out, i = (text or "").split(), {}, 0
    while i < len(tokens):
        if not tokens[i].startswith("--"):
            sys.exit(f"bad --args token {tokens[i]!r}")
        has_value = i + 1 < len(tokens) and not tokens[i + 1].startswith("--")
        out[tokens[i]] = tokens[i + 1] if has_value else None
        i += 2 if has_value else 1
    return out


def register(name, base=None, profile=None, patches=None, policy=None, d1=None,
             args=None, env=None, note=""):
    if not re.fullmatch(r"img_[a-z0-9_]{1,40}", name):
        sys.exit("config name must look like img_<lowercase>")
    configs = load_configs()
    if base and base not in configs:
        sys.exit(f"unknown base {base}")
    c = dict(configs[base]) if base else {"patches": ["000"], "policy": "lpm", "d1": False}
    c.pop("note", None)
    c["args"] = {**c.get("args", {}), **(args or {})}
    c["env"] = {**c.get("env", {}), **(env or {})}
    for key, value in (("profile", profile), ("patches", patches), ("policy", policy), ("d1", d1)):
        if value is not None:
            c[key] = value
    c["note"] = note
    if not c.get("profile"):
        sys.exit("need --base or --profile")
    if not re.fullmatch(r"candidate-b[^/]*\.json", Path(c["profile"]).name) or not (ROOT / c["profile"]).is_file():
        sys.exit("profile must be an existing submission/candidate-b*.json (submission format)")
    for prefix in c["patches"]:
        if len(list((ROOT / "patches").glob(prefix + "-*.patch"))) != 1:
            sys.exit(f"need exactly one patches/{prefix}-*.patch")
    if c["patches"][0] != "000" or any(not re.fullmatch(r"1\d\d", p) for p in c["patches"][1:]):
        sys.exit("base-line patches: 000 first, then 1xx patches made against the organizer base")
    if any(k in c["args"] for k in ("--port", "--schedule-policy")):
        sys.exit("--port/--schedule-policy are set by the runner; use --policy")
    if configs.get(name) not in (None, c):
        sys.exit(f"{name} already registered with a different definition")
    configs[name] = c
    CONFIGS.write_text(json.dumps(configs, indent=1, ensure_ascii=False) + "\n")
    return c


def cmd_new(a):
    env = dict(kv.split("=", 1) for kv in (a.env or []))
    register(a.name, a.base, a.profile, a.patches.split(",") if a.patches else None, a.policy,
             a.d1, parse_args_delta(a.args), env, a.note or "")
    print(f"registered {a.name}; next: scripts/l2.py add {a.name}")


def cmd_batch(a):
    catalog = json.loads(Path(a.catalog).read_text())
    for t in catalog["tasks"]:
        if t.get("skip"):
            continue
        if t.get("base") or t["config"] not in load_configs():
            register(t["config"], t.get("base"), t.get("profile"), t.get("patches"), t.get("policy"),
                     t.get("d1"), t.get("args"), t.get("env"), t.get("note", ""))
        add(t["config"], t.get("slug"), t.get("hint", 14), t.get("levels"), t.get("budget", 300),
            t.get("note"), t.get("ladder_mode", "fast"))


def cmd_add(a):
    add(a.config, a.slug, a.hint, a.levels, a.budget, a.note)


def add(config, slug=None, hint=14, levels=None, budget=300, note=None, mode="fast"):
    configs = load_configs()
    if config not in configs:
        sys.exit(f"unknown config {config}; see: scripts/l2.py configs")
    if isinstance(levels, str):
        levels = [int(x) for x in levels.split(",")]
    ladder = ({"mode": "levels", "levels": levels, "max_n": 30} if levels
              else {"mode": "official-climb", "max_n": 30} if mode == "official-climb"
              else {"mode": "fast", "hint": hint, "max_n": 30})
    spec = {"image_ref": BASE_IMAGE, "profiles": [configs[config]["profile"]],
            "service_name": SERVICE, "max_admission_wait_minutes": 2880,
            "matrix": [config], "ladder": ladder, "time_budget_minutes": budget,
            "notes": note or configs[config].get("note", "")}
    slug = slug or config.replace("_", "-")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(spec, f)
    out = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/trisol_test_daemon.py"),
                          "--root", str(ROOT), "--enqueue", f.name, "--slug", slug],
                         capture_output=True, text=True)
    Path(f.name).unlink()
    if out.returncode:
        sys.exit("enqueue failed: " + (out.stderr or out.stdout).strip())
    item = Path(out.stdout.strip().splitlines()[-1])
    (item / "APPROVED").write_text(f"approved_by: Claude (self-test approval delegated by the user)\n"
                                   f"approved_at: {now()}\nscope: {item.name} — Trisol self-test only\n")
    (item / "notes.md").write_text(f"Enqueued and approved by scripts/l2.py at {now()}.\n")
    print(f"queued {item.name}")


def cmd_ls(_):
    state = ROOT / "data/trisol_tests.json"
    items = json.loads(state.read_text()).get("items", {}) if state.exists() else {}
    print("queue:", "PAUSED (tests/queue/STOP)" if (QUEUE / "STOP").exists() else "running")
    for item in sorted(p for p in QUEUE.iterdir() if p.is_dir() and re.match(r"\d\d-", p.name)):
        spec = json.loads((item / "spec.json").read_text())
        status = items.get(item.name, {}).get("status") or ("approved" if (item / "APPROVED").exists() else "unapproved")
        print(f"  {item.name:24} {','.join(spec['matrix']):14} {status}")
    print("details: python3 scripts/test_status.py (daemon ledger + results)")


def cmd_pause(_):
    (QUEUE / "STOP").write_text(f"paused via scripts/l2.py at {now()}\n")
    print("paused")


def cmd_resume(_):
    (QUEUE / "STOP").unlink(missing_ok=True)
    print("resumed")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("configs").set_defaults(fn=cmd_configs)
    n = sub.add_parser("new")
    n.add_argument("name"); n.add_argument("--base"); n.add_argument("--profile")
    n.add_argument("--patches"); n.add_argument("--policy")
    n.add_argument("--d1", action=argparse.BooleanOptionalAction, default=None)
    n.add_argument("--args"); n.add_argument("--env", action="append"); n.add_argument("--note")
    n.set_defaults(fn=cmd_new)
    d = sub.add_parser("add")
    d.add_argument("config"); d.add_argument("--slug"); d.add_argument("--hint", type=int, default=14)
    d.add_argument("--levels"); d.add_argument("--budget", type=int, default=300); d.add_argument("--note")
    d.set_defaults(fn=cmd_add)
    b = sub.add_parser("batch"); b.add_argument("catalog"); b.set_defaults(fn=cmd_batch)
    for name, fn in (("ls", cmd_ls), ("pause", cmd_pause), ("resume", cmd_resume)):
        sub.add_parser(name).set_defaults(fn=fn)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
