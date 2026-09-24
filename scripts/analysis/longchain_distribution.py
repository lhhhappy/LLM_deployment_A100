#!/usr/bin/env python3
"""Offline workload and clone diagnostics against public S1, not representativeness certification."""
from __future__ import annotations

import argparse
from array import array
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sqlite3
import sys
import tempfile

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "s1-dev/harness"))
from s1_common import load_index, in_ttft_gate, Renderer
from s1_loadgen import build_gap_plan
from scripts.analysis.longchain import canonical, digest, file_digest, read_jsonl, tool_pairing


def stats(values):
    values = sorted(values)
    if not values:
        return {"n": 0}
    return {"n": len(values), "sum": sum(values), "mean": sum(values)/len(values),
            **{name: values[min(len(values)-1, int(q*len(values)))]
               for name, q in (("min", 0), ("p50", .5), ("p95", .95), ("p99", .99), ("max", 1))}}


def lcp(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def length_bands(lengths):
    edges = (("1-4", 1, 4), ("5-8", 5, 8), ("9-15", 9, 15),
             ("16-30", 16, 30), ("31-99", 31, 99), ("100+", 100, float("inf")))
    return {name: {"chains": sum(lo <= n <= hi for n in lengths),
                   "requests": sum(n for n in lengths if lo <= n <= hi)} for name, lo, hi in edges}


def normalized_block(block):
    """Normalize only actual call IDs, including their explicit textual references."""
    ids = [str(t.get("id") or t.get("tool_call_id")) for m in block
           for t in m.get("tool_calls", []) if t.get("id") or t.get("tool_call_id")]
    mapping = {old: f"__call_{i}__" for i, old in enumerate(dict.fromkeys(ids))}
    pattern = re.compile(r"(?<![\w-])(?:" + "|".join(re.escape(x) for x in
                         sorted(mapping, key=len, reverse=True)) + r")(?![\w-])") if mapping else None
    def visit(value, path=()):
        if isinstance(value, str):
            if path in (("role",), ("tool_calls", "name"), ("tool_calls", "type"),
                        ("tool_calls", "function", "name")):
                return value
            if path == ("content", "type") and value in ("text", "image_url", "input_text", "input_image"):
                return value
            if path and path[-1] in ("id", "tool_call_id", "call_id", "tool_use_id"):
                return mapping.get(value, value)
            return pattern.sub(lambda m: mapping[m.group()], value) if pattern else value
        if isinstance(value, list):
            return [visit(x, path) for x in value]
        if isinstance(value, dict):
            return {k: visit(v, path+(k,)) for k, v in value.items()}
        return value
    return visit(block)


def blocks(messages):
    result, i = [], 0
    while i < len(messages):
        j = i + 1
        if messages[i].get("tool_calls"):
            while j < len(messages) and messages[j].get("role") == "tool":
                j += 1
        block = messages[i:j]
        if tool_pairing(block) != "incomplete":
            result.append((digest(block), digest(normalized_block(block)), len(canonical(block))))
        i = j
    return result


def duplicate_stats(db, table):
    groups = db.execute(f"SELECT fp, count(*), count(DISTINCT session), count(DISTINCT chain), max(chars) "
                        f"FROM {table} GROUP BY fp HAVING count(*) > 1").fetchall()
    cross = [r for r in groups if r[2] > 1]
    return {"duplicate_groups": len(groups), "cross_session_groups": len(cross),
            "cross_session_occurrences": sum(r[1] for r in cross),
            "cross_session_redundant_occurrences": sum(r[1]-1 for r in cross),
            "cross_chain_groups": sum(r[3] > 1 for r in groups),
            "largest_cross_session_groups": [dict(zip(("sha256", "occurrences", "sessions", "chains", "chars"), r))
                                             for r in sorted(cross, key=lambda x: (-x[1], -x[4]))[:20]]}


def inspect(root, renderer, dbfile):
    rows, chains, grouped = load_index(str(root))
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest.get("status") == "BUILDING":
        raise ValueError(f"dataset is still BUILDING: {root}")
    provenance = {r["req_id"]: r for r in read_jsonl(root / "provenance.jsonl")} if (root / "provenance.jsonl").exists() else {}
    heads = {rs[0]["_req_id"]: cid for cid, rs in grouped.items()}
    db = sqlite3.connect(dbfile)
    db.executescript("""CREATE TABLE snapshots(rid TEXT PRIMARY KEY, blocks TEXT);
        CREATE TABLE prompts(fp TEXT, session TEXT, chain TEXT, chars INTEGER);
        CREATE TABLE introductions(fp TEXT, session TEXT, chain TEXT, chars INTEGER);
        CREATE TABLE normalized_introductions(fp TEXT, session TEXT, chain TEXT, chars INTEGER);""")
    head_tokens, seen = {}, set()
    for path in sorted((root / "bodies").rglob("*.jsonl.gz")):
        for body in read_jsonl(path):
            rid = body.get("req_id")
            if rid not in rows:
                continue
            if rid in seen:
                raise ValueError(f"duplicate body: {rid}")
            seen.add(rid)
            row = rows[rid]
            content = {k: v for k, v in body.items() if k != "req_id"}
            db.execute("INSERT INTO prompts VALUES(?,?,?,?)", (digest(content), row["pack"]+":"+row["session_id"], row["chain_id"], len(canonical(content))))
            db.execute("INSERT INTO snapshots VALUES(?,?)", (rid, json.dumps(blocks(body.get("messages", [])))))
            if renderer and rid in heads:
                tokens = array("I", renderer.tokenizer.encode(renderer.render(body), add_special_tokens=False))
                preamble = {**body, "messages": []}
                prefix = renderer.tokenizer.encode(renderer.render(preamble), add_special_tokens=False)
                head_tokens[heads[rid]] = {"tokens": tokens, "preamble_len": lcp(tokens, prefix),
                                          "source_req_id": provenance.get(rid, {}).get("source_req_id", rid),
                                          "req_id": rid, "session_id": row["session_id"], "pack": row["pack"]}
    if seen != set(rows):
        raise ValueError(f"body coverage mismatch: {len(seen)}/{len(rows)}")
    head_blocks, introduction_count = 0, 0
    for rs in grouped.values():
        previous = Counter()
        for index, row in enumerate(rs):
            current = json.loads(db.execute("SELECT blocks FROM snapshots WHERE rid=?", (row["_req_id"],)).fetchone()[0])
            counts = Counter(x[0] for x in current)
            if index == 0:
                head_blocks += len(current)
            else:
                new = counts - previous
                by_fp = {x[0]: x for x in current}
                for fp, n in new.items():
                    exact, normalized, chars = by_fp[fp]
                    values = (row["pack"]+":"+row["session_id"], row["chain_id"], chars)
                    for _ in range(n):
                        db.execute("INSERT INTO introductions VALUES(?,?,?,?)", (exact, *values))
                        db.execute("INSERT INTO normalized_introductions VALUES(?,?,?,?)", (normalized, *values))
                    introduction_count += n
            previous = counts
    cohort_path = root / "cohort.json"
    cohort = json.loads(cohort_path.read_text()) if cohort_path.exists() else {
        "chains": [{"chain_id": cid, "req_ids": [r["_req_id"] for r in rs]} for cid, rs in grouped.items()]}
    gaps, gap_stats = build_gap_plan(cohort["chains"], rows, 3600000)
    plans = list(read_jsonl(root / "event-plans.jsonl")) if (root / "event-plans.jsonl").exists() else []
    source_phases = Counter()
    for cid in grouped:
        target = chains[cid].get("source_chain_targets", chains[cid])
        source_phases.update(target.get("phases", {}))
    kinds = Counter(r.get("kind", "source") for r in provenance.values())
    origin_counts = Counter()
    source_uses = defaultdict(lambda: {"events": 0, "sessions": set(), "role": None})
    for rid, p in provenance.items():
        if p.get("kind") != "synthetic":
            continue
        origin_counts[p.get("gap_origin", "unknown")] += 1
        materials = [(field, p[field]["fingerprint"]) for field in ("query_material", "answer_material")
                     if isinstance(p.get(field), dict) and p[field].get("fingerprint")]
        if not materials and p.get("donor_fingerprint"):
            materials = [("donor_block", p["donor_fingerprint"])]
        for role, fp in materials:
            value = source_uses[role+":"+fp]
            value["events"] += 1
            value["sessions"].add(rows[rid]["session_id"])
            value["role"] = role
    uses = [{"material": fp, "events": v["events"], "sessions": len(v["sessions"]), "role": v["role"]} for fp, v in source_uses.items()]
    report = {"root": str(root), "manifest_sha256": file_digest(manifest_path) if manifest_path.exists() else None,
              "requests": len(rows), "chains": len(grouped), "sessions": len({(r["pack"], r["session_id"]) for r in rows.values()}),
              "kinds": dict(kinds), "four_gates": {gate: sum(in_ttft_gate(r, gate) for r in rows.values())
                  for gate in ("fast_intra", "overall_intra", "turn_start", "chain_start")},
              "chain_length": stats([len(rs) for rs in grouped.values()]),
              "chain_length_bands": length_bands([len(rs) for rs in grouped.values()]),
              "source_summary_chain_length": stats([chains[cid].get("source_chain_targets", chains[cid])["n_requests"] for cid in grouped]),
              "phase": dict(Counter(r.get("phase") for r in rows.values())),
              "prompt_tokens": stats([r["glm_tokens"] for r in rows.values()]),
              "frozen_uncached_tokens": stats([r["uncached_expected"] for r in rows.values()]),
              "rendered_added_tokens_from_provenance": stats([p["added_tokens"] for p in provenance.values() if "added_tokens" in p]),
              "output_budget": stats([r["max_output_i"] for r in rows.values()]),
              "gap_ms": stats([r.get("replay_gap_ms") or 0 for r in rows.values()]),
              "effective_gap_ms": stats([g for values in gaps.values() for g in values]), "gap_cap": gap_stats,
              "estimated_gap_rows": sum(bool(r.get("gap_imputed")) for r in rows.values()), "synthetic_gap_origins": dict(origin_counts),
              "source_plan_actual": {"source_all": dict(source_phases),
                  "actual_all": dict(Counter(r.get("phase") for r in rows.values())),
                  "planned_synthetic": dict(Counter(e["kind"] for p in plans for e in p["events"])),
                  "actual_synthetic": dict(Counter(p.get("event_kind", rows[rid]["phase"]) for rid,p in provenance.items() if p.get("kind") == "synthetic")),
                  "displaced_planned": dict(Counter(p["displaced_planned_kind"] for p in provenance.values() if p.get("displaced_planned_kind"))),
                  "context_pressure_events": sum(p.get("event_position_origin") == "explicit_context_pressure" for p in provenance.values()),
                  "replaced_behavior_references": sum(bool(e.get("replaced_reference_for_event")) for p in plans for e in p["events"])},
              "repetition": {"complete_prompt_excluding_req_id": duplicate_stats(db, "prompts"),
                  "introduced_blocks_exact": duplicate_stats(db, "introductions"),
                  "introduced_blocks_call_ids_normalized": duplicate_stats(db, "normalized_introductions"),
                  "introduced_block_occurrences": introduction_count, "initial_history_blocks_excluded_from_introductions": head_blocks,
                  "source_material_uses": sorted(uses, key=lambda x: (-x["events"], x["material"]))}}
    db.close()
    return report, head_tokens


def head_comparison(generated, source):
    source_by_req = {h["req_id"]: h for h in source.values()}
    pairs = []
    keys = sorted(generated)
    for i, a in enumerate(keys):
        ga = generated[a]
        sa = source_by_req.get(ga["source_req_id"])
        for b in keys[i+1:]:
            gb = generated[b]
            sb = source_by_req.get(gb["source_req_id"])
            if sa is None or sb is None:
                raise ValueError("corresponding source chain head unavailable")
            gl = lcp(ga["tokens"], gb["tokens"])
            sl = lcp(sa["tokens"], sb["tokens"])
            ge = max(0, gl-min(ga["preamble_len"], gb["preamble_len"]))
            se = max(0, sl-min(sa["preamble_len"], sb["preamble_len"]))
            pairs.append({"chain_a": a, "chain_b": b, "generated_lcp": gl, "source_lcp": sl,
                          "generated_beyond_system_tools": ge, "source_beyond_system_tools": se,
                          "added_shared_history_tokens": ge-se,
                          "same_source_session": (sa["pack"], sa["session_id"]) == (sb["pack"], sb["session_id"])})
    return {"pairs": len(pairs), "generated_lcp": stats([p["generated_lcp"] for p in pairs]),
            "source_lcp": stats([p["source_lcp"] for p in pairs]),
            "generated_beyond_system_tools": stats([p["generated_beyond_system_tools"] for p in pairs]),
            "source_beyond_system_tools": stats([p["source_beyond_system_tools"] for p in pairs]),
            "pairs_with_added_shared_history": sum(p["added_shared_history_tokens"] > 0 for p in pairs),
            "pairs_from_same_source_session": sum(p["same_source_session"] for p in pairs),
            "largest_added_shared_history": sorted(pairs, key=lambda p: (-p["added_shared_history_tokens"], -p["generated_lcp"]))[:20]}


def markdown(result):
    src, gen = result["source"], result["generated"]
    lines = ["# 合成长链分布与重复审计", "", "这是 CPU 分布诊断，不是完整性验收、线上代表性证明或 GPU 成绩。", "",
             f"生成集 `{gen['root']}`；manifest `{gen['manifest_sha256']}`。", "", "| 指标 | 公开 s1-dev | 生成集 |", "|---|---:|---:|"]
    for label, key in (("请求", "requests"), ("链", "chains"), ("session", "sessions")):
        lines.append(f"| {label} | {src[key]} | {gen[key]} |")
    for gate in src["four_gates"]:
        lines.append(f"| {gate} | {src['four_gates'][gate]} | {gen['four_gates'][gate]} |")
    for label, key in (("链长", "chain_length"), ("prompt", "prompt_tokens"), ("冻结新增输入", "frozen_uncached_tokens"),
                       ("输出预算", "output_budget"), ("gap ms", "gap_ms"), ("cap后gap ms", "effective_gap_ms")):
        lines.append(f"| {label} p50 / p95 / max | " + " | ".join(" / ".join(str(x[key].get(k)) for k in ("p50", "p95", "max")) for x in (src, gen)) + " |")
    lines += ["", "源摘要 / 计划 / 实际事件：", "", "```json", json.dumps(gen["source_plan_actual"], ensure_ascii=False, indent=2), "```", "",
              f"生成集估计gap {gen['estimated_gap_rows']} 条；{gen['gap_cap']['n_chains_compressed']} 链触发3600s cap。",
              "Phoenix end-to-start proxy 不等于工具并集与用户思考分解；无法据此宣称恢复了正式到达节奏。", "", "| 重复口径 | 公开 s1-dev 跨session组 | 生成集跨session组 |", "|---|---:|---:|"]
    for key in ("complete_prompt_excluding_req_id", "introduced_blocks_exact", "introduced_blocks_call_ids_normalized"):
        lines.append(f"| {key} | {src['repetition'][key]['cross_session_groups']} | {gen['repetition'][key]['cross_session_groups']} |")
    heads = result.get("head_prefix")
    if heads:
        lines += ["", f"GLM链首比较 {heads['pairs']} 对；较对应source新增共享历史的对数 {heads['pairs_with_added_shared_history']}；来源原本同session的分支对数 {heads['pairs_from_same_source_session']}。"]
    lines += ["", "口径：公开集是链前缀样本，链长/事件量不能直接作为完整线上分布。新增输入主表比较双方冻结uncached_expected；生成集另报provenance中的实际LCP新增，二者不混用。",
              "重复统计先剔除req_id；完整正文包含system/tools/messages。新增块是相邻快照完整消息/工具组多重集合的正差，保留历史与重建保留尾部不重复计数；每条链的入场历史单列，不当新增素材。归一化只改实际调用ID及其引用，保留参数与正文。短通用消息的重复不是自动失败。",
              "生成session独立后，源中同session不同chain的相同入场状态会变成跨session重复；必须对照源head，不能全部当作新克隆。链首LCP扣除两侧system/tools渲染前缀后，再减对应源head同口径值。此处只检测链首，不能证明中后段不存在跨链共享；新增块重复是补充指标。",
              "完整指纹组、来源素材引入次数、分位数及top重复组见同名JSON；没有以原self-check PASS代替独立完整验收。", ""]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--source-root", type=Path, default=REPO / "s1-dev/data/dev-combined-v1")
    ap.add_argument("--tok-dir", type=Path, default=REPO / "s1-dev/glm_tok")
    ap.add_argument("--out", type=Path, required=True, help="JSON path outside dataset roots; sibling .md is also written")
    ap.add_argument("--skip-head-tokens", action="store_true", help="omit token-level head comparison; explicitly reported")
    args = ap.parse_args(argv)
    root, source, out = args.root.resolve(), args.source_root.resolve(), args.out.resolve()
    if any(out == p or p in out.parents for p in (root, source)):
        ap.error("reports must be outside both datasets")
    renderer = None if args.skip_head_tokens else Renderer(str(args.tok_dir.resolve()))
    with tempfile.TemporaryDirectory(prefix="longchain-distribution-") as tmp:
        source_report, source_heads = inspect(source, renderer, Path(tmp)/"source.sqlite")
        print(f"SOURCE audited {source_report['requests']} requests", flush=True)
        generated_report, generated_heads = inspect(root, renderer, Path(tmp)/"generated.sqlite")
        print(f"GENERATED audited {generated_report['requests']} requests", flush=True)
        result = {"diagnostic_only": True, "source": source_report, "generated": generated_report,
                  "head_prefix": head_comparison(generated_heads, source_heads) if renderer else None,
                  "token_head_comparison_skipped": not bool(renderer)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    out.with_suffix(".md").write_text(markdown(result))
    print(f"WROTE {out}", flush=True)


if __name__ == "__main__":
    main()
