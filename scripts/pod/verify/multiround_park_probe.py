#!/usr/bin/env python3
"""Exclusive queue-job diagnostic, not a harness workload or an SLO result.

Run ON and OFF separately. OFF repeats the same completion cases to measure
numerical noise; compare never treats a missing yield or a missing arm as PASS.
Only this client's RIDs are aborted on failure. The service is never stopped.
"""

import argparse
import asyncio
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import time
import uuid


CASES = (("grid_minus", 196607, 0), ("grid_exact", 196608, 0),
         ("grid_plus", 196609, 0), ("prefix_fork", 196609, 4097))
TOKENS = 12


def prompt(n, seed):
    rng = random.Random(seed)
    # Ordinary valid token IDs, disjoint first token for A and B. This is a
    # synthetic state fixture; no public/hidden evaluation prompt is modified.
    return [rng.randrange(1000, 9000) for _ in range(n)]


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


class Probe:
    def __init__(self, args, http):
        self.args, self.http = args, http
        self.prefix = "124m-probe-" + uuid.uuid4().hex + "-"
        self.tasks, self.records, self.events, self.errors = {}, [], [], []
        self.admitted = set()
        self.log = args.server_log.open()
        self.log.seek(0, 2)  # Never consume a previous job's evidence.
        self.pending_line = ""
        self.event_out = (args.out / "events.jsonl").open("w")
        self.response_out = (args.out / "responses.jsonl").open("w")
        self.coverage = []

    def poll(self):
        lines = (self.pending_line + self.log.read()).split("\n")
        self.pending_line = lines.pop()
        for line in lines:
            if any(marker in line.lower() for marker in (
                "traceback (most recent call last)", "runtimeerror:",
                "assertionerror", "out of memory", "memory leak detected",
                "parked request changed live kv ownership")):
                self.errors.append(line)
            for marker in ("[ax-124m] ", "[ax-prefix-decision] "):
                if marker not in line:
                    continue
                try:
                    obj = json.loads(line.split(marker, 1)[1])
                except ValueError:
                    self.errors.append("malformed mechanism record: " + line)
                    continue
                if marker == "[ax-prefix-decision] ":
                    self.admitted.update(r for r in obj["admitted"] if r.startswith(self.prefix))
                else:
                    self.events.append(obj)
                    self.event_out.write(json.dumps(obj) + "\n")
                    self.event_out.flush()

    async def post(self, endpoint, payload=None, timeout=75):
        async with self.http.post(self.args.base_url + endpoint, json=payload,
                                  timeout=timeout) as response:
            text = await response.text()
            return response.status, json.loads(text) if text else {}

    async def flush(self, busy=False):
        # A busy response must not become an idle success by waiting for the
        # parked requests to finish. After HTTP completion, allow HiCache's
        # in-flight writes and the final overlap callback to retire normally.
        status, body = await self.post(f"/flush_cache?timeout={0 if busy else 5}", timeout=10)
        expected = (400, False) if busy else (200, True)
        if (status, body.get("success")) != expected:
            raise RuntimeError(f"flush expected {expected}, received {(status, body)}")
        return time.perf_counter()

    async def wait(self, predicate, seconds, description):
        stop = time.monotonic() + seconds
        while True:
            self.poll()
            if self.errors:
                raise RuntimeError("server error; see receipt/server.log")
            result = predicate()
            if result:
                return result
            if time.monotonic() >= stop:
                raise RuntimeError("NOT_COVERED: " + description)
            await asyncio.sleep(0.05)

    def send(self, case, role, ids):
        rid = self.prefix + case + "-" + role

        async def generate():
            started = time.perf_counter()
            status, body = await self.post("/generate", {
                "rid": rid, "input_ids": ids, "stream": False,
                "sampling_params": {"temperature": 0, "max_new_tokens": TOKENS,
                                    "ignore_eos": True},
                "return_logprob": True, "logprob_start_len": -1,
                "top_logprobs_num": 8,
            })
            record = dict(case=case, role=role, rid=rid, input_length=len(ids),
                          input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                          started=started, finished=time.perf_counter(), status=status,
                          output_ids=body.get("output_ids"), meta=body.get("meta_info", {}))
            if status != 200:
                record["error"] = body
            self.records.append(record)
            self.response_out.write(json.dumps(record) + "\n")
            self.response_out.flush()
            return record

        self.tasks[rid] = asyncio.create_task(generate())
        return rid

    def successful(self, record):
        meta = record["meta"]
        lp = meta.get("output_token_logprobs", [])
        ids = record["output_ids"]
        if (record["status"] != 200 or not ids or len(ids) != TOKENS
                or meta.get("completion_tokens") != TOKENS
                or meta.get("prompt_tokens") != record["input_length"]
                or len(lp) != TOKENS or [p[1] for p in lp] != ids
                or any(not math.isfinite(p[0]) for p in lp)
                or meta.get("finish_reason", {}).get("type") != "length"):
            raise RuntimeError("incomplete/non-finite/aborted output: " + record["rid"])

    async def pair(self, case, size, seed_size=0, abort_role=None):
        await self.flush()
        a_ids, b_ids = prompt(size, 12401), prompt(33537, 12402)
        if seed_size:
            seed = self.send(case, "seed", a_ids[:seed_size])
            self.successful(await self.tasks[seed])
            # Do not flush here: the next request must really hit a checkpoint.
        start_events = len(self.events)
        a = self.send(case, "A", a_ids)
        await self.wait(lambda: a in self.admitted, 15, case + ": A never admitted")
        await asyncio.sleep(1.2)
        if self.tasks[a].done():
            raise RuntimeError("NOT_COVERED: A completed before B arrived")
        b = self.send(case, "B", b_ids)
        event = None
        if self.args.mode == "on":
            event = await self.wait(lambda: next((e for e in self.events[start_events:]
                if e.get("event") == "yield" and e.get("a") == a and e.get("b") == b), None),
                12, case + ": no actual A -> B yield")
            if event["a_remaining"] <= 16384 or event["b_remaining"] <= 16384:
                raise RuntimeError("NOT_COVERED: not a multi-round rescue")
            flushed_at = await self.flush(busy=True)
            self.poll()
            busy_while_parked = not any(e.get("event") == "resume"
                and e.get("rid") == a and e["t"] <= flushed_at
                for e in self.events[start_events:])
        else:
            busy_while_parked = False
        if abort_role:
            target = a if abort_role == "A" else b
            # Prefix-scoped abort is intentional: no global abort_all on a
            # shared service, even though this is an exclusive queue job.
            status, _ = await self.post("/abort_request", {"rid": target}, timeout=10)
            if status != 200:
                raise RuntimeError("abort not accepted: " + target)
        results = await asyncio.wait_for(asyncio.gather(self.tasks[a], self.tasks[b]), 65)
        await self.flush()  # Require true idle, including pending cache writes.
        self.poll()
        case_events = self.events[start_events:]
        for record in results:
            if record["role"] == abort_role:
                terminated = any(e.get("event") == "terminal" and e.get("rid") == record["rid"]
                    and e.get("reason") == "abort" for e in case_events)
                if not terminated or record["meta"].get("finish_reason", {}).get("type") != "abort":
                    raise RuntimeError("NOT_COVERED: expected paused/active abort was not observed")
            else:
                self.successful(record)
        if event and abort_role != "A":
            if not any(e.get("event") == "resume" and e.get("rid") == a for e in case_events):
                raise RuntimeError("NOT_COVERED: A never resumed")
            for record in results:
                if record["role"] != abort_role:
                    count = sum(e.get("event") == "first_token" and e.get("rid") == record["rid"]
                                for e in case_events)
                    if count != 1:
                        raise RuntimeError(f"expected one first-token record, got {count}")
        elif self.args.mode == "off" and any(e.get("event") == "yield" for e in case_events):
            raise RuntimeError("OFF arm unexpectedly parked a request")
        hit = results[0]["meta"].get("cached_tokens", 0)
        if seed_size and not 0 < hit < size // 2:
            raise RuntimeError("NOT_COVERED: seeded A had no partial prefix hit")
        self.coverage.append(dict(case=case, a=a, b=b, yield_seen=event is not None,
            busy_flush_while_parked=busy_while_parked, idle_flush=True, cached_tokens=hit,
            abort_role=abort_role, transitions=Counter(e["event"] for e in case_events)))
        print("124M_CASE", case, json.dumps(self.coverage[-1]), flush=True)

    async def run(self):
        # OFF repeats all paired completions. ON uses exactly the same A/B
        # inputs and admission trigger; only their service order can differ.
        for repeat in range(2 if self.args.mode == "off" else 1):
            for name, size, seed_size in CASES:
                await self.pair(name + f".r{repeat}", size, seed_size)
        if self.args.mode == "on":
            for role in ("A", "B"):
                await self.pair("abort_" + role, 196609, abort_role=role)
            if not any(c["busy_flush_while_parked"] for c in self.coverage):
                raise RuntimeError("NOT_COVERED: flush never reached a still-parked pair")
        # After abort paths, make the allocator serve the same long case again.
        await self.pair("reuse.r0", 196609)

    async def cleanup(self):
        completed = {r["rid"] for r in self.records}
        if set(self.tasks) - completed:
            try:
                # wait_for can cancel an HTTP task while its server request is
                # still alive. Cancel our whole RID prefix, not only live tasks.
                await self.post("/abort_request", {"rid": self.prefix}, timeout=3)
            except Exception:
                pass
        pending = [t for t in self.tasks.values() if not t.done()]
        if pending:
            _, unfinished = await asyncio.wait(pending, timeout=8)
            for task in unfinished:
                task.cancel()
        # Do not flush here: a failed probe must not erase evidence or pretend
        # to have verified drain. Each successful case has its own flush receipt.


def compare(off, on):
    if off["mode"] != "off" or on["mode"] != "on" or any(
            r["status"] != "STATE_PASS" for r in (off, on)):
        raise ValueError("need complete OFF and ON state receipts")
    by_key = [{(r["case"], r["role"]): r for r in arm["responses"]} for arm in (off, on)]
    if any(len(lookup) != len(arm["responses"]) for lookup, arm in zip(by_key, (off, on))):
        raise ValueError("duplicate response identity")
    rows = []
    for name, _, _ in CASES:
        for role in ("A", "B"):
            a, b, c = (by_key[0][(name + ".r0", role)], by_key[0][(name + ".r1", role)],
                       by_key[1][(name + ".r0", role)])
            if len({r["input_sha256"] for r in (a, b, c)}) != 1:
                raise ValueError("input mismatch")
            aps, bps, cps = [r["meta"]["output_token_logprobs"] for r in (a, b, c)]
            if any(len(p) != TOKENS for p in (aps, bps, cps)):
                raise ValueError("truncated output")
            if any(not math.isfinite(lp[0]) for ps in (aps, bps, cps) for lp in ps):
                raise ValueError("non-finite output logprob")
            # After a token divergence the next step has different conditioning;
            # never compare those logprobs as if they measured numerical error.
            comparable = 0
            for i in range(TOKENS):
                if len({p[i][1] for p in (aps, bps, cps)}) != 1:
                    break
                comparable += 1
            noise = max((abs(aps[i][0] - bps[i][0]) for i in range(comparable)), default=None)
            delta = max((max(abs(cps[i][0] - aps[i][0]), abs(cps[i][0] - bps[i][0]))
                         for i in range(comparable)), default=None)
            # No made-up numerical tolerance: report raw deltas. If ON falls
            # outside the two-OFF envelope, this needs independent review.
            within = comparable == TOKENS and all(
                min(aps[i][0], bps[i][0]) <= cps[i][0] <= max(aps[i][0], bps[i][0])
                for i in range(TOKENS))
            rows.append(dict(case=name, role=role, common_conditioning_steps=comparable,
                             off_off_max_logprob_delta=noise, on_off_max_logprob_delta=delta,
                             within_observed_off_envelope=within))
    return dict(status="WITHIN_OBSERVED_OFF_NOISE" if all(r["within_observed_off_envelope"]
                    for r in rows) else "NUMERICAL_REVIEW_REQUIRED", cases=rows,
                scope="output evidence only; not full KV/KDA tensor equality or an SLO verdict")


async def run(args):
    import aiohttp  # Already used by the Pod's harness; no runtime installation.

    args.out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    async with aiohttp.ClientSession() as http:
        probe = Probe(args, http)
        status, error = "STATE_PASS", None
        try:
            await asyncio.wait_for(probe.run(), timeout=240)
        except Exception as exc:
            status, error = "INVALID", f"{type(exc).__name__}: {exc}"
        finally:
            await probe.cleanup()
            probe.poll()
            if probe.errors:
                status, error = "INVALID", "server errors during diagnostic"
            probe.log.close()
            probe.event_out.close()
            probe.response_out.close()
        result = dict(mode=args.mode, status=status, error=error, started=started, ended=time.time(),
                      cold_budget_s=8, request_budget_s=240, coverage=probe.coverage,
                      events=probe.events, responses=probe.records, server_errors=probe.errors,
                      not_covered=["full tensor equality", "high-KV-pressure rollback",
                                   "forced service-lease expiry", "global abort_all", "SLO"])
        save(args.out / "receipt.json", result)
        print("124M_PROBE", status, error or "numerical pairing still required", flush=True)
        return 0 if status == "STATE_PASS" else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    r = sub.add_parser("run")
    r.add_argument("--mode", choices=("on", "off"), required=True)
    r.add_argument("--base-url", default="http://127.0.0.1:30000")
    r.add_argument("--server-log", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--off", type=Path, required=True)
    c.add_argument("--on", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.action == "run":
        return asyncio.run(run(args))
    result = compare(json.loads(args.off.read_text()), json.loads(args.on.read_text()))
    save(args.out, result)
    print(result["status"])
    return 0 if result["status"] == "WITHIN_OBSERVED_OFF_NOISE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
