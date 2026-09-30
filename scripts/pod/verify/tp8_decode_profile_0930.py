#!/usr/bin/env python3
"""Bounded, isolated warm-prefix TP8 decode profile on an existing engine.

Not a replay, numerical equivalence check, or SLO/performance verdict. Never
changes engine flags, graphs, or scoring. Prime one frozen rendered prompt with
one output token, launch 32 identical-prefix streams, wait for every first token,
then profile the next five forwards. Profiled timings include profiler overhead.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import collections
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
import threading
import time
import urllib.error
import urllib.request


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def post(base, endpoint, body=None, timeout=180):
    data = json.dumps(body, ensure_ascii=False).encode() if body is not None else b""
    req = urllib.request.Request(base.rstrip("/")+endpoint, data=data,
                                 headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw.decode("utf-8", "replace")


def flushed(base, timeout=30):
    deadline = time.monotonic()+timeout
    attempts = []
    while True:
        try:
            response = post(base,"/flush_cache",timeout=10)
            attempts.append(dict(at_s=time.time(),response=response))
            if isinstance(response,dict) and response.get("success") is True:
                return dict(success=True,attempts=attempts)
        except Exception as error:
            attempts.append(dict(at_s=time.time(),error=repr(error)))
        if time.monotonic() >= deadline:
            return dict(success=False,attempts=attempts)
        time.sleep(.5)


def stop_profile(base):
    try:
        return dict(stopped=True,response=post(base,"/stop_profile",timeout=180))
    except urllib.error.HTTPError as error:
        detail = error.read(4096).decode("utf-8","replace")
        # num_steps automatically stops the profiler; the API then raises 500.
        if "Profiling is not in progress" in detail:
            return dict(stopped=False,already_automatically_stopped=True,status=error.code,response=detail)
        return dict(stopped=False,error=repr(error),status=error.code,response=detail)
    except Exception as error:
        return dict(stopped=False,error=repr(error))


def ram_output(path):
    logical_root=Path("/tmp/ax/codex").resolve()
    out=path.resolve()
    if logical_root not in out.parents or out.exists():
        raise ValueError("live probe requires a new directory below /tmp/ax/codex")
    existing=out.parent
    while not existing.exists():
        existing=existing.parent
    mounts=[]
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left,right=line.split(" - ",1)
        mount=Path(left.split()[4].replace("\\040"," "))
        if mount==existing or mount in existing.parents:
            mounts.append((len(mount.parts),right.split()[0],str(mount)))
    _,filesystem,mount=max(mounts)
    if filesystem!="tmpfs":
        raise ValueError(f"profile output is not RAM-backed: {filesystem} {mount}")
    out.mkdir(parents=True)
    return out,dict(logical_root=str(logical_root),resolved_output=str(out),filesystem=filesystem,mount=mount)


def prompt_from_frozen(root, rid, tok_dir, harness):
    sys.path.insert(0,str(harness))
    from s1_common import Renderer
    rows=[json.loads(line) for line in (root/"requests.jsonl").open() if line.strip()]
    def request_id(row):
        return f"{row['pack']}:{row['view']}:{row['logical_call_id']}"
    candidates=[r for r in rows if request_id(r)==rid] if rid else [r for r in rows if 6144 <= r["glm_tokens"] <= 10240]
    if not candidates:
        raise ValueError("no frozen request in selected 6–10k prompt range; supply --rid")
    row=min(candidates,key=lambda r:(abs(r["glm_tokens"]-8192),request_id(r)))
    rid=request_id(row)
    body=None
    with gzip.open(root/row["body_ref"],"rt") as handle:
        for line in handle:
            candidate=json.loads(line)
            if candidate["req_id"]==rid:
                body=candidate;break
    if body is None:
        raise ValueError(f"missing frozen body {rid}")
    renderer=Renderer(str(tok_dir));prompt=renderer.render(body);tokens=renderer.n_tokens(prompt)
    if not 6144 <= tokens <= 10240:
        raise ValueError(f"rendered prompt is outside 6–10k: {tokens}")
    return prompt,dict(req_id=rid,frozen_glm_tokens=row["glm_tokens"],rendered_tokens=tokens,
                       requests_sha256=hashlib.sha256((root/"requests.jsonl").read_bytes()).hexdigest(),
                       prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),body_ref=row["body_ref"])


class Streams:
    def __init__(self,base,prompt,count,outputs,tag,timeout):
        self.base,self.prompt,self.count,self.outputs,self.tag,self.timeout=base,prompt,count,outputs,tag,timeout
        self.lock=threading.Lock();self.first_ids=set();self.done_ids=set();self.results={}
        self.all_first=threading.Event();self.release=threading.Barrier(count)

    def request(self,index):
        rid=f"decode-profile-{self.tag}-{index}"
        result=dict(index=index,rid=rid,requested_output_tokens=self.outputs,events=0,meta_events=[])
        try:
            self.release.wait(timeout=30)
            result["client_dispatch_at_s"]=time.time()
            body=dict(text=self.prompt,rid=rid,stream=True,sampling_params=dict(max_new_tokens=self.outputs,temperature=0,ignore_eos=True))
            req=urllib.request.Request(self.base.rstrip("/")+"/generate",data=json.dumps(body).encode(),
                                       headers={"Content-Type":"application/json"},method="POST")
            done=False
            with urllib.request.urlopen(req,timeout=self.timeout) as response:
                for line in response:
                    if not line.startswith(b"data:"):
                        continue
                    payload=line[5:].strip()
                    if payload==b"[DONE]":
                        done=True;break
                    event=json.loads(payload);now=time.time();meta=event.get("meta_info") or {}
                    result["events"]+=1
                    result["meta_events"].append(dict(client_at_s=now,meta_info=meta))
                    emitted=int(meta.get("completion_tokens") or 0)
                    if emitted>0:
                        result["last_token_at_s"]=now
                        if "first_token_at_s" not in result:
                            result["first_token_at_s"]=now
                            with self.lock:
                                self.first_ids.add(index)
                                if len(self.first_ids)==self.count:
                                    self.all_first.set()
            result["sse_done"]=done
            result["final_meta"]=result["meta_events"][-1]["meta_info"] if result["meta_events"] else {}
            tokens=result["final_meta"].get("completion_tokens")
            if not done or tokens!=self.outputs:
                raise ValueError(f"incomplete output: done={done} completion_tokens={tokens}")
            result["tpot_s"]=(result["last_token_at_s"]-result["first_token_at_s"])/(tokens-1)
        except Exception as error:
            result["error"]=repr(error)
        finally:
            result["client_finish_at_s"]=time.time()
            with self.lock:
                self.done_ids.add(index);self.results[index]=result
        return result


def analyze_trace(path):
    import prof_ledger
    ledger=prof_ledger.analyze(str(path))
    with gzip.open(path,"rt") as handle:
        events=json.load(handle)["traceEvents"]
    kernels=collections.defaultdict(lambda:dict(calls=0,gpu_time_us=0.0))
    host=collections.defaultdict(lambda:dict(calls=0,cpu_time_us=0.0))
    transfers=collections.defaultdict(lambda:dict(calls=0,gpu_time_us=0.0))
    decode=[];cpu=[]
    for event in events:
        if event.get("ph")!="X" or "dur" not in event:
            continue
        name,category=event.get("name",""),event.get("cat","")
        if category=="kernel":
            kernels[name]["calls"]+=1;kernels[name]["gpu_time_us"]+=float(event["dur"])
        if category in ("cpu_op","cuda_runtime","cuda_driver"):
            host[(category,name)]["calls"]+=1;host[(category,name)]["cpu_time_us"]+=float(event["dur"])
        if category in ("gpu_memcpy","gpu_memset"):
            transfers[(category,name)]["calls"]+=1;transfers[(category,name)]["gpu_time_us"]+=float(event["dur"])
        if name.startswith("step[DECODE ") and category in ("gpu_user_annotation","user_annotation"):
            item=dict(name=name,ts_us=event["ts"],dur_us=event["dur"],args=event.get("args",{}))
            (decode if category=="gpu_user_annotation" else cpu).append(item)
    def start_intervals(spans):
        starts=sorted({event["ts_us"] for event in spans})
        return [(right-left)/1000 for left,right in zip(starts,starts[1:])]
    return dict(trace=path.name,ledger=ledger,gpu_decode_annotations=decode,cpu_decode_spans=cpu,
                cpu_step_start_intervals_ms=start_intervals(cpu),
                kernels=[dict(name=name,category=prof_ledger.cat_of(name),**item)
                         for name,item in sorted(kernels.items(),key=lambda pair:-pair[1]["gpu_time_us"])],
                host_operations=[dict(category=key[0],name=key[1],**item) for key,item in sorted(host.items(),key=lambda pair:-pair[1]["cpu_time_us"])],
                transfers=[dict(category=key[0],name=key[1],**item) for key,item in sorted(transfers.items(),key=lambda pair:-pair[1]["gpu_time_us"])],
                kernel_duration_note="Summed GPU kernel duration, not critical-path time; overlapping streams must use ledger busy union.",
                host_duration_note="CPU operations nest; summed durations double-count nested spans. Start intervals describe profiled annotated forwards, including gaps and export stalls, not unprofiled TPOT.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--tok-dir",type=Path,default=Path("/mnt/models"))
    parser.add_argument("--harness-dir",type=Path,default=Path("/tmp/ax/s1/s1-dev/harness"))
    parser.add_argument("--rid")
    parser.add_argument("--base-url",default="http://127.0.0.1:30000")
    parser.add_argument("--out-dir",type=Path,required=True)
    parser.add_argument("--server-log",type=Path,required=True)
    parser.add_argument("--tag",required=True)
    parser.add_argument("--batch",type=int,default=32,choices=[32])
    parser.add_argument("--outputs",type=int,default=128,choices=[64,128])
    parser.add_argument("--steps",type=int,default=5,choices=range(5,11))
    parser.add_argument("--request-timeout",type=int,default=180)
    parser.add_argument("--trace-budget-mib",type=int,default=200)
    parser.add_argument("--expected-requests-sha256",default="6170fd8204db4de8be4f99861d8cfa20ba36eab8b3317b8a3cd1343c5aef8ebb")
    args=parser.parse_args()
    out,storage=ram_output(args.out_dir);trace_dir=out/"traces";trace_dir.mkdir()
    receipt=dict(scope="isolated warm-prefix decode profile; not a replay or SLO/performance verdict",status="RUNNING",storage=storage,
                 started_at_s=time.time(),batch=args.batch,outputs=args.outputs,steps=args.steps,activities=["CPU","GPU"],
                 limitations=["Identical warm prefixes omit prefill interference, cache pressure, heterogeneous histories, and real gaps.",
                              "Host/network submission may stagger requests. Trace and server log batch evidence is required.",
                              "CPU/GPU profiling and trace export perturb timing. Gaps do not prove CPU starvation.",
                              "Kernel names classify heuristically; ambiguous GEMMs need launch/callsite correlation."])
    write(out/"receipt.json",receipt)
    profile_armed=False;controller=None;pending=[];executor=None
    log_offset=args.server_log.stat().st_size
    try:
        prompt,identity=prompt_from_frozen(args.root,args.rid,args.tok_dir,args.harness_dir);receipt["prompt"]=identity
        if identity["requests_sha256"]!=args.expected_requests_sha256:
            raise RuntimeError("frozen request index SHA mismatch")
        receipt["flush_before"]=flushed(args.base_url)
        if not receipt["flush_before"]["success"]:
            raise RuntimeError("pre-probe flush failed")
        prime=post(args.base_url,"/generate",dict(text=prompt,stream=False,rid=f"decode-prime-{args.tag}",
                   sampling_params=dict(max_new_tokens=1,temperature=0,ignore_eos=True)))
        receipt["prime_response"]=prime
        if prime.get("meta_info",{}).get("completion_tokens")!=1:
            raise RuntimeError("primer did not produce exactly one token")
        controller=Streams(args.base_url,prompt,args.batch,args.outputs,args.tag,args.request_timeout)
        executor=futures.ThreadPoolExecutor(args.batch)
        pending=[executor.submit(controller.request,i) for i in range(args.batch)]
        if not controller.all_first.wait(timeout=45):
            raise RuntimeError("not every stream emitted first token before profiling deadline")
        with controller.lock:
            receipt["at_profile_arm"]=dict(first_token_requests=len(controller.first_ids),finished_requests=len(controller.done_ids),at_s=time.time())
            if controller.done_ids:
                raise RuntimeError("a stream finished before profiler was armed")
        request=dict(output_dir=str(trace_dir),profile_id=args.tag,profile_prefix=args.tag,activities=["CPU","GPU"],
                     with_stack=False,record_shapes=False,merge_profiles=False,profile_by_stage=False,start_step=1,num_steps=args.steps)
        receipt["profile_request"]=request
        profile_armed=True
        receipt["start_response"]=post(args.base_url,"/start_profile",request,timeout=30)
        # Profiles stop automatically after the bounded window; continue draining
        # every stream even though export may temporarily stall the scheduler.
        for pending_request in pending:
            pending_request.result()
    except Exception as error:
        receipt["error"]=repr(error)
    finally:
        if profile_armed:
            receipt["stop_response"]=stop_profile(args.base_url)
        if executor is not None:
            executor.shutdown(wait=True,cancel_futures=False)
        receipt["drained_at_s"]=time.time()
        if controller is not None:
            results=[controller.results[i] for i in sorted(controller.results)]
            write(out/"requests.json",results)
            receipt["request_count"]=len(results);receipt["request_errors"]=sum("error" in row for row in results)
            receipt["cached_tokens"]=dict(collections.Counter(str(row.get("final_meta",{}).get("cached_tokens")) for row in results))
        receipt["flush_after"]=flushed(args.base_url,timeout=60)
        if not receipt["flush_after"]["success"]:
            receipt.setdefault("error","post-probe flush failed; inspect server before next job")
        if args.server_log.exists():
            with args.server_log.open("rb") as handle:
                handle.seek(log_offset);log=handle.read(8*1024**2+1)
            receipt["server_log_tail_complete"]=len(log)<=8*1024**2
            if receipt["server_log_tail_complete"]:
                (out/"server-probe.log").write_bytes(log)
                receipt["logged_decode_batches"]=collections.Counter(map(int,re.findall(rb"Decode batch[^\n]*#running-req: (\d+)",log)))
                stop=receipt.get("stop_response",{})
                if stop.get("status")==500 and b"RuntimeError: Profiling is not in progress" in log:
                    stop["already_automatically_stopped"]=True
                    stop["verified_from_probe_server_log"]=True
            else:
                receipt.setdefault("error","probe server log exceeded8MiB; original log retained")
        receipt["finished_at_s"]=time.time()
        write(out/"receipt.json",receipt)
    traces=sorted(trace_dir.glob("*.trace.json.gz"))
    receipt["traces"]=[dict(name=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in traces]
    receipt["trace_bytes"]=sum(p.stat().st_size for p in traces)
    if receipt["trace_bytes"]>args.trace_budget_mib*1024**2:
        receipt.setdefault("error","trace budget exceeded; evidence retained")
    ranks={int(match.group(1)) for p in traces if (match:=re.search(r"TP-(\d+)(?:-|\.)",p.name))}
    receipt["ranks"]=sorted(ranks)
    analyses=[]
    if ranks!=set(range(8)):
        receipt.setdefault("error",f"missing TP8 traces: {sorted(ranks)}")
    for path in traces:
        try:
            analysis=analyze_trace(path);analyses.append(analysis)
            batches={int(match.group(1)) for event in analysis["gpu_decode_annotations"] if (match:=re.search(r"bs=(\d+)",event["name"]))}
            analysis["trace_decode_batch_sizes"]=sorted(batches)
            decode_count=analysis["ledger"].get("classes",{}).get("decode",{}).get("n",0)
            if "error" in analysis["ledger"] or batches!={args.batch} or decode_count!=args.steps:
                receipt.setdefault("error",f"trace does not establish bounded decode batch32: {path.name}")
        except Exception as error:
            receipt.setdefault("error",f"trace analysis failed {path.name}: {error!r}")
    write(out/"profile-analysis.json",analyses)
    stop=receipt.get("stop_response",{})
    if profile_armed and not (stop.get("stopped") or stop.get("already_automatically_stopped")):
        receipt.setdefault("error","profiler stop was not verified")
    receipt["status"]="INVALID" if receipt.get("error") or receipt.get("request_errors") else "PROFILED_DIAGNOSTIC"
    write(out/"receipt.json",receipt)
    print(json.dumps({k:receipt.get(k) for k in ("status","error","batch","steps","ranks","trace_bytes","request_count","request_errors")}))
    return 0 if receipt["status"]=="PROFILED_DIAGNOSTIC" else 2


if __name__=="__main__":
    sys.dont_write_bytecode=True
    sys.exit(main())
