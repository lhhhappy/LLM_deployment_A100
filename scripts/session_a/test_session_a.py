"""SA-01..07: CPU-only controller/remote fault and protocol tests; never Trisol."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock
import base64

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
import remote
import runner


def row(n, passed=True, mean=.03):
    return {"N": n, "passed": passed, "status": "completed", "dev_gates": {str(i): {} for i in range(10)},
            "tpot": {"passed": passed, "tpot_mean": mean}}


class ProtocolTests(unittest.TestCase):
    def test_decorated_cli_frame(self):
        self.assertEqual(common.framed('Info: websocket connected\nSESSION_A_JSON:{"ok":true}\n'), {"ok": True})

    def test_missing_duplicate_or_nonobject_frame_rejected(self):
        for output in ('{}', 'SESSION_A_JSON:[]', 'SESSION_A_JSON:{}\nSESSION_A_JSON:{}', 'SESSION_A_JSON:bad'):
            with self.subTest(output=output), self.assertRaises(ValueError):
                common.framed(output)

    def test_incomplete_is_not_slo_fail(self):
        for bad in ({"levels": []}, {"levels": [row(6)]}, {"levels": [dict(row(10), status="aborted")]},
                    {"levels": [dict(row(10), passed=1)]}, {"levels": [dict(row(10), dev_gates={})]}):
            with self.assertRaises(ValueError):
                common.level_result(bad, 10)
        self.assertFalse(common.level_result({"levels": [row(10, False)]}, 10)["passed"])

    def test_metrics_missing_is_not_zero(self):
        self.assertFalse(common.metrics_summary('unrelated 0')["forward"]["available"])
        r = common.metrics_summary('sglang:queue_time_seconds_sum{model="x"} 3\nsglang:forward_execution_seconds_total 4\nsglang:mamba_available_tokens 100')
        self.assertTrue(all(v["available"] for v in r.values()))

    def test_patch_command_and_d1_switches(self):
        candidate = json.loads((runner.REPO / "submission/candidate-01.json").read_text())
        for name in common.POLICIES:
            command, env, patches = common.candidate_command(candidate, name, 30000)
            self.assertEqual(command[command.index("--port")+1], "30000")
            self.assertIn("--enable-metrics", command)
            self.assertEqual(common.ROLE_ENV in env, common.POLICIES[name][2] or common.ROLE_ENV in common.EXTRA_ENV.get(name, {}))
            self.assertEqual(command[command.index("--schedule-policy")+1], common.POLICIES[name][1])
            if name == "baseline":
                self.assertEqual(patches, ["000"])
            if name in ("spf", "spf_d1"):
                self.assertEqual(patches, ["000", "001", "002"])
        self.assertIn(common.ROLE_ENV, candidate["env"])  # original unchanged


class PlanningTests(unittest.TestCase):
    def trace(self, outcomes):
        observed = {}
        visited = []
        for outcome in outcomes:
            n, reason = common.next_baseline(observed)
            self.assertIsNone(reason)
            visited.append(n)
            observed[n] = outcome
        return visited, common.next_baseline(observed)

    def test_calibration_always_before_holdout_then_climb(self):
        self.assertEqual(self.trace([True, True, True, False]), ([6, 10, 14, 18], (None, "adjacent_bracket")))

    def test_descent_reuses_six_and_measures_two(self):
        self.assertEqual(self.trace([True, False]), ([6, 10], (None, "adjacent_bracket")))
        self.assertEqual(self.trace([False, False, True]), ([6, 10, 2], (None, "adjacent_bracket")))
        self.assertEqual(self.trace([False, False, False]), ([6, 10, 2], (None, "no_passing_rung")))

    def test_nonmonotone_and_upper_bound_are_not_capacity(self):
        self.assertEqual(self.trace([False, True])[1], (None, "non_monotone"))
        self.assertEqual(common.next_baseline({6: True, 10: True, 14: True}, 14), (None, "max_n_bound"))

    def test_monotonic_budget_final_reserve(self):
        clock = mock.Mock(return_value=10.)
        budget = common.Budget(100, 20, clock)
        self.assertEqual(budget.limit(200), 80)
        self.assertTrue(budget.fits(60, 20))
        self.assertFalse(budget.fits(61, 20))
        clock.return_value = 90
        with self.assertRaises(TimeoutError):
            budget.limit(1)
        self.assertEqual(budget.limit(100, final=True), 20)
        clock.return_value = 120
        with self.assertRaises(TimeoutError):
            budget.limit(1, final=True)

    def test_invalid_budgets_and_cli_fail_before_network(self):
        for seconds, reserve in ((20, 20), (20, -1), (float("inf"), 20), (float("nan"), 1)):
            with self.assertRaises(ValueError):
                common.Budget(seconds, reserve)
        with mock.patch.object(subprocess, "run", side_effect=AssertionError("unexpected subprocess")):
            for args in (["--minutes", "nan"], ["--port", "8000"], ["--run-id", "../escape"], ["--max-n", "11"]):
                with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                    runner.main(args)

    def test_planner_real_cohort_and_matrix_pair_cost(self):
        ctl = runner.Controller(runner.parser().parse_args(["--minutes", "480"]))
        self.assertAlmostEqual(ctl.plan["ladder_levels"][1]["measure_minutes"], 35)
        self.assertGreater(ctl.config_cost(10), ctl.level_seconds[10] + ctl.level_seconds[14])


class TransferTests(unittest.TestCase):
    def test_upload_batches_stay_below_per_argument_limit_and_roundtrip(self):
        data = b"a" * (48 * 1024 * 9 + 31)
        batches = list(runner.upload_batches(io.BytesIO(data), 8))
        self.assertEqual([len(b)//3 for b in batches], [8, 2])
        self.assertLessEqual(max(len(arg) for batch in batches for arg in batch), 65536)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/"upload").mkdir()
            for batch in batches:
                result = subprocess.run([sys.executable, "-B", "-c", runner.UPLOAD_CHUNKS, tmp, *batch], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(common.framed(result.stdout)["verified_chunks"], len(batch)//3)
            self.assertEqual(b"".join(p.read_bytes() for p in sorted((Path(tmp)/"upload").glob("*.part"))), data)

    def archive(self, name, kind=None):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            entry = tarfile.TarInfo(name)
            if kind:
                entry.type = kind
                entry.linkname = "/tmp/escape"
            else:
                entry.size = 3
            tar.addfile(entry, None if kind else io.BytesIO(b"abc"))
        buf.seek(0)
        return tarfile.open(fileobj=buf)

    def test_valid_archive_and_traversal_links_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.archive("nested/file") as archive:
                common.safe_extract(archive, tmp)
            self.assertEqual((Path(tmp)/"nested/file").read_bytes(), b"abc")
            for name, kind in (("../escape", None), ("/tmp/escape", None), ("link", tarfile.SYMTYPE), ("link", tarfile.LNKTYPE)):
                with self.archive(name, kind) as archive, self.assertRaises(ValueError):
                    common.safe_extract(archive, tmp)

    def test_bundle_manifest_and_bootstrap_detect_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "input.txt"
            src.write_text("data")
            archive = root / "payload.tar.gz"
            manifest = runner.bundle(archive, {"s1-dev/input.txt": src}, {"aime": {}})
            pod = root / "pod"
            (pod / "upload").mkdir(parents=True)
            (pod / "upload/000000.part").write_bytes(archive.read_bytes())
            result = subprocess.run([sys.executable, "-B", "-c", runner.BOOTSTRAP, str(pod), common.digest(archive)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(common.framed(result.stdout)["verified_files"], len(manifest))
            self.assertEqual((pod / "s1-dev/input.txt").read_text(), "data")
            with (pod / "upload/000000.part").open("ab") as f:
                f.write(b"corrupt")
            result = subprocess.run([sys.executable, "-B", "-c", runner.BOOTSTRAP, str(pod), common.digest(archive)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)

    def test_inputs_include_dependencies_and_reference_paths(self):
        files = runner.inputs()
        for name in ("scripts/serving_probe.py", "scripts/preflight_8gpu.py", "scripts/if_checks.py",
                     "scripts/session_a/remote.py", "submission/candidate-01.json", "reference/srt/runtime_context.py"):
            self.assertIn(name, files)
        self.assertFalse(any("__pycache__" in f for f in files))

    def test_export_changed_files_and_snapshot_hash(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(remote, "ART", Path(tmp)/"art"), mock.patch.object(remote, "WORK", Path(tmp)/"work"):
            remote.ART.mkdir(); remote.WORK.mkdir()
            (remote.ART / "file").write_text("first")
            initial = remote.export({})
            self.assertIn("file", initial["manifest"])
            self.assertEqual(initial["sha256"], common.digest(remote.WORK/"export.tar.gz"))
            unchanged = remote.export({"known": initial["manifest"]})
            self.assertEqual(unchanged["manifest"], {})
            (remote.ART/"file").write_text("changed")
            changed = remote.export({"known": initial["manifest"]})
            self.assertNotEqual(changed["manifest"]["file"], initial["manifest"]["file"])


class RemoteTests(unittest.TestCase):
    def test_cap_small_single_server_report_is_not_registry_pass(self):
        import cap_spot_check
        import serving_probe
        report = {"rows": [{"id": "q", "results": {"stock": {"interface_ok": True}}}],
                  "scores": {"stock": 1}, "source": {"sha256": "fixture"}, "passed": False}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(remote, "ROOT", Path(tmp)), mock.patch.object(remote, "ART", Path(tmp)/"art"):
            common.write_json(Path(tmp)/"cap_questions.json", {s: {"questions": [], "source": report["source"]} for s in ("aime", "gpqa")})
            with mock.patch.object(cap_spot_check, "compare", side_effect=lambda *a: json.loads(json.dumps(report))), mock.patch.object(serving_probe, "client_for"):
                result = remote.cap({"config": "stock", "id": "cap-job", "port": 30000})
            self.assertFalse(result["registry_sample_complete"])
            self.assertFalse(json.loads((remote.ART/"stock/cap_aime.json").read_text())["passed"])
            self.assertTrue((remote.ART/"stock/cap_history/cap-job/aime.json").exists())

    def test_real_patches_stack_in_inspection_copy_only(self):
        """Actual patch -p3 --fuzz=0 against the local reference in temporary copies."""
        import shutil
        import types
        import urllib.error
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "installed/sglang"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text('__version__="fixture"\n')
            patch_dir = root / "patches"
            patch_dir.mkdir()
            for prefix in ("000", "001", "002"):
                patches = list((runner.REPO / "patches").glob(prefix + "-*.patch"))
                if not patches:
                    continue
                patch = patches[0]
                shutil.copy2(patch, patch_dir / patch.name)
                for line in patch.read_text().splitlines():
                    if line.startswith("+++ b/python/sglang/"):
                        name = line.split()[1].removeprefix("b/python/sglang/")
                        source = runner.REPO / "src/sglang/python/sglang" / name
                        if source.exists():
                            target = package / name
                            target.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(source, target)
            before = {str(p.relative_to(package)): common.digest(p) for p in package.rglob("*") if p.is_file()}
            (root / "reference").mkdir()
            common.write_json(root/"reference-index.json", {})
            original_run = remote.run
            def probe(cmd, **kwargs):
                if cmd[0] == "patch":
                    return original_run(cmd, **kwargs)
                return {"command": cmd, "returncode": 0, "output": "CPU fixture; not hardware evidence"}
            with mock.patch.object(remote, "ROOT", root), mock.patch.object(remote, "ART", root/"artifacts"), mock.patch.object(remote, "WORK", root/"work"), mock.patch.object(remote.importlib.util, "find_spec", return_value=types.SimpleNamespace(origin=str(package/"__init__.py"))), mock.patch.object(remote, "run", side_effect=probe), mock.patch.object(remote.urllib.request, "urlopen", side_effect=urllib.error.URLError("offline")):
                result = remote.inspect({})
            self.assertEqual(result["patches"]["000"]["status"], "ok")
            # v0.5.20 line (001/002/004) is archived under patches/v0520 (decision 29): reported absent.
            self.assertEqual(result["patches"]["001"]["status"], "skip")
            self.assertIn(result["patches"]["002"]["status"], ("ok", "skip"))
            self.assertTrue((root/"artifacts/inspection/report.txt").is_file())
            self.assertFalse(result["internet"]["pypi"]["reachable"])
            self.assertEqual(before, {str(p.relative_to(package)): common.digest(p) for p in package.rglob("*") if p.is_file()})

    def test_remote_timeout_stops_owned_engine_and_marks_timeout(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(remote, "ART", Path(tmp)/"art"):
            (remote.ART/"jobs").mkdir(parents=True)
            spec = {"id": "job", "action": "measure", "timeout": .01, "deadline": 9999999999}
            common.write_json(Path(tmp)/"job.json", spec)
            child = mock.Mock(pid=54321)
            child.wait.side_effect = [subprocess.TimeoutExpired("fixture", .01), 0]
            with mock.patch.object(remote.subprocess, "Popen", return_value=child), mock.patch.object(remote, "kill_tree") as kill, mock.patch.object(remote, "stop_engine") as stop:
                remote.supervise(str(Path(tmp)/"job.json"))
            kill.assert_called_once_with(54321)
            stop.assert_called_once()
            self.assertEqual(json.loads((remote.ART/"jobs/job.status.json").read_text())["status"], "timeout")

    def test_stop_refuses_reused_pid(self):
        with mock.patch.object(remote, "process_info", return_value={"start": "new"}), mock.patch.object(remote.os, "killpg") as kill:
            with self.assertRaises(RuntimeError):
                remote.stop_engine({"pid": 1234, "start": "old"})
            kill.assert_not_called()


class FakeController(runner.Controller):
    def __init__(self, tmp, *, fail=None, spf=True, minutes=2000, argv=()):
        super().__init__(runner.parser().parse_args(["--minutes", str(minutes), *argv]))
        self.out = Path(tmp)/"run"
        self.calls = []
        self.fail = fail
        self.spf = spf

    def event(self, message):
        self.calls.append(("event", message))

    def transfer(self):
        self.uploaded = True
        self.calls.append(("transfer",))

    def job(self, action, config="baseline", n=None):
        self.calls.append((action, config, n))
        if self.fail == action:
            raise RuntimeError("injected " + action)
        if action == "inspect":
            return {"baseline_ready": True, "patches": {p: {"status": "ok" if p != "002" or self.spf else "skip"} for p in ("000", "001", "002", "004")}}
        if action == "measure":
            return row(n, n < 18 or config == "spf_d1", .025 if config == "spf_d1" else .03)
        return {}

    def sync(self, final=False):
        self.calls.append(("sync", final))
        if self.fail == "sync":
            raise RuntimeError("injected sync")

    def gpu_sync(self, final=False):
        self.calls.append(("gpu_sync", final))

    def remote(self, operation, *args, final=False):
        self.calls.append(("remote", operation, final))
        return {}


class MockPodController(runner.Controller):
    """Real controller/job/sync/manifest logic with an entirely local fake pod."""
    def __init__(self, args, pod_root):
        super().__init__(args)
        self.pod = Path(pod_root) / self.stamp
        self.pod_art = self.pod / "artifacts"
        self.pod_work = self.pod / "work"
        self.calls = []

    def event(self, message):
        self.calls.append(("event", message))

    def transfer(self):
        self.pod_art.mkdir(parents=True)
        self.pod_work.mkdir()
        for profile in self.profiles:
            target = self.pod / profile["remote"]
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(Path(profile["path"]).read_bytes())
        self.uploaded = True

    def gpu_sync(self, final=False):
        self.calls.append(("gpu_sync", final))

    def remote(self, operation, *args, final=False):
        self.calls.append((operation, *args))
        if operation == "launch":
            spec = json.loads(args[0])
            action, config = spec["action"], spec["config"]
            result = {}
            if action == "inspect":
                result = {"baseline_ready": True, "patches": {p: {"status": "ok"} for p in ("000", "001", "002", "004")}}
            elif action == "start":
                profile = self.pod / spec["profile"]
                assert common.digest(profile) == spec["profile_sha256"]
                command, env, patches = common.candidate_command(json.loads(profile.read_text()), config, spec["port"])
                common.write_json(self.pod_art / config / "engine-command.json", {
                    "argv": command, "env_overrides": env, "patches": patches,
                    "profile_sha256": spec["profile_sha256"]})
            elif action == "measure":
                n = spec["n"]
                folder = self.pod_art / config / f"N{n}"
                folder.mkdir(parents=True)
                (folder / "raw.jsonl").write_text('{}\n')
                common.write_json(folder / "run.json", {"config": {"N": n}})
                result = row(n, n < 18)
                result["paths"] = {key: f"{self.remote_root}/artifacts/{config}/N{n}/{name}"
                                   for key, name in (("raw", "raw.jsonl"), ("run", "run.json"))}
                common.write_json(folder / "result.json", result)
            common.write_json(self.pod_art / "jobs" / (spec["id"] + ".status.json"), {"status": "done", "result": result})
            return {"launched": spec["id"]}
        if operation == "status":
            return json.loads((self.pod_art / "jobs" / (args[0] + ".status.json")).read_text())
        if operation == "export":
            with mock.patch.object(remote, "ART", self.pod_art), mock.patch.object(remote, "WORK", self.pod_work):
                return remote.export(json.loads(args[0]))
        if operation == "read-export":
            data = (self.pod_work / "export.tar.gz").read_bytes()
            return {"data": base64.b64encode(data[int(args[0]):int(args[0])+int(args[1])]).decode()}
        if operation == "stop":
            return {"stopped": True}
        raise AssertionError(operation)


class ControllerTests(unittest.TestCase):
    def test_full_batch_order_pair_sync_and_best_cap(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            ctl = FakeController(tmp)
            ctl.execute()
        measured = [c for c in ctl.calls if c[0] == "measure"]
        self.assertEqual(measured[:4], [("measure", "baseline", n) for n in (6, 10, 14, 18)])
        self.assertEqual(measured[4:], [("measure", cfg, n) for cfg in ("spf", "spf_d1", "d1") for n in (14, 18)])
        self.assertLess(ctl.calls.index(("preflight", "baseline", None)), ctl.calls.index(measured[0]))
        self.assertLess(ctl.calls.index(("cap", "baseline", None)), ctl.calls.index(measured[0]))
        for measured_call in measured:
            i = ctl.calls.index(measured_call)
            self.assertIn(("sync", False), ctl.calls[i+1:i+3])
        self.assertEqual([c for c in ctl.calls if c[0] == "cap"][-1], ("cap", "spf_d1", None))
        self.assertEqual(ctl.summary["status"], "completed")
        self.assertIn(("remote", "stop", True), ctl.calls)

    def test_preflight_failure_never_reaches_ladder_and_finally_stops(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            ctl = FakeController(tmp, fail="preflight")
            ctl.execute()
        self.assertEqual(ctl.summary["status"], "aborted")
        self.assertFalse(any(c[0] == "measure" for c in ctl.calls))
        self.assertIn(("remote", "stop", True), ctl.calls)

    def test_sync_failure_does_not_skip_stop_or_second_pull(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            ctl = FakeController(tmp, fail="sync")
            ctl.execute()
        self.assertIn(("remote", "stop", True), ctl.calls)
        self.assertEqual(ctl.calls.count(("sync", True)), 2)
        self.assertEqual(ctl.summary["status"], "aborted")

    def test_missing_002_skips_spf_but_finishes_d1(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            ctl = FakeController(tmp, spf=False)
            ctl.execute()
        configs = {c[1] for c in ctl.calls if c[0] == "measure"}
        self.assertEqual(configs, {"baseline", "d1"})
        self.assertEqual(ctl.summary["status"], "completed")
        self.assertTrue(any("patch 002" in s for s in ctl.summary["skips"]))

    def test_tight_budget_keeps_n6_n10_and_reports_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            ctl = FakeController(tmp, minutes=100)
            ctl.execute()
        self.assertEqual([c[2] for c in ctl.calls if c[0] == "measure"], [6, 10])
        self.assertEqual(ctl.summary["status"], "incomplete")
        self.assertIsNone(ctl.summary["baseline"]["critical_p"])

    def test_dry_run_has_no_subprocess_network_or_writes(self):
        output = io.StringIO()
        with mock.patch.object(subprocess, "run", side_effect=AssertionError("subprocess")), mock.patch.object(subprocess, "Popen", side_effect=AssertionError("subprocess")), mock.patch.object(Path, "write_text", side_effect=AssertionError("write")), mock.patch.object(Path, "mkdir", side_effect=AssertionError("mkdir")), mock.patch.object(remote.urllib.request, "urlopen", side_effect=AssertionError("network")), redirect_stdout(output):
            self.assertEqual(runner.main(["--dry-run", "--run-id", "dry-test"]), 0)
        text = output.getvalue()
        self.assertIn('"n": 6', text)
        self.assertIn('"n": 10', text)
        self.assertIn("SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS", text)
        self.assertIn("--team arena --", text)
        self.assertIn("FINAL", text)
        self.assertNotIn("exec -i", text)


class DaemonContractTests(unittest.TestCase):
    def test_id_and_name_are_forwarded_to_exec_unchanged(self):
        for option, service in (("--service-id", "2102309548588015616"), ("--service", "lh-t1")):
            ctl = runner.Controller(runner.parser().parse_args([option, service, "--budget-minutes", "40"]))
            with mock.patch.object(runner.subprocess, "run", return_value=mock.Mock(returncode=0, stdout='SESSION_A_JSON:{}')) as run:
                ctl.bohr(["python3", "-c", "fixture"])
            self.assertEqual(run.call_args.args[0][4], service)
            self.assertIn("--team", run.call_args.args[0])

    def test_image_line_mapping(self):
        # T33: organizer-base line; B profile only for the D1-on config, A for the rest.
        paths = [runner.REPO / "submission/candidate-bA-0922e.json", runner.REPO / "submission/candidate-bB-0922f.json"]
        configs = ["img_a", "img_b", "img_b_off", "img_a_fcfs"]
        _, selected = common.select_profiles(paths, configs)
        self.assertEqual({c: Path(selected[c]["path"]).name[10:12] for c in configs},
                         {"img_a": "bA", "img_b": "bB", "img_b_off": "bA", "img_a_fcfs": "bA"})
        for config in configs:
            candidate = json.loads(Path(selected[config]["path"]).read_text())
            _, env, patches = common.candidate_command(candidate, config, 30000)
            self.assertEqual(patches, ["000", "101"] if config.startswith("img_b") else ["000"])
            self.assertEqual(env.get(common.ROLE_ENV), "154827,154829" if config == "img_b" else None)

    def test_config_deltas(self):
        # T33: registry args/env deltas apply on top of the profile; bare flags have no value.
        with mock.patch.dict(common.POLICIES, {"img_t": (["000", "101"], "lpm", True)}), \
             mock.patch.dict(common.EXTRA_ARGS, {"img_t": {"--chunked-prefill-size": 4096, "--enable-mixed-chunk": None,
                                                          "--page-size": 32}}), \
             mock.patch.dict(common.EXTRA_ENV, {"img_t": {common.ROLE_ENV: "154827"}}):
            candidate = json.loads((runner.REPO / "submission/candidate-bB-0922f.json").read_text())
            command, env, _ = common.candidate_command(candidate, "img_t", 30000)
            self.assertEqual(command[command.index("--chunked-prefill-size") + 1], "4096")
            self.assertEqual(command[command.index("--page-size") + 1], "32")
            self.assertEqual(command.count("--page-size"), 1)
            self.assertIn("--enable-mixed-chunk", command)
            self.assertEqual(env[common.ROLE_ENV], "154827")

    def test_profile_mapping_uses_contents_and_v12_stack(self):
        paths = sorted((runner.REPO / "submission").glob("candidate-b[0-3]*.json"), reverse=True)
        configs = [c for c in common.POLICIES if c != "stock" and not c.startswith("img_")]
        profiles, selected = common.select_profiles(paths, configs)
        self.assertEqual(len(profiles), 4)
        for config, prefix in (("baseline", "b0"), ("d1", "b1"), ("spf", "b2"), ("spf_d1", "b3"),
                               ("hrrn", "b0"), ("d1v12", "b1"), ("spf_d1v12", "b3")):
            self.assertIn("candidate-" + prefix, selected[config]["path"])
        candidate = json.loads(Path(selected["spf_d1v12"]["path"]).read_text())
        command, env, patches = common.candidate_command(candidate, "spf_d1v12", 30000)
        self.assertEqual(patches, ["000", "001", "002", "004"])
        self.assertEqual(command[command.index("--tool-call-parser") + 1], "glm47")
        self.assertIn(common.ROLE_ENV, env)

    def test_ambiguous_profiles_rejected(self):
        path = runner.REPO / "submission/candidate-b0-d0-baseline.json"
        with self.assertRaises(ValueError):
            common.select_profiles([path, path], ["baseline"])

    def test_explicit_matrix_baseline_first_exact_other_order(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            ctl = FakeController(tmp, argv=["--matrix", "hrrn,spf_d1v12,baseline,d1v12"])
            ctl.execute()
        measured = [c for c in ctl.calls if c[0] == "measure"]
        self.assertEqual(measured, [("measure", "baseline", n) for n in (6, 10, 14, 18)] +
                         [("measure", c, n) for c in ("hrrn", "spf_d1v12", "d1v12") for n in (14, 18)])
        self.assertEqual(ctl.summary["status"], "completed")

    def test_no_baseline_fast_honors_hint_and_matrix(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            ctl = FakeController(tmp, argv=["--matrix", "d1v12,hrrn", "--ladder-mode", "fast", "--hint", "22"])
            ctl.execute()
        measured = [c for c in ctl.calls if c[0] == "measure"]
        self.assertEqual(measured, [("measure", c, n) for c in ("d1v12", "hrrn") for n in (22, 18, 10, 14)])
        self.assertNotIn(("start", "baseline", None), ctl.calls)
        self.assertEqual(ctl.summary["status"], "completed")

    def test_no_baseline_official_starts_ten(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            ctl = FakeController(tmp, argv=["--matrix", "d1", "--ladder-mode", "official-climb"])
            ctl.execute()
        self.assertEqual([c[2] for c in ctl.calls if c[0] == "measure"], [10, 14, 18])

    def test_levels_runs_exact_order_even_after_failure(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            ctl = FakeController(tmp, argv=["--matrix", "spf,hrrn", "--ladder-mode", "levels", "--levels", "18,6,10"])
            ctl.execute()
        self.assertEqual([c for c in ctl.calls if c[0] == "measure"],
                         [("measure", c, n) for c in ("spf", "hrrn") for n in (18, 6, 10)])
        self.assertEqual(ctl.summary["status"], "completed")

    def test_baseline_levels_and_fast_keep_calibration_without_duplicates(self):
        for mode, options, expected in (("levels", ["--levels", "10,14,18"], [6, 10, 14, 18]),
                                        ("fast", ["--hint", "22"], [6, 10, 22, 18, 14])):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
                ctl = FakeController(tmp, argv=["--matrix", "baseline,d1", "--ladder-mode", mode, *options])
                ctl.execute()
                self.assertEqual([c[2] for c in ctl.calls if c[:2] == ("measure", "baseline")], expected)
                self.assertEqual([c[2] for c in ctl.calls if c[:2] == ("measure", "d1")], [14, 18])

    def test_max_n_and_max_levels_bound_measurements(self):
        for options, expected in ((["--max-n", "14"], [10, 14]),
                                  (["--max-levels", "1"], [10]),
                                  (["--max-levels", "3"], [10, 14, 18])):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
                ctl = FakeController(tmp, argv=["--matrix", "hrrn", *options])
                ctl.execute()
                self.assertEqual([c[2] for c in ctl.calls if c[0] == "measure"], expected)
                self.assertEqual(ctl.summary["status"], "completed" if len(expected) == 3 else "incomplete")
        args = runner.parser().parse_args(["--matrix", "spf", "--ladder-mode", "fast", "--hint", "2", "--max-n", "2"])
        self.assertEqual(runner.next_level(args, {}), (2, None))
        self.assertEqual(runner.next_level(args, {2: False}), (None, "no_passing_rung"))

    def test_explicit_missing_patch_fails_without_silent_matrix_rewrite(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            ctl = FakeController(tmp, spf=False, argv=["--matrix", "spf"])
            ctl.execute()
        self.assertEqual(ctl.summary["status"], "aborted")
        self.assertFalse(any(c[0] == "measure" for c in ctl.calls))
        self.assertIn("patch 002", ctl.summary["errors"][0])

    def test_invalid_contract_stops_before_controller(self):
        bad = (["--matrix", "stock"], ["--matrix", "spf,spf"], ["--matrix", "d1", "--hrrn"],
               ["--ladder-mode", "levels"], ["--max-levels", "0"], ["--levels", "3"],
               ["--matrix", "spf", "--ladder-mode", "fast", "--hint", "18", "--max-n", "14"],
               ["--collect-only"])
        from contextlib import redirect_stderr
        for argv in bad:
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()), mock.patch.object(runner, "Controller") as ctor:
                with self.assertRaises(SystemExit):
                    runner.main(["--minutes", "40", *argv])
                ctor.assert_not_called()

    def test_real_sync_manifest_and_zero_budget_crash_recovery(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            out, pod = Path(tmp) / "out", Path(tmp) / "pod"
            (out / "profiles").mkdir(parents=True)  # daemon owns this before launch
            profile = out / "profiles/base.json"
            profile.write_bytes((runner.REPO / "submission/candidate-b1-d1.json").read_bytes())
            argv = ["--service-id", "mock-id", "--run-id", "recovery", "--out", str(out), "--profiles", str(profile),
                    "--matrix", "d1", "--ladder-mode", "levels", "--levels", "6,10", "--budget-minutes", "1000"]
            ctl = MockPodController(runner.parser().parse_args(argv), pod)
            ctl.execute()
            manifest = json.loads((out / "session_result.json").read_text())
            self.assertEqual([r["N"] for r in manifest["results"]], [10, 6])
            for result in manifest["results"]:
                self.assertEqual(result["profile_sha256"], common.digest(profile))
                self.assertTrue((out / result["raw"]).is_file())
                self.assertFalse(Path(result["raw"]).is_absolute())
                self.assertFalse(result["candidate_exact"])
                self.assertEqual(result["p0"], {})
            self.assertEqual(ctl.gpu_out, "/sjtu/linhang/arena/runs/recovery")
            # Simulate a crash with no aggregate, plus one corrupted local file.
            (out / "summary.json").unlink()
            (out / "session_result.json").unlink()
            (out / "d1/N6/raw.jsonl").write_text("corrupt")
            argv[argv.index("--budget-minutes") + 1] = "0"
            created = []
            def controller(args):
                recovery = MockPodController(args, pod)
                created.append(recovery)
                return recovery
            with mock.patch.object(runner, "Controller", side_effect=controller), mock.patch.object(subprocess, "run", side_effect=AssertionError("real subprocess")):
                self.assertEqual(runner.main([*argv, "--collect-only"]), 0)
            recovery = created[0]
            self.assertEqual(json.loads((out / "session_result.json").read_text()), manifest)
            self.assertEqual(json.loads((out / "summary.json").read_text())["status"], "recovered")
            self.assertFalse(any(c[0] in ("launch", "stop") for c in recovery.calls))
            self.assertTrue(any(c[0] == "export" for c in recovery.calls))
            with self.assertRaises(ValueError):
                ctl.execute()  # stale output cannot be reused as a fresh run

    def test_remote_start_uses_transferred_candidate_and_writes_digest_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = root / "profiles/custom.json"
            candidate = json.loads((runner.REPO / "submission/candidate-b1-d1.json").read_text())
            candidate["env"]["PROFILE_FIXTURE"] = "selected"
            candidate["command"] += " --mem-fraction-static 0.73"
            common.write_json(profile, candidate)
            (root / "work/stock/sglang").mkdir(parents=True)
            (root / "patches").mkdir()
            for prefix in ("000", "001"):
                (root / "patches" / (prefix + "-fixture.patch")).write_text("mock patch")
            spec = {"config": "d1", "port": 30000, "profile": "profiles/custom.json",
                    "profile_sha256": common.digest(profile), "deadline": 9999999999}
            process = mock.Mock(pid=12345)
            process.poll.return_value = None
            with mock.patch.object(remote, "ROOT", root), mock.patch.object(remote, "WORK", root/"work"), \
                 mock.patch.object(remote, "ART", root/"artifacts"), mock.patch.object(remote, "stop_engine"), \
                 mock.patch.object(remote.socket, "socket"), mock.patch.object(remote, "run", return_value={"returncode": 0}), \
                 mock.patch.object(remote.subprocess, "Popen", return_value=process) as popen, \
                 mock.patch.object(remote, "process_info", return_value={"start": "fixture"}), \
                 mock.patch.object(remote.urllib.request, "urlopen", return_value=io.BytesIO(b'{"data":[{"id":"default"}]}')):
                self.assertTrue(remote.start(spec)["ready"])
            launch = popen.call_args_list[0]
            self.assertIn("0.73", launch.args[0])
            self.assertEqual(launch.kwargs["env"]["PROFILE_FIXTURE"], "selected")
            receipt = json.loads((root / "artifacts/d1/engine-command.json").read_text())
            self.assertEqual(receipt["profile_sha256"], common.digest(profile))
            self.assertEqual(receipt["patches"], ["000", "001"])

    def test_transfer_contains_all_selected_profile_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/"out"
            out.mkdir()
            args = runner.parser().parse_args(["--minutes", "40", "--matrix", "d1", "--out", str(out),
                                             "--profiles", str(runner.REPO/"submission/candidate-b1-d1.json")])
            ctl = runner.Controller(args)
            observed = {}
            original_bundle = runner.bundle
            def bundle(path, files, cap):
                observed.update({k: v.read_bytes() for k, v in files.items()})
                return original_bundle(path, files, cap)
            def bohr(command, **kwargs):
                if command[2] == runner.UPLOAD_CHUNKS:
                    return {"verified_chunks": (len(command)-4)//3}
                if command[2] == runner.BOOTSTRAP:
                    return {"verified_files": len(observed)}
                return {"created": True}
            with mock.patch.object(runner, "inputs", return_value={}), mock.patch.object(runner, "prepare_cap", return_value={}), \
                 mock.patch.object(runner, "bundle", side_effect=bundle), mock.patch.object(ctl, "bohr", side_effect=bohr), \
                 mock.patch.object(ctl, "event"):
                ctl.transfer()
            self.assertEqual(observed["profiles/000.json"], Path(ctl.profiles[0]["path"]).read_bytes())
            self.assertTrue(ctl.uploaded)

    def test_recovery_failure_never_launches_or_stops_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = runner.parser().parse_args(["--collect-only", "--minutes", "0", "--run-id", "missing", "--out", str(Path(tmp)/"out")])
            ctl = MockPodController(args, Path(tmp)/"missing-pod")
            self.assertEqual(ctl.collect(), 2)
            self.assertFalse(any(c[0] in ("launch", "stop") for c in ctl.calls))

    def test_profile_checksum_is_bound_to_manifest(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            args = runner.parser().parse_args(["--minutes", "1000", "--matrix", "baseline", "--ladder-mode", "levels",
                                             "--levels", "6,10", "--run-id", "binding", "--out", str(Path(tmp)/"out")])
            ctl = MockPodController(args, Path(tmp)/"pod")
            ctl.execute()
            common.write_json(ctl.out / "baseline/engine-command.json", {"profile_sha256": "wrong"})
            with self.assertRaises(ValueError):
                ctl.write_results()



if __name__ == "__main__":
    unittest.main()
