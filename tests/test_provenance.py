"""Regression tests for the provenance fixes found by the first Windows trace.

Each test pins one issue of the 2026-10-03 smoke run (experiments/SMOKE-30B):

1. `--expert-store` = auto produced "expert_store": null in the manifest;
2. read_mode was null although the runtime confirmed `asincrono`;
3. CPU identity was null (wrong attribute name);
4. final store statistics came from a stderr buffer and were empty;
5. BUILD-INFO.json must stay immutable (runtime validation goes elsewhere);
6. publication-quality must fail when required provenance is missing;
plus: the last forward passes of a run were missing from the trace
(TerminateProcess before the 250 ms flush) and must be detected; queue depth
must never be called exact unless every completion is exact.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from argine.research import fingerprint, iotimeline, rawlog, runner  # noqa: E402
from argine.research.trace import TraceMeta  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="argine2-prov-"))

# BUILD-INFO.json of llama-research/ as delivered on 2026-10-03. It is
# historical build provenance: if this test fails, someone rewrote it.
BUILD_INFO_SHA256 = "9aa3b12aec32b47cf784bba73df7ae257144f2f920f06989345d5be887917221"

SYSTEM_MINUS_STEADY = 5_000_000_000


def make_run(d: Path, *, decode_per_request=(3, 3), drop_tail_steps=0, read_mode="asincrono",
             store="C:\\modelli\\m.psxs", layers=2, bounded=False) -> list:
    """Write a synthetic raw.tsv + markers.jsonl and return responses.

    Two requests; each has one prompt step (2 tokens) and N decode steps.
    `drop_tail_steps` removes the last forward passes, as TerminateProcess did.
    """
    d.mkdir(parents=True, exist_ok=True)
    lines = [f"H\targine-raw\t1\t1000\t{1000 + SYSTEM_MINUS_STEADY}\t7",
             "C\tinstrumentation\targine2-research-v1",
             f"C\tstore\t{store}", "C\tbudget_bytes\t100000", f"C\tread_mode\t{read_mode}",
             "C\tstore_layers\t%d" % layers]
    markers, responses = [], []
    t, g, req = 10_000, 0, 1
    steps = []  # (graph, t, n_tokens)
    for ri, n_dec in enumerate(decode_per_request):
        pid = f"p{ri}"
        markers.append({"event": "request_start", "segment": "coding", "prompt_id": pid,
                        "system_ns": t - 5 + SYSTEM_MINUS_STEADY})
        for k in range(1 + n_dec):
            steps.append((g, t, 2 if k == 0 else 1))
            g += 1
            t += 100
        markers.append({"event": "request_end", "segment": "coding", "prompt_id": pid,
                        "system_ns": t + SYSTEM_MINUS_STEADY})
        responses.append({"prompt_id": pid, "predicted_n": n_dec + 1})
        t += 1000
    if drop_tail_steps:
        steps = steps[:-drop_tail_steps]
    for gi, (graph, ts, n) in enumerate(steps):
        for layer in range(layers):
            ids = "\t".join(["1", "2"] * n)
            lines.append(f"R\t{graph}\t{ts + layer}\t{layer}\t{n}\t2\t{ids}")
            lines.append(f"N\t{graph}\t{layer}\t0\t{ts + layer}\t{ts + layer + 5}\t2\t4")
            outcome = "m" if gi == 0 else "h"
            lines.append(f"A\t{ts + layer + 1}\t0\t{layer}\t1\t0\t{outcome}\t100\t{req if outcome == 'm' else 0}")
            if outcome == "m":
                flags = 2 if bounded else 1
                lines.append(f"I\t{req}\t0\t{layer}\t1\t0\t100\t{ts + layer + 1}\t{ts + layer + 3}"
                             f"\t{ts + layer + 2}\t{ts + layer + 4}\t1\t{flags}")
                req += 1
        lines.append(f"S\t{ts + 50}\t{graph}\t200\t100000\t{gi}\t{layers}\t0\t200\t0")
    (d / "raw.tsv").write_text("\n".join(lines) + "\n")
    (d / "markers.jsonl").write_text("".join(json.dumps(m) + "\n" for m in markers))
    return responses


class TestExpertStoreFingerprint(unittest.TestCase):
    def test_auto_resolves_next_to_model(self):
        model = TMP / "models" / "Qwen.gguf"
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(b"GGUF" + b"\0" * 64)
        (model.with_suffix(".psxs")).write_bytes(b"PSXS" + b"\1" * 64)
        r = runner.resolve_expert_store("auto", str(model))
        self.assertEqual(r["requested"], "auto")
        self.assertEqual(Path(r["resolved"]), model.with_suffix(".psxs").resolve())

    def test_auto_store_produces_non_null_fingerprint(self):
        model = TMP / "models2" / "M.gguf"
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(b"GGUF" * 100)
        model.with_suffix(".psxs").write_bytes(b"PSXS" * 100)
        r = runner.resolve_expert_store("auto", str(model))
        fp = fingerprint.collect(model=str(model), store=r["resolved"], store_required=True)
        st = fp["expert_store"]
        self.assertIsNotNone(st)
        for key in ("path", "bytes", "digest", "method"):
            self.assertTrue(st.get(key), key)
        self.assertNotIn("expert store fingerprint (path, size, checksum, method)", fp["missing"])

    def test_store_required_but_missing_fails_completeness(self):
        fp = fingerprint.collect(model=None, store=None, store_required=True)
        self.assertFalse(fp["publication_quality"])
        self.assertIn("expert store fingerprint (path, size, checksum, method)", fp["missing"])


class TestReadMode(unittest.TestCase):
    def test_effective_read_mode_from_trace(self):
        d = TMP / "rm"
        resp = make_run(d, read_mode="blocco:2")
        s = runner.process_run(d, TraceMeta(n_experts=8), resp)
        self.assertEqual(s["runtime_confirmation"]["read_mode"], "blocco:2")
        self.assertEqual(s["runtime_confirmation"]["expert_store_path"], "C:\\modelli\\m.psxs")
        self.assertTrue(s["runtime_confirmation"]["instrumentation_confirmed"])

    def test_requested_vs_effective_agreement(self):
        self.assertTrue(runner.read_modes_agree("asincrono", "asincrono"))
        self.assertTrue(runner.read_modes_agree("blocco", "blocco:2"))
        self.assertFalse(runner.read_modes_agree("sincrono", "asincrono"))
        self.assertFalse(runner.read_modes_agree("asincrono", None))


class TestCpuIdentity(unittest.TestCase):
    def test_family_strings_are_not_model_names(self):
        self.assertIsNone(fingerprint._clean_cpu("Intel64 Family 6 Model 186 Stepping 2, GenuineIntel"))
        self.assertIsNone(fingerprint._clean_cpu("CPU sconosciuta"))
        self.assertEqual(fingerprint._clean_cpu("  13th Gen Intel(R)  Core(TM) i5-1340P "),
                         "13th Gen Intel(R) Core(TM) i5-1340P")

    @unittest.skipUnless(sys.platform.startswith(("win", "linux")), "no supported CPU source")
    def test_cpu_name_non_null_on_supported_platforms(self):
        c = fingerprint.cpu_identity()
        self.assertTrue(c["name"], c)
        self.assertNotEqual(c["source"], "unavailable")
        if sys.platform.startswith("win"):
            self.assertTrue(c["source"].startswith(("windows-registry", "powershell")))

    def test_graceful_fallback(self):
        with mock.patch.object(fingerprint.sys, "platform", "plan9"), \
                mock.patch.object(fingerprint.platform, "processor", return_value=""):
            c = fingerprint.cpu_identity()
        self.assertEqual(c, {"name": None, "source": "unavailable"})

    def test_machine_identity_fields(self):
        m = fingerprint.machine_identity()
        for key in ("cpu", "cpu_source", "cores_physical", "cores_logical", "ram_total", "os"):
            self.assertIn(key, m)
        self.assertTrue(m["os"])


class TestFinalStatsAndCompleteness(unittest.TestCase):
    def test_final_store_stats_from_trace(self):
        d = TMP / "fs"
        resp = make_run(d)
        s = runner.process_run(d, TraceMeta(n_experts=8), resp)
        fs = s["final_store_stats"]
        self.assertTrue(fs["source"].startswith("research trace"))
        self.assertIsNotNone(fs["last_snapshot"])
        self.assertEqual(fs["trace_totals"]["reads_failed"], 0)
        self.assertEqual(fs["trace_totals"]["reads"], fs["trace_totals"]["outcomes"]["m"])
        self.assertEqual(fs["layers_served"], 2)
        self.assertIsNotNone(fs["expert_level_hit_rate"])

    def test_complete_trace_passes(self):
        d = TMP / "ok"
        resp = make_run(d)
        s = runner.process_run(d, TraceMeta(n_experts=8), resp)
        self.assertTrue(s["trace_completeness"]["complete"], s["trace_completeness"])

    def test_missing_tail_is_detected(self):
        """The defect of the first Windows run: the final forward pass(es) lost
        at termination, with a perfectly well-formed file."""
        d = TMP / "tail"
        resp = make_run(d, drop_tail_steps=1)
        s = runner.process_run(d, TraceMeta(n_experts=8), resp)
        tc = s["trace_completeness"]
        self.assertFalse(tc["complete"])
        last = tc["requests"][-1]
        self.assertEqual(last["decode_steps"], last["expected_decode_steps"] - 1)


class TestWorkloadOnlyStatistics(unittest.TestCase):
    def test_warmup_and_sentinel_are_excluded(self):
        d = TMP / "wl"
        resp = make_run(d)
        mk = [json.loads(l) for l in (d / "markers.jsonl").read_text().splitlines()]
        # the first forward pass (the only misses) happens inside a warm-up window
        mk.insert(0, {"event": "request_end", "segment": "warmup", "prompt_id": "warmup",
                      "system_ns": 10_050 + SYSTEM_MINUS_STEADY})
        mk.insert(0, {"event": "request_start", "segment": "warmup", "prompt_id": "warmup",
                      "system_ns": 9_990 + SYSTEM_MINUS_STEADY})
        mk.append({"event": "request_start", "segment": runner.SENTINEL_SEGMENT,
                   "prompt_id": runner.SENTINEL_SEGMENT, "system_ns": 10**9 + SYSTEM_MINUS_STEADY})
        (d / "markers.jsonl").write_text("".join(json.dumps(m) + "\n" for m in mk))
        fs = runner.process_run(d, TraceMeta(n_experts=8), resp)["final_store_stats"]
        self.assertGreater(fs["trace_totals"]["outcomes"].get("m", 0), 0)
        self.assertEqual(fs["workload"]["outcomes"].get("m", 0), 0)
        self.assertEqual(fs["expert_level_hit_rate"], 1.0)


class TestQueueDepthStatus(unittest.TestCase):
    def test_exact_only_when_all_completions_exact(self):
        d = TMP / "qd"
        make_run(d / "a", bounded=False)
        make_run(d / "b", bounded=True)
        qa = iotimeline.queue_depth(rawlog.parse(d / "a" / "raw.tsv").io)
        qb = iotimeline.queue_depth(rawlog.parse(d / "b" / "raw.tsv").io)
        self.assertEqual(qa["status"], "exact")
        self.assertEqual(qb["status"], "upper_bound")
        for q in (qa, qb):
            for key in ("exact_completion_share", "q_time_weighted_mean_over_span",
                        "q_time_weighted_mean_when_busy"):
                self.assertIn(key, q)
        self.assertNotIn("exact", qb["label"].split(" (")[0])


class TestBuildInfoImmutable(unittest.TestCase):
    def test_build_info_unchanged(self):
        p = ROOT / "llama-research" / "BUILD-INFO.json"
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), BUILD_INFO_SHA256)
        info = json.loads(p.read_text())
        self.assertIs(info["runtime_tested"], False, "delivery-time statement must stay as written")

    def test_validation_goes_to_a_separate_registry(self):
        d = TMP / "bin"
        d.mkdir(exist_ok=True)
        shutil.copy(ROOT / "llama-research" / "BUILD-INFO.json", d / "BUILD-INFO.json")
        before = (d / "BUILD-INFO.json").read_bytes()
        runner._append_validation(d, {"validation_run_id": "x-run00", "platform": "test",
                                      "instrumentation_confirmed": True,
                                      "expert_store_confirmed": True,
                                      "effective_read_mode": "asincrono"},
                                  d / "run", {"hardware": {"cpu": "test cpu"}},
                                  {"publication_quality": False})
        self.assertEqual((d / "BUILD-INFO.json").read_bytes(), before)
        rec = json.loads((d / "VALIDATIONS.jsonl").read_text().splitlines()[-1])
        self.assertEqual(rec["validation_run_id"], "x-run00")
        self.assertEqual(rec["effective_read_mode"], "asincrono")


@unittest.skipUnless(shutil.which("git"), "git not available")
class TestGitState(unittest.TestCase):
    def test_untracked_outputs_do_not_make_the_tree_dirty(self):
        import subprocess
        repo = TMP / "repo"
        repo.mkdir(exist_ok=True)
        (repo / "code.py").write_text("x = 1\n")
        g = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
        subprocess.run(g + ["init", "-q"], check=True)
        subprocess.run(g + ["add", "code.py"], check=True)
        subprocess.run(g + ["commit", "-qm", "c"], check=True)
        (repo / "run-00").mkdir()
        (repo / "run-00" / "raw.tsv").write_text("H\n")
        st = fingerprint.git_state(repo)
        self.assertTrue(st["commit"])
        self.assertFalse(st["dirty"], "untracked run outputs must not mark the tree dirty")
        (repo / "code.py").write_text("x = 2\n")
        self.assertTrue(fingerprint.git_state(repo)["dirty"])


class TestPublicationCheck(unittest.TestCase):
    def good_run(self):
        return {"validation": {"instrumentation_confirmed": True},
                "read_mode": {"effective": "asincrono"},
                "expert_store": {"effective": "C:\\m.psxs", "fingerprint_matches_effective": True},
                "final_store_stats": {"last_snapshot": {"hits": 1}},
                "trace_completeness": {"complete": True},
                "budget_violations": 0,
                "machine_state": {"warnings": []}}

    def test_complete_provenance_passes(self):
        c = runner.publication_check({"missing": []}, self.good_run())
        self.assertTrue(c["publication_quality"], c)

    def test_each_missing_item_fails(self):
        cases = [
            ("read_mode", {"effective": None}, "effective read mode"),
            ("expert_store", {"effective": None}, "effective expert store"),
            ("expert_store", {"effective": "C:\\m.psxs", "fingerprint_matches_effective": False},
             "does not match"),
            ("final_store_stats", None, "final store statistics"),
            ("trace_completeness", {"complete": False}, "trace completeness"),
            ("budget_violations", 3, "hard budget"),
            ("validation", {"instrumentation_confirmed": False}, "instrumentation"),
        ]
        for key, value, needle in cases:
            run = self.good_run()
            run[key] = value
            c = runner.publication_check({"missing": []}, run)
            self.assertFalse(c["publication_quality"], key)
            self.assertTrue(any(needle in m for m in c["missing"]), (key, c["missing"]))

    def test_fingerprint_gaps_fail(self):
        c = runner.publication_check({"missing": ["argine2 commit", "CPU identity"]}, self.good_run())
        self.assertFalse(c["publication_quality"])
        self.assertIn("argine2 commit", c["missing"])

    def test_low_ram_is_a_warning_not_a_failure(self):
        run = self.good_run()
        run["machine_state"] = {"warnings": ["available RAM before launch 3 GiB < threshold 8 GiB"]}
        c = runner.publication_check({"missing": []}, run)
        self.assertTrue(c["publication_quality"])
        self.assertEqual(len(c["warnings"]), 1)

    def test_fingerprint_completeness_requires_commit_and_cpu(self):
        with mock.patch.object(fingerprint, "git_state", return_value={"commit": None}), \
                mock.patch.object(fingerprint, "cpu_identity",
                                  return_value={"name": None, "source": "unavailable"}):
            fp = fingerprint.collect()
        self.assertFalse(fp["publication_quality"])
        self.assertIn("argine2 commit", fp["missing"])
        self.assertIn("CPU identity", fp["missing"])


class TestBackendProvenance(unittest.TestCase):
    """EXP-019 audit (2026-10-12): auto resolved to vulkan, the CPU-only research binary
    enumerated no Vulkan device, and the manifest recorded no backend at all."""

    def test_vulkan_without_devices_is_effective_cpu(self):
        b = runner.backend_provenance("auto", "vulkan", [])
        self.assertEqual((b["requested"], b["resolved"], b["effective"]), ("auto", "vulkan", "cpu"))
        self.assertEqual(b["vulkan_devices_enumerated"], [])

    def test_vulkan_with_devices(self):
        b = runner.backend_provenance("auto", "vulkan", ["Vulkan0: test iGPU"])
        self.assertEqual(b["effective"], "vulkan")

    def test_cpu(self):
        b = runner.backend_provenance("cpu", "cpu", None)
        self.assertEqual(b["effective"], "cpu")
        self.assertIsNone(b["vulkan_devices_enumerated"])

    def test_trace_completeness_declares_workload_scope(self):
        d = TMP / "scope"
        resp = make_run(d)
        s = runner.process_run(d, TraceMeta(n_experts=8), resp)
        self.assertIn("workload requests only", s["trace_completeness"]["scope"])


if __name__ == "__main__":
    unittest.main()
