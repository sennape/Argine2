"""EXP-027 runner: plan structure, record validation and result classification.

No model, no inference, no subprocess: only the pure functions of
tools/exp027_run.py are exercised.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import exp027_run as x  # noqa: E402


def good_record(threads: int, gen: float = 6.0, rnd: int = 1, model: str = "30b",
                idx: int = 1) -> dict:
    return {
        "event": "benchmark", "model": {"path": x.MODELS[model]["path"]},
        "plan": {"threads": threads, "threads_batch": 12, "backend": "cpu", "gpu_layers": 0,
                 "expert_store": True, "expert_cache_budget": 4 * 1024 ** 3},
        "runtime": {"backend": "cpu", "build": "llama-research-v2@abcdef12"},
        "metrics": {"runs": [{"gen_tok_s": gen, "prompt_tok_s": 40.0, "ttft_s": 2.0}],
                    "errors": [], "resources": {"ram_peak": 6 * 1024 ** 3}},
        "modo_lettura": "asincrono", "repeat": 1, "n_predict": 128, "attempts": [],
        "exp027": {"valid": True, "model_key": model, "threads": threads, "round": rnd,
                   "order_index": idx},
    }


class TestPlan(unittest.TestCase):
    def setUp(self):
        self.plan = x.build_plan()

    def test_24_launches_3_per_model_and_thread(self):
        self.assertEqual(len(self.plan), 24)
        self.assertEqual(x.check_plan(self.plan), [])
        for key in ("30b", "80b"):
            for t in (4, 6, 8, 12):
                self.assertEqual(sum(1 for i in self.plan
                                     if i["model_key"] == key and i["threads"] == t), 3)

    def test_order_is_the_protocol_order(self):
        seq = [(i["model_key"], i["round"], i["threads"]) for i in self.plan]
        expected = ([("30b", 1, t) for t in (4, 6, 12, 8)] + [("30b", 2, t) for t in (6, 8, 4, 12)]
                    + [("30b", 3, t) for t in (8, 12, 6, 4)] + [("80b", 1, t) for t in (12, 4, 8, 6)]
                    + [("80b", 2, t) for t in (4, 6, 12, 8)] + [("80b", 3, t) for t in (6, 8, 4, 12)])
        self.assertEqual(seq, expected)
        self.assertEqual([i["order_index"] for i in self.plan], list(range(1, 25)))

    def test_every_command_has_the_fixed_flags(self):
        for i in self.plan:
            a = x.benchmark_args(i)
            for flag, val in (("--repeat", "1"), ("--retries", "0"), ("--backend", "cpu"),
                              ("--expert-budget", "4G"), ("--read-mode", "asincrono"),
                              ("--n-predict", "128"), ("--threads", str(i["threads"]))):
                self.assertEqual(a[a.index(flag) + 1], val, (i["order_index"], flag))
            self.assertIn("--expert-store", a)
            self.assertNotIn("--dry-run", a)
            self.assertEqual(i["batch_threads"], 12)
            self.assertIn("--dry-run", x.benchmark_args(i, dry_run=True))

    def test_check_plan_catches_a_broken_plan(self):
        broken = self.plan[:-1]
        self.assertTrue(x.check_plan(broken))


class TestValidation(unittest.TestCase):
    def test_valid_record(self):
        it = x.build_plan()[0]
        self.assertEqual(x.validate(good_record(it["threads"]), it, 0, "abcdef12"), [])

    def test_mismatches_fail_loudly(self):
        it = x.build_plan()[0]
        cases = {
            "threads_batch": lambda r: r["plan"].__setitem__("threads_batch", 16),
            "threads": lambda r: r["plan"].__setitem__("threads", 8),
            "backend": lambda r: r["runtime"].__setitem__("backend", "vulkan"),
            "gpu": lambda r: r["plan"].__setitem__("gpu_layers", 49),
            "budget": lambda r: r["plan"].__setitem__("expert_cache_budget", 2 * 1024 ** 3),
            "read": lambda r: r.__setitem__("modo_lettura", "sincrono"),
            "read_missing": lambda r: r.__setitem__("modo_lettura", None),
            "repeat": lambda r: r.__setitem__("repeat", 3),
            "retry": lambda r: r.__setitem__("attempts", [{"attempt": 1, "oom": "x"}]),
            "errors": lambda r: r["metrics"].__setitem__("errors", ["run 1: boom"]),
            "binary": lambda r: r["runtime"].__setitem__("build", "llama-research@00000000"),
        }
        for name, mutate in cases.items():
            r = good_record(it["threads"])
            mutate(r)
            self.assertTrue(x.validate(r, it, 0, "abcdef12"), name)
        self.assertTrue(x.validate(good_record(it["threads"]), it, 1, "abcdef12"))
        self.assertTrue(x.validate(None, it, 0, "abcdef12"))

    def test_new_records_reads_only_appended_lines(self):
        d = Path(tempfile.mkdtemp())
        log = d / "runs.jsonl"
        log.write_text(json.dumps({"event": "old"}) + "\n", encoding="utf-8")
        off = log.stat().st_size
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"event": "benchmark", "n": 1}) + "\n")
        got = x.new_records(log, off)
        self.assertEqual([r["event"] for _, r in got], ["benchmark"])
        self.assertEqual(got[0][0], off)


def launches(values_by_thread, model="30b"):
    recs, idx = [], 0
    for t, vals in values_by_thread.items():
        for rnd, v in enumerate(vals, 1):
            idx += 1
            recs.append(good_record(t, v, rnd, model, idx))
    return recs


class TestAnalysis(unittest.TestCase):
    def test_clear_best(self):
        recs = launches({4: [6.50, 6.52, 6.49], 6: [6.00, 6.02, 5.99],
                         8: [5.70, 5.72, 5.69], 12: [5.20, 5.22, 5.19]})
        csv_text, s = x.analyze(recs)
        self.assertEqual(s["models"]["30b"]["verdict"], "CLEAR BEST")
        self.assertEqual(s["models"]["30b"]["best"], 4)
        self.assertFalse(s["models"]["30b"]["recommend_fourth_round"])
        self.assertIn("6.50 6.52 6.49", csv_text)
        self.assertEqual(s["models"]["80b"]["verdict"], "INCONCLUSIVE")   # no 80B data

    def test_practical_tie(self):
        recs = launches({4: [6.50, 6.40, 6.55], 6: [6.45, 6.52, 6.42],
                         8: [5.70, 5.72, 5.69], 12: [5.20, 5.22, 5.19]})
        s = x.analyze(recs)[1]["models"]["30b"]
        self.assertEqual(s["verdict"], "PRACTICAL TIE")
        self.assertEqual(s["tied"], [4, 6])
        self.assertTrue(s["recommend_fourth_round"])

    def test_inconclusive_when_launch_variation_is_large(self):
        recs = launches({4: [6.5, 5.5, 7.2], 6: [6.0, 6.0, 6.0],
                         8: [5.7, 5.7, 5.7], 12: [5.2, 5.2, 5.2]})
        self.assertEqual(x.analyze(recs)[1]["models"]["30b"]["verdict"], "INCONCLUSIVE")

    def test_inconclusive_when_round_effect_explains_the_gap(self):
        # every configuration drifts up by ~4% per round; best leads by 1.5%
        recs = launches({4: [6.09, 6.34, 6.59], 6: [6.00, 6.24, 6.49],
                         8: [5.70, 5.93, 6.17], 12: [5.20, 5.41, 5.62]})
        s = x.analyze(recs)[1]["models"]["30b"]
        self.assertNotEqual(s["verdict"], "CLEAR BEST")
        self.assertTrue(s["recommend_fourth_round"])

    def test_invalid_launches_are_excluded(self):
        recs = launches({4: [6.5, 6.5, 6.5], 6: [6.0, 6.0, 6.0], 8: [5.7] * 3, 12: [5.2] * 3})
        recs[0]["exp027"]["valid"] = False
        s = x.analyze(recs)[1]
        self.assertEqual(s["launches_valid"], 11)
        self.assertEqual(s["models"]["30b"]["verdict"], "INCONCLUSIVE")


class TestLargeFileFingerprints(unittest.TestCase):
    """Models/stores (~107 GB) must not be read in full before the launches."""

    def setUp(self):
        from argine.research import fingerprint
        self.fp = fingerprint
        self.d = Path(tempfile.mkdtemp())
        self.models = {}
        for key in ("30b", "80b"):
            m = self.d / f"{key}.gguf"
            m.write_bytes(b"m" * 1000)
            m.with_suffix(".psxs").write_bytes(b"s" * 2000)
            self.models[key] = {"name": key, "path": str(m)}
        (self.d / "bin").mkdir()
        (self.d / "bin" / "llama-server").write_bytes(b"binary")
        self.calls = []
        self._orig = fingerprint.file_checksum
        self._orig_cache = fingerprint._cache_path
        cache = self.d / "cache.json"
        cache.write_text("{}")
        fingerprint._cache_path = lambda: cache

        def spy(path, full=False):
            self.calls.append((Path(path).name, full))
            return self._orig(path, full)
        fingerprint.file_checksum = spy

    def tearDown(self):
        self.fp.file_checksum = self._orig
        self.fp._cache_path = self._orig_cache

    def _session(self):
        return {"provenance": x.session_provenance(self.d / "bin", self.models)}

    def test_pre_run_provenance_never_full_reads_models_or_stores(self):
        prov = self._session()["provenance"]
        large = {n for n, full in self.calls if full and n.endswith((".gguf", ".psxs"))}
        self.assertEqual(large, set())
        self.assertIn(("llama-server", True), self.calls)        # small binary: full hash ok
        e = prov["models"]["30b"]["model"]
        self.assertEqual(e["full_status"], "pending post-run")
        for k in ("canonical_path", "bytes", "mtime", "mtime_ns"):
            self.assertIn(k, e["pre_run"])
        self.assertEqual(e["fast"]["method"], "sha256")      # tiny file: fast == full method
        self.assertIn("power", prov)

    def test_known_full_hash_is_reused_from_the_cache_without_reading(self):
        m = Path(self.models["30b"]["path"])
        st = m.stat()
        key = f"{m.resolve()}|{st.st_size}|{int(st.st_mtime)}|sha256"
        self.fp._cache_path().write_text(json.dumps({key: "ab" * 32}))
        prov = self._session()["provenance"]
        self.assertEqual(prov["models"]["30b"]["model"]["full"]["digest"], "ab" * 32)
        self.assertTrue(prov["models"]["30b"]["model"]["full_status"].startswith("reused"))

    def test_post_run_finalisation_computes_pending_and_is_idempotent(self):
        session = self._session()
        hashed = []

        def fake(path):
            hashed.append(Path(path).name)
            return {"digest": "cd" * 32, "method": "sha256"}
        self.assertEqual(x.finalize_fingerprints(session, hasher=fake), [])
        self.assertEqual(sorted(hashed), ["30b.gguf", "30b.psxs", "80b.gguf", "80b.psxs"])
        self.assertTrue(session["fingerprints_final"])
        self.assertTrue(session["valid"])
        e = session["provenance"]["models"]["80b"]["expert_store"]
        self.assertEqual(e["full"]["digest"], "cd" * 32)
        self.assertEqual(e["full_status"], "computed post-run")
        hashed.clear()                                   # resume / re-finalise: no recomputation
        self.assertEqual(x.finalize_fingerprints(session, hasher=fake), [])
        self.assertEqual(hashed, [])

    def test_changed_file_invalidates_the_session(self):
        session = self._session()
        p = Path(self.models["80b"]["path"])
        p.write_bytes(b"m" * 1001)
        problems = x.finalize_fingerprints(session, hasher=lambda q: {"digest": "x", "method": "sha256"})
        self.assertTrue(problems)
        self.assertFalse(session["valid"])
        self.assertIn("changed", session["invalid_reason"])

    def test_preflight_does_not_fingerprint_anything(self):
        import subprocess as sp
        orig = sp.run

        class R:
            returncode = 0
            stdout = "llama-server --threads {t} --threads-batch 12 --n-gpu-layers 0"
            stderr = ""

        def fake_run(cmd, **kw):
            r = R()
            r.stdout = R.stdout.format(t=cmd[cmd.index("--threads") + 1])
            return r
        sp.run = fake_run
        try:
            res = x.preflight(x.build_plan(), self.d / "bin")
        finally:
            sp.run = orig
        self.assertEqual(len(res), 8)
        self.assertTrue(all(r["ok"] for r in res))
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
