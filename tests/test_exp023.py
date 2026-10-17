"""EXP-023 retrospective synthesis: source validation, JSD, windows, transient parsing, verdict.

No model, no inference, no subprocess: a synthetic EXP-007-shaped source tree is built in a
temporary directory and only the functions of tools/exp023_analyze.py are exercised.
"""

from __future__ import annotations

import ast
import csv
import io
import json
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import exp023_analyze as x  # noqa: E402
from argine.research.trace import Step, Trace, TraceMeta  # noqa: E402

SEG_LEN = {"natural-language": 130, "coding": 140, "reasoning-math": 128, "structured-json": 100}
ADAPT_HEADER = ["policy", "budget", "boundary", "steady_misses_per_step (last 128)"] + list(x.ADAPT_COLS.values())


def make_trace(seed: int, separated: bool = False) -> Trace:
    """Two layers, 6 experts, top-2. Each segment prefers its own experts."""
    rng = random.Random(seed)
    steps = [Step(graph=0, n_tokens=1, routes={0: [(0, 1)], 1: [(0, 1)]}, phase="decode", segment="warmup")]
    g = 1
    for s, name in enumerate(x.SEGMENT_ORDER):
        steps.append(Step(graph=g, n_tokens=5, routes={0: [(0, 1)] * 5, 1: [(2, 3)] * 5}, phase="prefill", segment=name))
        g += 1
        for _ in range(SEG_LEN[name]):
            routes = {}
            for layer in (0, 1):
                if separated:
                    pair = (s, s + 1) if layer == 0 else (5 - s, 4 - s)
                else:
                    pool = [s, s + 1, (s + 2) % 6] if rng.random() < 0.8 else list(range(6))
                    pair = tuple(sorted(rng.sample(pool, 2)))
                routes[layer] = [pair]
            steps.append(Step(graph=g, n_tokens=1, routes=routes, phase="decode", segment=name))
            g += 1
    steps.append(Step(graph=g, n_tokens=1, routes={0: [(4, 5)], 1: [(4, 5)]}, phase="decode", segment="flush-sentinel"))
    return Trace(meta=TraceMeta(model="synthetic", n_experts=6, top_k=2, layers=[0, 1]), steps=steps)


def adaptation_csv(neg: bool = True) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(ADAPT_HEADER)
    for pol in ("global-lru", "ablate-frozen", "adaptive-e64", "adaptive-e32"):
        for b in x.BUDGETS:
            for bd in x.BOUNDARIES:
                e64 = (-250.5 if (neg and bd == 2) else 100.0 + bd) + (7.0 if pol == "adaptive-e64" else 0.0)
                w.writerow([pol, b, bd, "10.0", "5.0", "6.0", f"{e64}", f"{max(e64, 0) + 40.0}", "30.0",
                            "True" if abs(e64) > 60 else "False", "12", "" if pol == "ablate-frozen" else "250"])
    return buf.getvalue()


def summary_json(trace: Trace, share=1.3, lag=-100) -> dict:
    idx, segs = x.counted_decode(trace)
    rng = x.segment_ranges(segs)
    budgets = {b: {"gain_vs_frozen_misses": 1000, "A_fixed_mean_allocation_gain": int(1000 * share),
                   "A_share_of_gain_mean_alloc": share, "oracle_potential_B_segment_switching": 500,
                   "reaction_lag_cost_vs_instant_piecewise": lag} for b in x.BUDGETS}
    return {"segments": [[n, a, b] for n, a, b in rng], "boundaries_counted": [r[1] for r in rng[1:]],
            "boundaries_trace": [idx[r[1]] for r in rng[1:]], "budgets": budgets}


def build_root(tmp: Path, separated: bool = False) -> Path:
    root = tmp / "repo"
    hits = io.StringIO()
    hw = csv.writer(hits, lineterminator="\n")
    hw.writerow(["campaign", "run", "qualified", "exp021_closure"])
    for i, src in enumerate(x.SOURCES):
        p = x.source_paths(src, root)
        tr = make_trace(seed=i, separated=separated)
        p["trace"].parent.mkdir(parents=True, exist_ok=True)
        tr.save(p["trace"])
        p["manifest"].write_text(json.dumps({"workload": "transition", "split": src["split"],
                                             "fingerprint": {"model": {"path": "C:\\m\\" + src["model_file"]}}}))
        t = root / src["temporal"]
        t.mkdir(parents=True, exist_ok=True)
        (t / "summary.json").write_text(json.dumps(summary_json(tr)))
        (t / "adaptation_v2.csv").write_text(adaptation_csv())
        (t / "segments.csv").write_text("policy\n")
        (t / "cross_segment.csv").write_text("policy\n")
        n, h = x.routing_sha(tr)
        (t / "replication.csv").write_text(
            "trace,workload_steps,workload_routing_sha256\n"
            f"experiments\\EXP-007-router-characterization\\raw\\{src['campaign']}\\{src['run']}\\trace.jsonl.gz,{n},{h}\n")
        hw.writerow([src["campaign"], src["run"], "True", "PASS"])
    x.source_paths(x.SOURCES[0], root)["runtime_hit_rates"].write_text(hits.getvalue())
    return root


def listing(root: Path):
    return sorted((str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*"))


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="argine2-exp023-"))
        self.root = build_root(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestSources(Base):
    def test_valid_synthetic_sources_pass(self):
        m = x.check(self.root)
        self.assertEqual([s["key"] for s in m], ["30b-dev", "30b-eval", "80b-dev", "80b-eval"])
        self.assertTrue(all(len(s["files"]) == 8 for s in m))

    def test_all_four_sources_required(self):
        with self.assertRaises(x.ProtocolError):
            x.check(self.root, x.SOURCES[:3])
        with self.assertRaises(x.ProtocolError):
            x.check(self.root, (x.SOURCES[0],) * 4)

    def test_each_required_file_missing_is_rejected(self):
        for name in list(x.COMPACT_FILES) + ["trace", "manifest"]:
            root = build_root(self.tmp / name)
            x.source_paths(x.SOURCES[1], root)[name].unlink()
            with self.assertRaises(x.ProtocolError, msg=name):
                x.check(root)

    def test_schema_and_identity_violations(self):
        src = x.SOURCES[2]
        p = x.source_paths(src, self.root)
        good = p["adaptation_v2.csv"].read_text()
        p["adaptation_v2.csv"].write_text(good.replace("excess64_positive_part", "renamed"))
        with self.assertRaises(x.ProtocolError):
            x.validate_source(src, self.root)
        p["adaptation_v2.csv"].write_text(good)
        rep = p["replication.csv"].read_text()
        p["replication.csv"].write_text(rep[:-10] + "0" * 9 + "\n")          # wrong routing hash
        with self.assertRaises(x.ProtocolError):
            x.validate_source(src, self.root)
        p["replication.csv"].write_text(rep)
        man = json.loads(p["manifest"].read_text())
        p["manifest"].write_text(json.dumps({**man, "split": "dev" if src["split"] == "eval" else "eval"}))
        with self.assertRaises(x.ProtocolError):
            x.validate_source(src, self.root)
        p["manifest"].write_text(json.dumps(man))
        s = json.loads(p["summary.json"].read_text())
        s["boundaries_counted"][0] += 1
        p["summary.json"].write_text(json.dumps(s))
        with self.assertRaises(x.ProtocolError):
            x.validate_source(src, self.root)

    def test_unqualified_run_rejected(self):
        h = x.source_paths(x.SOURCES[0], self.root)["runtime_hit_rates"]
        h.write_text(h.read_text().replace("80b-transition-eval-v2,run-00-4G,True,PASS",
                                           "80b-transition-eval-v2,run-00-4G,False,FAIL"))
        with self.assertRaises(x.ProtocolError):
            x.check(self.root)

    def test_check_writes_nothing(self):
        before = listing(self.root)
        proc_before = listing(x.PROCESSED) if x.PROCESSED.exists() else None
        x.check(self.root)
        self.assertEqual(listing(self.root), before)
        self.assertEqual(listing(x.PROCESSED) if x.PROCESSED.exists() else None, proc_before)


class TestNoProcesses(unittest.TestCase):
    def test_no_subprocess_or_model_launch_code(self):
        tree = ast.parse((ROOT / "tools" / "exp023_analyze.py").read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        self.assertTrue(imported.isdisjoint({"subprocess", "multiprocessing", "socket", "http", "urllib", "requests"}),
                        imported)
        calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertTrue(calls.isdisjoint({"system", "popen", "Popen", "spawn", "execv", "startfile"}), calls)
        self.assertFalse(hasattr(x, "subprocess"))


class TestJSD(unittest.TestCase):
    def test_identical_is_zero_and_disjoint_is_one(self):
        self.assertEqual(x.jsd({1: 3, 2: 1}, {1: 6, 2: 2}), 0.0)
        self.assertAlmostEqual(x.jsd({1: 5}, {2: 5}), 1.0)

    def test_symmetric_and_bounded(self):
        rng = random.Random(7)
        for _ in range(200):
            p = {k: rng.randint(0, 5) for k in range(8)}
            q = {k: rng.randint(0, 5) for k in range(8)}
            p[0] += 1
            q[1] += 1
            a, b = x.jsd(p, q), x.jsd(q, p)
            self.assertAlmostEqual(a, b, places=12)
            self.assertGreaterEqual(a, 0.0)
            self.assertLessEqual(a, 1.0)

    def test_zero_probability_entries(self):
        self.assertAlmostEqual(x.jsd({1: 1, 2: 1, 3: 0}, {1: 1, 2: 1}), 0.0)
        self.assertAlmostEqual(x.jsd({1: 1, 2: 0}, {1: 1, 2: 1}), x.jsd({1: 1}, {1: 1, 2: 1}))
        with self.assertRaises(x.ProtocolError):
            x.jsd({}, {1: 1})


class TestWindows(unittest.TestCase):
    def test_matched_64_step_windows(self):
        self.assertEqual(x.within_windows(0, 299), [(0, 64), (64, 128), (128, 192), (192, 256)])
        self.assertEqual(x.within_windows(897, 1125), [(897, 961), (961, 1025), (1025, 1089)])
        b, a = x.boundary_windows(("a", 0, 299), ("b", 299, 598))
        self.assertEqual((b, a), ((235, 299), (299, 363)))
        self.assertEqual(b[1] - b[0], 64)
        self.assertEqual(a[1] - a[0], 64)
        with self.assertRaises(x.ProtocolError):
            x.boundary_windows(("a", 0, 63), ("b", 63, 200))

    def test_no_within_window_crosses_a_boundary(self):
        for r0, r1 in ((0, 299), (299, 598), (10, 77), (5, 5)):
            for a, b in x.within_windows(r0, r1):
                self.assertTrue(r0 <= a < b <= r1)
                self.assertEqual(b - a, 64)

    def test_separated_segments_give_zero_within_and_one_at_boundaries(self):
        tr = make_trace(seed=0, separated=True)
        rows = x.jsd_rows(tr, "30b-dev", "30b", "dev")
        self.assertEqual(len(rows), 2 * 3)                                  # 2 layers x 3 boundaries
        for r in rows:
            self.assertEqual(r["within_median"], 0.0)                       # a crossing window would be > 0
            self.assertEqual(r["within_max"], 0.0)
            self.assertGreater(r["boundary_jsd"], 0.4)
            self.assertIsNone(r["ratio_boundary_over_within_median"])      # denominator 0 -> no ratio
        # within pairs per segment (130, 140, 128, 100 steps -> 2, 2, 2, 1 windows): 1, 1, 1, 0;
        # the reference of a boundary is the pairs of its old and new segment
        self.assertEqual([r["within_n"] for r in rows if r["layer"] == 0], [1 + 1, 1 + 1, 1 + 0])


class TestTransient(unittest.TestCase):
    def test_selects_exactly_adaptive_e64_and_frozen_with_signed_values(self):
        rows = x.transient_rows(adaptation_csv(), "30b-dev", "30b", "dev", x.SEGMENT_ORDER)
        self.assertEqual(len(rows), 9)
        self.assertEqual([(r["budget"], r["boundary"]) for r in rows],
                         [(b, d) for b in x.BUDGETS for d in x.BOUNDARIES])
        neg = [r for r in rows if r["boundary"] == 2]
        for r in neg:
            self.assertEqual(r["frozen_excess64"], -250.5)                 # signed, not abs()
            self.assertEqual(r["adaptive_excess64"], -243.5)
            self.assertEqual(r["frozen_excess64_positive_part"], 40.0)      # separate field
            self.assertAlmostEqual(r["delta_excess64_adaptive_minus_frozen"], 7.0)
        self.assertTrue(all(r["adaptive_allocation_convergence_steps"] == 250.0 for r in rows))
        self.assertEqual(rows[0]["before_segment"], "natural-language")
        self.assertEqual(rows[0]["after_segment"], "coding")

    def test_duplicates_and_missing_rows_rejected(self):
        text = adaptation_csv()
        lines = text.splitlines()
        dup = "\n".join(lines + [l for l in lines if l.startswith("adaptive-e64,4G,1,")]) + "\n"
        with self.assertRaises(x.ProtocolError):
            x.transient_rows(dup, "k", "30b", "dev", x.SEGMENT_ORDER)
        drop = "\n".join(l for l in lines if not l.startswith("ablate-frozen,1.5G,3,")) + "\n"
        with self.assertRaises(x.ProtocolError):
            x.transient_rows(drop, "k", "30b", "dev", x.SEGMENT_ORDER)


class TestVerdict(unittest.TestCase):
    def rows(self, gain=1000, mean_gain=1200, lag=-5, share=1.2):
        return [{"trace_key": "k", "budget": "4G", "controller_gain_vs_frozen_misses": gain,
                 "A_fixed_mean_allocation_gain": mean_gain, "A_share_of_gain_mean_alloc": share,
                 "reaction_lag_cost_vs_instant_piecewise": lag} for _ in range(12)]

    def test_formal_verdict_is_deterministic_and_takes_no_data(self):
        import inspect
        self.assertEqual(list(inspect.signature(x.formal_verdict).parameters), [])
        v = x.formal_verdict()
        self.assertEqual(v["verdict"], "FALSIFIED")
        self.assertIn("current", v["scope"])
        self.assertEqual(v["basis"]["replicated"], ["30B EVAL", "80B DEV", "80B EVAL"])
        self.assertEqual(x.formal_verdict(), v)

    def test_consistency_reports_numbers_without_a_threshold(self):
        a = x.fh029_consistency(self.rows(mean_gain=1001, share=1.001, lag=-1))
        b = x.fh029_consistency(self.rows(mean_gain=2400, share=2.4, lag=-1000))
        self.assertTrue(a["consistent_with_documented_FH029"] and b["consistent_with_documented_FH029"])
        self.assertEqual(a["A_share_of_gain_mean_alloc_min"], 1.001)          # near-1 share is not "failed"
        self.assertIn("not a decision rule", a["role"])
        r = self.rows()
        r[3] = dict(r[3], A_share_of_gain_mean_alloc=1.129)
        self.assertIn("112.9%", x.fh029_consistency(r)["rounding_note"])

    def test_contradiction_is_an_evidence_error_not_a_verdict(self):
        for rows in (self.rows(mean_gain=1000, share=1.0), self.rows(mean_gain=900, share=0.9), self.rows(lag=0),
                     self.rows(lag=12)):
            with self.assertRaises(x.EvidenceInconsistencyError):
                x.fh029_consistency(rows)
        self.assertFalse(issubclass(x.EvidenceInconsistencyError, x.ProtocolError))
        with self.assertRaises(x.ProtocolError):
            x.fh029_consistency(self.rows()[:11])                          # structural, not evidence


class TestAnalyze(Base):
    def test_outputs_complete_deterministic_and_only_in_out_dir(self):
        before = listing(self.root)
        o1, o2 = self.tmp / "out1", self.tmp / "out2"
        s = x.analyze(root=self.root, out_dir=o1)
        x.analyze(root=self.root, out_dir=o2)
        names = sorted(p.name for p in o1.iterdir())
        self.assertEqual(names, ["controller_summary.csv", "jsd_by_layer.csv", "jsd_summary.csv",
                                 "source_manifest.json", "summary.json", "transient_summary.csv"])
        for n in names:
            self.assertEqual((o1 / n).read_bytes(), (o2 / n).read_bytes(), n)
        self.assertEqual(listing(self.root), before)                       # sources untouched
        t = list(csv.DictReader(io.StringIO((o1 / "transient_summary.csv").read_text())))
        self.assertEqual(len(t), 4 * 3 * 3)
        self.assertEqual([r["trace_key"] for r in t][::9], ["30b-dev", "30b-eval", "80b-dev", "80b-eval"])
        j = list(csv.DictReader(io.StringIO((o1 / "jsd_by_layer.csv").read_text())))
        self.assertEqual(len(j), 4 * 2 * 3)
        self.assertEqual(s["final_status_logic"]["verdict"], "FALSIFIED")
        for key in ("historical_hypothesis", "established_evidence", "retrospective_jsd_operationalisation",
                    "cross_trace_transient_synthesis", "final_status_logic"):
            self.assertIn(key, s)
        self.assertEqual(s["historical_hypothesis"]["text"], x.HYPOTHESIS)
        self.assertGreater(s["cross_trace_transient_synthesis"]["negative_signed_excess64_values"], 0)

    def test_verdict_does_not_depend_on_jsd(self):
        a = x.analyze(root=self.root, out_dir=self.tmp / "a")
        sep = build_root(self.tmp / "sep", separated=True)
        b = x.analyze(root=sep, out_dir=self.tmp / "b")
        self.assertNotEqual(a["retrospective_jsd_operationalisation"]["pooled"],
                            b["retrospective_jsd_operationalisation"]["pooled"])
        self.assertEqual(a["final_status_logic"]["verdict"], b["final_status_logic"]["verdict"])

    def test_verdict_does_not_depend_on_consistency_numbers(self):
        a = x.analyze(root=self.root, out_dir=self.tmp / "a")
        other = build_root(self.tmp / "other")
        for src in x.SOURCES:                                              # very different, still consistent
            p = x.source_paths(src, other)["summary.json"]
            s = json.loads(p.read_text())
            for b in s["budgets"].values():
                b.update(A_fixed_mean_allocation_gain=2400, A_share_of_gain_mean_alloc=2.4,
                         reaction_lag_cost_vs_instant_piecewise=-1)
            p.write_text(json.dumps(s))
        b = x.analyze(root=other, out_dir=self.tmp / "b")
        self.assertNotEqual(a["established_evidence"]["consistency_check"],
                            b["established_evidence"]["consistency_check"])
        self.assertEqual(a["final_status_logic"], b["final_status_logic"])
        self.assertEqual(a["final_status_logic"], x.formal_verdict())

    def test_evidence_inconsistency_is_surfaced_and_writes_nothing(self):
        p = x.source_paths(x.SOURCES[1], self.root)["summary.json"]
        s = json.loads(p.read_text())
        s["budgets"]["2G"]["reaction_lag_cost_vs_instant_piecewise"] = 50
        p.write_text(json.dumps(s))
        out = self.tmp / "out"
        with self.assertRaises(x.EvidenceInconsistencyError) as cm:
            x.analyze(root=self.root, out_dir=out)
        self.assertIn("30b-eval 2G", str(cm.exception))
        self.assertFalse(out.exists())


class TestFreezeValidation(Base):
    """The whole active freeze is validated, and --check cannot succeed on a mismatch."""

    def setUp(self):
        super().setUp()
        self.manifest = x.check(self.root)
        self.fz = self.tmp / "analysis_freeze.json"
        self.reset()

    def reset(self):
        self.write(x.freeze_record(self.manifest, "a" * 40, None))

    def write(self, rec):
        self.fz.write_text(json.dumps(rec, indent=2))

    def tampered(self, fn):
        rec = json.loads(self.fz.read_text())
        fn(rec)
        self.write(rec)
        return x.freeze_problems(self.manifest, self.fz)

    def test_fully_matching_freeze_passes(self):
        self.assertEqual(x.freeze_problems(self.manifest, self.fz), [])

    def test_changed_jsd_window_detected(self):
        bad = self.tampered(lambda r: r["jsd"].update(window_counted_decode_steps=128))
        self.assertTrue(any("'jsd'" in b for b in bad), bad)

    def test_changed_verdict_logic_and_other_spec_fields_detected(self):
        self.assertTrue(self.tampered(lambda r: r.update(verdict_logic="FALSIFIED iff share > 1.13")))
        for k, v in (("transient", {"policies": ["adaptive-e32"]}), ("output_schema", {}), ("label", "frozen before data"),
                     ("statements", []), ("version", "EXP023-R2")):
            self.reset()
            self.assertTrue(self.tampered(lambda r: r.update({k: v})), k)
        self.reset()
        self.assertTrue(self.tampered(lambda r: r.update(extra_override=True)))      # unexpected key
        self.reset()
        self.assertTrue(self.tampered(lambda r: r.pop("jsd")))                       # missing key

    def test_changed_source_identity_detected(self):
        bad = self.tampered(lambda r: r["sources"][1].update(run="run-03-4G"))
        self.assertTrue(any("identities" in b for b in bad), bad)
        self.reset()
        self.assertTrue(self.tampered(lambda r: r["sources"].reverse()))
        self.reset()
        self.assertTrue(self.tampered(lambda r: r["sources"][2].update(workload_routing_sha256="0" * 64)))

    def test_hash_mismatches_still_detected(self):
        bad = self.tampered(lambda r: r["sources"][0]["files"]["trace"].update(sha256="0" * 64))
        self.assertTrue(any("fingerprints" in b for b in bad), bad)
        self.reset()
        self.assertIn("tools/exp023_analyze.py differs from the freeze", self.tampered(lambda r: r.update(tool_sha256="0")))
        self.reset()
        self.assertIn("tests/test_exp023.py differs from the freeze", self.tampered(lambda r: r.update(tests_sha256="0")))
        self.reset()
        self.assertTrue(self.tampered(lambda r: r.update(argine2_head_at_freeze=None)))
        self.reset()
        self.assertTrue(self.tampered(lambda r: r.update(supersedes="missing_archive.json")))

    def test_frozen_utc_is_not_a_scientific_field(self):
        self.assertEqual(self.tampered(lambda r: r.update(frozen_utc="1999-01-01T00:00:00+00:00")), [])

    def test_cli_check_cannot_succeed_on_a_mismatch(self):
        import contextlib
        from unittest import mock
        orig = x.source_paths

        def run_check():
            out, err = io.StringIO(), io.StringIO()
            with mock.patch.object(x, "check", return_value=self.manifest), \
                    mock.patch.object(x, "FREEZE", self.fz), \
                    mock.patch.object(x, "source_paths", lambda src, root=None: orig(src, self.root)), \
                    contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = x.main(["--check"])
            return rc, out.getvalue() + err.getvalue()
        rc, text = run_check()
        self.assertEqual(rc, 0, text)
        self.assertIn("freeze: matches", text)
        self.assertIn("check: OK", text)
        rec = json.loads(self.fz.read_text())
        rec["jsd"]["base"] = 10
        self.write(rec)
        rc, text = run_check()
        self.assertNotEqual(rc, 0)
        self.assertNotIn("check: OK", text)
        self.assertIn("MISMATCH", text)


class TestFrozenSpec(unittest.TestCase):
    def test_spec_values(self):
        self.assertEqual(x.WINDOW, 64)
        self.assertEqual(x.BUDGETS, ("4G", "2G", "1.5G"))
        self.assertEqual((x.POLICY_FROZEN, x.POLICY_ADAPTIVE), ("ablate-frozen", "adaptive-e64"))
        self.assertIn("NOT frozen before data", x.FREEZE_SPEC["label"])
        self.assertTrue(any("after the data existed" in s for s in x.FREEZE_SPEC["statements"]))
        self.assertEqual(x.HYPOTHESIS, "Transitions change the per-layer expert distribution measurably (JSD > "
                                       "within-segment noise) and cause a transient miss excess that adaptive "
                                       "allocation shortens.")


if __name__ == "__main__":
    unittest.main()
