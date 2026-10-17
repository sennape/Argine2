"""EXP-010 protocol tooling: selection, freeze, closure rule, CLI provenance.

Synthetic fixtures only: no EXP-007 trace is read or replayed here.
"""

from __future__ import annotations

import copy
import gzip
import json
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import exp010_analysis as X  # noqa: E402
from argine.research import trace as T  # noqa: E402

TOOL = Path(X.__file__).resolve()
README = f"""# EXP-010 synthetic

{X.MARK_BEGIN}
protocol body for tests
{X.MARK_END}

outside the protocol
"""


def write_csv(rows, p):
    X.write_csv(rows, Path(p))


class Fixture:
    """A throw-away repository root with frozen fake code, fake raw traces and
    DEV / EVAL output trees whose miss counts come from caller functions."""

    def __init__(self, dev_misses=None, eval_misses=None, selected=None):
        self.root = Path(tempfile.mkdtemp(prefix="exp010-"))
        (self.root / X.EXP_REL).mkdir(parents=True)
        (self.root / X.README_REL).write_text(README, encoding="utf-8")
        for f in X.CODE_FILES:
            p = self.root / f
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f"# fake {f}\n", encoding="utf-8")
        self.raw = self.root / X.RAW_REL
        self.sha = {}
        for (m, f), (dev, ev) in X.PAIRS.items():
            for camp in (dev, ev):
                p = self.raw / camp / "run-00-4G" / "trace.jsonl.gz"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(f"trace {camp}".encode())
                self.sha[camp] = X.sha256_file(p)
        X.freeze(self.root, TOOL)
        self.dev = self.root / X.EXP_REL / "dev"
        self.eval = self.root / X.EXP_REL / "eval"
        self.ea = self.root / X.EXP_REL / "eval-adaptive"
        self.dev_misses = dev_misses or (lambda m, f, b, v: 1000)
        self.eval_misses = eval_misses or (lambda m, f, b, role: 1000)
        self.write_dev()

    def ident(self, camp, m, f, split):
        return {"path": camp, "sha256": self.sha[camp],
                "meta": {"model": f"synthetic-{m.upper()}", "split": split,
                         "workload": f"{f}@abcdef123456"}}

    def write_dev(self):
        variants = [X.UNIFORM, "global-lru"] + [c["variant"] for c in X.CANDIDATES] + \
                   sorted(X.DIAGNOSTIC_VARIANTS - {"global-lru"})
        for (m, f), (dev, _ev) in X.PAIRS.items():
            rows = []
            for b, bb in X.BUDGETS.items():
                for v in variants:
                    rows.append({"policy": v, "budget_bytes": bb, "misses": self.dev_misses(m, f, b, v),
                                 "variant": v, "budget": b, "feasible": "True",
                                 "moves": 10, "bytes_moved": 1000})
            write_csv(rows, self.dev / dev / "adaptive.csv")
            man = {"command": "adaptive", "trace": self.ident(dev, m, f, "dev"),
                   "budgets_bytes": list(X.BUDGETS.values()), "phase": "decode",
                   "epochs": [32, 64, 128], "pinned": [0.0, 0.25, 0.5], "ablate": True}
            (self.dev / dev / "adaptive_manifest.json").write_text(json.dumps(man))

    def write_eval(self, sel, static_train="separate"):
        for (m, f), (dev, ev) in X.PAIRS.items():
            cand = X.CAND_BY_ID[sel["models"][m]["selected"]]
            rows = []
            for b, bb in X.BUDGETS.items():
                for role, pol in X.ROLE_POLICY.items():
                    rows.append({"policy": pol, "budget_bytes": bb, "budget": b, "feasible": "True",
                                 "misses": self.eval_misses(m, f, b, role), "hit_rate": 0.5,
                                 "bytes_per_token": 1.0, "stall_ms_per_token_modelled": 2.0,
                                 "train": static_train if role == "static_dev" else "-"})
                rows.append({"policy": "adaptive-e64", "budget_bytes": bb, "budget": b,
                             "feasible": "True", "misses": self.eval_misses(m, f, b, "adaptive_sel"),
                             "train": "online"})
            write_csv(rows, self.eval / ev / "policy_comparison.csv")
            cman = {"command": "compare", "trace": self.ident(ev, m, f, "eval"),
                    "train": self.ident(dev, m, f, "dev"), "train_mode": "separate",
                    "budgets_bytes": list(X.BUDGETS.values()), "phase": "decode"}
            (self.eval / ev / "compare_manifest.json").write_text(json.dumps(cman))
            arows = []
            for b, bb in X.BUDGETS.items():
                arows.append({"policy": X.UNIFORM, "variant": X.UNIFORM, "budget_bytes": bb,
                              "feasible": "True", "misses": self.eval_misses(m, f, b, "uniform")})
                arows.append({"policy": cand["variant"], "variant": cand["variant"],
                              "budget_bytes": bb, "feasible": "True",
                              "misses": self.eval_misses(m, f, b, "adaptive_sel"),
                              "moves": 5, "bytes_moved": 500, "hit_rate": 0.6})
            write_csv(arows, self.ea / ev / "adaptive.csv")
            aman = {"command": "adaptive", "trace": self.ident(ev, m, f, "eval"),
                    "budgets_bytes": list(X.BUDGETS.values()), "phase": "decode",
                    "epochs": [cand["epoch"]], "pinned": [cand["pinned"]] if cand["pinned"] else [0.0],
                    "ablate": False}
            (self.ea / ev / "adaptive_manifest.json").write_text(json.dumps(aman))

    def select(self):
        return X.build_selection(self.dev, self.root, self.raw, TOOL)

    def analyze(self, doc, **kw):
        return X.analyze(doc, self.eval, self.ea, self.root / "processed", self.root, self.raw,
                         self.dev, None, TOOL, **kw)

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)


def cells_with(g_by_cand, bytes_by_cand=None):
    """21 synthetic DEV cells where candidate c has miss ratio g_by_cand[c] in every cell."""
    cells = {}
    for f in X.FAMILIES:
        for b in X.BUDGETS:
            row = {X.UNIFORM: {"misses": "100000"}}
            for c in X.CANDIDATES:
                row[c["variant"]] = {"misses": str(100000 * g_by_cand[c["id"]]),
                                     "bytes_moved": str((bytes_by_cand or {}).get(c["id"], 0) / 21)}
            cells[(f, b)] = row
    return cells


class TestStats(unittest.TestCase):
    def test_geomean(self):
        self.assertAlmostEqual(X.geomean([2.0, 8.0]), 4.0)
        self.assertAlmostEqual(X.geomean([1.0]), 1.0)
        with self.assertRaises(ValueError):
            X.geomean([1.0, 0.0])

    def test_21_cell_equal_weighting(self):
        cells = {(f, b): {"a": {"misses": 200 if (f, b) == ("coding", "4G") else 100},
                          "s": {"misses": 100}} for f in X.FAMILIES for b in X.BUDGETS}
        self.assertEqual(len(cells), 21)
        self.assertAlmostEqual(X._g(cells, sorted(cells), "a", "s"), 2 ** (1 / 21))

    def test_ties_count_half(self):
        w = X.wins([1, 2, 3, 4], [2, 2, 2, 2])
        self.assertEqual((w["wins"], w["ties"], w["losses"], w["W"]), (1, 1, 2, 1.5))


class TestSelection(unittest.TestCase):
    def test_lowest_g_selected(self):
        s = X.select_model(cells_with({"e32": 1.0, "e64": 0.98, "e128": 0.97,
                                       "e64-pin25": 0.99, "e64-pin50": 1.01}))
        self.assertEqual(s["selected"], "e128")
        self.assertEqual(s["tie_break_path"], ["G_DEV"])
        self.assertAlmostEqual(s["candidates"]["e64"]["G_DEV"], 0.98)

    def test_tie_break_bytes_moved(self):
        s = X.select_model(cells_with({"e32": 0.95, "e64": 0.95, "e128": 0.96,
                                       "e64-pin25": 0.99, "e64-pin50": 0.99},
                                      {"e32": 5000, "e64": 4000}))
        self.assertEqual(s["selected"], "e64")
        self.assertEqual(s["tie_break_path"], ["G_DEV", "bytes_moved_total"])

    def test_tie_break_larger_epoch(self):
        s = X.select_model(cells_with({"e32": 0.95, "e64": 0.95, "e128": 0.95,
                                       "e64-pin25": 0.99, "e64-pin50": 0.99}))
        self.assertEqual(s["selected"], "e128")
        self.assertEqual(s["tie_break_path"], ["G_DEV", "bytes_moved_total", "larger_epoch"])

    def test_tie_break_no_pinning(self):
        s = X.select_model(cells_with({"e32": 0.99, "e64": 0.95, "e128": 0.99,
                                       "e64-pin25": 0.95, "e64-pin50": 0.95}))
        self.assertEqual(s["selected"], "e64")
        self.assertEqual(s["tie_break_path"][-1], "no_pinning")

    def test_tie_threshold(self):
        s = X.select_model(cells_with({"e32": 0.95, "e64": 0.95 + 5e-5, "e128": 0.99,
                                       "e64-pin25": 0.99, "e64-pin50": 0.99}, {"e32": 9, "e64": 1}))
        self.assertEqual(s["selected"], "e64")          # within 1e-4: bytes moved decides
        s = X.select_model(cells_with({"e32": 0.95, "e64": 0.95 + 2e-4, "e128": 0.99,
                                       "e64-pin25": 0.99, "e64-pin50": 0.99}, {"e32": 9, "e64": 1}))
        self.assertEqual(s["selected"], "e32")          # beyond 1e-4: G decides

    def test_one_config_per_model_end_to_end(self):
        def dm(m, f, b, v):
            best = "adaptive-e128" if m == "30b" else "adaptive-e64-pin50"
            return 900 if v == best else 1000
        fx = Fixture(dev_misses=dm)
        try:
            doc = fx.select()
            models = doc["selection"]["models"]
            self.assertEqual(set(models), {"30b", "80b"})
            self.assertEqual(models["30b"]["selected"], "e128")
            self.assertEqual(models["80b"]["selected"], "e64-pin50")
            for m in models.values():
                self.assertEqual(len(m["candidates"]), 5)
                self.assertTrue(all(c["n_cells"] == 21 for c in m["candidates"].values()))
            self.assertEqual(doc["selection"]["leakage"], X.LEAKAGE)
            self.assertIn("argine2_commit", doc["selection"]["provenance"])
        finally:
            fx.close()

    def test_select_never_reads_eval(self):
        fx = Fixture()
        try:
            sel0 = fx.select()["selection"]
            fx.write_eval(sel0)
            seen = []
            orig_csv, orig_json = X.read_csv, X.read_json
            with mock.patch.object(X, "read_csv", side_effect=lambda p: seen.append(str(p)) or orig_csv(p)), \
                 mock.patch.object(X, "read_json", side_effect=lambda p: seen.append(str(p)) or orig_json(p)):
                fx.select()
            self.assertTrue(seen)
            self.assertFalse([p for p in seen if "eval" in Path(p).parts or "eval-adaptive" in Path(p).parts])
        finally:
            fx.close()

    def test_eval_split_as_dev_rejected(self):
        fx = Fixture()
        try:
            dev = X.PAIRS[("30b", "coding")][0]
            p = fx.dev / dev / "adaptive_manifest.json"
            man = json.loads(p.read_text())
            man["trace"]["meta"]["split"] = "eval"
            p.write_text(json.dumps(man))
            with self.assertRaises(X.ProtocolError):
                fx.select()
        finally:
            fx.close()

    def test_wrong_trace_hash_rejected(self):
        fx = Fixture()
        try:
            dev = X.PAIRS[("80b", "transition")][0]
            p = fx.dev / dev / "adaptive_manifest.json"
            man = json.loads(p.read_text())
            man["trace"]["sha256"] = "0" * 64
            p.write_text(json.dumps(man))
            with self.assertRaises(X.ProtocolError):
                fx.select()
        finally:
            fx.close()

    def test_missing_cell_rejected(self):
        fx = Fixture()
        try:
            dev = X.PAIRS[("80b", "italian")][0]
            (fx.dev / dev / "adaptive.csv").unlink()
            with self.assertRaises(X.ProtocolError):
                fx.select()
        finally:
            fx.close()

    def test_missing_budget_row_rejected(self):
        fx = Fixture()
        try:
            dev = X.PAIRS[("30b", "italian")][0]
            p = fx.dev / dev / "adaptive.csv"
            rows = [r for r in X.read_csv(p) if not (r["variant"] == "adaptive-e64"
                                                    and r["budget"] == "1.5G")]
            write_csv(rows, p)
            with self.assertRaises(X.ProtocolError):
                fx.select()
        finally:
            fx.close()

    def test_duplicate_cell_rejected(self):
        fx = Fixture()
        try:
            dev = X.PAIRS[("30b", "coding")][0]
            p = fx.dev / dev / "adaptive.csv"
            rows = X.read_csv(p)
            write_csv(rows + rows[:1], p)
            with self.assertRaises(X.ProtocolError):
                fx.select()
        finally:
            fx.close()

    def test_unknown_candidate_rejected(self):
        fx = Fixture()
        try:
            dev = X.PAIRS[("30b", "coding")][0]
            p = fx.dev / dev / "adaptive.csv"
            rows = X.read_csv(p)
            extra = dict(rows[0], variant="adaptive-e256", policy="adaptive-e256")
            write_csv(rows + [extra], p)
            with self.assertRaises(X.ProtocolError):
                fx.select()
            write_csv(rows, p)
            doc = fx.select()
            doc["selection"]["models"]["30b"]["selected"] = "e256"
            doc["canonical_sha256"] = X.canonical_sha(doc["selection"])   # hash made valid
            with self.assertRaises(X.ProtocolError):
                X.validate_selection(doc, fx.root, TOOL)
        finally:
            fx.close()


class TestHashAndFreeze(unittest.TestCase):
    def test_canonical_hash_and_corruption(self):
        fx = Fixture()
        try:
            doc = fx.select()
            X.validate_selection(doc, fx.root, TOOL)
            doc2 = copy.deepcopy(doc)
            doc2["written_utc"] = "1999-01-01T00:00:00Z"            # timestamp is outside
            X.validate_selection(doc2, fx.root, TOOL)
            self.assertEqual(X.canonical_sha(doc["selection"]), X.canonical_sha(fx.select()["selection"]))
            doc3 = copy.deepcopy(doc)
            doc3["selection"]["models"]["30b"]["candidates"]["e64"]["G_DEV"] = 0.5
            with self.assertRaises(X.ProtocolError):
                X.validate_selection(doc3, fx.root, TOOL)
        finally:
            fx.close()

    def test_code_change_after_freeze_rejected(self):
        fx = Fixture()
        try:
            (fx.root / "argine/research/replay.py").write_text("# changed\n")
            with self.assertRaises(X.ProtocolError):
                fx.select()
        finally:
            fx.close()

    def test_protocol_change_rejected_but_outside_text_free(self):
        fx = Fixture()
        try:
            p = fx.root / X.README_REL
            p.write_text(p.read_text() + "\nresult added later\n")
            fx.select()                                              # outside markers: OK
            p.write_text(p.read_text().replace("protocol body", "protocol BODY"))
            with self.assertRaises(X.ProtocolError):
                fx.select()
        finally:
            fx.close()

    def test_crlf_does_not_change_protocol_hash(self):
        fx = Fixture()
        try:
            p = fx.root / X.README_REL
            h = X.protocol_sha(p)
            p.write_bytes(p.read_text().replace("\n", "\r\n").encode())
            self.assertEqual(X.protocol_sha(p), h)
        finally:
            fx.close()

    def test_refreeze_refused(self):
        fx = Fixture()
        try:
            with self.assertRaises(X.ProtocolError):
                X.freeze(fx.root, TOOL)
        finally:
            fx.close()


class TestClosureRule(unittest.TestCase):
    def test_model_classification(self):
        self.assertEqual(X.classify_model(0.98, 15, 0.97), "ADAPTIVE_BETTER")
        self.assertEqual(X.classify_model(1.01, 9, 0.97), "STATIC_BETTER_OR_EQUAL")
        self.assertEqual(X.classify_model(1.0, 10.5, 0.97), "STATIC_BETTER_OR_EQUAL")
        self.assertEqual(X.classify_model(0.98, 10.5, 0.97), "MIXED")      # W must exceed 10.5
        self.assertEqual(X.classify_model(1.02, 12, 0.97), "MIXED")
        self.assertEqual(X.classify_model(0.98, 15, 1.01), "MIXED")        # uniform guard

    def test_verdicts(self):
        def pm(a, b, ua=0.9, ub=0.9):
            return {"30b": {"classification": a, "G_uniform": ua},
                    "80b": {"classification": b, "G_uniform": ub}}
        self.assertEqual(X.verdict(pm("ADAPTIVE_BETTER", "ADAPTIVE_BETTER")), "SUPPORTED")
        self.assertEqual(X.verdict(pm("STATIC_BETTER_OR_EQUAL", "STATIC_BETTER_OR_EQUAL")), "FALSIFIED")
        self.assertEqual(X.verdict(pm("ADAPTIVE_BETTER", "MIXED")), "INCONCLUSIVE")
        self.assertEqual(X.verdict(pm("ADAPTIVE_BETTER", "STATIC_BETTER_OR_EQUAL")), "INCONCLUSIVE")
        self.assertEqual(X.verdict(pm("MIXED", "MIXED", 1.02, 1.03)), "FALSIFIED")   # uniform guard
        self.assertEqual(X.verdict(pm("MIXED", "MIXED", 1.02, 0.99)), "INCONCLUSIVE")

    def test_sensitivity_never_affects_verdict(self):
        rows = X.sensitivity_rows(0.995, "30b")
        self.assertEqual([r["threshold"] for r in rows], [0.005, 0.01, 0.02])
        self.assertTrue(rows[0]["adaptive_fewer_misses_by_at_least_t"])
        self.assertFalse(rows[1]["adaptive_fewer_misses_by_at_least_t"])
        self.assertTrue(all(r["affects_registry_verdict"] is False for r in rows))


def eval_fn(adapt_vs_static, adapt_vs_uniform=0.97, budget_override=None):
    def f(m, fam, b, role):
        r = adapt_vs_static[m]
        if budget_override and (m, b) in budget_override:
            r = budget_override[(m, b)]
        base = 100000
        return {"global_lru": 120000, "uniform": int(round(base * r / adapt_vs_uniform)),
                "twoq": 118000, "static_dev": base, "ref_static_eval_insample": 95000,
                "ref_belady": 60000, "adaptive_sel": int(round(base * r))}[role]
    return f


class TestAnalyze(unittest.TestCase):
    def run_case(self, avs, avu=0.97, override=None, **kw):
        fx = Fixture(eval_misses=eval_fn(avs, avu, override))
        try:
            doc = fx.select()
            fx.write_eval(doc["selection"], **kw)
            return fx.analyze(doc), fx
        except Exception:
            fx.close()
            raise

    def test_supported_and_budget_aggregates(self):
        s, fx = self.run_case({"30b": 0.995, "80b": 0.98},
                              override={("30b", "4G"): 1.01})
        try:
            self.assertEqual(s["verdict"], "SUPPORTED")       # 0.5% effect still SUPPORTED
            p = s["per_model"]["30b"]
            self.assertAlmostEqual(p["per_budget"]["4G"]["G_static"], 1.01, places=4)
            self.assertAlmostEqual(p["per_budget"]["2G"]["G_static"], 0.995, places=4)
            self.assertEqual(p["budget_reversals"], ["4G"])
            self.assertEqual(p["W"], 14)
            self.assertFalse(s["descriptive_only"]["sensitivity_affects_verdict"])
            for f in ("summary.json", "dev_selection.csv", "eval_table.csv", "decomposition.csv",
                      "subgroups.csv", "sensitivity.csv", "determinism_check.csv"):
                self.assertTrue((fx.root / "processed" / f).exists(), f)
            sub = X.read_csv(fx.root / "processed" / "subgroups.csv")
            self.assertEqual({r["subgroup"] for r in sub}, {"transition", "multi-turn"})
            self.assertTrue(all(r["n_cells"] == "3" for r in sub))
        finally:
            fx.close()

    def test_falsified(self):
        s, fx = self.run_case({"30b": 1.02, "80b": 1.0})
        fx.close()
        self.assertEqual(s["per_model"]["80b"]["W"], 10.5)     # all ties -> 21 * 0.5
        self.assertEqual(s["verdict"], "FALSIFIED")

    def test_inconclusive(self):
        s, fx = self.run_case({"30b": 0.98, "80b": 1.02})
        fx.close()
        self.assertEqual(s["verdict"], "INCONCLUSIVE")

    def test_uniform_guard_falsifies(self):
        s, fx = self.run_case({"30b": 0.98, "80b": 0.98}, avu=1.05)
        fx.close()
        self.assertEqual([s["per_model"][m]["classification"] for m in X.MODELS], ["MIXED", "MIXED"])
        self.assertEqual(s["verdict"], "FALSIFIED")

    def test_insample_static_rejected(self):
        with self.assertRaises(X.ProtocolError):
            self.run_case({"30b": 0.98, "80b": 0.98}, static_train="IN-SAMPLE / UPPER BOUND")

    def test_analyze_refuses_corrupted_selection(self):
        fx = Fixture()
        try:
            doc = fx.select()
            fx.write_eval(doc["selection"])
            doc["selection"]["models"]["80b"]["selected"] = "e32"
            with self.assertRaises(X.ProtocolError):
                fx.analyze(doc)
        finally:
            fx.close()

    def _bump_eval_adaptive(self, fx, variant, model="30b", family="coding", budget="2G"):
        ev = X.PAIRS[(model, family)][1]
        p = fx.ea / ev / "adaptive.csv"
        rows = X.read_csv(p)
        hit = 0
        for r in rows:
            if r["variant"] == variant and int(r["budget_bytes"]) == X.BUDGETS[budget]:
                r["misses"] = str(int(float(r["misses"])) + 1)
                hit += 1
        self.assertEqual(hit, 1)
        write_csv(rows, p)
        return ev

    def test_eval_internal_consistency_identical_succeeds(self):
        def dm(m, f, b, v):
            return 900 if v == "adaptive-e64" else 1000
        fx = Fixture(dev_misses=dm, eval_misses=eval_fn({"30b": 0.98, "80b": 0.98}))
        try:
            doc = fx.select()
            self.assertEqual(doc["selection"]["models"]["30b"]["selected"], "e64")
            fx.write_eval(doc["selection"])
            s = fx.analyze(doc)
            det = X.read_csv(fx.root / "processed" / "determinism_check.csv")
            internal = [r for r in det if r.get("stage") == "eval-internal"]
            self.assertEqual({r["policy"] for r in internal}, {X.UNIFORM, "adaptive-e64"})
            self.assertEqual(len(internal), 2 * 14 * 3)
            self.assertTrue(all(r["identical"] == "True" for r in internal))
            self.assertEqual(s["determinism"]["non_identical"], 0)
        finally:
            fx.close()

    def test_eval_internal_uniform_mismatch_rejected(self):
        fx = Fixture(eval_misses=eval_fn({"30b": 0.98, "80b": 0.98}))
        try:
            doc = fx.select()
            fx.write_eval(doc["selection"])
            ev = self._bump_eval_adaptive(fx, X.UNIFORM)
            with self.assertRaisesRegex(X.ProtocolError, "deterministic EVAL consistency mismatch") as cm:
                fx.analyze(doc)
            msg = str(cm.exception)
            self.assertIn(ev, msg)
            self.assertIn("2G", msg)
            self.assertIn(X.UNIFORM, msg)
            self.assertFalse((fx.root / "processed" / "summary.json").exists())
        finally:
            fx.close()

    def test_eval_internal_adaptive_e64_mismatch_rejected(self):
        def dm(m, f, b, v):
            return 900 if v == "adaptive-e64" else 1000
        fx = Fixture(dev_misses=dm, eval_misses=eval_fn({"30b": 0.98, "80b": 0.98}))
        try:
            doc = fx.select()
            fx.write_eval(doc["selection"])
            self._bump_eval_adaptive(fx, "adaptive-e64", model="80b", family="transition",
                                     budget="1.5G")
            with self.assertRaisesRegex(X.ProtocolError,
                                        "deterministic EVAL consistency mismatch for adaptive-e64"):
                fx.analyze(doc)
            self.assertFalse((fx.root / "processed" / "summary.json").exists())
        finally:
            fx.close()

    def test_eval_table_comparison_class(self):
        s, fx = self.run_case({"30b": 0.98, "80b": 0.98})
        try:
            rows = X.read_csv(fx.root / "processed" / "eval_table.csv")
            self.assertNotIn("deployable", rows[0])
            cls = {r["role"]: r["comparison_class"] for r in rows}
            self.assertEqual(cls, X.COMPARISON_CLASS)
            self.assertEqual(cls["static_dev"], "DEV-fitted workload-family-informed")
            note = {r["role"]: r["note"] for r in rows}
            self.assertIn("not a generic deployable baseline", note["static_dev"])
        finally:
            fx.close()

    def test_step5_script(self):
        def dm(m, f, b, v):
            return 900 if v == ("adaptive-e64-pin25" if m == "30b" else "adaptive-e32") else 1000
        fx = Fixture(dev_misses=dm)
        try:
            sel = fx.select()["selection"]
            txt = X.eval_adaptive_script(sel, "C:\\repo")
            calls = [l for l in txt.splitlines() if l.startswith("call argine.bat research adaptive")]
            self.assertEqual(len(calls), 14)
            self.assertEqual(sum("--epochs 64 --pinned 0.25 --no-ablate" in l for l in calls), 7)
            self.assertEqual(sum("--epochs 32 --pinned 0 --no-ablate" in l for l in calls), 7)
            self.assertTrue(all("--budgets 4G,2G,1.5G" in l and "-eval" in l for l in calls))
            self.assertIn("\r\n", txt)
            self.assertIn("REM Runs the DEV-selected configuration of each model on its 7 EVAL cells.", txt)
            self.assertIn("only the selected variant enters the primary comparison", txt)
            self.assertNotIn("ONLY", txt)
        finally:
            fx.close()


class TestCliProvenance(unittest.TestCase):
    """`compare` / `adaptive` record their inputs; flags behave as the protocol assumes."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="exp010-cli-"))
        cls.dev = cls._trace(1, "dev").save(cls.tmp / "dev.jsonl.gz")
        cls.ev = cls._trace(2, "eval").save(cls.tmp / "eval.jsonl.gz")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @staticmethod
    def _trace(seed, split):
        rng = random.Random(seed)
        steps = [T.Step(graph=g, n_tokens=1, phase=T.PHASE_DECODE,
                        routes={l: [tuple(sorted(rng.sample(range(16), 2)))] for l in range(4)})
                 for g in range(200)]
        meta = T.TraceMeta(model="synthetic-30B", n_experts=16, top_k=2, layers=list(range(4)),
                           expert_bytes={l: 1000 for l in range(4)}, split=split,
                           workload="coding@abcdef123456")
        return T.Trace(meta=meta, steps=steps)

    def test_budget_parsing(self):
        from argine.research.cli import _sizes
        self.assertEqual(_sizes("4G,2G,1.5G"), [4 << 30, 2 << 30, 3 << 29])

    def test_compare_records_train_trace(self):
        from argine.cli import main
        out = self.tmp / "cmp"
        self.assertEqual(main(["research", "compare", str(self.ev), "--budgets", "12000,20000",
                               "--train", str(self.dev), "--out", str(out)]), 0)
        man = json.loads((out / "compare_manifest.json").read_text())
        self.assertEqual(man["train_mode"], "separate")
        self.assertEqual(man["trace"]["meta"]["split"], "eval")
        self.assertEqual(man["train"]["meta"]["split"], "dev")
        self.assertEqual(man["train"]["sha256"], X.sha256_file(self.dev))
        rows = X.read_csv(out / "policy_comparison.csv")
        self.assertEqual({r["train"] for r in rows if r["policy"] == "layer-aware-static"}, {"separate"})

    def test_adaptive_selected_only(self):
        from argine.cli import main
        out = self.tmp / "ad"
        self.assertEqual(main(["research", "adaptive", str(self.ev), "--budgets", "12000,20000",
                               "--epochs", "64", "--pinned", "0.25", "--no-ablate",
                               "--out", str(out)]), 0)
        variants = {r["variant"] for r in X.read_csv(out / "adaptive.csv")}
        self.assertEqual(variants, {"per-layer-lru-uniform", "global-lru", "adaptive-e64",
                                    "adaptive-e64-pin25"})
        man = json.loads((out / "adaptive_manifest.json").read_text())
        self.assertEqual((man["epochs"], man["pinned"], man["ablate"]), ([64], [0.25], False))
        out2 = self.tmp / "ad2"
        self.assertEqual(main(["research", "adaptive", str(self.ev), "--budgets", "12000",
                               "--epochs", "128", "--pinned", "0", "--no-ablate",
                               "--out", str(out2)]), 0)
        self.assertEqual({r["variant"] for r in X.read_csv(out2 / "adaptive.csv")},
                         {"per-layer-lru-uniform", "global-lru", "adaptive-e128"})


if __name__ == "__main__":
    unittest.main()
