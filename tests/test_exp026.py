"""Tests for tools/exp026_run.py (EXP-026 protocol EXP026-P1).

No test starts a model, llama-server, a benchmark, a research trace, a replay or diskbench: every non-git
subprocess call made by the code under test is intercepted and fails the test. All data are synthetic.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import random
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("exp026_run", ROOT / "tools" / "exp026_run.py")
E = importlib.util.module_from_spec(_spec)
sys.modules["exp026_run"] = E
_spec.loader.exec_module(E)

SPAN = 8 / 3


class NoSubprocess:
    """Only `git` may run; anything else fails the test."""

    def __init__(self):
        self.calls = []
        self._real = subprocess.run

    def run(self, argv, *a, **kw):
        self.calls.append(list(argv))
        if argv and argv[0] == "git":
            return self._real(argv, *a, **kw)
        raise AssertionError(f"subprocess started: {argv}")

    def popen(self, argv, *a, **kw):
        raise AssertionError(f"Popen started: {argv}")


def _quiet(fn, *a):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        rc = fn(*a)
    return rc, out.getvalue()


def speeds(s1, beta, jitter=(0.0, 0.0, 0.0)):
    """P1 / P2 tok/s lists for a given beta (P1/P2 span 8/3): s2 = s1 * (3/8)^beta."""
    p1 = [s1 * (1 + j) for j in jitter]
    p2 = [s1 * (3 / 8) ** beta * (1 - j) for j in jitter]
    return p1, p2


def cells_for(b30=0.55, b80=0.38, jit=(0.0, 0.001, -0.001), drop=None):
    out = {}
    for m, b, s1 in (("30b", b30, 6.5), ("80b", b80, 6.9)):
        p1, p2 = speeds(s1, b, jit)
        for p, vals in (("P1", p1), ("P2", p2), ("P3", [v * 0.8 for v in p2])):
            ls = [{"outcome": "VALID", "gen_tok_s": v} for v in vals]
            if drop == (m, p):
                ls[0] = {"outcome": "STORE_RESIDENCY_ABORT", "gen_tok_s": None}
            out[(m, p)] = E.perf_cell(ls)
    return out


def mech(T, nt=0.4, er=(0.7, 0.8), M=5e8, m_over_F=None, depth=6.0, model="30b"):
    N = T * nt
    res = T - N
    er_lo, er_hi = res * er[0], res * er[1]
    F = E.MODELS[model]["footprint_bytes"]
    return {"graphs": 100, "T": T, "N": N, "Res": res, "E_r_lo": er_lo, "E_r_hi": er_hi,
            "E_nr_lo": res - er_hi, "E_nr_hi": res - er_lo, "M": M, "N_over_T": N / T,
            "m": (M / F) if m_over_F is None else m_over_F, "q_lo": er_lo / M, "q_hi": er_hi / M, "depth": depth}


# ---------------------------------------------------------------- 1-6 ----

class ArithmeticAndPlan(unittest.TestCase):
    def test_01_budget_over_F(self):
        self.assertEqual(E.MODELS["30b"]["footprint_bytes"], 1097072640)
        self.assertEqual(E.MODELS["80b"]["footprint_bytes"], 719585280)
        want = {"30b": {"P1": 4294967296, "P2": 1610612736, "P3": 1073741824},
                "80b": {"P1": 2818572288, "P2": 1056964608, "P3": 704643072}}
        for m, d in want.items():
            for p, b in d.items():
                self.assertEqual(E.BUDGETS[m][p][1], b)
                self.assertEqual(int(E.BUDGETS[m][p][0][:-1]) * (1 << 20), b)
                self.assertEqual(E.shard_count(b), 16)
        self.assertAlmostEqual(E.x_of("30b", "P1"), 4294967296 / 1097072640)
        self.assertAlmostEqual(E.x_of("80b", "P2"), 1056964608 / 719585280)
        for p in E.POINTS:
            self.assertLess(abs(E.x_of("80b", p) / E.x_of("30b", p) - 1), 1e-3)
        self.assertEqual(E.check_budgets(), [])

    def test_02_span(self):
        for m in E.MODEL_ORDER:
            self.assertAlmostEqual(E.x_of(m, "P1") / E.x_of(m, "P2"), SPAN, places=12)
        self.assertEqual(E.PRIMARY_POINTS, ("P1", "P2"))

    def test_03_model_launch_count(self):
        plan = E.build_plan()
        models = [x for x in plan if x["block"] in ("perf", "diag")]
        self.assertEqual(len(models), 36)
        self.assertEqual(sum(1 for x in models if x.get("role") == "primary"), 18)
        self.assertEqual(sum(1 for x in models if x.get("role") == "sensitivity"), 6)
        self.assertEqual(sum(1 for x in models if x["block"] == "diag"), 12)
        self.assertTrue(all(x["threads"] == 12 and x["model"] == "30b" and x["point"] in ("P1", "P2")
                            for x in models if x.get("role") == "sensitivity"))
        self.assertTrue(all(x["threads"] == 4 for x in models if x.get("role") != "sensitivity"))
        self.assertEqual(E.check_plan(plan), [])

    def test_04_diskbench_count(self):
        d = [x for x in E.build_plan() if x["block"] == "diskbench"]
        self.assertEqual(len(d), 60)
        self.assertEqual({x["readers"] for x in d}, {2, 4, 8, 16})
        self.assertEqual(sorted({x["slice_bytes"] for x in d}), [450560, 589824, 720896, 884736, 1290240])
        for x in d:
            self.assertIn("direct", x["argv"])
            self.assertNotIn("moe", x["argv"])

    def test_05_no_retry_no_replacement(self):
        self.assertEqual(E.NO_RETRY_RULES, {"no_retry": True, "no_replacement": True, "no_adaptive_pruning": True,
                                            "categorical_failures_add_launches": False, "retries_flag": 0})
        for x in E.build_plan():
            if x["block"] != "diskbench":
                self.assertEqual(E._flag(x["argv"], "--retries"), "0")
        with self.assertRaises(E.ProtocolError):
            E.perf_cell([{"outcome": "VALID", "gen_tok_s": 5.0}] * 4)
        # a tampered plan with an extra launch is rejected
        plan = E.build_plan()
        plan.append(dict(plan[-1], launch_index=61))
        self.assertTrue(E.check_plan(plan))

    def test_06_frozen_permutations(self):
        for b, n, cells in (("perf", 3, E.perf_cells()), ("diag", 2, E.diag_cells()),
                            ("diskbench", 3, E.diskbench_cells())):
            rounds = E.frozen_rounds(b)
            self.assertEqual(len(rounds), n)
            for r in rounds:
                self.assertEqual(sorted(r), sorted(c["cell"] for c in cells))
            self.assertEqual(rounds, E.frozen_rounds(b))           # deterministic
        self.assertEqual(len(E.frozen_rounds("perf")[0]), 8)
        self.assertEqual(len(E.frozen_rounds("diskbench")[0]), 20)
        tampered = E.build_plan()
        i = next(k for k, x in enumerate(tampered) if x["block"] == "perf" and x["round"] == 1)
        tampered[i] = dict(tampered[i], cell=tampered[i + 1]["cell"])
        self.assertTrue(any("not every cell exactly once" in p for p in E.check_plan(tampered)))


# ---------------------------------------------------------------- 7-16 ----

class EligibilityAndExplanandum(unittest.TestCase):
    def test_07_perf_two_of_three_not_evaluable(self):
        c = E.perf_cell([{"outcome": "VALID", "gen_tok_s": 5.0}, {"outcome": "VALID", "gen_tok_s": 5.2},
                         {"outcome": "NON_COMPLETION", "gen_tok_s": None}])
        self.assertFalse(c["evaluable"])
        self.assertEqual(c["status"], "NOT EVALUABLE")
        self.assertIn("two-value center", c["descriptive_only"]["center_kind"])
        self.assertIn("DESCRIPTIVE ONLY", c["descriptive_only"]["center_kind"])
        self.assertNotIn("center", {k for k in c if k != "descriptive_only"})

    def test_08_perf_three_of_three_evaluable(self):
        c = E.perf_cell([{"outcome": "VALID", "gen_tok_s": v} for v in (5.0, 5.4, 5.2)])
        self.assertTrue(c["evaluable"])
        self.assertEqual(c["center"], 5.2)

    def test_09_diag_one_of_two_not_evaluable(self):
        good = {"exp026": {"outcome": "VALID", "diagnostics": {"publication_quality": True, "trace_complete": True,
                                                              "violations": 0},
                           "trace_data": {"n_accounting_problems": 0, "mechanism": {"graphs": 10}}}}
        bad = json.loads(json.dumps(good))
        bad["exp026"]["outcome"] = "STORE_RESIDENCY_ABORT"
        self.assertTrue(E.diag_eligible(good))
        self.assertFalse(E.diag_eligible(bad))
        overlap = json.loads(json.dumps(good))
        overlap["exp026"]["trace_data"]["n_accounting_problems"] = 1
        self.assertFalse(E.diag_eligible(overlap))
        prim = E.explanandum(cells_for())
        T = {(m, p): [1.0, 1.0] for m in E.MODEL_ORDER for p in E.PRIMARY_POINTS}
        T[("80b", "P2")] = None                                  # 1/2 eligible -> cell missing
        self.assertEqual(E.traced_gate(prim, T)["status"], "NOT EVALUABLE")
        self.assertEqual(E.candidate_a({}, True, False)["status"], "NOT EVALUABLE")
        self.assertEqual(E.candidate_c({}, False)["status"], "NOT EVALUABLE")

    def test_10_beta(self):
        self.assertAlmostEqual(E.beta_from(6.0, 6.0 * (3 / 8) ** 0.5, 8 / 3, 1.0), 0.5)
        c = cells_for(jit=(0, 0, 0))
        r = E.explanandum(c)
        self.assertAlmostEqual(r["beta"]["30b"], 0.55, places=9)
        self.assertAlmostEqual(r["beta"]["80b"], 0.38, places=9)
        self.assertAlmostEqual(r["delta_beta"], 0.38 - 0.55, places=9)

    def test_11_envelope(self):
        lo, hi = E.envelope([6.0, 6.2, 5.8], [4.0, 4.1, 3.9], 2.0, 1.0)
        bs = [E.beta_from(a, b, 2.0, 1.0) for a in (6.0, 6.2, 5.8) for b in (4.0, 4.1, 3.9)]
        self.assertEqual((lo, hi), (min(bs), max(bs)))
        self.assertEqual(len(bs), 9)

    def test_12_reproduced(self):
        self.assertEqual(E.explanandum(cells_for(0.55, 0.38))["outcome"], "REPRODUCED")

    def test_13_reversed(self):
        self.assertEqual(E.explanandum(cells_for(0.38, 0.55))["outcome"], "REVERSED")

    def test_14_not_distinguished(self):
        self.assertEqual(E.explanandum(cells_for(0.50, 0.49, jit=(0.0, 0.05, -0.05)))["outcome"], "NOT DISTINGUISHED")

    def test_15_not_evaluable(self):
        r = E.explanandum(cells_for(drop=("80b", "P1")))
        self.assertEqual(r["outcome"], "NOT EVALUABLE")
        # P3 never defines beta: a failed P3 cell does not make the primary NOT EVALUABLE
        self.assertEqual(E.explanandum(cells_for(drop=("30b", "P3")))["outcome"], "REPRODUCED")

    def test_16_traced_gate(self):
        prim = E.explanandum(cells_for(0.55, 0.38))

        def T(b30, b80, jit=0.0):
            out = {}
            for m, b, t1 in (("30b", b30, 150.0), ("80b", b80, 145.0)):
                out[(m, "P1")] = [t1 * (1 + jit), t1 * (1 - jit)]
                out[(m, "P2")] = [t1 * SPAN ** b * (1 - jit), t1 * SPAN ** b * (1 + jit)]
            return out
        self.assertEqual(E.traced_gate(prim, T(0.6, 0.4))["status"], "PASS")
        self.assertEqual(E.traced_gate(prim, T(0.4, 0.6))["status"], "FAIL")              # opposite sign
        self.assertEqual(E.traced_gate(prim, T(0.50, 0.49, jit=0.05))["status"], "FAIL")  # overlap vs REPRODUCED
        nd = E.explanandum(cells_for(0.50, 0.49, jit=(0.0, 0.05, -0.05)))
        self.assertEqual(E.traced_gate(nd, T(0.50, 0.49, jit=0.05))["status"], "PASS")
        self.assertEqual(E.traced_gate({"outcome": "NOT EVALUABLE"}, T(0.6, 0.4))["status"], "NOT EVALUABLE")


# ---------------------------------------------------------------- 17-21 ----

class RoutingGeometryAccounting(unittest.TestCase):
    def test_17_ews_fraction_distinct_from_ews_over_F(self):
        c = Counter({(0, e): 10 for e in range(50)})            # layer 0: 3,059,712 B experts on the 30B
        r = E.routing_metrics(c, "30b")
        self.assertEqual(r["ews90_experts"], 45)
        self.assertAlmostEqual(r["historical_ews_fraction"], 45 / 6144)
        store = 128 * (24 * 3059712 + 24 * 2654208)
        self.assertAlmostEqual(r["byte_weighted_variant_ews_bytes_over_store_bytes"], 45 * 3059712 / store)
        self.assertNotAlmostEqual(r["historical_ews_fraction"], r["byte_weighted_variant_ews_bytes_over_store_bytes"])
        self.assertAlmostEqual(r["pressure_normalized_ews_bytes_over_F"], 45 * 3059712 / 1097072640)
        self.assertIn("NOT equivalent", E.ROUTING_TAXONOMY["byte_weighted_variant"])
        self.assertIn("not a decision endpoint", E.ROUTING_TAXONOMY["pressure_normalized_predictor"])

    def test_18_exact_slice_sizes(self):
        self.assertEqual(E.slice_sizes("30b"), [884736, 1290240])
        self.assertEqual(E.slice_sizes("80b"), [450560, 589824, 720896])
        self.assertEqual(E.reads_per_token("30b"), {884736: 960, 1290240: 192})
        self.assertEqual(E.reads_per_token("80b"), {450560: 960, 589824: 450, 720896: 30})
        leb30, leb80 = E.layer_expert_bytes("30b"), E.layer_expert_bytes("80b")
        self.assertEqual(Counter(leb30.values()), Counter({3059712: 24, 2654208: 24}))
        self.assertEqual({l for l, b in leb80.items() if b == 1622016}, {0, 1, 2})
        self.assertEqual(Counter(leb80.values()), Counter({1622016: 3, 1490944: 45}))
        self.assertEqual(max(E.whole_layer_bytes("30b").values()), 391643136)
        self.assertEqual(sorted(set(E.whole_layer_bytes("80b").values())), [763363328, 830472192])
        for m in E.MODEL_ORDER:
            self.assertEqual(E.footprint(m), E.MODELS[m]["footprint_bytes"])
        self.assertEqual(E.check_geometry(), [])

    def _graphs(self, nodes=None):
        routes = [(0, 10, 0, 5), (1, 100, 0, 1), (2, 300, 0, 1), (3, 500, 0, 1), (4, 700, 0, 1)]
        requests = [(0, 600, "natural-language"), (650, 900, "natural-language")]
        nodes = nodes or [(1, 110, 150), (1, 160, 200), (2, 310, 330), (3, 510, 520)]
        reads = [(115, 140, True, 1000), (120, 170, False, 2000), (312, 318, True, 500)]
        return E.graph_accounting(routes, requests, nodes, reads)

    def test_19_overlapping_reads_use_union(self):
        rows, problems = self._graphs()
        self.assertEqual(problems, [])
        self.assertEqual([r["graph"] for r in rows], [1, 2])     # g0 prefill, g3 next in another request
        g1 = rows[0]
        self.assertEqual(g1["lat_sum"], 75)                       # 25 + 50
        self.assertEqual(g1["read_union"], 55)                    # union [115, 170]
        self.assertEqual(g1["E_r_hi"], 45)                        # |[115,170] ∩ ([110,150] ∪ [160,200])|
        self.assertNotEqual(g1["E_r_hi"], g1["lat_sum"])
        self.assertEqual(E.union([[1, 5], [3, 9], [12, 13]]), [[1, 9], [12, 13]])
        self.assertEqual(E.intersection_length([[0, 10]], [[5, 20]]), 5)

    def test_20_E_r_bounds(self):
        g1 = self._graphs()[0][0]
        self.assertEqual((g1["T"], g1["Res"], g1["N"]), (200, 80, 120))
        self.assertEqual(g1["E_r_lo"], 25)                        # exact read only: [115, 140]
        self.assertEqual(g1["E_r_hi"], 45)
        self.assertLessEqual(g1["E_r_lo"], g1["E_r_hi"])
        self.assertEqual(g1["M"], 3000)
        self.assertIn("NOT", E.ACCOUNTING["kind"])

    def test_21_E_nr_bounds(self):
        rows = self._graphs()[0]
        g1 = rows[0]
        self.assertEqual(g1["E_nr_lo"], 80 - 45)
        self.assertEqual(g1["E_nr_hi"], 80 - 25)
        for r in rows:
            self.assertEqual(r["T"], r["N"] + r["Res"])           # exact part
            self.assertEqual(r["E_nr_lo"] + r["E_r_hi"], r["Res"])
            self.assertEqual(r["E_nr_hi"] + r["E_r_lo"], r["Res"])
        _rows, problems = self._graphs(nodes=[(1, 110, 150), (1, 140, 160), (2, 310, 330)])
        self.assertTrue(any("overlap" in p for p in problems))
        lm = E.launch_mechanism(rows, 1000)
        self.assertAlmostEqual(lm["N_over_T"], (120 + 180) / 400)
        self.assertAlmostEqual(lm["q_lo"], (25 + 6) / 3500)
        self.assertAlmostEqual(lm["q_hi"], (45 + 6) / 3500)
        self.assertAlmostEqual(lm["m"], 3500 / 2 / 1000)


# ---------------------------------------------------------------- 22-27 ----

class Candidates(unittest.TestCase):
    @staticmethod
    def m_values(e30, e80, jit=0.0):
        out = {}
        for m, e in (("30b", e30), ("80b", e80)):
            p1 = [0.2 * (1 + jit), 0.2 * (1 - jit)]
            out[m] = {"P1": p1, "P2": [v * (1 / SPAN) ** e for v in p1]}  # m_P2 = m_P1 * (x2/x1)^e
        return out

    def test_22_candidate_a(self):
        self.assertEqual(E.candidate_a(self.m_values(-1.0, -0.4), True, True)["status"], "SUPPORTED")
        self.assertIn("consistent with routing concentration",
                      E.candidate_a(self.m_values(-1.0, -0.4), True, True)["wording"])
        self.assertEqual(E.candidate_a(self.m_values(-1.0, -0.4), False, True)["status"], "UNRESOLVED")
        self.assertEqual(E.candidate_a(self.m_values(-0.4, -1.0), True, True)["status"], "FALSIFIED AS STATED")
        self.assertEqual(E.candidate_a(self.m_values(-0.8, -0.79, jit=0.2), True, True)["status"], "WEAKENED")
        mixed = self.m_values(-1.0, -0.4)
        mixed["80b"]["P2"] = [mixed["80b"]["P1"][0] * 1.3, mixed["80b"]["P1"][1] * 0.7]
        self.assertEqual(E.candidate_a(mixed, True, True)["status"], "UNRESOLVED")
        self.assertEqual(E.candidate_a(self.m_values(-1.0, -0.4), True, False)["status"], "NOT EVALUABLE")

    def test_23_candidate_b_bound_separation(self):
        fav = {"status": "FAVOURS"}
        dis = {"status": "DISFAVOURS"}
        ind = {"status": "INDETERMINATE"}
        low = {"80b": {p: [(1.0, 2.0), (1.1, 2.1)] for p in E.PRIMARY_POINTS},
               "30b": {p: [(2.5, 3.0), (2.6, 3.1)] for p in E.PRIMARY_POINTS}}
        high = {"80b": low["30b"], "30b": low["80b"]}
        touching = {"80b": {p: [(1.0, 2.6), (1.1, 2.1)] for p in E.PRIMARY_POINTS}, "30b": low["30b"]}
        d = [6.0, 7.0, 8.0, 9.0, 6.0, 7.0, 8.0, 9.0]
        self.assertEqual(E.candidate_b(fav, low, d, True)["status"], "SUPPORTED")
        self.assertEqual(E.candidate_b(dis, high, d, True)["status"], "FALSIFIED AS STATED")
        self.assertEqual(E.candidate_b(fav, touching, d, True)["status"], "WEAKENED")   # bounds overlap
        self.assertEqual(E.candidate_b(ind, low, d, True)["status"], "WEAKENED")
        self.assertEqual(E.candidate_b(fav, high, d, True)["status"], "UNRESOLVED")
        self.assertEqual(E.candidate_b(fav, low, d[:-1] + [17.0], True)["status"], "UNRESOLVED")
        self.assertEqual(E.candidate_b({"status": "NOT EVALUABLE"}, low, d, True)["status"], "NOT EVALUABLE")
        bm = {"30b": {2: [100, 101, 102], 4: [200, 201, 202], 8: [300, 301, 302]},
              "80b": {4: [150, 151, 152], 8: [250, 251, 252], 16: [350, 351, 352]}}
        self.assertEqual(E.device_comparison(bm)["status"], "FAVOURS")
        bm["80b"][16] = None
        self.assertEqual(E.device_comparison(bm)["status"], "NOT EVALUABLE")
        self.assertAlmostEqual(E.b_mix({884736: 1000.0, 1290240: 1000.0}, "30b"), 1000.0)
        b = E.b_mix({884736: 1000.0, 1290240: 2000.0}, "30b")
        w1, w2 = 960 * 884736, 192 * 1290240
        self.assertAlmostEqual(b, (w1 + w2) / (w1 / 1000 + w2 / 2000))

    @staticmethod
    def c_data(nr80, r80, share30=(0.5, 0.6)):
        return {"80b": {p: [{"E_nr_hi": nr80[p], "E_r_lo": r80, "E_nr_lo": nr80[p] / 2, "Res": 100.0}] * 2
                        for p in E.PRIMARY_POINTS},
                "30b": {p: [{"E_nr_hi": share30[1] * 100, "E_r_lo": 30.0, "E_nr_lo": share30[0] * 100,
                             "Res": 100.0}] * 2 for p in E.PRIMARY_POINTS}}

    def test_24_candidate_c_never_supported(self):
        rng = random.Random(7)
        for _ in range(500):
            d = {m: {p: [{"E_nr_hi": rng.uniform(0, 100), "E_r_lo": rng.uniform(0, 100),
                          "E_nr_lo": rng.uniform(0, 50), "Res": rng.uniform(100, 200)} for _ in range(2)]
                     for p in E.PRIMARY_POINTS} for m in E.MODEL_ORDER}
            self.assertNotEqual(E.candidate_c(d, True)["status"], "SUPPORTED")
        self.assertIn("NOT REACHABLE", E.CANDIDATE_RULES["c"]["SUPPORTED"])

    def test_25_c_unresolved_blocks_lifecycle_falsified(self):
        st = {"a": "FALSIFIED AS STATED", "b": "FALSIFIED AS STATED", "c": "UNRESOLVED", "d": "FALSIFIED AS STATED"}
        self.assertEqual(E.lifecycle("REPRODUCED", "PASS", st), "INCONCLUSIVE")
        for s in ("WEAKENED", "NOT EVALUABLE"):
            self.assertEqual(E.lifecycle("REPRODUCED", "PASS", dict(st, c=s)), "INCONCLUSIVE")

    def test_26_c_can_be_falsified_as_stated(self):
        d = self.c_data({"P1": 10.0, "P2": 12.0}, 60.0)
        self.assertEqual(E.candidate_c(d, True)["status"], "FALSIFIED AS STATED")
        one = self.c_data({"P1": 10.0, "P2": 70.0}, 60.0, share30=(0.1, 0.2))
        self.assertEqual(E.candidate_c(one, True)["status"], "WEAKENED")
        neither = self.c_data({"P1": 70.0, "P2": 70.0}, 60.0, share30=(0.1, 0.2))
        self.assertEqual(E.candidate_c(neither, True)["status"], "UNRESOLVED")
        notmore = self.c_data({"P1": 40.0, "P2": 40.0}, 30.0, share30=(0.5, 0.6))   # 0.40 <= 0.50 at both
        self.assertEqual(E.candidate_c(notmore, True)["status"], "WEAKENED")
        st = {k: "FALSIFIED AS STATED" for k in E.CANDIDATE_KEYS}
        self.assertEqual(E.lifecycle("REPRODUCED", "PASS", st), "FALSIFIED")

    def test_27_candidate_d_non_residency(self):
        hi = {"80b": {p: [0.6, 0.62] for p in E.PRIMARY_POINTS}, "30b": {p: [0.3, 0.32] for p in E.PRIMARY_POINTS}}
        lo = {"80b": hi["30b"], "30b": hi["80b"]}
        r = E.candidate_d(hi, True, True)
        self.assertEqual(r["status"], "SUPPORTED")
        self.assertIn("non-residency", r["quantity"])
        self.assertNotIn("pure compute", r["quantity"])
        self.assertEqual(E.candidate_d(hi, False, True)["status"], "WEAKENED")
        self.assertEqual(E.candidate_d(lo, True, True)["status"], "FALSIFIED AS STATED")
        self.assertEqual(E.candidate_d(hi, True, False)["status"], "NOT EVALUABLE")
        self.assertIn("never 'pure compute'", E.CANDIDATE_RULES["d"]["endpoint"])
        self.assertIn("NOT pure compute", E.ACCOUNTING["N_g"])

    def test_counterfactual_deterministic_and_both_bounds(self):
        def term(T, nt, er, M):
            N = T * nt
            res = T - N
            return {"N": N, "Res": res, "E_r_lo": res * er[0], "E_r_hi": res * er[1], "M": M}
        terms = {("30b", "P1"): term(150, 0.4, (0.7, 0.9), 2e8), ("30b", "P2"): term(250, 0.25, (0.7, 0.9), 6e8),
                 ("80b", "P1"): term(145, 0.5, (0.7, 0.9), 1e8), ("80b", "P2"): term(210, 0.36, (0.7, 0.9), 4e8)}
        a = E.counterfactual(terms, "M")
        self.assertEqual(a, E.counterfactual(terms, "M"))
        self.assertEqual(set(a["versions"]), {"lo", "hi"})
        self.assertEqual(a["toward_both"], all(v["toward"] for v in a["versions"].values()))
        n = E.counterfactual(terms, "N")
        self.assertEqual(n["versions"]["lo"]["beta80_cf"], n["versions"]["hi"]["beta80_cf"])   # N is exact
        with self.assertRaises(E.ProtocolError):
            E.counterfactual(terms, "C")
        self.assertIn("DESCRIPTIVE ACCOUNTING ONLY", E.COUNTERFACTUAL_RULE)


# ---------------------------------------------------------------- 28-34 ----

class LifecycleAndFreeze(unittest.TestCase):
    def test_28_lifecycle_supported(self):
        st = {"a": "SUPPORTED", "b": "UNRESOLVED", "c": "UNRESOLVED", "d": "FALSIFIED AS STATED"}
        self.assertEqual(E.lifecycle("REPRODUCED", "PASS", st), "SUPPORTED")
        self.assertEqual(E.lifecycle("REPRODUCED", "FAIL", st), "INCONCLUSIVE")

    def test_29_lifecycle_falsified_only_with_all_four(self):
        st = {k: "FALSIFIED AS STATED" for k in E.CANDIDATE_KEYS}
        self.assertEqual(E.lifecycle("REPRODUCED", "PASS", st), "FALSIFIED")
        for k in E.CANDIDATE_KEYS:
            self.assertEqual(E.lifecycle("REPRODUCED", "PASS", dict(st, **{k: "WEAKENED"})), "INCONCLUSIVE")

    def test_30_lifecycle_inconclusive_fallback(self):
        sup = {"a": "SUPPORTED", "b": "SUPPORTED", "c": "UNRESOLVED", "d": "SUPPORTED"}
        for o in ("NOT EVALUABLE", "NOT DISTINGUISHED", "REVERSED"):
            self.assertEqual(E.lifecycle(o, "PASS", sup), "INCONCLUSIVE")
        self.assertEqual(E.lifecycle("REPRODUCED", "NOT EVALUABLE", sup), "INCONCLUSIVE")
        self.assertEqual(E.lifecycle("REPRODUCED", "PASS", {k: "WEAKENED" for k in E.CANDIDATE_KEYS}), "INCONCLUSIVE")

    def test_31_not_reproduced_wording(self):
        t = E.not_reproduced_text("NOT DISTINGUISHED")
        self.assertIn("was not distinguishable from the 30B's", t)
        self.assertIn("does not retrospectively falsify EXP-006 or alter F-004", t)
        self.assertIn("The candidate-mechanism hypothesis is INCONCLUSIVE", t)
        self.assertIn("steeper than", E.not_reproduced_text("REVERSED"))
        self.assertIsNone(E.not_reproduced_text("REPRODUCED"))
        self.assertIsNone(E.not_reproduced_text("NOT EVALUABLE"))
        readme = E.protocol_text()
        self.assertIn(E.NOT_REPRODUCED_TEMPLATE.format(relation="[not distinguishable from | steeper than]"), readme)

    def test_32_rq009_mapping(self):
        allr = {"a": "SUPPORTED", "b": "WEAKENED", "c": "FALSIFIED AS STATED", "d": "WEAKENED"}
        self.assertEqual(E.rq009("REPRODUCED", allr)["status"], "ANSWERED")
        self.assertEqual(E.rq009("NOT DISTINGUISHED", allr)["status"], "ANSWERED")
        part = dict(allr, c="UNRESOLVED")
        r = E.rq009("REPRODUCED", part)
        self.assertEqual((r["status"], r["kind"], r["remaining"]), ("PARTIALLY ANSWERED", "narrowed", ["c"]))
        none = {k: "NOT EVALUABLE" for k in E.CANDIDATE_KEYS}
        self.assertEqual(E.rq009("REPRODUCED", none)["kind"], "unchanged, data note")
        self.assertEqual(E.rq009("NOT EVALUABLE", allr)["status"], "PARTIALLY ANSWERED")
        note = E.rq009("REVERSED", part)["note"]
        self.assertIn("consistent with a normalization/region difference", note)
        self.assertNotIn("caused by", note)

    def test_33_check_detects_tampered_hashes(self):
        now = E.freeze_hashes()
        self.assertEqual(set(now), {"protocol_sha256", "spec_sha256", "config_protocol_sha256", "tool_sha256",
                                    "tests_sha256", "exp022_dependency_sha256"})
        with tempfile.TemporaryDirectory() as d:
            fz = Path(d) / "protocol_freeze.json"
            fz.write_text(json.dumps(now), encoding="utf-8")
            self.assertEqual(E.freeze_status(fz, now)[0], True)
            for k in now:
                bad = dict(now, **{k: "0" * 64})
                fz.write_text(json.dumps(bad), encoding="utf-8")
                ok, why = E.freeze_status(fz, now)
                self.assertFalse(ok)
                self.assertIn(k, why)
            # the protocol hash follows the README block; the config hash follows the protocol object
            rd = Path(d) / "README.md"
            t = E.README.read_text(encoding="utf-8")
            rd.write_text(t.replace("No retries. No replacements.", "One retry. No replacements."), encoding="utf-8")
            self.assertNotEqual(E.protocol_text(rd), E.protocol_text())
            cf = Path(d) / "config.json"
            c = json.loads(E.CONFIG.read_text(encoding="utf-8"))
            cf.write_text(json.dumps(c), encoding="utf-8")
            self.assertEqual(E.config_protocol_sha256(cf), E.config_protocol_sha256())
            c["protocol"]["launches"] = 37
            cf.write_text(json.dumps(c), encoding="utf-8")
            self.assertNotEqual(E.config_protocol_sha256(cf), E.config_protocol_sha256())
            c["protocol"]["launches"] = 36
            c["protocol"]["measurements_made"] = True                    # closure flips it: not part of the hash
            cf.write_text(json.dumps(c), encoding="utf-8")
            self.assertEqual(E.config_protocol_sha256(cf), E.config_protocol_sha256())
            self.assertEqual(E.check_docs(config=cf), [])
        self.assertEqual(E.X22._lf_sha(E.EXP022_TOOL), E.EXP022_TOOL_SHA256)

    def test_34_no_outputs_required_before_measurement(self):
        # Lifecycle-aware (amendment A1, 2026-10-16, test-only): the live repository is checked against the
        # contract of its current state; the synthetic checks below are unchanged.
        self.assertEqual(E.check_outputs(), [])
        if not E.measurements_made():
            self.assertFalse((E.EXP / "results.csv").exists())
        else:
            self.assertTrue((E.EXP / "results.csv").exists())
            self.assertTrue((E.PROCESSED / "summary.json").exists())
            for p in E.RAW.iterdir():
                if p.name != ".gitkeep":
                    self.assertTrue(E.harness_session_dir(p), p.name)
        with tempfile.TemporaryDirectory() as d:
            exp = Path(d)
            for sub in ("raw", "processed", "figures"):
                (exp / sub).mkdir()
                (exp / sub / ".gitkeep").write_text("")
            cfg = exp / "config.json"
            cfg.write_text(json.dumps({"protocol": {"measurements_made": False}}), encoding="utf-8")
            self.assertEqual(E.check_outputs(exp, cfg), [])
            (exp / "raw" / "stray.json").write_text("{}")
            self.assertTrue(E.check_outputs(exp, cfg))
            (exp / "raw" / "stray.json").unlink()
            (exp / "raw" / "perf").mkdir()
            (exp / "raw" / "perf" / "session.json").write_text(json.dumps(
                {"experiment": E.EXP_ID, "protocol_version": E.PROTOCOL_VERSION, "block": "perf"}))
            self.assertEqual(E.check_outputs(exp, cfg), [])
            with mock.patch.object(E, "RAW", exp / "raw"), mock.patch.object(E, "EXP", exp), \
                    mock.patch.object(E, "PROCESSED", exp / "processed"):
                (exp / "raw" / "perf" / "session.json").unlink()
                (exp / "raw" / "perf").rmdir()
                rc, out = _quiet(E.cmd_analyze)
                self.assertEqual(rc, 1)
                self.assertFalse((exp / "results.csv").exists())
                self.assertEqual(sorted(p.name for p in (exp / "processed").iterdir()), [".gitkeep"])


class NoSideEffects(unittest.TestCase):
    def test_check_and_dry_run_start_nothing(self):
        rec = NoSubprocess()
        with mock.patch("subprocess.run", rec.run), mock.patch("subprocess.Popen", rec.popen):
            self.assertEqual(E.static_problems(), [])
            rc, out = _quiet(E.print_dry_run)
        self.assertEqual(rc, 0)
        self.assertIn("no subprocess started", out)
        self.assertTrue(all(c[0] == "git" for c in rec.calls))

    def test_historical_text_and_candidate_d(self):
        t = E.README.read_text(encoding="utf-8")
        self.assertIn(E.HISTORICAL_HYPOTHESIS, t)
        self.assertNotIn("(d)", E.HISTORICAL_HYPOTHESIS)
        self.assertIn(E.CANDIDATE_D_TEXT, (E.ROOT / "OPEN_QUESTIONS.md").read_text(encoding="utf-8"))
        self.assertIn("RQ-009", E.CANDIDATES["d"]["source"])
        self.assertEqual(E.check_docs(), [])

    def test_diskbench_classification(self):
        x = next(i for i in E.build_plan() if i["block"] == "diskbench")
        rec = {"event": "diskbench", "metrics": {"file": E.DISKBENCH_FILE, "risultati": [
            {"kind": "direct", "slice": x["slice_bytes"], "threads": x["readers"], "mb_s": 1500.0, "error": None}]}}
        self.assertEqual(E.classify_diskbench(rec, 1, x, 0, False)["outcome"], "VALID")
        self.assertEqual(E.classify_diskbench(None, 0, x, 1, False)["outcome"], "NON_COMPLETION")
        self.assertEqual(E.classify_diskbench(rec, 2, x, 0, False)["outcome"], "INVALID")
        wrong = json.loads(json.dumps(rec))
        wrong["metrics"]["risultati"][0]["threads"] = 32
        self.assertEqual(E.classify_diskbench(wrong, 1, x, 0, False)["outcome"], "INVALID")


class AnalyzeEndToEnd(unittest.TestCase):
    def test_synthetic_pipeline(self):
        recs = {"diskbench": [], "perf": [], "diag": []}
        for x in E.build_plan():
            if x["block"] == "diskbench":
                b = 1000.0 + 50 * x["readers"] + (300 if x["slice_bytes"] < 800000 else 0) + x["round"]
                recs["diskbench"].append({"exp026": {**x, "outcome": "VALID", "mb_s": b}})
            elif x["block"] == "perf":
                beta = {"30b": 0.55, "80b": 0.38}[x["model"]] if x["threads"] == 4 else 0.5
                s = 6.5 * (x["budget_over_F"] / E.x_of(x["model"], "P1")) ** beta * (1 + 0.001 * x["round"])
                recs["perf"].append({"exp026": {**x, "outcome": "VALID", "gen_tok_s": s}})
            else:
                recs["diag"].append({"exp026": {**x, "outcome": "VALID", "run_dir": "raw/diag/x",
                                                "diagnostics": {"publication_quality": True, "trace_complete": True,
                                                                "violations": 0}}})

        def loader(r):
            e = r["exp026"]
            beta = {"30b": 0.6, "80b": 0.4}[e["model"]]
            T = 150.0 * (E.x_of(e["model"], "P1") / e["budget_over_F"]) ** beta * (1 + 0.001 * e["round"])
            nt = {"30b": 0.3, "80b": 0.5}[e["model"]]
            m = 0.1 * (E.x_of(e["model"], "P1") / e["budget_over_F"]) ** {"30b": 1.0, "80b": 0.5}[e["model"]]
            return {"mechanism": mech(T, nt=nt, M=m * E.MODELS[e["model"]]["footprint_bytes"], m_over_F=m,
                                      model=e["model"]),
                    "n_accounting_problems": 0, "routing": None}
        text, s = E.analyze(recs, {b: {"status": "complete"} for b in E.BLOCK_ORDER}, trace_loader=loader)
        self.assertEqual(s["explanandum"]["outcome"], "REPRODUCED")
        self.assertEqual(s["gate"]["status"], "PASS")
        self.assertEqual(s["candidates"]["c"]["status"] != "SUPPORTED", True)
        self.assertIn(s["lifecycle_status"], ("SUPPORTED", "FALSIFIED", "INCONCLUSIVE"))
        self.assertEqual(len(text.strip().splitlines()), 1 + 60 + 24 + 12)
        self.assertIn("sensitivity", s)
        # one aborted primary launch -> NOT EVALUABLE -> INCONCLUSIVE, nothing replaced
        recs["perf"][next(i for i, r in enumerate(recs["perf"]) if r["exp026"]["cell"] == "80b-t4-P1")]["exp026"]["outcome"] = "STORE_RESIDENCY_ABORT"
        _t, s2 = E.analyze(recs, {}, trace_loader=loader)
        self.assertEqual(s2["explanandum"]["outcome"], "NOT EVALUABLE")
        self.assertEqual(s2["lifecycle_status"], "INCONCLUSIVE")
        self.assertTrue(all(v["status"] == "NOT EVALUABLE" for v in s2["candidates"].values()))


if __name__ == "__main__":
    unittest.main()
