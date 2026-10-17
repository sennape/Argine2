"""EXP-008 analysis: the frozen statistics and decision rule (no model, no data files)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import exp008_analysis as a  # noqa: E402


def rows(model="30b", f_p1=lambda w, b: (w + 1) * b, f_t=lambda w, b: (w + 1) * b):
    out = []
    for wi, w in enumerate(a.WORKLOADS):
        for bi, (bud, _) in enumerate(a.BUDGETS.items()):
            bb = 3 - bi
            out.append({"model": model, "workload": w, "budget": bud, "P0": float(bb),
                        "P1": float(f_p1(wi, bb)), "P2": float(f_p1(wi, bb)) + 0.5,
                        "gen_tok_s": float(f_t(wi, bb)), "expert_level_hit_rate": float(f_t(wi, bb)) / 100})
    return out


class TestStats(unittest.TestCase):
    def test_spearman_with_ties(self):
        self.assertAlmostEqual(a.spearman([1, 2, 3], [10, 20, 30]), 1.0)
        self.assertAlmostEqual(a.spearman([1, 2, 3], [30, 20, 10]), -1.0)
        self.assertIsNone(a.spearman([1, 1, 1], [1, 2, 3]))
        self.assertEqual(a.ranks([5, 1, 5]), [2.5, 1.0, 2.5])

    def test_pairs_p0_is_all_ties(self):
        r = rows()
        pc = a.pair_counts(r, "P0", "gen_tok_s")
        self.assertEqual(pc["n"], 63)
        self.assertEqual(pc["ties"], 63)
        self.assertEqual(pc["accuracy"], 0.5)
        self.assertEqual(a.pair_counts(r, "P1", "gen_tok_s")["accuracy"], 1.0)

    def test_lowo_exact_power_law_has_zero_error(self):
        r = rows(f_p1=lambda w, b: (w + 1) * b, f_t=lambda w, b: ((w + 1) * b) ** 0.5)
        self.assertAlmostEqual(a.lowo_errors(r, "P1", "gen_tok_s")["mae"], 0.0, places=9)
        self.assertGreater(a.lowo_errors(r, "P0", "gen_tok_s")["mae"], 0.0)

    def test_decision_supported_when_ews_predicts_workloads(self):
        res = a.analyse_split(rows("30b") + rows("80b"), a.PREDICTORS)
        self.assertEqual(res["gen_tok_s"]["decision"]["verdict"], "SUPPORTED")

    def test_decision_falsified_when_only_budget_matters(self):
        flat = lambda w, b: b * 1.0  # noqa: E731  target ignores workloads
        r = [dict(x, P1=x["P0"] * (1 + 0.001 * i), P2=x["P0"] * (1 - 0.001 * i))
             for m in ("30b", "80b") for i, x in enumerate(rows(m, f_t=flat))]
        res = a.analyse_split(r, a.PREDICTORS)
        self.assertEqual(res["gen_tok_s"]["decision"]["verdict"], "FALSIFIED")

    def test_decision_inconclusive_when_models_disagree(self):
        good = rows("30b")
        bad = [dict(x, P1=x["P0"], P2=x["P0"]) for x in rows("80b", f_t=lambda w, b: (7 - w) * b)]
        res = a.analyse_split(good + bad, a.PREDICTORS)
        self.assertNotEqual(res["gen_tok_s"]["decision"]["verdict"], "SUPPORTED")


if __name__ == "__main__":
    unittest.main()
