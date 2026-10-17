"""Tests for the temporal transition analysis (tools/transition_analysis.py)
and the replay fields it relies on."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from argine.research.controller import AdaptiveResidencyController  # noqa: E402
from argine.research.costmodel import CostModel  # noqa: E402
from argine.research.replay import adaptation_metrics, replay  # noqa: E402
from argine.research.trace import PHASE_DECODE, PHASE_PREFILL, Step, Trace, TraceMeta  # noqa: E402

import transition_analysis as TA  # noqa: E402


def seg_trace(seg_len=40, layers=4, n_exp=16, k=2, size=1000, sizes=None):
    """Warm-up, four segments with disjoint hot experts, sentinel."""
    steps = []
    g = 0

    def add(seg, experts_fn, n, phase=PHASE_DECODE):
        nonlocal g
        for t in range(n):
            routes = {l: [experts_fn(l, t)] for l in range(layers)}
            steps.append(Step(g, 1, routes, phase, seg))
            g += 1
    add("warmup", lambda l, t: [0, 1], 3)
    for s, seg in enumerate(("a", "b", "c", "d")):
        add(seg, lambda l, t, s=s: [4 * s + (t % 3), 4 * s + 3] if l % 2 == 0 else [s, (s + t) % n_exp], seg_len)
    add("flush-sentinel", lambda l, t: [0, 1], 1)
    meta = TraceMeta(model="synthetic", n_experts=n_exp, top_k=k, layers=list(range(layers)),
                     expert_bytes=sizes or {l: size for l in range(layers)})
    return Trace(meta, steps)


class TestDistance(unittest.TestCase):
    def test_distance_is_half_l1_over_budget(self):
        a = {0: 300, 1: 100}
        b = {0: 100, 1: 300}
        self.assertAlmostEqual(TA.dist(a, b, 400), 0.5)
        self.assertEqual(TA.dist(a, a, 400), 0.0)
        self.assertAlmostEqual(TA.dist(a, b, 400), TA.dist(b, a, 400))

    def test_distance_bounded_by_one(self):
        self.assertLessEqual(TA.dist({0: 400, 1: 0}, {0: 0, 1: 400}, 400), 1.0)


class TestCountedIndex(unittest.TestCase):
    def test_boundaries_are_first_counted_steps_of_new_segments(self):
        t = seg_trace()
        steps, segs, bounds = TA.counted_index(t)
        self.assertEqual(len(steps), 160)                  # warm-up and sentinel excluded
        self.assertEqual(bounds, [40, 80, 120])
        self.assertEqual([segs[b] for b in bounds], ["b", "c", "d"])
        self.assertEqual(TA.segment_ranges(segs)[0], ("a", 0, 40))


class TestReplayPerStepFields(unittest.TestCase):
    def test_per_step_bytes_and_accesses_sum_to_totals(self):
        t = seg_trace(sizes={0: 1000, 1: 1500, 2: 1000, 3: 1500})
        r = replay(t, AdaptiveResidencyController(epoch=8), 30000, phase=PHASE_DECODE)
        self.assertEqual(sum(r.per_step_bytes), r.bytes_missed)
        self.assertEqual(sum(r.per_step_accesses), r.accesses)
        self.assertEqual(sum(r.per_step_misses), r.misses)
        self.assertEqual(len(r.per_step_bytes), r.steps)


class TestScheduled(unittest.TestCase):
    def test_allocation_switches_before_the_scheduled_step(self):
        t = seg_trace()
        a1 = {l: 6000 if l == 0 else 2000 for l in range(4)}
        a2 = {l: 6000 if l == 1 else 2000 for l in range(4)}
        switch_at = 3 + 40              # first trace step of segment "b"
        run = TA.run_policy(t, "sched", lambda: TA.Scheduled({switch_at: a2}, a1, CostModel(), "s"),
                            12000, CostModel())
        self.assertEqual(TA.alloc_at(run, switch_at - 1), a1)
        self.assertEqual(TA.alloc_at(run, switch_at), a2)


class TestAdaptationV1Floor(unittest.TestCase):
    def test_v1_metric_cannot_report_less_than_its_window(self):
        """The historical adaptation_steps starts its search at k = window, so a
        segment with NO transient at all reports exactly 16: a floor, not a
        convergence time (EXP-007 transition analysis)."""
        flat = [100] * 300 + [120] * 300
        res = adaptation_metrics(flat, [300])
        self.assertEqual(res[0]["adaptation_steps"], 16)

    def test_v1_relative_band_hides_small_transients_at_high_miss_rates(self):
        # transient of +15 misses/step for 32 steps on a steady 200/step: inside 10% band
        seg = [215] * 32 + [200] * 268
        res = adaptation_metrics([200] * 300 + seg, [300])
        self.assertEqual(res[0]["adaptation_steps"], 16)
        self.assertGreater(res[0]["transient_excess_misses"], 0)


class TestCostWeightingFactor(unittest.TestCase):
    def test_cost_per_byte_is_the_same_for_every_expert_size(self):
        """With zero per-read overhead the stall of a node is bytes / bandwidth(qd):
        cost_per_miss(L) / bytes(L) does not depend on L, so cost weighting is a
        common factor (up to floating-point rounding)."""
        c = CostModel()
        self.assertEqual(c.per_read_overhead_us, 0.0)
        for m in range(1, 9):
            f = [c.node_stall_s(m, m * b) / m / b for b in (2654208, 3059712)]
            self.assertAlmostEqual(f[0] / f[1], 1.0, places=12)


class TestBlockBootstrap(unittest.TestCase):
    def test_ci_contains_mean_of_constant_series(self):
        lo, hi = TA.block_bootstrap([2.0] * 100, block=10, n=200)
        self.assertEqual((lo, hi), (2.0, 2.0))


if __name__ == "__main__":
    unittest.main()
