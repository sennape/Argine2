"""Tests of the Argine2 research layer (argine/research).

Like the rest of Argine's tests, many of these pin down a property that a
research result depends on, so that a regression cannot silently change a
published number:

* the trace converter handles llama.cpp's last-layer pruning (a prompt step has
  N tokens in every layer except the last, which computes only one);
* exact LRU stack distances agree with an actual LRU replay for every size;
* no policy ever exceeds its budget (the hard memory invariant);
* the Belady-style oracle is never beaten on uniform sizes;
* the DP allocator is optimal (checked against brute force);
* the frozen controller is identical to the uniform per-layer LRU (ablation
  baseline is what it claims to be);
* train/eval separation of the workload corpus is enforced.
"""

from __future__ import annotations

import gzip
import itertools
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from argine.research import (allocator, analysis, elasticity, iotimeline, metrics,  # noqa: E402
                             oracle, policies, rawlog, report, stats, trace as T, workloads)
from argine.research.controller import AdaptiveResidencyController  # noqa: E402
from argine.research.costmodel import CostModel  # noqa: E402
from argine.research.replay import InvariantViolation, replay  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="argine2-tests-"))


def synthetic_trace(n_steps=120, layers=4, n_experts=16, k=2, seed=1, skew=1.2,
                    size=1000, segments=None) -> T.Trace:
    rng = random.Random(seed)
    w = [1.0 / (i + 1) ** skew for i in range(n_experts)]
    steps = []
    for g in range(n_steps):
        routes = {}
        for l in range(layers):
            perm = list(range(n_experts))
            rng.shuffle(perm) if l % 2 else None
            chosen = set()
            while len(chosen) < k:
                chosen.add(perm[rng.choices(range(n_experts), weights=w)[0]])
            routes[l] = [tuple(sorted(chosen))]
        seg = None
        if segments:
            seg = segments[min(len(segments) - 1, g * len(segments) // n_steps)]
        steps.append(T.Step(graph=g, n_tokens=1, routes=routes, phase=T.PHASE_DECODE, segment=seg))
    meta = T.TraceMeta(model="synthetic", n_experts=n_experts, top_k=k, layers=list(range(layers)),
                       expert_bytes={l: size for l in range(layers)})
    return T.Trace(meta=meta, steps=steps)


class TestTraceFormat(unittest.TestCase):
    def test_roundtrip_gzip(self):
        tr = synthetic_trace(segments=["a", "b"])
        p = tr.save(TMP / "t.jsonl.gz")
        back = T.Trace.load(p)
        self.assertEqual(len(back.steps), len(tr.steps))
        self.assertEqual(back.steps[5].routes, tr.steps[5].routes)
        self.assertEqual(back.meta.expert_bytes, tr.meta.expert_bytes)
        self.assertEqual(back.segments(), ["a", "b"])

    def test_rejects_foreign_file(self):
        p = TMP / "x.jsonl"
        p.write_text('{"hello": 1}\n')
        with self.assertRaises(T.TraceError):
            T.Trace.load(p)

    def test_route_log_last_layer_pruning_and_truncation(self):
        # two layers; a 3-token prompt step whose LAST layer has 1 token
        # (llama.cpp computes only the output token there), then two decode
        # steps, then a truncated line
        lines = []
        for kind in ("gate", "up", "down"):
            lines.append(f"blk.0.ffn_{kind}_exps.weight 3 2 1 2 3 4 5 6")
        for kind in ("gate", "up", "down"):
            lines.append(f"blk.1.ffn_{kind}_exps.weight 1 2 7 0")
        for g in range(2):
            for l in (0, 1):
                for kind in ("gate", "up", "down"):
                    lines.append(f"blk.{l}.ffn_{kind}_exps.weight 1 2 {g} {g + 1}")
        lines.append("blk.0.ffn_gate_exps.weight 1 2 4")  # truncated: 1 id of 2
        p = TMP / "route.log"
        p.write_text("\n".join(lines) + "\n")
        tr, warns = T.from_route_log(p, n_experts=8)
        self.assertEqual(len(tr.steps), 3)
        self.assertEqual(tr.steps[0].n_tokens, 3, "step size is the max over layers")
        self.assertEqual(tr.steps[0].phase, T.PHASE_PREFILL)
        self.assertEqual(tr.steps[0].routes[1], [(7, 0)])
        self.assertEqual(tr.steps[1].phase, T.PHASE_DECODE)
        self.assertTrue(any("truncated" in w for w in warns))
        self.assertEqual(tr.steps[0].experts(0), (1, 2, 3, 4, 5, 6))

    def test_split_requests(self):
        tr = synthetic_trace(n_steps=10)
        tr.steps[0].phase = T.PHASE_PREFILL
        tr.steps[4].phase = T.PHASE_PREFILL
        self.assertEqual(T.split_requests(tr), [(0, 4), (4, 10)])


class TestMetrics(unittest.TestCase):
    def test_basic_measures(self):
        self.assertAlmostEqual(metrics.shannon_entropy([1, 1, 1, 1]), 2.0)
        self.assertEqual(metrics.shannon_entropy([5, 0, 0]), 0.0)
        self.assertAlmostEqual(metrics.gini([1, 1, 1, 1]), 0.0)
        self.assertGreater(metrics.gini([0, 0, 0, 10]), 0.7)
        self.assertEqual(metrics.ews([50, 30, 15, 5], 0.90), 3)
        self.assertEqual(metrics.ews([50, 30, 15, 5], 0.95), 3)
        self.assertEqual(metrics.ews([50, 30, 15, 5], 0.99), 4)
        self.assertAlmostEqual(metrics.top_coverage([50, 30, 15, 5], 1), 0.5)
        self.assertAlmostEqual(metrics.js_divergence({1: 1}, {1: 1}), 0.0)
        self.assertAlmostEqual(metrics.js_divergence({1: 1}, {2: 1}), 1.0)

    def test_working_set_and_reuse(self):
        ts = [(0, (1, 2)), (1, (1, 3)), (2, (1, 2))]
        self.assertAlmostEqual(metrics.working_set_mean(ts, 1), 2.0)
        self.assertAlmostEqual(metrics.working_set_mean(ts, 2), (2 + 3 + 3) / 3)
        rd, cold = metrics.reuse_distances(ts)
        self.assertEqual(cold, 3)
        self.assertEqual(sorted(rd), [1, 1, 2])

    def test_stack_distance_equals_lru_replay_for_every_size(self):
        """The elasticity curves rest on this equivalence."""
        tr = synthetic_trace(n_steps=80, layers=3, n_experts=12, k=3, seed=7)
        curves = elasticity.lru_curves(tr)
        for c in range(3, 13):
            alloc = {l: c * 1000 for l in tr.layers}
            r = replay(tr, policies.PerLayerLRU(alloc), sum(alloc.values()))
            self.assertEqual(r.misses, sum(curves[l][c] for l in tr.layers), f"c={c}")

    def test_characterize_shapes(self):
        ch = metrics.characterize(synthetic_trace())
        self.assertEqual(len(ch["per_layer"]), 4)
        row = ch["per_layer"][0]
        for key in ("entropy_bits", "gini", "ews95", "ws_tau16", "p_reuse_le4", "hot_set_stability"):
            self.assertIn(key, row)
        self.assertLessEqual(row["ews90"], row["ews95"])
        self.assertLessEqual(row["ews95"], row["ews99"])


class TestPolicies(unittest.TestCase):
    def all_policies(self, tr):
        counts = policies.counts_from_trace(tr)
        return [policies.GlobalLRU(), policies.ShardedLRU(4), policies.PerLayerLRU(),
                policies.LFU(), policies.LFU(halflife=16), policies.TwoQ(),
                policies.LeastStale("lru"), policies.LeastStale("mru"),
                policies.StaticPinned(counts, 1.0), policies.StaticPinned(counts, 0.5),
                policies.StaticPinned(counts, 1.0, per_layer=True),
                AdaptiveResidencyController(epoch=8), AdaptiveResidencyController(epoch=8, pinned_fraction=0.5),
                oracle.BeladyGlobal(tr)]

    def test_budget_invariant_every_policy(self):
        tr = synthetic_trace(n_steps=60, layers=4, n_experts=16, k=2, seed=3)
        for budget in (8000, 13000, 30000):
            for p in self.all_policies(tr):
                r = replay(tr, p, budget)  # raises InvariantViolation on breach
                self.assertLessEqual(r.peak_resident, budget, p.name)

    def test_invariant_violation_is_detected(self):
        class Cheater(policies.Policy):
            name = "cheater"

            def _access(self, step, layer, experts):
                self.resident_bytes += 10 ** 9
                return []
        with self.assertRaises(InvariantViolation):
            replay(synthetic_trace(n_steps=3), Cheater(), 1000)

    def test_belady_never_beaten_with_uniform_sizes(self):
        for seed in (1, 2, 3):
            tr = synthetic_trace(n_steps=70, layers=3, n_experts=10, k=2, seed=seed)
            for budget in (6000, 10000, 15000):
                best = replay(tr, oracle.BeladyGlobal(tr), budget).misses
                for p in self.all_policies(tr)[:-1]:
                    self.assertGreaterEqual(replay(tr, p, budget).misses, best, (p.name, seed, budget))

    def test_least_stale_lru_equals_global_lru(self):
        """Documented equivalence: in a global scope, stale-first + LRU among
        stale is exactly LRU."""
        tr = synthetic_trace(n_steps=60, seed=9)
        for b in (5000, 9000):
            self.assertEqual(replay(tr, policies.LeastStale("lru"), b).misses,
                             replay(tr, policies.GlobalLRU(), b).misses)

    def test_sharded_with_one_shard_is_global(self):
        tr = synthetic_trace(seed=4)
        self.assertEqual(replay(tr, policies.ShardedLRU(1), 7000).misses,
                         replay(tr, policies.GlobalLRU(), 7000).misses)

    def test_frozen_controller_equals_uniform(self):
        tr = synthetic_trace(seed=5)
        a = replay(tr, AdaptiveResidencyController(epoch=8, adaptive=False), 12000)
        b = replay(tr, policies.PerLayerLRU(), 12000)
        self.assertEqual(a.misses, b.misses)

    def test_controller_moves_memory_and_keeps_min_slots(self):
        # layer 0 hot and varied, other layers almost constant: memory should
        # flow toward layer 0
        tr = synthetic_trace(n_steps=200, layers=4, n_experts=16, k=2, seed=11, skew=0.0)
        for st in tr.steps:
            for l in (1, 2, 3):
                st.routes[l] = [(0, 1)]
        pol = AdaptiveResidencyController(epoch=16, hysteresis=0.0)
        replay(tr, pol, 4 * 6 * 1000)
        alloc = pol.allocation()
        self.assertGreater(alloc[0], alloc[1])
        for l, b in alloc.items():
            self.assertGreaterEqual(b // 1000, 2, "min_slots = top_k must hold")

    def test_overflow_is_reported_not_hidden(self):
        tr = synthetic_trace(n_steps=5, layers=2, n_experts=8, k=4)
        r = replay(tr, policies.PerLayerLRU({0: 2000, 1: 2000}), 4000)
        self.assertGreater(r.overflow_events, 0)


class TestAllocator(unittest.TestCase):
    def test_dp_matches_brute_force(self):
        rng = random.Random(2)
        curves = {}
        for l in range(3):
            c = [100.0]
            for _ in range(6):
                c.append(max(0.0, c[-1] - rng.randint(0, 30)))
            curves[l] = c
        sizes = {0: 2, 1: 3, 2: 2}
        budget = 10
        best = min((sum(curves[l][c[l]] for l in range(3)), c)
                   for c in itertools.product(range(7), repeat=3)
                   if sum(c[l] * sizes[l] for l in range(3)) <= budget)
        got = allocator.dp_optimal(curves, sizes, budget)
        self.assertAlmostEqual(allocator.total_cost(curves, got), best[0])

    def test_dp_not_worse_than_greedy_or_uniform(self):
        tr = synthetic_trace(n_steps=100, layers=4, n_experts=16, k=2, seed=6)
        curves = elasticity.lru_curves(tr)
        sizes = {l: 1000 for l in tr.layers}
        for b in (12000, 20000, 40000):
            dp = allocator.total_cost(curves, allocator.dp_optimal(curves, sizes, b, 2))
            gr = allocator.total_cost(curves, allocator.lookahead_greedy(curves, sizes, b, 2))
            un = allocator.total_cost(curves, allocator.uniform(tr.layers, sizes, b, 16, 2))
            self.assertLessEqual(dp, gr + 1e-9)
            self.assertLessEqual(dp, un + 1e-9)

    def test_min_slots_infeasible_raises(self):
        with self.assertRaises(ValueError):
            allocator.uniform([0, 1], {0: 10, 1: 10}, 30, 8, min_slots=2)


class TestRawLogAndIO(unittest.TestCase):
    RAW = ("H\targine-raw\t1\t1000\t5000000\t42\n"
           "C\tbudget_bytes\t4096\n"
           "R\t0\t1100\t0\t1\t2\t3\t5\n"
           "A\t1200\t0\t0\t3\t0\tm\t100\t1\n"
           "A\t1210\t0\t0\t5\t0\th\t100\t0\n"
           "I\t1\t0\t0\t3\t0\t100\t1300\t1700\t1500\t1800\t1\t1\n"
           "I\t2\t1\t0\t5\t0\t100\t1400\t1600\t1400\t1600\t1\t6\n"
           "N\t0\t0\t0\t1150\t1900\t2\t4\n"
           "E\t2000\t0\t3\t0\t100\t1700\t1800\t2\n"
           "S\t2100\t0\t100\t4096\t1\t1\t1\t100\t0\n"
           "Z\tfuture record\n"
           "R\t1\t3000\t0\t1\t2\t3")  # truncated tail, no newline

    def setUp(self):
        self.p = TMP / "raw.tsv"
        self.p.write_text(self.RAW)

    def test_parse(self):
        log = rawlog.parse(self.p)
        self.assertTrue(log.truncated_tail)
        self.assertEqual(log.unknown["Z"], 1, "unknown records are counted, not dropped silently")
        self.assertEqual(len(log.routes), 1)
        self.assertEqual(log.config["budget_bytes"], "4096")
        self.assertEqual(log.io[0].hidden_ns, 200)      # 1300 -> wait at 1500
        self.assertEqual(log.io[0].unhidden_ns, 200)
        self.assertEqual(log.io[1].hidden_ns, 0)        # synchronous
        self.assertEqual(log.evictions[0].resident_ns, 300)
        tr = rawlog.to_trace(log, markers=[{"event": "request_start", "segment": "coding",
                                            "system_ns": 5000000 - 1000 + 1050}])
        self.assertEqual(tr.steps[0].segment, "coding")

    def test_queue_depth_exact(self):
        log = rawlog.parse(self.p)
        q = iotimeline.queue_depth(log.io)
        # [1300,1700) and [1400,1600): depth 1,2,1 over 100,200,100 ns
        self.assertEqual(q["q_max"], 2)
        self.assertAlmostEqual(q["q_time_weighted_mean_over_span"], (100 + 400 + 100) / 400)
        self.assertEqual(q["label"], "upper bound (some completions bounded)")

    def test_node_attribution(self):
        log = rawlog.parse(self.p)
        att = rawlog.node_attributed_outcomes(log)
        self.assertEqual(att["unassigned_accesses"], 0)
        self.assertAlmostEqual(att["gate_node_hit_rate"], 0.5)


class TestStatsWorkloads(unittest.TestCase):
    def test_summary(self):
        s = stats.summarize([10.0, 11.0, 12.0])
        self.assertAlmostEqual(s["mean"], 11.0)
        self.assertAlmostEqual(s["sd"], 1.0)
        self.assertAlmostEqual(s["ci95_high"] - s["mean"], 4.303 / 3 ** 0.5, places=3)
        self.assertTrue(s["enough_repetitions"])
        self.assertFalse(stats.summarize([1.0, 2.0])["enough_repetitions"])

    def test_schedule_interleaves(self):
        order = stats.interleaved_schedule(["A", "B", "C"], 3)
        self.assertEqual(len(order), 9)
        self.assertNotEqual(order[:3], order[3:6], "the order must change between rounds")
        for cfg in "ABC":
            self.assertEqual(order.count(cfg), 3)

    def test_corpus_loads_and_is_disjoint(self):
        names = workloads.available()
        for required in ("natural-language", "coding", "reasoning-math", "structured-json",
                         "italian", "multi-turn", "transition"):
            self.assertIn(required, names)
        for n in names:
            w = workloads.load(n)
            self.assertIn("dev", w.prompts)
            self.assertIn("eval", w.prompts)
            self.assertEqual(len(w.sha256), 64)

    def test_corpus_rejects_leak(self):
        p = TMP / "leak.json"
        p.write_text(json.dumps({"name": "leak", "splits": {
            "dev": [{"id": "a", "prompt": "same"}], "eval": [{"id": "b", "prompt": "same"}]}}))
        with self.assertRaises(ValueError):
            workloads.load(str(p))


class TestPipeline(unittest.TestCase):
    def test_analysis_and_report(self):
        tr = synthetic_trace(n_steps=150, layers=4, n_experts=16, k=2, seed=8,
                             segments=["natural-language", "coding"])
        out = TMP / "analysis"
        analysis.run_all(tr, out, [10000, 16000, 30000], CostModel(), opt_grid_points=4)
        for f in ("characterization.json", "per_layer.csv", "reuse_cdf.csv", "lru_curves.csv",
                  "elasticity.csv", "heterogeneity.json", "policy_comparison.csv", "adaptive.csv",
                  "drift.csv"):
            self.assertTrue((out / f).exists(), f)
        p = report.build(out)
        text = p.read_text()
        self.assertIn("[LIMITATION]", text)       # in-sample warning present
        self.assertIn("belady-global", text)

    def test_cli_convert_and_replay(self):
        from argine.cli import main
        lines = []
        for g in range(40):
            for l in (0, 1):
                for kind in ("gate", "up", "down"):
                    lines.append(f"blk.{l}.ffn_{kind}_exps.weight 1 2 {g % 5} {(g * 3) % 7}")
        src = TMP / "r2.log"
        src.write_text("\n".join(lines) + "\n")
        dst = TMP / "r2.jsonl.gz"
        self.assertEqual(main(["research", "convert", str(src), "--to", str(dst),
                               "--n-experts", "8", "--expert-bytes", "100"]), 0)
        self.assertEqual(main(["research", "replay", str(dst), "--policy", "global-lru",
                               "--budget", "800"]), 0)


class TestRepositoryIntegrity(unittest.TestCase):
    """The repository is the research artifact: its links, registry and derived
    data must stay consistent with the raw data."""

    def test_markdown_links_resolve(self):
        sys.path.insert(0, str(ROOT / "tools"))
        import check_links
        self.assertEqual(check_links.check(), [])

    def test_experiment_registry_is_valid(self):
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "index_experiments.py"), "--check"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_historical_results_regenerate_identically(self):
        """Processed data must be reproducible from raw data (section 44)."""
        import hashlib
        files = sorted((ROOT / "experiments").glob("EXP-0*/results.csv"))

        def digest():
            return {f: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
        before = digest()
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "build_historical_data.py")],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(before, digest(), "results.csv changed when regenerated from raw data")

    def test_imported_logs_are_sanitised(self):
        for name in ("runs.jsonl", "perplexity.jsonl"):
            text = (ROOT / "data" / "raw" / "argine-v1" / name).read_text(encoding="utf-8")
            self.assertNotRegex(text, r"(?i)\balpel\b|LucaAlpe")


@unittest.skipUnless(shutil.which("g++"), "g++ not available")
class TestResearchInstrumentationCpp(unittest.TestCase):
    """Compile and run the C++ test of patches/argine-research.* against a
    real store built from the tiny MoE fixture."""

    def test_cpp(self):
        sys.path.insert(0, str(ROOT / "tests"))
        from make_tiny_model import make_tiny_moe
        from argine.gguf import read_gguf
        from argine import expert_store as es
        d = TMP / "cpp"
        d.mkdir(exist_ok=True)
        g = d / "tiny.gguf"
        make_tiny_moe(g)
        m = read_gguf(str(g))
        es.write_store(m, es.build_plan(m), d / "tiny.psxs")
        exe = d / "arxtest"
        pdir = ROOT / "patches"
        r = subprocess.run(["g++", "-O1", "-std=c++17", "-pthread",
                            str(pdir / "argine-store.cpp"), str(pdir / "argine-research.cpp"),
                            str(pdir / "argine-research-test.cpp"), "-o", str(exe)],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        r = subprocess.run([str(exe), str(d / "tiny.psxs"), str(m.n_layers), str(m.expert_count),
                            str(d / "trace.raw")], capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:])
        self.assertIn("all ok", r.stdout)


if __name__ == "__main__":
    unittest.main()
