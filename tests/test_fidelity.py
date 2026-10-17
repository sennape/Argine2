"""Regression tests for replay fidelity and the static-allocation optimum
(docs/research/REPLAY-FIDELITY-VALIDATION.md).

Numbered as in the validation requirement:
 1 runtime/replay identical sequence (REAL C++ store vs StoreReplay)
 2 shard mapping            3 per-shard capacity      4 byte-based eviction
 5 unequal expert sizes     6 grouped expert events   7 event atomicity
 8 exact LRU ordering       9 warm-start state       10 empty-start state
11 workload filtering      12 warm-up exclusion      13 flush-sentinel exclusion
14 gate-level metric semantics                       15 in-flight ('w') behaviour
16 optimal allocation never worse than feasible uniform (same objective)
17 budget invariant        18 deterministic replay
plus: stack-distance curves == replay under v2 semantics with prompt steps
that overflow partitions; legacy semantics still reproduce the superseded
numbers; the validator finds a planted divergence.
Windows v2 follow-up: per-thread pins with two thread pools (real store and
model), the old single-pool pin model is shown to diverge, sequence gaps are
classified, node completeness finds lost records.
"""

from __future__ import annotations

import itertools
import json
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from argine.research import allocator, elasticity, fidelity as F, policies, rawlog  # noqa: E402
from argine.research import storereplay as S  # noqa: E402
from argine.research.replay import counted_step, replay  # noqa: E402
from argine.research.trace import PHASE_DECODE, PHASE_PREFILL, Step, Trace, TraceMeta  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="argine2-fid-"))
MiB = 1 << 20


def mixed_trace(n_steps=160, layers=3, n_exp=12, k=2, seed=3, size=1000, prompt_every=40,
                prompt_tokens=6, segments=True) -> Trace:
    rng = random.Random(seed)
    steps = []
    for g in range(n_steps):
        ntok = prompt_tokens if g % prompt_every == 0 else 1
        routes = {}
        for l in range(layers):
            sel = []
            for _ in range(ntok):
                ch = set()
                while len(ch) < k:
                    ch.add(rng.choice([0, 1, 2, 3]) if rng.random() < 0.6 else rng.randrange(n_exp))
                sel.append(tuple(sorted(ch)))
            routes[l] = sel
        seg = None
        if segments:
            seg = "warmup" if g < 5 else ("flush-sentinel" if g == n_steps - 1 else "coding")
        steps.append(Step(g, ntok, routes, PHASE_PREFILL if ntok > 1 else PHASE_DECODE, seg))
    meta = TraceMeta(model="synthetic", n_experts=n_exp, top_k=k, layers=list(range(layers)),
                     expert_bytes={l: size for l in range(layers)})
    return Trace(meta, steps)


def sizes_fn(d):
    return lambda key: d[(key[0], key[2])]


# --------------------------------------------------------------------- 1

@unittest.skipUnless(shutil.which("g++"), "g++ not available")
class TestRealStoreEquivalence(unittest.TestCase):
    """The C++ store (argine-store.cpp) driven deterministically writes a
    format-v2 trace; StoreReplay must make every identical decision."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "tests"))
        from make_tiny_model import make_tiny_moe
        from argine.gguf import read_gguf
        from argine import expert_store as es
        d = TMP / "store"
        d.mkdir(exist_ok=True)
        g = d / "tiny.gguf"
        make_tiny_moe(g)
        m = read_gguf(str(g))
        cls.store = d / "tiny.psxs"
        es.write_store(m, es.build_plan(m), cls.store)
        cls.exe = d / "fidtest"
        p = ROOT / "patches"
        r = subprocess.run(["g++", "-O1", "-std=c++17", "-pthread", str(p / "argine-store.cpp"),
                            str(p / "argine-research.cpp"), str(p / "argine-store-fidelity-test.cpp"),
                            "-o", str(cls.exe)], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr[-2000:])

    def run_cfg(self, slices: int, shards: int, seed: int, pools: int = 1):
        """Run the driver until a run never takes the store's "everything in
        this shard is pinned" failure path.

        Which shard a slice lands in hashes the region's virtual address
        (ASLR), so on a small multi-shard budget a prompt node can, in some
        address layouts, pin more than one shard holds. That path waits 2 s
        and drops the request (EXP-021 runs never take it: 0 failures). It is a
        property of the layout, not of the replay under test, so a layout that
        avoids it is drawn again (fresh process = fresh layout), a bounded
        number of times."""
        d = TMP / f"run-{slices}-{shards}-{seed}-p{pools}"
        d.mkdir(exist_ok=True)
        outs = []
        for _attempt in range(8):
            r = subprocess.run([str(self.exe), str(self.store), "2", "8", str(d / "raw.tsv"),
                                str(8192 * slices), str(shards), "300", str(seed), str(pools)],
                               capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr)
            outs.append(r.stdout.strip())
            if "failures 0" in r.stdout:
                break
        self.assertIn("failures 0", r.stdout, outs)
        (d / "markers.jsonl").write_text("")
        log = rawlog.parse(d / "raw.tsv")
        st = F.load_runtime_stream(d, log)
        return log, st

    def test_every_decision_identical(self):
        # shard choice hashes real virtual addresses (ASLR), so configurations
        # keep enough headroom per shard never to hit the "all pinned" path;
        # 16 shards are covered by the end-to-end llama-server runs (EXP-021)
        for slices, shards, seed in ((26, 2, 7), (40, 4, 11), (30, 3, 5), (36, 3, 2)):
            log, st = self.run_cfg(slices, shards, seed)
            self.assertEqual(log.version, 2)
            self.assertEqual(F.reconstruct(log)["total"], 0)
            res = F.compare(st)                         # free-running, recorded shard map
            self.assertGreater(res["runtime_evictions"], 100, (slices, shards))
            self.assertEqual(res["outcome_disagreements"], 0, res["first_divergence"])
            self.assertEqual(res["victim_disagreements"], 0, res["first_divergence"])
            self.assertEqual(res["replay_evictions"], res["runtime_evictions"])
            self.assertEqual(res["runtime_metrics"], res["replay_metrics"])

    def test_two_thread_pools_every_decision_identical(self):
        """--threads != --threads-batch: prompt and decode graphs run on
        different threads, each keeping its pins until it runs again (the
        cause of the 5.29% disagreement of the first Windows v2 run)."""
        # 1 shard: no address-dependent placement, fully deterministic
        for slices, shards, seed in ((30, 1, 3), (40, 2, 9)):
            log, st = self.run_cfg(slices, shards, seed, pools=2)
            self.assertEqual(len(st.pools), 2)
            res = F.compare(st)
            self.assertGreater(res["runtime_evictions_past_pinned"], 50,
                               "the other pool's pins must actually be passed over")
            self.assertEqual(res["outcome_disagreements"], 0, res["first_divergence"])
            self.assertEqual(res["victim_disagreements"], 0, res["first_divergence"])
            self.assertEqual(res["skip_count_disagreements"], 0)
            self.assertEqual(res["replay_evictions_past_pinned"],
                             res["runtime_evictions_past_pinned"])
            old = F.compare(st, pin_model="single-pool", max_divergences=1)
            self.assertGreater(old["victim_disagreements"], 0,
                               "the single-pool pin model must fail here (test is sensitive)")
            self.assertEqual(old["replay_evictions_past_pinned"], 0)

    def test_wrong_shard_map_is_detected(self):
        """The validator is not trivially green: a different shard map diverges."""
        log, st = self.run_cfg(40, 4, 11)
        res = F.compare(st, S.HashShardMap(st.n_shards, st.sizes, 99))
        self.assertGreater(res["outcome_disagreements"], 0)
        self.assertIsNotNone(res["first_divergence"])

    def test_planted_divergence_is_located(self):
        """Flip one runtime outcome: the first divergence is exactly there."""
        log, st = self.run_cfg(26, 2, 7)
        i = next(j for j, r in enumerate(st.requests) if r.outcome == "h" and j > 500)
        st.requests[i].outcome = "m"
        res = F.compare(st, lockstep=True)
        self.assertEqual(res["first_divergence"]["index"], i)
        self.assertEqual(res["outcome_disagreements"], 1, "lockstep must not cascade")


# ----------------------------------------------------------------- 2, 3

class TestShards(unittest.TestCase):
    def test_runtime_shard_count_rule(self):
        self.assertEqual(S.runtime_shard_count(4 << 30), 16)
        self.assertEqual(S.runtime_shard_count(int(1.5 * (1 << 30))), 16)
        self.assertEqual(S.runtime_shard_count(96 * MiB), 3)
        self.assertEqual(S.runtime_shard_count(20 * MiB), 1)
        self.assertEqual(S.runtime_shard_count(20 * MiB, forced=16), 16)

    def test_per_shard_budgets_sum_exactly(self):
        for b, n in ((4 << 30, 16), (1610612737, 16), (1000, 3)):
            bs = S.shard_budgets(b, n)
            self.assertEqual(sum(bs), b)
            self.assertEqual(bs[0], b // n)
            self.assertEqual(bs[-1], b // n + b - (b // n) * n)

    def test_hash_is_the_runtime_mix64(self):
        # value from the C++ finaliser (fmix64(1)), fixed reference
        self.assertEqual(S.mix64(1), 0xB456BCFC34C2CB2C)
        m = S.HashShardMap(16, {(0, 0): 4096}, seed=1)
        self.assertEqual([m((0, e, 0)) for e in range(8)], [m((0, e, 0)) for e in range(8)])
        self.assertTrue(all(0 <= m((0, e, 0)) < 16 for e in range(64)))

    def test_recorded_map_from_v2_records(self):
        raw = ("H\targine-raw\t2\t0\t0\t1\nC\tbudget_bytes\t8192\nC\tcache_shards\t2\n"
               "R\t0\t5\t0\t1\t1\t3\nN\t0\t0\t0\t1\t100\t1\t1\n"
               "A\t10\t0\t0\t3\t0\tm\t4096\t1\t1\t0\n")
        d = TMP / "v2map"
        d.mkdir(exist_ok=True)
        (d / "raw.tsv").write_text(raw)
        (d / "markers.jsonl").write_text("")
        st = F.load_runtime_stream(d)
        self.assertEqual(st.version, 2)
        self.assertEqual(st.mapping[(0, 3, 0)], 1)


# ------------------------------------------------------------ 4, 5, 8, 17

def one_shard(budget, sizes):
    class M:
        n = 1
        exact = True
        label = "one"

        def __call__(self, k):
            return 0
    return S.StoreReplay(budget, M(), sizes_fn(sizes), 1)


class TestStoreReplayMechanics(unittest.TestCase):
    def test_byte_based_eviction_frees_enough(self):
        sizes = {(0, 0): 100, (1, 0): 300}
        rp = one_shard(400, sizes)
        for e in range(4):
            rp.begin_node()
            rp.access((0, e, 0))          # 4 x 100 = full
        rp.begin_node()
        d = rp.access((1, 0, 0))          # needs 300 -> evicts the 3 oldest
        self.assertEqual(d.victims, [(0, 0, 0), (0, 1, 0), (0, 2, 0)])
        self.assertLessEqual(rp.total_resident(), 400)

    def test_unequal_sizes_respected(self):
        sizes = {(0, 0): 3059712, (1, 0): 2654208}
        rp = one_shard(3059712 + 2 * 2654208, sizes)
        for key in ((1, 0, 0), (1, 1, 0), (0, 0, 0)):          # two small ones are oldest
            rp.begin_node()
            self.assertFalse(rp.access(key).hit)
        self.assertEqual(rp.evictions, 0)
        rp.begin_node()
        d = rp.access((0, 1, 0))
        self.assertEqual(len(d.victims), 2, "a 3.06 MB slice needs two 2.65 MB slots freed")

    def test_exact_lru_order(self):
        sizes = {(0, 0): 1}
        rp = one_shard(3, sizes)
        for e in (0, 1, 2):
            rp.begin_node()
            rp.access((0, e, 0))
        rp.begin_node()
        rp.access((0, 0, 0))              # 0 becomes MRU: order 1, 2, 0
        rp.begin_node()
        self.assertEqual(rp.access((0, 3, 0)).victims, [(0, 1, 0)])
        rp.begin_node()
        self.assertEqual(rp.access((0, 4, 0)).victims, [(0, 2, 0)])

    def test_budget_invariant_under_pressure(self):
        tr = mixed_trace()
        sizes = {(l, k): 1000 + 37 * l + 11 * k for l in range(3) for k in range(3)}
        for budget in (9000, 15000, 30000):
            for sd in range(3):
                rp = S.StoreReplay(budget, S.HashShardMap(3, sizes, sd), sizes_fn(sizes), 3)
                S.run(S.synthetic_stream(tr), rp)
                self.assertEqual(rp.violations, 0)
                self.assertLessEqual(rp.peak, budget)
                for s in range(3):
                    self.assertLessEqual(rp.resident[s], rp.budgets[s])


# --------------------------------------------------------------------- 6

class TestRequestPattern(unittest.TestCase):
    def test_gate_up_down_nodes_request_all_kinds_in_k_order(self):
        tr = mixed_trace(n_steps=1, layers=1, segments=False)
        reqs = list(S.synthetic_stream(tr))
        ex = tr.steps[0].experts(0)
        per_node = {}
        for r in reqs:
            per_node.setdefault(r.node, []).append(r)
        self.assertEqual(len(per_node), 3)
        expected = [(0, e, k) for k in (0, 1, 2) for e in ex]
        for nk, (node, rs) in enumerate(sorted(per_node.items())):
            self.assertEqual([r.key for r in rs], expected)
            self.assertTrue(all(r.node_kind == nk for r in rs))


# --------------------------------------------------------------------- 7

class TestAtomicity(unittest.TestCase):
    def test_runtime_is_not_event_atomic_but_counterfactuals_are(self):
        """Resident members requested LATER in the node are not protected by
        the store: they can be evicted by an earlier member's miss."""
        sizes = {(0, 0): 1, (1, 0): 1}
        rp = one_shard(2, sizes)
        rp.begin_node(); rp.access((0, 5, 0))        # LRU end
        rp.begin_node(); rp.access((1, 9, 0))
        rp.begin_node()                              # node needs (0,1) and (0,5)
        d1 = rp.access((0, 1, 0))
        self.assertEqual(d1.victims, [(0, 5, 0)], "member (0,5) not yet requested: evictable")
        self.assertFalse(rp.access((0, 5, 0)).hit)
        # an event-atomic counterfactual protects (0,5)
        lru = policies.LRUScope(2)
        lru.admit((0, 5), 1, set()); lru.admit((1, 9), 1, set())
        ok, _ = lru.admit((0, 1), 1, {(0, 1), (0, 5)})
        self.assertTrue(ok)
        self.assertIn((0, 5), lru)

    def test_pins_cover_current_node_only(self):
        sizes = {(0, 0): 1}
        rp = one_shard(2, sizes)
        rp.begin_node(); rp.access((0, 0, 0)); rp.access((0, 1, 0))
        d = rp.access((0, 2, 0))                     # both resident are pinned
        self.assertTrue(d.failed)
        rp.begin_node()                              # next node: pins released
        self.assertEqual(rp.access((0, 2, 0)).victims, [(0, 0, 0)])


    def test_pins_are_per_thread(self):
        """A thread that does not enter the node keeps its pins."""
        sizes = {(0, 0): 1}
        rp = one_shard(3, sizes)
        rp.begin_node([1]); rp.access((0, 0, 0), tid=1)          # prompt thread 1
        rp.begin_node([2]); rp.access((0, 1, 0), tid=2)          # decode thread 2
        rp.access((0, 2, 0), tid=2)
        rp.begin_node([2])                                       # thread 1 not in this pool
        d = rp.access((0, 3, 0), tid=2)
        self.assertEqual(d.victims, [(0, 1, 0)], "(0,0) is still pinned by thread 1")
        self.assertEqual(d.skipped, [1])
        rp.begin_node([1])                                       # thread 1 runs again
        self.assertTrue(rp.access((0, 0, 0), tid=1).hit)
        rp.begin_node([1, 2])
        d = rp.access((0, 4, 0), tid=1)
        self.assertEqual(d.victims, [(0, 2, 0)])

    def test_pin_counts_across_threads(self):
        """Two threads pinning the same slice: it stays pinned until both release."""
        sizes = {(0, 0): 1}
        rp = one_shard(2, sizes)
        rp.begin_node([1, 2]); rp.access((0, 0, 0), tid=1); rp.access((0, 0, 0), tid=2)
        rp.access((0, 1, 0), tid=1)
        rp.begin_node([1])
        self.assertTrue(rp.is_pinned((0, 0, 0)))
        self.assertEqual(rp.access((0, 2, 0), tid=1).victims, [(0, 1, 0)])
        rp.begin_node([2])
        self.assertFalse(rp.is_pinned((0, 0, 0)))


# ------------------------------------------------- record integrity (v2)

def _log_with(records, windows_ns=None):
    log = rawlog.RawLog("synthetic")
    for r in records:
        if r[0] == "A":
            _, t, tid, key, out, sh, sq = r
            log.accesses.append(rawlog.Access(t, tid, *key, out, 4096, 0, sh, sq))
        else:
            _, t, key, sh, sq = r
            log.evictions.append(rawlog.Eviction(t, *key, 4096, 0, 0, 1, sh, sq, 0))
    return log


class TestRecordIntegrity(unittest.TestCase):
    def test_gap_followed_by_workload_is_replay_relevant(self):
        log = _log_with([("A", 10, 1, (0, 1, 0), "m", 0, 0), ("A", 20, 1, (0, 2, 0), "m", 0, 2),
                         ("A", 30, 1, (0, 1, 0), "h", 0, 3)])
        g = F.seq_gaps(log, [])
        self.assertEqual(g["missing"], 1)
        self.assertEqual(g["shards_with_replay_relevant_seq_gaps"], 1)
        self.assertEqual(g["gaps"][0]["seq"], 1)

    def test_gap_in_sentinel_tail_is_not(self):
        log = _log_with([("A", 10, 1, (0, 1, 0), "m", 0, 0), ("A", 20, 1, (0, 2, 0), "m", 0, 1),
                         ("A", 110, 2, (0, 1, 0), "h", 0, 3), ("A", 120, 2, (0, 2, 0), "h", 0, 5)])
        g = F.seq_gaps(log, [(100, 200)])                       # flush-sentinel window
        self.assertEqual(g["missing"], 2)
        self.assertEqual(g["shards_with_seq_gaps"], 1)
        self.assertEqual(g["shards_with_replay_relevant_seq_gaps"], 0)
        self.assertEqual([x["inside_excluded_segment"] for x in g["gaps"]], [False, True],
                         "seq 2 lies between a workload and a sentinel record; seq 4 inside")

    def test_no_gaps_and_duplicates_counted(self):
        log = _log_with([("A", 10, 1, (0, 1, 0), "m", 0, 0), ("E", 11, (0, 1, 0), 0, 1),
                         ("A", 12, 1, (0, 2, 0), "m", 0, 2), ("A", 13, 1, (0, 2, 0), "h", 0, 2)])
        g = F.seq_gaps(log, [])
        self.assertEqual(g["missing"], 0)
        self.assertEqual(g["duplicates"], 1)

    def test_node_completeness_finds_lost_requests(self):
        log = rawlog.RawLog("synthetic")
        log.routes.append((0, 0, 0, 1, 2, [3, 5]))
        log.nodes.append(rawlog.Node(0, 0, 0, 100, 200, 2, 1))
        log.nodes.append(rawlog.Node(0, 0, 1, 300, 400, 2, 1))
        for k in range(3):
            for e in (3, 5):
                log.accesses.append(rawlog.Access(150, 1, 0, e, k, "m", 4096, 0, 0, 0))
                if (e, k) != (5, 2):
                    log.accesses.append(rawlog.Access(350, 1, 0, e, k, "h", 4096, 0, 0, 0))
        c = F.node_completeness(log, [])
        self.assertEqual(c["incomplete"], 1)
        self.assertEqual(c["details"][0]["node_kind"], 1)
        self.assertEqual(c["details"][0]["missing"], [(0, 5, 2)])
        self.assertEqual(F.node_completeness(log, [(250, 500)])["incomplete_in_workload"], 0)


def _stream(reqs, evictions, missing=None, size=4096, budget_slices=2):
    """One shard, slices of equal size. reqs: (key, outcome, seq, workload);
    evictions: (seq, key) as recorded by the runtime."""
    rq = [S.Request(i, 0, 0, k, "decode", wl, out, 0, sq, i, 1)
          for i, (k, out, sq, wl) in enumerate(reqs)]
    keys = {k for k, *_ in reqs} | {k for _s, k in evictions}
    return F.RuntimeStream(rq, {0: sorted((sq, k, 0) for sq, k in evictions)}, {},
                           {(k[0], k[2]): size for k in keys}, budget_slices * size, 1, 2,
                           {k: 0 for k in keys}, 0, len(rq), None, None, missing)


class TestClosureRegions(unittest.TestCase):
    """EXP-007 80B follow-up: full-trace, workload and termination-tail
    fidelity are separate questions; nothing is hidden."""
    A, B, C, D = (0, 1, 0), (0, 2, 0), (0, 3, 0), (0, 4, 0)

    def test_exact_stream_has_no_disagreement_anywhere(self):
        st = _stream([(self.A, "m", 0, True), (self.B, "m", 1, True), (self.C, "m", 3, False)],
                     [(2, self.A)])
        r = F.compare(st)
        self.assertEqual(r["event_disagreement_rate"], 0)
        self.assertEqual(r["regions"]["tail"]["requests"], 1)
        self.assertEqual(r["regions"]["workload"]["rt_evictions"], 0)
        self.assertEqual(r["regions"]["tail"]["rt_evictions"], 1)

    def test_tail_disagreement_after_record_loss_is_explained(self):
        # the runtime evicted B (not A) because a lost record (seq 2) had touched A
        st = _stream([(self.A, "m", 0, True), (self.B, "m", 1, True), (self.C, "m", 4, False)],
                     [(3, self.B)], missing={0: [2]})
        r = F.compare(st)
        self.assertEqual(r["regions"]["workload"]["disagreements"], 0)
        self.assertEqual(r["regions"]["tail"]["victim_disagreements"], 1)
        self.assertEqual(r["tail_disagreements_explained_by_record_loss"], 1)
        self.assertEqual(r["tail_disagreements_unexplained"], 0)
        self.assertGreater(r["event_disagreement_rate"], 0, "full-trace result is not hidden")

    def test_tail_disagreement_without_record_loss_is_unexplained(self):
        st = _stream([(self.A, "m", 0, True), (self.B, "m", 1, True), (self.C, "m", 3, False)],
                     [(2, self.B)])
        r = F.compare(st)
        self.assertEqual(r["tail_disagreements_unexplained"], 1)

    def test_loss_in_the_tail_after_the_request_does_not_explain_it(self):
        st = _stream([(self.A, "m", 0, True), (self.B, "m", 1, True), (self.C, "m", 3, False)],
                     [(2, self.B)], missing={0: [7]})
        self.assertEqual(F.compare(st)["tail_disagreements_unexplained"], 1)

    def test_pre_workload_disagreement_is_strict(self):
        # warm-up (not workload) disagreement BEFORE the last workload request
        st = _stream([(self.A, "m", 0, False), (self.B, "m", 1, False), (self.C, "m", 3, False),
                      (self.D, "m", 5, True)], [(2, self.B), (4, self.A)], missing={0: []})
        r = F.compare(st)
        self.assertGreater(r["regions"]["pre_workload"]["disagreements"], 0)
        self.assertEqual(r["tail_disagreements_unexplained"], 0)
        self.assertEqual(r["regions"]["tail"]["requests"], 0)

    def test_tool_verdicts(self):
        sys.path.insert(0, str(ROOT / "tools"))
        import validate_replay_fidelity as V
        base = {"reconstruct_inconsistencies": 0, "shards_with_replay_relevant_seq_gaps": 0,
                "incomplete_nodes_in_workload": 0, "rp_budget_violations": 0}
        wl = dict(base, wl_disagreements=0, wl_victim_disagreements=0, wl_skip_count_disagreements=0,
                  pre_workload_disagreements=0, pre_workload_skip_count_disagreements=0,
                  wl_rt_misses=5, wl_rp_misses=5, wl_rt_miss_bytes=9, wl_rp_miss_bytes=9,
                  wl_rt_evictions=3, wl_rp_evictions=3, wl_rp_failures=0, pre_workload_rp_failures=0)
        self.assertEqual(V._verdict(V.WORKLOAD, wl), "PASS")
        self.assertTrue(V._verdict(V.WORKLOAD, dict(wl, pre_workload_disagreements=1)).startswith("FAIL"))
        self.assertTrue(V._verdict(V.WORKLOAD, dict(wl, wl_rp_evictions=4)).startswith("FAIL"))
        self.assertTrue(V._verdict(V.WORKLOAD, dict(wl, incomplete_nodes_in_workload=1)).startswith("FAIL"))


# ---------------------------------------------------------------- 9, 10

class TestStartState(unittest.TestCase):
    def test_empty_start_misses_first_touch(self):
        sizes = {(0, 0): 1}
        rp = one_shard(10, sizes)
        rp.begin_node()
        self.assertFalse(rp.access((0, 0, 0)).hit)

    def test_warm_start_hits(self):
        sizes = {(0, 0): 1}
        rp = one_shard(3, sizes)
        rp.warm_start([(0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 3, 0)])   # last one does not fit
        self.assertEqual(rp.total_resident(), 3)
        rp.begin_node()
        self.assertTrue(rp.access((0, 2, 0)).hit)
        self.assertFalse(rp.access((0, 3, 0)).hit)

    def test_warmup_steps_serve_state_but_are_not_counted(self):
        tr = mixed_trace()
        # v2: warm-up steps are served (state) but excluded from the metric
        r = replay(tr, policies.GlobalLRU(), 6000, phase="decode")
        r_legacy = replay(tr, policies.GlobalLRU(), 6000, phase="decode", semantics="v1-legacy")
        n_dec_workload = sum(1 for s in tr.steps if counted_step(s, "decode"))
        self.assertEqual(r.steps, n_dec_workload)
        self.assertEqual(r_legacy.steps, sum(1 for s in tr.steps if s.phase == "decode"))


# ----------------------------------------------------------- 11, 12, 13

def fake_run(d: Path, windows):
    """Raw v2 log with one node per step and markers for warm-up/sentinel."""
    d.mkdir(parents=True, exist_ok=True)
    lines = ["H\targine-raw\t2\t0\t1000\t1", "C\tbudget_bytes\t100000", "C\tcache_shards\t1"]
    seq = 0
    for g in range(6):
        t = 1000 * (g + 1)
        lines.append(f"R\t{g}\t{t}\t0\t1\t1\t{g % 2}")
        lines.append(f"N\t{g}\t0\t0\t{t}\t{t + 500}\t1\t1")
        out = "m" if g < 2 else "h"
        lines.append(f"A\t{t + 10}\t0\t0\t{g % 2}\t0\t{out}\t4096\t{g + 1 if out == 'm' else 0}\t0\t{seq}")
        seq += 1
    (d / "raw.tsv").write_text("\n".join(lines) + "\n")
    mk = []
    for seg, a, b in windows:
        mk.append({"event": "request_start", "segment": seg, "prompt_id": seg, "system_ns": a + 1000})
        mk.append({"event": "request_end", "segment": seg, "prompt_id": seg, "system_ns": b + 1000})
    (d / "markers.jsonl").write_text("".join(json.dumps(m) + "\n" for m in mk))


class TestWorkloadFiltering(unittest.TestCase):
    def test_warmup_and_sentinel_excluded_from_both_sides(self):
        d = TMP / "wl"
        fake_run(d, [("warmup", 900, 2600), ("flush-sentinel", 5900, 6700)])
        st = F.load_runtime_stream(d)
        flags = [r.workload for r in st.requests]
        self.assertEqual(flags, [False, False, True, True, True, False])
        res = F.compare(st)
        # workload requests 3, all hits for runtime and replay
        self.assertEqual(res["runtime_metrics"]["slice_all_all_n"], 3)
        self.assertEqual(res["runtime_metrics"]["slice_all_all"], 1.0)
        self.assertEqual(res["replay_metrics"], res["runtime_metrics"])

    def test_metrics_ignore_non_workload(self):
        m = S.Metrics()
        m.add(S.Request(0, 0, 0, (0, 0, 0), "decode", False), False)
        m.add(S.Request(0, 0, 0, (0, 0, 1), "decode", True), True)
        self.assertEqual(m.rates()["slice_all_all"], 1.0)


# -------------------------------------------------------------------- 14

class TestGateMetric(unittest.TestCase):
    def test_gate_vs_all_nodes_vs_expert_event(self):
        m = S.Metrics()
        R = S.Request
        # gate node: expert 1 gets 2 of 3 slices -> partial; expert 2 all hit
        for kind, hit in ((0, True), (1, True), (2, False)):
            m.add(R(0, 0, 0, (0, 1, kind), "decode", True), hit)
        for kind in (0, 1, 2):
            m.add(R(0, 0, 0, (0, 2, kind), "decode", True), True)
        # up node re-requests: always hits, counted only in slice_all
        for e in (1, 2):
            for kind in (0, 1, 2):
                m.add(R(1, 0, 1, (0, e, kind), "decode", True), True)
        r = m.rates()
        self.assertAlmostEqual(r["slice_gate_decode"], 5 / 6)
        self.assertAlmostEqual(r["slice_all_decode"], 11 / 12)
        self.assertAlmostEqual(r["expert_event_decode"], 1 / 2)
        self.assertEqual(r["partial_expert_events_decode"], 1)


# -------------------------------------------------------------------- 15

class TestInFlight(unittest.TestCase):
    def test_waited_outcome_counts_as_hit_and_matches_replay(self):
        d = TMP / "w"
        d.mkdir(exist_ok=True)
        lines = ["H\targine-raw\t2\t0\t0\t1", "C\tbudget_bytes\t100000", "C\tcache_shards\t1",
                 "R\t0\t1\t0\t1\t1\t4", "N\t0\t0\t0\t1\t100\t1\t2",
                 "A\t10\t0\t0\t4\t0\tm\t4096\t1\t0\t0",
                 "A\t11\t1\t0\t4\t0\tw\t4096\t0\t0\t1"]      # 2nd thread waited for the read
        (d / "raw.tsv").write_text("\n".join(lines) + "\n")
        (d / "markers.jsonl").write_text("")
        st = F.load_runtime_stream(d)
        res = F.compare(st)
        self.assertEqual(res["outcome_disagreements"], 0)
        self.assertAlmostEqual(res["runtime_metrics"]["slice_all_all"], 0.5)


# -------------------------------------------------------------------- 16

class TestOptimalAllocation(unittest.TestCase):
    SIZES_30B = {l: (3059712 if l % 3 == 0 or l < 10 else 2654208) for l in range(48)}

    def curves(self, seed=0, n=40):
        rng = random.Random(seed)
        out = {}
        for l in range(48):
            c = [1000.0]
            for _ in range(n):
                c.append(max(0.0, c[-1] - rng.randint(0, 40) * (1 + l % 5)))
            out[l] = c
        return out

    def test_never_worse_than_uniform_with_the_sizes_that_broke_it(self):
        """Before 2026-10-03 the DP rounded 2.65 MB experts up to 3 MiB at
        4 GiB and lost to uniform (38,206 vs 36,473 misses)."""
        for seed in range(3):
            cur = self.curves(seed)
            for budget in (4 << 30, 2 << 30, 3 << 29):
                uni = allocator.uniform(list(range(48)), self.SIZES_30B, budget, 40, 8)
                opt = allocator.dp_optimal(cur, self.SIZES_30B, budget, 8)
                self.assertLessEqual(allocator.total_cost(cur, opt),
                                     allocator.total_cost(cur, uni) + 1e-9)
                self.assertLessEqual(sum(c * self.SIZES_30B[l] for l, c in opt.items()), budget)

    def test_exact_against_brute_force_with_unequal_sizes(self):
        rng = random.Random(9)
        for _ in range(20):
            cur = {}
            for l in range(3):
                c = [100.0]
                for _ in range(5):
                    c.append(max(0.0, c[-1] - rng.randint(0, 30)))
                cur[l] = c
            sizes = {0: rng.choice([3, 4, 5]), 1: rng.choice([3, 4, 7]), 2: rng.choice([2, 5])}
            budget = rng.randint(8, 30)
            best = min(sum(cur[l][c[l]] for l in range(3))
                       for c in itertools.product(range(6), repeat=3)
                       if sum(c[l] * sizes[l] for l in range(3)) <= budget)
            got = allocator.dp_optimal(cur, sizes, budget)
            self.assertAlmostEqual(allocator.total_cost(cur, got), best)

    def test_checked_variant_raises_on_violation(self):
        cur = {0: [10.0, 5.0, 0.0], 1: [10.0, 5.0, 0.0]}
        sizes = {0: 1, 1: 1}
        orig = allocator.dp_optimal
        try:
            allocator.dp_optimal = lambda *a, **k: {0: 0, 1: 0}      # a broken solver
            with self.assertRaises(allocator.AllocationInvariantError):
                allocator.dp_optimal_checked(cur, sizes, 4, 0, {"uniform": {0: 2, 1: 2}})
        finally:
            allocator.dp_optimal = orig

    def test_heterogeneity_on_real_size_classes(self):
        tr = mixed_trace(layers=6, n_exp=16, segments=True)
        for l in tr.layers:
            tr.meta.expert_bytes[l] = 3059712 if l % 2 else 2654208
        curves = elasticity.lru_curves(tr, "decode")
        for b in (40_000_000, 80_000_000):
            h = elasticity.heterogeneity(tr, curves, b, "decode")
            self.assertLessEqual(h["misses_optimal_static"], h["misses_uniform"])


# -------------------------------------------------------------------- 18

class TestDeterminism(unittest.TestCase):
    def test_replays_are_deterministic(self):
        tr = mixed_trace()
        sizes = {(l, k): 1000 for l in range(3) for k in range(3)}
        out = []
        for _ in range(2):
            rp = S.StoreReplay(12000, S.HashShardMap(4, sizes, 5), sizes_fn(sizes), 4)
            m, _ = S.run(S.synthetic_stream(tr), rp)
            out.append((m.rates(), rp.evictions))
        self.assertEqual(out[0], out[1])
        a = replay(tr, policies.ShardedLRU(4), 9000, phase="decode")
        b = replay(tr, policies.ShardedLRU(4), 9000, phase="decode")
        self.assertEqual((a.hits, a.misses, a.evictions), (b.hits, b.misses, b.evictions))


# ------------------------------------------------------------ v2 curves

class TestCurvesUnderV2(unittest.TestCase):
    def test_stack_curves_equal_replay_with_overflowing_prompt_steps(self):
        tr = mixed_trace(prompt_tokens=8, n_exp=12)       # prompt events up to 12 > slots
        cur = elasticity.lru_curves(tr, "decode")
        for c in (2, 3, 5, 8, 12):
            alloc = {l: c * 1000 for l in tr.layers}
            r = replay(tr, policies.PerLayerLRU(alloc), sum(alloc.values()), phase="decode")
            self.assertEqual(r.misses, sum(cur[l][c] for l in tr.layers), c)

    def test_min_slots_uses_counted_events_only(self):
        tr = mixed_trace(prompt_tokens=8)
        self.assertEqual(elasticity.min_slots(tr, "decode"), 2)


if __name__ == "__main__":
    unittest.main()
