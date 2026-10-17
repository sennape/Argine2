"""EXP-024 harness: frozen schedule, ON/OFF validation, paired analysis and decision rule.

No model, no inference, no subprocess, no llama-server: only the pure functions of
tools/exp024_run.py are exercised (trace files are written to a temporary directory).
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import exp024_run as x  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="argine2-exp024-"))

FROZEN = [("4G", "OFF"), ("4G", "ON"), ("1.5G", "OFF"), ("1.5G", "ON"),
          ("1.5G", "ON"), ("1.5G", "OFF"), ("4G", "ON"), ("4G", "OFF"),
          ("4G", "ON"), ("4G", "OFF"), ("1.5G", "ON"), ("1.5G", "OFF"),
          ("1.5G", "OFF"), ("1.5G", "ON"), ("4G", "OFF"), ("4G", "ON"),
          ("4G", "OFF"), ("4G", "ON"), ("1.5G", "OFF"), ("1.5G", "ON")]


def record(launch, gen=6.0, sid="S1", valid=True, attempt=1):
    return {"event": "benchmark", "model": {"path": x.MODEL_PATH, "file_size": 123},
            "plan": {"threads": 4, "threads_batch": 12, "backend": "cpu", "gpu_layers": 0,
                     "expert_store": True, "expert_cache_budget": launch["budget_bytes"]},
            "runtime": {"backend": "cpu", "build": "llama-research-v2@a0063720"},
            "metrics": {"runs": [{"gen_tok_s": gen, "prompt_tok_s": 20.0, "ttft_s": 3.0}], "errors": [],
                        "resources": {"ram_peak": 5, "cpu_mean_percent": 300.0}},
            "expert_cache": {"hit_pct": 90.0}, "modo_lettura": "asincrono", "repeat": 1,
            "n_predict": 128, "prompt_chars": 304, "attempts": [],
            "exp024": {"session_id": sid, "launch_index": launch["launch_index"], "attempt": attempt,
                       "round": launch["round"], "budget": launch["budget"],
                       "condition": launch["condition"], "valid": valid, "problems": []}}


def full_session(off=6.0, on_by_round=None, sid="S1"):
    """20 valid records; ON throughput per round given as overhead percentages."""
    on_by_round = on_by_round or {b: [1.0] * 5 for b in x.BUDGETS}
    out = []
    for l in x.build_schedule():
        if l["condition"] == "OFF":
            g = off
        else:
            g = off * (1 - on_by_round[l["budget"]][l["round"] - 1] / 100.0)
        out.append(record(l, gen=g, sid=sid))
    return out


class TestSchedule(unittest.TestCase):
    def setUp(self):
        self.s = x.build_schedule()

    def test_exactly_20_launches_and_frozen_order(self):
        self.assertEqual(len(self.s), 20)
        self.assertEqual([(l["budget"], l["condition"]) for l in self.s], FROZEN)
        self.assertEqual([l["launch_index"] for l in self.s], list(range(1, 21)))
        self.assertEqual(x.check_schedule(self.s), [])

    def test_five_per_budget_and_condition(self):
        for b in ("4G", "1.5G"):
            for c in ("OFF", "ON"):
                self.assertEqual(sum(1 for l in self.s if l["budget"] == b and l["condition"] == c), 5)

    def test_pairs_adjacent_in_every_round_and_order_varied(self):
        for rnd in range(1, 6):
            r = sorted([l for l in self.s if l["round"] == rnd], key=lambda l: l["position_in_round"])
            self.assertEqual(len(r), 4)
            for b in ("4G", "1.5G"):
                pos = [l["position_in_round"] for l in r if l["budget"] == b]
                self.assertEqual(abs(pos[0] - pos[1]), 1)
        bal = x.order_balance(self.s)
        self.assertEqual(bal["condition_first_per_budget"], {"4G": {"OFF": 3, "ON": 2}, "1.5G": {"OFF": 3, "ON": 2}})
        self.assertEqual(bal["budget_first_in_round"], {"4G": 3, "1.5G": 2})

    def test_check_detects_tampering(self):
        bad = [dict(l) for l in self.s]
        bad[0], bad[1] = dict(bad[1], launch_index=1), dict(bad[0], launch_index=2)
        self.assertTrue(x.check_schedule(bad))
        self.assertTrue(x.check_schedule(self.s[:-1]))

    def test_commands_identical_across_conditions_and_fixed_flags(self):
        for b in ("4G", "1.5G"):
            a = x.benchmark_args(b)
            for flag, val in (("--repeat", "1"), ("--n-predict", "128"), ("--retries", "0"),
                              ("--backend", "cpu"), ("--read-mode", "asincrono"), ("--threads", "4"),
                              ("--expert-budget", b)):
                self.assertEqual(a[a.index(flag) + 1], val)
            self.assertNotIn("--dry-run", a)
            self.assertNotIn("--research-trace", " ".join(a))

    def test_trace_names_encode_index_round_budget_condition(self):
        on = [l for l in self.s if l["condition"] == "ON"]
        self.assertEqual(on[0]["trace_file"], "raw/traces/launch-02-r1-4G-ON.tsv")
        self.assertTrue(all(l["trace_file"] is None for l in self.s if l["condition"] == "OFF"))
        self.assertEqual(len({l["would_be_trace_file"] for l in self.s}), 20)

    def test_generic_interleaver_is_not_used(self):
        """The repository's interleaved_schedule puts OFF first in every pair: not AB/BA."""
        from argine.research import stats
        gen = stats.interleaved_schedule(["OFF", "ON"], 5)
        self.assertEqual(gen[::2], ["OFF"] * 5)


class TestEnvironment(unittest.TestCase):
    def setUp(self):
        self.llama = ROOT / "llama-research-v2"
        self.base = x.base_env({"PATH": "p"}, self.llama)

    def test_on_sets_off_removes_trace_variable(self):
        s = x.build_schedule()
        on = x.launch_env(dict(self.base, ARGINE_RESEARCH_TRACE="stale"), s[1], TMP)
        off = x.launch_env(dict(self.base, ARGINE_RESEARCH_TRACE="stale"), s[0], TMP)
        self.assertTrue(on["ARGINE_RESEARCH_TRACE"].endswith("launch-02-r1-4G-ON.tsv"))
        self.assertNotIn("ARGINE_RESEARCH_TRACE", off)
        self.assertEqual(set(x.env_delta(self.base, on)), {"ARGINE_RESEARCH_TRACE"})
        self.assertEqual(x.env_delta(self.base, off), {})

    def test_unexpected_runtime_variables_are_refused(self):
        for k in ("ARGINE_RESEARCH_TRACE", "ARGINE_EXPERT_PRETOUCH", "ARGINE_PREFETCH", "PICOSTAR_X"):
            with self.assertRaises(x.ProtocolError):
                x.base_env({k: "1"}, self.llama)

    def test_other_llama_dir_is_refused(self):
        with self.assertRaises(x.ProtocolError):
            x.base_env({"ARGINE_LLAMA_DIR": str(TMP)}, self.llama)


class TestTraceValidation(unittest.TestCase):
    def setUp(self):
        self.s = x.build_schedule()
        self.dir = TMP / "traces"
        self.dir.mkdir(exist_ok=True)

    def _on(self, n):
        l = self.s[1]
        p = self.dir / f"on{n}.tsv"
        return l, p

    def test_on_valid(self):
        l, p = self._on(1)
        p.write_text("H\targine-raw\t2\t1\t2\t3\nC\tx\ty\n")
        log = f"  argine: expert store aperto: ...\n  argine: research trace attiva: {p}\n"
        self.assertEqual(x.check_trace_condition(l, log, p, [], [p.name]), [])

    def test_on_without_confirmation_or_wrong_path_or_empty(self):
        l, p = self._on(2)
        p.write_text("H\targine-raw\t2\t1\t2\t3\n")
        self.assertTrue(x.check_trace_condition(l, "no line", p, [], [p.name]))
        self.assertTrue(x.check_trace_condition(l, f"argine: research trace attiva: {p}.other", p, [], [p.name]))
        two = f"argine: research trace attiva: {p}\nargine: research trace attiva: {p}\n"
        self.assertTrue(x.check_trace_condition(l, two, p, [], [p.name]))
        e = self.dir / "empty.tsv"
        e.write_text("")
        self.assertTrue(x.check_trace_condition(l, f"argine: research trace attiva: {e}", e, [], [e.name]))
        missing = self.dir / "never.tsv"
        self.assertTrue(x.check_trace_condition(l, f"argine: research trace attiva: {missing}", missing, [], []))

    def test_on_bad_header_or_cannot_open(self):
        l, p = self._on(3)
        p.write_text("garbage\n")
        self.assertTrue(x.check_trace_condition(l, f"argine: research trace attiva: {p}", p, [], [p.name]))
        p.write_text("H\targine-raw\t2\t1\n")
        log = f"argine: research trace attiva: {p}\nargine: research trace: cannot open X\n"
        self.assertTrue(x.check_trace_condition(l, log, p, [], [p.name]))

    def test_off_valid_and_violations(self):
        l = self.s[0]
        p = self.dir / "off-would-be.tsv"
        self.assertEqual(x.check_trace_condition(l, "argine: expert store aperto", p, ["a"], ["a"]), [])
        self.assertTrue(x.check_trace_condition(l, "argine: research trace attiva: q", p, [], []))
        self.assertTrue(x.check_trace_condition(l, "", p, [], ["new.tsv"]))
        p.write_text("H\n")
        self.assertTrue(x.check_trace_condition(l, "", p, [], []))

    def test_store_log(self):
        ok = ("  argine: expert store aperto: " + x.STORE_PATH + ", 48 layer x 128 expert, lettura asincrono\n"
              "  argine: expert store IN USO: primo nodo MoE servito\n")
        self.assertEqual(x.check_store_log(ok), [])
        self.assertTrue(x.check_store_log(ok.replace("asincrono", "sincrono")))
        self.assertTrue(x.check_store_log(ok.replace(x.STORE_PATH, r"C:\other.psxs")))
        self.assertTrue(x.check_store_log(ok.replace("IN USO", "")))


class TestRecordValidation(unittest.TestCase):
    def setUp(self):
        self.l = x.build_schedule()[2]          # 1.5G OFF

    def test_valid(self):
        self.assertEqual(x.validate_record(record(self.l), self.l, 0, "a0063720", 123, 304), [])

    def test_each_violation_is_caught(self):
        bad = {
            "threads": lambda r: r["plan"].update(threads=12),
            "batch": lambda r: r["plan"].update(threads_batch=8),
            "backend": lambda r: r["runtime"].update(backend="vulkan"),
            "gpu": lambda r: r["plan"].update(gpu_layers=49),
            "budget": lambda r: r["plan"].update(expert_cache_budget=4 * 1024 ** 3),
            "store": lambda r: r["plan"].update(expert_store=False),
            "read": lambda r: r.update(modo_lettura="blocco"),
            "repeat": lambda r: r.update(repeat=5),
            "npredict": lambda r: r.update(n_predict=64),
            "prompt": lambda r: r.update(prompt_chars=10),
            "retry": lambda r: r.update(attempts=[{"attempt": 1}]),
            "errors": lambda r: r["metrics"].update(errors=["x"]),
            "notoks": lambda r: r["metrics"]["runs"][0].update(gen_tok_s=None),
            "model": lambda r: r["model"].update(path=r"C:\other.gguf"),
            "size": lambda r: r["model"].update(file_size=1),
            "build": lambda r: r["runtime"].update(build="llama-research-v2@deadbeef"),
        }
        for name, f in bad.items():
            r = record(self.l)
            f(r)
            self.assertTrue(x.validate_record(r, self.l, 0, "a0063720", 123, 304), name)
        self.assertTrue(x.validate_record(record(self.l), self.l, 1, "a0063720", 123, 304))
        self.assertTrue(x.validate_record(None, self.l, 0, "a0063720", 123, 304))


class TestAnalysis(unittest.TestCase):
    def test_sign_convention(self):
        self.assertAlmostEqual(x.overhead_pct(10.0, 9.8), 2.0)
        self.assertAlmostEqual(x.overhead_pct(10.0, 10.1), -1.0)
        with self.assertRaises(x.ProtocolError):
            x.overhead_pct(0.0, 1.0)

    def test_student_t_on_known_input(self):
        s = x.t_summary([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual((s["n"], s["df"]), (5, 4))
        self.assertAlmostEqual(s["mean"], 3.0)
        self.assertAlmostEqual(s["sd"], 1.5811388300841898)
        self.assertAlmostEqual(s["t_crit"], 2.776445105)
        self.assertAlmostEqual(s["ci_low"], 3.0 - 2.776445105 * 1.5811388300841898 / 5 ** 0.5)
        self.assertAlmostEqual(s["ci_high"], 4.963243, places=5)

    def test_pairing_by_round(self):
        ov = {"4G": [0.5, 1.0, 1.5, 2.0, 2.5], "1.5G": [-1.0, 0.0, 1.0, 0.0, 0.0]}
        pairs = x.paired_observations(full_session(on_by_round=ov), "S1")
        for b in x.BUDGETS:
            self.assertEqual([p["round"] for p in pairs[b]], [1, 2, 3, 4, 5])
            for p, want in zip(pairs[b], ov[b]):
                self.assertAlmostEqual(p["overhead_pct"], want)
                launches = {l["launch_index"]: l for l in x.build_schedule()}
                self.assertEqual(launches[p["off_launch"]]["round"], launches[p["on_launch"]]["round"])
                self.assertEqual(p["delta_ms_per_token"] > 1e-12, want > 0)   # slower ON = more ms/token

    def test_duplicates_retries_foreign_and_incomplete_are_refused(self):
        recs = full_session()
        with self.assertRaises(x.ProtocolError):
            x.paired_observations(recs + [recs[3]], "S1")                 # duplicate launch
        r = full_session()
        r[5]["exp024"]["attempt"] = 2
        with self.assertRaises(x.ProtocolError):
            x.paired_observations(r, "S1")                                # retry
        r = full_session()
        r[0]["exp024"]["session_id"] = "OTHER"
        with self.assertRaises(x.ProtocolError):
            x.paired_observations(r, "S1")                                # foreign session
        with self.assertRaises(x.ProtocolError):
            x.paired_observations(full_session()[:-1], "S1")              # incomplete
        r = full_session()
        r[7]["exp024"]["valid"] = False
        with self.assertRaises(x.ProtocolError):
            x.paired_observations(r, "S1")                                # invalid launch
        r = full_session()
        r[0]["exp024"]["condition"] = "ON"
        with self.assertRaises(x.ProtocolError):
            x.paired_observations(r, "S1")                                # schedule mismatch

    def test_analyze_end_to_end_on_toy_data(self):
        ov = {"4G": [0.5, 0.6, 0.4, 0.5, 0.5], "1.5G": [1.0, 1.1, 0.9, 1.0, 1.0]}
        text, summary = x.analyze(full_session(on_by_round=ov), "S1")
        self.assertEqual(summary["decision"]["verdict"], "SUPPORTED")
        self.assertEqual(len(text.strip().splitlines()), 11)              # header + 10 pairs
        self.assertAlmostEqual(summary["budgets"]["4G"]["overhead_pct"]["mean"], 0.5)
        self.assertEqual(summary["budgets"]["4G"]["gen_tok_s_off"]["n"], 5)


class TestDecisionRule(unittest.TestCase):
    def ci(self, lo, hi):
        return {"ci_low": lo, "ci_high": hi}

    def test_supported(self):
        d = x.decide({"4G": self.ci(-0.5, 1.9), "1.5G": self.ci(0.1, 1.99)})
        self.assertEqual(d["verdict"], "SUPPORTED")

    def test_falsified_at_one_budget(self):
        d = x.decide({"4G": self.ci(0.0, 1.0), "1.5G": self.ci(2.01, 4.0)})
        self.assertEqual(d["verdict"], "FALSIFIED")

    def test_inconclusive_when_crossing_or_touching(self):
        self.assertEqual(x.decide({"4G": self.ci(0.0, 1.0), "1.5G": self.ci(1.0, 3.0)})["verdict"], "INCONCLUSIVE")
        self.assertEqual(x.decide({"4G": self.ci(0.0, 2.0), "1.5G": self.ci(0.0, 1.0)})["verdict"], "INCONCLUSIVE")
        self.assertEqual(x.decide({"4G": self.ci(2.0, 3.0), "1.5G": self.ci(0.0, 1.0)})["verdict"], "INCONCLUSIVE")

    def test_threshold_frozen_and_budgets_required(self):
        self.assertEqual(x.THRESHOLD_PCT, 2.0)
        with self.assertRaises(x.ProtocolError):
            x.decide({"4G": self.ci(0, 1)})


def clean_snapshot(**kw):
    s = {"available": True, "head": "9b842d6e1e6d2508ccf244523384502b0ebbb295",
         "tracked": {f: True for f in x.REQUIRED_TRACKED}, "porcelain": [], "error": None}
    s.update(kw)
    return s


class TestRunGate(unittest.TestCase):
    """Real --run requires a committed, clean, frozen protocol (independent review 2026-10-12).
    Git state is injected: no git process, no model, no subprocess."""

    OK = (True, "matches the frozen protocol")

    def test_clean_committed_state_accepted(self):
        self.assertEqual(x.run_gate_problems(clean_snapshot(), *self.OK), [])
        g = x.require_run_gate(snapshot_fn=clean_snapshot, freeze_fn=lambda: self.OK)
        self.assertEqual(g["head"], "9b842d6e1e6d2508ccf244523384502b0ebbb295")
        self.assertTrue(g["porcelain_clean"])

    def test_required_files_are_exactly_the_protocol_files(self):
        self.assertEqual(set(x.REQUIRED_TRACKED), {
            "experiments/EXP-024-tracing-overhead/README.md",
            "experiments/EXP-024-tracing-overhead/config.json",
            "experiments/EXP-024-tracing-overhead/protocol_freeze.json",
            "tools/exp024_run.py", "tests/test_exp024.py"})

    def test_dirty_tracked_tree_rejected(self):
        bad = x.run_gate_problems(clean_snapshot(porcelain=[" M tools/exp024_run.py"]), *self.OK)
        self.assertTrue(any("not clean" in b for b in bad))

    def test_untracked_file_rejected(self):
        bad = x.run_gate_problems(clean_snapshot(porcelain=["?? notes.txt"]), *self.OK)
        self.assertTrue(any("not clean" in b for b in bad))

    def test_unreadable_status_rejected(self):
        self.assertTrue(x.run_gate_problems(clean_snapshot(porcelain=None), *self.OK))

    def test_missing_head_rejected(self):
        bad = x.run_gate_problems(clean_snapshot(head=None, error="unknown revision"), *self.OK)
        self.assertTrue(any("HEAD" in b for b in bad))

    def test_git_unavailable_rejected(self):
        bad = x.run_gate_problems({"available": False, "error": "git unavailable: timeout"}, *self.OK)
        self.assertTrue(bad)

    def test_each_untracked_protocol_file_rejected(self):
        for f in x.REQUIRED_TRACKED:
            tracked = {g: True for g in x.REQUIRED_TRACKED}
            tracked[f] = False
            bad = x.run_gate_problems(clean_snapshot(tracked=tracked), *self.OK)
            self.assertEqual(bad, [f"not tracked by Git: {f}"], f)
            missing = {g: True for g in x.REQUIRED_TRACKED if g != f}
            self.assertTrue(x.run_gate_problems(clean_snapshot(tracked=missing), *self.OK), f)

    def test_freeze_mismatch_rejected(self):
        bad = x.run_gate_problems(clean_snapshot(), False, "CHANGED since freeze: tool_sha256")
        self.assertEqual(bad, ["freeze: CHANGED since freeze: tool_sha256"])
        with self.assertRaises(x.ProtocolError):
            x.require_run_gate(snapshot_fn=clean_snapshot, freeze_fn=lambda: (False, "CHANGED"))

    def test_failed_gate_reaches_no_subprocess_session_or_trace(self):
        import argparse
        from unittest import mock

        def forbidden(*a, **k):
            raise AssertionError("a subprocess was reached although the gate failed")
        fail_snaps = [clean_snapshot(porcelain=["?? x"]), clean_snapshot(head=None),
                      clean_snapshot(tracked={})]
        for snap in fail_snaps:
            with mock.patch.object(x, "git_snapshot", return_value=snap), \
                    mock.patch.object(x.subprocess, "run", side_effect=forbidden), \
                    mock.patch.object(x, "preflight", side_effect=forbidden), \
                    mock.patch.object(x, "session_provenance", side_effect=forbidden), \
                    mock.patch.object(x, "_write_json", side_effect=forbidden), \
                    mock.patch.object(x, "freeze_status", return_value=self.OK):
                rc = x.run(argparse.Namespace(llama_dir=str(ROOT / "llama-research-v2")))
            self.assertEqual(rc, 2)
        with mock.patch.object(x, "git_snapshot", return_value=clean_snapshot()), \
                mock.patch.object(x.subprocess, "run", side_effect=forbidden), \
                mock.patch.object(x, "preflight", side_effect=forbidden), \
                mock.patch.object(x, "_write_json", side_effect=forbidden), \
                mock.patch.object(x, "freeze_status", return_value=(False, "CHANGED")):
            self.assertEqual(x.run(argparse.Namespace(llama_dir=str(ROOT / "llama-research-v2"))), 2)

    def test_dry_run_and_preflight_do_not_apply_the_gate(self):
        from unittest import mock
        with mock.patch.object(x, "require_run_gate", side_effect=AssertionError("gate called")):
            self.assertIn(x.print_dry_run(), (0, 1))


if __name__ == "__main__":
    unittest.main()
