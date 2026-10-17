"""Tests for tools/exp022_run.py (EXP-022 protocol EXP022-P1).

No test starts a model, llama-server, a benchmark, a research trace or a replay: every subprocess
call made by the code under test is intercepted and inspected.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("exp022_run", ROOT / "tools" / "exp022_run.py")
X = importlib.util.module_from_spec(_spec)
sys.modules["exp022_run"] = X
_spec.loader.exec_module(X)

MIB = 1 << 20


class Recorder:
    """Stands in for subprocess.run / Popen; records argv; only `git` is ever executed for real."""

    def __init__(self, fake_stdout: str = "", allow_git: bool = True):
        self.calls = []
        self.fake_stdout = fake_stdout
        self.allow_git = allow_git
        self._real = subprocess.run

    def run(self, argv, *a, **kw):
        self.calls.append(list(argv))
        if argv and argv[0] == "git" and self.allow_git:
            return self._real(argv, *a, **kw)
        return subprocess.CompletedProcess(argv, 0, stdout=self.fake_stdout, stderr="")

    def popen(self, argv, *a, **kw):
        self.calls.append(list(argv))
        raise AssertionError(f"Popen must never be called: {argv}")


def _quiet(fn, *a):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = fn(*a)
    return rc, out.getvalue() + err.getvalue()


def _item(model="30b", label="768M", phase="perf"):
    return next(x for x in X.build_plan() if x["model"] == model and x["budget"] == label and x["phase"] == phase)


def _llama_cmd(item, **override):
    flags = X.expected_llama_flags(item)
    flags.update(override)
    s = f"C:\\Argine2\\llama-research-v2\\llama-server.exe --model {X.MODELS[item['model']]['path']} --host 127.0.0.1"
    for k, v in flags.items():
        s += f" {k} {v}"
    return s + " --n-cpu-moe 48 --cache-ram 0 --jinja"


def _segment(item, shards=None, starts=1, abort=False, warn=False):
    m = X.MODELS[item["model"]]
    lines = []
    for _ in range(starts):
        lines.append("===== 2026-10-14 10:00:00 :: " + _llama_cmd(item))
    if warn:
        lines.append("argine: budget 512 MiB sotto il minimo teorico 792 MiB (tutti gli expert di un layer).")
    lines.append(f"argine: expert store aperto: {m['store']}, {m['n_layers']} layer x {m['n_experts']} expert, "
                 f"budget {item['budget_bytes'] >> 20} MiB, {shards or X.shard_count(item['budget_bytes'])} settori "
                 "di cache, cache dell'OS scavalcata, lettura asincrono")
    if abort:
        lines.append("ggml.c:123: argine: impossibile rendere residenti gli expert del layer 3 (blk.3.ffn_gate_exps.weight). "
                     "Residenti 511 MiB su un budget di 512 MiB, questo nodo ne chiede 300 su 512.")
    return "\n".join(lines) + "\n"


def _perf_record(item, gen=3.9, tokens=128, errors=(), budget=None, resident=None):
    b = item["budget_bytes"] if budget is None else budget
    return {"event": "benchmark", "model": {"path": X.MODELS[item["model"]]["path"]},
            "plan": {"threads": item["threads"], "threads_batch": 12, "ctx": 4096, "batch": 2048, "ubatch": 512,
                     "kv_type_k": "f16", "kv_type_v": "f16", "flash_attn": True, "parallel": 1, "backend": "cpu",
                     "gpu_layers": 0, "expert_store": True, "expert_cache_budget": b},
            "runtime": {"backend": "cpu", "build": "llama-research-v2@a0063720"},
            "modo_lettura": "asincrono", "repeat": 1, "n_predict": 128, "prompt_chars": 304, "attempts": [],
            "metrics": {"runs": [{"gen_tok_s": gen, "generated_tokens": tokens, "ttft_s": 3.5, "prompt_tok_s": 22.0}],
                        "errors": list(errors), "resources": {"ram_peak": 3 << 30}},
            "expert_cache": {"mib_budget": b >> 20, "mib_residenti": (b >> 20) - 8 if resident is None else resident,
                             "hit_pct": 50.0}}


def _rec(block, idx, budget, outcome, rate=None, eligible=False):
    model, phase = block.split("-")
    return {"exp022": {"block": block, "model": model, "phase": phase, "launch_index": idx, "round": 1,
                       "position_in_round": 1, "budget": budget, "budget_bytes": X.budget_bytes(model, budget),
                       "fraction_of_footprint": 1.0, "outcome": outcome, "reason": "",
                       "expert_level_hit_rate": rate, "h5_eligible": eligible},
            "metrics": {"runs": [{"gen_tok_s": 3.0}]} if phase == "perf" else None}


class BudgetsAndFootprints(unittest.TestCase):
    def test_exact_bytes_and_shards(self):
        want = {"30b": [1610612736, 1207959552, 1073741824, 805306368, 536870912],
                "80b": [1610612736, 805306368, 704643072, 536870912]}
        for m, bs in want.items():
            self.assertEqual([b for _, b, _ in X.BUDGETS[m]], bs)
            for _, b, _ in X.BUDGETS[m]:
                self.assertEqual(X.shard_count(b), 16)
        self.assertEqual(X.check_budgets(), [])
        self.assertEqual(X.shard_count(X.EXCLUDED_BUDGETS["80b-336M"]["bytes"]), 10)
        self.assertNotIn("336M", [lab for lab, _, _ in X.BUDGETS["80b"]])

    def test_fractions_of_footprint(self):
        f30 = [round(t["fraction_of_footprint"], 3) for t in X.budget_table("30b").values()]
        f80 = [round(t["fraction_of_footprint"], 3) for t in X.budget_table("80b").values()]
        self.assertEqual(f30, [1.468, 1.101, 0.979, 0.734, 0.489])
        self.assertEqual(f80, [2.238, 1.119, 0.979, 0.746])

    def test_footprints_from_tracked_evidence(self):
        d = X.derive_repo_footprints()
        self.assertEqual(d["30b"]["footprint_bytes"], 1097072640)
        self.assertEqual(d["80b"]["footprint_bytes"], 719585280)
        self.assertEqual(d["30b"]["whole_layer_bytes"], 391643136)
        self.assertEqual(d["80b"]["whole_layer_bytes"], 830472192)
        self.assertEqual(X.check_footprints(), [])
        self.assertAlmostEqual(1097072640 / MIB, 1046.2, places=1)
        self.assertAlmostEqual(719585280 / MIB, 686.2, places=1)

    def test_footprint_arithmetic(self):
        meta = {"expert_bytes": {"0": 100, "1": 200}, "top_k": 3, "n_experts": 10}
        self.assertEqual(X.footprint_from_meta(meta),
                         {"layers": 2, "top_k": 3, "n_experts": 10, "footprint_bytes": 900, "whole_layer_bytes": 2000})

    def test_h5_point_is_in_the_table(self):
        self.assertEqual(X.budget_bytes("30b", "768M"), X.H5["budget_bytes"])
        with self.assertRaises(X.ProtocolError):
            X.budget_bytes("80b", "336M")


class SourcesAndConventions(unittest.TestCase):
    def test_runtime_sources_certified(self):
        self.assertEqual(X.check_runtime_sources(), [])

    def test_timeouts_and_caps(self):
        self.assertEqual(X.check_timeout_sources(), [])
        self.assertEqual(X.cap_seconds("perf"), 2400)
        self.assertEqual(X.cap_seconds("diag"), 6000)

    def test_workloads(self):
        self.assertEqual(X.check_workloads(), [])

    def test_readme(self):
        self.assertEqual(X.check_readme(), [])
        self.assertIn(X.HISTORICAL_HYPOTHESIS, X.README.read_text(encoding="utf-8").replace("\r\n", "\n"))

    def test_usability_conventions(self):
        c = X.usability_category
        self.assertEqual(c(5.0), "INTERACTIVE")
        self.assertEqual(c(4.999), "DEGRADED")
        self.assertEqual(c(1.0), "DEGRADED")
        self.assertEqual(c(0.999), "NON_INTERACTIVE")
        self.assertIsNone(c(None))
        conv = X.protocol_spec()["usability_reporting_conventions"]
        self.assertEqual(set(conv) - {"note"}, {"INTERACTIVE", "DEGRADED", "NON_INTERACTIVE"})
        self.assertEqual([name for name, _ in X.USABILITY], ["INTERACTIVE", "DEGRADED", "NON_INTERACTIVE"])


class Plan(unittest.TestCase):
    def test_structure(self):
        p = X.build_plan()
        self.assertEqual(len(p), 45)
        self.assertEqual([sum(1 for x in p if x["block"] == b) for b in X.BLOCK_ORDER], [15, 10, 12, 8])
        self.assertEqual(X.check_plan(p), [])
        for m in X.MODEL_ORDER:
            self.assertEqual(X.DIAG_ROUNDS[m], X.PERF_ROUNDS[m][:2])
            for r in X.PERF_ROUNDS[m]:
                self.assertIn("1.5G", r)
            pairs = [(r[i], r[i + 1]) for r in X.PERF_ROUNDS[m] for i in range(len(r) - 1)]
            self.assertEqual(len(pairs), len(set(pairs)))
        self.assertEqual(sum(1 for x in p if x["block"] == "30b-diag" and x["budget"] == "768M"), 2)

    def test_commands(self):
        a = _item("30b", "512M", "perf")["argv"]
        self.assertEqual(a[:2], ["benchmark", X.MODELS["30b"]["path"]])
        for f, v in (("--threads", "12"), ("--expert-budget", "512M"), ("--ctx", "4096"), ("--ubatch", "512"),
                     ("--retries", "0"), ("--repeat", "1"), ("--backend", "cpu"), ("--read-mode", "asincrono")):
            self.assertEqual(a[a.index(f) + 1], v)
        d = _item("80b", "672M", "diag")["argv"]
        self.assertEqual(d[:2], ["research", "trace"])
        for f, v in (("--threads", "4"), ("--budgets", "672M"), ("--split", "eval"), ("--workload", "natural-language")):
            self.assertEqual(d[d.index(f) + 1], v)
        self.assertNotIn("--dry-run", a + d)

    def test_tampering_detected(self):
        p = X.build_plan()
        p[0]["argv"] = [v if v != "12" else "4" for v in p[0]["argv"]]
        self.assertTrue(any("--threads" in s for s in X.check_plan(p)))


class Classification(unittest.TestCase):
    def test_precedence(self):
        c = X.classify
        self.assertEqual(c(["bad"], [], [], ["abort"], True, 0, False, "")["outcome"], "INVALID")
        self.assertEqual(c([], ["viol"], [], [], True, 0, False, "")["outcome"], "INVALID")
        self.assertEqual(c([], [], [], ["argine: impossibile"], False, 1, True, "")["outcome"], "STORE_RESIDENCY_ABORT")
        self.assertEqual(c(["bad"], [], [], ["argine: impossibile"], False, 1, False, "")["outcome"], "INVALID")
        self.assertEqual(X.OUTCOMES, ("VALID", "STORE_RESIDENCY_ABORT", "NON_COMPLETION", "INVALID"))
        self.assertNotIn("INFEASIBLE", X.OUTCOMES)
        tax = X.protocol_spec()["outcome_taxonomy"]
        self.assertNotIn("INFEASIBLE", tax)
        self.assertEqual(tax["precedence"], "INVALID > STORE_RESIDENCY_ABORT > NON_COMPLETION > VALID")
        self.assertEqual(c([], [], [], [], True, 0, True, "")["outcome"], "NON_COMPLETION")
        self.assertEqual(c([], [], [], [], False, 0, False, "short")["outcome"], "NON_COMPLETION")
        self.assertEqual(c([], [], [], [], True, 3, False, "")["outcome"], "NON_COMPLETION")
        self.assertEqual(c([], [], [], [], True, 0, False, "complete")["outcome"], "VALID")

    def test_segment(self):
        it = _item("80b", "512M", "perf")
        seg = X.parse_server_segment(_segment(it, warn=True, abort=True))
        self.assertEqual(X.check_segment_provenance(seg, it), [])
        self.assertTrue(seg["min_budget_warning"])
        self.assertEqual(len(seg["abort_lines"]), 1)
        self.assertTrue(X.check_segment_provenance(X.parse_server_segment(_segment(it, shards=10)), it))
        self.assertTrue(X.check_segment_provenance(X.parse_server_segment(_segment(it, starts=2)), it))
        bad = _segment(it).replace("--threads-batch 12", "--threads-batch 4")
        self.assertTrue(any("--threads-batch" in s for s in X.check_segment_provenance(X.parse_server_segment(bad), it)))

    def test_perf_record(self):
        it = _item("30b", "768M", "perf")
        self.assertEqual(X.perf_record_problems(_perf_record(it), it, "a0063720"), [])
        self.assertTrue(X.perf_record_problems(_perf_record(it, budget=1610612736), it, "a0063720"))
        self.assertEqual(X.perf_invariant_problems(_perf_record(it), it), [])
        self.assertTrue(X.perf_invariant_problems(_perf_record(it, resident=769), it))
        self.assertEqual(X.perf_measured(_perf_record(it))[0], True)
        self.assertEqual(X.perf_measured(_perf_record(it, tokens=90))[0], False)
        self.assertEqual(X.perf_measured(_perf_record(it, errors=["warmup: boom"]))[0], False)
        self.assertEqual(X.perf_measured(_perf_record(it, errors=["ttft a caldo: boom"]))[0], True)
        self.assertEqual(X.perf_measured(None)[0], False)

    def test_diag(self):
        resp = [{"prompt_id": f"nl-eval-{i}", "predicted_n": 128} for i in (1, 2, 3)]
        self.assertTrue(X.diag_measured({"errors": []}, resp)[0])
        self.assertFalse(X.diag_measured({"errors": []}, resp[:2])[0])
        self.assertFalse(X.diag_measured({"errors": ["x"]}, resp)[0])
        self.assertFalse(X.diag_measured({"errors": []}, resp[:2] + [{"prompt_id": "nl-eval-3", "predicted_n": 7}])[0])
        self.assertEqual(X.diag_invariant_problems({"budget_violations": 0,
                                                    "final_store_stats": {"max_resident_minus_budget": -5}}), [])
        self.assertTrue(X.diag_invariant_problems({"budget_violations": 2}))


class H5Rule(unittest.TestCase):
    def _recs(self, rates, eligible=True):
        return [_rec("30b-diag", i + 1, "768M", "VALID", r, eligible) for i, r in enumerate(rates)]

    def test_falsified_strictly_above(self):
        d = X.h5_decision(self._recs([0.31, 0.32]), True)
        self.assertEqual((d["formal_status"], d["historical_prediction"]), ("FALSIFIED", "FALSIFIED"))
        d = X.h5_decision(self._recs([0.30, 0.30]), True)
        self.assertEqual((d["formal_status"], d["historical_prediction"]), ("INCONCLUSIVE", "NOT FALSIFIED BY THIS TEST"))

    def test_mean_not_vote(self):
        # the mean decides: one launch on the other side of 0.30 changes nothing by itself
        self.assertEqual(X.h5_decision(self._recs([0.29, 0.33]), True)["formal_status"], "FALSIFIED")
        self.assertEqual(X.h5_decision(self._recs([0.10, 0.45]), True)["formal_status"], "INCONCLUSIVE")
        self.assertEqual(X.h5_decision(self._recs([0.25, 0.34]), True)["formal_status"], "INCONCLUSIVE")
        self.assertAlmostEqual(X.h5_decision(self._recs([0.10, 0.45]), True)["aggregate"], 0.275)

    def test_no_positive_support(self):
        self.assertIsNone(X.H5["positive_support_threshold"])
        self.assertEqual(X.h5_decision(self._recs([0.0, 0.0]), True)["formal_status"], "INCONCLUSIVE")

    def test_amendment1_two_eligible_031_falsified(self):
        d = X.h5_decision(self._recs([0.31, 0.31]), True)
        self.assertTrue(d["evaluable"])
        self.assertEqual((d["historical_prediction"], d["formal_status"]), ("FALSIFIED", "FALSIFIED"))
        self.assertAlmostEqual(d["aggregate"], 0.31)

    def test_amendment1_two_eligible_zero_not_falsified(self):
        d = X.h5_decision(self._recs([0.00, 0.00]), True)
        self.assertTrue(d["evaluable"])
        self.assertEqual((d["historical_prediction"], d["formal_status"]), ("NOT FALSIFIED BY THIS TEST", "INCONCLUSIVE"))

    def test_amendment1_one_eligible_not_evaluable_even_if_099(self):
        recs = [_rec("30b-diag", 1, "768M", "VALID", 0.99, True),
                _rec("30b-diag", 2, "768M", "STORE_RESIDENCY_ABORT")]
        d = X.h5_decision(recs, True)
        self.assertFalse(d["evaluable"])
        self.assertEqual((d["historical_prediction"], d["formal_status"]), ("NOT EVALUABLE", "INCONCLUSIVE"))
        self.assertEqual(d["eligible_launches"], 1)
        # a VALID launch that fails eligibility (e.g. not publication-quality) counts the same way
        recs = [_rec("30b-diag", 1, "768M", "VALID", 0.99, True), _rec("30b-diag", 2, "768M", "VALID", 0.99, False)]
        self.assertEqual(X.h5_decision(recs, True)["historical_prediction"], "NOT EVALUABLE")
        self.assertEqual(X.h5_decision(self._recs([0.99]), True)["formal_status"], "INCONCLUSIVE")

    def test_amendment1_zero_eligible_not_evaluable(self):
        recs = [_rec("30b-diag", 1, "768M", "STORE_RESIDENCY_ABORT"), _rec("30b-diag", 2, "768M", "NON_COMPLETION")]
        d = X.h5_decision(recs, True)
        self.assertFalse(d["evaluable"])
        self.assertEqual((d["historical_prediction"], d["formal_status"]), ("NOT EVALUABLE", "INCONCLUSIVE"))
        self.assertEqual(X.h5_decision([], True)["historical_prediction"], "NOT EVALUABLE")
        self.assertFalse(X.h5_decision(self._recs([0.5, 0.5]), False)["evaluable"])

    def test_amendment1_rule_frozen(self):
        self.assertEqual(X.H5["minimum_eligible_launches"], 2)
        self.assertEqual(X.H5["scheduled_launches"], 2)
        self.assertEqual(sum(1 for x in X.build_plan() if x["block"] == "30b-diag" and x["budget"] == "768M"), 2)
        with self.assertRaises(X.ProtocolError):
            X.h5_decision(self._recs([0.1, 0.1, 0.1]), True)

    def test_other_points_ignored(self):
        recs = self._recs([0.1, 0.1]) + [_rec("30b-diag", 9, "1.5G", "VALID", 0.9, True),
                                         _rec("80b-diag", 1, "768M", "VALID", 0.9, True)]
        self.assertEqual(X.h5_decision(recs, True)["formal_status"], "INCONCLUSIVE")


class Gate(unittest.TestCase):
    def _snap(self, porcelain=(), anc=True, head="abc"):
        return {"available": True, "head": head, "tracked": {f: True for f in X.REQUIRED_TRACKED},
                "porcelain": list(porcelain), "error": None, "freeze_head_is_ancestor": anc,
                "registry_head": '{"run_dir": "old"}\n'}

    def test_clean(self):
        self.assertEqual(X.run_gate_problems(self._snap(), True, "ok"), [])

    def test_failures(self):
        self.assertTrue(X.run_gate_problems(self._snap([" M tools/exp022_run.py"]), True, "ok"))
        self.assertTrue(X.run_gate_problems(self._snap(["?? x.txt"]), True, "ok"))
        self.assertTrue(X.run_gate_problems(self._snap(anc=False), True, "ok"))
        self.assertTrue(X.run_gate_problems(self._snap(), False, "changed"))
        self.assertTrue(X.run_gate_problems(self._snap(head="def"), True, "ok", expected_head="abc"))
        s = self._snap()
        s["tracked"][X.REQUIRED_TRACKED[0]] = False
        self.assertTrue(X.run_gate_problems(s, True, "ok"))

    def test_registry_append_only(self):
        snap = self._snap([" M llama-research-v2/VALIDATIONS.jsonl"])
        ok_now = snap["registry_head"] + json.dumps(
            {"run_dir": "C:\\A\\experiments\\EXP-022-extreme-pressure\\raw\\30b-diag\\launch-01\\run-00-1.5G"}) + "\n"
        self.assertEqual(X.run_gate_problems(snap, True, "ok", ok_now), [])
        out_now = snap["registry_head"] + json.dumps({"run_dir": "C:\\A\\experiments\\EXP-007\\raw\\x"}) + "\n"
        self.assertTrue(X.run_gate_problems(snap, True, "ok", out_now))
        self.assertTrue(X.run_gate_problems(snap, True, "ok", '{"run_dir": "edited"}\n'))


class NoLaunch(unittest.TestCase):
    def test_check_and_dry_run_start_nothing_but_git(self):
        rec = Recorder()
        with mock.patch.object(X.subprocess, "run", rec.run), mock.patch.object(X.subprocess, "Popen", rec.popen), \
                mock.patch.object(X, "gguf_footprint", side_effect=AssertionError("GGUF read")):
            _quiet(X.main, ["--check"])
            n_check = len(rec.calls)
            rc, out = _quiet(X.main, ["--dry-run"])
        self.assertEqual(rc, 0)
        self.assertTrue(all(c[0] == "git" for c in rec.calls), rec.calls)
        self.assertEqual(len(rec.calls), n_check, "--dry-run started a subprocess")
        self.assertIn("no subprocess started", out)

    def test_preflight_only_dry_runs(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            exe = td / "llama-server.exe"
            exe.write_bytes(b"fake")
            model, store = td / "m.gguf", td / "m.psxs"
            model.write_bytes(b"x")
            store.write_bytes(b"x")
            models = json.loads(json.dumps(X.MODELS))
            models["30b"]["path"], models["30b"]["store"] = str(model), str(store)
            it = _item("30b", "1.5G", "perf")
            fake_out = "== COMANDO llama.cpp\n  " + _llama_cmd(it).replace(X.MODELS["30b"]["path"], str(model)) + "\n"
            rec = Recorder(fake_stdout=fake_out, allow_git=False)
            with mock.patch.object(X.subprocess, "run", rec.run), mock.patch.object(X.subprocess, "Popen", rec.popen), \
                    mock.patch.object(X, "MODELS", models), \
                    mock.patch.object(X, "static_problems", return_value=[]), \
                    mock.patch.object(X, "binary_path", return_value=exe), \
                    mock.patch.object(X, "REQUIRED_SERVER_SHA256", hashlib.sha256(b"fake").hexdigest()), \
                    mock.patch.object(X, "gguf_footprint", return_value={
                        "footprint_bytes": models["30b"]["footprint_bytes"],
                        "whole_layer_bytes": models["30b"]["whole_layer_bytes"]}), \
                    mock.patch.object(X, "previous_blocks_problems", return_value=([], None)), \
                    mock.patch.object(X, "RAW", td / "raw"), \
                    mock.patch.object(X, "machine_problems", return_value=([], {})), \
                    mock.patch.object(X, "freeze_status", return_value=(True, "ok")):
                r = X.preflight("30b-perf", td, parent={})
        self.assertTrue(r["ok"], r["problems"])
        self.assertEqual(len(rec.calls), 5)
        for c in rec.calls:
            self.assertIn("--dry-run", c)
            self.assertIn("benchmark", c)
            self.assertNotIn("trace", c)
            self.assertFalse(any("llama-server" in str(t) for t in c))

    def test_run_refused_before_any_subprocess(self):
        rec = Recorder(allow_git=False)
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(X.subprocess, "run", rec.run), mock.patch.object(X.subprocess, "Popen", rec.popen), \
                    mock.patch.object(X, "RAW", Path(td) / "raw"), \
                    mock.patch.object(X, "previous_blocks_problems", return_value=([], None)), \
                    mock.patch.object(X, "require_run_gate", side_effect=X.ProtocolError("dirty tree")):
                rc, _ = _quiet(X.run_block, "30b-perf", Path(td))
            self.assertFalse((Path(td) / "raw").exists())
        self.assertEqual(rc, 2)
        self.assertEqual(rec.calls, [])

    def test_later_block_refused_until_earlier_complete(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(X, "RAW", Path(td)):
                bad, _ = X.previous_blocks_problems("30b-diag")
                self.assertTrue(bad)
                (Path(td) / "30b-perf").mkdir()
                (Path(td) / "30b-perf" / "session.json").write_text(
                    json.dumps({"status": "STOPPED_INVALID at launch 3", "valid": False}), encoding="utf-8")
                bad, _ = X.previous_blocks_problems("30b-diag")
                self.assertTrue(bad)
                (Path(td) / "30b-perf" / "session.json").write_text(
                    json.dumps({"status": "complete", "valid": True, "checked_head_commit": "abc"}), encoding="utf-8")
                bad, head = X.previous_blocks_problems("30b-diag")
                self.assertEqual((bad, head), ([], "abc"))


class AbortCause(unittest.TestCase):
    LINE = ("ggml.c:1:argine: impossibile rendere residenti gli expert del layer 3 (blk.3.ffn_gate_exps.weight). "
            "Residenti 500 MiB su un budget di 512 MiB, questo nodo ne chiede {n} su 512. Alza ARGINE_EXPERT_BUDGET")

    def test_parse(self):
        p = X.parse_residency_abort(self.LINE.format(n=300))
        self.assertEqual((p["layer"], p["resident_mib"], p["budget_mib"], p["node_experts"], p["layer_experts"]),
                         (3, 500, 512, 300, 512))
        self.assertIsNone(X.parse_residency_abort("something else"))

    def test_rule(self):
        lb = {3: 1622016}
        big, small = self.LINE.format(n=400), self.LINE.format(n=100)   # 400 x 1.55 MiB > 512 MiB; 100 x < 512 MiB
        b = 536870912
        self.assertEqual(X.abort_cause([big], [(144, 144)], b, "80b", None, lb)["cause"], "capacity_or_pinning")
        self.assertEqual(X.abort_cause([big], [], b, "80b", None, lb)["cause"], "cause_unresolved")      # no coverage
        self.assertEqual(X.abort_cause([big], [(140, 144)], b, "80b", None, lb)["cause"], "cause_unresolved")
        self.assertEqual(X.abort_cause([small], [(144, 144)], b, "80b", 0, lb)["cause"], "cause_unresolved")
        self.assertEqual(X.abort_cause([small], [(144, 144)], b, "80b", None, lb)["cause"], "cause_unresolved")
        self.assertEqual(X.abort_cause([small], [(144, 144)], b, "80b", 2, lb)["cause"], "commit_or_read_failure")
        self.assertEqual(X.abort_cause(["unparsable abort"], [(144, 144)], b, "80b", None, lb)["cause"], "cause_unresolved")

    def test_tracked_layer_bytes(self):
        lb = X.layer_expert_bytes("80b")
        self.assertEqual(len(lb), 48)
        self.assertEqual(sum(lb.values()) * 10, X.MODELS["80b"]["footprint_bytes"])

    def test_envelope_keeps_abort_category(self):
        r = _rec("80b-perf", 1, "512M", "STORE_RESIDENCY_ABORT")
        self.assertEqual(X._perf_row(r)["usability"], "STORE_RESIDENCY_ABORT")
        self.assertIsNone(X._perf_row(r)["gen_tok_s"])


class AnalysisAndFreeze(unittest.TestCase):
    def test_analyze_synthetic(self):
        recs = []
        for b in X.BLOCK_ORDER:
            model, phase = b.split("-")
            for i, (lab, _, _) in enumerate(X.BUDGETS[model], 1):
                if phase == "perf":
                    r = _rec(b, i, lab, "STORE_RESIDENCY_ABORT" if lab == "512M" and model == "80b" else "VALID")
                    r["metrics"] = {"runs": [{"gen_tok_s": 6.0 if lab == "1.5G" else 2.0}]}
                else:
                    r = _rec(b, i, lab, "VALID", 0.05, eligible=(model == "30b" and lab == "768M"))
                recs.append(r)
        sessions = {b: {"status": "complete", "valid": True} for b in X.BLOCK_ORDER}
        text, s = X.analyze(recs, sessions)
        self.assertEqual(s["formal_status"], "INCONCLUSIVE")
        self.assertEqual(s["envelope"]["30b-perf"]["1.5G"]["envelope_label"], "INTERACTIVE")
        self.assertEqual(s["envelope"]["30b-perf"]["768M"]["envelope_label"], "DEGRADED")
        self.assertEqual(s["envelope"]["80b-perf"]["512M"]["envelope_label"], "STORE_RESIDENCY_ABORT")
        self.assertIn("outcome", text.splitlines()[0])

    def test_spec_contents(self):
        s = X.protocol_spec()
        self.assertEqual(s["historical_hypothesis_verbatim"], X.HISTORICAL_HYPOTHESIS)
        self.assertIsNone(s["h5"]["positive_support_threshold"])
        self.assertEqual(s["h5"]["threshold"], 0.30)
        self.assertEqual(len(s["launch_order"]), 45)
        self.assertEqual(s["budgets"]["30b"]["768M"]["bytes"], 805306368)
        self.assertEqual(s["completion_rule"]["cap_s"], {"perf": 2400, "diag": 6000})
        self.assertEqual(set(s["outcome_taxonomy"]) & set(X.OUTCOMES), set(X.OUTCOMES))
        self.assertIn("80b-336M", s["excluded_budgets"])
        self.assertEqual(s["runtime"]["server_sha256"], X.REQUIRED_SERVER_SHA256)
        self.assertEqual(set(X.freeze_hashes()), {"protocol_sha256", "spec_sha256", "tool_sha256", "tests_sha256"})

    def test_superseded_freeze_rejected(self):
        old = X.EXP / "protocol_freeze_pre_review_2026-10-14.json"
        if not old.exists():
            self.skipTest("archived pre-review freeze not present")
        o = json.loads(old.read_text(encoding="utf-8"))
        now = X.freeze_hashes()
        for k in ("spec_sha256", "tool_sha256", "tests_sha256"):
            self.assertNotEqual(o[k], now[k], k)
        with mock.patch.object(X, "FREEZE", old):
            ok, why = X.freeze_status()
        self.assertFalse(ok)
        self.assertIn("CHANGED", why)
        cur = json.loads(X.FREEZE.read_text(encoding="utf-8"))
        self.assertEqual(cur["supersedes"], old.name)
        self.assertEqual([a["id"] for a in cur["amendments"]], ["A1", "A2"])
        self.assertTrue(all(a["before_any_measurement"] for a in cur["amendments"]))

    def test_freeze_record_if_present(self):
        if not X.FREEZE.exists():
            self.skipTest("protocol_freeze.json not yet written")
        f = json.loads(X.FREEZE.read_text(encoding="utf-8"))
        self.assertEqual(f["protocol_version"], X.PROTOCOL_VERSION)
        self.assertEqual(f["spec"]["historical_hypothesis_verbatim"], X.HISTORICAL_HYPOTHESIS)
        ok, why = X.freeze_status()
        self.assertTrue(ok, why)


if __name__ == "__main__":
    unittest.main()
