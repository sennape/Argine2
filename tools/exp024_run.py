"""EXP-024 — cost of ENABLED research instrumentation (orchestration, provenance, analysis).

    python tools/exp024_run.py --dry-run     # print + check the frozen 20-launch schedule; no subprocess
    python tools/exp024_run.py --preflight   # static checks + `argine benchmark --dry-run` per budget and
                                             #   condition: reads the GGUF header, never loads the model,
                                             #   never starts llama-server, never runs inference
    python tools/exp024_run.py --freeze      # write protocol_freeze.json (protocol + tool hashes)
    python tools/exp024_run.py --run         # the 20 real launches (writes raw/); NOT resumable
    python tools/exp024_run.py --finalize    # post-session fingerprint finalisation only
    python tools/exp024_run.py --analyze     # raw/records.jsonl -> results.csv + processed/summary.json

What is measured
----------------
Research instrumentation ENABLED (`ARGINE_RESEARCH_TRACE` set) versus DISABLED
(variable absent) on the SAME `llama-research-v2` binary. It is not patched vs
unpatched: with tracing disabled the compiled hooks remain and each returns
after one atomic load (`arx_enabled`). With tracing enabled the full research
path runs: record formatting and buffered file writes, and on Windows the
precise completion wait (`arx_wait_precise`) used only while tracing. The
quantity is the overhead of that whole enabled path; it is not decomposed.

How the factor is toggled
-------------------------
Only through the environment of the `python -m argine benchmark` child process:
`LlamaServerProcess` passes `os.environ` (plus its own store variables) to
llama-server, and `argine benchmark` has no option of its own that sets the
variable. The command line of a budget is byte-identical in both conditions.
The parent environment must not carry ARGINE_RESEARCH_TRACE or any other
ARGINE_* / PICOSTAR_* variable except ARGINE_LLAMA_DIR and ARGINE_HOME;
otherwise the tool refuses to run.

Confirmation that the factor took effect (every launch):
* ON: the launch log (which contains the store's own stderr lines, echoed by
  `argine benchmark`) has exactly one `argine: research trace attiva: <path>`
  line naming the requested file; the file exists, is non-empty and starts
  with the `H argine-raw 2` header; no "cannot open" line.
* OFF: no `research trace` text at all in the log, the launch's would-be trace
  file does not exist, and raw/traces/ gained no file.

Replication unit
----------------
One complete `argine benchmark` launch: fresh llama-server, empty Expert Store,
the normal benchmark warm-up, exactly one measured request (`--repeat 1`,
`--n-predict 128`, `--retries 0`). Requests inside one process are never
counted as replicates. A failed launch stops the session; it is never retried,
because a retry would not be the preregistered launch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
EXP_ID = "EXP-024-tracing-overhead"
EXP = ROOT / "experiments" / EXP_ID
RAW = EXP / "raw"
TRACES = RAW / "traces"
LOGS = RAW / "logs"
README = EXP / "README.md"
FREEZE = EXP / "protocol_freeze.json"
MARK_BEGIN = "<!-- EXP-024 PROTOCOL-BEGIN -->"
MARK_END = "<!-- EXP-024 PROTOCOL-END -->"

MODEL_NAME = "Qwen3-30B-A3B Q4_K_M"
MODEL_PATH = r"C:\modelli\Qwen_Qwen3-30B-A3B-Q4_K_M.gguf"
STORE_PATH = r"C:\modelli\Qwen_Qwen3-30B-A3B-Q4_K_M.psxs"
DEFAULT_LLAMA_DIR = ROOT / "llama-research-v2"
INSTRUMENTATION_ID = "argine2-research-v2"
BUDGETS = {"4G": 4 * 1024 ** 3, "1.5G": 1610612736}
THREADS = 4
BATCH_THREADS = 12
READ_MODE = "asincrono"
BACKEND = "cpu"
N_PREDICT = 128
REPEAT = 1
RETRIES = 0
COOLDOWN_S = 60
N_ROUNDS = 5
CONDITIONS = ("OFF", "ON")
THRESHOLD_PCT = 2.0           # frozen hypothesis: overhead < 2% at every budget
CONFIDENCE = 0.95
TRACE_CONFIRM = "argine: research trace attiva:"
TRACE_CANNOT_OPEN = "research trace: cannot open"
TRACE_HEADER = "H\targine-raw\t2\t"
ALLOWED_ARGINE_ENV = {"ARGINE_LLAMA_DIR", "ARGINE_HOME"}

# The frozen execution order (README "Execution order"). Hard-coded on purpose:
# stats.interleaved_schedule(["OFF", "ON"], 5) gives OFF ON OFF ON ... (OFF always
# first), not the intended AB / BA alternation.
FROZEN_ROUNDS: Tuple[Tuple[Tuple[str, str], ...], ...] = (
    (("4G", "OFF"), ("4G", "ON"), ("1.5G", "OFF"), ("1.5G", "ON")),
    (("1.5G", "ON"), ("1.5G", "OFF"), ("4G", "ON"), ("4G", "OFF")),
    (("4G", "ON"), ("4G", "OFF"), ("1.5G", "ON"), ("1.5G", "OFF")),
    (("1.5G", "OFF"), ("1.5G", "ON"), ("4G", "OFF"), ("4G", "ON")),
    (("4G", "OFF"), ("4G", "ON"), ("1.5G", "OFF"), ("1.5G", "ON")),
)

# two-sided 95% Student-t critical values, df 1..10
T975 = {1: 12.706204736, 2: 4.302652730, 3: 3.182446305, 4: 2.776445105, 5: 2.570581836,
        6: 2.446911851, 7: 2.364624252, 8: 2.306004135, 9: 2.262157163, 10: 2.228138852}


class ProtocolError(RuntimeError):
    """A configuration, provenance or data condition that the frozen protocol forbids."""


# ------------------------------------------------------------- schedule ----

def trace_name(launch_index: int, rnd: int, budget: str, cond: str) -> str:
    return f"launch-{launch_index:02d}-r{rnd}-{budget}-{cond}.tsv"


def build_schedule() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rnd, items in enumerate(FROZEN_ROUNDS, 1):
        for pos, (budget, cond) in enumerate(items, 1):
            i = len(out) + 1
            name = trace_name(i, rnd, budget, cond)
            out.append({"launch_index": i, "round": rnd, "position_in_round": pos,
                        "budget": budget, "budget_bytes": BUDGETS[budget], "condition": cond,
                        "trace_file": (f"raw/traces/{name}" if cond == "ON" else None),
                        "would_be_trace_file": f"raw/traces/{name}"})
    return out


def benchmark_args(budget: str, dry_run: bool = False) -> List[str]:
    """The exact `argine benchmark` argument list; identical for OFF and ON."""
    a = ["benchmark", MODEL_PATH,
         "--backend", BACKEND,
         "--expert-store",
         "--expert-budget", budget,
         "--read-mode", READ_MODE,
         "--threads", str(THREADS),
         "--repeat", str(REPEAT),
         "--n-predict", str(N_PREDICT),
         "--retries", str(RETRIES),
         "--expert-stats"]
    if dry_run:
        a.append("--dry-run")
    return a


def command(budget: str, dry_run: bool = False) -> List[str]:
    return [sys.executable, "-m", "argine"] + benchmark_args(budget, dry_run)


def check_schedule(s: List[Dict[str, Any]]) -> List[str]:
    """Structural checks of the frozen schedule. Empty list = OK."""
    bad: List[str] = []
    if len(s) != 20:
        bad.append(f"{len(s)} launches, expected 20")
    if [x["launch_index"] for x in s] != list(range(1, len(s) + 1)):
        bad.append("launch indices are not 1..N in order")
    expected = [(b, c) for rnd in FROZEN_ROUNDS for (b, c) in rnd]
    if [(x["budget"], x["condition"]) for x in s] != expected:
        bad.append("order differs from the frozen schedule")
    for b in BUDGETS:
        for c in CONDITIONS:
            n = sum(1 for x in s if x["budget"] == b and x["condition"] == c)
            if n != N_ROUNDS:
                bad.append(f"{b}-{c}: {n} launches, expected {N_ROUNDS}")
    for rnd in range(1, N_ROUNDS + 1):
        r = [x for x in s if x["round"] == rnd]
        if len(r) != 4:
            bad.append(f"round {rnd}: {len(r)} launches, expected 4")
            continue
        for b in BUDGETS:
            pos = [x["position_in_round"] for x in r if x["budget"] == b]
            conds = sorted(x["condition"] for x in r if x["budget"] == b)
            if conds != ["OFF", "ON"]:
                bad.append(f"round {rnd} budget {b}: conditions {conds}, expected one OFF and one ON")
            elif abs(pos[0] - pos[1]) != 1:
                bad.append(f"round {rnd} budget {b}: OFF and ON are not adjacent")
    names = [x["would_be_trace_file"] for x in s]
    if len(set(names)) != len(names):
        bad.append("trace file names are not unique")
    for x in s:
        if (x["condition"] == "ON") != bool(x["trace_file"]):
            bad.append(f"launch {x['launch_index']}: trace file inconsistent with condition")
    for b in BUDGETS:
        a = benchmark_args(b)
        for flag, val in (("--repeat", "1"), ("--retries", "0"), ("--backend", "cpu"),
                          ("--expert-budget", b), ("--read-mode", READ_MODE),
                          ("--threads", str(THREADS)), ("--n-predict", str(N_PREDICT))):
            if flag not in a or a[a.index(flag) + 1] != val:
                bad.append(f"budget {b}: {flag} != {val}")
        if "--expert-store" not in a:
            bad.append(f"budget {b}: --expert-store missing")
    return bad


def order_balance(s: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Descriptive: how often each condition comes first within a budget, and budget order."""
    first = {b: {"OFF": 0, "ON": 0} for b in BUDGETS}
    budget_first = {b: 0 for b in BUDGETS}
    for rnd in range(1, N_ROUNDS + 1):
        r = sorted([x for x in s if x["round"] == rnd], key=lambda x: x["position_in_round"])
        budget_first[r[0]["budget"]] += 1
        for b in BUDGETS:
            first[b][[x for x in r if x["budget"] == b][0]["condition"]] += 1
    return {"condition_first_per_budget": first, "budget_first_in_round": budget_first}


# ---------------------------------------------------------- environment ----

def base_env(parent: Dict[str, str], llama_dir: Path) -> Dict[str, str]:
    """The child environment common to OFF and ON. Refuses unexpected ARGINE_* /
    PICOSTAR_* variables instead of silently clearing or keeping them."""
    extra = sorted(k for k in parent
                   if (k.upper().startswith("ARGINE_") and k.upper() not in ALLOWED_ARGINE_ENV)
                   or k.upper().startswith("PICOSTAR_"))
    if extra:
        raise ProtocolError("parent environment carries variables that could change the runtime: "
                            + ", ".join(extra) + " (unset them; ARGINE_RESEARCH_TRACE is set by the "
                            "tool only for ON launches)")
    cur = parent.get("ARGINE_LLAMA_DIR")
    if cur and Path(cur).resolve() != llama_dir.resolve():
        raise ProtocolError(f"ARGINE_LLAMA_DIR={cur} differs from the research runtime {llama_dir}")
    env = dict(parent)
    env["ARGINE_LLAMA_DIR"] = str(llama_dir)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def launch_env(base: Dict[str, str], launch: Dict[str, Any], exp_dir: Path = EXP) -> Dict[str, str]:
    env = dict(base)
    env.pop("ARGINE_RESEARCH_TRACE", None)
    if launch["condition"] == "ON":
        env["ARGINE_RESEARCH_TRACE"] = str((exp_dir / launch["trace_file"]).resolve())
    return env


def env_delta(base: Dict[str, str], env: Dict[str, str]) -> Dict[str, Any]:
    keys = sorted(set(base) | set(env))
    return {k: env.get(k) for k in keys if base.get(k) != env.get(k)}


# ----------------------------------------------------------- validation ----

def _norm(p: str) -> str:
    return os.path.normcase(os.path.normpath(p.strip().strip('"')))


def check_trace_condition(launch: Dict[str, Any], log_text: str, trace_path: Path,
                          traces_before: List[str], traces_after: List[str]) -> List[str]:
    """ON/OFF evidence that the experimental factor took effect. Empty = OK."""
    bad: List[str] = []
    lines = [l for l in log_text.splitlines() if TRACE_CONFIRM in l]
    new_files = sorted(set(traces_after) - set(traces_before))
    if TRACE_CANNOT_OPEN in log_text:
        bad.append("runtime could not open the research trace")
    if launch["condition"] == "ON":
        if len(lines) != 1:
            bad.append(f"{len(lines)} instrumentation confirmation lines (expected exactly 1)")
        else:
            got = lines[0].split(TRACE_CONFIRM, 1)[1]
            if _norm(got) != _norm(str(trace_path)):
                bad.append(f"instrumentation opened {got.strip()!r}, expected {str(trace_path)!r}")
        if not trace_path.exists():
            bad.append(f"trace file {trace_path} missing")
        else:
            if trace_path.stat().st_size <= 0:
                bad.append(f"trace file {trace_path} is empty")
            else:
                with open(trace_path, "rb") as fh:
                    head = fh.read(len(TRACE_HEADER)).decode("utf-8", "replace")
                if head != TRACE_HEADER:
                    bad.append(f"trace file does not start with the argine-raw v2 header ({head!r})")
        if new_files != [trace_path.name]:
            bad.append(f"files created in raw/traces: {new_files} (expected only {trace_path.name})")
    else:
        if "research trace" in log_text.lower():
            bad.append("OFF launch log mentions the research trace")
        if trace_path.exists():
            bad.append(f"OFF launch has a trace file {trace_path}")
        if new_files:
            bad.append(f"OFF launch created files in raw/traces: {new_files}")
    return bad


def check_store_log(log_text: str) -> List[str]:
    bad: List[str] = []
    opened = [l for l in log_text.splitlines() if "expert store aperto:" in l]
    if len(opened) != 1:
        bad.append(f"{len(opened)} 'expert store aperto' lines (expected 1)")
    else:
        m = re.search(r"expert store aperto:\s*(.+?\.psxs)", opened[0], flags=re.IGNORECASE)
        if not m or _norm(m.group(1)) != _norm(STORE_PATH):
            bad.append(f"store opened: {opened[0].strip()!r}, expected {STORE_PATH}")
        if f"lettura {READ_MODE}" not in opened[0]:
            bad.append(f"store read mode is not {READ_MODE}: {opened[0].strip()!r}")
    if "expert store IN USO" not in log_text:
        bad.append("store never reported IN USO")
    if "device Vulkan" in log_text:
        bad.append("Vulkan device handling appears in the log (backend must be cpu)")
    return bad


def validate_record(rec: Optional[Dict[str, Any]], launch: Dict[str, Any], exit_code: int,
                    binary_sha8: Optional[str], model_bytes: Optional[int],
                    prompt_chars: int) -> List[str]:
    """Benchmark record checks for one launch. Empty list = valid."""
    bad: List[str] = []
    if exit_code != 0:
        bad.append(f"exit status {exit_code}")
    if rec is None:
        bad.append("no single benchmark record appended to the shared log")
        return bad
    plan = rec.get("plan") or {}
    rt = rec.get("runtime") or {}
    met = rec.get("metrics") or {}
    runs = met.get("runs") or []
    model = rec.get("model") or {}
    checks = [
        (rec.get("event") == "benchmark", f"event {rec.get('event')!r}"),
        (_norm(str(model.get("path") or "")) == _norm(MODEL_PATH), f"model path {model.get('path')!r}"),
        (model_bytes is None or model.get("file_size") == model_bytes,
         f"model size {model.get('file_size')} != {model_bytes}"),
        (plan.get("threads") == THREADS, f"plan.threads {plan.get('threads')} != {THREADS}"),
        (plan.get("threads_batch") == BATCH_THREADS,
         f"plan.threads_batch {plan.get('threads_batch')} != {BATCH_THREADS}"),
        (plan.get("backend") == BACKEND and rt.get("backend") == BACKEND,
         f"backend plan={plan.get('backend')} runtime={rt.get('backend')} != cpu"),
        ((plan.get("gpu_layers") or 0) == 0, f"gpu_layers {plan.get('gpu_layers')} != 0"),
        (plan.get("expert_store") is True, "expert store not enabled in the plan"),
        (plan.get("expert_cache_budget") == launch["budget_bytes"],
         f"expert budget {plan.get('expert_cache_budget')} != {launch['budget_bytes']}"),
        (str(rec.get("modo_lettura") or "") == READ_MODE,
         f"read mode reported by the store {rec.get('modo_lettura')!r} != {READ_MODE}"),
        (rec.get("repeat") == REPEAT and len(runs) == REPEAT,
         f"repeat {rec.get('repeat')} / measured runs {len(runs)} != 1"),
        (rec.get("n_predict") == N_PREDICT, f"n_predict {rec.get('n_predict')} != {N_PREDICT}"),
        (rec.get("prompt_chars") == prompt_chars, f"prompt_chars {rec.get('prompt_chars')} != {prompt_chars}"),
        (not rec.get("attempts"), f"server start needed retries: {rec.get('attempts')}"),
        (not met.get("errors"), f"benchmark errors: {met.get('errors')}"),
        (bool(runs and runs[0].get("gen_tok_s")), "no generation tok/s"),
    ]
    bad += [msg for ok, msg in checks if not ok]
    if not binary_sha8 or binary_sha8 not in str(rt.get("build") or ""):
        bad.append(f"runtime build {rt.get('build')!r} does not carry binary sha {binary_sha8}")
    return bad


# ------------------------------------------------------------- analysis ----

def overhead_pct(off: float, on: float) -> float:
    """100 * (OFF - ON) / OFF. Positive = tracing slower; negative = tracing faster."""
    if not off or off <= 0:
        raise ProtocolError(f"invalid OFF throughput {off!r}")
    return 100.0 * (off - on) / off


def t_summary(values: List[float], confidence: float = CONFIDENCE) -> Dict[str, Any]:
    if confidence != 0.95:
        raise ProtocolError("only the frozen 95% interval is implemented")
    n = len(values)
    if n < 2 or (n - 1) not in T975:
        raise ProtocolError(f"t summary needs 2..11 values, got {n}")
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    se = sd / math.sqrt(n)
    t = T975[n - 1]
    return {"n": n, "mean": mean, "sd": sd, "se": se, "df": n - 1, "t_crit": t,
            "ci_low": mean - t * se, "ci_high": mean + t * se}


def describe(values: List[float]) -> Dict[str, Any]:
    m = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else None
    return {"n": len(values), "values": list(values), "mean": m, "median": statistics.median(values),
            "sd": sd, "cv": (sd / m if sd is not None and m else None),
            "min": min(values), "max": max(values)}


def decide(ci_by_budget: Dict[str, Dict[str, float]], threshold: float = THRESHOLD_PCT) -> Dict[str, Any]:
    """Frozen rule. SUPPORTED: CI upper < threshold at every budget. FALSIFIED: CI lower >
    threshold at at least one budget. INCONCLUSIVE otherwise (a CI touching or crossing it)."""
    if set(ci_by_budget) != set(BUDGETS):
        raise ProtocolError(f"decision needs exactly the budgets {sorted(BUDGETS)}")
    above = sorted(b for b, c in ci_by_budget.items() if c["ci_low"] > threshold)
    below = sorted(b for b, c in ci_by_budget.items() if c["ci_high"] < threshold)
    if above:
        return {"verdict": "FALSIFIED", "threshold_pct": threshold,
                "reason": f"95% CI lower bound > {threshold}% at {', '.join(above)}"}
    if len(below) == len(BUDGETS):
        return {"verdict": "SUPPORTED", "threshold_pct": threshold,
                "reason": f"95% CI upper bound < {threshold}% at every budget"}
    return {"verdict": "INCONCLUSIVE", "threshold_pct": threshold,
            "reason": "the 95% CI does not lie entirely below the threshold at "
                      + ", ".join(sorted(set(BUDGETS) - set(below))) + ", and no lower bound exceeds it"}


def _gen(r: Dict[str, Any]) -> float:
    return float(((r.get("metrics") or {}).get("runs") or [{}])[0]["gen_tok_s"])


def paired_observations(records: List[Dict[str, Any]], session_id: str) -> Dict[str, List[Dict[str, Any]]]:
    """Round-level OFF/ON pairs per budget. Refuses duplicates, retries, foreign sessions,
    invalid launches and incomplete designs instead of silently using what is there."""
    sched = {x["launch_index"]: x for x in build_schedule()}
    seen: Dict[int, Dict[str, Any]] = {}
    for r in records:
        e = r.get("exp024") or {}
        if e.get("session_id") != session_id:
            raise ProtocolError(f"record from another session: {e.get('session_id')!r}")
        i = int(e.get("launch_index", 0))
        if i in seen:
            raise ProtocolError(f"duplicate record for launch {i} (a retry is not a preregistered launch)")
        if int(e.get("attempt", 1)) != 1:
            raise ProtocolError(f"launch {i}: attempt {e.get('attempt')} (retries are not allowed)")
        if not e.get("valid"):
            raise ProtocolError(f"launch {i} is invalid: {e.get('problems')}")
        s = sched.get(i)
        if s is None or (e.get("budget"), e.get("condition"), e.get("round")) != (s["budget"], s["condition"], s["round"]):
            raise ProtocolError(f"launch {i} does not match the frozen schedule")
        seen[i] = r
    if sorted(seen) != list(range(1, 21)):
        raise ProtocolError(f"incomplete session: {len(seen)} of 20 valid launches")
    out: Dict[str, List[Dict[str, Any]]] = {b: [] for b in BUDGETS}
    for b in BUDGETS:
        for rnd in range(1, N_ROUNDS + 1):
            pair = {c: [r for r in seen.values() if r["exp024"]["budget"] == b
                        and r["exp024"]["round"] == rnd and r["exp024"]["condition"] == c] for c in CONDITIONS}
            if any(len(v) != 1 for v in pair.values()):
                raise ProtocolError(f"budget {b} round {rnd}: no unique OFF/ON pair")
            off, on = _gen(pair["OFF"][0]), _gen(pair["ON"][0])
            out[b].append({"round": rnd, "off_launch": pair["OFF"][0]["exp024"]["launch_index"],
                           "on_launch": pair["ON"][0]["exp024"]["launch_index"],
                           "gen_tok_s_off": off, "gen_tok_s_on": on,
                           "overhead_pct": overhead_pct(off, on),
                           "ms_per_token_off": 1000.0 / off, "ms_per_token_on": 1000.0 / on,
                           "delta_ms_per_token": 1000.0 / on - 1000.0 / off})
    return out


def _diag(records: List[Dict[str, Any]], b: str, c: str) -> Dict[str, Any]:
    rs = [r for r in records if r["exp024"]["budget"] == b and r["exp024"]["condition"] == c]

    def col(f):
        v = [f(r) for r in rs]
        v = [x for x in v if x is not None]
        return statistics.median(v) if v else None
    run0 = lambda r: ((r.get("metrics") or {}).get("runs") or [{}])[0]  # noqa: E731
    res = lambda r: (r.get("metrics") or {}).get("resources") or {}  # noqa: E731
    return {"ttft_s_median": col(lambda r: run0(r).get("ttft_s")),
            "prompt_tok_s_median": col(lambda r: run0(r).get("prompt_tok_s")),
            "ram_peak_median": col(lambda r: res(r).get("ram_peak")),
            "cpu_mean_percent_median": col(lambda r: res(r).get("cpu_mean_percent")),
            "store_hit_pct_median": col(lambda r: (r.get("expert_cache") or {}).get("hit_pct")),
            "read_modes": sorted({str(r.get("modo_lettura")) for r in rs})}


def analyze(records: List[Dict[str, Any]], session_id: str) -> Tuple[str, Dict[str, Any]]:
    pairs = paired_observations(records, session_id)
    rows, summary = [], {"experiment": EXP_ID, "session_id": session_id,
                         "primary_metric": "gen_tok_s; paired per round and budget",
                         "overhead_definition": "100 * (gen_tok_s_OFF - gen_tok_s_ON) / gen_tok_s_OFF",
                         "budgets": {}}
    ci_by_budget = {}
    for b, ps in pairs.items():
        off = [p["gen_tok_s_off"] for p in ps]
        on = [p["gen_tok_s_on"] for p in ps]
        ov = [p["overhead_pct"] for p in ps]
        dms = [p["delta_ms_per_token"] for p in ps]
        ci = t_summary(ov)
        ci_by_budget[b] = ci
        summary["budgets"][b] = {"pairs": ps, "gen_tok_s_off": describe(off), "gen_tok_s_on": describe(on),
                                 "overhead_pct": {**describe(ov), **ci},
                                 "delta_ms_per_token": {**describe(dms), **t_summary(dms)},
                                 "diagnostics": {c: _diag(records, b, c) for c in CONDITIONS}}
        for p in ps:
            rows.append({"budget": b, **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in p.items()}})
    summary["decision"] = decide(ci_by_budget)
    summary["note"] = ("raw effect and CI are reported whatever the verdict; secondary metrics are "
                       "descriptive only")
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue(), summary


# ---------------------------------------------------------------- freeze ----

def _lf_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def protocol_text(readme: Path = README) -> str:
    t = readme.read_text(encoding="utf-8").replace("\r\n", "\n")
    if t.count(MARK_BEGIN) != 1 or t.count(MARK_END) != 1:
        raise ProtocolError("README protocol markers missing or duplicated")
    return t.split(MARK_BEGIN, 1)[1].split(MARK_END, 1)[0]


def freeze_hashes() -> Dict[str, str]:
    return {"protocol_sha256": hashlib.sha256(protocol_text().encode("utf-8")).hexdigest(),
            "tool_sha256": _lf_sha(Path(__file__).resolve()),
            "tests_sha256": _lf_sha(ROOT / "tests" / "test_exp024.py")}


def freeze_status() -> Tuple[bool, str]:
    if not FREEZE.exists():
        return False, "protocol_freeze.json absent (run --freeze)"
    f = json.loads(FREEZE.read_text(encoding="utf-8"))
    now = freeze_hashes()
    diff = [k for k in now if f.get(k) != now[k]]
    return (not diff), ("matches the frozen protocol" if not diff else
                        "CHANGED since freeze: " + ", ".join(diff))


def cmd_freeze() -> int:
    now = freeze_hashes()
    supersedes = None
    if FREEZE.exists():
        old_bytes = FREEZE.read_bytes()
        old = json.loads(old_bytes.decode("utf-8"))
        if all(old.get(k) == v for k, v in now.items()):
            print("protocol_freeze.json already matches; nothing written")
            return 0
        archived = [p for p in sorted(EXP.glob("protocol_freeze_*.json")) if p.read_bytes() == old_bytes]
        if not archived:
            print("protocol_freeze.json exists and differs: first copy it byte-for-byte to a dated "
                  "protocol_freeze_<label>.json and document the amendment, then refreeze",
                  file=sys.stderr)
            return 1
        supersedes = archived[0].name
    snap = git_snapshot()
    rec = {"experiment": EXP_ID, "protocol_version": "EXP024-P1",
           "status": "frozen before any measurement; independent review completed 2026-10-12",
           "supersedes": supersedes,
           "protocol_hash_method": f"sha256 of the UTF-8, LF-normalised README text strictly between "
                                   f"'{MARK_BEGIN}' and '{MARK_END}'",
           "code_hash_method": "sha256 of the LF-normalised file bytes",
           **now, "argine2_at_freeze": {"head": snap.get("head"), "error": snap.get("error"),
                                        "porcelain": snap.get("porcelain"),
                                        "note": "the reviewed files are uncommitted at freeze time by "
                                                "construction; --run requires them committed and the tree clean"},
           "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    FREEZE.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {FREEZE.relative_to(ROOT)}: protocol {now['protocol_sha256'][:12]}, "
          f"tool {now['tool_sha256'][:12]}, tests {now['tests_sha256'][:12]}")
    return 0


# ------------------------------------------------------------- git gate ----
# A REAL run starts only from a committed, clean, frozen protocol (independent review,
# 2026-10-12). --dry-run and --preflight deliberately do not apply this gate, so the
# Windows command construction can be checked before the protocol is committed.

REQUIRED_TRACKED = (
    "experiments/EXP-024-tracing-overhead/README.md",
    "experiments/EXP-024-tracing-overhead/config.json",
    "experiments/EXP-024-tracing-overhead/protocol_freeze.json",
    "tools/exp024_run.py",
    "tests/test_exp024.py",
)
GIT_TIMEOUT_S = 60


def git_snapshot(root: Path = ROOT) -> Dict[str, Any]:
    """Raw Git state for the gate: HEAD, per-file tracked status, porcelain output.
    Only reads the repository; every failure is reported, never guessed."""
    def g(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                              timeout=GIT_TIMEOUT_S, errors="replace")
    snap: Dict[str, Any] = {"available": False, "head": None, "tracked": {}, "porcelain": None,
                            "error": None}
    try:
        h = g("rev-parse", "--verify", "HEAD^{commit}")
    except Exception as e:  # noqa: BLE001 (git missing, timeout)
        snap["error"] = f"git unavailable: {e}"
        return snap
    snap["available"] = True
    if h.returncode == 0 and h.stdout.strip():
        snap["head"] = h.stdout.strip()
    else:
        snap["error"] = (h.stderr or h.stdout or "").strip()[:300] or "HEAD does not resolve"
    try:
        for f in REQUIRED_TRACKED:
            snap["tracked"][f] = g("ls-files", "--error-unmatch", "--", f).returncode == 0
        s = g("status", "--porcelain")
        snap["porcelain"] = s.stdout.splitlines() if s.returncode == 0 else None
        if s.returncode != 0:
            snap["error"] = (snap["error"] or "") + " git status failed: " + (s.stderr or "").strip()[:300]
    except Exception as e:  # noqa: BLE001
        snap["error"] = f"git failed: {e}"
    return snap


def run_gate_problems(snap: Dict[str, Any], freeze_ok: bool, freeze_why: str) -> List[str]:
    """Pure decision of the real-run gate. Empty list = the run may start."""
    bad: List[str] = []
    if not snap.get("available"):
        bad.append(snap.get("error") or "git unavailable")
        return bad
    if not snap.get("head"):
        bad.append(f"HEAD does not resolve to a commit ({snap.get('error')})")
    tracked = snap.get("tracked") or {}
    for f in REQUIRED_TRACKED:
        if not tracked.get(f):
            bad.append(f"not tracked by Git: {f}")
    porcelain = snap.get("porcelain")
    if porcelain is None:
        bad.append("git status --porcelain could not be read")
    elif any(l.strip() for l in porcelain):
        bad.append("working tree not clean (git status --porcelain, untracked files included): "
                   + "; ".join(l for l in porcelain if l.strip())[:600])
    if not freeze_ok:
        bad.append("freeze: " + freeze_why)
    return bad


def require_run_gate(snapshot_fn=None, freeze_fn=None) -> Dict[str, Any]:
    """Raise ProtocolError unless the real run may start; return the checked state."""
    snap = (snapshot_fn or git_snapshot)()
    ok, why = (freeze_fn or freeze_status)()
    bad = run_gate_problems(snap, ok, why)
    if bad:
        raise ProtocolError("real run refused (no session, no model, no benchmark, no trace):\n  - "
                            + "\n  - ".join(bad))
    return {"checked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "head": snap["head"], "tracked": snap["tracked"], "porcelain_clean": True,
            "freeze": why}


# ------------------------------------------------------------ provenance ----

def benchmark_workload() -> Dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from argine import bench  # noqa: E402
    p = bench.DEFAULT_PROMPT
    return {"prompt_source": "argine.bench.DEFAULT_PROMPT (unchanged)", "prompt_text": p,
            "prompt_sha256": hashlib.sha256(p.encode("utf-8")).hexdigest(), "prompt_chars": len(p),
            "n_predict": N_PREDICT, "temperature": 0.0, "cache_prompt": False,
            "warmup": "argine.bench default (8-token request before the measured one), identical in both conditions"}


LARGE_FILE_POLICY = (
    "no full read of the model or Expert Store before or between the 20 launches: pre-session "
    "snapshot = canonical path, size, mtime, Argine2 fast fingerprint, plus a full SHA-256 only if "
    "already known without reading the file; missing full hashes are computed after launch 20 and "
    "the session is valid only if size and mtime did not change")


def _stat_snapshot(path: Path) -> Dict[str, Any]:
    st = path.stat()
    return {"path": str(path), "canonical_path": str(path.resolve()), "bytes": st.st_size,
            "mtime": st.st_mtime, "mtime_ns": st.st_mtime_ns,
            "mtime_iso": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds")}


def known_full_hash(path: Path, size: int, mtime: float) -> Optional[Dict[str, Any]]:
    """A full SHA-256 of exactly this file found WITHOUT reading it (same rule as EXP-027):
    Argine2's checksum cache, or a run manifest written after the file's mtime."""
    sys.path.insert(0, str(ROOT))
    from argine.research import fingerprint  # noqa: E402
    key = f"{path.resolve()}|{size}|{int(mtime)}|sha256"
    try:
        cache = json.loads(fingerprint._cache_path().read_text(encoding="utf-8"))  # noqa: SLF001
    except Exception:  # noqa: BLE001
        cache = {}
    if key in cache:
        return {"digest": cache[key], "method": "sha256", "source": f"Argine2 checksum cache, key {key}"}
    for m in sorted((ROOT / "experiments").glob("EXP-*/raw/*/run-*/manifest.json")):
        try:
            if m.stat().st_mtime < mtime:
                continue
            fp = json.loads(m.read_text(encoding="utf-8")).get("fingerprint") or {}
        except Exception:  # noqa: BLE001
            continue
        for k in ("model", "expert_store"):
            e = fp.get(k) or {}
            if (e.get("method") == "sha256" and e.get("digest") and e.get("bytes") == size
                    and str(e.get("path", "")).lower() == str(path).lower()):
                return {"digest": e["digest"], "method": "sha256",
                        "source": f"{m.relative_to(ROOT)} fingerprint.{k} (written after the file's mtime)"}
    return None


def power_state() -> Dict[str, Any]:
    out: Dict[str, Any] = {"note": "best effort; not mandatory"}
    if os.name == "nt":
        try:
            r = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True,
                               timeout=10, errors="replace")
            out["power_scheme"] = (r.stdout or "").strip() or None
        except Exception as e:  # noqa: BLE001
            out["power_scheme_error"] = str(e)[:200]
    try:
        import psutil  # type: ignore
        b = psutil.sensors_battery()
        out["ac_power"] = bool(b.power_plugged) if b is not None and b.power_plugged is not None else None
    except Exception as e:  # noqa: BLE001
        out["ac_power_error"] = str(e)[:200]
    return out


def free_ram() -> Optional[int]:
    try:
        import psutil  # type: ignore
        return int(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001
        return None


def binary_path(llama_dir: Path) -> Path:
    exe = llama_dir / "llama-server.exe"
    return exe if exe.exists() else llama_dir / "llama-server"


def build_info_check(llama_dir: Path) -> Tuple[Dict[str, Any], List[str]]:
    bad: List[str] = []
    p = llama_dir / "BUILD-INFO.json"
    if not p.exists():
        return {}, [f"{p} missing"]
    raw = p.read_bytes()
    info = json.loads(raw.decode("utf-8"))
    if "VULKAN" in str(info.get("cmake_flags", "")).upper():
        bad.append("BUILD-INFO cmake flags enable Vulkan: effective-CPU argument does not hold")
    if any("vulkan" in str(d).lower() for d in info.get("dll_dependencies") or []):
        bad.append("binary depends on vulkan-1.dll")
    if info.get("instrumentation") != INSTRUMENTATION_ID:
        bad.append(f"instrumentation {info.get('instrumentation')!r} != {INSTRUMENTATION_ID}")
    return {"path": str(p), "sha256": hashlib.sha256(raw).hexdigest(),
            "instrumentation": info.get("instrumentation"),
            "llama_cpp_commit": info.get("llama_cpp_commit"),
            "server_sha256_declared": (info.get("binaries") or {}).get("llama-server.exe"),
            "cmake_flags": info.get("cmake_flags"), "dll_dependencies": info.get("dll_dependencies")}, bad


def session_provenance(llama_dir: Path) -> Dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from argine.research import fingerprint  # noqa: E402
    bi, bad = build_info_check(llama_dir)
    if bad:
        raise ProtocolError("; ".join(bad))
    binfo = fingerprint.binary_info(binary_path(llama_dir))
    if bi.get("server_sha256_declared") and binfo.get("digest") != bi["server_sha256_declared"]:
        raise ProtocolError("llama-server SHA-256 differs from BUILD-INFO's declared hash")
    if not binfo.get("has_research_trace"):
        raise ProtocolError("binary lacks the research instrumentation marker")
    out: Dict[str, Any] = {"argine2": fingerprint.git_state(ROOT), "binary": binfo, "build_info": bi,
                           "effective_backend": {"value": "cpu", "basis": "--backend cpu, and BUILD-INFO "
                                                 "enables no Vulkan backend and no vulkan-1.dll dependency"},
                           "hardware": fingerprint.machine_identity(), "power": power_state(),
                           "free_ram_before_session": free_ram(), "python": sys.version.split()[0],
                           "platform": platform.platform(), "workload": benchmark_workload(),
                           "large_file_policy": LARGE_FILE_POLICY, "files": {}}
    for role, path in (("model", Path(MODEL_PATH)), ("expert_store", Path(STORE_PATH))):
        snap = _stat_snapshot(path)
        known = known_full_hash(path, snap["bytes"], snap["mtime"])
        out["files"][role] = {"pre_run": snap, "fast": fingerprint.file_checksum(path, full=False),
                              "full": known, "full_status": "reused (no file read)" if known else "pending post-run"}
    return out


def finalize_fingerprints(session: Dict[str, Any], llama_dir: Path, hasher=None) -> List[str]:
    """After launch 20: model/store size+mtime unchanged, missing full hashes computed,
    binary and BUILD-INFO hashes unchanged."""
    sys.path.insert(0, str(ROOT))
    from argine.research import fingerprint  # noqa: E402
    if hasher is None:
        hasher = lambda p: fingerprint.file_checksum(p, full=True)  # noqa: E731
    problems: List[str] = []
    for role, e in session["provenance"]["files"].items():
        pre = e["pre_run"]
        path = Path(pre["path"])

        def changed() -> Optional[str]:
            if not path.exists():
                return "file missing"
            st = path.stat()
            if st.st_size != pre["bytes"] or st.st_mtime_ns != pre["mtime_ns"]:
                return f"size/mtime changed (pre {pre['bytes']}/{pre['mtime_ns']}, now {st.st_size}/{st.st_mtime_ns})"
            return None
        why = changed()
        if why is None and not (e.get("full") or {}).get("digest"):
            h = hasher(path)
            why = changed()
            if why is None:
                e["full"] = {"digest": h.get("digest"), "method": h.get("method"),
                             "source": "computed after the 20 launches"}
                e["full_status"] = "computed post-run"
        e["post_run_check"] = {"checked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                               "unchanged": why is None, "problem": why}
        if why:
            problems.append(f"{role} {path}: {why}")
    b_now = hashlib.sha256(binary_path(llama_dir).read_bytes()).hexdigest()
    if b_now != session["provenance"]["binary"].get("digest"):
        problems.append("llama-server binary changed during the session")
    bi_now = hashlib.sha256((llama_dir / "BUILD-INFO.json").read_bytes()).hexdigest()
    if bi_now != session["provenance"]["build_info"]["sha256"]:
        problems.append("BUILD-INFO.json changed during the session")
    session["fingerprints_final"] = not problems
    session["valid"] = not problems
    if problems:
        session["invalid_reason"] = "; ".join(problems)
    return problems


# ------------------------------------------------------------------- run ----

def shared_log() -> Path:
    sys.path.insert(0, str(ROOT))
    from argine import telemetry  # noqa: E402
    return telemetry.log_path()


def new_records(log: Path, offset: int) -> List[Tuple[int, Dict[str, Any]]]:
    out: List[Tuple[int, Dict[str, Any]]] = []
    if not log.exists():
        return out
    with open(log, "rb") as fh:
        fh.seek(offset)
        pos = offset
        for raw in fh:
            line = raw.decode("utf-8", "replace").strip()
            if line:
                try:
                    out.append((pos, json.loads(line)))
                except json.JSONDecodeError:
                    pass
            pos += len(raw)
    return out


def _list_traces() -> List[str]:
    return sorted(p.name for p in TRACES.iterdir()) if TRACES.exists() else []


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def preflight(llama_dir: Path, parent: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Static checks plus `argine benchmark --dry-run` (GGUF header only; no llama-server,
    no model load, no inference). Returns {"ok": bool, "problems": [...], "checks": {...}}."""
    parent = dict(os.environ) if parent is None else parent
    problems: List[str] = []
    checks: Dict[str, Any] = {}
    sched = build_schedule()
    problems += check_schedule(sched)
    try:
        base = base_env(parent, llama_dir)
    except ProtocolError as e:
        problems.append(str(e))
        base = None
    if "ARGINE_RESEARCH_TRACE" in parent:
        problems.append("ARGINE_RESEARCH_TRACE is set in the parent environment")
    bp = binary_path(llama_dir)
    checks["binary"] = str(bp)
    if not bp.exists():
        problems.append(f"research binary missing: {bp}")
    bi, bad = build_info_check(llama_dir)
    checks["build_info"] = {k: bi.get(k) for k in ("sha256", "instrumentation", "llama_cpp_commit")}
    problems += bad
    for role, p in (("model", MODEL_PATH), ("expert_store", STORE_PATH)):
        ok = Path(p).exists()
        checks[role] = {"path": p, "exists": ok}
        if not ok:
            problems.append(f"{role} missing: {p}")
    if (RAW / "session.json").exists() or _read_jsonl(RAW / "records.jsonl"):
        problems.append("raw/session.json or raw/records.jsonl already exists: a new session needs a clean raw/")
    if _list_traces():
        problems.append(f"raw/traces is not empty: {_list_traces()[:5]}")
    if LOGS.exists() and any(LOGS.iterdir()):
        problems.append("raw/logs is not empty: a new session needs a clean raw/")
    ok_f, why = freeze_status()
    checks["freeze"] = why
    if not ok_f:
        problems.append("freeze: " + why)
    dry: Dict[str, Any] = {}
    if base is not None and not any("missing" in p for p in problems):
        for b in BUDGETS:
            per = {}
            for c in CONDITIONS:
                launch = next(x for x in sched if x["budget"] == b and x["condition"] == c)
                r = subprocess.run(command(b, dry_run=True), cwd=str(ROOT), env=launch_env(base, launch),
                                   capture_output=True, text=True, encoding="utf-8", errors="replace")
                txt = (r.stdout or "") + (r.stderr or "")
                cmdline = next((l.strip() for l in txt.splitlines() if "--threads-batch" in l), "")
                g = lambda pat: (re.search(pat, cmdline).group(1) if re.search(pat, cmdline) else None)  # noqa: E731
                per[c] = {"exit": r.returncode, "llama_command": cmdline,
                          "threads": g(r"--threads\s+(\d+)"), "threads_batch": g(r"--threads-batch\s+(\d+)"),
                          "n_gpu_layers": g(r"--n-gpu-layers\s+(\d+)"), "no_repack": "--no-repack" in cmdline,
                          "server_started": "avvio llama-server" in txt}
                if r.returncode != 0 or not cmdline:
                    problems.append(f"dry-run {b}-{c} failed (exit {r.returncode}): {txt[-400:]!r}")
                elif (per[c]["threads"] != str(THREADS) or per[c]["threads_batch"] != str(BATCH_THREADS)
                      or (per[c]["n_gpu_layers"] or "0") != "0" or per[c]["no_repack"] or per[c]["server_started"]):
                    problems.append(f"dry-run {b}-{c}: llama-server command differs from the protocol: {cmdline}")
            if per.get("OFF", {}).get("llama_command") != per.get("ON", {}).get("llama_command"):
                problems.append(f"budget {b}: llama-server command differs between OFF and ON")
            dry[b] = per
    checks["dry_run"] = dry
    return {"ok": not problems, "problems": problems, "checks": checks}


def run(args: argparse.Namespace) -> int:
    llama_dir = Path(args.llama_dir)
    try:
        gate = require_run_gate()                 # before any subprocess, session or trace
    except ProtocolError as e:
        print(f"PROTOCOL ERROR: {e}", file=sys.stderr)
        return 2
    pf = preflight(llama_dir)
    if not pf["ok"]:
        print("preflight failed; nothing was run:\n  - " + "\n  - ".join(pf["problems"]), file=sys.stderr)
        return 2
    try:
        gate_again = require_run_gate()           # the dry-runs must not have changed the tree
    except ProtocolError as e:
        print(f"PROTOCOL ERROR: {e}", file=sys.stderr)
        return 2
    if gate_again["head"] != gate["head"]:
        print("PROTOCOL ERROR: HEAD changed during preflight", file=sys.stderr)
        return 2
    sched = build_schedule()
    base = base_env(dict(os.environ), llama_dir)
    workload = benchmark_workload()
    session = {"experiment": EXP_ID,
               "session_id": datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8],
               "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "freeze": json.loads(FREEZE.read_text(encoding="utf-8")),
               "protocol": {"model": MODEL_NAME, "model_path": MODEL_PATH, "store_path": STORE_PATH,
                            "budgets": BUDGETS, "threads": THREADS, "batch_threads": BATCH_THREADS,
                            "read_mode": READ_MODE, "backend": BACKEND, "n_predict": N_PREDICT,
                            "repeat": REPEAT, "retries": RETRIES, "cooldown_s": COOLDOWN_S,
                            "expert_stats": True, "rounds": N_ROUNDS, "threshold_pct": THRESHOLD_PCT,
                            "factor": "ARGINE_RESEARCH_TRACE set (ON) / absent (OFF), same binary",
                            "replication_unit": "one complete benchmark launch, one measured request",
                            "decision_rule": "SUPPORTED: CI95 upper < 2% at 4G and 1.5G; FALSIFIED: CI95 "
                                             "lower > 2% at >= 1 budget; else INCONCLUSIVE"},
               "llama_dir": str(llama_dir),
               "schedule": [{**x, "command": command(x["budget"]),
                             "env_delta": env_delta(base, launch_env(base, x))} for x in sched],
               "order_balance": order_balance(sched),
               "git_gate": {"before_preflight": gate, "before_session": gate_again},
               "checked_head_commit": gate["head"],
               "preflight": pf, "provenance": session_provenance(llama_dir),
               "shared_log": str(shared_log()), "status": "started"}
    session_path, records_path = RAW / "session.json", RAW / "records.jsonl"
    _write_json(session_path, session)          # schedule recorded before launch 1
    bsha8 = (session["provenance"]["binary"].get("digest") or "")[:8] or None
    model_bytes = session["provenance"]["files"]["model"]["pre_run"]["bytes"]
    log = shared_log()
    TRACES.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    for n, x in enumerate(sched):
        if n:
            print(f"  wait {COOLDOWN_S}s")
            time.sleep(COOLDOWN_S)
        trace_path = (EXP / x["would_be_trace_file"]).resolve()
        if trace_path.exists():
            raise ProtocolError(f"{trace_path} already exists before launch {x['launch_index']}")
        env = launch_env(base, x)
        before = _list_traces()
        offset = log.stat().st_size if log.exists() else 0
        ram_before = free_ram()
        t_launch = datetime.now(timezone.utc).isoformat(timespec="seconds")
        print(f"[{x['launch_index']:>2}/20] round {x['round']}  {x['budget']}-{x['condition']}")
        logf = LOGS / f"launch-{x['launch_index']:02d}.log"
        if logf.exists():
            raise ProtocolError(f"{logf} already exists")
        with open(logf, "w", encoding="utf-8") as fh:
            proc = subprocess.run(command(x["budget"]), cwd=str(ROOT), env=env, stdout=fh,
                                  stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        log_text = logf.read_text(encoding="utf-8", errors="replace")
        appended = new_records(log, offset)
        bench = [(pos, r) for pos, r in appended if r.get("event") == "benchmark"]
        other = [r.get("event") for _, r in appended if r.get("event") != "benchmark"]
        rec = bench[0][1] if len(bench) == 1 else None
        problems = validate_record(rec, x, proc.returncode, bsha8, model_bytes, workload["prompt_chars"])
        problems += check_store_log(log_text)
        problems += check_trace_condition(x, log_text, trace_path, before, _list_traces())
        if len(bench) > 1:
            problems.append(f"{len(bench)} benchmark records appended (expected 1)")
        if other:
            problems.append(f"other events appended: {other}")
        out = dict(rec or {})
        out["exp024"] = {"experiment": EXP_ID, "session_id": session["session_id"],
                         "launch_index": x["launch_index"], "attempt": 1, "round": x["round"],
                         "position_in_round": x["position_in_round"], "budget": x["budget"],
                         "budget_bytes": x["budget_bytes"], "condition": x["condition"],
                         "trace_file": x["trace_file"],
                         "trace_bytes": trace_path.stat().st_size if trace_path.exists() else None,
                         "launch_utc": t_launch, "free_ram_before_launch": ram_before,
                         "exit_status": proc.returncode, "valid": not problems, "problems": problems,
                         "command": command(x["budget"]), "env_delta": env_delta(base, env),
                         "launch_log": str(logf.relative_to(EXP)),
                         "source": ({"shared_log": str(log), "byte_offset": bench[0][0], "ts": bench[0][1].get("ts")}
                                    if len(bench) == 1 else None)}
        _append_jsonl(records_path, out)
        if problems:
            session["status"] = f"INVALID: stopped at launch {x['launch_index']}"
            session["valid"] = False
            _write_json(session_path, session)
            print(f"LAUNCH {x['launch_index']} INVALID — session stopped (no retry):\n  - "
                  + "\n  - ".join(problems), file=sys.stderr)
            return 1
        print(f"      gen {_gen(out)} tok/s, valid")
    session["status"] = "launches complete"
    session["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write_json(session_path, session)
    return finalize(session, session_path, llama_dir)


def finalize(session: Dict[str, Any], session_path: Path, llama_dir: Path) -> int:
    print("finalising fingerprints after the 20 launches")
    problems = finalize_fingerprints(session, llama_dir)
    session["status"] = "complete" if not problems else "INVALID"
    _write_json(session_path, session)
    if problems:
        print("SESSION INVALID:\n  - " + "\n  - ".join(problems), file=sys.stderr)
        return 1
    print("session complete: run `python tools/exp024_run.py --analyze`")
    return 0


def cmd_finalize(args: argparse.Namespace) -> int:
    sp = RAW / "session.json"
    if not sp.exists():
        print("no raw/session.json", file=sys.stderr)
        return 1
    session = json.loads(sp.read_text(encoding="utf-8"))
    recs = _read_jsonl(RAW / "records.jsonl")
    try:
        paired_observations(recs, session["session_id"])
    except ProtocolError as e:
        print(f"cannot finalise: {e}", file=sys.stderr)
        return 1
    return finalize(session, sp, Path(args.llama_dir))


def cmd_analyze() -> int:
    sp = RAW / "session.json"
    if not sp.exists():
        print("no raw/session.json: the experiment has not been run", file=sys.stderr)
        return 1
    session = json.loads(sp.read_text(encoding="utf-8"))
    if session.get("status") != "complete" or not session.get("valid"):
        print(f"session status {session.get('status')!r}: no analysis of an incomplete or invalid session",
              file=sys.stderr)
        return 1
    text, summary = analyze(_read_jsonl(RAW / "records.jsonl"), session["session_id"])
    (EXP / "results.csv").write_text(text, encoding="utf-8", newline="")
    (EXP / "processed").mkdir(exist_ok=True)
    _write_json(EXP / "processed" / "summary.json", summary)
    print(text)
    for b, s in summary["budgets"].items():
        o = s["overhead_pct"]
        print(f"{b}: mean paired overhead {o['mean']:.3f}% (95% CI {o['ci_low']:.3f} .. {o['ci_high']:.3f})")
    print(f"decision: {summary['decision']['verdict']} — {summary['decision']['reason']}")
    return 0


# ------------------------------------------------------------------ main ----

def print_dry_run() -> int:
    s = build_schedule()
    print(f"{EXP_ID}: {len(s)} launches, {COOLDOWN_S}s cooldown between complete launches (none before launch 1)")
    print(f"model {MODEL_PATH}; store {STORE_PATH}; runtime {DEFAULT_LLAMA_DIR}")
    print(f"backend {BACKEND}, read mode {READ_MODE}, threads {THREADS}, threads-batch {BATCH_THREADS} "
          f"(planner), repeat {REPEAT}, n_predict {N_PREDICT}, retries {RETRIES}")
    for b in BUDGETS:
        print(f"command {b} (OFF and ON identical): " + " ".join(command(b)[1:]))
    for x in s:
        env = ("ARGINE_RESEARCH_TRACE=" + str((EXP / x["trace_file"]).resolve())) if x["trace_file"] \
            else "ARGINE_RESEARCH_TRACE absent"
        print(f"  {x['launch_index']:>2}  round {x['round']}  pos {x['position_in_round']}  "
              f"{x['budget']:>4}-{x['condition']:<3}  {env}")
    print("order balance: " + json.dumps(order_balance(s)))
    p = check_schedule(s)
    print("schedule check: " + ("OK" if not p else "; ".join(p)))
    ok, why = freeze_status()
    print(f"freeze: {why}")
    print(f"decision rule: SUPPORTED if CI95 upper < {THRESHOLD_PCT}% at both budgets; FALSIFIED if CI95 "
          f"lower > {THRESHOLD_PCT}% at >= 1 budget; otherwise INCONCLUSIVE")
    print("dry run: no subprocess started, no model read")
    return 0 if not p else 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    for f in ("--dry-run", "--preflight", "--freeze", "--run", "--finalize", "--analyze"):
        g.add_argument(f, action="store_true")
    ap.add_argument("--llama-dir", default=str(DEFAULT_LLAMA_DIR))
    a = ap.parse_args(argv)
    try:
        if a.dry_run:
            return print_dry_run()
        if a.preflight:
            r = preflight(Path(a.llama_dir))
            print(json.dumps(r["checks"], indent=2, ensure_ascii=False))
            print("preflight: " + ("OK" if r["ok"] else "FAIL\n  - " + "\n  - ".join(r["problems"])))
            return 0 if r["ok"] else 1
        if a.freeze:
            return cmd_freeze()
        if a.run:
            return run(a)
        if a.finalize:
            return cmd_finalize(a)
        return cmd_analyze()
    except ProtocolError as e:
        print(f"PROTOCOL ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
