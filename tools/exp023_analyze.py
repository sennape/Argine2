"""EXP-023 — retrospective, analysis-only synthesis of the four EXP-007 transition sessions.

    python tools/exp023_analyze.py --check     # validate the fixed sources; writes nothing
    python tools/exp023_analyze.py --freeze    # write analysis_freeze.json (fingerprints + spec)
    python tools/exp023_analyze.py --analyze   # write experiments/EXP-023-workload-drift/processed/*

What this tool is
-----------------
A deterministic reader of EXISTING evidence. It never starts a process: no
`argine benchmark`, no `argine research trace`, no llama-server, no inference,
no replay. It reads four fixed EXP-007 transition sessions (router traces of
completed, EXP-021-qualified runs and the temporal analyses already computed
from them by tools/transition_analysis.py) and writes only under
experiments/EXP-023-workload-drift/processed/.

Retrospective status
--------------------
The EXP-023 hypothesis predates the transition campaign, but the exact closure
operationalisation below is specified now, after all four traces exist and after
FH-028 and FH-029 are known. It is a retrospective synthesis and reproducibility
freeze, not an untouched confirmatory test.

A. Routing-distribution drift  [DERIVED, retrospective operationalisation]
   Decode phase, counted steps only (warm-up and flush sentinel excluded, exactly as
   transition_analysis.counted_index). Per MoE layer, the distribution of routed
   expert selections (each token's top-k selections counted once each).
   * boundary b: P_before = last W counted steps of the old segment,
                 P_after  = first W counted steps of the new segment;
   * within-segment reference: each segment is cut into consecutive,
     non-overlapping W-step windows from its start (complete windows only; the
     remainder at the segment end is not used), and the JSD of every adjacent pair
     of windows is computed. No window crosses a segment boundary;
   * reference for a boundary = the within pairs of its two adjacent segments
     (old and new); a pooled reference over all four segments is reported too;
   * W = 64, chosen for consistency with the post-boundary 64-step windows of the
     existing temporal analysis, not fitted to JSD results;
   * JSD base 2: M = (P+Q)/2, JSD = 1/2 sum_{P>0} P log2(P/M) + 1/2 sum_{Q>0} Q log2(Q/M).
     Terms with zero probability contribute 0 (lim x log x = 0); no smoothing. In [0, 1].
   Effect sizes only: no p-value and no success threshold.

B. Cross-trace transient synthesis
   From each adaptation_v2.csv: adaptive-e64 vs ablate-frozen at 4G / 2G / 1.5G and
   boundaries 1-3: signed excess16/32/64 (never converted to positive), the
   positive-part excess64 as a separate field, the noise reference and
   significance flag, half-life and the adaptive allocation convergence steps.
   From each summary.json: controller gain (gain_vs_frozen_misses), the fixed
   mean-allocation gain and its share, the segment-switching oracle potential and
   the reaction-lag cost against instantly switched segment allocations.

Formal status logic
-------------------
The hypothesis is conjunctive. Its clause "adaptive allocation shortens the
transient" is already FALSIFIED for the current controller (FH-029, replicated on
four transition traces), so the global verdict is deterministically FALSIFIED for
the current controller (`formal_verdict`, which takes no data). The FH-029 facts
are re-checked in the fixed sources as reproducibility evidence only
(`fh029_consistency`): the mean-allocation gain against the controller's own gain,
and the sign of the reaction-lag cost; the actual numbers are reported. A
contradiction raises EvidenceInconsistencyError (a provenance problem to
investigate), never a different verdict. Neither the JSD analysis nor any numeric
value is a decision threshold.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
EXP_ID = "EXP-023-workload-drift"
EXP = ROOT / "experiments" / EXP_ID
PROCESSED = EXP / "processed"
FREEZE = EXP / "analysis_freeze.json"
E7 = "experiments/EXP-007-router-characterization"

WINDOW = 64
BUDGETS = ("4G", "2G", "1.5G")
BOUNDARIES = (1, 2, 3)
POLICY_ADAPTIVE = "adaptive-e64"
POLICY_FROZEN = "ablate-frozen"
SEGMENT_ORDER = ("natural-language", "coding", "reasoning-math", "structured-json")
EXCLUDED_SEGMENTS = ("warmup", "flush-sentinel")
COMPACT_FILES = ("summary.json", "adaptation_v2.csv", "segments.csv", "cross_segment.csv", "replication.csv")
HYPOTHESIS = ("Transitions change the per-layer expert distribution measurably (JSD > within-segment "
              "noise) and cause a transient miss excess that adaptive allocation shortens.")

# The four fixed sources: the temporal analyses of EXP-007 and the run whose router trace
# each one replayed (resolved from each temporal directory's replication.csv; the qualified
# run-00-4G of each campaign in EXP-007 runtime-hit-rates.csv).
SOURCES: Tuple[Dict[str, str], ...] = (
    {"key": "30b-dev", "model": "30b", "split": "dev", "campaign": "30b-transition-dev-v2", "run": "run-00-4G",
     "temporal": f"{E7}/analysis-v3/30b-transition-run-00-4G-temporal", "model_file": "Qwen_Qwen3-30B-A3B-Q4_K_M.gguf"},
    {"key": "30b-eval", "model": "30b", "split": "eval", "campaign": "30b-transition-eval-v2", "run": "run-00-4G",
     "temporal": f"{E7}/analysis-v3/30b-transition-eval-run-00-4G-temporal", "model_file": "Qwen_Qwen3-30B-A3B-Q4_K_M.gguf"},
    {"key": "80b-dev", "model": "80b", "split": "dev", "campaign": "80b-transition-dev-v2", "run": "run-00-4G",
     "temporal": f"{E7}/analysis-v3/80b-transition-dev-run-00-4G-temporal", "model_file": "Qwen3-Next-80B-A3B-Instruct-Q3_K_M.gguf"},
    {"key": "80b-eval", "model": "80b", "split": "eval", "campaign": "80b-transition-eval-v2", "run": "run-00-4G",
     "temporal": f"{E7}/analysis-v3/80b-transition-eval-run-00-4G-temporal", "model_file": "Qwen3-Next-80B-A3B-Instruct-Q3_K_M.gguf"},
)

ADAPT_COLS = {
    "excess16": "excess16", "excess32": "excess32", "excess64": "excess64",
    "excess64_positive_part": "excess64_positive_part",
    "noise_se": "excess64_noise_se (sd_steady x sqrt64)",
    "significant": "excess64_significant (>2se)",
    "half_life": "half_life_steps (cum. excess reaches 50% of excess64)",
    "convergence": "allocation_convergence_steps (D to segment-final <= 1%)",
}
SUMMARY_KEYS = ("gain_vs_frozen_misses", "A_fixed_mean_allocation_gain", "A_share_of_gain_mean_alloc",
                "oracle_potential_B_segment_switching", "reaction_lag_cost_vs_instant_piecewise")


class ProtocolError(RuntimeError):
    """A source or a result that the fixed specification does not allow."""


# ------------------------------------------------------------------ paths ----

def source_paths(src: Dict[str, str], root: Path = ROOT) -> Dict[str, Path]:
    run_dir = root / E7 / "raw" / src["campaign"] / src["run"]
    t = root / src["temporal"]
    out = {f: t / f for f in COMPACT_FILES}
    out.update({"trace": run_dir / "trace.jsonl.gz", "manifest": run_dir / "manifest.json",
                "runtime_hit_rates": root / E7 / "runtime-hit-rates.csv"})
    return out


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _lf_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ------------------------------------------------------------------ trace ----

def load_trace(path: Path):
    sys.path.insert(0, str(ROOT))
    from argine.research.trace import Trace  # noqa: E402
    return Trace.load(path)


def counted_decode(trace) -> Tuple[List[int], List[str]]:
    """Trace indices and segments of counted decode steps (= transition_analysis.counted_index)."""
    idx, segs = [], []
    for i, st in enumerate(trace.steps):
        if st.phase == "decode" and st.segment not in EXCLUDED_SEGMENTS:
            idx.append(i)
            segs.append(st.segment)
    return idx, segs


def segment_ranges(segs: Sequence[str]) -> List[Tuple[str, int, int]]:
    out, start = [], 0
    for k in range(1, len(segs) + 1):
        if k == len(segs) or segs[k] != segs[start]:
            out.append((segs[start], start, k))
            start = k
    return out


def routing_sha(trace) -> Tuple[int, str]:
    """Same identity as transition_analysis.replication (workload steps, routing hash)."""
    h = hashlib.sha256()
    n = 0
    for st in trace.steps:
        if st.segment in EXCLUDED_SEGMENTS:
            continue
        n += 1
        h.update(json.dumps([st.segment, st.n_tokens, [sorted(st.experts(l)) for l in trace.layers]]).encode())
    return n, h.hexdigest()


def layer_counts(trace, step_idx: Sequence[int], k0: int, k1: int, layer: int) -> Counter:
    """Routed selections of `layer` over counted steps k0..k1-1 (each token's top-k once each)."""
    c: Counter = Counter()
    for k in range(k0, k1):
        for tok in trace.steps[step_idx[k]].routes.get(layer) or ():
            c.update(tok)
    return c


# -------------------------------------------------------------------- JSD ----

def jsd(p: Dict[Any, float], q: Dict[Any, float]) -> float:
    """Jensen-Shannon divergence, base 2, of two count or probability maps. Zero entries
    contribute 0; no smoothing. Result clamped at 0 against round-off."""
    sp, sq = float(sum(p.values())), float(sum(q.values()))
    if sp <= 0 or sq <= 0:
        raise ProtocolError("JSD of an empty distribution")
    keys = set(p) | set(q)
    total = 0.0
    for k in keys:
        a, b = p.get(k, 0) / sp, q.get(k, 0) / sq
        m = (a + b) / 2.0
        if a > 0:
            total += 0.5 * a * math.log2(a / m)
        if b > 0:
            total += 0.5 * b * math.log2(b / m)
    return min(1.0, max(0.0, total))


def within_windows(r0: int, r1: int, window: int = WINDOW) -> List[Tuple[int, int]]:
    """Consecutive non-overlapping complete windows from the segment start."""
    out, s = [], r0
    while s + window <= r1:
        out.append((s, s + window))
        s += window
    return out


def boundary_windows(prev: Tuple[str, int, int], nxt: Tuple[str, int, int], window: int = WINDOW):
    if prev[2] - prev[1] < window or nxt[2] - nxt[1] < window:
        raise ProtocolError(f"segment shorter than the {window}-step window at a boundary")
    if prev[2] != nxt[1]:
        raise ProtocolError("segments are not contiguous")
    return (prev[2] - window, prev[2]), (nxt[1], nxt[1] + window)


def _q(v: List[float], q: float) -> Optional[float]:
    if not v:
        return None
    s = sorted(v)
    pos = (len(s) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def jsd_rows(trace, key: str, model: str, split: str, window: int = WINDOW) -> List[Dict[str, Any]]:
    idx, segs = counted_decode(trace)
    rng = segment_ranges(segs)
    rows = []
    for layer in sorted(trace.layers):
        within: Dict[str, List[float]] = {}
        for name, r0, r1 in rng:
            ws = within_windows(r0, r1, window)
            within[name] = [jsd(layer_counts(trace, idx, a0, a1, layer), layer_counts(trace, idx, b0, b1, layer))
                            for (a0, a1), (b0, b1) in zip(ws, ws[1:])]
        pooled = [v for name, _a, _b in rng for v in within[name]]
        for b in range(1, len(rng)):
            prev, nxt = rng[b - 1], rng[b]
            (p0, p1), (n0, n1) = boundary_windows(prev, nxt, window)
            bj = jsd(layer_counts(trace, idx, p0, p1, layer), layer_counts(trace, idx, n0, n1, layer))
            ref = within[prev[0]] + within[nxt[0]]
            med = statistics.median(ref) if ref else None
            rows.append({"trace_key": key, "model": model, "split": split, "layer": layer, "boundary": b,
                         "before_segment": prev[0], "after_segment": nxt[0],
                         "before_window": f"{p0}-{p1}", "after_window": f"{n0}-{n1}",
                         "boundary_jsd": round(bj, 8), "within_n": len(ref),
                         "within_jsd_values": " ".join(f"{v:.8f}" for v in ref),
                         "within_median": None if med is None else round(med, 8),
                         "within_min": round(min(ref), 8) if ref else None,
                         "within_max": round(max(ref), 8) if ref else None,
                         "ratio_boundary_over_within_median": (round(bj / med, 6) if med else None),
                         "within_all_segments_median": round(statistics.median(pooled), 8) if pooled else None})
    return rows


def jsd_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Descriptive pooled summaries by trace (per boundary and all) and by model (all)."""
    out = []

    def summ(level, key, boundary, rs):
        bj = [r["boundary_jsd"] for r in rs]
        wm = [r["within_median"] for r in rs if r["within_median"] is not None]
        rt = [r["ratio_boundary_over_within_median"] for r in rs if r["ratio_boundary_over_within_median"] is not None]
        above_max = [r["boundary_jsd"] > r["within_max"] for r in rs if r["within_max"] is not None]
        out.append({"level": level, "key": key, "boundary": boundary, "n_rows": len(rs),
                    "boundary_jsd_median": round(statistics.median(bj), 8),
                    "boundary_jsd_q1": round(_q(bj, 0.25), 8), "boundary_jsd_q3": round(_q(bj, 0.75), 8),
                    "within_median_median": round(statistics.median(wm), 8) if wm else None,
                    "ratio_median": round(statistics.median(rt), 6) if rt else None,
                    "ratio_q1": round(_q(rt, 0.25), 6) if rt else None,
                    "ratio_q3": round(_q(rt, 0.75), 6) if rt else None,
                    "share_rows_boundary_above_within_max": (round(sum(above_max) / len(above_max), 6)
                                                             if above_max else None)})
    for key in sorted({r["trace_key"] for r in rows}, key=_trace_order):
        rs = [r for r in rows if r["trace_key"] == key]
        for b in sorted({r["boundary"] for r in rs}):
            summ("trace", key, str(b), [r for r in rs if r["boundary"] == b])
        summ("trace", key, "all", rs)
    for m in sorted({r["model"] for r in rows}):
        summ("model", m, "all", [r for r in rows if r["model"] == m])
    return out


def _trace_order(key: str) -> int:
    return [s["key"] for s in SOURCES].index(key) if key in [s["key"] for s in SOURCES] else 99


# -------------------------------------------------------------- transient ----

def _num(v: str) -> Optional[float]:
    v = (v or "").strip()
    if v == "":
        return None
    return float(v)


def transient_rows(adaptation_csv: str, key: str, model: str, split: str,
                   segments: Sequence[str]) -> List[Dict[str, Any]]:
    """adaptive-e64 vs ablate-frozen, exactly 3 budgets x 3 boundaries; signed values kept."""
    rows = list(csv.DictReader(io.StringIO(adaptation_csv)))
    missing = [c for c in ["policy", "budget", "boundary"] + list(ADAPT_COLS.values()) if c not in (rows[0] if rows else {})]
    if missing:
        raise ProtocolError(f"{key}: adaptation_v2.csv lacks columns {missing}")
    pick: Dict[Tuple[str, str, int], Dict[str, str]] = {}
    for r in rows:
        if r["policy"] not in (POLICY_ADAPTIVE, POLICY_FROZEN):
            continue
        k = (r["policy"], r["budget"], int(r["boundary"]))
        if k in pick:
            raise ProtocolError(f"{key}: duplicate adaptation_v2 row {k}")
        pick[k] = r
    out = []
    for budget in BUDGETS:
        for b in BOUNDARIES:
            try:
                f, a = pick[(POLICY_FROZEN, budget, b)], pick[(POLICY_ADAPTIVE, budget, b)]
            except KeyError as e:
                raise ProtocolError(f"{key}: missing adaptation_v2 row {e}") from None
            g = lambda r, c: _num(r[ADAPT_COLS[c]])  # noqa: E731
            out.append({"trace_key": key, "model": model, "split": split, "budget": budget, "boundary": b,
                        "before_segment": segments[b - 1], "after_segment": segments[b],
                        "frozen_excess16": g(f, "excess16"), "frozen_excess32": g(f, "excess32"),
                        "frozen_excess64": g(f, "excess64"),
                        "adaptive_excess16": g(a, "excess16"), "adaptive_excess32": g(a, "excess32"),
                        "adaptive_excess64": g(a, "excess64"),
                        "delta_excess64_adaptive_minus_frozen": round(g(a, "excess64") - g(f, "excess64"), 6),
                        "frozen_excess64_positive_part": g(f, "excess64_positive_part"),
                        "adaptive_excess64_positive_part": g(a, "excess64_positive_part"),
                        "frozen_noise_se": g(f, "noise_se"), "adaptive_noise_se": g(a, "noise_se"),
                        "frozen_significant": f[ADAPT_COLS["significant"]] == "True",
                        "adaptive_significant": a[ADAPT_COLS["significant"]] == "True",
                        "frozen_half_life_steps": g(f, "half_life"), "adaptive_half_life_steps": g(a, "half_life"),
                        "adaptive_allocation_convergence_steps": g(a, "convergence")})
    return out


def controller_rows(summary: Dict[str, Any], key: str, model: str, split: str) -> List[Dict[str, Any]]:
    out = []
    for budget in BUDGETS:
        b = (summary.get("budgets") or {}).get(budget)
        if b is None or any(k not in b for k in SUMMARY_KEYS):
            raise ProtocolError(f"{key}: summary.json lacks budget {budget} or one of {SUMMARY_KEYS}")
        out.append({"trace_key": key, "model": model, "split": split, "budget": budget,
                    "controller_gain_vs_frozen_misses": b["gain_vs_frozen_misses"],
                    "A_fixed_mean_allocation_gain": b["A_fixed_mean_allocation_gain"],
                    "A_share_of_gain_mean_alloc": b["A_share_of_gain_mean_alloc"],
                    "oracle_potential_B_segment_switching": b["oracle_potential_B_segment_switching"],
                    "reaction_lag_cost_vs_instant_piecewise": b["reaction_lag_cost_vs_instant_piecewise"]})
    return out


class EvidenceInconsistencyError(RuntimeError):
    """The reconstructed sources contradict the documented FH-029 evidence. This is a
    provenance / evidence failure to investigate; it never changes the scientific verdict."""


FH029_RECORD = {
    "clause": "adaptive allocation shortens the transient (temporal-reaction clause of the hypothesis)",
    "status": "FALSIFIED for the current controller",
    "established": "FH-029, EXP-007 30B transition DEV trace (temporal analysis, write-up TRANSITION-ADAPTIVE-GAIN.md)",
    "replicated": ["30B EVAL", "80B DEV", "80B EVAL"],
    "documented_facts": [
        "the controller's mean allocation, applied as a fixed split, gains more than the controller itself "
        "in every trace and budget (documented as >= 113%)",
        "against instantly switched per-segment splits, the reaction-lag cost is negative in every case",
    ],
}


def formal_verdict() -> Dict[str, Any]:
    """The EXP-023 global verdict. Deterministic: it derives from the historical FH-029 record
    (the conjunctive hypothesis has a clause already falsified for the current controller).
    It takes no data argument: neither the JSD characterisation nor the numerical
    consistency summaries can change it."""
    return {"verdict": "FALSIFIED",
            "scope": "for the current (untuned adaptive-e64) controller",
            "basis": FH029_RECORD,
            "reason": ("the historical hypothesis is conjunctive and its clause 'adaptive allocation shortens "
                       "the transient' is already falsified for the current controller by FH-029, replicated on "
                       "30B EVAL, 80B DEV and 80B EVAL"),
            "independent_of": ["the retrospective JSD operationalisation",
                               "the numerical consistency summaries (which are reproducibility evidence only)"],
            "does_not_mean": ["that workloads are stationary", "that workload dependence is falsified",
                              "that layer-aware allocation is useless",
                              "that no faster or different future controller could exploit transitions"]}


def fh029_consistency(ctrl: List[Dict[str, Any]], n_traces: int = 4) -> Dict[str, Any]:
    """Reproducibility check of the documented FH-029 facts against the fixed sources. It compares
    quantities with each other or with their documented sign (no new threshold) and reports the
    actual numbers. A contradiction raises EvidenceInconsistencyError; it is not a verdict input."""
    if len(ctrl) != n_traces * len(BUDGETS):
        raise ProtocolError(f"{len(ctrl)} controller rows, expected {n_traces * len(BUDGETS)}")
    gains_exceed = [r["A_fixed_mean_allocation_gain"] > r["controller_gain_vs_frozen_misses"] for r in ctrl]
    lag_negative = [r["reaction_lag_cost_vs_instant_piecewise"] < 0 for r in ctrl]
    shares = [r["A_share_of_gain_mean_alloc"] for r in ctrl]
    lags = [r["reaction_lag_cost_vs_instant_piecewise"] for r in ctrl]
    lo = min(range(len(ctrl)), key=lambda i: shares[i])
    out = {"role": "reproducibility / consistency evidence; not a decision rule",
           "rows": len(ctrl),
           "mean_allocation_gain_exceeds_controller_gain": f"{sum(gains_exceed)} of {len(ctrl)}",
           "reaction_lag_cost_negative": f"{sum(lag_negative)} of {len(ctrl)}",
           "A_share_of_gain_mean_alloc_min": shares[lo],
           "A_share_of_gain_mean_alloc_min_at": f"{ctrl[lo]['trace_key']} {ctrl[lo]['budget']}",
           "A_share_of_gain_mean_alloc_max": max(shares),
           "reaction_lag_cost_range": [min(lags), max(lags)],
           "rounding_note": (f"the minimum share is {shares[lo]} = {shares[lo] * 100:.1f}%; the historical "
                             "'>= 113%' wording is true only after rounding to an integer percentage. FH-029 is "
                             "not modified here; this is flagged for a separate documentary correction"),
           "consistent_with_documented_FH029": all(gains_exceed) and all(lag_negative)}
    if not out["consistent_with_documented_FH029"]:
        bad = [f"{r['trace_key']} {r['budget']}" for r, g, l in zip(ctrl, gains_exceed, lag_negative) if not (g and l)]
        raise EvidenceInconsistencyError("reconstructed sources contradict the documented FH-029 replication at "
                                         + ", ".join(bad) + ": investigate provenance; the verdict is not changed")
    return out


# ------------------------------------------------------------- validation ----

def validate_source(src: Dict[str, str], root: Path = ROOT, trace=None) -> Dict[str, Any]:
    """All structural and provenance checks of one source. Raises ProtocolError; reads only."""
    p = source_paths(src, root)
    for name, path in p.items():
        if not path.exists():
            raise ProtocolError(f"{src['key']}: required source file missing: {path}")
    summary = json.loads(p["summary.json"].read_text(encoding="utf-8"))
    segs = [s[0] for s in summary.get("segments") or []]
    if tuple(segs) != SEGMENT_ORDER:
        raise ProtocolError(f"{src['key']}: segments {segs} != {SEGMENT_ORDER}")
    if len(summary.get("boundaries_counted") or []) != 3:
        raise ProtocolError(f"{src['key']}: expected 3 boundaries")
    if sorted((summary.get("budgets") or {}).keys()) != sorted(BUDGETS):
        raise ProtocolError(f"{src['key']}: budgets {sorted(summary.get('budgets', {}))} != {sorted(BUDGETS)}")
    trace = trace if trace is not None else load_trace(p["trace"])
    idx, csegs = counted_decode(trace)
    rng = segment_ranges(csegs)
    if [[n, a, b] for n, a, b in rng] != summary["segments"]:
        raise ProtocolError(f"{src['key']}: trace segments {rng} differ from the temporal analysis {summary['segments']}")
    if [r[1] for r in rng[1:]] != summary["boundaries_counted"]:
        raise ProtocolError(f"{src['key']}: boundaries differ from the temporal analysis")
    if [idx[b] for b in summary["boundaries_counted"]] != summary["boundaries_trace"]:
        raise ProtocolError(f"{src['key']}: trace-step boundaries differ from the temporal analysis")
    n, h = routing_sha(trace)
    rep = list(csv.DictReader(io.StringIO(p["replication.csv"].read_text(encoding="utf-8"))))
    tail = f"{src['campaign']}/{src['run']}/trace.jsonl.gz"
    match = [r for r in rep if r["trace"].replace("\\", "/").endswith(tail)]
    if len(match) != 1:
        raise ProtocolError(f"{src['key']}: replication.csv does not name {tail} exactly once")
    if (int(match[0]["workload_steps"]), match[0]["workload_routing_sha256"]) != (n, h):
        raise ProtocolError(f"{src['key']}: routing identity differs from replication.csv ({n}, {h[:12]})")
    man = json.loads(p["manifest"].read_text(encoding="utf-8"))
    mpath = str(((man.get("fingerprint") or {}).get("model") or {}).get("path") or "")
    if man.get("workload") != "transition" or man.get("split") != src["split"] or not mpath.endswith(src["model_file"]):
        raise ProtocolError(f"{src['key']}: run manifest identity {man.get('workload')}/{man.get('split')}/{mpath}")
    qual = [r for r in csv.DictReader(io.StringIO(p["runtime_hit_rates"].read_text(encoding="utf-8")))
            if r["campaign"] == src["campaign"] and r["run"] == src["run"]]
    if len(qual) != 1 or qual[0]["qualified"] != "True" or qual[0]["exp021_closure"] != "PASS":
        raise ProtocolError(f"{src['key']}: run is not a qualified EXP-007 run in runtime-hit-rates.csv")
    transient_rows(p["adaptation_v2.csv"].read_text(encoding="utf-8"), src["key"], src["model"], src["split"], segs)
    controller_rows(summary, src["key"], src["model"], src["split"])
    for (n0, a, b) in rng:
        if b - a < WINDOW:
            raise ProtocolError(f"{src['key']}: segment {n0} shorter than {WINDOW} counted steps")
    return {"key": src["key"], "model": src["model"], "split": src["split"],
            "campaign": src["campaign"], "run": src["run"], "temporal_dir": src["temporal"],
            "segments": summary["segments"], "boundaries_counted": summary["boundaries_counted"],
            "boundaries_trace": summary["boundaries_trace"], "counted_decode_steps": len(idx),
            "workload_steps": n, "workload_routing_sha256": h,
            "routing_identity_source": f"{src['temporal']}/replication.csv row {match[0]['trace']}",
            "layers": len(trace.layers), "n_experts": trace.meta.n_experts, "top_k": trace.meta.top_k,
            "run_manifest": {"workload": man.get("workload"), "split": man.get("split"),
                             "workload_sha256": man.get("workload_sha256"), "model_path": mpath,
                             "publication_quality": man.get("publication_quality")},
            "exp007_qualification": {"qualified": qual[0]["qualified"], "exp021_closure": qual[0]["exp021_closure"]},
            "files": {name: {"path": str(path.relative_to(root)).replace("\\", "/"), "sha256": sha256_file(path)}
                      for name, path in p.items()}}


def check(root: Path = ROOT, sources: Sequence[Dict[str, str]] = SOURCES) -> List[Dict[str, Any]]:
    """Validate the fixed source set; writes nothing."""
    if [s["key"] for s in sources] != ["30b-dev", "30b-eval", "80b-dev", "80b-eval"]:
        raise ProtocolError("the source set must be exactly the four fixed transition sessions")
    return [validate_source(s, root) for s in sources]


FREEZE_VERSION = "EXP023-R1"


def freeze_keys() -> set:
    return {"experiment", "version", "supersedes", *FREEZE_SPEC, "sources", "tool_sha256", "tests_sha256",
            "argine2_head_at_freeze", "frozen_utc"}


def _source_record(s: Dict[str, Any]) -> Dict[str, Any]:
    return {"key": s["key"], "model": s["model"], "split": s["split"], "campaign": s["campaign"],
            "run": s["run"], "temporal_dir": s["temporal_dir"],
            "workload_routing_sha256": s["workload_routing_sha256"], "files": s["files"]}


def freeze_record(manifest: List[Dict[str, Any]], head: Optional[str], supersedes: Optional[str]) -> Dict[str, Any]:
    """The complete freeze record (written by --freeze, re-derived by freeze_problems)."""
    return {"experiment": EXP_ID, "version": FREEZE_VERSION, "supersedes": supersedes, **FREEZE_SPEC,
            "sources": [_source_record(s) for s in manifest],
            "tool_sha256": _lf_sha(Path(__file__).resolve()),
            "tests_sha256": _lf_sha(ROOT / "tests" / "test_exp023.py"),
            "argine2_head_at_freeze": head,
            "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def _norm(v: Any) -> Any:
    return json.loads(json.dumps(v))


def freeze_problems(manifest: List[Dict[str, Any]], freeze_path: Optional[Path] = None) -> List[str]:
    """Validate the whole active freeze: experiment/version, every FREEZE_SPEC field, no extra keys,
    the exact four source identities in order with routing identities and file fingerprints, the
    tool and test fingerprints, and a well-formed freeze HEAD. frozen_utc is not compared."""
    freeze_path = freeze_path or FREEZE
    if not freeze_path.exists():
        return ["analysis_freeze.json absent (run --freeze)"]
    f = json.loads(freeze_path.read_text(encoding="utf-8"))
    bad = []
    if f.get("experiment") != EXP_ID or f.get("version") != FREEZE_VERSION:
        bad.append(f"experiment/version {f.get('experiment')}/{f.get('version')} != {EXP_ID}/{FREEZE_VERSION}")
    extra, missing = sorted(set(f) - freeze_keys()), sorted(freeze_keys() - set(f))
    if extra or missing:
        bad.append(f"freeze keys differ (extra {extra}, missing {missing})")
    for k, v in FREEZE_SPEC.items():
        if _norm(f.get(k)) != _norm(v):
            bad.append(f"frozen specification field '{k}' differs from FREEZE_SPEC")
    want = [_source_record(s) for s in manifest]
    got = f.get("sources") or []
    ident = lambda r: [r.get(k) for k in ("key", "model", "split", "campaign", "run", "temporal_dir")]  # noqa: E731
    if [ident(r) for r in got] != [ident(r) for r in want]:
        bad.append("frozen source identities or their order differ")
    for w, g in zip(want, got):
        if g.get("workload_routing_sha256") != w["workload_routing_sha256"]:
            bad.append(f"{w['key']} routing identity differs from the freeze")
        if _norm(g.get("files")) != _norm(w["files"]):
            bad.append(f"{w['key']} source file paths or fingerprints differ from the freeze")
    if f.get("tool_sha256") != _lf_sha(Path(__file__).resolve()):
        bad.append("tools/exp023_analyze.py differs from the freeze")
    t = ROOT / "tests" / "test_exp023.py"
    if not t.exists() or f.get("tests_sha256") != _lf_sha(t):
        bad.append("tests/test_exp023.py differs from the freeze")
    h = f.get("argine2_head_at_freeze")
    if not (isinstance(h, str) and len(h) == 40 and all(c in "0123456789abcdef" for c in h)):
        bad.append(f"argine2_head_at_freeze is not a commit id: {h!r}")
    sup = f.get("supersedes")
    if sup is not None and not (freeze_path.parent / str(sup)).exists():
        bad.append(f"superseded freeze {sup} is not archived next to the active freeze")
    return bad


# --------------------------------------------------------------- analysis ----

def _csv(rows: List[Dict[str, Any]]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if v is None else v) for k, v in r.items()})
    return buf.getvalue()


def analyze(root: Path = ROOT, sources: Sequence[Dict[str, str]] = SOURCES,
            out_dir: Optional[Path] = None, manifest: Optional[List[Dict[str, Any]]] = None,
            window: int = WINDOW) -> Dict[str, Any]:
    """Compute the outputs from the given sources and write them to out_dir only."""
    out_dir = out_dir or PROCESSED
    manifest = manifest if manifest is not None else [validate_source(s, root) for s in sources]
    jrows, trows, crows = [], [], []
    for src in sources:
        p = source_paths(src, root)
        tr = load_trace(p["trace"])
        summary = json.loads(p["summary.json"].read_text(encoding="utf-8"))
        segs = [s[0] for s in summary["segments"]]
        jrows += jsd_rows(tr, src["key"], src["model"], src["split"], window)
        trows += transient_rows(p["adaptation_v2.csv"].read_text(encoding="utf-8"), src["key"], src["model"],
                                src["split"], segs)
        crows += controller_rows(summary, src["key"], src["model"], src["split"])
    jsum = jsd_summary(jrows)
    consistency = fh029_consistency(crows, n_traces=len(sources))
    v = formal_verdict()
    signed = [r["adaptive_excess64"] for r in trows] + [r["frozen_excess64"] for r in trows]
    summary_out = {
        "experiment": EXP_ID,
        "historical_hypothesis": {"text": HYPOTHESIS, "status_note": "kept verbatim; conjunctive"},
        "established_evidence": {
            "FH-028": "historical adaptation_steps has a floor at 16 and is not used; adaptation_v2.csv is used",
            "FH-029": "the controller does not win by reacting to transitions; replicated on 30B DEV/EVAL and 80B DEV/EVAL",
            "consistency_check": consistency},
        "retrospective_jsd_operationalisation": {
            "label": "[DERIVED, retrospective operationalisation]; specified after the data existed; not a confirmatory test",
            "window_counted_decode_steps": window, "jsd_base": 2,
            "reference": "adjacent within-segment window pairs of the old and new segment",
            "pooled": [r for r in jsum if r["boundary"] == "all"],
            "note": "effect sizes only; no p-value and no success threshold; does not set the verdict"},
        "cross_trace_transient_synthesis": {
            "comparison": f"{POLICY_ADAPTIVE} vs {POLICY_FROZEN}; budgets {list(BUDGETS)}; boundaries {list(BOUNDARIES)}",
            "rows": len(trows),
            "adaptive_excess64_below_frozen": sum(1 for r in trows if r["delta_excess64_adaptive_minus_frozen"] < 0),
            "adaptive_excess64_above_frozen": sum(1 for r in trows if r["delta_excess64_adaptive_minus_frozen"] > 0),
            "negative_signed_excess64_values": sum(1 for x in signed if x < 0),
            "wording": "boundary transient = signed departure from the new-segment steady state; positive part reported separately",
            "controller": crows},
        "final_status_logic": v,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "source_manifest.json").write_text(json.dumps({"experiment": EXP_ID, "sources": manifest}, indent=2) + "\n",
                                                  encoding="utf-8", newline="\n")
    (out_dir / "jsd_by_layer.csv").write_text(_csv(jrows), encoding="utf-8", newline="")
    (out_dir / "jsd_summary.csv").write_text(_csv(jsum), encoding="utf-8", newline="")
    (out_dir / "transient_summary.csv").write_text(_csv(trows), encoding="utf-8", newline="")
    (out_dir / "controller_summary.csv").write_text(_csv(crows), encoding="utf-8", newline="")
    (out_dir / "summary.json").write_text(json.dumps(summary_out, indent=2) + "\n", encoding="utf-8", newline="\n")
    return summary_out


# ----------------------------------------------------------------- freeze ----

FREEZE_SPEC = {
    "label": "retrospective synthesis / reproducibility freeze (NOT frozen before data)",
    "statements": [
        "the original EXP-023 hypothesis (2026-10-03) predates the transition campaign",
        "the exact closure operationalisation is frozen now (2026-10-12)",
        "all four source traces and their temporal analyses already exist",
        "FH-028 and FH-029 outcomes are already known",
        "the JSD operationalisation was specified after the data existed",
        "no untouched-holdout or confirmatory claim is made",
    ],
    "jsd": {"window_counted_decode_steps": WINDOW, "base": 2, "phase": "decode, counted steps only",
            "unit": "routed selections per layer (each token's top-k once each)",
            "definition": "M=(P+Q)/2; JSD = 1/2 sum_{P>0} P log2(P/M) + 1/2 sum_{Q>0} Q log2(Q/M); zero terms 0; no smoothing",
            "boundary": "last W counted steps of the old segment vs first W of the new segment",
            "within_reference": ("non-overlapping complete W-step windows from each segment start; JSD of adjacent "
                                 "pairs; reference of a boundary = pairs of its old and new segment; pooled over all "
                                 "four segments reported too"),
            "outputs": "effect sizes and descriptive summaries only"},
    "transient": {"policies": [POLICY_FROZEN, POLICY_ADAPTIVE], "budgets": list(BUDGETS),
                  "boundaries": list(BOUNDARIES), "source": "adaptation_v2.csv (signed values kept)",
                  "summary_keys": list(SUMMARY_KEYS)},
    "verdict_logic": ("deterministic: FALSIFIED for the current controller, from the historical FH-029 record "
                      "(conjunctive hypothesis; temporal-reaction clause falsified on 30B DEV and replicated on 30B "
                      "EVAL, 80B DEV, 80B EVAL). Neither the JSD characterisation nor the numerical summaries are "
                      "inputs. The FH-029 facts are re-checked as reproducibility evidence (mean-allocation gain vs "
                      "controller gain; sign of the reaction-lag cost); a contradiction is an evidence/provenance "
                      "inconsistency error to investigate, never a verdict change."),
    "output_schema": {"source_manifest.json": "sources with file paths and SHA-256",
                      "jsd_by_layer.csv": ["trace_key", "model", "split", "layer", "boundary", "before_segment",
                                           "after_segment", "before_window", "after_window", "boundary_jsd",
                                           "within_n", "within_jsd_values", "within_median", "within_min",
                                           "within_max", "ratio_boundary_over_within_median",
                                           "within_all_segments_median"],
                      "jsd_summary.csv": ["level", "key", "boundary", "n_rows", "boundary_jsd_median",
                                          "boundary_jsd_q1", "boundary_jsd_q3", "within_median_median",
                                          "ratio_median", "ratio_q1", "ratio_q3",
                                          "share_rows_boundary_above_within_max"],
                      "transient_summary.csv": "36 rows: 4 traces x 3 budgets x 3 boundaries",
                      "controller_summary.csv": "12 rows: 4 traces x 3 budgets",
                      "summary.json": ["historical_hypothesis", "established_evidence",
                                       "retrospective_jsd_operationalisation", "cross_trace_transient_synthesis",
                                       "final_status_logic"]},
}


def git_head(root: Path = ROOT) -> Optional[str]:
    """HEAD commit read from .git files (no git process is started)."""
    g = root / ".git"
    try:
        h = (g / "HEAD").read_text(encoding="utf-8").strip()
        if not h.startswith("ref:"):
            return h
        ref = h.split(None, 1)[1]
        f = g / ref
        if f.exists():
            return f.read_text(encoding="utf-8").strip()
        for line in (g / "packed-refs").read_text(encoding="utf-8").splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
    except OSError:
        return None
    return None


def cmd_freeze() -> int:
    manifest = check()
    supersedes = None
    if FREEZE.exists():
        old = FREEZE.read_bytes()
        archived = [p for p in sorted(EXP.glob("analysis_freeze_*.json")) if p.read_bytes() == old]
        if not archived:
            print("analysis_freeze.json exists: copy it byte-for-byte to a dated analysis_freeze_<label>.json "
                  "and document the amendment before refreezing", file=sys.stderr)
            return 1
        supersedes = archived[0].name
    rec = freeze_record(manifest, git_head(), supersedes)
    FREEZE.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {FREEZE.relative_to(ROOT)}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--freeze", action="store_true")
    g.add_argument("--analyze", action="store_true")
    a = ap.parse_args(argv)
    try:
        if a.check:
            m = check()
            for s in m:
                print(f"{s['key']}: {s['campaign']}/{s['run']}  counted decode {s['counted_decode_steps']}  "
                      f"segments {[x[2] - x[1] for x in s['segments']]}  routing {s['workload_routing_sha256'][:12]}  "
                      f"EXP-007 {s['exp007_qualification']}  OK")
            ctrl = []
            for src in SOURCES:
                sj = json.loads(source_paths(src)["summary.json"].read_text(encoding="utf-8"))
                ctrl += controller_rows(sj, src["key"], src["model"], src["split"])
            c = fh029_consistency(ctrl)
            print(f"FH-029 consistency (reproducibility evidence, not a decision rule): mean-allocation gain > "
                  f"controller gain {c['mean_allocation_gain_exceeds_controller_gain']}; reaction-lag cost < 0 "
                  f"{c['reaction_lag_cost_negative']}; min share {c['A_share_of_gain_mean_alloc_min']} "
                  f"({c['A_share_of_gain_mean_alloc_min_at']})")
            print(f"formal verdict (deterministic, from FH-029): {formal_verdict()['verdict']} "
                  f"{formal_verdict()['scope']}")
            fp = freeze_problems(m)
            if fp:
                print("freeze: MISMATCH\n  - " + "\n  - ".join(fp), file=sys.stderr)
                print("check: FAILED (freeze mismatch; nothing written)", file=sys.stderr)
                return 2
            print("freeze: matches")
            print("check: OK (nothing written)")
            return 0
        if a.freeze:
            return cmd_freeze()
        m = check()
        fp = freeze_problems(m)
        if fp:
            raise ProtocolError("sources or tool differ from analysis_freeze.json: " + "; ".join(fp))
        s = analyze(manifest=m)
        print(json.dumps(s["final_status_logic"], indent=2))
        return 0
    except ProtocolError as e:
        print(f"PROTOCOL ERROR: {e}", file=sys.stderr)
        return 2
    except EvidenceInconsistencyError as e:
        print(f"EVIDENCE INCONSISTENCY (investigate; the verdict is unchanged): {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
