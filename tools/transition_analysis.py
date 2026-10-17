"""Temporal, segment-aware analysis of allocation policies on a transition trace.

    python tools/transition_analysis.py TRACE OUT_DIR [--budgets 4G,2G,1.5G]
           [--coding-trace TRACE] [--replicas TRACE ...]

Where does the adaptive controller's gain over the uniform split come from:
a better layer allocation in general (A), or faster reaction to workload
transitions (B)? Everything here is a counterfactual replay of ONE router
trace (category: counterfactual / oracle; see REPLAY-FIDELITY-VALIDATION.md).

Outputs (all in OUT_DIR):
  segments.csv            per policy x budget x segment
  boundaries.csv          per policy x budget x boundary x window
  allocation_timeline.csv per epoch: per-layer slots of every adaptive variant
  allocation_distance.csv per epoch: distance of the allocation to reference vectors
  allocation_boundary.csv per boundary: allocation before / +16 / +32 / +64 / end
  attribution.csv         gain decomposition (A vs B) and oracle potentials
  cross_segment.csv       misses of segment s under the static optimum of segment s'
  adaptation_v2.csv       transient metrics that do not saturate (old adaptation.csv kept)
  adaptation_v1_audit.csv the old metric recomputed, with its floor/threshold diagnosis
  pinning.csv             pinned-set composition around boundaries
  ablations.csv           hysteresis and cost-weighting comparison
  cost_weighting_check.csv  scale-invariance proof
  bootstrap.csv           within-trace block-bootstrap CIs (paired, per segment)
  stationarity.csv        transition vs stationary coding characterization
  replication.csv         routing identity of the campaign's traces
  summary.json            headline numbers used in the write-up
  figures/*.png

Distance metric (explicit): for two allocations a, b (bytes per layer) under
budget B,  D(a, b) = 1/2 * sum_L |a_L - b_L| / B. It is the minimum fraction of
the budget that must move between layers to turn a into b (each moved byte
leaves one layer and enters another), in [0, 1].
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from argine.research import allocator, elasticity  # noqa: E402
from argine.research.controller import AdaptiveResidencyController  # noqa: E402
from argine.research.costmodel import CostModel  # noqa: E402
from argine.research.io import write_csv, write_json  # noqa: E402
from argine.research.metrics import characterize  # noqa: E402
from argine.research.policies import GlobalLRU, PerLayerLRU  # noqa: E402
from argine.research.replay import adaptation_metrics, counted_step, replay  # noqa: E402
from argine.research.trace import PHASE_DECODE, Trace  # noqa: E402

GiB = 1 << 30
PHASE = PHASE_DECODE
WINDOWS = (("pre32", -32, 0), ("post16", 0, 16), ("post32", 0, 32), ("post64", 0, 64))
STEADY = 128   # last N steps of a segment = its steady-state window


def parse_size(s: str) -> int:
    s = s.strip().upper()
    mult = {"G": GiB, "M": 1 << 20, "K": 1 << 10}
    if s[-1] in mult:
        return int(float(s[:-1]) * mult[s[-1]])
    return int(s)


def fmt_b(b: int) -> str:
    g = b / GiB
    return f"{g:g}G"


# ------------------------------------------------------------------ setup --

class Run:
    """One replay with per-epoch controller snapshots."""

    def __init__(self, name: str, res, snaps: List[Dict], budget: int):
        self.name, self.res, self.snaps, self.budget = name, res, snaps, budget


def run_policy(trace: Trace, name: str, make: Callable[[], object], budget: int,
               cost: CostModel) -> Run:
    pol = make()
    snaps: List[Dict] = []
    orig = pol.end_step

    def hooked(step):
        orig(step)
        al = pol.allocation() if hasattr(pol, "allocation") else None
        snaps.append({"step": step,
                      "alloc": dict(al) if al else None,
                      "moves": getattr(pol, "moves", 0),
                      "bytes_moved": getattr(pol, "bytes_moved", 0),
                      "pinned": ({l: set(v) for l, v in pol.pinned.items()}
                                 if getattr(pol, "pinned_fraction", 0) else None)})
    pol.end_step = hooked
    res = replay(trace, pol, budget, phase=PHASE, cost=cost)
    return Run(name, res, snaps, budget)


def variants(cost: CostModel) -> List[Tuple[str, Callable[[], object]]]:
    C = AdaptiveResidencyController
    return [
        ("global-lru", lambda: GlobalLRU()),
        ("per-layer-lru-uniform", lambda: PerLayerLRU(label="per-layer-lru-uniform")),
        ("ablate-frozen", lambda: C(epoch=64, adaptive=False, cost=cost)),
        ("adaptive-e32", lambda: C(epoch=32, cost=cost)),
        ("adaptive-e64", lambda: C(epoch=64, cost=cost)),
        ("adaptive-e128", lambda: C(epoch=128, cost=cost)),
        ("adaptive-e64-pin25", lambda: C(epoch=64, pinned_fraction=0.25, cost=cost)),
        ("adaptive-e64-pin50", lambda: C(epoch=64, pinned_fraction=0.5, cost=cost)),
        ("ablate-no-hysteresis", lambda: C(epoch=64, hysteresis=0.0, cost=cost)),
        ("ablate-no-cost-weighting", lambda: C(epoch=64, cost_weighting=False, cost=cost)),
    ]


ADAPTIVE = ("adaptive-e32", "adaptive-e64", "adaptive-e128", "adaptive-e64-pin25",
            "adaptive-e64-pin50", "ablate-no-hysteresis", "ablate-no-cost-weighting")


def counted_index(trace: Trace) -> Tuple[List[int], List[str], List[int]]:
    """Trace step index and segment of every counted step; boundaries as
    indices into the counted series (first counted step of a new segment)."""
    steps, segs = [], []
    for i, st in enumerate(trace.steps):
        if counted_step(st, PHASE):
            steps.append(i)
            segs.append(st.segment)
    bounds = [k for k in range(1, len(segs)) if segs[k] != segs[k - 1]]
    return steps, segs, bounds


def segment_ranges(segs: List[str]) -> List[Tuple[str, int, int]]:
    out, start = [], 0
    for k in range(1, len(segs) + 1):
        if k == len(segs) or segs[k] != segs[start]:
            out.append((segs[start], start, k))
            start = k
    return out


def alloc_at(run: Run, trace_step: int) -> Optional[Dict[int, int]]:
    """Allocation in force while `trace_step` is served = after end_step of the
    previous served step (snapshots are taken after every end_step)."""
    if not run.snaps or run.snaps[0]["alloc"] is None:
        return None
    prev = None
    for s in run.snaps:
        if s["step"] >= trace_step:
            break
        prev = s
    return (prev or run.snaps[0])["alloc"]


def dist(a: Dict[int, int], b: Dict[int, int], budget: int) -> float:
    ks = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in ks) / budget


def moved_between(run: Run, t0: int, t1: int) -> Dict[str, float]:
    """Allocation movement between trace steps t0 (exclusive) and t1 (inclusive):
    controller block moves, bytes moved, epochs with movement, peak per-epoch
    movement (fraction of budget)."""
    if not run.snaps or run.snaps[0]["alloc"] is None:
        return {"moves": 0, "bytes_moved": 0, "epochs_with_moves": 0,
                "peak_epoch_move_frac": 0.0, "mean_epoch_move_frac": 0.0}
    prev = None
    moves = bm = 0
    per_epoch = []
    for s in run.snaps:
        if prev is not None and t0 < s["step"] <= t1:
            dm = s["moves"] - prev["moves"]
            db = s["bytes_moved"] - prev["bytes_moved"]
            if dm or db:
                moves += dm
                bm += db
                per_epoch.append(db / run.budget)
        prev = s
    return {"moves": moves, "bytes_moved": bm, "epochs_with_moves": len(per_epoch),
            "peak_epoch_move_frac": round(max(per_epoch), 5) if per_epoch else 0.0,
            "mean_epoch_move_frac": round(statistics.mean(per_epoch), 5) if per_epoch else 0.0}


# ------------------------------------------------------- segment oracles --

def masked_trace(trace: Trace, keep_segment: str) -> Trace:
    """Same steps (all still SERVED), but only `keep_segment` is counted:
    other steps get a segment label the metric excludes."""
    t = copy.copy(trace)
    t.steps = []
    for st in trace.steps:
        s2 = copy.copy(st)
        if st.segment != keep_segment:
            s2.segment = "warmup"
        t.steps.append(s2)
    return t


def optimal_allocations(trace: Trace, budget: int, seg_names: Sequence[str]) -> Dict[str, Dict[int, int]]:
    sizes = {l: trace.meta.bytes_of(l) for l in trace.layers}
    k = elasticity.min_slots(trace, PHASE)
    uni = allocator.uniform(trace.layers, sizes, budget, trace.meta.n_experts, k)
    out = {"uniform": allocator.to_bytes(uni, sizes)}
    curves = elasticity.lru_curves(trace, PHASE)
    out["global-opt"] = allocator.to_bytes(
        allocator.dp_optimal_checked(curves, sizes, budget, k, {"uniform": uni}), sizes)
    for s in seg_names:
        c = elasticity.lru_curves(masked_trace(trace, s), PHASE)
        out[f"opt:{s}"] = allocator.to_bytes(
            allocator.dp_optimal_checked(c, sizes, budget, k, {"uniform": uni}), sizes)
    return out


class Scheduled(AdaptiveResidencyController):
    """Frozen controller (same serving code path as the adaptive variants)
    whose allocation is switched to a given vector at given trace steps,
    before the step is served. An oracle when the schedule is fitted on
    the evaluated trace."""

    def __init__(self, schedule: Dict[int, Dict[int, int]], first: Dict[int, int], cost, label):
        super().__init__(epoch=0, adaptive=False, initial=first, cost=cost, label=label)
        self.schedule = schedule

    def end_step(self, step: int) -> None:
        nxt = self.schedule.get(step + 1)
        if nxt is not None:
            self.set_allocation(nxt)


def round_to_budget(mean_slots: Dict[int, float], sizes: Dict[int, int], budget: int,
                    min_slots: int) -> Dict[int, int]:
    slots = {l: max(min_slots, int(math.floor(v))) for l, v in mean_slots.items()}
    used = sum(slots[l] * sizes[l] for l in slots)
    while used > budget:  # should not happen; shave the largest
        l = max(slots, key=lambda x: slots[x])
        slots[l] -= 1
        used -= sizes[l]
    for l in sorted(mean_slots, key=lambda x: -(mean_slots[x] - math.floor(mean_slots[x]))):
        if used + sizes[l] <= budget:
            slots[l] += 1
            used += sizes[l]
    return {l: slots[l] * sizes[l] for l in slots}


# ----------------------------------------------------------------- tables --

def window_rows(trace, run: Run, steps, segs, bounds, rng_seg, steady_of) -> List[Dict]:
    pm, pa, pb, ps = (run.res.per_step_misses, run.res.per_step_accesses,
                      run.res.per_step_bytes, run.res.per_step_stall)
    rows = []
    for j, b in enumerate(bounds):
        seg_new = segs[b]
        steady = steady_of[seg_new]
        a_before = alloc_at(run, steps[b - 1] + 1) if b else None
        for wname, lo, hi in WINDOWS + (("steady", None, None),):
            if wname == "steady":
                s0, s1 = [(r0, r1) for nm, r0, r1 in rng_seg if nm == seg_new][0]
                lo_k, hi_k = s1 - STEADY, s1
            else:
                lo_k, hi_k = b + lo, b + hi
            ks = range(lo_k, hi_k)
            m = sum(pm[k] for k in ks)
            a = sum(pa[k] for k in ks)
            n = len(ks)
            excess = sum(pm[k] - steady for k in ks) if wname != "pre32" else None
            mv = moved_between(run, steps[lo_k] - 1, steps[hi_k - 1])
            a_end = alloc_at(run, steps[hi_k - 1] + 1)
            changed = (sum(1 for l in a_before if a_before[l] != a_end.get(l))
                       if (a_before and a_end and wname != "pre32") else None)
            mag = (round(dist(a_before, a_end, run.budget), 5)
                   if (a_before and a_end and wname != "pre32") else None)
            rows.append({"policy": run.name, "budget": fmt_b(run.budget), "boundary": j + 1,
                         "boundary_counted_step": b, "boundary_trace_step": steps[b],
                         "from_segment": segs[b - 1], "to_segment": seg_new, "window": wname,
                         "steps": n, "misses_per_step": round(m / n, 3),
                         "hit_rate": round(1 - m / a, 5) if a else None,
                         "bytes_per_step": round(sum(pb[k] for k in ks) / n, 1),
                         "stall_ms_per_step": round(1000 * sum(ps[k] for k in ks) / n, 4),
                         "steady_misses_per_step": round(steady, 3),
                         "excess_misses_vs_new_steady": round(excess, 1) if excess is not None else None,
                         "moves": mv["moves"], "bytes_moved": mv["bytes_moved"],
                         "layers_changed_vs_pre": changed, "allocation_change_D": mag})
    return rows


def block_bootstrap(diff: Sequence[float], block: int = 32, n: int = 2000, seed: int = 7):
    rnd = random.Random(seed)
    N = len(diff)
    if N == 0:
        return None, None
    nb = max(1, N // block)
    means = []
    for _ in range(n):
        tot = 0.0
        cnt = 0
        for _b in range(nb):
            s = rnd.randrange(0, max(1, N - block + 1))
            chunk = diff[s:s + block]
            tot += sum(chunk)
            cnt += len(chunk)
        means.append(tot / cnt)
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


# ------------------------------------------------------------------- main --

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("out")
    ap.add_argument("--budgets", default="4G,2G,1.5G")
    ap.add_argument("--coding-trace", default=None)
    ap.add_argument("--replicas", nargs="*", default=[])
    a = ap.parse_args()
    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    trace = Trace.load(a.trace)
    cost = CostModel()
    budgets = [parse_size(x) for x in a.budgets.split(",")]
    steps, segs, bounds = counted_index(trace)
    rng_seg = segment_ranges(segs)
    seg_names = [nm for nm, _a, _b in rng_seg]
    sizes = {l: trace.meta.bytes_of(l) for l in trace.layers}
    kmin = elasticity.min_slots(trace, PHASE)
    print("segments", rng_seg, "boundaries (counted idx)", bounds,
          "trace steps", [steps[b] for b in bounds], flush=True)

    seg_rows, bnd_rows, tl_rows, dist_rows, ab_rows, attr_rows = [], [], [], [], [], []
    cross_rows, adapt2_rows, audit_rows, pin_rows, abl_rows, boot_rows = [], [], [], [], [], []
    summary: Dict = {"segments": rng_seg, "boundaries_counted": bounds,
                     "boundaries_trace": [steps[b] for b in bounds], "budgets": {}}
    per_step_store: Dict[str, Dict[str, List[float]]] = {}

    # hot sets per segment and layer (EWS90 of the segment's counted steps)
    hot: Dict[str, Dict[int, set]] = {}
    for nm, r0, r1 in rng_seg:
        cnt: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for k in range(r0, r1):
            st = trace.steps[steps[k]]
            for l in trace.layers:
                for e in st.experts(l):
                    cnt[l][e] += 1
        hot[nm] = {}
        for l, c in cnt.items():
            tot = sum(c.values())
            acc, s = 0, set()
            for e, x in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])):
                if acc >= 0.9 * tot:
                    break
                s.add(e)
                acc += x
            hot[nm][l] = s

    for B in budgets:
        bname = fmt_b(B)
        print("budget", bname, flush=True)
        runs: Dict[str, Run] = {}
        for name, make in variants(cost):
            runs[name] = run_policy(trace, name, make, B, cost)
            print("  ", name, runs[name].res.misses, flush=True)
        # steady state per segment: per policy (each policy's own steady state)
        # ---------------------------------------------------- segments
        for name, run in runs.items():
            r = run.res
            for nm, r0, r1 in rng_seg:
                ks = range(r0, r1)
                m = sum(r.per_step_misses[k] for k in ks)
                acc = sum(r.per_step_accesses[k] for k in ks)
                mv = moved_between(run, steps[r0] - 1, steps[r1 - 1])
                seg_rows.append({"policy": name, "budget": bname, "segment": nm,
                                 "counted_steps": f"{r0}-{r1 - 1}",
                                 "trace_steps": f"{steps[r0]}-{steps[r1 - 1]}",
                                 "tokens": r1 - r0, "hit_rate": round(1 - m / acc, 5),
                                 "misses": m, "misses_per_token": round(m / (r1 - r0), 3),
                                 "bytes_read": sum(r.per_step_bytes[k] for k in ks),
                                 "stall_ms_per_token": round(
                                     1000 * sum(r.per_step_stall[k] for k in ks) / (r1 - r0), 4),
                                 **mv})
            per_step_store[f"{bname}|{name}"] = {"misses": r.per_step_misses}
        # ---------------------------------------------------- boundaries
        for name, run in runs.items():
            steady_of = {}
            for nm, r0, r1 in rng_seg:
                steady_of[nm] = sum(run.res.per_step_misses[r1 - STEADY:r1]) / STEADY
            bnd_rows += window_rows(trace, run, steps, segs, bounds, rng_seg, steady_of)
        # ---------------------------------------------------- oracle allocations
        opt = optimal_allocations(trace, B, seg_names)
        # allocation timelines and distances
        for name in ADAPTIVE:
            run = runs[name]
            for s in run.snaps:
                if s["alloc"] is None:
                    continue
                # record epochs only (allocation can change only at epoch ends)
                pol_epoch = {"adaptive-e32": 32, "adaptive-e128": 128}.get(name, 64)
                if (s["step"] + 1) % pol_epoch:
                    continue
                k_c = sum(1 for t in steps if t <= s["step"])
                seg_now = segs[min(k_c, len(segs) - 1)] if k_c else segs[0]
                prev_seg = None
                for j, (nm, r0, r1) in enumerate(rng_seg):
                    if nm == seg_now and j:
                        prev_seg = rng_seg[j - 1][0]
                row = {"policy": name, "budget": bname, "trace_step": s["step"],
                       "counted_steps_done": k_c, "segment": seg_now,
                       "D_to_uniform": round(dist(s["alloc"], opt["uniform"], B), 5),
                       "D_to_global_opt": round(dist(s["alloc"], opt["global-opt"], B), 5),
                       "D_to_current_segment_opt": round(dist(s["alloc"], opt[f"opt:{seg_now}"], B), 5),
                       "D_to_previous_segment_opt": (round(dist(s["alloc"], opt[f"opt:{prev_seg}"], B), 5)
                                                     if prev_seg else None),
                       "moves_cumulative": s["moves"], "bytes_moved_cumulative": s["bytes_moved"]}
                dist_rows.append(row)
                if name in ("adaptive-e64", "adaptive-e32", "adaptive-e128"):
                    for l, by in s["alloc"].items():
                        tl_rows.append({"policy": name, "budget": bname, "trace_step": s["step"],
                                        "layer": l, "slots": by // sizes[l]})
            # per boundary allocation vectors
            for j, b in enumerate(bounds):
                seg_new = segs[b]
                end_k = [r1 for nm, r0, r1 in rng_seg if nm == seg_new][0]
                a_pre = alloc_at(run, steps[b])
                vecs = {"pre": a_pre}
                for h in (16, 32, 64):
                    vecs[f"+{h}"] = alloc_at(run, steps[min(b + h, len(steps) - 1)])
                vecs["end"] = alloc_at(run, steps[end_k - 1] + 1)
                gain = {l: vecs["end"][l] - a_pre[l] for l in a_pre}
                gainers = sorted((l for l in gain if gain[l] > 0), key=lambda l: -gain[l])
                losers = sorted((l for l in gain if gain[l] < 0), key=lambda l: gain[l])
                row = {"policy": name, "budget": bname, "boundary": j + 1,
                       "from_segment": segs[b - 1], "to_segment": seg_new}
                for k_, v in vecs.items():
                    row[f"D_pre_to_{k_}"] = round(dist(a_pre, v, B), 5)
                    row[f"D_{k_}_to_new_seg_opt"] = round(dist(v, opt[f"opt:{seg_new}"], B), 5)
                    row[f"D_{k_}_to_old_seg_opt"] = round(dist(v, opt[f"opt:{segs[b - 1]}"], B), 5)
                    row[f"D_{k_}_to_global_opt"] = round(dist(v, opt["global-opt"], B), 5)
                row["D_old_seg_opt_to_new_seg_opt"] = round(
                    dist(opt[f"opt:{segs[b - 1]}"], opt[f"opt:{seg_new}"], B), 5)
                row["layers_gaining_pre_to_end"] = " ".join(f"{l}:+{gain[l] // sizes[l]}" for l in gainers[:8])
                row["layers_losing_pre_to_end"] = " ".join(f"{l}:{gain[l] // sizes[l]}" for l in losers[:8])
                row["n_layers_changed_pre_to_end"] = len(gainers) + len(losers)
                opt_shift = {l: opt[f"opt:{seg_new}"][l] - opt[f"opt:{segs[b - 1]}"][l] for l in a_pre}
                num = sum(gain[l] * opt_shift[l] for l in a_pre)
                den = math.sqrt(sum(g * g for g in gain.values()) * sum(o * o for o in opt_shift.values()))
                row["cos_adaptive_shift_vs_optimal_shift"] = round(num / den, 4) if den else None
                ab_rows.append(row)

        # ---------------------------------------------------- attribution
        e64 = runs["adaptive-e64"]
        frozen = runs["ablate-frozen"]
        uni = runs["per-layer-lru-uniform"]
        # adaptive's own mean allocation over counted steps, as a fixed allocation
        mean_slots = defaultdict(float)
        for k in range(len(steps)):
            al = alloc_at(e64, steps[k])
            for l, by in al.items():
                mean_slots[l] += by / sizes[l] / len(steps)
        a_mean = round_to_budget(mean_slots, sizes, B, kmin)
        # adaptive's eventual allocation per segment (mean over its steady window)
        a_seg_adapt = {}
        for nm, r0, r1 in rng_seg:
            ms = defaultdict(float)
            for k in range(r1 - STEADY, r1):
                for l, by in alloc_at(e64, steps[k]).items():
                    ms[l] += by / sizes[l] / STEADY
            a_seg_adapt[nm] = round_to_budget(ms, sizes, B, kmin)
        a_end = alloc_at(e64, steps[-1] + 1)

        def static(alloc, label):
            return run_policy(trace, label, lambda: AdaptiveResidencyController(
                epoch=0, adaptive=False, initial=alloc, cost=cost, label=label), B, cost)

        def piecewise(per_seg: Dict[str, Dict[int, int]], label):
            sched = {}
            for nm, r0, r1 in rng_seg[1:]:
                # switch before the new segment's prompt step (first step of the segment)
                first_trace = min(i for i, st in enumerate(trace.steps) if st.segment == nm)
                sched[first_trace] = per_seg[nm]
            first = per_seg[rng_seg[0][0]]
            return run_policy(trace, label, lambda: Scheduled(sched, first, cost, label), B, cost)

        extra = {
            "static:adaptive-mean": static(a_mean, "static:adaptive-mean"),
            "static:adaptive-final": static(a_end, "static:adaptive-final"),
            "static:global-opt (oracle)": static(opt["global-opt"], "static:global-opt"),
            "piecewise:adaptive-segment-steady (instant switch)":
                piecewise(a_seg_adapt, "piecewise:adaptive-segment-steady"),
            "piecewise:segment-opt (oracle, instant switch)":
                piecewise({nm: opt[f"opt:{nm}"] for nm in seg_names}, "piecewise:segment-opt"),
        }
        for s in seg_names:
            extra[f"static:opt:{s} (oracle)"] = static(opt[f"opt:{s}"], f"static:opt:{s}")
        M = {name: r.res.misses for name, r in runs.items()}
        M.update({name: r.res.misses for name, r in extra.items()})
        base_u, base_f, ad = M["per-layer-lru-uniform"], M["ablate-frozen"], M["adaptive-e64"]
        for name in list(M):
            attr_rows.append({"budget": bname, "policy_or_allocation": name, "misses": M[name],
                              "hit_rate": round(1 - M[name] / e64.res.accesses, 6),
                              "misses_vs_frozen": M[name] - base_f,
                              "misses_vs_uniform": M[name] - base_u})
        for nm, r0, r1 in rng_seg:
            for src in seg_names:
                r = extra[f"static:opt:{src} (oracle)"].res
                cross_rows.append({"budget": bname, "evaluated_segment": nm, "allocation_from": src,
                                   "misses": sum(r.per_step_misses[r0:r1])})
            for lbl, r in (("frozen-uniform", frozen.res), ("global-opt", extra["static:global-opt (oracle)"].res),
                           ("adaptive-e64", e64.res)):
                cross_rows.append({"budget": bname, "evaluated_segment": nm, "allocation_from": lbl,
                                   "misses": sum(r.per_step_misses[r0:r1])})
        gen = {}
        for nm, r0, r1 in rng_seg:
            row = {r["allocation_from"]: r["misses"] for r in cross_rows
                   if r["budget"] == bname and r["evaluated_segment"] == nm}
            u_, d_ = row["frozen-uniform"], row[nm]
            off = [row[x] for x in seg_names if x != nm]
            gen[nm] = {"generic_share (other segments' optima)": round((u_ - sum(off) / len(off)) / (u_ - d_), 3),
                       "global_opt_share": round((u_ - row["global-opt"]) / (u_ - d_), 3),
                       "adaptive_e64_share": round((u_ - row["adaptive-e64"]) / (u_ - d_), 3)}
        G = base_f - ad
        A_part = base_f - M["static:adaptive-mean"]
        A_final = base_f - M["static:adaptive-final"]
        lag = M["piecewise:adaptive-segment-steady (instant switch)"] - ad
        P_A = base_f - M["static:global-opt (oracle)"]
        P_B = M["static:global-opt (oracle)"] - M["piecewise:segment-opt (oracle, instant switch)"]
        # boundary-local share of the gain: adaptive-e64 vs frozen in post64 windows
        d = [frozen.res.per_step_misses[k] - e64.res.per_step_misses[k] for k in range(len(steps))]
        in_post = set()
        for b in bounds:
            in_post.update(range(b, min(b + 64, len(steps))))
        gain_post = sum(d[k] for k in in_post)
        seg1 = rng_seg[0]
        gain_seg1 = sum(d[k] for k in range(seg1[1], seg1[2]))
        summary["budgets"][bname] = {
            "misses": {k: v for k, v in M.items()},
            "accesses": e64.res.accesses,
            "gain_vs_frozen_misses": G, "gain_vs_uniform_misses": base_u - ad,
            "A_fixed_mean_allocation_gain": A_part,
            "A_fixed_final_allocation_gain": A_final,
            "A_share_of_gain_mean_alloc": round(A_part / G, 3) if G else None,
            "reaction_lag_cost_vs_instant_piecewise": lag,
            "oracle_potential_A_global_static": P_A,
            "oracle_potential_B_segment_switching": P_B,
            "gain_in_post64_windows": gain_post, "post64_steps": len(in_post),
            "gain_outside_post64": G - gain_post, "outside_steps": len(steps) - len(in_post),
            "gain_in_first_segment_no_transition": gain_seg1,
            "uniform_vs_frozen_misses": base_u - base_f,
            "share_of_segment_optimum_gain": gen,
        }
        # ---------------------------------------------------- bootstrap (paired, within trace)
        for lbl, other in (("adaptive-e64 - ablate-frozen", frozen), ("adaptive-e64 - per-layer-lru-uniform", uni)):
            dd = [e64.res.per_step_misses[k] - other.res.per_step_misses[k] for k in range(len(steps))]
            for nm, r0, r1 in rng_seg + [("whole", 0, len(steps))]:
                lo, hi = block_bootstrap(dd[r0:r1])
                boot_rows.append({"budget": bname, "contrast": lbl, "segment": nm,
                                  "mean_diff_misses_per_step": round(statistics.mean(dd[r0:r1]), 3),
                                  "ci95_lo": round(lo, 3), "ci95_hi": round(hi, 3),
                                  "block": 32, "resamples": 2000})
        # ---------------------------------------------------- adaptation metric v1 audit + v2
        for name, run in runs.items():
            pm = run.res.per_step_misses
            v1 = adaptation_metrics(pm, bounds)
            for j, (b, row) in enumerate(zip(bounds, v1)):
                end_k = bounds[j + 1] if j + 1 < len(bounds) else len(pm)
                seg = pm[b:end_k]
                steady_half = sum(seg[len(seg) // 2:]) / (len(seg) - len(seg) // 2)
                roll16 = sum(seg[:16]) / 16
                sd = statistics.pstdev(seg[len(seg) // 2:])
                audit_rows.append({
                    "policy": name, "budget": bname, "boundary": j + 1,
                    "adaptation_steps_v1": row.get("adaptation_steps"),
                    "at_floor_16": row.get("adaptation_steps") == 16,
                    "steady_v1": round(steady_half, 3),
                    "first_window_mean_misses": round(roll16, 3),
                    "threshold_v1 (steady x 1.10)": round(1.1 * steady_half, 3),
                    "first_window_below_threshold": roll16 <= 1.1 * steady_half + 1e-9,
                    "tolerance_band_misses_per_step": round(0.1 * steady_half, 3),
                    "first_window_excess_per_step": round(roll16 - steady_half, 3),
                    "steady_per_step_sd": round(sd, 3),
                    "band_in_sd_units": round(0.1 * steady_half / sd, 2) if sd else None})
                # v2: absolute, non-saturating
                steady = sum(pm[end_k - STEADY:end_k]) / STEADY
                exc = [pm[k] - steady for k in range(b, min(b + 64, end_k))]
                cum = list(_accum(exc))
                total64 = cum[-1] if cum else 0.0
                half = next((i + 1 for i, c in enumerate(cum) if total64 > 0 and c >= 0.5 * total64), None)
                se64 = statistics.pstdev(pm[end_k - STEADY:end_k]) * math.sqrt(64)
                conv = None
                if run.snaps and run.snaps[0]["alloc"] is not None:
                    a_fin = alloc_at(run, steps[end_k - 1] + 1)
                    for k in range(b, end_k):
                        if dist(alloc_at(run, steps[k]), a_fin, B) <= 0.01:
                            conv = k - b
                            break
                adapt2_rows.append({
                    "policy": name, "budget": bname, "boundary": j + 1,
                    "steady_misses_per_step (last 128)": round(steady, 3),
                    "excess16": round(sum(exc[:16]), 1), "excess32": round(sum(exc[:32]), 1),
                    "excess64": round(total64, 1),
                    "excess64_positive_part": round(sum(max(0.0, x) for x in exc), 1),
                    "excess64_noise_se (sd_steady x sqrt64)": round(se64, 1),
                    "excess64_significant (>2se)": abs(total64) > 2 * se64,
                    "half_life_steps (cum. excess reaches 50% of excess64)": half,
                    "allocation_convergence_steps (D to segment-final <= 1%)": conv})
        # ---------------------------------------------------- pinning
        for name in ("adaptive-e64-pin25", "adaptive-e64-pin50"):
            run = runs[name]
            for j, b in enumerate(bounds):
                old, new = segs[b - 1], segs[b]
                end_k = [r1 for nm, r0, r1 in rng_seg if nm == new][0]
                for h in (0, 16, 32, 64, 128, 256):
                    if b + h >= end_k:
                        continue
                    t = steps[b + h]
                    snap = None
                    for s in run.snaps:
                        if s["step"] >= t:
                            break
                        snap = s
                    pins = snap["pinned"] if snap else None
                    if not pins:
                        continue
                    tot = sum(len(v) for v in pins.values())
                    old_only = sum(1 for l, v in pins.items() for e in v
                                   if e in hot[old].get(l, ()) and e not in hot[new].get(l, ()))
                    new_hot = sum(1 for l, v in pins.items() for e in v if e in hot[new].get(l, ()))
                    # live: accessed in the next 32 counted steps
                    used = defaultdict(set)
                    for k in range(b + h, min(b + h + 32, len(steps))):
                        st = trace.steps[steps[k]]
                        for l in trace.layers:
                            used[l].update(st.experts(l))
                    live = sum(1 for l, v in pins.items() for e in v if e in used[l])
                    dead_bytes = sum(sizes[l] for l, v in pins.items() for e in v if e not in used[l])
                    pin_rows.append({"policy": name, "budget": bname, "boundary": j + 1,
                                     "from_segment": old, "to_segment": new, "steps_after_boundary": h,
                                     "pinned_experts": tot,
                                     "pinned_bytes_frac_of_budget": round(
                                         sum(sizes[l] * len(v) for l, v in pins.items()) / B, 4),
                                     "share_old_hot_not_new_hot": round(old_only / tot, 4) if tot else None,
                                     "share_in_new_hot_set": round(new_hot / tot, 4) if tot else None,
                                     "share_live_next32": round(live / tot, 4) if tot else None,
                                     "dead_pinned_bytes_frac_of_budget": round(dead_bytes / B, 4)})
        # ---------------------------------------------------- ablations
        for name in ("adaptive-e64", "ablate-no-hysteresis", "ablate-no-cost-weighting",
                     "adaptive-e64-pin25", "adaptive-e64-pin50", "ablate-frozen"):
            run = runs[name]
            ex = [r for r in adapt2_rows if r["policy"] == name and r["budget"] == bname]
            abl_rows.append({"budget": bname, "policy": name, "misses": run.res.misses,
                             "hit_rate": round(run.res.hit_rate, 6),
                             "delta_misses_vs_e64": run.res.misses - e64.res.misses,
                             "moves": run.snaps[-1]["moves"] if run.snaps else 0,
                             "bytes_moved": run.snaps[-1]["bytes_moved"] if run.snaps else 0,
                             "sum_excess64_over_boundaries": round(sum(r["excess64"] for r in ex), 1),
                             "sum_excess16_over_boundaries": round(sum(r["excess16"] for r in ex), 1)})
        summary["budgets"][bname]["optimal_allocation_distances"] = {
            "D(uniform, global-opt)": round(dist(opt["uniform"], opt["global-opt"], B), 4),
            **{f"D(global-opt, opt:{s})": round(dist(opt["global-opt"], opt[f"opt:{s}"], B), 4)
               for s in seg_names},
            **{f"D(opt:{seg_names[i]}, opt:{seg_names[i + 1]})": round(
                dist(opt[f"opt:{seg_names[i]}"], opt[f"opt:{seg_names[i + 1]}"], B), 4)
               for i in range(len(seg_names) - 1)},
            "D(adaptive-mean, global-opt)": round(dist(a_mean, opt["global-opt"], B), 4),
            "D(adaptive-mean, uniform)": round(dist(a_mean, opt["uniform"], B), 4)}

    # -------------------------------------------------------- cost-weighting check
    cw_rows = cost_weighting_check(trace, budgets, cost)
    # -------------------------------------------------------- stationarity
    st_rows = stationarity(trace, rng_seg, steps, a.coding_trace)
    # -------------------------------------------------------- replication
    rep_rows = replication([a.trace] + list(a.replicas))

    for fname, rows in (("segments.csv", seg_rows), ("boundaries.csv", bnd_rows),
                        ("allocation_timeline.csv", tl_rows), ("allocation_distance.csv", dist_rows),
                        ("allocation_boundary.csv", ab_rows), ("attribution.csv", attr_rows),
                        ("cross_segment.csv", cross_rows), ("adaptation_v2.csv", adapt2_rows),
                        ("adaptation_v1_audit.csv", audit_rows), ("pinning.csv", pin_rows),
                        ("ablations.csv", abl_rows), ("cost_weighting_check.csv", cw_rows),
                        ("bootstrap.csv", boot_rows), ("stationarity.csv", st_rows),
                        ("replication.csv", rep_rows)):
        if rows:
            write_csv(rows, out / fname)
    summary["distance_metric"] = "D(a,b) = 1/2 * sum_L |a_L - b_L| / budget"
    summary["steady_window"] = f"last {STEADY} counted steps of each segment"
    write_json(summary, out / "summary.json")
    json.dump(per_step_store, open(out / "per_step_misses.json", "w"))
    figures(out, trace, steps, segs, bounds, rng_seg, per_step_store, dist_rows, tl_rows, pin_rows,
            summary, budgets)
    print("wrote", out)
    return 0


def _accum(xs):
    s = 0.0
    for x in xs:
        s += x
        yield s


class _TieAudit(AdaptiveResidencyController):
    """Weighted controller that, at every reallocation decision, also computes
    the decision the UNWEIGHTED controller would take on the same state and
    records any disagreement (receiver, donor, hysteresis test)."""

    def _reallocate(self):
        ctx = self.ctx
        slots = self._slots()
        free = ctx.budget - sum(slots[l] * ctx.bytes_of(l) for l in ctx.layers)
        limit = self.max_move * ctx.budget
        moved = 0
        n = ctx.n_experts
        U = self._utility
        while moved < limit:
            cand = [l for l in ctx.layers if slots[l] < n]
            recv = max(cand, key=lambda l: (self._value(l, slots[l]), -l), default=None)
            ru = max(cand, key=lambda l: (U(l, slots[l]), -l), default=None)
            if recv != ru:
                self.flips.append(("receiver", recv, ru, U(recv, slots[recv]), U(ru, slots[ru]),
                                   ctx.bytes_of(recv) != ctx.bytes_of(ru)))
            if recv is None:
                break
            need = ctx.bytes_of(recv)
            v_add = self._value(recv, slots[recv])
            if v_add <= 0:
                break
            if free >= need:
                slots[recv] += 1
                free -= need
                moved += need
                continue
            donors = [l for l in ctx.layers if l != recv and slots[l] > self.min_slots]
            if not donors:
                break
            don = min(donors, key=lambda l: (self._value(l, slots[l] - 1), l))
            du = min(donors, key=lambda l: (U(l, slots[l] - 1), l))
            if don != du:
                self.flips.append(("donor", don, du, U(don, slots[don] - 1), U(du, slots[du] - 1),
                                   ctx.bytes_of(don) != ctx.bytes_of(du)))
            k = -(-need // ctx.bytes_of(don))
            if slots[don] - k < self.min_slots:
                break
            v_rem = sum(self._value(don, slots[don] - i) for i in range(1, k + 1)) / k
            u_rem = sum(U(don, slots[don] - i) for i in range(1, k + 1)) / k
            tw = v_add <= (1.0 + self.hyst) * v_rem + 1e-18
            tu = U(recv, slots[recv]) <= (1.0 + self.hyst) * u_rem + 1e-18
            if tw != tu:
                self.flips.append(("hysteresis-test", recv, don, U(recv, slots[recv]), u_rem, None))
            if tw:
                break
            slots[don] -= k
            free += k * ctx.bytes_of(don) - need
            slots[recv] += 1
            moved += need
            self.moves += 1
        self.bytes_moved += moved
        self.set_allocation({l: slots[l] * ctx.bytes_of(l) for l in ctx.layers})


def cost_weighting_check(trace: Trace, budgets, cost: CostModel) -> List[Dict]:
    """Cost weighting multiplies every slot value by cost_per_miss(L)/bytes(L).
    With the calibrated cost model (per-read overhead 0) that factor is
    1/bandwidth(qd) for EVERY layer within an epoch, so in exact arithmetic the
    decisions are the same as unweighted. This audit replays the weighted
    controller and checks, at every single decision, what the unweighted
    controller would decide on the same state."""
    rows = []
    sz = sorted({trace.meta.bytes_of(l) for l in trace.layers})
    for B in budgets:
        p = _TieAudit(epoch=64, cost=cost)
        p.flips = []
        r = replay(trace, p, B, phase=PHASE, cost=cost)
        q = AdaptiveResidencyController(epoch=64, cost_weighting=False, cost=cost)
        r2 = replay(trace, q, B, phase=PHASE, cost=cost)
        ties = sum(1 for f in p.flips if f[0] != "hysteresis-test" and f[3] == f[4])
        cross = sum(1 for f in p.flips if f[5])
        facs = []
        for m in range(1, 9):
            fs = [cost.node_stall_s(m, m * b) / m / b for b in sz]
            facs.append(abs(fs[0] - fs[-1]) / fs[0])
        rows.append({"budget": fmt_b(B), "weighted_misses": r.misses, "weighted_moves": p.moves,
                     "unweighted_misses": r2.misses, "unweighted_moves": q.moves,
                     "decisions_that_differ": len(p.flips),
                     "of_which_exact_utility_ties": ties,
                     "of_which_between_the_two_expert_sizes": cross,
                     "hysteresis_test_flips": sum(1 for f in p.flips if f[0] == "hysteresis-test"),
                     "max_relative_difference_of_cost_per_byte_between_sizes": f"{max(facs):.1e}",
                     "first_flips": json.dumps([list(f[:5]) for f in p.flips[:3]])})
    return rows


def _sub(trace: Trace, idx: Sequence[int]) -> Trace:
    t = copy.copy(trace)
    t.steps = [trace.steps[i] for i in idx]
    return t


def _char_row(t: Trace, label: str) -> Dict:
    ch = characterize(t, PHASE)["summary"]
    g = ch["global_working_set"]
    sp = ch["layer_spread"]
    m = lambda k: round(sp[k]["mean"], 4) if sp.get(k) else None  # noqa: E731
    return {"source": label, "decode_tokens": ch["tokens"],
            "EWS90_global_frac": g.get("ews90_frac"), "EWS95_global_frac": g.get("ews95_frac"),
            "EWS99_global_frac": g.get("ews99_frac"),
            "ews95_frac_layer_mean": m("ews95_frac"),
            "hot_set_stability_mean": m("hot_set_stability"), "overlap_prev_mean": m("overlap_prev"),
            "reuse_median_mean": m("reuse_median"), "entropy_norm_mean": m("entropy_norm"),
            "ws_tau16_mean": m("ws_tau16")}


def stationarity(trace: Trace, rng_seg, steps, coding_trace: Optional[str]) -> List[Dict]:
    """Routing statistics depend on window length, so windows of equal length
    are compared: each 299-token transition segment, the whole transition
    trace, and the stationary coding trace (whole, and its first 299 decode
    tokens)."""
    rows = [_char_row(_sub(trace, steps), "transition: all 4 segments (workload decode)")]
    for nm, r0, r1 in rng_seg:
        rows.append(_char_row(_sub(trace, steps[r0:r1]), f"transition: {nm}"))
    if coding_trace:
        ct = Trace.load(coding_trace)
        cs = [i for i, st in enumerate(ct.steps) if counted_step(st, PHASE)]
        rows.append(_char_row(_sub(ct, cs), "coding (stationary): whole workload decode"))
        n = rng_seg[0][2] - rng_seg[0][1]
        rows.append(_char_row(_sub(ct, cs[:n]), f"coding (stationary): first {n} decode tokens"))
    return rows


def replication(paths: Sequence[str]) -> List[Dict]:
    rows = []
    for p in paths:
        t = Trace.load(p)
        h = hashlib.sha256()
        n = 0
        for st in t.steps:
            if st.segment in ("warmup", "flush-sentinel"):
                continue
            n += 1
            h.update(json.dumps([st.segment, st.n_tokens,
                                 [sorted(st.experts(l)) for l in t.layers]]).encode())
        rows.append({"trace": p, "workload_steps": n, "workload_routing_sha256": h.hexdigest()})
    return rows


# --------------------------------------------------------------- figures --

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# colour follows the policy in every figure (validated with the dataviz palette checker)
COLORS = {"adaptive-e64": "#2a78d6", "adaptive-e64-pin50": "#eb6834", "ablate-frozen": "#1baf7a",
          "global-lru": "#eda100", "per-layer-lru-uniform": "#4a3aa7", "adaptive-e64-pin25": "#e87ba4",
          "adaptive-e32": "#008300", "adaptive-e128": "#e34948"}


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.6)
    ax.tick_params(colors="#52514e", labelsize=8)


def _roll(xs, w):
    out, s = [], 0.0
    q: List[float] = []
    for x in xs:
        q.append(x)
        s += x
        if len(q) > w:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def figures(out, trace, steps, segs, bounds, rng_seg, store, dist_rows, tl_rows, pin_rows, summary, budgets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = out / "figures"
    bn = [fmt_b(b) for b in budgets]
    color = COLORS

    def bounds_on(ax):
        for b in bounds:
            ax.axvline(b, color="#8a8984", linewidth=0.8, linestyle=":")
        for nm, r0, r1 in rng_seg:
            ax.text((r0 + r1) / 2, 1.0, nm, transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=7, color="#52514e")

    # 1. gain of adaptive-e64 (and pin50) over frozen, rolling 32, per budget
    fig, axes = plt.subplots(len(bn), 1, figsize=(9, 2.4 * len(bn)), sharex=True)
    for ax, b in zip(axes, bn):
        fz = store[f"{b}|ablate-frozen"]["misses"]
        for p in ("adaptive-e64", "adaptive-e64-pin50"):
            d = [f - x for f, x in zip(fz, store[f"{b}|{p}"]["misses"])]
            ax.plot(_roll(d, 32), color=color[p], linewidth=1.6, label=p)
        ax.axhline(0, color="#52514e", linewidth=0.8)
        bounds_on(ax)
        ax.set_ylabel(f"{b}\nmisses/step saved\nvs frozen (roll 32)", fontsize=8, color="#52514e")
        _style(ax)
    axes[0].legend(fontsize=7, frameon=False, ncol=3, loc="upper right", bbox_to_anchor=(1, 1.35))
    axes[-1].set_xlabel("counted decode step", fontsize=8, color="#52514e")
    fig.tight_layout()
    fig.savefig(fd / "gain_over_time.png", dpi=130)
    plt.close(fig)

    # 2. misses per step (rolling 16) for main policies, per budget
    fig, axes = plt.subplots(len(bn), 1, figsize=(9, 2.4 * len(bn)), sharex=True)
    for ax, b in zip(axes, bn):
        for p in ("global-lru", "ablate-frozen", "adaptive-e64", "adaptive-e64-pin50"):
            ax.plot(_roll(store[f"{b}|{p}"]["misses"], 16), color=color[p], linewidth=1.4, label=p)
        bounds_on(ax)
        ax.set_ylabel(f"{b}\nmisses/step\n(roll 16)", fontsize=8, color="#52514e")
        _style(ax)
    axes[0].legend(fontsize=7, frameon=False, ncol=4, loc="upper right", bbox_to_anchor=(1, 1.35))
    axes[-1].set_xlabel("counted decode step", fontsize=8, color="#52514e")
    fig.tight_layout()
    fig.savefig(fd / "misses_over_time.png", dpi=130)
    plt.close(fig)

    # 3. allocation distance of adaptive-e64 to references, per budget
    fig, axes = plt.subplots(len(bn), 1, figsize=(9, 2.4 * len(bn)), sharex=True)
    refs = (("D_to_uniform", "uniform", PALETTE[1]), ("D_to_global_opt", "whole-trace optimum", PALETTE[0]),
            ("D_to_current_segment_opt", "current segment's optimum", PALETTE[2]))
    for ax, b in zip(axes, bn):
        rows = [r for r in dist_rows if r["policy"] == "adaptive-e64" and r["budget"] == b]
        x = [r["counted_steps_done"] for r in rows]
        for key, lbl, c in refs:
            ax.plot(x, [r[key] for r in rows], color=c, linewidth=1.6, marker="o", markersize=3, label=lbl)
        bounds_on(ax)
        ax.set_ylabel(f"{b}\nD (fraction of\nbudget)", fontsize=8, color="#52514e")
        _style(ax)
    axes[0].legend(fontsize=7, frameon=False, ncol=3, loc="upper right", bbox_to_anchor=(1, 1.35))
    axes[-1].set_xlabel("counted decode steps done (epoch ends of adaptive-e64)", fontsize=8, color="#52514e")
    fig.tight_layout()
    fig.savefig(fd / "allocation_distance.png", dpi=130)
    plt.close(fig)

    # 4. per-layer allocation heatmap (adaptive-e64, middle budget), slots relative to uniform
    b = bn[len(bn) // 2]
    rows = [r for r in tl_rows if r["policy"] == "adaptive-e64" and r["budget"] == b]
    if rows:
        ts = sorted({r["trace_step"] for r in rows})
        layers = sorted({r["layer"] for r in rows})
        grid = {(r["trace_step"], r["layer"]): r["slots"] for r in rows}
        first = {l: grid[(ts[0], l)] for l in layers}
        mat = [[grid[(t, l)] - first[l] for t in ts] for l in layers]
        vmax = max(1, max(abs(v) for row in mat for v in row))
        fig, ax = plt.subplots(figsize=(9, 4))
        im = ax.imshow(mat, aspect="auto", origin="lower", cmap="RdBu", vmin=-vmax, vmax=vmax,
                       extent=[ts[0], ts[-1], layers[0] - 0.5, layers[-1] + 0.5])
        for nm in [s for s, _a, _b in rng_seg][1:]:
            t0 = min(i for i, st in enumerate(trace.steps) if st.segment == nm)
            ax.axvline(t0, color="#0b0b0b", linewidth=0.8, linestyle=":")
        fig.colorbar(im, label="slots vs uniform start")
        ax.set_xlabel("trace step", fontsize=8)
        ax.set_ylabel("layer", fontsize=8)
        ax.set_title(f"adaptive-e64 per-layer allocation, {b} (blue = more slots)", fontsize=9)
        fig.tight_layout()
        fig.savefig(fd / "allocation_heatmap.png", dpi=130)
        plt.close(fig)

    # 5. decomposition bars per budget (misses saved vs frozen)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    keys = (("adaptive-e64", "adaptive-e64 (online)"),
            ("static:adaptive-mean", "fixed: adaptive's mean allocation"),
            ("static:global-opt (oracle)", "fixed: whole-trace optimum (oracle)"),
            ("piecewise:segment-opt (oracle, instant switch)", "per-segment optimum, instant switch (oracle)"))
    w = 0.2
    for i, (k, lbl) in enumerate(keys):
        vals = [summary["budgets"][b]["misses"]["ablate-frozen"] - summary["budgets"][b]["misses"][k] for b in bn]
        ax.bar([j + (i - 1.5) * w for j in range(len(bn))], vals, width=w * 0.92, color=PALETTE[i], label=lbl)
    ax.set_xticks(range(len(bn)))
    ax.set_xticklabels(bn)
    ax.set_ylabel("misses saved vs frozen uniform", fontsize=8, color="#52514e")
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(fd / "gain_decomposition.png", dpi=130)
    plt.close(fig)

    # 6. pinning: share of pins that are old-phase-only, after each boundary
    if pin_rows:
        fig, axes = plt.subplots(1, len(bn), figsize=(9, 3.0), sharey=True)
        styles = {1: "-", 2: "--", 3: ":"}
        for ax, b in zip(axes, bn):
            for p in ("adaptive-e64-pin25", "adaptive-e64-pin50"):
                for j in sorted({r["boundary"] for r in pin_rows}):
                    rows = [r for r in pin_rows if r["policy"] == p and r["budget"] == b and r["boundary"] == j]
                    ax.plot([r["steps_after_boundary"] for r in rows],
                            [r["share_old_hot_not_new_hot"] for r in rows],
                            color=COLORS[p], linewidth=1.6, linestyle=styles.get(j, "-"), marker="o",
                            markersize=3, label=f"{p}, boundary {j}" if b == bn[0] else None)
            ax.set_title(b, fontsize=9)
            ax.set_xlabel("steps after boundary", fontsize=8)
            _style(ax)
        axes[0].set_ylabel("pinned experts hot only\nin the OLD segment", fontsize=8)
        axes[0].legend(fontsize=6, frameon=False)
        fig.tight_layout()
        fig.savefig(fd / "pinning_stale_share.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
