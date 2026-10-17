"""Splitting one expert-memory budget M = M_1 + ... + M_L across layers.

All allocators take per-layer cost curves: curve[L][c] is the cost (misses, or
modelled stall) of giving layer L exactly c expert slots. They return slots
per layer; multiply by the per-layer expert size for bytes.

* uniform            — the implicit default of most runtimes: an equal share.
* lookahead_greedy   — utility-based allocation with lookahead (Qureshi &
                       Patt, "Utility-Based Cache Partitioning", MICRO 2006):
                       repeatedly give the next block of slots to the layer
                       whose best extension saves most cost per byte. Handles
                       non-convex curves; optimal when curves are convex.
* dp_optimal         — exact minimum of sum_L curve[L][c_L] subject to
                       sum_L c_L * size_L <= M over exact byte sizes
                       (min-plus convolution per size class, then exact
                       combination of classes). numpy only accelerates.

Every allocator respects `min_slots` per layer (the largest event of that
layer must fit, otherwise the event-atomic contract cannot be honoured).
"""

from __future__ import annotations

import math
from bisect import bisect_right
from typing import Dict, List, Optional, Sequence, Tuple

try:  # optional acceleration; results are identical without it
    import numpy as _np  # type: ignore
except Exception:  # noqa: BLE001
    _np = None


Curves = Dict[int, Sequence[float]]


def uniform(layers: Sequence[int], sizes: Dict[int, int], budget: int,
            n_experts: int, min_slots: int = 0) -> Dict[int, int]:
    """Equal bytes per layer, rounded down to whole experts."""
    per = budget // max(1, len(layers))
    out = {}
    for l in layers:
        c = min(n_experts, per // max(1, sizes[l]))
        out[l] = max(c, min(min_slots, n_experts))
    _check(out, sizes, budget)
    return out


def _check(slots: Dict[int, int], sizes: Dict[int, int], budget: int) -> None:
    used = sum(c * sizes[l] for l, c in slots.items())
    if used > budget:
        raise ValueError(f"allocation uses {used} bytes > budget {budget}: "
                         f"min_slots cannot be honoured at this budget")


def lookahead_greedy(curves: Curves, sizes: Dict[int, int], budget: int,
                     min_slots: int = 0) -> Dict[int, int]:
    layers = sorted(curves)
    slots = {l: min(min_slots, len(curves[l]) - 1) for l in layers}
    left = budget - sum(slots[l] * sizes[l] for l in layers)
    if left < 0:
        raise ValueError("budget below min_slots for every layer")
    while True:
        best = None  # (ratio, layer, delta)
        for l in layers:
            cur = curves[l]
            c0 = slots[l]
            s = sizes[l]
            max_d = min(len(cur) - 1 - c0, left // s)
            if max_d <= 0:
                continue
            base = cur[c0]
            for d in range(1, max_d + 1):
                gain = base - cur[c0 + d]
                if gain <= 0:
                    continue
                r = gain / (d * s)
                if best is None or r > best[0] + 1e-15:
                    best = (r, l, d)
        if best is None:
            break
        _r, l, d = best
        slots[l] += d
        left -= d * sizes[l]
    _check(slots, sizes, budget)
    return slots


class AllocationInvariantError(AssertionError):
    """An allocator returned a worse objective than a feasible baseline."""


def _class_curve(curves: Curves, layers: Sequence[int], lo: int):
    """Min-plus convolution of the curves of `layers`: best[t] = minimum total
    cost with exactly t slots spread over these layers (each layer >= lo
    slots), plus back-pointers. Exact; integer slot counts only."""
    best = {0: 0.0}
    back: List[Dict[int, int]] = []
    for l in layers:
        cur = curves[l]
        top = len(cur) - 1
        lo_l = min(lo, top)
        nxt: Dict[int, float] = {}
        arg: Dict[int, int] = {}
        for t, v in best.items():
            for c in range(lo_l, top + 1):
                val = v + float(cur[c])
                tt = t + c
                if tt not in nxt or val < nxt[tt] - 1e-12:
                    nxt[tt] = val
                    arg[tt] = c
        best = nxt
        back.append(arg)
    return best, back


def _class_curve_np(curves: Curves, layers: Sequence[int], lo: int):
    INF = float("inf")
    best = _np.array([0.0])
    back = []
    for l in layers:
        cur = _np.asarray(curves[l], dtype=float)
        top = len(cur) - 1
        lo_l = min(lo, top)
        size = len(best) + top
        nxt = _np.full(size, INF)
        arg = _np.full(size, -1, dtype=_np.int64)
        for c in range(lo_l, top + 1):
            cand = best + cur[c]
            seg = nxt[c:c + len(best)]
            better = cand < seg - 1e-12
            seg[better] = cand[better]
            arg[c:c + len(best)][better] = c
        best = nxt
        back.append(arg)
    out = {t: float(v) for t, v in enumerate(best) if _np.isfinite(v)}
    backd = [{t: int(a[t]) for t in range(len(a)) if a[t] >= 0} for a in back]
    return out, backd


def _unwind(back: List[Dict[int, int]], layers: Sequence[int], t: int) -> Dict[int, int]:
    slots = {}
    for l, arg in zip(reversed(layers), reversed(back)):
        c = arg[t]
        slots[l] = c
        t -= c
    return slots


def dp_optimal(curves: Curves, sizes: Dict[int, int], budget: int, min_slots: int = 0,
               max_units: int = 0) -> Dict[int, int]:
    """EXACT minimum of sum_L curve[L][c_L] s.t. sum_L c_L * size_L <= budget,
    c_L >= min_slots, over integer slot counts and EXACT byte sizes.

    Method: layers are grouped by expert size; within a size class the class
    cost for every total slot count is a min-plus convolution (exact); classes
    are then combined by enumerating the slot totals of all classes but the
    last, which takes as many slots as the remaining bytes allow (class costs
    are non-increasing in slots because per-layer miss curves are).

    History: an earlier version solved the problem on a byte grid capped at
    8192 cells and rounded every size UP to the grid. On Qwen3-30B-A3B at
    4 GiB that charged 3 MiB for 2.53 MiB experts (+18.5%), excluded the
    uniform allocation from the feasible set, and produced an "optimal"
    allocation with MORE misses than uniform. `max_units` is accepted for
    compatibility and ignored. See docs/research/REPLAY-FIDELITY-VALIDATION.md.
    """
    layers = sorted(curves)
    classes: Dict[int, List[int]] = {}
    for l in layers:
        classes.setdefault(int(sizes[l]), []).append(l)
    order = sorted(classes)            # deterministic
    if len(order) > 4:
        raise ValueError(f"{len(order)} distinct expert sizes: exact combination not "
                         "implemented beyond 4 size classes")
    conv = _class_curve_np if _np is not None else _class_curve
    per = []
    for sz in order:
        best, back = conv(curves, classes[sz], min_slots)
        per.append((sz, classes[sz], best, back))
    # prefix minimum for the last class: best cost with AT MOST t slots
    sz_last, lay_last, best_last, back_last = per[-1]
    ts = sorted(best_last)
    pref: List[Tuple[int, float, int]] = []   # (t, cost, argt)
    run_c, run_t = float("inf"), None
    for t in ts:
        if best_last[t] < run_c - 1e-12:
            run_c, run_t = best_last[t], t
        pref.append((t, run_c, run_t))
    pref_t = [p[0] for p in pref]
    best_total, best_pick = float("inf"), None

    def rec(i: int, used: int, cost: float, pick: List[int]):
        nonlocal best_total, best_pick
        if i == len(per) - 1:
            room = (budget - used) // sz_last
            j = bisect_right(pref_t, room) - 1
            if j < 0:
                return
            _t, c_last, t_arg = pref[j]
            if cost + c_last < best_total - 1e-12:
                best_total, best_pick = cost + c_last, pick + [t_arg]
            return
        sz, _lay, best, _back = per[i]
        for t in sorted(best):
            u2 = used + t * sz
            if u2 > budget:
                break
            rec(i + 1, u2, cost + best[t], pick + [t])

    rec(0, 0, 0.0, [])
    if best_pick is None:
        raise ValueError("no feasible allocation (budget below min_slots for every layer)")
    slots: Dict[int, int] = {}
    for (sz, lay, _best, back), t in zip(per, best_pick):
        slots.update(_unwind(back, lay, t))
    _check(slots, sizes, budget)
    return slots


def dp_optimal_checked(curves: Curves, sizes: Dict[int, int], budget: int, min_slots: int = 0,
                       baselines: Optional[Dict[str, Dict[int, int]]] = None) -> Dict[int, int]:
    """dp_optimal plus the invariant: never worse than any FEASIBLE baseline
    allocation evaluated with the same curves (raises otherwise)."""
    slots = dp_optimal(curves, sizes, budget, min_slots)
    opt = total_cost(curves, slots)
    for name, b in (baselines or {}).items():
        if sum(c * sizes[l] for l, c in b.items()) > budget:
            continue
        if any(c < min_slots for c in b.values()):
            continue
        if opt > total_cost(curves, b) + 1e-9:
            raise AllocationInvariantError(
                f"optimal objective {opt} > feasible baseline '{name}' {total_cost(curves, b)}")
    return slots


def total_cost(curves: Curves, slots: Dict[int, int]) -> float:
    return float(sum(curves[l][min(c, len(curves[l]) - 1)] for l, c in slots.items()))


def to_bytes(slots: Dict[int, int], sizes: Dict[int, int]) -> Dict[int, int]:
    return {l: c * sizes[l] for l, c in slots.items()}
