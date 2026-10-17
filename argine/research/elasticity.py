"""Layer Memory Elasticity Curves and Stall Cost per Byte (sections 15-16).

For every layer L the elasticity curve E_L(m) is the expected stall avoided by
giving L m resident bytes, relative to the minimum allotment that the
event-atomic contract allows (the largest event of L must fit):

    E_L(m) = [misses_L(c_min) - misses_L(c)] * cost_per_miss_L,   m = c * size_L

and its discrete derivative dE_L/dm is the marginal value of one more byte.
The question of section 15 — "does meaningful layer-specific elasticity
exist?" — is answered by how much dE_L/dm differs between layers at the same
total budget, measured here as the spread of the marginal value at the uniform
allocation, and by the in-sample gap between the uniform and the optimal
static allocation.

Curves come from exact event-atomic LRU stack distances (metrics.py) for
per-layer LRU, and from Belady-style simulation (oracle.py) for the bound.
They are computed IN-SAMPLE on the trace given; using them to choose an
allocation for a different trace is the controller's job, not this module's.

Stall Cost per Byte (SCB) at allotment c:
    SCB_L(c) = (misses_L(c) - misses_L(c+1)) * cost_per_miss_L / size_L / tokens
i.e. seconds of stall per generated token avoided per additional resident byte.
cost_per_miss_L is taken from the cost model at the trace's mean misses per
node, which keeps the metric interpretable (no fitted model, no network).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from . import allocator
from .costmodel import CostModel
from .metrics import lru_miss_curve, lru_stack_distances
from .oracle import layer_event_sets, opt_misses
from .trace import PHASE_DECODE, Trace


def layer_event_stream(trace: Trace, layer: int, phase: Optional[str] = PHASE_DECODE,
                       semantics: str = "v2"):
    """(event sets SERVED in this layer, per-event 'counted' flags), with the
    same served/counted rules as replay.replay()."""
    from .replay import counted_step, served_steps
    sets, flags = [], []
    for i in served_steps(trace, phase, semantics):
        st = trace.steps[i]
        ex = st.experts(layer)
        if ex:
            sets.append(ex)
            flags.append(counted_step(st, phase, semantics))
    return sets, flags


def lru_curves(trace: Trace, phase: Optional[str] = PHASE_DECODE,
               max_slots: Optional[int] = None, semantics: str = "v2") -> Dict[int, List[int]]:
    """misses[c] of a per-layer LRU, c = 0..n_experts, counting only counted
    events but serving all (exact; equal to replay of PerLayerLRU)."""
    n = max_slots if max_slots is not None else trace.meta.n_experts
    out = {}
    for l in trace.layers:
        sets, flags = layer_event_stream(trace, l, phase, semantics)
        dist = lru_stack_distances(sets)
        mask = [f for S, f in zip(sets, flags) for _ in sorted(set(S))]
        out[l] = lru_miss_curve([d for d, m in zip(dist, mask) if m], n)
    return out


def opt_curves(trace: Trace, grid: Sequence[int], phase: Optional[str] = PHASE_DECODE,
               semantics: str = "v2") -> Dict[int, Dict[int, int]]:
    out = {}
    for l in trace.layers:
        sets, flags = layer_event_stream(trace, l, phase, semantics)
        out[l] = {c: opt_misses(sets, c, flags) for c in grid}
    return out


def densify(points: Dict[int, int], n: int) -> List[int]:
    """Step-interpolate a curve known on a grid to every c = 0..n (taking, for
    each c, the value at the largest grid point <= c: an upper bound on misses,
    since misses are non-increasing in c)."""
    xs = sorted(points)
    out = []
    j = 0
    for c in range(n + 1):
        while j + 1 < len(xs) and xs[j + 1] <= c:
            j += 1
        out.append(points[xs[j]] if xs and xs[0] <= c else points[xs[0]])
    return out


def min_slots(trace: Trace, phase: Optional[str] = PHASE_DECODE, semantics: str = "v2") -> int:
    """Largest COUNTED event in any layer: the smallest per-layer partition in
    which every counted event can be atomic (larger served events, e.g.
    prompt steps when counting decode, overflow and are streamed)."""
    from .replay import counted_step
    m = 0
    for st in trace.steps:
        if not counted_step(st, phase, semantics):
            continue
        for l in trace.layers:
            m = max(m, len(st.experts(l)))
    return m


def cost_per_miss_s(trace: Trace, cost: CostModel, layer: int, misses_per_node: float) -> float:
    """Stall of one expert miss in `layer`, at a typical node with
    `misses_per_node` misses (queue depth effect included)."""
    m = max(1.0, misses_per_node)
    b = trace.meta.bytes_of(layer)
    return cost.node_stall_s(int(round(m)), int(round(m * b))) / m


def elasticity_rows(trace: Trace, curves: Dict[int, List[int]], cost: CostModel,
                    misses_per_node: float, phase: Optional[str] = PHASE_DECODE,
                    step: int = 1) -> List[Dict[str, object]]:
    tokens = max(1, trace.n_tokens(phase))
    cmin = min_slots(trace, phase)
    rows = []
    for l, cur in curves.items():
        size = trace.meta.bytes_of(l)
        cpm = cost_per_miss_s(trace, cost, l, misses_per_node)
        base = cur[min(cmin, len(cur) - 1)]
        for c in range(cmin, len(cur), step):
            nxt = cur[c + 1] if c + 1 < len(cur) else cur[c]
            rows.append({
                "layer": l, "slots": c, "bytes": c * size, "misses": cur[c],
                "misses_per_token": cur[c] / tokens,
                "E_stall_avoided_ms_per_token": 1000.0 * (base - cur[c]) * cpm / tokens,
                "dE_dm_ms_per_token_per_MiB": 1000.0 * (cur[c] - nxt) * cpm / tokens
                                               / (size / 2**20),
                "scb_s_per_token_per_byte": (cur[c] - nxt) * cpm / tokens / size,
            })
    return rows


def heterogeneity(trace: Trace, curves: Dict[int, List[int]], budget: int,
                  phase: Optional[str] = PHASE_DECODE) -> Dict[str, object]:
    """How unequal is the marginal value of memory across layers at this budget?

    Reports, in-sample:
      * misses with the uniform allocation and with the DP-optimal static
        allocation, and the relative excess of uniform (uniform_excess);
      * the spread (min/median/max) of the marginal misses saved by ONE more
        slot at the uniform allocation, per layer;
      * the optimal allocation itself (slots per layer) and its spread.
    A uniform_excess near 0 falsifies "uniform allocation is suboptimal" for
    this trace and budget.
    """
    sizes = {l: trace.meta.bytes_of(l) for l in trace.layers}
    n = trace.meta.n_experts
    cmin = min_slots(trace, phase)
    uni = allocator.uniform(trace.layers, sizes, budget, n, cmin)
    try:
        # exact over the same objective and constraints, and checked against the
        # uniform split (feasible by construction): optimal <= uniform
        opt = allocator.dp_optimal_checked(curves, sizes, budget, cmin, {"uniform": uni})
    except ValueError as e:
        return {"budget_bytes": budget, "error": str(e)}
    m_uni = allocator.total_cost(curves, uni)
    m_opt = allocator.total_cost(curves, opt)
    marg = []
    for l in trace.layers:
        c = uni[l]
        cur = curves[l]
        marg.append(cur[c] - cur[c + 1] if c + 1 < len(cur) else 0)
    ms = sorted(marg)
    alloc = sorted(opt.values())
    return {
        "budget_bytes": budget,
        "uniform_slots_per_layer": uni[trace.layers[0]],
        "misses_uniform": m_uni,
        "misses_optimal_static": m_opt,
        "uniform_excess": (m_uni - m_opt) / m_opt if m_opt else (0.0 if m_uni == 0 else float("inf")),
        "marginal_at_uniform_min": ms[0],
        "marginal_at_uniform_median": ms[len(ms) // 2],
        "marginal_at_uniform_max": ms[-1],
        "optimal_slots_min": alloc[0],
        "optimal_slots_median": alloc[len(alloc) // 2],
        "optimal_slots_max": alloc[-1],
        "optimal_allocation": {str(l): c for l, c in sorted(opt.items())},
    }
