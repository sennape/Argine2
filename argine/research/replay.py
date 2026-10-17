"""Shadow policy simulator: replay one real router trace through many policies.

Contribution D of Argine2. The model runs once; its routing decisions are
recorded (trace.py); every policy then sees exactly the same sequence of
events, so differences between policies are differences between policies,
not between runs.

What is exact and what is modelled
----------------------------------
[exact]    hits, misses, bytes read, evictions, per-layer misses, residency
           invariant — given the trace and the event-atomic contract.
[modelled] stall time, via costmodel.CostModel. Always reported next to the
           exact quantities and labelled as modelled.

The hard memory invariant is checked after every event: a policy that ever
holds more than its budget makes the replay fail loudly, because a result that
violates the budget is not a result.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .costmodel import CostModel
from .policies import Context, Policy
from .trace import PHASE_DECODE, Trace


class InvariantViolation(RuntimeError):
    pass


@dataclass
class ReplayResult:
    policy: str
    describe: Dict[str, object]
    budget: int
    phase: str
    steps: int
    tokens: int
    events: int
    accesses: int
    hits: int
    misses: int
    bytes_missed: int
    evictions: int
    overflow_events: int
    overflow_misses: int
    stall_s_modelled: float
    controller_s: float
    wall_s: float
    peak_resident: int
    per_layer: Dict[int, Dict[str, int]] = field(default_factory=dict)
    per_step_misses: List[int] = field(default_factory=list)
    per_step_bytes: List[int] = field(default_factory=list)      # counted steps, bytes missed
    per_step_accesses: List[int] = field(default_factory=list)   # counted steps, expert selections
    per_step_stall: List[float] = field(default_factory=list)
    allocation_timeline: List[Dict[str, object]] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return self.hits / self.accesses if self.accesses else 0.0

    def row(self) -> Dict[str, object]:
        t = max(1, self.tokens)
        return {
            "policy": self.policy, "budget_bytes": self.budget, "phase": self.phase,
            "tokens": self.tokens, "events": self.events, "accesses": self.accesses,
            "hits": self.hits, "misses": self.misses, "hit_rate": round(self.hit_rate, 6),
            "misses_per_token": round(self.misses / t, 4),
            "bytes_per_token": round(self.bytes_missed / t, 1),
            "evictions": self.evictions,
            "stall_ms_per_token_modelled": round(1000.0 * self.stall_s_modelled / t, 4),
            "overflow_events": self.overflow_events,
            "peak_resident_bytes": self.peak_resident,
            "controller_ms": round(1000.0 * self.controller_s, 3),
            "replay_wall_s": round(self.wall_s, 3),
        }


def make_context(trace: Trace, budget: int, sizes: Optional[Dict[int, int]] = None) -> Context:
    sz = dict(sizes or trace.meta.expert_bytes)
    for l in trace.layers:
        sz.setdefault(l, 1)
    return Context(layers=list(trace.layers), n_experts=trace.meta.n_experts, size=sz,
                   budget=int(budget), top_k=trace.meta.top_k)


SEMANTICS = ("v2", "v1-legacy")
EXCLUDED_SEGMENTS = ("warmup", "flush-sentinel")


def served_steps(trace: Trace, phase: Optional[str], semantics: str = "v2") -> List[int]:
    """Indices of the steps whose events a policy SERVES (updates state with).

    v2: every step, as in the runtime (prompt steps load experts too).
    v1-legacy: only steps of `phase` (the Argine2 0.1 behaviour, which let a
    decode-only replay skip the cache pollution of prompt processing).
    """
    if semantics == "v1-legacy":
        return [i for i, st in enumerate(trace.steps) if phase is None or st.phase == phase]
    return list(range(len(trace.steps)))


def counted_step(st, phase: Optional[str], semantics: str = "v2") -> bool:
    """Whether a step's events enter the metric (v2: phase AND workload-only)."""
    if phase is not None and st.phase != phase:
        return False
    if semantics == "v2" and st.segment in EXCLUDED_SEGMENTS:
        return False
    return True


def replay(trace: Trace, policy: Policy, budget: int, *, phase: Optional[str] = PHASE_DECODE,
           cost: Optional[CostModel] = None, sizes: Optional[Dict[int, int]] = None,
           warmup_steps: int = 0, timeline_every: int = 0,
           semantics: str = "v2") -> ReplayResult:
    """Replay `trace` through `policy` with `budget` bytes.

    phase:          which steps COUNT in the metric: "decode" (default), "prefill",
                    or None for all. Under v2 semantics every step updates the
                    policy's state regardless of `phase`, and warm-up /
                    flush-sentinel steps never count.
    semantics:      "v2" (default) or "v1-legacy" — the earlier behaviour in which
                    only `phase` steps were served at all and warm-up steps were
                    counted. Kept only to reproduce superseded reports; see
                    docs/research/REPLAY-FIDELITY-VALIDATION.md.
    warmup_steps:   additionally exclude the first N counted steps from the metric.
    timeline_every: record the policy's per-layer allocation every N served steps.
    """
    if semantics not in SEMANTICS:
        raise ValueError(f"semantics must be one of {SEMANTICS}")
    cost = cost or CostModel()
    ctx = make_context(trace, budget, sizes)
    t_start = time.perf_counter()
    policy.start(ctx)
    if policy.resident_bytes > budget:
        raise InvariantViolation(f"{policy.name}: starts with {policy.resident_bytes} > {budget}")

    layers = ctx.layers
    per_layer = {l: {"accesses": 0, "misses": 0} for l in layers}
    hits = misses = acc = events = bmiss = 0
    stall = 0.0
    ctrl = 0.0
    peak = policy.resident_bytes
    per_step_m: List[int] = []
    per_step_s: List[float] = []
    per_step_b: List[int] = []
    per_step_a: List[int] = []
    timeline: List[Dict[str, object]] = []
    n_sel = 0
    n_counted = 0
    tokens = 0
    for i in served_steps(trace, phase, semantics):
        st = trace.steps[i]
        counted = counted_step(st, phase, semantics)
        if counted:
            counted = n_counted >= warmup_steps
            n_counted += 1
        n_sel += 1
        step_m = 0
        step_s = 0.0
        step_b = 0
        step_a = 0
        for l in layers:
            ex = st.experts(l)
            if not ex:
                continue
            missed = policy.access(i, l, ex)
            if policy.resident_bytes > budget:
                raise InvariantViolation(
                    f"{policy.name}: {policy.resident_bytes} bytes resident > budget {budget} "
                    f"after step {i} layer {l}")
            peak = max(peak, policy.resident_bytes)
            if counted:
                b = ctx.bytes_of(l)
                m = len(missed)
                events += 1
                acc += len(ex)
                misses += m
                hits += len(ex) - m
                bmiss += m * b
                per_layer[l]["accesses"] += len(ex)
                per_layer[l]["misses"] += m
                s = cost.node_stall_s(m, m * b)
                stall += s
                step_m += m
                step_s += s
                step_b += m * b
                step_a += len(ex)
        t0 = time.perf_counter()
        policy.end_step(i)
        ctrl += time.perf_counter() - t0
        if counted:
            tokens += st.n_tokens
            per_step_m.append(step_m)
            per_step_s.append(step_s)
            per_step_b.append(step_b)
            per_step_a.append(step_a)
        if timeline_every and n_sel % timeline_every == 0:
            al = policy.allocation()
            if al is not None:
                timeline.append({"step": i, "allocation": dict(al)})

    return ReplayResult(
        policy=policy.name, describe={**policy.describe(), "semantics": semantics,
                                      "category": getattr(policy, "category", "counterfactual")},
        budget=int(budget),
        phase=phase or "all", steps=len(per_step_m), tokens=tokens, events=events,
        accesses=acc, hits=hits, misses=misses, bytes_missed=bmiss,
        evictions=policy.evictions, overflow_events=policy.overflow_events,
        overflow_misses=policy.overflow_misses, stall_s_modelled=stall,
        controller_s=ctrl + getattr(policy, "monitor_s", 0.0),
        wall_s=time.perf_counter() - t_start, peak_resident=peak, per_layer=per_layer,
        per_step_misses=per_step_m, per_step_stall=per_step_s, allocation_timeline=timeline,
        per_step_bytes=per_step_b, per_step_accesses=per_step_a)


def compare(trace: Trace, factories: Sequence[Tuple[str, "callable"]], budgets: Sequence[int],
            **kw) -> List[ReplayResult]:
    """Replay every policy at every budget. `factories` builds a fresh policy
    per run (policies are stateful)."""
    out = []
    for b in budgets:
        for _name, make in factories:
            out.append(replay(trace, make(), b, **kw))
    return out


def adaptation_metrics(per_step_misses: Sequence[int], boundaries: Sequence[int],
                       window: int = 16, tolerance: float = 0.10) -> List[Dict[str, object]]:
    """Behaviour around workload transitions (section 22).

    For each boundary b (index into the per-step series): the steady miss rate
    of the new segment is the mean over its last half; the adaptation time is
    the number of steps after b until the rolling mean (`window`) first falls
    within `tolerance` of that steady value; the transient excess is the sum of
    misses above steady during that time.

    LIMITATION (2026-10-04, EXP-007 transition analysis; definition kept
    unchanged so historical adaptation.csv files stay reproducible):
    `adaptation_steps` cannot be smaller than `window` (the search starts at
    k = window), and the tolerance is RELATIVE to the steady miss rate. Under
    memory pressure the steady rate is high, the band (10% of it) exceeds the
    transient, and the metric returns its floor (16) for almost every policy:
    on the 30B transition trace 27/30 rows at 2 GiB and 28/30 at 1.5 GiB.
    Use the absolute transient metrics of tools/transition_analysis.py
    (adaptation_v2.csv: excess16/32/64 against the last-128-step steady state,
    with a noise reference) instead.
    """
    out = []
    edges = list(boundaries) + [len(per_step_misses)]
    for j, b in enumerate(edges[:-1]):
        end = edges[j + 1]
        seg = per_step_misses[b:end]
        if len(seg) < 2 * window:
            out.append({"boundary_step": b, "note": "segment too short"})
            continue
        steady = sum(seg[len(seg) // 2:]) / float(len(seg) - len(seg) // 2)
        t_adapt = None
        for k in range(window, len(seg) + 1):
            roll = sum(seg[k - window:k]) / float(window)
            if roll <= steady * (1.0 + tolerance) + 1e-9:
                t_adapt = k
                break
        excess = sum(max(0.0, x - steady) for x in seg[:t_adapt or len(seg)])
        pre = per_step_misses[max(0, b - window):b]
        out.append({"boundary_step": b, "steady_misses_per_step": round(steady, 3),
                    "pre_boundary_misses_per_step": round(sum(pre) / len(pre), 3) if pre else None,
                    "adaptation_steps": t_adapt, "transient_excess_misses": round(excess, 1)})
    return out
