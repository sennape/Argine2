"""Offline references that know the future (section 18).

They are not deployable. They answer: how far is an online policy from what
perfect knowledge of the trace would have allowed?

* BeladyGlobal / BeladyPerLayer — evict the resident expert whose next use is
  farthest in the future (never-used-again first). For single-key requests and
  uniform sizes this is Belady's MIN, which is optimal. Events here are
  multi-key (all members pinned together) and sizes may differ between layers;
  farthest-next-use is then a strong reference but NOT proven optimal.
  Results are labelled "Belady-style".
* opt_layer_curve — per-layer misses of Belady-style replacement for every
  slot count, used to find the best possible static per-layer allocation.
* oracle_static_allocation — the per-layer allocation that minimises replayed
  misses ON THIS SAME TRACE (in-sample). An upper bound for any static
  layer-aware allocation, not an achievable result.
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .policies import Context, Key, LRUScope, Policy
from .trace import PHASE_DECODE, Trace

INF = float("inf")


def event_list(trace: Trace, phase: Optional[str] = PHASE_DECODE, semantics: str = "v2"
               ) -> List[Tuple[int, int, Tuple[int, ...]]]:
    """The exact event sequence replay.replay() SERVES (not only counts)."""
    from .replay import served_steps
    out = []
    for i in served_steps(trace, phase, semantics):
        st = trace.steps[i]
        for l in trace.layers:
            ex = st.experts(l)
            if ex:
                out.append((i, l, ex))
    return out


def next_uses(events: Sequence[Tuple[int, int, Tuple[int, ...]]]) -> List[Dict[int, float]]:
    """For event j, {expert: index of the next event using (layer, expert)}."""
    nxt: Dict[Key, float] = {}
    out: List[Dict[int, float]] = [dict() for _ in events]
    for j in range(len(events) - 1, -1, -1):
        _i, l, ex = events[j]
        d = out[j]
        for e in ex:
            d[e] = nxt.get((l, e), INF)
        for e in ex:
            nxt[(l, e)] = j
    return out


class _BeladyBase(Policy):
    family = "oracle"
    needs_future = True

    category = "oracle"

    def __init__(self, trace: Trace, phase: Optional[str] = PHASE_DECODE,
                 semantics: str = "v2") -> None:
        super().__init__()
        self.events = event_list(trace, phase, semantics)
        self.nu = next_uses(self.events)

    def start(self, ctx: Context) -> None:
        super().start(ctx)
        self.j = 0
        self.version: Dict[Key, int] = {}
        self.res: Dict[Key, int] = {}

    def _expect(self, step: int, layer: int, experts: List[int]) -> Dict[int, float]:
        if self.j >= len(self.events):
            raise RuntimeError("oracle: more events than the trace it was built on")
        i, l, ex = self.events[self.j]
        if i != step or l != layer or tuple(experts) != tuple(ex):
            raise RuntimeError("oracle: replayed events differ from the oracle's trace "
                               "(different phase filter or trace?)")
        nu = self.nu[self.j]
        self.j += 1
        return nu


class BeladyGlobal(_BeladyBase):
    name = "belady-global"

    def start(self, ctx: Context) -> None:
        super().start(ctx)
        self.heap: List[Tuple[float, int, Key]] = []

    def _access(self, step, layer, experts):
        nu = self._expect(step, layer, experts)
        b = self.ctx.bytes_of(layer)
        members = {(layer, e) for e in experts}
        misses = []
        overflow = False
        for e in experts:
            k = (layer, e)
            v = self.version.get(k, 0) + 1
            self.version[k] = v
            if k in self.res:
                heapq.heappush(self.heap, (-nu[e], v, k))
                continue
            misses.append(e)
            ok = True
            held = []
            while self.resident_bytes + b > self.ctx.budget:
                victim = None
                while self.heap:
                    negn, ver, kk = heapq.heappop(self.heap)
                    if kk not in self.res or self.version.get(kk) != ver:
                        continue
                    if kk in members:
                        held.append((negn, ver, kk))
                        continue
                    victim = kk
                    break
                if victim is None:
                    ok = False
                    break
                self.resident_bytes -= self.res.pop(victim)
                self.evictions += 1
            for h in held:
                heapq.heappush(self.heap, h)
            if ok:
                self.res[k] = b
                self.resident_bytes += b
                heapq.heappush(self.heap, (-nu[e], v, k))
            else:
                overflow = True
                self.overflow_misses += 1
        if overflow:
            self.overflow_events += 1
        return misses


class BeladyPerLayer(_BeladyBase):
    """Belady-style replacement inside fixed per-layer allotments."""
    family = "oracle"

    def __init__(self, trace: Trace, allocation: Dict[int, int],
                 phase: Optional[str] = PHASE_DECODE, label: str = "belady-per-layer",
                 semantics: str = "v2") -> None:
        super().__init__(trace, phase, semantics)
        self.alloc = dict(allocation)
        self.name = label

    def start(self, ctx: Context) -> None:
        super().start(ctx)
        if sum(self.alloc.get(l, 0) for l in ctx.layers) > ctx.budget:
            raise ValueError("allocation exceeds budget")
        self.heaps: Dict[int, List[Tuple[float, int, Key]]] = {l: [] for l in ctx.layers}
        self.used: Dict[int, int] = {l: 0 for l in ctx.layers}

    def allocation(self):
        return dict(self.alloc)

    def _access(self, step, layer, experts):
        nu = self._expect(step, layer, experts)
        b = self.ctx.bytes_of(layer)
        cap = self.alloc.get(layer, 0)
        heap = self.heaps[layer]
        members = {(layer, e) for e in experts}
        misses = []
        overflow = False
        for e in experts:
            k = (layer, e)
            v = self.version.get(k, 0) + 1
            self.version[k] = v
            if k in self.res:
                heapq.heappush(heap, (-nu[e], v, k))
                continue
            misses.append(e)
            ok = b <= cap
            held = []
            while ok and self.used[layer] + b > cap:
                victim = None
                while heap:
                    negn, ver, kk = heapq.heappop(heap)
                    if kk not in self.res or self.version.get(kk) != ver:
                        continue
                    if kk in members:
                        held.append((negn, ver, kk))
                        continue
                    victim = kk
                    break
                if victim is None:
                    ok = False
                    break
                freed = self.res.pop(victim)
                self.used[layer] -= freed
                self.resident_bytes -= freed
                self.evictions += 1
            for h in held:
                heapq.heappush(heap, h)
            if ok:
                self.res[k] = b
                self.used[layer] += b
                self.resident_bytes += b
                heapq.heappush(heap, (-nu[e], v, k))
            else:
                overflow = True
                self.overflow_misses += 1
        if overflow:
            self.overflow_events += 1
        return misses


def opt_misses(event_sets: Sequence[Tuple[int, ...]], slots: int,
               counted: Optional[Sequence[bool]] = None) -> int:
    """Belady-style misses of ONE layer with `slots` expert slots; with
    `counted`, all events are served but only flagged events are counted."""
    n = len(event_sets)
    nxt: Dict[int, float] = {}
    nu: List[Dict[int, float]] = [dict() for _ in range(n)]
    for j in range(n - 1, -1, -1):
        for e in event_sets[j]:
            nu[j][e] = nxt.get(e, INF)
        for e in event_sets[j]:
            nxt[e] = j
    res: Dict[int, int] = {}
    ver: Dict[int, int] = {}
    heap: List[Tuple[float, int, int]] = []
    misses = 0
    for j, S in enumerate(event_sets):
        members = set(S)
        cnt = counted is None or counted[j]
        for e in sorted(members):
            v = ver.get(e, 0) + 1
            ver[e] = v
            if e in res:
                heapq.heappush(heap, (-nu[j][e], v, e))
                continue
            misses += 1 if cnt else 0
            if slots <= 0:
                continue
            held = []
            ok = True
            while len(res) >= slots:
                victim = None
                while heap:
                    a, vv, kk = heapq.heappop(heap)
                    if kk not in res or ver.get(kk) != vv:
                        continue
                    if kk in members:
                        held.append((a, vv, kk))
                        continue
                    victim = kk
                    break
                if victim is None:
                    ok = False
                    break
                del res[victim]
            for h in held:
                heapq.heappush(heap, h)
            if ok:
                res[e] = 1
                heapq.heappush(heap, (-nu[j][e], v, e))
    return misses


def layer_event_sets(trace: Trace, layer: int, phase: Optional[str] = PHASE_DECODE
                     ) -> List[Tuple[int, ...]]:
    return [st.experts(layer) for st in trace.steps
            if (phase is None or st.phase == phase) and st.experts(layer)]


def opt_layer_curve(trace: Trace, layer: int, slot_grid: Sequence[int],
                    phase: Optional[str] = PHASE_DECODE) -> Dict[int, int]:
    sets = layer_event_sets(trace, layer, phase)
    return {c: opt_misses(sets, c) for c in slot_grid}
