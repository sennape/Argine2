"""AdaptiveResidencyController — replay prototype (sections 19-21).

STATUS: replay-only prototype. Section 19 of the Argine2 requirement places
the runtime controller AFTER Phase 1 characterization; this module exists so
that the hypothesis can be tested counterfactually (and falsified) on recorded
traces before any runtime code is written. It never sees the future: every
decision uses only events that have already been replayed.

Design (deterministic and interpretable, no learned model)
----------------------------------------------------------
* Layout: one LRU partition per layer (PerLayerLRU). The controller decides
  the bytes of each partition; the partitions enforce them, so the global
  hard budget holds at every event (replay checks it).
* Monitor: for every layer an ONLINE event-atomic LRU stack-distance
  histogram h_L[d] (exact, Fenwick tree), decayed by `decay` at every epoch so
  that it follows workload drift. h_L[d] estimates the hits gained by the
  (d+1)-th slot of layer L: the classic utility monitor of utility-based cache
  partitioning (Qureshi & Patt, MICRO 2006), here with layers in place of
  cores.
* Value of memory: marginal stall avoided per byte,
      v_add(L)    = h_L[c_L]     * cost_per_miss_L / size_L
      v_remove(L) = h_L[c_L - 1] * cost_per_miss_L / size_L
  with cost_per_miss_L from the cost model at the observed misses per node.
* Decision: every `epoch` steps, move one slot-sized block at a time from the
  layer with the lowest v_remove to the layer with the highest v_add while
  v_add > (1 + hysteresis) * v_remove, up to `max_move` of the budget per
  epoch. Donors shrink first (evicting their LRU unpinned experts), then
  receivers grow.
* Optional pinning (`pinned_fraction` of each partition): experts are pinned by
  decayed frequency x stability (share of recent epochs in which the expert
  was in the layer's EWS90 set), not by frequency alone.

Ablation switches (section 33): adaptive=False (frozen initial allocation),
estimator="freq" (frequency ranks instead of reuse distances),
cost_weighting=False (count misses, ignore bytes and queue depth),
hysteresis=0, pinned_fraction=0.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from typing import Deque, Dict, List, Optional, Sequence, Set

from .costmodel import CostModel
from .metrics import _Fenwick
from .policies import Context, Key, PerLayerLRU, uniform_bytes


class OnlineStackDistance:
    """Event-atomic LRU stack distances for one scope, computed online."""

    def __init__(self, capacity: int = 1 << 16) -> None:
        self.cap = capacity
        self.bit = _Fenwick(capacity)
        self.last: Dict[int, int] = {}
        self.pos = 0
        self.active = 0

    def _compact(self) -> None:
        order = sorted(self.last.items(), key=lambda kv: kv[1])
        self.cap = max(self.cap, 4 * len(order) + 1024)
        self.bit = _Fenwick(self.cap)
        self.last = {}
        self.pos = 0
        for e, _p in order:
            self.pos += 1
            self.bit.add(self.pos, 1)
            self.last[e] = self.pos
        self.active = len(order)

    def event(self, members: Sequence[int]) -> List[Optional[int]]:
        if self.pos + len(members) >= self.cap:
            self._compact()
        out: List[Optional[int]] = []
        for e in members:
            p = self.last.get(e)
            out.append(None if p is None else self.active - self.bit.sum(p))
        for e in members:
            p = self.last.get(e)
            if p is not None:
                self.bit.add(p, -1)
                self.active -= 1
            self.pos += 1
            self.bit.add(self.pos, 1)
            self.active += 1
            self.last[e] = self.pos
        return out


class AdaptiveResidencyController(PerLayerLRU):
    family = "allocation"

    def __init__(self, *, epoch: int = 64, decay: float = 0.5, hysteresis: float = 0.10,
                 max_move: float = 0.05, estimator: str = "stack", cost_weighting: bool = True,
                 pinned_fraction: float = 0.0, adaptive: bool = True,
                 initial: Optional[Dict[int, int]] = None, cost: Optional[CostModel] = None,
                 stability_epochs: int = 4, label: str = "") -> None:
        super().__init__(allocation=initial)
        if estimator not in ("stack", "freq"):
            raise ValueError("estimator must be 'stack' or 'freq'")
        self.epoch = int(epoch)
        self.decay = float(decay)
        self.hyst = float(hysteresis)
        self.max_move = float(max_move)
        self.estimator = estimator
        self.cost_weighting = cost_weighting
        self.pinned_fraction = float(pinned_fraction)
        self.adaptive = adaptive
        self.cost = cost or CostModel()
        self.stability_epochs = int(stability_epochs)
        self.name = label or self._auto_name()
        self.monitor_s = 0.0

    def _auto_name(self) -> str:
        parts = ["adaptive" if self.adaptive else "frozen", f"e{self.epoch}"]
        if self.estimator != "stack":
            parts.append("freq")
        if not self.cost_weighting:
            parts.append("nocost")
        if self.hyst == 0:
            parts.append("nohyst")
        if self.pinned_fraction:
            parts.append(f"pin{int(round(self.pinned_fraction * 100))}")
        return "-".join(parts)

    def describe(self):
        return {"name": self.name, "family": self.family, "epoch": self.epoch,
                "decay": self.decay, "hysteresis": self.hyst, "max_move": self.max_move,
                "estimator": self.estimator, "cost_weighting": self.cost_weighting,
                "pinned_fraction": self.pinned_fraction, "adaptive": self.adaptive,
                "online": True, "status": "replay prototype"}

    # ------------------------------------------------------------ state ---
    def start(self, ctx: Context) -> None:
        super().start(ctx)
        self.min_slots = max(1, ctx.top_k)
        self.mon = {l: OnlineStackDistance() for l in ctx.layers}
        self.hist: Dict[int, Dict[int, float]] = {l: {} for l in ctx.layers}
        self.freq: Dict[int, Counter] = {l: Counter() for l in ctx.layers}
        self.top_hist: Dict[int, Deque[Set[int]]] = {
            l: deque(maxlen=self.stability_epochs) for l in ctx.layers}
        self.pinned: Dict[int, Set[int]] = {l: set() for l in ctx.layers}
        self.ep_misses = 0
        self.ep_nodes = 0
        self.mpn = 1.0  # running misses per node
        self.moves = 0
        self.bytes_moved = 0
        self.monitor_s = 0.0

    # ---------------------------------------------------------- serving ---
    def _access(self, step, layer, experts):
        t0 = time.perf_counter()
        d = self.mon[layer].event(experts)
        h = self.hist[layer]
        for dist in d:
            if dist is not None:
                h[dist] = h.get(dist, 0.0) + 1.0
        self.freq[layer].update(experts)
        self.monitor_s += time.perf_counter() - t0

        sc = self.scopes[layer]
        b = self.ctx.bytes_of(layer)
        protect = {(layer, e) for e in experts} | {(layer, e) for e in self.pinned[layer]}
        misses = []
        overflow = False
        before = sc.evictions
        for e in experts:
            k = (layer, e)
            if k in sc:
                sc.touch(k)
                continue
            misses.append(e)
            ok, freed = sc.admit(k, b, protect)
            self.resident_bytes -= freed
            if ok:
                self.resident_bytes += b
            else:
                overflow = True
                self.overflow_misses += 1
        self.evictions += sc.evictions - before
        if overflow:
            self.overflow_events += 1
        self.ep_misses += len(misses)
        self.ep_nodes += 1
        return misses

    # --------------------------------------------------------- decisions ---
    def _cpm(self, layer: int) -> float:
        if not self.cost_weighting:
            return 1.0
        m = max(1, int(round(self.mpn)))
        b = self.ctx.bytes_of(layer)
        return self.cost.node_stall_s(m, m * b) / m

    def _utility(self, layer: int, slot_index: int) -> float:
        """Estimated hits per epoch of the slot with this 0-based index."""
        if slot_index < 0:
            return 0.0
        if self.estimator == "stack":
            return self.hist[layer].get(slot_index, 0.0)
        ranked = sorted(self.freq[layer].values(), reverse=True)
        return float(ranked[slot_index]) if slot_index < len(ranked) else 0.0

    def _value(self, layer: int, slot_index: int) -> float:
        u = self._utility(layer, slot_index)
        if not self.cost_weighting:
            return u
        return u * self._cpm(layer) / self.ctx.bytes_of(layer)

    def end_step(self, step: int) -> None:
        if self.epoch <= 0 or (step + 1) % self.epoch:
            return
        t0 = time.perf_counter()
        if self.ep_nodes:
            self.mpn = 0.5 * self.mpn + 0.5 * (self.ep_misses / self.ep_nodes)
        self.ep_misses = self.ep_nodes = 0
        if self.adaptive:
            self._reallocate()
        if self.pinned_fraction > 0:
            self._repin()
        for l in self.ctx.layers:
            h = self.hist[l]
            for k in list(h):
                h[k] *= self.decay
                if h[k] < 1e-3:
                    del h[k]
            f = self.freq[l]
            for k in list(f):
                f[k] *= self.decay
        self.monitor_s += time.perf_counter() - t0

    def _slots(self) -> Dict[int, int]:
        return {l: sc.capacity // self.ctx.bytes_of(l) for l, sc in self.scopes.items()}

    def _reallocate(self) -> None:
        ctx = self.ctx
        slots = self._slots()
        free = ctx.budget - sum(slots[l] * ctx.bytes_of(l) for l in ctx.layers)
        limit = self.max_move * ctx.budget
        moved = 0
        n = ctx.n_experts
        while moved < limit:
            recv = max((l for l in ctx.layers if slots[l] < n),
                       key=lambda l: (self._value(l, slots[l]), -l), default=None)
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
            k = -(-need // ctx.bytes_of(don))  # slots the donor must give
            if slots[don] - k < self.min_slots:
                break
            v_rem = sum(self._value(don, slots[don] - i) for i in range(1, k + 1)) / k
            if v_add <= (1.0 + self.hyst) * v_rem + 1e-18:
                break
            slots[don] -= k
            free += k * ctx.bytes_of(don) - need
            slots[recv] += 1
            moved += need
            self.moves += 1
        self.bytes_moved += moved
        self.set_allocation({l: slots[l] * ctx.bytes_of(l) for l in ctx.layers})

    def _repin(self) -> None:
        for l in self.ctx.layers:
            f = self.freq[l]
            tot = sum(f.values())
            top: Set[int] = set()
            acc = 0.0
            for e, x in sorted(f.items(), key=lambda kv: (-kv[1], kv[0])):
                if acc >= 0.90 * tot:
                    break
                top.add(e)
                acc += x
            self.top_hist[l].append(top)
            hist = self.top_hist[l]

            def stab(e: int) -> float:
                return sum(1 for s in hist if e in s) / float(len(hist) or 1)

            cap_slots = int(self.pinned_fraction * self._slots()[l])
            cap_slots = max(0, min(cap_slots, self._slots()[l] - self.min_slots))
            ranked = sorted(f, key=lambda e: (-(f[e] * stab(e)), e))
            self.pinned[l] = set(ranked[:cap_slots])
