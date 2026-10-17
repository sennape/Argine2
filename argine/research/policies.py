"""Cache and memory-allocation policies for counterfactual replay.

All policies follow the same EVENT-ATOMIC contract, which is also the contract
of Argine's runtime: one event is one MoE node, i.e. the set of experts one
forward pass needs in one layer. Every member of the event must be resident
at the same time (the runtime pins them until the node is computed), so a
policy may never evict a member of the current event to make room for another.
Admission and eviction happen at event granularity; within an event members
are processed in ascending expert id, which makes every policy deterministic.
(Flattening events into sequential accesses changes policy rankings; see
"Reproducible Evaluation of MoE Expert Caching", arXiv 2608.07911.)

Budgets are in bytes. The unit of residency is one expert of one layer (its
gate, up and down slices together), keyed by (layer, expert). The runtime
caches the three slices independently; modelling them as one unit is an
approximation, exact when the three are always loaded and evicted together
(the gate node loads all three, so they are loaded together).

If an event cannot fit in the capacity its policy may use (e.g. a prompt step
needing more experts than a layer's partition), it cannot be atomic. Members
are then streamed through the scope in processing order: when only members of
the event remain resident, the least recently used member is evicted to admit
the next one ("overflow"). After the event the scope holds the last members
processed — exactly what an LRU stack of that capacity would hold, so exact
stack-distance curves stay equal to replay (tested). Replay reports how often
this happened.

Categories (never mixed in a table without the label):
* counterfactual — implementable policies evaluated on a recorded trace;
* oracle         — uses future knowledge (oracle.py): a bound, not a result;
* runtime-reproduction — only storereplay.StoreReplay.
Policies fitted on the trace they are evaluated on are additionally marked
in-sample (IN-SAMPLE / UPPER BOUND).
"""

from __future__ import annotations

import heapq
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

Key = Tuple[int, int]


@dataclass
class Context:
    layers: List[int]
    n_experts: int
    size: Dict[int, int]          # bytes of one resident expert, per layer
    budget: int                   # bytes
    top_k: int = 0

    def bytes_of(self, layer: int) -> int:
        return int(self.size.get(layer, 1))


class Policy:
    """Base class. Subclasses implement `_access` and maintain `resident_bytes`."""

    name = "base"
    family = "cache"          # "cache" | "allocation" | "oracle"
    category = "counterfactual"
    needs_future = False      # True only for offline oracles

    def __init__(self) -> None:
        self.ctx: Optional[Context] = None
        self.resident_bytes = 0
        self.evictions = 0
        self.overflow_events = 0
        self.overflow_misses = 0

    # --- lifecycle -----------------------------------------------------
    def start(self, ctx: Context) -> None:
        self.ctx = ctx
        self.resident_bytes = 0
        self.evictions = 0
        self.overflow_events = 0
        self.overflow_misses = 0

    def access(self, step: int, layer: int, experts: Sequence[int]) -> List[int]:
        """Serve one event. Returns the experts that missed (sorted)."""
        return self._access(step, layer, sorted(set(experts)))

    def end_step(self, step: int) -> None:
        pass

    def allocation(self) -> Optional[Dict[int, int]]:
        """Bytes allotted per layer, for partitioned policies (else None)."""
        return None

    def describe(self) -> Dict[str, object]:
        return {"name": self.name, "family": self.family}

    def _access(self, step: int, layer: int, experts: List[int]) -> List[int]:
        raise NotImplementedError


# ------------------------------------------------------------- LRU scope ---

class LRUScope:
    """An LRU set of keys with a byte capacity, honouring event atomicity."""

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.od: "OrderedDict[Key, int]" = OrderedDict()  # key -> bytes, oldest first
        self.used = 0
        self.evictions = 0

    def __contains__(self, k: Key) -> bool:
        return k in self.od

    def touch(self, k: Key) -> None:
        self.od.move_to_end(k)

    def evict_one(self, protect: Set[Key]) -> Optional[int]:
        for k in self.od:
            if k not in protect:
                b = self.od.pop(k)
                self.used -= b
                self.evictions += 1
                return b
        return None

    def admit(self, k: Key, b: int, protect: Set[Key]) -> Tuple[bool, int]:
        """Make room and insert. Returns (admitted, bytes_freed)."""
        if b > self.capacity:
            return False, 0
        freed = 0
        while self.used + b > self.capacity:
            got = self.evict_one(protect)
            if got is None:
                return False, freed
            freed += got
        self.od[k] = b
        self.used += b
        return True, freed

    def shrink_to(self, capacity: int, protect: Set[Key] = frozenset()) -> int:  # type: ignore[assignment]
        self.capacity = int(capacity)
        freed = 0
        while self.used > self.capacity:
            got = self.evict_one(protect)
            if got is None:
                break
            freed += got
        return freed


def _serve_lru(scope: LRUScope, layer: int, experts: List[int], b: int,
               pol: Policy) -> List[int]:
    members = {(layer, e) for e in experts}
    misses: List[int] = []
    overflow = False
    for e in experts:
        k = (layer, e)
        if k in scope:
            scope.touch(k)
            continue
        misses.append(e)
        ok, freed = scope.admit(k, b, members)
        pol.resident_bytes -= freed
        if not ok and b <= scope.capacity:
            # overflow: stream members through, evicting the LRU member
            got = 0
            while scope.used + b > scope.capacity:
                g = scope.evict_one(frozenset())
                if g is None:
                    break
                got += g
            pol.resident_bytes -= got
            ok, freed = scope.admit(k, b, members)
            pol.resident_bytes -= freed
            overflow = True
            pol.overflow_misses += 1
        if ok:
            pol.resident_bytes += b
        else:
            overflow = True
            pol.overflow_misses += 1
    if overflow:
        pol.overflow_events += 1
    return misses


# ------------------------------------------------------------- policies ---

class GlobalLRU(Policy):
    """One LRU over all layers, one byte budget."""
    name = "global-lru"

    def start(self, ctx: Context) -> None:
        super().start(ctx)
        self.scope = LRUScope(ctx.budget)

    def _access(self, step, layer, experts):
        out = _serve_lru(self.scope, layer, experts, self.ctx.bytes_of(layer), self)
        self.evictions = self.scope.evictions
        return out


def _mix64(x: int) -> int:
    # fmix64 from MurmurHash3: the same family the runtime uses for sharding
    x &= 0xFFFFFFFFFFFFFFFF
    x ^= x >> 33
    x = (x * 0xff51afd7ed558ccd) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 33
    x = (x * 0xc4ceb9fe1a85ec53) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 33
    return x


class ShardedLRU(Policy):
    """Model of Argine's current runtime cache: keys hashed into n shards, each
    an independent LRU with budget/n bytes (argine-store.cpp, `Shard`).

    The runtime hashes (region address, offset) of each slice; addresses are
    not reproducible offline, so keys are spread with fmix64 over
    (layer, expert, seed). The partition is therefore statistically, not
    exactly, the runtime's. With n = 1 this is GlobalLRU.
    """
    name = "sharded-lru"

    def __init__(self, shards: int = 16, seed: int = 0) -> None:
        super().__init__()
        self.n = max(1, int(shards))
        self.seed = seed
        self.name = f"sharded-lru-{self.n}"

    def start(self, ctx: Context) -> None:
        super().start(ctx)
        base = ctx.budget // self.n
        self.scopes = [LRUScope(base) for _ in range(self.n)]
        self.scopes[-1].capacity += ctx.budget - base * self.n

    def _shard(self, layer: int, e: int) -> int:
        return _mix64((layer << 32) ^ (e << 8) ^ self.seed) % self.n

    def _access(self, step, layer, experts):
        b = self.ctx.bytes_of(layer)
        members = {(layer, e) for e in experts}
        misses = []
        overflow = False
        for e in experts:
            k = (layer, e)
            sc = self.scopes[self._shard(layer, e)]
            if k in sc:
                sc.touch(k)
                continue
            misses.append(e)
            ok, freed = sc.admit(k, b, members)
            self.resident_bytes -= freed
            if ok:
                self.resident_bytes += b
            else:
                overflow = True
                self.overflow_misses += 1
        if overflow:
            self.overflow_events += 1
        self.evictions = sum(s.evictions for s in self.scopes)
        return misses

    def describe(self):
        return {"name": self.name, "family": self.family, "shards": self.n, "seed": self.seed}


class PerLayerLRU(Policy):
    """One LRU per layer with a fixed byte allotment per layer.

    `allocation=None` means uniform: the budget is split evenly in bytes.
    This is the family in which "layer-aware" allocations are compared.
    """
    name = "per-layer-lru"
    family = "allocation"

    def __init__(self, allocation: Optional[Dict[int, int]] = None, label: str = "") -> None:
        super().__init__()
        self._alloc_in = dict(allocation) if allocation else None
        if label:
            self.name = label

    def start(self, ctx: Context) -> None:
        super().start(ctx)
        alloc = self._alloc_in or uniform_bytes(ctx)
        total = sum(alloc.get(l, 0) for l in ctx.layers)
        if total > ctx.budget:
            raise ValueError(f"{self.name}: allocation {total} exceeds budget {ctx.budget}")
        self.scopes = {l: LRUScope(alloc.get(l, 0)) for l in ctx.layers}

    def _access(self, step, layer, experts):
        sc = self.scopes[layer]
        before = sc.evictions
        out = _serve_lru(sc, layer, experts, self.ctx.bytes_of(layer), self)
        self.evictions += sc.evictions - before
        return out

    def allocation(self):
        return {l: s.capacity for l, s in self.scopes.items()}

    def set_allocation(self, alloc: Dict[int, int]) -> None:
        """Change allotments: shrink first (evicting), then grow."""
        for l, cap in alloc.items():
            sc = self.scopes[l]
            if cap < sc.capacity:
                before = sc.evictions
                self.resident_bytes -= sc.shrink_to(cap)
                self.evictions += sc.evictions - before
        for l, cap in alloc.items():
            sc = self.scopes[l]
            if cap >= sc.capacity:
                sc.capacity = int(cap)


def uniform_bytes(ctx: Context) -> Dict[int, int]:
    """Even split in bytes, rounded down to whole experts per layer."""
    per = ctx.budget // max(1, len(ctx.layers))
    return {l: (per // ctx.bytes_of(l)) * ctx.bytes_of(l) for l in ctx.layers}


class LFU(Policy):
    """Least-frequently-used over all layers (counts persist across evictions
    unless `in_cache_only`), ties broken by least recent use. Optional decay:
    counts are halved every `halflife` steps (0 = no decay)."""
    name = "lfu"

    def __init__(self, halflife: int = 0, in_cache_only: bool = False) -> None:
        super().__init__()
        self.halflife = int(halflife)
        self.in_cache_only = in_cache_only
        if halflife:
            self.name = f"lfu-decay{halflife}"

    def start(self, ctx: Context) -> None:
        super().start(ctx)
        self.count: Dict[Key, float] = {}
        self.last: Dict[Key, int] = {}
        self.res: Dict[Key, int] = {}
        self.heap: List[Tuple[float, int, Key]] = []
        self.clock = 0
        self.scale = 1.0

    def _push(self, k: Key) -> None:
        heapq.heappush(self.heap, (self.count[k], self.last[k], k))

    def _victim(self, protect: Set[Key]) -> Optional[Key]:
        held = []
        found = None
        while self.heap:
            c, t, k = heapq.heappop(self.heap)
            if k not in self.res or self.count.get(k) != c or self.last.get(k) != t:
                continue  # stale entry
            if k in protect:
                held.append((c, t, k))
                continue
            found = k
            break
        for h in held:
            heapq.heappush(self.heap, h)
        return found

    def _access(self, step, layer, experts):
        b = self.ctx.bytes_of(layer)
        members = {(layer, e) for e in experts}
        misses = []
        overflow = False
        for e in experts:
            k = (layer, e)
            self.clock += 1
            self.count[k] = self.count.get(k, 0.0) + self.scale
            self.last[k] = self.clock
            if k in self.res:
                self._push(k)
                continue
            misses.append(e)
            if b > self.ctx.budget:
                overflow = True
                continue
            ok = True
            while self.resident_bytes + b > self.ctx.budget:
                v = self._victim(members)
                if v is None:
                    ok = False
                    break
                self.resident_bytes -= self.res.pop(v)
                self.evictions += 1
                if self.in_cache_only:
                    self.count.pop(v, None)
            if ok:
                self.res[k] = b
                self.resident_bytes += b
                self._push(k)
            else:
                overflow = True
                self.overflow_misses += 1
        if overflow:
            self.overflow_events += 1
        return misses

    def end_step(self, step: int) -> None:
        if self.halflife and step > 0 and step % self.halflife == 0:
            # halve all counts lazily by doubling the increment for the future
            self.scale *= 2.0
            if self.scale > 1e12:  # renormalise
                for k in self.count:
                    self.count[k] /= self.scale
                self.scale = 1.0
                self.heap = [(self.count[k], self.last[k], k) for k in self.res]
                heapq.heapify(self.heap)


class TwoQ(Policy):
    """2Q (Johnson & Shasha, VLDB 1994), byte-based, global scope.
    A1in: FIFO of first-time residents (kin of capacity); A1out: ghost keys of
    recent A1in evictions (kout of capacity, keys only, no bytes); Am: LRU of
    keys re-referenced after leaving A1in."""
    name = "2q"

    def __init__(self, kin: float = 0.25, kout: float = 0.5) -> None:
        super().__init__()
        self.kin, self.kout = kin, kout

    def start(self, ctx: Context) -> None:
        super().start(ctx)
        self.a1in: "OrderedDict[Key, int]" = OrderedDict()
        self.am: "OrderedDict[Key, int]" = OrderedDict()
        self.a1out: "OrderedDict[Key, int]" = OrderedDict()
        self.in_bytes = 0
        self.out_bytes = 0

    def _reclaim(self, protect: Set[Key]) -> bool:
        cap = self.ctx.budget
        if self.in_bytes > self.kin * cap:
            for k in self.a1in:
                if k not in protect:
                    b = self.a1in.pop(k)
                    self.in_bytes -= b
                    self.resident_bytes -= b
                    self.evictions += 1
                    self.a1out[k] = b
                    self.out_bytes += b
                    while self.out_bytes > self.kout * cap and self.a1out:
                        _, ob = self.a1out.popitem(last=False)
                        self.out_bytes -= ob
                    return True
        for od, is_in in ((self.am, False), (self.a1in, True)):
            for k in od:
                if k not in protect:
                    b = od.pop(k)
                    self.resident_bytes -= b
                    if is_in:
                        self.in_bytes -= b
                    self.evictions += 1
                    return True
        return False

    def _access(self, step, layer, experts):
        b = self.ctx.bytes_of(layer)
        members = {(layer, e) for e in experts}
        misses = []
        overflow = False
        for e in experts:
            k = (layer, e)
            if k in self.am:
                self.am.move_to_end(k)
                continue
            if k in self.a1in:
                continue
            misses.append(e)
            ok = True
            while self.resident_bytes + b > self.ctx.budget:
                if not self._reclaim(members):
                    ok = False
                    break
            if not ok:
                overflow = True
                self.overflow_misses += 1
                continue
            if k in self.a1out:
                self.out_bytes -= self.a1out.pop(k)
                self.am[k] = b
            else:
                self.a1in[k] = b
                self.in_bytes += b
            self.resident_bytes += b
        if overflow:
            self.overflow_events += 1
        return misses


class LeastStale(Policy):
    """Stale-first eviction over all layers.

    "Least-Stale" is described as evicting "blocks untouched in the current
    forward cycle first" (arXiv 2608.07911, citing SpecMD, arXiv 2602.03921).
    The order AMONG stale blocks is not specified in the description available
    to us, and it matters:

    * variant "lru": oldest stale first. In a global scope every block touched
      in the current cycle is more recent than every stale one, so this variant
      is provably identical to GlobalLRU; it is kept to make that visible.
    * variant "mru": most recently used stale block first. Under the cyclic
      layer order of a forward pass, the most recent stale blocks belong to
      the layers that will be needed LAST in the next cycle, which is what a
      Belady-like choice would prefer.

    [OPEN] Which variant matches the published policy must be checked against
    the SpecMD paper; results are reported under the variant's name.
    """

    def __init__(self, variant: str = "mru") -> None:
        super().__init__()
        if variant not in ("lru", "mru"):
            raise ValueError("variant must be lru or mru")
        self.variant = variant
        self.name = f"least-stale-{variant}"

    def start(self, ctx: Context) -> None:
        super().start(ctx)
        self.od: "OrderedDict[Key, int]" = OrderedDict()  # oldest first
        self.step_of: Dict[Key, int] = {}
        self.cur = -1

    def _victim(self, protect: Set[Key]) -> Optional[Key]:
        if self.variant == "lru":
            for k in self.od:
                if k not in protect:
                    return k
            return None
        # mru among stale: walk from the newest end past current-cycle blocks
        for k in reversed(self.od):
            if k in protect:
                continue
            if self.step_of.get(k, -1) < self.cur:
                return k
        for k in self.od:  # nothing stale: plain LRU among non-members
            if k not in protect:
                return k
        return None

    def _access(self, step, layer, experts):
        self.cur = step
        b = self.ctx.bytes_of(layer)
        members = {(layer, e) for e in experts}
        misses = []
        overflow = False
        for e in experts:
            k = (layer, e)
            if k in self.od:
                self.od.move_to_end(k)
                self.step_of[k] = step
                continue
            misses.append(e)
            ok = True
            while self.resident_bytes + b > self.ctx.budget:
                v = self._victim(members)
                if v is None:
                    ok = False
                    break
                self.resident_bytes -= self.od.pop(v)
                self.evictions += 1
            if ok:
                self.od[k] = b
                self.step_of[k] = step
                self.resident_bytes += b
            else:
                overflow = True
                self.overflow_misses += 1
        if overflow:
            self.overflow_events += 1
        return misses


class StaticPinned(Policy):
    """Pinned hot set + dynamic LRU for the rest of the budget.

    The pinned set is chosen OFFLINE from `train_counts` ((layer, expert) ->
    selections, normally from a development trace): keys are taken in
    decreasing score order until `pinned_fraction` of the budget is used, where
    score = count / bytes (frequency per byte) by default. The remaining budget
    (at least `reserve` bytes, enough for the largest event) is an LRU.

    pinned_fraction = 1.0 gives "static frequency" (everything static except
    the streaming reserve); 0.25 / 0.5 give the hot-pinning experiment of
    section 21. When `per_layer=True` the pinned set and the LRU are both
    partitioned evenly by layer ("static hot set per layer").
    """
    family = "allocation"

    def __init__(self, train_counts: Dict[Key, int], pinned_fraction: float = 1.0,
                 per_layer: bool = False, label: str = "", in_sample: bool = False) -> None:
        super().__init__()
        self.train = dict(train_counts)
        self.frac = float(pinned_fraction)
        self.per_layer = per_layer
        self.in_sample = in_sample
        self.name = label or (("static-hot-per-layer" if per_layer else "static-frequency")
                              if pinned_fraction >= 0.999 else
                              f"pinned{int(round(pinned_fraction * 100))}+lru")

    def start(self, ctx: Context) -> None:
        super().start(ctx)
        k = max(ctx.top_k, 1)
        biggest = max(ctx.bytes_of(l) for l in ctx.layers)
        self.pinned: Set[Key] = set()
        if self.per_layer:
            per = ctx.budget // len(ctx.layers)
            self.scopes: Dict[int, LRUScope] = {}
            for l in ctx.layers:
                b = ctx.bytes_of(l)
                reserve = k * b
                pin_cap = min(int(per * self.frac), per - reserve)
                cand = sorted(((c, e) for (ll, e), c in self.train.items() if ll == l),
                              key=lambda x: (-x[0], x[1]))
                used = 0
                for _c, e in cand:
                    if used + b > pin_cap:
                        break
                    self.pinned.add((l, e))
                    used += b
                self.resident_bytes += used
                self.scopes[l] = LRUScope(((per - used) // b) * b)
        else:
            reserve = k * biggest
            pin_cap = min(int(ctx.budget * self.frac), ctx.budget - reserve)
            cand = sorted(self.train.items(),
                          key=lambda kv: (-kv[1] / ctx.bytes_of(kv[0][0]), kv[0]))
            used = 0
            for key, _c in cand:
                b = ctx.bytes_of(key[0])
                if used + b > pin_cap:
                    continue
                self.pinned.add(key)
                used += b
            self.resident_bytes += used
            self.lru = LRUScope(ctx.budget - used)
        self.pinned_bytes = self.resident_bytes

    def _access(self, step, layer, experts):
        b = self.ctx.bytes_of(layer)
        rest = [e for e in experts if (layer, e) not in self.pinned]
        if not rest:
            return []
        scope = self.scopes[layer] if self.per_layer else self.lru
        # members of the event that are pinned need no protection in the LRU
        out = _serve_lru(scope, layer, rest, b, self)
        if self.per_layer:
            self.evictions = sum(s.evictions for s in self.scopes.values())
        else:
            self.evictions = self.lru.evictions
        return out

    def describe(self):
        return {"name": self.name, "family": self.family, "pinned_fraction": self.frac,
                "per_layer": self.per_layer, "pinned_keys": len(getattr(self, "pinned", ())),
                "train": "in-sample" if self.in_sample else "separate trace"}


def counts_from_trace(trace, phase=None) -> Dict[Key, int]:
    """(layer, expert) -> number of events containing it (event units, which
    is what residency pays for). `phase` restricts to prefill or decode steps."""
    out: Dict[Key, int] = {}
    for i, l, ex in trace.events():
        if phase is not None and trace.steps[i].phase != phase:
            continue
        for e in ex:
            out[(l, e)] = out.get((l, e), 0) + 1
    return out
