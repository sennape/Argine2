"""StoreReplay — an executable model of Argine's Expert Store cache (runtime reproduction).

This module reproduces, decision by decision, the replacement algorithm of
`patches/argine-store.cpp` (`prenota`, `completa`, `psx_cache_unpin`) and the
request pattern of `patches/argine-experts.cpp` (`compute_forward`). It is the
ONLY replay in Argine2 that is presented as a reproduction of the runtime; all
other policies in `policies.py` are counterfactuals, and the Belady-style
policies in `oracle.py` are bounds. The full state machine, with line-level
references, is in docs/research/REPLAY-FIDELITY-VALIDATION.md.

Runtime semantics reproduced here
---------------------------------
* Unit of residency: one SLICE = (layer, expert, kind), kind in gate/up/down.
  An expert is not a unit: its three slices are cached, evicted and hashed
  independently and may be partly resident.
* Bytes: need = slice size rounded up to the store alignment (4096), for every
  slice, whatever the I/O mode (psx_slice_read_bytes).
* Shards: n = budget // 32 MiB clamped to [1, 16] (or ARGINE_EXPERT_SHARDS);
  shard budget = budget // n, the last shard takes the remainder. A slice's
  shard is mix64(region_address * phi + offset) % n: it depends on virtual
  addresses, so it is either READ from a format-v2 trace (exact) or DRAWN from
  the same hash family with random region constants (statistical, labelled).
* Per shard: one LRU list. Hit = move to MRU. Miss = evict from the LRU end,
  one slot at a time, skipping pinned or still-loading slots, until
  resident + need <= shard budget; then insert at MRU (before the read
  completes). Failure to find a victim => the runtime waits and eventually
  aborts the node; here it is counted and the slice is not inserted.
* Pins are PER THREAD (`thread_local` list in argine-experts.cpp): a slice
  is pinned from the moment a thread requests it until THAT thread enters its
  next MoE node. Each thread of the pool computing a node unpins at node entry,
  before the barrier that precedes all requests of the node. If every graph is
  computed by the same threads (Linux EXP-021 runs: --threads 4
  --threads-batch 4), the pinned set while node N is served is exactly
  {slices already requested in node N}. If decode and prompt graphs use
  different thread pools (Windows reference runs: --threads 4
  --threads-batch 12), a thread that is not in the current pool keeps the
  pins of the last node it ran — e.g. decode threads keep the last decode
  graph's layer-47 slices pinned through the whole next prompt. `begin_node`
  therefore takes the set of threads entering the node (EXP-021 Windows v2
  follow-up). Requested-later members of the same node are NOT protected
  (the runtime is not event-atomic).
* Request pattern per layer: the gate node requests all three kinds of its
  experts (k = 0 .. 3n-1, kind = k // n in order gate, up, down; expert =
  sorted(experts)[k % n]); the up and the down node request the same slices
  again (normally all hits, which refresh recency).
* Threads take slices k in fetch_add order, so the order of requests within a
  node is k order up to races. A format-v2 trace gives the true per-shard order
  (seq numbers); a synthetic stream uses k order.

Metrics (identical definitions for runtime and replay; see fidelity.py)
* slice_all        hits / slice requests, all nodes
* slice_gate       hits / slice requests of gate nodes        (expert-level hit
                                                               rate of the runtime)
* expert_event     share of (step, layer, expert) whose 3 slices all hit in the
                   gate node
Each restricted to workload steps (warm-up and flush sentinel excluded) and to
a phase (decode / prefill / all).
"""

from __future__ import annotations

import random
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

MIX_MASK = 0xFFFFFFFFFFFFFFFF
PHI = 0x9E3779B97F4A7C15
MIN_SHARD_BYTES = 32 << 20
MAX_SHARDS = 16
ALIGN = 4096

KIND_GATE, KIND_UP, KIND_DOWN = 0, 1, 2
NODE_KINDS = (KIND_GATE, KIND_UP, KIND_DOWN)

SliceKey = Tuple[int, int, int]  # (layer, expert, kind)


def mix64(x: int) -> int:
    """MurmurHash3 fmix64, as `mix64` in argine-store.cpp."""
    x &= MIX_MASK
    x ^= x >> 33
    x = (x * 0xFF51AFD7ED558CCD) & MIX_MASK
    x ^= x >> 33
    x = (x * 0xC4CEB9FE1A85EC53) & MIX_MASK
    x ^= x >> 33
    return x


def runtime_shard_count(budget: int, forced: int = 0) -> int:
    """psx_cache_new_n: n = budget / 32 MiB clamped to [1, 16]; forced wins."""
    if forced:
        return max(1, int(forced))
    if not budget:
        return MAX_SHARDS
    return max(1, min(MAX_SHARDS, budget // MIN_SHARD_BYTES))


def shard_budgets(budget: int, n: int) -> List[int]:
    base = budget // n if budget else 0
    out = [base] * n
    if budget:
        out[-1] += budget - base * n
    return out


def round_up(n: int, a: int = ALIGN) -> int:
    return -(-int(n) // a) * a


class HashShardMap:
    """Statistical model of the runtime shard mapping.

    The runtime key is (region address, offset); the address is not
    recorded in format-v1 traces. This map uses the same hash with a random
    64-bit constant per (layer, kind) standing in for region*phi + tensor
    offset, and offset step = slice bytes. Different seeds = different
    plausible address layouts.
    """

    def __init__(self, n: int, slice_bytes: Dict[Tuple[int, int], int], seed: int = 0):
        self.n = n
        rng = random.Random(seed)
        self.base = {lk: rng.getrandbits(64) for lk in slice_bytes}
        self.step = dict(slice_bytes)
        self.label = f"hash-model(seed={seed})"
        self.exact = False

    def __call__(self, key: SliceKey) -> int:
        l, e, k = key
        return mix64(self.base[(l, k)] + e * self.step[(l, k)]) % self.n


class RecordedShardMap:
    """Exact runtime mapping read from a format-v2 trace (A/E records)."""

    def __init__(self, n: int, mapping: Dict[SliceKey, int]):
        self.n = n
        self.map = dict(mapping)
        self.label = "recorded (format v2)"
        self.exact = True

    def __call__(self, key: SliceKey) -> int:
        return self.map[key]


@dataclass
class Decision:
    hit: bool
    victims: List[SliceKey] = field(default_factory=list)
    skipped: List[int] = field(default_factory=list)   # pinned slots passed per victim
    failed: bool = False
    shard: int = 0


class StoreReplay:
    """The runtime's sharded slice LRU with node-scoped pins."""

    category = "runtime-reproduction"

    def __init__(self, budget: int, shard_map, size_of: Callable[[SliceKey], int],
                 n_shards: Optional[int] = None):
        self.n = n_shards or shard_map.n
        self.budgets = shard_budgets(budget, self.n)
        self.budget = budget
        self.shard_of = shard_map
        self.size_of = size_of
        self.lru: List["OrderedDict[SliceKey, int]"] = [OrderedDict() for _ in range(self.n)]
        self.resident = [0] * self.n
        self.pins: Dict[SliceKey, int] = {}           # key -> pin count (all threads)
        self.held: Dict[Optional[int], List[SliceKey]] = {}   # thread -> its pins
        self.evictions = 0
        self.failures = 0
        self.violations = 0
        self.peak = 0

    # -- state ----------------------------------------------------------------
    @property
    def pinned(self) -> set:
        return {k for k, c in self.pins.items() if c > 0}

    def is_pinned(self, key: SliceKey) -> bool:
        return self.pins.get(key, 0) > 0

    def _pin(self, key: SliceKey, tid: Optional[int]) -> None:
        self.pins[key] = self.pins.get(key, 0) + 1
        self.held.setdefault(tid, []).append(key)

    def _release(self, tid: Optional[int]) -> None:
        for k in self.held.pop(tid, ()):
            c = self.pins.get(k, 0) - 1
            if c > 0:
                self.pins[k] = c
            else:
                self.pins.pop(k, None)

    def begin_node(self, threads: Optional[Iterable[Optional[int]]] = None) -> None:
        """Threads entering a node drop the pins of their previous node, before
        the barrier. `threads=None`: every thread enters (one pool)."""
        if threads is None:
            self.pins.clear()
            self.held.clear()
            return
        for t in threads:
            self._release(t)

    def warm_start(self, keys_lru_to_mru: Iterable[SliceKey]) -> None:
        for k in keys_lru_to_mru:
            s = self.shard_of(k)
            b = self.size_of(k)
            if k not in self.lru[s] and self.resident[s] + b <= self.budgets[s]:
                self.lru[s][k] = b
                self.resident[s] += b

    def resident_keys(self) -> List[SliceKey]:
        return [k for od in self.lru for k in od]

    def total_resident(self) -> int:
        return sum(self.resident)

    # -- one request ------------------------------------------------------------
    def access(self, key: SliceKey, tid: Optional[int] = None) -> Decision:
        s = self.shard_of(key)
        od = self.lru[s]
        if key in od:
            od.move_to_end(key)
            self._pin(key, tid)
            return Decision(True, shard=s)
        need = self.size_of(key)
        d = Decision(False, shard=s)
        while self.budgets[s] and self.resident[s] + need > self.budgets[s]:
            victim = None
            skipped = 0
            for k in od:                       # LRU end first
                if self.pins.get(k, 0) > 0:
                    skipped += 1
                    continue
                victim = k
                break
            if victim is None:
                d.failed = True
                self.failures += 1
                return d
            self.resident[s] -= od.pop(victim)
            self.evictions += 1
            d.victims.append(victim)
            d.skipped.append(skipped)
        od[key] = need
        self.resident[s] += need
        if self.budgets[s] and self.resident[s] > self.budgets[s]:
            self.violations += 1
        self._pin(key, tid)
        self.peak = max(self.peak, self.total_resident())
        return d


# --------------------------------------------------------------- streams ---

@dataclass
class Request:
    """One slice request, in the order a replay must serve it."""
    node: int                 # global node index (0, 1, 2, ...)
    graph: int
    node_kind: int            # kind of the MUL_MAT_ID node that requested it
    key: SliceKey
    phase: str                # decode | prefill
    workload: bool            # False for warm-up / flush-sentinel steps
    # runtime facts, when the stream comes from a runtime trace
    outcome: Optional[str] = None
    shard: Optional[int] = None
    seq: Optional[int] = None
    t_ns: Optional[int] = None
    tid: Optional[int] = None  # requesting thread (runtime trace)


DECODE_POOL, BATCH_POOL = "decode-pool", "batch-pool"


def synthetic_stream(trace, excluded_segments=("warmup", "flush-sentinel"),
                     split_pools: bool = False) -> Iterator[Request]:
    """The request stream the runtime would generate for a router trace:
    for every step and layer, gate node (3 kinds, k order), up node, down node.

    split_pools: llama.cpp ran with --threads != --threads-batch, so prompt
    and decode graphs are computed by different threads. A router trace says
    nothing about threads, so each pool is modelled as ONE thread (`tid` =
    pool name): its pins persist while the other pool computes. This is an
    approximation (the main thread belongs to both pools); on the Windows v2
    trace it leaves 1.25% event disagreement and −0.01 pp, against 5.29% and
    +0.09 pp for a single-pool model (EXP-021 follow-up). Exact replay needs
    the runtime trace (fidelity.compare)."""
    node = 0
    for step in trace.steps:
        phase = "decode" if step.n_tokens == 1 else "prefill"
        wl = step.segment not in excluded_segments
        for layer in trace.layers:
            ex = step.experts(layer)
            if not ex:
                continue
            for nk in NODE_KINDS:
                for kind in NODE_KINDS:
                    for e in ex:
                        yield Request(node, step.graph, nk, (layer, e, kind), phase, wl,
                                      tid=(BATCH_POOL if phase == "prefill" else DECODE_POOL)
                                      if split_pools else None)
                node += 1


def slice_sizes_from_model(model) -> Dict[Tuple[int, int], int]:
    """(layer, kind) -> aligned slice bytes, from a GGUF model."""
    from .. import expert_store as es
    n = int(getattr(model, "expert_count", 0) or 0)
    out = {}
    names = {"gate": KIND_GATE, "up": KIND_UP, "down": KIND_DOWN}
    for layer, kinds in es.expert_tensors(model).items():
        for kname, t in kinds.items():
            out[(int(layer), names[kname])] = round_up(int(t.n_bytes) // n)
    return out


@dataclass
class Tally:
    req: int = 0
    hit: int = 0

    def add(self, hit: bool) -> None:
        self.req += 1
        self.hit += 1 if hit else 0

    @property
    def rate(self) -> Optional[float]:
        return self.hit / self.req if self.req else None


class Metrics:
    """The same metric definitions for runtime outcomes and replay decisions."""

    def __init__(self) -> None:
        self.t: Dict[Tuple[str, str], Tally] = {}
        self._expert: Dict[Tuple[int, int, int], List[int]] = {}   # (node, layer, expert) -> [n, hits]
        self._expert_phase: Dict[Tuple[int, int, int], str] = {}

    def add(self, r: Request, hit: bool) -> None:
        if not r.workload:
            return
        for ph in (r.phase, "all"):
            self.t.setdefault(("slice_all", ph), Tally()).add(hit)
            if r.node_kind == KIND_GATE:
                self.t.setdefault(("slice_gate", ph), Tally()).add(hit)
        if r.node_kind == KIND_GATE:
            k = (r.node, r.key[0], r.key[1])
            v = self._expert.setdefault(k, [0, 0])
            v[0] += 1
            v[1] += 1 if hit else 0
            self._expert_phase[k] = r.phase

    def rates(self) -> Dict[str, Optional[float]]:
        out = {}
        for (m, ph), tl in sorted(self.t.items()):
            out[f"{m}_{ph}"] = tl.rate
            out[f"{m}_{ph}_n"] = tl.req
        for ph in ("decode", "prefill", "all"):
            ev = [v for k, v in self._expert.items() if ph == "all" or self._expert_phase[k] == ph]
            out[f"expert_event_{ph}"] = (sum(1 for v in ev if v[1] == v[0]) / len(ev)) if ev else None
            out[f"expert_event_{ph}_n"] = len(ev)
            out[f"partial_expert_events_{ph}"] = sum(1 for v in ev if 0 < v[1] < v[0])
        return out


def run(stream: Iterable[Request], replay: StoreReplay) -> Tuple[Metrics, StoreReplay]:
    """Requests with a `tid` release only that thread's pins at node entry
    (split thread pools); requests without one release every pin."""
    m = Metrics()
    cur = None
    for r in stream:
        if r.node != cur:
            replay.begin_node([r.tid] if r.tid is not None else None)
            cur = r.node
        d = replay.access(r.key, r.tid)
        m.add(r, d.hit)
    return m, replay
