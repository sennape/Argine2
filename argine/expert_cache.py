"""Expert cache SSD-backed con eviction LRU, prefetch asincrono e telemetria.

Questo modulo e' la base della Milestone 3. Fa due cose:

1. Fornisce una cache expert REALE: legge le slice degli expert direttamente
   dal GGUF su NVMe (pread/mmap), le tiene in RAM sotto un budget rigido e ne
   espelle le meno usate. Serve a misurare davvero throughput SSD e latenza.

2. Fornisce un simulatore di routing (`simulate`) che fa passare una traccia di
   token attraverso la cache e produce le metriche del KPI: hit rate, byte letti
   da disco, MB/s, tok/s proiettati.

Nota di onesta' tecnica: per sostituire il caricamento degli expert *dentro*
llama.cpp serve una patch C++. Qui misuriamo il sottosistema di memoria/IO in
isolamento, che e' il fattore che decide se un MoE da 16-20 GB gira su 16 GB.
"""

from __future__ import annotations

import math
import os
import random
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .gguf import GGUFModel, TensorInfo
from .util import GB, MB, fmt_bytes


# --------------------------------------------------------------- indicizza ---

@dataclass
class ExpertSlice:
    """Una slice = (layer, expert_id, tensore) come intervallo di byte nel file."""
    layer: int
    expert: int
    tensor: str
    file: str
    offset: int
    nbytes: int

    @property
    def key(self) -> Tuple[int, int, str]:
        return (self.layer, self.expert, self.tensor)


@dataclass
class ExpertIndex:
    n_experts: int = 0
    n_used: int = 0
    layers: List[int] = field(default_factory=list)
    slices: Dict[Tuple[int, int], List[ExpertSlice]] = field(default_factory=dict)
    bytes_per_expert_layer: int = 0
    total_expert_bytes: int = 0
    shared_bytes: int = 0

    def get(self, layer: int, expert: int) -> List[ExpertSlice]:
        return self.slices.get((layer, expert), [])


def build_expert_index(model: GGUFModel) -> ExpertIndex:
    """Costruisce l'indice delle slice expert dai metadati del GGUF."""
    idx = ExpertIndex(
        n_experts=model.expert_count,
        n_used=model.expert_used_count or 1,
        shared_bytes=model.shared_bytes,
        total_expert_bytes=model.expert_bytes,
    )
    if not model.is_moe or model.expert_count <= 0:
        return idx

    per_layer_bytes: Dict[int, int] = {}
    for t in model.expert_tensors:
        layer = t.layer_index
        if layer is None:
            continue
        # tensore expert: l'ultimo asse e' l'indice di expert -> slice contigue
        n_exp_axis = t.dims[-1] if t.dims else model.expert_count
        if n_exp_axis != model.expert_count:
            # layout inatteso: dividiamo comunque in parti uguali
            n_exp_axis = model.expert_count
        slice_bytes = t.n_bytes // max(1, n_exp_axis)
        for e in range(n_exp_axis):
            # file e offset vengono dal tensore: con un GGUF splittato ogni
            # shard ha la propria area dati
            sl = ExpertSlice(
                layer=layer, expert=e, tensor=t.name,
                file=t.file or model.path,
                offset=(t.abs_offset if t.file else model.data_offset + t.offset)
                       + e * slice_bytes,
                nbytes=slice_bytes,
            )
            idx.slices.setdefault((layer, e), []).append(sl)
        per_layer_bytes[layer] = per_layer_bytes.get(layer, 0) + slice_bytes

    idx.layers = sorted({k[0] for k in idx.slices})
    if per_layer_bytes:
        idx.bytes_per_expert_layer = int(sum(per_layer_bytes.values()) / len(per_layer_bytes))
    return idx


# ------------------------------------------------------------------ reader ---

class SliceReader:
    """Lettore di slice dal GGUF. real=False non tocca il disco (solo modello)."""

    def __init__(self, real: bool = True, simulated_mb_s: float = 1800.0):
        self.real = real
        self.simulated_mb_s = simulated_mb_s
        self._fds: Dict[str, int] = {}
        self._lock = threading.Lock()
        self.bytes_read = 0
        self.read_time = 0.0
        self.reads = 0

    def _fd(self, path: str) -> int:
        with self._lock:
            fd = self._fds.get(path)
            if fd is None:
                flags = os.O_RDONLY
                if hasattr(os, "O_BINARY"):
                    flags |= os.O_BINARY  # Windows
                fd = os.open(path, flags)
                self._fds[path] = fd
            return fd

    def read(self, slices: Sequence[ExpertSlice]) -> int:
        total = 0
        t0 = time.perf_counter()
        for sl in slices:
            if self.real:
                data = os.pread(self._fd(sl.file), sl.nbytes, sl.offset) \
                    if hasattr(os, "pread") else self._read_seek(sl)
                total += len(data)
            else:
                total += sl.nbytes
        dt = time.perf_counter() - t0
        if not self.real:
            dt = total / (self.simulated_mb_s * MB)
        with self._lock:
            self.bytes_read += total
            self.read_time += dt
            self.reads += 1
        return total

    def _read_seek(self, sl: ExpertSlice) -> bytes:
        fd = self._fd(sl.file)
        with self._lock:
            os.lseek(fd, sl.offset, os.SEEK_SET)
            return os.read(fd, sl.nbytes)

    @property
    def throughput_mb_s(self) -> float:
        if self.read_time <= 0:
            return 0.0
        return self.bytes_read / MB / self.read_time

    def close(self) -> None:
        with self._lock:
            for fd in self._fds.values():
                try:
                    os.close(fd)
                except OSError:
                    pass
            self._fds.clear()


# ------------------------------------------------------------------- cache ---

@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    bytes_loaded: int = 0
    bytes_evicted: int = 0
    prefetch_issued: int = 0
    prefetch_useful: int = 0
    load_time: float = 0.0
    peak_resident: int = 0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    @property
    def prefetch_accuracy(self) -> float:
        return self.prefetch_useful / self.prefetch_issued if self.prefetch_issued else 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["lookups"] = self.lookups
        d["hit_rate"] = round(self.hit_rate, 4)
        d["prefetch_accuracy"] = round(self.prefetch_accuracy, 4)
        return d


class LRUExpertCache:
    """Cache LRU degli expert con budget in byte rigido.

    policy: 'lru' | '2q' | 'lfu'
    """

    def __init__(self, budget_bytes: int, index: ExpertIndex, reader: SliceReader,
                 policy: str = "lru", keep_bytes: bool = False):
        if budget_bytes <= 0:
            raise ValueError("budget della expert cache deve essere > 0")
        self.budget = int(budget_bytes)
        self.index = index
        self.reader = reader
        self.policy = policy.lower()
        self.keep_bytes = keep_bytes
        self.stats = CacheStats()
        self.resident = 0
        self._main: "OrderedDict[Tuple[int,int], int]" = OrderedDict()   # key -> nbytes
        self._ghost: "OrderedDict[Tuple[int,int], int]" = OrderedDict()  # 2Q: ingresso
        self._freq: Dict[Tuple[int, int], int] = {}
        self._payload: Dict[Tuple[int, int], bytes] = {}
        self._inflight: Dict[Tuple[int, int], threading.Event] = {}
        self._from_prefetch: set[Tuple[int, int]] = set()
        self._lock = threading.RLock()

    # ------------------------------------------------------------- lookup --
    def __contains__(self, key: Tuple[int, int]) -> bool:
        with self._lock:
            return key in self._main

    def touch(self, key: Tuple[int, int]) -> None:
        with self._lock:
            if key in self._main:
                self._main.move_to_end(key)
            self._freq[key] = self._freq.get(key, 0) + 1

    def get(self, layer: int, expert: int, from_prefetch: bool = False) -> bool:
        """Ritorna True se hit, False se miss (in tal caso carica la slice)."""
        key = (layer, expert)
        with self._lock:
            if key in self._main:
                self._main.move_to_end(key)
                self._freq[key] = self._freq.get(key, 0) + 1
                if not from_prefetch:
                    self.stats.hits += 1
                    # il prefetch e' stato "utile" solo se questo hit e' su una
                    # slice caricata in anticipo e non ancora consumata
                    if key in self._from_prefetch:
                        self._from_prefetch.discard(key)
                        self.stats.prefetch_useful += 1
                return True
            if not from_prefetch:
                self.stats.misses += 1
        self._load(key)
        return False

    def prefetch(self, layer: int, expert: int) -> None:
        key = (layer, expert)
        with self._lock:
            if key in self._main or key in self._inflight:
                return
            self.stats.prefetch_issued += 1
        self._load(key, prefetch=True)

    # --------------------------------------------------------------- load --
    def _load(self, key: Tuple[int, int], prefetch: bool = False) -> None:
        layer, expert = key
        slices = self.index.get(layer, expert)
        if not slices:
            return
        nbytes = sum(s.nbytes for s in slices)
        ev = threading.Event()
        with self._lock:
            waiting = self._inflight.get(key)
            if waiting is not None:
                pass
            else:
                self._inflight[key] = ev
        if waiting is not None:
            waiting.wait(timeout=30)
            return
        try:
            t0 = time.perf_counter()
            self.reader.read(slices)
            dt = time.perf_counter() - t0
            with self._lock:
                if prefetch:
                    self._from_prefetch.add(key)
                self._evict_for(nbytes)
                self._main[key] = nbytes
                self._main.move_to_end(key)
                self._freq[key] = self._freq.get(key, 0) + 1
                self.resident += nbytes
                self.stats.bytes_loaded += nbytes
                self.stats.load_time += dt
                self.stats.peak_resident = max(self.stats.peak_resident, self.resident)
        finally:
            with self._lock:
                self._inflight.pop(key, None)
            ev.set()

    def _evict_for(self, nbytes: int) -> None:
        while self.resident + nbytes > self.budget and self._main:
            victim = self._pick_victim()
            size = self._main.pop(victim, 0)
            self._payload.pop(victim, None)
            self._from_prefetch.discard(victim)
            self.resident -= size
            self.stats.evictions += 1
            self.stats.bytes_evicted += size

    def _pick_victim(self) -> Tuple[int, int]:
        if self.policy == "lfu" and self._freq:
            candidates = [k for k in self._main]
            return min(candidates, key=lambda k: (self._freq.get(k, 0), list(self._main).index(k)))
        # lru / 2q: il meno recente
        return next(iter(self._main))

    # -------------------------------------------------------------- stato --
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "policy": self.policy,
                "budget": self.budget,
                "resident": self.resident,
                "entries": len(self._main),
                "utilization": round(self.resident / self.budget, 4) if self.budget else 0.0,
                **self.stats.to_dict(),
            }


# ---------------------------------------------------------------- prefetch ---

class AsyncPrefetcher:
    """Carica gli expert del layer successivo mentre il layer corrente calcola."""

    def __init__(self, cache: LRUExpertCache, workers: int = 2, enabled: bool = True):
        self.cache = cache
        self.enabled = enabled
        self.pool = ThreadPoolExecutor(max_workers=max(1, workers),
                                       thread_name_prefix="argine-prefetch") if enabled else None
        self._pending: List[Any] = []

    def submit(self, layer: int, experts: Iterable[int]) -> None:
        if not self.enabled or self.pool is None:
            return
        for e in experts:
            self._pending.append(self.pool.submit(self.cache.prefetch, layer, e))
        if len(self._pending) > 64:
            self._pending = [f for f in self._pending if not f.done()]

    def drain(self, timeout: float = 5.0) -> None:
        t0 = time.time()
        for f in list(self._pending):
            remaining = timeout - (time.time() - t0)
            if remaining <= 0:
                break
            try:
                f.result(timeout=remaining)
            except Exception:
                pass
        self._pending.clear()

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)


# -------------------------------------------------------------- simulatore ---

@dataclass
class SimResult:
    tokens: int = 0
    layers: int = 0
    topk: int = 0
    cache: Dict[str, Any] = field(default_factory=dict)
    disk_bytes: int = 0
    disk_mb_s: float = 0.0
    io_time_s: float = 0.0
    wall_s: float = 0.0
    real_io: bool = True
    skew: float = 0.0
    prefetch: bool = True
    projected_tok_s: Optional[float] = None
    compute_tok_s: Optional[float] = None
    io_bound: bool = False
    budget: int = 0
    shared_bytes: int = 0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _zipf_weights(n: int, skew: float) -> List[float]:
    if skew <= 0:
        return [1.0 / n] * n
    raw = [1.0 / ((i + 1) ** skew) for i in range(n)]
    tot = sum(raw)
    return [r / tot for r in raw]


def _sample_experts(rng: random.Random, weights: List[float], topk: int,
                    n: int, rotate: int = 0) -> List[int]:
    """Estrae topk expert distinti secondo una distribuzione (rotate sposta la
    'zona calda' per layer, cosi' layer diversi hanno expert caldi diversi)."""
    chosen: set[int] = set()
    guard = 0
    while len(chosen) < min(topk, n) and guard < topk * 20:
        guard += 1
        r = rng.random()
        acc = 0.0
        for i, w in enumerate(weights):
            acc += w
            if r <= acc:
                chosen.add((i + rotate) % n)
                break
    while len(chosen) < min(topk, n):
        chosen.add(rng.randrange(n))
    return sorted(chosen)


def simulate(model: GGUFModel, *, cache_budget: int, tokens: int = 256,
             topk: Optional[int] = None, skew: float = 0.9, policy: str = "lru",
             prefetch: bool = True, real_io: bool = True, seed: int = 1234,
             compute_tok_s: Optional[float] = None,
             simulated_mb_s: float = 1800.0) -> SimResult:
    """Fa passare `tokens` token attraverso la expert cache e misura hit rate e IO.

    skew: 0 = routing uniforme (caso peggiore), 1.0+ = expert caldi marcati
          (caso realistico: il routing dei MoE e' fortemente sbilanciato).
    """
    index = build_expert_index(model)
    res = SimResult(real_io=real_io, skew=skew, prefetch=prefetch,
                    budget=cache_budget, shared_bytes=model.shared_bytes,
                    compute_tok_s=compute_tok_s)
    if not model.is_moe or not index.slices:
        res.notes.append("il modello non e' MoE: nessun expert da mettere in cache")
        return res

    n = index.n_experts
    k = topk or index.n_used or 1
    layers = index.layers
    res.tokens, res.layers, res.topk = tokens, len(layers), k

    reader = SliceReader(real=real_io, simulated_mb_s=simulated_mb_s)
    cache = LRUExpertCache(cache_budget, index, reader, policy=policy)
    pf = AsyncPrefetcher(cache, workers=2, enabled=prefetch)
    rng = random.Random(seed)
    weights = _zipf_weights(n, skew)

    t0 = time.perf_counter()
    try:
        for _tok in range(tokens):
            routing = [
                _sample_experts(rng, weights, k, n, rotate=li * 7 % max(1, n))
                for li, _l in enumerate(layers)
            ]
            for li, layer in enumerate(layers):
                wanted = routing[li]
                # prefetch del layer successivo mentre "calcoliamo" questo
                if prefetch and li + 1 < len(layers):
                    pf.submit(layers[li + 1], routing[li + 1])
                for e in wanted:
                    cache.get(layer, e)
        pf.drain(timeout=10)
    finally:
        pf.close()
        snap = cache.snapshot()
        res.cache = snap
        res.disk_bytes = reader.bytes_read
        res.io_time_s = reader.read_time
        res.disk_mb_s = reader.throughput_mb_s
        reader.close()

    res.wall_s = time.perf_counter() - t0

    # tok/s proiettati: il token deve attendere l'IO dei propri miss
    io_per_token = res.io_time_s / tokens if tokens else 0.0
    if compute_tok_s and compute_tok_s > 0:
        compute_per_token = 1.0 / compute_tok_s
        # il prefetch sovrappone parte dell'IO al calcolo
        overlap = 0.6 if prefetch else 0.0
        effective_io = max(0.0, io_per_token - compute_per_token * overlap)
        res.projected_tok_s = 1.0 / (compute_per_token + effective_io)
        res.io_bound = effective_io > compute_per_token
    else:
        res.projected_tok_s = 1.0 / io_per_token if io_per_token > 0 else None
        res.io_bound = True
        res.notes.append(
            "senza --compute-tok-s il valore proiettato considera solo l'IO "
            "(limite superiore teorico)"
        )
    if res.cache.get("hit_rate", 0) > 0.98 and skew > 0:
        res.notes.append("hit rate molto alto: il budget copre quasi tutti gli expert caldi")
    return res


# ------------------------------------------------------------------ report ---

def budget_table(model: GGUFModel, plan_budget: int, kv_bytes: int,
                 runtime_bytes: int, expert_cache: int) -> List[Tuple[str, int]]:
    rows = [
        ("shared tensors (residenti)", model.shared_bytes),
        ("expert cache", expert_cache),
        ("KV cache", kv_bytes),
        ("runtime", runtime_bytes),
    ]
    rows.append(("TOTALE", sum(v for _k, v in rows)))
    return rows


def residency_plan(model: GGUFModel, expert_cache_budget: int) -> Dict[str, Any]:
    """Quanti expert stanno in RAM col budget dato: hot vs cold."""
    idx = build_expert_index(model)
    per_slice = idx.bytes_per_expert_layer or 1
    n_layers = max(1, len(idx.layers))
    slots = int(expert_cache_budget // per_slice) if per_slice else 0
    per_layer_slots = slots / n_layers if n_layers else 0
    coverage = min(1.0, per_layer_slots / idx.n_experts) if idx.n_experts else 0.0
    return {
        "experts": idx.n_experts,
        "experts_used_per_token": idx.n_used,
        "moe_layers": n_layers,
        "bytes_per_expert_per_layer": per_slice,
        "expert_bytes_total": idx.total_expert_bytes,
        "shared_bytes": idx.shared_bytes,
        "cache_budget": expert_cache_budget,
        "cache_slots": slots,
        "slots_per_layer": round(per_layer_slots, 2),
        "coverage": round(coverage, 4),
        "cold_bytes": max(0, idx.total_expert_bytes - expert_cache_budget),
    }
