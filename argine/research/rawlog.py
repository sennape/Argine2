"""Parser of the runtime's raw research log ("argine-raw v1").

The C++ side (patches/argine-research.{h,cpp}) writes one tab-separated record
per line; see the header of argine-research.h for the authoritative format.
This module turns that log into structured Python objects and, from them, into
a router trace (``trace.Trace``) plus tables for the I/O timeline, node timing
and evictions.

The parser is strict about what it does not understand: unknown record types
are counted and reported, never silently dropped.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .trace import (PHASE_DECODE, PHASE_PREFILL, Step, Trace, TraceMeta)

FLAG_COMPLETE_EXACT = 1
FLAG_COMPLETE_BOUND = 2
FLAG_SYNCHRONOUS = 4
FLAG_FALLBACK = 8

KIND_NAMES = {0: "gate", 1: "up", 2: "down"}


@dataclass
class IORecord:
    req: int
    tid: int
    layer: int
    expert: int
    kind: int
    bytes: int
    issue_ns: int
    complete_ns: int
    wait_start_ns: int
    wait_end_ns: int
    ok: bool
    flags: int

    @property
    def exact(self) -> bool:
        return bool(self.flags & FLAG_COMPLETE_EXACT)

    @property
    def synchronous(self) -> bool:
        return bool(self.flags & FLAG_SYNCHRONOUS)

    @property
    def latency_ns(self) -> int:
        return max(0, self.complete_ns - self.issue_ns)

    @property
    def hidden_ns(self) -> int:
        """Part of [issue, complete] before the issuing thread started waiting."""
        if self.synchronous:
            return 0
        return max(0, min(self.complete_ns, self.wait_start_ns) - self.issue_ns)

    @property
    def unhidden_ns(self) -> int:
        return max(0, self.latency_ns - self.hidden_ns)


@dataclass
class Access:
    t_ns: int
    tid: int
    layer: int
    expert: int
    kind: int
    outcome: str      # h, m, w, x
    bytes: int
    req: int
    # format v2 only (None in v1 logs): owning shard and per-shard sequence
    # number of the decisive LRU state change, assigned under the shard mutex
    shard: Optional[int] = None
    seq: Optional[int] = None


@dataclass
class Eviction:
    t_ns: int
    layer: int
    expert: int
    kind: int
    bytes: int
    loaded_ns: int
    last_use_ns: int
    uses: int
    shard: Optional[int] = None      # format v2
    seq: Optional[int] = None        # format v2
    skipped: Optional[int] = None    # format v2: pinned/loading slots passed over

    @property
    def resident_ns(self) -> int:
        return max(0, self.t_ns - self.loaded_ns) if self.loaded_ns else 0


@dataclass
class Undo:
    """v2: a slot inserted for a miss and removed because its read failed."""
    t_ns: int
    layer: int
    expert: int
    kind: int
    bytes: int
    shard: int
    seq: int


@dataclass
class Node:
    graph: int
    layer: int
    kind: int
    start_ns: int
    ready_ns: int
    n_serve: int
    nth: int

    @property
    def residency_ns(self) -> int:
        return max(0, self.ready_ns - self.start_ns)


@dataclass
class Snapshot:
    t_ns: int
    graph: int
    resident: int
    budget: int
    hits: int
    misses: int
    evictions: int
    bytes_read: int
    violations: int


@dataclass
class RawLog:
    path: str
    steady_ns: int = 0
    system_ns: int = 0
    pid: int = 0
    config: Dict[str, str] = field(default_factory=dict)
    routes: List[Tuple[int, int, int, int, int, List[int]]] = field(default_factory=list)
    accesses: List[Access] = field(default_factory=list)
    io: List[IORecord] = field(default_factory=list)
    evictions: List[Eviction] = field(default_factory=list)
    undos: List[Undo] = field(default_factory=list)
    version: int = 1
    nodes: List[Node] = field(default_factory=list)
    snapshots: List[Snapshot] = field(default_factory=list)
    unknown: Counter = field(default_factory=Counter)
    malformed: int = 0
    truncated_tail: bool = False

    def system_to_steady(self, system_ns: int) -> int:
        return system_ns - (self.system_ns - self.steady_ns)

    def summary(self) -> Dict[str, object]:
        return {
            "routes": len(self.routes), "accesses": len(self.accesses), "io": len(self.io),
            "evictions": len(self.evictions), "undos": len(self.undos),
            "format_version": self.version, "nodes": len(self.nodes),
            "snapshots": len(self.snapshots), "malformed": self.malformed,
            "unknown_types": dict(self.unknown), "truncated_tail": self.truncated_tail,
            "config": dict(self.config),
        }


def parse(path: str | Path, *, keep_accesses: bool = True) -> RawLog:
    """Parse a raw log. `keep_accesses=False` skips A records (they are the
    bulk of the file) when only routing and I/O are needed."""
    log = RawLog(path=str(path))
    import gzip
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:  # type: ignore[operator]
        lines = fh.readlines()
    if lines and not lines[-1].endswith("\n"):
        # the process was terminated mid-write: the last line is partial
        log.truncated_tail = True
        lines = lines[:-1]
    for line in lines:
        f = line.rstrip("\n").split("\t")
        t = f[0]
        try:
            if t == "H":
                if len(f) < 6 or f[1] != "argine-raw":
                    raise ValueError("bad header")
                log.steady_ns, log.system_ns, log.pid = int(f[3]), int(f[4]), int(f[5])
                log.version = int(f[2])
            elif t == "C":
                log.config[f[1]] = f[2] if len(f) > 2 else ""
            elif t == "R":
                g, ts, layer, n_tok, n_used = (int(x) for x in f[1:6])
                ids = [int(x) for x in f[6:]]
                if len(ids) != n_tok * n_used:
                    raise ValueError("id count")
                log.routes.append((g, ts, layer, n_tok, n_used, ids))
            elif t == "A":
                if keep_accesses:
                    a = Access(int(f[1]), int(f[2]), int(f[3]), int(f[4]),
                               int(f[5]), f[6], int(f[7]), int(f[8]))
                    if len(f) >= 11:
                        a.shard, a.seq = int(f[9]), int(f[10])
                    log.accesses.append(a)
            elif t == "I":
                log.io.append(IORecord(int(f[1]), int(f[2]), int(f[3]), int(f[4]), int(f[5]),
                                       int(f[6]), int(f[7]), int(f[8]), int(f[9]), int(f[10]),
                                       f[11] == "1", int(f[12])))
            elif t == "E":
                ev = Eviction(*(int(x) for x in f[1:9]))
                if len(f) >= 12:
                    ev.shard, ev.seq, ev.skipped = int(f[9]), int(f[10]), int(f[11])
                log.evictions.append(ev)
            elif t == "U":
                log.undos.append(Undo(*(int(x) for x in f[1:8])))
            elif t == "N":
                log.nodes.append(Node(*(int(x) for x in f[1:8])))
            elif t == "S":
                log.snapshots.append(Snapshot(*(int(x) for x in f[1:10])))
            else:
                log.unknown[t] += 1
        except (ValueError, IndexError, TypeError):
            log.malformed += 1
    return log


def read_markers(path: str | Path) -> List[Dict[str, object]]:
    """Workload markers written by the harness (one JSON object per line)."""
    out = []
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def to_trace(log: RawLog, *, meta: Optional[TraceMeta] = None,
             markers: Optional[List[Dict[str, object]]] = None) -> Trace:
    """Build a router trace from the R records.

    Segments (workload markers) are assigned by time: the harness records the
    system time at which each request starts, the H record pairs the system and
    steady clocks, and a step belongs to the last request started before it.
    """
    by_graph: Dict[int, Dict[int, List[Tuple[int, ...]]]] = {}
    n_tok: Dict[int, int] = {}
    t0: Dict[int, int] = {}
    top_k = 0
    for g, ts, layer, n, k, ids in log.routes:
        by_graph.setdefault(g, {})[layer] = [tuple(ids[i * k:(i + 1) * k]) for i in range(n)]
        # max over layers: the last layer only computes the output tokens
        n_tok[g] = max(n_tok.get(g, 0), n)
        t0.setdefault(g, ts)
        t0[g] = min(t0[g], ts)
        top_k = max(top_k, k)

    starts: List[Tuple[int, str, str]] = []
    for mk in markers or []:
        if mk.get("event") == "request_start" and "system_ns" in mk:
            starts.append((log.system_to_steady(int(mk["system_ns"])),  # type: ignore[arg-type]
                           str(mk.get("segment") or ""), str(mk.get("prompt_id") or "")))
    starts.sort()

    steps: List[Step] = []
    for g in sorted(by_graph):
        seg = None
        for ts, s, _pid in starts:
            if ts <= t0[g]:
                seg = s or None
            else:
                break
        steps.append(Step(graph=g, n_tokens=n_tok[g], routes=by_graph[g],
                          phase=PHASE_PREFILL if n_tok[g] > 1 else PHASE_DECODE,
                          segment=seg, t_ns=t0[g]))
    m = meta or TraceMeta()
    if not m.layers:
        m.layers = sorted({l for st in steps for l in st.routes})
    if not m.top_k:
        m.top_k = top_k
    if not m.n_experts and log.config.get("store_experts"):
        m.n_experts = int(log.config["store_experts"])
    return Trace(meta=m, steps=steps)


def io_rows(log: RawLog) -> List[Dict[str, object]]:
    """The I records as flat rows (CSV-ready)."""
    rows = []
    for r in log.io:
        rows.append({
            "request_id": r.req, "thread": r.tid, "layer": r.layer, "expert": r.expert,
            "kind": KIND_NAMES.get(r.kind, r.kind), "bytes": r.bytes,
            "issue_ns": r.issue_ns, "complete_ns": r.complete_ns,
            "wait_start_ns": r.wait_start_ns, "wait_end_ns": r.wait_end_ns,
            "ok": int(r.ok), "flags": r.flags,
            "completion": "exact" if r.exact else "upper_bound",
            "synchronous": int(r.synchronous),
            "latency_ns": r.latency_ns, "hidden_ns": r.hidden_ns, "unhidden_ns": r.unhidden_ns,
        })
    return rows


def node_rows(log: RawLog) -> List[Dict[str, object]]:
    return [{"graph": n.graph, "layer": n.layer, "kind": KIND_NAMES.get(n.kind, n.kind),
             "start_ns": n.start_ns, "ready_ns": n.ready_ns, "residency_ns": n.residency_ns,
             "n_serve": n.n_serve, "threads": n.nth} for n in log.nodes]


def eviction_rows(log: RawLog) -> List[Dict[str, object]]:
    return [{"t_ns": e.t_ns, "layer": e.layer, "expert": e.expert,
             "kind": KIND_NAMES.get(e.kind, e.kind), "bytes": e.bytes,
             "resident_ns": e.resident_ns, "uses": e.uses} for e in log.evictions]


def node_attributed_outcomes(log: RawLog, keep=None) -> Dict[str, object]:
    """Attribute every slice access to the MoE node whose residency window
    contains it, and compute hit rates two ways.

    Why: for every layer the runtime runs three MUL_MAT_ID nodes (gate, up,
    down) and EACH of them requests the slices of all three kinds; the gate
    node loads them, the other two re-pin slices that are already resident.
    The runtime's hit counter therefore includes two guaranteed hits per
    slice and over-states the expert-level hit rate. `gate_node_hit_rate`
    counts only the first node of each layer, which is the one that pays the
    misses (verified: misses in up/down nodes are reported separately).
    """
    import bisect
    nodes = sorted(log.nodes, key=lambda n: n.start_ns)
    starts = [n.start_ns for n in nodes]
    by: Dict[Tuple[int, str], int] = {}
    unassigned = 0
    for a in log.accesses:
        if keep is not None and not keep(a.t_ns):
            continue
        i = bisect.bisect_right(starts, a.t_ns) - 1
        if i < 0 or a.t_ns > nodes[i].ready_ns:
            unassigned += 1
            continue
        k = (nodes[i].kind, a.outcome)
        by[k] = by.get(k, 0) + 1
    tot = sum(by.values())
    miss = sum(v for (k, o), v in by.items() if o == "m")
    first = sum(v for (k, o), v in by.items() if k == 0)
    miss_first = sum(v for (k, o), v in by.items() if k == 0 and o == "m")
    return {
        "accesses_by_node_kind_and_outcome": {f"{KIND_NAMES.get(k, k)}:{o}": v
                                              for (k, o), v in sorted(by.items())},
        "unassigned_accesses": unassigned,
        "runtime_style_hit_rate": 1 - miss / tot if tot else None,
        "gate_node_hit_rate": 1 - miss_first / first if first else None,
        "misses_outside_gate_nodes": miss - miss_first,
    }


def graph_timing(log: RawLog) -> List[Dict[str, object]]:
    """Per forward pass: wall time spanned by MoE nodes, time spent in the
    residency phase (blocked on experts) and the remainder (compute + sync).

    The residency phase of a node runs from thread 0 entering the expert
    store's compute_forward to the barrier after all reads; no matrix
    multiplication happens in it, so it is a direct measure of residency stall
    for that node (it includes hit bookkeeping and the barrier itself).
    The span of a graph is first node start -> last node ready, and therefore
    excludes the non-MoE work before the first and after the last MoE layer:
    `span_ns` is a lower bound of the forward pass time.
    """
    per: Dict[int, List[Node]] = {}
    for n in log.nodes:
        per.setdefault(n.graph, []).append(n)
    out = []
    for g in sorted(per):
        ns = per[g]
        start = min(n.start_ns for n in ns)
        end = max(n.ready_ns for n in ns)
        res = sum(n.residency_ns for n in ns)
        out.append({"graph": g, "nodes": len(ns), "span_ns": end - start,
                    "residency_ns": res, "other_ns": max(0, (end - start) - res)})
    return out
