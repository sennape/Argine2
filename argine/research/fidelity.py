"""Runtime-vs-replay fidelity validation, event by event.

Input: one research run directory (raw.tsv + markers.jsonl). Output: whether a
replay policy makes the SAME decisions the Expert Store made, and where the two
state machines first stop agreeing.

Three checks, from weakest assumption to strongest:

1. reconstruct  (mapping-free, any format)
   Rebuild the runtime's own resident set from its records alone (insert on
   'm', remove on E and U) and verify that every 'h'/'w' was resident and every
   'm' was absent. Validates the event stream (missing / duplicate / misordered
   records) without modelling any policy.

2. free-running replay  (any format)
   Feed the runtime's request stream to StoreReplay with its own state and
   compare outcomes and evictions decision by decision. With a format-v1 trace
   the shard mapping must be drawn from the hash family (statistical); with a
   format-v2 trace it is the recorded one (exact).

3. lockstep replay  (format v2 only)
   Like 2, but before every decision the replay's shard state is the runtime's
   (it applies the runtime's outcome and victims after predicting them). Every
   disagreement is then an independent semantic difference, not a cascade.

The first divergence is reported with the state needed to explain it.
"""

from __future__ import annotations

import bisect
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from . import rawlog
from .storereplay import (KIND_GATE, HashShardMap, Metrics, RecordedShardMap, Request,
                          StoreReplay, round_up, runtime_shard_count)

EXCLUDED = ("warmup", "flush-sentinel")


@dataclass
class RuntimeStream:
    requests: List[Request]
    evictions_by_shard: Dict[int, List[Tuple[int, Tuple[int, int, int], int]]]  # seq,key,skipped
    undos_by_shard: Dict[int, List[Tuple[int, Tuple[int, int, int]]]]
    sizes: Dict[Tuple[int, int], int]
    budget: int
    n_shards: int
    version: int
    mapping: Dict[Tuple[int, int, int], int]
    unassigned: int
    nodes: int
    # node index -> threads entering it (the pool of its graph: llama.cpp uses
    # --threads for decode graphs and --threads-batch for prompt graphs, which
    # may be different threads). None: one pool, every thread enters.
    node_threads: Optional[Dict[int, frozenset]] = None
    pools: Optional[Dict[int, List[int]]] = None
    # format v2: per shard, the sequence numbers that were produced but never
    # reached the file (lost records)
    missing_seq_by_shard: Optional[Dict[int, List[int]]] = None


def _windows(log: rawlog.RawLog, markers: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
    open_: Dict[str, int] = {}
    out = []
    for m in markers:
        if m.get("segment") not in EXCLUDED or "system_ns" not in m:
            continue
        t = log.system_to_steady(int(m["system_ns"]))
        if m.get("event") == "request_start":
            open_[m["segment"]] = t
        elif m.get("event") == "request_end" and m["segment"] in open_:
            out.append((open_.pop(m["segment"]), t))
    for _seg, t in open_.items():
        out.append((t, float("inf")))
    return out


def load_runtime_stream(run_dir, log: Optional[rawlog.RawLog] = None) -> RuntimeStream:
    from pathlib import Path
    d = Path(run_dir)
    raw = d / "raw.tsv" if (d / "raw.tsv").exists() else d / "raw.tsv.gz"
    log = log or rawlog.parse(raw)
    markers = rawlog.read_markers(d / "markers.jsonl")
    win = _windows(log, markers)
    ntok: Dict[int, int] = {}
    for g, _ts, _l, n, _k, _ids in log.routes:
        ntok[g] = max(ntok.get(g, 0), n)
    nodes = sorted(log.nodes, key=lambda n: n.start_ns)
    starts = [n.start_ns for n in nodes]
    per_node: Dict[int, List[rawlog.Access]] = defaultdict(list)
    unassigned = 0
    for a in log.accesses:
        i = bisect.bisect_right(starts, a.t_ns) - 1
        if i < 0 or a.t_ns > nodes[i].ready_ns:
            unassigned += 1
            continue
        per_node[i].append(a)
    v2 = all(a.seq is not None for a in log.accesses[:1000]) and log.accesses
    reqs: List[Request] = []
    sizes: Dict[Tuple[int, int], int] = {}
    mapping: Dict[Tuple[int, int, int], int] = {}
    for i in sorted(per_node):
        nd = nodes[i]
        phase = "decode" if ntok.get(nd.graph, 1) == 1 else "prefill"
        acc = per_node[i]
        if v2:
            acc.sort(key=lambda a: (a.shard, a.seq))
        else:
            acc.sort(key=lambda a: a.t_ns)
        for a in acc:
            wl = not any(lo <= a.t_ns <= hi for lo, hi in win)
            key = (a.layer, a.expert, a.kind)
            sizes[(a.layer, a.kind)] = max(sizes.get((a.layer, a.kind), 0), round_up(a.bytes))
            if a.shard is not None:
                mapping[key] = a.shard
            reqs.append(Request(i, nd.graph, nd.kind, key, phase, wl, a.outcome, a.shard, a.seq,
                                a.t_ns, a.tid))
    ev: Dict[int, List] = defaultdict(list)
    for e in log.evictions:
        if e.shard is not None:
            ev[e.shard].append((e.seq, (e.layer, e.expert, e.kind), e.skipped))
            mapping[(e.layer, e.expert, e.kind)] = e.shard
    for v in ev.values():
        v.sort()
    un: Dict[int, List] = defaultdict(list)
    for u in log.undos:
        un[u.shard].append((u.seq, (u.layer, u.expert, u.kind)))
    budget = int(log.config.get("budget_bytes", "0") or 0)
    shards = int(log.config.get("cache_shards", "0") or 0) or runtime_shard_count(budget)
    # Thread pools: every thread of the pool computing a graph enters every
    # node of it (and drops its pins there). The pool of a node is identified
    # by its thread count (N record nth); its members are the threads seen
    # requesting in nodes with that count. A pool member that never requests
    # holds no pins, so leaving it out changes nothing.
    pool: Dict[int, set] = defaultdict(set)
    for i, acc in per_node.items():
        for a in acc:
            pool[nodes[i].nth].add(a.tid)
    node_threads = {i: frozenset(pool[nodes[i].nth]) for i in per_node}
    missing = None
    if v2:
        present: Dict[int, set] = defaultdict(set)
        for a in log.accesses:
            present[a.shard].add(a.seq)
        for e in log.evictions:
            present[e.shard].add(e.seq)
        for u in log.undos:
            present[u.shard].add(u.seq)
        missing = {sh: sorted(set(range(max(q) + 1)) - q) for sh, q in present.items()
                   if len(q) != max(q) + 1}
    return RuntimeStream(reqs, dict(ev), dict(un), sizes, budget, shards, 2 if v2 else 1,
                         mapping, unassigned, len(nodes), node_threads,
                         {k: sorted(v) for k, v in pool.items()}, missing)


def thread_pool_config(run_dir) -> Dict[str, Any]:
    """--threads / --threads-batch the run's llama-server was started with
    (manifest). Different values = two thread pools, each holding its own
    pins while the other computes."""
    import json
    from pathlib import Path
    try:
        m = json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))
        args = ((m.get("fingerprint") or {}).get("runtime_flags") or {}).get("llama_args") or []
    except Exception:  # noqa: BLE001
        return {}

    def val(flag):
        try:
            return int(args[args.index(flag) + 1])
        except (ValueError, IndexError):
            return None
    t, tb = val("--threads"), val("--threads-batch")
    return {"threads": t, "threads_batch": tb,
            "split": t is not None and tb is not None and t != tb}


# ------------------------------------------------------- 0. record integrity

def _in(t: int, win: List[Tuple[float, float]]) -> bool:
    return any(lo <= t <= hi for lo, hi in win)


def seq_gaps(log: rawlog.RawLog, windows: List[Tuple[float, float]]) -> Dict[str, Any]:
    """Every missing per-shard sequence number (format v2), classified.

    The store increments a shard's sequence exactly once per A, E or U record
    it emits, under the shard mutex, and every increment is traced while the
    trace is active. A missing number is therefore a record that was produced
    and then lost on the way to the file. It is REPLAY-RELEVANT unless no
    workload record of the same shard follows it: a lost record that is only
    followed by warm-up / flush-sentinel records cannot change any counted
    decision (decisions are replayed per shard, shards are independent).
    """
    per: Dict[int, Dict[int, Tuple[int, str, Tuple[int, int, int], Optional[int]]]] = defaultdict(dict)
    dups = 0
    for a in log.accesses:
        if a.seq is None:
            return {"applicable": False}
        if a.seq in per[a.shard]:
            dups += 1
        per[a.shard][a.seq] = (a.t_ns, "A", (a.layer, a.expert, a.kind), a.tid)
    for e in log.evictions:
        if e.seq in per[e.shard]:
            dups += 1
        per[e.shard][e.seq] = (e.t_ns, "E", (e.layer, e.expert, e.kind), None)
    for u in log.undos:
        if u.seq in per[u.shard]:
            dups += 1
        per[u.shard][u.seq] = (u.t_ns, "U", (u.layer, u.expert, u.kind), None)
    gaps = []
    for sh in sorted(per):
        d = per[sh]
        top = max(d)
        missing = [q for q in range(top + 1) if q not in d]
        if not missing:
            continue
        # last workload record of this shard (by sequence number)
        wl_seqs = [q for q, r in d.items() if r[1] == "A" and not _in(r[0], windows)]
        last_wl = max(wl_seqs) if wl_seqs else -1
        for q in missing:
            prev = max((x for x in d if x < q), default=None)
            nxt = min((x for x in d if x > q), default=None)
            rel = q < last_wl
            seg_prev = d[prev][0] if prev is not None else None
            gaps.append({
                "shard": sh, "seq": q, "replay_relevant": rel,
                "class": ("followed by workload records: replay-relevant" if rel else
                          "after the last workload record of the shard (warm-up / flush "
                          "sentinel tail): replay-irrelevant"),
                "prev": {"seq": prev, "t_ns": seg_prev, "type": d[prev][1], "key": d[prev][2],
                         "tid": d[prev][3]} if prev is not None else None,
                "next": {"seq": nxt, "t_ns": d[nxt][0], "type": d[nxt][1], "key": d[nxt][2],
                         "tid": d[nxt][3]} if nxt is not None else None,
                # the lost record happened between prev and next
                "inside_excluded_segment": bool(seg_prev is not None and _in(seg_prev, windows)
                                                and nxt is not None and _in(d[nxt][0], windows)),
            })
    return {"applicable": True, "duplicates": dups, "missing": len(gaps),
            "shards_with_seq_gaps": len({g["shard"] for g in gaps}),
            "shards_with_replay_relevant_seq_gaps": len({g["shard"] for g in gaps
                                                         if g["replay_relevant"]}),
            "gaps": gaps}


def node_completeness(log: rawlog.RawLog, windows: List[Tuple[float, float]]) -> Dict[str, Any]:
    """Independent of sequence numbers: every MoE node must request exactly the
    slices its router decision implies (all three kinds of the layer's
    selected experts, in each of the gate / up / down nodes). Finds lost A
    records that a sequence check cannot see (the last records of a shard)."""
    route: Dict[Tuple[int, int], set] = {}
    for g, _ts, l, _n, _k, ids in log.routes:
        route.setdefault((g, l), set()).update(int(x) for x in ids)
    nodes = sorted(log.nodes, key=lambda n: n.start_ns)
    starts = [n.start_ns for n in nodes]
    seen: Dict[int, set] = defaultdict(set)
    for a in log.accesses:
        i = bisect.bisect_right(starts, a.t_ns) - 1
        if 0 <= i and a.t_ns <= nodes[i].ready_ns:
            seen[i].add((a.layer, a.expert, a.kind))
    bad = []
    for i, nd in enumerate(nodes):
        exp = {(nd.layer, e, k) for e in route.get((nd.graph, nd.layer), ()) for k in range(3)}
        got = seen.get(i, set())
        if exp != got:
            bad.append({"graph": nd.graph, "layer": nd.layer, "node_kind": nd.kind,
                        "missing": sorted(exp - got), "unexpected": sorted(got - exp),
                        "excluded_segment": _in(nd.start_ns, windows)})
    return {"nodes": len(nodes), "incomplete": len(bad),
            "incomplete_in_workload": sum(1 for b in bad if not b["excluded_segment"]),
            "details": bad[:50]}


# ------------------------------------------------------------ 1. reconstruct

def reconstruct(log: rawlog.RawLog) -> Dict[str, Any]:
    """Mapping-free consistency of the runtime's own records."""
    events = []
    for a in log.accesses:
        events.append((a.t_ns, 1, "A", a))
    for e in log.evictions:
        events.append((e.t_ns, 0, "E", e))
    for u in log.undos:
        events.append((u.t_ns, 0, "U", u))
    v2 = bool(log.accesses) and log.accesses[0].seq is not None
    if v2:
        # exact: per shard, in seq order; shards are independent
        per: Dict[int, List] = defaultdict(list)
        for a in log.accesses:
            per[a.shard].append((a.seq, "A", a))
        for e in log.evictions:
            per[e.shard].append((e.seq, "E", e))
        for u in log.undos:
            per[u.shard].append((u.seq, "U", u))
        seqs = [sorted(x for x in v) for v in per.values()]
        ordered = [x for s in seqs for x in s]
        gaps = sum(1 for s in seqs if [x[0] for x in s] != list(range(len(s))))
    else:
        ordered = [(t, kind, obj) for t, _p, kind, obj in sorted(events, key=lambda x: (x[0], x[1]))]
        gaps = None
    res = set()
    bad = {"hit_not_resident": 0, "miss_already_resident": 0, "evict_not_resident": 0}
    first = None
    for _k, kind, o in ordered:
        key = (o.layer, o.expert, o.kind)
        if kind == "A":
            if o.outcome in ("h", "w"):
                if key not in res:
                    bad["hit_not_resident"] += 1
                    first = first or {"record": "A", "key": key, "outcome": o.outcome, "t_ns": o.t_ns}
            elif o.outcome == "m":
                if key in res:
                    bad["miss_already_resident"] += 1
                    first = first or {"record": "A", "key": key, "outcome": "m", "t_ns": o.t_ns}
                res.add(key)
        else:
            if key not in res:
                bad["evict_not_resident"] += 1
                first = first or {"record": kind, "key": key, "t_ns": o.t_ns}
            res.discard(key)
    return {"order": "per-shard seq (exact)" if v2 else "timestamps across threads (approximate)",
            "inconsistencies": bad, "total": sum(bad.values()), "first": first,
            "shards_with_seq_gaps": gaps, "records": len(ordered)}


# ------------------------------------------------------- 2/3. replay compare

@dataclass
class Divergence:
    index: int
    node: int
    graph: int
    key: Tuple[int, int, int]
    shard: int
    kind: str
    runtime: Dict[str, Any]
    replay: Dict[str, Any]
    context: Dict[str, Any] = field(default_factory=dict)


def _runtime_victims(st: RuntimeStream, shard: int, prev_seq: int, seq: int):
    lst = st.evictions_by_shard.get(shard, [])
    lo = bisect.bisect_right(lst, (prev_seq, (10**9,), 10**9))
    hi = bisect.bisect_left(lst, (seq, (-1,), -1))
    return [(k, sk) for _s, k, sk in lst[lo:hi]]


def compare(st: RuntimeStream, shard_map=None, lockstep: bool = False,
            max_divergences: int = 20, pin_model: str = "per-thread") -> Dict[str, Any]:
    """Serve the runtime's request stream with StoreReplay and compare.

    pin_model: "per-thread" (runtime semantics: each thread drops its own pins
    when it enters a node of its pool) or "single-pool" (every thread drops
    its pins at every node: the model used before the Windows v2 follow-up,
    exact only when one thread pool computes every graph)."""
    if pin_model not in ("per-thread", "single-pool"):
        raise ValueError(pin_model)
    if shard_map is None:
        if st.version < 2:
            raise ValueError("format-v1 trace: pass a statistical shard map (HashShardMap)")
        shard_map = RecordedShardMap(st.n_shards, st.mapping)
    if lockstep and not getattr(shard_map, "exact", False):
        raise ValueError("lockstep needs the recorded shard mapping (format v2)")
    rp = StoreReplay(st.budget, shard_map, lambda k: st.sizes[(k[0], k[2])], st.n_shards)
    rt_m, rp_m = Metrics(), Metrics()
    last_seq: Dict[int, int] = defaultdict(lambda: -1)
    recent: Dict[int, List] = defaultdict(list)
    cur = None
    n = agree = outcome_dis = victim_dis = skip_dis = rt_skip_ev = rp_skip_ev = 0
    tot = {"rt_miss": 0, "rp_miss": 0, "rt_miss_bytes": 0, "rp_miss_bytes": 0,
           "rt_miss_wl": 0, "rp_miss_wl": 0}
    divs: List[Divergence] = []
    # Regions of the request stream (EXP-007 80B follow-up). The measured
    # workload is held to strict equality; so is everything BEFORE its last
    # request (warm-up, gaps between requests), because a difference there
    # would carry into the workload state. Only the TAIL after the last
    # workload request (flush sentinel, process termination) is reported
    # separately, and a tail difference counts as explained only if its shard
    # had already lost records (missing sequence numbers below the request's
    # own): shards are independent, so lost records in one shard cannot move
    # another shard's decisions.
    last_wl = max((i for i, q in enumerate(st.requests) if q.workload), default=-1)
    missing = st.missing_seq_by_shard or {}

    def _region(i, q):
        if q.workload:
            return "workload"
        return "pre_workload" if i < last_wl else "tail"
    reg = {k: {"requests": 0, "disagreements": 0, "outcome_disagreements": 0,
               "victim_disagreements": 0, "skip_count_disagreements": 0,
               "rt_misses": 0, "rp_misses": 0, "rt_miss_bytes": 0, "rp_miss_bytes": 0,
               "rt_evictions": 0, "rp_evictions": 0, "rp_failures": 0}
           for k in ("workload", "pre_workload", "tail")}
    tail_explained = tail_unexplained = 0
    for idx, r in enumerate(st.requests):
        if r.node != cur:
            rp.begin_node(st.node_threads.get(r.node)
                          if (st.node_threads and pin_model == "per-thread") else None)
            cur = r.node
        s = shard_map(r.key)
        od = rp.lru[s]
        tail = [(k, rp.is_pinned(k)) for k in list(od.keys())[:12]]
        res_before = rp.resident[s]
        d = rp.access(r.key, r.tid if pin_model == "per-thread" else None)
        rt_hit = r.outcome in ("h", "w")
        sz = rp.size_of(r.key)
        if not rt_hit:
            tot["rt_miss"] += 1
            tot["rt_miss_bytes"] += sz
            tot["rt_miss_wl"] += 1 if r.workload else 0
        if not d.hit:
            tot["rp_miss"] += 1
            tot["rp_miss_bytes"] += sz
            tot["rp_miss_wl"] += 1 if r.workload else 0
        rt_m.add(r, rt_hit)
        rp_m.add(r, d.hit)
        n += 1
        rt_vict = None
        if getattr(shard_map, "exact", False) and r.seq is not None:
            rt_vict = _runtime_victims(st, s, last_seq[s], r.seq)
            last_seq[s] = r.seq
        same_outcome = rt_hit == d.hit
        same_victims = rt_vict is None or [k for k, _ in rt_vict] == d.victims
        rp_skip_ev += sum(1 for x in d.skipped if x)
        g = reg[_region(idx, r)]
        g["requests"] += 1
        g["rt_misses"] += 0 if rt_hit else 1
        g["rp_misses"] += 0 if d.hit else 1
        g["rt_miss_bytes"] += 0 if rt_hit else sz
        g["rp_miss_bytes"] += 0 if d.hit else sz
        g["rp_evictions"] += len(d.victims)
        g["rp_failures"] += 1 if d.failed else 0
        if rt_vict is not None:
            g["rt_evictions"] += len(rt_vict)
            rt_skip_ev += sum(1 for _k, x in rt_vict if x)
            if same_victims and [x for _k, x in rt_vict] != list(d.skipped):
                skip_dis += 1   # same victim, different number of pinned slots passed
                g["skip_count_disagreements"] += 1
        if not (same_outcome and same_victims):
            g["disagreements"] += 1
            g["outcome_disagreements"] += 0 if same_outcome else 1
            g["victim_disagreements"] += 0 if same_victims else 1
            if _region(idx, r) == "tail":
                lost = r.seq is not None and any(q < r.seq for q in missing.get(s, ()))
                tail_explained += 1 if lost else 0
                tail_unexplained += 0 if lost else 1
        if same_outcome and same_victims:
            agree += 1
        else:
            outcome_dis += 0 if same_outcome else 1
            victim_dis += 0 if same_victims else 1
            if len(divs) < max_divergences:
                divs.append(Divergence(
                    idx, r.node, r.graph, r.key, s,
                    "outcome" if not same_outcome else "victims",
                    {"outcome": r.outcome, "victims": rt_vict, "seq": r.seq},
                    {"hit": d.hit, "victims": d.victims, "skipped": d.skipped,
                     "failed": d.failed},
                    {"shard_budget": rp.budgets[s], "resident_before": res_before,
                     "resident_after": rp.resident[s],
                     "lru_tail_before (oldest first, pinned?)": tail,
                     "pinned_total": len(rp.pinned),
                     "pins_by_thread": {str(t): len(v) for t, v in rp.held.items()},
                     "requesting_thread": r.tid,
                     "previous_requests_in_shard": list(recent[s][-10:]),
                     "phase": r.phase, "node_kind": r.node_kind,
                     "region": _region(idx, r),
                     "missing_seq_in_shard_before": [q for q in missing.get(s, ())
                                                     if r.seq is not None and q < r.seq][:50]}))
            if lockstep:
                _force(rp, s, r, rt_hit, rt_vict)
        recent[s].append((r.key, r.outcome, "hit" if d.hit else "miss"))
        if len(recent[s]) > 40:
            del recent[s][:20]
    return {
        "mode": "lockstep" if lockstep else "free-running",
        "shard_map": shard_map.label,
        "requests": n, "agree": agree,
        "event_disagreement_rate": (n - agree) / n if n else None,
        "outcome_disagreements": outcome_dis, "victim_disagreements": victim_dis,
        "skip_count_disagreements": skip_dis if rt_skip_ev or rp_skip_ev or skip_dis else 0,
        "runtime_evictions_past_pinned": rt_skip_ev, "replay_evictions_past_pinned": rp_skip_ev,
        "pin_model": pin_model, "thread_pools": st.pools,
        "first_divergence": divs[0].__dict__ if divs else None,
        "divergences": [d.__dict__ for d in divs],
        "runtime_metrics": rt_m.rates(), "replay_metrics": rp_m.rates(),
        "replay_evictions": rp.evictions, "replay_failures": rp.failures,
        "replay_budget_violations": rp.violations, "replay_peak_resident": rp.peak,
        "runtime_evictions": sum(len(v) for v in st.evictions_by_shard.values()) or None,
        "totals": tot,
        "regions": reg,
        "last_workload_request_index": last_wl,
        "tail_disagreements_explained_by_record_loss": tail_explained,
        "tail_disagreements_unexplained": tail_unexplained,
        "shard_budgets": rp.budgets,
    }


def _force(rp: StoreReplay, s: int, r: Request, rt_hit: bool, rt_vict) -> None:
    """Make the replay shard state equal to the runtime's after this decision."""
    od = rp.lru[s]
    # undo the replay's own decision
    if r.key in od and not rt_hit:
        pass
    for k, _sk in (rt_vict or []):
        if k in od:
            rp.resident[s] -= od.pop(k)
    if rt_hit:
        if r.key not in od:
            od[r.key] = rp.size_of(r.key)
            rp.resident[s] += od[r.key]
        od.move_to_end(r.key)
    else:
        if r.key not in od:
            od[r.key] = rp.size_of(r.key)
            rp.resident[s] += od[r.key]
        od.move_to_end(r.key)
    # victims the replay chose but the runtime did not: put them back at the LRU end
    # (they were the oldest unpinned slots)
    rp._pin(r.key, r.tid)


def hash_ensemble(st: RuntimeStream, seeds: Iterable[int]) -> List[Dict[str, Any]]:
    out = []
    for sd in seeds:
        res = compare(st, HashShardMap(st.n_shards, st.sizes, sd), lockstep=False,
                      max_divergences=1)
        out.append({"seed": sd, **{k: res[k] for k in ("event_disagreement_rate",
                                                       "outcome_disagreements",
                                                       "replay_evictions")},
                    "replay": res["replay_metrics"], "runtime": res["runtime_metrics"]})
    return out
