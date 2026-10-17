"""Routing characterization: how routed expert accesses actually behave.

Contribution A/B of Argine2. Every function here takes a router trace and
returns plain numbers; nothing depends on a cache policy. Definitions are
written next to the code because they ARE the contribution: a metric whose
definition is ambiguous cannot be falsified.

Definitions (per layer L, over the selected phase, default: decode tokens)
------------------------------------------------------------------------
accesses          number of (token, expert) selections in L
count_L(e)        selections of expert e in L
p_L(e)            count_L(e) / accesses
entropy_bits      H_L = -sum_e p_L(e) log2 p_L(e)
entropy_norm      H_L / log2(n_experts)            (1 = uniform routing)
effective_experts 2 ** H_L                         (perplexity of the router)
gini              Gini coefficient of count_L over ALL n_experts (unused = 0)
topN_coverage     share of accesses that go to the N most selected experts
EWS_q             Effective Working Set: the minimum number of experts whose
                  selections cover a fraction q of the accesses (q = .90, .95,
                  .99). EWS_q / n_experts is its fractional form, and
                  EWS_q * expert_bytes(L) its size in bytes.
ws_tau            Denning working set: mean number of distinct experts selected
                  in L during the last tau tokens (tau = 1, 4, 16, 64, 256).
                  Unlike EWS (whole-trace), it measures short-term demand.
reuse distance    for consecutive selections of the same expert in L, the gap
                  in tokens (1 = selected again by the next token).
stack distance    event-atomic LRU stack distance: at the start of the step in
                  which e is selected, the number of distinct experts of L
                  used more recently than e. A per-layer LRU with c slots hits
                  e iff stack_distance < c (exact, see `lru_stack_distances`).
overlap_prev      mean |S_t ∩ S_{t-1}| / k between consecutive tokens' selected
                  sets (temporal locality at distance 1).
hot_set_stability Jaccard similarity of the EWS95 expert sets computed on the
                  first and on the second half of the trace.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .trace import PHASE_DECODE, Trace

INF = float("inf")
DEFAULT_TAUS = (1, 4, 16, 64, 256)
DEFAULT_QS = (0.90, 0.95, 0.99)


# -------------------------------------------------------------- basics ---

def shannon_entropy(counts: Iterable[int]) -> float:
    c = [x for x in counts if x > 0]
    tot = float(sum(c))
    if tot <= 0:
        return 0.0
    return -sum((x / tot) * math.log2(x / tot) for x in c)


def gini(counts: Sequence[int]) -> float:
    """Gini coefficient (0 = perfectly even, -> 1 = all mass on one expert)."""
    xs = sorted(counts)
    n = len(xs)
    tot = float(sum(xs))
    if n == 0 or tot <= 0:
        return 0.0
    cum = 0.0
    for i, x in enumerate(xs, 1):
        cum += i * x
    return (2.0 * cum) / (n * tot) - (n + 1.0) / n


def top_coverage(counts: Sequence[int], n: int) -> float:
    tot = float(sum(counts))
    if tot <= 0:
        return 0.0
    return sum(sorted(counts, reverse=True)[:n]) / tot


def ews(counts: Sequence[int], q: float) -> int:
    """Minimum number of experts covering a fraction q of the accesses."""
    tot = float(sum(counts))
    if tot <= 0:
        return 0
    need = q * tot
    acc = 0.0
    for i, x in enumerate(sorted(counts, reverse=True), 1):
        acc += x
        if acc + 1e-9 >= need:
            return i
    return len(counts)


def ews_set(counter: Counter, q: float) -> set:
    tot = float(sum(counter.values()))
    if tot <= 0:
        return set()
    out, acc = set(), 0.0
    for e, x in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
        out.add(e)
        acc += x
        if acc + 1e-9 >= q * tot:
            break
    return out


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / float(len(a | b))


def js_divergence(p: Dict[int, float], q: Dict[int, float]) -> float:
    """Jensen-Shannon divergence in bits (0 = identical, 1 = disjoint)."""
    keys = set(p) | set(q)
    sp, sq = float(sum(p.values())) or 1.0, float(sum(q.values())) or 1.0
    out = 0.0
    for k in keys:
        a, b = p.get(k, 0) / sp, q.get(k, 0) / sq
        m = 0.5 * (a + b)
        if a > 0:
            out += 0.5 * a * math.log2(a / m)
        if b > 0:
            out += 0.5 * b * math.log2(b / m)
    return out


# --------------------------------------------------------- per-layer data ---

def layer_counts(trace: Trace, phase: Optional[str] = PHASE_DECODE) -> Dict[int, Counter]:
    out: Dict[int, Counter] = {l: Counter() for l in trace.layers}
    for l in trace.layers:
        c = out[l]
        for _t, sel in trace.token_selections(l, phase):
            c.update(sel)
    return out


def full_counts(counter: Counter, n_experts: int) -> List[int]:
    return [counter.get(e, 0) for e in range(n_experts)]


def token_sets(trace: Trace, layer: int, phase: Optional[str] = PHASE_DECODE
               ) -> List[Tuple[int, Tuple[int, ...]]]:
    return list(trace.token_selections(layer, phase))


def working_set_mean(tsets: List[Tuple[int, Tuple[int, ...]]], tau: int) -> float:
    """Mean |distinct experts in the last tau tokens| (sliding window)."""
    if not tsets:
        return 0.0
    window: List[Tuple[int, ...]] = []
    live: Counter = Counter()
    tot = 0.0
    n = 0
    for _t, sel in tsets:
        window.append(sel)
        live.update(set(sel))
        if len(window) > tau:
            old = window.pop(0)
            for e in set(old):
                live[e] -= 1
                if live[e] == 0:
                    del live[e]
        tot += len(live)
        n += 1
    return tot / n


def reuse_distances(tsets: List[Tuple[int, Tuple[int, ...]]]) -> Tuple[List[int], int]:
    """Token gaps between consecutive selections of the same expert, and the
    number of first (cold) selections."""
    last: Dict[int, int] = {}
    out: List[int] = []
    cold = 0
    for t, sel in tsets:
        for e in set(sel):
            if e in last:
                out.append(t - last[e])
            else:
                cold += 1
            last[e] = t
    return out, cold


class _Fenwick:
    def __init__(self, n: int) -> None:
        self.n = n
        self.t = [0] * (n + 1)

    def add(self, i: int, v: int) -> None:
        while i <= self.n:
            self.t[i] += v
            i += i & -i

    def sum(self, i: int) -> int:
        s = 0
        while i > 0:
            s += self.t[i]
            i -= i & -i
        return s


def lru_stack_distances(event_sets: Sequence[Tuple[int, ...]]) -> List[float]:
    """Event-atomic LRU stack distances, one per access, in access order.

    `event_sets` is the sequence of expert sets of ONE cache scope (one layer
    for per-layer caches). For each access the distance is the number of
    distinct experts used more recently than it, measured at the start of the
    event (so members of the same event never count against each other).
    Within an event, ties are broken by expert id (higher id = more recent),
    matching `policies.PerLayerLRU`, which touches members in sorted order.
    Cold accesses get +inf.

    Property used by elasticity and oracle code: an event-atomic LRU with c
    slots (c >= the largest event) hits an access iff its distance < c.
    """
    n = sum(len(s) for s in event_sets) + 1
    bit = _Fenwick(n)
    last: Dict[int, int] = {}
    pos = 0
    active = 0
    out: List[float] = []
    for S in event_sets:
        members = sorted(set(S))
        for e in members:
            if e in last:
                out.append(float(active - bit.sum(last[e])))
            else:
                out.append(INF)
        for e in members:
            if e in last:
                bit.add(last[e], -1)
                active -= 1
            pos += 1
            bit.add(pos, 1)
            active += 1
            last[e] = pos
    return out


def lru_miss_curve(distances: Sequence[float], max_slots: int) -> List[int]:
    """misses[c] for c = 0..max_slots from stack distances."""
    hist = Counter()
    cold = 0
    for d in distances:
        if d == INF:
            cold += 1
        else:
            hist[int(d)] += 1
    total = len(distances)
    out = []
    hits = 0
    for c in range(max_slots + 1):
        if c > 0:
            hits += hist.get(c - 1, 0)
        out.append(total - hits)
    return out


def overlap_prev(tsets: List[Tuple[int, Tuple[int, ...]]]) -> float:
    prev: Optional[set] = None
    tot, n = 0.0, 0
    for _t, sel in tsets:
        s = set(sel)
        if prev is not None and s:
            tot += len(s & prev) / float(len(s))
            n += 1
        prev = s
    return tot / n if n else 0.0


def coactivation(tsets: List[Tuple[int, Tuple[int, ...]]], n_experts: int,
                 top: int = 10) -> List[Dict[str, float]]:
    """The most frequent expert pairs selected by the same token, with lift
    = P(a,b) / (P(a) P(b)); lift > 1 means they co-occur more than by chance."""
    pair: Counter = Counter()
    single: Counter = Counter()
    n_tok = 0
    for _t, sel in tsets:
        s = sorted(set(sel))
        n_tok += 1
        single.update(s)
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                pair[(s[i], s[j])] += 1
    out = []
    for (a, b), c in pair.most_common(top):
        pa, pb = single[a] / n_tok, single[b] / n_tok
        out.append({"a": a, "b": b, "count": c, "p_ab": c / n_tok,
                    "lift": (c / n_tok) / (pa * pb) if pa * pb > 0 else 0.0})
    return out


# ------------------------------------------------------------ summaries ---

def layer_summary(trace: Trace, layer: int, phase: Optional[str] = PHASE_DECODE,
                  taus: Sequence[int] = DEFAULT_TAUS,
                  qs: Sequence[float] = DEFAULT_QS) -> Dict[str, object]:
    n_exp = trace.meta.n_experts
    tsets = token_sets(trace, layer, phase)
    counter: Counter = Counter()
    for _t, sel in tsets:
        counter.update(sel)
    counts = full_counts(counter, n_exp)
    acc = sum(counts)
    H = shannon_entropy(counts)
    row: Dict[str, object] = {
        "layer": layer,
        "tokens": len(tsets),
        "accesses": acc,
        "distinct_experts": sum(1 for x in counts if x > 0),
        "entropy_bits": H,
        "entropy_norm": H / math.log2(n_exp) if n_exp > 1 else 0.0,
        "effective_experts": 2 ** H if acc else 0.0,
        "gini": gini(counts),
        "top1_coverage": top_coverage(counts, 1),
        "top5_coverage": top_coverage(counts, 5),
        "top10_coverage": top_coverage(counts, 10),
    }
    eb = trace.meta.expert_bytes.get(layer)
    for q in qs:
        k = ews(counts, q)
        tag = f"ews{int(round(q * 100))}"
        row[tag] = k
        row[tag + "_frac"] = k / n_exp if n_exp else 0.0
        if eb:
            row[tag + "_bytes"] = k * eb
    for tau in taus:
        row[f"ws_tau{tau}"] = working_set_mean(tsets, tau)
    rd, cold = reuse_distances(tsets)
    row["cold_selections"] = cold
    row["reuse_mean"] = sum(rd) / len(rd) if rd else 0.0
    row["reuse_median"] = _median(rd) if rd else 0.0
    for n in (1, 4, 16, 64):
        row[f"p_reuse_le{n}"] = sum(1 for d in rd if d <= n) / len(rd) if rd else 0.0
    row["overlap_prev"] = overlap_prev(tsets)
    half = len(tsets) // 2
    a: Counter = Counter()
    b: Counter = Counter()
    for i, (_t, sel) in enumerate(tsets):
        (a if i < half else b).update(sel)
    row["hot_set_stability"] = jaccard(ews_set(a, 0.95), ews_set(b, 0.95)) if half else 0.0
    return row


def _median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def reuse_cdf_rows(trace: Trace, phase: Optional[str] = PHASE_DECODE,
                   points: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
                   ) -> List[Dict[str, object]]:
    """P(reuse_distance <= N) per layer and for all layers pooled."""
    rows = []
    pooled: List[int] = []
    for l in trace.layers:
        rd, _ = reuse_distances(token_sets(trace, l, phase))
        pooled.extend(rd)
        for n in points:
            rows.append({"layer": l, "N": n,
                         "p_le": sum(1 for d in rd if d <= n) / len(rd) if rd else 0.0})
    for n in points:
        rows.append({"layer": "all", "N": n,
                     "p_le": sum(1 for d in pooled if d <= n) / len(pooled) if pooled else 0.0})
    return rows


def frequency_rows(trace: Trace, phase: Optional[str] = PHASE_DECODE) -> List[Dict[str, object]]:
    rows = []
    for l, c in layer_counts(trace, phase).items():
        tot = float(sum(c.values())) or 1.0
        for rank, (e, x) in enumerate(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])), 1):
            rows.append({"layer": l, "rank": rank, "expert": e, "count": x, "share": x / tot})
    return rows


def global_working_set(trace: Trace, phase: Optional[str] = PHASE_DECODE,
                       qs: Sequence[float] = DEFAULT_QS) -> Dict[str, object]:
    """EWS over all (layer, expert) keys together, in experts and in bytes.

    In bytes the keys are ordered by count, not by count per byte; with equal
    expert sizes in every layer the two orders coincide.
    """
    keys: Counter = Counter()
    for l, c in layer_counts(trace, phase).items():
        for e, x in c.items():
            keys[(l, e)] += x
    tot = float(sum(keys.values()))
    n_keys = len(trace.layers) * trace.meta.n_experts
    out: Dict[str, object] = {"keys_total": n_keys, "keys_used": len(keys),
                              "accesses": int(tot)}
    if tot <= 0:
        return out
    ordered = sorted(keys.items(), key=lambda kv: (-kv[1], kv[0]))
    for q in qs:
        acc, n, b = 0.0, 0, 0
        for (l, _e), x in ordered:
            acc += x
            n += 1
            b += trace.meta.expert_bytes.get(l, 0)
            if acc + 1e-9 >= q * tot:
                break
        tag = f"ews{int(round(q * 100))}"
        out[tag] = n
        out[tag + "_frac"] = n / n_keys if n_keys else 0.0
        if trace.meta.expert_bytes:
            out[tag + "_bytes"] = b
    total_bytes = sum(trace.meta.expert_bytes.get(l, 0) for l in trace.layers) * trace.meta.n_experts
    if total_bytes:
        out["expert_bytes_total"] = total_bytes
    return out


def segment_drift(trace: Trace, phase: Optional[str] = PHASE_DECODE) -> List[Dict[str, object]]:
    """Per workload segment and layer: entropy, EWS95, and the Jensen-Shannon
    divergence of the expert distribution from the previous segment."""
    segs = trace.segments()
    rows = []
    prev: Dict[int, Counter] = {}
    for s in segs:
        sub = trace.select(segment=s)
        cur = layer_counts(sub, phase)
        for l, c in cur.items():
            counts = full_counts(c, trace.meta.n_experts)
            rows.append({"segment": s, "layer": l, "accesses": sum(counts),
                         "entropy_bits": shannon_entropy(counts),
                         "ews95": ews(counts, 0.95),
                         "jsd_from_prev": js_divergence(dict(prev[l]), dict(c))
                         if l in prev else None})
        prev = cur
    return rows


def characterize(trace: Trace, phase: Optional[str] = PHASE_DECODE,
                 taus: Sequence[int] = DEFAULT_TAUS) -> Dict[str, object]:
    """Everything above for every layer, plus a global summary."""
    per_layer = [layer_summary(trace, l, phase, taus) for l in trace.layers]

    def spread(key: str) -> Dict[str, float]:
        xs = [float(r[key]) for r in per_layer if r.get(key) is not None]
        if not xs:
            return {}
        mean = sum(xs) / len(xs)
        sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
        return {"min": min(xs), "max": max(xs), "mean": mean, "sd": sd,
                "cv": sd / mean if mean else 0.0,
                "argmin": per_layer[xs.index(min(xs))]["layer"],
                "argmax": per_layer[xs.index(max(xs))]["layer"]}

    summary = {
        "phase": phase or "all",
        "tokens": trace.n_tokens(phase),
        "steps": len([s for s in trace.steps if phase is None or s.phase == phase]),
        "layers": len(trace.layers),
        "n_experts": trace.meta.n_experts,
        "top_k": trace.meta.top_k,
        "uniform_entropy_bits": math.log2(trace.meta.n_experts) if trace.meta.n_experts > 1 else 0,
        "layer_spread": {k: spread(k) for k in
                         ("entropy_norm", "gini", "ews90_frac", "ews95_frac", "ews99_frac",
                          "ws_tau16", "overlap_prev", "hot_set_stability", "reuse_median")},
        "global_working_set": global_working_set(trace, phase),
    }
    return {"summary": summary, "per_layer": per_layer}
