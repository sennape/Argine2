"""Exact asynchronous I/O timeline (section 9).

Input: the I records of a raw research log, one per storage read, each with a
unique request id, issue and completion timestamps, and the interval in which
the issuing thread was blocked waiting.

Q(t) = number of reads actually outstanding at time t = #(issue <= t < complete).
This is the queue depth seen by the software, reconstructed from per-request
timestamps — not the wait-derived proxy Argine printed before Argine2
("(ns_read/wall) x (reads/blocks)"), which is a proxy and is labelled so
wherever it appears.

Exactness: a completion flagged COMPLETE_EXACT was observed by a wait that
returned for that request; one flagged COMPLETE_BOUND was found already done
when polled, so its timestamp is an upper bound. Q(t) built on upper bounds
over-estimates the depth during those intervals. The share of exact
completions is reported with every result; Q(t) statistics are labelled
"upper bound" whenever that share is below 100%.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from .rawlog import IORecord


def _pct(sorted_xs: Sequence[float], q: float) -> float:
    if not sorted_xs:
        return 0.0
    i = min(len(sorted_xs) - 1, max(0, int(round(q * (len(sorted_xs) - 1)))))
    return float(sorted_xs[i])


def queue_depth(records: Sequence[IORecord]) -> Dict[str, object]:
    ev: List[Tuple[int, int]] = []
    for r in records:
        if r.complete_ns > r.issue_ns:
            ev.append((r.issue_ns, +1))
            ev.append((r.complete_ns, -1))
    if not ev:
        return {"reads": len(records), "note": "no reads with positive duration"}
    ev.sort(key=lambda x: (x[0], x[1]))  # completions before issues at equal time
    t_first, t_last = ev[0][0], ev[-1][0]
    q = 0
    qmax = 0
    t_prev = t_first
    area = 0
    busy = 0
    hist: Dict[int, int] = defaultdict(int)
    for t, d in ev:
        dt = t - t_prev
        if dt > 0:
            area += q * dt
            hist[q] += dt
            if q > 0:
                busy += dt
        q += d
        qmax = max(qmax, q)
        t_prev = t
    span = t_last - t_first
    exact = sum(1 for r in records if r.exact)
    status = queue_depth_status(records)
    return {
        "reads": len(records),
        "exact_completion_share": exact / len(records),
        # "exact" ONLY when every issue/completion interval is exact; otherwise
        # completions stamped as upper bounds lengthen intervals and every Q
        # statistic below is an upper bound of the true queue depth.
        "status": status,
        "label": ("exact" if status == "exact" else
                  "upper bound (some completions bounded)"),
        "span_ms": span / 1e6,
        "busy_ms": busy / 1e6,
        "q_time_weighted_mean_over_span": area / span if span else 0.0,
        "q_time_weighted_mean_when_busy": area / busy if busy else 0.0,
        "q_max": qmax,
        "q_time_share": {str(k): v / span for k, v in sorted(hist.items())} if span else {},
    }


def queue_depth_status(records: Sequence[IORecord]) -> str:
    """'exact' iff every read has an exact completion (issue times are always
    taken immediately before the read is issued); otherwise 'upper_bound'."""
    if not records:
        return "no_data"
    return "exact" if all(r.exact for r in records) else "upper_bound"


def latency(records: Sequence[IORecord]) -> Dict[str, object]:
    lat = sorted(r.latency_ns / 1e6 for r in records)
    ex = sorted(r.latency_ns / 1e6 for r in records if r.exact)
    out: Dict[str, object] = {"reads": len(lat)}
    for name, xs in (("all", lat), ("exact_only", ex)):
        if xs:
            out[name] = {"mean_ms": sum(xs) / len(xs), "p50_ms": _pct(xs, .5),
                         "p90_ms": _pct(xs, .9), "p99_ms": _pct(xs, .99), "max_ms": xs[-1]}
    return out


def overlap(records: Sequence[IORecord]) -> Dict[str, object]:
    """Hidden vs unhidden I/O.

    hidden   = part of a read's [issue, complete] before its thread began waiting
               (the thread was issuing or serving other slices meanwhile);
    unhidden = the rest: time during which the thread was blocked on it.
    In Argine's runtime the barrier after the reads precedes the matrix
    multiplication, so "hidden" means hidden behind other residency work of
    the same node, never behind compute.
    """
    tot = sum(r.latency_ns for r in records)
    hid = sum(r.hidden_ns for r in records)
    sync = sum(1 for r in records if r.synchronous)
    return {"io_ms": tot / 1e6, "hidden_ms": hid / 1e6, "unhidden_ms": (tot - hid) / 1e6,
            "hidden_fraction": hid / tot if tot else 0.0,
            "unhidden_fraction": (tot - hid) / tot if tot else 0.0,
            "synchronous_reads": sync}


def blocked_time(records: Sequence[IORecord]) -> Dict[str, object]:
    """Per thread, the union of its wait intervals (time blocked on reads)."""
    per: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for r in records:
        if r.wait_end_ns > r.wait_start_ns:
            per[r.tid].append((r.wait_start_ns, r.wait_end_ns))
    out = {}
    total = 0
    for tid, iv in per.items():
        iv = sorted(set(iv))
        cur_s, cur_e = iv[0]
        acc = 0
        for s, e in iv[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                acc += cur_e - cur_s
                cur_s, cur_e = s, e
        acc += cur_e - cur_s
        out[str(tid)] = acc / 1e6
        total += acc
    return {"per_thread_ms": out, "total_ms": total / 1e6}


def bandwidth(records: Sequence[IORecord]) -> Dict[str, object]:
    qd = queue_depth(records)
    by = sum(r.bytes for r in records)
    busy = float(qd.get("busy_ms") or 0.0)
    return {"bytes": by, "mb_s_when_busy": (by / 1e6) / (busy / 1e3) if busy else 0.0}


def analyse(records: Sequence[IORecord]) -> Dict[str, object]:
    ok = [r for r in records if r.ok]
    return {"queue_depth": queue_depth(ok), "latency": latency(ok), "overlap": overlap(ok),
            "blocked": blocked_time(ok), "bandwidth": bandwidth(ok),
            "failed_reads": len(records) - len(ok)}


def queue_series(records: Sequence[IORecord], bucket_ms: float = 1.0) -> List[Dict[str, float]]:
    """Q(t) averaged per time bucket, for plotting."""
    ev = sorted([(r.issue_ns, 1) for r in records] + [(r.complete_ns, -1) for r in records],
                key=lambda x: (x[0], x[1]))
    if not ev:
        return []
    b = int(bucket_ms * 1e6)
    t0 = ev[0][0]
    out = []
    q = 0
    t_prev = t0
    cur_b = 0
    area = 0
    for t, d in ev:
        while t >= t0 + (cur_b + 1) * b:
            edge = t0 + (cur_b + 1) * b
            area += q * (edge - t_prev)
            out.append({"t_ms": cur_b * bucket_ms, "q_mean": area / b})
            area = 0
            t_prev = edge
            cur_b += 1
        area += q * (t - t_prev)
        t_prev = t
        q += d
    return out
