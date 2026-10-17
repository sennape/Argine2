"""Analysis pipeline: one trace in, every Phase-1/Phase-2 table out.

    characterize  -> characterization.json, per_layer.csv, reuse_cdf.csv,
                     expert_frequency.csv, coactivation.csv, drift.csv
    elasticity    -> lru_curves.csv, elasticity.csv, heterogeneity.json
    compare       -> policy_comparison.csv (+ oracle gaps), per_layer_misses.csv
    adaptive      -> adaptive.csv, allocation_timeline.csv, adaptation.csv

Every table is written as CSV/JSON next to the figures made from it. All
numbers are deterministic functions of the trace and the cost model, except
the columns whose name ends in `_wall_s` / `controller_ms` (timings of the
replay itself), which are written to `overhead.csv` only.

Train / evaluation separation (section 25): policies that learn offline
(static frequency, static hot set, layer-aware static allocation) are fitted on
`train`, evaluated on `trace`. When no separate train trace is given they are
fitted on the evaluation trace itself and every row says `train=in-sample`:
such rows are upper bounds for those policies, not results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import allocator, elasticity, metrics
from .controller import AdaptiveResidencyController
from .costmodel import CostModel
from .io import write_csv, write_json
from .oracle import BeladyGlobal, BeladyPerLayer
from .policies import (GlobalLRU, LeastStale, LFU, PerLayerLRU, ShardedLRU, StaticPinned,
                       TwoQ, counts_from_trace)
from .replay import adaptation_metrics, replay
from .trace import PHASE_DECODE, Trace

GiB = 1 << 30
MiB = 1 << 20

Factory = Callable[[int], object]


def fmt_budget(b: int) -> str:
    return f"{b / GiB:g}G" if b >= GiB else f"{b / MiB:g}M"


def per_layer_floor(trace: Trace, phase: Optional[str] = PHASE_DECODE) -> int:
    """Smallest budget at which every layer can hold its own largest event:
    the floor of any strictly layer-partitioned design."""
    k = elasticity.min_slots(trace, phase)
    return sum(k * trace.meta.bytes_of(l) for l in trace.layers)


# ------------------------------------------------------------ characterize

def characterize_to(trace: Trace, out: Path, phase: Optional[str] = PHASE_DECODE) -> Dict:
    out.mkdir(parents=True, exist_ok=True)
    ch = metrics.characterize(trace, phase)
    ch["summary"]["per_layer_partition_floor_bytes"] = per_layer_floor(trace, phase)
    write_json({"meta": trace.meta.to_json(), "summary": ch["summary"]},
               out / "characterization.json")
    write_csv(ch["per_layer"], out / "per_layer.csv")
    write_csv(metrics.reuse_cdf_rows(trace, phase), out / "reuse_cdf.csv")
    write_csv(metrics.frequency_rows(trace, phase), out / "expert_frequency.csv")
    co = []
    for l in trace.layers:
        for r in metrics.coactivation(metrics.token_sets(trace, l, phase),
                                      trace.meta.n_experts, top=5):
            co.append({"layer": l, **r})
    write_csv(co, out / "coactivation.csv")
    if trace.segments():
        write_csv(metrics.segment_drift(trace, phase), out / "drift.csv")
    return ch


# -------------------------------------------------------------- elasticity

def elasticity_to(trace: Trace, budgets: Sequence[int], out: Path, cost: CostModel,
                  phase: Optional[str] = PHASE_DECODE, opt_grid_points: int = 0) -> Dict:
    out.mkdir(parents=True, exist_ok=True)
    curves = elasticity.lru_curves(trace, phase)
    rows = []
    for l, cur in curves.items():
        for c, m in enumerate(cur):
            rows.append({"layer": l, "slots": c, "misses_lru": m})
    if opt_grid_points:
        n = trace.meta.n_experts
        k = elasticity.min_slots(trace, phase)
        grid = sorted({k} | {max(k, round(k + i * (n - k) / (opt_grid_points - 1)))
                             for i in range(opt_grid_points)})
        opt = elasticity.opt_curves(trace, grid, phase)
        idx = {(r["layer"], r["slots"]): r for r in rows}
        for l, pts in opt.items():
            for c, m in pts.items():
                idx[(l, c)]["misses_opt"] = m
    write_csv(rows, out / "lru_curves.csv")
    # misses per node at the middle budget, for the per-miss cost
    mpn = _misses_per_node(trace, budgets[len(budgets) // 2] if budgets else GiB, phase)
    el = elasticity.elasticity_rows(trace, curves, cost, mpn, phase)
    write_csv(el, out / "elasticity.csv")
    het = []
    floor = per_layer_floor(trace, phase)
    for b in budgets:
        if b < floor:
            het.append({"budget_bytes": b, "infeasible": True,
                        "reason": f"below the per-layer partition floor ({floor} bytes)"})
            continue
        het.append(elasticity.heterogeneity(trace, curves, b, phase))
    write_json({"misses_per_node_for_cost": mpn, "budgets": het,
                "per_layer_partition_floor_bytes": floor}, out / "heterogeneity.json")
    return {"curves": curves, "heterogeneity": het}


def _misses_per_node(trace: Trace, budget: int, phase: Optional[str]) -> float:
    r = replay(trace, GlobalLRU(), budget, phase=phase)
    return r.misses / r.events if r.events else 0.0


# ---------------------------------------------------------------- policies

IN_SAMPLE_LABEL = "IN-SAMPLE / UPPER BOUND"
FITTED = {"static-frequency", "static-hot-per-layer", "layer-aware-static"}
ORACLES = {"belady-global", "oracle-static-lru", "belady-per-layer-opt"}


def category_of(name: str) -> str:
    """runtime-reproduction | counterfactual | oracle | measured (never mixed)."""
    if name in ORACLES:
        return "oracle"
    if name.startswith("store-replay"):
        return "runtime-reproduction"
    if name.startswith("runtime"):
        return "measured"
    return "counterfactual"


def policy_factories(trace: Trace, train: Optional[Trace], cost: CostModel,
                     phase: Optional[str] = PHASE_DECODE, include_oracle: bool = True,
                     include_opt_static: bool = False) -> List[Tuple[str, Factory, str]]:
    """(name, factory(budget) -> Policy, train_source).

    `sharded-lru-16-expert` was called `sharded-lru-16` before 2026-10-03. It
    shards whole EXPERTS with a made-up hash and is event-atomic, so it is a
    counterfactual approximation of the store, not a reproduction of it (the
    reproduction is storereplay.StoreReplay; see REPLAY-FIDELITY-VALIDATION.md).
    """
    in_sample = train is None
    tr = train or trace
    src = IN_SAMPLE_LABEL if in_sample else "separate"
    counts = counts_from_trace(tr, phase)
    train_curves = elasticity.lru_curves(tr, phase)
    eval_curves = elasticity.lru_curves(trace, phase)
    sizes = {l: trace.meta.bytes_of(l) for l in trace.layers}
    k = elasticity.min_slots(trace, phase)

    def layer_aware(curves):
        def make(b):
            uni = allocator.uniform(trace.layers, sizes, b, trace.meta.n_experts, k)
            slots = allocator.dp_optimal_checked(curves, sizes, b, k, {"uniform": uni})
            return PerLayerLRU(allocator.to_bytes(slots, sizes))
        return make

    def named(p, name):
        p.name = name
        return p

    fs: List[Tuple[str, Factory, str]] = [
        ("global-lru", lambda b: GlobalLRU(), "-"),
        ("sharded-lru-16-expert", lambda b: named(ShardedLRU(16), "sharded-lru-16-expert"), "-"),
        ("per-layer-lru-uniform", lambda b: named(PerLayerLRU(), "per-layer-lru-uniform"), "-"),
        ("lfu", lambda b: LFU(), "-"),
        ("2q", lambda b: TwoQ(), "-"),
        ("least-stale-lru", lambda b: LeastStale("lru"), "-"),
        ("least-stale-mru", lambda b: LeastStale("mru"), "-"),
        ("static-frequency", lambda b: StaticPinned(counts, 1.0, in_sample=in_sample), src),
        ("static-hot-per-layer",
         lambda b: StaticPinned(counts, 1.0, per_layer=True, in_sample=in_sample), src),
        ("layer-aware-static",
         lambda b: named(layer_aware(train_curves)(b), "layer-aware-static"), src),
        ("adaptive-e64", lambda b: AdaptiveResidencyController(epoch=64, cost=cost), "online"),
    ]
    if include_oracle:
        fs.append(("belady-global", lambda b: BeladyGlobal(trace, phase), "oracle"))
        fs.append(("oracle-static-lru",
                   lambda b: named(layer_aware(eval_curves)(b), "oracle-static-lru"), "oracle"))
    if include_opt_static:
        n = trace.meta.n_experts
        grid = sorted({k, n} | {max(k, int(k + i * (n - k) / 15)) for i in range(16)})
        opt = {l: elasticity.densify(v, n) for l, v in elasticity.opt_curves(trace, grid, phase).items()}

        def make_bpl(b):
            uni = allocator.uniform(trace.layers, sizes, b, trace.meta.n_experts, k)
            slots = allocator.dp_optimal_checked(opt, sizes, b, k, {"uniform": uni})
            return BeladyPerLayer(trace, allocator.to_bytes(slots, sizes), phase,
                                  label="belady-per-layer-opt")
        fs.append(("belady-per-layer-opt", make_bpl, "oracle"))
    return fs


PARTITIONED = {"per-layer-lru-uniform", "static-hot-per-layer", "layer-aware-static",
               "adaptive-e64", "oracle-static-lru", "belady-per-layer-opt"}


def store_replay_row(trace: Trace, budget: int, slice_bytes: Dict, phase: Optional[str],
                     shard_seeds: Sequence[int] = (0, 1, 2, 3, 4), n_shards: int = 0,
                     shard_map=None, split_pools: bool = False) -> Dict:
    """Runtime-reproduction row: StoreReplay on the trace's request stream.
    Metric = expert-event hit (all 3 slices of an expert hit in the gate node),
    the definition that matches the expert-unit counterfactual policies."""
    from . import storereplay as S
    n = n_shards or S.runtime_shard_count(budget)
    maps = [shard_map] if shard_map is not None else [S.HashShardMap(n, slice_bytes, sd)
                                                      for sd in shard_seeds]
    vals = []
    for mp in maps:
        m, rp = S.run(S.synthetic_stream(trace, split_pools=split_pools), S.StoreReplay(budget, mp,
                      lambda k: slice_bytes[(k[0], k[2])], n))
        r = m.rates()
        ph = phase or "all"
        vals.append((r[f"expert_event_{ph}"], r[f"slice_gate_{ph}"], r[f"expert_event_{ph}_n"],
                     r[f"partial_expert_events_{ph}"], rp.evictions, rp.failures, rp.violations))
    k = len(vals)
    mean = lambda i: sum(v[i] for v in vals) / k  # noqa: E731
    sd = (sum((v[0] - mean(0)) ** 2 for v in vals) / (k - 1)) ** 0.5 if k > 1 else 0.0
    ev = int(vals[0][2])
    return {"policy": "store-replay", "category": "runtime-reproduction",
            "budget_bytes": budget, "phase": phase or "all", "accesses": ev,
            "hits": round(mean(0) * ev), "misses": ev - round(mean(0) * ev),
            "hit_rate": round(mean(0), 6), "hit_rate_sd_over_shard_maps": round(sd, 6),
            "slice_gate_hit_rate": round(mean(1), 6), "partial_expert_events": round(mean(3)),
            "evictions_slices": round(mean(4)), "overflow_events": round(mean(5)),
            "budget_violations": max(v[6] for v in vals),
            "metric": "expert-event hit (3 slices) — slice-level store model",
            "shard_map": maps[0].label if k == 1 else f"hash-model x{k} seeds (statistical)",
            "pin_model": "per-pool (threads != threads-batch)" if split_pools else "single pool"}


def compare_to(trace: Trace, budgets: Sequence[int], out: Path, cost: CostModel,
               train: Optional[Trace] = None, phase: Optional[str] = PHASE_DECODE,
               include_opt_static: bool = False, slice_bytes: Optional[Dict] = None,
               runtime_rows: Optional[List[Dict]] = None,
               split_pools: bool = False) -> List[Dict]:
    out.mkdir(parents=True, exist_ok=True)
    floor = per_layer_floor(trace, phase)
    fs = policy_factories(trace, train, cost, phase, include_oracle=True,
                          include_opt_static=include_opt_static)
    rows, overhead, per_layer = [], [], []
    for b in budgets:
        bel = None
        block = []
        for name, make, src in fs:
            if name in PARTITIONED and b < floor:
                block.append({"policy": name, "budget_bytes": b, "budget": fmt_budget(b),
                              "train": src, "feasible": False, "category": category_of(name),
                              "note": f"below per-layer partition floor {fmt_budget(floor)}"})
                continue
            res = replay(trace, make(b), b, phase=phase, cost=cost)
            row = res.row()
            row.update({"budget": fmt_budget(b), "train": src, "feasible": True,
                        "category": category_of(name), "in_sample": src == IN_SAMPLE_LABEL,
                        "metric": "expert-event hit (expert = unit)"})
            overhead.append({"policy": name, "budget": fmt_budget(b),
                             "replay_wall_s": row.pop("replay_wall_s"),
                             "controller_ms": row.pop("controller_ms")})
            block.append(row)
            if name == "belady-global":
                bel = res
            if name in ("global-lru", "per-layer-lru-uniform", "layer-aware-static",
                        "adaptive-e64", "belady-global"):
                for l, d in res.per_layer.items():
                    per_layer.append({"policy": name, "budget": fmt_budget(b), "layer": l, **d})
        if slice_bytes:
            row = store_replay_row(trace, b, slice_bytes, phase, split_pools=split_pools)
            row.update({"budget": fmt_budget(b), "train": "-", "feasible": True})
            block.insert(0, row)
        for rr in runtime_rows or []:
            if int(rr["budget_bytes"]) == b:
                block.insert(0, {**rr, "budget": fmt_budget(b), "train": "-", "feasible": True,
                                 "category": "measured"})
        for row in block:
            if row.get("feasible") and bel is not None and row.get("category") == "counterfactual":
                row["oracle_gap_misses"] = ((row["misses"] - bel.misses) / bel.misses
                                            if bel.misses else None)
                lru = next((r for r in block if r["policy"] == "global-lru"), None)
                if lru and lru.get("misses") is not None and lru["misses"] > bel.misses:
                    # share of the LRU -> oracle gap that this policy closes
                    row["gap_closed_vs_lru"] = ((lru["misses"] - row["misses"]) /
                                                (lru["misses"] - bel.misses))
        rows.extend(block)
    write_csv(rows, out / "policy_comparison.csv")
    write_csv(overhead, out / "overhead.csv")
    write_csv(per_layer, out / "per_layer_misses.csv")
    return rows


# ---------------------------------------------------------------- adaptive

def adaptive_to(trace: Trace, budgets: Sequence[int], out: Path, cost: CostModel,
                phase: Optional[str] = PHASE_DECODE, epochs: Sequence[int] = (32, 64, 128),
                pinned: Sequence[float] = (0.0, 0.25, 0.5), ablate: bool = True,
                timeline_budget: Optional[int] = None) -> List[Dict]:
    out.mkdir(parents=True, exist_ok=True)
    floor = per_layer_floor(trace, phase)
    variants: List[Tuple[str, Callable[[], object]]] = [
        ("per-layer-lru-uniform", lambda: PerLayerLRU(label="per-layer-lru-uniform")),
        ("global-lru", lambda: GlobalLRU()),
    ]
    for e in epochs:
        variants.append((f"adaptive-e{e}", lambda e=e: AdaptiveResidencyController(epoch=e, cost=cost)))
    for pf in pinned:
        if pf:
            variants.append((f"adaptive-e64-pin{int(pf * 100)}",
                             lambda pf=pf: AdaptiveResidencyController(epoch=64, pinned_fraction=pf,
                                                                       cost=cost)))
    if ablate:
        variants += [
            ("ablate-frozen", lambda: AdaptiveResidencyController(epoch=64, adaptive=False, cost=cost)),
            ("ablate-freq-estimator", lambda: AdaptiveResidencyController(epoch=64, estimator="freq",
                                                                          cost=cost)),
            ("ablate-no-cost-weighting", lambda: AdaptiveResidencyController(
                epoch=64, cost_weighting=False, cost=cost)),
            ("ablate-no-hysteresis", lambda: AdaptiveResidencyController(epoch=64, hysteresis=0.0,
                                                                         cost=cost)),
        ]
    rows, timeline, adapt, overhead = [], [], [], []
    tb = timeline_budget or (budgets[len(budgets) // 2] if budgets else None)
    boundaries = _segment_boundaries(trace, phase)
    for b in budgets:
        for name, make in variants:
            if name != "global-lru" and b < floor:
                rows.append({"variant": name, "budget": fmt_budget(b), "feasible": False})
                continue
            pol = make()
            res = replay(trace, pol, b, phase=phase, cost=cost,
                         timeline_every=16 if (b == tb and name == "adaptive-e64") else 0)
            row = res.row()
            overhead.append({"variant": name, "budget": fmt_budget(b),
                             "replay_wall_s": row.pop("replay_wall_s"),
                             "controller_ms": row.pop("controller_ms"),
                             "controller_ms_per_token": 1000.0 * res.controller_s / max(1, res.tokens)})
            row.update({"variant": name, "budget": fmt_budget(b), "feasible": True,
                        "moves": getattr(pol, "moves", None),
                        "bytes_moved": getattr(pol, "bytes_moved", None)})
            rows.append(row)
            for snap in res.allocation_timeline:
                for l, by in snap["allocation"].items():
                    timeline.append({"step": snap["step"], "layer": l,
                                     "slots": by // trace.meta.bytes_of(l)})
            if boundaries:
                for a in adaptation_metrics(res.per_step_misses, boundaries):
                    adapt.append({"variant": name, "budget": fmt_budget(b), **a})
    write_csv(rows, out / "adaptive.csv")
    write_csv(overhead, out / "adaptive_overhead.csv")
    if timeline:
        write_csv(timeline, out / "allocation_timeline.csv")
    if adapt:
        write_csv(adapt, out / "adaptation.csv")
    return rows


def _segment_boundaries(trace: Trace, phase: Optional[str]) -> List[int]:
    """Indices into replay's per-step series (COUNTED steps only)."""
    from .replay import counted_step
    idx = []
    prev = None
    k = 0
    for st in trace.steps:
        if not counted_step(st, phase):
            continue
        if st.segment != prev and prev is not None:
            idx.append(k)
        prev = st.segment
        k += 1
    return idx


def run_all(trace: Trace, out: Path, budgets: Sequence[int], cost: CostModel,
            train: Optional[Trace] = None, phase: Optional[str] = PHASE_DECODE,
            opt_grid_points: int = 9, adaptive: bool = True,
            slice_bytes: Optional[Dict] = None, runtime_rows: Optional[List[Dict]] = None,
            split_pools: bool = False) -> Dict:
    out.mkdir(parents=True, exist_ok=True)
    write_json({"cost_model": cost.to_json(), "phase": phase or "all",
                "replay_semantics": "v2 (all steps served; metric = phase, workload-only)",
                "allocator": "exact (min-plus per size class), invariant optimal <= uniform",
                "store_replay_pin_model": ("per-pool (threads != threads-batch)" if split_pools
                                           else "single pool"),
                "budgets_bytes": list(budgets),
                "train": "separate" if train is not None else "in-sample",
                "trace_meta": trace.meta.to_json(),
                "train_meta": train.meta.to_json() if train is not None else None},
               out / "analysis_manifest.json")
    characterize_to(trace, out, phase)
    elasticity_to(trace, budgets, out, cost, phase, opt_grid_points)
    compare_to(trace, budgets, out, cost, train, phase, slice_bytes=slice_bytes,
               runtime_rows=runtime_rows, split_pools=split_pools)
    if adaptive:
        adaptive_to(trace, budgets, out, cost, phase)
    return {"out": str(out)}
