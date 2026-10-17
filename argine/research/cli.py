"""`argine research ...` — the research command family of Argine2.

    argine research trace MODEL --expert-store --budgets 4G,2G --workload coding
    argine research convert route.log --to trace.jsonl.gz --n-experts 128
    argine research characterize TRACE --out DIR
    argine research replay TRACE --policy global-lru --budget 2G
    argine research compare TRACE --budgets 4G,3G,2G,1.5G --train DEV_TRACE --out DIR
    argine research elasticity TRACE --budgets 4G,2G --out DIR
    argine research oracle TRACE --budgets 4G,2G --out DIR
    argine research adaptive TRACE --budgets 2G --out DIR
    argine research analyze TRACE --budgets ... --out DIR   (all of the above)
    argine research iotimeline RUN_DIR
    argine research predict RUNS.csv
    argine research report DIR
    argine research fingerprint --model M --binary B
    argine research schedule --configs A,B,C --repeat 3

Every command writes files (CSV / JSON / JSONL); the screen output is a
summary of what was written, never the only copy of a result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from . import analysis, fingerprint, history, report, stats, workloads
from .costmodel import CostModel
from .io import read_csv, write_csv, write_json
from .trace import PHASE_DECODE, PHASE_PREFILL, Trace

POLICIES = ("global-lru", "sharded-lru-16-expert", "per-layer-lru-uniform", "lfu", "2q",
            "least-stale-lru", "least-stale-mru", "static-frequency", "static-hot-per-layer",
            "layer-aware-static", "adaptive-e64", "belady-global", "oracle-static-lru")


def _size(s: str) -> int:
    from ..util import parse_size
    v = parse_size(s)
    if not v:
        raise argparse.ArgumentTypeError(f"not a size: {s}")
    return int(v)


def _sizes(s: str) -> List[int]:
    return [_size(x) for x in s.split(",") if x.strip()]


def _phase(s: str) -> Optional[str]:
    return None if s == "all" else s


def _load(path: str, args) -> Trace:
    tr = Trace.load(path)
    if getattr(args, "expert_bytes", None):
        for l in tr.layers:
            tr.meta.expert_bytes[l] = int(args.expert_bytes)
    if not tr.meta.expert_bytes:
        print("warning: the trace has no expert sizes: budgets are interpreted as numbers of "
              "experts. Pass --expert-bytes N or convert with --gguf.", file=sys.stderr)
    return tr


def _cost(args) -> CostModel:
    return CostModel.load(getattr(args, "cost", None))


def _common_trace_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("trace", help="router trace (.jsonl or .jsonl.gz)")
    p.add_argument("--phase", default="decode", choices=["decode", "prefill", "all"],
                   help="which steps to analyse (default decode: the generation regime)")
    p.add_argument("--expert-bytes", type=int, default=None, metavar="N",
                   help="override bytes per resident expert for every layer")
    p.add_argument("--cost", default=None, metavar="JSON",
                   help="cost model parameters (default: measured diskbench curve, see "
                        "argine/research/costmodel.py)")


# ------------------------------------------------------------- commands ---

def cmd_convert(args) -> int:
    from . import rawlog, trace as T
    src = Path(args.source)
    if args.kind == "auto":
        with src.open("r", encoding="utf-8", errors="replace") as fh:
            head = fh.readline()
        kind = "raw" if head.startswith("H\targine-raw") else "route-log"
    else:
        kind = args.kind
    if kind == "route-log":
        tr, warns = T.from_route_log(src, model=args.model_name or "", n_experts=args.n_experts or 0,
                                     provenance=args.provenance or "")
    else:
        log = rawlog.parse(src)
        markers = rawlog.read_markers(args.markers) if args.markers else []
        tr = rawlog.to_trace(log, markers=markers)
        warns = [f"{log.malformed} malformed records"] if log.malformed else []
        if log.truncated_tail:
            warns.append("last line truncated (process terminated mid-write): dropped")
    if args.gguf:
        from ..gguf import read_gguf
        from .runner import expert_bytes_from_model
        m = read_gguf(args.gguf)
        tr.meta.expert_bytes = expert_bytes_from_model(m)
        tr.meta.model = tr.meta.model or Path(args.gguf).name
        if not tr.meta.n_experts:
            tr.meta.n_experts = int(m.expert_count or 0)
    elif args.expert_bytes:
        tr.meta.expert_bytes = {l: int(args.expert_bytes) for l in tr.layers}
    if args.request is not None:
        reqs = T.split_requests(tr)
        a, b = reqs[args.request]
        tr = tr.select(graphs=(a, b))
        tr.meta.notes.append(f"request {args.request} of {len(reqs)} (steps {a}..{b - 1})")
    if args.split:
        tr.meta.split = args.split
    for n in args.note or []:
        tr.meta.notes.append(n)
    tr.save(args.to)
    for w in warns:
        print("warning:", w, file=sys.stderr)
    print(f"wrote {args.to}: {len(tr.steps)} steps, {tr.n_tokens('decode')} decode tokens, "
          f"{tr.n_tokens('prefill')} prompt tokens, {len(tr.layers)} layers")
    return 0


def cmd_characterize(args) -> int:
    tr = _load(args.trace, args)
    ch = analysis.characterize_to(tr, Path(args.out), _phase(args.phase))
    s = ch["summary"]
    print(json.dumps({k: s[k] for k in ("tokens", "layers", "n_experts", "top_k")}))
    print(f"wrote {args.out}/characterization.json, per_layer.csv, reuse_cdf.csv, ...")
    return 0


def _factory(name: str, tr: Trace, train: Optional[Trace], cost: CostModel, phase):
    for n, make, src in analysis.policy_factories(tr, train, cost, phase):
        if n == name:
            return make, src
    raise SystemExit(f"unknown policy {name} (known: {', '.join(POLICIES)})")


def cmd_replay(args) -> int:
    from .replay import replay
    tr = _load(args.trace, args)
    train = _load(args.train, args) if args.train else None
    make, src = _factory(args.policy, tr, train, _cost(args), _phase(args.phase))
    res = replay(tr, make(args.budget), args.budget, phase=_phase(args.phase), cost=_cost(args))
    row = res.row()
    row["train"] = src
    print(json.dumps(row, indent=2))
    if args.out:
        write_json({"result": row, "per_layer": res.per_layer}, args.out)
    return 0


def cmd_compare(args) -> int:
    tr = _load(args.trace, args)
    train = _load(args.train, args) if args.train else None
    rows = analysis.compare_to(tr, args.budgets, Path(args.out), _cost(args), train,
                               _phase(args.phase), include_opt_static=args.opt_static)
    _write_command_manifest(Path(args.out) / "compare_manifest.json", "compare", args, tr, train,
                            {"train_mode": "separate" if train is not None else "in-sample",
                             "opt_static": bool(args.opt_static)})
    print(f"wrote {args.out}/policy_comparison.csv ({len(rows)} rows)")
    return 0


def cmd_elasticity(args) -> int:
    tr = _load(args.trace, args)
    analysis.elasticity_to(tr, args.budgets, Path(args.out), _cost(args), _phase(args.phase),
                           args.opt_grid)
    print(f"wrote {args.out}/lru_curves.csv, elasticity.csv, heterogeneity.json")
    return 0


def cmd_oracle(args) -> int:
    tr = _load(args.trace, args)
    from .oracle import BeladyGlobal
    from .replay import replay
    rows = []
    for b in args.budgets:
        r = replay(tr, BeladyGlobal(tr, _phase(args.phase)), b, phase=_phase(args.phase),
                   cost=_cost(args)).row()
        r.pop("replay_wall_s", None)
        r.pop("controller_ms", None)
        rows.append(r)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    write_csv(rows, Path(args.out) / "oracle.csv")
    analysis.elasticity_to(tr, args.budgets, Path(args.out), _cost(args), _phase(args.phase),
                           args.opt_grid)
    print(f"wrote {args.out}/oracle.csv and the in-sample optimal static allocations")
    return 0


def cmd_adaptive(args) -> int:
    tr = _load(args.trace, args)
    epochs = [int(x) for x in args.epochs.split(",")]
    pinned = [float(x) for x in args.pinned.split(",")]
    rows = analysis.adaptive_to(tr, args.budgets, Path(args.out), _cost(args), _phase(args.phase),
                                epochs, pinned, ablate=not args.no_ablate)
    _write_command_manifest(Path(args.out) / "adaptive_manifest.json", "adaptive", args, tr, None,
                            {"epochs": epochs, "pinned": pinned, "ablate": not args.no_ablate,
                             "variants": [r.get("variant") for r in rows]})
    print(f"wrote {args.out}/adaptive.csv ({len(rows)} rows)")
    return 0


def _sha256_file(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _trace_identity(path: str, tr: Trace) -> dict:
    m = tr.meta
    return {"path": str(path), "sha256": _sha256_file(path),
            "meta": {"model": m.model, "split": m.split, "workload": m.workload,
                     "n_experts": m.n_experts, "top_k": m.top_k, "n_layers": len(m.layers)}}


def _write_command_manifest(path: Path, command: str, args, tr: Trace, train: Optional[Trace],
                            extra: dict) -> None:
    """Provenance of a `compare` / `adaptive` output directory (added 2026-10-11 for EXP-010):
    which trace was replayed and, for `compare --train`, which trace fitted the offline
    policies. Results are unchanged; this file only records inputs."""
    import datetime
    man = {"command": command, "argv": sys.argv[1:],
           "trace": _trace_identity(args.trace, tr),
           "train": _trace_identity(args.train, train) if train is not None else None,
           "budgets_bytes": list(args.budgets), "phase": args.phase,
           "cost": getattr(args, "cost", None),
           "written_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
               "%Y-%m-%dT%H:%M:%SZ")}
    man.update(extra)
    write_json(man, path)


def measured_runtime_rows(runs_dir: str, phase: Optional[str]) -> List[dict]:
    """One measured row per budget (mean over repetitions) with the metric that
    matches the expert-unit table: expert-event hit, phase, workload-only."""
    from . import fidelity as F
    from .storereplay import Metrics
    import statistics
    per: dict = {}
    for run in sorted(Path(runs_dir).iterdir()):
        if not (run / "raw.tsv").exists():
            continue
        st = F.load_runtime_stream(run)
        m = Metrics()
        for r in st.requests:
            m.add(r, r.outcome in ("h", "w"))
        rt = m.rates()
        ph = phase or "all"
        per.setdefault(st.budget, []).append((run.name, rt[f"expert_event_{ph}"],
                                              rt[f"slice_gate_{ph}"], rt[f"expert_event_{ph}_n"]))
    rows = []
    for b, v in sorted(per.items()):
        hr = [x[1] for x in v]
        rows.append({"policy": "runtime (measured)", "budget_bytes": b, "phase": phase or "all",
                     "accesses": v[0][3], "hit_rate": round(statistics.mean(hr), 6),
                     "hit_rate_sd_over_runs": round(statistics.stdev(hr), 6) if len(hr) > 1 else 0.0,
                     "slice_gate_hit_rate": round(statistics.mean(x[2] for x in v), 6),
                     "runs": " ".join(x[0] for x in v),
                     "metric": "expert-event hit (3 slices), measured by the runtime trace"})
    return rows


def cmd_analyze(args) -> int:
    tr = _load(args.trace, args)
    train = _load(args.train, args) if args.train else None
    slice_bytes = None
    split_pools = False
    if args.slice_bytes_from:
        from . import fidelity as F
        slice_bytes = F.load_runtime_stream(Path(args.slice_bytes_from)).sizes
        split_pools = F.thread_pool_config(Path(args.slice_bytes_from)).get("split", False)
    runtime_rows = measured_runtime_rows(args.runtime_runs, _phase(args.phase)) \
        if args.runtime_runs else None
    analysis.run_all(tr, Path(args.out), args.budgets, _cost(args), train, _phase(args.phase),
                     args.opt_grid, adaptive=not args.no_adaptive, slice_bytes=slice_bytes,
                     runtime_rows=runtime_rows, split_pools=split_pools)
    p = report.build(args.out, title=args.title or "Argine2 analysis")
    print(f"wrote {args.out} (report: {p})")
    return 0


def cmd_reprocess(args) -> int:
    """Re-derive trace, tables and summary.json of a run from its raw log.
    The run's manifest.json is a record of the run and is not modified."""
    from .runner import process_run
    from .trace import TraceMeta
    d = Path(args.run)
    meta = TraceMeta(model=args.model_name or "", n_experts=args.n_experts or 0,
                     provenance="argine research reprocess of a recorded run")
    old = d / "trace.jsonl.gz"
    if old.exists():  # keep model metadata (expert sizes, top-k) of the original processing
        meta = Trace.load(old).meta
        meta.notes.append("reprocessed by argine research reprocess")
    s = process_run(d, meta)
    q = s["io"]["queue_depth"]
    print(json.dumps({"trace_complete": s["trace_completeness"]["complete"],
                      "queue_depth_status": q.get("status"),
                      "exact_completion_share": q.get("exact_completion_share"),
                      "runtime_confirmation": s["runtime_confirmation"],
                      "expert_level_hit_rate": s["final_store_stats"]["expert_level_hit_rate"],
                      "text_provenance": s["text_provenance"]["verdict"],
                      "conversation_history": s["conversation_history"]["verdict"]},
                     indent=2))
    return 0


def cmd_fidelity(args) -> int:
    import runpy
    sys.argv = ["validate_replay_fidelity", args.runs, args.out, "--seeds", str(args.seeds),
                "--pin-model", args.pin_model]
    runpy.run_path(str(Path(__file__).resolve().parents[2] / "tools" / "validate_replay_fidelity.py"),
                   run_name="__main__")
    return 0


def cmd_iotimeline(args) -> int:
    from . import iotimeline, rawlog
    d = Path(args.run)
    raw = d / "raw.tsv" if d.is_dir() else d
    log = rawlog.parse(raw, keep_accesses=False)
    res = iotimeline.analyse(log.io)
    out = Path(args.out) if args.out else (raw.parent / "iotimeline.json")
    write_json(res, out)
    write_csv(iotimeline.queue_series(log.io, args.bucket_ms), out.with_name("queue_series.csv"))
    print(json.dumps(res["queue_depth"], indent=2))
    return 0


def cmd_predict(args) -> int:
    from . import predictor
    runs = []
    for r in read_csv(args.runs):
        runs.append({k: (float(v) if k != "label" and v not in ("", None) else v)
                     for k, v in r.items()})
    res = predictor.evaluate(runs, _cost(args))
    write_json(res, args.out)
    print(f"median |error| = {res['median_abs_pct_error']} %  -> {args.out}")
    return 0


def cmd_report(args) -> int:
    p = report.build(args.dir, title=args.title or "Argine2 research report")
    print(f"wrote {p}")
    return 0


def cmd_fingerprint(args) -> int:
    fp = fingerprint.collect(model=args.model, store=args.store, binary=args.binary,
                             workload_file=args.workload, full_checksum=args.full_checksum)
    print(json.dumps(fp, indent=2, default=str))
    return 0 if fp["publication_quality"] else 3


def cmd_schedule(args) -> int:
    order = stats.interleaved_schedule(args.configs.split(","), args.repeat)
    print(" ".join(order))
    return 0


def cmd_workloads(args) -> int:
    for name in workloads.available(args.version):
        w = workloads.load(name, args.version)
        print(f"{w.name:22s} {w.sha256[:12]}  " +
              "  ".join(f"{k}:{len(v)}" for k, v in w.prompts.items()) + f"  {w.description}")
        mt = any(x.multi_turn for v in w.prompts.values() for x in v)
        sem = history.semantics(w.history, mt)
        if sem:
            print(f"{'':22s} conversation history {sem['version']}: "
                  f"assistant_source={sem['assistant_source']} "
                  f"reasoning_content_in_history={str(sem['reasoning_content_in_history']).lower()} "
                  f"empty_final_content_policy={sem['empty_final_content_policy']} "
                  f"provenance_text={sem['provenance_text']}")
    return 0


# ------------------------------------------------------------- registry ---

def register(sub, core) -> None:
    """Add `research` to the main parser. `core` is argine.cli (for the
    runtime arguments shared with `run` / `benchmark`)."""
    rp = sub.add_parser("research", help="Argine2 research commands (trace, replay, report, ...)",
                        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    rs = rp.add_subparsers(dest="research_command", metavar="research-command")

    p = rs.add_parser("trace", help="instrumented end-to-end runs of a versioned workload")
    core._add_model_arg(p)
    core._add_runtime_args(p)
    core._add_expert_store_args(p)
    p.add_argument("--workload", required=True, help="workload name (see `research workloads`)")
    p.add_argument("--workload-version", default=workloads.CURRENT)
    p.add_argument("--split", default="dev", choices=["dev", "eval"])
    p.add_argument("--budgets", default=None, metavar="SIZE,SIZE",
                   help="expert budgets to run (interleaved); default: --expert-budget")
    p.add_argument("--repeat", type=int, default=1, help="repetitions per budget (3-5 for claims)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cooldown", type=int, default=20, metavar="SEC")
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--full-checksum", action="store_true",
                   help="full SHA-256 of model and store (slow once, then cached)")
    p.add_argument("--min-start-available", default="8G", metavar="SIZE",
                   help="warn (never invalidate) when system-available RAM right before "
                        "launch is below SIZE (default 8G; 0 disables): different starting "
                        "machine states are a suspected cause of session drift (EXP-018)")
    p.add_argument("--out", required=True, help="run directory (e.g. experiments/EXP-.../raw)")
    p.add_argument("--port", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--host", default="127.0.0.1", help=argparse.SUPPRESS)
    p.set_defaults(func=lambda a: __import__("argine.research.runner", fromlist=["x"]).cmd_trace(a))

    p = rs.add_parser("convert", help="route.log / raw log -> router trace")
    p.add_argument("source")
    p.add_argument("--to", required=True)
    p.add_argument("--kind", default="auto", choices=["auto", "route-log", "raw"])
    p.add_argument("--markers", default=None, help="markers.jsonl (raw logs)")
    p.add_argument("--n-experts", type=int, default=None)
    p.add_argument("--model-name", default=None)
    p.add_argument("--gguf", default=None, help="read exact expert sizes from the model")
    p.add_argument("--expert-bytes", type=int, default=None)
    p.add_argument("--request", type=int, default=None,
                   help="keep only the N-th request (0-based; requests start at a prompt step)")
    p.add_argument("--split", default=None, choices=["dev", "eval"])
    p.add_argument("--provenance", default=None)
    p.add_argument("--note", action="append")
    p.set_defaults(func=cmd_convert)

    p = rs.add_parser("characterize", help="routing statistics per layer (Phase 1)")
    _common_trace_args(p)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_characterize)

    p = rs.add_parser("replay", help="replay one policy at one budget")
    _common_trace_args(p)
    p.add_argument("--policy", required=True, choices=POLICIES)
    p.add_argument("--budget", required=True, type=_size)
    p.add_argument("--train", default=None, help="development trace for offline policies")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_replay)

    p = rs.add_parser("compare", help="all policies x budgets on one trace")
    _common_trace_args(p)
    p.add_argument("--budgets", required=True, type=_sizes)
    p.add_argument("--train", default=None)
    p.add_argument("--opt-static", action="store_true",
                   help="also Belady inside the optimal per-layer allocation (slow)")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_compare)

    p = rs.add_parser("elasticity", help="per-layer memory elasticity curves")
    _common_trace_args(p)
    p.add_argument("--budgets", required=True, type=_sizes)
    p.add_argument("--opt-grid", type=int, default=0, help="also Belady curves on N grid points")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_elasticity)

    p = rs.add_parser("oracle", help="offline upper bounds (Belady-style, optimal static)")
    _common_trace_args(p)
    p.add_argument("--budgets", required=True, type=_sizes)
    p.add_argument("--opt-grid", type=int, default=9)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_oracle)

    p = rs.add_parser("adaptive", help="AdaptiveResidencyController (replay) + ablations")
    _common_trace_args(p)
    p.add_argument("--budgets", required=True, type=_sizes)
    p.add_argument("--epochs", default="32,64,128")
    p.add_argument("--pinned", default="0,0.25,0.5")
    p.add_argument("--no-ablate", action="store_true")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_adaptive)

    p = rs.add_parser("analyze", help="characterize + elasticity + compare + adaptive + report")
    _common_trace_args(p)
    p.add_argument("--budgets", required=True, type=_sizes)
    p.add_argument("--train", default=None)
    p.add_argument("--opt-grid", type=int, default=9)
    p.add_argument("--no-adaptive", action="store_true")
    p.add_argument("--slice-bytes-from", default=None, metavar="RUN_DIR",
                   help="read exact (layer, kind) slice sizes from a run's raw.tsv and add the "
                        "runtime-reproduction row (StoreReplay)")
    p.add_argument("--runtime-runs", default=None, metavar="DIR",
                   help="add measured runtime rows (mean over repetitions per budget) from the "
                        "run directories in DIR")
    p.add_argument("--title", default=None)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_analyze)

    p = rs.add_parser("reprocess", help="re-derive a run's tables and summary from raw.tsv")
    p.add_argument("run", help="run directory (contains raw.tsv, markers.jsonl, responses.jsonl)")
    p.add_argument("--model-name", default=None)
    p.add_argument("--n-experts", type=int, default=None)
    p.set_defaults(func=cmd_reprocess)

    p = rs.add_parser("fidelity", help="runtime-vs-replay validation of recorded runs (EXP-021)")
    p.add_argument("runs", help="directory of run directories")
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--pin-model", default="per-thread", choices=["per-thread", "single-pool"])
    p.set_defaults(func=cmd_fidelity)

    p = rs.add_parser("iotimeline", help="queue depth Q(t) (exact or upper bound) and latency from a run")
    p.add_argument("run", help="run directory or raw.tsv")
    p.add_argument("--bucket-ms", type=float, default=1.0)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_iotimeline)

    p = rs.add_parser("predict", help="fit and validate the per-token performance model")
    p.add_argument("runs", help="CSV with ms_per_token, bytes_missed_per_token, reads_per_token, "
                                "nodes_per_token, qd")
    p.add_argument("--cost", default=None)
    p.add_argument("--out", default="prediction.json")
    p.set_defaults(func=cmd_predict)

    p = rs.add_parser("report", help="REPORT.md + figures from an analysis directory")
    p.add_argument("dir")
    p.add_argument("--title", default=None)
    p.set_defaults(func=cmd_report)

    p = rs.add_parser("fingerprint", help="build/model/workload fingerprint (section 28)")
    p.add_argument("--model", default=None)
    p.add_argument("--store", default=None)
    p.add_argument("--binary", default=None)
    p.add_argument("--workload", default=None)
    p.add_argument("--full-checksum", action="store_true")
    p.set_defaults(func=cmd_fingerprint)

    p = rs.add_parser("schedule", help="interleaved order for repeated runs")
    p.add_argument("--configs", required=True)
    p.add_argument("--repeat", type=int, default=3)
    p.set_defaults(func=cmd_schedule)

    p = rs.add_parser("workloads", help="list the versioned workload corpus")
    p.add_argument("--version", default=workloads.CURRENT)
    p.set_defaults(func=cmd_workloads)

    rp.set_defaults(func=lambda a: (rp.print_help(), 2)[1])
