"""`argine research report`: a deterministic report from stored analysis data.

Input: a directory written by `argine research analyze` (and optionally run
directories written by `argine research trace`). Output: REPORT.md plus
figures/ (PNG, when matplotlib is installed) whose source data are the CSV
files already in the directory. The report never computes anything that is
not already stored: it reads, formats, and labels.

Every statement the report makes carries a taxonomy tag:
[MEASURED] read from data, [DERIVED] computed from measured data by a stated
method, [LIMITATION] what the data cannot support.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .io import read_csv, read_json

GiB = 1 << 30


def _f(x, nd=3) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a" if x in (None, "") else str(x)
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.{nd}g}" if abs(v) < 1 else f"{v:.{nd}f}".rstrip("0").rstrip(".")


def _table(rows: List[Dict], cols: List[str], headers: Optional[List[str]] = None) -> str:
    h = headers or cols
    out = ["| " + " | ".join(h) + " |", "|" + "---|" * len(h)]
    for r in rows:
        out.append("| " + " | ".join(_f(r.get(c)) if c not in ("policy", "variant", "budget", "train",
                                                                "runs", "shard_map")
                                     else str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def figures(d: Path) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return []
    fig_dir = d / "figures"
    fig_dir.mkdir(exist_ok=True)
    made = []
    pl = read_csv(d / "per_layer.csv") if (d / "per_layer.csv").exists() else []
    if pl:
        layers = [int(r["layer"]) for r in pl]
        for key, label, fname in (("entropy_norm", "normalised routing entropy", "entropy_by_layer.png"),
                                  ("ews95_frac", "EWS95 / n_experts", "ews95_by_layer.png")):
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.bar(layers, [float(r[key]) for r in pl], color="#4C72B0")
            ax.set_xlabel("layer")
            ax.set_ylabel(label)
            ax.set_title(f"{label} by layer (source: per_layer.csv)", fontsize=9)
            fig.tight_layout()
            fig.savefig(fig_dir / fname, dpi=120)
            plt.close(fig)
            made.append(fname)
    if (d / "reuse_cdf.csv").exists():
        rows = read_csv(d / "reuse_cdf.csv")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        pts = [(int(r["N"]), float(r["p_le"])) for r in rows if r["layer"] == "all"]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label="all layers")
        for lay in sorted({r["layer"] for r in rows if r["layer"] != "all"}, key=int)[::12]:
            q = [(int(r["N"]), float(r["p_le"])) for r in rows if r["layer"] == lay]
            ax.plot([p[0] for p in q], [p[1] for p in q], alpha=.5, label=f"layer {lay}")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("reuse distance N (tokens)")
        ax.set_ylabel("P(reuse distance <= N)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / "reuse_cdf.png", dpi=120)
        plt.close(fig)
        made.append("reuse_cdf.png")
    if (d / "policy_comparison.csv").exists():
        rows = [r for r in read_csv(d / "policy_comparison.csv") if r.get("feasible") == "True"]
        pols = []
        for r in rows:
            if r["policy"] not in pols:
                pols.append(r["policy"])
        for metric, label, fname in (("hit_rate", "hit rate (expert level)", "hit_rate_vs_budget.png"),
                                     ("misses_per_token", "misses / token", "misses_vs_budget.png"),
                                     ("stall_ms_per_token_modelled",
                                      "modelled stall ms / token", "stall_vs_budget.png")):
            fig, ax = plt.subplots(figsize=(7, 4))
            cmap = plt.get_cmap("tab20")
            for i, p in enumerate(pols):
                # measured / store-replay rows carry only the metrics they can
                # report (no modelled stall for the runtime): skip blanks
                q = sorted(((int(r["budget_bytes"]) / GiB, float(r[metric])) for r in rows
                            if r["policy"] == p and r.get(metric) not in (None, "")))
                if not q:
                    continue
                oracle_like = p.startswith(("belady", "oracle"))
                ax.plot([x for x, _ in q], [y for _, y in q], "--" if oracle_like else "-",
                        marker="." if not oracle_like else "x", color=cmap(i % 20),
                        linewidth=2.2 if oracle_like else 1.4, label=p)
            ax.set_xlabel("expert budget (GiB)")
            ax.set_ylabel(label)
            ax.legend(fontsize=6, ncol=2)
            fig.tight_layout()
            fig.savefig(fig_dir / fname, dpi=120)
            plt.close(fig)
            made.append(fname)
    if (d / "allocation_timeline.csv").exists():
        rows = read_csv(d / "allocation_timeline.csv")
        steps = sorted({int(r["step"]) for r in rows})
        layers = sorted({int(r["layer"]) for r in rows})
        grid = {(int(r["step"]), int(r["layer"])): int(r["slots"]) for r in rows}
        fig, ax = plt.subplots(figsize=(7, 3.5))
        im = ax.imshow([[grid.get((s, l), 0) for s in steps] for l in layers], aspect="auto",
                       origin="lower", extent=[steps[0], steps[-1], layers[0], layers[-1]])
        fig.colorbar(im, label="slots")
        ax.set_xlabel("step")
        ax.set_ylabel("layer")
        ax.set_title("adaptive allocation over time (allocation_timeline.csv)", fontsize=9)
        fig.tight_layout()
        fig.savefig(fig_dir / "allocation_timeline.png", dpi=120)
        plt.close(fig)
        made.append("allocation_timeline.png")
    return made


def build(d: str | Path, title: str = "Argine2 research report") -> Path:
    d = Path(d)
    md: List[str] = [f"# {title}", "",
                     "Generated by `argine research report` from the CSV/JSON files in this "
                     "directory. It contains no number that is not stored next to it.", ""]
    man = read_json(d / "analysis_manifest.json") if (d / "analysis_manifest.json").exists() else {}
    meta = man.get("trace_meta", {})
    md += ["## Inputs", "",
           f"- model: `{meta.get('model', '?')}` — {meta.get('n_experts')} experts, top-k "
           f"{meta.get('top_k')}, {len(meta.get('layers', []))} MoE layers",
           f"- trace source: `{meta.get('source', '?')}`; provenance: {meta.get('provenance', '?')}",
           f"- workload: `{meta.get('workload') or 'not recorded'}`, split: "
           f"`{meta.get('split') or 'not recorded'}`",
           f"- phase analysed: `{man.get('phase', '?')}`; offline policies trained "
           f"**{man.get('train', '?')}**", ""]
    for n in meta.get("notes", []) or []:
        md.append(f"- note: {n}")
    if man.get("train") == "in-sample":
        md += ["", "[LIMITATION] Offline policies (static frequency, static hot set, "
               "layer-aware static) were fitted on the same trace they are evaluated on: "
               "their rows are in-sample upper bounds, not results.", ""]
    ch = read_json(d / "characterization.json") if (d / "characterization.json").exists() else None
    if ch:
        s = ch["summary"]
        md += ["## Routing characterization", "",
               f"[MEASURED] {s['tokens']} tokens, {s['layers']} layers, {s['n_experts']} experts, "
               f"top-k {s['top_k']} (phase {s['phase']}).", ""]
        sp = s["layer_spread"]
        rows = []
        for k in ("entropy_norm", "gini", "ews90_frac", "ews95_frac", "ews99_frac", "ws_tau16",
                  "overlap_prev", "hot_set_stability", "reuse_median"):
            v = sp.get(k) or {}
            rows.append({"metric": k, **v})
        md += ["[DERIVED] Spread of per-layer metrics (definitions in `argine/research/metrics.py`):", "",
               "| metric | min (layer) | max (layer) | mean | CV across layers |", "|---|---|---|---|---|"]
        for r in rows:
            md.append(f"| {r['metric']} | {_f(r.get('min'))} ({r.get('argmin')}) | "
                      f"{_f(r.get('max'))} ({r.get('argmax')}) | {_f(r.get('mean'))} | {_f(r.get('cv'))} |")
        g = s.get("global_working_set", {})
        md += ["", f"[DERIVED] Global effective working set: EWS90 = {g.get('ews90')} experts "
               f"({_f(g.get('ews90_frac'))} of {g.get('keys_total')}), EWS95 = {g.get('ews95')} "
               f"({_f(g.get('ews95_frac'))}), EWS99 = {g.get('ews99')} ({_f(g.get('ews99_frac'))}).",
               f"[DERIVED] Per-layer partition floor (every layer holding its largest event): "
               f"{_f(s.get('per_layer_partition_floor_bytes', 0) / GiB)} GiB.", ""]
    if (d / "heterogeneity.json").exists():
        h = read_json(d / "heterogeneity.json")
        rows = [b for b in h["budgets"] if not b.get("infeasible") and "error" not in b]
        if rows:
            md += ["## Layer heterogeneity of memory value (in-sample)", "",
                   "[DERIVED] Misses of a per-layer LRU with the uniform split vs the optimal "
                   "static split for the SAME objective (exact LRU stack-distance curves, exact "
                   "byte sizes, same minimum per layer), computed on this trace: an IN-SAMPLE "
                   "upper bound on what any static per-layer split can gain. The solver is "
                   "checked to be never worse than uniform.", "",
                   _table([{**r, "budget": f"{r['budget_bytes'] / GiB:g}G"} for r in rows],
                          ["budget", "misses_uniform", "misses_optimal_static", "uniform_excess",
                           "optimal_slots_min", "optimal_slots_median", "optimal_slots_max"]), ""]
    if (d / "policy_comparison.csv").exists():
        allrows = read_csv(d / "policy_comparison.csv")
        rows = [r for r in allrows if r.get("feasible") == "True"]
        legacy = not any(r.get("category") for r in rows)
        md += ["## Policy comparison", "",
               "Hits, misses and bytes are exact for the replay; stall is modelled (cost model "
               "in `analysis_manifest.json`). All rows use the same metric unless stated: "
               "expert-event hit rate (an expert counts as a hit only if all three of its slices "
               "were resident), counted steps of the analysed phase, warm-up and flush sentinel "
               "excluded. Categories are never mixed in one table.", ""]
        if legacy:
            md += ["[LIMITATION] This analysis predates the category labels (replay semantics "
                   "v1-legacy). See docs/research/REPLAY-FIDELITY-VALIDATION.md.", "",
                   _table(rows, ["budget", "policy", "train", "hit_rate", "misses_per_token",
                                 "stall_ms_per_token_modelled", "oracle_gap_misses"]), ""]
        else:
            groups = [
                ("Measured runtime [MEASURED]", lambda r: r.get("category") == "measured",
                 ["budget", "policy", "hit_rate", "hit_rate_sd_over_runs", "slice_gate_hit_rate", "runs"]),
                ("Runtime reproduction [DERIVED] — StoreReplay (validated event-by-event, EXP-021)",
                 lambda r: r.get("category") == "runtime-reproduction",
                 ["budget", "policy", "hit_rate", "hit_rate_sd_over_shard_maps", "slice_gate_hit_rate",
                  "partial_expert_events", "shard_map"]),
                ("Counterfactual policies (implementable, online or trace-independent)",
                 lambda r: r.get("category") == "counterfactual" and r.get("in_sample") != "True",
                 ["budget", "policy", "hit_rate", "misses_per_token",
                  "stall_ms_per_token_modelled", "oracle_gap_misses", "gap_closed_vs_lru"]),
                ("Counterfactual policies — IN-SAMPLE / UPPER BOUND (fitted on the evaluated trace; "
                 "not deployable competitors)",
                 lambda r: r.get("category") == "counterfactual" and r.get("in_sample") == "True",
                 ["budget", "policy", "train", "hit_rate", "misses_per_token", "oracle_gap_misses"]),
                ("Oracles (future knowledge: bounds, not results)",
                 lambda r: r.get("category") == "oracle",
                 ["budget", "policy", "hit_rate", "misses_per_token", "stall_ms_per_token_modelled"]),
            ]
            for title, pred, cols in groups:
                sel = [r for r in rows if pred(r)]
                if sel:
                    md += [f"### {title}", "", _table(sel, cols), ""]
        infeas = [r for r in allrows if r.get("feasible") == "False"]
        if infeas:
            md += [f"[DERIVED] {len(infeas)} (policy, budget) pairs were infeasible: "
                   "strictly layer-partitioned policies cannot run below the per-layer "
                   "partition floor.", ""]
    if (d / "adaptive.csv").exists():
        rows = [r for r in read_csv(d / "adaptive.csv") if r.get("feasible") == "True"]
        md += ["## Adaptive allocation and ablations (replay prototype)", "",
               _table(rows, ["budget", "variant", "hit_rate", "misses_per_token",
                             "stall_ms_per_token_modelled", "moves"]), ""]
    if (d / "adaptation.csv").exists():
        md += ["## Workload transitions", "", _table(read_csv(d / "adaptation.csv"),
               ["variant", "budget", "boundary_step", "steady_misses_per_step",
                "adaptation_steps", "transient_excess_misses"]), ""]
    figs = figures(d)
    if figs:
        md += ["## Figures", ""] + [f"![{f}](figures/{f})" for f in figs] + [""]
    else:
        md += ["[LIMITATION] matplotlib not available: figures not generated (the CSV "
               "sources are in this directory).", ""]
    p = d / "REPORT.md"
    p.write_text("\n".join(md) + "\n", encoding="utf-8")
    return p
