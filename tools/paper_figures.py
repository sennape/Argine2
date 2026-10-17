from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _save(fig: plt.Figure, pdf_path: Path, png_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        metadata={
            "Creator": "Argine2 tools/paper_figures.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        metadata={"Software": "Argine2 tools/paper_figures.py"},
    )
    plt.close(fig)
    print(f"Created: {pdf_path.relative_to(ROOT)}")
    print(f"Created: {png_path.relative_to(ROOT)}")


def figure1(write: bool) -> None:
    """Figure 1 — system architecture."""
    if not write:
        print("Fig. 1 check: no data dependencies")
        return

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    def box(x, y, w, h, text, fs=9):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.06",
                fill=False,
                linewidth=1.2,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.1,
            )
        )

    box(4.5, 5.45, 3.0, 0.65, "MoE token execution", 10)
    box(4.75, 4.25, 2.5, 0.65, "Router / gating", 10)
    box(
        4.0,
        1.8,
        4.0,
        1.75,
        "Expert-memory pool in DRAM\nHard cap M\n\nLayer-wise resident expert slices",
        9,
    )
    box(
        0.45,
        2.1,
        2.35,
        1.15,
        "Allocation / retention policy\nper-layer budgets + eviction",
        8.5,
    )
    box(
        9.15,
        2.0,
        2.35,
        1.35,
        "NVMe expert store\nnon-resident expert slices",
        8.5,
    )
    box(
        3.55,
        0.25,
        4.9,
        0.75,
        "hit / reuse   ·   bytes read   ·   queue depth\n token latency   ·   throughput",
        8,
    )

    arrow(6, 5.45, 6, 4.92)
    arrow(6, 4.25, 6, 3.58)
    arrow(2.8, 2.67, 3.98, 2.67)
    arrow(8.02, 2.67, 9.13, 2.67)
    arrow(6, 1.8, 6, 1.02)

    ax.text(6.18, 5.15, "token", fontsize=7.5, va="center")
    ax.text(6.18, 3.92, "selected experts", fontsize=7.5, va="center")
    ax.text(3.38, 2.89, "allocation", fontsize=7.5, ha="center")
    ax.text(8.58, 2.91, "miss", fontsize=7.5, ha="center")
    ax.text(8.58, 2.35, "async direct I/O", fontsize=7.5, ha="center")

    fig.tight_layout()
    out = ROOT / "docs" / "figures"
    _save(
        fig,
        out / "argine_system_architecture.pdf",
        out / "argine_system_architecture.png",
    )


def _exp007_frame() -> tuple[Path, pd.DataFrame]:
    base = ROOT / "experiments" / "EXP-007-router-characterization" / "analysis-v3"
    df = pd.read_csv(base / "cross-cell-summary.csv")
    df = df[df["train"].eq("in-sample")].copy()
    return base, df


def figure2(write: bool) -> None:
    """Figure 2 — layer heterogeneity and uniform split penalty."""
    base, df = _exp007_frame()
    cells = (
        df[df["budget"].eq("4G")][["model", "analysis"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    def load(model: str) -> pd.DataFrame:
        series = []
        for analysis in cells.loc[cells["model"].eq(model), "analysis"]:
            path = base / analysis / "per_layer.csv"
            s = pd.read_csv(path).set_index("layer")["ews90_frac"].rename(analysis)
            series.append(s)
        return pd.concat(series, axis=1)

    p30 = load("30b")
    p80 = load("80b")
    assert p30.shape[1] == 14 and p80.shape[1] == 14, (
        f"Fig. 2: expected 14 cells/model, got 30B={p30.shape[1]}, 80B={p80.shape[1]}"
    )

    cv = lambda p: p.std(axis=0, ddof=0) / p.mean(axis=0)
    c30, c80 = cv(p30), cv(p80)

    u = 100 * df.groupby(["model", "budget"])["uniform_excess_in_sample"].agg(
        ["median", "min", "max"]
    )
    budgets = ["4G", "2G", "1.5G"]

    print(f"Fig. 2: 30B layer-EWS90 CV {c30.min():.3f}..{c30.max():.3f}")
    print(f"Fig. 2: 80B layer-EWS90 CV {c80.min():.3f}..{c80.max():.3f}")
    for model in ["30b", "80b"]:
        for budget in budgets:
            r = u.loc[(model, budget)]
            print(
                f"Fig. 2: {model} {budget} uniform excess "
                f"median={r['median']:.3f}% range={r['min']:.3f}..{r['max']:.3f}%"
            )

    if not write:
        return

    fig, axs = plt.subplots(1, 3, figsize=(10.8, 3.5))
    for ax, p, title in [
        (axs[0], p30, "(a) 30B layer demand"),
        (axs[1], p80, "(b) 80B layer demand"),
    ]:
        line = ax.plot(p.index, p.median(axis=1), linewidth=1.6)[0]
        ax.fill_between(
            p.index,
            p.min(axis=1),
            p.max(axis=1),
            alpha=0.18,
            color=line.get_color(),
        )
        ax.set_title(title)
        ax.set_xlabel("MoE layer")
        ax.set_ylim(0, 0.8)
        ax.spines[["top", "right"]].set_visible(False)

    axs[0].set_ylabel("EWS90 fraction of layer experts")

    x = np.arange(3)

    def vals(model: str) -> np.ndarray:
        return np.array(
            [
                [
                    u.loc[(model, b), "median"],
                    u.loc[(model, b), "min"],
                    u.loc[(model, b), "max"],
                ]
                for b in budgets
            ]
        )

    a, z = vals("30b"), vals("80b")
    axs[2].errorbar(
        x - 0.07,
        a[:, 0],
        yerr=np.vstack((a[:, 0] - a[:, 1], a[:, 2] - a[:, 0])),
        fmt="o-",
        capsize=3,
        label="30B",
    )
    axs[2].errorbar(
        x + 0.07,
        z[:, 0],
        yerr=np.vstack((z[:, 0] - z[:, 1], z[:, 2] - z[:, 0])),
        fmt="s-",
        capsize=3,
        label="80B",
    )
    axs[2].set_xticks(x, budgets)
    axs[2].set_xlabel("Expert-memory budget")
    axs[2].set_ylabel("Uniform-allocation excess misses (%)")
    axs[2].set_title("(c) Uniform split penalty")
    axs[2].legend(frameon=False, fontsize=8)
    axs[2].spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = ROOT / "experiments" / "EXP-007-router-characterization" / "figures"
    _save(
        fig,
        out / "layer_heterogeneity_and_uniform_penalty.pdf",
        out / "layer_heterogeneity_and_uniform_penalty.png",
    )


def figure3(write: bool) -> None:
    """Figure 3 — allocation gain versus future-informed selection headroom."""
    _, df = _exp007_frame()
    df["adaptive_pp"] = 100 * (df["hit_e64"] - df["hit_global_lru"])
    df["selection_pp"] = 100 * (df["hit_belady_global"] - df["hit_e64"])

    budgets = ["4G", "2G", "1.5G"]
    models = ["30b", "80b"]
    counts = df.groupby(["model", "budget"]).size()

    assert all(counts.get((m, b), 0) == 14 for m in models for b in budgets), (
        "Fig. 3: expected exactly 14 cells for every model/budget"
    )

    s = df.groupby(["model", "budget"])[["adaptive_pp", "selection_pp"]].agg(
        ["median", "min", "max"]
    )

    for m in models:
        for b in budgets:
            print(
                f"Fig. 3: {m} {b} n={counts[(m,b)]} "
                f"adaptive={s.loc[(m,b),('adaptive_pp','median')]:.3f} pp "
                f"[{s.loc[(m,b),('adaptive_pp','min')]:.3f},"
                f"{s.loc[(m,b),('adaptive_pp','max')]:.3f}] "
                f"selection={s.loc[(m,b),('selection_pp','median')]:.3f} pp "
                f"[{s.loc[(m,b),('selection_pp','min')]:.3f},"
                f"{s.loc[(m,b),('selection_pp','max')]:.3f}]"
            )

    if not write:
        return

    fig, axs = plt.subplots(1, 2, figsize=(8.7, 3.8), sharey=True)
    x = np.arange(3)

    for ax, m in zip(axs, models):
        q = s.loc[m]
        adaptive = np.array(
            [
                [
                    q.loc[b, ("adaptive_pp", "median")],
                    q.loc[b, ("adaptive_pp", "min")],
                    q.loc[b, ("adaptive_pp", "max")],
                ]
                for b in budgets
            ]
        )
        selection = np.array(
            [
                [
                    q.loc[b, ("selection_pp", "median")],
                    q.loc[b, ("selection_pp", "min")],
                    q.loc[b, ("selection_pp", "max")],
                ]
                for b in budgets
            ]
        )

        ax.errorbar(
            x - 0.08,
            adaptive[:, 0],
            yerr=np.vstack(
                (
                    adaptive[:, 0] - adaptive[:, 1],
                    adaptive[:, 2] - adaptive[:, 0],
                )
            ),
            fmt="o-",
            capsize=3,
            label="Adaptive-e64 gain over global LRU",
        )
        ax.errorbar(
            x + 0.08,
            selection[:, 0],
            yerr=np.vstack(
                (
                    selection[:, 0] - selection[:, 1],
                    selection[:, 2] - selection[:, 0],
                )
            ),
            fmt="s-",
            capsize=3,
            label="Belady-style headroom over adaptive-e64",
        )
        ax.axhline(0, linestyle="--", linewidth=1, color="0.45")
        ax.set_xticks(x, budgets)
        ax.set_xlabel("Expert-memory budget")
        ax.set_title("(a) 30B" if m == "30b" else "(b) 80B")
        ax.spines[["top", "right"]].set_visible(False)

    axs[0].set_ylabel("Hit-rate difference (percentage points)")
    axs[0].legend(frameon=False, fontsize=7)

    fig.tight_layout()
    out = ROOT / "experiments" / "EXP-007-router-characterization" / "figures"
    _save(
        fig,
        out / "allocation_vs_selection.pdf",
        out / "allocation_vs_selection.png",
    )


def figure4(write: bool) -> None:
    """Figure 4 — workload drift versus adaptation mechanism."""
    base = ROOT / "experiments" / "EXP-023-workload-drift"
    j = pd.read_csv(base / "processed" / "jsd_summary.csv")
    c = pd.read_csv(base / "processed" / "controller_summary.csv")

    j = j[(j["level"] == "trace") & (j["boundary"].astype(str) == "all")].copy()
    assert len(j) == 4, f"Fig. 4: expected 4 aggregate trace rows, got {len(j)}"
    assert len(c) == 12, f"Fig. 4: expected 12 controller rows, got {len(c)}"

    print(
        "Fig. 4: median boundary/within ratios = "
        + ", ".join(f"{r.key}:{r.ratio_median:.3f}" for r in j.itertuples())
    )

    if not write:
        return

    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.8))

    ax = axs[0]
    x = list(range(len(j)))
    within = j["within_median_median"].astype(float).to_numpy()
    boundary = j["boundary_jsd_median"].astype(float).to_numpy()

    for i, (a, b) in enumerate(zip(within, boundary)):
        ax.plot([i, i], [a, b], linewidth=1, color="0.75", zorder=1)
    ax.scatter(x, within, s=34, label="Within segment", zorder=3)
    ax.scatter(x, boundary, s=34, label="Segment boundary", zorder=3)
    ax.set_xticks(
        x,
        [str(k).replace("-", " ").upper() for k in j["key"]],
        rotation=30,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel("Jensen-Shannon divergence")
    ax.set_title("(a) Workload drift")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axs[1]
    order = ["4G", "2G", "1.5G"]
    xpos = {b: i for i, b in enumerate(order)}
    styles = {
        ("30b", "dev"): ("C0", "o", "30B DEV", -0.12),
        ("30b", "eval"): ("C0", "s", "30B EVAL", -0.04),
        ("80b", "dev"): ("C1", "o", "80B DEV", 0.04),
        ("80b", "eval"): ("C1", "s", "80B EVAL", 0.12),
    }

    for key, (color, marker, label, offset) in styles.items():
        z = c[(c["model"] == key[0]) & (c["split"] == key[1])]
        ax.scatter(
            [xpos[b] + offset for b in z["budget"]],
            z["A_share_of_gain_mean_alloc"].astype(float),
            s=34,
            marker=marker,
            color=color,
            label=label,
            zorder=3,
        )

    ax.axhline(1.0, linestyle="--", linewidth=1, color="0.45")
    ax.text(
        0.02,
        1.015,
        "100% of controller gain",
        transform=ax.get_yaxis_transform(),
        fontsize=8,
        va="bottom",
    )
    ax.set_xticks(range(len(order)), order)
    ax.set_xlabel("Expert-memory budget")
    ax.set_ylabel("Fixed-mean gain / controller gain")
    ax.set_title("(b) Adaptation mechanism")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = base / "figures"
    _save(
        fig,
        out / "workload_drift_vs_adaptation.pdf",
        out / "workload_drift_vs_adaptation.png",
    )


def figure5(write: bool) -> None:
    """Figure 5 — measured versus predicted token latency."""
    base = ROOT / "experiments" / "EXP-025-performance-model"
    df = pd.read_csv(base / "results.csv")
    assert len(df) == 5, f"Fig. 5: expected 5 configurations, got {len(df)}"

    median_ape = float(df["abs_pct_error"].median())
    assert median_ape < 15.0, f"Fig. 5: median APE no longer below 15%: {median_ape}"
    print(f"Fig. 5: median LOO APE = {median_ape:.6f}%")

    if not write:
        return

    x = df["measured_ms"].astype(float).to_numpy()
    y = df["predicted_ms"].astype(float).to_numpy()
    labels = df["label"].astype(str).tolist()

    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    ax.scatter(x, y, s=42, zorder=3)

    lo = min(np.r_[x, y]) - 12
    hi = max(np.r_[x, y]) + 12
    ax.plot([lo, hi], [lo, hi], "--", linewidth=1, color="0.45", zorder=1)

    for a, b, label in zip(x, y, labels):
        ax.annotate(
            label,
            (a, b),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Measured latency (ms/token)")
    ax.set_ylabel("Predicted latency (ms/token)")
    ax.text(
        0.04,
        0.94,
        f"Median LOO APE = {median_ape:.2f}%",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out")

    fig.tight_layout()
    out = base / "figures"
    _save(
        fig,
        out / "performance_model_predicted_vs_measured.pdf",
        out / "performance_model_predicted_vs_measured.png",
    )


def figure6(write: bool) -> None:
    """Figure 6 — extreme pressure and matched-F cross-model regimes."""
    b22 = ROOT / "experiments" / "EXP-022-extreme-pressure"
    d22 = pd.read_csv(b22 / "results.csv")

    perf = d22[(d22["block"] == "30b-perf") & (d22["outcome"] == "VALID")].copy()
    diag = d22[
        (d22["model"] == "30b") & d22["expert_level_hit_rate"].notna()
    ].copy()

    cp = (
        perf.groupby(["budget", "fraction_of_footprint"])["gen_tok_s"]
        .agg(["mean", "min", "max", "count"])
        .reset_index()
        .sort_values("fraction_of_footprint")
    )
    cd = (
        diag.groupby(["budget", "fraction_of_footprint"])["expert_level_hit_rate"]
        .agg(["mean", "min", "max", "count"])
        .reset_index()
        .sort_values("fraction_of_footprint")
    )

    assert set(cp["count"]) == {3}, "Fig. 6: EXP022 expected 3 perf replicas/budget"
    assert set(cd["count"]) == {2}, "Fig. 6: EXP022 expected 2 diagnostic replicas/budget"

    bt = float(cp.loc[cp["budget"] == "1.5G", "mean"].iloc[0])
    bh = float(cd.loc[cd["budget"] == "1.5G", "mean"].iloc[0])

    cp[["rel", "relmin", "relmax"]] = 100 * cp[["mean", "min", "max"]] / bt
    cd[["rel", "relmin", "relmax"]] = 100 * cd[["mean", "min", "max"]] / bh

    b26 = ROOT / "experiments" / "EXP-026-30b-vs-80b-regimes"
    d26 = pd.read_csv(b26 / "results.csv")
    q = d26[
        (d26["block"] == "perf")
        & (d26["threads"] == 4)
        & (d26["outcome"] == "VALID")
        & d26["model"].isin(["30b", "80b"])
    ].copy()

    footprint = {"30b": 1_097_072_640, "80b": 719_585_280}
    q["budget_over_F"] = q.apply(
        lambda r: r["budget_bytes"] / footprint[r["model"]], axis=1
    )

    g = (
        q.groupby(["model", "point", "budget_over_F"])["gen_tok_s"]
        .agg(["median", "min", "max", "count"])
        .reset_index()
    )
    assert all(g["count"] == 3), "Fig. 6: EXP026 expected 3 replicas/cell"

    retained_512 = float(
        100 * cp.loc[cp["budget"] == "512M", "mean"].iloc[0] / bt
    )
    retained_hit_768 = float(
        100 * cd.loc[cd["budget"] == "768M", "mean"].iloc[0] / bh
    )
    print(f"Fig. 6: 512M throughput retained = {retained_512:.2f}%")
    print(f"Fig. 6: 768M expert-hit retained = {retained_hit_768:.3f}%")

    if not write:
        return

    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.8))

    ax = axs[0]
    ax.errorbar(
        cp["fraction_of_footprint"],
        cp["rel"],
        yerr=np.vstack((cp["rel"] - cp["relmin"], cp["relmax"] - cp["rel"])),
        fmt="o-",
        capsize=3,
        label="Generation throughput",
    )
    ax.errorbar(
        cd["fraction_of_footprint"],
        cd["rel"],
        yerr=np.vstack((cd["rel"] - cd["relmin"], cd["relmax"] - cd["rel"])),
        fmt="s--",
        capsize=3,
        label="Expert-level hit rate",
    )
    ax.axvline(1.0, linestyle=":", linewidth=1, color="0.45")
    ax.text(
        1.01,
        0.05,
        "one-token footprint",
        rotation=90,
        transform=ax.get_xaxis_transform(),
        fontsize=7,
        va="bottom",
    )
    ax.set_xlabel("Expert-memory budget / one-token footprint")
    ax.set_ylabel("Relative to 1.5G baseline (%)")
    ax.set_ylim(-3, 108)
    ax.set_title("(a) Extreme pressure: 30B")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axs[1]
    styles = {"30b": ("C0", "o", "30B"), "80b": ("C1", "s", "80B")}
    for model in ["30b", "80b"]:
        z = g[g["model"] == model].copy()
        color, marker, label = styles[model]

        ax.errorbar(
            z["budget_over_F"],
            z["median"],
            yerr=np.vstack((z["median"] - z["min"], z["max"] - z["median"])),
            fmt=marker,
            linestyle="none",
            capsize=3,
            color=color,
            label=label,
        )

        primary = z[z["point"].isin(["P1", "P2"])].sort_values("budget_over_F")
        ax.plot(
            primary["budget_over_F"],
            primary["median"],
            "-",
            color=color,
            linewidth=1.5,
        )

        p3 = z[z["point"] == "P3"]
        ax.scatter(
            p3["budget_over_F"],
            p3["median"],
            marker=marker,
            facecolors="white",
            edgecolors=color,
            s=38,
            zorder=4,
        )

    ax.set_xlabel("Expert-memory budget / one-token footprint")
    ax.set_ylabel("Generation throughput (tok/s)")
    ax.set_title("(b) Matched pressure: EXP-026")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([1.0, 1.5, 2.0, 3.0, 4.0]))
    ax.xaxis.set_major_formatter(FixedFormatter(["1", "1.5", "2", "3", "4"]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.legend(frameon=False, fontsize=8)
    ax.text(
        0.05,
        0.95,
        "Primary P1→P2: β30 = 0.476\nβ80 = 0.495   Δβ = +0.019",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )
    ax.text(
        0.05,
        0.76,
        "Open markers: P3 (secondary)",
        transform=ax.transAxes,
        va="top",
        fontsize=7,
    )
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = b26 / "figures"
    _save(
        fig,
        out / "cross_regime_synthesis.pdf",
        out / "cross_regime_synthesis.png",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate inputs and regenerate the six main Argine paper figures."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate figure source data and invariants without writing files.",
    )
    args = parser.parse_args()

    write = not args.check
    figure1(write)
    figure2(write)
    figure3(write)
    figure4(write)
    figure5(write)
    figure6(write)

    if args.check:
        print("paper_figures.py: CHECK OK")
    else:
        print("paper_figures.py: all six figures regenerated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
