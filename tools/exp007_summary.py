"""Cross-cell summary of the EXP-007 analyses (read-only over existing outputs).

    python tools/exp007_summary.py            # writes analysis-v3/cross-cell-summary.csv
                                              # and runtime-hit-rates.csv (if raw/ is present)
    python tools/exp007_summary.py --check    # compare with the files on disk; exit 1 if different

Reads, for every `analysis-v3/<cell>/` that has an `analysis_manifest.json`:
* REPORT.md                -> global EWS90/95/99 fraction (the line written by `research report`)
* heterogeneity.json       -> uniform excess over the optimal static per-layer LRU split (in-sample)
* policy_comparison.csv    -> global-lru, per-layer-lru-uniform, 2q, belady-global hit rates
* adaptive.csv             -> adaptive-e32/e64/e128 (+ moves), pin25/pin50, no-hysteresis,
                              no-cost-weighting, frequency-estimator, frozen

runtime-hit-rates.csv lists, for every run under `raw/` (smoke folders excluded),
the workload-only runtime hit rates recorded in its `manifest.json`
(`final_store_stats.expert_level_hit_rate` = gate-slice hits over prefill +
decode; `runtime_style_hit_rate`) and its EXP-021 closure verdict from
`fidelity/<campaign>-regions/` when it exists, else `fidelity/<campaign>/` (format-v1 `30b-coding-dev` was validated
in EXP-021 itself and is marked so).

concentration-vs-opportunity.csv: exploratory rank correlations over cells
(see build_rho); descriptive only.

No number is computed from raw traces here and nothing is re-run: the tables only
collect values already stored next to each analysis or run. All analyses are trained
IN-SAMPLE (see `train` column). Every policy result collected here (global-lru,
per-layer-lru-uniform, 2q, belady-global, the adaptive variants and the optimal
static split behind uniform_excess_in_sample) comes from those in-sample analyses.
None of them is a deployable or out-of-sample result. The in-sample-fitted static
policies (static-frequency, static-hot-per-layer, layer-aware-static) are not
collected.
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "experiments" / "EXP-007-router-characterization" / "analysis-v3"
OUT = A / "cross-cell-summary.csv"
OUT_RHO = A / "concentration-vs-opportunity.csv"
E = A.parent
OUT_RT = E / "runtime-hit-rates.csv"
BUDGETS = {4294967296: "4G", 2147483648: "2G", 1610612736: "1.5G"}
EWS = re.compile(r"EWS90 = \d+ experts \(([\d.]+) of \d+\), EWS95 = \d+ \(([\d.]+)\), "
                 r"EWS99 = \d+ \(([\d.]+)\)")
POLICIES = ["global-lru", "per-layer-lru-uniform", "2q", "belady-global"]
VARIANTS = {"adaptive-e32": "e32", "adaptive-e64": "e64", "adaptive-e128": "e128",
            "adaptive-e64-pin25": "e64_pin25", "adaptive-e64-pin50": "e64_pin50",
            "adaptive-e64-nohyst": "e64_nohyst", "adaptive-e64-nocost": "e64_nocost",
            "adaptive-e64-freq": "e64_freq", "frozen-e64": "frozen"}


def _rows(p: Path):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def cell(d: Path):
    m = json.loads((d / "analysis_manifest.json").read_text(encoding="utf-8"))
    t = m.get("trace_meta") or {}
    model = "80b" if "80B" in str(t.get("model", "")) else "30b"
    wl = str(t.get("workload", "")).split("@")
    ews = EWS.search((d / "REPORT.md").read_text(encoding="utf-8"))
    het = {BUDGETS.get(int(b["budget_bytes"])): b
           for b in json.loads((d / "heterogeneity.json").read_text(encoding="utf-8"))["budgets"]}
    pol = {(r["policy"], r["budget"]): r for r in _rows(d / "policy_comparison.csv")}
    ada = {(r["policy"], r["budget"]): r for r in _rows(d / "adaptive.csv")}
    out = []
    for b in ("4G", "2G", "1.5G"):
        r = {"model": model, "workload": wl[0], "workload_sha12": wl[1] if len(wl) > 1 else "",
             "split": t.get("split"), "analysis": d.name, "train": m.get("train"), "budget": b,
             "ews90_frac": ews.group(1) if ews else "", "ews95_frac": ews.group(2) if ews else "",
             "ews99_frac": ews.group(3) if ews else "",
             "uniform_excess_in_sample": f"{float(het[b]['uniform_excess']):.6f}" if b in het else ""}
        for p in POLICIES:
            r[f"hit_{p.replace('-', '_')}"] = pol.get((p, b), {}).get("hit_rate", "")
        for v, k in VARIANTS.items():
            row = ada.get((v, b), {})
            r[f"hit_{k}"] = row.get("hit_rate", "")
            r[f"moves_{k}"] = row.get("moves", "")
        out.append(r)
    return out


def build() -> str:
    rows = []
    for d in sorted(A.iterdir()):
        if d.is_dir() and (d / "analysis_manifest.json").exists():
            rows += cell(d)
    rows.sort(key=lambda r: (r["model"], r["workload"], r["split"], r["analysis"],
                             ["4G", "2G", "1.5G"].index(r["budget"])))
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def _ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):            # average ranks for ties
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0
        i = j + 1
    return r


def _spearman(a, b):
    ra, rb = _ranks(a), _ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb) if va and vb else float("nan")


def build_rho(summary_csv: str) -> str:
    """Exploratory, post hoc: Spearman rank correlation, over cells, between the
    global EWS90 fraction (lower = more concentrated) and three measures of
    allocation opportunity. n = 14 cells per model, 28 pooled. Descriptive only:
    no test, and trace lengths differ between workloads (not length-normalised)."""
    rows = list(csv.DictReader(io.StringIO(summary_csv)))
    out = []
    for scope in ("30b", "80b", "pooled"):
        for b in ("4G", "2G", "1.5G"):
            x = [r for r in rows if r["budget"] == b and scope in ("pooled", r["model"])]
            e = [float(r["ews90_frac"]) for r in x]
            meas = {
                "uniform_excess_in_sample": [float(r["uniform_excess_in_sample"]) for r in x],
                "adaptive_e64_minus_global_lru": [float(r["hit_e64"]) - float(r["hit_global_lru"]) for r in x],
                "adaptive_e64_minus_per_layer_uniform": [float(r["hit_e64"]) - float(r["hit_per_layer_lru_uniform"])
                                                         for r in x],
            }
            for k, v in meas.items():
                out.append({"scope": scope, "budget": b, "n_cells": len(x), "measure": k,
                            "spearman_rho_vs_ews90_frac": f"{_spearman(e, v):.3f}"})
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(out[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(out)
    return buf.getvalue()


def build_runtime() -> str:
    raw = E / "raw"
    rows = []
    for camp in sorted(p for p in raw.iterdir() if p.is_dir() and "smoke" not in p.name):
        fid = {}
        # a `<campaign>-regions` validation (workload / tail verdicts, EXP-021
        # §14) supersedes the original strict-only output kept beside it
        fcsv = E / "fidelity" / f"{camp.name}-regions" / "fidelity_runs.csv"
        if not fcsv.exists():
            fcsv = E / "fidelity" / camp.name / "fidelity_runs.csv"
        if fcsv.exists():
            fid = {r["run"]: r for r in _rows(fcsv)}
        for run in sorted(camp.glob("run-*/manifest.json")):
            m = json.loads(run.read_text(encoding="utf-8"))
            fs = m.get("final_store_stats") or {}
            f = fid.get(run.parent.name, {})
            if f:
                closure = f.get("exp021_closure", "")
            elif camp.name == "30b-coding-dev":
                closure = "EXP-021 (format v1, validated there)"
            else:
                closure = "no fidelity record"
            fp = (m.get("fingerprint") or {}).get("model") or {}
            model = "80b" if "80B" in json.dumps(fp) or camp.name.startswith("80b") else "30b"
            rows.append({"campaign": camp.name, "run": run.parent.name, "model": model,
                         "workload": m.get("workload"), "workload_sha12": str(m.get("workload_sha256", ""))[:12],
                         "split": m.get("split"), "budget": m.get("budget"),
                         "expert_level_hit_rate": fs.get("expert_level_hit_rate"),
                         "runtime_style_hit_rate": fs.get("runtime_style_hit_rate"),
                         "trace_complete": (m.get("trace_completeness") or {}).get("complete"),
                         "exp021_closure": closure,
                         "qualified": closure == "PASS" or closure.startswith("EXP-021 (format v1")})
    if not rows:
        return None
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def main(argv) -> int:
    summary = build()
    outs = [(OUT, summary), (OUT_RHO, build_rho(summary))]
    rt = build_runtime() if (E / "raw").exists() else None
    if rt is not None:
        outs.append((OUT_RT, rt))
    else:
        print("note: no run manifests under raw/; runtime-hit-rates.csv not checked")
    if "--check" in argv:
        bad = [p for p, t in outs if not (p.exists() and p.read_text(encoding="utf-8") == t)]
        print("EXP-007 summaries up to date" if not bad else
              "differs from the source files: " + ", ".join(p.name for p in bad))
        return 0 if not bad else 1
    for p, t in outs:
        p.write_text(t, encoding="utf-8", newline="")
        print(f"wrote {p.relative_to(ROOT)} ({t.count(chr(10)) - 1} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
