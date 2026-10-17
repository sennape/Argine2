"""Interpretable per-token performance model (Contribution F, section 26).

    T_token = T_compute + T_unhidden_IO + T_sync + T_controller

with, for one configuration (model x workload x budget x policy):

    T_unhidden_IO ~ a * (bytes_missed_per_token / BW(qd)) + b * reads_per_token
    T_sync        ~ s * moe_nodes_per_token        (barriers, hit bookkeeping)
    T_compute     ~ c                              (per model; fitted)
    T_controller  measured directly (0 for static policies)

The coefficients (c, s, a, b) are fitted with non-negative least squares on
measured configurations and have physical meaning: c is compute ms/token, s is
ms per MoE node, a is the inverse of the achieved fraction of device bandwidth
(1 = full device curve), b is ms of overhead per read. Prediction error is
reported with leave-one-out cross-validation; the target of section 26 is a
median absolute error below 10-15%. Failing it is a result.

Inputs are plain dictionaries so that the model can be fitted from runtime
traces (runner.py writes them) or from any other source with the same fields:
    ms_per_token, bytes_missed_per_token, reads_per_token, nodes_per_token,
    qd (mean queue depth when busy), controller_ms_per_token (optional).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from .costmodel import CostModel

FEATURES = ("compute", "sync_nodes", "io_bytes_over_bw", "reads")


def design_row(run: Dict[str, float], cost: CostModel) -> List[float]:
    bw = cost.bandwidth(max(1.0, float(run.get("qd", 1.0)))) / cost.efficiency  # device curve
    return [1.0,
            float(run.get("nodes_per_token", 0.0)),
            1000.0 * float(run.get("bytes_missed_per_token", 0.0)) / bw,   # ms at full device BW
            float(run.get("reads_per_token", 0.0))]


def nnls(X: Sequence[Sequence[float]], y: Sequence[float], iters: int = 5000) -> List[float]:
    """Non-negative least squares by projected coordinate descent (small p)."""
    p = len(X[0])
    w = [0.0] * p
    col_sq = [sum(r[j] * r[j] for r in X) for j in range(p)]
    resid = list(y)
    for _ in range(iters):
        change = 0.0
        for j in range(p):
            if col_sq[j] == 0:
                continue
            g = sum(X[i][j] * resid[i] for i in range(len(X)))
            new = max(0.0, w[j] + g / col_sq[j])
            d = new - w[j]
            if d:
                for i in range(len(X)):
                    resid[i] -= d * X[i][j]
                w[j] = new
                change = max(change, abs(d))
        if change < 1e-12:
            break
    return w


def fit(runs: Sequence[Dict[str, float]], cost: Optional[CostModel] = None) -> Dict[str, object]:
    cost = cost or CostModel()
    X = [design_row(r, cost) for r in runs]
    y = [float(r["ms_per_token"]) - float(r.get("controller_ms_per_token", 0.0)) for r in runs]
    w = nnls(X, y)
    return {"coefficients": dict(zip(FEATURES, w)), "n": len(runs)}


def predict(model: Dict[str, object], run: Dict[str, float],
            cost: Optional[CostModel] = None) -> float:
    cost = cost or CostModel()
    w = [model["coefficients"][f] for f in FEATURES]  # type: ignore[index]
    x = design_row(run, cost)
    return sum(a * b for a, b in zip(w, x)) + float(run.get("controller_ms_per_token", 0.0))


def evaluate(runs: Sequence[Dict[str, float]], cost: Optional[CostModel] = None
             ) -> Dict[str, object]:
    """Leave-one-out: fit on all runs but one, predict the left-out one."""
    cost = cost or CostModel()
    rows = []
    for i, r in enumerate(runs):
        train = [x for j, x in enumerate(runs) if j != i]
        if len(train) < len(FEATURES):
            break
        m = fit(train, cost)
        pred = predict(m, r, cost)
        meas = float(r["ms_per_token"])
        rows.append({"label": r.get("label", str(i)), "measured_ms": meas,
                     "predicted_ms": pred, "abs_pct_error": abs(pred - meas) / meas * 100})
    errs = sorted(x["abs_pct_error"] for x in rows)
    med = (errs[len(errs) // 2] if len(errs) % 2 else
           0.5 * (errs[len(errs) // 2 - 1] + errs[len(errs) // 2])) if errs else None
    return {"loo": rows, "median_abs_pct_error": med,
            "meets_target_15pct": (med is not None and med < 15.0),
            "full_fit": fit(runs, cost) if runs else None,
            "note": "needs at least 5 configurations for leave-one-out with 4 coefficients"}
