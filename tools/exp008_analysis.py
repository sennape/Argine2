"""EXP-008 — EWS as a predictor: deterministic analysis over existing EXP-007 data.

    python tools/exp008_analysis.py            # writes experiments/EXP-008-working-set/processed/

No model, no inference, no trace recording. Inputs (read-only):
* EXP-007 run-00-4G routing traces (raw/<campaign>/run-00-4G/trace.jsonl.gz)
* EXP-007 run manifests' expert-level hit rate (runtime-hit-rates.csv, qualified runs only)
* EXP-007 responses.jsonl of every qualified run (generation tok/s)

The procedure is the one frozen in the EXP-008 README ("Analysis plan,
frozen before predictor-comparison statistics were computed"). Constants
below must match it; the README records this file's SHA-256 at freeze time.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import math
import random
import statistics
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
E7 = ROOT / "experiments" / "EXP-007-router-characterization"
OUT = ROOT / "experiments" / "EXP-008-working-set" / "processed"

WORKLOADS = ["natural-language", "coding", "reasoning-math", "structured-json",
             "italian", "transition", "multi-turn-v2"]
MODELS = ["30b", "80b"]
BUDGETS = {"4G": 4 * 1024 ** 3, "2G": 2 * 1024 ** 3, "1.5G": 1536 * 1024 ** 2}
PRIMARY_SPLIT = "eval"          # methodology: claims on EVAL; DEV = replication (secondary)
TRUNC_DECODE_TOKENS = 320       # minimum common decode length across cells
DENNING_TAU = 16                # the Denning window reported by `research characterize`
BOOT_N = 2000
BOOT_SEED = 20261011
PREDICTORS = ["P0", "P1", "P2"]
TARGETS = ["gen_tok_s", "expert_level_hit_rate"]
EXCLUDED_SEGMENTS = ("warmup", "flush-sentinel")


# ------------------------------------------------------------- statistics --

def ranks(v: Sequence[float]) -> List[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0 + 1
        i = j + 1
    return r


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    rx, ry = ranks(x), ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return None if sx == 0 or sy == 0 else sxy / (sx * sy)


def ols(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    mx, my = statistics.mean(x), statistics.mean(y)
    sxx = sum((a - mx) ** 2 for a in x)
    if sxx == 0:
        return my, 0.0
    b = sum((a - mx) * (c - my) for a, c in zip(x, y)) / sxx
    return my - b * mx, b


def lowo_errors(rows: List[Dict[str, Any]], pred: str, target: str) -> Dict[str, Any]:
    """Leave-one-workload-out: log(target) = a + b log(predictor), OLS on the
    other 6 workloads (18 rows), predict the held-out workload's 3 rows."""
    errs, preds = [], []
    for w in sorted({r["workload"] for r in rows}):
        train = [r for r in rows if r["workload"] != w]
        test = [r for r in rows if r["workload"] == w]
        a, b = ols([math.log(r[pred]) for r in train], [math.log(r[target]) for r in train])
        for r in test:
            p = math.exp(a + b * math.log(r[pred]))
            errs.append(abs(p - r[target]))
            preds.append({"workload": w, "budget": r["budget"], "observed": r[target],
                          "predicted": p})
    mae = statistics.mean(errs)
    return {"mae": mae, "relative_mae": mae / statistics.mean(r[target] for r in rows),
            "n": len(errs), "predictions": preds}


def pair_counts(rows: List[Dict[str, Any]], pred: str, target: str) -> Dict[str, Any]:
    """At fixed budget, every workload pair: is the predictor ordering the same as
    the target ordering? Accuracy = (concordant + 0.5 ties) / n."""
    c = d = t = 0
    for bud in BUDGETS:
        sub = [r for r in rows if r["budget"] == bud]
        for r1, r2 in combinations(sub, 2):
            sp = (r1[pred] > r2[pred]) - (r1[pred] < r2[pred])
            st = (r1[target] > r2[target]) - (r1[target] < r2[target])
            if sp == 0 or st == 0:
                t += 1
            elif sp == st:
                c += 1
            else:
                d += 1
    n = c + d + t
    return {"concordant": c, "discordant": d, "ties": t, "n": n,
            "accuracy": (c + 0.5 * t) / n if n else None}


def compare(stats: Dict[str, Dict[str, Any]], x: str) -> Dict[str, str]:
    """Per family: 'X' (X strictly better than P0), 'P0' (P0 strictly better),
    'tie'. Pairs: X better iff accuracy > 0.5 (P0 is 0.5 by construction)."""
    out = {}
    rx, r0 = abs(stats[x]["rho"] or 0), abs(stats["P0"]["rho"] or 0)
    out["spearman"] = "X" if rx > r0 else ("P0" if r0 > rx else "tie")
    mx, m0 = stats[x]["mae"], stats["P0"]["mae"]
    out["fit"] = "X" if mx < m0 else ("P0" if m0 < mx else "tie")
    acc = stats[x]["pair_accuracy"]
    out["pairs"] = "X" if acc > 0.5 else ("P0" if acc < 0.5 else "tie")
    return out


def decide(per_model: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """Frozen rule. per_model[model][predictor] = {rho, mae, pair_accuracy}."""
    cmp = {m: {x: compare(per_model[m], x) for x in ("P1", "P2")} for m in per_model}

    def beats(m: str, x: str) -> bool:
        v = list(cmp[m][x].values())
        return v.count("X") >= 2 and "P0" not in v

    def p0_holds(m: str, x: str) -> bool:      # P0 matches or outperforms X in >= 2 families
        return sum(1 for v in cmp[m][x].values() if v in ("P0", "tie")) >= 2

    supported = [x for x in ("P1", "P2") if all(beats(m, x) for m in per_model)]
    falsified = all(p0_holds(m, x) for m in per_model for x in ("P1", "P2"))
    verdict = "SUPPORTED" if supported else ("FALSIFIED" if falsified else "INCONCLUSIVE")
    return {"verdict": verdict, "supported_by": supported, "comparisons": cmp}


# ------------------------------------------------------------------ data --

def _qualified() -> Dict[Tuple[str, str, str, str], List[Dict[str, str]]]:
    out: Dict[Tuple[str, str, str, str], List[Dict[str, str]]] = {}
    with open(E7 / "runtime-hit-rates.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["qualified"] == "True":
                out.setdefault((r["model"], r["workload"], r["split"], r["budget"]), []).append(r)
    return out


def run_tok_s(campaign: str, run: str) -> float:
    """Same formula as runner.cmd_trace: sum(predicted_n) / sum(predicted_ms) over
    the workload responses with predicted_per_second."""
    rs = [json.loads(l) for l in (E7 / "raw" / campaign / run / "responses.jsonl")
          .read_text(encoding="utf-8").splitlines() if l.strip()]
    dec = [r for r in rs if r.get("predicted_per_second")]
    return sum(r["predicted_n"] for r in dec) / (sum(r["predicted_ms"] for r in dec) / 1000.0)


def runs_csv_tok_s(campaign: str) -> Dict[str, float]:
    p = E7 / "raw" / campaign / "runs.csv"
    out: Dict[str, float] = {}
    if not p.exists():
        return out
    with open(p, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for d in sorted((E7 / "raw" / campaign).glob(f"run-{int(r['run']):02d}-*")):
                if r.get("gen_tok_s"):
                    out[d.name] = float(r["gen_tok_s"])
    return out


def trace_campaign(model: str, workload: str, split: str) -> str:
    cands = [c.name for c in (E7 / "raw").iterdir()
             if c.name.startswith(f"{model}-{workload}-{split}") and "replacement" not in c.name
             and "smoke" not in c.name and (c / "run-00-4G" / "trace.jsonl.gz").exists()]
    if model == "30b" and workload == "coding" and split == "dev":
        cands = ["30b-coding-dev"]
    if len(cands) != 1:
        raise RuntimeError(f"ambiguous or missing run-00-4G trace for {model} {workload} {split}: {cands}")
    return cands[0]


def working_set_predictors(trace, truncate: Optional[int]) -> Dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from argine.research import metrics  # noqa: E402
    from argine.research.trace import PHASE_DECODE  # noqa: E402
    dec = [i for i, s in enumerate(trace.steps)
           if s.phase == PHASE_DECODE and (s.segment or "") not in EXCLUDED_SEGMENTS]
    keep = set(dec[:truncate] if truncate else dec)
    t = copy.copy(trace)
    t.steps = [s for i, s in enumerate(trace.steps) if i in keep]
    g = metrics.global_working_set(t)
    eb = trace.meta.expert_bytes
    ws_bytes = 0.0
    for layer in t.layers:
        ws = metrics.working_set_mean(metrics.token_sets(t, layer, PHASE_DECODE), DENNING_TAU)
        ws_bytes += ws * (eb.get(layer) or eb.get(str(layer)) or 0)
    if not g.get("ews95_bytes") or not ws_bytes or not g.get("expert_bytes_total"):
        raise RuntimeError("expert sizes missing from the trace metadata: cannot express working sets in bytes")
    return {"decode_tokens": t.n_tokens(PHASE_DECODE), "ews95": g.get("ews95"),
            "ews95_bytes": g.get("ews95_bytes"), "denning_ws_bytes": ws_bytes,
            "expert_bytes_total": g.get("expert_bytes_total")}


def build_table() -> List[Dict[str, Any]]:
    sys.path.insert(0, str(ROOT))
    from argine.research.trace import Trace  # noqa: E402
    q = _qualified()
    rows: List[Dict[str, Any]] = []
    for model in MODELS:
        for split in ("eval", "dev"):
            for w in WORKLOADS:
                camp = trace_campaign(model, w, split)
                tr = Trace.load(E7 / "raw" / camp / "run-00-4G" / "trace.jsonl.gz")
                ws = working_set_predictors(tr, TRUNC_DECODE_TOKENS)
                full = working_set_predictors(tr, None)
                for bud, bb in BUDGETS.items():
                    runs = q.get((model, w, split, bud), [])
                    toks, srcs = [], []
                    for r in runs:
                        v = run_tok_s(r["campaign"], r["run"])
                        ref = runs_csv_tok_s(r["campaign"]).get(r["run"])
                        toks.append(v)
                        srcs.append(f"{r['campaign']}/{r['run']}"
                                    + ("" if ref is not None else " (no runs.csv: responses.jsonl)")
                                    + (f" [runs.csv {ref:.6f}]" if ref is not None else ""))
                    hits = [float(r["expert_level_hit_rate"]) for r in runs]
                    rows.append({
                        "model": model, "workload": w, "split": split, "budget": bud,
                        "budget_bytes": bb, "trace": f"{camp}/run-00-4G/trace.jsonl.gz",
                        "trace_decode_tokens_used": ws["decode_tokens"],
                        "expert_bytes_total": ws["expert_bytes_total"],
                        "ews95_bytes_320": ws["ews95_bytes"],
                        "denning_ws16_bytes_320": round(ws["denning_ws_bytes"], 1),
                        "P0": bb / ws["expert_bytes_total"],
                        "P1": bb / ws["ews95_bytes"],
                        "P2": bb / ws["denning_ws_bytes"],
                        "P1_full_trace": bb / full["ews95_bytes"],
                        "P2_full_trace": bb / full["denning_ws_bytes"],
                        "full_trace_decode_tokens": full["decode_tokens"],
                        "n_qualified_runs": len(runs),
                        "gen_tok_s_runs": " ".join(f"{x:.4f}" for x in toks),
                        "gen_tok_s": statistics.median(toks) if toks else None,
                        "expert_level_hit_rate_runs": " ".join(f"{x:.6f}" for x in hits),
                        "expert_level_hit_rate": statistics.median(hits) if hits else None,
                        "provenance": "; ".join(srcs),
                    })
    return rows


def analyse_split(rows: List[Dict[str, Any]], preds: Sequence[str]) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    for target in TARGETS:
        per_model: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for m in MODELS:
            rr = [r for r in rows if r["model"] == m]
            per_model[m] = {}
            for p in preds:
                f = lowo_errors(rr, p, target)
                pc = pair_counts(rr, p, target)
                per_model[m][p] = {
                    "rho": spearman([r[p] for r in rr], [r[target] for r in rr]),
                    "n": len(rr), "distinct_predictor_values": len({r[p] for r in rr}),
                    "mae": f["mae"], "relative_mae": f["relative_mae"],
                    "pair_accuracy": pc["accuracy"], "pairs": pc,
                    "fixed_budget_rho": {b: spearman([r[p] for r in rr if r["budget"] == b],
                                                     [r[target] for r in rr if r["budget"] == b])
                                         for b in BUDGETS},
                    "lowo_predictions": f["predictions"]}
        res[target] = {"per_model": per_model}
        if list(preds) == PREDICTORS:
            res[target]["decision"] = decide(per_model)
    return res


def bootstrap(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rng = random.Random(BOOT_SEED)
    out = []
    for target in TARGETS:
        for m in MODELS:
            rr = [r for r in rows if r["model"] == m]
            by_w = {w: [r for r in rr if r["workload"] == w] for w in WORKLOADS}
            draws: Dict[str, List[float]] = {k: [] for k in ("P0", "P1", "P2", "P1-P0", "P2-P0")}
            for _ in range(BOOT_N):
                sample = [r for w in rng.choices(WORKLOADS, k=len(WORKLOADS)) for r in by_w[w]]
                rho = {p: spearman([r[p] for r in sample], [r[target] for r in sample]) for p in PREDICTORS}
                if any(v is None for v in rho.values()):
                    continue
                for p in PREDICTORS:
                    draws[p].append(rho[p])
                draws["P1-P0"].append(abs(rho["P1"]) - abs(rho["P0"]))
                draws["P2-P0"].append(abs(rho["P2"]) - abs(rho["P0"]))
            for k, v in draws.items():
                v.sort()
                out.append({"target": target, "model": m, "quantity": f"rho {k}" if "-" not in k
                            else f"|rho {k.split('-')[0]}| - |rho P0|", "resamples": len(v),
                            "p2.5": round(v[int(0.025 * len(v))], 4), "median": round(statistics.median(v), 4),
                            "p97.5": round(v[int(0.975 * len(v)) - 1], 4)})
    return out


def _csv(rows: List[Dict[str, Any]]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    table = build_table()
    primary = [r for r in table if r["split"] == PRIMARY_SPLIT]
    dev = [r for r in table if r["split"] == "dev"]
    res_primary = analyse_split(primary, PREDICTORS)
    res_dev = analyse_split(dev, PREDICTORS)
    expl_full = analyse_split([{**r, "P1": r["P1_full_trace"], "P2": r["P2_full_trace"]}
                               for r in primary], PREDICTORS)
    pooled = {t: {p: spearman([r[p] for r in primary], [r[t] for r in primary]) for p in PREDICTORS}
              for t in TARGETS}
    boot = bootstrap(primary)

    def fmt(x):
        return None if x is None else round(x, 6)

    (OUT / "predictor_table.csv").write_text(_csv([{k: (fmt(v) if isinstance(v, float) else v)
                                                    for k, v in r.items()} for r in table]),
                                             encoding="utf-8", newline="")
    corr, fits, pairs = [], [], []
    for label, res in (("primary (EVAL)", res_primary), ("replication (DEV)", res_dev),
                       ("exploratory: full-trace P1/P2 (EVAL)", expl_full)):
        for t in TARGETS:
            for m in MODELS:
                for p in PREDICTORS:
                    s = res[t]["per_model"][m][p]
                    corr.append({"analysis": label, "target": t, "model": m, "predictor": p,
                                 "scope": "all 21 rows", "n": s["n"],
                                 "distinct_predictor_values": s["distinct_predictor_values"],
                                 "spearman_rho": fmt(s["rho"])})
                    for b, v in s["fixed_budget_rho"].items():
                        corr.append({"analysis": label, "target": t, "model": m, "predictor": p,
                                     "scope": f"fixed budget {b} (7 workloads)", "n": 7,
                                     "distinct_predictor_values": len({r[p] for r in (primary if 'DEV' not in label else dev) if r['model'] == m and r['budget'] == b}),
                                     "spearman_rho": fmt(v)})
                    fits.append({"analysis": label, "target": t, "model": m, "predictor": p,
                                 "method": "LOWO, OLS log(target)~log(predictor)", "n_predictions": 21,
                                 "mae": fmt(s["mae"]), "relative_mae": fmt(s["relative_mae"])})
                    pairs.append({"analysis": label, "target": t, "model": m, "predictor": p,
                                  **s["pairs"], "accuracy": fmt(s["pairs"]["accuracy"])})
    (OUT / "correlations.csv").write_text(_csv(corr), encoding="utf-8", newline="")
    (OUT / "fit_errors.csv").write_text(_csv(fits), encoding="utf-8", newline="")
    (OUT / "falsification_pairs.csv").write_text(_csv(pairs), encoding="utf-8", newline="")
    (OUT / "bootstrap.csv").write_text(_csv(boot), encoding="utf-8", newline="")

    def strip(res):
        return {t: {"decision": res[t].get("decision"),
                    "per_model": {m: {p: {k: v for k, v in s.items() if k not in ("lowo_predictions",)}
                                      for p, s in res[t]["per_model"][m].items()}
                                  for m in MODELS}} for t in TARGETS}
    summary = {
        "experiment": "EXP-008-working-set",
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "constants": {"primary_split": PRIMARY_SPLIT, "truncate_decode_tokens": TRUNC_DECODE_TOKENS,
                      "denning_tau": DENNING_TAU, "budgets": BUDGETS, "bootstrap_n": BOOT_N,
                      "bootstrap_seed": BOOT_SEED},
        "observations": {"primary": len(primary), "replication_dev": len(dev),
                         "cells_with_fewer_than_3_runs": [
                             {k: r[k] for k in ("model", "workload", "split", "budget", "n_qualified_runs")}
                             for r in table if r["n_qualified_runs"] != 3]},
        "primary_throughput_verdict": res_primary["gen_tok_s"]["decision"]["verdict"],
        "secondary_hit_rate_verdict": res_primary["expert_level_hit_rate"]["decision"]["verdict"],
        "replication_dev_verdicts": {t: res_dev[t]["decision"]["verdict"] for t in TARGETS},
        "primary": strip(res_primary), "replication_dev": strip(res_dev),
        "exploratory_full_trace": strip(expl_full),
        "exploratory_pooled_spearman_eval": pooled,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                      encoding="utf-8", newline="")
    print(json.dumps({"primary_throughput_verdict": summary["primary_throughput_verdict"],
                      "secondary_hit_rate_verdict": summary["secondary_hit_rate_verdict"],
                      "replication_dev": summary["replication_dev_verdicts"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
