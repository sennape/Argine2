"""Derive the data files of the historical experiments (EXP-001..006, 011..018)
from Argine's imported raw logs (data/raw/argine-v1).

    python tools/build_historical_data.py

For every experiment it writes:
    experiments/<id>/raw/records.jsonl   the exact raw records used (subset of runs.jsonl
                                         or perplexity.jsonl, unchanged)
    experiments/<id>/results.csv         one row per configuration: repetitions and
                                         their summary statistics
Selection is by timestamp window + configuration, written out below, so that
anyone can check which raw record supports which number.

Nothing is interpolated or corrected. Where the README of Argine quotes a number
that is not in the logs (e.g. the store's internal cache counters, which the logs
of that period did not record), the experiment README says so.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from argine.research.io import write_csv # noqa: E402
from argine.research.stats import summarize  # noqa: E402

RAW = ROOT / "data" / "raw" / "argine-v1"
EXP = ROOT / "experiments"


def load(name):
    return [json.loads(l) for l in (RAW / name).read_text(encoding="utf-8").splitlines() if l.strip()]


RUNS = load("runs.jsonl")
PPL = load("perplexity.jsonl")


def between(r, a, b):
    return a <= r["ts_iso"] <= b


def gen(r):
    return [x.get("gen_tok_s") for x in (r.get("metrics") or {}).get("runs", [])]


def write(exp_id, records, rows, notes=None):
    d = EXP / exp_id
    (d / "raw").mkdir(parents=True, exist_ok=True)
    with open(
        d / "raw" / "records.jsonl",
        "w",
        encoding="utf-8",
        newline="\n",
    ) as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    write_csv(rows, d / "results.csv")
    if notes:
        (d / "processed").mkdir(exist_ok=True)
        with open(
            d / "processed" / "notes.json",
            "w",
            encoding="utf-8",
            newline="\n",
        ) as fh:
            fh.write(json.dumps(notes, indent=2) + "\n")


def speed_row(label, recs, **extra):
    vals = [v for r in recs for v in gen(r) if v is not None]
    s = summarize(vals)
    return {"config": label, "n_reps": s.get("n"), "gen_tok_s_mean": s.get("mean"),
            "gen_tok_s_median": s.get("median"), "sd": s.get("sd"), "cv": s.get("cv"),
            "ci95_low": s.get("ci95_low"), "ci95_high": s.get("ci95_high"),
            "values": " ".join(f"{v:.2f}" for v in vals),
            "records": " ".join(r["ts_iso"] for r in recs), **extra}


def model_is(r, name):
    return (r.get("model") or {}).get("name", "").startswith(name)


def threads(r):
    return (r.get("plan") or {}).get("threads")


def budget(r):
    b = r.get("expert_budget") or (r.get("plan") or {}).get("expert_cache_budget")
    return int(b) if b else None


def build_(r):
    return (r.get("runtime") or {}).get("build")


def exponent(b_hi, v_hi, b_lo, v_lo):
    """d log(speed) / d log(residency) between two points (Argine's method)."""
    return math.log(v_hi / v_lo) / math.log(b_hi / b_lo)


def main():
    # EXP-001 thread sweep -------------------------------------------------
    recs, rows = [], []
    for a, b, label in (("2026-09-27T15:54", "2026-09-27T15:57", "Qwen3-8B dense Q4_K_M, official b11214, sweep 1"),
                        ("2026-09-27T16:08", "2026-09-27T16:11", "Qwen3-8B dense Q4_K_M, official b11214, sweep 2"),
                        ("2026-09-30T18:04", "2026-09-30T18:11", "Qwen3-30B-A3B IQ2_XXS resident, old patched build (OpenMP)"),
                        ("2026-09-30T19:32", "2026-09-30T19:36", "Qwen3-30B-A3B Q4_K_M + store 6G, old patched build (OpenMP)")):
        sel = [r for r in RUNS if r["event"] == "benchmark_sweep" and between(r, a, b + ":59")]
        recs += sel
        for t in sorted({threads(r) for r in sel}):
            rows.append(speed_row(f"{label} | threads={t}", [r for r in sel if threads(r) == t],
                                  threads=t, build=build_(sel[0])))
    write("EXP-001-thread-sweep", recs, rows)

    # EXP-002 repack ----------------------------------------------------------
    sel = [r for r in RUNS if r["event"] == "benchmark" and r["ts_iso"][:16] in
           ("2026-09-27T16:27", "2026-09-27T16:35")]
    rows = [speed_row("repack ON (default llama.cpp)", [sel[0]], ram_peak=sel[0]["metrics"]["resources"]["ram_peak"]),
            speed_row("repack OFF (Argine default)", [sel[1]], ram_peak=sel[1]["metrics"]["resources"]["ram_peak"])]
    write("EXP-002-repack", sel, rows, {"order": "AAA then BBB (not interleaved)"})

    # EXP-003 direct I/O vs mmap: diskbench + store_bench --------------------
    sel = [r for r in RUNS if r["event"] in ("diskbench", "store_bench")]
    rows = []
    for r in sel:
        m = r.get("metrics") or {}
        for x in m.get("risultati", []) or []:
            rows.append({"ts_iso": r["ts_iso"], "event": "diskbench", "path": x["kind"],
                         "slice_bytes": x["slice"], "threads": x["threads"], "mb_s": x["mb_s"],
                         "latency_ms": x.get("latency_ms"), "cache_ratio": x.get("cache_ratio")})
        for x in m.get("strade", []) or []:
            rows.append({"ts_iso": r["ts_iso"], "event": "store_bench", "path": x["label"],
                         "threads": m.get("threads"), "mb_s": x["mb_s"],
                         "us_per_expert": x.get("us_per_expert")})
    write("EXP-003-direct-io", sel, rows)

    # EXP-004 three read layers, one binary ---------------------------------
    sel = [r for r in RUNS if r["event"] == "benchmark_sweep_budget" and r.get("modo_lettura")
           and model_is(r, "Qwen3 30B") and between(r, "2026-10-03T13:53", "2026-10-03T14:18")]
    rows = []
    for b in sorted({budget(r) for r in sel}, reverse=True):
        for mode in ("sincrono", "blocco", "asincrono"):
            rr = [r for r in sel if budget(r) == b and r["modo_lettura"] == mode]
            rows.append(speed_row(f"budget={b / 2**30:g}G mode={mode}", rr, budget_bytes=b,
                                  read_mode=mode, resident_fraction=rr[0]["frazione_residente"]))
    write("EXP-004-async-store", sel, rows, {"order": "per budget: sincrono, blocco, asincrono "
                                                      "(same order at every budget, not rotated)"})

    # EXP-005 30B budget sweeps (all sweeps, grouped by binary) --------------
    sel = [r for r in RUNS if r["event"] == "benchmark_sweep_budget" and model_is(r, "Qwen3 30B")
           and not r.get("modo_lettura")]
    rows = []
    groups = {}
    for r in sel:
        groups.setdefault((r["ts_iso"][:13], build_(r)), []).append(r)
    for (hour, bld), rr in sorted(groups.items()):
        for b in sorted({budget(x) for x in rr}, reverse=True):
            pts = [x for x in rr if budget(x) == b]
            rows.append(speed_row(f"{hour} {bld} budget={b / 2**30:g}G", pts, session=hour,
                                  build=bld, budget_bytes=b,
                                  resident_fraction=pts[0].get("frazione_residente")))
    write("EXP-005-30b-budget-sweep", sel, rows)

    # EXP-006 80B --------------------------------------------------------------
    sel = [r for r in RUNS if model_is(r, "Qwen3-Next-80B") and gen(r)]
    rows = []
    for r in sel:
        rows.append(speed_row(f"{r['ts_iso']} {r['event']} budget={budget(r) / 2**30:g}G "
                              f"mode={r.get('modo_lettura') or 'not recorded'}", [r],
                              build=build_(r), budget_bytes=budget(r),
                              resident_fraction=r.get("frazione_residente"),
                              ram_peak=r["metrics"]["resources"]["ram_peak"]))
    curve = [r for r in sel if r["event"] == "benchmark_sweep_budget"]
    hi, lo = curve[0], curve[-1]
    vhi = summarize(gen(hi))["median"]
    vlo = summarize(gen(lo))["median"]
    write("EXP-006-80b-budget-sweep", sel, rows,
          {"exponent_median_endpoints": exponent(hi["frazione_residente"], vhi,
                                                 lo["frazione_residente"], vlo),
           "method": "Argine's two-endpoint exponent on medians; Argine reported 0.384 +- 0.043"})

    # EXP-011 binary rebuild -------------------------------------------------
    sel = [r for r in RUNS if r["event"] == "benchmark" and r["ts_iso"][:16] in
           ("2026-10-01T19:41", "2026-10-01T19:46", "2026-10-01T20:22")]
    rows = [speed_row(f"IQ2_S resident, t4, build={build_(r)}", [r], build=build_(r)) for r in sel]
    write("EXP-011-binary-rebuild", sel, rows)

    # EXP-012 dense 14B control ------------------------------------------------
    sel = [r for r in RUNS if r["event"] == "benchmark" and r["ts_iso"][:16] in
           ("2026-10-02T21:20", "2026-10-02T21:23", "2026-10-01T20:22")]
    ppl = [p for p in PPL if p["chunk"] == 60 and ("14B" in p["model"] or "IQ2S" in p["model"])]
    rows = [speed_row(f"{(r['model'] or {}).get('name')} {(r['model'] or {}).get('quantization')} "
                      f"{r['ts_iso']}", [r]) for r in sel]
    rows += [{"config": f"perplexity {Path(p['model'].replace(chr(92), '/')).name}", "ppl": p["ppl"],
              "ppl_err": p["errore"], "signature": p["firma"]} for p in ppl]
    write("EXP-012-dense-14b-control", sel + ppl, rows)

    # EXP-013 quality per bit + EXP-017 calibration text -------------------
    rows = [{"model_file": Path(p["model"].replace(chr(92), "/")).name, "ppl": p["ppl"],
             "ppl_err": p["errore"], "signature": p["firma"], "text_hash": p["testo_hash"]}
            for p in PPL]
    write("EXP-013-quality-per-bit", [p for p in PPL if p["chunk"] == 60], [r for r in rows if "chunk=60" in r["signature"]])
    write("EXP-017-calibration-text", PPL, rows)

    # EXP-014 conversation -----------------------------------------------------
    sel = [r for r in RUNS if r["event"] == "benchmark_conversazione"]
    rows = []
    for r in sel:
        m = r["metrics"]
        for key in ("con_cache", "senza_cache"):
            for t in m.get(key, []) or []:
                rows.append({"prefix_reuse": key == "con_cache", **t})
    write("EXP-014-conversation-prefix-reuse", sel, rows)

    # EXP-016 co-activation reorder ---------------------------------------------
    sel = [r for r in RUNS if r["event"] == "reorder" or (r["event"] == "benchmark" and r["ts_iso"][:16] in
                                                            ("2026-09-27T21:00", "2026-09-27T22:45", "2026-09-27T22:47"))]
    rows = [speed_row(f"{r['ts_iso']} {(r['model'] or {}).get('path', '').split(chr(92))[-1]}", [r])
            for r in sel if r["event"] == "benchmark"]
    write("EXP-016-coactivation-reorder", sel, rows,
          {"warning": "the raw records do not label which benchmark used the reordered file; "
                      "the model path field identifies it where present"})

    # EXP-018 reproducibility audit -----------------------------------------
    pairs = []
    by_cfg = {}
    for r in RUNS:
        if r["event"] not in ("benchmark", "benchmark_sweep_budget") or not gen(r):
            continue
        plan = r.get("plan") or {}
        # identical plan only: model, quantisation, binary, threads, store on/off,
        # expert budget (store) or expert cache (mmap), context, prompt
        key = ((r.get("model") or {}).get("name"), (r.get("model") or {}).get("quantization"),
               build_(r), threads(r), bool(plan.get("expert_store")), budget(r), plan.get("ctx"),
               r.get("prompt_chars"))
        by_cfg.setdefault(key, []).append(r)
    rows = []
    for key, rr in by_cfg.items():
        if len(rr) < 2:
            continue
        meds = [summarize(gen(r))["median"] for r in rr]
        within = [summarize(gen(r))["cv"] for r in rr if len(gen(r)) > 1]
        rows.append({"model": key[0], "quant": key[1], "build": key[2], "threads": key[3],
                     "store": key[4], "budget_bytes": key[5], "ctx": key[6],
                     "read_modes": " ".join(sorted({str(r.get("modo_lettura") or "unrecorded")
                                                    for r in rr})),
                     "sessions": len(rr),
                     "median_min": min(meds), "median_max": max(meds),
                     "between_run_spread": (max(meds) - min(meds)) / min(meds),
                     "within_run_cv_max": max(within) if within else None,
                     "records": " ".join(r["ts_iso"] for r in rr)})
        pairs += rr
    rows.sort(key=lambda x: -x["between_run_spread"])
    write("EXP-018-reproducibility-audit", pairs, rows)
    print("historical experiment data written")


if __name__ == "__main__":
    main()
