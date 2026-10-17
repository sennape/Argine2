"""EXP-010: DEV-selected, frozen, EVAL-tested retrospective holdout (replay only).

    python tools/exp010_analysis.py verify-pairs [--out CSV]
    python tools/exp010_analysis.py freeze                       (once, before any real replay)
    python tools/exp010_analysis.py check-freeze
    python tools/exp010_analysis.py select  --dev DEV_ROOT --out selection.json
    python tools/exp010_analysis.py show    --selection selection.json
    python tools/exp010_analysis.py script-eval-adaptive --selection selection.json --out step5.bat
    python tools/exp010_analysis.py analyze --selection selection.json --eval EVAL_ROOT
                                            --eval-adaptive EVAL_ADAPTIVE_ROOT --out PROCESSED

The protocol is in experiments/EXP-010-adaptive-allocation/README.md, between the
PROTOCOL-BEGIN / PROTOCOL-END markers; its SHA-256 (LF-normalised text between the
markers) is frozen in protocol_freeze.json together with this tool and the replay code.

Leakage: B, RETROSPECTIVE HOLDOUT. EVAL traces and the EVAL outcomes of every adaptive
candidate were inspected during EXP-007. This is not a preregistration.

`select` reads DEV outputs only; it has no EVAL argument and never opens an EVAL path.
`analyze` refuses to run without a selection.json whose canonical hash verifies and
whose provenance matches the frozen code. The closure rule is directional and has no
effect-size threshold; the 0.5 / 1 / 2% sensitivity table is descriptive only.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]

EXPERIMENT_ID = "EXP-010-adaptive-allocation"
PROTOCOL_VERSION = "EXP010-P1a (pre-execution amendment, frozen 2026-10-11)"
LEAKAGE = "B - RETROSPECTIVE HOLDOUT"
FRAMING = "DEV-selected, frozen, EVAL-tested retrospective holdout"
CLOSURE_RULE_ID = "EXP010-CR1-directional (no effect-size threshold)"
COMMIT_NOTE = "Argine2 commit unavailable / repository not a git checkout"

EXP_REL = "experiments/EXP-010-adaptive-allocation"
RAW_REL = "experiments/EXP-007-router-characterization/raw"
PRIOR_REL = "experiments/EXP-007-router-characterization/analysis-v3"
RHR_REL = "experiments/EXP-007-router-characterization/runtime-hit-rates.csv"
README_REL = EXP_REL + "/README.md"
FREEZE_REL = EXP_REL + "/protocol_freeze.json"
MARK_BEGIN = "<!-- EXP-010 PROTOCOL-BEGIN -->"
MARK_END = "<!-- EXP-010 PROTOCOL-END -->"

CODE_FILES = ("argine/research/controller.py", "argine/research/analysis.py",
              "argine/research/replay.py", "argine/research/policies.py",
              "argine/research/allocator.py", "argine/research/elasticity.py",
              "argine/research/costmodel.py", "argine/research/cli.py",
              "argine/research/trace.py", "argine/research/metrics.py")

MODELS = ("30b", "80b")
FAMILIES = ("natural-language", "coding", "reasoning-math", "structured-json", "italian",
            "transition", "multi-turn-v2")
SUBGROUPS = {"transition": ("transition",), "multi-turn": ("multi-turn-v2",)}
BUDGETS = {"4G": 4 << 30, "2G": 2 << 30, "1.5G": 3 << 29}
BUDGET_BY_BYTES = {v: k for k, v in BUDGETS.items()}
N_CELLS = len(FAMILIES) * len(BUDGETS)          # 21 per model
HALF = N_CELLS / 2.0                            # 10.5
TIE_DELTA_G = 1e-4
SENSITIVITY = (0.005, 0.01, 0.02)

# The frozen DEV candidate space (id, adaptive.csv variant, epoch, pinned fraction).
CANDIDATES = (
    {"id": "e32", "variant": "adaptive-e32", "epoch": 32, "pinned": 0.0},
    {"id": "e64", "variant": "adaptive-e64", "epoch": 64, "pinned": 0.0},
    {"id": "e128", "variant": "adaptive-e128", "epoch": 128, "pinned": 0.0},
    {"id": "e64-pin25", "variant": "adaptive-e64-pin25", "epoch": 64, "pinned": 0.25},
    {"id": "e64-pin50", "variant": "adaptive-e64-pin50", "epoch": 64, "pinned": 0.5},
)
CAND_BY_ID = {c["id"]: c for c in CANDIDATES}
CAND_BY_VARIANT = {c["variant"]: c for c in CANDIDATES}
UNIFORM = "per-layer-lru-uniform"
DIAGNOSTIC_VARIANTS = {"global-lru", "ablate-frozen", "ablate-freq-estimator",
                       "ablate-no-cost-weighting", "ablate-no-hysteresis"}
FIXED_PARAMETERS = {"decay": 0.5, "hysteresis": 0.10, "max_move": 0.05, "estimator": "stack",
                    "cost_weighting": True, "stability_epochs": 4, "min_slots": "top_k",
                    "initial_allocation": "uniform", "cost_model": "default CostModel"}

# EVAL roles in policy_comparison.csv (compare --train DEV).
ROLE_POLICY = {"global_lru": "global-lru", "uniform": UNIFORM, "twoq": "2q",
               "static_dev": "layer-aware-static",
               "ref_static_eval_insample": "oracle-static-lru", "ref_belady": "belady-global"}
COMPARISON_CLASS = {"global_lru": "no-fit baseline", "uniform": "no-fit baseline",
                    "twoq": "online baseline",
                    "static_dev": "DEV-fitted workload-family-informed",
                    "adaptive_sel": "online replay controller; DEV-selected configuration",
                    "ref_static_eval_insample": "reference", "ref_belady": "reference"}
TRAIN_INDEPENDENT = ("global-lru", "sharded-lru-16-expert", UNIFORM, "lfu", "2q",
                     "least-stale-lru", "least-stale-mru", "adaptive-e64", "belady-global",
                     "oracle-static-lru")


def _pairs() -> Dict[Tuple[str, str], Tuple[str, str]]:
    """(model, family) -> (DEV campaign, EVAL campaign); verified folder names."""
    p = {}
    for m in MODELS:
        for f in FAMILIES:
            if f == "multi-turn-v2":
                p[(m, f)] = (f"{m}-multi-turn-v2-dev", f"{m}-multi-turn-v2-eval")
            elif m == "30b" and f == "coding":
                p[(m, f)] = ("30b-coding-dev", "30b-coding-eval-v2")      # DEV: trace format v1
            else:
                p[(m, f)] = (f"{m}-{f}-dev-v2", f"{m}-{f}-eval-v2")
    return p


PAIRS = _pairs()


def prior_cell(model: str, family: str, split: str) -> str:
    """analysis-v3 folder of the same run-00-4G trace (for determinism_check only)."""
    if split == "dev" and family == "coding":
        return "run-00-4G" if model == "30b" else "80b-run-00-4G"
    if split == "dev" and family == "transition" and model == "30b":
        return "30b-transition-run-00-4G"
    return f"{model}-{family}-{split}-run-00-4G"


def trace_rel(campaign: str) -> str:
    return f"{RAW_REL}/{campaign}/run-00-4G/trace.jsonl.gz"


class ProtocolError(Exception):
    pass


# ------------------------------------------------------------------ hashing ---

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def protocol_text(readme: Path) -> str:
    t = readme.read_text(encoding="utf-8").replace("\r\n", "\n")
    if t.count(MARK_BEGIN) != 1 or t.count(MARK_END) != 1:
        raise ProtocolError(f"{readme}: protocol markers missing or duplicated")
    a = t.index(MARK_BEGIN) + len(MARK_BEGIN)
    b = t.index(MARK_END)
    if b <= a:
        raise ProtocolError("protocol markers out of order")
    return t[a:b]


def protocol_sha(readme: Path) -> str:
    return sha256_bytes(protocol_text(readme).encode("utf-8"))


def canonical_sha(payload) -> str:
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_bytes(s.encode("utf-8"))


CANONICAL_METHOD = ("sha256 of json.dumps(selection, sort_keys=True, separators=(',', ':'), "
                    "ensure_ascii=False) encoded as UTF-8; timestamps live outside 'selection'")


def current_hashes(root: Path, tool_path: Optional[Path] = None) -> Dict:
    return {"protocol_sha256": protocol_sha(root / README_REL),
            "tool_sha256": sha256_file(tool_path or Path(__file__).resolve()),
            "code_sha256": {f: sha256_file(root / f) for f in CODE_FILES}}


def freeze(root: Path, tool_path: Optional[Path] = None) -> Dict:
    p = root / FREEZE_REL
    if p.exists():
        raise ProtocolError(f"{p} exists: the protocol is already frozen (never re-freeze "
                            "silently; record an amendment instead)")
    h = current_hashes(root, tool_path)
    doc = {"experiment": EXPERIMENT_ID, "protocol_version": PROTOCOL_VERSION,
           "leakage": LEAKAGE, "framing": FRAMING, "closure_rule": CLOSURE_RULE_ID,
           "protocol_hash_method": "sha256 of the UTF-8, LF-normalised README text strictly "
                                   f"between '{MARK_BEGIN}' and '{MARK_END}'",
           "code_hash_method": "sha256 of the raw file bytes",
           "argine2_commit": COMMIT_NOTE, **h,
           "frozen_utc": _utc()}
    p.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return doc


def check_freeze(root: Path, tool_path: Optional[Path] = None) -> Tuple[Dict, List[str]]:
    p = root / FREEZE_REL
    if not p.exists():
        raise ProtocolError(f"{p} missing: freeze the protocol before any real replay")
    fz = json.loads(p.read_text(encoding="utf-8"))
    h = current_hashes(root, tool_path)
    bad = []
    for k in ("protocol_sha256", "tool_sha256"):
        if fz.get(k) != h[k]:
            bad.append(f"{k}: frozen {fz.get(k)} != current {h[k]}")
    for f, v in h["code_sha256"].items():
        if fz.get("code_sha256", {}).get(f) != v:
            bad.append(f"{f}: frozen {fz.get('code_sha256', {}).get(f)} != current {v}")
    if fz.get("protocol_version") != PROTOCOL_VERSION:
        bad.append("protocol_version differs from the tool")
    return fz, bad


def _utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------- I/O ---

def read_csv(p: Path) -> List[Dict[str, str]]:
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_json(p: Path) -> Dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write_csv(rows: List[Dict], p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    cols: List[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _rel(p: Path, root: Path) -> str:
    try:
        return Path(p).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return Path(p).as_posix()


def family_of(workload: Optional[str]) -> str:
    return (workload or "").split("@", 1)[0]


def model_of(meta_model: Optional[str]) -> Optional[str]:
    s = (meta_model or "").lower()
    if "30b" in s:
        return "30b"
    if "80b" in s:
        return "80b"
    return None


# ------------------------------------------------------------------ stats ---

def geomean(xs: Sequence[float]) -> float:
    xs = list(xs)
    if not xs:
        raise ValueError("geometric mean of an empty sequence")
    if any(x <= 0 for x in xs):
        raise ValueError("geometric mean needs positive ratios")
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def ratio(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        raise ProtocolError(f"miss counts must be positive (got {a}, {b})")
    return a / b


def _sign(g: float) -> int:
    return (g > 1.0) - (g < 1.0)


def wins(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    w = sum(1 for x, y in zip(a, b) if x < y)
    t = sum(1 for x, y in zip(a, b) if x == y)
    l = sum(1 for x, y in zip(a, b) if x > y)
    return {"wins": w, "ties": t, "losses": l, "W": w + 0.5 * t}


def classify_model(g_pool: float, w: float, g_uniform: float) -> str:
    """Frozen model-level rule (MIXED is internal, never a registry status)."""
    if g_pool < 1.0 and w > HALF and g_uniform <= 1.0:
        return "ADAPTIVE_BETTER"
    if g_pool >= 1.0 and w <= HALF:
        return "STATIC_BETTER_OR_EQUAL"
    return "MIXED"


def verdict(per_model: Dict[str, Dict]) -> str:
    """Frozen experiment rule over both models."""
    labels = [per_model[m]["classification"] for m in MODELS]
    if all(x == "ADAPTIVE_BETTER" for x in labels):
        return "SUPPORTED"
    if all(x == "STATIC_BETTER_OR_EQUAL" for x in labels):
        return "FALSIFIED"
    if all(per_model[m]["G_uniform"] > 1.0 for m in MODELS):
        return "FALSIFIED"
    return "INCONCLUSIVE"


def sensitivity_rows(g_pool: float, model: str) -> List[Dict]:
    out = []
    for t in SENSITIVITY:
        out.append({"model": model, "threshold": t, "G_pool": g_pool,
                    "adaptive_fewer_misses_by_at_least_t": g_pool <= 1.0 - t,
                    "static_fewer_misses_by_at_least_t": g_pool >= 1.0 / (1.0 - t),
                    "within_plus_minus_t": (1.0 - t) < g_pool < 1.0 / (1.0 - t),
                    "affects_registry_verdict": False})
    return out


# -------------------------------------------------------------- DEV select ---

def _check_manifest(man: Dict, split: str, model: str, family: str, expected_sha: Optional[str],
                    where: str, train: bool = False) -> None:
    key = "train" if train else "trace"
    t = man.get(key)
    if not t:
        raise ProtocolError(f"{where}: manifest has no '{key}' identity")
    meta = t.get("meta", {})
    if meta.get("split") != split:
        raise ProtocolError(f"{where}: {key} split is {meta.get('split')!r}, expected {split!r}")
    if family_of(meta.get("workload")) != family:
        raise ProtocolError(f"{where}: {key} workload {meta.get('workload')!r} is not {family!r}")
    if model_of(meta.get("model")) != model:
        raise ProtocolError(f"{where}: {key} model {meta.get('model')!r} is not {model!r}")
    if expected_sha is not None and t.get("sha256") != expected_sha:
        raise ProtocolError(f"{where}: {key} sha256 {t.get('sha256')} differs from the raw "
                            f"EXP-007 trace {expected_sha}")
    budgets = sorted(int(b) for b in man.get("budgets_bytes", []))
    if budgets != sorted(BUDGETS.values()):
        raise ProtocolError(f"{where}: budgets {budgets} are not 4G,2G,1.5G")
    if man.get("phase") not in (None, "decode"):
        raise ProtocolError(f"{where}: phase {man.get('phase')!r} is not decode")


def _index_rows(rows: Iterable[Dict[str, str]], name_col: str, where: str,
                allowed: Optional[set] = None) -> Dict[Tuple[str, str], Dict[str, str]]:
    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    for r in rows:
        name = r.get(name_col) or r.get("policy")
        if not name:
            continue
        if allowed is not None and name.startswith("adaptive-") and name not in allowed:
            raise ProtocolError(f"{where}: unknown candidate variant {name!r} (the frozen grid "
                                "has five candidates)")
        bb = r.get("budget_bytes")
        if bb in (None, ""):
            continue
        b = BUDGET_BY_BYTES.get(int(float(bb)))
        if b is None:
            raise ProtocolError(f"{where}: unexpected budget {bb}")
        if str(r.get("feasible", "True")) != "True":
            raise ProtocolError(f"{where}: {name} at {b} is infeasible")
        k = (name, b)
        if k in out:
            raise ProtocolError(f"{where}: duplicate row {name} at {b}")
        out[k] = r
    return out


def load_dev_cells(dev_root: Path, raw_root: Optional[Path], repo: Path) -> Tuple[Dict, Dict]:
    """{model: {(family, budget): {variant: row}}} and the DEV source provenance."""
    cells: Dict[str, Dict] = {m: {} for m in MODELS}
    sources: Dict[str, Dict] = {}
    allowed = set(CAND_BY_VARIANT)
    for (m, f), (dev, _ev) in PAIRS.items():
        d = dev_root / dev
        csvp, manp = d / "adaptive.csv", d / "adaptive_manifest.json"
        if not csvp.exists() or not manp.exists():
            raise ProtocolError(f"missing DEV cell {m}/{f}: {d} needs adaptive.csv and "
                                "adaptive_manifest.json")
        man = read_json(manp)
        exp_sha = sha256_file(raw_root / dev / "run-00-4G" / "trace.jsonl.gz") if raw_root else None
        _check_manifest(man, "dev", m, f, exp_sha, str(d))
        idx = _index_rows(read_csv(csvp), "variant", str(csvp), allowed)
        for b in BUDGETS:
            need = [UNIFORM] + [c["variant"] for c in CANDIDATES]
            miss = [v for v in need if (v, b) not in idx]
            if miss:
                raise ProtocolError(f"{csvp}: missing {miss} at {b}")
            cells[m][(f, b)] = {v: idx[(v, b)] for v in need}
        sources[dev] = {"model": m, "family": f, "adaptive_csv": _rel(csvp, repo),
                        "adaptive_csv_sha256": sha256_file(csvp),
                        "adaptive_manifest_sha256": sha256_file(manp),
                        "trace_sha256": man["trace"]["sha256"]}
    for m in MODELS:
        if len(cells[m]) != N_CELLS:
            raise ProtocolError(f"{m}: {len(cells[m])} DEV cells, expected {N_CELLS}")
    return cells, sources


def select_model(cells: Dict[Tuple[str, str], Dict[str, Dict]]) -> Dict:
    """Frozen rule: lowest G_DEV; |dG| < 1e-4 -> fewer bytes moved -> larger epoch -> no pin."""
    if len(cells) != N_CELLS:
        raise ProtocolError(f"{len(cells)} cells, expected {N_CELLS}")
    stats = {}
    for c in CANDIDATES:
        rs = [ratio(float(cells[k][c["variant"]]["misses"]), float(cells[k][UNIFORM]["misses"]))
              for k in sorted(cells)]
        bm = sum(float(cells[k][c["variant"]].get("bytes_moved") or 0) for k in cells)
        mv = sum(float(cells[k][c["variant"]].get("moves") or 0) for k in cells)
        stats[c["id"]] = {"G_DEV": geomean(rs), "n_cells": len(rs), "bytes_moved_total": bm,
                          "moves_total": mv}
    g_min = min(s["G_DEV"] for s in stats.values())
    tied = [cid for cid, s in stats.items() if s["G_DEV"] - g_min < TIE_DELTA_G]
    path = ["G_DEV"]
    if len(tied) > 1:
        path.append("bytes_moved_total")
        bmin = min(stats[c]["bytes_moved_total"] for c in tied)
        tied = [c for c in tied if stats[c]["bytes_moved_total"] == bmin]
    if len(tied) > 1:
        path.append("larger_epoch")
        emax = max(CAND_BY_ID[c]["epoch"] for c in tied)
        tied = [c for c in tied if CAND_BY_ID[c]["epoch"] == emax]
    if len(tied) > 1:
        path.append("no_pinning")
        tied = sorted(tied, key=lambda c: CAND_BY_ID[c]["pinned"])[:1]
    sel = tied[0]
    for s in stats.values():
        s["G_DEV"] = round(s["G_DEV"], 12)
    return {"selected": sel, **{k: CAND_BY_ID[sel][k] for k in ("variant", "epoch", "pinned")},
            "tie_break_path": path, "tie_break_used": len(path) > 1, "candidates": stats}


def build_selection(dev_root: Path, repo: Path, raw_root: Optional[Path],
                    tool_path: Optional[Path] = None) -> Dict:
    fz, bad = check_freeze(repo, tool_path)
    if bad:
        raise ProtocolError("frozen protocol/code mismatch:\n  " + "\n  ".join(bad))
    cells, sources = load_dev_cells(dev_root, raw_root, repo)
    models = {m: select_model(cells[m]) for m in MODELS}
    selection = {
        "experiment": EXPERIMENT_ID, "protocol_version": PROTOCOL_VERSION,
        "leakage": LEAKAGE, "framing": FRAMING,
        "selection_level": "per model (one configuration for 30B, one for 80B)",
        "objective": "G_DEV = geometric mean over 7 workloads x 3 budgets (21 equal-weight "
                     "cells) of misses(candidate) / misses(per-layer-lru-uniform), "
                     "expert-event misses, decode, DEV only",
        "tie_break": ["lower G_DEV", f"|dG| < {TIE_DELTA_G}: fewer total bytes moved",
                      "larger epoch", "no pinning"],
        "candidate_set": [dict(c) for c in CANDIDATES],
        "fixed_parameters": dict(FIXED_PARAMETERS),
        "models": models, "dev_sources": sources,
        "provenance": {"code_sha256": fz["code_sha256"], "tool_sha256": fz["tool_sha256"],
                       "protocol_sha256": fz["protocol_sha256"],
                       "freeze_file_sha256": sha256_file(repo / FREEZE_REL),
                       "argine2_commit": COMMIT_NOTE},
    }
    return {"selection": selection, "canonical_sha256": canonical_sha(selection),
            "canonical_method": CANONICAL_METHOD, "written_utc": _utc()}


def validate_selection(doc: Dict, repo: Path, tool_path: Optional[Path] = None,
                       require_freeze: bool = True) -> Dict:
    sel = doc.get("selection")
    if not isinstance(sel, dict):
        raise ProtocolError("selection.json has no 'selection' payload")
    if canonical_sha(sel) != doc.get("canonical_sha256"):
        raise ProtocolError("selection.json canonical hash does not verify (edited or corrupted)")
    if sel.get("experiment") != EXPERIMENT_ID or sel.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolError("selection.json belongs to another experiment or protocol version")
    if [c["id"] for c in sel.get("candidate_set", [])] != [c["id"] for c in CANDIDATES]:
        raise ProtocolError("selection.json candidate set differs from the frozen grid")
    for m in MODELS:
        s = sel.get("models", {}).get(m)
        if not s or s.get("selected") not in CAND_BY_ID:
            raise ProtocolError(f"selection.json: unknown or missing candidate for {m}")
        c = CAND_BY_ID[s["selected"]]
        if (s.get("variant"), s.get("epoch"), s.get("pinned")) != (c["variant"], c["epoch"], c["pinned"]):
            raise ProtocolError(f"selection.json: {m} configuration fields disagree with {c['id']}")
    if require_freeze:
        fz, bad = check_freeze(repo, tool_path)
        if bad:
            raise ProtocolError("frozen protocol/code mismatch:\n  " + "\n  ".join(bad))
        pv = sel.get("provenance", {})
        for k in ("code_sha256", "tool_sha256", "protocol_sha256"):
            if pv.get(k) != fz.get(k):
                raise ProtocolError(f"selection.json {k} differs from the current frozen state")
    return sel


# ------------------------------------------------------------- EVAL analyze ---

def load_eval_cells(sel: Dict, eval_root: Path, ea_root: Path, raw_root: Optional[Path],
                    repo: Path) -> Tuple[Dict, List[Dict], Dict]:
    """{model: {(family, budget): {role: row}}}, consistency rows, EVAL sources."""
    cells: Dict[str, Dict] = {m: {} for m in MODELS}
    consistency: List[Dict] = []
    sources: Dict[str, Dict] = {}
    allowed = set(CAND_BY_VARIANT)
    for (m, f), (dev, ev) in PAIRS.items():
        cand = CAND_BY_ID[sel["models"][m]["selected"]]
        dcmp, dad = eval_root / ev, ea_root / ev
        for p in (dcmp / "policy_comparison.csv", dcmp / "compare_manifest.json",
                  dad / "adaptive.csv", dad / "adaptive_manifest.json"):
            if not p.exists():
                raise ProtocolError(f"missing EVAL input for {m}/{f}: {p}")
        cman = read_json(dcmp / "compare_manifest.json")
        aman = read_json(dad / "adaptive_manifest.json")
        ev_sha = sha256_file(raw_root / ev / "run-00-4G" / "trace.jsonl.gz") if raw_root else None
        dev_sha = sha256_file(raw_root / dev / "run-00-4G" / "trace.jsonl.gz") if raw_root else None
        _check_manifest(cman, "eval", m, f, ev_sha, str(dcmp))
        _check_manifest(cman, "dev", m, f, dev_sha, str(dcmp), train=True)
        if cman.get("train_mode") != "separate":
            raise ProtocolError(f"{dcmp}: static comparator was not fitted on a separate DEV trace")
        _check_manifest(aman, "eval", m, f, ev_sha, str(dad))
        if cand["epoch"] not in aman.get("epochs", []) or (
                cand["pinned"] and cand["pinned"] not in aman.get("pinned", [])):
            raise ProtocolError(f"{dad}: run does not contain the selected {cand['id']}")
        pc = _index_rows(read_csv(dcmp / "policy_comparison.csv"), "policy",
                         str(dcmp / "policy_comparison.csv"))
        ad = _index_rows(read_csv(dad / "adaptive.csv"), "variant", str(dad / "adaptive.csv"),
                         allowed)
        for b in BUDGETS:
            roles = {}
            for role, pol in ROLE_POLICY.items():
                r = pc.get((pol, b))
                if r is None:
                    raise ProtocolError(f"{dcmp}: missing {pol} at {b}")
                roles[role] = r
            if roles["static_dev"].get("train") != "separate":
                raise ProtocolError(f"{dcmp}: layer-aware-static at {b} is "
                                    f"{roles['static_dev'].get('train')!r}, not 'separate'")
            a = ad.get((cand["variant"], b))
            if a is None:
                raise ProtocolError(f"{dad}: missing {cand['variant']} at {b}")
            roles["adaptive_sel"] = a
            cells[m][(f, b)] = roles
            for pol, src_row in ((UNIFORM, ad.get((UNIFORM, b))),
                                 ("adaptive-e64", ad.get(("adaptive-e64", b)))):
                c = pc.get((pol, b))
                if src_row is not None and c is not None:
                    # P1a: duplicate deterministic rows are a HARD invariant (no tolerance).
                    new_misses = int(float(src_row["misses"]))
                    compare_misses = int(float(c["misses"]))
                    identical = new_misses == compare_misses
                    consistency.append({"stage": "eval-internal", "model": m, "family": f,
                                        "split": "eval", "budget": b, "policy": pol,
                                        "new_misses": new_misses,
                                        "prior_misses": compare_misses,
                                        "prior_source": "eval/compare (same frozen run set)",
                                        "diff": new_misses - compare_misses,
                                        "identical": identical})
                    if not identical:
                        raise ProtocolError(
                            f"{ev}/{b}: deterministic EVAL consistency mismatch for {pol}: "
                            f"eval-adaptive misses={new_misses}, compare misses={compare_misses}")
        sources[ev] = {"model": m, "family": f, "dev_campaign": dev,
                       "policy_comparison_sha256": sha256_file(dcmp / "policy_comparison.csv"),
                       "adaptive_csv_sha256": sha256_file(dad / "adaptive.csv"),
                       "eval_trace_sha256": cman["trace"]["sha256"],
                       "train_trace_sha256": cman["train"]["sha256"]}
    return cells, consistency, sources


def _num(r: Dict, k: str) -> Optional[float]:
    v = r.get(k)
    return None if v in (None, "") else float(v)


def eval_table(cells: Dict) -> List[Dict]:
    out = []
    for m in MODELS:
        for (f, b), roles in sorted(cells[m].items()):
            lru, bel = float(roles["global_lru"]["misses"]), float(roles["ref_belady"]["misses"])
            for role, r in roles.items():
                mi = float(r["misses"])
                out.append({"model": m, "family": f, "budget": b, "role": role,
                            "policy": r.get("policy") or r.get("variant"),
                            "comparison_class": COMPARISON_CLASS[role],
                            "note": {"static_dev": "DEV-fitted, workload-family-informed, frozen; "
                                                   "not a generic deployable baseline",
                                     "ref_static_eval_insample": "IN-SAMPLE on EVAL: reference only",
                                     "ref_belady": "future-informed reference, not a proven bound"
                                     }.get(role, ""),
                            "misses": int(mi), "hit_rate": _num(r, "hit_rate"),
                            "bytes_per_token": _num(r, "bytes_per_token"),
                            "stall_ms_per_token_MODELLED": _num(r, "stall_ms_per_token_modelled"),
                            "moves": _num(r, "moves"), "bytes_moved": _num(r, "bytes_moved"),
                            "gap_closed_vs_lru_descriptive": ((lru - mi) / (lru - bel)
                                                              if lru > bel else None)})
    return out


def _g(cells_m: Dict, keys: List, a: str, b: str) -> float:
    return geomean([ratio(float(cells_m[k][a]["misses"]), float(cells_m[k][b]["misses"]))
                    for k in keys])


def model_summary(cells_m: Dict) -> Dict:
    keys = sorted(cells_m)
    if len(keys) != N_CELLS:
        raise ProtocolError(f"{len(keys)} EVAL cells, expected {N_CELLS}")
    a = [float(cells_m[k]["adaptive_sel"]["misses"]) for k in keys]
    s = [float(cells_m[k]["static_dev"]["misses"]) for k in keys]
    wl = wins(a, s)
    g_pool = _g(cells_m, keys, "adaptive_sel", "static_dev")
    g_uni = _g(cells_m, keys, "adaptive_sel", "uniform")
    out = {"G_pool": g_pool, "G_uniform": g_uni, **wl, "n_cells": len(keys),
           "classification": classify_model(g_pool, wl["W"], g_uni),
           "G_adaptive_vs_global_lru_descriptive": _g(cells_m, keys, "adaptive_sel", "global_lru"),
           "G_adaptive_vs_2q_descriptive": _g(cells_m, keys, "adaptive_sel", "twoq"),
           "per_budget": {}, "per_workload": {}}
    for b in BUDGETS:
        kb = [k for k in keys if k[1] == b]
        ab = [float(cells_m[k]["adaptive_sel"]["misses"]) for k in kb]
        sb = [float(cells_m[k]["static_dev"]["misses"]) for k in kb]
        out["per_budget"][b] = {"G_static": _g(cells_m, kb, "adaptive_sel", "static_dev"),
                                "G_uniform": _g(cells_m, kb, "adaptive_sel", "uniform"),
                                **wins(ab, sb), "n_cells": len(kb)}
    for f in FAMILIES:
        out["per_workload"][f] = {
            b: {"ratio_adaptive_vs_static": ratio(float(cells_m[(f, b)]["adaptive_sel"]["misses"]),
                                                  float(cells_m[(f, b)]["static_dev"]["misses"])),
                "ratio_adaptive_vs_uniform": ratio(float(cells_m[(f, b)]["adaptive_sel"]["misses"]),
                                                   float(cells_m[(f, b)]["uniform"]["misses"]))}
            for b in BUDGETS}
    out["budget_reversals"] = [b for b, v in out["per_budget"].items()
                               if _sign(v["G_static"]) != _sign(g_pool)]
    return out


def decomposition_rows(cells: Dict) -> List[Dict]:
    stages = [("1 global LRU -> per-layer uniform (layer isolation / partitioning)", "uniform",
               "global_lru"),
              ("2 per-layer uniform -> DEV-fitted static (uses DEV workload-family information)",
               "static_dev", "uniform"),
              ("3 DEV-fitted static -> DEV-selected adaptive (dynamic adaptive gain)",
               "adaptive_sel", "static_dev"),
              ("direct: per-layer uniform -> DEV-selected adaptive", "adaptive_sel", "uniform"),
              ("direct: global LRU -> DEV-selected adaptive", "adaptive_sel", "global_lru")]
    out = []
    for m in MODELS:
        keys = sorted(cells[m])
        for scope, kk in [("all 21 cells", keys)] + [(f"budget {b}", [k for k in keys if k[1] == b])
                                                      for b in BUDGETS]:
            for name, a, b in stages:
                out.append({"model": m, "scope": scope, "stage": name,
                            "G_misses_ratio": _g(cells[m], kk, a, b), "n_cells": len(kk)})
    return out


def subgroup_rows(cells: Dict) -> List[Dict]:
    out = []
    for m in MODELS:
        for name, fams in SUBGROUPS.items():
            kk = sorted(k for k in cells[m] if k[0] in fams)
            a = [float(cells[m][k]["adaptive_sel"]["misses"]) for k in kk]
            s = [float(cells[m][k]["static_dev"]["misses"]) for k in kk]
            out.append({"model": m, "subgroup": name, "n_cells": len(kk),
                        "G_static": _g(cells[m], kk, "adaptive_sel", "static_dev"),
                        "G_uniform": _g(cells[m], kk, "adaptive_sel", "uniform"),
                        **wins(a, s), "decides_status": False})
    return out


def determinism_rows(sel: Dict, dev_root: Optional[Path], eval_root: Path, ea_root: Path,
                     prior_root: Optional[Path]) -> List[Dict]:
    """New frozen-code outputs vs analysis-v3 (audit only: the new run wins by protocol)."""
    out: List[Dict] = []
    if prior_root is None or not prior_root.exists():
        return [{"stage": "prior", "note": "analysis-v3 not available: no comparison made"}]

    def cmp(stage, m, f, split, newp, priorp, col, names):
        if not newp.exists() or not priorp.exists():
            out.append({"stage": stage, "model": m, "family": f, "split": split,
                        "note": f"missing {newp if not newp.exists() else priorp}"})
            return
        new = _index_rows(read_csv(newp), col, str(newp))
        pri = _index_rows(read_csv(priorp), col, str(priorp))
        for (name, b), r in sorted(new.items()):
            if name not in names or (name, b) not in pri:
                continue
            nm, pm = int(float(r["misses"])), int(float(pri[(name, b)]["misses"]))
            out.append({"stage": stage, "model": m, "family": f, "split": split, "budget": b,
                        "policy": name, "new_misses": nm, "prior_misses": pm,
                        "prior_source": _rel(priorp, REPO), "diff": nm - pm,
                        "identical": nm == pm})

    adaptive_names = {UNIFORM, "global-lru"} | set(CAND_BY_VARIANT) | DIAGNOSTIC_VARIANTS
    for (m, f), (dev, ev) in PAIRS.items():
        if dev_root is not None:
            cmp("dev-adaptive", m, f, "dev", dev_root / dev / "adaptive.csv",
                prior_root / prior_cell(m, f, "dev") / "adaptive.csv", "variant", adaptive_names)
        cmp("eval-compare", m, f, "eval", eval_root / ev / "policy_comparison.csv",
            prior_root / prior_cell(m, f, "eval") / "policy_comparison.csv", "policy",
            set(TRAIN_INDEPENDENT))
        cmp("eval-adaptive", m, f, "eval", ea_root / ev / "adaptive.csv",
            prior_root / prior_cell(m, f, "eval") / "adaptive.csv", "variant", adaptive_names)
    return out


def analyze(sel_doc: Dict, eval_root: Path, ea_root: Path, out: Path, repo: Path,
            raw_root: Optional[Path], dev_root: Optional[Path], prior_root: Optional[Path],
            tool_path: Optional[Path] = None, require_freeze: bool = True) -> Dict:
    sel = validate_selection(sel_doc, repo, tool_path, require_freeze)
    cells, consistency, sources = load_eval_cells(sel, eval_root, ea_root, raw_root, repo)
    per_model = {m: model_summary(cells[m]) for m in MODELS}
    v = verdict(per_model)
    out.mkdir(parents=True, exist_ok=True)
    dev_rows = []
    for m in MODELS:
        s = sel["models"][m]
        for cid, st in s["candidates"].items():
            dev_rows.append({"model": m, "candidate": cid, **{k: CAND_BY_ID[cid][k] for k in
                                                              ("variant", "epoch", "pinned")},
                             **st, "selected": cid == s["selected"],
                             "tie_break_path": " > ".join(s["tie_break_path"])})
    write_csv(dev_rows, out / "dev_selection.csv")
    write_csv(eval_table(cells), out / "eval_table.csv")
    write_csv(decomposition_rows(cells), out / "decomposition.csv")
    write_csv(subgroup_rows(cells), out / "subgroups.csv")
    sens = [r for m in MODELS for r in sensitivity_rows(per_model[m]["G_pool"], m)]
    write_csv(sens, out / "sensitivity.csv")
    det = consistency + determinism_rows(sel, dev_root, eval_root, ea_root, prior_root)
    write_csv(det, out / "determinism_check.csv")
    pooled = geomean([ratio(float(cells[m][k]["adaptive_sel"]["misses"]),
                            float(cells[m][k]["static_dev"]["misses"]))
                      for m in MODELS for k in cells[m]])
    summary = {
        "experiment": EXPERIMENT_ID, "protocol_version": PROTOCOL_VERSION,
        "framing": FRAMING, "leakage": LEAKAGE, "closure_rule": CLOSURE_RULE_ID,
        "primary_comparison": "DEV-selected adaptive vs DEV-fitted, workload-family-informed "
                              "static allocation applied unchanged to EVAL",
        "primary_metric": "expert-event misses (decode, replay v2, expert-unit, event-atomic)",
        "selection_canonical_sha256": sel_doc["canonical_sha256"],
        "selected": {m: sel["models"][m]["selected"] for m in MODELS},
        "per_model": per_model, "verdict": v,
        "verdict_note": "applies to the surviving retrospective-holdout sub-question only; "
                        "'approach the oracle' has no verdict (no original threshold)",
        "descriptive_only": {
            "pooled_two_model_G_adaptive_vs_static": pooled,
            "comparisons_already_seen_on_EVAL": ["adaptive vs global LRU",
                                                 "adaptive vs per-layer uniform",
                                                 "adaptive vs 2Q"],
            "sensitivity_thresholds": list(SENSITIVITY),
            "sensitivity_affects_verdict": False,
            "stall_is": "MODELLED, not measured"},
        "determinism": {"rows": len(det),
                        "non_identical": sum(1 for r in det if r.get("identical") is False)},
        "eval_sources": sources, "written_utc": _utc()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    return summary


# -------------------------------------------------------- pair verification ---

def verify_pairs(repo: Path) -> List[Dict]:
    """Metadata-only check of the 14 DEV/EVAL pairs (loads traces, replays nothing)."""
    sys.path.insert(0, str(repo))
    from argine.research.trace import Trace
    from argine.research.replay import counted_step
    rhr = {}
    if (repo / RHR_REL).exists():
        for r in read_csv(repo / RHR_REL):
            if r.get("run") == "run-00-4G":
                rhr[r["campaign"]] = r
    out = []
    for (m, f), (dev, ev) in PAIRS.items():
        row = {"model": m, "family": f}
        for split, camp in (("dev", dev), ("eval", ev)):
            run = repo / RAW_REL / camp / "run-00-4G"
            tp = run / "trace.jsonl.gz"
            row[f"{split}_campaign"] = camp
            row[f"{split}_trace_exists"] = tp.exists()
            if not tp.exists():
                continue
            tr = Trace.load(str(tp))
            man = read_json(run / "manifest.json") if (run / "manifest.json").exists() else {}
            row[f"{split}_trace_sha256"] = sha256_file(tp)
            row[f"{split}_meta_split"] = tr.meta.split
            row[f"{split}_workload"] = tr.meta.workload
            row[f"{split}_model_file"] = tr.meta.model
            row[f"{split}_decode_steps"] = sum(1 for s in tr.steps if counted_step(s, "decode"))
            row[f"{split}_manifest_budget"] = man.get("budget")
            row[f"{split}_history_semantics"] = (man.get("conversation_history_semantics") or {}
                                                 ).get("version", "-")
            q = rhr.get(camp, {})
            row[f"{split}_exp021"] = q.get("exp021_closure")
            row[f"{split}_qualified"] = q.get("qualified")
        ok = (row.get("dev_trace_exists") and row.get("eval_trace_exists")
              and row.get("dev_meta_split") == "dev" and row.get("eval_meta_split") == "eval"
              and family_of(row.get("dev_workload")) == f == family_of(row.get("eval_workload"))
              and row.get("dev_workload") == row.get("eval_workload")
              and model_of(row.get("dev_model_file")) == m == model_of(row.get("eval_model_file"))
              and row.get("dev_manifest_budget") == "4G" == row.get("eval_manifest_budget")
              and row.get("dev_qualified") == "True" == row.get("eval_qualified")
              and (f != "multi-turn-v2" or (row.get("dev_history_semantics") == "v2"
                                            == row.get("eval_history_semantics"))))
        row["length_mismatch"] = row.get("dev_decode_steps") != row.get("eval_decode_steps")
        row["pair_ok"] = bool(ok)
        out.append(row)
    return out


# ------------------------------------------------------------- step-5 script ---

def eval_adaptive_script(sel: Dict, repo_dir: str, argine: str = "argine.bat") -> str:
    lines = ["@echo off",
             "REM EXP-010 STEP 5, generated from selection.json (do not edit by hand).",
             "REM Runs the DEV-selected configuration of each model on its 7 EVAL cells.",
             "REM The adaptive command may also emit mandatory baseline/base rows; only the "
             "selected variant enters the primary comparison.",
             f"cd /d \"{repo_dir}\""]
    for m in MODELS:
        c = CAND_BY_ID[sel["models"][m]["selected"]]
        lines.append(f"REM {m}: selected {c['id']} (variant {c['variant']})")
        pinned = "0" if not c["pinned"] else f"{c['pinned']:g}"
        for f in FAMILIES:
            ev = PAIRS[(m, f)][1]
            tr = trace_rel(ev).replace("/", "\\")
            out = f"{EXP_REL}/eval-adaptive/{ev}".replace("/", "\\")
            lines.append(f"call {argine} research adaptive {tr} --budgets 4G,2G,1.5G "
                         f"--epochs {c['epoch']} --pinned {pinned} --no-ablate --out {out}")
            lines.append("if errorlevel 1 exit /b 1")
    lines.append("echo EXP-010 STEP 5 done.")
    return "\r\n".join(lines) + "\r\n"


# --------------------------------------------------------------------- CLI ---

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", default=str(REPO), help=argparse.SUPPRESS)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("verify-pairs")
    p.add_argument("--out", default=None)
    sub.add_parser("freeze")
    sub.add_parser("check-freeze")
    p = sub.add_parser("select")
    p.add_argument("--dev", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--raw", default=None, help="EXP-007 raw root (default: the repository's)")
    p = sub.add_parser("show")
    p.add_argument("--selection", required=True)
    p = sub.add_parser("script-eval-adaptive")
    p.add_argument("--selection", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("analyze")
    p.add_argument("--selection", required=True)
    p.add_argument("--eval", required=True)
    p.add_argument("--eval-adaptive", required=True)
    p.add_argument("--dev", default=None, help="DEV root (determinism check of DEV outputs)")
    p.add_argument("--prior", default=None, help="analysis-v3 root (default: the repository's)")
    p.add_argument("--raw", default=None)
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    repo = Path(a.repo)
    try:
        if a.cmd == "verify-pairs":
            rows = verify_pairs(repo)
            if a.out:
                write_csv(rows, Path(a.out))
            for r in rows:
                print(f"{r['model']} {r['family']:17s} {r['dev_campaign']:30s} -> "
                      f"{r['eval_campaign']:30s} steps {r.get('dev_decode_steps')}/"
                      f"{r.get('eval_decode_steps')} ok={r['pair_ok']}")
            return 0 if all(r["pair_ok"] for r in rows) else 1
        if a.cmd == "freeze":
            doc = freeze(repo)
            print(json.dumps(doc, indent=2))
            return 0
        if a.cmd == "check-freeze":
            _fz, bad = check_freeze(repo)
            print("frozen state OK" if not bad else "MISMATCH:\n  " + "\n  ".join(bad))
            return 0 if not bad else 2
        if a.cmd == "select":
            raw = Path(a.raw) if a.raw else repo / RAW_REL
            doc = build_selection(Path(a.dev), repo, raw)
            outp = Path(a.out)
            if outp.exists():
                raise ProtocolError(f"{outp} exists: a selection is made once; never overwrite it")
            outp.parent.mkdir(parents=True, exist_ok=True)
            outp.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            for m in MODELS:
                s = doc["selection"]["models"][m]
                print(f"{m}: selected {s['selected']} (tie-break path: {' > '.join(s['tie_break_path'])})")
            print(f"canonical sha256: {doc['canonical_sha256']}")
            return 0
        if a.cmd == "show":
            doc = read_json(Path(a.selection))
            sel = validate_selection(doc, repo, require_freeze=False)
            for m in MODELS:
                s = sel["models"][m]
                print(f"{m}: SELECTED {s['selected']}  (epoch {s['epoch']}, pinned {s['pinned']})")
                for cid, st in s["candidates"].items():
                    print(f"    {cid:10s} G_DEV={st['G_DEV']:.6f}  bytes_moved={st['bytes_moved_total']:.0f}")
            print(f"canonical sha256 verified: {doc['canonical_sha256']}")
            return 0
        if a.cmd == "script-eval-adaptive":
            sel = validate_selection(read_json(Path(a.selection)), repo)
            Path(a.out).write_text(eval_adaptive_script(sel, str(repo.resolve())), encoding="utf-8", newline="")
            print(f"wrote {a.out}")
            return 0
        if a.cmd == "analyze":
            raw = Path(a.raw) if a.raw else repo / RAW_REL
            prior = Path(a.prior) if a.prior else repo / PRIOR_REL
            s = analyze(read_json(Path(a.selection)), Path(a.eval), Path(a.eval_adaptive),
                        Path(a.out), repo, raw, Path(a.dev) if a.dev else None, prior)
            for m in MODELS:
                pm = s["per_model"][m]
                print(f"{m}: G_pool={pm['G_pool']:.6f} W={pm['W']} G_uniform={pm['G_uniform']:.6f}"
                      f" -> {pm['classification']}")
            print(f"VERDICT: {s['verdict']}")
            return 0
    except ProtocolError as e:
        print(f"PROTOCOL ERROR: {e}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
