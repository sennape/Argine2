"""EXP-026 — 30B vs 80B regimes (protocol EXP026-P1: orchestration, classification, analysis).

    python tools/exp026_run.py --check                 # static validation only: protocol constants, exact
                                                       #   budgets, model geometry re-derived from tracked
                                                       #   EXP-007 evidence, runtime/diskbench sources, frozen
                                                       #   orders, docs/config agreement, empty raw/, freeze.
                                                       #   Only `git` is ever started; no model file is opened.
    python tools/exp026_run.py --dry-run               # print the frozen schedule and every command; no subprocess
    python tools/exp026_run.py --freeze                # write protocol_freeze.json (prospective freeze)
    python tools/exp026_run.py --preflight --block B   # machine/runtime checks (+ `argine benchmark --dry-run` for
                                                       #   model blocks: GGUF headers only, never llama-server)
    python tools/exp026_run.py --run --block B         # one block (writes raw/<B>/); NOT resumable, no retries
    python tools/exp026_run.py --finalize --block B    # post-block fingerprint finalisation only
    python tools/exp026_run.py --analyze               # raw/ -> results.csv + processed/summary.json

Blocks, in this frozen order: diskbench (device calibration, 60 runs), perf (24 untraced model launches:
18 primary + 6 sensitivity), diag (12 traced model launches). 36 scheduled model launches in total.

What EXP-026 asks (historical, 2026-10-03): why does the 80B behave differently from the 30B?
Primary explanandum: the pressure-response slope difference Δβ = β80 − β30, β = Δln(gen tok/s) / Δln(budget/F)
between P1 and P2, from untraced launches at 4 generation threads for both models. The candidate explanations
(a)-(c) of the EXP-026 README and (d) of RQ-009 are discriminated with frozen per-candidate rules; the
accounting is a BOUND decomposition (never an exact T = C + E_r + E_nr identity).

Replication unit: one complete launch. No retries, no replacements, no adaptive pruning.
Per-launch validation (provenance, log parsing, outcome classification) is reused unchanged from
tools/exp022_run.py, whose LF-normalised SHA-256 is pinned below and in the freeze.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
EXP_ID = "EXP-026-30b-vs-80b-regimes"
EXP = ROOT / "experiments" / EXP_ID
RAW = EXP / "raw"
PROCESSED = EXP / "processed"
FIGURES = EXP / "figures"
README = EXP / "README.md"
CONFIG = EXP / "config.json"
FREEZE = EXP / "protocol_freeze.json"
TESTS = ROOT / "tests" / "test_exp026.py"
OPEN_QUESTIONS = ROOT / "OPEN_QUESTIONS.md"
MARK_BEGIN = "<!-- EXP-026 PROTOCOL-BEGIN -->"
MARK_END = "<!-- EXP-026 PROTOCOL-END -->"
PROTOCOL_VERSION = "EXP026-P1"
FREEZE_DATE = "2026-10-15"
PRE_FREEZE_HEAD = "527540cee4e97b7ecc337c57f95d767f47798473"

MIB = 1 << 20
GIB = 1 << 30

# ------------------------------------------------------- reused EXP-022 code ----

EXP022_TOOL = ROOT / "tools" / "exp022_run.py"
EXP022_TOOL_SHA256 = "cced9415e82bb6d495064c5ae2e6a9dd63e901a0f52f7967706d219a902b4e81"   # LF-normalised
EXP022_REUSED = ("classify", "parse_server_segment", "check_segment_provenance", "check_llama_command",
                 "expected_llama_flags", "perf_record_problems", "perf_invariant_problems", "perf_measured",
                 "diag_manifest_problems", "diag_invariant_problems", "diag_measured", "diag_diagnostics",
                 "abort_cause", "check_runtime_sources", "check_timeout_sources", "check_workloads",
                 "cap_seconds", "base_env", "binary_path", "machine_problems", "orphan_servers", "free_ram",
                 "session_provenance", "finalize_fingerprints", "registry_append_only", "routing_sha",
                 "gguf_footprint", "_git", "_sha", "_lf_sha", "describe")


def _load_exp022():
    spec = importlib.util.spec_from_file_location("exp022_run_for_exp026", EXP022_TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


X22 = _load_exp022()

# ------------------------------------------------------------ hypothesis ----

HISTORICAL_QUESTION = ("Why does the 80B behave differently from the 30B (requirement section 56, "
                       "question 10)?")
HISTORICAL_HYPOTHESIS = (
    "[HYPOTHESIS] Competing explanations, to be discriminated: (a) routing concentration (smaller EWS "
    "fraction); (b) smaller slices (440-704 KiB vs 930 KiB) raise queue depth per byte; (c) the 80B is "
    "bookkeeping-bound (commit/decommit/lock 56% of its I/O bucket) rather than read-bound.")
CANDIDATE_D_TEXT = "(d) different compute (Gated DeltaNet in 36 layers)"
CANDIDATES = {
    "a": {"wording": "(a) routing concentration (smaller EWS fraction)", "source": "EXP-026 README hypothesis (2026-10-03)"},
    "b": {"wording": "(b) smaller slices (440-704 KiB vs 930 KiB) raise queue depth per byte",
          "source": "EXP-026 README hypothesis (2026-10-03)"},
    "c": {"wording": "(c) the 80B is bookkeeping-bound (commit/decommit/lock 56% of its I/O bucket) rather than "
                     "read-bound", "source": "EXP-026 README hypothesis (2026-10-03)"},
    "d": {"wording": CANDIDATE_D_TEXT,
          "source": "pre-existing RQ-009 candidate list (OPEN_QUESTIONS.md); not part of the historical EXP-026 "
                    "hypothesis quotation"},
}
CANDIDATE_KEYS = ("a", "b", "c", "d")

# ----------------------------------------------------------------- models ----

L30_DOWN_1260 = (0, 1, 2, 3, 4, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 42, 43, 44, 45, 46, 47)
L80_DOWN_704 = (0, 1, 2)
ALL_LAYERS = tuple(range(48))

MODELS: Dict[str, Dict[str, Any]] = {
    "30b": {
        "name": "Qwen3-30B-A3B Q4_K_M",
        "path": r"C:\modelli\Qwen_Qwen3-30B-A3B-Q4_K_M.gguf",
        "store": r"C:\modelli\Qwen_Qwen3-30B-A3B-Q4_K_M.psxs",
        "n_layers": 48, "n_experts": 128, "top_k": 8,
        "footprint_bytes": 1097072640,
        "whole_layer_bytes_max": 391643136,
        "slices": {  # kind -> {bytes: layers}
            "gate": {884736: ALL_LAYERS},
            "up": {884736: ALL_LAYERS},
            "down": {1290240: L30_DOWN_1260, 884736: tuple(l for l in ALL_LAYERS if l not in L30_DOWN_1260)},
        },
        "geometry_source": "experiments/EXP-007-router-characterization/analysis-v3/"
                           "30b-natural-language-dev-run-00-4G/analysis_manifest.json",
    },
    "80b": {
        "name": "Qwen3-Next-80B-A3B-Instruct Q3_K_M",
        "path": r"C:\modelli\Qwen3-Next-80B-A3B-Instruct-Q3_K_M.gguf",
        "store": r"C:\modelli\Qwen3-Next-80B-A3B-Instruct-Q3_K_M.psxs",
        "n_layers": 48, "n_experts": 512, "top_k": 10,
        "footprint_bytes": 719585280,
        "whole_layer_bytes_max": 830472192,
        "slices": {
            "gate": {450560: ALL_LAYERS},
            "up": {450560: ALL_LAYERS},
            "down": {720896: L80_DOWN_704, 589824: tuple(l for l in ALL_LAYERS if l not in L80_DOWN_704)},
        },
        "geometry_source": "experiments/EXP-007-router-characterization/analysis-v3/"
                           "80b-natural-language-dev-run-00-4G/analysis_manifest.json",
    },
}
MODEL_ORDER = ("30b", "80b")
GEOMETRY_PROVENANCE = (
    "slice sizes by (layer, kind) verified 2026-10-15 from every EXP-022 diagnostic io.csv (aligned store-read "
    "bytes; every (layer, kind) group uniform), the trace metadata expert_bytes per layer, and the GGUF-derived "
    "expert_bytes total of the benchmark record; F and the whole-layer sizes are re-derived by --check from the "
    "tracked EXP-007 trace_meta")

# historical EWS-fraction premise (EXP-007, natural-language, 4 GiB; descriptive only, never a decision input)
EXP007_EWS90 = {"30b": {"eval": (2179, 6144), "dev": (2069, 6144)},
                "80b": {"eval": (6331, 24576), "dev": (6403, 24576)}}


def layer_expert_bytes(model: str) -> Dict[int, int]:
    """Bytes of one expert of each layer (gate + up + down aligned slices), from the frozen slice table."""
    out = {l: 0 for l in range(MODELS[model]["n_layers"])}
    for _kind, sizes in MODELS[model]["slices"].items():
        for b, layers in sizes.items():
            for l in layers:
                out[l] += b
    return out


def footprint(model: str) -> int:
    return MODELS[model]["top_k"] * sum(layer_expert_bytes(model).values())


def whole_layer_bytes(model: str) -> Dict[int, int]:
    return {l: b * MODELS[model]["n_experts"] for l, b in layer_expert_bytes(model).items()}


def slice_sizes(model: str) -> List[int]:
    return sorted({b for sizes in MODELS[model]["slices"].values() for b in sizes})


def reads_per_token(model: str) -> Dict[int, int]:
    """Routed-expert store reads per decode token at zero hits, by slice size (top_k x (layer, kind) count)."""
    c: Counter = Counter()
    for _kind, sizes in MODELS[model]["slices"].items():
        for b, layers in sizes.items():
            c[b] += MODELS[model]["top_k"] * len(layers)
    return dict(sorted(c.items()))


# ---------------------------------------------------------------- budgets ----

POINTS = ("P1", "P2", "P3")
PRIMARY_POINTS = ("P1", "P2")
POINT_ROLES = {"P1": "primary slope endpoint", "P2": "primary slope endpoint",
               "P3": "secondary sub-F probe (not used for the primary beta)"}
BUDGETS: Dict[str, Dict[str, Tuple[str, int]]] = {
    "30b": {"P1": ("4096M", 4294967296), "P2": ("1536M", 1610612736), "P3": ("1024M", 1073741824)},
    "80b": {"P1": ("2688M", 2818572288), "P2": ("1008M", 1056964608), "P3": ("672M", 704643072)},
}
SHARD_MIN_BYTES = 32 * MIB
MAX_SHARDS = 16
NOT_IN_GRID = ("no 512 MiB block", "no ~1.10 F block", "nothing below 512 MiB")


def shard_count(budget: int) -> int:
    return int(max(1, min(MAX_SHARDS, budget // SHARD_MIN_BYTES)))


def x_of(model: str, point: str) -> float:
    """budget / F (the primary normalization)."""
    return BUDGETS[model][point][1] / MODELS[model]["footprint_bytes"]


def budget_table(model: str) -> Dict[str, Dict[str, Any]]:
    wl = whole_layer_bytes(model)
    out = {}
    for p in POINTS:
        label, b = BUDGETS[model][p]
        out[p] = {"label": label, "bytes": b, "mib": b / MIB, "budget_over_F": x_of(model, p),
                  "shards": shard_count(b), "per_shard_bytes": b // shard_count(b), "role": POINT_ROLES[p],
                  "below_every_whole_layer": b < min(wl.values()),
                  "below_largest_whole_layer": b < max(wl.values())}
    return out


# ---------------------------------------------------------------- runtime ----

DEFAULT_LLAMA_DIR = ROOT / "llama-research-v2"
PRIMARY_THREADS = 4
SENSITIVITY_THREADS = 12
SENSITIVITY_MODEL = "30b"
SENSITIVITY_POINTS = ("P1", "P2")
BATCH_THREADS = X22.BATCH_THREADS            # 12: planner max(threads, min(logical, physical)) on the i5-1340P
CTX, BATCH, UBATCH, KV_TYPE = X22.CTX, X22.BATCH, X22.UBATCH, X22.KV_TYPE
READ_MODE = "asincrono"
BACKEND = "cpu"
N_PREDICT = 128
REPEAT = 1
RETRIES = 0
COOLDOWN_S = 60
SETTLE_S = 5
STORE_READ_SEMANTICS = {
    "cli": "--read-mode asincrono",
    "environment": {"ARGINE_EXPERT_READ": "asincrono", "ARGINE_EXPERT_DIRECT": "1", "ARGINE_EXPERT_PRETOUCH": "0"},
    "shards": MAX_SHARDS,
    "windows": "per-thread handles opened with FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED; 4096-aligned slice "
               "reads; several overlapped reads in flight per handle; synchronous fallback only when no handle or "
               "event is available (recorded per read in the trace flags)",
    "recorded_per_launch": "store-open line (budget, 16 shards, 'cache dell'OS scavalcata', 'lettura asincrono'); "
                           "diag manifests: read_mode requested/effective, direct_io=1, pretouch=0, cache_shards",
}
RUNTIME_SOURCE_SNIPPETS = {
    "patches/argine-store.cpp": ("DWORD flags = (direct ? FILE_FLAG_NO_BUFFERING : FILE_FLAG_RANDOM_ACCESS)",
                                 "| FILE_FLAG_OVERLAPPED;"),
    "patches/argine-experts.cpp": ("ARGINE_EXPERT_DIRECT   1 (default) scavalca la cache dell'OS",),
    "argine/cli.py": ('MODI_LETTURA = ("sincrono", "blocco", "asincrono")', 'env["ARGINE_EXPERT_READ"] = modo',
                      'env["ARGINE_EXPERT_DIRECT"] = "1" if diretto else "0"'),
}

# ---------------------------------------------------------------- workloads ----

PERF_PROMPT_SHA256 = X22.PERF_PROMPT_SHA256
PERF_PROMPT_CHARS = X22.PERF_PROMPT_CHARS
DIAG_WORKLOAD = dict(X22.DIAG_WORKLOAD)       # natural-language v1 EVAL, 3 requests x 128 tokens, sha 71b170bc...
WORKLOAD_NOTE = ("the trace tool accepts only versioned workloads, so the diagnostic blocks use natural-language v1 "
                 "EVAL (the EXP-022 diagnostic workload), while the explanandum is measured on the untraced default "
                 "benchmark prompt; the traced-slope agreement gate links the two")

# ---------------------------------------------------------------- diskbench ----

DISKBENCH_FILE = MODELS["80b"]["store"]
DISKBENCH_SIZES = (450560, 589824, 720896, 884736, 1290240)
DISKBENCH_READERS = (2, 4, 8, 16)
DISKBENCH_ROUNDS = 3
DISKBENCH_SECONDS = 4
DISKBENCH_KIND = "direct"
DISKBENCH_CAP_S = 300
DISKBENCH_PAIRS = ((2, 4), (4, 8), (8, 16))   # (30B readers N, 80B readers 2N): approximately equal bytes in flight
DISKBENCH_ROLE = (
    "device-level calibration only, not an emulator of the expert store. The store issues overlapped "
    "(asynchronous) uncached reads from per-thread handles, several in flight per handle, inside per-layer "
    "waves. `argine diskbench --kinds direct` issues one blocking uncached ReadFile per Python thread "
    "(FILE_FLAG_NO_BUFFERING, no FILE_FLAG_OVERLAPPED) at random 4096-aligned offsets. It reproduces only: "
    "uncached/direct reads, alignment, slice size and a controlled number of concurrent reads. It does NOT "
    "reproduce the store's `asincrono` semantics, its wave structure or its offsets. `moe` mode is not used.")
DISKBENCH_SOURCE_SNIPPETS = {
    "argine/diskbench.py": ("def bench_positioned(", "flags = FILE_FLAG_NO_BUFFERING if self.direct else FILE_FLAG_RANDOM_ACCESS",
                            'ALL_KINDS = ("mmap", "pread", "direct", "moe")'),
    "argine/cli.py": ('event="diskbench"', "if args.json:"),
    "argine/util.py": ('env = os.environ.get("ARGINE_HOME") or os.environ.get("PICOSTAR_HOME")',
                       "nuova = not p.exists()"),
}
DISKBENCH_TELEMETRY = (
    "each diskbench run gets ARGINE_HOME = raw/diskbench/argine-home (created by the harness BEFORE the first run, "
    "so argine.util.app_dir never imports another history into it); diskbench's telemetry line therefore lands in "
    "raw/diskbench/argine-home/logs/runs.jsonl and is copied into raw/diskbench/records.jsonl; stdout/stderr go "
    "to raw/diskbench/logs/run-NN.log")

# ------------------------------------------------------------- schedule ----

BLOCK_ORDER = ("diskbench", "perf", "diag")
SEEDS = {"perf": 2602601, "diag": 2602602, "diskbench": 2602603}
PERMUTATION_RULE = ("round r (1-based) of block B: cells in canonical order, then random.Random(SEEDS[B] * 100 + r)"
                    ".shuffle(cells); the expanded orders are stored in the freeze spec and re-derived by --check")
PERF_ROUNDS = 3
DIAG_ROUNDS = 2
EXPECTED = {"perf_primary": 18, "perf_sensitivity": 6, "diag": 12, "model_launches": 36, "diskbench_runs": 60}
OUTCOMES = ("VALID", "STORE_RESIDENCY_ABORT", "NON_COMPLETION", "INVALID")
DISKBENCH_OUTCOMES = ("VALID", "NON_COMPLETION", "INVALID")
NO_RETRY_RULES = {"no_retry": True, "no_replacement": True, "no_adaptive_pruning": True,
                  "categorical_failures_add_launches": False, "retries_flag": RETRIES}


class ProtocolError(RuntimeError):
    """A configuration, provenance or data condition that the frozen protocol forbids."""


def perf_cells() -> List[Dict[str, Any]]:
    cells = [{"cell": f"{m}-t{PRIMARY_THREADS}-{p}", "model": m, "point": p, "threads": PRIMARY_THREADS,
              "role": "primary"} for m in MODEL_ORDER for p in POINTS]
    cells += [{"cell": f"{SENSITIVITY_MODEL}-t{SENSITIVITY_THREADS}-{p}", "model": SENSITIVITY_MODEL, "point": p,
               "threads": SENSITIVITY_THREADS, "role": "sensitivity"} for p in SENSITIVITY_POINTS]
    return cells


def diag_cells() -> List[Dict[str, Any]]:
    return [{"cell": f"{m}-t{PRIMARY_THREADS}-{p}-diag", "model": m, "point": p, "threads": PRIMARY_THREADS,
             "role": "diagnostic"} for m in MODEL_ORDER for p in POINTS]


def diskbench_cells() -> List[Dict[str, Any]]:
    return [{"cell": f"s{s}-n{n}", "slice_bytes": s, "readers": n} for s in DISKBENCH_SIZES for n in DISKBENCH_READERS]


def frozen_rounds(block: str) -> List[List[str]]:
    cells = {"perf": perf_cells, "diag": diag_cells, "diskbench": diskbench_cells}[block]()
    n = {"perf": PERF_ROUNDS, "diag": DIAG_ROUNDS, "diskbench": DISKBENCH_ROUNDS}[block]
    out = []
    for r in range(1, n + 1):
        ids = [c["cell"] for c in cells]
        random.Random(SEEDS[block] * 100 + r).shuffle(ids)
        out.append(ids)
    return out


def perf_args(model: str, label: str, threads: int, dry_run: bool = False) -> List[str]:
    a = ["benchmark", MODELS[model]["path"],
         "--backend", BACKEND, "--expert-store", "--expert-budget", label,
         "--read-mode", READ_MODE, "--threads", str(threads),
         "--ctx", str(CTX), "--batch", str(BATCH), "--ubatch", str(UBATCH),
         "--repeat", str(REPEAT), "--n-predict", str(N_PREDICT), "--retries", str(RETRIES),
         "--expert-stats"]
    if dry_run:
        a.append("--dry-run")
    return a


def diag_args(model: str, label: str, threads: int, out_dir: str) -> List[str]:
    return ["research", "trace", MODELS[model]["path"],
            "--backend", BACKEND, "--expert-store", "--budgets", label, "--repeat", str(REPEAT),
            "--read-mode", READ_MODE, "--threads", str(threads),
            "--ctx", str(CTX), "--batch", str(BATCH), "--ubatch", str(UBATCH),
            "--retries", str(RETRIES), "--expert-stats",
            "--workload", DIAG_WORKLOAD["name"], "--workload-version", DIAG_WORKLOAD["version"],
            "--split", DIAG_WORKLOAD["split"], "--cooldown", "0", "--out", out_dir]


def diskbench_args(slice_bytes: int, readers: int) -> List[str]:
    return ["diskbench", DISKBENCH_FILE, "--kinds", DISKBENCH_KIND, "--no-sequential",
            "--slice", str(slice_bytes), "--threads", str(readers), "--seconds", str(DISKBENCH_SECONDS), "--json"]


def build_plan() -> List[Dict[str, Any]]:
    plan: List[Dict[str, Any]] = []
    for block in BLOCK_ORDER:
        cells = {c["cell"]: c for c in {"perf": perf_cells, "diag": diag_cells, "diskbench": diskbench_cells}[block]()}
        i = 0
        for rnd, order in enumerate(frozen_rounds(block), 1):
            for pos, cid in enumerate(order, 1):
                i += 1
                c = cells[cid]
                item: Dict[str, Any] = {"block": block, "phase": block, "cell": cid, "launch_index": i,
                                        "global_index": len(plan) + 1, "round": rnd, "position_in_round": pos}
                if block == "diskbench":
                    item.update({"slice_bytes": c["slice_bytes"], "readers": c["readers"], "cap_s": DISKBENCH_CAP_S,
                                 "launch_dir": f"raw/diskbench/run-{i:02d}",
                                 "argv": diskbench_args(c["slice_bytes"], c["readers"])})
                else:
                    label, b = BUDGETS[c["model"]][c["point"]]
                    item.update({"model": c["model"], "point": c["point"], "role": c["role"], "threads": c["threads"],
                                 "batch_threads": BATCH_THREADS, "budget": label, "budget_bytes": b,
                                 "budget_over_F": x_of(c["model"], c["point"]),
                                 "cap_s": X22.cap_seconds(block), "launch_dir": f"raw/{block}/launch-{i:02d}"})
                    item["argv"] = (perf_args(c["model"], label, c["threads"]) if block == "perf" else
                                    diag_args(c["model"], label, c["threads"],
                                              str(Path("experiments") / EXP_ID / item["launch_dir"])))
                plan.append(item)
    return plan


def block_plan(block: str) -> List[Dict[str, Any]]:
    if block not in BLOCK_ORDER:
        raise ProtocolError(f"unknown block {block!r}; frozen blocks: {', '.join(BLOCK_ORDER)}")
    return [x for x in build_plan() if x["block"] == block]


def command(item: Dict[str, Any]) -> List[str]:
    return [sys.executable, "-m", "argine"] + list(item["argv"])


def _flag(argv: Sequence[str], flag: str) -> Optional[str]:
    return argv[list(argv).index(flag) + 1] if flag in argv else None


# ===================================================== frozen decision rules ====

ENVELOPE_RULE = ("for each model, beta is computed for every pairing of one P1 value with one P2 value (3 x 3 "
                 "for performance cells, 2 x 2 for diagnostic cells); the envelope is [min, max] of those betas")
EXPLANANDUM_RULE = {
    "beta": "beta_m = (ln s_P1 - ln s_P2) / (ln x_P1 - ln x_P2), s = gen tok/s (cell median of 3 VALID untraced "
            "launches at 4 generation threads), x = budget / F",
    "delta_beta": "delta_beta = beta_80b - beta_30b (point value from the cell medians)",
    "envelope": ENVELOPE_RULE,
    "REPRODUCED": "beta80 < beta30 and the envelopes do not overlap (max env80 < min env30)",
    "REVERSED": "beta80 > beta30 and the envelopes do not overlap (min env80 > max env30)",
    "NOT DISTINGUISHED": "the envelopes overlap",
    "NOT EVALUABLE": "a P1 or P2 primary cell of either model fails eligibility (fewer than 3 of 3 VALID)",
    "historical_direction": "the 80B had the flatter speed/residency response (F-004: 0.384 vs 0.50-0.58)",
    "no_imported_thresholds": "no pooled-SD threshold, no CV threshold, no p-value",
}
ELIGIBILITY_RULES = {
    "performance_cell": "3 of 3 scheduled launches VALID; otherwise the cell is NOT EVALUABLE for every decision; "
                        "categorical outcomes are reported; numeric values of VALID launches may be shown labelled "
                        "DESCRIPTIVE ONLY (for 2 VALID launches the centre is the two-value mean, labelled as such) "
                        "and never feed a candidate or lifecycle decision",
    "diagnostic_launch": "outcome VALID, manifest publication_quality true, trace complete over the workload, "
                         "budget_violations == 0, node intervals disjoint and node residency <= T in every eligible "
                         "decode graph, at least one eligible decode graph",
    "diagnostic_cell": "2 of 2 scheduled launches eligible (EXP-022 A1 precedent); otherwise every mechanism "
                       "decision that needs the cell is NOT EVALUABLE",
    "diskbench_comparison": "all 3 round values VALID for every (slice, readers) cell the comparison needs",
}
ACCOUNTING = {
    "kind": "BOUND DECOMPOSITION ONLY (verified on existing EXP-022 traces 2026-10-15; an exact "
            "T = C + E_r + E_nr identity is NOT frozen)",
    "eligible_decode_graph": "graph g with n_tokens == 1 whose next graph g+1 is also a decode graph in the same "
                             "workload request (warm-up and flush-sentinel segments excluded)",
    "T_g": "t0(g+1) - t0(g), t0 = the graph's earliest router (R record) timestamp",
    "Res_g": "sum over the graph's MoE nodes of (ready - start), thread-0 view; nodes must be disjoint (checked)",
    "N_g": "T_g - Res_g: exact; NON-RESIDENCY (compute-inclusive) time, NOT pure compute: non-MoE layers, attention, "
           "Gated DeltaNet, MoE matmul outside residency, sampling, synchronisation outside the store, tracing cost",
    "reads_of_g": "store reads (I records) issued in [t0(g), t0(g+1))",
    "U_all": "union of [issue, complete] over reads_of_g (complete may be an upper bound)",
    "U_ex": "union of [issue, complete] over reads_of_g with exact completion only",
    "E_r": "E_r_lo = |U_ex ∩ ResU_g| <= E_r <= E_r_hi = |U_all ∩ ResU_g|, ResU_g = union of node intervals",
    "E_nr": "E_nr = Res_g - E_r, so E_nr_lo = Res_g - E_r_hi <= E_nr <= E_nr_hi = Res_g - E_r_lo",
    "unions": "read intervals overlap heavily (mean concurrency 6-9 in EXP-022 samples): unions, never sums of latencies",
    "M_g": "sum of aligned store-read bytes of reads_of_g (a wait on another thread's read is an access, not a read)",
    "per_launch": "means over eligible decode graphs: T, N, Res, E_r_lo/hi, E_nr_lo/hi, M; N/T = sum N / sum T; "
                  "q_lo/hi = sum E_r_lo/hi / sum M; m = mean M / F; depth = sum read latency / sum |U_all|",
}
COUNTERFACTUAL_RULE = (
    "DESCRIPTIVE ACCOUNTING ONLY, not a causal intervention, no significance. Traced cell means per (model, point). "
    "For a term X in {N, E_r, E_nr}: X80(P2) is replaced by X80(P1) * X30(P2) / X30(P1), the other 80B P2 terms "
    "stay observed, T80'(P2) = N + E_r + E_nr; for X = M the 80B P2 read bytes are rescaled the same way and "
    "E_r80(P2) is scaled with them (q held). beta80' uses the observed T80(P1). 'Moves beta80 toward beta30' means "
    "|beta80' - beta30| < |beta80 - beta30| on traced betas. Bounded terms are evaluated at both admissible bound "
    "versions (lo: E_r = E_r_lo, E_nr = Res - E_r_lo; hi: E_r = E_r_hi, E_nr = Res - E_r_hi); a directional "
    "statement counts only if both versions move beta80 toward beta30.")
GATE_RULE = {
    "beta_traced": "per model, from traced decode T (launch means) at P1 and P2: beta = (ln T_P2 - ln T_P1) / "
                   "(ln x_P1 - ln x_P2); point from cell means, envelope over the 2 x 2 launch pairings",
    "FAIL": "sign(delta_beta_traced) differs from sign(primary delta_beta), OR the traced envelopes overlap while the "
            "primary outcome is REPRODUCED or REVERSED",
    "NOT EVALUABLE": "primary explanandum NOT EVALUABLE, or a P1/P2 diagnostic cell of either model not 2/2 eligible",
    "PASS": "otherwise",
    "effect": "if the gate is not PASS every candidate test is NOT EVALUABLE; the untraced performance result stands; "
              "traced results are never primary throughput results",
}
CANDIDATE_RULES = {
    "a": {
        "premise": "historical EWS-fraction metric (EWS90 routed experts / total routed experts) reported separately",
        "endpoint": "m = decode store-read bytes / eligible decode graphs / F per diagnostic launch; e_a = "
                    "(ln m_P2 - ln m_P1) / (ln x_P2 - ln x_P1), envelope over the 2 x 2 launch pairings",
        "UNRESOLVED_first": "the 4 e_a values do not all have the same sign, or some m <= 0",
        "SUPPORTED": "max |e_a|80 < min |e_a|30 AND the M counterfactual moves beta80 toward beta30 at both bounds",
        "FALSIFIED AS STATED": "min |e_a|80 >= max |e_a|30",
        "WEAKENED": "the |e_a| envelopes overlap",
        "UNRESOLVED": "otherwise (80B smaller without overlap but the counterfactual does not move beta80 toward "
                      "beta30 at both bounds)",
        "NOT EVALUABLE": "a P1/P2 diagnostic cell not 2/2 eligible, or the gate is not PASS",
        "wording": "consistent with routing concentration contributing to the regime difference",
    },
    "b": {
        "device": "B_mix per model and round; comparison k: 80B at 2N vs 30B at N, N in {2, 4, 8}; favours 80B if "
                  "min over rounds B_mix80 > max over rounds B_mix30, disfavours if max80 < min30, else "
                  "indeterminate; device FAVOURS if all three favour, DISFAVOURS if all three disfavour, else "
                  "INDETERMINATE",
        "trace": "q = E_r / M (bounded); at a point 80B LOWER if max(80B q_hi over eligible launches) < min(30B q_lo), "
                 "HIGHER if min(80B q_lo) > max(30B q_hi), else indeterminate; trace LOWER/HIGHER if so at both "
                 "P1 and P2, else INDETERMINATE",
        "SUPPORTED": "device FAVOURS and trace LOWER",
        "FALSIFIED AS STATED": "device DISFAVOURS and trace HIGHER",
        "WEAKENED": "(FAVOURS and trace INDETERMINATE) or (device INDETERMINATE and LOWER)",
        "UNRESOLVED": "device and trace contradict (FAVOURS+HIGHER or DISFAVOURS+LOWER), a relevant traced depth "
                      "(P1/P2, both models) outside [2, 16], or any other evaluable combination",
        "NOT EVALUABLE": "required diagnostic cells not eligible, diskbench comparison incomplete, or gate not PASS",
        "limitation": "diskbench does not reproduce store asincrono behaviour; device evidence is device-level only",
    },
    "c": {
        "counters": "whole-process store counters (commit/decommit/lock/read) are DIAGNOSTIC ONLY",
        "FALSIFIED AS STATED": "at BOTH P1 and P2, in BOTH eligible 80B diagnostic launches: E_nr_hi < E_r_lo",
        "WEAKENED": "the falsification condition holds at exactly one of P1, P2; OR at both P1 and P2 "
                    "max over 80B launches of E_nr_hi / Res <= min over 30B launches of E_nr_lo / Res (the 80B is "
                    "demonstrably not more non-read dominated than the 30B)",
        "UNRESOLVED": "otherwise",
        "SUPPORTED": "NOT REACHABLE with existing instrumentation (E_nr also contains barrier waits and hit "
                     "lookups); no positive bookkeeping rule exists; inability to support (c) does not make it false",
        "NOT EVALUABLE": "required diagnostic cells not eligible or gate not PASS",
    },
    "d": {
        "endpoint": "N / T per launch: non-residency (compute-inclusive) time share; never 'pure compute'; Gated "
                    "DeltaNet itself is not isolated",
        "SUPPORTED": "at P1 AND P2: min(80B N/T) > max(30B N/T) AND the N counterfactual moves beta80 toward beta30",
        "FALSIFIED AS STATED": "at BOTH P1 and P2: max(80B N/T) < min(30B N/T)",
        "WEAKENED": "otherwise when evaluable",
        "NOT EVALUABLE": "required diagnostic cells not eligible or gate not PASS",
    },
}
LIFECYCLE_RULES = {
    "SUPPORTED": "explanandum REPRODUCED AND gate PASS AND at least one of (a), (b), (c), (d) SUPPORTED under its "
                 "own frozen rule (others may be anything)",
    "FALSIFIED": "explanandum REPRODUCED AND gate PASS AND all four candidates FALSIFIED AS STATED under their own "
                 "rules; UNRESOLVED, WEAKENED or NOT EVALUABLE never count as falsified",
    "INCONCLUSIVE": "every other case (explanandum NOT EVALUABLE, NOT DISTINGUISHED or REVERSED; gate failure; no "
                    "supported candidate while at least one candidate is not falsified)",
    "during_freeze": "PLANNED, measurements_made = false",
}
NOT_REPRODUCED_TEMPLATE = (
    "EXPLANANDUM NOT REPRODUCED. At matched budget/F (P1→P2, 4 generation threads, current runtime) the 80B's "
    "pressure-response exponent was {relation} the 30B's. The historical regime difference did not reproduce in the "
    "matched-F region on the current runtime. This does not retrospectively falsify EXP-006 or alter F-004. Those "
    "were measured over different budget/F regions on earlier binaries and remain valid descriptions of those "
    "sessions. The candidate-mechanism hypothesis is INCONCLUSIVE: its object was absent at matched budget/F, so the "
    "candidates could not be discriminated against it. Secondary results are reported descriptively.")
NOT_REPRODUCED_RELATION = {"NOT DISTINGUISHED": "not distinguishable from", "REVERSED": "steeper than"}
RQ009_RULES = {
    "ANSWERED": "explanandum evaluable and all four candidates resolved (SUPPORTED, WEAKENED or FALSIFIED AS STATED)",
    "PARTIALLY ANSWERED": "explanandum evaluable and at least one candidate resolved; the Unknown text is narrowed to "
                          "the remaining candidates",
    "otherwise": "RQ-009 remains PARTIALLY ANSWERED with a data note; it never regresses",
    "not_reproduced_note": "the historical slope difference is not reproduced at matched budget/F; consistent with a "
                           "normalization/region difference",
    "forbidden_wording": "caused by normalization",
    "open_questions_edit": "OPEN_QUESTIONS.md is not modified by the freeze; it is updated only at closure",
}
ROUTING_TAXONOMY = {
    "historical_ews_fraction": "EWS90 routed experts / total routed experts (the literal 'smaller EWS fraction'); "
                               "EXP-007 values are historical/descriptive premises only",
    "byte_weighted_variant": "EWS90 bytes / store bytes: NOT equivalent (expert size varies by layer in both models); "
                             "reported separately, labelled",
    "pressure_normalized_predictor": "EWS90 bytes / F with exact per-layer expert sizes: a predictor under budget/F "
                                     "normalization; not the historical statement and not a decision endpoint",
    "mechanism_consequence": "candidate (a) endpoint m and its elasticity e_a (see CANDIDATE_RULES['a'])",
}
TRACING_LIMITATIONS = (
    "traced and untraced launches are never pooled; traced T is used only for beta_traced, shares and bounds",
    "no tracing-overhead claim outside EXP-024's tested condition",
    "node residency is the thread-0 view; reads of other threads are counted by interval",
    "some completions are upper bounds, so E_r, E_nr, q and depth are bounds or bound-derived",
    "N is non-residency time and includes tracing cost",
)
KNOWN_LIMITATIONS = (
    "observational cross-model comparison: no randomization across models; architecture, quantization, expert "
    "count, slice sizes, routing and F differ together; only run order is randomized",
    "4 generation threads is the 80B's EXP-027 optimum at 4 GiB; the 30B sensitivity at 12 threads covers P1/P2 only",
    "the 80B is below every whole-layer size at P3 (672 MiB < 728 MiB): layer fit changes with pressure there",
    "one session per block, one workload per block type; the diagnostic workload differs from the performance prompt",
    "2 diagnostic launches per cell give weak non-overlap evidence",
    "diskbench is a device-level calibration, not a store emulator; 80B 2N vs 30B N is about 5% more bytes in flight "
    "on the 80B side (mean read 499,712 vs 952,320 B)",
    "the matched-F region (P1->P2 = 3.92F->1.47F) differs from the historical 80B region (5.97F->2.24F)",
    "causal language is limited to 'consistent with' / 'falsified as stated'",
)
REPORTING = {
    "secondary": "R = ms/token80 / ms/token30 at P1, P2, P3 (cell medians, launch ranges), descriptive",
    "sensitivity": "30B at 12 threads: beta (P1->P2) and envelope, operational comparison with the 80B at 4 threads; "
                   "descriptive, not part of the primary decision",
    "p3": "P3 performance and mechanism values are descriptive",
}


# ================================================================= checks ====

def check_plan(plan: List[Dict[str, Any]]) -> List[str]:
    bad: List[str] = []
    by = {b: [x for x in plan if x["block"] == b] for b in BLOCK_ORDER}
    if [x["block"] for x in plan] != sorted((x["block"] for x in plan), key=BLOCK_ORDER.index):
        bad.append("blocks are not in the frozen order")
    perf = by["perf"]
    n_prim = sum(1 for x in perf if x["role"] == "primary")
    n_sens = sum(1 for x in perf if x["role"] == "sensitivity")
    got = {"perf_primary": n_prim, "perf_sensitivity": n_sens, "diag": len(by["diag"]),
           "model_launches": len(perf) + len(by["diag"]), "diskbench_runs": len(by["diskbench"])}
    for k, v in EXPECTED.items():
        if got[k] != v:
            bad.append(f"{k}: {got[k]} scheduled, frozen {v}")
    for b in BLOCK_ORDER:
        rows = by[b]
        cells = sorted({c["cell"] for c in {"perf": perf_cells, "diag": diag_cells, "diskbench": diskbench_cells}[b]()})
        if [x["launch_index"] for x in rows] != list(range(1, len(rows) + 1)):
            bad.append(f"{b}: launch indices are not 1..N")
        for rnd in sorted({x["round"] for x in rows}):
            ids = [x["cell"] for x in rows if x["round"] == rnd]
            if sorted(ids) != cells:
                bad.append(f"{b} round {rnd}: not every cell exactly once")
        for x in rows:
            a = x["argv"]
            if "--dry-run" in a:
                bad.append(f"{b} {x['launch_index']}: a real launch carries --dry-run")
            if b == "diskbench":
                want = {"--kinds": DISKBENCH_KIND, "--slice": str(x["slice_bytes"]), "--threads": str(x["readers"]),
                        "--seconds": str(DISKBENCH_SECONDS)}
                if a[1] != DISKBENCH_FILE or "--no-sequential" not in a or "--json" not in a or "moe" in a:
                    bad.append(f"diskbench {x['launch_index']}: file/flags differ from the frozen command")
            else:
                want = {"--backend": BACKEND, "--read-mode": READ_MODE, "--threads": str(x["threads"]),
                        "--ctx": str(CTX), "--batch": str(BATCH), "--ubatch": str(UBATCH),
                        "--repeat": str(REPEAT), "--retries": str(RETRIES)}
                want["--expert-budget" if b == "perf" else "--budgets"] = x["budget"]
                if b == "perf":
                    want["--n-predict"] = str(N_PREDICT)
                else:
                    want.update({"--workload": DIAG_WORKLOAD["name"], "--workload-version": DIAG_WORKLOAD["version"],
                                 "--split": DIAG_WORKLOAD["split"]})
                if "--expert-store" not in a or "--expert-stats" not in a:
                    bad.append(f"{b} {x['launch_index']}: --expert-store/--expert-stats missing")
                if x["threads"] != (SENSITIVITY_THREADS if x.get("role") == "sensitivity" else PRIMARY_THREADS):
                    bad.append(f"{b} {x['launch_index']}: thread rule violated")
            for f, v in want.items():
                if _flag(a, f) != v:
                    bad.append(f"{b} {x['launch_index']}: {f} = {_flag(a, f)!r}, expected {v!r}")
    return bad


def check_geometry(root: Path = ROOT) -> List[str]:
    bad: List[str] = []
    for m in MODEL_ORDER:
        meta = json.loads((root / MODELS[m]["geometry_source"]).read_text(encoding="utf-8"))["trace_meta"]
        eb = {int(k): int(v) for k, v in meta["expert_bytes"].items()}
        if eb != layer_expert_bytes(m):
            bad.append(f"{m}: per-layer expert bytes of the tracked trace_meta differ from the frozen slice table")
        if int(meta["top_k"]) != MODELS[m]["top_k"] or int(meta["n_experts"]) != MODELS[m]["n_experts"]:
            bad.append(f"{m}: top_k / n_experts differ from the tracked trace_meta")
        if footprint(m) != MODELS[m]["footprint_bytes"]:
            bad.append(f"{m}: F re-derived {footprint(m)} != frozen {MODELS[m]['footprint_bytes']}")
        if max(whole_layer_bytes(m).values()) != MODELS[m]["whole_layer_bytes_max"]:
            bad.append(f"{m}: largest whole layer differs")
        if sum(b * n for b, n in reads_per_token(m).items()) != MODELS[m]["footprint_bytes"]:
            bad.append(f"{m}: reads-per-token weights do not reconstruct F")
        for kind, sizes in MODELS[m]["slices"].items():
            layers = sorted(l for ls in sizes.values() for l in ls)
            if layers != list(range(MODELS[m]["n_layers"])):
                bad.append(f"{m} {kind}: slice table does not cover every layer exactly once")
            for b in sizes:
                if b % 4096:
                    bad.append(f"{m} {kind}: {b} is not 4096-aligned")
    if sorted(DISKBENCH_SIZES) != sorted(set(slice_sizes("30b")) | set(slice_sizes("80b"))):
        bad.append("diskbench sizes are not exactly the verified distinct slice sizes")
    return bad


def check_budgets() -> List[str]:
    bad: List[str] = []
    sys.path.insert(0, str(ROOT))
    from argine.util import parse_size  # noqa: E402
    for m in MODEL_ORDER:
        for p in POINTS:
            label, b = BUDGETS[m][p]
            if parse_size(label) != b:
                bad.append(f"{m} {p}: parse_size({label}) = {parse_size(label)}, frozen {b}")
            if b % MIB or b % MAX_SHARDS or b < 512 * MIB or shard_count(b) != MAX_SHARDS:
                bad.append(f"{m} {p}: not a whole-MiB, 16-shard budget >= 512 MiB")
        if abs(x_of(m, "P1") / x_of(m, "P2") - 8 / 3) > 1e-9:
            bad.append(f"{m}: P1/P2 span is not 8/3")
    for p in POINTS:
        if abs(x_of("80b", p) / x_of("30b", p) - 1) > 1e-3:
            bad.append(f"{p}: budget/F not matched within 0.1% across models")
    return bad


def check_sources(root: Path = ROOT) -> List[str]:
    bad = list(X22.check_runtime_sources(root))
    for group in (RUNTIME_SOURCE_SNIPPETS, DISKBENCH_SOURCE_SNIPPETS):
        for rel, needles in group.items():
            text = (root / rel).read_text(encoding="utf-8", errors="replace")
            for n in needles:
                if n not in text:
                    bad.append(f"{rel}: expected text not found: {n!r}")
    sys.path.insert(0, str(root))
    from argine import cli  # noqa: E402
    for s in DISKBENCH_SIZES:
        a = cli.build_parser().parse_args(diskbench_args(s, 8))
        if (a.kinds != [DISKBENCH_KIND] or a.slice != [str(s)] or a.threads != ["8"] or not a.no_sequential
                or not a.json or float(a.seconds) != DISKBENCH_SECONDS or a.file != DISKBENCH_FILE):
            bad.append(f"argine diskbench parser does not read the frozen command as intended (slice {s})")
    return bad


def check_dependency() -> List[str]:
    bad = []
    if X22._lf_sha(EXP022_TOOL) != EXP022_TOOL_SHA256:  # noqa: SLF001
        bad.append("tools/exp022_run.py differs from the pinned version whose functions EXP-026 reuses")
    for name in EXP022_REUSED:
        if not hasattr(X22, name):
            bad.append(f"tools/exp022_run.py lacks {name}")
    if X22.READ_MODE != READ_MODE or X22.BACKEND != BACKEND or X22.N_PREDICT != N_PREDICT or X22.RETRIES != 0:
        bad.append("EXP-022 runtime constants reused by EXP-026 changed")
    for m in MODEL_ORDER:
        for k in ("path", "store", "n_layers", "n_experts", "top_k", "footprint_bytes"):
            if X22.MODELS[m][k] != MODELS[m][k]:
                bad.append(f"EXP-022 model table differs for {m} {k} (reused log/provenance checks read it)")
    return bad


def _readme_text(readme: Path = README) -> str:
    return readme.read_text(encoding="utf-8").replace("\r\n", "\n")


def check_docs(readme: Path = README, config: Path = CONFIG, oq: Path = OPEN_QUESTIONS) -> List[str]:
    bad: List[str] = []
    t = _readme_text(readme)
    if HISTORICAL_HYPOTHESIS not in t:
        bad.append("README does not contain the 2026-10-03 hypothesis verbatim")
    if HISTORICAL_QUESTION not in t:
        bad.append("README does not contain the historical research question verbatim")
    if t.count(MARK_BEGIN) != 1 or t.count(MARK_END) != 1 or t.index(MARK_BEGIN) > t.index(MARK_END):
        bad.append("README protocol markers missing, duplicated or reversed")
    if CANDIDATE_D_TEXT not in oq.read_text(encoding="utf-8"):
        bad.append("OPEN_QUESTIONS.md no longer contains candidate (d) verbatim")
    cfg = json.loads(config.read_text(encoding="utf-8"))
    st = re.search(r"\|\s*Status\s*\|\s*\*\*([A-Z]+)\*\*", t)
    if not st or st.group(1) != cfg.get("status"):
        bad.append("README status and config.json status differ")
    proto = dict(cfg.get("protocol") or {})
    proto.pop("measurements_made", None)
    want = config_protocol()
    want.pop("measurements_made", None)
    if proto != want:
        bad.append("config.json protocol object differs from the harness (regenerate it, do not hand-edit)")
    return bad


def measurements_made(config: Path = CONFIG) -> bool:
    return bool((json.loads(config.read_text(encoding="utf-8")).get("protocol") or {}).get("measurements_made"))


def check_outputs(exp: Path = EXP, config: Path = CONFIG) -> List[str]:
    bad: List[str] = []
    for d in ("raw", "processed", "figures"):
        if not (exp / d).is_dir():
            bad.append(f"{d}/ missing")
    if not measurements_made(config):
        if (exp / "raw").is_dir():
            for p in sorted((exp / "raw").iterdir()):
                if p.name == ".gitkeep":
                    continue
                if not harness_session_dir(p):
                    bad.append(f"raw/{p.name} is not a session written by this harness (measurements_made is false)")
        for d in ("processed", "figures"):
            extra = sorted(p.name for p in (exp / d).iterdir() if p.name != ".gitkeep") if (exp / d).is_dir() else []
            if extra:
                bad.append(f"{d}/ contains {extra} while measurements_made is false")
        if (exp / "results.csv").exists():
            bad.append("results.csv exists while measurements_made is false")
    return bad


def harness_session_dir(p: Path) -> bool:
    """A raw/<block> directory created by --run of this frozen protocol (session.json names it)."""
    s = p / "session.json"
    if not (p.is_dir() and p.name in BLOCK_ORDER and s.exists()):
        return False
    try:
        d = json.loads(s.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    return d.get("experiment") == EXP_ID and d.get("protocol_version") == PROTOCOL_VERSION and d.get("block") == p.name


def static_problems() -> List[str]:
    """Everything --check verifies. Repository files only; no model, store, binary or device is touched."""
    problems: List[str] = []
    for name, f in (("plan", lambda: check_plan(build_plan())), ("geometry", check_geometry),
                    ("budgets", check_budgets), ("sources", check_sources), ("dependency", check_dependency),
                    ("timeouts", X22.check_timeout_sources), ("workloads", X22.check_workloads),
                    ("docs", check_docs), ("outputs", check_outputs)):
        try:
            problems += [f"{name}: {p}" for p in f()]
        except Exception as e:  # noqa: BLE001
            problems.append(f"{name}: check failed: {e}")
    return problems


# ============================================================ spec / freeze ====

def config_protocol() -> Dict[str, Any]:
    """The structured protocol object of config.json (generated; never hand-edited)."""
    return {
        "version": PROTOCOL_VERSION,
        "freeze": f"experiments/{EXP_ID}/protocol_freeze.json",
        "tool": "tools/exp026_run.py", "tests": "tests/test_exp026.py",
        "kind": "prospective; frozen before any EXP-026 measurement",
        "primary_explanandum": "delta_beta = beta_80b - beta_30b, beta = dln(gen_tok_s)/dln(budget/F) P1->P2, untraced, "
                               "4 generation threads; outcomes REPRODUCED / REVERSED / NOT DISTINGUISHED / NOT EVALUABLE",
        "F30": MODELS["30b"]["footprint_bytes"], "F80": MODELS["80b"]["footprint_bytes"],
        "models": {m: {"name": MODELS[m]["name"], "path": MODELS[m]["path"], "store": MODELS[m]["store"],
                       "top_k": MODELS[m]["top_k"], "n_experts": MODELS[m]["n_experts"],
                       "footprint_bytes": MODELS[m]["footprint_bytes"],
                       "whole_layer_bytes_max": MODELS[m]["whole_layer_bytes_max"],
                       "slice_sizes": slice_sizes(m)} for m in MODEL_ORDER},
        "budgets_bytes": {m: {p: {"label": BUDGETS[m][p][0], "bytes": BUDGETS[m][p][1]} for p in POINTS}
                          for m in MODEL_ORDER},
        "points": {p: POINT_ROLES[p] for p in POINTS},
        "threads": {"primary_and_diagnostic": PRIMARY_THREADS,
                    "sensitivity": {"model": SENSITIVITY_MODEL, "threads": SENSITIVITY_THREADS,
                                    "points": list(SENSITIVITY_POINTS)}},
        "prompt_threads": BATCH_THREADS,
        "workloads": {"performance": {"prompt": "argine.bench.DEFAULT_PROMPT", "sha256": PERF_PROMPT_SHA256,
                                      "n_predict": N_PREDICT, "tracing": False},
                      "diagnostic": {"name": DIAG_WORKLOAD["name"], "version": DIAG_WORKLOAD["version"],
                                     "split": DIAG_WORKLOAD["split"], "sha256": DIAG_WORKLOAD["sha256"],
                                     "tracing": True}},
        "blocks": list(BLOCK_ORDER),
        "performance_replicas": 3, "diagnostic_replicas": 2,
        "launches": EXPECTED["model_launches"],
        "launch_breakdown": {k: EXPECTED[k] for k in ("perf_primary", "perf_sensitivity", "diag")},
        "diskbench": {"sizes": list(DISKBENCH_SIZES), "readers": list(DISKBENCH_READERS),
                      "repetitions": DISKBENCH_ROUNDS, "runs": EXPECTED["diskbench_runs"], "kind": DISKBENCH_KIND,
                      "file": DISKBENCH_FILE, "seconds": DISKBENCH_SECONDS, "role": "device-level calibration only"},
        **NO_RETRY_RULES,
        "performance_eligibility": "3/3 VALID", "diagnostic_eligibility": "2/2 eligible",
        "outcome_taxonomy": list(OUTCOMES), "outcome_precedence": "INVALID > STORE_RESIDENCY_ABORT > NON_COMPLETION > VALID",
        "seeds": dict(SEEDS), "cooldown_s": COOLDOWN_S,
        "accounting": "BOUND DECOMPOSITION ONLY",
        "decision_rules": {"explanandum": EXPLANANDUM_RULE, "gate": GATE_RULE, "candidates": CANDIDATE_RULES},
        "lifecycle_rules": LIFECYCLE_RULES, "rq009_rules": RQ009_RULES,
        "measurements_made": False,
    }


def protocol_spec() -> Dict[str, Any]:
    plan = build_plan()
    return {
        "experiment": EXP_ID, "protocol_version": PROTOCOL_VERSION,
        "freeze_kind": "prospective: written before any EXP-026 measurement; no measurement exists",
        "historical_question_verbatim": HISTORICAL_QUESTION,
        "historical_hypothesis_verbatim": HISTORICAL_HYPOTHESIS,
        "candidates": CANDIDATES,
        "explanandum": EXPLANANDUM_RULE,
        "models": {m: {"name": MODELS[m]["name"], "path": MODELS[m]["path"], "store": MODELS[m]["store"],
                       "n_layers": MODELS[m]["n_layers"], "n_experts": MODELS[m]["n_experts"],
                       "top_k": MODELS[m]["top_k"], "footprint_bytes": MODELS[m]["footprint_bytes"],
                       "slices": {k: {str(b): list(ls) for b, ls in v.items()} for k, v in MODELS[m]["slices"].items()},
                       "layer_expert_bytes": {str(l): b for l, b in layer_expert_bytes(m).items()},
                       "whole_layer_bytes": {str(l): b for l, b in whole_layer_bytes(m).items()},
                       "reads_per_token": {str(b): n for b, n in reads_per_token(m).items()},
                       "geometry_source": MODELS[m]["geometry_source"]} for m in MODEL_ORDER},
        "geometry_provenance": GEOMETRY_PROVENANCE,
        "historical_ews_premise_exp007": EXP007_EWS90,
        "budgets": {m: budget_table(m) for m in MODEL_ORDER}, "not_in_grid": list(NOT_IN_GRID),
        "runtime": {"binary": "llama-research-v2/llama-server.exe", "server_sha256": X22.REQUIRED_SERVER_SHA256,
                    "build_info_sha256": X22.REQUIRED_BUILD_INFO_SHA256, "llama_cpp_commit": X22.REQUIRED_LLAMA_CPP_COMMIT,
                    "instrumentation": X22.INSTRUMENTATION_ID, "backend": BACKEND, "store_read": STORE_READ_SEMANTICS,
                    "ctx": CTX, "batch": BATCH, "ubatch": UBATCH, "kv_type": KV_TYPE, "batch_threads": BATCH_THREADS,
                    "primary_threads": PRIMARY_THREADS,
                    "sensitivity": {"model": SENSITIVITY_MODEL, "threads": SENSITIVITY_THREADS,
                                    "points": list(SENSITIVITY_POINTS)},
                    "repeat": REPEAT, "retries": RETRIES},
        "workloads": {"performance": {"prompt": "argine.bench.DEFAULT_PROMPT", "sha256": PERF_PROMPT_SHA256,
                                      "chars": PERF_PROMPT_CHARS, "n_predict": N_PREDICT, "tracing": False},
                      "diagnostic": DIAG_WORKLOAD, "note": WORKLOAD_NOTE},
        "diskbench": {"role": DISKBENCH_ROLE, "file": DISKBENCH_FILE, "kind": DISKBENCH_KIND, "sizes": list(DISKBENCH_SIZES),
                      "readers": list(DISKBENCH_READERS), "rounds": DISKBENCH_ROUNDS, "seconds": DISKBENCH_SECONDS,
                      "cap_s": DISKBENCH_CAP_S, "pairs_30b_80b": [list(p) for p in DISKBENCH_PAIRS],
                      "b_mix": "B_mix(N) = sum bytes / sum (bytes / B(slice, N)), bytes = reads_per_token x slice",
                      "telemetry": DISKBENCH_TELEMETRY, "outcomes": list(DISKBENCH_OUTCOMES), "cooldown_s": 0},
        "blocks": list(BLOCK_ORDER), "seeds": SEEDS, "permutation_rule": PERMUTATION_RULE,
        "orders": {b: frozen_rounds(b) for b in BLOCK_ORDER},
        "launch_order": [{k: x.get(k) for k in ("block", "launch_index", "round", "position_in_round", "cell",
                                                 "model", "point", "threads", "budget", "budget_bytes",
                                                 "slice_bytes", "readers")} for x in plan],
        "expected_counts": EXPECTED, "no_retry": NO_RETRY_RULES,
        "cooldown_s": {"between_model_launches": COOLDOWN_S, "diskbench": 0},
        "completion_caps_s": {"perf": X22.cap_seconds("perf"), "diag": X22.cap_seconds("diag"),
                              "diskbench": DISKBENCH_CAP_S},
        "outcome_taxonomy": {"models": list(OUTCOMES), "precedence": "INVALID > STORE_RESIDENCY_ABORT > "
                             "NON_COMPLETION > VALID (tools/exp022_run.classify, unchanged; store aborts are causally "
                             "neutral, never 'infeasible')", "diskbench": list(DISKBENCH_OUTCOMES),
                             "every_scheduled_launch_is_kept": True},
        "eligibility": ELIGIBILITY_RULES, "accounting": ACCOUNTING, "counterfactual": COUNTERFACTUAL_RULE,
        "gate": GATE_RULE, "candidates_rules": CANDIDATE_RULES, "routing_taxonomy": ROUTING_TAXONOMY,
        "lifecycle": LIFECYCLE_RULES, "not_reproduced_template": NOT_REPRODUCED_TEMPLATE,
        "not_reproduced_relation": NOT_REPRODUCED_RELATION, "rq009": RQ009_RULES, "reporting": REPORTING,
        "tracing_limitations": list(TRACING_LIMITATIONS), "known_limitations": list(KNOWN_LIMITATIONS),
        "reused_exp022": {"file": "tools/exp022_run.py", "lf_sha256": EXP022_TOOL_SHA256, "functions": list(EXP022_REUSED)},
    }


def spec_sha256() -> str:
    return hashlib.sha256(json.dumps(protocol_spec(), sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def protocol_text(readme: Path = README) -> str:
    t = _readme_text(readme)
    if t.count(MARK_BEGIN) != 1 or t.count(MARK_END) != 1:
        raise ProtocolError("README protocol markers missing or duplicated")
    return t.split(MARK_BEGIN, 1)[1].split(MARK_END, 1)[0]


def config_protocol_sha256(config: Path = CONFIG) -> str:
    p = dict(json.loads(config.read_text(encoding="utf-8")).get("protocol") or {})
    p.pop("measurements_made", None)
    return hashlib.sha256(json.dumps(p, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def freeze_hashes() -> Dict[str, str]:
    return {"protocol_sha256": hashlib.sha256(protocol_text().encode("utf-8")).hexdigest(),
            "spec_sha256": spec_sha256(),
            "config_protocol_sha256": config_protocol_sha256(),
            "tool_sha256": X22._lf_sha(Path(__file__).resolve()),  # noqa: SLF001
            "tests_sha256": X22._lf_sha(TESTS),  # noqa: SLF001
            "exp022_dependency_sha256": X22._lf_sha(EXP022_TOOL)}  # noqa: SLF001


HASH_METHODS = {
    "protocol_sha256": "sha256 of the UTF-8 README text strictly between the EXP-026 markers, AFTER CRLF->LF normalisation",
    "spec_sha256": "sha256 of json.dumps(protocol_spec(), sort_keys=True, ensure_ascii=False) (UTF-8; no line endings involved)",
    "config_protocol_sha256": "sha256 of json.dumps(config.json['protocol'] without 'measurements_made', sort_keys=True, "
                              "ensure_ascii=False) (parsed JSON; independent of line endings)",
    "tool_sha256": "sha256 of tools/exp026_run.py bytes AFTER CRLF->LF normalisation",
    "tests_sha256": "sha256 of tests/test_exp026.py bytes AFTER CRLF->LF normalisation",
    "exp022_dependency_sha256": "sha256 of tools/exp022_run.py bytes AFTER CRLF->LF normalisation",
}


def freeze_status(freeze: Path = FREEZE, now: Optional[Dict[str, str]] = None) -> Tuple[bool, str]:
    if not freeze.exists():
        return False, "protocol_freeze.json absent (run --freeze)"
    f = json.loads(freeze.read_text(encoding="utf-8"))
    now = freeze_hashes() if now is None else now
    diff = [k for k in now if f.get(k) != now[k]]
    return (not diff), ("matches the frozen protocol" if not diff else "CHANGED since freeze: " + ", ".join(diff))


def cmd_freeze() -> int:
    problems = static_problems()
    if problems:
        print("refusing to freeze: --check fails:\n  - " + "\n  - ".join(problems), file=sys.stderr)
        return 1
    now = freeze_hashes()
    if FREEZE.exists():
        old = json.loads(FREEZE.read_text(encoding="utf-8"))
        if all(old.get(k) == v for k, v in now.items()):
            print("protocol_freeze.json already matches; nothing written")
            return 0
        print("protocol_freeze.json exists and differs: archive it byte-for-byte as protocol_freeze_<label>.json "
              "and document the amendment before refreezing", file=sys.stderr)
        return 1
    snap = git_snapshot()
    rec = {"experiment": EXP_ID, "protocol_version": PROTOCOL_VERSION, "freeze_date": FREEZE_DATE,
           "status": "prospective freeze before any EXP-026 measurement",
           "pre_freeze_head": PRE_FREEZE_HEAD,
           "argine2_at_freeze": {"head": snap.get("head"), "error": snap.get("error"), "porcelain": snap.get("porcelain"),
                                 "note": "the protocol files are uncommitted at freeze time by construction; --run "
                                         "requires them committed, HEAD descending from pre_freeze_head, and a clean tree"},
           "hash_methods": HASH_METHODS, **now,
           "workload": {"diagnostic": {k: DIAG_WORKLOAD[k] for k in ("name", "version", "split", "file", "sha256")},
                        "performance_prompt_sha256": PERF_PROMPT_SHA256},
           "models": {m: {"name": MODELS[m]["name"], "path": MODELS[m]["path"], "store": MODELS[m]["store"]}
                      for m in MODEL_ORDER},
           "runtime_identity": {"server_sha256": X22.REQUIRED_SERVER_SHA256,
                                "build_info_sha256": X22.REQUIRED_BUILD_INFO_SHA256,
                                "llama_cpp_commit": X22.REQUIRED_LLAMA_CPP_COMMIT,
                                "source": "llama-research-v2/BUILD-INFO.json (checked statically by --check)"},
           "seeds": SEEDS, "orders": {b: frozen_rounds(b) for b in BLOCK_ORDER},
           "scheduled_model_launches": EXPECTED["model_launches"],
           "scheduled_diskbench_runs": EXPECTED["diskbench_runs"],
           "measurements_made": False,
           "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "spec": protocol_spec()}
    FREEZE.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {FREEZE.relative_to(ROOT)}: " + ", ".join(f"{k} {v[:12]}" for k, v in now.items()))
    return 0


# ============================================================== git gate ====

REQUIRED_TRACKED = (
    f"experiments/{EXP_ID}/README.md", f"experiments/{EXP_ID}/config.json",
    f"experiments/{EXP_ID}/protocol_freeze.json", "tools/exp026_run.py", "tests/test_exp026.py",
    "tools/exp022_run.py",
)


def git_snapshot(root: Path = ROOT) -> Dict[str, Any]:
    snap: Dict[str, Any] = {"available": False, "head": None, "tracked": {}, "porcelain": None, "error": None,
                            "freeze_head_is_ancestor": None, "registry_head": None}
    try:
        h = X22._git(root, "rev-parse", "--verify", "HEAD^{commit}")  # noqa: SLF001
    except Exception as e:  # noqa: BLE001
        snap["error"] = f"git unavailable: {e}"
        return snap
    snap["available"] = True
    if h.returncode == 0 and h.stdout.strip():
        snap["head"] = h.stdout.strip()
    else:
        snap["error"] = (h.stderr or h.stdout or "").strip()[:300] or "HEAD does not resolve"
    try:
        for f in REQUIRED_TRACKED:
            snap["tracked"][f] = X22._git(root, "ls-files", "--error-unmatch", "--", f).returncode == 0  # noqa: SLF001
        s = X22._git(root, "status", "--porcelain")  # noqa: SLF001
        snap["porcelain"] = s.stdout.splitlines() if s.returncode == 0 else None
        snap["freeze_head_is_ancestor"] = X22._git(root, "merge-base", "--is-ancestor", PRE_FREEZE_HEAD,  # noqa: SLF001
                                                   "HEAD").returncode == 0
        r = X22._git(root, "show", f"HEAD:{X22.VALIDATIONS_REGISTRY}")  # noqa: SLF001
        snap["registry_head"] = r.stdout if r.returncode == 0 else None
    except Exception as e:  # noqa: BLE001
        snap["error"] = f"git failed: {e}"
    return snap


def run_gate_problems(snap: Dict[str, Any], freeze_ok: bool, freeze_why: str,
                      registry_now: Optional[str] = None, expected_head: Optional[str] = None) -> List[str]:
    bad: List[str] = []
    if not snap.get("available"):
        return [snap.get("error") or "git unavailable"]
    if not snap.get("head"):
        bad.append(f"HEAD does not resolve to a commit ({snap.get('error')})")
    for f in REQUIRED_TRACKED:
        if not (snap.get("tracked") or {}).get(f):
            bad.append(f"not tracked by Git: {f}")
    porcelain = snap.get("porcelain")
    if porcelain is None:
        bad.append("git status --porcelain could not be read")
    else:
        dirty = [l for l in porcelain if l.strip()]
        reg = [l for l in dirty if l[3:].strip().replace("\\", "/") == X22.VALIDATIONS_REGISTRY and l[:2] == " M"]
        other = [l for l in dirty if l not in reg]
        if other:
            bad.append("working tree not clean (git status --porcelain, untracked included): " + "; ".join(other)[:600])
        if reg:
            bad += X22.registry_append_only(snap.get("registry_head"), registry_now or "", f"experiments/{EXP_ID}/raw/")
    if snap.get("freeze_head_is_ancestor") is not True:
        bad.append("the pre-freeze HEAD is not an ancestor of the current HEAD (or could not be checked)")
    if expected_head and snap.get("head") != expected_head:
        bad.append(f"HEAD {snap.get('head')} differs from the HEAD of the earlier blocks {expected_head}")
    if not freeze_ok:
        bad.append("freeze: " + freeze_why)
    return bad


def require_run_gate(expected_head: Optional[str] = None, snapshot_fn=None, freeze_fn=None) -> Dict[str, Any]:
    snap = (snapshot_fn or git_snapshot)()
    ok, why = (freeze_fn or freeze_status)()
    reg = ROOT / X22.VALIDATIONS_REGISTRY
    bad = run_gate_problems(snap, ok, why, reg.read_text(encoding="utf-8") if reg.exists() else None, expected_head)
    if bad:
        raise ProtocolError("real run refused (no session, no model, no benchmark, no trace, no diskbench):\n  - "
                            + "\n  - ".join(bad))
    return {"checked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "head": snap["head"],
            "tracked": snap["tracked"], "freeze": why}


# ============================================================ arithmetic ====

def beta_from(s1: float, s2: float, x1: float, x2: float) -> float:
    """beta = (ln s1 - ln s2) / (ln x1 - ln x2); s = speed (tok/s or 1/T), x = budget/F."""
    return (math.log(s1) - math.log(s2)) / (math.log(x1) - math.log(x2))


def envelope(v1: Sequence[float], v2: Sequence[float], x1: float, x2: float) -> Tuple[float, float]:
    """[min, max] of beta over every pairing of one P1 value with one P2 value."""
    bs = [beta_from(a, b, x1, x2) for a in v1 for b in v2]
    return min(bs), max(bs)


def compare_envelopes(env80: Tuple[float, float], env30: Tuple[float, float]) -> str:
    if env80[1] < env30[0]:
        return "REPRODUCED"
    if env80[0] > env30[1]:
        return "REVERSED"
    return "NOT DISTINGUISHED"


def _sign(v: float) -> int:
    return (v > 0) - (v < 0)


def perf_cell(launches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """launches: the scheduled launches of one cell, each {'outcome', 'gen_tok_s'}. 3/3 VALID or NOT EVALUABLE."""
    if len(launches) > 3:
        raise ProtocolError(f"{len(launches)} launches in a performance cell; exactly 3 are scheduled "
                            "(no retry, no replacement)")
    valid = [float(l["gen_tok_s"]) for l in launches if l.get("outcome") == "VALID" and l.get("gen_tok_s")]
    out: Dict[str, Any] = {"scheduled": 3, "recorded": len(launches),
                           "outcomes": Counter(l.get("outcome") for l in launches), "valid_values": valid}
    if len(launches) == 3 and len(valid) == 3:
        out.update({"evaluable": True, "center": statistics.median(valid), "center_kind": "median of 3 VALID"})
    else:
        out.update({"evaluable": False, "status": "NOT EVALUABLE",
                    "descriptive_only": (None if not valid else
                                         {"center": statistics.mean(valid) if len(valid) == 2 else statistics.median(valid),
                                          "center_kind": ("two-value center (mean of 2 VALID launches), DESCRIPTIVE ONLY"
                                                          if len(valid) == 2 else f"{len(valid)} VALID launch(es), DESCRIPTIVE ONLY"),
                                          "values": valid})})
    out["outcomes"] = dict(out["outcomes"])
    return out


def explanandum(cells: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, Any]:
    """cells[(model, point)] = perf_cell(...) for the primary (4-thread) cells."""
    need = [(m, p) for m in MODEL_ORDER for p in PRIMARY_POINTS]
    missing = [f"{m}-{p}" for m, p in need if not (cells.get((m, p)) or {}).get("evaluable")]
    if missing:
        return {"outcome": "NOT EVALUABLE", "reason": "primary cell(s) not 3/3 VALID: " + ", ".join(missing)}
    beta, env = {}, {}
    for m in MODEL_ORDER:
        x1, x2 = x_of(m, "P1"), x_of(m, "P2")
        beta[m] = beta_from(cells[(m, "P1")]["center"], cells[(m, "P2")]["center"], x1, x2)
        env[m] = envelope(cells[(m, "P1")]["valid_values"], cells[(m, "P2")]["valid_values"], x1, x2)
    outcome = compare_envelopes(env["80b"], env["30b"])
    return {"outcome": outcome, "beta": beta, "envelope": env, "delta_beta": beta["80b"] - beta["30b"]}


def traced_gate(primary: Dict[str, Any], T: Dict[Tuple[str, str], Optional[List[float]]]) -> Dict[str, Any]:
    """T[(model, point)] = per-launch mean decode T of an ELIGIBLE (2/2) diagnostic cell, else None."""
    if primary.get("outcome") == "NOT EVALUABLE":
        return {"status": "NOT EVALUABLE", "reason": "primary explanandum NOT EVALUABLE"}
    need = [(m, p) for m in MODEL_ORDER for p in PRIMARY_POINTS]
    if any(not T.get(k) for k in need):
        return {"status": "NOT EVALUABLE", "reason": "a P1/P2 diagnostic cell is not 2/2 eligible"}
    beta, env = {}, {}
    for m in MODEL_ORDER:
        x1, x2 = x_of(m, "P1"), x_of(m, "P2")
        t1, t2 = T[(m, "P1")], T[(m, "P2")]
        beta[m] = beta_from(1 / statistics.mean(t1), 1 / statistics.mean(t2), x1, x2)
        env[m] = envelope([1 / t for t in t1], [1 / t for t in t2], x1, x2)
    d_tr = beta["80b"] - beta["30b"]
    cls = compare_envelopes(env["80b"], env["30b"])
    fail = []
    if _sign(d_tr) != _sign(primary["delta_beta"]):
        fail.append("sign of delta_beta_traced differs from the primary delta_beta")
    if primary["outcome"] in ("REPRODUCED", "REVERSED") and cls == "NOT DISTINGUISHED":
        fail.append("traced envelopes overlap while the primary outcome is " + primary["outcome"])
    return {"status": "FAIL" if fail else "PASS", "reason": "; ".join(fail) or "agreement", "beta_traced": beta,
            "envelope_traced": env, "delta_beta_traced": d_tr, "traced_classification": cls}


# ======================================================= bound accounting ====

def union(intervals: Iterable[Sequence[int]]) -> List[List[int]]:
    iv = sorted([int(a), int(b)] for a, b in intervals if b > a)
    out: List[List[int]] = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def length(u: Sequence[Sequence[int]]) -> int:
    return sum(b - a for a, b in u)


def intersection_length(u: Sequence[Sequence[int]], v: Sequence[Sequence[int]]) -> int:
    i = j = tot = 0
    while i < len(u) and j < len(v):
        a, b = max(u[i][0], v[j][0]), min(u[i][1], v[j][1])
        if b > a:
            tot += b - a
        if u[i][1] < v[j][1]:
            i += 1
        else:
            j += 1
    return tot


def graph_accounting(routes: Iterable[Sequence[Any]], requests: Sequence[Tuple[int, int, str]],
                     nodes: Iterable[Tuple[int, int, int]], reads: Iterable[Tuple[int, int, bool, int]]
                     ) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Per eligible decode graph: T, Res, N (exact) and the E_r / E_nr bounds.

    routes: (graph, ts, layer, n_tokens, ...); requests: (start, end, segment) in the trace clock;
    nodes: (graph, start_ns, ready_ns); reads: (issue_ns, complete_ns, exact, bytes)."""
    t0: Dict[int, int] = {}
    ntok: Dict[int, int] = {}
    for r in routes:
        g, ts, n = int(r[0]), int(r[1]), int(r[3])
        t0[g] = min(t0.get(g, ts), ts)
        ntok[g] = max(ntok.get(g, 0), n)
    by_node: Dict[int, List[List[int]]] = {}
    for g, a, b in nodes:
        by_node.setdefault(int(g), []).append([int(a), int(b)])
    rd = sorted((int(a), int(b), bool(e), int(by)) for a, b, e, by in reads)
    issues = [r[0] for r in rd]

    def req_of(t: int) -> Optional[int]:
        for i, (a, b, seg) in enumerate(requests):
            if a <= t <= b and seg not in ("warmup", "flush-sentinel"):
                return i
        return None

    import bisect
    rows: List[Dict[str, Any]] = []
    problems: List[str] = []
    for g in sorted(t0):
        if ntok[g] != 1 or (g + 1) not in t0 or ntok[g + 1] != 1:
            continue
        r1, r2 = req_of(t0[g]), req_of(t0[g + 1])
        if r1 is None or r1 != r2:
            continue
        T = t0[g + 1] - t0[g]
        iv = sorted(by_node.get(g, []))
        res = sum(b - a for a, b in iv)
        res_u = union(iv)
        if length(res_u) != res:
            problems.append(f"graph {g}: node residency intervals overlap (sum {res} != union {length(res_u)})")
        if res > T:
            problems.append(f"graph {g}: node residency {res} > T {T}")
        lo, hi = bisect.bisect_left(issues, t0[g]), bisect.bisect_left(issues, t0[g + 1])
        rs = rd[lo:hi]
        u_all = union([a, b] for a, b, _e, _by in rs)
        u_ex = union([a, b] for a, b, e, _by in rs if e)
        er_hi = intersection_length(u_all, res_u)
        er_lo = intersection_length(u_ex, res_u)
        rows.append({"graph": g, "T": T, "Res": res, "N": T - res, "E_r_lo": er_lo, "E_r_hi": er_hi,
                     "E_nr_lo": res - er_hi, "E_nr_hi": res - er_lo, "M": sum(by for *_x, by in rs),
                     "reads": len(rs), "lat_sum": sum(b - a for a, b, _e, _by in rs), "read_union": length(u_all)})
    return rows, problems


def launch_mechanism(rows: List[Dict[str, Any]], F: int) -> Dict[str, Any]:
    n = len(rows)
    if not n:
        return {"graphs": 0}
    s = {k: sum(r[k] for r in rows) for k in ("T", "Res", "N", "E_r_lo", "E_r_hi", "E_nr_lo", "E_nr_hi", "M",
                                              "lat_sum", "read_union", "reads")}
    return {"graphs": n, **{k: s[k] / n for k in ("T", "Res", "N", "E_r_lo", "E_r_hi", "E_nr_lo", "E_nr_hi", "M")},
            "N_over_T": s["N"] / s["T"], "m": s["M"] / n / F,
            "q_lo": (s["E_r_lo"] / s["M"]) if s["M"] else None, "q_hi": (s["E_r_hi"] / s["M"]) if s["M"] else None,
            "depth": (s["lat_sum"] / s["read_union"]) if s["read_union"] else None,
            "E_nr_share_lo": s["E_nr_lo"] / s["Res"] if s["Res"] else None,
            "E_nr_share_hi": s["E_nr_hi"] / s["Res"] if s["Res"] else None,
            "reads_per_graph": s["reads"] / n}


def counterfactual(terms: Dict[Tuple[str, str], Dict[str, float]], X: str) -> Dict[str, Any]:
    """Descriptive accounting only. terms[(model, point)] = traced cell means with keys N, Res, E_r_lo, E_r_hi, M.
    Returns beta80, beta30, the counterfactual beta80 per bound version, and whether each moves toward beta30."""
    if X not in ("N", "E_r", "E_nr", "M"):
        raise ProtocolError(f"counterfactual term {X!r} is not frozen")
    x1, x2 = x_of("80b", "P1"), x_of("80b", "P2")
    y1, y2 = x_of("30b", "P1"), x_of("30b", "P2")

    def T(e: Dict[str, float]) -> float:
        return e["N"] + e["Res"]

    b80 = beta_from(1 / T(terms[("80b", "P1")]), 1 / T(terms[("80b", "P2")]), x1, x2)
    b30 = beta_from(1 / T(terms[("30b", "P1")]), 1 / T(terms[("30b", "P2")]), y1, y2)
    out: Dict[str, Any] = {"term": X, "beta80": b80, "beta30": b30, "versions": {}}
    for v in ("lo", "hi"):
        def comp(e: Dict[str, float]) -> Dict[str, float]:
            er = e["E_r_lo"] if v == "lo" else e["E_r_hi"]
            return {"N": e["N"], "E_r": er, "E_nr": e["Res"] - er, "M": e["M"]}
        c1, c2 = comp(terms[("80b", "P1")]), comp(terms[("80b", "P2")])
        d1, d2 = comp(terms[("30b", "P1")]), comp(terms[("30b", "P2")])
        if not d1[X] or (X == "M" and not c2["M"]):
            out["versions"][v] = {"beta80_cf": None, "toward": False, "reason": "zero denominator"}
            continue
        new = dict(c2)
        if X == "M":
            mp = c1["M"] * d2["M"] / d1["M"]
            new["E_r"] = c2["E_r"] / c2["M"] * mp
            new["M"] = mp
        else:
            new[X] = c1[X] * d2[X] / d1[X]
        t2 = new["N"] + new["E_r"] + new["E_nr"]
        t1 = c1["N"] + c1["E_r"] + c1["E_nr"]
        if t2 <= 0 or t1 <= 0:
            out["versions"][v] = {"beta80_cf": None, "toward": False, "reason": "non-positive time"}
            continue
        bcf = beta_from(1 / t1, 1 / t2, x1, x2)
        out["versions"][v] = {"beta80_cf": bcf, "toward": abs(bcf - b30) < abs(b80 - b30)}
    out["toward_both"] = all(out["versions"][v]["toward"] for v in ("lo", "hi"))
    return out


# ============================================================ candidates ====

def candidate_a(m: Dict[str, Dict[str, List[float]]], cf_toward_both: Optional[bool], evaluable: bool) -> Dict[str, Any]:
    """m[model][point] = per-launch m values (2 eligible launches each)."""
    if not evaluable:
        return {"status": "NOT EVALUABLE"}
    e: Dict[str, List[float]] = {}
    for mod in MODEL_ORDER:
        a, b = m[mod]["P1"], m[mod]["P2"]
        if any(v <= 0 for v in a + b):
            return {"status": "UNRESOLVED", "reason": "m <= 0 in a launch"}
        x1, x2 = x_of(mod, "P1"), x_of(mod, "P2")
        e[mod] = [(math.log(q) - math.log(p)) / (math.log(x2) - math.log(x1)) for p in a for q in b]
    signs = {_sign(v) for vs in e.values() for v in vs}
    if len(signs) != 1:
        return {"status": "UNRESOLVED", "reason": "e_a values disagree in direction", "e_a": e}
    a80 = [abs(v) for v in e["80b"]]
    a30 = [abs(v) for v in e["30b"]]
    base = {"e_a": e, "abs_envelope": {"80b": (min(a80), max(a80)), "30b": (min(a30), max(a30))},
            "counterfactual_toward_both": cf_toward_both}
    if max(a80) < min(a30):
        if cf_toward_both:
            return {**base, "status": "SUPPORTED",
                    "wording": "consistent with routing concentration contributing to the regime difference"}
        return {**base, "status": "UNRESOLVED",
                "reason": "80B elasticity smaller without overlap, but the M counterfactual does not move beta80 "
                          "toward beta30 at both bounds"}
    if min(a80) >= max(a30):
        return {**base, "status": "FALSIFIED AS STATED"}
    return {**base, "status": "WEAKENED"}


def b_mix(values: Dict[int, float], model: str) -> float:
    """values[slice_bytes] = device MB/s at one readers count. Model-weighted effective bandwidth."""
    w = {s: n * s for s, n in reads_per_token(model).items()}
    return sum(w.values()) / sum(w[s] / values[s] for s in w)


def device_comparison(bmix: Dict[str, Dict[int, Optional[List[float]]]]) -> Dict[str, Any]:
    """bmix[model][N] = the 3 per-round B_mix values (None if any needed diskbench cell is not 3/3 VALID)."""
    per = []
    for n30, n80 in DISKBENCH_PAIRS:
        a, b = bmix["80b"].get(n80), bmix["30b"].get(n30)
        if not a or not b or len(a) != DISKBENCH_ROUNDS or len(b) != DISKBENCH_ROUNDS:
            return {"status": "NOT EVALUABLE", "reason": f"diskbench incomplete for 80B N={n80} / 30B N={n30}"}
        st = "favours" if min(a) > max(b) else "disfavours" if max(a) < min(b) else "indeterminate"
        per.append({"30b_readers": n30, "80b_readers": n80, "80b": a, "30b": b, "result": st})
    res = {p["result"] for p in per}
    overall = "FAVOURS" if res == {"favours"} else "DISFAVOURS" if res == {"disfavours"} else "INDETERMINATE"
    return {"status": overall, "comparisons": per}


def _q_point(q80: List[Tuple[float, float]], q30: List[Tuple[float, float]]) -> str:
    if max(h for _l, h in q80) < min(l for l, _h in q30):
        return "LOWER"
    if min(l for l, _h in q80) > max(h for _l, h in q30):
        return "HIGHER"
    return "INDETERMINATE"


def candidate_b(device: Dict[str, Any], q: Dict[str, Dict[str, List[Tuple[float, float]]]],
                depths: List[Optional[float]], evaluable: bool) -> Dict[str, Any]:
    if not evaluable or device.get("status") == "NOT EVALUABLE":
        return {"status": "NOT EVALUABLE", "device": device.get("status")}
    pts = {p: _q_point(q["80b"][p], q["30b"][p]) for p in PRIMARY_POINTS}
    trace = "LOWER" if set(pts.values()) == {"LOWER"} else "HIGHER" if set(pts.values()) == {"HIGHER"} else "INDETERMINATE"
    dev = device["status"]
    base = {"device": dev, "trace": trace, "trace_points": pts, "depths": depths}
    if any(d is None or d < 2 or d > 16 for d in depths):
        return {**base, "status": "UNRESOLVED", "reason": "a relevant traced in-flight depth lies outside [2, 16]"}
    if dev == "FAVOURS" and trace == "LOWER":
        return {**base, "status": "SUPPORTED"}
    if dev == "DISFAVOURS" and trace == "HIGHER":
        return {**base, "status": "FALSIFIED AS STATED"}
    if (dev == "FAVOURS" and trace == "INDETERMINATE") or (dev == "INDETERMINATE" and trace == "LOWER"):
        return {**base, "status": "WEAKENED"}
    reason = ("device and trace contradict" if {dev, trace} in ({"FAVOURS", "HIGHER"}, {"DISFAVOURS", "LOWER"})
              else "no frozen positive or negative pattern")
    return {**base, "status": "UNRESOLVED", "reason": reason}


def candidate_c(e: Dict[str, Dict[str, List[Dict[str, float]]]], evaluable: bool) -> Dict[str, Any]:
    """e[model][point] = per-launch dicts with E_r_lo, E_nr_lo, E_nr_hi, Res. SUPPORTED is never returned."""
    if not evaluable:
        return {"status": "NOT EVALUABLE"}
    cond = {p: all(l["E_nr_hi"] < l["E_r_lo"] for l in e["80b"][p]) for p in PRIMARY_POINTS}
    share = {p: max(l["E_nr_hi"] / l["Res"] for l in e["80b"][p]) <= min(l["E_nr_lo"] / l["Res"] for l in e["30b"][p])
             for p in PRIMARY_POINTS}
    base = {"falsification_condition": cond, "not_more_non_read_dominated": share,
            "supported_reachable": False}
    if all(cond.values()):
        return {**base, "status": "FALSIFIED AS STATED"}
    if sum(cond.values()) == 1 or all(share.values()):
        return {**base, "status": "WEAKENED"}
    return {**base, "status": "UNRESOLVED"}


def candidate_d(nt: Dict[str, Dict[str, List[float]]], cf_toward_both: Optional[bool], evaluable: bool) -> Dict[str, Any]:
    """nt[model][point] = per-launch N/T (non-residency, compute-inclusive share; never 'pure compute')."""
    if not evaluable:
        return {"status": "NOT EVALUABLE"}
    higher = {p: min(nt["80b"][p]) > max(nt["30b"][p]) for p in PRIMARY_POINTS}
    lower = {p: max(nt["80b"][p]) < min(nt["30b"][p]) for p in PRIMARY_POINTS}
    base = {"quantity": "N/T, non-residency (compute-inclusive) time share", "higher": higher, "lower": lower,
            "counterfactual_toward_both": cf_toward_both}
    if all(higher.values()) and cf_toward_both:
        return {**base, "status": "SUPPORTED"}
    if all(lower.values()):
        return {**base, "status": "FALSIFIED AS STATED"}
    return {**base, "status": "WEAKENED"}


RESOLVED = ("SUPPORTED", "WEAKENED", "FALSIFIED AS STATED")


def lifecycle(primary_outcome: str, gate_status: str, statuses: Dict[str, str]) -> str:
    if primary_outcome != "REPRODUCED" or gate_status != "PASS":
        return "INCONCLUSIVE"
    if any(statuses.get(k) == "SUPPORTED" for k in CANDIDATE_KEYS):
        return "SUPPORTED"
    if all(statuses.get(k) == "FALSIFIED AS STATED" for k in CANDIDATE_KEYS):
        return "FALSIFIED"
    return "INCONCLUSIVE"


def rq009(primary_outcome: str, statuses: Dict[str, str]) -> Dict[str, Any]:
    resolved = [k for k in CANDIDATE_KEYS if statuses.get(k) in RESOLVED]
    remaining = [k for k in CANDIDATE_KEYS if k not in resolved]
    note = RQ009_RULES["not_reproduced_note"] if primary_outcome in ("NOT DISTINGUISHED", "REVERSED") else None
    if primary_outcome == "NOT EVALUABLE":
        return {"status": "PARTIALLY ANSWERED", "kind": "unchanged, data note", "resolved": resolved,
                "remaining": remaining, "note": note}
    if not remaining:
        return {"status": "ANSWERED", "kind": "all four resolved", "resolved": resolved, "remaining": [], "note": note}
    if resolved:
        return {"status": "PARTIALLY ANSWERED", "kind": "narrowed", "resolved": resolved, "remaining": remaining,
                "note": note}
    return {"status": "PARTIALLY ANSWERED", "kind": "unchanged, data note", "resolved": [], "remaining": remaining,
            "note": note}


def not_reproduced_text(primary_outcome: str) -> Optional[str]:
    rel = NOT_REPRODUCED_RELATION.get(primary_outcome)
    return NOT_REPRODUCED_TEMPLATE.format(relation=rel) if rel else None


def routing_metrics(counter: Counter, model: str, q: float = 0.9) -> Dict[str, Any]:
    """counter[(layer, expert)] = decode selections. Three different quantities, never substituted."""
    sys.path.insert(0, str(ROOT))
    from argine.research.metrics import ews_set  # noqa: E402
    s = ews_set(counter, q)
    leb = layer_expert_bytes(model)
    total_experts = MODELS[model]["n_layers"] * MODELS[model]["n_experts"]
    store_bytes = sum(leb.values()) * MODELS[model]["n_experts"]
    ews_bytes = sum(leb[int(l)] for l, _e in s)
    return {"ews90_experts": len(s),
            "historical_ews_fraction": len(s) / total_experts,
            "byte_weighted_variant_ews_bytes_over_store_bytes": ews_bytes / store_bytes,
            "pressure_normalized_ews_bytes_over_F": ews_bytes / MODELS[model]["footprint_bytes"]}


# ============================================================ raw -> data ====

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def diag_launch_data(run_dir: Path, model: str) -> Dict[str, Any]:
    """Bound accounting and routing of one traced launch, from its raw.tsv and markers.jsonl."""
    sys.path.insert(0, str(ROOT))
    from argine.research import rawlog  # noqa: E402
    log = rawlog.parse(run_dir / "raw.tsv", keep_accesses=False)
    requests: List[Tuple[int, int, str]] = []
    start = None
    for mk in rawlog.read_markers(run_dir / "markers.jsonl"):
        if mk.get("event") == "request_start":
            start = (log.system_to_steady(int(mk["system_ns"])), str(mk.get("segment") or ""))
        elif mk.get("event") == "request_end" and start is not None:
            requests.append((start[0], log.system_to_steady(int(mk["system_ns"])), start[1]))
            start = None
    rows, problems = graph_accounting(log.routes, requests, [(n.graph, n.start_ns, n.ready_ns) for n in log.nodes],
                                      [(r.issue_ns, r.complete_ns, r.exact, r.bytes) for r in log.io])
    mech = launch_mechanism(rows, MODELS[model]["footprint_bytes"])
    counter: Counter = Counter()
    for g, ts, layer, n, k, ids in log.routes:
        if n == 1 and any(a <= ts <= b and seg not in ("warmup", "flush-sentinel") for a, b, seg in requests):
            for e in ids:
                counter[(int(layer), int(e))] += 1
    return {"mechanism": mech, "accounting_problems": problems[:20], "n_accounting_problems": len(problems),
            "routing": routing_metrics(counter, model) if counter else None,
            "synchronous_reads": sum(1 for r in log.io if r.synchronous),
            "bound_completions": sum(1 for r in log.io if not r.exact), "reads_total": len(log.io)}


def diag_eligible(rec: Dict[str, Any]) -> bool:
    e = rec["exp026"]
    d = e.get("diagnostics") or {}
    data = e.get("trace_data") or {}
    return (e.get("outcome") == "VALID" and d.get("publication_quality") is True and d.get("trace_complete") is True
            and (d.get("violations") or 0) == 0 and data.get("n_accounting_problems") == 0
            and ((data.get("mechanism") or {}).get("graphs") or 0) > 0)


def analyze(records: Dict[str, List[Dict[str, Any]]], sessions: Dict[str, Dict[str, Any]],
            trace_loader=None) -> Tuple[str, Dict[str, Any]]:
    trace_loader = trace_loader or (lambda r: diag_launch_data(EXP / r["exp026"]["run_dir"], r["exp026"]["model"]))
    summary: Dict[str, Any] = {"experiment": EXP_ID, "protocol_version": PROTOCOL_VERSION,
                               "blocks": {b: (sessions.get(b) or {}).get("status") for b in BLOCK_ORDER}}
    # ---- diskbench
    B: Dict[Tuple[int, int, int], float] = {}
    for r in records.get("diskbench", []):
        e = r["exp026"]
        if e["outcome"] == "VALID":
            B[(e["slice_bytes"], e["readers"], e["round"])] = float(e["mb_s"])
    bmix: Dict[str, Dict[int, Optional[List[float]]]] = {m: {} for m in MODEL_ORDER}
    for m in MODEL_ORDER:
        for n in DISKBENCH_READERS:
            vals = []
            for rnd in range(1, DISKBENCH_ROUNDS + 1):
                if all((s, n, rnd) in B for s in slice_sizes(m)):
                    vals.append(b_mix({s: B[(s, n, rnd)] for s in slice_sizes(m)}, m))
            bmix[m][n] = vals if len(vals) == DISKBENCH_ROUNDS else None
    device = device_comparison(bmix)
    summary["diskbench"] = {"b_mix": bmix, "device": device}
    # ---- performance
    pc: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    for r in records.get("perf", []):
        e = r["exp026"]
        pc.setdefault((e["model"], e["point"], e["threads"]), []).append({"outcome": e["outcome"], "gen_tok_s": e.get("gen_tok_s")})
    cells = {k: perf_cell(v) for k, v in pc.items()}
    primary = explanandum({(m, p): cells.get((m, p, PRIMARY_THREADS), {}) for m in MODEL_ORDER for p in POINTS})
    summary["performance_cells"] = {f"{m}-t{t}-{p}": c for (m, p, t), c in cells.items()}
    summary["explanandum"] = primary
    summary["not_reproduced_text"] = not_reproduced_text(primary["outcome"])
    R = {}
    for p in POINTS:
        c80, c30 = cells.get(("80b", p, PRIMARY_THREADS)) or {}, cells.get(("30b", p, PRIMARY_THREADS)) or {}
        if c80.get("evaluable") and c30.get("evaluable"):
            R[p] = {"R_ms_per_token_80_over_30": c30["center"] / c80["center"], "label": "secondary, descriptive"}
    summary["secondary_R"] = R
    s1, s2 = cells.get(("30b", "P1", SENSITIVITY_THREADS)) or {}, cells.get(("30b", "P2", SENSITIVITY_THREADS)) or {}
    if s1.get("evaluable") and s2.get("evaluable"):
        x1, x2 = x_of("30b", "P1"), x_of("30b", "P2")
        summary["sensitivity"] = {"beta_30b_t12": beta_from(s1["center"], s2["center"], x1, x2),
                                  "envelope_30b_t12": envelope(s1["valid_values"], s2["valid_values"], x1, x2),
                                  "label": "descriptive; not part of the primary decision"}
    # ---- diagnostics
    dl: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in records.get("diag", []):
        e = r["exp026"]
        if e.get("outcome") == "VALID" and e.get("run_dir"):
            try:
                e["trace_data"] = trace_loader(r)
            except Exception as ex:  # noqa: BLE001
                e["trace_data"] = {"error": str(ex)[:300]}
        dl.setdefault((e["model"], e["point"]), []).append(r)
    mech: Dict[Tuple[str, str], Optional[List[Dict[str, Any]]]] = {}
    for m in MODEL_ORDER:
        for p in POINTS:
            rs = dl.get((m, p), [])
            if len(rs) > 2:
                raise ProtocolError(f"{len(rs)} diagnostic launches in {m} {p}; exactly 2 are scheduled")
            ok = len(rs) == 2 and all(diag_eligible(r) for r in rs)
            mech[(m, p)] = [r["exp026"]["trace_data"]["mechanism"] for r in rs] if ok else None
    summary["diagnostic_cells"] = {f"{m}-{p}": {"eligible_2_of_2": mech[(m, p)] is not None, "launches": mech[(m, p)]}
                                   for m in MODEL_ORDER for p in POINTS}
    gate = traced_gate(primary, {k: ([l["T"] for l in v] if v else None) for k, v in mech.items()})
    summary["gate"] = gate
    ok = gate["status"] == "PASS"
    cf = {}
    if ok:
        terms = {(m, p): {k: statistics.mean(l[k] for l in mech[(m, p)]) for k in ("N", "Res", "E_r_lo", "E_r_hi", "M")}
                 for m in MODEL_ORDER for p in PRIMARY_POINTS}
        cf = {X: counterfactual(terms, X) for X in ("M", "N", "E_r", "E_nr")}
    summary["counterfactuals"] = {"label": "DESCRIPTIVE ACCOUNTING ONLY", **cf}

    def per(key: str) -> Dict[str, Dict[str, List[Any]]]:
        return {m: {p: [l[key] for l in mech[(m, p)]] for p in PRIMARY_POINTS} for m in MODEL_ORDER}
    st: Dict[str, Dict[str, Any]] = {}
    st["a"] = candidate_a(per("m") if ok else {}, cf.get("M", {}).get("toward_both"), ok)
    st["b"] = candidate_b(device, {m: {p: [(l["q_lo"], l["q_hi"]) for l in mech[(m, p)]] for p in PRIMARY_POINTS}
                                   for m in MODEL_ORDER} if ok else {},
                          [l["depth"] for m in MODEL_ORDER for p in PRIMARY_POINTS for l in mech[(m, p)]] if ok else [],
                          ok)
    st["c"] = candidate_c({m: {p: mech[(m, p)] for p in PRIMARY_POINTS} for m in MODEL_ORDER} if ok else {}, ok)
    st["d"] = candidate_d(per("N_over_T") if ok else {}, cf.get("N", {}).get("toward_both"), ok)
    statuses = {k: v["status"] for k, v in st.items()}
    summary["candidates"] = st
    summary["lifecycle_status"] = lifecycle(primary["outcome"], gate["status"], statuses)
    summary["rq009"] = rq009(primary["outcome"], statuses)
    summary["routing"] = {"historical_premise_exp007": EXP007_EWS90,
                          "per_launch": {f"{e['exp026']['model']}-{e['exp026']['point']}-L{e['exp026']['launch_index']:02d}":
                                         (e["exp026"].get("trace_data") or {}).get("routing")
                                         for rs in dl.values() for e in rs},
                          "taxonomy": ROUTING_TAXONOMY}
    rows: List[Dict[str, Any]] = []
    for b in BLOCK_ORDER:
        for r in sorted(records.get(b, []), key=lambda x: x["exp026"]["launch_index"]):
            e = r["exp026"]
            row = {k: e.get(k) for k in ("block", "launch_index", "round", "position_in_round", "cell", "model", "point",
                                         "threads", "budget", "budget_bytes", "slice_bytes", "readers", "outcome")}
            row["gen_tok_s"] = e.get("gen_tok_s") if e.get("outcome") == "VALID" else None
            row["mb_s"] = e.get("mb_s") if e.get("outcome") == "VALID" else None
            mm = (e.get("trace_data") or {}).get("mechanism") or {}
            for k in ("T", "N_over_T", "m", "q_lo", "q_hi", "E_r_lo", "E_r_hi", "E_nr_lo", "E_nr_hi", "depth"):
                row[k] = mm.get(k)
            row["diag_eligible"] = diag_eligible(r) if b == "diag" else None
            rows.append(row)
    buf = io.StringIO()
    fields = list(rows[0]) if rows else ["block"]
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue(), summary


# ================================================================== run ====

def previous_blocks_problems(block: str) -> Tuple[List[str], Optional[str]]:
    bad: List[str] = []
    head = None
    for b in BLOCK_ORDER[:BLOCK_ORDER.index(block)]:
        s = RAW / b / "session.json"
        if not s.exists():
            bad.append(f"block {b} has not been run (frozen order {' -> '.join(BLOCK_ORDER)})")
            continue
        d = json.loads(s.read_text(encoding="utf-8"))
        if d.get("status") != "complete" or not d.get("valid"):
            bad.append(f"block {b} is {d.get('status')!r}: later blocks are refused")
        head = head or d.get("checked_head_commit")
    return bad, head


def _model_dry_runs(block: str, base: Dict[str, str]) -> Tuple[List[str], Dict[str, Any]]:
    """`argine benchmark --dry-run` per distinct (model, budget, threads): GGUF headers only, never llama-server."""
    problems: List[str] = []
    out: Dict[str, Any] = {}
    seen = set()
    for x in block_plan(block):
        key = (x["model"], x["budget"], x["threads"])
        if key in seen:
            continue
        seen.add(key)
        r = subprocess.run([sys.executable, "-m", "argine"] + perf_args(x["model"], x["budget"], x["threads"], dry_run=True),
                           cwd=str(ROOT), env=base, capture_output=True, text=True, encoding="utf-8", errors="replace")
        txt = (r.stdout or "") + (r.stderr or "")
        cmdline = next((l.strip() for l in txt.splitlines() if "--threads-batch" in l), "")
        started = "avvio llama-server" in txt
        out["-".join(map(str, key))] = {"exit": r.returncode, "llama_command": cmdline, "server_started": started}
        if r.returncode != 0 or not cmdline or started:
            problems.append(f"dry-run {key} failed or started a server (exit {r.returncode})")
        else:
            problems += [f"dry-run {key}: {p}" for p in X22.check_llama_command(cmdline, x)]
    return problems, out


def preflight(block: str, llama_dir: Path, parent: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    parent = dict(os.environ) if parent is None else parent
    problems = static_problems()
    checks: Dict[str, Any] = {}
    base = None
    try:
        base = X22.base_env(parent, llama_dir)
    except ProtocolError as e:
        problems.append(str(e))
    except Exception as e:  # noqa: BLE001
        problems.append(str(e))
    if "ARGINE_RESEARCH_TRACE" in parent:
        problems.append("ARGINE_RESEARCH_TRACE is set in the parent environment")
    if block == "diskbench":
        if not Path(DISKBENCH_FILE).exists():
            problems.append(f"diskbench file missing: {DISKBENCH_FILE}")
    else:
        bp = X22.binary_path(llama_dir)
        if not bp.exists():
            problems.append(f"research binary missing: {bp}")
        elif X22._sha(bp) != X22.REQUIRED_SERVER_SHA256:  # noqa: SLF001
            problems.append("llama-server SHA-256 differs from the frozen binary")
        for m in MODEL_ORDER:
            for role in ("path", "store"):
                if not Path(MODELS[m][role]).exists():
                    problems.append(f"{m} {role} missing: {MODELS[m][role]}")
        if not any("missing" in p for p in problems):
            for m in MODEL_ORDER:
                try:
                    g = X22.gguf_footprint(m)
                    checks[f"gguf_footprint_{m}"] = g
                    if g["footprint_bytes"] != MODELS[m]["footprint_bytes"] or g["whole_layer_bytes"] != MODELS[m]["whole_layer_bytes_max"]:
                        problems.append(f"{m}: GGUF header geometry differs from the frozen F / whole layer")
                except Exception as e:  # noqa: BLE001
                    problems.append(f"{m}: GGUF header derivation failed: {e}")
    prev, _ = previous_blocks_problems(block)
    problems += prev
    bd = RAW / block
    if bd.exists() and any(bd.iterdir()):
        problems.append(f"{bd} is not empty: a block is never resumed or repeated")
    mp, ms = X22.machine_problems()
    problems += mp
    checks["machine"] = ms
    ok_f, why = freeze_status()
    checks["freeze"] = why
    if not ok_f:
        problems.append("freeze: " + why)
    if block != "diskbench" and base is not None and not any("missing" in p for p in problems):
        dp, checks["dry_run"] = _model_dry_runs(block, base)
        problems += dp
    return {"ok": not problems, "problems": problems, "checks": checks}


def _size(p: Path) -> int:
    return p.stat().st_size if p.exists() else 0


def _read_from(p: Path, offset: int) -> str:
    if not p.exists():
        return ""
    with open(p, "rb") as fh:
        fh.seek(offset)
        return fh.read().decode("utf-8", "replace")


def classify_diskbench(rec: Optional[Dict[str, Any]], n_records: int, item: Dict[str, Any], rc: Optional[int],
                       timed_out: bool) -> Dict[str, Any]:
    """INVALID > NON_COMPLETION > VALID for one diskbench run."""
    if n_records > 1:
        return {"outcome": "INVALID", "reason": f"{n_records} diskbench telemetry records appended"}
    if rec is not None:
        met = rec.get("metrics") or {}
        res = met.get("risultati") or []
        if os.path.normcase(str(met.get("file"))) != os.path.normcase(DISKBENCH_FILE):
            return {"outcome": "INVALID", "reason": f"diskbench file {met.get('file')!r}"}
        if len(res) != 1 or res[0].get("kind") != DISKBENCH_KIND or res[0].get("slice") != item["slice_bytes"] \
                or res[0].get("threads") != item["readers"]:
            return {"outcome": "INVALID", "reason": f"diskbench result does not match the scheduled cell: {res}"[:300]}
    if timed_out:
        return {"outcome": "NON_COMPLETION", "reason": "wall-clock cap exceeded"}
    if rec is None:
        return {"outcome": "NON_COMPLETION", "reason": "no diskbench telemetry record"}
    r0 = rec["metrics"]["risultati"][0]
    if rc != 0 or r0.get("error") or not r0.get("mb_s") or float(r0["mb_s"]) <= 0:
        return {"outcome": "NON_COMPLETION", "reason": f"exit {rc}, error {r0.get('error')!r}, mb_s {r0.get('mb_s')}"}
    return {"outcome": "VALID", "reason": "complete"}


def run_diskbench_block(llama_dir: Path, gate: Dict[str, Any], pf: Dict[str, Any]) -> int:
    plan = block_plan("diskbench")
    bd = RAW / "diskbench"
    home = bd / "argine-home"
    (home / "logs").mkdir(parents=True, exist_ok=False)   # created first: app_dir must never import another history
    runs_log = home / "logs" / "runs.jsonl"
    base = X22.base_env(dict(os.environ), llama_dir)
    env = dict(base)
    env["ARGINE_HOME"] = str(home)
    f = Path(DISKBENCH_FILE)
    session = {"experiment": EXP_ID, "protocol_version": PROTOCOL_VERSION, "block": "diskbench",
               "session_id": datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8],
               "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "freeze": json.loads(FREEZE.read_text(encoding="utf-8")), "schedule": [{**x, "command": command(x)} for x in plan],
               "git_gate": gate, "checked_head_commit": gate["head"], "preflight": pf,
               "telemetry": DISKBENCH_TELEMETRY,
               "provenance": {"file": {"path": str(f), "bytes": f.stat().st_size, "mtime_ns": f.stat().st_mtime_ns},
                              "diskbench_py_sha256": X22._lf_sha(ROOT / "argine" / "diskbench.py")},  # noqa: SLF001
               "status": "started", "valid": None}
    sp, rp = bd / "session.json", bd / "records.jsonl"
    _write_json(sp, session)
    for x in plan:
        logf = bd / "logs" / f"run-{x['launch_index']:02d}.log"
        logf.parent.mkdir(parents=True, exist_ok=True)
        if logf.exists():
            raise ProtocolError(f"diskbench run {x['launch_index']} output already exists")
        off = _size(runs_log)
        print(f"[diskbench {x['launch_index']:>2}/{len(plan)}] round {x['round']}  slice {x['slice_bytes']}  readers {x['readers']}")
        timed_out, rc = False, None
        with open(logf, "w", encoding="utf-8") as fh:
            try:
                rc = subprocess.run(command(x), cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT, text=True,
                                    encoding="utf-8", errors="replace", timeout=x["cap_s"]).returncode
            except subprocess.TimeoutExpired:
                timed_out = True
        appended = [json.loads(l) for l in _read_from(runs_log, off).splitlines() if l.strip().startswith("{")]
        recs = [r for r in appended if r.get("event") == "diskbench"]
        rec = recs[0] if len(recs) == 1 else None
        cls = classify_diskbench(rec, len(recs), x, rc, timed_out)
        r0 = ((rec or {}).get("metrics") or {}).get("risultati") or [{}]
        out = {"telemetry": rec, "exp026": {"experiment": EXP_ID, "session_id": session["session_id"],
               **{k: x[k] for k in ("block", "cell", "launch_index", "round", "position_in_round", "slice_bytes",
                                    "readers", "cap_s")}, "attempt": 1, "exit_status": rc, "timed_out": timed_out, **cls,
               "mb_s": r0[0].get("mb_s"), "phys_mb_s": r0[0].get("phys_mb_s"), "iops": r0[0].get("iops"),
               "latency_ms": r0[0].get("latency_ms"), "command": command(x), "log": str(logf.relative_to(EXP))}}
        _append_jsonl(rp, out)
        print(f"      {cls['outcome']}: {cls['reason'][:160]}")
        if cls["outcome"] == "INVALID":
            session.update({"status": f"STOPPED_INVALID at run {x['launch_index']}", "valid": False,
                            "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
            _write_json(sp, session)
            return 1
    st = f.stat()
    unchanged = st.st_size == session["provenance"]["file"]["bytes"] and st.st_mtime_ns == session["provenance"]["file"]["mtime_ns"]
    session.update({"status": "complete" if unchanged else "INVALID", "valid": unchanged,
                    "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    if not unchanged:
        session["invalid_reason"] = "diskbench file size/mtime changed during the block"
    _write_json(sp, session)
    print("block " + session["status"])
    return 0 if unchanged else 1


def run_model_block(block: str, llama_dir: Path, gate: Dict[str, Any], pf: Dict[str, Any]) -> int:
    plan = block_plan(block)
    base = X22.base_env(dict(os.environ), llama_dir)
    bd = RAW / block
    session = {"experiment": EXP_ID, "protocol_version": PROTOCOL_VERSION, "block": block,
               "session_id": datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8],
               "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "freeze": json.loads(FREEZE.read_text(encoding="utf-8")), "llama_dir": str(llama_dir),
               "schedule": [{**x, "command": command(x)} for x in plan], "git_gate": gate,
               "checked_head_commit": gate["head"], "preflight": pf,
               "provenance": {m: X22.session_provenance(llama_dir, m) for m in MODEL_ORDER},
               "status": "started", "valid": None}
    sp, rp = bd / "session.json", bd / "records.jsonl"
    _write_json(sp, session)
    bsha = session["provenance"]["30b"]["binary"].get("digest")
    runs_log, srv_log = X22._logs()  # noqa: SLF001
    for n, x in enumerate(plan):
        if n:
            time.sleep(COOLDOWN_S)
        harness: List[str] = []
        if X22.orphan_servers():
            harness.append("llama-server process(es) running before the launch")
        logf = bd / "logs" / f"launch-{x['launch_index']:02d}.log"
        logf.parent.mkdir(parents=True, exist_ok=True)
        if logf.exists() or (EXP / x["launch_dir"]).exists():
            raise ProtocolError(f"launch {x['launch_index']} output already exists")
        o_runs, o_srv = _size(runs_log), _size(srv_log)
        ram_before = X22.free_ram()
        t_launch = datetime.now(timezone.utc).isoformat(timespec="seconds")
        print(f"[{block} {x['launch_index']:>2}/{len(plan)}] round {x['round']}  {x['cell']}  {x['budget']}")
        t0 = time.monotonic()
        timed_out, rc = False, None
        if not harness:
            with open(logf, "w", encoding="utf-8") as fh:
                try:
                    rc = subprocess.run(command(x), cwd=str(ROOT), env=base, stdout=fh, stderr=subprocess.STDOUT,
                                        text=True, encoding="utf-8", errors="replace", timeout=x["cap_s"]).returncode
                except subprocess.TimeoutExpired:
                    timed_out = True
        elapsed = time.monotonic() - t0
        time.sleep(SETTLE_S)
        if X22.orphan_servers():
            sys.path.insert(0, str(ROOT))
            from argine import runtime as rtmod  # noqa: E402
            rtmod.kill_orphans([o["pid"] for o in X22.orphan_servers()])
            time.sleep(SETTLE_S)
            if X22.orphan_servers():
                harness.append("llama-server still running after the launch ended (could not be terminated)")
        seg = X22.parse_server_segment(_read_from(srv_log, o_srv))
        prov = X22.check_segment_provenance(seg, x) if not harness else []
        inv: List[str] = []
        extra: Dict[str, Any] = {}
        reads_failed = None
        if block == "perf":
            appended = [json.loads(l) for l in _read_from(runs_log, o_runs).splitlines() if l.strip().startswith("{")]
            bench = [r for r in appended if r.get("event") == "benchmark"]
            if len(bench) > 1:
                harness.append(f"{len(bench)} benchmark records appended")
            rec = bench[0] if len(bench) == 1 else None
            if rec is not None:
                prov += X22.perf_record_problems(rec, x, (bsha or "")[:8] or None)
                inv += X22.perf_invariant_problems(rec, x)
                extra["benchmark_record"] = rec
                run0 = ((rec.get("metrics") or {}).get("runs") or [{}])[0]
                extra["gen_tok_s"] = run0.get("gen_tok_s")
            done, why = X22.perf_measured(rec)
        else:
            man, summ, resp, rd = _diag_artifacts(x)
            if man is not None:
                prov += X22.diag_manifest_problems(man, summ, x, bsha)
                inv += X22.diag_invariant_problems(man)
                extra["diagnostics"] = X22.diag_diagnostics(man, summ, resp)
                reads_failed = ((man.get("final_store_stats") or {}).get("trace_totals") or {}).get("reads_failed")
                extra["run_dir"] = str(rd.relative_to(EXP)) if rd else None
                if rd is not None and (rd / "trace.jsonl.gz").exists():
                    try:
                        extra["routing_steps"], extra["routing_sha256"] = X22.routing_sha(rd / "trace.jsonl.gz")
                    except Exception as e:  # noqa: BLE001
                        extra["routing_error"] = str(e)[:300]
            done, why = X22.diag_measured(man, resp)
        cls = X22.classify(prov, inv, harness, seg["abort_lines"], done, rc, timed_out, why)
        if cls["outcome"] == "STORE_RESIDENCY_ABORT":
            cls["abort_cause"] = X22.abort_cause(seg["abort_lines"], seg["coverage"], x["budget_bytes"], x["model"],
                                                 reads_failed, layer_expert_bytes(x["model"]))
        rec_out = {"exp026": {"experiment": EXP_ID, "session_id": session["session_id"],
                              **{k: x[k] for k in ("block", "cell", "model", "point", "role", "threads", "launch_index",
                                                   "round", "position_in_round", "budget", "budget_bytes",
                                                   "budget_over_F", "cap_s", "launch_dir")},
                              "attempt": 1, "launch_utc": t_launch, "free_ram_before_launch": ram_before,
                              "exit_status": rc, "timed_out": timed_out, "elapsed_s": elapsed, **cls,
                              "abort_lines": seg["abort_lines"][:3], "min_budget_warning": seg["min_budget_warning"],
                              "provenance_problems": prov, "invariant_problems": inv, "harness_problems": harness,
                              **{k: v for k, v in extra.items() if k != "benchmark_record"},
                              "command": command(x), "launch_log": str(logf.relative_to(EXP))}}
        if "benchmark_record" in extra:
            rec_out["benchmark_record"] = extra["benchmark_record"]
        _append_jsonl(rp, rec_out)
        print(f"      {cls['outcome']}: {cls['reason'][:160]}")
        if cls["outcome"] == "INVALID":
            session.update({"status": f"STOPPED_INVALID at launch {x['launch_index']}", "valid": False,
                            "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
            _write_json(sp, session)
            print("INVALID launch: block stopped (no retry, no resume)", file=sys.stderr)
            return 1
    session["status"] = "launches complete"
    _write_json(sp, session)
    return finalize(session, sp, llama_dir)


def _diag_artifacts(item: Dict[str, Any]):
    d = EXP / item["launch_dir"]
    runs = sorted(d.glob("run-*")) if d.exists() else []
    if len(runs) != 1:
        return None, None, [], None
    rd = runs[0]
    man = json.loads((rd / "manifest.json").read_text(encoding="utf-8")) if (rd / "manifest.json").exists() else None
    summ = json.loads((rd / "summary.json").read_text(encoding="utf-8")) if (rd / "summary.json").exists() else None
    return man, summ, _read_jsonl(rd / "responses.jsonl"), rd


def finalize(session: Dict[str, Any], sp: Path, llama_dir: Path) -> int:
    problems: List[str] = []
    for m in MODEL_ORDER:
        problems += [f"{m}: {p}" for p in X22.finalize_fingerprints({"provenance": session["provenance"][m]}, llama_dir)]
    session["status"] = "complete" if not problems else "INVALID"
    session["valid"] = not problems
    session["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if problems:
        session["invalid_reason"] = "; ".join(problems)
    _write_json(sp, session)
    print("block " + session["status"] + ("" if not problems else ": " + "; ".join(problems)))
    return 0 if not problems else 1


def run_block(block: str, llama_dir: Path) -> int:
    prev, expected_head = previous_blocks_problems(block)
    if prev:
        print("PROTOCOL ERROR: " + "; ".join(prev), file=sys.stderr)
        return 2
    try:
        gate = require_run_gate(expected_head)
        pf = preflight(block, llama_dir)
        if not pf["ok"]:
            print("preflight failed; nothing was run:\n  - " + "\n  - ".join(pf["problems"]), file=sys.stderr)
            return 2
        gate2 = require_run_gate(expected_head)
        if gate2["head"] != gate["head"]:
            raise ProtocolError("HEAD changed during preflight")
    except ProtocolError as e:
        print(f"PROTOCOL ERROR: {e}", file=sys.stderr)
        return 2
    if block == "diskbench":
        return run_diskbench_block(llama_dir, {"before_preflight": gate, "before_session": gate2, "head": gate["head"]}, pf)
    return run_model_block(block, llama_dir, {"before_preflight": gate, "before_session": gate2, "head": gate["head"]}, pf)


def cmd_finalize(block: str, llama_dir: Path) -> int:
    sp = RAW / block / "session.json"
    if block == "diskbench" or not sp.exists():
        print("nothing to finalise for this block", file=sys.stderr)
        return 1
    s = json.loads(sp.read_text(encoding="utf-8"))
    if s.get("status") != "launches complete":
        print(f"block status {s.get('status')!r}: nothing to finalise", file=sys.stderr)
        return 1
    return finalize(s, sp, llama_dir)


def cmd_analyze() -> int:
    sessions, records = {}, {}
    for b in BLOCK_ORDER:
        sp = RAW / b / "session.json"
        if sp.exists():
            sessions[b] = json.loads(sp.read_text(encoding="utf-8"))
            records[b] = _read_jsonl(RAW / b / "records.jsonl")
    if not any(records.values()):
        print("no records: EXP-026 has not been run; nothing written", file=sys.stderr)
        return 1
    text, summary = analyze(records, sessions)
    (EXP / "results.csv").write_text(text, encoding="utf-8", newline="")
    _write_json(PROCESSED / "summary.json", summary)
    print(text)
    print(f"explanandum {summary['explanandum']['outcome']}; gate {summary['gate']['status']}; candidates "
          + ", ".join(f"{k}={v['status']}" for k, v in summary["candidates"].items())
          + f"; lifecycle {summary['lifecycle_status']}; RQ-009 {summary['rq009']['status']}")
    return 0


# ================================================================== main ====

def print_dry_run() -> int:
    plan = build_plan()
    print(f"{EXP_ID} {PROTOCOL_VERSION}: {EXPECTED['model_launches']} scheduled model launches "
          f"({EXPECTED['perf_primary']} primary + {EXPECTED['perf_sensitivity']} sensitivity + {EXPECTED['diag']} "
          f"diagnostic) and {EXPECTED['diskbench_runs']} diskbench runs; blocks {' -> '.join(BLOCK_ORDER)}")
    for m in MODEL_ORDER:
        print(f"{m}: F = {MODELS[m]['footprint_bytes']} B; slices {slice_sizes(m)}; reads/token {reads_per_token(m)}")
        for p, t in budget_table(m).items():
            print(f"    {p} {t['label']:>6} = {t['bytes']:>10} B  {t['budget_over_F']:.4f} F  {t['shards']} shards  {t['role']}")
    for x in plan:
        tag = (f"s{x['slice_bytes']} n{x['readers']}" if x["block"] == "diskbench" else
               f"{x['cell']:<18} {x['budget']:>6}")
        print(f"  {x['block']:<9} {x['launch_index']:>2}  r{x['round']} p{x['position_in_round']:>2}  {tag}  cap {x['cap_s']} s  "
              + " ".join(x["argv"]))
    p = check_plan(plan)
    print("plan check: " + ("OK" if not p else "; ".join(p)))
    print(f"freeze: {freeze_status()[1]}")
    print("dry run: no subprocess started, no model, store or device read; preflight NOT performed")
    return 0 if not p else 1


def cmd_check() -> int:
    problems = static_problems()
    ok, why = freeze_status()
    plan = build_plan()
    print(f"plan: " + ", ".join(f"{b} {sum(1 for x in plan if x['block'] == b)}" for b in BLOCK_ORDER)
          + f"; model launches {EXPECTED['model_launches']}; diskbench runs {EXPECTED['diskbench_runs']}")
    for m in MODEL_ORDER:
        print(f"{m}: F = {footprint(m)} B; budget/F " + ", ".join(f"{p} {x_of(m, p):.4f}" for p in POINTS))
    print(f"freeze: {why}")
    if problems:
        print("check: FAILED\n  - " + "\n  - ".join(problems))
        return 2
    if not ok:
        print("check: FAILED (freeze)")
        return 2
    print("check: OK (nothing written; no model, store, binary or device read)")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    for f in ("--check", "--dry-run", "--freeze", "--preflight", "--run", "--finalize", "--analyze"):
        g.add_argument(f, action="store_true")
    ap.add_argument("--block", choices=BLOCK_ORDER)
    ap.add_argument("--llama-dir", default=str(DEFAULT_LLAMA_DIR))
    a = ap.parse_args(argv)
    try:
        if a.check:
            return cmd_check()
        if a.dry_run:
            return print_dry_run()
        if a.freeze:
            return cmd_freeze()
        if a.analyze:
            return cmd_analyze()
        if not a.block:
            ap.error("--preflight, --run and --finalize need --block")
        if a.preflight:
            r = preflight(a.block, Path(a.llama_dir))
            print(json.dumps(r["checks"], indent=2, ensure_ascii=False, default=str))
            print("preflight: " + ("OK" if r["ok"] else "FAIL\n  - " + "\n  - ".join(r["problems"])))
            return 0 if r["ok"] else 1
        if a.run:
            return run_block(a.block, Path(a.llama_dir))
        return cmd_finalize(a.block, Path(a.llama_dir))
    except ProtocolError as e:
        print(f"PROTOCOL ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
