"""EXP-022 — extreme memory pressure (protocol, orchestration, classification, analysis).

    python tools/exp022_run.py --check                 # repository-only validation: protocol constants,
                                                       #   exact budget bytes, footprints re-derived from
                                                       #   tracked EXP-007 evidence, shard rule and abort
                                                       #   signature in the BUILD-INFO-certified patch
                                                       #   sources, timeouts, schedule, freeze. Only `git`
                                                       #   is ever started; no model file is opened.
    python tools/exp022_run.py --dry-run               # print the frozen plan and every command; no subprocess
    python tools/exp022_run.py --freeze                # write protocol_freeze.json (prospective freeze)
    python tools/exp022_run.py --preflight --block B   # machine/runtime checks + `argine benchmark --dry-run`
                                                       #   per configuration (reads GGUF headers only; never
                                                       #   starts llama-server, never runs inference)
    python tools/exp022_run.py --run --block B         # the launches of one block (writes raw/<B>/); NOT resumable
    python tools/exp022_run.py --finalize --block B    # post-block fingerprint finalisation only
    python tools/exp022_run.py --analyze               # raw/ -> results.csv + processed/summary.json

Blocks, in this frozen order: 30b-perf, 30b-diag, 80b-perf, 80b-diag.

What the experiment separates
-----------------------------
A. H-5 mechanism test (30B, primary). The historical 2026-10-03 prediction is kept verbatim. Its pre-data
   falsifier ">30% expert-level hits at 768 MiB" is operationalised with the EXP-020 metric
   (workload gate-slice `expert_level_hit_rate`) from the TRACED diagnostic block. Above 30% -> FALSIFIED.
   Otherwise the prediction is "not falsified by this test" and the formal status is INCONCLUSIVE: no
   positive support threshold existed before the data, and none is introduced.
B. RQ-008 usability envelope (both models; the 80B arm is exploratory). Measured UNTRACED throughput and
   completion, labelled with frozen reporting conventions. It never sets the H-5 verdict.

Replication unit: one complete launch (fresh process, empty Expert Store). No retries.
Outcomes per launch: VALID, STORE_RESIDENCY_ABORT, NON_COMPLETION, INVALID (see classify()).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import inspect
import io
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
EXP_ID = "EXP-022-extreme-pressure"
EXP = ROOT / "experiments" / EXP_ID
RAW = EXP / "raw"
FIDELITY = EXP / "fidelity"
README = EXP / "README.md"
CONFIG = EXP / "config.json"
FREEZE = EXP / "protocol_freeze.json"
TESTS = ROOT / "tests" / "test_exp022.py"
MARK_BEGIN = "<!-- EXP-022 PROTOCOL-BEGIN -->"
MARK_END = "<!-- EXP-022 PROTOCOL-END -->"
PROTOCOL_VERSION = "EXP022-P1"

MIB = 1 << 20
GIB = 1 << 30

# ------------------------------------------------------------ hypothesis ----

HISTORICAL_HYPOTHESIS = (
    "[HYPOTHESIS] Prediction written before measuring (from the EXP-009 pilot): on Qwen3-30B-A3B, below "
    "one token's expert footprint (~1.02 GiB, 384 experts) the runtime's sharded LRU collapses (pilot "
    "replay: 0-20% expert hits at 1 GiB, ~0-2% at 768 MiB), while frequency-based retention would keep "
    "~30-45%. If the runtime keeps more than 30% expert-level hits at 768 MiB, the prediction is falsified.")

H5 = {
    "model": "30b",
    "budget": "768M",
    "budget_bytes": 805306368,
    "phase": "diag",
    "threshold": 0.30,
    "comparison": "strictly greater than",
    "metric_field": "manifest.final_store_stats.expert_level_hit_rate",
    "metric_definition": (
        "EXP-020 workload gate-slice hit rate: slice hits over slice accesses made in the gate (first) "
        "MUL_MAT_ID node of each layer, prefill + decode, workload requests only (warm-up and flush "
        "sentinel excluded); computed by argine.research.rawlog.node_attributed_outcomes from the research "
        "trace of the launch. It is the runtime field `expert_level_hit_rate`, not the expert-event "
        "decode-only counterfactual metric of the EXP-009 pilot."),
    "scheduled_launches": 2,
    "aggregation": "arithmetic mean of the two scheduled 30B diag launches at 768 MiB, both of which must be "
                   "H-5-eligible; individual launches and dispersion reported; no launch-level vote",
    "eligibility": "launch outcome VALID, manifest publication_quality true, trace complete over the "
                   "workload, budget_violations == 0",
    "minimum_eligible_launches": 2,
    "falsified_if": "both scheduled launches eligible and their mean > 0.30",
    "otherwise": "both eligible and mean <= 0.30: NOT FALSIFIED BY THIS TEST; formal status INCONCLUSIVE",
    "not_evaluable": "fewer than 2 eligible launches for any reason (or the 30B diag block incomplete): H-5 "
                     "NOT EVALUABLE, formal status INCONCLUSIVE, reported with the outcomes observed at 768 MiB; "
                     "no replacement launch and no retry",
    "positive_support_threshold": None,
    "positive_support_note": (
        "No positive support threshold is introduced. The 2026-10-03 prediction recorded only a "
        "falsification condition; any support bound chosen now (e.g. <=10%, <=2%, a fraction of the 1.5G "
        "anchor) would be selected after the pilot values were known. Agreement with the predicted range "
        "may be reported descriptively only."),
}

# ----------------------------------------------------------------- models ----

MODELS: Dict[str, Dict[str, Any]] = {
    "30b": {
        "name": "Qwen3-30B-A3B Q4_K_M",
        "path": r"C:\modelli\Qwen_Qwen3-30B-A3B-Q4_K_M.gguf",
        "store": r"C:\modelli\Qwen_Qwen3-30B-A3B-Q4_K_M.psxs",
        "role": "primary (H-5 mechanism test + usability envelope)",
        "gen_threads": 12,
        "n_layers": 48, "n_experts": 128, "top_k": 8,
        "footprint_bytes": 1097072640,          # = 1046.2 MiB = 1.0217 GiB
        "whole_layer_bytes": 391643136,         # = 373.5 MiB (largest layer x all experts)
        "footprint_source": "experiments/EXP-007-router-characterization/analysis-v3/"
                            "30b-natural-language-dev-run-00-4G/analysis_manifest.json",
    },
    "80b": {
        "name": "Qwen3-Next-80B-A3B-Instruct Q3_K_M",
        "path": r"C:\modelli\Qwen3-Next-80B-A3B-Instruct-Q3_K_M.gguf",
        "store": r"C:\modelli\Qwen3-Next-80B-A3B-Instruct-Q3_K_M.psxs",
        "role": "exploratory (usability envelope + descriptive diagnostics; no frozen prediction)",
        "gen_threads": 4,
        "n_layers": 48, "n_experts": 512, "top_k": 10,
        "footprint_bytes": 719585280,           # = 686.2 MiB = 0.6702 GiB
        "whole_layer_bytes": 830472192,         # = 792.0 MiB
        "footprint_source": "experiments/EXP-007-router-characterization/analysis-v3/"
                            "80b-natural-language-dev-run-00-4G/analysis_manifest.json",
    },
}
MODEL_ORDER = ("30b", "80b")
FOOTPRINT_DEFINITION = (
    "one decode token's routed-expert footprint as the store accounts it: sum over the 48 MoE layers of "
    "top_k x the bytes the store needs for one expert of that layer (gate + up + down slices, each rounded "
    "up to 4096 B: argine.research.runner.expert_bytes_from_model). The shared expert of the 80B is not in "
    "the store. Derived from existing metadata, not measured by EXP-022.")

# exact bytes; the label is what the CLI receives (argine.util.parse_size: M = MiB, G = GiB)
BUDGETS: Dict[str, Tuple[Tuple[str, int, str], ...]] = {
    "30b": (("1.5G", 1610612736, "in-session anchor"),
            ("1152M", 1207959552, "above the one-token footprint"),
            ("1024M", 1073741824, "approximately at the footprint"),
            ("768M", 805306368, "H-5 primary falsification point"),
            ("512M", 536870912, "far below the footprint")),
    "80b": (("1.5G", 1610612736, "in-session anchor"),
            ("768M", 805306368, "above the one-token footprint"),
            ("672M", 704643072, "approximately at the footprint"),
            ("512M", 536870912, "below the footprint")),
}
EXCLUDED_BUDGETS = {
    "80b-336M": {"bytes": 352321536,
                 "reason": "below 512 MiB the store uses floor(budget / 32 MiB) < 16 shards (10 at 336 MiB): "
                           "a second structural variable. Not part of EXP022-P1. If ever run, it is a "
                           "separately labelled exploratory feasibility probe, never pooled with or compared "
                           "directly against the 16-shard arm."},
}

# store shard rule (patches/argine-store.cpp, psx_cache_new_n): n = min(16, floor(budget / 32 MiB)), >= 1
SHARD_MIN_BYTES = 32 * MIB
MAX_SHARDS = 16
SHARD_RULE_SOURCE_SNIPPETS = ("const uint64_t minimo_per_settore = 32ull << 20;",
                              "if (n > 16) {", "n = (size_t) (budget_bytes / minimo_per_settore);")

# ---------------------------------------------------------------- runtime ----

DEFAULT_LLAMA_DIR = ROOT / "llama-research-v2"
INSTRUMENTATION_ID = "argine2-research-v2"
REQUIRED_SERVER_SHA256 = "a0063720ec4bb5d1ba3e4f7c0387dd1bc6a239b0ce038ba3d755351dab3f5cf3"
REQUIRED_BUILD_INFO_SHA256 = "cc76046a98ce03b445e83d5948bcae22b7c2e46376e9f3cc414505b2bde97d04"
REQUIRED_LLAMA_CPP_COMMIT = "254b177307b14d055bbe4c050c9c441a1e06d8a4"
BATCH_THREADS = 12            # planner: max(threads, min(logical, physical)) = 12 on the i5-1340P; verified
CTX = 4096
BATCH = 2048
UBATCH = 512
KV_TYPE = "f16"
READ_MODE = "asincrono"
BACKEND = "cpu"
N_PREDICT = 128
REPEAT = 1
RETRIES = 0
COOLDOWN_S = 60
SETTLE_S = 5                  # after a launch ends: wait before the orphan check
MIN_START_AVAILABLE = 8 * GIB  # free RAM required to START a block; per launch a warning only
BROWSER_PROCESSES = ("chrome.exe", "msedge.exe", "firefox.exe", "brave.exe", "opera.exe")

# completion caps: built from existing argine timeout conventions, fixed before any sub-1.5G data
READY_TIMEOUT_S = 600         # argine.server.LlamaServerProcess.wait_ready(timeout=600.0)
REQUEST_TIMEOUT_S = 1800      # argine.bench._post / _post_stream_ttft(timeout=1800.0)
MEASURED_REQUESTS = {"perf": 1, "diag": 3}
NORMAL_DURATION_REFERENCE = {
    "perf": "EXP-024 30B 1.5G untraced launches: process monitor duration 46.2-49.0 s (raw/records.jsonl)",
    "diag": "EXP-007 natural-language EVAL 1.5G traced runs: workload wall 108-109 s (30B), 89-94 s (80B)",
}

# Store-residency abort (patches/argine-experts.cpp, GGML_ABORT): the runtime could not make a node's slices
# resident. The same path is taken for pinned-capacity exhaustion, commit failure and read failure, so the
# signature alone is causally NEUTRAL (amendment 2, 2026-10-14, before any EXP-022 measurement).
RESIDENCY_ABORT_SIGNATURE = "argine: impossibile rendere residenti gli expert del layer"
RESIDENCY_ABORT_SOURCE_SNIPPETS = (
    "argine: impossibile rendere residenti gli expert del layer %u (%s). ",
    " MiB su un budget di %",
    "ne chiede %u su %u. Alza ARGINE_EXPERT_BUDGET",
    "id.layer, src0->name, st.bytes_resident >> 20, st.budget >> 20, n_serve,",
)
RESIDENCY_ABORT_RE = re.compile(
    r"impossibile rendere residenti gli expert del layer (\d+) \(([^)]*)\)\. Residenti (\d+) MiB su un "
    r"budget di (\d+) MiB, questo nodo ne chiede (\d+) su (\d+)")
COVERAGE_RE = re.compile(r"copertura:\s*(\d+) tensori di expert su (\d+)")
ABORT_CAUSES = ("capacity_or_pinning", "commit_or_read_failure", "cause_unresolved")
ABORT_CAUSE_RULE = {
    "capacity_or_pinning": "the parsed abort line names layer L and n requested experts, the launch's log shows "
                           "full store coverage ('copertura: N tensori di expert su N', so all three slice kinds "
                           "of the node are pinned together), and n x expert_bytes[L] (all three kinds, aligned; "
                           "tracked EXP-007 trace_meta) > the budget: the node's pinned set cannot fit in the "
                           "whole budget, whatever else happened",
    "commit_or_read_failure": "capacity rule not met, and the launch's research trace records at least one "
                              "failed read (reads_failed > 0)",
    "cause_unresolved": "every other case (always for untraced launches unless the capacity rule holds); no "
                        "runtime or binary change is made to obtain an attribution",
}
MIN_BUDGET_WARNING = "sotto il minimo teorico"      # startup warning only (budget < whole layer)
STORE_OPEN = "argine: expert store aperto:"

# ---------------------------------------------------------------- workloads ----

PERF_PROMPT_SHA256 = "b1677835fc599087f01f08bf9ab3a07f088351e443740209de52be40778095f3"
PERF_PROMPT_CHARS = 304
DIAG_WORKLOAD = {"name": "natural-language", "version": "v1", "split": "eval",
                 "file": "workloads/v1/natural-language.json",
                 "sha256": "71b170bc0fde211e3694fede63192c72ea70ef7f561791ac28b82ebac25105cc",
                 "requests": 3, "n_predict": 128,
                 "split_rationale": "repository rule 'tune on dev, claim on eval' (workloads/README.md); "
                                    "nothing in EXP-022 is tuned on traces, and EXP-007 already holds "
                                    "natural-language EVAL runs of both models at >= 1.5 GiB"}

# ------------------------------------------------------------- schedule ----

BLOCK_ORDER = ("30b-perf", "30b-diag", "80b-perf", "80b-diag")
PERF_ROUNDS: Dict[str, Tuple[Tuple[str, ...], ...]] = {
    # 3 x 5 Latin rectangle: every budget once per round, three different positions, no ordered
    # neighbour pair repeated, 6 decreasing and 6 increasing transitions
    "30b": (("1.5G", "1152M", "1024M", "768M", "512M"),
            ("1152M", "512M", "768M", "1024M", "1.5G"),
            ("512M", "1024M", "1152M", "1.5G", "768M")),
    # first three rows of the 4 x 4 Williams square A B D C / B C A D / C D B A
    "80b": (("1.5G", "768M", "512M", "672M"),
            ("768M", "672M", "1.5G", "512M"),
            ("672M", "512M", "768M", "1.5G")),
}
DIAG_ROUNDS: Dict[str, Tuple[Tuple[str, ...], ...]] = {m: PERF_ROUNDS[m][:2] for m in MODEL_ORDER}

OUTCOMES = ("VALID", "STORE_RESIDENCY_ABORT", "NON_COMPLETION", "INVALID")
AMENDMENTS = (
    {"id": "A1", "date": "2026-10-14", "before_any_measurement": True,
     "change": "H-5 needs BOTH scheduled 30B / 768 MiB diag launches eligible (minimum_eligible_launches 1 -> 2); "
               "fewer than 2 eligible -> NOT EVALUABLE, formal status INCONCLUSIVE"},
    {"id": "A2", "date": "2026-10-14", "before_any_measurement": True,
     "change": "per-launch outcome INFEASIBLE renamed STORE_RESIDENCY_ABORT (causally neutral); cause recorded "
               "as capacity_or_pinning / commit_or_read_failure / cause_unresolved by a frozen evidence rule"},
)
USABILITY = (("INTERACTIVE", 5.0), ("DEGRADED", 1.0), ("NON_INTERACTIVE", 0.0))
ALLOWED_ARGINE_ENV = {"ARGINE_LLAMA_DIR", "ARGINE_HOME"}
VALIDATIONS_REGISTRY = "llama-research-v2/VALIDATIONS.jsonl"


class ProtocolError(RuntimeError):
    """A configuration, provenance or data condition that the frozen protocol forbids."""


# =============================================================== derived ====

def budget_table(model: str) -> Dict[str, Dict[str, Any]]:
    out = {}
    for label, b, role in BUDGETS[model]:
        f = MODELS[model]["footprint_bytes"]
        out[label] = {"bytes": b, "mib": b / MIB, "gib": b / GIB, "role": role,
                      "fraction_of_footprint": b / f, "shards": shard_count(b),
                      "per_shard_bytes": b // shard_count(b),
                      "below_whole_layer": b < MODELS[model]["whole_layer_bytes"]}
    return out


def budget_bytes(model: str, label: str) -> int:
    for lab, b, _ in BUDGETS[model]:
        if lab == label:
            return b
    raise ProtocolError(f"{model}: budget {label!r} is not in the frozen protocol")


def shard_count(budget: int) -> int:
    n = budget // SHARD_MIN_BYTES
    return int(max(1, min(MAX_SHARDS, n)))


def cap_seconds(phase: str) -> int:
    return READY_TIMEOUT_S + REQUEST_TIMEOUT_S * MEASURED_REQUESTS[phase]


def usability_category(gen_tok_s: Optional[float]) -> Optional[str]:
    """Frozen REPORTING convention (RQ-008 roadmap proposal), never a hypothesis threshold."""
    if gen_tok_s is None:
        return None
    for name, lo in USABILITY:
        if gen_tok_s >= lo:
            return name
    return "NON_INTERACTIVE"


def footprint_from_meta(meta: Dict[str, Any]) -> Dict[str, int]:
    eb = {int(k): int(v) for k, v in (meta.get("expert_bytes") or {}).items()}
    if not eb:
        raise ProtocolError("trace metadata without expert_bytes")
    k, n = int(meta["top_k"]), int(meta["n_experts"])
    return {"layers": len(eb), "top_k": k, "n_experts": n,
            "footprint_bytes": sum(eb.values()) * k, "whole_layer_bytes": max(eb.values()) * n}


def derive_repo_footprints(root: Path = ROOT) -> Dict[str, Dict[str, Any]]:
    """Re-derive F and the whole-layer size from the TRACKED EXP-007 analysis manifests."""
    out = {}
    for m in MODEL_ORDER:
        p = root / MODELS[m]["footprint_source"]
        d = json.loads(p.read_text(encoding="utf-8"))
        out[m] = {**footprint_from_meta(d["trace_meta"]), "source": MODELS[m]["footprint_source"]}
    return out


# ============================================================== schedule ====

def perf_args(model: str, label: str, dry_run: bool = False) -> List[str]:
    a = ["benchmark", MODELS[model]["path"],
         "--backend", BACKEND, "--expert-store", "--expert-budget", label,
         "--read-mode", READ_MODE, "--threads", str(MODELS[model]["gen_threads"]),
         "--ctx", str(CTX), "--batch", str(BATCH), "--ubatch", str(UBATCH),
         "--repeat", str(REPEAT), "--n-predict", str(N_PREDICT), "--retries", str(RETRIES),
         "--expert-stats"]
    if dry_run:
        a.append("--dry-run")
    return a


def diag_args(model: str, label: str, out_dir: str) -> List[str]:
    return ["research", "trace", MODELS[model]["path"],
            "--backend", BACKEND, "--expert-store", "--budgets", label, "--repeat", str(REPEAT),
            "--read-mode", READ_MODE, "--threads", str(MODELS[model]["gen_threads"]),
            "--ctx", str(CTX), "--batch", str(BATCH), "--ubatch", str(UBATCH),
            "--retries", str(RETRIES), "--expert-stats",
            "--workload", DIAG_WORKLOAD["name"], "--workload-version", DIAG_WORKLOAD["version"],
            "--split", DIAG_WORKLOAD["split"], "--cooldown", "0", "--out", out_dir]


def launch_dir_rel(block: str, i: int) -> str:
    return f"raw/{block}/launch-{i:02d}"


def build_plan() -> List[Dict[str, Any]]:
    plan: List[Dict[str, Any]] = []
    for block in BLOCK_ORDER:
        model, phase = block.split("-")
        rounds = PERF_ROUNDS[model] if phase == "perf" else DIAG_ROUNDS[model]
        i = 0
        for rnd, order in enumerate(rounds, 1):
            for pos, label in enumerate(order, 1):
                i += 1
                b = budget_bytes(model, label)
                item = {"block": block, "model": model, "phase": phase, "launch_index": i,
                        "global_index": len(plan) + 1, "round": rnd, "position_in_round": pos,
                        "budget": label, "budget_bytes": b,
                        "fraction_of_footprint": b / MODELS[model]["footprint_bytes"],
                        "threads": MODELS[model]["gen_threads"], "batch_threads": BATCH_THREADS,
                        "cap_s": cap_seconds(phase), "launch_dir": launch_dir_rel(block, i)}
                item["argv"] = (perf_args(model, label) if phase == "perf"
                                else diag_args(model, label, str(Path("experiments") / EXP_ID / item["launch_dir"])))
                plan.append(item)
    return plan


def block_plan(block: str) -> List[Dict[str, Any]]:
    if block not in BLOCK_ORDER:
        raise ProtocolError(f"unknown block {block!r}; frozen blocks: {', '.join(BLOCK_ORDER)}")
    return [x for x in build_plan() if x["block"] == block]


def command(item: Dict[str, Any]) -> List[str]:
    return [sys.executable, "-m", "argine"] + list(item["argv"])


def fidelity_command(item: Dict[str, Any]) -> List[str]:
    d = Path("experiments") / EXP_ID
    return ["python", "tools/validate_replay_fidelity.py", str(d / item["launch_dir"]),
            str(d / "fidelity" / item["block"] / f"launch-{item['launch_index']:02d}")]


def _flag(argv: Sequence[str], flag: str) -> Optional[str]:
    return argv[list(argv).index(flag) + 1] if flag in argv else None


def check_plan(plan: List[Dict[str, Any]]) -> List[str]:
    bad: List[str] = []
    if [x["block"] for x in plan] != sorted((x["block"] for x in plan), key=BLOCK_ORDER.index):
        bad.append("blocks are not in the frozen order")
    for block in BLOCK_ORDER:
        model, phase = block.split("-")
        rows = [x for x in plan if x["block"] == block]
        rounds = PERF_ROUNDS[model] if phase == "perf" else DIAG_ROUNDS[model]
        labels = [lab for lab, _, _ in BUDGETS[model]]
        if len(rows) != len(rounds) * len(labels):
            bad.append(f"{block}: {len(rows)} launches, expected {len(rounds) * len(labels)}")
        if [x["launch_index"] for x in rows] != list(range(1, len(rows) + 1)):
            bad.append(f"{block}: launch indices are not 1..N")
        for rnd, order in enumerate(rounds, 1):
            if sorted(order, key=labels.index) != labels:
                bad.append(f"{block} round {rnd}: not every budget exactly once")
        for lab in labels:
            pos = [r.index(lab) for r in rounds]
            if len(set(pos)) != len(pos):
                bad.append(f"{block}: budget {lab} repeats a position across rounds")
        if phase == "perf" and len(rounds) != 3:
            bad.append(f"{block}: {len(rounds)} rounds, expected 3")
        if phase == "diag" and len(rounds) != 2:
            bad.append(f"{block}: {len(rounds)} rounds, expected 2 launches per budget")
        if phase == "perf" and any("1.5G" not in r for r in rounds):
            bad.append(f"{block}: the 1.5G anchor is missing from a round")
        for x in rows:
            a = x["argv"]
            want = {"--backend": BACKEND, "--read-mode": READ_MODE, "--threads": str(MODELS[model]["gen_threads"]),
                    "--ctx": str(CTX), "--batch": str(BATCH), "--ubatch": str(UBATCH),
                    "--repeat": str(REPEAT), "--retries": str(RETRIES)}
            want["--expert-budget" if phase == "perf" else "--budgets"] = x["budget"]
            if phase == "perf":
                want["--n-predict"] = str(N_PREDICT)
            else:
                want.update({"--workload": DIAG_WORKLOAD["name"], "--workload-version": DIAG_WORKLOAD["version"],
                             "--split": DIAG_WORKLOAD["split"]})
            for f, v in want.items():
                if _flag(a, f) != v:
                    bad.append(f"{block} launch {x['launch_index']}: {f} = {_flag(a, f)!r}, expected {v!r}")
            if "--expert-store" not in a or "--expert-stats" not in a:
                bad.append(f"{block} launch {x['launch_index']}: --expert-store/--expert-stats missing")
            if "--dry-run" in a:
                bad.append(f"{block} launch {x['launch_index']}: a real launch carries --dry-run")
    return bad


def check_budgets() -> List[str]:
    bad: List[str] = []
    sys.path.insert(0, str(ROOT))
    from argine.util import parse_size  # noqa: E402
    for m in MODEL_ORDER:
        for label, b, _ in BUDGETS[m]:
            if parse_size(label) != b:
                bad.append(f"{m} {label}: parse_size gives {parse_size(label)}, frozen {b}")
            if b % MIB:
                bad.append(f"{m} {label}: not a whole number of MiB")
            if shard_count(b) != MAX_SHARDS:
                bad.append(f"{m} {label}: {shard_count(b)} shards (the main arm requires {MAX_SHARDS})")
            if b % MAX_SHARDS:
                bad.append(f"{m} {label}: not divisible by {MAX_SHARDS}")
        labels = [lab for lab, _, _ in BUDGETS[m]]
        if labels[0] != "1.5G":
            bad.append(f"{m}: the first budget is not the 1.5G anchor")
    for k, v in EXCLUDED_BUDGETS.items():
        if shard_count(v["bytes"]) == MAX_SHARDS:
            bad.append(f"excluded probe {k} would not change the shard count: exclusion reason invalid")
    if budget_bytes(H5["model"], H5["budget"]) != H5["budget_bytes"]:
        bad.append("H-5 primary point bytes differ from the budget table")
    return bad


# ============================================================== sources ====

def _lf_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_runtime_sources(root: Path = ROOT) -> List[str]:
    """The shard rule and the abort signature, read from the patch sources whose SHA-256 BUILD-INFO
    certifies for the measured binary (so they describe that binary, not a later edit)."""
    bad: List[str] = []
    bi_path = root / "llama-research-v2" / "BUILD-INFO.json"
    if _sha(bi_path) != REQUIRED_BUILD_INFO_SHA256:
        bad.append(f"BUILD-INFO.json SHA-256 differs from the frozen {REQUIRED_BUILD_INFO_SHA256[:12]}")
    bi = json.loads(bi_path.read_text(encoding="utf-8"))
    if bi.get("llama_cpp_commit") != REQUIRED_LLAMA_CPP_COMMIT:
        bad.append(f"llama.cpp commit {bi.get('llama_cpp_commit')!r} != frozen")
    if (bi.get("binaries") or {}).get("llama-server.exe") != REQUIRED_SERVER_SHA256:
        bad.append("BUILD-INFO declares a different llama-server.exe SHA-256")
    if bi.get("instrumentation") != INSTRUMENTATION_ID:
        bad.append(f"instrumentation {bi.get('instrumentation')!r} != {INSTRUMENTATION_ID}")
    if "VULKAN" in str(bi.get("cmake_flags", "")).upper():
        bad.append("BUILD-INFO enables Vulkan")
    certified = bi.get("patch_sources_sha256") or {}
    for name, needles in (("argine-store.cpp", SHARD_RULE_SOURCE_SNIPPETS),
                          ("argine-experts.cpp", RESIDENCY_ABORT_SOURCE_SNIPPETS + (MIN_BUDGET_WARNING, STORE_OPEN,
                                                                                  "copertura: %"))):
        p = root / "patches" / name
        if certified.get(name) not in (_sha(p), _lf_sha(p)):
            bad.append(f"patches/{name} is not the source BUILD-INFO certifies")
        text = p.read_text(encoding="utf-8", errors="replace")
        for n in needles:
            if n not in text:
                bad.append(f"patches/{name}: expected text not found: {n!r}")
    return bad


def check_timeout_sources() -> List[str]:
    sys.path.insert(0, str(ROOT))
    from argine import bench, server  # noqa: E402
    bad: List[str] = []
    if inspect.signature(server.LlamaServerProcess.wait_ready).parameters["timeout"].default != READY_TIMEOUT_S:
        bad.append("argine.server.LlamaServerProcess.wait_ready timeout default changed")
    for f in (bench._post, bench._post_stream_ttft):  # noqa: SLF001
        if inspect.signature(f).parameters["timeout"].default != REQUEST_TIMEOUT_S:
            bad.append(f"argine.bench.{f.__name__} timeout default changed")
    return bad


def check_workloads(root: Path = ROOT) -> List[str]:
    sys.path.insert(0, str(root))
    from argine import bench  # noqa: E402
    bad: List[str] = []
    p = bench.DEFAULT_PROMPT
    if hashlib.sha256(p.encode("utf-8")).hexdigest() != PERF_PROMPT_SHA256 or len(p) != PERF_PROMPT_CHARS:
        bad.append("argine.bench.DEFAULT_PROMPT differs from the frozen performance prompt")
    wf = root / DIAG_WORKLOAD["file"]
    if _sha(wf) != DIAG_WORKLOAD["sha256"]:
        bad.append(f"{DIAG_WORKLOAD['file']} SHA-256 differs from the frozen workload")
    w = json.loads(wf.read_text(encoding="utf-8"))
    ev = (w.get("splits") or {}).get(DIAG_WORKLOAD["split"]) or []
    if len(ev) != DIAG_WORKLOAD["requests"] or int(w.get("n_predict", 0)) != DIAG_WORKLOAD["n_predict"]:
        bad.append("diagnostic workload: request count or n_predict differs from the protocol")
    return bad


def check_footprints(root: Path = ROOT) -> List[str]:
    bad: List[str] = []
    d = derive_repo_footprints(root)
    for m in MODEL_ORDER:
        for k in ("footprint_bytes", "whole_layer_bytes", "top_k", "n_experts"):
            if d[m][k] != MODELS[m][k]:
                bad.append(f"{m}: {k} re-derived {d[m][k]} != frozen {MODELS[m][k]}")
        if d[m]["layers"] != MODELS[m]["n_layers"]:
            bad.append(f"{m}: {d[m]['layers']} layers in the metadata")
    return bad


def check_readme(readme: Path = README) -> List[str]:
    bad: List[str] = []
    t = readme.read_text(encoding="utf-8").replace("\r\n", "\n")
    if HISTORICAL_HYPOTHESIS not in t:
        bad.append("README does not contain the 2026-10-03 hypothesis verbatim")
    if t.count(MARK_BEGIN) != 1 or t.count(MARK_END) != 1:
        bad.append("README protocol markers missing or duplicated")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    st = re.search(r"\|\s*Status\s*\|\s*\*\*([A-Z]+)\*\*", t)
    if not st or st.group(1) != cfg.get("status"):
        bad.append("README status and config.json status differ")
    return bad


def static_problems() -> List[str]:
    """Everything --check verifies. Repository files only; no model, store or binary is read."""
    problems: List[str] = []
    for name, f in (("plan", lambda: check_plan(build_plan())), ("budgets", check_budgets),
                    ("footprints", check_footprints), ("runtime sources", check_runtime_sources),
                    ("timeouts", check_timeout_sources), ("workloads", check_workloads),
                    ("readme", check_readme)):
        try:
            problems += [f"{name}: {p}" for p in f()]
        except Exception as e:  # noqa: BLE001
            problems.append(f"{name}: check failed: {e}")
    return problems


# ================================================================ freeze ====

def protocol_text(readme: Path = README) -> str:
    t = readme.read_text(encoding="utf-8").replace("\r\n", "\n")
    if t.count(MARK_BEGIN) != 1 or t.count(MARK_END) != 1:
        raise ProtocolError("README protocol markers missing or duplicated")
    return t.split(MARK_BEGIN, 1)[1].split(MARK_END, 1)[0]


def protocol_spec() -> Dict[str, Any]:
    """The machine-readable protocol recorded in the freeze."""
    plan = build_plan()
    return {
        "experiment": EXP_ID, "protocol_version": PROTOCOL_VERSION,
        "freeze_kind": "prospective: written before any EXP-022 launch; no measurement exists",
        "amendments": list(AMENDMENTS),
        "historical_hypothesis_verbatim": HISTORICAL_HYPOTHESIS,
        "h5": H5,
        "outputs": {"A": "H-5 mechanism test (30B, traced diag block, historical >30% falsifier)",
                    "B": "RQ-008 usability envelope (untraced perf blocks; descriptive; does not set H-5)"},
        "model_roles": {m: MODELS[m]["role"] for m in MODEL_ORDER},
        "models": {m: {k: MODELS[m][k] for k in ("name", "path", "store", "gen_threads", "n_layers",
                                                  "n_experts", "top_k", "footprint_bytes",
                                                  "whole_layer_bytes", "footprint_source")}
                   for m in MODEL_ORDER},
        "footprint_definition": FOOTPRINT_DEFINITION,
        "budgets": {m: budget_table(m) for m in MODEL_ORDER},
        "excluded_budgets": EXCLUDED_BUDGETS,
        "shard_rule": {"rule": "n = min(16, floor(budget / 32 MiB)), >= 1; per-shard budget = budget / n",
                       "source": "patches/argine-store.cpp psx_cache_new_n (certified by BUILD-INFO)"},
        "runtime": {"binary": "llama-research-v2/llama-server.exe", "server_sha256": REQUIRED_SERVER_SHA256,
                    "build_info_sha256": REQUIRED_BUILD_INFO_SHA256, "llama_cpp_commit": REQUIRED_LLAMA_CPP_COMMIT,
                    "instrumentation": INSTRUMENTATION_ID, "backend": BACKEND, "read_mode": READ_MODE,
                    "direct_io": "on (store-open line 'cache dell'OS scavalcata')", "pretouch": "off (forced 0 by argine)",
                    "ctx": CTX, "batch": BATCH, "ubatch": UBATCH, "kv_type": KV_TYPE, "flash_attn": "on",
                    "gpu_layers": 0, "parallel": 1, "batch_threads": BATCH_THREADS,
                    "gen_threads": {m: MODELS[m]["gen_threads"] for m in MODEL_ORDER},
                    "thread_scope_note": "EXP-027 established these generation-thread optima at 4 GiB only; "
                                         "they are a frozen configuration choice here, not a claim of "
                                         "optimality below 1.5 GiB",
                    "repeat": REPEAT, "retries": RETRIES, "expert_stats": True,
                    "no_runtime_changes": "llama.cpp, expert store, shard and wait logic, allocator and "
                                          "binary are measured as they are"},
        "performance_phase": {"tracing": "OFF (ARGINE_RESEARCH_TRACE absent)",
                              "reason": "EXP-024: < 2% supported at 4G only, inconclusive at 1.5G, unknown below",
                              "tool": "argine benchmark", "prompt": "argine.bench.DEFAULT_PROMPT",
                              "prompt_sha256": PERF_PROMPT_SHA256, "prompt_chars": PERF_PROMPT_CHARS,
                              "n_predict": N_PREDICT, "cache_prompt": False,
                              "warmup": "argine.bench default (8-token request), not measured",
                              "rounds": 3, "rounds_order": PERF_ROUNDS,
                              "primary_metric": "generation tok/s of the single measured request per launch"},
        "diagnostic_phase": {"tracing": "ON (argine research trace, one budget per invocation)",
                             "workload": DIAG_WORKLOAD, "launches_per_budget": 2, "rounds_order": DIAG_ROUNDS,
                             "runs_after": "the same model's performance block is complete",
                             "reason_two_launches": "shard assignment hashes the store region pointer, so it may "
                                                    "differ between launches",
                             "objectives": ["workload gate-slice expert_level_hit_rate", "phase semantics",
                                            "misses / read bytes", "pin waits (w records)", "resident bytes vs budget",
                                            "invariant violations", "generated-token completeness",
                                            "misses_outside_gate_nodes", "routing / workload identity",
                                            "qualification"],
                             "traced_throughput": "secondary only"},
        "blocks": list(BLOCK_ORDER),
        "launch_order": [{k: x[k] for k in ("block", "launch_index", "round", "position_in_round", "budget",
                                             "budget_bytes")} for x in plan],
        "cooldown_s": COOLDOWN_S,
        "completion_rule": {"cap_s": {p: cap_seconds(p) for p in MEASURED_REQUESTS},
                            "formula": "READY_TIMEOUT_S (600, LlamaServerProcess.wait_ready) + REQUEST_TIMEOUT_S "
                                       "(1800, argine.bench request timeout) x measured requests",
                            "normal_duration_reference": NORMAL_DURATION_REFERENCE,
                            "token_completeness": {"perf": "measured request generated_tokens == 128",
                                                   "diag": "3 workload responses without error, predicted_n == 128 each"},
                            "fixed_before_data": True},
        "outcome_taxonomy": {
            "INVALID": "provenance / configuration / invariant failure or a harness safety condition; stops the block",
            "STORE_RESIDENCY_ABORT": f"the frozen store-residency abort signature {RESIDENCY_ABORT_SIGNATURE!r} "
                                     "was observed in the launch's llama-server log segment: the runtime could not "
                                     "make the requested node's slices resident, but the signature alone does not "
                                     "distinguish pinned-capacity exhaustion from commit/read failure; a scientific "
                                     "outcome, the block continues; the budget is NOT called physically infeasible",
            "abort_cause": {"labels": list(ABORT_CAUSES), "rule": ABORT_CAUSE_RULE},
            "NON_COMPLETION": "no abort signature, but the workload did not complete within the cap or the "
                              "token-completeness rule; a scientific outcome",
            "VALID": "provenance verified, measured workload complete, exit status 0, no abort signature",
            "precedence": "INVALID > STORE_RESIDENCY_ABORT > NON_COMPLETION > VALID",
            "on_store_residency_abort_or_non_completion": "the block continues with the next scheduled launch; no budget "
                                               "is removed adaptively; nothing is retried"},
        "usability_reporting_conventions": {"INTERACTIVE": ">= 5 tok/s", "DEGRADED": ">= 1 and < 5 tok/s",
                                            "NON_INTERACTIVE": "< 1 tok/s",
                                            "note": "reporting labels for VALID performance launches only; "
                                                    "not hypothesis thresholds; I/O_DOMINATED is not used (no "
                                                    "validated estimator exists)"},
        "replay_scope": {"role": "diagnostic / mechanism only; never replaces measured throughput or the "
                                 "measured runtime hit rate",
                         "tool": "tools/validate_replay_fidelity.py per diag launch, after the session",
                         "conditional_fidelity": "report whether w / U records occurred; EXP-021-level fidelity is "
                                                 "not claimed where those unvalidated paths occur; disagreement is "
                                                 "reported, never used to exclude a launch",
                         "counterfactuals": "LFU / static-frequency / Belady may be reported as mechanism context; "
                                            "sharded-lru-16-expert is not a runtime model (FH-025)"},
        "exclusion_and_qualification": {
            "h5_eligibility": H5["eligibility"],
            "replacement_runs": "none under EXP022-P1",
            "stopped_block": "a block stopped by an INVALID launch is not resumed or repeated under this freeze",
            "pooling": "no pooling across blocks or sessions without explicit labelling"},
        "machine_state": {"min_free_ram_to_start_block_bytes": MIN_START_AVAILABLE,
                          "per_launch": "free RAM before launch recorded; below 8 GiB is a warning",
                          "browsers_closed": list(BROWSER_PROCESSES), "no_orphan_llama_server": True,
                          "ac_power": "must not be reported as battery"},
        "registry_side_effect": f"`argine research trace` appends one line per run to the tracked "
                                f"{VALIDATIONS_REGISTRY}; the block gate accepts only append-only lines whose "
                                "run_dir lies under this experiment's raw/",
    }


def spec_sha256() -> str:
    return hashlib.sha256(json.dumps(protocol_spec(), sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def freeze_hashes() -> Dict[str, str]:
    return {"protocol_sha256": hashlib.sha256(protocol_text().encode("utf-8")).hexdigest(),
            "spec_sha256": spec_sha256(),
            "tool_sha256": _lf_sha(Path(__file__).resolve()),
            "tests_sha256": _lf_sha(TESTS)}


def freeze_status() -> Tuple[bool, str]:
    if not FREEZE.exists():
        return False, "protocol_freeze.json absent (run --freeze)"
    f = json.loads(FREEZE.read_text(encoding="utf-8"))
    now = freeze_hashes()
    diff = [k for k in now if f.get(k) != now[k]]
    return (not diff), ("matches the frozen protocol" if not diff else "CHANGED since freeze: " + ", ".join(diff))


def cmd_freeze() -> int:
    problems = static_problems()
    if problems:
        print("refusing to freeze: --check fails:\n  - " + "\n  - ".join(problems), file=sys.stderr)
        return 1
    now = freeze_hashes()
    supersedes = None
    if FREEZE.exists():
        old_bytes = FREEZE.read_bytes()
        old = json.loads(old_bytes.decode("utf-8"))
        if all(old.get(k) == v for k, v in now.items()):
            print("protocol_freeze.json already matches; nothing written")
            return 0
        archived = [p for p in sorted(EXP.glob("protocol_freeze_*.json")) if p.read_bytes() == old_bytes]
        if not archived:
            print("protocol_freeze.json exists and differs: copy it byte-for-byte to a dated "
                  "protocol_freeze_<label>.json and document the amendment, then refreeze", file=sys.stderr)
            return 1
        supersedes = archived[0].name
    snap = git_snapshot()
    rec = {"experiment": EXP_ID, "protocol_version": PROTOCOL_VERSION,
           "status": "prospective freeze before any EXP-022 launch; amendments A1-A2 applied before any measurement",
           "amendments": list(AMENDMENTS),
           "supersedes": supersedes,
           "protocol_hash_method": f"sha256 of the UTF-8, LF-normalised README text strictly between "
                                   f"'{MARK_BEGIN}' and '{MARK_END}'",
           "spec_hash_method": "sha256 of json.dumps(protocol_spec(), sort_keys=True, ensure_ascii=False)",
           "code_hash_method": "sha256 of the LF-normalised file bytes",
           **now,
           "argine2_at_freeze": {"head": snap.get("head"), "error": snap.get("error"),
                                 "porcelain": snap.get("porcelain"),
                                 "note": "the protocol files are uncommitted at freeze time by construction; "
                                         "--run requires them committed, HEAD descending from this commit, and "
                                         "a clean tree"},
           "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "spec": protocol_spec()}
    FREEZE.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {FREEZE.relative_to(ROOT)}: protocol {now['protocol_sha256'][:12]}, spec {now['spec_sha256'][:12]}, "
          f"tool {now['tool_sha256'][:12]}, tests {now['tests_sha256'][:12]}")
    return 0


# ============================================================== git gate ====

REQUIRED_TRACKED = (
    f"experiments/{EXP_ID}/README.md",
    f"experiments/{EXP_ID}/config.json",
    f"experiments/{EXP_ID}/protocol_freeze.json",
    "tools/exp022_run.py",
    "tests/test_exp022.py",
)
GIT_TIMEOUT_S = 60


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                          timeout=GIT_TIMEOUT_S, errors="replace")


def git_snapshot(root: Path = ROOT, freeze_head: Optional[str] = None) -> Dict[str, Any]:
    snap: Dict[str, Any] = {"available": False, "head": None, "tracked": {}, "porcelain": None,
                            "error": None, "freeze_head_is_ancestor": None, "registry_head": None}
    try:
        h = _git(root, "rev-parse", "--verify", "HEAD^{commit}")
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
            snap["tracked"][f] = _git(root, "ls-files", "--error-unmatch", "--", f).returncode == 0
        s = _git(root, "status", "--porcelain")
        snap["porcelain"] = s.stdout.splitlines() if s.returncode == 0 else None
        if freeze_head:
            snap["freeze_head_is_ancestor"] = _git(root, "merge-base", "--is-ancestor", freeze_head,
                                                   "HEAD").returncode == 0
        r = _git(root, "show", f"HEAD:{VALIDATIONS_REGISTRY}")
        snap["registry_head"] = r.stdout if r.returncode == 0 else None
    except Exception as e:  # noqa: BLE001
        snap["error"] = f"git failed: {e}"
    return snap


def registry_append_only(head_text: Optional[str], now_text: str, raw_prefix: str) -> List[str]:
    """The only change the gate tolerates: lines appended to VALIDATIONS.jsonl by EXP-022 traced runs."""
    if head_text is None:
        return [f"{VALIDATIONS_REGISTRY} not in HEAD"]
    h, n = head_text.replace("\r\n", "\n"), now_text.replace("\r\n", "\n")
    if not n.startswith(h):
        return [f"{VALIDATIONS_REGISTRY} changed other than by appending"]
    bad = []
    norm = lambda s: s.replace("\\", "/").lower()  # noqa: E731
    for line in [l for l in n[len(h):].splitlines() if l.strip()]:
        try:
            run_dir = str(json.loads(line).get("run_dir") or "")
        except json.JSONDecodeError:
            bad.append(f"{VALIDATIONS_REGISTRY}: appended line is not JSON")
            continue
        if norm(raw_prefix) not in norm(run_dir):
            bad.append(f"{VALIDATIONS_REGISTRY}: appended line for {run_dir!r}, outside {raw_prefix}")
    return bad


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
        reg = [l for l in dirty if l[3:].strip().replace("\\", "/") == VALIDATIONS_REGISTRY and l[:2] == " M"]
        other = [l for l in dirty if l not in reg]
        if other:
            bad.append("working tree not clean (git status --porcelain, untracked included): "
                       + "; ".join(other)[:600])
        if reg:
            bad += registry_append_only(snap.get("registry_head"), registry_now or "", f"experiments/{EXP_ID}/raw/")
    if snap.get("freeze_head_is_ancestor") is False:
        bad.append("the freeze's recorded HEAD is not an ancestor of the current HEAD")
    if snap.get("freeze_head_is_ancestor") is None:
        bad.append("ancestry of the freeze commit could not be checked")
    if expected_head and snap.get("head") != expected_head:
        bad.append(f"HEAD {snap.get('head')} differs from the HEAD of the earlier blocks {expected_head}")
    if not freeze_ok:
        bad.append("freeze: " + freeze_why)
    return bad


def _freeze_head() -> Optional[str]:
    if not FREEZE.exists():
        return None
    return (json.loads(FREEZE.read_text(encoding="utf-8")).get("argine2_at_freeze") or {}).get("head")


def require_run_gate(expected_head: Optional[str] = None, snapshot_fn=None, freeze_fn=None) -> Dict[str, Any]:
    snap = (snapshot_fn or (lambda: git_snapshot(freeze_head=_freeze_head())))()
    ok, why = (freeze_fn or freeze_status)()
    reg = ROOT / VALIDATIONS_REGISTRY
    bad = run_gate_problems(snap, ok, why, reg.read_text(encoding="utf-8") if reg.exists() else None,
                            expected_head)
    if bad:
        raise ProtocolError("real run refused (no session, no model, no benchmark, no trace):\n  - "
                            + "\n  - ".join(bad))
    return {"checked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "head": snap["head"], "tracked": snap["tracked"], "freeze": why}


# ============================================================ log parsing ====

def parse_server_segment(text: str) -> Dict[str, Any]:
    """The part of the shared llama-server.log written during one launch."""
    lines = text.replace("\r\n", "\n").split("\n")
    return {"starts": [l for l in lines if l.startswith("===== ")],
            "store_open": [l.strip() for l in lines if STORE_OPEN in l],
            "abort_lines": [l.strip() for l in lines if RESIDENCY_ABORT_SIGNATURE in l],
            "coverage": [(int(m.group(1)), int(m.group(2))) for m in (COVERAGE_RE.search(l) for l in lines) if m],
            "min_budget_warning": any(MIN_BUDGET_WARNING in l for l in lines),
            "vulkan": any("vulkan" in l.lower() and "device" in l.lower() for l in lines)}


def _norm(p: str) -> str:
    return os.path.normcase(os.path.normpath(str(p).strip().strip('"')))


def expected_llama_flags(item: Dict[str, Any]) -> Dict[str, str]:
    return {"--threads": str(item["threads"]), "--threads-batch": str(BATCH_THREADS),
            "--ctx-size": str(CTX), "--batch-size": str(BATCH), "--ubatch-size": str(UBATCH),
            "--n-gpu-layers": "0", "--cache-type-k": KV_TYPE, "--cache-type-v": KV_TYPE,
            "--flash-attn": "on", "--parallel": "1"}


def check_llama_command(cmdline: str, item: Dict[str, Any]) -> List[str]:
    bad: List[str] = []
    toks = cmdline.split()
    if not any(_norm(t) == _norm(MODELS[item["model"]]["path"]) for t in toks):
        bad.append("llama-server command does not name the frozen model")
    for f, v in expected_llama_flags(item).items():
        got = toks[toks.index(f) + 1] if f in toks else None
        if got != v:
            bad.append(f"llama-server {f} = {got!r}, expected {v!r}")
    if "--no-repack" in toks:
        bad.append("llama-server command carries --no-repack (store buffer type disabled)")
    return bad


def check_segment_provenance(seg: Dict[str, Any], item: Dict[str, Any]) -> List[str]:
    """Provenance verifiable even when the process aborted: one server start, the frozen command,
    one store-open line with the frozen store, budget, shard count, direct I/O and read mode."""
    bad: List[str] = []
    m = MODELS[item["model"]]
    if len(seg["starts"]) != 1:
        bad.append(f"{len(seg['starts'])} llama-server starts in the launch's log segment (expected 1)")
    else:
        if "llama-research-v2" not in seg["starts"][0].replace("\\", "/"):
            bad.append("llama-server start does not use llama-research-v2")
        bad += check_llama_command(seg["starts"][0].split(" :: ", 1)[-1], item)
    if len(seg["store_open"]) != 1:
        bad.append(f"{len(seg['store_open'])} store-open lines (expected 1)")
    else:
        line = seg["store_open"][0]
        want = (f"{m['n_layers']} layer x {m['n_experts']} expert", f"budget {item['budget_bytes'] >> 20} MiB",
                f"{shard_count(item['budget_bytes'])} settori di cache", "cache dell'OS scavalcata",
                f"lettura {READ_MODE}")
        for w in want:
            if w not in line:
                bad.append(f"store-open line lacks {w!r}: {line!r}")
        mm = re.search(r"expert store aperto:\s*(.+?\.psxs)", line, flags=re.IGNORECASE)
        if not mm or _norm(mm.group(1)) != _norm(m["store"]):
            bad.append(f"store opened is not {m['store']}")
    if seg["vulkan"]:
        bad.append("Vulkan device handling appears in the log (backend must be cpu)")
    return bad


# ======================================================== per-phase checks ====

def perf_record_problems(rec: Dict[str, Any], item: Dict[str, Any], binary_sha8: Optional[str]) -> List[str]:
    plan = rec.get("plan") or {}
    rt = rec.get("runtime") or {}
    model = rec.get("model") or {}
    checks = [
        (rec.get("event") == "benchmark", f"event {rec.get('event')!r}"),
        (_norm(str(model.get("path") or "")) == _norm(MODELS[item["model"]]["path"]), f"model path {model.get('path')!r}"),
        (plan.get("threads") == item["threads"], f"plan.threads {plan.get('threads')}"),
        (plan.get("threads_batch") == BATCH_THREADS, f"plan.threads_batch {plan.get('threads_batch')}"),
        (plan.get("ctx") == CTX and plan.get("batch") == BATCH and plan.get("ubatch") == UBATCH,
         f"plan ctx/batch/ubatch {plan.get('ctx')}/{plan.get('batch')}/{plan.get('ubatch')}"),
        (plan.get("kv_type_k") == KV_TYPE and plan.get("kv_type_v") == KV_TYPE, "plan KV type"),
        (plan.get("flash_attn") is True and plan.get("parallel") == 1, "plan flash_attn/parallel"),
        (plan.get("backend") == BACKEND and rt.get("backend") == BACKEND, "backend not cpu"),
        ((plan.get("gpu_layers") or 0) == 0, f"gpu_layers {plan.get('gpu_layers')}"),
        (plan.get("expert_store") is True, "expert store not enabled in the plan"),
        (plan.get("expert_cache_budget") == item["budget_bytes"], f"expert budget {plan.get('expert_cache_budget')}"),
        (str(rec.get("modo_lettura") or "") == READ_MODE, f"read mode {rec.get('modo_lettura')!r}"),
        (rec.get("repeat") == REPEAT and rec.get("n_predict") == N_PREDICT, "repeat / n_predict"),
        (rec.get("prompt_chars") == PERF_PROMPT_CHARS, f"prompt_chars {rec.get('prompt_chars')}"),
        (not rec.get("attempts"), f"server start needed retries: {rec.get('attempts')}"),
        (bool(binary_sha8) and binary_sha8 in str(rt.get("build") or ""), f"runtime build {rt.get('build')!r}"),
    ]
    return [msg for ok, msg in checks if not ok]


def perf_invariant_problems(rec: Dict[str, Any], item: Dict[str, Any]) -> List[str]:
    ec = rec.get("expert_cache") or {}
    bad = []
    if ec.get("mib_budget") is not None and ec["mib_budget"] != item["budget_bytes"] >> 20:
        bad.append(f"store reports budget {ec['mib_budget']} MiB")
    if ec.get("mib_residenti") is not None and ec.get("mib_budget") is not None and ec["mib_residenti"] > ec["mib_budget"]:
        bad.append(f"resident {ec['mib_residenti']} MiB > budget {ec['mib_budget']} MiB")
    return bad


def perf_measured(rec: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
    if rec is None:
        return False, "no benchmark record"
    met = rec.get("metrics") or {}
    runs = met.get("runs") or []
    errs = [str(e) for e in met.get("errors") or []]
    if len(runs) != REPEAT:
        return False, f"{len(runs)} measured runs"
    r = runs[0]
    if any(e.startswith("warmup") or e.startswith("run 1") for e in errs):
        return False, "error in the warm-up or measured request: " + "; ".join(errs)[:300]
    if not r.get("gen_tok_s") or r.get("generated_tokens") != N_PREDICT:
        return False, f"generated {r.get('generated_tokens')} of {N_PREDICT} tokens"
    return True, ("complete" + (f" (later errors: {'; '.join(errs)[:200]})" if errs else ""))


def diag_manifest_problems(man: Dict[str, Any], summ: Optional[Dict[str, Any]], item: Dict[str, Any],
                           binary_sha: Optional[str]) -> List[str]:
    fp = man.get("fingerprint") or {}
    rf = fp.get("runtime_flags") or {}
    rc = (summ or {}).get("runtime_confirmation") or {}
    cfg = ((summ or {}).get("raw") or {}).get("config") or {}
    val = man.get("validation") or {}
    be = (rf.get("backend") or {})
    checks = [
        (man.get("budget") == item["budget"], f"manifest budget {man.get('budget')!r}"),
        (man.get("workload") == DIAG_WORKLOAD["name"] and man.get("workload_sha256") == DIAG_WORKLOAD["sha256"],
         "workload identity"),
        (man.get("split") == DIAG_WORKLOAD["split"], f"split {man.get('split')!r}"),
        (rf.get("threads") == item["threads"], f"runtime_flags.threads {rf.get('threads')}"),
        (be.get("effective") == BACKEND and be.get("requested") == BACKEND, f"backend {be}"),
        (val.get("instrumentation_confirmed") is True, "instrumentation not confirmed"),
        (val.get("binary_sha256") == binary_sha == REQUIRED_SERVER_SHA256, "binary SHA-256"),
        (val.get("build_info_sha256") == REQUIRED_BUILD_INFO_SHA256, "BUILD-INFO SHA-256"),
        ((man.get("read_mode") or {}).get("agree") is True, "read mode requested != effective"),
        ((man.get("expert_store") or {}).get("fingerprint_matches_effective") is True, "store fingerprint"),
        (_norm(str((man.get("expert_store") or {}).get("effective") or "")) == _norm(MODELS[item["model"]]["store"]),
         "effective store path"),
        (str(cfg.get("budget_bytes")) == str(item["budget_bytes"]), f"runtime budget_bytes {cfg.get('budget_bytes')}"),
        (str(cfg.get("cache_shards")) == str(shard_count(item["budget_bytes"])), f"cache_shards {cfg.get('cache_shards')}"),
        (str(cfg.get("direct_io")) == "1" and str(cfg.get("pretouch")) == "0", "direct I/O / pretouch"),
        (rc.get("read_mode") == READ_MODE, f"runtime read mode {rc.get('read_mode')!r}"),
    ]
    bad = [msg for ok, msg in checks if not ok]
    bad += check_llama_command(" ".join(str(x) for x in (rf.get("llama_args") or [])), item)
    return bad


def diag_invariant_problems(man: Dict[str, Any]) -> List[str]:
    bad = []
    if (man.get("budget_violations") or 0) != 0:
        bad.append(f"hard budget violated: {man.get('budget_violations')} violations")
    mr = (man.get("final_store_stats") or {}).get("max_resident_minus_budget")
    if mr is not None and mr > 0:
        bad.append(f"resident exceeded the budget by {mr} B")
    return bad


def diag_measured(man: Optional[Dict[str, Any]], responses: List[Dict[str, Any]]) -> Tuple[bool, str]:
    if man is None:
        return False, "no manifest"
    if man.get("errors"):
        return False, "workload errors: " + "; ".join(map(str, man["errors"]))[:300]
    if len(responses) != DIAG_WORKLOAD["requests"]:
        return False, f"{len(responses)} workload responses"
    short = [r.get("prompt_id") for r in responses if r.get("predicted_n") != DIAG_WORKLOAD["n_predict"]]
    if short:
        return False, f"token-incomplete responses: {short}"
    return True, "complete"


def classify(provenance: List[str], invariant: List[str], harness: List[str], abort_lines: List[str],
             measured_complete: bool, exit_status: Optional[int], timed_out: bool, why: str) -> Dict[str, Any]:
    """Frozen precedence INVALID > STORE_RESIDENCY_ABORT > NON_COMPLETION > VALID."""
    if provenance or invariant or harness:
        return {"outcome": "INVALID", "reason": "; ".join(provenance + invariant + harness)[:1000]}
    if abort_lines:
        return {"outcome": "STORE_RESIDENCY_ABORT", "reason": abort_lines[0][:500]}
    if timed_out:
        return {"outcome": "NON_COMPLETION", "reason": "wall-clock cap exceeded"}
    if not measured_complete or exit_status != 0:
        return {"outcome": "NON_COMPLETION",
                "reason": why if not measured_complete else f"exit status {exit_status}"}
    return {"outcome": "VALID", "reason": why}


# ============================================================== analysis ====

def describe(values: List[float]) -> Optional[Dict[str, Any]]:
    if not values:
        return None
    m = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else None
    return {"n": len(values), "values": list(values), "mean": m, "median": statistics.median(values),
            "sd": sd, "cv": (sd / m if sd is not None and m else None), "min": min(values), "max": max(values)}


def layer_expert_bytes(model: str, root: Path = ROOT) -> Dict[int, int]:
    """Per-layer bytes of one expert (gate + up + down, aligned), from the tracked EXP-007 trace_meta."""
    d = json.loads((root / MODELS[model]["footprint_source"]).read_text(encoding="utf-8"))
    return {int(k): int(v) for k, v in d["trace_meta"]["expert_bytes"].items()}


def parse_residency_abort(line: str) -> Optional[Dict[str, Any]]:
    m = RESIDENCY_ABORT_RE.search(line)
    if not m:
        return None
    return {"layer": int(m.group(1)), "tensor": m.group(2), "resident_mib": int(m.group(3)),
            "budget_mib": int(m.group(4)), "node_experts": int(m.group(5)), "layer_experts": int(m.group(6))}


def abort_cause(abort_lines: List[str], coverage: List[Tuple[int, int]], budget: int, model: str,
                reads_failed: Optional[int], layer_bytes: Optional[Dict[int, int]] = None) -> Dict[str, Any]:
    """Frozen, evidence-only attribution of a STORE_RESIDENCY_ABORT (ABORT_CAUSE_RULE). Never guesses."""
    parsed = parse_residency_abort(abort_lines[0]) if abort_lines else None
    out: Dict[str, Any] = {"parsed_abort": parsed, "coverage": coverage[-1] if coverage else None,
                           "reads_failed": reads_failed, "node_demand_bytes": None}
    full_cov = bool(coverage) and coverage[-1][0] == coverage[-1][1] and coverage[-1][1] > 0
    if parsed is not None:
        lb = layer_bytes if layer_bytes is not None else layer_expert_bytes(model)
        if parsed["layer"] in lb:
            out["node_demand_bytes"] = parsed["node_experts"] * lb[parsed["layer"]]
    if parsed is not None and full_cov and out["node_demand_bytes"] is not None and out["node_demand_bytes"] > budget:
        out["cause"] = "capacity_or_pinning"
    elif reads_failed is not None and reads_failed > 0:
        out["cause"] = "commit_or_read_failure"
    else:
        out["cause"] = "cause_unresolved"
    return out


def h5_decision(records: List[Dict[str, Any]], block_complete: bool) -> Dict[str, Any]:
    pts = [r for r in records if r["exp022"]["block"] == f"{H5['model']}-diag"
           and r["exp022"]["budget"] == H5["budget"]]
    elig = [r for r in pts if r["exp022"].get("h5_eligible")]
    vals = [float(r["exp022"]["expert_level_hit_rate"]) for r in elig]
    base = {"rule": H5, "launches": [{"launch_index": r["exp022"]["launch_index"], "outcome": r["exp022"]["outcome"],
                                      "h5_eligible": r["exp022"].get("h5_eligible"),
                                      "expert_level_hit_rate": r["exp022"].get("expert_level_hit_rate")} for r in pts]}
    if len(pts) > H5["scheduled_launches"] or len({r["exp022"]["launch_index"] for r in pts}) != len(pts):
        raise ProtocolError(f"{len(pts)} records at the H-5 point; exactly {H5['scheduled_launches']} are scheduled "
                            "(no replacement launch, no retry)")
    if not block_complete or len(vals) < H5["minimum_eligible_launches"]:
        return {**base, "evaluable": False, "aggregate": None, "historical_prediction": "NOT EVALUABLE",
                "formal_status": "INCONCLUSIVE", "eligible_launches": len(vals),
                "reason": "the 30B diag block is incomplete" if not block_complete else
                          f"{len(vals)} of the {H5['scheduled_launches']} scheduled 30B / 768 MiB launches are "
                          f"H-5-eligible; the rule needs {H5['minimum_eligible_launches']}"}
    agg = statistics.mean(vals)
    falsified = agg > H5["threshold"]
    return {**base, "evaluable": True, "aggregate": agg, "dispersion": describe(vals), "eligible_launches": len(vals),
            "historical_prediction": "FALSIFIED" if falsified else "NOT FALSIFIED BY THIS TEST",
            "formal_status": "FALSIFIED" if falsified else "INCONCLUSIVE",
            "reason": (f"mean workload gate-slice hit rate {agg:.6f} > {H5['threshold']}" if falsified else
                       f"mean workload gate-slice hit rate {agg:.6f} <= {H5['threshold']}: not falsified; "
                       "no positive support criterion was pre-specified")}


def routing_sha(trace_path: Path) -> Tuple[int, str]:
    """Routing identity of the workload steps (warm-up and flush sentinel excluded); same formula as
    tools/exp023_analyze.routing_sha."""
    sys.path.insert(0, str(ROOT))
    from argine.research.trace import Trace  # noqa: E402
    tr = Trace.load(trace_path)
    h = hashlib.sha256()
    n = 0
    for st in tr.steps:
        if st.segment in ("warmup", "flush-sentinel"):
            continue
        n += 1
        h.update(json.dumps([st.segment, st.n_tokens, [sorted(st.experts(l)) for l in tr.layers]]).encode())
    return n, h.hexdigest()


def _perf_row(r: Dict[str, Any]) -> Dict[str, Any]:
    e = r["exp022"]
    run0 = ((r.get("metrics") or {}).get("runs") or [{}])[0] if r.get("metrics") else {}
    res = ((r.get("metrics") or {}).get("resources") or {}) if r.get("metrics") else {}
    ec = r.get("expert_cache") or {}
    g = run0.get("gen_tok_s") if e["outcome"] == "VALID" else None
    return {"gen_tok_s": g, "ms_per_token": (1000.0 / g if g else None),
            "usability": usability_category(g) if e["outcome"] == "VALID" else e["outcome"],
            "ttft_s": run0.get("ttft_s"), "prompt_tok_s": run0.get("prompt_tok_s"),
            "ram_peak": res.get("ram_peak"),
            "ram_peak_minus_budget": (res["ram_peak"] - e["budget_bytes"]) if res.get("ram_peak") else None,
            "store_hit_pct_runtime_style_process_total": ec.get("hit_pct"),
            "store_mib_read_process_total": ec.get("mib_letti"), "ms_attesa_sum_threads": ec.get("ms_attesa"),
            "attese": ec.get("attese"), "coda_proxy": ec.get("coda")}


def analyze(records: List[Dict[str, Any]], sessions: Dict[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    complete = {b: (sessions.get(b) or {}).get("status") == "complete" and bool((sessions.get(b) or {}).get("valid"))
                for b in BLOCK_ORDER}
    rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {"experiment": EXP_ID, "protocol_version": PROTOCOL_VERSION,
                               "blocks_complete": complete, "envelope": {}, "diagnostics": {}}
    for r in sorted(records, key=lambda x: (BLOCK_ORDER.index(x["exp022"]["block"]), x["exp022"]["launch_index"])):
        e = r["exp022"]
        row = {k: e.get(k) for k in ("block", "launch_index", "phase", "model", "round", "position_in_round",
                                     "budget", "budget_bytes", "fraction_of_footprint", "outcome",
                                     "elapsed_s", "min_budget_warning")}
        row["outcome_reason"] = (e.get("reason") or "")[:200]
        row["abort_cause"] = (e.get("abort_cause") or {}).get("cause")
        if e["phase"] == "perf":
            row.update({k: v for k, v in _perf_row(r).items() if k in ("gen_tok_s", "ms_per_token", "usability",
                                                                      "ttft_s", "prompt_tok_s", "ram_peak")})
        else:
            row.update({"expert_level_hit_rate": e.get("expert_level_hit_rate"),
                        "runtime_style_hit_rate_labelled": e.get("runtime_style_hit_rate"),
                        "h5_eligible": e.get("h5_eligible"), "traced_gen_tok_s_secondary": e.get("traced_gen_tok_s")})
        rows.append(row)
    for b in BLOCK_ORDER:
        model, phase = b.split("-")
        rs = [r for r in records if r["exp022"]["block"] == b]
        cells = {}
        for label, by, role in BUDGETS[model]:
            c = [r for r in rs if r["exp022"]["budget"] == label]
            counts = {o: sum(1 for r in c if r["exp022"]["outcome"] == o) for o in OUTCOMES}
            causes = {k: sum(1 for r in c if (r["exp022"].get("abort_cause") or {}).get("cause") == k) for k in ABORT_CAUSES}
            cell: Dict[str, Any] = {"bytes": by, "role": role, "fraction_of_footprint": by / MODELS[model]["footprint_bytes"],
                                    "outcome_counts": counts, "store_residency_abort_causes": causes}
            if phase == "perf":
                pr = [_perf_row(r) for r in c]
                vals = [p["gen_tok_s"] for p in pr if p["gen_tok_s"]]
                labels = sorted({p["usability"] for p in pr})
                cell.update({"gen_tok_s_valid": describe(vals),
                             "usability_labels": labels,
                             "envelope_label": labels[0] if len(labels) == 1 else "MIXED: " + ", ".join(labels),
                             "secondary": pr})
            else:
                hr = [float(r["exp022"]["expert_level_hit_rate"]) for r in c if r["exp022"].get("expert_level_hit_rate") is not None
                      and r["exp022"]["outcome"] == "VALID"]
                cell.update({"expert_level_hit_rate_valid": describe(hr),
                             "launch_diagnostics": [r["exp022"].get("diagnostics") for r in c]})
            cells[label] = cell
        (summary["envelope"] if phase == "perf" else summary["diagnostics"])[b] = cells
        if phase == "diag":
            shas = sorted({r["exp022"].get("routing_sha256") for r in rs if r["exp022"]["outcome"] == "VALID"} - {None})
            summary["diagnostics"][b]["routing_identity"] = {"distinct_sha256_over_valid_launches": shas,
                                                             "identical": len(shas) <= 1}
    summary["h5"] = h5_decision(records, complete[f"{H5['model']}-diag"])
    summary["formal_status"] = summary["h5"]["formal_status"]
    summary["notes"] = ["usability labels are frozen reporting conventions, not hypothesis thresholds",
                        "STORE_RESIDENCY_ABORT is its own category, never converted to NON_INTERACTIVE, and does not by "
                        "itself mean the budget is physically infeasible",
                        "the 80B arm is exploratory: descriptive only",
                        "runtime-style hit rates and process-total store counters are labelled and include warm-up",
                        "replay fidelity, where run, is diagnostic and never excludes a launch"]
    buf = io.StringIO()
    fields: List[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    w = csv.DictWriter(buf, fieldnames=fields or ["block"], lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue(), summary


# ============================================================ provenance ====

LARGE_FILE_POLICY = (
    "no full read of the model or Expert Store before or between a block's launches: pre-block snapshot = "
    "path, size, mtime, Argine2 fast fingerprint, plus a full SHA-256 only if already known without reading "
    "the file; missing full hashes are computed after the block and the block is valid only if size and "
    "mtime did not change")


def _stat_snapshot(path: Path) -> Dict[str, Any]:
    st = path.stat()
    return {"path": str(path), "bytes": st.st_size, "mtime": st.st_mtime, "mtime_ns": st.st_mtime_ns}


def known_full_hash(path: Path, size: int, mtime: float) -> Optional[Dict[str, Any]]:
    sys.path.insert(0, str(ROOT))
    from argine.research import fingerprint  # noqa: E402
    key = f"{path.resolve()}|{size}|{int(mtime)}|sha256"
    try:
        cache = json.loads(fingerprint._cache_path().read_text(encoding="utf-8"))  # noqa: SLF001
    except Exception:  # noqa: BLE001
        cache = {}
    if key in cache:
        return {"digest": cache[key], "method": "sha256", "source": f"Argine2 checksum cache, key {key}"}
    return None


def free_ram() -> Optional[int]:
    try:
        import psutil  # type: ignore
        return int(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001
        return None


def running_processes() -> List[str]:
    try:
        import psutil  # type: ignore
        return [str(p.info.get("name") or "").lower() for p in psutil.process_iter(["name"])]
    except Exception:  # noqa: BLE001
        return []


def orphan_servers() -> List[Dict[str, Any]]:
    sys.path.insert(0, str(ROOT))
    from argine import runtime as rtmod  # noqa: E402
    return rtmod.find_orphans()


def ac_power() -> Optional[bool]:
    try:
        import psutil  # type: ignore
        b = psutil.sensors_battery()
        return None if b is None or b.power_plugged is None else bool(b.power_plugged)
    except Exception:  # noqa: BLE001
        return None


def machine_problems() -> Tuple[List[str], Dict[str, Any]]:
    bad: List[str] = []
    fr = free_ram()
    procs = running_processes()
    browsers = sorted({p for p in procs if p in BROWSER_PROCESSES})
    orph = orphan_servers()
    ac = ac_power()
    if fr is None:
        bad.append("free RAM unknown (psutil missing)")
    elif fr < MIN_START_AVAILABLE:
        bad.append(f"free RAM {fr} B < {MIN_START_AVAILABLE} B required to start a block")
    if browsers:
        bad.append("browser processes running (close them before a performance-sensitive block): " + ", ".join(browsers))
    if orph:
        bad.append(f"{len(orph)} llama-server process(es) already running")
    if ac is False:
        bad.append("running on battery")
    return bad, {"free_ram": fr, "browsers": browsers, "orphans": len(orph), "ac_power": ac}


def binary_path(llama_dir: Path) -> Path:
    exe = llama_dir / "llama-server.exe"
    return exe if exe.exists() else llama_dir / "llama-server"


def gguf_footprint(model: str) -> Dict[str, int]:
    """Preflight only: the same derivation from the GGUF header (no tensor data is read)."""
    sys.path.insert(0, str(ROOT))
    from argine.gguf import read_gguf  # noqa: E402
    from argine.research.runner import expert_bytes_from_model  # noqa: E402
    m = read_gguf(Path(MODELS[model]["path"]))
    return footprint_from_meta({"expert_bytes": expert_bytes_from_model(m), "top_k": m.expert_used_count,
                                "n_experts": m.expert_count})


def base_env(parent: Dict[str, str], llama_dir: Path) -> Dict[str, str]:
    extra = sorted(k for k in parent if (k.upper().startswith("ARGINE_") and k.upper() not in ALLOWED_ARGINE_ENV)
                   or k.upper().startswith("PICOSTAR_"))
    if extra:
        raise ProtocolError("parent environment carries variables that could change the runtime: " + ", ".join(extra))
    cur = parent.get("ARGINE_LLAMA_DIR")
    if cur and Path(cur).resolve() != llama_dir.resolve():
        raise ProtocolError(f"ARGINE_LLAMA_DIR={cur} differs from the research runtime {llama_dir}")
    env = dict(parent)
    env["ARGINE_LLAMA_DIR"] = str(llama_dir)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _block_dir(block: str) -> Path:
    return RAW / block


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def previous_blocks_problems(block: str) -> Tuple[List[str], Optional[str]]:
    bad: List[str] = []
    head = None
    for b in BLOCK_ORDER[:BLOCK_ORDER.index(block)]:
        s = _block_dir(b) / "session.json"
        if not s.exists():
            bad.append(f"block {b} has not been run (frozen order {' -> '.join(BLOCK_ORDER)})")
            continue
        d = json.loads(s.read_text(encoding="utf-8"))
        if d.get("status") != "complete" or not d.get("valid"):
            bad.append(f"block {b} is {d.get('status')!r}: later blocks are refused")
        head = head or d.get("checked_head_commit")
    return bad, head


def preflight(block: str, llama_dir: Path, parent: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Machine and runtime checks plus `argine benchmark --dry-run` per budget of the block."""
    parent = dict(os.environ) if parent is None else parent
    plan = block_plan(block)
    model = plan[0]["model"]
    problems = static_problems()
    checks: Dict[str, Any] = {}
    try:
        base = base_env(parent, llama_dir)
    except ProtocolError as e:
        problems.append(str(e))
        base = None
    if "ARGINE_RESEARCH_TRACE" in parent:
        problems.append("ARGINE_RESEARCH_TRACE is set in the parent environment")
    bp = binary_path(llama_dir)
    if not bp.exists():
        problems.append(f"research binary missing: {bp}")
    elif _sha(bp) != REQUIRED_SERVER_SHA256:
        problems.append("llama-server SHA-256 differs from the frozen binary")
    for role in ("path", "store"):
        if not Path(MODELS[model][role]).exists():
            problems.append(f"{role} missing: {MODELS[model][role]}")
    if not problems:
        try:
            g = gguf_footprint(model)
            checks["gguf_footprint"] = g
            for k in ("footprint_bytes", "whole_layer_bytes"):
                if g[k] != MODELS[model][k]:
                    problems.append(f"{model}: {k} from the GGUF header {g[k]} != frozen {MODELS[model][k]}")
        except Exception as e:  # noqa: BLE001
            problems.append(f"GGUF header derivation failed: {e}")
    prev, _ = previous_blocks_problems(block)
    problems += prev
    bd = _block_dir(block)
    if bd.exists() and any(bd.iterdir()):
        problems.append(f"{bd} is not empty: a block is never resumed or repeated")
    mp, ms = machine_problems()
    problems += mp
    checks["machine"] = ms
    ok_f, why = freeze_status()
    checks["freeze"] = why
    if not ok_f:
        problems.append("freeze: " + why)
    dry: Dict[str, Any] = {}
    if base is not None and not any("missing" in p for p in problems):
        for label, _, _ in BUDGETS[model]:
            item = next(x for x in plan if x["budget"] == label)
            r = subprocess.run([sys.executable, "-m", "argine"] + perf_args(model, label, dry_run=True), cwd=str(ROOT),
                               env=base, capture_output=True, text=True, encoding="utf-8", errors="replace")
            txt = (r.stdout or "") + (r.stderr or "")
            cmdline = next((l.strip() for l in txt.splitlines() if "--threads-batch" in l), "")
            per = {"exit": r.returncode, "llama_command": cmdline, "server_started": "avvio llama-server" in txt}
            if r.returncode != 0 or not cmdline or per["server_started"]:
                problems.append(f"dry-run {label} failed or started a server (exit {r.returncode})")
            else:
                problems += [f"dry-run {label}: {p}" for p in check_llama_command(cmdline, item)]
            dry[label] = per
    checks["dry_run"] = dry
    return {"ok": not problems, "problems": problems, "checks": checks}


def session_provenance(llama_dir: Path, model: str) -> Dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from argine.research import fingerprint  # noqa: E402
    out: Dict[str, Any] = {"argine2": fingerprint.git_state(ROOT), "binary": fingerprint.binary_info(binary_path(llama_dir)),
                           "build_info_sha256": _sha(llama_dir / "BUILD-INFO.json"),
                           "hardware": fingerprint.machine_identity(), "python": sys.version.split()[0],
                           "platform": platform.platform(), "large_file_policy": LARGE_FILE_POLICY, "files": {}}
    for role, p in (("model", Path(MODELS[model]["path"])), ("expert_store", Path(MODELS[model]["store"]))):
        snap = _stat_snapshot(p)
        known = known_full_hash(p, snap["bytes"], snap["mtime"])
        out["files"][role] = {"pre_run": snap, "fast": fingerprint.file_checksum(p, full=False), "full": known,
                              "full_status": "reused (no file read)" if known else "pending post-block"}
    return out


def finalize_fingerprints(session: Dict[str, Any], llama_dir: Path, hasher=None) -> List[str]:
    sys.path.insert(0, str(ROOT))
    from argine.research import fingerprint  # noqa: E402
    hasher = hasher or (lambda p: fingerprint.file_checksum(p, full=True))
    problems: List[str] = []
    for role, e in session["provenance"]["files"].items():
        pre = e["pre_run"]
        path = Path(pre["path"])

        def changed() -> Optional[str]:
            if not path.exists():
                return "file missing"
            st = path.stat()
            return None if (st.st_size == pre["bytes"] and st.st_mtime_ns == pre["mtime_ns"]) else "size/mtime changed"
        why = changed()
        if why is None and not (e.get("full") or {}).get("digest"):
            h = hasher(path)
            why = changed()
            if why is None:
                e["full"] = {"digest": h.get("digest"), "method": h.get("method"), "source": "computed after the block"}
                e["full_status"] = "computed post-block"
        e["post_run_check"] = {"unchanged": why is None, "problem": why}
        if why:
            problems.append(f"{role} {path}: {why}")
    if _sha(binary_path(llama_dir)) != session["provenance"]["binary"].get("digest"):
        problems.append("llama-server binary changed during the block")
    if _sha(llama_dir / "BUILD-INFO.json") != session["provenance"]["build_info_sha256"]:
        problems.append("BUILD-INFO.json changed during the block")
    return problems


# =================================================================== run ====

def _size(p: Path) -> int:
    return p.stat().st_size if p.exists() else 0


def _read_from(p: Path, offset: int) -> str:
    if not p.exists():
        return ""
    with open(p, "rb") as fh:
        fh.seek(offset)
        return fh.read().decode("utf-8", "replace")


def _logs() -> Tuple[Path, Path]:
    sys.path.insert(0, str(ROOT))
    from argine import telemetry  # noqa: E402
    from argine.util import logs_dir  # noqa: E402
    return telemetry.log_path(), logs_dir() / "llama-server.log"


def _diag_artifacts(item: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[Dict[str, Any]], Optional[Path]]:
    d = EXP / item["launch_dir"]
    runs = sorted(d.glob("run-*")) if d.exists() else []
    if len(runs) != 1:
        return None, None, [], None
    rd = runs[0]
    man = json.loads((rd / "manifest.json").read_text(encoding="utf-8")) if (rd / "manifest.json").exists() else None
    summ = json.loads((rd / "summary.json").read_text(encoding="utf-8")) if (rd / "summary.json").exists() else None
    resp = _read_jsonl(rd / "responses.jsonl")
    return man, summ, resp, rd


def diag_diagnostics(man: Dict[str, Any], summ: Optional[Dict[str, Any]], resp: List[Dict[str, Any]]) -> Dict[str, Any]:
    fs = man.get("final_store_stats") or {}
    wl = fs.get("workload") or {}
    s = summ or {}
    dec = [r for r in resp if r.get("predicted_ms")]
    return {"workload_outcomes": wl.get("outcomes"), "workload_reads_ok": wl.get("reads_ok"),
            "workload_read_payload_bytes": wl.get("read_payload_bytes"), "workload_evictions": wl.get("evictions"),
            "w_records": (wl.get("outcomes") or {}).get("w", 0), "undo_records": ((s.get("raw") or {}).get("undos")),
            "misses_outside_gate_nodes_all_records": (s.get("hit_rate_semantics") or {}).get("misses_outside_gate_nodes"),
            "unassigned_accesses_all_records": (s.get("hit_rate_semantics") or {}).get("unassigned_accesses"),
            "max_resident_minus_budget": fs.get("max_resident_minus_budget"), "violations": man.get("budget_violations"),
            "trace_complete": (man.get("trace_completeness") or {}).get("complete"),
            "truncated_tail": (s.get("raw") or {}).get("truncated_tail"),
            "queue_depth_status": ((s.get("io") or {}).get("queue_depth") or {}).get("status"),
            "publication_quality": man.get("publication_quality"), "missing": man.get("missing"),
            "traced_gen_tok_s_secondary": (sum(r["predicted_n"] for r in dec) / (sum(r["predicted_ms"] for r in dec) / 1000.0)
                                           if dec else None)}


def run_block(block: str, llama_dir: Path) -> int:
    prev, expected_head = previous_blocks_problems(block)
    if prev:
        print("PROTOCOL ERROR: " + "; ".join(prev), file=sys.stderr)
        return 2
    try:
        gate = require_run_gate(expected_head)
    except ProtocolError as e:
        print(f"PROTOCOL ERROR: {e}", file=sys.stderr)
        return 2
    pf = preflight(block, llama_dir)
    if not pf["ok"]:
        print("preflight failed; nothing was run:\n  - " + "\n  - ".join(pf["problems"]), file=sys.stderr)
        return 2
    try:
        gate2 = require_run_gate(expected_head)
    except ProtocolError as e:
        print(f"PROTOCOL ERROR: {e}", file=sys.stderr)
        return 2
    if gate2["head"] != gate["head"]:
        print("PROTOCOL ERROR: HEAD changed during preflight", file=sys.stderr)
        return 2
    plan = block_plan(block)
    model = plan[0]["model"]
    base = base_env(dict(os.environ), llama_dir)
    bd = _block_dir(block)
    session = {"experiment": EXP_ID, "protocol_version": PROTOCOL_VERSION, "block": block,
               "session_id": datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8],
               "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "freeze": json.loads(FREEZE.read_text(encoding="utf-8")), "llama_dir": str(llama_dir),
               "schedule": [{**x, "command": command(x)} for x in plan],
               "git_gate": {"before_preflight": gate, "before_session": gate2},
               "checked_head_commit": gate["head"], "preflight": pf,
               "provenance": session_provenance(llama_dir, model), "status": "started", "valid": None}
    sp, rp = bd / "session.json", bd / "records.jsonl"
    _write_json(sp, session)
    bsha = session["provenance"]["binary"].get("digest")
    runs_log, srv_log = _logs()
    for n, x in enumerate(plan):
        if n:
            time.sleep(COOLDOWN_S)
        harness: List[str] = []
        orph = orphan_servers()
        if orph:
            harness.append(f"{len(orph)} llama-server process(es) running before the launch")
        logf = bd / "logs" / f"launch-{x['launch_index']:02d}.log"
        logf.parent.mkdir(parents=True, exist_ok=True)
        if logf.exists() or (EXP / x["launch_dir"]).exists():
            raise ProtocolError(f"launch {x['launch_index']} output already exists")
        o_runs, o_srv = _size(runs_log), _size(srv_log)
        ram_before = free_ram()
        t_launch = datetime.now(timezone.utc).isoformat(timespec="seconds")
        print(f"[{block} {x['launch_index']:>2}/{len(plan)}] round {x['round']}  {x['budget']}")
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
        if orphan_servers():
            sys.path.insert(0, str(ROOT))
            from argine import runtime as rtmod  # noqa: E402
            rtmod.kill_orphans([o["pid"] for o in orphan_servers()])
            time.sleep(SETTLE_S)
            if orphan_servers():
                harness.append("llama-server still running after the launch ended (could not be terminated)")
        seg = parse_server_segment(_read_from(srv_log, o_srv))
        prov = check_segment_provenance(seg, x) if not harness else []
        inv: List[str] = []
        out: Dict[str, Any] = {}
        diag_extra: Dict[str, Any] = {}
        reads_failed: Optional[int] = None
        if x["phase"] == "perf":
            appended = [json.loads(l) for l in _read_from(runs_log, o_runs).splitlines() if l.strip().startswith("{")]
            bench = [r for r in appended if r.get("event") == "benchmark"]
            if len(bench) > 1:
                harness.append(f"{len(bench)} benchmark records appended")
            rec = bench[0] if len(bench) == 1 else None
            if rec is not None:
                prov += perf_record_problems(rec, x, (bsha or "")[:8] or None)
                inv += perf_invariant_problems(rec, x)
                out = dict(rec)
            done, why = perf_measured(rec)
        else:
            man, summ, resp, rd = _diag_artifacts(x)
            if man is not None:
                prov += diag_manifest_problems(man, summ, x, bsha)
                inv += diag_invariant_problems(man)
                diag_extra["diagnostics"] = diag_diagnostics(man, summ, resp)
                reads_failed = ((man.get("final_store_stats") or {}).get("trace_totals") or {}).get("reads_failed")
                fs = man.get("final_store_stats") or {}
                diag_extra["expert_level_hit_rate"] = fs.get("expert_level_hit_rate")
                diag_extra["runtime_style_hit_rate"] = fs.get("runtime_style_hit_rate")
                diag_extra["traced_gen_tok_s"] = diag_extra["diagnostics"]["traced_gen_tok_s_secondary"]
                if rd is not None and (rd / "trace.jsonl.gz").exists():
                    try:
                        diag_extra["routing_steps"], diag_extra["routing_sha256"] = routing_sha(rd / "trace.jsonl.gz")
                    except Exception as e:  # noqa: BLE001
                        diag_extra["routing_error"] = str(e)[:300]
                out = {"manifest_path": str(rd.relative_to(EXP)) if rd else None}
            done, why = diag_measured(man, resp)
        cls = classify(prov, inv, harness, seg["abort_lines"], done, rc, timed_out, why)
        if cls["outcome"] == "STORE_RESIDENCY_ABORT":
            cls["abort_cause"] = abort_cause(seg["abort_lines"], seg["coverage"], x["budget_bytes"], x["model"],
                                             reads_failed)
        h5_ok = (x["phase"] == "diag" and cls["outcome"] == "VALID" and diag_extra.get("expert_level_hit_rate") is not None
                 and (diag_extra.get("diagnostics") or {}).get("publication_quality") is True
                 and (diag_extra.get("diagnostics") or {}).get("trace_complete") is True)
        out["exp022"] = {"experiment": EXP_ID, "session_id": session["session_id"], **{k: x[k] for k in (
            "block", "model", "phase", "launch_index", "round", "position_in_round", "budget", "budget_bytes",
            "fraction_of_footprint", "cap_s", "launch_dir")}, "attempt": 1, "launch_utc": t_launch,
            "free_ram_before_launch": ram_before, "ram_warning": bool(ram_before is not None and ram_before < MIN_START_AVAILABLE),
            "exit_status": rc, "timed_out": timed_out, "elapsed_s": elapsed, **cls,
            "min_budget_warning": seg["min_budget_warning"], "abort_lines": seg["abort_lines"][:3],
            "provenance_problems": prov, "invariant_problems": inv, "harness_problems": harness,
            "h5_eligible": h5_ok, **diag_extra, "command": command(x), "launch_log": str(logf.relative_to(EXP))}
        _append_jsonl(rp, out)
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


def finalize(session: Dict[str, Any], sp: Path, llama_dir: Path) -> int:
    problems = finalize_fingerprints(session, llama_dir)
    session["status"] = "complete" if not problems else "INVALID"
    session["valid"] = not problems
    session["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if problems:
        session["invalid_reason"] = "; ".join(problems)
    _write_json(sp, session)
    print("block " + session["status"] + ("" if not problems else ": " + "; ".join(problems)))
    return 0 if not problems else 1


def cmd_finalize(block: str, llama_dir: Path) -> int:
    sp = _block_dir(block) / "session.json"
    if not sp.exists():
        print("no session.json for this block", file=sys.stderr)
        return 1
    s = json.loads(sp.read_text(encoding="utf-8"))
    if s.get("status") != "launches complete":
        print(f"block status {s.get('status')!r}: nothing to finalise", file=sys.stderr)
        return 1
    return finalize(s, sp, llama_dir)


def cmd_analyze() -> int:
    sessions, records = {}, []
    for b in BLOCK_ORDER:
        sp = _block_dir(b) / "session.json"
        if sp.exists():
            sessions[b] = json.loads(sp.read_text(encoding="utf-8"))
            records += _read_jsonl(_block_dir(b) / "records.jsonl")
    if not records:
        print("no records: EXP-022 has not been run", file=sys.stderr)
        return 1
    for r in records:
        fr = FIDELITY / r["exp022"]["block"] / f"launch-{r['exp022']['launch_index']:02d}" / "fidelity_runs.json"
        if fr.exists():
            r["exp022"].setdefault("diagnostics", {})["replay_fidelity"] = json.loads(fr.read_text(encoding="utf-8"))
    text, summary = analyze(records, sessions)
    (EXP / "results.csv").write_text(text, encoding="utf-8", newline="")
    _write_json(EXP / "processed" / "summary.json", summary)
    print(text)
    print(f"H-5: {summary['h5']['historical_prediction']} -> formal status {summary['formal_status']} "
          f"({summary['h5']['reason']})")
    return 0


# ================================================================== main ====

def print_dry_run() -> int:
    plan = build_plan()
    print(f"{EXP_ID} {PROTOCOL_VERSION}: {len(plan)} launches in blocks {' -> '.join(BLOCK_ORDER)}")
    for m in MODEL_ORDER:
        print(f"{m}: {MODELS[m]['role']}; F = {MODELS[m]['footprint_bytes']} B "
              f"({MODELS[m]['footprint_bytes'] / MIB:.1f} MiB); threads {MODELS[m]['gen_threads']} / batch {BATCH_THREADS}")
        for lab, t in budget_table(m).items():
            print(f"    {lab:>6} = {t['bytes']:>10} B  {t['fraction_of_footprint']:.3f} F  {t['shards']} shards  {t['role']}")
    for x in plan:
        print(f"  {x['block']:<9} {x['launch_index']:>2}  r{x['round']} p{x['position_in_round']}  {x['budget']:>6}  "
              f"cap {x['cap_s']} s  " + " ".join(x["argv"]))
    print("post-session replay diagnostics (not part of a block):")
    for x in plan:
        if x["phase"] == "diag":
            print("  " + " ".join(fidelity_command(x)))
    p = check_plan(plan)
    print("plan check: " + ("OK" if not p else "; ".join(p)))
    print(f"freeze: {freeze_status()[1]}")
    print("dry run: no subprocess started, no model read")
    return 0 if not p else 1


def cmd_check() -> int:
    problems = static_problems()
    ok, why = freeze_status()
    plan = build_plan()
    print(f"plan: {len(plan)} launches in blocks " + ", ".join(
        f"{b} ({sum(1 for x in plan if x['block'] == b)})" for b in BLOCK_ORDER))
    for m in MODEL_ORDER:
        print(f"{m}: F = {MODELS[m]['footprint_bytes']} B, whole layer {MODELS[m]['whole_layer_bytes']} B "
              f"(tracked source {MODELS[m]['footprint_source']})")
    print(f"H-5: {H5['model']} {H5['budget']} = {H5['budget_bytes']} B, falsified iff mean {H5['metric_field']} "
          f"> {H5['threshold']}; no positive support threshold")
    print(f"freeze: {why}")
    if problems:
        print("check: FAILED\n  - " + "\n  - ".join(problems))
        return 2
    if not ok:
        print("check: FAILED (freeze)")
        return 2
    print("check: OK (nothing written; no model, store or binary read)")
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
            print(json.dumps(r["checks"], indent=2, ensure_ascii=False))
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
