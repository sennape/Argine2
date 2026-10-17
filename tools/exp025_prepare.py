#!/usr/bin/env python3
"""
EXP-025 primary dataset preprocessor.

Constructs the frozen EXP-025 primary predictor dataset from the completed
EXP-022 30b-diag traces. This tool performs extraction, validation,
decode-only attribution and replica aggregation only. It does not fit,
evaluate or predict with the EXP-025 model.

Frozen protocol commit:
    26da7819f79a114743aa8f38645c71e70e84ed7d
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from argine.research.iotimeline import queue_depth
from argine.research.rawlog import parse as parse_rawlog
from argine.research.trace import Trace


FREEZE_COMMIT = "26da7819f79a114743aa8f38645c71e70e84ed7d"

EXPECTED_BUDGETS = ("1.5G", "1152M", "1024M", "768M", "512M")
EXPECTED_REPLICAS_PER_BUDGET = 2
EXPECTED_REQUESTS_PER_REPLICA = 3
EXPECTED_PREDICTED_N_PER_REQUEST = 128
EXPECTED_GENERATED_TOKENS = 384
EXPECTED_WORKLOAD_DECODE_STEPS = 381
WORKLOAD_SEGMENT = "natural-language"

REPLICA_FIELDS = (
    "label", "launch", "budget", "source_run_dir",
    "ms_per_token", "gen_tok_s", "generated_tokens",
    "workload_decode_steps", "bytes_missed_per_token",
    "reads_per_token", "nodes_per_token", "qd", "q_status",
    "q_exact_completion_share", "controller_ms_per_token",
    "decode_reads", "decode_bytes", "decode_nodes", "io_total",
    "io_req_unique_matches", "io_req_unmatched",
    "io_req_multiply_matched", "accesses_unassigned_to_node",
    "io_graph_unmapped", "trace_complete", "publication_quality",
    "violations",
)

# Predictor-facing CSV: keep every non-label field numeric because the current
# research CLI float-converts non-label columns.
RUN_FIELDS = (
    "label", "ms_per_token", "bytes_missed_per_token",
    "reads_per_token", "nodes_per_token", "qd",
    "controller_ms_per_token", "qd_upper_bound",
)


class Exp025Error(RuntimeError):
    pass


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_one_csv_row(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise Exp025Error(f"{path}: expected one data row, found {len(rows)}")
    return rows[0]


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in {"true", "1", "yes"}:
        return True
    if v in {"false", "0", "no"}:
        return False
    raise Exp025Error(f"cannot parse boolean value {value!r}")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise Exp025Error(f"{path}:{lineno}: invalid JSON: {e}") from e
            if not isinstance(obj, dict):
                raise Exp025Error(f"{path}:{lineno}: expected JSON object")
            out.append(obj)
    return out


def single_run_dir(launch_dir: Path) -> Path:
    candidates = sorted(
        p for p in launch_dir.iterdir()
        if p.is_dir() and p.name.startswith("run-")
    )
    if len(candidates) != 1:
        raise Exp025Error(
            f"{launch_dir}: expected one run-* directory, found "
            f"{[p.name for p in candidates]}"
        )
    return candidates[0]


def step_attr(step: Any, *names: str) -> Any:
    for name in names:
        if hasattr(step, name):
            return getattr(step, name)
    return None


def step_graph(step: Any) -> int:
    g = step_attr(step, "graph", "graph_id")
    if g is None:
        raise Exp025Error(
            "Trace Step has no graph/graph_id; refusing to infer graph IDs"
        )
    return int(g)


def step_phase(step: Any) -> str:
    phase = step_attr(step, "phase")
    if phase:
        return str(phase)
    n_tokens = step_attr(step, "n_tokens")
    if n_tokens is None:
        raise Exp025Error(
            "Trace Step has neither phase nor n_tokens; cannot classify decode"
        )
    return "decode" if int(n_tokens) == 1 else "prefill"


def step_segment(step: Any) -> str:
    seg = step_attr(step, "segment")
    if seg is not None:
        return str(seg)

    meta = step_attr(step, "meta")
    if meta is not None:
        if isinstance(meta, Mapping) and "segment" in meta:
            return str(meta["segment"])
        if hasattr(meta, "segment"):
            return str(getattr(meta, "segment"))

    raise Exp025Error(
        "Trace Step has no segment metadata; refusing to guess workload windows"
    )


def trace_graph_map(trace: Trace) -> Dict[int, Any]:
    out: Dict[int, Any] = {}
    for step in trace.steps:
        g = step_graph(step)
        if g in out:
            raise Exp025Error(f"duplicate Trace Step graph id {g}")
        out[g] = step
    return out


def node_for_access(nodes_sorted: Sequence[Any], starts: Sequence[int], t_ns: int) -> Any | None:
    # Exact attribution semantics from rawlog.node_attributed_outcomes().
    i = bisect.bisect_right(starts, t_ns) - 1
    if i < 0 or t_ns > nodes_sorted[i].ready_ns:
        return None
    return nodes_sorted[i]


def workload_response_metrics(run_dir: Path) -> Tuple[int, float, float]:
    responses = load_jsonl(run_dir / "responses.jsonl")
    workload = [
        r for r in responses
        if str(r.get("segment", r.get("workload", ""))) == WORKLOAD_SEGMENT
    ]

    if len(workload) != EXPECTED_REQUESTS_PER_REPLICA:
        raise Exp025Error(
            f"{run_dir}: expected {EXPECTED_REQUESTS_PER_REPLICA} workload "
            f"responses, found {len(workload)}"
        )

    predicted_n: List[int] = []
    predicted_ms: List[float] = []
    for r in workload:
        if r.get("error") not in (None, ""):
            raise Exp025Error(f"{run_dir}: response error={r.get('error')!r}")
        if "predicted_n" not in r or "predicted_ms" not in r:
            raise Exp025Error(f"{run_dir}: response missing predicted_n/predicted_ms")
        predicted_n.append(int(r["predicted_n"]))
        predicted_ms.append(float(r["predicted_ms"]))

    if any(n != EXPECTED_PREDICTED_N_PER_REQUEST for n in predicted_n):
        raise Exp025Error(
            f"{run_dir}: expected predicted_n={EXPECTED_PREDICTED_N_PER_REQUEST} "
            f"for every request, got {predicted_n}"
        )

    generated_tokens = sum(predicted_n)
    if generated_tokens != EXPECTED_GENERATED_TOKENS:
        raise Exp025Error(
            f"{run_dir}: generated_tokens={generated_tokens}, "
            f"expected {EXPECTED_GENERATED_TOKENS}"
        )

    total_ms = sum(predicted_ms)
    if total_ms <= 0:
        raise Exp025Error(f"{run_dir}: non-positive total predicted_ms")

    gen_tok_s = generated_tokens / (total_ms / 1000.0)
    ms_per_token = total_ms / generated_tokens
    return generated_tokens, gen_tok_s, ms_per_token


def process_replica(repo: Path, launch_dir: Path) -> Dict[str, Any]:
    launch_row = read_one_csv_row(launch_dir / "runs.csv")
    budget = str(launch_row["budget"]).strip()

    if budget not in EXPECTED_BUDGETS:
        raise Exp025Error(f"{launch_dir}: unexpected budget {budget!r}")

    violations = int(launch_row["violations"])
    trace_complete = parse_bool(launch_row["trace_complete"])
    publication_quality = parse_bool(launch_row["publication_quality"])

    if violations != 0:
        raise Exp025Error(f"{launch_dir}: violations={violations}, expected 0")
    if not trace_complete:
        raise Exp025Error(f"{launch_dir}: trace_complete is false")
    if not publication_quality:
        raise Exp025Error(f"{launch_dir}: publication_quality is false")

    run_dir = single_run_dir(launch_dir)
    generated_tokens, gen_tok_s_calc, ms_per_token = workload_response_metrics(run_dir)
    gen_tok_s_saved = float(launch_row["gen_tok_s"])

    if not math.isclose(gen_tok_s_calc, gen_tok_s_saved, rel_tol=2e-7, abs_tol=2e-7):
        raise Exp025Error(
            f"{launch_dir}: gen_tok_s mismatch: responses={gen_tok_s_calc:.10g}, "
            f"runs.csv={gen_tok_s_saved:.10g}"
        )

    trace = Trace.load(run_dir / "trace.jsonl.gz")
    graph_to_step = trace_graph_map(trace)

    workload_decode_graphs = {
        g for g, step in graph_to_step.items()
        if step_phase(step) == "decode" and step_segment(step) == WORKLOAD_SEGMENT
    }
    if len(workload_decode_graphs) != EXPECTED_WORKLOAD_DECODE_STEPS:
        raise Exp025Error(
            f"{launch_dir}: workload decode graphs={len(workload_decode_graphs)}, "
            f"expected {EXPECTED_WORKLOAD_DECODE_STEPS}"
        )

    log = parse_rawlog(run_dir / "raw.tsv", keep_accesses=True)
    nodes_sorted = sorted(log.nodes, key=lambda n: n.start_ns)
    starts = [n.start_ns for n in nodes_sorted]

    accesses_by_req: Dict[int, List[Any]] = defaultdict(list)
    for access in log.accesses:
        accesses_by_req[int(access.req)].append(access)

    io_total = len(log.io)
    io_unique = 0
    io_unmatched = 0
    io_multi = 0
    access_unassigned = 0
    io_graph_unmapped = 0
    decode_io: List[Any] = []

    for io_rec in log.io:
        matches = accesses_by_req.get(int(io_rec.req), [])
        if not matches:
            io_unmatched += 1
            continue
        if len(matches) != 1:
            io_multi += 1
            continue

        io_unique += 1
        access = matches[0]
        node = node_for_access(nodes_sorted, starts, int(access.t_ns))
        if node is None:
            access_unassigned += 1
            continue

        graph = int(node.graph)
        step = graph_to_step.get(graph)
        if step is None:
            io_graph_unmapped += 1
            continue

        if step_phase(step) == "decode" and step_segment(step) == WORKLOAD_SEGMENT:
            decode_io.append(io_rec)

    if io_unmatched or io_multi:
        raise Exp025Error(
            f"{launch_dir}: I/O req join failed: unmatched={io_unmatched}, "
            f"multiply_matched={io_multi}"
        )
    if access_unassigned:
        raise Exp025Error(
            f"{launch_dir}: {access_unassigned} I/O-matched accesses "
            "could not be attributed to a Node"
        )
    if io_graph_unmapped:
        raise Exp025Error(
            f"{launch_dir}: {io_graph_unmapped} I/O records mapped to Nodes "
            "whose graph has no Trace Step"
        )

    decode_nodes = [n for n in log.nodes if int(n.graph) in workload_decode_graphs]
    if not decode_io:
        raise Exp025Error(f"{launch_dir}: no workload decode I/O records")
    if not decode_nodes:
        raise Exp025Error(f"{launch_dir}: no workload decode Nodes")

    q = queue_depth(decode_io)
    if "q_time_weighted_mean_when_busy" not in q:
        raise Exp025Error(f"{launch_dir}: queue_depth has no busy mean: {q}")

    qd = float(q["q_time_weighted_mean_when_busy"])
    q_status = str(q["status"])
    q_exact_share = float(q["exact_completion_share"])

    decode_reads = len(decode_io)
    decode_bytes = sum(int(r.bytes) for r in decode_io)
    decode_node_count = len(decode_nodes)
    denom = float(generated_tokens)

    return {
        "label": f"{budget}-{launch_dir.name}",
        "launch": launch_dir.name,
        "budget": budget,
        "source_run_dir": str(run_dir.relative_to(repo)).replace("\\", "/"),
        "ms_per_token": ms_per_token,
        "gen_tok_s": gen_tok_s_calc,
        "generated_tokens": generated_tokens,
        "workload_decode_steps": len(workload_decode_graphs),
        "bytes_missed_per_token": decode_bytes / denom,
        "reads_per_token": decode_reads / denom,
        "nodes_per_token": decode_node_count / denom,
        "qd": qd,
        "q_status": q_status,
        "q_exact_completion_share": q_exact_share,
        "controller_ms_per_token": 0.0,
        "decode_reads": decode_reads,
        "decode_bytes": decode_bytes,
        "decode_nodes": decode_node_count,
        "io_total": io_total,
        "io_req_unique_matches": io_unique,
        "io_req_unmatched": io_unmatched,
        "io_req_multiply_matched": io_multi,
        "accesses_unassigned_to_node": access_unassigned,
        "io_graph_unmapped": io_graph_unmapped,
        "trace_complete": trace_complete,
        "publication_quality": publication_quality,
        "violations": violations,
    }


def aggregate(replicas: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    by_budget: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in replicas:
        by_budget[str(row["budget"])].append(row)

    counts = {budget: len(by_budget[budget]) for budget in EXPECTED_BUDGETS}
    if any(n != EXPECTED_REPLICAS_PER_BUDGET for n in counts.values()):
        raise Exp025Error(
            "primary dataset NOT EVALUABLE: every budget requires both scheduled "
            f"replicas; counts={counts}"
        )

    scalar_fields = (
        "ms_per_token", "bytes_missed_per_token", "reads_per_token",
        "nodes_per_token", "qd", "controller_ms_per_token",
    )

    runs: List[Dict[str, Any]] = []
    provenance: Dict[str, Any] = {}

    for budget in EXPECTED_BUDGETS:
        rows = sorted(by_budget[budget], key=lambda r: str(r["launch"]))
        out: Dict[str, Any] = {"label": budget}
        for field in scalar_fields:
            out[field] = mean(float(r[field]) for r in rows)

        q_status = "exact" if all(r["q_status"] == "exact" for r in rows) else "upper_bound"
        out["qd_upper_bound"] = 1.0 if q_status == "upper_bound" else 0.0
        runs.append(out)

        provenance[budget] = {
            "replicas": [r["launch"] for r in rows],
            "aggregation": "arithmetic_mean_of_replica_level_scalars",
            "q_status": q_status,
            "replica_q_status": [r["q_status"] for r in rows],
        }

    return runs, provenance


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def git_output(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=repo, text=True,
            encoding="utf-8", errors="replace",
        ).strip()
    except Exception:
        return ""


def build(repo: Path, *, write: bool) -> Dict[str, Any]:
    source = repo / "experiments" / "EXP-022-extreme-pressure" / "raw" / "30b-diag"
    out_dir = repo / "experiments" / "EXP-025-performance-model" / "processed"

    if not source.is_dir():
        raise Exp025Error(f"missing source directory: {source}")

    launch_dirs = sorted(
        p for p in source.iterdir()
        if p.is_dir() and p.name.startswith("launch-")
    )
    if len(launch_dirs) != 10:
        raise Exp025Error(f"expected 10 launch-* directories, found {len(launch_dirs)}")

    replicas: List[Dict[str, Any]] = []
    for launch_dir in launch_dirs:
        print(f"[EXP-025] processing {launch_dir.name} ...", flush=True)
        replicas.append(process_replica(repo, launch_dir))

    runs, configuration_provenance = aggregate(replicas)

    report = {
        "experiment": "EXP-025-performance-model",
        "freeze_commit": FREEZE_COMMIT,
        "source": "EXP-022-extreme-pressure/raw/30b-diag",
        "purpose": "preprocess_only_no_fit_no_prediction",
        "generated_tokens_per_replica": EXPECTED_GENERATED_TOKENS,
        "workload_decode_steps_per_replica": EXPECTED_WORKLOAD_DECODE_STEPS,
        "replica_aggregation": "arithmetic_mean_of_replica_level_scalars",
        "replicas": len(replicas),
        "configurations": len(runs),
        "budgets": list(EXPECTED_BUDGETS),
        "configuration_provenance": configuration_provenance,
        "working_tree_head_at_preprocess": git_output(repo, "rev-parse", "HEAD"),
    }

    if write:
        write_csv(out_dir / "replicas.csv", REPLICA_FIELDS, replicas)
        write_csv(out_dir / "runs.csv", RUN_FIELDS, runs)
        (out_dir / "preprocess_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"[EXP-025] wrote {out_dir / 'replicas.csv'}")
        print(f"[EXP-025] wrote {out_dir / 'runs.csv'}")
        print(f"[EXP-025] wrote {out_dir / 'preprocess_report.json'}")
    else:
        print("[EXP-025] validation-only mode: no files written")

    print("[EXP-025] primary preprocessing: OK")
    print(f"[EXP-025] replicas={len(replicas)} configurations={len(runs)}")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the frozen EXP-025 primary dataset from EXP-022 traces."
    )
    parser.add_argument(
        "--check", action="store_true",
        help="validate and derive everything but do not write processed files",
    )
    args = parser.parse_args(argv)

    try:
        build(repo_root(), write=not args.check)
    except Exp025Error as e:
        print(f"[EXP-025] ERROR: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
