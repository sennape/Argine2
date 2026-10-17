"""Runtime-vs-replay fidelity on recorded research runs (EXP-021).

    python tools/validate_replay_fidelity.py RUNS_DIR OUT_DIR [--seeds 3]

For each run directory (raw.tsv, markers.jsonl, trace.jsonl.gz):
* reconstruct: consistency of the runtime's own records (mapping-free);
* BEFORE: the comparison the earlier report made — `sharded-lru-16` (expert
  units, event-atomic, decode-only state and metric, warm-up counted) against
  the runtime's workload-only gate-node slice hit rate over ALL phases;
* AFTER: StoreReplay on the runtime's request stream, same metric definitions
  as the runtime, with the recorded shard map (format v2: exact) or a hash
  ensemble of `--seeds` maps (format v1: statistical);
* matched expert-unit counterfactual: the old policy under v2 semantics with
  the metric that matches it (expert-event hit, decode, workload-only).
Record integrity (format v2): every missing per-shard sequence number is
classified (replay-relevant or not), and every MoE node is checked for the
complete set of slice requests its router decision implies.

`--pin-model single-pool` reproduces the model used before the Windows v2
follow-up (every thread drops its pins at every node); the default
`per-thread` model is the runtime's (each thread drops its own pins when it
enters a node of the pool computing that graph).

Writes fidelity_runs.csv, fidelity_runs.json and per-run divergence files.

Verdicts for format-v2 runs (EXP-007 80B follow-up). Three questions are kept
apart instead of folded into one:
* full_trace_fidelity            strict equality over EVERY record of the
  server's lifetime (the original EXP-021 criterion). Kept as a diagnostic.
* workload_fidelity              strict equality over the measured workload
  AND everything before its last request (warm-up, between requests): no
  disagreement of any kind, identical misses / miss bytes / evictions over the
  workload, no replay-relevant sequence gap, no incomplete workload node.
* termination_tail_completeness  whether the records after the last workload
  request (flush sentinel, cut by process termination) are complete, and
  whether every disagreement there is explained by records the shard had
  already lost. An UNEXPLAINED tail disagreement fails closure: it would be a
  real replay error, wherever it happens.
exp021_closure = workload_fidelity PASS and no unexplained tail disagreement.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from argine.research import fidelity as F, rawlog  # noqa: E402
from argine.research.io import write_csv, write_json  # noqa: E402
from argine.research.policies import ShardedLRU  # noqa: E402
from argine.research.replay import replay  # noqa: E402
from argine.research.storereplay import HashShardMap, RecordedShardMap  # noqa: E402
from argine.research.trace import Trace  # noqa: E402


FULL_TRACE = ("reconstruct_inconsistencies == 0", "shards_with_replay_relevant_seq_gaps == 0",
              "incomplete_nodes_in_workload == 0", "event_disagreement_rate == 0",
              "rt_misses == rp_misses", "rt_miss_bytes == rp_miss_bytes",
              "rt_evictions == rp_evictions", "rp_budget_violations == 0", "rp_failures == 0")
WORKLOAD = ("reconstruct_inconsistencies == 0", "shards_with_replay_relevant_seq_gaps == 0",
            "incomplete_nodes_in_workload == 0",
            "wl_disagreements == 0", "wl_victim_disagreements == 0",
            "wl_skip_count_disagreements == 0",
            "pre_workload_disagreements == 0", "pre_workload_skip_count_disagreements == 0",
            "wl_rt_misses == wl_rp_misses", "wl_rt_miss_bytes == wl_rp_miss_bytes",
            "wl_rt_evictions == wl_rp_evictions",
            "rp_budget_violations == 0", "wl_rp_failures == 0", "pre_workload_rp_failures == 0")


def _verdict(conds, row):
    failed = [c for c in conds if not eval(c, {}, dict(row))]  # noqa: S307 (fixed strings)
    return "PASS" if not failed else "FAIL: " + "; ".join(failed)


def one(run: Path, seeds: int, pin_model: str = "per-thread") -> dict:
    t0 = time.time()
    log = rawlog.parse(run / "raw.tsv")
    st = F.load_runtime_stream(run, log)
    rc = F.reconstruct(log)
    tr = Trace.load(run / "trace.jsonl.gz")
    old = replay(tr, ShardedLRU(16), st.budget, phase="decode", semantics="v1-legacy")
    matched = replay(tr, ShardedLRU(16), st.budget, phase="decode")
    if st.version >= 2:
        maps = [RecordedShardMap(st.n_shards, st.mapping)]
    else:
        maps = [HashShardMap(st.n_shards, st.sizes, sd) for sd in range(seeds)]
    res = [F.compare(st, mp, max_divergences=5, pin_model=pin_model) for mp in maps]
    win = F._windows(log, rawlog.read_markers(run / "markers.jsonl"))
    gaps = F.seq_gaps(log, win)
    comp = F.node_completeness(log, win)
    tp = F.thread_pool_config(run)
    rt = res[0]["runtime_metrics"]

    def m(key):
        vals = [r["replay_metrics"][key] for r in res]
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    row = {"run": run.name, "budget_bytes": st.budget, "format": st.version,
           "shards": st.n_shards, "shard_budget_bytes": res[0]["shard_budgets"][0],
           "shard_map": maps[0].label if len(maps) == 1 else f"hash-model x{len(maps)} (statistical)",
           "reconstruct_inconsistencies": rc["total"],
           "pin_model": pin_model,
           "threads": tp.get("threads"), "threads_batch": tp.get("threads_batch"),
           "thread_pools": json.dumps(st.pools, sort_keys=True),
           "shards_with_seq_gaps": gaps.get("shards_with_seq_gaps"),
           "shards_with_replay_relevant_seq_gaps": gaps.get("shards_with_replay_relevant_seq_gaps"),
           "missing_seq_numbers": gaps.get("missing"),
           "incomplete_nodes": comp["incomplete"],
           "incomplete_nodes_in_workload": comp["incomplete_in_workload"],
           "before_runtime_hit": rt["slice_gate_all"],
           "before_replay_hit": old.hit_rate,
           "before_abs_delta_pp": 100 * (old.hit_rate - rt["slice_gate_all"]),
           "before_metric_note": "runtime: gate slices, all phases, workload; replay: expert units, decode only, warm-up counted",
           }
    for key in ("slice_gate_all", "slice_gate_decode", "slice_gate_prefill", "expert_event_decode",
                "slice_all_all"):
        mu, sd = m(key)
        row[f"rt_{key}"] = rt[key]
        row[f"rp_{key}"] = mu
        row[f"rp_{key}_sd"] = sd
        row[f"delta_pp_{key}"] = 100 * (mu - rt[key])
    row["matched_counterfactual_expert_event_decode"] = matched.hit_rate
    row["matched_counterfactual_delta_pp_vs_rt_expert_event"] = 100 * (matched.hit_rate - rt["expert_event_decode"])
    dis = [r["event_disagreement_rate"] for r in res]
    row["event_disagreement_rate"] = statistics.mean(dis)
    row["event_disagreement_rate_sd"] = statistics.stdev(dis) if len(dis) > 1 else 0.0
    row["rt_misses"] = res[0]["totals"]["rt_miss"]
    row["rp_misses"] = statistics.mean(r["totals"]["rp_miss"] for r in res)
    row["rt_miss_bytes"] = res[0]["totals"]["rt_miss_bytes"]
    row["rp_miss_bytes"] = statistics.mean(r["totals"]["rp_miss_bytes"] for r in res)
    row["rt_evictions"] = len(log.evictions)
    row["rp_evictions"] = statistics.mean(r["replay_evictions"] for r in res)
    row["rp_failures"] = max(r["replay_failures"] for r in res)
    row["rp_budget_violations"] = max(r["replay_budget_violations"] for r in res)
    row["outcome_disagreements"] = res[0]["outcome_disagreements"]
    row["victim_disagreements"] = res[0]["victim_disagreements"]
    row["skip_count_disagreements"] = res[0]["skip_count_disagreements"]
    row["rt_evictions_past_pinned"] = res[0]["runtime_evictions_past_pinned"]
    row["rp_evictions_past_pinned"] = res[0]["replay_evictions_past_pinned"]
    row["requests"] = res[0]["requests"]
    r0 = res[0]
    for reg_name, prefix in (("workload", "wl_"), ("pre_workload", "pre_workload_"), ("tail", "tail_")):
        for k, v in r0["regions"][reg_name].items():
            row[prefix + k] = v
    row["last_workload_request_index"] = r0["last_workload_request_index"]
    row["tail_disagreements_explained_by_record_loss"] = r0["tail_disagreements_explained_by_record_loss"]
    row["tail_disagreements_unexplained"] = r0["tail_disagreements_unexplained"]
    fd = res[0]["first_divergence"]
    row["first_divergence_index"] = fd["index"] if fd else None
    if st.version >= 2:
        row["full_trace_fidelity"] = _verdict(FULL_TRACE, row)
        row["workload_fidelity"] = _verdict(WORKLOAD, row)
        complete = (not row["missing_seq_numbers"] and not row["incomplete_nodes"]
                    and row["tail_disagreements"] == 0)
        row["termination_tail_completeness"] = (
            "COMPLETE" if complete else
            f"INCOMPLETE: {row['missing_seq_numbers']} lost records, {row['incomplete_nodes']} "
            f"incomplete node(s) ({row['incomplete_nodes_in_workload']} in workload), "
            f"{row['tail_disagreements']} tail disagreement(s): "
            f"{row['tail_disagreements_explained_by_record_loss']} explained by record loss, "
            f"{row['tail_disagreements_unexplained']} unexplained")
        ok = row["workload_fidelity"] == "PASS" and row["tail_disagreements_unexplained"] == 0
        row["exp021_closure"] = "PASS" if ok else (
            "FAIL: " + ("workload_fidelity " + row["workload_fidelity"]
                        if row["workload_fidelity"] != "PASS" else "unexplained tail disagreement"))
    else:
        row["full_trace_fidelity"] = row["workload_fidelity"] = "n/a (format v1)"
        row["termination_tail_completeness"] = "n/a (format v1)"
        row["exp021_closure"] = "n/a (format v1: shard map not recorded)"
    row["seconds"] = round(time.time() - t0, 1)
    return row, {"reconstruct": rc, "seq_gaps": gaps, "node_completeness": comp,
                 "pin_model": pin_model, "thread_pools": st.pools,
                 "divergences_seed0": res[0]["divergences"],
                 "first_divergence_seed0": fd}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("out")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--pin-model", default="per-thread", choices=["per-thread", "single-pool"])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in sorted(p for p in Path(a.runs).iterdir() if (p / "raw.tsv").exists()):
        row, detail = one(run, a.seeds, a.pin_model)
        rows.append(row)
        write_json(detail, out / f"divergence-{run.name}.json")
        print(f"{run.name}: rt {row['rt_slice_gate_all']:.4f} rp {row['rp_slice_gate_all']:.4f} "
              f"before-delta {row['before_abs_delta_pp']:+.2f}pp disagree {row['event_disagreement_rate']:.4f} "
              f"closure {row['exp021_closure']} | full-trace {row['full_trace_fidelity']} | "
              f"tail {row['termination_tail_completeness']}",
              flush=True)
        write_csv(rows, out / "fidelity_runs.csv")
    write_json(rows, out / "fidelity_runs.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
