"""`argine research trace`: instrumented end-to-end runs (Phase 1).

One invocation runs a versioned workload against llama-server with the expert
store and the research instrumentation active, for one or more expert budgets
and repetitions, in an interleaved order. Every run directory receives:

    raw.tsv               the runtime's raw research log (argine-raw v1)
    markers.jsonl         request start/end markers written by this harness
    responses.jsonl       per-request llama.cpp timings and generated-text provenance
                          (hashes and lengths, never the text: see textprov.py)
    trace.jsonl.gz        router trace (with workload segments)
    io.csv                one row per storage read (exact I/O timeline)
    nodes.csv             residency phase of every MoE node
    graph_timing.csv      per forward pass: span, residency, remainder
    evictions.csv         every eviction, with resident duration
    summary.json          I/O timeline (queue depth with exact/upper-bound status),
                          final store statistics derived from the trace, trace
                          completeness, runtime confirmation, budget check
    manifest.json         fingerprint (section 28), requested vs effective store and
                          read mode, machine state, validation metadata, and the
                          consolidated publication-quality check

Nothing here decides anything about residency: it observes.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import fingerprint, history, iotimeline, rawlog, textprov, workloads
from .io import write_csv, write_json
from .stats import interleaved_schedule
from .trace import TraceMeta

TRACE_LINE = "research trace attiva"
STORE_IN_USE_LINE = "expert store IN USO"
INSTRUMENTATION_ID = "argine2-research-v2"
# v1 lacks the shard/seq fields needed for event-level replay validation but is
# otherwise a complete record; both are accepted as confirmed instrumentation.
INSTRUMENTATION_IDS = ("argine2-research-v1", "argine2-research-v2")
SENTINEL_SEGMENT = "flush-sentinel"
# The C++ side writes its stdio buffer to the OS at most every 250 ms, at node
# boundaries; llama-server is terminated, not exited. A request issued more
# than this long after the last workload request flushes everything before it.
FLUSH_INTERVAL_S = 0.25


def available_ram() -> Optional[int]:
    """System-wide available RAM in bytes (psutil), or None."""
    try:
        import psutil  # type: ignore
        return int(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001
        return None


def expert_bytes_from_model(m: Any) -> Dict[int, int]:
    """Bytes the store needs to make one expert of each layer resident:
    gate + up + down slices, each rounded up to the 4096-byte alignment the
    store reads with (psx_slice_read_bytes)."""
    from .. import expert_store as es
    out: Dict[int, int] = {}
    n = int(getattr(m, "expert_count", 0) or 0)
    if n <= 0:
        return out
    for layer, kinds in es.expert_tensors(m).items():
        tot = 0
        for _kind, t in kinds.items():
            per = int(t.n_bytes) // n
            tot += -(-per // 4096) * 4096
        out[int(layer)] = tot
    return out


def _post_chat(port: int, messages: List[Dict[str, str]], n_predict: int, seed: int,
               cache_prompt: bool, timeout: float = 3600.0) -> Dict[str, Any]:
    import http.client
    body = json.dumps({"messages": messages, "max_tokens": n_predict, "temperature": 0.0,
                       "top_k": 1, "seed": seed, "cache_prompt": cache_prompt,
                       "stream": False}).encode("utf-8")
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("POST", "/v1/chat/completions", body=body,
                     headers={"Content-Type": "application/json", "Connection": "close"})
        resp = conn.getresponse()
        raw = resp.read()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {raw[:300]!r}")
        return json.loads(raw.decode("utf-8"))
    finally:
        conn.close()


def _marker(fh, event: str, **kw: Any) -> None:
    rec = {"event": event, "system_ns": time.time_ns(), **kw}
    fh.write(json.dumps(rec) + "\n")
    fh.flush()


def run_workload(port: int, wl: workloads.Workload, split: str, run_dir: Path,
                 seed: int = 42, warmup: bool = True) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(run_dir / "markers.jsonl", "w", encoding="utf-8") as mk, \
            open(run_dir / "responses.jsonl", "w", encoding="utf-8") as rs:
        if warmup:
            _marker(mk, "request_start", segment="warmup", prompt_id="warmup")
            try:
                _post_chat(port, [{"role": "user", "content": "Hi"}], 8, seed, False)
            except Exception as e:  # noqa: BLE001
                print(f"  warm-up failed: {e}", file=sys.stderr)
            _marker(mk, "request_end", segment="warmup", prompt_id="warmup")
        v2 = wl.history is not None
        plan_samples: List[Dict[str, Any]] = []
        for p in wl.split(split):
            turns = p.turns or [{"content": p.prompt, "segment": p.segment,
                                 "n_predict": p.n_predict}]
            messages: List[Dict[str, str]] = []
            sample_plan = {"sample_id": p.id, "turns_expected": len(turns), "turns": []}
            if p.turns:
                plan_samples.append(sample_plan)
            stopped = False
            for ti, turn in enumerate(turns):
                if stopped:
                    sample_plan["turns"].append({"turn_index": ti + 1,
                                                 "turn_history_status": history.SKIPPED})
                    continue
                seg = str(turn.get("segment", p.segment))
                npred = int(turn.get("n_predict", p.n_predict))
                messages.append({"role": "user", "content": str(turn["content"])})
                pid = f"{p.id}" + (f"#t{ti + 1}" if p.turns else "")
                _marker(mk, "request_start", segment=seg, prompt_id=pid)
                t0 = time.perf_counter()
                try:
                    obj = _post_chat(port, messages, npred, seed, cache_prompt=bool(p.turns))
                    err = None
                except Exception as e:  # noqa: BLE001
                    obj, err = {}, str(e)
                wall = time.perf_counter() - t0
                _marker(mk, "request_end", segment=seg, prompt_id=pid)
                tm = (obj.get("timings") if isinstance(obj, dict) else None) or {}
                prov = textprov.record(obj, err, tm.get("predicted_n"))
                if prov["text_provenance"] not in textprov.OK_STATES:
                    print(f"  TEXT PROVENANCE {prov['text_provenance']} for {pid}: "
                          f"{prov['text_provenance_detail']}", file=sys.stderr)
                hist_fields: Dict[str, Any] = {}
                if not v2:
                    # v1 history: `content` as returned (possibly ""), exactly as
                    # before the 2026-10-04 provenance fix; records unchanged.
                    messages.append({"role": "assistant", "content": textprov.history_content(obj)})
                elif p.turns:
                    # v2 history: non-empty final content only, never reasoning.
                    htext, hist_fields = history.turn(obj, prov)
                    if htext is not None:
                        messages.append({"role": "assistant", "content": htext})
                    else:
                        stopped = True
                        print(f"  CONVERSATION HISTORY {hist_fields['turn_history_status']} for "
                              f"{pid}: sample {p.id} invalid, remaining turns not sent",
                              file=sys.stderr)
                    ctx_tok = [tm.get(k) for k in ("cache_n", "prompt_n", "predicted_n")]
                    hist_fields = {"history_semantics": "v2", "sample_id": p.id,
                                   "turn_index": ti + 1, "turns_expected": len(turns),
                                   "followed_by_user_turn": ti + 1 < len(turns),
                                   **hist_fields,
                                   "context_tokens": (sum(ctx_tok) if all(isinstance(x, int)
                                                                          for x in ctx_tok) else None)}
                    sample_plan["turns"].append({"turn_index": ti + 1, "prompt_id": pid,
                                                 "turn_history_status":
                                                     hist_fields["turn_history_status"]})
                rec = {"prompt_id": pid, "workload": wl.name, "workload_sha256": wl.sha256,
                       "split": split, "segment": seg, "n_predict": npred, "wall_s": wall,
                       "error": err, **prov,
                       "prompt_n": tm.get("prompt_n"), "prompt_ms": tm.get("prompt_ms"),
                       "predicted_n": tm.get("predicted_n"), "predicted_ms": tm.get("predicted_ms"),
                       "predicted_per_second": tm.get("predicted_per_second"),
                       "prompt_per_second": tm.get("prompt_per_second"),
                       "cache_n": tm.get("cache_n"), **hist_fields}
                rs.write(json.dumps(rec) + "\n")
                out.append(rec)
    sem = history.semantics(wl.history, any(x.multi_turn for x in wl.split(split)))
    if sem is not None:
        write_json({"workload": wl.name, "workload_sha256": wl.sha256, "split": split,
                    "conversation_history_semantics": sem, "samples": plan_samples},
                   run_dir / history.FILE)
    return out


def flush_sentinel(port: int, run_dir: Path, seed: int = 42) -> Optional[str]:
    """One tiny request after the workload, more than FLUSH_INTERVAL_S later.

    Its first MoE node flushes every record of the workload to the file. Its
    own records may be lost at termination; they are labelled with the
    segment `flush-sentinel` and excluded from every statistic.
    """
    time.sleep(FLUSH_INTERVAL_S * 2)
    with open(run_dir / "markers.jsonl", "a", encoding="utf-8") as mk:
        _marker(mk, "request_start", segment=SENTINEL_SEGMENT, prompt_id=SENTINEL_SEGMENT)
        err = None
        try:
            _post_chat(port, [{"role": "user", "content": "."}], 2, seed, False)
        except Exception as e:  # noqa: BLE001
            err = str(e)
        _marker(mk, "request_end", segment=SENTINEL_SEGMENT, prompt_id=SENTINEL_SEGMENT)
    time.sleep(FLUSH_INTERVAL_S * 2)
    return err


def runtime_confirmation(log: rawlog.RawLog) -> Dict[str, Any]:
    """What the runtime itself declared, from structured C records."""
    c = log.config
    return {
        "source": "trace C records (structured)",
        "instrumentation": c.get("instrumentation"),
        "instrumentation_confirmed": c.get("instrumentation") in INSTRUMENTATION_IDS,
        "event_level_validation_possible": c.get("instrumentation") == INSTRUMENTATION_ID,
        "expert_store_path": c.get("store"),
        "budget_bytes": int(c["budget_bytes"]) if c.get("budget_bytes", "").isdigit() else None,
        "read_mode": c.get("read_mode"),
        "direct_io": c.get("direct_io"),
        "cache_shards": c.get("cache_shards"),
        "pretouch": c.get("pretouch"),
        "expert_store_serving": len(log.accesses) > 0 or len(log.routes) > 0,
    }


def trace_completeness(log: rawlog.RawLog, tr, markers: List[Dict[str, Any]],
                       responses: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Does the trace hold every forward pass the server reported?

    For each workload request: decode steps inside its [start, end] markers
    must equal predicted_n - 1 (the first generated token comes from the prompt
    step), and every step must cover all MoE layers. The flush sentinel and the
    warm-up are not checked.
    """
    n_layers = len(tr.layers)
    starts = {}
    for m in markers:
        if m.get("event") in ("request_start", "request_end") and "system_ns" in m:
            starts.setdefault(m.get("prompt_id"), {})[m["event"]] = \
                log.system_to_steady(int(m["system_ns"]))
    rows = []
    for r in responses or []:
        pid = r.get("prompt_id")
        w = starts.get(pid, {})
        if "request_start" not in w or "request_end" not in w or not r.get("predicted_n"):
            rows.append({"prompt_id": pid, "checked": False})
            continue
        st = [x for x in tr.steps if x.t_ns is not None
              and w["request_start"] <= x.t_ns <= w["request_end"]]
        dec = sum(1 for x in st if x.phase == "decode")
        partial = sum(1 for x in st if len(x.routes) < n_layers)
        expected = int(r["predicted_n"]) - 1
        rows.append({"prompt_id": pid, "checked": True, "decode_steps": dec,
                     "expected_decode_steps": expected, "partial_steps": partial,
                     "complete": dec == expected and partial == 0})
    checked = [x for x in rows if x.get("checked")]
    return {"requests": rows,
            "complete": bool(checked) and all(x["complete"] for x in checked),
            "method": "decode steps per request vs llama.cpp predicted_n - 1; all layers present",
            "scope": ("workload requests only: the warm-up and flush-sentinel requests are not "
                      "checked; termination-tail record loss is reported separately by "
                      "tools/validate_replay_fidelity.py (termination_tail_completeness)")}


EXCLUDED_SEGMENTS = ("warmup", SENTINEL_SEGMENT)


def excluded_windows(log: rawlog.RawLog, markers: List[Dict[str, Any]]) -> List[tuple]:
    """Steady-clock windows of the warm-up and flush-sentinel requests."""
    open_: Dict[str, int] = {}
    out = []
    for m in markers:
        if m.get("segment") not in EXCLUDED_SEGMENTS or "system_ns" not in m:
            continue
        t = log.system_to_steady(int(m["system_ns"]))
        if m.get("event") == "request_start":
            open_[m["segment"]] = t
        elif m.get("event") == "request_end" and m["segment"] in open_:
            out.append((open_.pop(m["segment"]), t))
    for seg, t in open_.items():  # started, never ended: exclude to the end
        out.append((t, float("inf")))
    return out


def _workload_totals(log: rawlog.RawLog, windows: List[tuple]) -> Dict[str, Any]:
    def keep(t: int) -> bool:
        return not any(a <= t <= b for a, b in windows)
    outcomes: Dict[str, int] = {}
    for a in log.accesses:
        if keep(a.t_ns):
            outcomes[a.outcome] = outcomes.get(a.outcome, 0) + 1
    ok = [r for r in log.io if r.ok and keep(r.issue_ns)]
    sem = rawlog.node_attributed_outcomes(log, keep)
    return {"excluded_segments": list(EXCLUDED_SEGMENTS),
            "excluded_windows_ns": [[a, b if b != float("inf") else None] for a, b in windows],
            "outcomes": outcomes, "reads_ok": len(ok),
            "reads_failed": sum(1 for r in log.io if not r.ok and keep(r.issue_ns)),
            "read_payload_bytes": sum(r.bytes for r in ok),
            "evictions": sum(1 for e in log.evictions if keep(e.t_ns)),
            "runtime_style_hit_rate": sem["runtime_style_hit_rate"],
            "expert_level_hit_rate": sem["gate_node_hit_rate"]}


def final_store_stats(log: rawlog.RawLog, markers: Optional[List[Dict[str, Any]]] = None
                      ) -> Dict[str, Any]:
    """Final Expert Store statistics derived from structured trace records.

    Not from stderr: the store's own report is printed every 5 s only with
    ARGINE_EXPERT_VERBOSE and lives in a bounded line buffer. The last S
    snapshot is the store's authoritative counter set at the last forward-pass
    boundary; the trace totals count every A/I/E record, including the final
    forward pass after that snapshot.
    """
    last = log.snapshots[-1] if log.snapshots else None
    outcomes: Dict[str, int] = {}
    for a in log.accesses:
        outcomes[a.outcome] = outcomes.get(a.outcome, 0) + 1
    ok = [r for r in log.io if r.ok]
    layers = sorted({a.layer for a in log.accesses})
    sem = rawlog.node_attributed_outcomes(log)
    return {
        "source": "research trace (structured S/A/I/E records)",
        "last_snapshot": None if last is None else {
            "graph": last.graph, "resident_bytes": last.resident, "budget_bytes": last.budget,
            "hits": last.hits, "misses": last.misses, "evictions": last.evictions,
            "bytes_read": last.bytes_read, "violations": last.violations},
        "peak_resident_bytes": max((x.resident for x in log.snapshots), default=None),
        "max_resident_minus_budget": max((x.resident - x.budget for x in log.snapshots),
                                         default=None),
        "trace_totals": {
            "scope": "every record of the server's lifetime, incl. warm-up and flush sentinel",
            "accesses": len(log.accesses), "outcomes": outcomes,
            "reads": len(log.io), "reads_ok": len(ok), "reads_failed": len(log.io) - len(ok),
            "read_payload_bytes": sum(r.bytes for r in ok),
            "evictions": len(log.evictions)},
        "layers_served": len(layers),
        "store_layers": int(log.config["store_layers"]) if log.config.get("store_layers", "").isdigit() else None,
        "runtime_style_hit_rate_all_records": sem["runtime_style_hit_rate"],
        "expert_level_hit_rate_all_records": sem["gate_node_hit_rate"],
        # Workload-only figures exclude the warm-up and the flush sentinel:
        # these are the numbers to cite.
        "workload": _workload_totals(log, excluded_windows(log, markers or [])),
        "runtime_style_hit_rate": None,
        "expert_level_hit_rate": None,
    }


def process_run(run_dir: Path, meta: TraceMeta,
                responses: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Raw log -> trace and tables. Usable on its own to re-derive a run."""
    raw = run_dir / "raw.tsv"
    if not raw.exists() and (run_dir / "raw.tsv.gz").exists():
        raw = run_dir / "raw.tsv.gz"
    log = rawlog.parse(raw)
    markers = rawlog.read_markers(run_dir / "markers.jsonl")
    if responses is None and (run_dir / "responses.jsonl").exists():
        responses = [json.loads(l) for l in (run_dir / "responses.jsonl")
                     .read_text(encoding="utf-8").splitlines() if l.strip()]
    plan = (json.loads((run_dir / history.FILE).read_text(encoding="utf-8"))
            if (run_dir / history.FILE).exists() else None)
    tr = rawlog.to_trace(log, meta=meta, markers=markers)
    tr.save(run_dir / "trace.jsonl.gz")
    write_csv(rawlog.io_rows(log), run_dir / "io.csv")
    write_csv(rawlog.node_rows(log), run_dir / "nodes.csv")
    write_csv(rawlog.graph_timing(log), run_dir / "graph_timing.csv")
    write_csv(rawlog.eviction_rows(log), run_dir / "evictions.csv")
    fs = final_store_stats(log, markers)
    fs["runtime_style_hit_rate"] = fs["workload"]["runtime_style_hit_rate"]
    fs["expert_level_hit_rate"] = fs["workload"]["expert_level_hit_rate"]
    last = log.snapshots[-1] if log.snapshots else None
    summary = {
        "raw": log.summary(),
        "runtime_confirmation": runtime_confirmation(log),
        "io": iotimeline.analyse(log.io),
        "access_outcomes": fs["trace_totals"]["outcomes"],
        "hit_rate_semantics": rawlog.node_attributed_outcomes(log),
        "final_store_stats": fs,
        "trace_completeness": trace_completeness(log, tr, markers, responses),
        "text_provenance": textprov.summarize(responses),
        "conversation_history": history.eligibility(responses, plan),
        "budget_invariant": {
            "violations": last.violations if last else None,
            "max_resident_over_budget": fs["max_resident_minus_budget"],
        },
        "trace": {"steps": len(tr.steps), "decode_tokens": tr.n_tokens("decode"),
                  "prefill_tokens": tr.n_tokens("prefill"), "segments": tr.segments()},
    }
    write_json(summary, run_dir / "summary.json")
    return summary


def resolve_expert_store(requested: Any, model_path: str) -> Dict[str, Any]:
    """The .psxs the CLI asks the runtime to open (same rule as
    argine.cli._default_store_path for `auto`), as an absolute path."""
    if not requested:
        return {"requested": None, "resolved": None}
    if requested == "auto":
        p = Path(model_path).with_suffix(".psxs")
    else:
        p = Path(str(requested)).expanduser()
    try:
        p = p.resolve()
    except Exception:  # noqa: BLE001
        p = p.absolute()
    return {"requested": str(requested), "resolved": str(p)}


def read_modes_agree(requested: Optional[str], effective: Optional[str]) -> bool:
    """`blocco` requests the runtime's default block size, reported as blocco:N."""
    if not requested or not effective:
        return False
    if requested == effective:
        return True
    return requested == "blocco" and effective.startswith("blocco:")


def _same_path(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b:
        return False
    norm = lambda x: str(x).replace("\\", "/").rstrip("/").lower()  # noqa: E731
    return norm(a) == norm(b)


def publication_check(fp: Dict[str, Any], run: Dict[str, Any]) -> Dict[str, Any]:
    """Consolidated check: fingerprint completeness + run-level provenance.

    Machine-state thresholds are warnings, never failures (historical runs are
    not invalidated by RAM levels)."""
    missing = list(fp.get("missing") or [])
    if not run["validation"]["instrumentation_confirmed"]:
        missing.append("instrumentation not confirmed by the runtime")
    if not run["read_mode"].get("effective"):
        missing.append("effective read mode (runtime confirmation)")
    st = run["expert_store"]
    if not st.get("effective"):
        missing.append("effective expert store path (runtime confirmation)")
    elif not st.get("fingerprint_matches_effective"):
        missing.append("expert store fingerprint does not match the store the runtime opened")
    if not run.get("final_store_stats") or not run["final_store_stats"].get("last_snapshot"):
        missing.append("final store statistics")
    if not run["trace_completeness"].get("complete"):
        missing.append("trace completeness (forward passes missing from the trace)")
    if (run.get("budget_violations") or 0) != 0:
        missing.append("hard budget violated")
    tp = run.get("text_provenance")
    if tp is not None and not tp.get("complete"):
        missing.append(f"generated-text provenance ({tp.get('verdict')}: {tp.get('counts')})")
    ch = run.get("conversation_history")
    if ch is not None and ch.get("applicable") and not ch.get("eligible"):
        missing.append(f"conversation history (multi-turn v2 not EXP-007 eligible: "
                       f"{ch.get('turn_status_counts')})")
    warnings = list(run["machine_state"].get("warnings") or [])
    return {"publication_quality": not missing, "missing": missing, "warnings": warnings}


def backend_provenance(requested: Optional[str], resolved: Optional[str],
                       vulkan_devices: Optional[List[str]]) -> Dict[str, Any]:
    """Requested vs resolved vs effective llama.cpp backend (EXP-019 audit, 2026-10-12).

    `resolved` is Argine's choice (hardware.resolve_backend, e.g. auto -> vulkan on an
    iGPU). Before 2026-10-12 it was only printed, never written to the manifest. When it
    is vulkan but the binary enumerates no Vulkan device (a CPU-only build), llama.cpp
    runs on the CPU and --n-gpu-layers has no effect.
    """
    devs = list(vulkan_devices) if vulkan_devices else []
    if resolved == "vulkan":
        eff, src = (("vulkan", "Vulkan devices enumerated by the binary") if devs else
                    ("cpu", "no Vulkan device enumerated by the binary: llama.cpp falls back "
                            "to the CPU"))
    elif resolved == "cpu":
        eff, src = "cpu", "resolved backend cpu"
    else:
        eff, src = None, "unknown"
    return {"requested": requested, "resolved": resolved,
            "vulkan_devices_enumerated": devs if resolved == "vulkan" else None,
            "effective": eff, "effective_source": src}


def cmd_trace(args: Any) -> int:
    from .. import cli as core
    from ..util import parse_size, fmt_bytes

    wl = workloads.load(args.workload, args.workload_version)
    if not getattr(args, "expert_store", None):
        core.error("research trace needs --expert-store: the instrumentation lives in the "
                   "expert store of the patched llama-server")
        return core.EXIT_USAGE
    if getattr(args, "expert_cache", None) and not getattr(args, "expert_budget", None):
        core.info("--expert-cache with --expert-store is read as the store budget "
                  "(--expert-budget)")
        args.expert_budget = args.expert_cache
        args.expert_cache = None
    budgets = [b.strip() for b in (args.budgets or args.expert_budget or "").split(",") if b.strip()]
    if not budgets:
        core.error("give the expert budget(s): --expert-budget 2G or --budgets 4G,3G,2G")
        return core.EXIT_USAGE
    for b in budgets:
        if not parse_size(b):
            core.error(f"not a size: {b}")
            return core.EXIT_USAGE
    min_start = parse_size(getattr(args, "min_start_available", None) or "0") or 0
    requested_read_mode = core._modo_lettura(getattr(args, "read_mode", None))
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    order = interleaved_schedule(budgets, max(1, args.repeat))
    core.info(f"research trace: workload {wl.name} [{args.split}] sha256 {wl.sha256[:12]}, "
              f"{len(order)} runs in order {' '.join(order)}")
    hsem = history.semantics(wl.history, any(x.multi_turn for x in wl.split(args.split)))
    if hsem:
        core.info("  conversation history " + hsem["version"] + ": assistant_source="
                  f"{hsem['assistant_source']}, reasoning_content_in_history="
                  f"{str(hsem['reasoning_content_in_history']).lower()}, empty_final_content_policy="
                  f"{hsem['empty_final_content_policy']}, provenance_text={hsem['provenance_text']}")
    model = core._load_model(args.model)
    store_req = resolve_expert_store(args.expert_store, args.model)
    meta_base = dict(model=Path(args.model).name, n_experts=int(model.expert_count or 0),
                     top_k=int(model.expert_used_count or 0),
                     expert_bytes=expert_bytes_from_model(model),
                     workload=f"{wl.name}@{wl.sha256[:12]}", split=args.split,
                     provenance="argine research trace (patch 0003 instrumentation)")
    results = []
    for i, blabel in enumerate(order):
        run_dir = root / f"run-{i:02d}-{blabel}"
        run_dir.mkdir(parents=True, exist_ok=True)
        args.expert_budget = blabel
        args.research_trace = str(run_dir / "raw.tsv")
        core.info(f"run {i + 1}/{len(order)}: budget {blabel} -> {run_dir}")

        warnings: List[str] = []
        avail_before = available_ram()
        if min_start and avail_before is not None and avail_before < min_start:
            msg = (f"available RAM before launch {fmt_bytes(avail_before)} < threshold "
                   f"{fmt_bytes(min_start)}: the starting machine state differs from a clean "
                   "one; close other applications for publication runs")
            core.warn(msg)
            warnings.append(msg)
        code, ctx = core._run_common(args)
        if ctx is None:
            return code
        proc, port = ctx["proc"], ctx["upstream"]
        lines = list(getattr(proc, "_psx_lines", [])) + proc.tail(400)
        stderr_trace = any(TRACE_LINE in ln for ln in lines)
        if not stderr_trace:
            proc.stop()
            core.error("llama-server did not confirm the research trace: this binary lacks "
                       "patch 0003. Point ARGINE_LLAMA_DIR at llama-research/.")
            return core.EXIT_RUNTIME
        t0 = time.time()
        responses = run_workload(port, wl, args.split, run_dir, seed=args.seed,
                                 warmup=not args.no_warmup)
        wall = time.time() - t0
        sentinel_error = flush_sentinel(port, run_dir, seed=args.seed)
        lines = list(getattr(proc, "_psx_lines", [])) + proc.tail(400)
        stderr_in_use = any(STORE_IN_USE_LINE in ln for ln in lines)
        stats = proc.monitor.stop() if proc.monitor else None
        store_status = proc.expert_store_status()
        proc.stop()
        time.sleep(2.0)  # let the OS flush the trace file and release the process memory
        avail_after = available_ram()

        meta = TraceMeta(**meta_base)
        summary = process_run(run_dir, meta, responses)
        rc = summary["runtime_confirmation"]
        effective_store = rc.get("expert_store_path")
        fp = fingerprint.collect(
            model=args.model, store=effective_store or store_req["resolved"],
            store_required=True,
            binary=str(ctx["rt"].binary("server")), workload_file=str(wl.path),
            runtime_flags={"expert_budget": blabel, "requested_read_mode": requested_read_mode,
                           "effective_read_mode": rc.get("read_mode"),
                           "threads": ctx["plan"].threads, "ctx": ctx["plan"].ctx,
                           "backend": backend_provenance(getattr(args, "backend", None),
                                                         ctx.get("backend"),
                                                         ctx.get("vulkan_devices")),
                           "llama_args": getattr(proc, "args", None)},
            full_checksum=args.full_checksum)
        binary_path = Path(str(ctx["rt"].binary("server")))
        build_info = binary_path.parent / "BUILD-INFO.json"
        validation = {
            "validation_run_id": (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                  + f"-{str((fp['binary'] or {}).get('digest', ''))[:8]}-run{i:02d}"),
            "platform": platform.platform(),
            "instrumentation_confirmed": bool(stderr_trace and rc["instrumentation_confirmed"]),
            "instrumentation_sources": {"stderr_line": stderr_trace,
                                        "trace_config": rc.get("instrumentation")},
            "expert_store_confirmed": bool(stderr_in_use or rc["expert_store_serving"]),
            "expert_store_sources": {"stderr_in_use_line": stderr_in_use,
                                     "trace_records": rc["expert_store_serving"]},
            "effective_read_mode": rc.get("read_mode"),
            "binary_sha256": (fp["binary"] or {}).get("digest"),
            "build_info_sha256": (hashlib.sha256(build_info.read_bytes()).hexdigest()
                                  if build_info.exists() else None),
            "trace_complete": summary["trace_completeness"]["complete"],
        }
        min_avail = stats.system_available_min if stats else None
        run = {
            "expert_store": {**store_req, "effective": effective_store,
                             "resolved_matches_effective": _same_path(store_req["resolved"], effective_store),
                             "fingerprint_matches_effective": _same_path(
                                 (fp.get("expert_store") or {}).get("path"), effective_store)
                             and bool((fp.get("expert_store") or {}).get("digest"))},
            "read_mode": {"requested": requested_read_mode, "effective": rc.get("read_mode"),
                          "effective_source": "trace C record read_mode",
                          "stderr_reported": (store_status or {}).get("lettura"),
                          "agree": read_modes_agree(requested_read_mode, rc.get("read_mode"))},
            "machine_state": {"available_before_launch": avail_before,
                              "available_min_during_run": min_avail,
                              "available_after_shutdown": avail_after,
                              "ram_total": (fp.get("hardware") or {}).get("ram_total"),
                              "start_threshold": min_start or None,
                              "below_start_threshold": bool(min_start and avail_before is not None
                                                            and avail_before < min_start),
                              "warnings": warnings},
            "final_store_stats": summary["final_store_stats"],
            "trace_completeness": summary["trace_completeness"],
            "text_provenance": summary["text_provenance"],
            "conversation_history": summary["conversation_history"],
            "budget_violations": summary["budget_invariant"]["violations"],
            "validation": validation,
        }
        check = publication_check(fp, run)
        if not run["read_mode"]["agree"]:
            core.warn(f"requested read mode {requested_read_mode}, runtime used "
                      f"{rc.get('read_mode')}: the effective mode is what was measured")
        manifest = {"run_index": i, "budget": blabel, "workload": wl.name,
                    "workload_sha256": wl.sha256, "workload_version": wl.version,
                    "conversation_history_semantics": history.semantics(
                        wl.history, any(x.multi_turn for x in wl.split(args.split))),
                    "split": args.split, "seed": args.seed,
                    "wall_s": wall, "flush_sentinel_error": sentinel_error,
                    **run, **check, "fingerprint": fp,
                    "store_status_stderr": {**(store_status or {}),
                                            "note": "informational only: bounded stderr line "
                                                    "buffer, not used for statistics"},
                    "resources": stats.to_dict() if stats else None,
                    "responses": len(responses),
                    "errors": [r["error"] for r in responses if r.get("error")]}
        write_json(manifest, run_dir / "manifest.json")
        _append_validation(binary_path.parent, validation, run_dir, fp, check)
        dec = [r for r in responses if r.get("predicted_per_second")]
        tps = (sum(r["predicted_n"] for r in dec) / (sum(r["predicted_ms"] for r in dec) / 1000.0)
               if dec else None)
        q = summary["io"]["queue_depth"]
        results.append({"run": i, "budget": blabel, "gen_tok_s": tps,
                        "violations": summary["budget_invariant"]["violations"],
                        "reads": summary["raw"]["io"],
                        "q_status": q.get("status"),
                        "exact_completion_share": q.get("exact_completion_share"),
                        "q_tw_mean_span": q.get("q_time_weighted_mean_over_span"),
                        "q_tw_mean_active": q.get("q_time_weighted_mean_when_busy"),
                        "expert_level_hit_rate": run["final_store_stats"]["expert_level_hit_rate"],
                        "runtime_style_hit_rate": run["final_store_stats"]["runtime_style_hit_rate"],
                        "available_before_launch": avail_before, "available_min": min_avail,
                        "trace_complete": validation["trace_complete"],
                        "publication_quality": check["publication_quality"]})
        core.ok(f"  {blabel}: {tps:.2f} tok/s" if tps else f"  {blabel}: no timings")
        ch = summary["conversation_history"]
        if ch.get("applicable"):
            (core.ok if ch.get("eligible") else core.warn)(
                f"  conversation history v2: {ch['verdict']} (turns: {ch['turn_status_counts']})")
        if check["missing"]:
            core.warn("  not publication-quality: " + "; ".join(check["missing"]))
        if i + 1 < len(order) and args.cooldown:
            time.sleep(args.cooldown)
    write_csv(results, root / "runs.csv")
    return core.EXIT_OK


def _append_validation(bin_dir: Path, validation: Dict[str, Any], run_dir: Path,
                       fp: Dict[str, Any], check: Dict[str, Any]) -> None:
    """Append-only run-level validation registry next to the binary.

    BUILD-INFO.json is historical build provenance and is never written here:
    runtime validation of a binary is recorded in VALIDATIONS.jsonl instead.
    """
    rec = {**validation, "run_dir": str(run_dir),
           "cpu": (fp.get("hardware") or {}).get("cpu"),
           "publication_quality": check["publication_quality"],
           "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    try:
        with open(bin_dir / "VALIDATIONS.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
    except OSError:
        pass
