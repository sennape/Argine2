"""EXP-027 — current-runtime generation thread sweep (orchestration + provenance).

    python tools/exp027_run.py plan                 # print the 24-launch plan (no model, no inference)
    python tools/exp027_run.py preflight            # --dry-run of the 8 configurations: checks the
                                                    #   llama-server command (threads, threads-batch,
                                                    #   gpu layers) without loading or running a model
    python tools/exp027_run.py run                  # execute the 24 launches (writes raw/)
    python tools/exp027_run.py run --resume         # continue an interrupted session
    python tools/exp027_run.py finalize             # (re)run post-run fingerprint finalisation only
    python tools/exp027_run.py analyze              # raw/records.jsonl -> results.csv + processed/summary.json

What this tool is
-----------------
Orchestration and provenance only. It never changes runtime behaviour: every
launch is the ordinary `python -m argine benchmark` (what `argine.bat
benchmark` runs) with a fixed, fully explicit argument list. The configuration
cannot be changed silently:

* the command for each launch is built from constants below and recorded;
* `--retries 0` (no automatic degradation), `--repeat 1`, `--backend cpu`;
* after every launch the benchmark record is validated against the intended
  configuration (threads, threads_batch = 12, backend cpu, gpu layers 0, store
  on, 4 GiB budget, read mode reported by the store = asincrono, one request,
  no errors, no retry). Any mismatch, non-zero exit, missing record or error
  stops the session with a non-zero exit status ("fail loudly").

`--threads-batch` is not a CLI option of `argine benchmark`: the planner sets it
to max(threads, min(logical, physical)) = 12 on the reference i5-1340P. The
tool does not override it; it verifies it twice: in `preflight` (from the
llama-server command printed by --dry-run) and after every launch (from the
plan stored in the record). On a machine where it is not 12, the tool refuses.

Replication unit
----------------
One complete benchmark launch (fresh process, empty Expert Store) with exactly
one measured request (`--repeat 1`). Requests inside one process share the
populated store and process state and are not independent replicates.

Raw data
--------
* raw/session.json   provenance of the session: full SHA-256 of the research
                     binary and BUILD-INFO, git state (commit recorded as
                     missing if the tree is not a git checkout — never
                     invented), machine identity, best-effort power scheme and
                     AC state, benchmark workload identity, the exact 24-launch
                     plan and every command line. Models and Expert Stores
                     (~107 GB) are NOT read in full before the measurements
                     (that would perturb file cache, NVMe and thermal state):
                     pre-run snapshot = canonical path, size, mtime, Argine2
                     fast fingerprint, plus a full SHA-256 only if already
                     known (checksum cache / earlier manifest, no file read).
                     Missing full hashes are computed after launch 24; the
                     session is valid only if size and mtime are unchanged.
* raw/records.jsonl  one line per launch attempt: the benchmark record copied
                     unchanged from the shared runs.jsonl, plus an `exp027`
                     block (session, model, round, order index, threads, batch
                     threads, budget, backend, read mode, launch time, free RAM
                     before launch, exit status, validation verdict, source
                     log identity).
* raw/logs/launch-NN.log   full stdout/stderr of each launch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
EXP_ID = "EXP-027-current-thread-sweep"
EXP = ROOT / "experiments" / EXP_ID
RAW = EXP / "raw"

MODELS = {
    "30b": {"name": "Qwen3-30B-A3B Q4_K_M",
            "path": r"C:\modelli\Qwen_Qwen3-30B-A3B-Q4_K_M.gguf"},
    "80b": {"name": "Qwen3-Next-80B-A3B-Instruct Q3_K_M",
            "path": r"C:\modelli\Qwen3-Next-80B-A3B-Instruct-Q3_K_M.gguf"},
}
THREADS = (4, 6, 8, 12)
BATCH_THREADS = 12
EXPERT_BUDGET = "4G"
EXPERT_BUDGET_BYTES = 4 * 1024 ** 3
READ_MODE = "asincrono"
BACKEND = "cpu"
N_PREDICT = 128
REPEAT = 1
RETRIES = 0
COOLDOWN_S = 60
# rotation within each model (one model completely before the other)
SCHEDULE = {
    "30b": [(1, [4, 6, 12, 8]), (2, [6, 8, 4, 12]), (3, [8, 12, 6, 4])],
    "80b": [(1, [12, 4, 8, 6]), (2, [4, 6, 12, 8]), (3, [6, 8, 4, 12])],
}
MODEL_ORDER = ("30b", "80b")
DEFAULT_LLAMA_DIR = ROOT / "llama-research-v2"

# result classification (see README "Closure criterion")
CV_INCONCLUSIVE = 0.05        # between-launch CV above this in any config -> INCONCLUSIVE
CLEAR_GAP_SD = 2.0            # gap best-second >= 2 x pooled SD and no range overlap -> CLEAR BEST


# ----------------------------------------------------------------- plan ----

def build_plan(models: Optional[Dict[str, Dict[str, str]]] = None) -> List[Dict[str, Any]]:
    models = models or MODELS
    plan: List[Dict[str, Any]] = []
    for key in MODEL_ORDER:
        for rnd, order in SCHEDULE[key]:
            for pos, t in enumerate(order, 1):
                plan.append({"order_index": len(plan) + 1, "model_key": key,
                             "model": models[key]["name"], "model_path": models[key]["path"],
                             "round": rnd, "position_in_round": pos, "threads": t,
                             "batch_threads": BATCH_THREADS, "expert_budget": EXPERT_BUDGET,
                             "backend": BACKEND, "read_mode": READ_MODE,
                             "repeat": REPEAT, "retries": RETRIES, "n_predict": N_PREDICT})
    return plan


def benchmark_args(item: Dict[str, Any], dry_run: bool = False) -> List[str]:
    """The exact `argine benchmark` argument list of one launch."""
    a = ["benchmark", item["model_path"],
         "--backend", BACKEND,
         "--expert-store",
         "--expert-budget", EXPERT_BUDGET,
         "--read-mode", READ_MODE,
         "--threads", str(item["threads"]),
         "--repeat", str(REPEAT),
         "--n-predict", str(N_PREDICT),
         "--retries", str(RETRIES),
         "--expert-stats"]
    if dry_run:
        a.append("--dry-run")
    return a


def command(item: Dict[str, Any], dry_run: bool = False) -> List[str]:
    return [sys.executable, "-m", "argine"] + benchmark_args(item, dry_run)


def check_plan(plan: List[Dict[str, Any]]) -> List[str]:
    """Structural checks of the plan; returns the list of problems (empty = ok)."""
    bad: List[str] = []
    if len(plan) != 24:
        bad.append(f"{len(plan)} launches, expected 24")
    counts: Dict[Tuple[str, int], int] = {}
    for it in plan:
        counts[(it["model_key"], it["threads"])] = counts.get((it["model_key"], it["threads"]), 0) + 1
        a = benchmark_args(it)
        for flag, val in (("--repeat", "1"), ("--retries", "0"), ("--backend", "cpu"),
                          ("--expert-budget", "4G"), ("--read-mode", READ_MODE),
                          ("--threads", str(it["threads"]))):
            if flag not in a or a[a.index(flag) + 1] != val:
                bad.append(f"launch {it['order_index']}: {flag} != {val}")
        if "--expert-store" not in a:
            bad.append(f"launch {it['order_index']}: --expert-store missing")
        if it["batch_threads"] != 12:
            bad.append(f"launch {it['order_index']}: batch threads {it['batch_threads']}")
    for key in MODEL_ORDER:
        for t in THREADS:
            if counts.get((key, t)) != 3:
                bad.append(f"{key} threads={t}: {counts.get((key, t), 0)} launches, expected 3")
    keys = [it["model_key"] for it in plan]
    if keys != ["30b"] * 12 + ["80b"] * 12:
        bad.append("models are not run in two consecutive blocks (30B then 80B)")
    return bad


def print_plan(plan: List[Dict[str, Any]]) -> None:
    print(f"{EXP_ID}: {len(plan)} launches, {COOLDOWN_S}s wait between launches")
    for it in plan:
        print(f"  {it['order_index']:>2}  {it['model_key']}  round {it['round']}  "
              f"threads {it['threads']:>2}  batch {it['batch_threads']}  :: "
              + " ".join(benchmark_args(it)))
    problems = check_plan(plan)
    print("plan check: " + ("OK" if not problems else "; ".join(problems)))


# ------------------------------------------------------------ provenance ----

def benchmark_workload() -> Dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from argine import bench  # noqa: E402
    p = bench.DEFAULT_PROMPT
    return {"prompt_source": "argine.bench.DEFAULT_PROMPT (unchanged)",
            "prompt_sha256": hashlib.sha256(p.encode("utf-8")).hexdigest(),
            "prompt_chars": len(p), "n_predict": N_PREDICT,
            "temperature": 0.0, "cache_prompt": False,
            "warmup": "argine.bench default (8-token request before the measured one)"}


LARGE_FILE_POLICY = (
    "no full read of model/store files before or during the 24 launches: pre-run snapshot "
    "(canonical path, size, mtime, Argine2 fast fingerprint) plus any full SHA-256 already "
    "known from the checksum cache or earlier manifests; missing full hashes are computed "
    "after the last launch and the session is valid only if size and mtime did not change")


def _stat_snapshot(path: Path) -> Dict[str, Any]:
    st = path.stat()
    return {"path": str(path), "canonical_path": str(path.resolve()), "bytes": st.st_size,
            "mtime": st.st_mtime, "mtime_ns": st.st_mtime_ns,
            "mtime_iso": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds")}


def known_full_hash(path: Path, size: int, mtime: float) -> Optional[Dict[str, Any]]:
    """A full SHA-256 already computed for exactly this file, found WITHOUT
    reading the file: (1) Argine2's checksum cache, keyed on canonical path,
    size, mtime and method; (2) a run manifest under experiments/ whose
    fingerprint used method sha256 for the same path and size and that was
    written after the file's mtime. None if neither exists."""
    sys.path.insert(0, str(ROOT))
    from argine.research import fingerprint  # noqa: E402
    key = f"{path.resolve()}|{size}|{int(mtime)}|sha256"
    try:
        cache = json.loads(fingerprint._cache_path().read_text(encoding="utf-8"))  # noqa: SLF001
    except Exception:  # noqa: BLE001
        cache = {}
    if key in cache:
        return {"digest": cache[key], "method": "sha256",
                "source": f"Argine2 checksum cache ({fingerprint._cache_path()}), key {key}"}  # noqa: SLF001
    for m in sorted((ROOT / "experiments").glob("EXP-*/raw/*/run-*/manifest.json")):
        try:
            if m.stat().st_mtime < mtime:
                continue
            fp = json.loads(m.read_text(encoding="utf-8")).get("fingerprint") or {}
        except Exception:  # noqa: BLE001
            continue
        for k in ("model", "expert_store"):
            e = fp.get(k) or {}
            if (e.get("method") == "sha256" and e.get("digest") and e.get("bytes") == size
                    and str(e.get("path", "")).lower() == str(path).lower()):
                return {"digest": e["digest"], "method": "sha256",
                        "source": f"{m.relative_to(ROOT)} fingerprint.{k} (written after the file's mtime)"}
    return None


def power_state() -> Dict[str, Any]:
    """Best-effort, optional: active Windows power scheme and AC/battery state."""
    out: Dict[str, Any] = {}
    if os.name == "nt":
        try:
            r = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True,
                               timeout=10, errors="replace")
            out["power_scheme"] = (r.stdout or "").strip() or None
        except Exception as e:  # noqa: BLE001
            out["power_scheme_error"] = str(e)[:200]
    try:
        import psutil  # type: ignore
        b = psutil.sensors_battery()
        out["ac_power"] = (bool(b.power_plugged) if b is not None and b.power_plugged is not None
                           else None)
        out["battery_percent"] = getattr(b, "percent", None) if b is not None else None
    except Exception as e:  # noqa: BLE001
        out["ac_power_error"] = str(e)[:200]
    out["note"] = "best effort; not mandatory"
    return out


def session_provenance(llama_dir: Path, models: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    """Pre-run provenance. Reads the research binary fully (small); for the
    models and Expert Stores it only stats the files, uses the Argine2 fast
    fingerprint (first and last 16 MiB, cached) and reuses an already-known full
    SHA-256 if one exists. It never reads a model or store file completely."""
    sys.path.insert(0, str(ROOT))
    from argine.research import fingerprint  # noqa: E402
    out: Dict[str, Any] = {
        "argine2": fingerprint.git_state(ROOT),
        "binary": fingerprint.binary_info(llama_dir / "llama-server.exe"
                                          if (llama_dir / "llama-server.exe").exists()
                                          else llama_dir / "llama-server"),
        "hardware": fingerprint.machine_identity(),
        "power": power_state(),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "models": {}, "workload": benchmark_workload(),
        "large_file_policy": LARGE_FILE_POLICY,
    }
    for key in MODEL_ORDER:
        mp = Path(models[key]["path"])
        entry: Dict[str, Any] = {"name": models[key]["name"]}
        for role, path in (("model", mp), ("expert_store", mp.with_suffix(".psxs"))):
            snap = _stat_snapshot(path)
            known = known_full_hash(path, snap["bytes"], snap["mtime"])
            entry[role] = {"pre_run": snap,
                           "fast": fingerprint.file_checksum(path, full=False),
                           "full": known,
                           "full_status": "reused (no file read)" if known else "pending post-run"}
        out["models"][key] = entry
    return out


def finalize_fingerprints(session: Dict[str, Any], hasher=None) -> List[str]:
    """After the last launch: verify size/mtime against the pre-run snapshot,
    compute any pending full SHA-256, and verify again after hashing.

    Returns the list of problems (empty = all files unchanged and fully
    fingerprinted). Entries already finalised are not recomputed."""
    if hasher is None:
        sys.path.insert(0, str(ROOT))
        from argine.research import fingerprint  # noqa: E402
        hasher = lambda p: fingerprint.file_checksum(p, full=True)  # noqa: E731
    problems: List[str] = []
    for key, entry in session["provenance"]["models"].items():
        for role in ("model", "expert_store"):
            e = entry[role]
            pre = e["pre_run"]
            path = Path(pre["path"])

            def changed() -> Optional[str]:
                if not path.exists():
                    return "file missing"
                st = path.stat()
                if st.st_size != pre["bytes"] or st.st_mtime_ns != pre["mtime_ns"]:
                    return (f"size/mtime changed (pre {pre['bytes']}/{pre['mtime_ns']}, "
                            f"now {st.st_size}/{st.st_mtime_ns})")
                return None

            why = changed()
            if why is None and (e.get("full") is None or not e["full"].get("digest")):
                h = hasher(path)
                why = changed()
                if why is None:
                    e["full"] = {"digest": h.get("digest"), "method": h.get("method"),
                                 "source": "computed after the 24 launches"}
                    e["full_status"] = "computed post-run"
            e["post_run_check"] = {"checked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                   "unchanged": why is None, "problem": why}
            if why:
                problems.append(f"{key} {role} {path}: {why}")
            elif e["full_status"].startswith("reused"):
                e["full_status"] = "reused (no file read); size/mtime verified post-run"
    session["fingerprints_final"] = not problems
    session["valid"] = not problems
    if problems:
        session["valid"] = False
        session["invalid_reason"] = "model or Expert Store file changed during the session: " + "; ".join(problems)
    return problems


def free_ram() -> Optional[int]:
    try:
        import psutil  # type: ignore
        return int(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001
        return None


def shared_log() -> Path:
    sys.path.insert(0, str(ROOT))
    from argine import telemetry  # noqa: E402
    return telemetry.log_path()


def new_records(log: Path, offset: int) -> List[Tuple[int, Dict[str, Any]]]:
    """Records appended to the shared log after byte `offset`, with their offsets."""
    out: List[Tuple[int, Dict[str, Any]]] = []
    if not log.exists():
        return out
    with open(log, "rb") as fh:
        fh.seek(offset)
        pos = offset
        for raw in fh:
            line = raw.decode("utf-8", "replace").strip()
            if line:
                try:
                    out.append((pos, json.loads(line)))
                except json.JSONDecodeError:
                    pass
            pos += len(raw)
    return out


def validate(rec: Optional[Dict[str, Any]], item: Dict[str, Any], exit_code: int,
             binary_sha8: Optional[str]) -> List[str]:
    """Everything that must hold for a launch to count. Empty list = valid."""
    bad: List[str] = []
    if exit_code != 0:
        bad.append(f"exit status {exit_code}")
    if rec is None:
        bad.append("no benchmark record appended to the shared log")
        return bad
    plan = rec.get("plan") or {}
    rt = rec.get("runtime") or {}
    met = rec.get("metrics") or {}
    checks = [
        (plan.get("threads") == item["threads"], f"plan.threads {plan.get('threads')} != {item['threads']}"),
        (plan.get("threads_batch") == BATCH_THREADS, f"plan.threads_batch {plan.get('threads_batch')} != {BATCH_THREADS}"),
        (plan.get("backend") == BACKEND and rt.get("backend") == BACKEND,
         f"backend plan={plan.get('backend')} runtime={rt.get('backend')} != cpu"),
        ((plan.get("gpu_layers") or 0) == 0, f"gpu_layers {plan.get('gpu_layers')} != 0"),
        (plan.get("expert_store") is True, "expert store not enabled in the plan"),
        (plan.get("expert_cache_budget") == EXPERT_BUDGET_BYTES,
         f"expert budget {plan.get('expert_cache_budget')} != {EXPERT_BUDGET_BYTES}"),
        (str(rec.get("modo_lettura") or "").startswith(READ_MODE),
         f"read mode reported by the store: {rec.get('modo_lettura')!r} (expected {READ_MODE})"),
        (rec.get("repeat") == REPEAT and len(met.get("runs") or []) == REPEAT,
         f"repeat {rec.get('repeat')} / runs {len(met.get('runs') or [])} != 1"),
        (rec.get("n_predict") == N_PREDICT, f"n_predict {rec.get('n_predict')} != {N_PREDICT}"),
        (not rec.get("attempts"), f"server start needed retries: {rec.get('attempts')}"),
        (not met.get("errors"), f"benchmark errors: {met.get('errors')}"),
        (bool((met.get("runs") or [{}])[0].get("gen_tok_s")), "no generation tok/s"),
    ]
    bad += [msg for ok, msg in checks if not ok]
    if binary_sha8 and binary_sha8 not in str(rt.get("build") or ""):
        bad.append(f"runtime build {rt.get('build')!r} does not carry binary sha {binary_sha8}")
    return bad


# ------------------------------------------------------------------- run ----

def _write_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _env(llama_dir: Path) -> Dict[str, str]:
    env = dict(os.environ)
    cur = env.get("ARGINE_LLAMA_DIR")
    if cur and Path(cur).resolve() != llama_dir.resolve():
        raise SystemExit(f"ARGINE_LLAMA_DIR={cur} differs from the requested research runtime "
                         f"{llama_dir}: refusing to guess which binary to measure")
    env["ARGINE_LLAMA_DIR"] = str(llama_dir)
    env.pop("ARGINE_RESEARCH_TRACE", None)   # no tracing in this experiment
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def preflight(plan: List[Dict[str, Any]], llama_dir: Path) -> List[Dict[str, Any]]:
    """--dry-run of each distinct configuration: no model load, no inference."""
    env = _env(llama_dir)
    seen, out = set(), []
    for it in plan:
        key = (it["model_key"], it["threads"])
        if key in seen:
            continue
        seen.add(key)
        r = subprocess.run(command(it, dry_run=True), cwd=str(ROOT), env=env,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        txt = (r.stdout or "") + (r.stderr or "")
        t = re.search(r"--threads\s+(\d+)", txt)
        tb = re.search(r"--threads-batch\s+(\d+)", txt)
        ngl = re.search(r"--n-gpu-layers\s+(\d+)", txt)
        res = {"model_key": it["model_key"], "threads": it["threads"], "exit": r.returncode,
               "llama_threads": int(t.group(1)) if t else None,
               "llama_threads_batch": int(tb.group(1)) if tb else None,
               "llama_gpu_layers": int(ngl.group(1)) if ngl else None}
        res["ok"] = (r.returncode == 0 and res["llama_threads"] == it["threads"]
                     and res["llama_threads_batch"] == BATCH_THREADS
                     and (res["llama_gpu_layers"] or 0) == 0)
        if not res["ok"]:
            res["output_tail"] = txt[-2000:]
        out.append(res)
        print(f"  preflight {it['model_key']} threads {it['threads']:>2}: "
              f"llama --threads {res['llama_threads']} --threads-batch {res['llama_threads_batch']} "
              f"--n-gpu-layers {res['llama_gpu_layers']}  -> {'OK' if res['ok'] else 'FAIL'}")
    return out


def run(args: argparse.Namespace) -> int:
    plan = build_plan()
    problems = check_plan(plan)
    if problems:
        print("plan check failed: " + "; ".join(problems), file=sys.stderr)
        return 2
    llama_dir = Path(args.llama_dir)
    for key in MODEL_ORDER:
        mp = Path(MODELS[key]["path"])
        if not mp.exists() or not mp.with_suffix(".psxs").exists():
            print(f"missing model or expert store: {mp} / {mp.with_suffix('.psxs')}", file=sys.stderr)
            return 2
    records_path, session_path = RAW / "records.jsonl", RAW / "session.json"
    done: Dict[int, Dict[str, Any]] = {}
    if args.resume:
        if not session_path.exists():
            print("--resume: no raw/session.json", file=sys.stderr)
            return 2
        session = json.loads(session_path.read_text(encoding="utf-8"))
        for r in _read_jsonl(records_path):
            e = r.get("exp027") or {}
            if e.get("session_id") == session["session_id"] and e.get("valid"):
                done[int(e["order_index"])] = r
        session.setdefault("resumes", []).append(datetime.now(timezone.utc).isoformat(timespec="seconds"))
    else:
        if _read_jsonl(records_path):
            print("raw/records.jsonl already has records: use --resume, or move them aside "
                  "(they are never overwritten)", file=sys.stderr)
            return 2
        print("preflight (dry-run, no inference):")
        pf = preflight(plan, llama_dir)
        if not all(p["ok"] for p in pf):
            print("preflight failed: configuration differs from the protocol; nothing was run",
                  file=sys.stderr)
            return 2
        print("pre-run provenance: full SHA-256 of the research binary; models and Expert "
              "Stores are only stat-ed and fast-fingerprinted (full hashes after the launches)")
        session = {"experiment": EXP_ID,
                   "session_id": datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8],
                   "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "protocol": {"threads": list(THREADS), "batch_threads": BATCH_THREADS,
                                "expert_budget": EXPERT_BUDGET, "read_mode": READ_MODE,
                                "backend": BACKEND, "n_predict": N_PREDICT, "repeat": REPEAT,
                                "retries": RETRIES, "cooldown_s": COOLDOWN_S,
                                "expert_stats": True, "research_trace": False,
                                "replication_unit": "one complete benchmark launch"},
                   "llama_dir": str(llama_dir),
                   "plan": [{**it, "command": command(it)} for it in plan],
                   "preflight": pf,
                   "provenance": session_provenance(llama_dir, MODELS),
                   "shared_log": str(shared_log())}
        RAW.mkdir(parents=True, exist_ok=True)
    session_path.write_text(json.dumps(session, indent=2, ensure_ascii=False), encoding="utf-8")
    bsha = ((session["provenance"].get("binary") or {}).get("digest") or "")[:8] or None
    env = _env(llama_dir)
    log = shared_log()
    todo = [it for it in plan if it["order_index"] not in done]
    if not todo and session.get("fingerprints_final"):
        print("session already complete and finalised; nothing to do")
        return 0
    for n, it in enumerate(todo):
        if n:
            print(f"  wait {COOLDOWN_S}s")
            time.sleep(COOLDOWN_S)
        attempt = 1 + sum(1 for r in _read_jsonl(records_path)
                          if (r.get("exp027") or {}).get("session_id") == session["session_id"]
                          and int((r.get("exp027") or {}).get("order_index", 0)) == it["order_index"])
        offset = log.stat().st_size if log.exists() else 0
        ram_before = free_ram()
        t_launch = datetime.now(timezone.utc).isoformat(timespec="seconds")
        print(f"[{it['order_index']:>2}/24] {it['model_key']} round {it['round']} threads {it['threads']}")
        logf = RAW / "logs" / f"launch-{it['order_index']:02d}-a{attempt}.log"
        logf.parent.mkdir(parents=True, exist_ok=True)
        with open(logf, "w", encoding="utf-8") as fh:
            proc = subprocess.run(command(it), cwd=str(ROOT), env=env, stdout=fh,
                                  stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                  errors="replace")
        appended = new_records(log, offset)
        bench = [(pos, r) for pos, r in appended if r.get("event") == "benchmark"
                 and str((r.get("model") or {}).get("path")) == it["model_path"]]
        other = [r.get("event") for _, r in appended if r.get("event") != "benchmark"]
        rec = bench[0][1] if len(bench) == 1 else None
        problems = validate(rec, it, proc.returncode, bsha)
        if len(bench) > 1:
            problems.append(f"{len(bench)} benchmark records appended (expected 1)")
        if other:
            problems.append(f"other events appended: {other}")
        out = dict(rec or {})
        out["exp027"] = {
            "experiment": EXP_ID, "session_id": session["session_id"],
            "order_index": it["order_index"], "attempt": attempt,
            "model_key": it["model_key"], "model": it["model"], "round": it["round"],
            "position_in_round": it["position_in_round"], "threads": it["threads"],
            "batch_threads": BATCH_THREADS, "expert_budget": EXPERT_BUDGET,
            "backend": BACKEND, "read_mode": READ_MODE,
            "launch_utc": t_launch, "free_ram_before_launch": ram_before,
            "exit_status": proc.returncode, "valid": not problems, "problems": problems,
            "command": command(it), "launch_log": str(logf.relative_to(EXP)),
            "source": ({"shared_log": str(log), "byte_offset": bench[0][0],
                        "ts": bench[0][1].get("ts"), "ts_iso": bench[0][1].get("ts_iso")}
                       if len(bench) == 1 else None),
        }
        _write_jsonl(records_path, out)
        if problems:
            session["status"] = f"stopped at launch {it['order_index']}"
            session_path.write_text(json.dumps(session, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"LAUNCH {it['order_index']} INVALID — session stopped:\n  - "
                  + "\n  - ".join(problems), file=sys.stderr)
            print("Inspect the launch log; continue later with: python tools/exp027_run.py run --resume",
                  file=sys.stderr)
            return 1
        g = out["metrics"]["runs"][0].get("gen_tok_s")
        print(f"      gen {g} tok/s, valid")
    session["status"] = "launches complete"
    session["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    session_path.write_text(json.dumps(session, indent=2, ensure_ascii=False), encoding="utf-8")
    return finalize(session, session_path)


def finalize(session: Dict[str, Any], session_path: Path) -> int:
    """Post-run fingerprint finalisation (idempotent: finalised entries are kept)."""
    print("all 24 launches valid; finalising fingerprints (full SHA-256 of models and "
          "Expert Stores not already known; this reads ~107 GB once, after the measurements)")
    problems = finalize_fingerprints(session)
    session["status"] = "complete" if not problems else "invalid"
    session_path.write_text(json.dumps(session, indent=2, ensure_ascii=False), encoding="utf-8")
    if problems:
        print("SESSION INVALID — a model or Expert Store file changed:\n  - "
              + "\n  - ".join(problems), file=sys.stderr)
        return 1
    print("session complete with full fingerprints: run `python tools/exp027_run.py analyze`")
    return 0


def cmd_finalize(_args: argparse.Namespace) -> int:
    session_path = RAW / "session.json"
    if not session_path.exists():
        print("no raw/session.json", file=sys.stderr)
        return 1
    session = json.loads(session_path.read_text(encoding="utf-8"))
    valid = {int(r["exp027"]["order_index"]) for r in _read_jsonl(RAW / "records.jsonl")
             if (r.get("exp027") or {}).get("session_id") == session["session_id"]
             and r["exp027"].get("valid")}
    if valid != set(range(1, 25)):
        print(f"only {len(valid)} of 24 launches valid: finish the session with "
              "`run --resume` first", file=sys.stderr)
        return 1
    return finalize(session, session_path)


# --------------------------------------------------------------- analyse ----

def _launch_row(r: Dict[str, Any]) -> Dict[str, Any]:
    e = r["exp027"]
    run0 = (r.get("metrics") or {}).get("runs", [{}])[0]
    res = (r.get("metrics") or {}).get("resources") or {}
    return {"model_key": e["model_key"], "threads": e["threads"], "round": e["round"],
            "order_index": e["order_index"], "gen_tok_s": run0.get("gen_tok_s"),
            "prompt_tok_s": run0.get("prompt_tok_s"), "ttft_s": run0.get("ttft_s"),
            "ram_peak": res.get("ram_peak")}


def classify(by_threads: Dict[int, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """CLEAR BEST / PRACTICAL TIE / INCONCLUSIVE for one model (see README)."""
    vals = {t: [x["gen_tok_s"] for x in v if x["gen_tok_s"]] for t, v in by_threads.items()}
    if any(len(v) < 3 for v in vals.values()) or len(vals) < 2:
        return {"verdict": "INCONCLUSIVE", "reason": "fewer than 3 valid launches in some configuration"}
    means = {t: statistics.mean(v) for t, v in vals.items()}
    sds = {t: statistics.stdev(v) for t, v in vals.items()}
    cvs = {t: sds[t] / means[t] for t in vals}
    ranked = sorted(means, key=lambda t: -means[t])
    best, second = ranked[0], ranked[1]
    gap = means[best] - means[second]
    pooled = ((sds[best] ** 2 + sds[second] ** 2) / 2) ** 0.5
    # order/session effect: spread of round means of normalised throughput
    rounds: Dict[int, List[float]] = {}
    for t, v in by_threads.items():
        for x in v:
            if x["gen_tok_s"]:
                rounds.setdefault(x["round"], []).append(x["gen_tok_s"] / means[t])
    rmeans = [statistics.mean(v) for v in rounds.values()]
    round_spread = (max(rmeans) - min(rmeans)) if rmeans else 0.0
    info = {"best": best, "second": second, "gap_tok_s": round(gap, 4),
            "gap_rel": round(gap / means[best], 4), "pooled_sd_top2": round(pooled, 4),
            "max_cv": round(max(cvs.values()), 4), "round_effect_rel": round(round_spread, 4),
            "means": {t: round(m, 4) for t, m in means.items()}}
    if max(cvs.values()) > CV_INCONCLUSIVE:
        return {**info, "verdict": "INCONCLUSIVE",
                "reason": f"between-launch CV {max(cvs.values()):.1%} > {CV_INCONCLUSIVE:.0%}"}
    separated = min(vals[best]) > max(vals[second]) and gap >= CLEAR_GAP_SD * pooled
    if separated and round_spread < gap / means[best]:
        return {**info, "verdict": "CLEAR BEST",
                "reason": f"no range overlap, gap >= {CLEAR_GAP_SD:g} x pooled SD, "
                          "and the round (order/session) effect is smaller than the gap"}
    if separated:
        return {**info, "verdict": "INCONCLUSIVE",
                "reason": "the best is separated from the second, but the round (order/session) "
                          "effect is at least as large as the gap"}

    def _tied(t: int) -> bool:
        pool = ((sds[best] ** 2 + sds[t] ** 2) / 2) ** 0.5
        return (means[best] - means[t] < CLEAR_GAP_SD * pool) or not (min(vals[best]) > max(vals[t]))
    return {**info, "verdict": "PRACTICAL TIE", "tied": sorted(t for t in ranked if _tied(t)),
            "reason": "top configurations within the observed between-launch variation"}


def analyze(records: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    valid = [r for r in records if (r.get("exp027") or {}).get("valid")]
    rows = [_launch_row(r) for r in valid]
    out_rows, summary = [], {"experiment": EXP_ID, "launches_valid": len(rows),
                             "launches_total": len(records), "models": {}}
    for key in MODEL_ORDER:
        by_t: Dict[int, List[Dict[str, Any]]] = {}
        for x in rows:
            if x["model_key"] == key:
                by_t.setdefault(x["threads"], []).append(x)
        for t in THREADS:
            v = sorted(by_t.get(t, []), key=lambda x: x["order_index"])
            g = [x["gen_tok_s"] for x in v if x["gen_tok_s"]]
            med = lambda k: (statistics.median([x[k] for x in v if x[k] is not None])  # noqa: E731
                             if any(x[k] is not None for x in v) else None)
            out_rows.append({
                "model": key, "threads": t, "batch_threads": BATCH_THREADS,
                "expert_budget": EXPERT_BUDGET, "n_launches": len(g),
                "gen_tok_s_launches": " ".join(f"{x:.2f}" for x in g),
                "gen_tok_s_mean": round(statistics.mean(g), 4) if g else None,
                "gen_tok_s_median": round(statistics.median(g), 4) if g else None,
                "gen_tok_s_sd": round(statistics.stdev(g), 4) if len(g) > 1 else None,
                "gen_tok_s_cv": round(statistics.stdev(g) / statistics.mean(g), 4) if len(g) > 1 else None,
                "gen_tok_s_min": min(g) if g else None, "gen_tok_s_max": max(g) if g else None,
                "prompt_tok_s_median": med("prompt_tok_s"), "ttft_s_median": med("ttft_s"),
                "ram_peak_max": max((x["ram_peak"] for x in v if x["ram_peak"]), default=None),
                "rounds": " ".join(str(x["round"]) for x in v)})
        verdict = classify(by_t) if by_t else {"verdict": "INCONCLUSIVE", "reason": "no data"}
        verdict["recommend_fourth_round"] = verdict["verdict"] != "CLEAR BEST"
        summary["models"][key] = verdict
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(out_rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(out_rows)
    return buf.getvalue(), summary


def cmd_analyze(_args: argparse.Namespace) -> int:
    recs = _read_jsonl(RAW / "records.jsonl")
    if not recs:
        print("no raw/records.jsonl: nothing to analyse (the experiment has not been run)",
              file=sys.stderr)
        return 1
    text, summary = analyze(recs)
    (EXP / "results.csv").write_text(text, encoding="utf-8", newline="")
    (EXP / "processed").mkdir(exist_ok=True)
    (EXP / "processed" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(text)
    for k, v in summary["models"].items():
        print(f"{k}: {v['verdict']} — {v.get('reason')}"
              + ("  (recommend the conditional fourth round)" if v["recommend_fourth_round"] else ""))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    p = sub.add_parser("preflight")
    p.add_argument("--llama-dir", default=str(DEFAULT_LLAMA_DIR))
    p = sub.add_parser("run")
    p.add_argument("--llama-dir", default=str(DEFAULT_LLAMA_DIR))
    p.add_argument("--resume", action="store_true")
    sub.add_parser("finalize")
    sub.add_parser("analyze")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        print_plan(build_plan())
        return 0 if not check_plan(build_plan()) else 1
    if a.cmd == "preflight":
        res = preflight(build_plan(), Path(a.llama_dir))
        return 0 if all(r["ok"] for r in res) else 1
    if a.cmd == "run":
        return run(a)
    if a.cmd == "finalize":
        return cmd_finalize(a)
    return cmd_analyze(a)


if __name__ == "__main__":
    sys.exit(main())
