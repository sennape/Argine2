"""Logging JSONL: una riga per esecuzione, confrontabile fra configurazioni."""

from __future__ import annotations

import json
import os
import platform
import socket
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import __version__
from .util import logs_dir

RUNS_LOG = "runs.jsonl"


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "to_dict"):
        try:
            return _jsonable(obj.to_dict())
        except Exception:
            pass
    return str(obj)


def log_path() -> Path:
    return logs_dir() / RUNS_LOG


def write_run(record: Dict[str, Any], path: Optional[Path] = None) -> Path:
    """Aggiunge una riga JSONL con il record di esecuzione."""
    p = path or log_path()
    payload = {
        "ts": time.time(),
        "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "argine_version": __version__,
        "host": socket.gethostname(),
        "platform": platform.platform(),
    }
    payload.update(_jsonable(record))
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return p


def ultima_build(percorso_modello: str, escludi_build: str = "",
                 path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """L'ultimo benchmark dello stesso modello con una build DIVERSA.

    Esiste per un confronto che mi e' sfuggito tre volte. Su Qwen3-30B-A3B
    IQ2_S, stesso file e stessi flag, la build ufficiale b11214 ha dato 13.53
    tok/s e la nostra patchata 7.45: 1.82x di differenza attribuita per mezz'ora
    alla quantizzazione, perche' la variabile ARGINE_LLAMA_DIR si era persa
    aprendo un terminale nuovo e nessuno lo diceva. Due misure di velocita' su
    binari diversi non si possono mettere nella stessa frase, e lo storico sa
    quale binario era in uso: tanto vale che lo dica.
    """
    migliore: Optional[Dict[str, Any]] = None
    for r in read_runs(path):
        if r.get("event") != "benchmark":
            continue
        # Il percorso e non il nome: `model.name` viene dai metadati del GGUF e
        # quattro quantizzazioni dello stesso modello lo condividono.
        if str((r.get("model") or {}).get("path") or "") != str(percorso_modello):
            continue
        b = str(((r.get("runtime") or {}).get("build")) or "")
        if not b or (escludi_build and b == escludi_build):
            continue
        tps = (r.get("metrics") or {}).get("gen_tok_s")
        migliore = {"build": b, "tok_s": tps, "quando": r.get("ts_iso")}
    return migliore


def read_runs(path: Optional[Path] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    p = path or log_path()
    if not p.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if limit:
        rows = rows[-limit:]
    return rows


def build_record(*, event: str, hardware: Any = None, model: Any = None,
                 plan: Any = None, runtime: Any = None, metrics: Any = None,
                 extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"event": event}
    if hardware is not None:
        rec["hardware"] = _hw_summary(hardware)
    if model is not None:
        rec["model"] = _model_summary(model)
    if plan is not None:
        rec["plan"] = _jsonable(plan)
    if runtime is not None:
        rec["runtime"] = _jsonable(runtime)
    if metrics is not None:
        rec["metrics"] = _jsonable(metrics)
    if extra:
        rec.update(_jsonable(extra))
    return rec


def _hw_summary(hw: Any) -> Dict[str, Any]:
    try:
        return {
            "cpu": hw.cpu_brand,
            "cores_physical": hw.cpu_cores_physical,
            "cores_logical": hw.cpu_cores_logical,
            "ram_total": hw.ram_total,
            "ram_available": hw.ram_available,
            "gpus": [g.name for g in hw.gpus],
            "vulkan": hw.vulkan_runtime,
            "backends": hw.backends,
        }
    except Exception:
        return _jsonable(hw)


def _model_summary(m: Any) -> Dict[str, Any]:
    try:
        return {
            "path": m.path,
            "name": m.name,
            "file_size": m.total_size_on_disk,
            "architecture": m.architecture,
            "quantization": m.quantization,
            "bits_per_weight": round(m.bits_per_weight, 3) if m.bits_per_weight else None,
            "params_total": m.n_params_total,
            "params_active": m.n_params_active,
            "moe": m.is_moe,
            "expert_count": m.expert_count,
            "expert_used": m.expert_used_count,
            "shared_bytes": m.shared_bytes,
            "expert_bytes": m.expert_bytes,
            "n_layers": m.n_layers,
            "n_ctx_train": m.n_ctx_train,
        }
    except Exception:
        return _jsonable(m)
