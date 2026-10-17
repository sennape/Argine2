"""Build fingerprinting (section 28).

Every measurement records: Argine2 commit, llama.cpp commit/build, binary
fingerprint, compile flags, model checksum, Expert Store checksum, runtime
flags and workload version. A result whose fingerprint is incomplete is marked
`publication_quality: false` and must not be cited as a publication-quality
measurement.

Checksums of multi-GiB files: a full SHA-256 of a 35 GiB model takes minutes,
so two methods exist and the method is always recorded:
  * "sha256"      full-file SHA-256 (use --full-checksum);
  * "sha256-fast" SHA-256 of (size, first 16 MiB, last 16 MiB). It detects a
                  different file with overwhelming probability, but not an
                  edit in the middle; it is labelled as such.
Results are cached by (path, size, mtime) so a full hash is paid once.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

FAST_CHUNK = 16 << 20
RESEARCH_MARKER = b"argine2-research-v"
STORE_MARKER = b"argine: expert store"


def _cache_path() -> Path:
    try:
        from ..util import app_dir
        d = app_dir() / "research"
    except Exception:  # noqa: BLE001
        d = Path.home() / ".argine2"
    d.mkdir(parents=True, exist_ok=True)
    return d / "checksum-cache.json"


def file_checksum(path: str | Path, full: bool = False) -> Dict[str, object]:
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    st = p.stat()
    method = "sha256" if full else "sha256-fast"
    key = f"{p.resolve()}|{st.st_size}|{int(st.st_mtime)}|{method}"
    cache: Dict[str, str] = {}
    cp = _cache_path()
    try:
        cache = json.loads(cp.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        cache = {}
    if key in cache:
        digest = cache[key]
    else:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            if full or st.st_size <= 2 * FAST_CHUNK:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
                if not full:
                    method = "sha256"  # small file: the fast method is the full one
            else:
                h.update(str(st.st_size).encode())
                h.update(fh.read(FAST_CHUNK))
                fh.seek(st.st_size - FAST_CHUNK)
                h.update(fh.read(FAST_CHUNK))
        digest = h.hexdigest()
        cache[key] = digest
        try:
            cp.write_text(json.dumps(cache, indent=0), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    return {"path": str(p), "bytes": st.st_size, "method": method, "digest": digest}


def git_state(repo: Optional[str | Path] = None) -> Dict[str, object]:
    repo = Path(repo or Path(__file__).resolve().parents[2])
    try:
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10)
        if head.returncode != 0:
            return {"commit": None, "note": "not a git repository (initialise it before "
                                            "publishing measurements)"}
        # dirty = tracked files modified. Untracked files (the run's own output
        # directory, earlier runs) do not change the code that produced a result.
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain",
                                "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10)
        return {"commit": head.stdout.strip(), "dirty": bool(dirty.stdout.strip()),
                "dirty_definition": "tracked files modified (untracked files ignored)"}
    except Exception as e:  # noqa: BLE001
        return {"commit": None, "note": f"git unavailable: {e}"}


def binary_info(path: Optional[str | Path]) -> Dict[str, object]:
    if not path:
        return {"path": None}
    p = Path(path)
    out: Dict[str, object] = dict(file_checksum(p, full=True))
    try:
        data = p.read_bytes()
        out["has_expert_store"] = STORE_MARKER in data
        out["has_research_trace"] = RESEARCH_MARKER in data
    except Exception:  # noqa: BLE001
        pass
    info = p.parent / "BUILD-INFO.json"
    if info.exists():
        try:
            out["build_info"] = json.loads(info.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            out["build_info"] = {"error": "unreadable BUILD-INFO.json"}
    return out


GENERIC_CPU_NAMES = {"", "cpu sconosciuta", "unknown", "x86_64", "amd64", "intel64", "arm64"}


def _clean_cpu(name: Optional[str]) -> Optional[str]:
    """None for empty/generic strings, including Windows' 'Family N Model N
    Stepping N' identifiers, which name a CPU family, not a model."""
    if not name:
        return None
    n = " ".join(str(name).split())
    low = n.lower()
    if low in GENERIC_CPU_NAMES:
        return None
    if "family" in low and "model" in low and "stepping" in low:
        return None
    return n


def cpu_identity() -> Dict[str, object]:
    """A stable CPU model name and where it came from.

    Sources, in order: Windows registry (ProcessorNameString, the same string
    Task Manager shows), Win32_Processor via PowerShell, /proc/cpuinfo,
    macOS sysctl. platform.processor() is NOT accepted on Windows, where it
    returns a family/model/stepping string, not a model name. Nothing is ever
    hard-coded; if no source answers, name is None and source "unavailable".
    """
    tries = []
    if sys.platform.startswith("win"):
        try:
            import winreg  # type: ignore
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                tries.append(("windows-registry:ProcessorNameString",
                              winreg.QueryValueEx(k, "ProcessorNameString")[0]))
        except Exception:  # noqa: BLE001
            pass
        try:
            r = subprocess.run(["powershell", "-NoProfile", "-Command",
                                "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name"],
                               capture_output=True, text=True, timeout=20)
            tries.append(("powershell:Win32_Processor.Name", r.stdout.strip()))
        except Exception:  # noqa: BLE001
            pass
    elif sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
                if line.lower().startswith("model name"):
                    tries.append(("/proc/cpuinfo:model name", line.split(":", 1)[1]))
                    break
        except Exception:  # noqa: BLE001
            pass
    elif sys.platform == "darwin":
        try:
            r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                               capture_output=True, text=True, timeout=10)
            tries.append(("sysctl:machdep.cpu.brand_string", r.stdout.strip()))
        except Exception:  # noqa: BLE001
            pass
    if not sys.platform.startswith("win"):
        tries.append(("platform.processor", platform.processor()))
    for src, name in tries:
        n = _clean_cpu(name)
        if n:
            return {"name": n, "source": src}
    return {"name": None, "source": "unavailable"}


def machine_identity() -> Dict[str, object]:
    cpu = cpu_identity()
    out: Dict[str, object] = {"cpu": cpu["name"], "cpu_source": cpu["source"],
                              "cores_physical": None, "cores_logical": os.cpu_count(),
                              "ram_total": None, "os": platform.platform(),
                              "os_release": platform.release(), "os_version": platform.version(),
                              "machine": platform.machine()}
    try:
        import psutil  # type: ignore
        out["cores_physical"] = psutil.cpu_count(logical=False)
        out["cores_logical"] = psutil.cpu_count(logical=True)
        out["ram_total"] = int(psutil.virtual_memory().total)
    except Exception:  # noqa: BLE001
        pass
    return out


def collect(*, model: Optional[str] = None, store: Optional[str] = None,
            binary: Optional[str] = None, workload_file: Optional[str] = None,
            runtime_flags: Optional[Dict[str, object]] = None,
            full_checksum: bool = False, store_required: bool = False) -> Dict[str, object]:
    """`store_required=True` (any run with --expert-store) makes the Expert
    Store fingerprint mandatory for publication quality."""
    fp: Dict[str, object] = {
        "argine2": git_state(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "binary": binary_info(binary),
        "model": file_checksum(model, full_checksum) if model else None,
        "expert_store": file_checksum(store, full_checksum) if store else None,
        "expert_store_required": bool(store_required),
        "workload": file_checksum(workload_file, True) if workload_file else None,
        "runtime_flags": runtime_flags or {},
        "hardware": machine_identity(),
    }
    fp["publication_quality"], fp["missing"] = completeness(fp)
    return fp


def completeness(fp: Dict[str, object]) -> tuple:
    missing = []
    if not (fp.get("argine2") or {}).get("commit"):  # type: ignore[union-attr]
        missing.append("argine2 commit")
    b = fp.get("binary") or {}
    if not b.get("digest"):  # type: ignore[union-attr]
        missing.append("binary fingerprint")
    if not (b.get("build_info") or {}).get("llama_cpp_commit"):  # type: ignore[union-attr]
        missing.append("llama.cpp commit / compile flags (BUILD-INFO.json)")
    if not (fp.get("model") or {}).get("digest"):  # type: ignore[union-attr]
        missing.append("model checksum")
    store = fp.get("expert_store")
    if (fp.get("expert_store_required") or store is not None) and not (store or {}).get("digest"):  # type: ignore[union-attr]
        missing.append("expert store fingerprint (path, size, checksum, method)")
    if not (fp.get("workload") or {}).get("digest"):  # type: ignore[union-attr]
        missing.append("workload version")
    if not (fp.get("hardware") or {}).get("cpu"):  # type: ignore[union-attr]
        missing.append("CPU identity")
    return (not missing), missing
