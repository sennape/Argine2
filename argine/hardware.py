"""Hardware Detector.

Rileva CPU, RAM, GPU, backend disponibili e spazio disco.
Progettato per Windows 11 ma con fallback cross-platform (utile per test).
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore

from .util import GB, MB, fmt_bytes, is_windows

# Vendor noti per GPU che condividono la RAM di sistema (iGPU)
_INTEGRATED_HINTS = (
    "iris", "uhd graphics", "hd graphics", "vega", "radeon graphics",
    "apple m", "adreno", "mali", "microsoft basic display",
)


@dataclass
class GPUInfo:
    name: str
    vendor: str = "unknown"
    dedicated_vram: Optional[int] = None  # byte, None se sconosciuto
    driver_version: Optional[str] = None
    integrated: bool = False
    source: str = ""  # come e' stata rilevata

    @property
    def shares_system_ram(self) -> bool:
        return self.integrated


@dataclass
class DiskInfo:
    path: str
    total: int
    free: int


@dataclass
class HardwareReport:
    os_name: str = ""
    os_version: str = ""
    machine: str = ""
    python: str = ""

    cpu_brand: str = ""
    cpu_cores_physical: Optional[int] = None
    cpu_cores_logical: Optional[int] = None
    cpu_freq_mhz: Optional[float] = None
    cpu_flags: List[str] = field(default_factory=list)
    cpu_p_cores: Optional[int] = None   # core con SMT (performance)
    cpu_e_cores: Optional[int] = None   # core senza SMT (efficient)

    ram_total: int = 0
    ram_available: int = 0
    swap_total: int = 0

    gpus: List[GPUInfo] = field(default_factory=list)
    vulkan_runtime: bool = False
    vulkan_icds: List[str] = field(default_factory=list)

    disks: List[DiskInfo] = field(default_factory=list)

    backends: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    # --------------------------------------------------------------- helpers
    @property
    def gpu_shared_memory_budget(self) -> int:
        """Su Windows la 'shared GPU memory' di una iGPU vale ~50% della RAM.

        Non e' memoria in piu': viene prelevata dalla RAM di sistema. Serve al
        Memory Planner per evitare di contare due volte lo stesso spazio.
        """
        if any(g.shares_system_ram for g in self.gpus):
            return self.ram_total // 2
        return 0

    @property
    def has_vulkan_gpu(self) -> bool:
        return self.vulkan_runtime and bool(self.gpus)

    @property
    def is_hybrid_cpu(self) -> bool:
        """CPU ibrida Intel (P-core con SMT + E-core senza SMT).

        Se ogni core fosse SMT avremmo logici = 2 x fisici; ogni core in meno
        rispetto a quel conto e' un E-core.
        """
        p, l = self.cpu_cores_physical, self.cpu_cores_logical
        if not p or not l:
            return False
        return p < l < 2 * p

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["gpu_shared_memory_budget"] = self.gpu_shared_memory_budget
        return d


# --------------------------------------------------------------------- CPU --

def _read_windows_cpu_brand() -> str:
    try:
        import winreg  # type: ignore

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        )
        with key:
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except Exception:
        return ""


def _read_linux_cpu_brand() -> str:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return ""


def _cpu_flags() -> List[str]:
    """Flag SIMD rilevanti per llama.cpp (avx2/avx512/f16c/fma...)."""
    wanted = {"avx", "avx2", "avx512f", "avx512_vnni", "avx_vnni", "f16c", "fma", "amx_int8", "amx_bf16"}
    found: List[str] = []
    try:
        if is_windows():
            # Non esiste un'API semplice: usiamo il minimo garantito dal brand moderno.
            # llama.cpp (build win-cpu-x64) sceglie a runtime la variante migliore,
            # quindi qui l'informazione e' solo diagnostica.
            return []
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if line.lower().startswith("flags"):
                    flags = set(line.split(":", 1)[1].split())
                    found = sorted(wanted & flags)
                    break
    except Exception:
        pass
    return found


# --------------------------------------------------------------------- GPU --

def _detect_gpus_windows() -> List[GPUInfo]:
    gpus: List[GPUInfo] = []
    ps = (
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,AdapterRAM,DriverVersion,AdapterCompatibility | "
        "ConvertTo-Json -Compress"
    )
    out = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps])
    if out:
        try:
            data = json.loads(out)
            if isinstance(data, dict):
                data = [data]
            for item in data:
                gpus.append(_gpu_from_wmi(item, "Win32_VideoController"))
        except Exception:
            pass
    if not gpus:
        out = _run(["wmic", "path", "win32_VideoController", "get", "Name,AdapterRAM,DriverVersion", "/format:csv"])
        if out:
            lines = [l for l in out.splitlines() if l.strip()]
            if len(lines) > 1:
                cols = [c.strip() for c in lines[0].split(",")]
                for line in lines[1:]:
                    vals = [v.strip() for v in line.split(",")]
                    item = dict(zip(cols, vals))
                    gpus.append(_gpu_from_wmi(item, "wmic"))
    return gpus


def _gpu_from_wmi(item: Dict[str, Any], source: str) -> GPUInfo:
    name = str(item.get("Name") or item.get("name") or "GPU sconosciuta").strip()
    vram = item.get("AdapterRAM")
    try:
        vram = int(vram) if vram not in (None, "", "0") else None
    except (TypeError, ValueError):
        vram = None
    # WMI riporta AdapterRAM su 32 bit: valori >= 4 GiB vanno in overflow.
    if vram is not None and vram <= 0:
        vram = None
    low = name.lower()
    vendor = "intel" if "intel" in low else "nvidia" if "nvidia" in low or "geforce" in low else \
        "amd" if "amd" in low or "radeon" in low else str(item.get("AdapterCompatibility") or "unknown").lower()
    integrated = any(h in low for h in _INTEGRATED_HINTS)
    return GPUInfo(
        name=name,
        vendor=vendor,
        dedicated_vram=None if integrated else vram,
        driver_version=(str(item.get("DriverVersion")) if item.get("DriverVersion") else None),
        integrated=integrated,
        source=source,
    )


def _detect_gpus_linux() -> List[GPUInfo]:
    gpus: List[GPUInfo] = []
    out = _run(["bash", "-lc", "lspci -nn 2>/dev/null | grep -Ei 'vga|3d|display'"])
    for line in (out or "").splitlines():
        if not line.strip():
            continue
        name = line.split(":", 2)[-1].strip()
        low = name.lower()
        vendor = "intel" if "intel" in low else "nvidia" if "nvidia" in low else "amd" if "amd" in low or "ati" in low else "unknown"
        gpus.append(GPUInfo(name=name, vendor=vendor,
                            integrated=any(h in low for h in _INTEGRATED_HINTS),
                            source="lspci"))
    return gpus


def _detect_vulkan() -> tuple[bool, List[str]]:
    """Ritorna (runtime presente, lista ICD trovati)."""
    icds: List[str] = []
    if is_windows():
        present = False
        for root in (os.environ.get("SystemRoot", r"C:\Windows"),):
            for sub in ("System32", "SysWOW64"):
                if (Path(root) / sub / "vulkan-1.dll").exists():
                    present = True
        try:
            import winreg  # type: ignore

            for hive in (winreg.HKEY_LOCAL_MACHINE,):
                for path in (r"SOFTWARE\Khronos\Vulkan\Drivers",):
                    try:
                        with winreg.OpenKey(hive, path) as key:
                            i = 0
                            while True:
                                try:
                                    nm, _val, _t = winreg.EnumValue(key, i)
                                    icds.append(nm)
                                    i += 1
                                except OSError:
                                    break
                    except OSError:
                        continue
        except Exception:
            pass
        return present or bool(icds), icds

    # Linux/macOS: utile solo per i test
    if shutil.which("vulkaninfo"):
        icds.append("vulkaninfo")
    for p in ("/usr/share/vulkan/icd.d", "/etc/vulkan/icd.d"):
        d = Path(p)
        if d.is_dir():
            icds.extend(sorted(f.name for f in d.glob("*.json")))
    lib = any(Path(p).exists() for p in (
        "/usr/lib/x86_64-linux-gnu/libvulkan.so.1", "/usr/lib/libvulkan.so.1", "/usr/local/lib/libvulkan.so.1"))
    return bool(icds) or lib, icds


# -------------------------------------------------------------------- disk --

def _detect_disks(extra_paths: Optional[List[str]] = None) -> List[DiskInfo]:
    disks: List[DiskInfo] = []
    seen: set[str] = set()
    candidates: List[str] = []
    if psutil is not None:
        for part in psutil.disk_partitions(all=False):
            if "cdrom" in part.opts or part.fstype == "":
                continue
            candidates.append(part.mountpoint)
    else:
        candidates.append(os.path.abspath(os.sep))
    candidates.append(str(Path.home()))
    for p in extra_paths or []:
        candidates.append(p)
    for path in candidates:
        try:
            usage = shutil.disk_usage(path)
        except OSError:
            continue
        key = f"{usage.total}"
        if key in seen:
            continue
        seen.add(key)
        disks.append(DiskInfo(path=path, total=usage.total, free=usage.free))
    # filtra i mount di servizio (immagini read-only, overlay minuscoli)
    big = [d for d in disks if d.total >= GB]
    return big or disks


# --------------------------------------------------------------------- run --

def _run(cmd: List[str], timeout: int = 20) -> Optional[str]:
    try:
        creationflags = 0x08000000 if is_windows() else 0  # CREATE_NO_WINDOW
        res = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            creationflags=creationflags,
        )
        if res.returncode == 0:
            return res.stdout
        return res.stdout or None
    except Exception:
        return None


# ---------------------------------------------------------------- override --

def _apply_overrides(r: HardwareReport) -> None:
    """Override via variabili d'ambiente, per validare piani su hardware diverso.

        ARGINE_FAKE_RAM=16G           simula la RAM totale
        ARGINE_FAKE_RAM_FREE=12G      simula la RAM libera
        ARGINE_FAKE_GPU="Intel(R) Iris(R) Xe Graphics"
        ARGINE_FAKE_CORES=12/16       core fisici/logici
        ARGINE_FAKE_VULKAN=1
    """
    from .util import parse_size  # import locale per evitare cicli

    changed = False
    v = os.environ.get("ARGINE_FAKE_RAM")
    if v:
        r.ram_total = int(parse_size(v) or r.ram_total)
        r.ram_available = min(r.ram_available or r.ram_total, int(r.ram_total * 0.75))
        changed = True
    v = os.environ.get("ARGINE_FAKE_RAM_FREE")
    if v:
        r.ram_available = int(parse_size(v) or r.ram_available)
        changed = True
    v = os.environ.get("ARGINE_FAKE_GPU")
    if v:
        low = v.lower()
        r.gpus = [GPUInfo(
            name=v,
            vendor="intel" if "intel" in low else "unknown",
            integrated=any(h in low for h in _INTEGRATED_HINTS),
            source="ARGINE_FAKE_GPU",
        )]
        changed = True
    v = os.environ.get("ARGINE_FAKE_CORES")
    if v:
        try:
            phys, _, log = v.partition("/")
            r.cpu_cores_physical = int(phys)
            r.cpu_cores_logical = int(log or phys)
            changed = True
        except ValueError:
            pass
    v = os.environ.get("ARGINE_FAKE_VULKAN")
    if v:
        r.vulkan_runtime = v not in ("0", "false", "no")
        changed = True
    if changed:
        r.notes.append("attenzione: hardware parzialmente simulato via ARGINE_FAKE_* "
                       "(le misure reali non sono attendibili)")


# ------------------------------------------------------------------ report --

def detect(extra_disk_paths: Optional[List[str]] = None) -> HardwareReport:
    r = HardwareReport()
    r.os_name = platform.system()
    r.os_version = platform.version() if not is_windows() else f"{platform.release()} (build {platform.version()})"
    r.machine = platform.machine()
    r.python = sys.version.split()[0]

    r.cpu_brand = (_read_windows_cpu_brand() if is_windows() else _read_linux_cpu_brand()) or platform.processor() or "CPU sconosciuta"
    if psutil is not None:
        r.cpu_cores_physical = psutil.cpu_count(logical=False)
        r.cpu_cores_logical = psutil.cpu_count(logical=True)
        try:
            freq = psutil.cpu_freq()
            r.cpu_freq_mhz = float(freq.max or freq.current) if freq else None
        except Exception:
            r.cpu_freq_mhz = None
        vm = psutil.virtual_memory()
        r.ram_total = int(vm.total)
        r.ram_available = int(vm.available)
        try:
            r.swap_total = int(psutil.swap_memory().total)
        except Exception:
            r.swap_total = 0
    else:
        r.cpu_cores_logical = os.cpu_count()
        r.notes.append("psutil non installato: RAM e metriche di processo non disponibili.")
    r.cpu_flags = _cpu_flags()

    r.gpus = _detect_gpus_windows() if is_windows() else _detect_gpus_linux()
    r.vulkan_runtime, r.vulkan_icds = _detect_vulkan()

    _apply_overrides(r)

    # dopo gli eventuali override: logici = 2*P + E e fisici = P + E => P = logici - fisici
    if r.is_hybrid_cpu:
        r.cpu_p_cores = r.cpu_cores_logical - r.cpu_cores_physical
        r.cpu_e_cores = r.cpu_cores_physical - r.cpu_p_cores

    r.disks = _detect_disks(extra_disk_paths)

    r.backends = ["cpu"]
    if r.has_vulkan_gpu:
        r.backends.append("vulkan")
    elif r.gpus and not r.vulkan_runtime:
        r.notes.append(
            "GPU rilevata ma runtime Vulkan assente: installa/aggiorna il driver grafico "
            "per abilitare il backend vulkan."
        )

    if r.ram_total and r.ram_total < 15 * GB:
        r.notes.append(f"RAM totale {fmt_bytes(r.ram_total)}: i default di Argine sono tarati su 16 GiB.")
    if any(g.shares_system_ram for g in r.gpus):
        r.notes.append(
            "GPU integrata: la memoria 'shared GPU' viene prelevata dalla RAM di sistema, "
            "quindi l'offload su Vulkan non libera RAM."
        )
    return r


def resolve_backend(requested: str, hw: HardwareReport) -> tuple[str, List[str]]:
    """Risolve --backend {auto,cpu,vulkan} nel backend effettivo.

    Ritorna (backend, messaggi). In 'auto' preferisce vulkan solo se c'e' una GPU
    Vulkan *non* integrata, oppure una iGPU con runtime Vulkan valido: sulle iGPU
    il guadagno e' modesto ma il prompt processing migliora.
    """
    msgs: List[str] = []
    requested = (requested or "auto").lower()
    if requested == "cpu":
        return "cpu", msgs
    if requested == "vulkan":
        if not hw.has_vulkan_gpu:
            msgs.append("backend vulkan richiesto ma non rilevato: verifica driver e vulkan-1.dll.")
        return "vulkan", msgs
    # auto
    if hw.has_vulkan_gpu:
        msgs.append(f"auto -> vulkan ({hw.gpus[0].name})")
        return "vulkan", msgs
    msgs.append("auto -> cpu (nessuna GPU Vulkan utilizzabile)")
    return "cpu", msgs
