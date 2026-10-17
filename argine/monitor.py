"""Campionamento risorse del processo llama-server (RAM, CPU, disco, page fault)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore

from .util import GB, MB


@dataclass
class Sample:
    t: float
    rss: int = 0
    uss: Optional[int] = None
    cpu_percent: float = 0.0
    read_bytes: int = 0          # I/O esplicito del processo
    sys_read_bytes: int = 0      # I/O a livello di disco, TUTTO il sistema
    page_faults: int = 0
    system_available: int = 0


@dataclass
class ResourceStats:
    samples: List[Sample] = field(default_factory=list)
    duration: float = 0.0
    error: Optional[str] = None

    @property
    def ram_peak(self) -> int:
        return max((s.rss for s in self.samples), default=0)

    @property
    def ram_mean(self) -> float:
        if not self.samples:
            return 0.0
        return sum(s.rss for s in self.samples) / len(self.samples)

    @property
    def cpu_mean(self) -> float:
        vals = [s.cpu_percent for s in self.samples[1:]] or [s.cpu_percent for s in self.samples]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def cpu_peak(self) -> float:
        return max((s.cpu_percent for s in self.samples), default=0.0)

    @property
    def disk_read_total(self) -> int:
        if len(self.samples) < 2:
            return 0
        return max(0, self.samples[-1].read_bytes - self.samples[0].read_bytes)

    @property
    def disk_read_mb_s(self) -> float:
        if self.elapsed <= 0:
            return 0.0
        return self.disk_read_total / MB / self.elapsed

    @property
    def sys_disk_read_total(self) -> int:
        """Byte letti dal disco a livello di sistema.

        Necessario perche' su Windows il contatore di I/O del processo NON vede
        le pagine caricate dalla memoria mappata: quelle le serve il memory
        manager, non ReadFile. Su un modello piu' grande della RAM e' proprio
        quel traffico che vogliamo misurare.
        """
        if len(self.samples) < 2:
            return 0
        return max(0, self.samples[-1].sys_read_bytes - self.samples[0].sys_read_bytes)

    @property
    def sys_disk_read_mb_s(self) -> float:
        if self.elapsed <= 0:
            return 0.0
        return self.sys_disk_read_total / MB / self.elapsed

    @property
    def page_faults_total(self) -> int:
        if len(self.samples) < 2:
            return 0
        return max(0, self.samples[-1].page_faults - self.samples[0].page_faults)

    @property
    def system_available_min(self) -> int:
        return min((s.system_available for s in self.samples), default=0)

    @property
    def elapsed(self) -> float:
        if self.duration > 0:
            return self.duration
        return self.samples[-1].t if self.samples else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ram_peak": self.ram_peak,
            "ram_mean": int(self.ram_mean),
            "cpu_mean_percent": round(self.cpu_mean, 1),
            "cpu_peak_percent": round(self.cpu_peak, 1),
            "disk_read_total": self.disk_read_total,
            "disk_read_mb_s": round(self.disk_read_mb_s, 2),
            "sys_disk_read_total": self.sys_disk_read_total,
            "sys_disk_read_mb_s": round(self.sys_disk_read_mb_s, 2),
            "page_faults": self.page_faults_total,
            "system_available_min": self.system_available_min,
            "samples": len(self.samples),
            "duration_s": round(self.elapsed, 3),
            "error": self.error,
        }


class ProcessMonitor:
    """Campiona un processo e i suoi figli in un thread separato."""

    def __init__(self, pid: int, interval: float = 0.25):
        self.pid = pid
        self.interval = interval
        self.stats = ResourceStats()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0 = 0.0

    def start(self) -> "ProcessMonitor":
        if psutil is None:
            self.stats.error = "psutil non disponibile"
            return self
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="argine-monitor")
        self._thread.start()
        return self

    def stop(self) -> ResourceStats:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        self.stats.duration = time.perf_counter() - self._t0 if self._t0 else 0.0
        return self.stats

    # ------------------------------------------------------------------ loop
    def _loop(self) -> None:
        try:
            proc = psutil.Process(self.pid)
        except Exception as e:  # processo gia' morto
            self.stats.error = f"processo non monitorabile: {e}"
            return
        try:
            proc.cpu_percent(None)
        except Exception:
            pass
        while not self._stop.is_set():
            try:
                self.stats.samples.append(self._sample(proc))
            except psutil.NoSuchProcess:
                break
            except Exception as e:
                self.stats.error = str(e)
                break
            self._stop.wait(self.interval)

    def _sample(self, proc) -> Sample:
        procs = [proc]
        try:
            procs += proc.children(recursive=True)
        except Exception:
            pass
        rss = 0
        cpu = 0.0
        read_bytes = 0
        faults = 0
        for p in procs:
            try:
                mi = p.memory_info()
                rss += int(getattr(mi, "rss", 0) or 0)
                faults += int(getattr(mi, "num_page_faults", 0) or 0)
                cpu += float(p.cpu_percent(None) or 0.0)
                try:
                    io = p.io_counters()
                    read_bytes += int(getattr(io, "read_bytes", 0) or 0)
                except Exception:
                    pass
            except Exception:
                continue
        try:
            avail = int(psutil.virtual_memory().available)
        except Exception:
            avail = 0
        sys_read = 0
        try:
            dio = psutil.disk_io_counters()
            if dio is not None:
                sys_read = int(getattr(dio, "read_bytes", 0) or 0)
        except Exception:
            pass
        return Sample(
            t=time.perf_counter() - self._t0,
            rss=rss, cpu_percent=cpu, read_bytes=read_bytes,
            sys_read_bytes=sys_read, page_faults=faults, system_available=avail,
        )
