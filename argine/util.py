"""Utility condivise: parsing/formatting dimensioni, path applicativi, output colorato."""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

KB = 1024
MB = 1024 ** 2
GB = 1024 ** 3

_SIZE_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([kmgtKMGT]?)([iI]?)[bB]?\s*$")

_UNIT_MULT = {"": 1, "k": KB, "m": MB, "g": GB, "t": GB * 1024}


def parse_size(value: str | int | float | None) -> Optional[int]:
    """Converte '11G', '11GB', '11GiB', '512M', '8192' in byte.

    Ritorna None se value e' None/vuoto. Solleva ValueError se non parsabile.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    value = str(value).strip()
    if not value:
        return None
    m = _SIZE_RE.match(value)
    if not m:
        raise ValueError(f"dimensione non valida: {value!r} (usa ad es. 11G, 512M, 2048)")
    num, unit, _ = m.groups()
    return int(float(num) * _UNIT_MULT[unit.lower()])


def fmt_bytes(n: Optional[float], decimals: int = 2) -> str:
    """Formatta byte in stringa leggibile (base 1024)."""
    if n is None:
        return "n/d"
    n = float(n)
    sign = "-" if n < 0 else ""
    n = abs(n)
    for unit, div in (("TiB", GB * 1024), ("GiB", GB), ("MiB", MB), ("KiB", KB)):
        if n >= div:
            return f"{sign}{n / div:.{decimals}f} {unit}"
    return f"{sign}{n:.0f} B"


def fmt_gb(n: Optional[float]) -> str:
    if n is None:
        return "n/d"
    return f"{n / GB:.2f} GiB"


def fmt_count(n: Optional[float]) -> str:
    """Formatta un conteggio di parametri: 7.62 B, 30.53 B, 512 M."""
    if n is None:
        return "n/d"
    n = float(n)
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.2f} {unit}"
    return f"{n:.0f}"


def app_dir() -> Path:
    """Directory dati di Argine (binari llama.cpp, log, cache)."""
    env = os.environ.get("ARGINE_HOME") or os.environ.get("PICOSTAR_HOME")
    if env:
        p = Path(env)
    elif os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        p = Path(base) / "Argine"
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
        p = Path(base) / "argine"
    nuova = not p.exists()
    p.mkdir(parents=True, exist_ok=True)
    if nuova:
        _eredita_storico(p)
    return p


def _eredita_storico(nuova: Path) -> None:
    """Porta avanti lo storico delle misure dalla cartella del vecchio nome.

    `runs.jsonl` e `perplexity.jsonl` contengono ogni misura mai presa, con le
    firme che rendono confrontabili le perplessita' fra loro. Ricominciare da
    zero perche' il progetto ha cambiato nome significherebbe buttare l'unica
    cosa che non si puo' rifare in fretta: due giorni di misure su una macchina
    specifica. Si copiano solo i log, una volta sola, alla prima esecuzione.
    """
    import shutil

    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        vecchia = Path(base) / "PicoStar16"
    else:
        base = (os.environ.get("XDG_DATA_HOME")
                or os.path.join(os.path.expanduser("~"), ".local", "share"))
        vecchia = Path(base) / "picostar16"
    if not (vecchia / "logs").is_dir():
        return
    try:
        (nuova / "logs").mkdir(parents=True, exist_ok=True)
        portati = []
        for nome in ("runs.jsonl", "perplexity.jsonl", "testo-ppl.txt",
                     "testo-imatrix.txt"):
            src = vecchia / "logs" / nome
            if src.is_file() and not (nuova / "logs" / nome).exists():
                shutil.copy2(src, nuova / "logs" / nome)
                portati.append(nome)
        if portati:
            print(f"[argine] storico delle misure ereditato da {vecchia.name}: "
                  f"{', '.join(portati)}")
    except Exception:  # noqa: BLE001
        pass  # non e' un motivo per non partire


def runtimes_dir() -> Path:
    p = app_dir() / "runtimes"
    p.mkdir(parents=True, exist_ok=True)
    return p


def logs_dir() -> Path:
    p = app_dir() / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def is_windows() -> bool:
    return os.name == "nt"


# ---------------------------------------------------------------- output ----

_NO_COLOR = bool(os.environ.get("NO_COLOR")) or not sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if _NO_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def bold(t: str) -> str:
    return _c("1", t)


def dim(t: str) -> str:
    return _c("2", t)


def green(t: str) -> str:
    return _c("32", t)


def yellow(t: str) -> str:
    return _c("33", t)


def red(t: str) -> str:
    return _c("31", t)


def cyan(t: str) -> str:
    return _c("36", t)


def header(title: str) -> str:
    line = "-" * max(0, 62 - len(title))
    return bold(f"== {title} {line}")


def kv(label: str, value: str, width: int = 24) -> str:
    return f"  {label:<{width}} {value}"


def info(msg: str) -> None:
    print(f"{cyan('[argine]')} {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"{yellow('[argine]')} {msg}", flush=True)


def error(msg: str) -> None:
    print(f"{red('[argine]')} {msg}", file=sys.stderr, flush=True)


def ok(msg: str) -> None:
    print(f"{green('[argine]')} {msg}", flush=True)


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0
        return False
