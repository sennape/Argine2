"""Benchmark: tok/s, time-to-first-token, RAM, page fault, disco, CPU/GPU."""

from __future__ import annotations

import http.client
import json
import re
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .util import GB, MB, fmt_bytes, is_windows

DEFAULT_PROMPT = (
    "Spiega in modo conciso ma completo come funziona la memoria virtuale in un "
    "sistema operativo moderno, includendo paginazione, page fault, working set e "
    "il ruolo del file di paging su SSD. Poi elenca tre conseguenze pratiche per "
    "un'applicazione che usa memory mapping su file piu' grandi della RAM fisica."
)


@dataclass
class BenchRun:
    prompt_tokens: int = 0
    generated_tokens: int = 0
    prompt_ms: float = 0.0
    predicted_ms: float = 0.0
    ttft_s: Optional[float] = None
    wall_s: float = 0.0
    stop_reason: str = ""

    @property
    def prompt_tok_s(self) -> Optional[float]:
        if self.prompt_ms <= 0 or not self.prompt_tokens:
            return None
        return self.prompt_tokens / (self.prompt_ms / 1000.0)

    @property
    def gen_tok_s(self) -> Optional[float]:
        if self.predicted_ms <= 0 or not self.generated_tokens:
            return None
        return self.generated_tokens / (self.predicted_ms / 1000.0)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["prompt_tok_s"] = round(self.prompt_tok_s, 2) if self.prompt_tok_s else None
        d["gen_tok_s"] = round(self.gen_tok_s, 2) if self.gen_tok_s else None
        return d


@dataclass
class BenchResult:
    runs: List[BenchRun] = field(default_factory=list)
    ttft_warm_s: Optional[float] = None   # con prefisso gia' in KV cache
    gpu_util_mean: Optional[float] = None
    gpu_util_peak: Optional[float] = None
    resources: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def _agg(self, attr: str) -> Optional[float]:
        vals = [getattr(r, attr) for r in self.runs]
        vals = [v for v in vals if v]
        return statistics.median(vals) if vals else None

    @property
    def prompt_tok_s(self) -> Optional[float]:
        vals = [r.prompt_tok_s for r in self.runs if r.prompt_tok_s]
        return statistics.median(vals) if vals else None

    @property
    def gen_tok_s(self) -> Optional[float]:
        vals = [r.gen_tok_s for r in self.runs if r.gen_tok_s]
        return statistics.median(vals) if vals else None

    @property
    def ttft_s(self) -> Optional[float]:
        return self._agg("ttft_s")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "runs": [r.to_dict() for r in self.runs],
            "prompt_tok_s_median": round(self.prompt_tok_s, 2) if self.prompt_tok_s else None,
            "gen_tok_s_median": round(self.gen_tok_s, 2) if self.gen_tok_s else None,
            "ttft_s_median": round(self.ttft_s, 3) if self.ttft_s else None,
            "ttft_warm_s": round(self.ttft_warm_s, 3) if self.ttft_warm_s else None,
            "gpu_util_mean": self.gpu_util_mean,
            "gpu_util_peak": self.gpu_util_peak,
            "resources": self.resources,
            "errors": self.errors,
        }


# ------------------------------------------------------------------ http -----

def _post(port: int, path: str, payload: Dict[str, Any], timeout: float = 1800.0) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("POST", path, body=body,
                     headers={"Content-Type": "application/json", "Connection": "close"})
        resp = conn.getresponse()
        raw = resp.read()
        if resp.status != 200:
            raise RuntimeError(f"{path} -> HTTP {resp.status}: {raw[:400].decode('utf-8', 'replace')}")
        return json.loads(raw.decode("utf-8"))
    finally:
        conn.close()


def _post_stream_ttft(port: int, path: str, payload: Dict[str, Any],
                      timeout: float = 1800.0) -> tuple[Optional[float], float, int, Dict[str, Any]]:
    """Esegue una richiesta in streaming e misura il time-to-first-token."""
    payload = dict(payload)
    payload["stream"] = True
    body = json.dumps(payload).encode("utf-8")
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    ttft: Optional[float] = None
    ntok = 0
    timings: Dict[str, Any] = {}
    t0 = time.perf_counter()
    try:
        conn.request("POST", path, body=body,
                     headers={"Content-Type": "application/json", "Accept": "text/event-stream",
                              "Connection": "close"})
        resp = conn.getresponse()
        if resp.status != 200:
            raw = resp.read()
            raise RuntimeError(f"{path} -> HTTP {resp.status}: {raw[:400].decode('utf-8','replace')}")
        buf = b""
        while True:
            chunk = resp.read(1)
            if not chunk:
                break
            buf += chunk
            if not buf.endswith(b"\n\n") and not buf.endswith(b"\n"):
                continue
            for line in buf.split(b"\n"):
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data in (b"[DONE]", b""):
                    continue
                try:
                    obj = json.loads(data.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                piece = _extract_piece(obj)
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    ntok += 1
                if isinstance(obj.get("timings"), dict):
                    timings = obj["timings"]
            buf = b""
    finally:
        conn.close()
    return ttft, time.perf_counter() - t0, ntok, timings


def _extract_piece(obj: Dict[str, Any]) -> str:
    if "content" in obj and isinstance(obj["content"], str):
        return obj["content"]
    choices = obj.get("choices")
    if isinstance(choices, list) and choices:
        d = choices[0].get("delta") or {}
        if isinstance(d.get("content"), str):
            return d["content"]
        if isinstance(choices[0].get("text"), str):
            return choices[0]["text"]
    return ""


# -------------------------------------------------------------- GPU util -----

class GpuSampler:
    """Campiona l'utilizzo GPU su Windows tramite i contatori prestazioni."""

    def __init__(self, seconds: int = 30):
        self.seconds = max(3, seconds)
        self.values: List[float] = []
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> "GpuSampler":
        if not is_windows():
            return self
        self._thread = threading.Thread(target=self._loop, daemon=True, name="argine-gpu")
        self._thread.start()
        return self

    def _loop(self) -> None:
        ps = (
            "$s=0; "
            f"1..{self.seconds} | ForEach-Object {{ "
            "try { $c = (Get-Counter '\\GPU Engine(*)\\Utilization Percentage' "
            "-ErrorAction Stop).CounterSamples | Measure-Object -Property CookedValue -Sum; "
            "Write-Output $c.Sum } catch { Write-Output 'NA' }; Start-Sleep -Seconds 1 }"
        )
        try:
            proc = subprocess.Popen(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                creationflags=0x08000000,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                if self._stop.is_set():
                    proc.terminate()
                    break
                line = line.strip()
                if not line or line == "NA":
                    continue
                try:
                    self.values.append(min(100.0, float(line.replace(",", "."))))
                except ValueError:
                    continue
        except Exception:
            pass

    def stop(self) -> tuple[Optional[float], Optional[float]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        if not self.values:
            return None, None
        return round(sum(self.values) / len(self.values), 1), round(max(self.values), 1)


# --------------------------------------------------------------- benchmark ---

def run_benchmark(port: int, *, prompt: str = DEFAULT_PROMPT, n_predict: int = 128,
                  repeat: int = 3, warmup: bool = True,
                  monitor_stats=None, on_progress=None) -> BenchResult:
    """Esegue il benchmark contro un llama-server/gateway gia' avviato."""
    result = BenchResult()

    if warmup:
        try:
            _post(port, "/completion", {"prompt": "ciao", "n_predict": 8,
                                        "stream": False, "cache_prompt": False}, timeout=600)
        except Exception as e:
            result.errors.append(f"warmup: {e}")

    gpu = GpuSampler(seconds=max(10, repeat * 20)).start()

    for i in range(repeat):
        if on_progress:
            on_progress(i + 1, repeat)
        run = BenchRun()
        t0 = time.perf_counter()
        # 1) streaming: TTFT reale
        try:
            ttft, wall, ntok, timings = _post_stream_ttft(
                port, "/completion",
                {"prompt": prompt, "n_predict": n_predict, "cache_prompt": False,
                 "temperature": 0.0, "timings_per_token": False},
            )
            run.ttft_s = ttft
            run.wall_s = wall
            run.generated_tokens = ntok
            if timings:
                _apply_timings(run, timings)
        except Exception as e:
            result.errors.append(f"run {i+1} (stream): {e}")

        # 2) non-streaming: timings precisi di llama.cpp
        if not run.prompt_ms or not run.predicted_ms:
            try:
                obj = _post(port, "/completion",
                            {"prompt": prompt, "n_predict": n_predict, "stream": False,
                             "cache_prompt": False, "temperature": 0.0})
                t = obj.get("timings") or {}
                _apply_timings(run, t)
                run.stop_reason = str(obj.get("stop_type") or obj.get("stopped_eos") or "")
                if not run.wall_s:
                    run.wall_s = time.perf_counter() - t0
            except Exception as e:
                result.errors.append(f"run {i+1} (timings): {e}")
        result.runs.append(run)

    # TTFT con il prefisso gia' in cache: e' il caso reale di una conversazione,
    # dove solo i token nuovi vengono elaborati. Quello a freddo e' il caso
    # peggiore (primo messaggio); riportarne uno solo inganna.
    try:
        _post(port, "/completion", {"prompt": prompt, "n_predict": 1,
                                    "stream": False, "cache_prompt": True}, timeout=1800)
        ttft_warm, _w, _n, _t = _post_stream_ttft(
            port, "/completion",
            {"prompt": prompt + "\nContinua:", "n_predict": 8,
             "cache_prompt": True, "temperature": 0.0},
        )
        result.ttft_warm_s = ttft_warm
    except Exception as e:
        result.errors.append(f"ttft a caldo: {e}")

    result.gpu_util_mean, result.gpu_util_peak = gpu.stop()
    if monitor_stats is not None:
        try:
            result.resources = monitor_stats.to_dict()
        except Exception:
            pass
    return result


@dataclass
class TurnoResult:
    turno: int = 0
    prompt_tokens: int = 0
    nuovi_tokens: int = 0      # token aggiunti rispetto al turno precedente
    ttft_s: Optional[float] = None
    gen_tok_s: Optional[float] = None
    prompt_ms: float = 0.0
    cached_tokens: int = 0     # quanti token llama.cpp dice di aver riusato

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _testo_turno(n: int, parole: int = 120) -> str:
    """Un turno di conversazione plausibile e diverso dagli altri.

    Diverso conta: se i turni fossero identici il prefisso crescerebbe in modo
    innaturalmente comprimibile, e misureremmo la ripetizione invece della
    conversazione.
    """
    temi = ["la paginazione su disco", "la quantizzazione a due bit",
            "la barriera fra i thread", "la cache degli expert",
            "la banda di memoria", "il costo di azzerare una pagina",
            "la profondita' di coda delle letture", "il riuso del prefisso"]
    t = temi[n % len(temi)]
    base = (f"Domanda {n}: riprendendo il discorso precedente, approfondisci "
            f"{t} e spiega come si misura, quali errori di metodo sono tipici in "
            f"quella misura, e cosa cambierebbe su una macchina con meta' della "
            f"memoria. ")
    riempi = (f"Considera anche il punto {n} dell'elenco e le implicazioni "
              f"pratiche per chi usa il sistema ogni giorno. ")
    out = base
    while len(out.split()) < parole:
        out += riempi
    return out + "\n"


def run_conversazione(port: int, *, turni: int = 6, parole_per_turno: int = 120,
                      n_predict: int = 48, cache_prompt: bool = True,
                      on_progress=None) -> List[TurnoResult]:
    """Misura il TTFT turno dopo turno su una conversazione che cresce.

    Perche' serve un benchmark a parte: tutte le altre misure del progetto
    mandano prompt **indipendenti**, uno per volta. Una conversazione vera e' un
    prefisso che cresce — turno 1, poi 1+2, poi 1+2+3 — e il riuso della KV cache
    esiste esattamente per quel caso. Misurare prompt indipendenti e poi
    progettare una cache del prefisso significa progettare contro un banco di
    prova che non riproduce il caso d'uso.

    `cache_prompt=False` serve a ottenere il termine di paragone: lo stesso
    identico carico senza riuso, cioe' quanto costerebbe ogni turno rifacendo il
    prefill da zero.
    """
    fuori: List[TurnoResult] = []
    conversazione = ""
    precedenti = 0
    for i in range(1, turni + 1):
        if on_progress:
            on_progress(i, turni)
        conversazione += _testo_turno(i, parole_per_turno)
        r = TurnoResult(turno=i)
        try:
            ttft, _wall, ntok, timings = _post_stream_ttft(
                port, "/completion",
                {"prompt": conversazione, "n_predict": n_predict,
                 "cache_prompt": bool(cache_prompt), "temperature": 0.0},
            )
            r.ttft_s = ttft
            tmp = BenchRun()
            tmp.generated_tokens = ntok
            if timings:
                _apply_timings(tmp, timings)
            r.prompt_tokens = tmp.prompt_tokens
            r.prompt_ms = tmp.prompt_ms
            r.gen_tok_s = tmp.gen_tok_s
            r.cached_tokens = int((timings or {}).get("prompt_n_cached")
                                  or (timings or {}).get("cache_n") or 0)
        except Exception:  # noqa: BLE001
            pass
        r.nuovi_tokens = max(0, r.prompt_tokens - precedenti)
        precedenti = r.prompt_tokens or precedenti
        # la risposta del modello fa parte del prefisso del turno dopo
        conversazione += f"Risposta {i}: " + ("testo " * 30) + "\n"
        fuori.append(r)
    return fuori


def format_conversazione(con: List[TurnoResult],
                         senza: Optional[List[TurnoResult]] = None) -> List[str]:
    righe = ["  turno  token prompt   nuovi     TTFT   riusati"]
    for r in con:
        righe.append(
            f"  {r.turno:>5}  {r.prompt_tokens:>12}  {r.nuovi_tokens:>6}  "
            f"{(r.ttft_s or 0):>6.2f}s  {r.cached_tokens:>8}")
    if senza:
        righe.append("")
        righe.append("  stessa conversazione senza riuso del prefisso:")
        righe.append("  turno  token prompt     TTFT   rapporto")
        for a, b in zip(senza, con):
            rapp = (a.ttft_s / b.ttft_s) if (a.ttft_s and b.ttft_s) else 0.0
            righe.append(
                f"  {a.turno:>5}  {a.prompt_tokens:>12}  {(a.ttft_s or 0):>6.2f}s  "
                f"{rapp:>8.1f}x")
    return righe


def _apply_timings(run: BenchRun, t: Dict[str, Any]) -> None:
    run.prompt_tokens = int(t.get("prompt_n") or run.prompt_tokens or 0)
    run.prompt_ms = float(t.get("prompt_ms") or run.prompt_ms or 0.0)
    pn = int(t.get("predicted_n") or 0)
    if pn:
        run.generated_tokens = pn
    run.predicted_ms = float(t.get("predicted_ms") or run.predicted_ms or 0.0)
