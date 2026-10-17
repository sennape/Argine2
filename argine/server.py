"""Avvio di llama-server + gateway HTTP Argine (API OpenAI-compatibile).

Argine non reimplementa l'inferenza: avvia llama-server su una porta interna
e davanti mette un piccolo gateway che
  - inoltra /v1/* (chat/completions, models, embeddings, ...) senza modifiche;
  - aggiunge /argine/* con piano di memoria, telemetria e stato runtime.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import __version__
from . import anthropic_api as anth
from .gguf import GGUFModel
from .hardware import HardwareReport
from .monitor import ProcessMonitor
from .planner import RuntimePlan
from .runtime import LlamaRuntime
from .util import GB, MB, fmt_bytes, info, is_windows, logs_dir, ok, warn, error


# ---------------------------------------------------------------- args ------

def build_llama_args(rt: LlamaRuntime, model_path: str, plan: RuntimePlan,
                     host: str, port: int, extra: Optional[List[str]] = None) -> List[str]:
    """Traduce il RuntimePlan in argomenti per llama-server."""
    args: List[str] = [
        str(rt.binary("server")),
        "--model", str(model_path),
        "--host", host,
        "--port", str(port),
        "--ctx-size", str(plan.ctx),
        "--batch-size", str(plan.batch),
        "--ubatch-size", str(plan.ubatch),
        "--threads", str(plan.threads),
        "--threads-batch", str(plan.threads_batch),
        "--n-gpu-layers", str(plan.gpu_layers),
        "--cache-type-k", plan.kv_type_k,
        "--cache-type-v", plan.kv_type_v,
        "--alias", "local",
        # un solo slot: e' un runtime locale monoutente, e 4 slot (il default di
        # llama-server) moltiplicano lo stato per richiesta senza servire a nulla
        "--parallel", str(max(1, plan.parallel)),
    ]
    if plan.llama_verbose:
        # con -v llama.cpp dichiara le dimensioni reali dei buffer: e' il dato
        # con cui tariamo il Memory Planner
        args += ["-v"]

    if plan.flash_attn:
        if "--flash-attn" in rt._help_text():
            # build recenti: --flash-attn [on|off|auto]; build vecchie: flag booleano
            if "[on|off|auto]" in rt._help_text() or "on|off" in rt._help_text():
                args += ["--flash-attn", "on"]
            else:
                args += ["--flash-attn"]
    if not plan.repack and rt.supports_flag("--no-repack"):
        # evita la copia SIMD dei pesi: fondamentale quando il modello e' al
        # limite della RAM, perche' quella copia vale due terzi dei pesi Q4
        args += ["--no-repack"]
    if not plan.mmap:
        args += ["--no-mmap"]
    if plan.mlock:
        args += ["--mlock"]

    # Placement degli expert MoE (Milestone 2)
    if plan.cpu_moe and plan.gpu_layers > 0:
        if rt.supports_flag("--n-cpu-moe"):
            args += ["--n-cpu-moe", str(plan.n_cpu_moe)]
        elif rt.supports_flag("--cpu-moe"):
            args += ["--cpu-moe"]
        elif rt.supports_flag("--override-tensor"):
            args += ["--override-tensor", r"\.ffn_(gate|up|down)_exps\.=CPU"]

    # ---- decodifica speculativa ----
    if plan.spec_type and rt.supports_flag("--spec-type"):
        args += ["--spec-type", plan.spec_type]
        if plan.draft_model:
            args += ["--spec-draft-model", str(plan.draft_model)]
            if plan.draft_ctx:
                args += ["--spec-draft-ctx-size", str(plan.draft_ctx)]
        if plan.draft_n_max:
            args += ["--spec-draft-n-max", str(plan.draft_n_max)]
        if plan.draft_n_min:
            args += ["--spec-draft-n-min", str(plan.draft_n_min)]
        if plan.draft_p_min is not None:
            args += ["--spec-draft-p-min", str(plan.draft_p_min)]

    # ---- riuso della KV cache ----
    #
    # --cache-ram va passato SEMPRE, anche a zero. Il default di llama.cpp e'
    # 8192 MiB di stati di prompt tenuti in RAM: su una macchina da 16 GB e'
    # meta' del budget, allocata fuori da ogni contabilita' del planner. Non e'
    # mai esplosa nei benchmark perche' ripetono un prompt corto, ma il tetto di
    # RAM rigido — il requisito centrale del progetto — non esiste finche' quel
    # numero lo sceglie qualcun altro.
    if rt.supports_flag("--cache-ram"):
        mib = int(plan.estimate.prompt_cache) >> 20
        args += ["--cache-ram", str(mib)]
    if plan.prompt_cache_dir and rt.supports_flag("--slot-save-path"):
        Path(plan.prompt_cache_dir).mkdir(parents=True, exist_ok=True)
        args += ["--slot-save-path", str(plan.prompt_cache_dir)]
    if plan.cache_reuse and rt.supports_flag("--cache-reuse"):
        args += ["--cache-reuse", str(plan.cache_reuse)]

    if plan.jinja and rt.supports_flag("--jinja"):
        args += ["--jinja"]
    if rt.supports_flag("--no-webui") and os.environ.get("ARGINE_NO_WEBUI"):
        args += ["--no-webui"]

    if extra:
        args += list(extra)
    return args


def free_port(preferred: Optional[int] = None) -> int:
    if preferred:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", preferred))
                return preferred
            except OSError:
                pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False




# ---------------------------------------------------------------- job object --

def _assegna_a_job_windows(pid: int):
    """Lega il processo figlio alla vita del nostro, su Windows.

    Un Job Object con KILL_ON_JOB_CLOSE garantisce che llama-server muoia quando
    moriamo noi, anche se veniamo terminati di colpo e il nostro codice di
    pulizia non arriva a girare (e' quello che succede rispondendo "S" alla
    domanda "Terminare il processo batch?" dopo un Ctrl+C).

    Ritorna l'handle del job, da tenere aperto: se lo si chiude, i figli muoiono.
    """
    if not is_windows():
        return None
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
                ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [("ReadOperationCount", ctypes.c_ulonglong),
                        ("WriteOperationCount", ctypes.c_ulonglong),
                        ("OtherOperationCount", ctypes.c_ulonglong),
                        ("ReadTransferCount", ctypes.c_ulonglong),
                        ("WriteTransferCount", ctypes.c_ulonglong),
                        ("OtherTransferCount", ctypes.c_ulonglong)]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
        PROCESS_SET_QUOTA, PROCESS_TERMINATE = 0x0100, 0x0001

        job = k32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(
                job, 9, ctypes.byref(info), ctypes.sizeof(info)):
            k32.CloseHandle(job)
            return None
        handle = k32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid)
        if not handle:
            k32.CloseHandle(job)
            return None
        esito = k32.AssignProcessToJobObject(job, handle)
        k32.CloseHandle(handle)
        if not esito:
            k32.CloseHandle(job)
            return None
        return job
    except Exception:
        return None


# ------------------------------------------------------------- backend ------

# Righe con cui llama.cpp dichiara le dimensioni reali dei buffer allocati.
_BUFFER_LINE_RE = re.compile(
    r"(model buffer size|KV buffer size|KV self size|compute buffer size|output buffer size)",
    re.IGNORECASE,
)

# Righe della build patchata sull'expert store. Vanno catturate separatamente
# perche' sono la SOLA prova che la funzione sia davvero attiva: impostare una
# variabile d'ambiente non garantisce che il motore l'abbia onorata, e un tool
# che stampa "attivo" senza verificarlo mente al suo utente.
_ARGINE_LINE_RE = re.compile(r"^\s*argine:", re.IGNORECASE)


class LlamaServerProcess:
    """Gestisce il ciclo di vita del processo llama-server."""

    def __init__(self, args: List[str], log_file: Optional[Path] = None,
                 verbose: bool = False, env: Optional[Dict[str, str]] = None):
        self.args = args
        self.verbose = verbose
        self.extra_env = env or {}
        self._job = None
        self.log_file = log_file or (logs_dir() / "llama-server.log")
        self.proc: Optional[subprocess.Popen] = None
        self.monitor: Optional[ProcessMonitor] = None
        self._lines: List[str] = []
        self._psx_lines: List[str] = []
        self._load_lines: List[str] = []   # conservate: contengono i buffer reali
        self._reader: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        creationflags = 0
        if is_windows():
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # per il CTRL+C pulito
        child_env = None
        if self.extra_env:
            child_env = dict(os.environ)
            child_env.update(self.extra_env)
        self.proc = subprocess.Popen(
            self.args,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
            creationflags=creationflags,
        )
        # lega la vita del figlio alla nostra: niente orfani se veniamo uccisi
        self._job = _assegna_a_job_windows(self.proc.pid)
        self._reader = threading.Thread(target=self._pump, daemon=True, name="llama-log")
        self._reader.start()
        self.monitor = ProcessMonitor(self.proc.pid).start()

    def _pump(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        try:
            with open(self.log_file, "a", encoding="utf-8") as fh:
                fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} :: {' '.join(self.args)}\n")
                for line in self.proc.stdout:
                    fh.write(line)
                    fh.flush()
                    clean = line.rstrip("\n")
                    with self._lock:
                        self._lines.append(clean)
                        # Le righe dei buffer vanno catturate per CONTENUTO, non per
                        # posizione: con -v llama.cpp stampa prima una riga per ogni
                        # tensore del modello (centinaia), quindi i buffer arrivano
                        # molto dopo l'inizio del log.
                        if len(self._load_lines) < 80 and _BUFFER_LINE_RE.search(clean):
                            self._load_lines.append(clean)
                        if _ARGINE_LINE_RE.search(clean):
                            # Anello: il resoconto si ripete ogni 5 secondi e le
                            # righe che contano sono le ULTIME, non le prime. La
                            # prima versione teneva le prime 40 e buttava via
                            # proprio quelle di fine esecuzione.
                            self._psx_lines.append(clean.strip())
                            if len(self._psx_lines) > 400:
                                del self._psx_lines[:200]
                        if len(self._lines) > 400:
                            del self._lines[:200]
                    if self.verbose:
                        sys.stderr.write(f"  | {line}")
        except Exception:
            pass

    def tail(self, n: int = 30) -> List[str]:
        with self._lock:
            return list(self._lines[-n:])

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def wait_ready(self, port: int, timeout: float = 600.0) -> bool:
        """Attende che llama-server risponda su /health."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if not self.alive:
                return False
            try:
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
                conn.request("GET", "/health")
                resp = conn.getresponse()
                body = resp.read()
                conn.close()
                if resp.status == 200:
                    return True
                # 503 = modello in caricamento
            except Exception:
                pass
            time.sleep(0.4)
        return False

    def stop(self, timeout: float = 15.0) -> None:
        if self.proc is None:
            return
        if self.monitor is not None:
            self.monitor.stop()
        if self.proc.poll() is not None:
            return
        try:
            self.proc.terminate()
            self.proc.wait(timeout=timeout)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass

    # Buffer che llama.cpp dichiara al caricamento. Sono la verita' di riferimento
    # contro cui tarare il Memory Planner: il nostro modello analitico non puo'
    # conoscere l'allocatore di ggml meglio di ggml stesso.
    _BUF_PATTERNS = (
        (r"model buffer size\s*=\s*([\d.,]+)\s*MiB", "model"),
        (r"(?:KV self size|kv_cache.*?size)\s*=\s*([\d.,]+)\s*MiB", "kv"),
        (r"compute buffer size\s*=\s*([\d.,]+)\s*MiB", "compute"),
        (r"output buffer size\s*=\s*([\d.,]+)\s*MiB", "output"),
    )

    # Le righe del resoconto dello store, tradotte in numeri.
    #
    # Esistono perche' il dato piu' importante prodotto da questo progetto viveva
    # soltanto nella scrollback di un terminale: il tasso di hit. Su
    # Qwen3-Next-80B l'11,7% di residenza ha dato il 91,2% di hit, ed e' quel
    # numero — non la velocita' — che ha spiegato perche' due previsioni di fila
    # fossero sbagliate. `runs.jsonl` registrava velocita', TTFT, RAM e risorse, e
    # non lo registrava: una misura che spiega le altre e che nessun file conserva
    # e' una misura che va rifatta ogni volta che serve.
    #
    # Il formato e' quello che il C++ stampa su stderr; se cambia, questi numeri
    # diventano `None` invece di sbagliati, e i test confrontano il parser con le
    # righe vere.
    _RE_CACHE = (
        (r"(\d+)\s+hit,\s*(\d+)\s+miss\s*\(([\d.]+)%\s*hit\),\s*(\d+)\s+sfratti,"
         r"\s*(\d+)\s*MiB letti.*?(\d+)/(\d+)\s*MiB residenti",
         ("hit", "miss", "hit_pct", "sfratti", "mib_letti", "mib_residenti",
          "mib_budget")),
        (r"lettura\s+(\d+)\s*\(\d+%\).*?pagine nuove\s+(\d+)\s*\(\d+%\).*?"
         r"commit\s+(\d+)\s*\(\d+%\).*?decommit\s+(\d+)\s*\(\d+%\).*?"
         r"attesa\s+(\d+)\s*\(\d+%\).*?lock\s+(\d+)\s*\(\d+%\)",
         ("ms_lettura", "ms_pagine", "ms_commit", "ms_decommit", "ms_attesa",
          "ms_lock")),
        (r"dentro il nostro codice\s+(\d+)\s*ms su\s+(\d+)\s*nodi.*?"
         r"I/O\s+(\d+)\s*ms.*?attesa fra thread\s+(\d+)\s*ms",
         ("ms_nostro_codice", "nodi_moe", "ms_io", "ms_attesa_thread")),
        (r"parete\s+(\d+)\s*ms;\s*profondita' di coda media\s+([\d.]+).*?"
         r"banda\s+(\d+)\s*MB/s aggregati,\s*(\d+)\s*MB/s per blocco",
         ("ms_parete", "coda", "mbs_aggregati", "mbs_per_blocco")),
        (r"(\d+)\s+attese,\s*([\d.]+)\s+letture in volo per attesa",
         ("attese", "letture_per_attesa")),
        (r"per miss:\s*([\d.]+)\s*ms di lettura,\s*([\d.]+)\s*ms di pagine nuove,"
         r"\s*([\d.]+)\s*ms di commit",
         ("ms_lettura_miss", "ms_pagine_miss", "ms_commit_miss")),
    )

    @classmethod
    def statistiche_cache(cls, righe: Any) -> Dict[str, float]:
        """I numeri del resoconto della expert cache, da righe di log."""
        out: Dict[str, float] = {}
        for riga in righe or []:
            for schema, nomi in cls._RE_CACHE:
                m = re.search(schema, riga)
                if not m:
                    continue
                for nome, valore in zip(nomi, m.groups()):
                    out[nome] = float(valore) if "." in valore else int(valore)
        return out

    def expert_store_status(self) -> Dict[str, Any]:
        """Lo store ha davvero preso i tensori degli expert?

        Esiste perche' ho commesso l'errore di stampare "expert serviti dallo
        store" solo per aver impostato una variabile d'ambiente, senza
        controllare che il motore l'avesse onorata. Il risultato e' stato un
        benchmark il cui esito non significava niente, e non potevo saperlo
        guardando l'output. La prova sta nel log di llama-server: il nome del
        buffer e la sua dimensione.
        """
        out: Dict[str, Any] = {"aperto": False, "in_uso": False, "bytes": 0,
                               "righe": [], "stats": None, "resoconto": [],
                               "lettura": None, "copertura": None, "numeri": {}}
        with self._lock:
            out["righe"] = list(self._psx_lines)
            righe_buffer = list(self._load_lines)
        for riga in righe_buffer:
            if "ARGINE" not in riga.upper():
                continue
            m = re.search(r"=\s*([0-9.]+)\s*MiB", riga)
            if m:
                byte = int(float(m.group(1)) * (1 << 20))
                out["bytes"] = max(out["bytes"], byte)
                # la passata di stima della memoria di llama.cpp alloca buffer
                # finti da zero byte: solo un buffer non vuoto e' una prova.
                # Nota: senza -v llama-server non stampa affatto queste righe,
                # quindi la loro assenza non prova nulla. La prova sta nelle
                # righe che il nostro codice scrive direttamente su stderr.
        for riga in out["righe"]:
            bassa = riga.lower()
            if "expert store aperto" in bassa:
                out["aperto"] = True
                # Quale strato di lettura ha davvero preso il motore. Si legge
                # dal log e non dalla variabile d'ambiente che gli abbiamo
                # passato per la stessa ragione di tutto il resto di questo
                # metodo: una misura etichettata con cio' che abbiamo *chiesto*
                # invece di cio' che e' *avvenuto* e' esattamente l'errore che
                # ha reso insignificante un benchmark intero.
                m = re.search(r"lettura\s+([a-z]+(?::\d+)?)", bassa)
                if m:
                    out["lettura"] = m.group(1)
            if "expert store in uso" in bassa:
                out["in_uso"] = True
                out["aperto"] = True
            if "expert cache" in bassa and ":" in riga:
                out["stats"] = riga.split(":", 1)[-1].strip()
            # Quanti tensori di expert sono finiti davvero nel nostro buffer.
            # Fra "in uso" e "in uso per tutto" c'e' una differenza invisibile: con
            # una parte dei tensori sulla memoria mappata la riga "IN USO" compare
            # comunque, la cache riporta hit e miss plausibili, e il tetto di RAM e'
            # falso di quella parte.
            m = re.search(r"copertura:\s*(\d+)\s+tensori di expert su\s+(\d+)", bassa)
            if m:
                out["copertura"] = (int(m.group(1)), int(m.group(2)))

        # l'ultimo blocco di resoconto: dalla riga "expert cache" piu' recente in poi
        ultimo = -1
        for i, riga in enumerate(out["righe"]):
            if "expert cache" in riga.lower():
                ultimo = i
        out["resoconto"] = list(out["righe"][ultimo:]) if ultimo >= 0 else []
        out["numeri"] = self.statistiche_cache(out["resoconto"])
        return out

    def reported_buffers(self) -> Dict[str, Any]:
        """Dimensioni reali dei buffer dichiarate da llama.cpp al caricamento.

        Distinzione essenziale: un buffer "Mapped" e' una mappatura del file
        (memoria VIRTUALE, la residenza la decide la page cache del sistema
        operativo), mentre gli altri sono allocazioni fisiche vere. Sommarli
        insieme porta a conclusioni sbagliate: su un modello piu' grande della
        RAM il mapped vale quanto tutto il file.
        """
        from .util import MB as _MB

        with self._lock:
            lines = list(self._load_lines)
        out: Dict[str, float] = {}
        mapped = 0.0
        detail: List[str] = []
        for line in lines:
            for pattern, key in self._BUF_PATTERNS:
                m = re.search(pattern, line, re.IGNORECASE)
                if not m:
                    continue
                try:
                    mib = float(m.group(1).replace(",", ""))
                except ValueError:
                    continue
                if key == "model" and re.search(r"mapped", line, re.IGNORECASE):
                    mapped += mib
                else:
                    out[key] = out.get(key, 0.0) + mib
                detail.append(line.strip())
                break
        if not out and not mapped:
            return {}
        res: Dict[str, Any] = {f"{k}_bytes": int(v * _MB) for k, v in out.items()}
        res["mapped_bytes"] = int(mapped * _MB)
        res["allocated_bytes"] = int(sum(out.values()) * _MB)
        res["total_bytes"] = res["allocated_bytes"] + res["mapped_bytes"]
        res["lines"] = detail[:20]
        return res

    def detect_oom(self) -> Optional[str]:
        """Cerca nei log indizi di out-of-memory / allocazione fallita."""
        needles = (
            "failed to allocate", "unable to allocate", "cannot allocate",
            "out of memory", "std::bad_alloc", "ggml_backend_alloc",
            "insufficient memory", "buffer of size",
        )
        for line in self.tail(200):
            low = line.lower()
            if any(n in low for n in needles):
                return line.strip()
        return None


# ------------------------------------------------------------- gateway ------

@dataclass
class GatewayState:
    upstream_port: int
    plan: RuntimePlan
    model: GGUFModel
    hardware: HardwareReport
    runtime_build: str = ""
    backend: str = "cpu"
    started_at: float = field(default_factory=time.time)
    backend_proc: Optional[LlamaServerProcess] = None
    expert_stats: Optional[Dict[str, Any]] = None


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"Argine/{__version__}"
    state: GatewayState = None  # type: ignore[assignment]

    # log HTTP silenzioso salvo verbose
    def log_message(self, fmt: str, *args: Any) -> None:
        if os.environ.get("ARGINE_HTTP_LOG"):
            sys.stderr.write("  http: " + fmt % args + "\n")

    # ------------------------------------------------------------ routing --
    def do_GET(self) -> None:
        if self._argine_route():
            return
        self._proxy("GET")

    def do_POST(self) -> None:
        if self._argine_route():
            return
        if self.path.split("?", 1)[0] == "/v1/messages":
            return self._messages()
        self._proxy("POST")

    def do_DELETE(self) -> None:
        self._proxy("DELETE")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS")

    # ---------------------------------------------------------- endpoints --
    def _argine_route(self) -> bool:
        path = self.path.split("?", 1)[0]
        st = self.state
        if path == "/argine/status":
            proc = st.backend_proc
            mon = proc.monitor.stats if (proc and proc.monitor) else None
            body = {
                "argine": __version__,
                "uptime_s": round(time.time() - st.started_at, 1),
                "backend": st.backend,
                "llama_build": st.runtime_build,
                "upstream_port": st.upstream_port,
                "model": {
                    "name": st.model.name,
                    "path": st.model.path,
                    "architecture": st.model.architecture,
                    "quantization": st.model.quantization,
                    "moe": st.model.is_moe,
                    "expert_count": st.model.expert_count,
                    "expert_used": st.model.expert_used_count,
                    "params_total": st.model.n_params_total,
                    "params_active": st.model.n_params_active,
                },
                "alive": bool(proc and proc.alive),
                "resources": mon.to_dict() if mon else None,
            }
            return self._json(200, body)
        if path == "/argine/plan":
            return self._json(200, st.plan.to_dict())
        if path == "/argine/memory":
            est = st.plan.estimate
            proc = st.backend_proc
            mon = proc.monitor.stats if (proc and proc.monitor) else None
            body = {
                "budget": st.plan.budget,
                "estimate": est.to_dict(),
                "measured": mon.to_dict() if mon else None,
                "ram_total": st.hardware.ram_total,
            }
            return self._json(200, body)
        if path == "/argine/experts":
            return self._json(200, st.expert_stats or {
                "moe": st.model.is_moe,
                "note": "telemetria expert disponibile con 'argine experts simulate'",
                "expert_count": st.model.expert_count,
                "expert_used": st.model.expert_used_count,
                "shared_bytes": st.model.shared_bytes,
                "expert_bytes": st.model.expert_bytes,
                "cache_budget": st.plan.expert_cache_budget,
                "streaming": st.plan.moe_streaming,
            })
        if path in ("/argine", "/argine/"):
            return self._json(200, {
                "service": "Argine",
                "version": __version__,
                "endpoints": ["/v1/chat/completions", "/v1/completions", "/v1/models",
                              "/v1/messages", "/health", "/argine/status",
                              "/argine/plan", "/argine/memory", "/argine/experts"],
                "dialetti": {"openai": "/v1/chat/completions", "anthropic": "/v1/messages"},
            })
        return False

    def _json(self, code: int, body: Dict[str, Any]) -> bool:
        data = json.dumps(body, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        return True

    # ------------------------------------------------- dialetto Anthropic --
    def _messages(self) -> None:
        """`/v1/messages`: traduce, inoltra a `/v1/chat/completions`, ritraduce.

        La traduzione sta in `anthropic_api`, che e' fatto di funzioni pure: qui
        resta solo il trasporto. La divisione non e' estetica — e' quello che
        rende la parte difficile verificabile senza avviare un modello.
        """
        st = self.state
        length = int(self.headers.get("Content-Length") or 0)
        grezzo = self.rfile.read(length) if length else b""
        try:
            req = json.loads(grezzo or b"{}")
        except json.JSONDecodeError as e:
            return self._json_anth(400, anth.ApiError(f"JSON non valido: {e}").to_dict())
        try:
            oai = anth.richiesta_in_openai(req)
        except anth.ApiError as e:
            return self._json_anth(400, e.to_dict())

        modello = str(req.get("model") or st.model.name)
        sequenze = req.get("stop_sequences") if isinstance(req.get("stop_sequences"), list) else None
        corpo = json.dumps(oai, ensure_ascii=False).encode("utf-8")
        intestazioni = {"Content-Type": "application/json",
                        "Content-Length": str(len(corpo)),
                        "Connection": "close",
                        "Accept-Encoding": "identity"}
        try:
            conn = http.client.HTTPConnection("127.0.0.1", st.upstream_port, timeout=3600)
            conn.request("POST", "/v1/chat/completions", body=corpo, headers=intestazioni)
            resp = conn.getresponse()
        except Exception as e:  # noqa: BLE001
            return self._json_anth(502, {
                "type": "error",
                "error": {"type": "api_error",
                          "message": f"backend llama-server non raggiungibile: {e}"},
            })

        if resp.status >= 400:
            # L'errore del backend si rigira nella forma Anthropic: un client che
            # parla quel dialetto legge gli errori con quello schema, e tradurre
            # tutto tranne la parte che si legge quando le cose vanno male e' il
            # modo piu' sicuro di far sembrare l'integrazione rotta.
            try:
                dentro = json.loads(resp.read() or b"{}")
            except json.JSONDecodeError:
                dentro = {}
            messaggio = ((dentro.get("error") or {}).get("message")
                         if isinstance(dentro.get("error"), dict) else None)
            self._json_anth(resp.status, {
                "type": "error",
                "error": {"type": "api_error",
                          "message": messaggio or f"errore dal backend ({resp.status})"},
            })
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
            return

        if oai.get("stream"):
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                for pezzo in anth.eventi_da_chunk_openai(
                        anth.righe_sse(resp.read), modello, sequenze):
                    self.wfile.write(b"%X\r\n" % len(pezzo) + pezzo + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception:  # noqa: BLE001
                pass
        else:
            try:
                dentro = json.loads(resp.read() or b"{}")
            except json.JSONDecodeError as e:
                self._json_anth(502, {
                    "type": "error",
                    "error": {"type": "api_error",
                              "message": f"risposta del backend non interpretabile: {e}"},
                })
                return
            self._json_anth(200, anth.risposta_in_anthropic(dentro, modello, sequenze))
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass

    def _json_anth(self, code: int, body: Dict[str, Any]) -> None:
        self._json(code, body)

    # -------------------------------------------------------------- proxy --
    def _proxy(self, method: str) -> None:
        st = self.state
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None

        headers = {}
        for k, v in self.headers.items():
            lk = k.lower()
            if lk in ("host", "connection", "content-length", "accept-encoding",
                      "transfer-encoding", "keep-alive", "proxy-connection"):
                continue
            headers[k] = v
        headers["Connection"] = "close"
        headers["Accept-Encoding"] = "identity"
        if body is not None:
            headers["Content-Length"] = str(len(body))

        try:
            conn = http.client.HTTPConnection("127.0.0.1", st.upstream_port, timeout=3600)
            conn.request(method, self.path, body=body, headers=headers)
            resp = conn.getresponse()
        except Exception as e:
            self._json(502, {"error": {
                "message": f"backend llama-server non raggiungibile: {e}",
                "type": "argine_upstream_error",
            }})
            return

        ctype = resp.getheader("Content-Type", "application/octet-stream")
        clen = resp.getheader("Content-Length")
        self.send_response(resp.status)
        self._cors()
        self.send_header("Content-Type", ctype)
        passthrough = ("cache-control", "x-request-id", "openai-organization")
        for k, v in resp.getheaders():
            if k.lower() in passthrough:
                self.send_header(k, v)
        if clen is not None:
            self.send_header("Content-Length", clen)
            self.end_headers()
            try:
                remaining = int(clen)
                while remaining > 0:
                    chunk = resp.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            except Exception:
                pass
        else:
            # streaming (SSE): rilanciamo in chunked
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    self.wfile.write(b"%X\r\n" % len(chunk) + chunk + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception:
                pass
        try:
            conn.close()
        except Exception:
            pass


class ArgineGateway:
    def __init__(self, host: str, port: int, state: GatewayState):
        self.host = host
        self.port = port
        handler = type("_BoundHandler", (_Handler,), {"state": state})
        self.httpd = ThreadingHTTPServer((host, port), handler)
        self.httpd.daemon_threads = True
        self._thread: Optional[threading.Thread] = None

    def serve_forever_background(self) -> None:
        self._thread = threading.Thread(target=self.httpd.serve_forever,
                                        kwargs={"poll_interval": 0.3},
                                        daemon=True, name="argine-gateway")
        self._thread.start()

    def shutdown(self) -> None:
        try:
            self.httpd.shutdown()
            self.httpd.server_close()
        except Exception:
            pass
