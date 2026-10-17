"""Runtime manager: procura llama.cpp GIA' COMPILATO.

L'utente finale non compila nulla. Alla prima esecuzione Argine scarica lo
zip ufficiale delle release di llama.cpp per il backend richiesto e lo estrae
in %LOCALAPPDATA%\\Argine\\runtimes.

Se sei offline puoi indicare una copia gia' presente:
    set ARGINE_LLAMA_DIR=C:\\percorso\\con\\llama-server.exe
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .util import (GB, MB, fmt_bytes, info, is_windows, ok, runtimes_dir, warn)

REPO = "ggml-org/llama.cpp"
API_RELEASES = f"https://api.github.com/repos/{REPO}/releases?per_page=40"
RELEASE_HTML = f"https://github.com/{REPO}/releases/expanded_assets/"
DOWNLOAD_BASE = f"https://github.com/{REPO}/releases/download"

# Build noto e funzionante, usato se l'API GitHub non e' raggiungibile
PINNED_BUILD = "b10456"

USER_AGENT = "Argine/0.1 (+local runtime manager)"

# Pattern degli asset per piattaforma/backend
ASSET_PATTERNS: Dict[Tuple[str, str], List[str]] = {
    ("windows", "cpu"): [r"^llama-.*-bin-win-cpu-x64\.zip$", r"^llama-.*-bin-win-avx2-x64\.zip$"],
    ("windows", "vulkan"): [r"^llama-.*-bin-win-vulkan-x64\.zip$"],
    ("windows", "sycl"): [r"^llama-.*-bin-win-sycl-x64\.zip$"],
    ("linux", "cpu"): [r"^llama-.*-bin-ubuntu-x64\.tar\.gz$"],
    ("linux", "vulkan"): [r"^llama-.*-bin-ubuntu-vulkan-x64\.tar\.gz$"],
    ("darwin", "cpu"): [r"^llama-.*-bin-macos-arm64\.tar\.gz$", r"^llama-.*-bin-macos-x64\.tar\.gz$"],
    ("darwin", "vulkan"): [r"^llama-.*-bin-macos-arm64\.tar\.gz$"],
}

BIN_NAMES = {
    "server": ("llama-server.exe", "llama-server"),
    "cli": ("llama-cli.exe", "llama-cli"),
    "bench": ("llama-bench.exe", "llama-bench"),
    # strumenti per la qualita': senza questi ogni affermazione su una
    # quantizzazione e' un'opinione
    "perplexity": ("llama-perplexity.exe", "llama-perplexity"),
    "quantize": ("llama-quantize.exe", "llama-quantize"),
    "imatrix": ("llama-imatrix.exe", "llama-imatrix"),
}


class RuntimeError_(RuntimeError):
    """Errore nel procurare/avviare llama.cpp."""


@dataclass
class LlamaRuntime:
    root: Path
    backend: str
    build: str = "custom"
    _help_cache: Optional[str] = field(default=None, repr=False, compare=False)
    _store_cache: Optional[bool] = field(default=None, repr=False, compare=False)

    def binary(self, kind: str = "server") -> Path:
        names = BIN_NAMES[kind]
        for name in names:
            direct = self.root / name
            if direct.exists():
                return direct
        for name in names:
            found = sorted(self.root.rglob(name))
            if found:
                return found[0]
        raise RuntimeError_(
            f"binario {names[0]} non trovato in {self.root}. "
            "Se hai impostato ARGINE_LLAMA_DIR verifica che contenga llama-server."
        )

    def version(self) -> str:
        try:
            res = _run([str(self.binary("server")), "--version"], timeout=30)
            text = (res.stdout or "") + (res.stderr or "")
            # formati visti: "version: 0.1.0-dev (build 10456, commit abc123)"
            #                "version: 4321 (abc123)"
            m = re.search(r"build\s+(\d+),\s*commit\s+([0-9a-f]+)", text)
            if m:
                return f"b{m.group(1)} ({m.group(2)})"
            m = re.search(r"version:\s*(\d+)\s*\(([0-9a-f]+)\)", text)
            if m:
                return f"b{m.group(1)} ({m.group(2)})"
            first = next((l for l in text.splitlines() if l.strip()), "")
            return first.strip() or self.build
        except Exception:
            return self.build

    def list_devices(self) -> List[str]:
        """Enumera i device del backend (verifica reale della disponibilita' Vulkan)."""
        devices: List[str] = []
        for args in (["--list-devices"], ["--help"]):
            try:
                res = _run([str(self.binary("server"))] + args, timeout=40)
            except Exception:
                continue
            text = (res.stdout or "") + (res.stderr or "")
            if args == ["--list-devices"]:
                capture = False
                for line in text.splitlines():
                    s = line.strip()
                    if s.lower().startswith("available devices"):
                        capture = True
                        continue
                    if capture:
                        if not s:
                            break
                        if s.lower() in ("(none)", "none", "-"):
                            continue
                        # scarta le righe di log del server che finiscono
                        # mescolate all'elenco (es. "0.00.000 I srv ...")
                        if re.match(r"^[\d.]+\s+[A-Z]\s", s) or "llama_server:" in s:
                            continue
                        devices.append(s)
                if devices:
                    return devices
            # fallback: righe ggml del tipo "ggml_vulkan: 0 = Intel(R) Iris(R) Xe ..."
            for line in text.splitlines():
                if "ggml_vulkan" in line and "=" in line:
                    devices.append(line.split(":", 1)[-1].strip())
            if devices:
                return devices
        return devices

    def supports_flag(self, flag: str) -> bool:
        return flag in self._help_text()

    def has_binary(self, kind: str) -> bool:
        """Il runtime contiene questo eseguibile?"""
        try:
            return self.binary(kind).exists()
        except Exception:
            return False

    def supports_expert_store(self) -> bool:
        """Questo binario contiene il buffer type dell'expert store?

        Non si puo' chiedere a `--help`: lo store si configura per variabili
        d'ambiente, proprio per tenere la patch a llama.cpp piccola, quindi non
        compare fra le opzioni. Si cerca il marcatore dentro il file.

        Serve perche' un binario ufficiale ignora le variabili in silenzio: il
        benchmark parte, gira, e produce un numero che descrive il percorso
        normale mentre l'utente crede di star misurando lo store. E' successo,
        ed e' costato due misure.
        """
        if self._store_cache is not None:
            return self._store_cache
        self._store_cache = False
        marcatore = b"ARGINE_EXPERT_STORE"

        # Dove cercare dipende da come e' stata costruita la build, e la prima
        # versione di questo controllo guardava solo l'eseguibile: su Windows
        # ggml e' collegato staticamente e il marcatore ci sta dentro, su Linux
        # e' in libggml-cpu.so e l'eseguibile non lo contiene. Il risultato era
        # un falso "non supportato" proprio sulla build con cui stavo provando.
        candidati: List[Path] = []
        try:
            candidati.append(self.binary("server"))
        except Exception:
            pass
        try:
            for f in sorted(self.root.glob("*")):
                if f.is_file() and f.suffix.lower() in (".so", ".dll", ".dylib"):
                    candidati.append(f)
                elif f.is_file() and ".so." in f.name:  # libggml-cpu.so.0.25.3
                    candidati.append(f)
        except Exception:
            pass

        for f in candidati:
            try:
                with open(f, "rb") as fh:
                    coda = b""
                    while True:
                        pezzo = fh.read(4 << 20)
                        if not pezzo:
                            break
                        if marcatore in coda + pezzo:
                            self._store_cache = True
                            return self._store_cache
                        # il marcatore puo' cadere a cavallo di due blocchi
                        coda = pezzo[-len(marcatore):]
            except Exception:
                continue
        return self._store_cache

    def _help_text(self) -> str:
        if self._help_cache is None:
            try:
                res = _run([str(self.binary("server")), "--help"], timeout=40)
                self._help_cache = (res.stdout or "") + (res.stderr or "")
            except Exception:
                self._help_cache = ""
        return self._help_cache


# ------------------------------------------------------------------- http ----

def _http_get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _run(cmd: List[str], timeout: int = 30) -> subprocess.CompletedProcess:
    creationflags = 0x08000000 if is_windows() else 0
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          creationflags=creationflags)


def _platform_key() -> str:
    s = platform.system().lower()
    if s.startswith("win"):
        return "windows"
    if s == "darwin":
        return "darwin"
    return "linux"


# --------------------------------------------------------- release resolve ---

def _releases_from_api() -> List[dict]:
    try:
        raw = _http_get(API_RELEASES, timeout=25)
        data = json.loads(raw.decode("utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _assets_from_html(tag: str) -> List[str]:
    """Fallback: legge i nomi degli asset dalla pagina HTML della release."""
    try:
        html = _http_get(RELEASE_HTML + tag, timeout=25).decode("utf-8", "replace")
    except Exception:
        return []
    return sorted(set(re.findall(r"/releases/download/[^/]+/([^\"'>\s]+)", html)))


def resolve_asset(backend: str, build: Optional[str] = None) -> Tuple[str, str]:
    """Trova (tag, nome_asset) per il backend richiesto.

    Nota: la release 'latest' di llama.cpp puo' non contenere binari (e' un tag
    puntatore), quindi scorriamo le release piu' recenti finche' non troviamo
    l'asset giusto.
    """
    plat = _platform_key()
    patterns = ASSET_PATTERNS.get((plat, backend))
    if not patterns:
        raise RuntimeError_(f"nessun binario precompilato noto per {plat}/{backend}")
    regexes = [re.compile(p) for p in patterns]

    candidates: List[Tuple[str, List[str]]] = []
    if build:
        candidates.append((build, _assets_from_html(build)))
    else:
        for rel in _releases_from_api():
            tag = str(rel.get("tag_name") or "")
            if not re.fullmatch(r"b\d+", tag):
                continue
            names = [a.get("name", "") for a in rel.get("assets", [])]
            candidates.append((tag, names))
        if not candidates:
            # API non disponibile (rate limit / offline): prova via HTML
            for tag in _recent_tags_from_html():
                candidates.append((tag, _assets_from_html(tag)))
        if not candidates:
            candidates.append((PINNED_BUILD, _assets_from_html(PINNED_BUILD)))

    for tag, names in candidates:
        for rx in regexes:
            for name in names:
                if rx.match(name):
                    return tag, name

    if build:
        raise RuntimeError_(f"la build {build} non contiene un asset per {plat}/{backend}")
    # ultimo tentativo: nome costruito a mano sul build pinnato
    guessed = _guess_asset_name(PINNED_BUILD, plat, backend)
    if guessed:
        warn(f"API GitHub non disponibile: uso il build pinnato {PINNED_BUILD}")
        return PINNED_BUILD, guessed
    raise RuntimeError_(
        f"impossibile individuare un binario llama.cpp per {plat}/{backend}. "
        "Scarica manualmente lo zip dalle release di llama.cpp e imposta ARGINE_LLAMA_DIR."
    )


def _recent_tags_from_html() -> List[str]:
    try:
        html = _http_get(f"https://github.com/{REPO}/releases", timeout=25).decode("utf-8", "replace")
    except Exception:
        return []
    tags = re.findall(r"/releases/tag/(b\d+)", html)
    seen: List[str] = []
    for t in tags:
        if t not in seen:
            seen.append(t)
    return seen[:15]


def _guess_asset_name(tag: str, plat: str, backend: str) -> Optional[str]:
    if plat == "windows":
        suffix = {"cpu": "win-cpu-x64.zip", "vulkan": "win-vulkan-x64.zip",
                  "sycl": "win-sycl-x64.zip"}.get(backend)
    elif plat == "linux":
        suffix = {"cpu": "ubuntu-x64.tar.gz", "vulkan": "ubuntu-vulkan-x64.tar.gz"}.get(backend)
    else:
        suffix = "macos-arm64.tar.gz"
    if not suffix:
        return None
    return f"llama-{tag}-bin-{suffix}"


# ------------------------------------------------------------- acquisizione --

def _install_dir(backend: str, tag: str) -> Path:
    return runtimes_dir() / f"{tag}-{backend}"


def _is_installed(d: Path) -> bool:
    if not d.is_dir():
        return False
    for names in (BIN_NAMES["server"],):
        for n in names:
            if (d / n).exists() or list(d.rglob(n)):
                return True
    return False


def _download(url: str, dest: Path, quiet: bool = False) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            t0 = time.time()
            with open(tmp, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    if not quiet and total:
                        pct = done * 100 // total
                        speed = done / max(1e-6, time.time() - t0)
                        sys.stdout.write(
                            f"\r  download {pct:3d}%  {fmt_bytes(done)}/{fmt_bytes(total)}"
                            f"  {fmt_bytes(speed)}/s   "
                        )
                        sys.stdout.flush()
        if not quiet:
            sys.stdout.write("\n")
        tmp.replace(dest)
    except urllib.error.HTTPError as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError_(f"download fallito ({e.code}) da {url}") from e
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError_(f"download fallito da {url}: {e}") from e


def _extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            for member in z.namelist():
                target = (dest / member).resolve()
                if not str(target).startswith(str(dest.resolve())):
                    raise RuntimeError_(f"archivio sospetto: path traversal su {member}")
            z.extractall(dest)
    else:
        with tarfile.open(archive, "r:*") as t:
            for member in t.getmembers():
                target = (dest / member.name).resolve()
                if not str(target).startswith(str(dest.resolve())):
                    raise RuntimeError_(f"archivio sospetto: path traversal su {member.name}")
            t.extractall(dest)
    # rendi eseguibili i binari su POSIX
    if not is_windows():
        for kind in BIN_NAMES.values():
            for name in kind:
                for f in dest.rglob(name):
                    try:
                        f.chmod(f.stat().st_mode | 0o755)
                    except OSError:
                        pass
    # appiattisci una eventuale cartella 'build/bin'
    for sub in ("build/bin", "bin"):
        cand = dest / sub
        if cand.is_dir() and not any((dest / n).exists() for n in BIN_NAMES["server"]):
            for f in cand.iterdir():
                try:
                    shutil.move(str(f), str(dest / f.name))
                except Exception:
                    pass


def _variabile(nome: str) -> Optional[str]:
    """Il valore della variabile, accettando anche il vecchio nome PICOSTAR_*.

    Il progetto si chiamava PicoStar16. Ignorare in silenzio un
    `PICOSTAR_LLAMA_DIR` scritto per abitudine sarebbe il peggiore dei
    comportamenti possibili: Argine ricadrebbe sulla build ufficiale senza dire
    niente, e due misure di velocita' prese su binari diversi sembrerebbero una
    differenza fra due modelli. E' successo davvero, prima della rinomina, solo
    perche' la variabile si era persa aprendo un terminale nuovo — e ha fatto
    attribuire alla quantizzazione un fattore 1,74 che in realtà proveniva dal cambio di binario/build; EXP-011 non isola quale differenza di build lo causasse
    """
    v = os.environ.get(nome)
    if v:
        return v
    vecchio = nome.replace("ARGINE_", "PICOSTAR_", 1)
    v = os.environ.get(vecchio)
    if v:
        warn(f"{vecchio} e' il nome vecchio: uso il suo valore, ma rinominala in "
             f"{nome} (il nome vecchio sparira')")
        return v
    return None


def _etichetta_locale(root: Path) -> str:
    """Nome della build per una copia locale indicata da ARGINE_LLAMA_DIR.

    Era la stringa fissa "ARGINE_LLAMA_DIR", uguale per qualunque cartella.
    Tre build compilate in modo diverso finivano nello storico con la stessa
    etichetta, e il controllo che avvisa quando il binario cambia non poteva
    vedere la differenza — cioe' falliva proprio nel caso in cui serve.

    Qui dentro ci va il nome della cartella e le prime cifre dell'impronta di
    llama-server: la cartella perche' e' leggibile, l'impronta perche'
    ricompilare nello stesso posto e' il modo piu' facile di cambiare binario
    senza accorgersene.
    """
    etichetta = root.name or "locale"
    try:
        h = hashlib.sha256()
        with open(LlamaRuntime(root=root, backend="cpu",
                                 build="?").binary("server"), "rb") as fh:
            for pezzo in iter(lambda: fh.read(1 << 20), b""):
                h.update(pezzo)
        return f"{etichetta}@{h.hexdigest()[:8]}"
    except Exception:  # noqa: BLE001
        return etichetta


def llama_dir_configurato() -> tuple[Optional[str], Optional[str], bool]:
    """(valore, nome della variabile che l'ha fornito, punta a una build valida).

    Serve a distinguere due guasti che si somigliavano e che il messaggio di
    `setup` confondeva: non riuscire a scaricare i binari, e averne indicata a
    mano una copia che non c'e'. Nel secondo caso ritentare il download non
    cambia niente — non si stava scaricando nulla — e il consiglio "indica una
    copia locale" ripete quello che e' gia' impostato e rotto.

    Si legge senza stampare l'avviso sul nome vecchio, perche' chi chiama questa
    funzione lo sta per dire in modo piu' preciso.
    """
    for nome in ("ARGINE_LLAMA_DIR", "PICOSTAR_LLAMA_DIR"):
        v = os.environ.get(nome)
        if v:
            return v, nome, _is_installed(Path(v).expanduser())
    return None, None, False


def build_accanto_al_progetto(percorso_rotto: str) -> Optional[Path]:
    """Cerca una build valida col nome dell'ultimo componente di un percorso che
    non esiste piu', dentro la cartella del progetto.

    Quando ARGINE_LLAMA_DIR punta nel vuoto, nove volte su dieci il motivo e' che
    la cartella del progetto e' stata spostata o rinominata e la build e' ancora
    dentro, chiamata allo stesso modo. Il pacchetto e' installato in editable, e
    quindi il suo `__file__` dice dove sta il progetto adesso. Non e' indovinare:
    e' guardare nel solo posto in cui ha senso guardare.
    """
    # Non `Path(...).name`: su Linux la barra rovesciata non e' un separatore, e
    # un percorso Windows tornerebbe intero come "nome". Funzionerebbe comunque
    # sulla macchina di destinazione, e questo e' il motivo peggiore per lasciare
    # un bug in piedi: non si vede, e nessun test lo prende.
    nome = percorso_rotto.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    if not nome or ":" in nome:
        return None
    try:
        progetto = Path(__file__).resolve().parents[1]
    except (OSError, IndexError):
        return None
    candidata = progetto / nome
    if _is_installed(candidata):
        return candidata
    return None


def ensure_runtime(backend: str, build: Optional[str] = None,
                   quiet: bool = False, offline: bool = False) -> LlamaRuntime:
    """Ritorna un LlamaRuntime pronto all'uso, scaricandolo se necessario."""
    env_dir = _variabile("ARGINE_LLAMA_DIR")
    if env_dir:
        root = Path(env_dir).expanduser()
        if not _is_installed(root):
            raise RuntimeError_(f"ARGINE_LLAMA_DIR={root} non contiene llama-server")
        return LlamaRuntime(root=root, backend=backend,
                            build=_etichetta_locale(root))

    # runtime gia' installati per questo backend: prendi il build piu' recente
    existing = sorted(
        (d for d in runtimes_dir().iterdir()
         if d.is_dir() and d.name.endswith(f"-{backend}") and _is_installed(d)),
        key=lambda d: _build_num(d.name), reverse=True,
    )
    if build:
        target = _install_dir(backend, build)
        if _is_installed(target):
            return LlamaRuntime(root=target, backend=backend, build=build)
    elif existing:
        d = existing[0]
        return LlamaRuntime(root=d, backend=backend, build=d.name.rsplit("-", 1)[0])

    if offline:
        raise RuntimeError_(
            f"nessun runtime {backend} installato e modalita' offline attiva. "
            "Esegui 'argine setup' con rete disponibile, oppure imposta ARGINE_LLAMA_DIR."
        )

    tag, asset = resolve_asset(backend, build)
    dest = _install_dir(backend, tag)
    if _is_installed(dest):
        return LlamaRuntime(root=dest, backend=backend, build=tag)

    url = f"{DOWNLOAD_BASE}/{tag}/{asset}"
    if not quiet:
        info(f"llama.cpp {tag} ({backend}) non presente: lo scarico una volta sola")
        info(f"  {url}")
    archive = runtimes_dir() / asset
    if not archive.exists():
        _download(url, archive, quiet=quiet)
    _extract(archive, dest)
    try:
        archive.unlink()
    except OSError:
        pass
    if not _is_installed(dest):
        raise RuntimeError_(f"estrazione completata ma llama-server non trovato in {dest}")
    rt = LlamaRuntime(root=dest, backend=backend, build=tag)
    if not quiet:
        ok(f"runtime pronto: {rt.binary('server')}")
    return rt


def _build_num(name: str) -> int:
    m = re.match(r"b(\d+)", name)
    return int(m.group(1)) if m else 0


def installed_runtimes() -> List[LlamaRuntime]:
    out: List[LlamaRuntime] = []
    for d in sorted(runtimes_dir().iterdir()) if runtimes_dir().exists() else []:
        if d.is_dir() and _is_installed(d):
            backend = d.name.rsplit("-", 1)[-1]
            out.append(LlamaRuntime(root=d, backend=backend, build=d.name.rsplit("-", 1)[0]))
    return out

def runtime_in_uso() -> Optional[LlamaRuntime]:
    """Il runtime che verrebbe usato adesso, senza scaricare niente.

    `installed_runtimes()` guarda solo nella cartella dati, e per questo
    `doctor` sapeva dire "non installato" mentre ARGINE_LLAMA_DIR puntava a una
    build perfettamente buona. Ma la diagnosi deve riguardare *il binario che
    partira'*, non quelli che abbiamo archiviato: e' esattamente la differenza
    che una volta ha fatto attribuire alla quantizzazione un fattore 1,74 che in realtà proveniva dal cambio di binario/build; EXP-011 non isola quale differenza di build lo causasse.
    """
    valore, _var, valido = llama_dir_configurato()
    if valore and valido:
        root = Path(valore).expanduser()
        return LlamaRuntime(root=root, backend="cpu",
                            build=_etichetta_locale(root))
    rts = installed_runtimes()
    return rts[0] if rts else None


def find_orphans() -> List[Dict[str, Any]]:
    """Trova processi llama-server rimasti attivi da esecuzioni precedenti.

    Su Windows un Ctrl+C sul terminale puo' terminare Python senza che la
    pulizia arrivi a girare: llama-server viene avviato in un gruppo di processi
    separato e non riceve il segnale. Restano orfani a occupare gigabyte, e il
    piano di memoria successivo trova poca RAM libera senza sapere perche'.
    """
    try:
        import psutil
    except ImportError:
        return []
    fuori: List[Dict[str, Any]] = []
    proprio = os.getpid()
    for proc in psutil.process_iter(["pid", "name", "create_time"]):
        try:
            nome = (proc.info.get("name") or "").lower()
            if not nome.startswith("llama-server"):
                continue
            if proc.info["pid"] == proprio:
                continue
            rss = 0
            try:
                rss = int(proc.memory_info().rss)
            except Exception:
                pass
            fuori.append({"pid": proc.info["pid"], "rss": rss,
                          "since": proc.info.get("create_time")})
        except Exception:
            continue
    return fuori


def kill_orphans(pids: Optional[List[int]] = None) -> int:
    """Termina i llama-server orfani. Ritorna quanti ne ha chiusi."""
    try:
        import psutil
    except ImportError:
        return 0
    obiettivi = pids if pids is not None else [o["pid"] for o in find_orphans()]
    chiusi = 0
    for pid in obiettivi:
        try:
            proc = psutil.Process(pid)
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except Exception:
                proc.kill()
            chiusi += 1
        except Exception:
            continue
    return chiusi
