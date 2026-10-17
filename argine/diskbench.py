"""Misura del tetto reale dello storage, sullo stesso file e sullo stesso
pattern di accesso che usa il motore.

Perche' esiste questo modulo
----------------------------
Su Qwen3-30B-A3B Q4_K_M la generazione e' limitata dall'I/O: la CPU resta al
20-42% mentre il disco legge 76-83 MiB per token a 290-562 MB/s. Tutte le
modifiche al *modo* in cui usiamo la memoria mappata (tre varianti di prefetch,
riordino degli expert per localita', batch piu' piccoli) hanno lasciato quel
numero invariato.

Restano due spiegazioni possibili, e sono incompatibili:

  A. il disco fisico non sa fare di meglio con questo pattern;
  B. il percorso di page fault della memoria mappata non riesce a tenere
     occupata la coda dell'NVMe, e letture esplicite ci arriverebbero.

La differenza vale settimane di lavoro, quindi la si misura invece di
argomentarla. Lo stesso file, le stesse slice, gli stessi offset casuali,
tre percorsi diversi:

  mmap      quello che fa llama.cpp oggi: mappa il file e tocca le pagine,
            lasciando al memory manager il compito di procurarsele
  pread     lettura posizionata con la cache dell'OS attiva
  direct    lettura posizionata che scavalca la cache
            (FILE_FLAG_NO_BUFFERING su Windows, O_DIRECT su Linux)
  seq       lettura sequenziale a blocchi grandi, senza cache: il tetto
            nominale del dispositivo, per avere un riferimento

Ogni percorso viene provato a piu' livelli di parallelismo, perche' l'ipotesi B
e' precisamente un'ipotesi sulla profondita' di coda.

Lettura dei risultati
---------------------
Vengono riportati due throughput distinti:

  logico   byte consegnati al chiamante / tempo
  fisico   byte letti dal dispositivo / tempo, dal contatore di sistema

Se logico >> fisico, quella prova stava pescando dalla page cache e il suo
numero non e' un dato sul disco. `direct` non ha questo problema per
costruzione, ed e' per questo che e' la prova che decide.
"""

from __future__ import annotations

import ctypes
import os
import platform
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from .util import fmt_bytes

try:  # psutil e' un requisito del progetto, ma questo modulo resta usabile senza
    import psutil
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

IS_WINDOWS = os.name == "nt"
SECTOR = 4096  # allineamento richiesto dall'I/O non bufferizzato


# ---------------------------------------------------------------------------
# risultati
# ---------------------------------------------------------------------------


@dataclass
class DiskResult:
    kind: str
    slice_bytes: int
    threads: int
    bytes_logical: int
    seconds: float
    bytes_physical: int | None = None
    ops: int = 0
    error: str | None = None

    @property
    def mb_s(self) -> float:
        if self.seconds <= 0:
            return 0.0
        return self.bytes_logical / self.seconds / 1e6

    @property
    def phys_mb_s(self) -> float | None:
        if self.bytes_physical is None or self.seconds <= 0:
            return None
        return self.bytes_physical / self.seconds / 1e6

    @property
    def iops(self) -> float:
        if self.seconds <= 0:
            return 0.0
        return self.ops / self.seconds

    @property
    def latency_ms(self) -> float:
        """Latenza media per operazione vista da un singolo thread."""
        if self.ops <= 0:
            return 0.0
        return self.seconds * self.threads / self.ops * 1000.0

    @property
    def cache_ratio(self) -> float | None:
        """Quota di byte serviti senza toccare il disco. None se non misurabile."""
        if self.bytes_physical is None or self.bytes_logical <= 0:
            return None
        served = self.bytes_logical - self.bytes_physical
        return max(0.0, min(1.0, served / self.bytes_logical))


# ---------------------------------------------------------------------------
# buffer allineati
# ---------------------------------------------------------------------------


def _alloc_aligned(size: int):
    """Buffer allineato al settore, necessario per l'I/O non bufferizzato.

    Restituisce (oggetto_da_tenere_vivo, indirizzo).
    """
    if IS_WINDOWS:
        MEM_COMMIT_RESERVE = 0x1000 | 0x2000
        PAGE_READWRITE = 0x04
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.VirtualAlloc.restype = ctypes.c_void_p
        kernel32.VirtualAlloc.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_ulong,
            ctypes.c_ulong,
        ]
        addr = kernel32.VirtualAlloc(None, size, MEM_COMMIT_RESERVE, PAGE_READWRITE)
        if not addr:
            raise OSError(ctypes.get_last_error(), "VirtualAlloc")
        return _VirtualAllocGuard(addr, size), addr
    # su POSIX una mappatura anonima e' allineata alla pagina per definizione
    import mmap as _mmap

    buf = _mmap.mmap(-1, size)
    addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    return buf, addr


class _VirtualAllocGuard:
    def __init__(self, addr: int, size: int) -> None:
        self.addr = addr
        self.size = size

    def close(self) -> None:
        if self.addr:
            MEM_RELEASE = 0x8000
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong]
            kernel32.VirtualFree(ctypes.c_void_p(self.addr), 0, MEM_RELEASE)
            self.addr = 0

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# percorso 1: memoria mappata (quello che fa llama.cpp)
# ---------------------------------------------------------------------------


class _Mapping:
    """Mappatura in sola lettura dell'intero file, con l'indirizzo base esposto.

    Serve l'indirizzo perche' le pagine vengono toccate con una `memcpy` via
    ctypes: ctypes rilascia il GIL durante la chiamata, quindi piu' thread
    Python generano page fault davvero in parallelo. Passando invece per lo
    slicing dell'oggetto `mmap` la copia avverrebbe col GIL in mano e la prova
    a piu' thread misurerebbe l'interprete, non il disco.
    """

    def __init__(self, path: Path) -> None:
        self.size = path.stat().st_size
        self._keep: list = []
        if IS_WINDOWS:
            GENERIC_READ = 0x80000000
            FILE_SHARE_READ = 0x00000001
            OPEN_EXISTING = 3
            PAGE_READONLY = 0x02
            FILE_MAP_READ = 0x04
            k = ctypes.WinDLL("kernel32", use_last_error=True)
            k.CreateFileW.restype = ctypes.c_void_p
            k.CreateFileW.argtypes = [
                ctypes.c_wchar_p,
                ctypes.c_ulong,
                ctypes.c_ulong,
                ctypes.c_void_p,
                ctypes.c_ulong,
                ctypes.c_ulong,
                ctypes.c_void_p,
            ]
            h = k.CreateFileW(str(path), GENERIC_READ, FILE_SHARE_READ, None, OPEN_EXISTING, 0, None)
            if h in (None, 0) or h == 2**64 - 1:
                raise OSError(ctypes.get_last_error(), f"CreateFileW {path}")
            k.CreateFileMappingW.restype = ctypes.c_void_p
            k.CreateFileMappingW.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_ulong,
                ctypes.c_ulong,
                ctypes.c_ulong,
                ctypes.c_wchar_p,
            ]
            m = k.CreateFileMappingW(ctypes.c_void_p(h), None, PAGE_READONLY, 0, 0, None)
            if not m:
                raise OSError(ctypes.get_last_error(), "CreateFileMappingW")
            k.MapViewOfFile.restype = ctypes.c_void_p
            k.MapViewOfFile.argtypes = [
                ctypes.c_void_p,
                ctypes.c_ulong,
                ctypes.c_ulong,
                ctypes.c_ulong,
                ctypes.c_size_t,
            ]
            base = k.MapViewOfFile(ctypes.c_void_p(m), FILE_MAP_READ, 0, 0, 0)
            if not base:
                raise OSError(ctypes.get_last_error(), "MapViewOfFile")
            self.base = int(base)
            self._win = (k, h, m)
        else:
            import mmap as _mmap

            f = open(path, "rb")
            mm = _mmap.mmap(f.fileno(), 0, access=_mmap.ACCESS_COPY)
            self.base = ctypes.addressof(ctypes.c_char.from_buffer(mm))
            self._keep = [mm, f]
            self._win = None

    def close(self) -> None:
        if self._win:
            k, h, m = self._win
            k.UnmapViewOfFile.argtypes = [ctypes.c_void_p]
            k.UnmapViewOfFile(ctypes.c_void_p(self.base))
            k.CloseHandle.argtypes = [ctypes.c_void_p]
            k.CloseHandle(ctypes.c_void_p(m))
            k.CloseHandle(ctypes.c_void_p(h))
            self._win = None
        for obj in reversed(self._keep):
            try:
                obj.close()
            except Exception:
                pass
        self._keep = []


# ---------------------------------------------------------------------------
# percorso 2 e 3: letture posizionate, con e senza cache
# ---------------------------------------------------------------------------


class _PositionedReader:
    """Lettura a un offset arbitrario, con un handle per thread.

    `direct=True` scavalca la cache dell'OS: su Windows FILE_FLAG_NO_BUFFERING,
    su Linux O_DIRECT. In quel caso offset, lunghezza e indirizzo del buffer
    devono essere multipli del settore.
    """

    def __init__(self, path: Path, direct: bool) -> None:
        self.path = path
        self.direct = direct
        self.size = path.stat().st_size

    # -- Windows ----------------------------------------------------------
    def _open_win(self):
        GENERIC_READ = 0x80000000
        FILE_SHARE_READ = 0x00000001
        OPEN_EXISTING = 3
        FILE_FLAG_NO_BUFFERING = 0x20000000
        FILE_FLAG_RANDOM_ACCESS = 0x10000000
        flags = FILE_FLAG_NO_BUFFERING if self.direct else FILE_FLAG_RANDOM_ACCESS
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.CreateFileW.restype = ctypes.c_void_p
        k.CreateFileW.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_void_p,
        ]
        h = k.CreateFileW(
            str(self.path), GENERIC_READ, FILE_SHARE_READ, None, OPEN_EXISTING, flags, None
        )
        if h in (None, 0) or h == 2**64 - 1:
            raise OSError(ctypes.get_last_error(), f"CreateFileW {self.path}")
        return k, h

    # -- POSIX ------------------------------------------------------------
    def _open_posix(self):
        flags = os.O_RDONLY
        if self.direct and hasattr(os, "O_DIRECT"):
            flags |= os.O_DIRECT
        return os.open(str(self.path), flags)

    def make_worker(self) -> tuple[Callable[[int, int, int], int], Callable[[], None]]:
        """Restituisce (leggi(offset, lunghezza, indirizzo_buffer), chiudi)."""
        if IS_WINDOWS:
            k, h = self._open_win()

            class _OVERLAPPED(ctypes.Structure):
                _fields_ = [
                    ("Internal", ctypes.c_void_p),
                    ("InternalHigh", ctypes.c_void_p),
                    ("Offset", ctypes.c_ulong),
                    ("OffsetHigh", ctypes.c_ulong),
                    ("hEvent", ctypes.c_void_p),
                ]

            k.ReadFile.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_ulong,
                ctypes.POINTER(ctypes.c_ulong),
                ctypes.POINTER(_OVERLAPPED),
            ]
            ov = _OVERLAPPED()
            got = ctypes.c_ulong(0)

            def read(offset: int, length: int, addr: int) -> int:
                # su un handle sincrono l'OVERLAPPED serve solo a dire "leggi
                # qui": la chiamata ritorna a lettura completata e il file
                # pointer condiviso non entra in gioco, quindi un handle per
                # thread basta e avanza.
                ov.Offset = offset & 0xFFFFFFFF
                ov.OffsetHigh = (offset >> 32) & 0xFFFFFFFF
                ok = k.ReadFile(
                    ctypes.c_void_p(h),
                    ctypes.c_void_p(addr),
                    length,
                    ctypes.byref(got),
                    ctypes.byref(ov),
                )
                if not ok:
                    err = ctypes.get_last_error()
                    if err != 38:  # ERROR_HANDLE_EOF
                        raise OSError(err, "ReadFile")
                return int(got.value)

            def close() -> None:
                k.CloseHandle.argtypes = [ctypes.c_void_p]
                k.CloseHandle(ctypes.c_void_p(h))

            return read, close

        fd = self._open_posix()
        libc = ctypes.CDLL(None, use_errno=True)
        libc.pread.restype = ctypes.c_ssize_t
        libc.pread.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_long]

        def read(offset: int, length: int, addr: int) -> int:
            n = libc.pread(fd, ctypes.c_void_p(addr), length, offset)
            if n < 0:
                raise OSError(ctypes.get_errno(), "pread")
            return int(n)

        def close() -> None:
            os.close(fd)

        return read, close


# ---------------------------------------------------------------------------
# il misuratore
# ---------------------------------------------------------------------------


def _sys_read_bytes() -> int | None:
    if psutil is None:
        return None
    try:
        c = psutil.disk_io_counters()
        return int(c.read_bytes) if c else None
    except Exception:
        return None


def _offsets(size: int, slice_bytes: int, seed: int) -> Callable[[], int]:
    """Generatore di offset casuali allineati al settore.

    Ogni thread riceve il proprio generatore con un seme diverso: se
    condividessero lo stesso stato si sincronizzerebbero sulle stesse pagine e
    la prova misurerebbe la cache invece del disco.
    """
    rnd = random.Random(seed)
    span = max(1, (size - slice_bytes) // SECTOR)

    def nxt() -> int:
        return rnd.randrange(span) * SECTOR

    return nxt


def _run_timed(
    n_threads: int,
    seconds: float,
    body: Callable[[int, Callable[[], bool]], tuple[int, int]],
) -> tuple[int, int, float]:
    """Esegue `body` su n thread per `seconds`, poi somma byte e operazioni.

    `body(indice_thread, ancora_attivo)` deve ciclare finche' `ancora_attivo()`
    e restituire (byte, operazioni).
    """
    stop = threading.Event()
    results: list[tuple[int, int]] = [(0, 0)] * n_threads
    errors: list[BaseException] = []
    ready = threading.Barrier(n_threads + 1)

    def alive() -> bool:
        return not stop.is_set()

    def worker(i: int) -> None:
        try:
            ready.wait()
            results[i] = body(i, alive)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
            stop.set()
            try:
                ready.wait(0.1)
            except Exception:
                pass

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(n_threads)]
    for t in threads:
        t.start()
    ready.wait()
    t0 = time.perf_counter()
    stop.wait(seconds)
    stop.set()
    for t in threads:
        t.join(timeout=30)
    elapsed = time.perf_counter() - t0
    if errors:
        raise errors[0]
    return sum(b for b, _ in results), sum(o for _, o in results), elapsed


def bench_mmap(path: Path, slice_bytes: int, threads: int, seconds: float) -> DiskResult:
    mapping = _Mapping(path)
    memmove = ctypes.memmove
    try:
        bufs = []
        addrs = []
        for _ in range(threads):
            keep, addr = _alloc_aligned(slice_bytes)
            bufs.append(keep)
            addrs.append(addr)
        base = mapping.base
        size = mapping.size

        def body(i: int, alive: Callable[[], bool]) -> tuple[int, int]:
            nxt = _offsets(size, slice_bytes, 1000 + i)
            dst = addrs[i]
            got = 0
            ops = 0
            while alive():
                off = nxt()
                # la copia tocca ogni pagina della slice, che e' esattamente
                # cio' che fa il kernel di ggml quando legge i pesi
                memmove(ctypes.c_void_p(dst), ctypes.c_void_p(base + off), slice_bytes)
                got += slice_bytes
                ops += 1
            return got, ops

        before = _sys_read_bytes()
        by, ops, el = _run_timed(threads, seconds, body)
        after = _sys_read_bytes()
    finally:
        mapping.close()
    phys = None if before is None or after is None else max(0, after - before)
    return DiskResult("mmap", slice_bytes, threads, by, el, phys, ops)


def bench_positioned(
    path: Path, slice_bytes: int, threads: int, seconds: float, direct: bool
) -> DiskResult:
    reader = _PositionedReader(path, direct=direct)
    size = reader.size
    workers = []
    bufs = []
    try:
        for _ in range(threads):
            read, close = reader.make_worker()
            keep, addr = _alloc_aligned(slice_bytes)
            workers.append((read, close, addr))
            bufs.append(keep)

        def body(i: int, alive: Callable[[], bool]) -> tuple[int, int]:
            read, _close, addr = workers[i]
            nxt = _offsets(size, slice_bytes, 2000 + i)
            got = 0
            ops = 0
            while alive():
                got += read(nxt(), slice_bytes, addr)
                ops += 1
            return got, ops

        before = _sys_read_bytes()
        by, ops, el = _run_timed(threads, seconds, body)
        after = _sys_read_bytes()
    finally:
        for _r, close, _a in workers:
            try:
                close()
            except Exception:
                pass
    phys = None if before is None or after is None else max(0, after - before)
    return DiskResult("direct" if direct else "pread", slice_bytes, threads, by, el, phys, ops)


def bench_moe(path: Path, slice_bytes: int, threads: int, seconds: float) -> DiskResult:
    """Il pattern vero: un'ondata di letture, poi una barriera.

    Le prove `direct` e `mmap` hanno parallelismo illimitato, il motore no. Per
    ogni layer servono 8 expert x 3 tensori = 24 slice, tutte prima di poter
    calcolare; finito il layer si ricomincia. Quindi la profondita' di coda
    utile e' limitata e, soprattutto, **conta la latenza della lettura piu'
    lenta**, non quella media: la barriera aspetta l'ultima.

    E' il tetto realistico. Se `direct` senza barriera fa 3 GB/s e questa ne fa
    1, il numero da usare per decidere e' 1.
    """
    reader = _PositionedReader(path, direct=True)
    size = reader.size
    workers = []
    bufs = []
    barriera = threading.Barrier(threads)
    try:
        for _ in range(threads):
            read, close = reader.make_worker()
            keep, addr = _alloc_aligned(slice_bytes)
            workers.append((read, close, addr))
            bufs.append(keep)

        def body(i: int, alive: Callable[[], bool]) -> tuple[int, int]:
            read, _close, addr = workers[i]
            nxt = _offsets(size, slice_bytes, 3000 + i)
            got = 0
            ops = 0
            while alive():
                got += read(nxt(), slice_bytes, addr)
                ops += 1
                try:
                    barriera.wait(timeout=1.0)
                except threading.BrokenBarrierError:
                    break
                except Exception:
                    break
            barriera.abort()  # libera chi sta ancora aspettando
            return got, ops

        before = _sys_read_bytes()
        by, ops, el = _run_timed(threads, seconds, body)
        after = _sys_read_bytes()
    finally:
        for _r, close, _a in workers:
            try:
                close()
            except Exception:
                pass
    phys = None if before is None or after is None else max(0, after - before)
    return DiskResult("moe", slice_bytes, threads, by, el, phys, ops)


def bench_sequential(path: Path, block: int, seconds: float) -> DiskResult:
    """Tetto nominale: lettura sequenziale a blocchi grandi, cache scavalcata."""
    reader = _PositionedReader(path, direct=True)
    read, close = reader.make_worker()
    keep, addr = _alloc_aligned(block)
    size = reader.size
    try:

        def body(i: int, alive: Callable[[], bool]) -> tuple[int, int]:
            off = 0
            got = 0
            ops = 0
            while alive():
                if off + block > size:
                    off = 0
                got += read(off, block, addr)
                off += block
                ops += 1
            return got, ops

        before = _sys_read_bytes()
        by, ops, el = _run_timed(1, seconds, body)
        after = _sys_read_bytes()
    finally:
        close()
    phys = None if before is None or after is None else max(0, after - before)
    return DiskResult("seq", block, 1, by, el, phys, ops)


# ---------------------------------------------------------------------------
# suite
# ---------------------------------------------------------------------------

DEFAULT_SLICES = (917504, 2883584)  # 896 KiB (un tensore expert), 2.75 MiB (la tripletta)
DEFAULT_THREADS = (1, 4, 8, 16, 32)
DEFAULT_KINDS = ("mmap", "pread", "direct", "moe")
ALL_KINDS = ("mmap", "pread", "direct", "moe")


def purge_cache(nbytes: int) -> int:
    """Svuota la page cache occupando memoria privata e rilasciandola.

    Non esiste un modo non privilegiato di dire a Windows "dimentica questo
    file". Chiedere molta memoria privata ottiene lo stesso effetto per vie
    indirette: il memory manager taglia la standby list per soddisfarci, e le
    pagine del modello sono la parte piu' grande di quella lista.

    Serve solo a rendere onesta la prova `mmap`; `direct` non ne ha bisogno.
    Restituisce i byte effettivamente toccati.
    """
    if nbytes <= 0:
        return 0
    pezzo = 256 << 20
    toccati = 0
    tenuti: list = []
    try:
        while toccati < nbytes:
            n = min(pezzo, nbytes - toccati)
            try:
                keep, addr = _alloc_aligned(n)
            except Exception:
                break
            ctypes.memset(ctypes.c_void_p(addr), 1, n)
            tenuti.append(keep)
            toccati += n
    finally:
        for k in tenuti:
            try:
                k.close()
            except Exception:
                pass
        tenuti.clear()
    return toccati


@dataclass
class Suite:
    path: Path
    slices: Sequence[int] = DEFAULT_SLICES
    threads: Sequence[int] = DEFAULT_THREADS
    kinds: Sequence[str] = DEFAULT_KINDS
    seconds: float = 4.0
    sequential: bool = True
    purge: int = 0
    results: list[DiskResult] = field(default_factory=list)

    def run(self, progress: Callable[[str], None] | None = None) -> list[DiskResult]:
        def say(msg: str) -> None:
            if progress:
                progress(msg)

        if self.sequential:
            say("sequenziale (tetto del dispositivo)...")
            self.results.append(bench_sequential(self.path, 8 << 20, self.seconds))
        for kind in self.kinds:
            for sl in self.slices:
                for th in self.threads:
                    if self.purge and kind != "direct":
                        say(f"svuoto la cache ({fmt_bytes(self.purge)})...")
                        purge_cache(self.purge)
                    say(f"{kind}: slice {fmt_bytes(sl)}, {th} thread...")
                    try:
                        if kind == "mmap":
                            r = bench_mmap(self.path, sl, th, self.seconds)
                        elif kind == "moe":
                            r = bench_moe(self.path, sl, th, self.seconds)
                        else:
                            r = bench_positioned(
                                self.path, sl, th, self.seconds, direct=(kind == "direct")
                            )
                    except Exception as exc:  # noqa: BLE001
                        r = DiskResult(kind, sl, th, 0, 0.0, None, 0, error=str(exc))
                    self.results.append(r)
        return self.results

    def best(self, kind: str) -> DiskResult | None:
        cands = [r for r in self.results if r.kind == kind and not r.error]
        return max(cands, key=lambda r: r.mb_s) if cands else None

    # il migliore throughput di disco mai osservato durante la generazione vera,
    # con la memoria mappata di llama.cpp (build MSVC ufficiale, Q4_K_M 17.35 GiB)
    PRODUZIONE_MB_S = 562.0

    def verdict(self, coda_osservata: Optional[float] = None) -> list[str]:
        """Tradurre la misura in una decisione, non in una curiosita'.

        Il confronto che decide e' `direct` contro il throughput gia' osservato
        in produzione, perche':

          - `direct` scavalca la cache per costruzione, quindi il suo numero e'
            un dato sul dispositivo e non sulla RAM;
          - i 562 MB/s di produzione sono il meglio che il percorso mmap ha
            saputo fare sul carico vero, che e' proprio la grandezza che
            vogliamo superare.

        La prova `mmap` sintetica resta come controllo, ma su una macchina dove
        il modello sta quasi tutto in page cache il suo throughput logico dice
        quanto e' veloce la RAM. Per questo non viene usata per il verdetto a
        meno che non abbia davvero mancato la cache.
        """
        out: list[str] = []
        mm = self.best("mmap")
        dr = self.best("direct")
        pr = self.best("pread")
        sq = self.best("seq")
        mo = self.best("moe")

        if sq:
            out.append(
                f"tetto sequenziale del dispositivo: {sq.mb_s:.0f} MB/s "
                f"({sq.mb_s / self.PRODUZIONE_MB_S:.1f}x il throughput visto in generazione)"
            )
        # il numero che decide e' quello col pattern realistico, se c'e'
        deciso = mo or dr
        if deciso is None:
            out.append(
                "nessuna prova con I/O diretto e' andata a buon fine: senza quella non c'e' "
                "verdetto, perche' e' la sola immune alla page cache"
            )
            if mm:
                out.append(f"  (mmap ha dato {mm.mb_s:.0f} MB/s logici, cache inclusa)")
            return out

        if dr:
            out.append(
                f"letture esplicite senza cache: {dr.mb_s:.0f} MB/s con {dr.threads} thread, "
                f"slice {fmt_bytes(dr.slice_bytes)}, {dr.iops:.0f} IOPS"
            )
        if mo:
            out.append(
                f"stesse letture con la barriera per layer (pattern reale): {mo.mb_s:.0f} MB/s "
                f"con {mo.threads} thread"
                + (
                    f", cioe' il {mo.mb_s / dr.mb_s * 100:.0f}% di quanto si ottiene senza "
                    "sincronizzazione"
                    if dr and dr.mb_s > 0
                    else ""
                )
            )
        if mm:
            nota = ""
            c = mm.cache_ratio
            if c is not None and c > 0.25:
                nota = f" (ma il {c * 100:.0f}% veniva dalla page cache: numero sulla RAM)"
            out.append(f"memoria mappata: {mm.mb_s:.0f} MB/s logici{nota}")
        if pr:
            out.append(f"letture posizionate con cache: {pr.mb_s:.0f} MB/s logici")

        rapporto = deciso.mb_s / self.PRODUZIONE_MB_S
        out.append(
            f"in generazione il percorso mmap di llama.cpp ha reso {self.PRODUZIONE_MB_S:.0f} MB/s "
            f"sullo stesso disco: il pattern '{deciso.kind}' rende {rapporto:.1f}x tanto"
        )
        if rapporto >= 1.8:
            out.append(
                "VERDETTO: il disco sa fare molto piu' di quanto il percorso di page fault "
                "riesca a chiedergli. Il collo di bottiglia e' software e sta dove pensavo: "
                f"il buffer type con I/O esplicito ha un margine fino a {rapporto:.1f}x sui "
                "byte per secondo, quindi vale il lavoro. Il guadagno finale sara' inferiore "
                "(resta il calcolo da fare), ma il tetto e' reale."
            )
        elif rapporto <= 1.2:
            out.append(
                "VERDETTO: il disco non ha piu' niente da dare con questo pattern. I "
                f"{self.PRODUZIONE_MB_S:.0f} MB/s osservati sono il dispositivo, non il "
                "software: riscrivere il percorso di I/O non produrrebbe un solo token al "
                "secondo in piu'. La mia ipotesi sul page fault e' sbagliata, e il buffer "
                "type va cancellato, non rimandato. L'unica leva che resta e' ridurre i byte "
                "per token: quantizzazione piu' aggressiva degli expert, doppia precisione, "
                "decodifica speculativa (che genera piu' token con le stesse letture)."
            )
        else:
            out.append(
                f"VERDETTO: margine stretto ({rapporto:.1f}x). Non abbastanza per giustificare "
                "un buffer type da solo: prima conviene la decodifica speculativa, che "
                "moltiplica i token a letture invariate ed e' gia' implementata."
            )
        out.extend(self._leve(deciso, coda_osservata))
        return out

    # Sopra questa profondita' di coda la leva del parallelismo e' in parte
    # disponibile: ci sono letture da sovrapporre. Sotto, non c'e' niente da
    # sovrapporre e conta solo come si legge.
    CODA_PARALLELISMO = 1.5

    def _leve(self, deciso: DiskResult,
              coda_osservata: Optional[float] = None) -> list[str]:
        """Quale delle due leve conviene tirare, *nel regime in cui si lavora*.

        Due versioni sbagliate prima di questa, e vale la pena che restino scritte
        perche' sono lo stesso errore due volte.

        La prima guardava solo il parallelismo: se da 1 a N thread il throughput
        saliva di almeno 1,3x concludeva "conviene emettere piu' letture insieme
        piuttosto che letture piu' grandi". Misura giusta, consiglio inservibile:
        descriveva il dispositivo a 32 thread.

        La seconda — scritta per correggere la prima, lo stesso giorno — inchiodava
        una costante, 0,75 miss per nodo, e concludeva che la coda reale e' ~1 e il
        parallelismo non e' disponibile. Quella costante veniva da una misura al 49%
        di residenza. Misurata ai budget che contano per un modello piu' grande, la
        coda vale **2,09 a 4 GiB e 2,49 a 1,5 GiB**, con 2,30 e 4,99 miss per nodo:
        tre volte e sei volte la costante. Avevo sostituito un verdetto tarato sul
        regime sbagliato con un verdetto tarato su un altro regime sbagliato, nella
        stessa funzione.

        La morale e' che il regime non e' una costante del programma: e' un dato
        della misura. Quindi ora si passa da fuori, e senza di esso il verdetto non
        si pronuncia. Il numero lo stampa lo store con `--expert-stats`
        ("profondita' di coda media").
        """
        out: list[str] = []

        def a_un_thread(slice_bytes: int) -> Optional[DiskResult]:
            return next((r for r in self.results
                         if r.kind == deciso.kind and r.threads == 1
                         and r.slice_bytes == slice_bytes and not r.error
                         and r.mb_s > 0), None)

        # leva A: profondita' di coda, misurata sullo stesso slice
        uno = a_un_thread(deciso.slice_bytes)
        scala_thread = (deciso.mb_s / uno.mb_s) if (uno and deciso.threads > 1) else None
        if scala_thread is not None:
            out.append(
                f"leva 'piu' letture insieme': da 1 a {deciso.threads} thread si passa da "
                f"{uno.mb_s:.0f} a {deciso.mb_s:.0f} MB/s ({scala_thread:.1f}x)")

        # leva B: dimensione della lettura, a un solo thread (il regime reale)
        a_un = [r for r in self.results
                if r.kind == deciso.kind and r.threads == 1 and not r.error and r.mb_s > 0]
        scala_slice = None
        if len(a_un) >= 2:
            piccola = min(a_un, key=lambda r: r.slice_bytes)
            grande = max(a_un, key=lambda r: r.slice_bytes)
            if piccola.slice_bytes != grande.slice_bytes:
                scala_slice = grande.mb_s / piccola.mb_s
                out.append(
                    f"leva 'letture piu' grandi': a un thread, da {fmt_bytes(piccola.slice_bytes)} "
                    f"a {fmt_bytes(grande.slice_bytes)} si passa da {piccola.mb_s:.0f} a "
                    f"{grande.mb_s:.0f} MB/s ({scala_slice:.2f}x)")

        if scala_thread is None and scala_slice is None:
            return out

        if coda_osservata is None:
            out.append(
                "quale leva sia disponibile dipende dalla profondita' di coda a cui si "
                "lavora, che e' un dato della tua esecuzione e non di questo benchmark: "
                "la stampa lo store con --expert-stats ('profondita' di coda media'). "
                "Rilancia con --coda N per avere il verdetto.")
            return out

        # Le due leve non sono indipendenti, e presentarle come tali e' stato il
        # terzo difetto di questo verdetto. A byte totali fissi per nodo, letture
        # piu' grandi sono anche *meno numerose*, quindi alzare la dimensione
        # abbassa la coda: non si sommano, si scambiano. Il modo onesto di dirlo e'
        # indicare la cella migliore della tabella, dove la combinazione e' una.
        migliore = max((r for r in self.results
                        if r.kind == deciso.kind and not r.error and r.mb_s > 0),
                       key=lambda r: r.mb_s, default=None)
        if migliore is not None:
            out.append(
                f"il massimo della tabella e' {migliore.mb_s:.0f} MB/s con slice "
                f"{fmt_bytes(migliore.slice_bytes)} e {migliore.threads} thread. "
                f"Attenzione: a byte totali fissi per nodo le due leve si scambiano — "
                f"letture piu' grandi sono anche meno numerose, quindi alzare la "
                f"dimensione abbassa la coda.")

        out.append(f"alla profondita' di coda osservata ({coda_osservata:.2f}):")
        if coda_osservata >= self.CODA_PARALLELISMO:
            if scala_thread is not None and scala_thread >= 1.3:
                out.append(
                    f"  entrambe le leve sono in gioco: ci sono letture da sovrapporre, e il "
                    f"dispositivo le premia ({scala_thread:.1f}x da 1 a {deciso.threads} "
                    f"thread). La coda cresce da sola col calare della residenza, quindi su un "
                    f"modello piu' grande questa leva si tira da se'.")
            if scala_slice is not None and scala_slice >= 1.1:
                out.append(
                    f"  e le letture grandi rendono comunque ({scala_slice:.2f}x): tenere "
                    f"contigue le tre matrici di un expert resta giusto.")
        else:
            if scala_slice is not None and scala_slice >= 1.1:
                out.append(
                    f"  LEVA DA TIRARE: letture piu' grandi ({scala_slice:.2f}x). A questa coda "
                    f"non c'e' niente da sovrapporre, qualunque cosa il dispositivo sappia fare "
                    f"a {deciso.threads} thread.")
            elif scala_thread is not None and scala_thread >= 1.3:
                out.append(
                    f"  NESSUNA LEVA UTILE QUI: il dispositivo scala con la coda "
                    f"({scala_thread:.1f}x) ma a questa coda non arrivi, e le letture grandi non "
                    f"rendono "
                    + (f"({scala_slice:.2f}x)" if scala_slice is not None else "(non misurato)")
                    + ". Il guadagno va cercato nel ridurre i miss.")
            else:
                out.append(
                    "  NESSUNA DELLE DUE LEVE rende molto: un solo thread quasi satura il "
                    "dispositivo e le letture grandi non aggiungono. Quello che paga e' non "
                    "passare per il page fault, ed e' gia' fatto.")
        return out


def format_results(results: Iterable[DiskResult]) -> list[str]:
    righe = [
        f"  {'percorso':<8} {'slice':>9} {'thread':>6} {'MB/s':>7} "
        f"{'disco':>7} {'IOPS':>7} {'lat ms':>7} {'cache':>6}"
    ]
    for r in results:
        if r.error:
            testo = r.error.splitlines()[0][:48]
            righe.append(f"  {r.kind:<8} {fmt_bytes(r.slice_bytes):>9} {r.threads:>6}   {testo}")
            continue
        phys = f"{r.phys_mb_s:.0f}" if r.phys_mb_s is not None else "-"
        cache = f"{r.cache_ratio * 100:.0f}%" if r.cache_ratio is not None else "-"
        righe.append(
            f"  {r.kind:<8} {fmt_bytes(r.slice_bytes):>9} {r.threads:>6} "
            f"{r.mb_s:>7.0f} {phys:>7} {r.iops:>7.0f} {r.latency_ms:>7.2f} {cache:>6}"
        )
    return righe
