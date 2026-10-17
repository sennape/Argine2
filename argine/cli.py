"""CLI di Argine: probe / inspect / plan / run / benchmark / experts / setup / logs."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import __version__
from . import bench as benchmod
from . import diskbench as dbmod
from . import expert_cache as ec
from . import expert_store as esmod
from . import hardware as hwmod
from . import planner as planmod
from . import quality as qmod
from . import quantize as qzmod
from . import reorder as reordermod
from . import runtime as rtmod
from . import server as srv
from . import telemetry
from .gguf import GGUFError, GGUFModel, find_gguf, read_gguf
from .util import (GB, MB, bold, cyan, dim, error, fmt_bytes, fmt_count, fmt_gb,
                   green, header, info, is_windows, kv, logs_dir, ok, parse_size,
                   red, runtimes_dir, warn, yellow)

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_INFEASIBLE = 3
EXIT_RUNTIME = 4


# ============================================================== formatting ===

def _print_hw(hw: hwmod.HardwareReport, devices: Optional[List[str]] = None) -> None:
    print(header("SISTEMA"))
    print(kv("OS", f"{hw.os_name} {hw.os_version}"))
    print(kv("Python", hw.python))
    print(kv("Argine", __version__))

    print(header("CPU"))
    print(kv("modello", hw.cpu_brand))
    print(kv("core / thread", f"{hw.cpu_cores_physical or '?'} core / {hw.cpu_cores_logical or '?'} thread"))
    if hw.is_hybrid_cpu and hw.cpu_p_cores:
        print(kv("architettura", f"ibrida: {hw.cpu_p_cores} P-core + {hw.cpu_e_cores} E-core"))
    if hw.cpu_freq_mhz:
        print(kv("frequenza", f"{hw.cpu_freq_mhz/1000:.2f} GHz"))
    if hw.cpu_flags:
        print(kv("SIMD", ", ".join(hw.cpu_flags)))

    print(header("MEMORIA"))
    print(kv("RAM totale", fmt_bytes(hw.ram_total)))
    print(kv("RAM libera", fmt_bytes(hw.ram_available)))
    if hw.swap_total:
        print(kv("file di paging", fmt_bytes(hw.swap_total)))
    if hw.gpu_shared_memory_budget:
        print(kv("GPU shared (max)", fmt_bytes(hw.gpu_shared_memory_budget) + dim("  (presa dalla RAM)")))
    budget, notes = planmod.safe_budget(hw, None)
    print(kv("budget consigliato", green(fmt_bytes(budget))))

    print(header("GPU"))
    if not hw.gpus:
        print(kv("", dim("nessuna GPU rilevata")))
    for g in hw.gpus:
        tag = "integrata" if g.integrated else "dedicata"
        extra = f", VRAM {fmt_bytes(g.dedicated_vram)}" if g.dedicated_vram else ""
        drv = f", driver {g.driver_version}" if g.driver_version else ""
        print(kv(g.name, f"{tag}{extra}{drv}"))
    print(kv("runtime Vulkan", green("presente") if hw.vulkan_runtime else red("assente")))
    if hw.vulkan_icds:
        print(kv("ICD Vulkan", ", ".join(Path(p).name for p in hw.vulkan_icds[:4])))
    if devices:
        print(kv("device llama.cpp", ""))
        for d in devices:
            print(f"      - {d}")

    print(header("BACKEND DISPONIBILI"))
    print(kv("", ", ".join(green(b) for b in hw.backends)))

    print(header("DISCO"))
    for d in hw.disks:
        print(kv(d.path, f"{fmt_bytes(d.free)} liberi su {fmt_bytes(d.total)}"))

    if hw.notes:
        print(header("NOTE"))
        for n in hw.notes:
            print(f"  {yellow('!')} {n}")
    print()


def _print_model(m: GGUFModel, plan: Optional[planmod.RuntimePlan] = None) -> None:
    print(header("MODELLO"))
    print(kv("file", m.path))
    if m.shards:
        print(kv("shard", f"{len(m.shards)} file, totale {fmt_bytes(m.total_size_on_disk)}"))
    print(kv("dimensione", fmt_bytes(m.total_size_on_disk)))
    print(kv("nome", m.name))
    print(kv("architettura", m.architecture))
    print(kv("GGUF", f"v{m.gguf_version}, {m.tensor_count} tensori"))

    print(header("QUANTIZZAZIONE"))
    print(kv("dominante", m.quantization))
    if m.bits_per_weight:
        print(kv("bit per peso", f"{m.bits_per_weight:.2f} bpw"))
    for name, count, nbytes in m.quant_mix[:6]:
        share = nbytes / m.tensor_bytes * 100 if m.tensor_bytes else 0
        print(kv(f"  {name}", f"{count} tensori, {fmt_bytes(nbytes)} ({share:.1f}%)"))

    print(header("PARAMETRI"))
    print(kv("totali", fmt_count(m.n_params_total)))
    print(kv("attivi per token", fmt_count(m.n_params_active)))
    print(kv("layer", str(m.n_layers)))
    print(kv("embedding", str(m.n_embd)))
    print(kv("head (Q / KV)", f"{m.n_head} / {m.n_head_kv}"))
    print(kv("head dim (K / V)", f"{m.head_dim_k} / {m.head_dim_v}"))
    print(kv("context di training", str(m.n_ctx_train)))

    print(header("MIXTURE OF EXPERTS"))
    if not m.is_moe:
        print(kv("MoE", "no (modello dense)"))
    else:
        print(kv("MoE", green("si")))
        print(kv("expert totali", str(m.expert_count)))
        print(kv("expert per token", str(m.expert_used_count)))
        if m.expert_shared_count:
            print(kv("shared expert", str(m.expert_shared_count)))
        print(kv("layer MoE", str(m.n_moe_layers)))
        print(kv("tensori shared", fmt_bytes(m.shared_bytes)))
        print(kv("tensori expert", fmt_bytes(m.expert_bytes)))
        print(kv("byte per expert", fmt_bytes(m.expert_bytes_per_expert)))
        print(kv("byte expert/layer", fmt_bytes(m.expert_bytes_per_expert_per_layer)))

    if plan is not None:
        _print_plan(plan, m)
    print()


def _print_plan(p: planmod.RuntimePlan, m: GGUFModel) -> None:
    print(header("PIANO DI MEMORIA"))
    print(kv("budget RAM", bold(fmt_bytes(p.budget))))
    for label, value in p.estimate.rows():
        if label == "expert cache" and p.moe_streaming:
            label = "expert residenti (stimati)"
        print(kv(f"  {label}", fmt_bytes(value)))
    total = p.estimate.total
    pct = total / p.budget * 100 if p.budget else 0
    color = green if pct <= 92 else yellow if pct <= 100 else red
    print(kv("  TOTALE stimato", color(f"{fmt_bytes(total)}  ({pct:.0f}% del budget)")))
    if p.estimate.gpu_shared:
        print(kv("  di cui nel buffer GPU", fmt_bytes(p.estimate.gpu_shared)
                 + dim("  (shared memory, non aggiuntiva)")))
    if p.estimate.gpu_dedicated:
        print(kv("  spostati in VRAM", fmt_bytes(p.estimate.gpu_dedicated)))

    print(header("CONFIGURAZIONE RUNTIME"))
    print(kv("backend", p.backend))
    print(kv("context", str(p.ctx)))
    print(kv("batch / ubatch", f"{p.batch} / {p.ubatch}"))
    print(kv("threads", f"{p.threads} (batch {p.threads_batch})"))
    print(kv("KV cache", f"K={p.kv_type_k.upper()} V={p.kv_type_v.upper()}"))
    print(kv("flash attention", "on" if p.flash_attn else "off"))
    print(kv("gpu-layers", str(p.gpu_layers)))
    print(kv("mmap / mlock", f"{'on' if p.mmap else 'off'} / {'on' if p.mlock else 'off'}"))
    if getattr(p, "_prefetch_depth", None):
        print(kv("prefetch expert", f"{p._prefetch_depth} layer avanti"))
    if p.spec_type:
        dettaglio = f"{p.spec_type}, fino a {p.draft_n_max} token"
        if p.draft_model:
            dettaglio += f", draft {Path(p.draft_model).name}"
        print(kv("speculativa", dettaglio))
    if m.is_moe:
        print(kv("expert su CPU", "si" if p.cpu_moe else "no"))
        print(kv("expert streaming", "si" if p.moe_streaming else "no (tutti residenti)"))
        if p.expert_cache_budget:
            cov = p.expert_cache_budget / m.expert_bytes if m.expert_bytes else 0
            print(kv("expert residenti (stima)",
                     f"{fmt_bytes(p.expert_cache_budget)} ({cov*100:.0f}% degli expert)"))
            # Questa riga nasce prima dell'expert store, quando la residenza era
            # davvero fuori dal nostro controllo. Con lo store attivo diceva
            # l'esatto contrario del vero — ed e' la riga che l'utente legge per
            # capire se il tetto c'e'. Dal pre-volo di uno sweep si vedeva
            # "expert residenti 8.00 GiB" seguito da "residenza governata da
            # mmap", mentre il tetto era nostro e rigido.
            if p.expert_store:
                print(kv("", dim("tetto rigido: la residenza e' governata dallo "
                                 "store, non dalla page cache dell'OS")))
            else:
                print(kv("", dim("residenza governata da mmap + page cache "
                                 "dell'OS, non da Argine: e' una stima, non un "
                                 "tetto (lo diventa con --expert-store)")))

    if p.decisions:
        print(header("DECISIONI DEL PLANNER"))
        for d in p.decisions:
            mark = yellow("~") if d.startswith("[safety]") else dim("-")
            print(f"  {mark} {d}")
    if p.warnings:
        print(header("AVVISI"))
        for w in p.warnings:
            print(f"  {yellow('!')} {w}")
    if not p.feasible:
        print()
        error(f"configurazione NON avviabile: {p.failure_reason}")


# =================================================================== probe ===

def cmd_probe(args: argparse.Namespace) -> int:
    hw = hwmod.detect()
    devices: List[str] = []
    if args.deep:
        for backend in ("vulkan", "cpu"):
            try:
                rt = rtmod.ensure_runtime(backend, quiet=True, offline=args.offline)
            except Exception as e:
                if backend == "vulkan":
                    hw.notes.append(f"runtime vulkan non verificabile: {e}")
                continue
            devs = rt.list_devices()
            devices.extend(f"[{backend}] {d}" for d in devs)
            if backend == "vulkan" and not devs:
                hw.notes.append("il binario Vulkan non elenca device: il backend ricadra' su CPU")
                if "vulkan" in hw.backends:
                    hw.backends.remove("vulkan")
            break
    if args.json:
        payload = hw.to_dict()
        payload["llama_devices"] = devices
        budget, notes = planmod.safe_budget(hw, None)
        payload["recommended_budget"] = budget
        payload["budget_notes"] = notes
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    else:
        _print_hw(hw, devices)
    return EXIT_OK


# ================================================================= inspect ===

def _load_model(path: str) -> GGUFModel:
    p = find_gguf(path)
    return read_gguf(p)


def cmd_inspect(args: argparse.Namespace) -> int:
    m = _load_model(args.model)
    hw = hwmod.detect(extra_disk_paths=[str(Path(m.path).parent)])
    backend, msgs = hwmod.resolve_backend(args.backend, hw)
    req = planmod.PlanRequest(
        ctx=args.ctx, ram_limit=parse_size(args.ram_limit), backend=backend,
        kv_type=args.kv_type, gpu_layers=args.gpu_layers,
        expert_cache=parse_size(args.expert_cache),
        spec=getattr(args, "spec", None),
        draft_model=getattr(args, "draft_model", None),
        draft_n_max=getattr(args, "draft_n_max", None),
        draft_ctx=getattr(args, "draft_ctx", None),
    )
    p = planmod.plan(m, hw, req)
    for msg in msgs:
        p.decisions.insert(0, msg)

    if args.json:
        print(json.dumps({
            "model": telemetry._model_summary(m),
            "metadata_keys": sorted(k for k in m.metadata),
            "plan": p.to_dict(),
            "hardware": telemetry._hw_summary(hw),
        }, indent=2, ensure_ascii=False, default=str))
    else:
        _print_model(m, p)
        if m.is_moe:
            res = ec.residency_plan(m, p.expert_cache_budget or m.expert_bytes)
            print(header("RESIDENZA EXPERT"))
            print(kv("slot in cache", str(res["cache_slots"])))
            print(kv("slot per layer", f"{res['slots_per_layer']} su {res['experts']} expert"))
            print(kv("copertura", f"{res['coverage']*100:.1f}%"))
            print(kv("byte 'cold' su SSD", fmt_bytes(res["cold_bytes"])))
            print()
    return EXIT_OK if p.feasible else EXIT_INFEASIBLE


# ==================================================================== plan ===

def cmd_plan(args: argparse.Namespace) -> int:
    return cmd_inspect(args)


# ===================================================================== run ===

def _degrade(p: planmod.RuntimePlan, m: GGUFModel, hw: hwmod.HardwareReport) -> Optional[str]:
    """Applica un ulteriore gradino di sicurezza dopo un fallimento reale."""
    if p.gpu_layers > 0:
        layers = m.n_layers or 32
        newv = 0 if p.gpu_layers <= max(1, layers // 4) else p.gpu_layers // 2
        p.gpu_layers = newv
        return f"gpu-layers -> {newv}"
    if p.kv_type_k == "f16":
        p.kv_type_k = p.kv_type_v = "q8_0"
        p.flash_attn = True
        return "KV cache -> Q8_0"
    if p.kv_type_k == "q8_0":
        p.kv_type_k = p.kv_type_v = "q4_0"
        return "KV cache -> Q4_0"
    for c in planmod.CTX_LADDER:
        if c < p.ctx:
            p.ctx = c
            return f"context -> {c}"
    for b in planmod.BATCH_LADDER:
        if b < p.batch:
            p.batch = b
            p.ubatch = min(p.ubatch, b)
            return f"batch -> {b}"
    if not p.mmap:
        p.mmap = True
        return "mmap -> on"
    if m.is_moe and not p.cpu_moe:
        p.cpu_moe = True
        p.n_cpu_moe = m.n_moe_layers or (m.n_layers or 0)
        return "expert -> CPU"
    return None


def _start_stack(m: GGUFModel, hw: hwmod.HardwareReport, p: planmod.RuntimePlan,
                 rt: rtmod.LlamaRuntime, args: argparse.Namespace,
                 gateway_port: Optional[int]) -> tuple[srv.LlamaServerProcess, int]:
    upstream = srv.free_port()
    llama_args = srv.build_llama_args(rt, m.path, p, "127.0.0.1", upstream,
                                      extra=args.llama_arg or None)
    if args.verbose:
        info("comando: " + " ".join(llama_args))
    env = {}
    route_log = getattr(args, "route_log", None)
    if route_log:
        env["ARGINE_ROUTE_LOG"] = str(Path(route_log).expanduser())
    prefetch = getattr(args, "prefetch", None)
    if prefetch is not None:
        env["ARGINE_PREFETCH"] = str(max(0, int(prefetch)))
    pf_threads = getattr(args, "prefetch_threads", None)
    if pf_threads is not None:
        env["ARGINE_PREFETCH_THREADS"] = str(max(1, int(pf_threads)))
    env.update(_expert_store_env(args, m, rt, p))
    # Argine2: research instrumentation (patch 0003). Observation only.
    research_trace = getattr(args, "research_trace", None)
    if research_trace:
        env["ARGINE_RESEARCH_TRACE"] = str(Path(research_trace).expanduser())
    proc = srv.LlamaServerProcess(llama_args, verbose=args.verbose, env=env or None)
    proc.start()
    return proc, upstream


def _decidi_repack(args: argparse.Namespace) -> Optional[bool]:
    """Repack acceso o spento, tenendo conto di un conflitto in llama.cpp.

    `--no-repack` in llama.cpp non spegne soltanto il repack: imposta
    `no_extra_bufts`, che **disabilita tutti gli extra buffer type**. L'expert
    store e' un extra buffer type, quindi passare `--no-repack` lo spegne.
    Restano aperti i file e le variabili d'ambiente, ma nessun tensore gli viene
    assegnato: lo store sembra attivo e non serve un byte.

    Il costo di lasciare il repack acceso, misurato su Qwen3-30B-A3B Q4_K_M con
    lo store attivo: 229.50 MiB, perche' gli expert appartengono a noi e al
    repack restano solo i tensori shared. 230 MiB per servirne 16350 dallo store
    e' un cambio conveniente, quindi con `--expert-store` il repack resta
    acceso, a meno che non lo si chieda esplicitamente.
    """
    if getattr(args, "no_repack", False):
        if getattr(args, "expert_store", None):
            warn("--no-repack e --expert-store sono incompatibili: in llama.cpp "
                 "--no-repack disabilita tutti gli extra buffer type, store compreso. "
                 "Tengo il repack acceso (costa ~230 MiB sui soli tensori shared).")
            return True
        return False
    if getattr(args, "repack", False):
        return True
    if getattr(args, "expert_store", None):
        # il default di Argine e' repack spento; con lo store va invertito,
        # altrimenti lo store non riceve nessun tensore
        return True
    return None


def _avvisa_build_cambiata(m: GGUFModel, rt: rtmod.LlamaRuntime) -> None:
    """Dice che il binario e' cambiato rispetto all'ultima misura dello stesso file.

    Serve perche' confrontare due velocita' prese su build diverse e' un errore
    silenzioso, e l'ho commesso: su qwen30b-IQ2S, stesso file e stessi flag, la
    build ufficiale b11214 ha dato 13.53 tok/s e la nostra patchata 7.45. Per
    mezz'ora ho attribuito quel 1.82x alla quantizzazione, perche' aprendo un
    terminale nuovo si era persa ARGINE_LLAMA_DIR e l'unico indizio era una
    parola in fondo a una riga di log.
    """
    try:
        prec = telemetry.ultima_build(str(m.path), escludi_build=str(rt.build))
    except Exception:  # noqa: BLE001
        return
    if not prec:
        return
    warn(f"la build e' cambiata: l'ultima misura di questo modello era su "
         f"{prec['build']}, questa e' su {rt.build}")
    if prec.get("tok_s"):
        print(dim(f"  allora: {prec['tok_s']} tok/s ({prec.get('quando') or '?'}). "
                  f"I due numeri NON sono confrontabili: cambia il binario, non "
                  f"il modello."))
    # Chiedere al binario, non indovinare dall'etichetta. La versione
    # precedente confrontava rt.build con la stringa "ARGINE_LLAMA_DIR", e
    # quando ho reso quell'etichetta piu' informativa la condizione e' diventata
    # sempre vera: annunciava "questa build non contiene l'expert store" a una
    # build che lo conteneva. Una riga scritta per segnalare le affermazioni non
    # verificate, che ne faceva una.
    try:
        ha_store = rt.supports_expert_store()
    except Exception:  # noqa: BLE001
        return
    if not ha_store:
        print(dim("  (questa build non contiene l'expert store: --expert-store "
                  "non avrebbe effetto)"))


def _piano_per_misura(m: GGUFModel, hw: hwmod.HardwareReport,
                      args: argparse.Namespace) -> Optional[planmod.RuntimePlan]:
    """Un piano anche per i comandi di misura (`ppl`, `imatrix`).

    Non serve a lanciare llama-server ma a due cose che senza di lui sbagliano:
    il numero di thread, che su CPU ibride non e' "tutti", e soprattutto il
    budget della cache degli expert. Senza piano `_expert_store_env` non trova
    nessun budget, non imposta ARGINE_EXPERT_BUDGET, e sul lato C++ zero
    significa *nessun tetto*: la cache cresce finche' contiene tutti gli expert.
    Lo store continua a funzionare, ma la cosa per cui esiste — il tetto rigido
    — sparisce, e la riga che dice "dentro il budget di RAM" diventa falsa.
    """
    try:
        return planmod.plan(m, hw, planmod.PlanRequest(
            backend="cpu", ctx=getattr(args, "ctx", None),
            ram_limit=parse_size(getattr(args, "ram_limit", None)),
            expert_store=bool(getattr(args, "expert_store", None)),
            expert_store_budget=parse_size(getattr(args, "expert_budget", None)),
        ))
    except Exception:  # noqa: BLE001
        return None


# I tre strati di lettura, in ordine di comparsa. Sono selezionabili a runtime
# perche' il loro confronto e' un risultato del progetto, e un risultato che
# nessuno puo' rifare e' un'opinione: le prime misure erano su tre binari diversi
# in tre momenti diversi, e due di quei binari non esistono piu'.
MODI_LETTURA = ("sincrono", "blocco", "asincrono")


def _modo_lettura(v: Optional[str]) -> str:
    """Normalizza il modo di lettura. Accetta `blocco:4` oltre a `blocco`."""
    if not v:
        return "asincrono"
    s = str(v).strip().lower()
    base = s.split(":", 1)[0]
    if base not in MODI_LETTURA:
        error(f"--read-mode: {v} non e' un modo di lettura "
              f"({', '.join(MODI_LETTURA)}, oppure blocco:N)")
        raise SystemExit(EXIT_USAGE)
    if ":" in s and base != "blocco":
        error(f"--read-mode: solo `blocco` prende una dimensione, non `{base}`")
        raise SystemExit(EXIT_USAGE)
    if base == "blocco" and ":" in s:
        try:
            n = int(s.split(":", 1)[1])
        except ValueError:
            error(f"--read-mode: `{s}` non ha un numero dopo i due punti")
            raise SystemExit(EXIT_USAGE)
        if not 1 <= n <= 8:
            error(f"--read-mode blocco:{n}: la dimensione va fra 1 e 8")
            raise SystemExit(EXIT_USAGE)
    return s


def _conferma_modo_lettura(st: Optional[Dict[str, Any]], chiesto: str) -> bool:
    """Il motore ha usato il modo che gli abbiamo chiesto?

    Si guarda il log, non la variabile d'ambiente che abbiamo impostato noi. E'
    la stessa regola che vale per lo store: una misura etichettata con cio' che
    si e' chiesto invece di cio' che e' avvenuto e' il modo piu' economico di
    produrre una tabella sbagliata.
    """
    visto = (st or {}).get("lettura")
    if not visto:
        # Qui la prima versione tornava True con la scusa "nessun log da cui
        # dedurlo". Era un buco, e la realta' lo ha infilato il giorno stesso: il
        # binario davvero in uso era di tre ore prima, non conosceva
        # ARGINE_EXPERT_READ, e uno sweep sui tre modi avrebbe prodotto tre
        # colonne identiche spacciate per un confronto.
        #
        # L'assenza non e' ambigua se lo store e' aperto: quella riga la stampa
        # qualunque binario capace di store, quindi se c'e' la riga e NON c'e' il
        # modo, il binario e' anteriore all'interruttore. E' informazione, non
        # silenzio.
        if (st or {}).get("aperto"):
            error(f"questo llama-server non sa scegliere lo strato di lettura: "
                  f"ho chiesto {chiesto} e lui non dichiara nessun modo")
            print(dim("  E' un binario anteriore a --read-mode. Il numero che segue "
                      "e' stato prodotto dal suo percorso predefinito, e un "
                      "confronto fra modi darebbe colonne identiche."))
            print(dim("  Controlla quale binario stai usando: `argine doctor`."))
            return False
        return True  # store non aperto: il problema e' un altro e lo dice gia'
    if visto.split(":", 1)[0] != chiesto.split(":", 1)[0] or (
            ":" in chiesto and visto != chiesto):
        warn(f"ho chiesto lettura {chiesto} e il motore ha usato {visto}: "
             f"la riga che segue NON misura cio' che dice")
        return False
    return True


def _expert_store_env(args: argparse.Namespace, m: GGUFModel,
                      rt: Optional[rtmod.LlamaRuntime] = None,
                      plan: Optional[planmod.RuntimePlan] = None) -> Dict[str, str]:
    """Variabili per servire gli expert dallo store invece che da mmap.

    Sono variabili d'ambiente e non argomenti di llama-server perche' la patch
    lato C++ resta cosi' minuscola: nessuna modifica al parsing degli argomenti
    di llama.cpp, quindi molto meno da mantenere a ogni rebase.
    """
    scelta = getattr(args, "expert_store", None)
    if not scelta:
        return {}

    # Il binario sa fare lo store? Un llama-server ufficiale ignora le variabili
    # d'ambiente in silenzio: il benchmark parte, gira e restituisce un numero
    # che descrive il percorso normale. Meglio fermarsi qui che consegnare una
    # misura che non misura quello che dice.
    if rt is not None and not rt.supports_expert_store():
        error(f"questo llama-server non contiene l'expert store (build={rt.build})")
        print(f"  binario in uso: {rt.binary('server')}")
        print()
        print("  Serve la build patchata. Nella stessa finestra del terminale:")
        print(cyan("    set ARGINE_LLAMA_DIR=") +
              cyan(str(Path(__file__).resolve().parents[1] / "llama-patched")))
        print(dim("  (la variabile vale solo per quella finestra: riaprendo il "
                  "terminale va rimessa)"))
        raise SystemExit(EXIT_USAGE)
    if scelta == "auto":
        percorso = _default_store_path(m.path)
        if not percorso.exists():
            warn(f"nessun expert store in {percorso.name}: si usa la memoria mappata "
                 f"come prima. Lo costruisci con `argine store build {Path(m.path).name}`")
            return {}
    else:
        percorso = Path(scelta).expanduser()
        if not percorso.exists():
            error(f"expert store non trovato: {percorso}")
            raise SystemExit(EXIT_USAGE)

    env = {"ARGINE_EXPERT_STORE": str(percorso)}
    diretto = getattr(args, "expert_direct", True)
    env["ARGINE_EXPERT_DIRECT"] = "1" if diretto else "0"
    if getattr(args, "verbose", False) or getattr(args, "expert_stats", False):
        env["ARGINE_EXPERT_VERBOSE"] = "1"

    # PRETOUCH azzera ogni slice (memset) prima di leggerla, per attribuire il
    # costo dei page fault separatamente da quello dell'I/O. E' uno strumento di
    # diagnosi, e costa una scrittura piena per ogni miss.
    #
    # Lo imponiamo a 0 invece di lasciarlo assente, per la stessa ragione per cui
    # --cache-ram si passa sempre: cio' che non si dichiara lo decide qualcun
    # altro. Qui "qualcun altro" era una variabile rimasta in ambiente da un
    # esperimento di giorni prima, che ha rallentato in silenzio una misura —
    # e non di una quantita' costante: il memset si paga per ogni miss, cioe'
    # proprio sulla variabile che lo sweep fa variare, quindi la curva ne
    # risultava piu' ripida del vero.
    pretouch_voluto = bool(getattr(args, "pretouch", False))
    env["ARGINE_EXPERT_PRETOUCH"] = "1" if pretouch_voluto else "0"
    if not pretouch_voluto:
        for nome in ("ARGINE_EXPERT_PRETOUCH", "PICOSTAR_EXPERT_PRETOUCH"):
            v = os.environ.get(nome)
            if v and v[0] not in "0nNfF":
                warn(f"{nome}={v} e' impostata nel tuo ambiente: la ignoro "
                     f"(azzera ogni slice prima di leggerla, e falserebbe la "
                     f"misura in proporzione ai miss). Per togliertela di mezzo: "
                     f"setx {nome} \"\"")
                break

    # Quale strato di lettura. Si dichiara sempre, come PRETOUCH e per lo stesso
    # motivo: il modo cambia la velocita' del 7-11%, quindi una variabile rimasta
    # in ambiente da una misura precedente sposterebbe il risultato piu' di
    # quanto valga l'effetto che si sta cercando.
    modo = _modo_lettura(getattr(args, "read_mode", None))
    env["ARGINE_EXPERT_READ"] = modo
    for nome in ("ARGINE_EXPERT_READ", "PICOSTAR_EXPERT_READ"):
        v = os.environ.get(nome)
        if v and v.strip() != modo:
            warn(f"{nome}={v} e' impostata nel tuo ambiente: uso {modo} "
                 f"(--read-mode). Per togliertela di mezzo: setx {nome} \"\"")
            break

    # Il budget lo decide il planner, che sa quanta RAM resta davvero. Senza
    # questo l'utente doveva indovinare un numero contro il piano, e sbagliarlo
    # significava far paginare la macchina: e' capitato con --expert-budget 8G
    # su un piano che lasciava agli expert 6.37 GiB.
    budget = parse_size(getattr(args, "expert_budget", None))
    dal_piano = False
    if not budget and plan is not None and plan.expert_cache_budget:
        budget = int(plan.expert_cache_budget)
        dal_piano = True
    if budget:
        env["ARGINE_EXPERT_BUDGET"] = str(int(budget))
    # "chiesto", non "attivo": la conferma arriva solo dal log di llama-server,
    # e la stampa _report_expert_store. Affermare qui che gli expert "sono
    # serviti dallo store" era falso: la variabile d'ambiente si imposta anche
    # su un binario che non la guarda nemmeno.
    info(f"chiedo lo store {percorso.name}"
         + (f", budget {fmt_bytes(budget)}"
            + (" (dal planner)" if dal_piano else " (da --expert-budget)")
            if budget else ", senza tetto")
         + (", cache dell'OS scavalcata" if diretto else ", cache dell'OS attiva")
         + " (conferma dopo il caricamento)")
    if not diretto:
        warn("con la cache dell'OS attiva la RAM viene contata due volte: il budget "
             "limita la nostra cache, non il totale")
    if not budget:
        # Rete di sicurezza per qualunque chiamante futuro: senza budget il lato
        # C++ non sfratta niente e la cache arriva a contenere tutti gli expert,
        # cioe' 16 GiB di memoria privata che su Windows finisce nel file di
        # paging invece che in pagine riciclabili. E' il contrario di cio' per
        # cui lo store esiste, quindi va detto, non lasciato dedurre.
        warn("nessun tetto sulla cache: senza budget non viene sfrattato niente "
             "e la memoria cresce fino a contenere tutti gli expert "
             f"({fmt_bytes(m.expert_bytes)}). Passa --expert-budget per "
             "rimettere il tetto rigido.")
    return env


def _add_expert_store_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--expert-store", nargs="?", const="auto", default=None,
                   metavar="FILE",
                   help="servi gli expert dall'expert store con letture esplicite "
                        "invece che dalla memoria mappata (richiede una build "
                        "patchata). Senza valore usa <modello>.psxs")
    p.add_argument("--expert-budget", default=None, metavar="SIZE",
                   help="tetto rigido di RAM per gli expert, es. 6G")
    p.add_argument("--expert-direct", dest="expert_direct", action="store_true",
                   default=True,
                   help="scavalca la cache dell'OS (default): condizione per un "
                        "budget davvero rigido")
    p.add_argument("--no-expert-direct", dest="expert_direct", action="store_false",
                   help="usa la cache dell'OS: piu' veloce a cache calda, ma la RAM "
                        "viene contata due volte")
    p.add_argument("--pretouch", action="store_true",
                   help="azzera ogni slice prima di leggerla, per separare il "
                        "costo dei page fault da quello dell'I/O. Strumento di "
                        "diagnosi: rallenta in proporzione ai miss, quindi non "
                        "usarlo per misurare la velocita'")
    p.add_argument("--read-mode", default=None, metavar="MODO",
                   help="strato di lettura dello store: asincrono (default, emetti "
                        "e continua), blocco[:N] (N slice in volo, si aspetta il "
                        "blocco), sincrono (una per volta). Serve a confrontarli "
                        "con lo stesso binario")
    p.add_argument("--expert-stats", action="store_true",
                   help="stampa hit, miss e sfratti della expert cache alla fine")


def _report_expert_store(proc: srv.LlamaServerProcess,
                         args: argparse.Namespace) -> Dict[str, Any]:
    """Verifica che lo store sia davvero in uso, e lo dice.

    Se e' stato chiesto e non risulta attivo, il risultato del benchmark non
    parla dello store: parla del percorso normale. Dirlo forte e' l'unico modo
    di non confondere un esperimento fallito con un'ipotesi smentita.
    """
    if not getattr(args, "expert_store", None):
        return {}
    st = proc.expert_store_status()
    print(header("EXPERT STORE"))
    if st["in_uso"]:
        ok("IN USO: gli expert arrivano dallo store con letture esplicite")
        if st["bytes"]:
            print(kv("tensori serviti", fmt_bytes(st["bytes"])))
    elif st["aperto"]:
        error("aperto ma NON in uso: nessun tensore expert e' stato assegnato al "
              "buffer dello store")
        print(dim("  Causa piu' probabile: --no-repack, che in llama.cpp disabilita "
                  "tutti gli extra buffer type e non solo il repack."))
        print(dim("  Il risultato di questo benchmark descrive il percorso normale con "
                  "memoria mappata, non lo store."))
    else:
        error("NON aperto: llama-server non ha nemmeno visto lo store")
        print(dim("  Cause tipiche: binario non patchato (controlla ARGINE_LLAMA_DIR), "
                  "store di un'altra versione, o file illeggibile."))
    for riga in st["righe"]:
        print(dim("  " + riga))
    if st["stats"]:
        print(kv("cache", st["stats"]))
    return st


def _report_expert_cache(proc: srv.LlamaServerProcess,
                         args: argparse.Namespace) -> Dict[str, Any]:
    """Statistiche della expert cache, dopo la chiusura del processo.

    Vanno lette qui e non al caricamento: llama-server le stampa uscendo, quindi
    al momento del report iniziale non esistono ancora. Sono il dato che dice se
    il budget e' stato rispettato, cioe' se la promessa centrale del progetto
    ("11 GiB e non un byte di piu'") e' stata mantenuta o solo dichiarata.
    """
    if not getattr(args, "expert_store", None):
        return {}
    st = proc.expert_store_status()
    blocco = st.get("resoconto") or []
    if not blocco:
        if st["in_uso"]:
            warn("nessun resoconto della cache ricevuto: llama-server lo stampa ogni "
                 "5 secondi con --expert-stats, quindi su esecuzioni molto brevi puo' "
                 "non fare in tempo")
        return st
    print(header("EXPERT CACHE"))
    for riga in blocco:
        print("  " + riga.replace("argine: ", ""))
    _avvisa_copertura(st)
    return st


def _avvisa_copertura(st: Dict[str, Any], stampa: bool = True) -> bool:
    """Lo store ha preso TUTTI i tensori di expert, o solo una parte?

    Una copertura parziale e' il peggiore degli esiti possibili perche' si
    traveste da successo: "IN USO" compare, la cache riporta hit e miss
    plausibili, il budget risulta rispettato, e intanto una fetta degli expert
    vive in memoria mappata, dove il tetto di RAM non vale. Le cause sono reali e
    tutte silenziose: un `ffn_gate_up_exps` fuso, slice non allineate di un solo
    tipo, un numero di expert oltre il limite della build.

    Restituisce True se la copertura e' piena o ignota.
    """
    cop = (st or {}).get("copertura")
    if not cop:
        # Lo stesso ragionamento di _conferma_modo_lettura: se lo store e' aperto
        # e la riga della copertura manca, il binario e' anteriore a quel
        # controllo. Non si puo' dire se la copertura sia piena, e va detto che
        # non si puo' dire, invece di tacere e lasciar credere che sia piena.
        if (st or {}).get("aperto") and stampa:
            warn("questo llama-server non riporta la copertura dei tensori: non "
                 "posso confermare che lo store li serva tutti (binario anteriore "
                 "al controllo). `argine doctor` dice quale stai usando.")
        return True
    presi, attesi = cop
    if not attesi or presi >= attesi:
        return True
    if stampa:
        error(f"copertura parziale: lo store serve {presi} tensori di expert su "
              f"{attesi} ({100.0 * presi / attesi:.0f}%)")
        print(dim("  Gli altri stanno sulla memoria mappata: il budget non e' un "
                  "tetto sulla RAM totale, e questo numero non descrive lo store "
                  "da solo."))
        print(dim("  Cause tipiche: gate e up fusi in un tensore solo, slice non "
                  "allineate per un solo tipo, piu' expert del limite della build."))
    return False


def _report_buffer_calibration(proc: srv.LlamaServerProcess,
                               p: planmod.RuntimePlan) -> Dict[str, Any]:
    """Confronta la stima del planner con i buffer dichiarati da llama.cpp.

    Serve a non fidarsi del proprio modello analitico: se lo scostamento e'
    grande, il piano di memoria sta mentendo e va ricalibrato.
    """
    buf = proc.reported_buffers()
    if not buf:
        return {}
    allocato = buf.get("allocated_bytes", 0)
    mappato = buf.get("mapped_bytes", 0)

    # Solo le allocazioni fisiche vanno confrontate con la nostra stima: un
    # buffer "Mapped" e' spazio virtuale, e quanta RAM occupi davvero lo decide
    # la page cache del sistema operativo.
    est = p.estimate.kv_cache + p.estimate.activations + p.estimate.repack
    if not p.moe_streaming:
        est += p.estimate.weights_resident
    if est:
        delta = (allocato - est) / est * 100
        riga = (f"buffer allocati da llama.cpp {fmt_bytes(allocato)} "
                f"vs stima {fmt_bytes(est)} ({delta:+.0f}%)")
        (info if abs(delta) <= 25 else warn)(riga)
        buf["delta_percent"] = round(delta, 1)
    buf["estimated_bytes"] = est

    if mappato:
        info(f"file mappato {fmt_bytes(mappato)} (memoria virtuale): quanto ne "
             "resta in RAM lo decide la page cache del sistema")
        if p.moe_streaming:
            warn("i tensori degli expert NON sono governati da Argine: llama.cpp "
                 "mappa l'intero file e la residenza parziale e' affidata al "
                 "sistema operativo. Il budget della expert cache e' una stima "
                 "di quanto restera' residente, non un limite imposto.")
    return buf


def _run_common(args: argparse.Namespace) -> tuple[int, Optional[Dict[str, Any]]]:
    """Prepara piano e runtime, avvia llama-server con retry progressivi.

    Ritorna (exit_code, contesto) dove contesto contiene proc/plan/porte.
    """
    try:
        m = _load_model(args.model)
    except GGUFError as e:
        error(str(e))
        return EXIT_USAGE, None

    orfani = rtmod.find_orphans()
    if orfani:
        totale = sum(o["rss"] or 0 for o in orfani)
        warn(f"{len(orfani)} processo/i llama-server di esecuzioni precedenti "
             f"ancora attivi, che occupano {fmt_bytes(totale)}:")
        for o in orfani:
            print(f"      pid {o['pid']}  {fmt_bytes(o['rss'])}")
        if getattr(args, "kill_orphans", False):
            n = rtmod.kill_orphans([o["pid"] for o in orfani])
            ok(f"terminati {n} processi; la RAM tornera' disponibile in pochi secondi")
            time.sleep(3)
        else:
            print("  Falsano il piano di memoria: chiudili con "
                  + bold("--kill-orphans") + " oppure 'taskkill /IM llama-server.exe /F'")

    hw = hwmod.detect(extra_disk_paths=[str(Path(m.path).parent)])
    backend, msgs = hwmod.resolve_backend(args.backend, hw)
    for msg in msgs:
        info(msg)

    if args.auto:
        info("profilo automatico attivo: context, KV, batch e offload scelti dal planner")

    req = planmod.PlanRequest(
        ctx=None if args.auto else args.ctx,
        batch=args.batch, ubatch=args.ubatch, threads=args.threads,
        kv_type=None if args.auto else args.kv_type,
        gpu_layers=None if args.auto else args.gpu_layers,
        ram_limit=parse_size(args.ram_limit), backend=backend,
        mmap=not args.no_mmap, mlock=args.mlock,
        expert_cache=parse_size(args.expert_cache),
        allow_streaming=not args.no_streaming, auto=args.auto,
        jinja=not args.no_jinja, parallel=args.parallel,
        llama_verbose=args.llama_verbose,
        repack=_decidi_repack(args),
        expert_store=bool(getattr(args, "expert_store", None)),
        expert_store_budget=parse_size(getattr(args, "expert_budget", None)),
        spec=args.spec, draft_model=args.draft_model,
        draft_n_max=args.draft_n_max, draft_ctx=args.draft_ctx,
        prompt_cache_dir=args.prompt_cache, cache_reuse=args.cache_reuse,
        prompt_cache_states=getattr(args, "prompt_cache_states", None),
    )
    try:
        p = planmod.plan(m, hw, req)
    except ValueError as e:
        error(str(e))
        return EXIT_USAGE, None

    _print_model(m, p) if args.verbose else _print_plan(p, m)

    if not p.feasible:
        print()
        error("Argine non avvia il modello: la stima supera il budget RAM.")
        print("  Prova:")
        for riga in _consigli_non_avviabile(m, hw, p, args):
            print("    - " + riga)
        telemetry.write_run(telemetry.build_record(
            event="run_refused", hardware=hw, model=m, plan=p,
            extra={"reason": p.failure_reason}))
        return EXIT_INFEASIBLE, None

    if args.dry_run:
        try:
            rt = rtmod.ensure_runtime(backend, build=args.llama_build, quiet=True,
                                     offline=True)
            print(header("COMANDO llama.cpp"))
            print("  " + " ".join(srv.build_llama_args(rt, m.path, p, "127.0.0.1", 8081)))
        except Exception as e:
            warn(f"runtime non ancora installato ({e}); esegui 'argine setup'")
        return EXIT_OK, None

    try:
        rt = rtmod.ensure_runtime(backend, build=args.llama_build, offline=args.offline)
    except Exception as e:
        error(f"runtime llama.cpp non disponibile: {e}")
        return EXIT_RUNTIME, None

    devs: Optional[List[str]] = None
    if backend == "vulkan":
        devs = rt.list_devices()
        if devs:
            info("device Vulkan: " + "; ".join(devs))
        else:
            warn("nessun device Vulkan enumerato dal binario: llama.cpp usera' la CPU")

    attempts: List[Dict[str, Any]] = []
    proc: Optional[srv.LlamaServerProcess] = None
    upstream = 0
    for attempt in range(1 + max(0, args.retries)):
        proc, upstream = _start_stack(m, hw, p, rt, args, None)
        info(f"avvio llama-server (porta interna {upstream})...")
        ready = proc.wait_ready(upstream, timeout=args.load_timeout)
        if ready:
            nota = (" (richiesto; nessun device Vulkan enumerato: esecuzione CPU)"
                    if backend == "vulkan" and not devs else "")
            ok(f"modello caricato in RAM/mmap; backend={backend}{nota}, build={rt.build}")
            _avvisa_build_cambiata(m, rt)
            _report_buffer_calibration(proc, p)
            _report_expert_store(proc, args)
            break
        oom = proc.detect_oom()
        tail = proc.tail(12)
        proc.stop()
        attempts.append({"attempt": attempt + 1, "oom": oom, "tail": tail})
        if not proc.alive and oom:
            error(f"llama-server ha esaurito la memoria: {oom}")
        else:
            error("llama-server non e' diventato pronto entro il timeout")
        step = _degrade(p, m, hw)
        if step is None or attempt >= args.retries:
            for line in tail:
                print(dim("  | " + line))
            telemetry.write_run(telemetry.build_record(
                event="run_failed", hardware=hw, model=m, plan=p,
                runtime={"backend": backend, "build": rt.build},
                extra={"attempts": attempts}))
            return EXIT_RUNTIME, None
        warn(f"riprovo con configurazione ridotta: {step}")
        p.estimate = planmod.estimate_memory(m, p, hw)
        p.decisions.append(f"[retry] {step}")

    assert proc is not None
    return EXIT_OK, {
        "model": m, "hw": hw, "plan": p, "rt": rt, "proc": proc,
        "upstream": upstream, "backend": backend, "vulkan_devices": devs,
        "attempts": attempts,
    }


def cmd_run(args: argparse.Namespace) -> int:
    code, ctx = _run_common(args)
    if ctx is None:
        return code
    m, hw, p, rt, proc = ctx["model"], ctx["hw"], ctx["plan"], ctx["rt"], ctx["proc"]
    upstream = ctx["upstream"]

    port = args.port
    if not srv.port_is_free(port, args.host):
        warn(f"porta {port} occupata: ne scelgo una libera")
        port = srv.free_port()

    state = srv.GatewayState(
        upstream_port=upstream, plan=p, model=m, hardware=hw,
        runtime_build=rt.version(), backend=ctx["backend"], backend_proc=proc,
    )
    gw = srv.ArgineGateway(args.host, port, state)
    gw.serve_forever_background()

    print()
    print(header("ARGINE PRONTO"))
    print(kv("API OpenAI", green(f"http://{args.host}:{port}/v1/chat/completions")))
    print(kv("stato", f"http://{args.host}:{port}/argine/status"))
    print(kv("memoria", f"http://{args.host}:{port}/argine/memory"))
    print(kv("web UI llama.cpp", f"http://127.0.0.1:{upstream}/"))
    print(kv("log", str(logs_dir())))
    print()
    print(dim("  Esempio:"))
    print(dim(f'    curl http://{args.host}:{port}/v1/chat/completions -H "Content-Type: application/json" '
              '-d "{\\"model\\":\\"local\\",\\"messages\\":[{\\"role\\":\\"user\\",\\"content\\":\\"Ciao\\"}]}"'))
    print()
    info("Ctrl+C per fermare.")

    t0 = time.time()
    stop = {"flag": False}

    def _handler(signum, frame):
        stop["flag"] = True

    try:
        signal.signal(signal.SIGINT, _handler)
    except Exception:
        pass

    try:
        while not stop["flag"]:
            if not proc.alive:
                oom = proc.detect_oom()
                error("llama-server e' terminato inaspettatamente" + (f": {oom}" if oom else ""))
                for line in proc.tail(15):
                    print(dim("  | " + line))
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        info("arresto...")
        gw.shutdown()
        stats = proc.monitor.stop() if proc.monitor else None
        proc.stop()
        rec = telemetry.build_record(
            event="run", hardware=hw, model=m, plan=p,
            runtime={"backend": ctx["backend"], "build": rt.build, "port": port,
                     "upstream_port": upstream},
            metrics={"uptime_s": round(time.time() - t0, 1),
                     "resources": stats.to_dict() if stats else None,
                     "buffers": proc.reported_buffers(),
                     "oom": proc.detect_oom()},
            extra={"attempts": ctx["attempts"]},
        )
        path = telemetry.write_run(rec)
        ok(f"telemetria salvata in {path}")
    return EXIT_OK


# =============================================================== benchmark ===

def _modello_esiste_o_errore(args: argparse.Namespace) -> Optional[int]:
    """None se il modello si risolve; un codice d'uscita se non esiste.

    Da chiamare **prima** di qualunque ciclo di sweep. Dentro il ciclo, un avvio
    fallito viene saltato e si passa alla configurazione successiva: e' il
    comportamento giusto per una configurazione che non ci sta in RAM, ed e'
    quello sbagliato per un file che non esiste, perche' non esistera' nemmeno al
    giro dopo. Visto dall'utente: il percorso sbagliato produceva quattro
    fallimenti identici con tre pause di raffreddamento da 20 secondi in mezzo,
    e il messaggio utile — l'elenco dei file che ci sono davvero — scorreva via
    in cima.

    Distinguere i guasti che si ripetono da quelli che dipendono dalla
    configurazione e' il punto: gli uni fermano tutto subito, gli altri no.
    """
    try:
        find_gguf(args.model)
    except GGUFError as e:
        error(str(e))
        return EXIT_USAGE
    except OSError as e:
        error(f"{args.model}: {e}")
        return EXIT_USAGE
    return None


def _bench_prompt(args: argparse.Namespace) -> str:
    prompt = benchmod.DEFAULT_PROMPT
    if args.prompt:
        prompt = args.prompt
    elif args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    if args.prompt_tokens:
        # allunga il prompt fino a ~N token (stima 3.5 char/token)
        target = int(args.prompt_tokens * 3.5)
        while len(prompt) < target:
            prompt += "\n" + benchmod.DEFAULT_PROMPT
    return prompt


def cmd_benchmark(args: argparse.Namespace) -> int:
    if getattr(args, "sweep_expert_budget", None):
        return _cmd_benchmark_sweep_budget(args)
    if args.sweep_threads:
        return _cmd_benchmark_sweep(args)
    if args.sweep_prompt:
        return _cmd_benchmark_sweep_prompt(args)
    if getattr(args, "conversazione", None):
        return _cmd_benchmark_conversazione(args)
    code, ctx = _run_common(args)
    if ctx is None:
        return code
    m, hw, p, rt, proc = ctx["model"], ctx["hw"], ctx["plan"], ctx["rt"], ctx["proc"]
    upstream = ctx["upstream"]

    prompt = _bench_prompt(args)

    print()
    info(f"benchmark: {args.repeat} ripetizioni, n_predict={args.n_predict}")

    def progress(i: int, tot: int) -> None:
        sys.stdout.write(f"\r  run {i}/{tot} ...   ")
        sys.stdout.flush()

    res = benchmod.run_benchmark(
        upstream, prompt=prompt, n_predict=args.n_predict, repeat=args.repeat,
        warmup=not args.no_warmup, monitor_stats=None, on_progress=progress,
    )
    sys.stdout.write("\r" + " " * 30 + "\r")
    stats = proc.monitor.stop() if proc.monitor else None
    if stats is not None:
        res.resources = stats.to_dict()
    proc.stop()
    st_cache = _report_expert_cache(proc, args)

    print(header("RISULTATI"))
    if len(res.runs) > 1:
        # la mediana nasconde la deriva: mostriamo ogni ripetizione
        print(kv("per ripetizione", dim("gen tok/s | prompt tok/s | TTFT")))
        for i, r in enumerate(res.runs, 1):
            g = f"{r.gen_tok_s:.2f}" if r.gen_tok_s else "n/d"
            pp = f"{r.prompt_tok_s:.1f}" if r.prompt_tok_s else "n/d"
            tt = f"{r.ttft_s:.2f} s" if r.ttft_s else "n/d"
            print(kv(f"  run {i}", f"{g:>8} | {pp:>8} | {tt:>7}"))
    print(kv("prompt tokens/sec", f"{res.prompt_tok_s:.1f}" if res.prompt_tok_s else "n/d"))
    print(kv("generation tokens/sec", bold(f"{res.gen_tok_s:.2f}") if res.gen_tok_s else "n/d"))
    print(kv("time to first token", f"{res.ttft_s:.3f} s" if res.ttft_s else "n/d"))
    if res.ttft_warm_s:
        risparmio = ((res.ttft_s - res.ttft_warm_s) / res.ttft_s * 100
                     if res.ttft_s else 0)
        print(kv("  con prefisso in cache",
                 f"{res.ttft_warm_s:.3f} s" + (f"  ({risparmio:.0f}% in meno)"
                                               if risparmio > 5 else "")))
    _report_drift(res)
    if stats is not None:
        print(kv("RAM peak", fmt_bytes(stats.ram_peak)))
        print(kv("RAM media", fmt_bytes(stats.ram_mean)))
        print(kv("RAM libera minima", fmt_bytes(stats.system_available_min)))
        # su Windows il contatore include i soft fault (pagine gia' in page cache):
        # solo il volume letto da disco dice se stiamo davvero facendo I/O
        # Su Windows l'I/O del processo NON include le pagine della memoria
        # mappata: per lo streaming di un modello piu' grande della RAM conta
        # il traffico a livello di disco, non quello del processo.
        print(kv("page faults", f"{stats.page_faults_total:,}".replace(",", ".")))
        print(kv("I/O del processo",
                 f"{fmt_bytes(stats.disk_read_total)} @ {stats.disk_read_mb_s:.1f} MB/s"
                 + dim("  (esclude la memoria mappata)")))
        sysread = stats.sys_disk_read_total
        etichetta = f"{fmt_bytes(sysread)} @ {stats.sys_disk_read_mb_s:.1f} MB/s"
        if sysread > 256 * MB:
            print(kv("lettura disco (sistema)", bold(etichetta)
                     + dim("  <- streaming in corso")))
        else:
            print(kv("lettura disco (sistema)", etichetta))
        cores = hw.cpu_cores_logical or 1
        quota = stats.cpu_mean / cores if cores else 0
        print(kv("CPU media / picco",
                 f"{stats.cpu_mean:.0f}% / {stats.cpu_peak:.0f}% "
                 + dim(f"({quota:.0f}% / {stats.cpu_peak/cores:.0f}% "
                       f"di {cores} thread)")))
        # con abbastanza thread e poca CPU usata, il tempo se ne va altrove.
        # Soglia sull'utilizzo relativo: 572% su 8 thread e' il 72% del possibile,
        # e con 40 GiB letti da disco significa che si sta aspettando.
        if p.threads >= 4 and sysread > 256 * MB:
            utilizzo = stats.cpu_mean / (p.threads * 100.0)
            if utilizzo < 0.85:
                print(kv("", yellow(
                    f"CPU al {utilizzo*100:.0f}% di quanto potrebbe con "
                    f"{p.threads} thread, mentre il disco legge: il collo di "
                    "bottiglia e' l'I/O, non il calcolo")))
                per_token = sysread / max(1, sum(
                    r.generated_tokens for r in res.runs) or 1)
                print(kv("", dim(f"circa {fmt_bytes(per_token)} letti da SSD "
                                 "per token generato")))
    print(kv("GPU media / picco",
             f"{res.gpu_util_mean}% / {res.gpu_util_peak}%" if res.gpu_util_mean is not None else "n/d"))
    budget_pct = (stats.ram_peak / p.budget * 100) if (stats and p.budget) else None
    if budget_pct is not None:
        col = green if budget_pct <= 100 else red
        print(kv("RAM peak vs budget", col(f"{budget_pct:.0f}%")))
    if res.errors:
        print(header("ERRORI"))
        for e in res.errors:
            print(f"  {yellow('!')} {e}")

    rec = telemetry.build_record(
        event="benchmark", hardware=hw, model=m, plan=p,
        runtime={"backend": ctx["backend"], "build": rt.build},
        metrics=res.to_dict(),
        extra={"prompt_chars": len(prompt), "n_predict": args.n_predict,
               "repeat": args.repeat, "attempts": ctx["attempts"],
               "buffers": proc.reported_buffers(),
               # Il tasso di hit e la ripartizione dei tempi: il dato che spiega
               # gli altri, e che prima viveva solo nella scrollback.
               "expert_cache": (st_cache or {}).get("numeri") or None,
               "expert_copertura": (st_cache or {}).get("copertura"),
               "modo_lettura": (st_cache or {}).get("lettura")},
    )
    path = telemetry.write_run(rec)
    print()
    ok(f"risultati salvati in {path}")
    if args.json:
        print(json.dumps(rec, indent=2, ensure_ascii=False, default=str))
    return EXIT_OK


_RE_PER_MISS = re.compile(
    r"per miss:\s*([0-9.]+)\s*ms di lettura,\s*([0-9.]+)\s*ms di pagine nuove,"
    r"\s*([0-9.]+)\s*ms di commit")


def _tempi_per_miss(st: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Estrae dal resoconto dello store i millisecondi per miss.

    Sono tre numeri che il lato C++ stampa gia' da sempre e che nessuno leggeva in
    modo strutturato. Servono a spiegare la pendenza della curva
    velocita'/residenza invece di constatarla: se al calare del budget cresce il
    tempo di *lettura* per miss, il costo e' nel dispositivo; se cresce quello
    delle *pagine nuove*, e' nel sistema di memoria.

    E' anche la misura che decide l'ipotesi sul `PRETOUCH`: se quel memset non
    aggiunge lavoro ma lo sposta, allora senza `PRETOUCH` la lettura per miss deve
    crescere di quanto, con `PRETOUCH`, valgono le pagine nuove.

    I tre numeri escono dallo stesso parser che legge tutto il resoconto
    (`srv.LlamaServerProcess.statistiche_cache`): due regex sulla stessa riga
    sarebbero due fonti che possono divergere, ed e' il modo in cui `samples + 2`
    e' diventato falso in silenzio.
    """
    chiavi = ("ms_lettura_miss", "ms_pagine_miss", "ms_commit_miss")
    numeri = (st or {}).get("numeri")
    if numeri is None:
        numeri = srv.LlamaServerProcess.statistiche_cache(
            (st or {}).get("resoconto") or [])
    return {k: numeri.get(k) for k in chiavi}


def _prevola_budget(args: argparse.Namespace,
                    budgets: List[int]) -> tuple[List[int], List[int], str]:
    """Calcola il piano di memoria per ogni budget *prima* di avviare qualcosa.

    `planmod.plan()` non lancia llama-server: costa millisecondi, e sa gia' dire
    se una configurazione e' avviabile. Scoprirlo configurazione per
    configurazione, con 20 secondi di raffreddamento in mezzo, significa far
    aspettare l'utente per un'informazione che avevamo tutta in partenza — lo
    stesso difetto del percorso inesistente, in un'altra forma.

    Qui il salto *e'* legittimo: un budget piu' grande della RAM libera non ci
    sta, e non e' un errore. Ma dirlo subito, tutto insieme, cambia la differenza
    fra una tabella in due secondi e un minuto di pause per poi scoprire che
    restavano due punti.

    Ritorna (budget richiesti, quelli avviabili, tabella da stampare).
    """
    try:
        m = read_gguf(find_gguf(args.model))
        hw = hwmod.detect(extra_disk_paths=[str(Path(m.path).parent)])
        backend, _ = hwmod.resolve_backend(getattr(args, "backend", "auto"), hw)
    except Exception as e:  # noqa: BLE001
        # Il pre-volo e' un servizio, non un cancello: se non riesce, si procede
        # come prima e ogni configurazione dira' la sua.
        warn(f"pre-volo non riuscito ({e}): procedo configurazione per configurazione")
        return budgets, budgets, ""

    righe = [header("PRE-VOLO: QUALI BUDGET SONO AVVIABILI")]
    righe.append(kv("expert nel modello", fmt_bytes(m.expert_bytes)))
    righe.append(bold(f"  {'budget':>10}{'residente':>11}{'stima totale':>15}{'esito':>34}"))
    fattibili: List[int] = []
    for b in budgets:
        try:
            p = planmod.plan(m, hw, planmod.PlanRequest(
                backend=backend, ctx=getattr(args, "ctx", None),
                batch=getattr(args, "batch", None),
                ubatch=getattr(args, "ubatch", None),
                threads=getattr(args, "threads", None),
                kv_type=getattr(args, "kv_type", None),
                gpu_layers=getattr(args, "gpu_layers", None),
                ram_limit=parse_size(getattr(args, "ram_limit", None)),
                mmap=not getattr(args, "no_mmap", False),
                mlock=getattr(args, "mlock", False),
                allow_streaming=not getattr(args, "no_streaming", False),
                repack=_decidi_repack(args),
                expert_store=True, expert_store_budget=b,
                prompt_cache_states=getattr(args, "prompt_cache_states", None),
            ))
        except Exception as e:  # noqa: BLE001
            righe.append(f"  {fmt_bytes(b, 1):>10}{'n/d':>11}{'n/d':>15}"
                         + red(f"{str(e)[:32]:>34}"))
            continue
        fr = f"{b / m.expert_bytes * 100:.0f}%" if m.expert_bytes else "n/d"
        if p.feasible:
            fattibili.append(b)
            esito = green("avviabile")
        else:
            esito = yellow("oltre il budget RAM")
        righe.append(f"  {fmt_bytes(b, 1):>10}{fr:>11}"
                     f"{fmt_bytes(p.estimate.total, 1):>15}{esito:>34}")
    righe.append("")
    if fattibili and len(fattibili) < len(budgets):
        saltati = [fmt_bytes(b, 1) for b in budgets if b not in fattibili]
        righe.append(dim(f"  salto {', '.join(saltati)} senza avviare niente: "
                         f"il piano dice che non ci stanno."))
        righe.append("")
    return budgets, fattibili, "\n".join(righe)


ERRORE_RELATIVO_ASSUNTO = 0.03
SPAN_MINIMO_PER_VERDETTO = 2.0


def _dispersione_relativa(res: Any) -> Optional[float]:
    """Errore relativo di una misura di velocita', dalle sue ripetizioni.

    Misurare la rumorosita' invece di assumerla: le ripetizioni ci sono gia', e
    l'incertezza dell'esponente dipende da questo numero. Con meno di due
    ripetizioni valide non c'e' nulla da stimare e si ritorna None, cosi' chi
    chiama usa il valore assunto *e lo dichiara*.
    """
    try:
        vals = [r.gen_tok_s for r in getattr(res, "runs", []) if r.gen_tok_s]
    except Exception:  # noqa: BLE001
        return None
    if len(vals) < 2:
        return None
    import statistics

    media = statistics.fmean(vals)
    if media <= 0:
        return None
    # deviazione standard della mediana approssimata con quella campionaria:
    # con tre ripetizioni una stima piu' raffinata sarebbe finta precisione
    return statistics.stdev(vals) / media


def esponente_residenza(
    rows: List[Dict[str, Any]],
) -> Optional[tuple[float, float, float, float]]:
    """(esponente, calo della residenza, calo della velocita', incertezza).

    L'esponente e' d(log velocita') / d(log residenza) fra il punto con piu'
    residenza e quello con meno. Dice in una cifra se la dimensione del modello ha
    un tetto: 0 significa che la residenza non conta e si puo' crescere quanto si
    vuole, 1 che la velocita' e' proporzionale alla frazione residente e quindi un
    modello tre volte piu' grande va tre volte piu' piano.

    Si confrontano gli estremi e non una regressione su tutti i punti perche' con
    tre o quattro misure una regressione darebbe una precisione che i dati non
    hanno; e ritorna None quando i punti utili sono meno di due, invece di
    inventare una pendenza da un punto solo.

    L'incertezza c'e' perche' la prima versione di questa funzione non l'aveva, e
    su due punti distanti solo 1,33x ha stampato "la velocita' scende in modo
    sensibile" da un esponente di 0,47 che valeva 0,47 +/- 0,15: un intervallo che
    scavalca la soglia del verdetto. Il denominatore e' log(calo della residenza),
    quindi **un'escursione piccola amplifica il rumore**: con 1,33x l'incertezza e'
    0,15, con 2x la stessa rumorosita' ne da' 0,06. Per questo il verdetto si
    pronuncia solo sopra un'escursione minima.

    L'errore per punto si prende dalla dispersione fra le ripetizioni quando c'e';
    altrimenti si assume il 3%, e si dichiara di averlo assunto.
    """
    utili = [r for r in rows if r.get("gen_tok_s") and r.get("frazione")]
    if len(utili) < 2:
        return None
    utili.sort(key=lambda r: r["frazione"], reverse=True)
    a, z = utili[0], utili[-1]
    rapporto_res = a["frazione"] / z["frazione"]
    rapporto_vel = a["gen_tok_s"] / z["gen_tok_s"]
    if rapporto_res <= 1 or rapporto_vel <= 0:
        return None
    err = max(a.get("err_rel") or ERRORE_RELATIVO_ASSUNTO,
              z.get("err_rel") or ERRORE_RELATIVO_ASSUNTO)
    incertezza = math.sqrt(2) * err / math.log(rapporto_res)
    return (math.log(rapporto_vel) / math.log(rapporto_res),
            rapporto_res, rapporto_vel, incertezza)


def _cmd_benchmark_sweep_budget(args: argparse.Namespace) -> int:
    """Misura la velocita' al variare del tetto di RAM per gli expert.

    Serve a rispondere, senza scaricare niente, alla domanda che decide se questa
    architettura scala: *cosa succede quando il set di expert e' molto piu'
    grande del budget?* Un modello con tre volte gli expert e lo stesso budget si
    comporta come questo modello con un terzo del budget — quello che cambia e' il
    tasso di miss, e il tasso di miss e' l'unica variabile che conta.

    Il README afferma che "con lo store la dimensione non ha un tetto". E' una
    previsione, non una misura: la curva prodotta qui la conferma o la smentisce
    prima di impegnare decine di GiB di banda su un modello piu' grande.

    La frazione fra budget e byte totali degli expert e' stampata accanto a ogni
    riga, perche' e' quella — non i GiB in assoluto — che si trasferisce a un
    altro modello.
    """
    budgets: List[int] = []
    for v in str(args.sweep_expert_budget).replace(" ", "").split(","):
        if not v:
            continue
        try:
            b = parse_size(v)
        except ValueError as e:
            error(f"--sweep-expert-budget: {e}")
            return EXIT_USAGE
        if b:
            budgets.append(b)
    budgets = sorted(set(budgets), reverse=True)
    if not budgets:
        error("--sweep-expert-budget vuole una lista di dimensioni, es. 8G,6G,4G,3G")
        return EXIT_USAGE

    if not getattr(args, "expert_store", None):
        error("--sweep-expert-budget ha senso solo con --expert-store: senza store "
              "gli expert stanno in memoria mappata e il budget non e' un tetto")
        return EXIT_USAGE

    fatale = _modello_esiste_o_errore(args)
    if fatale is not None:
        return fatale

    budgets, fattibili, nota = _prevola_budget(args, budgets)
    if nota:
        print(nota)
    if len(fattibili) < 2:
        error("meno di due configurazioni avviabili: non c'e' una curva da "
              "misurare, e due punti su quattro darebbero una pendenza "
              "costruita sul nulla.")
        print("  Prova:")
        print("    - chiudi browser e IDE: il planner limita il budget alla RAM "
              "davvero libera, non al totale")
        print("    - scegli budget piu' bassi, dentro cio' che il piano lascia "
              "agli expert (vedi la colonna 'esito' sopra)")
        return EXIT_INFEASIBLE
    budgets = fattibili

    # I modi di lettura da confrontare. Senza --sweep-read-mode e' uno solo e
    # tutto l'output resta quello di prima; con piu' di uno i modi si alternano
    # DENTRO ogni budget, non uno sweep dopo l'altro, perche' cosi' una deriva
    # termica o un processo che si sveglia colpisce tutti e tre invece di
    # premiarne uno.
    modi: List[str] = []
    for v in str(getattr(args, "sweep_read_mode", None) or "").replace(" ", "").split(","):
        if v:
            modi.append(_modo_lettura(v))
    if not modi:
        modi = [_modo_lettura(getattr(args, "read_mode", None))]
    modi = list(dict.fromkeys(modi))

    prompt = _bench_prompt(args)
    rows: List[Dict[str, Any]] = []
    info(f"sweep budget expert: {', '.join(fmt_bytes(b, 1) for b in budgets)} "
         f"({args.repeat} ripetizioni ciascuno)")
    if len(modi) > 1:
        info(f"per ogni budget, {len(modi)} strati di lettura: {', '.join(modi)}")

    expert_bytes: Optional[int] = None
    for n_cfg, (b, modo) in enumerate(
            [(b, modo) for b in budgets for modo in modi]):
        if n_cfg and args.cooldown > 0:
            info(f"pausa di raffreddamento {args.cooldown}s "
                 "(evita che la deriva termica falsi il confronto)")
            time.sleep(args.cooldown)
        print()
        print(header(f"BUDGET EXPERT = {fmt_bytes(b, 1)}"
                     + (f", LETTURA {modo.upper()}" if len(modi) > 1 else "")))
        sweep_args = argparse.Namespace(**vars(args))
        sweep_args.expert_budget = str(b)
        sweep_args.sweep_expert_budget = None
        sweep_args.read_mode = modo
        sweep_args.verbose = False
        code, ctx = _run_common(sweep_args)
        if ctx is None:
            warn(f"budget={fmt_bytes(b, 1)}: avvio non riuscito, salto")
            continue
        m, hw, p, rt, proc = ctx["model"], ctx["hw"], ctx["plan"], ctx["rt"], ctx["proc"]
        if expert_bytes is None:
            expert_bytes = m.expert_bytes
        res = benchmod.run_benchmark(
            ctx["upstream"], prompt=prompt, n_predict=args.n_predict,
            repeat=args.repeat, warmup=not args.no_warmup,
        )
        stats = proc.monitor.stop() if proc.monitor else None
        if stats is not None:
            res.resources = stats.to_dict()
        proc.stop()
        # Il resoconto della cache lo stampa llama-server uscendo, quindi va letto
        # qui. Lo sweep non lo faceva, e con --expert-stats non si vedeva niente:
        # i tempi per miss sono l'unico dato che spiega *perche'* la curva ha la
        # pendenza che ha, e sono quelli che servono per l'ipotesi sul PRETOUCH.
        st_cache = _report_expert_cache(proc, args)
        stato = proc.expert_store_status()
        onorato = _conferma_modo_lettura(stato, modo)
        # L'avviso lo ha gia' stampato _report_expert_cache: qui serve solo il
        # verdetto, per marcare la riga come non confrontabile.
        if not _avvisa_copertura(stato, stampa=False):
            onorato = False
        frazione = (b / expert_bytes) if expert_bytes else None
        rows.append({
            "budget": b,
            "modo": modo,
            "modo_onorato": onorato,
            "frazione": frazione,
            "gen_tok_s": res.gen_tok_s,
            "err_rel": _dispersione_relativa(res),
            "ttft_s": res.ttft_s,
            "ram_peak": stats.ram_peak if stats else None,
            **_tempi_per_miss(st_cache),
        })
        print(kv("generation tokens/sec",
                 bold(f"{res.gen_tok_s:.2f}") if res.gen_tok_s else "n/d"))
        telemetry.write_run(telemetry.build_record(
            event="benchmark_sweep_budget", hardware=hw, model=m, plan=p,
            runtime={"backend": ctx["backend"], "build": rt.build},
            metrics=res.to_dict(),
            extra={"expert_budget": b, "expert_bytes": expert_bytes,
                   "frazione_residente": frazione, "modo_lettura": modo,
                   "modo_onorato": onorato,
                   "expert_cache": (st_cache or {}).get("numeri") or None,
                   "expert_copertura": (st_cache or {}).get("copertura")}))

    if not rows:
        error("nessuna configurazione ha prodotto risultati")
        return EXIT_RUNTIME

    # Con piu' modi, la tabella che interessa e' il confronto fra loro a parita'
    # di budget, e le curve vanno calcolate una per modo: mescolarle darebbe un
    # esponente che non descrive nessuno dei tre.
    if len(modi) > 1:
        _tabella_modi_lettura(rows, modi, expert_bytes)
        for modo in modi:
            suoi = [r for r in rows if r["modo"] == modo]
            if len(suoi) >= 2:
                print()
                print(header(f"CURVA, LETTURA {modo.upper()}"))
                _stampa_curva(suoi, expert_bytes)
        print()
        return EXIT_OK

    print()
    print(header("CURVA VELOCITA' / RESIDENZA"))
    _stampa_curva(rows, expert_bytes)
    print()
    return EXIT_OK


def _tabella_modi_lettura(rows: List[Dict[str, Any]], modi: List[str],
                          expert_bytes: Optional[int]) -> None:
    """Confronto fra gli strati di lettura, budget per budget.

    La colonna `vs` usa come riferimento il PRIMO modo della lista, che per
    convenzione e' il piu' vecchio: si legge "quanto ha guadagnato il nuovo".

    Perche' una tabella a parte e non una riga nella curva: l'esponente e' un
    rapporto fra gli estremi, e un modo puo' averlo migliore solo per essere piu'
    lento in alto. E' successo — il blocco da 2 sembrava il migliore dei tre con
    questo criterio, pur essendo piu' lento dell'asincrono a ogni budget. Quello
    che conta e' il valore al punto di lavoro, quindi i valori vanno messi uno
    accanto all'altro.
    """
    print()
    print(header("STRATI DI LETTURA A CONFRONTO"))
    if expert_bytes:
        print(kv("expert totali", fmt_bytes(expert_bytes)))
    rif = modi[0]
    intestazione = f"  {'budget':>10}{'residente':>11}"
    for modo in modi:
        intestazione += f"{modo:>12}"
    intestazione += f"{'vs ' + rif:>14}"
    print(bold(intestazione))
    budgets = list(dict.fromkeys(r["budget"] for r in rows))
    sospetti = False
    for b in budgets:
        per_modo = {r["modo"]: r for r in rows if r["budget"] == b}
        fr = next((r["frazione"] for r in per_modo.values() if r["frazione"]), None)
        riga = f"  {fmt_bytes(b, 1):>10}{(f'{fr * 100:.0f}%' if fr else 'n/d'):>11}"
        migliore = max((r["gen_tok_s"] or 0.0) for r in per_modo.values())
        for modo in modi:
            r = per_modo.get(modo)
            v = r["gen_tok_s"] if r else None
            if not r or not r.get("modo_onorato", True):
                sospetti = True
            # Prima si allinea, poi si evidenzia: i codici colore contano come
            # caratteri per l'allineamento e grassettare prima sfalsa la colonna.
            cella = f"{(f'{v:.2f}' if v else 'n/d'):>12}"
            riga += bold(cella) if (v and v >= migliore - 1e-9) else cella
        base = per_modo.get(rif, {}).get("gen_tok_s")
        ultimo = per_modo.get(modi[-1], {}).get("gen_tok_s")
        delta = (f"{(ultimo / base - 1) * 100:+.1f}%"
                 if (base and ultimo) else "n/d")
        riga += f"{delta:>14}"
        print(riga)

    # Il rumore e' la cosa che rende leggibile o illeggibile un +7%: va stampato
    # accanto, non lasciato al lettore. La prima versione di questa misura dava
    # +7-11% con un run per punto e un rumore dell'1-4%, cioe' un margine di due
    # o tre volte il rumore su un campione di uno.
    errori = [r["err_rel"] for r in rows if r.get("err_rel")]
    if errori:
        peggiore = max(errori) * 100
        print()
        print(kv("dispersione fra ripetizioni",
                 f"fino a {peggiore:.1f}% (il margine fra i modi va letto contro "
                 f"questo numero)"))
    if sospetti:
        warn("almeno un punto manca o non ha usato il modo richiesto: le colonne "
             "non sono tutte confrontabili")


def _stampa_curva(rows: List[Dict[str, Any]], expert_bytes: Optional[int]) -> None:
    """La curva velocita'/residenza di una serie omogenea di punti."""
    if expert_bytes:
        print(kv("expert totali", fmt_bytes(expert_bytes)))
    print(bold(f"  {'budget':>10}{'residente':>11}{'gen tok/s':>12}"
               f"{'TTFT':>9}{'RAM peak':>11}{'vs il max':>11}"))
    # "vs pieno" era falso: il primo punto non e' la residenza piena (qui il 24%
    # di 16,35 GiB), e' solo il budget piu' alto che la RAM libera ha permesso.
    # Chiamarlo "pieno" invitava a leggere 1.00x come "nessuna penalita'".
    pieno = next((r["gen_tok_s"] for r in rows if r["gen_tok_s"]), None)
    for r in rows:
        gen = f"{r['gen_tok_s']:.2f}" if r["gen_tok_s"] else "n/d"
        fr = f"{r['frazione'] * 100:.0f}%" if r["frazione"] else "n/d"
        ttft = f"{r['ttft_s']:.2f}" if r["ttft_s"] else "n/d"
        ram = fmt_bytes(r["ram_peak"], 1) if r["ram_peak"] else "n/d"
        rel = (f"{r['gen_tok_s'] / pieno:.2f}x"
               if (pieno and r["gen_tok_s"]) else "n/d")
        print(f"  {fmt_bytes(r['budget'], 1):>10}{fr:>11}{gen:>12}"
              f"{ttft:>9}{ram:>11}{rel:>11}")

    # I tempi per miss, se lo store li ha riportati: dicono *dove* va il tempo in
    # piu' quando il budget scende, invece di limitarsi a dire che e' andato.
    con_miss = [r for r in rows if r.get("ms_lettura_miss") is not None]
    if con_miss:
        print()
        print(bold(f"  {'budget':>10}{'lettura/miss':>14}{'pagine/miss':>13}"
                   f"{'commit/miss':>13}"))
        for r in con_miss:
            print(f"  {fmt_bytes(r['budget'], 1):>10}"
                  f"{r['ms_lettura_miss']:>13.3f}m"
                  f"{r['ms_pagine_miss']:>12.3f}m"
                  f"{r['ms_commit_miss']:>12.3f}m")
        if len(con_miss) >= 2:
            a, z = con_miss[0], con_miss[-1]
            cresciuta = [(n, a[k], z[k]) for n, k in
                         (("lettura", "ms_lettura_miss"),
                          ("pagine nuove", "ms_pagine_miss"),
                          ("commit", "ms_commit_miss"))
                         if a[k] and z[k] and z[k] > a[k] * 1.1]
            if cresciuta:
                print(dim("  cresce al calare del budget: " + ", ".join(
                    f"{n} {v1:.3f}->{v2:.3f} ms ({v2/v1:.2f}x)"
                    for n, v1, v2 in cresciuta)))
            else:
                print(dim("  nessuna voce cresce: il costo per miss e' costante, "
                          "quindi la velocita' scende solo perche' i miss aumentano."))

    # La lettura che interessa: quanto si paga per ogni dimezzamento della
    # residenza. Se il prezzo e' lineare la dimensione non ha un tetto davvero;
    # se collassa, il tetto c'e' e questa curva dice dove.
    lettura = esponente_residenza(rows)
    if lettura is not None:
        esponente, rapporto_res, rapporto_vel, inc = lettura
        print()
        print(kv("residenza ridotta di", f"{rapporto_res:.2f}x"))
        print(kv("velocita' ridotta di", f"{rapporto_vel:.2f}x"))
        print(kv("esponente", f"{esponente:.2f} +/- {inc:.2f}"))
        basso, alto = esponente - inc, esponente + inc
        soglie = [s for s in (0.35, 0.8) if basso < s < alto]
        if rapporto_res < SPAN_MINIMO_PER_VERDETTO or soglie:
            warn(f"nessun verdetto: l'intervallo [{basso:.2f}, {alto:.2f}] non "
                 f"distingue fra i casi.")
            if rapporto_res < SPAN_MINIMO_PER_VERDETTO:
                print(f"  L'escursione misurata e' {rapporto_res:.2f}x, troppo "
                      f"piccola: il denominatore e' log(escursione), quindi un "
                      f"intervallo corto")
                print(f"  amplifica il rumore. Servono almeno "
                      f"{SPAN_MINIMO_PER_VERDETTO:.0f}x fra il budget piu' alto e "
                      f"il piu' basso.")
                piu_alto = max(r["budget"] for r in rows)
                print(dim(f"  Con il punto alto a {fmt_bytes(piu_alto, 1)}, il "
                          f"basso dovrebbe scendere a "
                          f"{fmt_bytes(piu_alto / SPAN_MINIMO_PER_VERDETTO, 1)} "
                          f"o meno."))
        elif alto < 0.35:
            ok("la velocita' regge: un modello con molti piu' expert e lo stesso "
               "budget resterebbe utilizzabile")
        elif alto < 0.8:
            warn("la velocita' scende in modo sensibile: un modello molto piu' "
                 "grande sarebbe usabile ma non interattivo")
        else:
            error("la velocita' crolla quasi in proporzione: il tetto esiste, e "
                  "l'affermazione \"la dimensione non ha un tetto\" va corretta")
        print(dim("  L'esponente e' d(log velocita')/d(log residenza): 0 = la "
                  "residenza non conta, 1 = la velocita' e' proporzionale a essa."))


def _cmd_benchmark_sweep(args: argparse.Namespace) -> int:
    """Ripete il benchmark variando il numero di thread e sceglie il migliore.

    Su CPU ibride Intel (P-core + E-core) il valore ottimale non e' deducibile
    dal conteggio dei core: va misurato.
    """
    try:
        values = sorted({int(v) for v in str(args.sweep_threads).replace(" ", "").split(",") if v})
    except ValueError:
        error("--sweep-threads vuole una lista di interi, es. 4,6,8,12")
        return EXIT_USAGE
    if not values:
        error("--sweep-threads e' vuoto")
        return EXIT_USAGE

    fatale = _modello_esiste_o_errore(args)
    if fatale is not None:
        return fatale

    prompt = _bench_prompt(args)
    rows: List[Dict[str, Any]] = []
    info(f"sweep thread: {', '.join(str(v) for v in values)} "
         f"({args.repeat} ripetizioni ciascuno)")

    for n_cfg, t in enumerate(values):
        if n_cfg and args.cooldown > 0:
            info(f"pausa di raffreddamento {args.cooldown}s "
                 "(evita che la deriva termica falsi il confronto)")
            time.sleep(args.cooldown)
        print()
        print(header(f"THREADS = {t}"))
        sweep_args = argparse.Namespace(**vars(args))
        sweep_args.threads = t
        sweep_args.sweep_threads = None
        sweep_args.verbose = False
        code, ctx = _run_common(sweep_args)
        if ctx is None:
            warn(f"threads={t}: avvio non riuscito, salto")
            continue
        m, hw, p, rt, proc = ctx["model"], ctx["hw"], ctx["plan"], ctx["rt"], ctx["proc"]
        res = benchmod.run_benchmark(
            ctx["upstream"], prompt=prompt, n_predict=args.n_predict,
            repeat=args.repeat, warmup=not args.no_warmup,
        )
        stats = proc.monitor.stop() if proc.monitor else None
        if stats is not None:
            res.resources = stats.to_dict()
        proc.stop()
        rows.append({
            "threads": t,
            "gen_tok_s": res.gen_tok_s,
            "prompt_tok_s": res.prompt_tok_s,
            "ttft_s": res.ttft_s,
            "ram_peak": stats.ram_peak if stats else None,
            "cpu_mean": stats.cpu_mean if stats else None,
        })
        print(kv("generation tokens/sec",
                 bold(f"{res.gen_tok_s:.2f}") if res.gen_tok_s else "n/d"))
        print(kv("prompt tokens/sec", f"{res.prompt_tok_s:.1f}" if res.prompt_tok_s else "n/d"))
        telemetry.write_run(telemetry.build_record(
            event="benchmark_sweep", hardware=hw, model=m, plan=p,
            runtime={"backend": ctx["backend"], "build": rt.build},
            metrics=res.to_dict(), extra={"sweep_threads": t}))

    if not rows:
        error("nessuna configurazione ha prodotto risultati")
        return EXIT_RUNTIME

    print()
    print(header("CONFRONTO"))
    print(bold(f"  {'threads':>8}{'gen tok/s':>12}{'prompt tok/s':>14}{'TTFT':>9}{'RAM peak':>11}"))
    best = _pick_best_threads(rows)
    for r in rows:
        gen = f"{r['gen_tok_s']:.2f}" if r["gen_tok_s"] else "n/d"
        pp = f"{r['prompt_tok_s']:.1f}" if r["prompt_tok_s"] else "n/d"
        ttft = f"{r['ttft_s']:.2f}" if r["ttft_s"] else "n/d"
        ram = fmt_bytes(r["ram_peak"], 1) if r["ram_peak"] else "n/d"
        mark = " <-- migliore" if best and r["threads"] == best["threads"] else ""
        print(f"  {r['threads']:>8}{gen:>12}{pp:>14}{ttft:>9}{ram:>11}" + green(mark))
    if best:
        print()
        fastest = max((r for r in rows if r["gen_tok_s"]),
                      key=lambda r: r["gen_tok_s"], default=None)
        detail = f"{best['gen_tok_s']:.2f} tok/s"
        if best.get("ttft_s"):
            detail += f", TTFT {best['ttft_s']:.2f} s"
        ok(f"valore consigliato: --threads {best['threads']} ({detail})")
        if fastest and fastest["threads"] != best["threads"]:
            gain = (fastest["gen_tok_s"] / best["gen_tok_s"] - 1) * 100
            print(dim(f"  --threads {fastest['threads']} genera solo il {gain:.1f}% in piu' "
                      f"ma risponde piu' tardi: non vale lo scambio."))
    _thermal_note(rows)
    if args.json:
        print(json.dumps(rows, indent=2, default=str))
    return EXIT_OK


def _report_drift(res: benchmod.BenchResult) -> None:
    """Confronta prima e ultima ripetizione.

    Su un portatile la velocita' cala man mano che la CPU scalda: il valore utile
    per una generazione lunga e' quello sostenuto, non quello a freddo.
    """
    vals = [r.gen_tok_s for r in res.runs if r.gen_tok_s]
    if len(vals) < 2:
        return
    primo, ultimo = vals[0], vals[-1]
    calo = (primo - ultimo) / primo * 100 if primo else 0.0
    print(kv("a freddo / sostenuto", f"{primo:.2f} / {bold(f'{ultimo:.2f}')} tok/s"))
    if calo >= 5:
        print(kv("", yellow(f"la velocita' cala del {calo:.0f}% mentre la CPU scalda: "
                            f"per generazioni lunghe conta {ultimo:.2f} tok/s")))


def _cmd_benchmark_conversazione(args: argparse.Namespace) -> int:
    """Misura il TTFT turno dopo turno su una conversazione che cresce.

    E' il caso d'uso vero, e fino ad ora non lo misuravamo: ogni altra misura
    manda prompt indipendenti, mentre una chat e' un prefisso che si allunga. La
    cache del prefisso esiste per questo, quindi senza questo banco non si puo'
    decidere se valga la pena costruirne una su disco.
    """
    turni = max(2, int(args.conversazione))
    code, ctx = _run_common(args)
    if ctx is None:
        return code
    m, hw, p, rt, proc = ctx["model"], ctx["hw"], ctx["plan"], ctx["rt"], ctx["proc"]

    print()
    info(f"conversazione simulata: {turni} turni, il prefisso cresce a ogni turno")

    def avanza(i: int, n: int) -> None:
        sys.stdout.write(f"\r  turno {i}/{n}...".ljust(40))
        sys.stdout.flush()

    con = benchmod.run_conversazione(ctx["upstream"], turni=turni,
                                     n_predict=args.n_predict,
                                     cache_prompt=True, on_progress=avanza)
    senza = None
    if not args.no_baseline:
        info("ripeto senza riuso del prefisso, per il termine di paragone")
        senza = benchmod.run_conversazione(ctx["upstream"], turni=turni,
                                           n_predict=args.n_predict,
                                           cache_prompt=False, on_progress=avanza)
    sys.stdout.write("\r" + " " * 40 + "\r")

    print(header("CONVERSAZIONE"))
    for riga in benchmod.format_conversazione(con, senza):
        print(riga)

    # Il giudizio: il riuso regge mentre la conversazione cresce, o si perde?
    utili = [r for r in con if r.ttft_s]
    if len(utili) >= 2:
        primo, ultimo = utili[0].ttft_s, utili[-1].ttft_s
        print()
        if ultimo <= primo * 2:
            ok(f"il TTFT resta stabile mentre il prefisso cresce "
               f"({primo:.2f}s -> {ultimo:.2f}s): il riuso in RAM regge, e una "
               f"cache su disco servirebbe solo a riprendere dopo un riavvio o "
               f"ad alternare fra conversazioni diverse")
        else:
            warn(f"il TTFT cresce comunque ({primo:.2f}s -> {ultimo:.2f}s): il "
                 f"riuso in RAM non copre il caso, e una cache piu' capiente "
                 f"cambierebbe l'esperienza")

    stats = proc.monitor.stop() if proc.monitor else None
    proc.stop()
    telemetry.write_run(telemetry.build_record(
        event="benchmark_conversazione", hardware=hw, model=m, plan=p,
        runtime={"backend": ctx["backend"], "build": rt.build},
        metrics={"con_cache": [r.to_dict() for r in con],
                 "senza_cache": [r.to_dict() for r in senza] if senza else None},
        extra={"risorse": stats.to_dict() if stats else None}))
    if args.json:
        print(json.dumps([r.to_dict() for r in con], indent=2))
    return EXIT_OK


def _cmd_benchmark_sweep_prompt(args: argparse.Namespace) -> int:
    """Misura la velocita' al variare della lunghezza del contesto.

    Serve al KPI: "velocita' interattiva accettabile" non significa nulla senza
    dire a quale lunghezza di conversazione. L'attenzione costa, e su una
    macchina a corto di RAM anche la KV cache compete con i pesi.
    """
    try:
        lunghezze = sorted({int(v) for v in str(args.sweep_prompt).replace(" ", "").split(",") if v})
    except ValueError:
        error("--sweep-prompt vuole una lista di interi, es. 128,512,2048")
        return EXIT_USAGE
    if not lunghezze:
        error("--sweep-prompt e' vuoto")
        return EXIT_USAGE

    code, ctx = _run_common(args)
    if ctx is None:
        return code
    m, hw, p, rt, proc = ctx["model"], ctx["hw"], ctx["plan"], ctx["rt"], ctx["proc"]

    righe: List[Dict[str, Any]] = []
    info(f"sweep contesto: {', '.join(str(v) for v in lunghezze)} token")
    for n in lunghezze:
        if n > p.ctx:
            warn(f"{n} token superano il context configurato ({p.ctx}): salto")
            continue
        prompt = benchmod.DEFAULT_PROMPT
        while len(prompt) < int(n * 3.5):
            prompt += "\n" + benchmod.DEFAULT_PROMPT
        print()
        print(header(f"CONTESTO ~{n} TOKEN"))
        res = benchmod.run_benchmark(
            ctx["upstream"], prompt=prompt, n_predict=args.n_predict,
            repeat=args.repeat, warmup=False,
        )
        righe.append({
            "prompt_tokens": n,
            "gen_tok_s": res.gen_tok_s,
            "prompt_tok_s": res.prompt_tok_s,
            "ttft_s": res.ttft_s,
        })
        print(kv("generazione", bold(f"{res.gen_tok_s:.2f} tok/s") if res.gen_tok_s else "n/d"))
        print(kv("prompt", f"{res.prompt_tok_s:.1f} tok/s" if res.prompt_tok_s else "n/d"))
        print(kv("TTFT", f"{res.ttft_s:.2f} s" if res.ttft_s else "n/d"))
        telemetry.write_run(telemetry.build_record(
            event="benchmark_ctx", hardware=hw, model=m, plan=p,
            runtime={"backend": ctx["backend"], "build": rt.build},
            metrics=res.to_dict(), extra={"prompt_tokens": n}))

    stats = proc.monitor.stop() if proc.monitor else None
    proc.stop()

    if not righe:
        error("nessuna lunghezza misurata")
        return EXIT_RUNTIME

    print()
    print(header("VELOCITA' IN FUNZIONE DEL CONTESTO"))
    print(bold(f"  {'token':>8}{'gen tok/s':>12}{'prompt tok/s':>14}{'TTFT':>9}{'vs primo':>10}"))
    base = righe[0]["gen_tok_s"]
    for r in righe:
        g = f"{r['gen_tok_s']:.2f}" if r["gen_tok_s"] else "n/d"
        pp = f"{r['prompt_tok_s']:.1f}" if r["prompt_tok_s"] else "n/d"
        tt = f"{r['ttft_s']:.2f}" if r["ttft_s"] else "n/d"
        rel = (f"{(r['gen_tok_s']/base - 1)*100:+.0f}%"
               if (base and r["gen_tok_s"]) else "")
        print(f"  {r['prompt_tokens']:>8}{g:>12}{pp:>14}{tt:>9}{rel:>10}")
    if len(righe) >= 2 and base and righe[-1]["gen_tok_s"]:
        calo = (1 - righe[-1]["gen_tok_s"] / base) * 100
        if calo >= 10:
            print()
            warn(f"da {righe[0]['prompt_tokens']} a {righe[-1]['prompt_tokens']} token "
                 f"la generazione perde il {calo:.0f}%: dichiarare un solo valore di "
                 "tok/s senza indicare la lunghezza del contesto e' fuorviante")
    if stats is not None:
        print()
        print(kv("RAM peak", fmt_bytes(stats.ram_peak)))
        print(kv("RAM libera minima", fmt_bytes(stats.system_available_min)))
        print(kv("disk read", f"{fmt_bytes(stats.disk_read_total)} @ "
                              f"{stats.disk_read_mb_s:.1f} MB/s"))
    if args.json:
        print(json.dumps(righe, indent=2, default=str))
    return EXIT_OK


def _pick_best_threads(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Scelta del numero di thread.

    La generazione satura presto (e' limitata dalla banda di memoria), quindi un
    +1% di tok/s non giustifica un TTFT peggiore: fra le configurazioni entro il
    3% dalla migliore, preferiamo quella che risponde prima.
    """
    valid = [r for r in rows if r.get("gen_tok_s")]
    if not valid:
        return None
    top = max(r["gen_tok_s"] for r in valid)
    contenders = [r for r in valid if r["gen_tok_s"] >= top * 0.97]
    with_ttft = [r for r in contenders if r.get("ttft_s")]
    if with_ttft:
        return min(with_ttft, key=lambda r: r["ttft_s"])
    return min(contenders, key=lambda r: r["threads"])


def _thermal_note(rows: List[Dict[str, Any]]) -> None:
    """Segnala il caso in cui il prompt processing peggiora al crescere dei
    thread di generazione: su un portatile e' il budget termico, non il parametro."""
    pp = [(r["threads"], r["prompt_tok_s"]) for r in rows if r.get("prompt_tok_s")]
    if len(pp) < 3:
        return
    pp.sort()
    decrescente = all(pp[i][1] > pp[i + 1][1] for i in range(len(pp) - 1))
    calo = (pp[0][1] - pp[-1][1]) / pp[0][1] * 100 if pp[0][1] else 0
    if decrescente and calo >= 8:
        print()
        warn(f"il prompt processing cala del {calo:.0f}% al crescere dei thread, "
             "pur avendo --threads-batch costante: e' il budget termico/energetico "
             "della CPU, non l'effetto del parametro. Su questo portatile velocita' "
             "di prompt e di generazione si tolgono corrente a vicenda.")



# ================================================================= reorder ===

def cmd_reorder(args: argparse.Namespace) -> int:
    """Riscrive il GGUF con gli expert co-attivati contigui su disco."""
    m = _load_model(args.model)
    if not m.is_moe:
        error(f"{Path(m.path).name} non e' un modello MoE: niente da riordinare")
        return EXIT_USAGE

    if not Path(args.trace).exists():
        error(f"traccia non trovata: {args.trace}")
        print("  La traccia si ottiene con una build di llama.cpp patchata:")
        print("    argine run modello.gguf --route-log C:\\percorso\\route.log")
        print("  (vedi patches/README.md)")
        return EXIT_USAGE

    info(f"leggo la traccia {args.trace}...")
    tr = reordermod.parse_trace(args.trace)
    s = tr.summary()
    if not tr.layers:
        error("la traccia non contiene scelte di routing utilizzabili")
        return EXIT_USAGE
    print(header("TRACCIA"))
    print(kv("layer osservati", str(s["layers"])))
    print(kv("osservazioni", str(s["osservazioni"])))
    print(kv("expert distinti", str(s["expert_visti"])))
    print(kv("quota dei 5 piu' usati", f"{s['quota_top5']*100:.1f}%"))

    plan = reordermod.plan_reorder(m, tr)
    problemi = reordermod.verify_plan(m, plan)
    if problemi:
        print()
        error("riordino rifiutato:")
        for x in problemi:
            print(f"  - {x}")
        return EXIT_USAGE

    print(header("EFFETTO STIMATO"))
    print(kv("blocchi per token", f"{plan.media_prima:.2f} -> {plan.media_dopo:.2f}"))
    guadagno = plan.guadagno
    col = green if guadagno >= 1.15 else yellow
    print(kv("letture risparmiate", col(f"{guadagno:.2f}x")))
    if guadagno < 1.05:
        warn("guadagno trascurabile: il routing di questa traccia e' gia' quasi "
             "contiguo, oppure la traccia e' troppo corta")

    if args.dry_run:
        print()
        info("dry-run: nessun file scritto")
        return EXIT_OK

    out = args.output or str(Path(m.path).with_name(
        Path(m.path).stem + "-reordered.gguf"))
    dimensione = Path(m.path).stat().st_size
    libero = shutil.disk_usage(str(Path(out).parent)).free
    if libero < dimensione * 1.05:
        error(f"spazio insufficiente: servono {fmt_bytes(dimensione)}, "
              f"liberi {fmt_bytes(libero)}")
        return EXIT_USAGE

    print()
    info(f"scrivo {out}")
    res = reordermod.rewrite(m, plan, out, progress=lambda t: info("  " + t))
    print()
    ok(f"riordinato: {res['tensori']} tensori in {res['layer']} layer, "
       f"{fmt_bytes(res['byte_riscritti'])} riscritti")
    print(kv("guadagno", f"{res['guadagno']:.2f}x meno letture per token"))
    print(dim("  Il modello e' matematicamente identico: la permutazione degli "
              "expert e' applicata anche alle righe del router."))
    telemetry.write_run(telemetry.build_record(
        event="reorder", model=m, metrics=res, extra={"trace": args.trace}))
    if args.json:
        print(json.dumps(res, indent=2, default=str))
    return EXIT_OK


# ===================================================================== ppl ===

def cmd_ppl(args: argparse.Namespace) -> int:
    """Misura la perplessita': quanto il modello e' sorpreso da un testo.

    Serve a dare un numero al prezzo che si paga quantizzando. Senza, "un po'
    peggio ma piu' veloce" resta un'opinione, e una quantizzazione rotta e' una
    quantizzazione veloce.
    """
    m = _load_model(args.model)
    hw = hwmod.detect(extra_disk_paths=[str(Path(m.path).parent)])
    backend, _msgs = hwmod.resolve_backend("cpu", hw)
    try:
        rt = rtmod.ensure_runtime(backend, build=args.llama_build, offline=args.offline)
    except Exception as e:  # noqa: BLE001
        error(f"runtime llama.cpp non disponibile: {e}")
        return EXIT_RUNTIME
    if not rt.has_binary("perplexity"):
        error("questo runtime non contiene llama-perplexity")
        print(f"  cartella in uso: {rt.root}")
        print("  Serve la build patchata, che lo include:")
        print(cyan("    set ARGINE_LLAMA_DIR=") +
              cyan(str(Path(__file__).resolve().parents[1] / "llama-patched")))
        return EXIT_USAGE

    testo = Path(args.text) if args.text else qmod.testo_di_prova(
        logs_dir() / "testo-ppl.txt")
    if not testo.exists():
        error(f"testo non trovato: {testo}")
        return EXIT_USAGE

    piano = _piano_per_misura(m, hw, args)
    # I thread li sceglie il planner, non una regola inventata qui: su CPU
    # ibride Intel conta i soli P-core, ed e' una logica tarata misurando. Cosi'
    # `ppl` e `benchmark` usano lo stesso numero e i tempi restano confrontabili.
    threads = args.threads or (piano.threads if piano else None) \
        or max(1, (hw.cpu_cores_physical or 4))
    print(header("PERPLESSITA'"))
    print(kv("modello", Path(m.path).name))
    print(kv("testo", f"{testo.name} ({fmt_bytes(testo.stat().st_size)})"))
    print(kv("contesto", str(args.ctx)))
    print(kv("blocchi", str(args.chunks) if args.chunks else "tutti"))
    print(kv("thread", str(threads)))
    div = qmod.diversita(testo)
    print(kv("diversita' del testo", f"{div*100:.0f}% di frasi distinte"))
    if not args.text:
        warn("testo di ripiego generato da Argine: serve a confrontare due "
             "quantizzazioni fra loro, NON a produrre un numero paragonabile a "
             "quelli pubblicati (per quelli serve wikitext-2-raw)")
    if div < 0.5:
        warn("questo testo si ripete: dopo il primo giro anche un modello rotto "
             "predice bene la ripetizione, la perplessita' esce bassa per tutti e "
             "il confronto non distingue niente. Usane un altro con --text.")
    env = _expert_store_env(args, m, rt, piano) if getattr(args, "expert_store", None) else {}
    print()

    ultimo = [0]

    def avanzamento(i: int, ppl: float) -> None:
        if i != ultimo[0]:
            ultimo[0] = i
            sys.stdout.write(f"\r  blocco {i}, ppl parziale {ppl:.4f}".ljust(70))
            sys.stdout.flush()

    res = qmod.run_perplexity(rt.binary("perplexity"), m.path, str(testo),
                              ctx=args.ctx, threads=threads, chunks=args.chunks,
                              env=env or None, progress=avanzamento,
                              timeout=args.timeout)
    sys.stdout.write("\r" + " " * 70 + "\r")

    if res.errore_msg and res.ppl <= 0:
        error(res.errore_msg)
        for riga in res.raw_tail:
            print(dim("  " + riga))
        return EXIT_RUNTIME

    print(header("RISULTATO"))
    print(kv("perplessita'", f"{res.ppl:.4f} +/- {res.errore:.4f}"))
    print(kv("blocchi valutati", str(res.chunk)))
    print(kv("durata", f"{res.secondi:.0f}s"))
    print(dim(f"  firma della misura: {res.firma}"))

    storico_path = logs_dir() / "perplexity.jsonl"
    storico = qmod.carica(storico_path)
    altri = qmod.confrontabili(storico, res.firma, escludi=str(m.path))
    rif = qmod.trova_riferimento(storico, res.firma, escludi=str(m.path),
                                 nome=getattr(args, "riferimento", "") or "")
    if getattr(args, "riferimento", None) and rif is None:
        warn(f"nessuna misura di '{args.riferimento}' con questa firma: "
             f"uso il migliore disponibile")
        rif = qmod.trova_riferimento(storico, res.firma, escludi=str(m.path))

    if altri:
        print()
        print(header("CONFRONTO"))
        # Tutti i modelli confrontabili, non solo uno: con tre file in giro
        # sceglierne uno e tacere sugli altri e' il modo piu' rapido di leggere
        # il numero sbagliato.
        righe = [(Path(r.model).name, r.ppl, r.errore) for r in altri]
        righe.append((Path(m.path).name + "  <-- questo", res.ppl, res.errore))
        largh = max(len(n) for n, _p, _e in righe)
        for nome, ppl, err in sorted(righe, key=lambda x: x[1]):
            print(f"  {nome:<{largh}}  {ppl:8.4f} +/- {err:.4f}")
        print()
        if rif:
            print(kv("riferimento", Path(rif.model).name))
            c = qmod.Confronto(riferimento=rif, candidato=res)
            for riga in c.giudizio():
                print("  " + riga)
            if len(altri) > 1:
                print(dim("  (il riferimento e' il migliore fra i confrontabili; "
                          "scegline un altro con --riferimento NOME)"))
    else:
        print()
        info("nessuna misura precedente con la stessa firma: questo diventa il "
             "riferimento per i confronti futuri")

    qmod.salva(res, storico_path)
    telemetry.write_run(telemetry.build_record(
        event="perplexity", model=m, metrics=res.to_dict()))
    if args.json:
        print(json.dumps(res.to_dict(), indent=2, ensure_ascii=False))
    return EXIT_OK


# ============================================================== quantize =====

def _runtime_con(kind: str, args: argparse.Namespace, m: GGUFModel):
    """Runtime che contiene un binario preciso, o un errore che dice cosa fare."""
    hw = hwmod.detect(extra_disk_paths=[str(Path(m.path).parent)])
    backend, _msgs = hwmod.resolve_backend("cpu", hw)
    rt = rtmod.ensure_runtime(backend, build=args.llama_build,
                              offline=getattr(args, "offline", False))
    if not rt.has_binary(kind):
        raise RuntimeError(
            f"questo runtime non contiene llama-{kind}.\n"
            f"  cartella in uso: {rt.root}\n"
            f"  Serve la build patchata, che lo include:\n"
            f"    set ARGINE_LLAMA_DIR="
            f"{Path(__file__).resolve().parents[1] / 'llama-patched'}")
    return hw, rt


def _stampa_stima(m: GGUFModel, st: qzmod.Stima, budget: int) -> None:
    print(header("COSA USCIRA'"))
    for v in st.voci:
        etichetta = v.classe
        if v.tipo_dopo and v.tipo_dopo != "invariato":
            etichetta += f" -> {v.tipo_dopo}"
        delta = ""
        if v.risparmio > 0:
            delta = f"  (-{fmt_bytes(v.risparmio)})"
        print(kv(etichetta,
                 f"{fmt_bytes(v.byte_prima)} -> {fmt_bytes(v.byte_dopo)}{delta}"))
    print()
    print(kv("file totale", f"{fmt_bytes(st.byte_prima)} -> "
                            f"{fmt_bytes(st.byte_dopo)} "
                            f"(-{fmt_bytes(st.risparmio)})"))
    p = m.n_params_total
    if p:
        print(kv("bit per peso", f"{m.bits_per_weight:.2f} -> "
                                 f"{st.byte_dopo * 8.0 / p:.2f}"))
    print(kv("budget RAM", fmt_bytes(budget)))
    if st.byte_dopo <= budget * 0.90:
        ok(f"ci sta in RAM con {fmt_bytes(budget - st.byte_dopo)} di margine: "
           f"lo streaming da disco non serve piu' e il collo di bottiglia "
           f"misurato sparisce")
    elif st.byte_dopo <= budget:
        warn("ci sta appena: resta poco per KV cache e contesto, la cache degli "
             "expert andra' comunque tenuta")
    else:
        warn(f"NON ci sta: mancano {fmt_bytes(st.byte_dopo - budget)}. "
             f"Serve un tipo piu' piccolo, oppure lo store resta necessario.")
        scelto, note = qzmod.scegli_tipo_per_budget(m, budget)
        for n in note:
            print(dim("    " + n))
        if scelto:
            print(cyan(f"    --experts {scelto}"))
    for a in st.avvisi:
        print(dim("  nota: " + a))


def cmd_quantize(args: argparse.Namespace) -> int:
    """Riscrive gli expert con meno bit per farli entrare in RAM.

    Non e' un compromesso fra qualita' e velocita' in astratto: e' il tentativo
    di eliminare la causa unica di lentezza che questo progetto ha misurato,
    cioe' leggere 16 GiB di expert da disco a ogni generazione.
    """
    m = _load_model(args.model)
    tq = qzmod.tipo(args.experts)
    if tq is None:
        error(f"tipo sconosciuto: {args.experts}")
        print("  tipi noti, dal piu' piccolo:")
        for t in qzmod.tipi_disponibili():
            if 1.5 <= t.bpw <= 6.6:
                marchio = " (richiede imatrix)" if t.imatrix else ""
                print(f"    {t.nome:<9} {t.bpw:5.2f} bit{marchio}")
        return EXIT_USAGE

    hw = hwmod.detect(extra_disk_paths=[str(Path(m.path).parent)])
    budget, note_budget = planmod.safe_budget(hw, parse_size(args.ram_limit)
                                              if args.ram_limit else None)

    print(header("QUANTIZZAZIONE EXPERT"))
    print(kv("modello", Path(m.path).name))
    print(kv("quantizzazione attuale", m.quantization))
    print(kv("expert -> ", f"{tq.nome} ({tq.bpw:.2f} bit per peso)"))
    if args.rest:
        print(kv("resto del modello ->", args.rest))
    else:
        print(kv("resto del modello", "invariato"))
    if tq.nota:
        print(dim("  " + tq.nota))
    print()

    st = qzmod.stima(m, tq.nome, args.rest, args.embd)
    _stampa_stima(m, st, budget)
    for n in note_budget:
        print(dim("  budget: " + n))

    imat = Path(args.imatrix) if args.imatrix else None
    if imat is None:
        auto = logs_dir() / (Path(m.path).stem + ".imatrix")
        if auto.exists():
            imat = auto
            info(f"uso la matrice di importanza trovata accanto ai log: {imat.name}")

    print()
    pre = qzmod.controlla(m, tq.nome, imat, args.rest, args.allow_requantize,
                          usa_tensor_type=not args.no_tensor_type)
    for a in pre.avvisi:
        warn(a)
    if not pre.ok:
        for e in pre.errori:
            error(e)
        return EXIT_USAGE

    if args.dry_run:
        print()
        info("--dry-run: nessun file prodotto")
        return EXIT_OK

    try:
        _hw, rt = _runtime_con("quantize", args, m)
    except Exception as e:  # noqa: BLE001
        error(str(e))
        return EXIT_RUNTIME

    binario = rt.binary("quantize")
    tt = qzmod.supporta_tensor_type(binario)
    if tt is False and not args.rest:
        error("questa build di llama-quantize non accetta --tensor-type, "
              "quindi non puo' quantizzare i soli expert.")
        print("  Due strade: aggiornare llama.cpp, oppure quantizzare tutto "
              "il modello con --rest " + tq.nome)
        return EXIT_USAGE
    if tt is None:
        warn("non sono riuscito a leggere l'elenco delle opzioni di "
             "llama-quantize: provo comunque con --tensor-type")

    out = Path(args.out) if args.out else (
        Path(m.path).with_name(Path(m.path).stem + f"-experts-{tq.nome}.gguf"))
    libero = shutil.disk_usage(str(out.parent)).free
    if libero < st.byte_dopo * 1.05:
        error(f"spazio insufficiente in {out.parent}: servono circa "
              f"{fmt_bytes(st.byte_dopo)}, liberi {fmt_bytes(libero)}")
        return EXIT_USAGE
    if out.exists() and not args.force:
        error(f"esiste gia': {out}  (--force per sovrascrivere)")
        return EXIT_USAGE

    threads = args.threads or hw.cpu_cores_physical or 8
    print()
    info(f"scrivo {out}")
    print(dim(f"  thread {threads}; la quantizzazione legge tutto il modello "
              f"una volta, quindi durera' quanto la lettura di "
              f"{fmt_bytes(m.total_size_on_disk)} da disco piu' il calcolo"))
    print()

    ultimo = [0]

    def avanzamento(fatti: int, totale: int, nome: str) -> None:
        if fatti != ultimo[0]:
            ultimo[0] = fatti
            pct = 100.0 * fatti / totale if totale else 0.0
            sys.stdout.write(f"\r  tensore {fatti}/{totale} ({pct:.0f}%) "
                             f"{nome[:40]}".ljust(78))
            sys.stdout.flush()

    res = qzmod.run_quantize(
        binario, m.path, str(out), tq.nome, resto_type=args.rest,
        embd_type=args.embd, imatrix=str(imat) if imat else None,
        threads=threads, consenti_riquantizzazione=args.allow_requantize,
        usa_tensor_type=(tt is not False) and not args.no_tensor_type,
        progress=avanzamento, timeout=args.timeout)
    sys.stdout.write("\r" + " " * 78 + "\r")

    if res.errore_msg:
        error(res.errore_msg)
        if res.comando:
            # Il comando esatto, perche' quando llama-quantize rifiuta un
            # argomento non dice quale: vederli tutti e' l'unico modo di
            # trovarlo, e riprovarlo a mano e' il modo piu' rapido di isolarlo.
            print(dim("  comando eseguito:"))
            print(dim("    " + " ".join(res.comando)))
        for riga in res.coda:
            print(dim("  " + riga))
        return EXIT_RUNTIME

    print(header("FATTO"))
    print(kv("file", str(out)))
    print(kv("dimensione", f"{fmt_bytes(res.byte_finali)} "
                           f"(stimata {fmt_bytes(st.byte_dopo)}, "
                           f"scarto {100.0*abs(res.byte_finali-st.byte_dopo)/max(1,st.byte_dopo):.1f}%)"))
    print(kv("durata", f"{res.secondi/60:.1f} min"))
    print()
    info("il numero che conta adesso e' la perplessita': senza, "
         "'piu' piccolo e piu' veloce' non dice se e' ancora lo stesso modello")
    print(cyan(f"  argine ppl {out} --ctx 512 --chunks 18"))
    print(dim("  (stessi parametri della misura di riferimento, altrimenti i "
              "due numeri non si possono confrontare)"))

    telemetry.write_run(telemetry.build_record(
        event="quantize", model=m, metrics=res.to_dict(),
        extra={"stima_byte": st.byte_dopo}))
    if args.json:
        print(json.dumps(res.to_dict(), indent=2, ensure_ascii=False))
    return EXIT_OK


def cmd_imatrix(args: argparse.Namespace) -> int:
    """Calcola la matrice di importanza: quali pesi meritano i bit che restano.

    A 2 bit la griglia e' cosi' grossolana che distribuirla uniformemente
    distrugge il modello. La matrice misura quanto ogni colonna di ogni tensore
    contribuisce all'uscita su un testo di calibrazione, e la quantizzazione la
    usa per spendere la precisione dove serve. llama.cpp non produce affatto i
    tipi IQ2 senza: non e' un'opzione di qualita', e' un prerequisito.
    """
    m = _load_model(args.model)
    try:
        hw, rt = _runtime_con("imatrix", args, m)
    except Exception as e:  # noqa: BLE001
        error(str(e))
        return EXIT_RUNTIME

    if args.text:
        testo = Path(args.text)
    else:
        # File diverso da quello della perplessita', e non per ordine: il testo
        # della perplessita' entra nella firma della misura, e allungarlo
        # renderebbe il riferimento gia' preso inconfrontabile con tutto cio'
        # che viene dopo.
        # Qui invece serve lungo: la matrice si calcola su un blocco alla volta,
        # e un testo che finisce prima dei blocchi richiesti li fa finire li'.
        blocchi = args.chunks or 128
        parole = int(args.ctx * blocchi / 1.2) + 2000
        testo = qmod.testo_di_prova(logs_dir() / "testo-imatrix.txt",
                                    parole=parole, seed=23)
    if not testo.exists():
        error(f"testo non trovato: {testo}")
        return EXIT_USAGE

    out = Path(args.out) if args.out else (
        logs_dir() / (Path(m.path).stem + ".imatrix"))
    piano = _piano_per_misura(m, hw, args)
    threads = args.threads or (piano.threads if piano else None) \
        or max(1, hw.cpu_cores_physical or 4)

    print(header("MATRICE DI IMPORTANZA"))
    print(kv("modello", Path(m.path).name))
    print(kv("testo", f"{testo.name} ({fmt_bytes(testo.stat().st_size)})"))
    print(kv("contesto", str(args.ctx)))
    print(kv("blocchi", str(args.chunks) if args.chunks else "tutti"))
    print(kv("thread", str(threads)))
    print(kv("uscita", str(out)))
    div = qmod.diversita(testo)
    print(kv("diversita' del testo", f"{div*100:.0f}% di frasi distinte"))
    if not args.text:
        warn("testo di ripiego generato da Argine. Per la matrice conta piu' "
             "che per la perplessita': misura quali pesi servono, e un testo "
             "povero produce una matrice che protegge le cose sbagliate. Un "
             "corpus vario (wikitext, o testi tuoi) via --text da' un risultato "
             "migliore.")
    if m.is_moe:
        info(f"modello MoE con {m.expert_count} expert per layer: perche' la "
             f"matrice copra anche gli expert rari servono molti blocchi "
             f"(qualche centinaio), non i 18 che bastano a una perplessita'")
    env = _expert_store_env(args, m, rt, piano) if getattr(args, "expert_store", None) else {}
    if env:
        info("expert store richiesto anche qui: gli expert si leggono con "
             "letture esplicite invece che dalla memoria mappata")
    print()

    ultimo = [0]

    def avanzamento(n: int) -> None:
        if n != ultimo[0]:
            ultimo[0] = n
            sys.stdout.write(f"\r  blocco {n}".ljust(60))
            sys.stdout.flush()

    res = qzmod.run_imatrix(rt.binary("imatrix"), m.path, str(testo), str(out),
                            ctx=args.ctx, threads=threads, chunks=args.chunks,
                            env=env or None, progress=avanzamento,
                            timeout=args.timeout)
    sys.stdout.write("\r" + " " * 60 + "\r")

    if res.errore_msg:
        error(res.errore_msg)
        if res.comando:
            print(dim("  comando eseguito:"))
            print(dim("    " + " ".join(res.comando)))
        for riga in res.coda:
            print(dim("  " + riga))
        return EXIT_RUNTIME

    print(header("RISULTATO"))
    print(kv("matrice", str(out)))
    print(kv("dimensione", fmt_bytes(res.byte)))
    print(kv("blocchi", str(res.chunk)))
    print(kv("durata", f"{res.secondi/60:.1f} min"))
    print()
    info("ora la quantizzazione IQ2 e' possibile:")
    print(cyan(f"  argine quantize {m.path} "
               f"--experts IQ2_XXS --imatrix {out} --allow-requantize"))

    telemetry.write_run(telemetry.build_record(
        event="imatrix", model=m, metrics=res.to_dict()))
    if args.json:
        print(json.dumps(res.to_dict(), indent=2, ensure_ascii=False))
    return EXIT_OK


# =============================================================== diskbench ===

def cmd_diskbench(args: argparse.Namespace) -> int:
    """Misura il tetto dello storage sul pattern di accesso degli expert.

    Esiste per rispondere a una domanda che decide dove va il lavoro: i
    290-562 MB/s che si vedono in generazione sono il disco o il percorso
    software? Finche' la risposta non e' misurata, qualunque ottimizzazione
    dell'I/O e' una scommessa.
    """
    percorso = Path(args.file) if args.file else None
    if percorso is None:
        error("serve un file grande su cui misurare: passa il GGUF del modello")
        return EXIT_USAGE
    if not percorso.exists():
        try:
            percorso = Path(find_gguf(str(percorso)))
        except GGUFError as exc:
            error(str(exc))
            return EXIT_USAGE

    dimensione = percorso.stat().st_size
    hw = hwmod.detect(extra_disk_paths=[str(percorso.parent)])
    ram = hw.ram_total or 0
    print(header("MISURA DELLO STORAGE"))
    print(kv("file", percorso.name))
    print(kv("dimensione", fmt_bytes(dimensione)))
    if ram:
        print(kv("RAM totale", fmt_bytes(ram)))
        if dimensione < ram:
            warn(f"il file ({fmt_bytes(dimensione)}) sta dentro la RAM "
                 f"({fmt_bytes(ram)}): le prove con cache attiva misureranno la "
                 "RAM, non il disco. Il verdetto si basa su 'direct', che non "
                 "e' influenzato, ma usa --purge per rendere onesto il controllo.")

    grezze = args.slice or [str(s) for s in dbmod.DEFAULT_SLICES]
    slices = []
    for x in grezze:
        n = parse_size(x) or 0
        if n < dbmod.SECTOR:
            error(f"slice troppo piccola: {x} (minimo {dbmod.SECTOR} byte)")
            return EXIT_USAGE
        # l'I/O non bufferizzato pretende multipli del settore
        slices.append(n - (n % dbmod.SECTOR))
    threads = [int(t) for t in (args.threads or dbmod.DEFAULT_THREADS)]
    kinds = list(args.kinds)
    suite = dbmod.Suite(percorso, slices=slices, threads=threads, kinds=kinds,
                        seconds=args.seconds, sequential=not args.no_sequential,
                        purge=parse_size(args.purge) or 0)
    if args.baseline and args.baseline > 0:
        suite.PRODUZIONE_MB_S = float(args.baseline)
    prove = (1 if suite.sequential else 0) + len(kinds) * len(slices) * len(threads)
    print(kv("prove", f"{prove} x {args.seconds:g}s = circa "
                      f"{prove * args.seconds / 60:.1f} min"))
    print()

    suite.run(progress=lambda m: info(m))

    print()
    print(header("RISULTATI"))
    for riga in dbmod.format_results(suite.results):
        print(riga)
    print(dim("  MB/s = byte consegnati; disco = byte letti davvero dal "
              "dispositivo (contatore di sistema); cache = quota servita "
              "senza toccare il disco"))
    print()
    print(header("DECISIONE"))
    for riga in suite.verdict(getattr(args, 'coda', None)):
        if riga.startswith("VERDETTO"):
            print()
            testo = riga
            print("  " + (green(testo) if "vale il lavoro" in testo else
                          yellow(testo) if "margine stretto" in testo else red(testo)))
        else:
            print(f"  - {riga}")

    telemetry.write_run(telemetry.build_record(
        event="diskbench",
        metrics={
            "file": str(percorso), "bytes": dimensione,
            "risultati": [
                {"kind": r.kind, "slice": r.slice_bytes, "threads": r.threads,
                 "mb_s": round(r.mb_s, 1),
                 "phys_mb_s": None if r.phys_mb_s is None else round(r.phys_mb_s, 1),
                 "iops": round(r.iops, 1), "latency_ms": round(r.latency_ms, 3),
                 "cache_ratio": None if r.cache_ratio is None else round(r.cache_ratio, 3),
                 "error": r.error}
                for r in suite.results
            ],
        }))
    if args.json:
        print(json.dumps([r.__dict__ for r in suite.results], indent=2, default=str))
    return EXIT_OK


# =================================================================== store ===

def _default_store_path(model_path: str) -> Path:
    return Path(model_path).with_suffix(".psxs")


def _consigli_non_avviabile(m: GGUFModel, hw: Any, p: planmod.RuntimePlan,
                            args: argparse.Namespace) -> List[str]:
    """Cosa fare quando il piano non sta nel budget: solo le cose applicabili.

    La versione precedente stampava quattro consigli fissi, e nel caso che conta
    piu' di tutti — expert store attivo, `--expert-budget` chiesto a mano, ctx e
    KV cache gia' scesi al minimo dalle riduzioni automatiche — **tre dei quattro
    non si applicavano**: "un GGUF piu' piccolo" (con lo store la dimensione del
    file non e' il vincolo, i residenti sono 1,35 GiB su 35,67), "--ctx 1024
    --kv-type q4_0" (era gia' a 512 e Q4_0) e "lascia attivo l'expert streaming"
    (era attivo). L'unica cosa da fare stava in un avviso trenta righe sopra.

    E' lo stesso difetto del messaggio di `setup` che ripeteva l'impostazione
    rotta: un elenco di rimedi che include quello che hai gia' fatto non e' un
    aiuto, e' rumore che nasconde l'unica riga utile.
    """
    out: List[str] = []
    mancano = p.estimate.total - p.budget

    # 1. Il budget degli expert chiesto a mano che non ci sta: e' la causa, e la
    #    via piu' corta e' una delle due qui sotto.
    chiesto = parse_size(getattr(args, "expert_budget", None))
    if p.expert_store and chiesto and p.expert_cache_budget >= chiesto:
        # Il numero sostenibile lo dice il planner, non lo ricalcoliamo noi: la
        # prima versione di questa riga lo rifaceva a mano e stampava "0 B" dove
        # l'avviso del planner diceva 512 MiB. Due numeri in contrasto nella
        # stessa schermata sono peggio di nessun numero.
        resta = p.expert_cache_sostenibile
        out.append(
            f"abbassa --expert-budget: con questa RAM libera il piano lascia agli "
            f"expert {fmt_bytes(resta)}, non {fmt_bytes(chiesto)}"
            if resta else
            f"abbassa --expert-budget: {fmt_bytes(chiesto)} non ci stanno con "
            f"questa RAM libera")
        out.append(
            f"oppure libera {fmt_bytes(mancano)} di RAM e tieni "
            f"{fmt_bytes(chiesto)}: serve per confrontare la misura con un'altra "
            f"presa allo stesso budget")

    # 2. RAM libera contro RAM totale: un numero da raggiungere, non un invito
    #    generico a chiudere il browser.
    libera = getattr(hw, "ram_available", 0) or 0
    totale = getattr(hw, "ram_total", 0) or 0
    if libera and totale and libera < totale * 0.5:
        out.append(
            f"chiudi applicazioni: libere {fmt_bytes(libera)} su {fmt_bytes(totale)}, "
            f"e il piano si dimensiona su quelle libere perche' pianificare su "
            f"memoria occupata da altri significa pianificare di paginare")

    # 3. Il contesto, solo se c'e' ancora margine per scenderlo.
    if p.ctx > planmod.MIN_CTX:
        out.append(f"--ctx {max(planmod.MIN_CTX, p.ctx // 2)} "
                   f"(ora {p.ctx})")
    if p.kv_type_k.lower() in ("f16", "bf16", "f32"):
        out.append(f"--kv-type q4_0 (ora K={p.kv_type_k})")

    # 4. Lo streaming degli expert, solo se e' spento.
    if m.is_moe and not (p.moe_streaming or p.expert_store):
        out.append("lascia attivo l'expert streaming (non usare --no-streaming)")

    # 5. Un modello piu' piccolo: solo se sono i residenti a non entrare, perche'
    #    con lo store il file da 35 GiB contribuisce per 1,35.
    if p.estimate.weights_resident + p.estimate.repack > p.budget:
        out.append(f"un GGUF piu' piccolo: i soli pesi residenti sono "
                   f"{fmt_bytes(p.estimate.weights_resident + p.estimate.repack)}, "
                   f"oltre il budget di {fmt_bytes(p.budget)}")
    if not out:
        out.append(f"--ram-limit piu' alto: mancano {fmt_bytes(mancano)}")
    return out


def _conta_verifica(m: GGUFModel, samples: int) -> Tuple[int, int]:
    """Quanti gruppi verifichera' `esmod.verify`, e su quanti layer.

    Si chiede alla stessa funzione che li scegliera', invece di ricalcolarlo:
    prima il messaggio diceva `samples + 2` per conto suo, ed era un numero
    destinato a diventare falso al primo cambio nella scelta dei campioni. Lo e'
    diventato: ora i gruppi sono uno per layer piu' i campioni piu' gli estremi,
    senza duplicati.
    """
    layers = sorted(esmod.expert_tensors(m))
    coppie = esmod.coppie_di_verifica(layers, m.expert_count, samples)
    return len(coppie), len({l for l, _ in coppie})


def _slice_per_tipo(piano: Any) -> List[Tuple[str, List[Tuple[int, int]]]]:
    """Per ogni tipo di tensore, le dimensioni di slice distinte e quanti layer.

    Serve perche' la dimensione **non e' una proprieta' del modello**: dipende
    dalla quantizzazione, e i quantizzatori la variano per layer. Su
    Qwen3-Next-80B-A3B Q3_K_M, `down` e' Q4_K su 45 layer e Q5_K su 3, quindi le
    slice sono 576 KiB su 45 layer e 704 su 3.
    """
    per_tipo: List[Tuple[str, List[Tuple[int, int]]]] = []
    for k, nome in enumerate(esmod.KINDS):
        layer_per_dim: Dict[int, set] = {}
        for g in piano.groups:
            layer_per_dim.setdefault(g.slabs[k].n_bytes, set()).add(g.layer)
        voci = sorted(((dim, len(layers)) for dim, layers in layer_per_dim.items()),
                      key=lambda x: -x[1])
        per_tipo.append((nome, voci))
    return per_tipo


def _stampa_slice_per_tipo(piano: Any) -> None:
    """Le dimensioni di lettura, una riga per tipo.

    La versione precedente stampava le tre slice del **primo gruppo** sotto
    l'etichetta "tre letture da", come se descrivessero il modello. Su un modello
    a quantizzazione mista non lo fanno: su Qwen3-Next il layer 0 ha `down` da 704
    KiB mentre 45 layer su 48 lo hanno da 576, e chi moltiplicava i numeri a
    schermo per 512 expert e 48 layer otteneva 38,1 GiB contro i 34,31 dichiarati
    due righe sopra. Due numeri sulla stessa schermata che non si moltiplicano fra
    loro fanno sospettare lo strumento, ed e' giusto che lo facciano.
    """
    n_gruppi = len(piano.groups) or 1
    medio = piano.raw_bytes / n_gruppi
    print(kv("expert completo", f"{fmt_bytes(medio)} in media "
                                f"({n_gruppi} gruppi)"))
    for nome, voci in _slice_per_tipo(piano):
        if len(voci) == 1:
            testo = fmt_bytes(voci[0][0])
        else:
            # varia per layer: dirlo, e dire quanto spesso
            testo = ", ".join(f"{fmt_bytes(dim)} su {n} layer" for dim, n in voci)
        print(kv(f"  lettura {nome}", testo))


def cmd_store(args: argparse.Namespace) -> int:
    """Costruisce, ispeziona, verifica o misura l'expert store."""
    m = _load_model(args.model)
    store = Path(args.store) if args.store else _default_store_path(m.path)

    if args.action == "info":
        if not store.exists():
            error(f"nessuno store in {store}")
            print(f"  lo costruisci con:  argine store build {Path(m.path).name}")
            return EXIT_USAGE
        try:
            idx = esmod.read_index(store)
        except esmod.StoreError as exc:
            error(str(exc))
            return EXIT_RUNTIME
        print(header("EXPERT STORE"))
        print(kv("file", store.name))
        print(kv("dimensione", fmt_bytes(store.stat().st_size)))
        print(kv("layer", str(len(idx.layers))))
        print(kv("expert per layer", str(idx.n_experts)))
        print(kv("gruppi", str(len(idx.entries))))
        print(kv("expert completo", fmt_bytes(idx.group_bytes)))
        print(kv("tre letture da", ", ".join(fmt_bytes(x) for x in idx.slice_bytes)))
        print(kv("allineamento", f"{idx.alignment} byte"))
        grezzi = sum(e.raw_bytes for e in idx.entries)
        print(kv("pesi", fmt_bytes(grezzi)))
        print(kv("riempimento", fmt_bytes(store.stat().st_size - idx.data_offset - grezzi)))
        print(dim("  Ogni slice allineata al settore e leggibile con "
                  "FILE_FLAG_NO_BUFFERING direttamente nella memoria del suo tensore: "
                  "nessuna copia, e la RAM non viene contata due volte, che e' la "
                  "condizione per un budget rigido."))
        return EXIT_OK

    if args.action == "verify":
        if not store.exists():
            error(f"nessuno store in {store}")
            return EXIT_USAGE
        quanti, quanti_layer = _conta_verifica(m, args.samples)
        info(f"confronto {quanti} gruppi con il GGUF, byte per byte "
             f"(ogni layer coperto: {quanti_layer})...")
        problemi = esmod.verify(m, store, samples=args.samples,
                                progress=lambda s: info("  " + s))
        if problemi:
            error("lo store NON corrisponde al modello:")
            for x in problemi:
                print(f"  - {x}")
            return EXIT_RUNTIME
        ok(f"{quanti} gruppi identici al GGUF, tutti i {quanti_layer} layer "
           f"coperti (estremi del file inclusi)")
        return EXIT_OK

    if args.action == "bench":
        if not store.exists():
            warn(f"nessuno store in {store}: misuro solo le strade sul GGUF")
            store_arg = None
        else:
            store_arg = store
        print(header("COME ARRIVANO GLI EXPERT"))
        print(kv("thread", str(args.threads)))
        print(kv("durata per strada", f"{args.seconds:g}s"))
        print(dim("  mmap   = page fault dal GGUF          3 letture disallineate, cache si'"))
        print(dim("  gguf   = letture esplicite dal GGUF   3 letture disallineate, cache si'"))
        print(dim("  storeb = letture esplicite dallo store 3 letture allineate,   cache si'"))
        print(dim("  store  = come storeb                  3 letture allineate,    cache no"))
        print(dim("  Tre letture su ogni strada: mmap->gguf isola il page fault, "
                  "gguf->storeb l'allineamento, storeb->store la cache."))
        print()
        res = esmod.bench_paths(m, store_arg, threads=args.threads,
                                seconds=args.seconds,
                                purge=parse_size(args.purge) or 0)
        if not res:
            error("nessuna misura possibile su questo modello")
            return EXIT_RUNTIME
        for riga in esmod.format_paths(res):
            print(riga)
        print()
        print(header("LETTURA"))
        gruppo = 0
        if store_arg:
            try:
                gruppo = esmod.read_index(store_arg).group_bytes
            except esmod.StoreError:
                gruppo = 0
        for riga in esmod.paths_verdict(res, group_bytes=gruppo):
            print(f"  - {riga}")
        telemetry.write_run(telemetry.build_record(
            event="store_bench", model=m,
            metrics={"threads": args.threads,
                     "strade": [{"label": r.label, "mb_s": round(r.mb_s, 1),
                                 "experts_s": round(r.experts_s, 1),
                                 "us_per_expert": round(r.us_per_expert, 1),
                                 "error": r.error} for r in res]}))
        return EXIT_OK

    # ------------------------------------------------------------ build ---
    problemi = esmod.check_model(m)
    if problemi:
        error("questo modello non puo' produrre un expert store:")
        for x in problemi:
            print(f"  - {x}")
        return EXIT_USAGE

    ordine: Optional[Dict[int, List[int]]] = None
    if args.trace:
        if not Path(args.trace).exists():
            error(f"traccia non trovata: {args.trace}")
            return EXIT_USAGE
        info(f"leggo la traccia {args.trace} per l'ordine fisico...")
        tr = reordermod.parse_trace(args.trace)
        if not tr.layers:
            warn("la traccia non contiene routing utilizzabile: uso l'ordine naturale")
        else:
            plan_r = reordermod.plan_reorder(m, tr)
            ordine = {l: list(p) for l, p in plan_r.perms.items()}
            print(kv("blocchi per token", f"{plan_r.media_prima:.2f} -> "
                                         f"{plan_r.media_dopo:.2f}"))
            print(dim("  Qui la permutazione e' solo layout: l'indice dello store fa da "
                      "indirezione, quindi i pesi del router NON vanno toccati e il "
                      "modello resta identico."))

    piano = esmod.build_plan(m, order=ordine)
    serve = piano.total_bytes
    libero = shutil.disk_usage(str(store.parent)).free
    print(header("PIANO"))
    print(kv("destinazione", str(store)))
    print(kv("dimensione", fmt_bytes(serve)))
    print(kv("  di cui pesi", fmt_bytes(piano.raw_bytes)))
    print(kv("  riempimento", fmt_bytes(piano.padding_bytes)))
    _stampa_slice_per_tipo(piano)
    print(kv("spazio libero", fmt_bytes(libero)))
    if libero < serve * 1.02:
        error(f"spazio insufficiente: servono {fmt_bytes(serve)}, "
              f"liberi {fmt_bytes(libero)}")
        return EXIT_USAGE
    if args.dry_run:
        print()
        info("dry-run: nessun file scritto")
        return EXIT_OK

    print()
    info("scrivo lo store (una passata sul GGUF, niente di grande in memoria)...")
    res = esmod.write_store(m, piano, store, progress=lambda s: info("  " + s))
    print()
    ok(f"store scritto: {res['gruppi']} gruppi, {fmt_bytes(res['bytes'])}")
    print(kv("pesi", fmt_bytes(res["expert_bytes"])))
    print(kv("riempimento", fmt_bytes(res["padding_bytes"])))
    print(kv("impronta dei pesi", str(res["digest"])))

    if not args.no_verify:
        print()
        info("verifica: confronto dei byte con il GGUF...")
        problemi = esmod.verify(m, store, samples=args.samples)
        if problemi:
            error("lo store NON corrisponde al modello: lo considero inutilizzabile")
            for x in problemi:
                print(f"  - {x}")
            return EXIT_RUNTIME
        quanti, quanti_layer = _conta_verifica(m, args.samples)
        ok(f"{quanti} gruppi verificati byte per byte, tutti i {quanti_layer} "
           f"layer coperti")
    telemetry.write_run(telemetry.build_record(
        event="store_build", model=m, metrics=res,
        extra={"trace": args.trace, "ordinato": bool(ordine)}))
    if args.json:
        print(json.dumps(res, indent=2, default=str))
    return EXIT_OK


# ================================================================= experts ===

def cmd_experts(args: argparse.Namespace) -> int:
    m = _load_model(args.model)
    if not m.is_moe:
        error(f"{Path(m.path).name} non e' un modello MoE (nessun tensore expert).")
        return EXIT_USAGE
    hw = hwmod.detect(extra_disk_paths=[str(Path(m.path).parent)])

    if args.action == "report":
        budget = parse_size(args.cache_budget)
        if budget is None:
            req = planmod.PlanRequest(ram_limit=parse_size(args.ram_limit), backend="cpu")
            p = planmod.plan(m, hw, req)
            budget = p.expert_cache_budget or m.expert_bytes
            kvb = p.estimate.kv_cache
            rtb = p.estimate.runtime + p.estimate.activations
        else:
            kvb = planmod.kv_cache_bytes(m, args.ctx or 4096, "f16", "f16")
            rtb = planmod.RUNTIME_FIXED
        res = ec.residency_plan(m, budget)
        print(header("BUDGET MEMORIA MoE"))
        for label, value in ec.budget_table(m, budget, kvb, rtb, budget):
            print(kv(label, fmt_bytes(value)))
        print(header("RESIDENZA"))
        print(kv("expert totali", f"{res['experts']} ({res['experts_used_per_token']} attivi/token)"))
        print(kv("layer MoE", str(res["moe_layers"])))
        print(kv("unita' di trasferimento", fmt_bytes(res["bytes_per_expert_per_layer"])))
        print(kv("slot in cache", f"{res['cache_slots']} ({res['slots_per_layer']}/layer)"))
        print(kv("copertura hot", f"{res['coverage']*100:.1f}%"))
        print(kv("cold su SSD", fmt_bytes(res["cold_bytes"])))
        print()
        if args.json:
            print(json.dumps(res, indent=2))
        return EXIT_OK

    # ---- simulate ----
    budget = parse_size(args.cache_budget)
    if budget is None:
        req = planmod.PlanRequest(ram_limit=parse_size(args.ram_limit), backend="cpu")
        p = planmod.plan(m, hw, req)
        budget = p.expert_cache_budget or int(4 * GB)
        info(f"budget expert cache dal planner: {fmt_bytes(budget)}")

    info(f"simulazione: {args.tokens} token, policy={args.policy}, "
         f"skew={args.skew}, prefetch={'on' if not args.no_prefetch else 'off'}, "
         f"IO {'reale' if not args.no_io else 'modellato'}")
    res = ec.simulate(
        m, cache_budget=budget, tokens=args.tokens, topk=args.topk,
        skew=args.skew, policy=args.policy, prefetch=not args.no_prefetch,
        real_io=not args.no_io, compute_tok_s=args.compute_tok_s,
    )
    c = res.cache
    print(header("EXPERT CACHE"))
    print(kv("policy / budget", f"{c.get('policy')} / {fmt_bytes(c.get('budget'))}"))
    print(kv("slice residenti", f"{c.get('entries')} ({fmt_bytes(c.get('resident'))})"))
    print(kv("lookup", f"{c.get('lookups'):,}".replace(",", ".")))
    hr = c.get("hit_rate", 0.0)
    col = green if hr >= 0.8 else yellow if hr >= 0.5 else red
    print(kv("hit rate", col(f"{hr*100:.1f}%")))
    print(kv("hit / miss", f"{c.get('hits')} / {c.get('misses')}"))
    print(kv("evictions", str(c.get("evictions"))))
    if not args.no_prefetch:
        print(kv("prefetch emessi/utili",
                 f"{c.get('prefetch_issued')} / {c.get('prefetch_useful')} "
                 f"({c.get('prefetch_accuracy', 0)*100:.0f}%)"))
    print(header("SSD STREAMING"))
    print(kv("byte letti", fmt_bytes(res.disk_bytes)))
    print(kv("tempo IO", f"{res.io_time_s:.2f} s"))
    print(kv("throughput", bold(f"{res.disk_mb_s:.0f} MB/s")))
    print(kv("byte per token", fmt_bytes(res.disk_bytes / max(1, res.tokens))))
    print(header("PROIEZIONE"))
    print(kv("token simulati", f"{res.tokens} x {res.layers} layer x top-{res.topk}"))
    if res.projected_tok_s:
        print(kv("tok/s proiettati", bold(f"{res.projected_tok_s:.2f}")))
    print(kv("limite", "IO (SSD)" if res.io_bound else "calcolo (CPU)"))
    for n in res.notes:
        print(f"  {yellow('!')} {n}")
    print()

    rec = telemetry.build_record(
        event="expert_simulation", hardware=hw, model=m,
        metrics=res.to_dict(),
        extra={"cache_budget": budget},
    )
    path = telemetry.write_run(rec)
    ok(f"risultati salvati in {path}")
    if args.json:
        print(json.dumps(res.to_dict(), indent=2, default=str))
    return EXIT_OK


# =================================================================== setup ===

def cmd_setup(args: argparse.Namespace) -> int:
    # Se e' impostata una cartella locale di llama.cpp, `setup` non scarica
    # niente: usa quella. Se quella cartella non c'e', il guasto non e' di rete,
    # e dirlo come se lo fosse manda l'utente a ritentare un download che non
    # verra' mai tentato. E' successo subito dopo la rinomina: la variabile era
    # rimasta a puntare la cartella col nome vecchio, e il messaggio consigliava
    # "indica una copia locale con ARGINE_LLAMA_DIR" — cioe' esattamente la cosa
    # che era impostata e rotta.
    valore, var, valido = rtmod.llama_dir_configurato()
    if valore and not valido:
        nuovo_nome = var.replace("PICOSTAR_", "ARGINE_")
        error(f"{var}={valore}")
        error("  questa cartella non contiene llama-server. Finche' la variabile")
        error("  e' impostata, setup non scarica niente: usa solo quella.")
        print()
        # Il caso quasi sempre vero quando questo scatta: la cartella del
        # progetto e' stata spostata o rinominata, e la build e' ancora dentro,
        # con lo stesso nome, accanto al pacchetto installato in editable.
        # Cercarla li' trasforma due passaggi di diagnosi in un copia-incolla.
        candidata = rtmod.build_accanto_al_progetto(valore)
        if candidata:
            info(f"Ho trovato una build con lo stesso nome ('{candidata.name}') "
                 f"dentro il progetto: e' quasi certamente quella che cercavi.")
            print(f"    setx {nuovo_nome} \"{candidata}\"")
        else:
            info("Per rimettere a posto, una delle due:")
            print(f"    setx {nuovo_nome} \"C:\\percorso\\della\\build\"   "
                  f"per usare la tua build patchata")
            print(f"    setx {var} \"\"   "
                  f"per far scaricare i binari ufficiali")
        print()
        info("Poi apri un terminale nuovo: setx non cambia quello corrente.")
        if var.startswith("PICOSTAR_"):
            warn(f"{var} e' il nome vecchio. Quello nuovo vince quando ci sono "
                 f"entrambi, quindi non e' urgente, ma puoi liberartene con "
                 f"'setx {var} \"\"'.")
        return EXIT_RUNTIME

    hw = hwmod.detect()
    targets: List[str] = []
    if args.backend in ("both", "all"):
        targets = ["cpu"] + (["vulkan"] if hw.vulkan_runtime else [])
        if "vulkan" not in targets:
            warn("runtime Vulkan non rilevato: scarico solo il backend CPU")
    else:
        targets = [args.backend]

    failures = 0
    for backend in targets:
        try:
            info(f"preparo il runtime {backend}...")
            rt = rtmod.ensure_runtime(backend, build=args.llama_build)
            ok(f"{backend}: {rt.binary('server')}  [{rt.version()}]")
            devs = rt.list_devices()
            if devs:
                for d in devs:
                    print(f"      - {d}")
        except Exception as e:
            failures += 1
            error(f"{backend}: {e}")

    print()
    print(header("RUNTIME INSTALLATI"))
    for rt in rtmod.installed_runtimes():
        print(kv(f"{rt.build} / {rt.backend}", str(rt.root)))
    print(kv("cartella dati", str(runtimes_dir().parent)))
    print()
    return EXIT_OK if failures == 0 else EXIT_RUNTIME


# ==================================================================== logs ===

def cmd_logs(args: argparse.Namespace) -> int:
    rows = telemetry.read_runs(limit=args.last)
    if not rows:
        info(f"nessuna esecuzione registrata in {telemetry.log_path()}")
        return EXIT_OK
    if args.json:
        for r in rows:
            print(json.dumps(r, ensure_ascii=False, default=str))
        return EXIT_OK
    print(header(f"ULTIME {len(rows)} ESECUZIONI  ({telemetry.log_path()})"))
    hdr = f"  {'quando':<17}{'evento':<19}{'modello':<26}{'ctx':>6}{'KV':>6}{'bk':>8}{'tok/s':>8}{'RAMpk':>10}"
    print(bold(hdr))
    for r in rows:
        model = (r.get("model") or {}).get("name") or ""
        plan = r.get("plan") or {}
        met = r.get("metrics") or {}
        tok = met.get("gen_tok_s_median")
        if tok is None:
            tok = (met.get("projected_tok_s") or "")
        res = met.get("resources") or {}
        rampk = res.get("ram_peak")
        print(f"  {str(r.get('ts_iso'))[5:16]:<17}"
              f"{str(r.get('event'))[:18]:<19}"
              f"{model[:25]:<26}"
              f"{str(plan.get('ctx') or ''):>6}"
              f"{str(plan.get('kv_type_k') or '').upper():>6}"
              f"{str(plan.get('backend') or ''):>8}"
              f"{(f'{tok:.2f}' if isinstance(tok,(int,float)) else ''):>8}"
              f"{(fmt_bytes(rampk, 1) if rampk else ''):>10}")
    print()
    return EXIT_OK


# ================================================================== doctor ===

def cmd_doctor(args: argparse.Namespace) -> int:
    problems = 0
    print(header("DIAGNOSTICA"))
    try:
        import psutil  # noqa: F401
        print(kv("psutil", green("ok")))
    except ImportError:
        problems += 1
        print(kv("psutil", red("mancante") + "  ->  pip install psutil"))
    hw = hwmod.detect()
    print(kv("RAM", fmt_bytes(hw.ram_total)))
    print(kv("backend", ", ".join(hw.backends)))
    # La variabile che decide *quale* binario parte. Era l'unica cosa importante
    # che doctor non guardava: diceva "non installato" mentre ARGINE_LLAMA_DIR
    # puntava a una build buona, e non diceva niente quando puntava nel vuoto.
    # Il README affermava che doctor sa dire se il binario contiene l'expert
    # store: era falso, e questo e' il punto in cui lo diventa vero.
    valore, var, valido = rtmod.llama_dir_configurato()
    if valore:
        vecchio = "  " + yellow("(nome vecchio)") if var != "ARGINE_LLAMA_DIR" else ""
        if valido:
            print(kv(var, valore + vecchio))
        else:
            problems += 1
            print(kv(var, red("cartella senza llama-server") + vecchio))
            print(kv("", dim(valore)))
            candidata = rtmod.build_accanto_al_progetto(valore)
            if candidata:
                print(kv("", f'setx ARGINE_LLAMA_DIR "{candidata}"'))
    else:
        print(kv("ARGINE_LLAMA_DIR", dim("non impostata: si usa la build scaricata")))

    rts = rtmod.installed_runtimes()
    for rt in rts:
        print(kv(f"runtime scaricato {rt.backend}", f"{rt.build} -> {rt.root}"))
    if not rts and not (valore and valido):
        problems += 1
        print(kv("runtime llama.cpp", yellow("nessuno utilizzabile")
                 + "  ->  argine setup, oppure imposta ARGINE_LLAMA_DIR"))

    # Lo store non compare fra le opzioni di --help, perche' si configura per
    # variabili d'ambiente: un binario ufficiale le ignora in silenzio. Dirlo qui
    # e' l'unico modo per accorgersene prima di una misura, invece che dopo.
    in_uso = rtmod.runtime_in_uso()
    if in_uso is not None:
        store = in_uso.supports_expert_store()
        print(kv("binario in uso", f"{in_uso.build} -> {in_uso.binary('server')}"))
        print(kv("expert store nel binario",
                 green("presente") if store else
                 yellow("assente") + "  ->  --expert-store verrebbe ignorato"))
    orfani = rtmod.find_orphans()
    if orfani:
        problems += 1
        tot = sum(o["rss"] or 0 for o in orfani)
        print(kv("llama-server orfani",
                 red(f"{len(orfani)} attivi, {fmt_bytes(tot)} occupati")
                 + "  ->  argine run ... --kill-orphans"))
    else:
        print(kv("llama-server orfani", green("nessuno")))
    d = shutil.disk_usage(str(runtimes_dir()))
    print(kv("spazio dati", f"{fmt_bytes(d.free)} liberi"))
    print(kv("cartella dati", str(runtimes_dir().parent)))
    print(kv("log", str(telemetry.log_path())))
    print()
    return EXIT_OK if problems == 0 else EXIT_RUNTIME


# =================================================================== parser ==

def _add_model_arg(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("model", help="file .gguf (o cartella che ne contiene uno)")


def _add_runtime_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--backend", choices=["auto", "cpu", "vulkan"], default="auto",
                    help="backend di inferenza (default: auto)")
    sp.add_argument("--ram-limit", default=None, metavar="SIZE",
                    help="budget RAM massimo, es. 11G (default: automatico)")
    sp.add_argument("--ctx", type=int, default=None, help="context size in token")
    sp.add_argument("--batch", type=int, default=None, help="logical batch size")
    sp.add_argument("--ubatch", type=int, default=None, help="physical batch size")
    sp.add_argument("--threads", type=int, default=None, help="thread di generazione")
    sp.add_argument("--kv-type", default=None, metavar="TYPE",
                    help="tipo KV cache: f16, q8_0, q4_0")
    sp.add_argument("--gpu-layers", type=int, default=None,
                    help="layer offloadati su GPU (solo backend vulkan)")
    sp.add_argument("--expert-cache", default=None, metavar="SIZE",
                    help="budget della expert cache per modelli MoE, es. 5G")
    sp.add_argument("--no-streaming", action="store_true",
                    help="vieta l'expert streaming (richiede tutti gli expert in RAM)")
    sp.add_argument("--no-mmap", action="store_true", help="disabilita il memory mapping")
    sp.add_argument("--mlock", action="store_true", help="blocca i pesi in RAM (sconsigliato su 16 GB)")
    sp.add_argument("--no-jinja", action="store_true",
                    help="non usare il template Jinja del modello (utile se la chat API da' errore)")
    sp.add_argument("--spec", default=None, metavar="MODO",
                    help="decodifica speculativa: 'ngram' (nessun modello draft, "
                         "costo di RAM trascurabile) oppure 'draft' con --draft-model")
    sp.add_argument("--draft-model", default=None, metavar="GGUF",
                    help="modello draft piccolo per --spec draft (es. Qwen3-0.6B Q4)")
    sp.add_argument("--draft-n-max", type=int, default=None,
                    help="token proposti a ogni passo dalla speculativa (default 8)")
    sp.add_argument("--draft-ctx", type=int, default=None,
                    help="context del modello draft")
    sp.add_argument("--no-repack", action="store_true",
                    help="disattiva la copia SIMD dei pesi: libera molta RAM "
                         "(su un Q4 circa due terzi del modello) al costo di "
                         "un po' di velocita' su CPU")
    sp.add_argument("--repack", action="store_true",
                    help="forza la copia SIMD anche se il planner la toglierebbe")
    sp.add_argument("--parallel", type=int, default=1,
                    help="slot concorrenti di llama-server (default 1: uso locale monoutente)")
    sp.add_argument("--kill-orphans", action="store_true",
                    help="termina i llama-server rimasti attivi da esecuzioni "
                         "precedenti, che occupano RAM e falsano il piano")
    sp.add_argument("--prompt-cache", default=None, metavar="DIR",
                    help="salva la KV cache degli slot su disco, per riprendere "
                         "una conversazione senza riprocessare il prompt dopo un riavvio")
    sp.add_argument("--cache-reuse", type=int, default=0, metavar="N",
                    help="riusa porzioni di KV cache anche quando il prefisso cambia "
                         "a meta' (dimensione minima del blocco, 0 = disattivo)")
    sp.add_argument("--prompt-cache-states", type=int, default=None, metavar="N",
                    help="quanti prompt gia' elaborati tenere in RAM per non "
                         "rifare il prefill (default 2, uno costa quanto la KV "
                         "cache del contesto; 0 li disattiva). Senza questo "
                         "llama.cpp ne userebbe fino a 8 GiB, fuori dal budget")
    sp.add_argument("--prefetch", type=int, default=None, metavar="N",
                    help="profondita' di prefetch degli expert: chiede al sistema "
                         "gli expert dei prossimi N layer (richiede la build patchata; "
                         "0 = disattivo)")
    sp.add_argument("--prefetch-threads", type=int, default=None, metavar="M",
                    help="thread dedicati al prefetch (default 2): il throughput "
                         "dell'SSD dipende da quanti fault sono in volo insieme")
    sp.add_argument("--route-log", default=None, metavar="FILE",
                    help="registra le scelte del router MoE su file (richiede una "
                         "build di llama.cpp patchata, vedi patches/README.md)")
    sp.add_argument("--llama-verbose", action="store_true",
                    help="log verboso di llama.cpp: stampa le dimensioni reali dei buffer "
                         "e permette a Argine di tarare la stima di memoria")
    sp.add_argument("--auto", action="store_true",
                    help="profilo automatico: ignora ctx/kv/gpu-layers e li calcola")
    sp.add_argument("--llama-build", default=None, metavar="bNNNNN",
                    help="forza una build specifica di llama.cpp")
    sp.add_argument("--llama-arg", action="append", default=None, metavar="ARG",
                    help="argomento aggiuntivo passato a llama-server (ripetibile)")
    sp.add_argument("--retries", type=int, default=2,
                    help="tentativi con configurazione ridotta dopo un fallimento")
    sp.add_argument("--load-timeout", type=float, default=900.0,
                    help="secondi massimi di attesa per il caricamento del modello")
    sp.add_argument("--offline", action="store_true",
                    help="non scaricare nulla: usa solo runtime gia' presenti")
    sp.add_argument("--dry-run", action="store_true", help="mostra il piano e il comando, senza avviare")
    sp.add_argument("-v", "--verbose", action="store_true", help="output dettagliato di llama-server")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="argine",
        description="Argine - runtime LLM locale ottimizzato per PC con 16 GB di RAM.",
        epilog="Esempi:\n"
               "  argine probe\n"
               "  argine inspect Qwen3-8B-Q4_K_M.gguf\n"
               "  argine run Qwen3-8B-Q4_K_M.gguf --ram-limit 11G --backend auto\n"
               "  argine benchmark Qwen3-8B-Q4_K_M.gguf --n-predict 128\n"
               "  argine experts simulate Qwen3-30B-A3B-Q4_K_M.gguf --cache-budget 5G\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--version", action="version", version=f"Argine {__version__}")
    sub = ap.add_subparsers(dest="command", metavar="comando")

    p = sub.add_parser("probe", help="mostra CPU, RAM, GPU, backend e disco")
    p.add_argument("--json", action="store_true")
    p.add_argument("--deep", action="store_true",
                   help="interroga i binari llama.cpp per enumerare i device reali")
    p.add_argument("--offline", action="store_true")
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("inspect", help="analizza un GGUF e stima la RAM necessaria")
    _add_model_arg(p)
    p.add_argument("--backend", choices=["auto", "cpu", "vulkan"], default="auto")
    p.add_argument("--ram-limit", default=None, metavar="SIZE")
    p.add_argument("--ctx", type=int, default=None)
    p.add_argument("--kv-type", default=None)
    p.add_argument("--gpu-layers", type=int, default=None)
    p.add_argument("--expert-cache", default=None, metavar="SIZE")
    p.add_argument("--spec", default=None, metavar="MODO")
    p.add_argument("--draft-model", default=None, metavar="GGUF")
    p.add_argument("--draft-n-max", type=int, default=None)
    p.add_argument("--draft-ctx", type=int, default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("plan", help="alias di inspect: mostra solo il piano di memoria")
    _add_model_arg(p)
    p.add_argument("--backend", choices=["auto", "cpu", "vulkan"], default="auto")
    p.add_argument("--ram-limit", default=None, metavar="SIZE")
    p.add_argument("--ctx", type=int, default=None)
    p.add_argument("--kv-type", default=None)
    p.add_argument("--gpu-layers", type=int, default=None)
    p.add_argument("--expert-cache", default=None, metavar="SIZE")
    p.add_argument("--spec", default=None, metavar="MODO")
    p.add_argument("--draft-model", default=None, metavar="GGUF")
    p.add_argument("--draft-n-max", type=int, default=None)
    p.add_argument("--draft-ctx", type=int, default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("run", help="avvia il modello e l'API locale OpenAI-compatibile")
    _add_model_arg(p)
    _add_runtime_args(p)
    p.add_argument("--port", type=int, default=8080, help="porta dell'API Argine (default 8080)")
    p.add_argument("--host", default="127.0.0.1", help="indirizzo di ascolto (default 127.0.0.1)")
    _add_expert_store_args(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("benchmark", help="misura tok/s, TTFT, RAM, disco, CPU/GPU")
    _add_model_arg(p)
    _add_runtime_args(p)
    p.add_argument("--n-predict", type=int, default=128, help="token da generare per run")
    p.add_argument("--repeat", type=int, default=3, help="numero di ripetizioni")
    p.add_argument("--prompt", default=None, help="prompt da usare")
    p.add_argument("--prompt-file", default=None, help="file con il prompt")
    p.add_argument("--prompt-tokens", type=int, default=None,
                   help="allunga il prompt fino a circa N token")
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--cooldown", type=int, default=20, metavar="SEC",
                   help="pausa fra le configurazioni dello sweep, per lasciare "
                        "raffreddare la CPU (default 20s, 0 per disattivarla)")
    p.add_argument("--sweep-prompt", default=None, metavar="N,N,N",
                   help="misura la velocita' a diverse lunghezze di contesto "
                        "(es. 128,512,2048): il KPI dipende da quanto e' lunga "
                        "la conversazione")
    p.add_argument("--conversazione", type=int, default=None, metavar="TURNI",
                   help="simula una conversazione di N turni, con il prefisso "
                        "che cresce a ogni turno, e misura il TTFT turno per "
                        "turno: e' il caso d'uso vero, diverso da prompt "
                        "indipendenti di lunghezza crescente")
    p.add_argument("--no-baseline", action="store_true",
                   help="salta la ripetizione senza riuso del prefisso")
    p.add_argument("--sweep-expert-budget", default=None, metavar="SIZE,SIZE,SIZE",
                   help="misura la velocita' al variare del tetto di RAM per gli "
                        "expert, es. 8G,6G,4G,3G. Predice come si comporterebbe un "
                        "modello con molti piu' expert senza scaricarlo: cio' che "
                        "conta e' la frazione residente, non i GiB")
    p.add_argument("--sweep-read-mode", default=None, metavar="MODO,MODO",
                   help="dentro lo sweep sul budget, ripete ogni punto con piu' "
                        "strati di lettura (es. sincrono,blocco,asincrono) e li "
                        "confronta a parita' di binario, di file e di ora")
    p.add_argument("--sweep-threads", default=None, metavar="N,N,N",
                   help="ripete il benchmark con piu' valori di --threads e li confronta "
                        "(consigliato sulle CPU ibride Intel, es. 4,6,8,12)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--port", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--host", default="127.0.0.1", help=argparse.SUPPRESS)
    _add_expert_store_args(p)
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("experts", help="analisi e simulazione della expert cache (MoE)")
    p.add_argument("action", choices=["report", "simulate"])
    _add_model_arg(p)
    p.add_argument("--cache-budget", default=None, metavar="SIZE",
                   help="budget della expert cache, es. 5G")
    p.add_argument("--ram-limit", default=None, metavar="SIZE")
    p.add_argument("--ctx", type=int, default=None)
    p.add_argument("--tokens", type=int, default=256, help="token da simulare")
    p.add_argument("--topk", type=int, default=None, help="expert attivi per token")
    p.add_argument("--skew", type=float, default=0.9,
                   help="sbilanciamento del routing (0 = uniforme, 1+ = expert caldi)")
    p.add_argument("--policy", choices=["lru", "2q", "lfu"], default="lru")
    p.add_argument("--no-prefetch", action="store_true")
    p.add_argument("--no-io", action="store_true",
                   help="non leggere da disco: usa un modello di throughput")
    p.add_argument("--compute-tok-s", type=float, default=None,
                   help="tok/s misurati a expert residenti (per la proiezione)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_experts)

    p = sub.add_parser("reorder",
                       help="riscrive un GGUF MoE con gli expert co-attivati contigui")
    _add_model_arg(p)
    p.add_argument("--trace", required=True, metavar="FILE",
                   help="traccia del routing (vedi patches/README.md)")
    p.add_argument("-o", "--output", default=None, metavar="GGUF",
                   help="file di destinazione (default: <nome>-reordered.gguf)")
    p.add_argument("--dry-run", action="store_true",
                   help="calcola il guadagno senza scrivere nulla")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_reorder)

    p = sub.add_parser("ppl",
                       help="misura la perplessita': il prezzo in qualita' di una "
                            "quantizzazione")
    _add_model_arg(p)
    p.add_argument("--text", default=None, metavar="FILE",
                   help="testo di riferimento (consigliato wikitext-2-raw); "
                        "assente, ne viene generato uno di ripiego")
    p.add_argument("--ctx", type=int, default=512,
                   help="finestra di contesto (default 512, come da prassi)")
    p.add_argument("--chunks", type=int, default=None,
                   help="quanti blocchi valutare: meno blocchi, misura piu' rapida "
                        "e piu' rumorosa")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--riferimento", default=None, metavar="NOME",
                   help="modello contro cui confrontarsi (default: il migliore "
                        "fra quelli misurati con la stessa firma)")
    p.add_argument("--ram-limit", default=None, metavar="SIZE")
    p.add_argument("--timeout", type=float, default=7200.0)
    p.add_argument("--llama-build", default=None, metavar="bNNNNN")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--json", action="store_true")
    _add_expert_store_args(p)
    p.set_defaults(func=cmd_ppl)

    p = sub.add_parser("quantize",
                       help="riscrive gli expert con meno bit per farli entrare "
                            "in RAM (precisione mista)")
    _add_model_arg(p)
    p.add_argument("--experts", default="IQ2_XXS", metavar="TIPO",
                   help="tipo per i tensori degli expert (default IQ2_XXS, "
                        "2,06 bit). I tipi IQ2 richiedono --imatrix; Q2_K no")
    p.add_argument("--rest", default=None, metavar="TIPO",
                   help="tipo per tutto il resto; assente, resta com'e'. "
                        "Toccarlo aggiunge errore e non restituisce GiB")
    p.add_argument("--embd", default=None, metavar="TIPO",
                   help="tipo per token_embd/output (spesso conviene tenerli alti)")
    p.add_argument("--imatrix", default=None, metavar="FILE",
                   help="matrice di importanza prodotta da 'argine imatrix'")
    p.add_argument("--out", default=None, metavar="FILE")
    p.add_argument("--ram-limit", default=None, metavar="SIZE",
                   help="budget RAM contro cui verificare la dimensione stimata")
    p.add_argument("--allow-requantize", action="store_true",
                   help="consenti di partire da un modello gia' quantizzato "
                        "(arrotonda due volte: peggio che partire da BF16)")
    p.add_argument("--no-tensor-type", action="store_true",
                   help="non usare --tensor-type: quantizza tutto uniformemente")
    p.add_argument("--dry-run", action="store_true",
                   help="calcola solo quanto uscira' grande e se ci sta")
    p.add_argument("--force", action="store_true")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--timeout", type=float, default=21600.0)
    p.add_argument("--llama-build", default=None, metavar="bNNNNN")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_quantize)

    p = sub.add_parser("imatrix",
                       help="calcola la matrice di importanza, prerequisito dei "
                            "tipi IQ2")
    _add_model_arg(p)
    p.add_argument("--text", default=None, metavar="FILE",
                   help="testo di calibrazione: piu' e' vario, meglio la matrice "
                        "protegge i pesi che contano")
    p.add_argument("--out", default=None, metavar="FILE")
    p.add_argument("--ctx", type=int, default=512)
    p.add_argument("--chunks", type=int, default=None,
                   help="quanti blocchi valutare; per un MoE con molti expert "
                        "servono centinaia perche' anche i rari vengano visti")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--ram-limit", default=None, metavar="SIZE")
    p.add_argument("--timeout", type=float, default=43200.0)
    p.add_argument("--llama-build", default=None, metavar="bNNNNN")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--json", action="store_true")
    _add_expert_store_args(p)
    p.set_defaults(func=cmd_imatrix)

    p = sub.add_parser("store",
                       help="expert store: pesi degli expert in un file fatto per "
                            "essere letto (una lettura allineata per expert)")
    p.add_argument("action", choices=["build", "info", "verify", "bench"])
    _add_model_arg(p)
    p.add_argument("--store", default=None, metavar="FILE",
                   help="file dello store (default: <modello>.psxs)")
    p.add_argument("--trace", default=None, metavar="FILE",
                   help="traccia del routing: ordina fisicamente gli expert "
                        "co-attivati. Qui e' solo layout, i pesi non cambiano")
    p.add_argument("--samples", type=int, default=24,
                   help="gruppi da confrontare col GGUF nella verifica")
    p.add_argument("--no-verify", action="store_true",
                   help="salta la verifica dopo la costruzione (sconsigliato)")
    p.add_argument("--threads", type=int, default=8,
                   help="thread per 'store bench' (default 8)")
    p.add_argument("--seconds", type=float, default=4.0,
                   help="durata di ogni strada in 'store bench'")
    p.add_argument("--purge", default="0", metavar="BYTE",
                   help="svuota la page cache prima di ogni strada occupando "
                        "questa memoria (es. 10G): rende confrontabili i percorsi "
                        "bufferizzati con quelli a cache scavalcata")
    p.add_argument("--dry-run", action="store_true",
                   help="mostra il piano senza scrivere")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_store)

    p = sub.add_parser("diskbench",
                       help="misura il tetto dello storage sul pattern degli expert")
    p.add_argument("file", nargs="?", default=None, metavar="FILE",
                   help="file grande su cui misurare (di norma il GGUF del modello)")
    p.add_argument("--slice", action="append", default=None, metavar="BYTE",
                   help="dimensione della lettura (ripetibile; default 896K e 2816K, "
                        "cioe' un tensore expert e la tripletta gate/up/down)")
    p.add_argument("--threads", nargs="+", default=None, metavar="N",
                   help="livelli di parallelismo da provare (default 1 4 8 16 32)")
    p.add_argument("--coda", type=float, default=None, metavar="N",
                   help="profondita' di coda a cui lavori davvero, dalla riga "
                        "'profondita' di coda media' che lo store stampa con "
                        "--expert-stats. Senza questo numero il verdetto su come "
                        "leggere non viene dato, perche' dipende dal regime: misurata "
                        "sulla macchina di riferimento vale 2,09 a 4 GiB di budget e "
                        "2,49 a 1,5 GiB")
    p.add_argument("--kinds", nargs="+", default=list(dbmod.DEFAULT_KINDS),
                   choices=list(dbmod.ALL_KINDS),
                   help="percorsi di I/O da confrontare: mmap (quello di llama.cpp), "
                        "pread (letture posizionate con cache), direct (senza cache), "
                        "moe (direct + barriera per layer, il pattern reale)")
    p.add_argument("--baseline", type=float, default=dbmod.Suite.PRODUZIONE_MB_S,
                   metavar="MB_S",
                   help="throughput di disco osservato in generazione su questa "
                        f"macchina, per il confronto (default {dbmod.Suite.PRODUZIONE_MB_S:.0f})")
    p.add_argument("--seconds", type=float, default=4.0,
                   help="durata di ogni singola prova (default 4)")
    p.add_argument("--purge", default="0", metavar="BYTE",
                   help="svuota la page cache occupando questa memoria prima di "
                        "ogni prova non diretta (es. 8G): rende onesto il "
                        "confronto con mmap su file piu' piccoli della RAM")
    p.add_argument("--no-sequential", action="store_true",
                   help="salta la lettura sequenziale di riferimento")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_diskbench)

    p = sub.add_parser("setup", help="scarica llama.cpp precompilato (una volta sola)")
    p.add_argument("--backend", choices=["cpu", "vulkan", "both"], default="both")
    p.add_argument("--llama-build", default=None, metavar="bNNNNN")
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("logs", help="mostra lo storico JSONL delle esecuzioni")
    p.add_argument("--last", type=int, default=20)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_logs)

    p = sub.add_parser("doctor", help="verifica l'installazione")
    p.set_defaults(func=cmd_doctor)

    # Argine2: the research command family (argine/research/cli.py)
    from .research.cli import register as _register_research
    _register_research(sub, sys.modules[__name__])
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if not getattr(args, "command", None):
        ap.print_help()
        return EXIT_USAGE
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print()
        info("interrotto")
        return 130
    except GGUFError as e:
        error(str(e))
        return EXIT_USAGE
    except ValueError as e:
        error(str(e))
        return EXIT_USAGE
    except rtmod.RuntimeError_ as e:
        error(str(e))
        return EXIT_RUNTIME


if __name__ == "__main__":
    sys.exit(main())
