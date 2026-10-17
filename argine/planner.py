"""Memory Planner.

Calcola un budget RAM sicuro e una configurazione runtime che ci stia dentro,
applicando in ordine la "scala di sicurezza" richiesta dalla specifica:

    riduzione context -> riduzione batch -> KV Q8 -> KV Q4
    -> riduzione GPU offload -> lazy loading / expert streaming -> errore

Contabilizza: model weights, KV cache, activations, runtime buffers,
GPU shared memory (le iGPU prendono RAM di sistema) e riserva OS.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .gguf import REPACK_TYPES as _REPACK_TYPES
from .gguf import GGUFModel, KV_TYPE_BYTES
from .hardware import HardwareReport
from .util import GB, MB, fmt_bytes

# Scale di riduzione (ordine di preferenza: dal piu' generoso al piu' stretto)
CTX_LADDER = [32768, 16384, 12288, 8192, 6144, 4096, 3072, 2048, 1536, 1024, 768, 512]
BATCH_LADDER = [2048, 1024, 512, 256, 128]
UBATCH_LADDER = [512, 256, 128, 64]
KV_LADDER = ["f16", "q8_0", "q4_0"]

DEFAULT_CTX_16GB = 4096
MIN_CTX = 512

# Overhead fisso del processo llama-server + vocabolario + strutture interne
RUNTIME_FIXED = 384 * MB
# Buffer di scratch per la pipeline di I/O (mmap readahead, prefetch expert)
IO_SCRATCH = 96 * MB


# Quanti stati di prompt tenere in cache, come default.
#
# Uno stato costa quanto l'intera KV cache del contesto: su Qwen3-30B-A3B a
# ctx 4096 sono ~400 MiB. Il default di llama.cpp, 8192 MiB, ne terrebbe venti —
# su una macchina da 16 GB e' una scelta fatta per macchine diverse da questa.
#
# Due e' il minimo che serve davvero: uno per la conversazione in corso, uno per
# quella precedente, cosi' alternando due chat non si ripaga il prefill ogni
# volta. Oltre, il guadagno cala in fretta e la RAM e' l'unica cosa che qui
# scarseggia.
STATI_PROMPT_IN_CACHE = 2


@dataclass
class MemoryEstimate:
    weights_resident: int = 0
    kv_cache: int = 0
    activations: int = 0
    runtime: int = 0
    expert_cache: int = 0
    repack: int = 0           # copia SIMD dei pesi tenuti su CPU (llama.cpp)
    draft: int = 0            # modello draft per la decodifica speculativa
    # Cache dei prompt gia' elaborati che llama-server tiene in RAM per non
    # rifare il prefill quando una conversazione riparte dallo stesso prefisso.
    # Ha un suo limite, e **il default di llama.cpp e' 8192 MiB**: su una
    # macchina da 16 GB e' piu' di meta' del budget, allocata fuori da ogni
    # nostra contabilita'. Finora non e' esplosa solo perche' i benchmark
    # ripetono un prompt corto; in una conversazione vera cresce.
    prompt_cache: int = 0
    # Quota che il driver GPU alloca dalla RAM di sistema (iGPU). NON va sommata
    # al totale: su una iGPU i pesi vivono nel buffer Vulkan *invece che* nel
    # working set del processo, non in aggiunta. E' un vincolo separato, da
    # confrontare col tetto di shared memory del driver.
    gpu_shared: int = 0
    # Su GPU discreta i pesi escono dalla RAM ed entrano nella VRAM.
    gpu_dedicated: int = 0

    @property
    def total(self) -> int:
        return (self.weights_resident + self.kv_cache + self.activations
                + self.runtime + self.expert_cache + self.repack + self.draft
                + self.prompt_cache)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["total"] = self.total
        return d

    def rows(self) -> List[tuple[str, int]]:
        rows = [
            ("model weights (residenti)", self.weights_resident),
            ("repack SIMD (copia CPU)", self.repack),
            ("modello draft (speculativa)", self.draft),
            ("expert cache", self.expert_cache),
            ("KV cache", self.kv_cache),
            ("cache dei prompt", self.prompt_cache),
            ("activations / compute", self.activations),
            ("runtime buffers", self.runtime),
        ]
        return [(k, v) for k, v in rows if v > 0]


@dataclass
class RuntimePlan:
    # parametri runtime
    ctx: int = DEFAULT_CTX_16GB
    batch: int = 2048
    ubatch: int = 512
    threads: int = 0
    threads_batch: int = 0
    kv_type_k: str = "f16"
    kv_type_v: str = "f16"
    gpu_layers: int = 0
    flash_attn: bool = True
    mmap: bool = True
    mlock: bool = False
    backend: str = "cpu"
    jinja: bool = True
    parallel: int = 1        # slot concorrenti di llama-server
    llama_verbose: bool = False
    repack: bool = True      # copia SIMD dei pesi: piu' veloce ma costa RAM

    # Decodifica speculativa: un draft propone piu' token, il modello grande li
    # verifica in un solo passaggio, ammortizzando la lettura dei pesi shared.
    spec_type: Optional[str] = None      # es. draft-simple, ngram-cache
    draft_model: Optional[str] = None    # percorso del GGUF draft
    draft_n_max: int = 0
    draft_n_min: int = 0
    draft_p_min: Optional[float] = None
    draft_ctx: int = 0
    draft_bytes: int = 0

    # Riuso della KV cache. llama-server riusa gia' il prefisso condiviso fra
    # richieste; queste due opzioni aggiungono la persistenza su disco fra
    # riavvii e il riuso quando il prefisso cambia a meta'.
    prompt_cache_dir: Optional[str] = None
    cache_reuse: int = 0
    # Quanti stati di prompt llama-server puo' tenere in RAM. Finisce in
    # --cache-ram come un numero di MiB: senza, vale il default di llama.cpp,
    # 8192 MiB, che su questa macchina e' meta' del budget fuori contabilita'.
    prompt_cache_states: int = STATI_PROMPT_IN_CACHE

    # MoE / streaming
    moe_streaming: bool = False          # experts non tutti residenti
    cpu_moe: bool = False                # experts su CPU, resto su GPU
    n_cpu_moe: int = 0                   # numero di layer MoE tenuti su CPU
    expert_cache_budget: int = 0         # byte riservati alla expert cache
    expert_store: bool = False           # expert letti dallo store, budget nostro
    # Quanto il piano lascerebbe agli expert da se', prima che --expert-budget lo
    # sovrascriva. Serve perche' con un budget forzato `expert_cache_budget`
    # diventa il numero CHIESTO e quello sostenibile va perso: il messaggio di
    # rifiuto lo ricalcolava per conto suo e otteneva 0 B dove l'avviso diceva
    # 512 MiB, cioe' due numeri in contrasto nella stessa schermata.
    expert_cache_sostenibile: int = 0

    # contabilita'
    budget: int = 0
    estimate: MemoryEstimate = field(default_factory=MemoryEstimate)
    decisions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    feasible: bool = True
    failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["estimate"] = self.estimate.to_dict()
        return d


# ------------------------------------------------------------------ budget --

def safe_budget(hw: HardwareReport, requested: Optional[int] = None) -> tuple[int, List[str]]:
    """Budget RAM sicuro. Se l'utente passa --ram-limit lo rispettiamo,
    limitandolo a cio' che e' fisicamente sensato."""
    notes: List[str] = []
    total = hw.ram_total or 16 * GB

    # Riserva per OS + shell + browser: 25% della RAM, min 3.5 GiB, max 6 GiB
    reserve = int(min(max(total * 0.25, int(3.5 * GB)), 6 * GB))
    auto = total - reserve

    if requested is None:
        available = hw.ram_available or auto
        # Pianificare su RAM che non c'e' significa pianificare di paginare.
        # Se il sistema ne ha meno libera del budget teorico, e' quella libera
        # a comandare: teniamo un margine per non lasciare la macchina a secco.
        margine = 512 * MB
        if available and available - margine < auto:
            effettivo = max(int(1 * GB), available - margine)
            notes.append(
                f"budget limitato a {fmt_bytes(effettivo)} dalla RAM realmente "
                f"libera ({fmt_bytes(available)}), non dal massimo teorico "
                f"{fmt_bytes(auto)}: pianificare su memoria occupata da altri "
                "processi significa pianificare di paginare"
            )
            notes.append(
                "  chiudendo altre applicazioni il budget si alza da solo: "
                "usa --ram-limit per forzarlo"
            )
            return effettivo, notes
        notes.append(f"budget automatico: {fmt_bytes(auto)} (riserva sistema {fmt_bytes(reserve)})")
        return int(auto), notes

    budget = int(requested)
    hard_cap = int(total * 0.90)
    if budget > hard_cap:
        notes.append(
            f"--ram-limit {fmt_bytes(budget)} supera il 90% della RAM fisica: "
            f"ridotto a {fmt_bytes(hard_cap)} per non far swappare il sistema"
        )
        budget = hard_cap
    available = hw.ram_available or auto
    if budget > available:
        notes.append(
            f"attenzione: budget {fmt_bytes(budget)} > RAM attualmente libera "
            f"{fmt_bytes(available)}; chiudi applicazioni o riduci --ram-limit"
        )
    return budget, notes


# ---------------------------------------------------------------- estimate --

def kv_cache_bytes(model: GGUFModel, ctx: int, k_type: str, v_type: str) -> int:
    layers = model.n_layers or 32
    n_kv = model.n_head_kv or model.n_head or 8
    dk = model.head_dim_k or 128
    dv = model.head_dim_v or dk
    bk = KV_TYPE_BYTES.get(k_type.lower(), 2.0)
    bv = KV_TYPE_BYTES.get(v_type.lower(), 2.0)
    per_token = n_kv * (dk * bk + dv * bv)
    return int(layers * ctx * per_token)


def prompt_cache_bytes(model: GGUFModel, ctx: int, k_type: str, v_type: str,
                       stati: int = STATI_PROMPT_IN_CACHE) -> int:
    """RAM che llama-server puo' usare per i prompt gia' elaborati.

    Uno stato salvato e' la KV cache del contesto intero, quindi si calcola con
    la stessa funzione: e' lo stesso tensore, copiato altrove.
    """
    if stati <= 0:
        return 0
    return kv_cache_bytes(model, ctx, k_type, v_type) * int(stati)


def activation_bytes(model: GGUFModel, ctx: int, ubatch: int, flash_attn: bool) -> int:
    """Buffer di calcolo di ggml.

    Calibrato su misure reali: la voce che domina non sono le activation dei
    layer ma la proiezione finale sul vocabolario, perche' i modelli recenti
    hanno vocabolari enormi (Qwen3: 151.936 token). Con ubatch 512 quel singolo
    tensore intermedio vale da solo ~300 MiB.
    """
    n_embd = model.n_embd or 4096
    n_head = model.n_head or 32
    n_ff = model.n_ff or (n_embd * 3)
    vocab = _vocab_size(model)

    # Tarato sul buffer dichiarato da llama.cpp: 108.01 MiB per Qwen3-8B con
    # ubatch 512, n_embd 4096, n_ff 12288. Il modello qui sotto ne calcola 104.
    # Nota: NON c'e' un intermedio grande quanto vocab x ubatch, perche' ggml
    # seleziona i token di output prima della proiezione finale.
    graph = ubatch * 4 * (n_embd * 4 + n_ff * 3)
    # buffer dei logits restituiti (solo gli ultimi token)
    logits = vocab * 4 * 8
    # matrice di attenzione: senza flash attention scala con ctx
    attn = 0 if flash_attn else ubatch * ctx * n_head * 4
    # strutture del tokenizer/vocabolario tenute in RAM da llama.cpp
    vocab_tables = vocab * 96
    return int(graph + logits + attn + vocab_tables)


def _vocab_size(model: GGUFModel) -> int:
    toks = model.metadata.get("tokenizer.ggml.tokens")
    if toks is not None:
        try:
            return len(toks)
        except TypeError:
            pass
    v = model.arch_key("vocab_size")
    if v:
        try:
            return int(v)
        except (TypeError, ValueError):
            pass
    return 151936  # default Qwen


def estimate_memory(model: GGUFModel, plan: RuntimePlan, hw: HardwareReport) -> MemoryEstimate:
    est = MemoryEstimate()

    if plan.moe_streaming and model.is_moe:
        est.weights_resident = model.shared_bytes
        est.expert_cache = plan.expert_cache_budget
    else:
        est.weights_resident = model.tensor_bytes or model.total_size_on_disk

    # Copia SIMD dei pesi: solo per la parte che resta sulla CPU
    if plan.repack:
        cpu_frac = 1.0 - _offload_fraction(model, plan)
        repackable = model.repackable_bytes
        if plan.moe_streaming and model.is_moe:
            # in streaming i pesi degli expert non sono tutti residenti: llama.cpp
            # ripacchetterebbe comunque tutto, per questo lo streaming lo esclude
            repackable = sum(t.n_bytes for t in model.shared_tensors
                             if t.type_name in _REPACK_TYPES and len(t.dims) >= 2
                             and not t.name.startswith("token_embd"))
        est.repack = int(repackable * max(0.0, cpu_frac))

    est.draft = plan.draft_bytes
    est.kv_cache = kv_cache_bytes(model, plan.ctx, plan.kv_type_k, plan.kv_type_v)
    est.prompt_cache = prompt_cache_bytes(model, plan.ctx, plan.kv_type_k,
                                          plan.kv_type_v, plan.prompt_cache_states)
    est.activations = activation_bytes(model, plan.ctx, plan.ubatch, plan.flash_attn)
    est.runtime = RUNTIME_FIXED

    # Su GPU integrata l'offload non libera RAM: la shared memory *e'* RAM.
    if plan.backend == "vulkan" and plan.gpu_layers > 0:
        igpu = any(g.shares_system_ram for g in hw.gpus) or not hw.gpus
        if igpu:
            frac = _offload_fraction(model, plan)
            est.gpu_shared = int(est.weights_resident * frac)
        else:
            # GPU discreta: i layer offloadati escono dal budget RAM
            frac = _offload_fraction(model, plan)
            vram = hw.gpus[0].dedicated_vram or 0
            moved = min(int(est.weights_resident * frac), vram)
            est.weights_resident -= moved
            est.gpu_dedicated = moved
    return est


def _offload_fraction(model: GGUFModel, plan: RuntimePlan) -> float:
    layers = model.n_layers or 32
    if plan.gpu_layers <= 0:
        return 0.0
    n = min(plan.gpu_layers, layers + 1)
    frac = n / float(layers + 1)
    if plan.cpu_moe and model.is_moe and model.tensor_bytes:
        # gli expert restano su CPU: sulla GPU va solo la parte shared
        frac *= model.shared_bytes / float(model.tensor_bytes)
    return min(1.0, frac)


# -------------------------------------------------------------------- plan --

@dataclass
class PlanRequest:
    ctx: Optional[int] = None
    batch: Optional[int] = None
    ubatch: Optional[int] = None
    threads: Optional[int] = None
    kv_type: Optional[str] = None        # applica a K e V
    kv_type_k: Optional[str] = None
    kv_type_v: Optional[str] = None
    gpu_layers: Optional[int] = None
    ram_limit: Optional[int] = None
    backend: str = "cpu"
    flash_attn: Optional[bool] = None
    mmap: bool = True
    mlock: bool = False
    expert_cache: Optional[int] = None   # byte; None = automatico
    allow_streaming: bool = True
    auto: bool = False
    jinja: bool = True
    parallel: int = 1
    llama_verbose: bool = False
    repack: Optional[bool] = None   # None = decide il planner
    # Expert store: gli expert vengono letti da un file con letture esplicite e
    # tenuti in una cache con budget rigido, invece che dalla memoria mappata con
    # residenza governata dall'OS. Cambia due cose nel piano:
    #  - il repack NON puo' essere spento, perche' in llama.cpp --no-repack
    #    imposta no_extra_bufts e disabilita anche il buffer type dello store;
    #  - la memoria degli expert diventa un numero che scegliamo noi, non una
    #    stima su cio' che l'OS decidera' di tenere.
    expert_store: bool = False
    expert_store_budget: Optional[int] = None
    spec: Optional[str] = None           # 'ngram' | 'draft' | nome esatto | 'none'
    draft_model: Optional[str] = None
    draft_n_max: Optional[int] = None
    draft_ctx: Optional[int] = None
    prompt_cache_dir: Optional[str] = None
    cache_reuse: Optional[int] = None
    prompt_cache_states: Optional[int] = None


def plan(model: GGUFModel, hw: HardwareReport, req: PlanRequest) -> RuntimePlan:
    p = RuntimePlan()
    p.backend = req.backend
    p.mmap = req.mmap
    p.mlock = req.mlock
    p.jinja = req.jinja
    p.parallel = max(1, int(req.parallel or 1))
    p.llama_verbose = req.llama_verbose
    # Repack disattivato per default. Misurato su i5-1340P (AVX2, niente
    # AVX-512 ne' AMX) con Qwen3-8B-Q4_K_M: senza repack 8.24 tok/s sostenuti e
    # 5.04 GiB di picco, con repack 7.05 tok/s e 8.16 GiB. Non e' uno scambio
    # fra RAM e velocita': su questa classe di CPU e' peggio su entrambi.
    # Con --repack si puo' rimisurare su hardware che abbia i kernel adatti.
    p.expert_store = bool(req.expert_store)
    p.repack = False if req.repack is None else bool(req.repack)
    if p.expert_store and not p.repack:
        # vedi il commento sul gradino "repack": senza extra buffer type lo store
        # non riceve tensori, quindi qui non c'e' scelta da fare
        p.repack = True
    _setup_speculative(p, req)
    p.prompt_cache_dir = req.prompt_cache_dir
    p.cache_reuse = int(req.cache_reuse or 0)
    if req.prompt_cache_states is not None:
        p.prompt_cache_states = max(0, int(req.prompt_cache_states))
    if p.prompt_cache_states:
        p.decisions.append(
            f"cache dei prompt: {p.prompt_cache_states} stati "
            f"(il default di llama.cpp e' 8192 MiB, fuori dal nostro budget)")
    if p.repack and req.repack and not p.expert_store:
        p.warnings.append(
            f"--repack aggiunge {fmt_bytes(model.repackable_bytes)} di RAM: "
            "verifica con 'benchmark --repack' e '--no-repack' che sul tuo "
            "processore serva davvero"
        )
    elif p.repack and p.expert_store and model.is_moe:
        # con lo store il repack riguarda solo i tensori shared, perche' gli
        # expert appartengono al nostro buffer type: misurato 229.50 MiB su
        # Qwen3-30B-A3B Q4_K_M, contro 16.35 GiB serviti dallo store
        shared_repack = sum(
            t.n_bytes for t in model.shared_tensors
            if t.type_name in _REPACK_TYPES and len(t.dims) >= 2
            and not t.name.startswith("token_embd")
        )
        p.decisions.append(
            f"repack limitato ai tensori shared ({fmt_bytes(shared_repack)}): "
            "gli expert appartengono al buffer type dello store"
        )

    p.budget, budget_notes = safe_budget(hw, req.ram_limit)
    p.decisions.extend(budget_notes)

    # ------------------------------------------------------------- threads --
    logical = hw.cpu_cores_logical or 8
    physical = hw.cpu_cores_physical or max(1, logical // 2)
    if req.threads and req.threads > 0:
        p.threads = req.threads
    elif hw.is_hybrid_cpu and hw.cpu_p_cores:
        # CPU ibrida: un thread per P-core, nient'altro.
        #
        # Il commento qui sopra diceva gia' "partiamo dai thread dei soli
        # P-core", ma il codice scriveva `cpu_p_cores * 2`, cioe' i fratelli
        # SMT dei P-core. Su un i7 ibrido (4 P + 8 E) faceva 8, ed e' il valore
        # con cui sono state prese tutte le misure di questo progetto.
        #
        # Misurato su Qwen3-30B-A3B, generazione, tre ripetizioni per punto:
        #
        #     thread   Q4_K_M + store   IQ2_XXS residente
        #        4          5.93               7.08
        #        6          --                 5.19
        #        8          4.23               4.84
        #       12          --                 4.01
        #
        # Monotona decrescente in entrambe le quantizzazioni: +40% e +46%
        # passando da 8 a 4. Non e' un effetto del tipo di pesi, e' la barriera.
        # ggml sincronizza i thread a ogni nodo del grafo e aspetta il piu'
        # lento: un E-core e' molto piu' lento di un P-core, e due thread sullo
        # stesso P-core si contendono le stesse unita' vettoriali senza
        # aggiungere lavoro fatto. In entrambi i casi il thread in piu' non
        # porta calcolo, porta un altro partecipante da attendere.
        p.threads = max(2, min(physical, hw.cpu_p_cores))
        p.decisions.append(
            f"threads={p.threads} (CPU ibrida: {hw.cpu_p_cores} P-core + "
            f"{hw.cpu_e_cores} E-core; un thread per P-core, misurato +40-46% "
            f"contro il doppio. Su un'altra CPU verificalo con "
            f"'benchmark --sweep-threads')"
        )
    else:
        p.threads = max(1, min(physical, max(1, physical - 1)) if physical > 4 else physical)
        p.decisions.append(f"threads={p.threads} (core fisici {physical}, logici {logical})")
    # il prompt processing e' compute-bound: li' gli E-core aiutano
    p.threads_batch = max(p.threads, min(logical, physical))

    # ----------------------------------------------------------- context ----
    ctx_train = model.n_ctx_train or 32768
    if req.ctx:
        p.ctx = int(req.ctx)
        if p.ctx > ctx_train:
            p.warnings.append(
                f"--ctx {p.ctx} supera il context di training ({ctx_train}): "
                "la qualita' puo' degradare"
            )
    else:
        p.ctx = min(DEFAULT_CTX_16GB, ctx_train)
        p.decisions.append(f"context iniziale {p.ctx} (default 16 GB)")

    # ------------------------------------------------------------- batch ----
    p.batch = int(req.batch) if req.batch else 2048
    p.ubatch = int(req.ubatch) if req.ubatch else min(512, p.batch)
    if p.ubatch > p.batch:
        p.ubatch = p.batch

    # ---------------------------------------------------------- KV types ----
    kt = (req.kv_type_k or req.kv_type or "f16").lower()
    vt = (req.kv_type_v or req.kv_type or "f16").lower()
    for t in (kt, vt):
        if t not in KV_TYPE_BYTES:
            raise ValueError(f"--kv-type {t} non supportato (usa: {', '.join(KV_LADDER)})")
    p.kv_type_k, p.kv_type_v = kt, vt

    # --------------------------------------------------------- flash attn ---
    p.flash_attn = True if req.flash_attn is None else bool(req.flash_attn)
    if p.kv_type_k != p.kv_type_v and not p.flash_attn:
        p.warnings.append("KV K/V di tipo diverso senza flash attention: llama.cpp potrebbe rifiutare")
    if (p.kv_type_k != "f16" or p.kv_type_v != "f16") and not p.flash_attn:
        p.warnings.append("KV cache quantizzata richiede flash attention: la abilito")
        p.flash_attn = True

    # -------------------------------------------------------- MoE / experts -
    igpu = any(g.shares_system_ram for g in hw.gpus)
    if model.is_moe:
        p.cpu_moe = (req.backend == "vulkan")
        if p.cpu_moe:
            p.n_cpu_moe = model.n_moe_layers or (model.n_layers or 0)
            p.decisions.append(
                f"MoE: expert su CPU, tensori shared su GPU ({p.n_cpu_moe} layer MoE)"
            )

    # ------------------------------------------------------- GPU layers -----
    layers = model.n_layers or 32
    if req.gpu_layers is not None:
        p.gpu_layers = int(req.gpu_layers)
    elif req.backend == "vulkan":
        p.gpu_layers = layers + 1  # tutti + output layer, poi la ladder riduce
        p.decisions.append(f"gpu-layers={p.gpu_layers} (offload completo, verra' ridotto se serve)")
    else:
        p.gpu_layers = 0
    if req.backend != "vulkan":
        p.gpu_layers = 0

    # ------------------------------------------- expert cache / streaming ---
    if model.is_moe and req.allow_streaming:
        # budget provvisorio: cio' che resta dopo shared + KV + runtime
        if req.expert_cache is not None:
            p.expert_cache_budget = int(req.expert_cache)
        else:
            p.expert_cache_budget = 0  # deciso dalla ladder solo se necessario

    # --------------------------------------------------------- ladder -------
    _fit(model, hw, p, req)

    p.estimate = estimate_memory(model, p, hw)

    # Su iGPU la 'shared GPU memory' ha un tetto (tipicamente meta' della RAM):
    # se lo superiamo, il driver fallisce l'allocazione. Riduciamo l'offload.
    shared_cap = hw.gpu_shared_memory_budget
    if igpu and shared_cap and p.estimate.gpu_shared > shared_cap and req.gpu_layers is None:
        layers = model.n_layers or 32
        while p.gpu_layers > 0 and estimate_memory(model, p, hw).gpu_shared > shared_cap:
            p.gpu_layers = max(0, p.gpu_layers - max(1, layers // 8))
        p.estimate = estimate_memory(model, p, hw)
        p.decisions.append(
            f"[safety] gpu-layers limitato a {p.gpu_layers}: la shared GPU memory "
            f"non puo' superare {fmt_bytes(shared_cap)}"
        )

    if igpu and p.gpu_layers > 0:
        p.warnings.append(
            "GPU integrata: i layer offloadati usano 'shared GPU memory', "
            "cioe' la stessa RAM di sistema. Nessun risparmio di RAM."
        )
    return p


def _fit(model: GGUFModel, hw: HardwareReport, p: RuntimePlan, req: PlanRequest) -> None:
    """Applica la scala di sicurezza fino a rientrare nel budget."""
    budget = p.budget
    targets: Dict[str, Any] = {
        "ctx": p.ctx, "batch": p.batch, "ubatch": p.ubatch,
        "gpu_layers": p.gpu_layers, "kv": p.kv_type_k,
    }
    start = {"ctx": p.ctx, "batch": p.batch, "ubatch": p.ubatch,
             "gpu_layers": p.gpu_layers, "kv": p.kv_type_k}
    user_locked = {
        "ctx": req.ctx is not None,
        "batch": req.batch is not None,
        "kv": (req.kv_type or req.kv_type_k or req.kv_type_v) is not None,
        "gpu": req.gpu_layers is not None,
        "expert": req.expert_cache is not None,
    }

    def est() -> int:
        return estimate_memory(model, p, hw).total

    if est() <= budget:
        p.estimate = estimate_memory(model, p, hw)
        p.feasible = True
        return

    steps: List[tuple[str, Any]] = []

    # 0) disattivazione del repack SIMD, se l'utente l'ha forzato con --repack.
    # Di default e' gia' spento: su CPU senza AVX-512/AMX misura peggio sia in
    # RAM sia in velocita'. Resta come primo gradino perche' libera circa due
    # terzi del peso di un modello Q4, molto piu' di qualsiasi altra riduzione.
    if p.repack and not p.expert_store:
        steps.append(("repack", False))
    elif p.repack and p.expert_store:
        # Spegnere il repack qui sarebbe un risparmio apparente e un disastro
        # reale: `--no-repack` in llama.cpp imposta no_extra_bufts, che
        # disabilita tutti gli extra buffer type, incluso quello dello store. Il
        # risultato e' uno store aperto che non serve un solo tensore, cioe' una
        # funzione spenta senza che nessuno lo dica. E' esattamente il bug che
        # ha reso inutile la prima misura dello store.
        p.decisions.append(
            "- repack tenuto acceso: con l'expert store non e' un gradino "
            "disponibile, perche' --no-repack disabiliterebbe il buffer type "
            "dello store insieme al repack"
        )

    # 0-bis) la cache dei prompt, prima di toccare il contesto.
    #
    # E' l'unica riduzione che non toglie **niente** al modello: rinunciarci
    # significa rifare il prefill quando si riprende una conversazione, cioe'
    # pagare secondi, non perdere capacita'. Il contesto, il batch e la
    # precisione della KV cache cambiano invece cosa il modello puo' fare o
    # quanto bene lo fa. Quindi si sacrifica per prima, e si scende a uno stato
    # prima di arrivare a zero: con uno la conversazione in corso resta coperta.
    if p.prompt_cache_states > 1:
        steps.append(("prompt_cache", 1))
    if p.prompt_cache_states > 0:
        steps.append(("prompt_cache", 0))

    # 1) riduzione context
    if not user_locked["ctx"]:
        for c in CTX_LADDER:
            if c < p.ctx:
                steps.append(("ctx", c))
    # 2) riduzione batch
    if not user_locked["batch"]:
        for b in BATCH_LADDER:
            if b < p.batch:
                steps.append(("batch", b))
        for u in UBATCH_LADDER:
            if u < p.ubatch:
                steps.append(("ubatch", u))
    # 3/4) KV Q8 -> Q4
    if not user_locked["kv"]:
        steps.append(("kv", "q8_0"))
        steps.append(("kv", "q4_0"))
    # 5) riduzione GPU offload
    if not user_locked["gpu"] and p.gpu_layers > 0:
        layers = model.n_layers or 32
        for frac in (0.75, 0.5, 0.25, 0.0):
            steps.append(("gpu", int(layers * frac)))
    # 6) lazy loading / expert streaming
    if model.is_moe and req.allow_streaming:
        steps.append(("streaming", True))
    steps.append(("mmap", True))

    for kind, value in steps:
        if est() <= budget:
            break
        if kind == "repack":
            p.repack = False
            p.decisions.append(
                "[safety] repack SIMD disattivato: libera "
                f"{fmt_bytes(model.repackable_bytes)} di RAM"
            )
        elif kind == "prompt_cache":
            if value >= p.prompt_cache_states:
                continue
            prima = p.estimate.prompt_cache or prompt_cache_bytes(
                model, p.ctx, p.kv_type_k, p.kv_type_v, p.prompt_cache_states)
            p.prompt_cache_states = value
            p.decisions.append(
                f"[safety] cache dei prompt {value} stati: libera "
                f"{fmt_bytes(prima - prompt_cache_bytes(model, p.ctx, p.kv_type_k, p.kv_type_v, value))}"
                + (" (si ripaghera' il prefill riprendendo una conversazione, "
                   "ma il modello non perde niente)" if value == 0 else "")
            )
        elif kind == "ctx":
            if value >= p.ctx:
                continue
            p.ctx = value
        elif kind == "batch":
            p.batch = value
            p.ubatch = min(p.ubatch, value)
        elif kind == "ubatch":
            p.ubatch = value
        elif kind == "kv":
            p.kv_type_k = p.kv_type_v = value
            p.flash_attn = True
        elif kind == "gpu":
            if value >= p.gpu_layers:
                continue
            p.gpu_layers = value
        elif kind == "streaming":
            p.moe_streaming = True
            # tutto cio' che resta dopo shared/KV/activations va alla expert cache
            p.expert_cache_budget = 0
            base = estimate_memory(model, p, hw).total
            free = budget - base
            min_cache = min(_min_expert_cache(model), model.expert_bytes)
            cache = max(min_cache, int(free * 0.92))
            p.expert_cache_budget = min(cache, model.expert_bytes)
            p.decisions.append(
                f"[safety] expert streaming attivo: {fmt_bytes(model.shared_bytes)} di "
                f"tensori shared residenti, {fmt_bytes(model.expert_bytes)} di expert "
                "serviti da SSD tramite cache"
            )
        elif kind == "mmap":
            if not p.mmap:
                p.mmap = True
                p.decisions.append("[safety] mmap forzato (lazy loading dei pesi)")
            p.mlock = False

    # Se e' stato attivato l'expert streaming, la memoria liberata e' molta:
    # ripristiniamo i parametri sacrificati prima, in ordine di valore d'uso.
    if p.moe_streaming:
        _relax(model, hw, p, req, targets)

    # log compatto: un rigo per parametro, valore iniziale -> finale
    for label, attr in (("context", "ctx"), ("batch", "batch"), ("ubatch", "ubatch"),
                        ("gpu-layers", "gpu_layers")):
        if getattr(p, attr) != start[attr]:
            p.decisions.append(f"[safety] {label} {start[attr]} -> {getattr(p, attr)}")
    if p.kv_type_k != start["kv"]:
        p.decisions.append(
            f"[safety] KV cache {start['kv'].upper()} -> {p.kv_type_k.upper()}")

    p.estimate = estimate_memory(model, p, hw)
    if p.estimate.total > budget:
        p.feasible = False
        over = p.estimate.total - budget
        p.failure_reason = (
            f"stima {fmt_bytes(p.estimate.total)} oltre il budget {fmt_bytes(budget)} "
            f"di {fmt_bytes(over)} anche dopo tutte le riduzioni disponibili"
        )
    else:
        p.feasible = True
        if p.ctx < MIN_CTX:
            p.feasible = False
            p.failure_reason = f"context ridotto sotto il minimo utile ({MIN_CTX})"


# Alias comodi -> valori accettati da llama.cpp
SPEC_ALIASES = {
    "none": None,
    "off": None,
    "ngram": "ngram-cache",     # nessun modello draft: costo di RAM trascurabile
    "ngram-simple": "ngram-simple",
    "ngram-cache": "ngram-cache",
    "ngram-mod": "ngram-mod",
    "draft": "draft-simple",    # richiede --draft-model
    "draft-simple": "draft-simple",
}


def _setup_speculative(p: RuntimePlan, req: PlanRequest) -> None:
    """Configura la decodifica speculativa e ne contabilizza il costo in RAM.

    Due famiglie molto diverse su una macchina a corto di memoria:
      - draft model: piu' efficace, ma il draft occupa RAM che toglieremmo
        alla expert cache;
      - n-gram: propone token dal contesto gia' visto, senza alcun modello,
        quindi costa quasi nulla. Su testo ripetitivo (codice, documenti)
        e' spesso il compromesso giusto qui.
    """
    spec = (req.spec or "").strip().lower() or None
    if spec is None and req.draft_model:
        spec = "draft"
    if spec is None:
        return
    if spec not in SPEC_ALIASES:
        raise ValueError(
            f"--spec {spec} non riconosciuto (usa: {', '.join(sorted(SPEC_ALIASES))})"
        )
    p.spec_type = SPEC_ALIASES[spec]
    if p.spec_type is None:
        return

    p.draft_n_max = int(req.draft_n_max or 8)
    p.draft_n_min = 1

    if p.spec_type.startswith("draft"):
        if not req.draft_model:
            raise ValueError(
                "--spec draft richiede --draft-model con il percorso di un GGUF "
                "piccolo (es. Qwen3-0.6B-Q4_K_M.gguf). "
                "In alternativa usa --spec ngram, che non richiede alcun modello."
            )
        p.draft_model = str(req.draft_model)
        p.draft_ctx = int(req.draft_ctx or min(p.ctx, 4096))
        p.draft_bytes = _draft_footprint(p.draft_model, p.draft_ctx)
    else:
        p.decisions.append(
            f"speculativa {p.spec_type}: nessun modello draft, costo di RAM trascurabile"
        )


def _draft_footprint(path: str, ctx: int) -> int:
    """Pesi + KV cache del modello draft. Se non lo si riesce a leggere, stima prudente."""
    from .gguf import GGUFError, read_gguf

    try:
        d = read_gguf(path)
    except (GGUFError, OSError):
        return 1 * GB
    kv = kv_cache_bytes(d, ctx, "f16", "f16")
    return int(d.tensor_bytes + kv + 128 * MB)


def _min_expert_cache(model: GGUFModel) -> int:
    """Cache minima sensata: gli expert di alcuni token, per non thrashare."""
    per_layer = model.expert_bytes_per_expert_per_layer or (48 * MB)
    used = max(1, model.expert_used_count)
    return max(int(per_layer * used * 3), 512 * MB)


def _relax(model: GGUFModel, hw: HardwareReport, p: RuntimePlan,
           req: PlanRequest, targets: Dict[str, Any]) -> None:
    """Riporta ctx/batch/KV/offload al valore migliore che sta ancora nel budget.

    Lo streaming degli expert e' una scelta architetturale, non un degrado di
    qualita': una volta attivo non ha senso restare con context 512. La expert
    cache assorbe tutto cio' che resta.
    """
    budget = p.budget
    min_cache = min(_min_expert_cache(model), model.expert_bytes)

    def fits(cache: int = min_cache) -> bool:
        saved = p.expert_cache_budget
        p.expert_cache_budget = cache
        total = estimate_memory(model, p, hw).total
        p.expert_cache_budget = saved
        return total <= budget

    restored: List[str] = []

    # 1) context: e' il parametro che l'utente percepisce di piu'
    target_ctx = int(targets["ctx"])
    if p.ctx < target_ctx and req.ctx is None:
        old = p.ctx
        for c in CTX_LADDER:
            if c > target_ctx or c <= p.ctx:
                continue
            prev = p.ctx
            p.ctx = c
            if fits():
                break
            p.ctx = prev
        if p.ctx > old:
            restored.append(f"context {old} -> {p.ctx}")

    # 2) offload degli shared tensor su GPU: con gli expert su CPU costa poco
    if p.cpu_moe and req.gpu_layers is None:
        target_gpu = int(targets["gpu_layers"])
        if p.gpu_layers < target_gpu:
            prev = p.gpu_layers
            p.gpu_layers = target_gpu
            if fits():
                restored.append(f"gpu-layers {prev} -> {target_gpu} (solo tensori shared)")
            else:
                p.gpu_layers = prev

    # 3) batch: incide sulla velocita' di prompt processing
    if req.batch is None:
        for name, ladder, target in (("ubatch", UBATCH_LADDER, targets["ubatch"]),
                                     ("batch", BATCH_LADDER, targets["batch"])):
            cur = getattr(p, name)
            if cur >= target:
                continue
            old = cur
            for v in ladder:
                if v > target or v <= getattr(p, name):
                    continue
                prev = getattr(p, name)
                setattr(p, name, v)
                if fits():
                    break
                setattr(p, name, prev)
            if getattr(p, name) > old:
                restored.append(f"{name} {old} -> {getattr(p, name)}")
    if p.ubatch > p.batch:
        p.ubatch = p.batch

    # 4) precisione KV: la alziamo solo se resta spazio abbondante per la cache
    if req.kv_type is None and req.kv_type_k is None and req.kv_type_v is None:
        for t in ("f16", "q8_0"):
            if KV_TYPE_BYTES[t] <= KV_TYPE_BYTES[p.kv_type_k]:
                continue
            prev_k, prev_v = p.kv_type_k, p.kv_type_v
            p.kv_type_k = p.kv_type_v = t
            # richiediamo il doppio della cache minima per giustificare il costo
            if fits(cache=min(model.expert_bytes, min_cache * 2)):
                restored.append(f"KV cache {prev_k.upper()} -> {t.upper()}")
                break
            p.kv_type_k, p.kv_type_v = prev_k, prev_v

    # 5) la expert cache assorbe tutta la memoria residua
    p.expert_cache_budget = 0
    base = estimate_memory(model, p, hw).total
    free = budget - base
    cache = max(min_cache, int(free * 0.92))
    p.expert_cache_budget = min(cache, model.expert_bytes)

    if req.expert_store and req.expert_store_budget:
        # Con l'expert store la residenza degli expert non e' una stima su cio'
        # che l'OS terra' in cache: e' un tetto che imponiamo noi. Il piano deve
        # usare QUEL numero, altrimenti dichiara un totale e la macchina ne
        # occupa un altro.
        #
        # E' successo: piano con 6.37 GiB di expert stimati, store avviato con 8
        # GiB, picco al 106% del budget e 206 MiB di RAM libera. Il benchmark
        # che ne e' uscito misurava una macchina che paginava, e il calo del 13%
        # fra la prima e l'ultima ripetizione l'avevo attribuito al calore senza
        # poterlo distinguere dalla paginazione.
        richiesto = min(int(req.expert_store_budget), model.expert_bytes)
        p.expert_cache_sostenibile = p.expert_cache_budget
        if richiesto > p.expert_cache_budget:
            p.warnings.append(
                f"--expert-budget {fmt_bytes(richiesto)} supera lo spazio che il piano "
                f"lascia agli expert ({fmt_bytes(p.expert_cache_budget)}): la macchina "
                "andra' a paginare e il risultato dira' poco. Togli --expert-budget per "
                "far scegliere il numero al planner."
            )
        p.expert_cache_budget = richiesto
        p.decisions.append(
            f"[store] expert cache fissata a {fmt_bytes(p.expert_cache_budget)} "
            "da --expert-budget: il piano usa questo numero, non una stima"
        )
    elif req.expert_store:
        p.decisions.append(
            f"[store] expert cache {fmt_bytes(p.expert_cache_budget)} decisa dal "
            "planner e imposta allo store come tetto rigido"
        )
    else:
        p.decisions.append(
            f"[relax] expert cache finale {fmt_bytes(p.expert_cache_budget)} "
            f"su {fmt_bytes(model.expert_bytes)} di expert"
        )
