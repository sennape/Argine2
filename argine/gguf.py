"""Parser GGUF puro-Python (nessuna dipendenza nativa).

Legge header, metadata KV e tabella dei tensori. Da qui derivano:
  - architettura, quantizzazione dominante, numero parametri
  - detection MoE (expert_count / expert_used_count / tensori *_exps)
  - contabilita' separata shared tensors vs expert tensors (per M2/M3)
  - geometria necessaria al Memory Planner per la KV cache

Formato: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

GGUF_MAGIC = b"GGUF"

# ----------------------------------------------------------- tipi metadata --
(
    T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32, T_FLOAT32, T_BOOL,
    T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64,
) = range(13)

_SCALAR = {
    T_UINT8: ("<B", 1), T_INT8: ("<b", 1),
    T_UINT16: ("<H", 2), T_INT16: ("<h", 2),
    T_UINT32: ("<I", 4), T_INT32: ("<i", 4),
    T_FLOAT32: ("<f", 4), T_BOOL: ("<?", 1),
    T_UINT64: ("<Q", 8), T_INT64: ("<q", 8),
    T_FLOAT64: ("<d", 8),
}

# ------------------------------------------------------------- tipi ggml -----
# nome, block_size, type_size (byte per blocco)
GGML_TYPES: Dict[int, Tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 40),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66),
    36: ("MXFP4", 32, 17),
}

# Byte per elemento della KV cache, per tipo accettato da llama.cpp
KV_TYPE_BYTES = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 34 / 32,
    "q5_1": 24 / 32,
    "q5_0": 22 / 32,
    "q4_1": 20 / 32,
    "q4_0": 18 / 32,
}

# Tipi che llama.cpp ripacchetta in un buffer CPU aggiuntivo (repack / extra bufts).
# Q6_K e i tipi K a 5-6 bit non sono ripacchettati: verificato sui log di llama.cpp.
REPACK_TYPES = {"Q4_0", "Q4_1", "Q4_K", "Q2_K", "IQ4_NL", "IQ4_XS"}

_EXPERT_RE = re.compile(r"(_exps\.|\.experts\.|ffn_(gate|up|down)_exps)")
_EXPERT_SHARED_RE = re.compile(r"_shexp\.")


class GGUFError(Exception):
    pass


@dataclass
class TensorInfo:
    name: str
    dims: List[int]
    ggml_type: int
    offset: int                 # offset relativo all'area dati del proprio file
    file: str = ""              # shard che contiene il tensore
    data_offset: int = 0        # inizio dell'area dati in quel file

    @property
    def abs_offset(self) -> int:
        """Posizione assoluta nel file: serve a leggere il tensore da disco."""
        return self.data_offset + self.offset

    @property
    def type_name(self) -> str:
        return GGML_TYPES.get(self.ggml_type, (f"TYPE_{self.ggml_type}", 1, 0))[0]

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n

    @property
    def n_bytes(self) -> int:
        name, blk, tsize = GGML_TYPES.get(self.ggml_type, ("?", 1, 0))
        if tsize == 0:
            return 0
        ne = self.n_elements
        # il primo asse deve essere multiplo di blk; arrotondiamo per sicurezza
        nblocks = (ne + blk - 1) // blk
        return nblocks * tsize

    @property
    def is_expert(self) -> bool:
        """Tensore di expert routed (escluso lo shared expert, sempre residente)."""
        return bool(_EXPERT_RE.search(self.name)) and not _EXPERT_SHARED_RE.search(self.name)

    @property
    def layer_index(self) -> Optional[int]:
        m = re.search(r"blk\.(\d+)\.", self.name)
        return int(m.group(1)) if m else None


@dataclass
class GGUFModel:
    path: str
    file_size: int = 0
    gguf_version: int = 0
    tensor_count: int = 0
    kv_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    tensors: List[TensorInfo] = field(default_factory=list)
    data_offset: int = 0
    alignment: int = 32
    shards: List[str] = field(default_factory=list)  # se il modello e' splittato

    # ------------------------------------------------------------- metadata
    @property
    def architecture(self) -> str:
        return str(self.metadata.get("general.architecture", "unknown"))

    def arch_key(self, suffix: str, default: Any = None) -> Any:
        return self.metadata.get(f"{self.architecture}.{suffix}", default)

    @property
    def name(self) -> str:
        return str(self.metadata.get("general.name", Path(self.path).stem))

    @property
    def n_layers(self) -> Optional[int]:
        v = self.arch_key("block_count")
        return int(v) if v is not None else None

    @property
    def n_embd(self) -> Optional[int]:
        v = self.arch_key("embedding_length")
        return int(v) if v is not None else None

    @property
    def n_head(self) -> Optional[int]:
        v = self.arch_key("attention.head_count")
        if isinstance(v, list):
            v = max(v) if v else None
        return int(v) if v is not None else None

    @property
    def n_head_kv(self) -> Optional[int]:
        v = self.arch_key("attention.head_count_kv")
        if isinstance(v, list):
            v = max(v) if v else None
        if v is None:
            return self.n_head
        return int(v)

    @property
    def head_dim_k(self) -> Optional[int]:
        v = self.arch_key("attention.key_length")
        if v is not None:
            return int(v)
        if self.n_embd and self.n_head:
            return self.n_embd // self.n_head
        return None

    @property
    def head_dim_v(self) -> Optional[int]:
        v = self.arch_key("attention.value_length")
        if v is not None:
            return int(v)
        return self.head_dim_k

    @property
    def n_ff(self) -> Optional[int]:
        """Dimensione della FFN (per i MoE: quella di un singolo expert)."""
        for key in ("feed_forward_length", "expert_feed_forward_length"):
            v = self.arch_key(key)
            if isinstance(v, list):
                v = max(v) if v else None
            if v is not None:
                return int(v)
        return None

    @property
    def n_ctx_train(self) -> Optional[int]:
        v = self.arch_key("context_length")
        return int(v) if v is not None else None

    @property
    def rope_freq_base(self) -> Optional[float]:
        v = self.arch_key("rope.freq_base")
        return float(v) if v is not None else None

    # ------------------------------------------------------------------ MoE
    @property
    def expert_count(self) -> int:
        v = self.arch_key("expert_count", 0)
        try:
            return int(v or 0)
        except (TypeError, ValueError):
            return 0

    @property
    def expert_used_count(self) -> int:
        v = self.arch_key("expert_used_count", 0)
        try:
            return int(v or 0)
        except (TypeError, ValueError):
            return 0

    @property
    def expert_shared_count(self) -> int:
        v = self.arch_key("expert_shared_count", 0)
        try:
            return int(v or 0)
        except (TypeError, ValueError):
            return 0

    @property
    def is_moe(self) -> bool:
        return self.expert_count > 1 or any(t.is_expert for t in self.tensors)

    # ------------------------------------------------------------ dimensioni
    @property
    def n_params_total(self) -> int:
        return sum(t.n_elements for t in self.tensors)

    @property
    def expert_tensors(self) -> List[TensorInfo]:
        return [t for t in self.tensors if t.is_expert]

    @property
    def shared_tensors(self) -> List[TensorInfo]:
        return [t for t in self.tensors if not t.is_expert]

    @property
    def expert_bytes(self) -> int:
        return sum(t.n_bytes for t in self.expert_tensors)

    @property
    def shared_bytes(self) -> int:
        return sum(t.n_bytes for t in self.shared_tensors)

    @property
    def tensor_bytes(self) -> int:
        return self.shared_bytes + self.expert_bytes

    @property
    def n_params_active(self) -> Optional[int]:
        """Parametri realmente attivati per token.

        Per un MoE: shared + (expert_used / expert_count) * expert_params.
        """
        if not self.is_moe:
            return self.n_params_total
        ec, eu = self.expert_count, self.expert_used_count
        if ec <= 0:
            return None
        if eu <= 0:
            eu = 1
        shared_p = sum(t.n_elements for t in self.shared_tensors)
        expert_p = sum(t.n_elements for t in self.expert_tensors)
        return int(shared_p + expert_p * (eu / ec))

    @property
    def expert_bytes_per_expert(self) -> Optional[int]:
        """Byte occupati da un singolo expert su tutti i layer."""
        if not self.is_moe or self.expert_count <= 0:
            return None
        return int(self.expert_bytes / self.expert_count)

    @property
    def expert_bytes_per_expert_per_layer(self) -> Optional[int]:
        """Byte di un expert in un singolo layer: e' l'unita' di trasferimento
        della expert cache SSD->RAM (M3)."""
        if not self.is_moe or self.expert_count <= 0:
            return None
        layers = {t.layer_index for t in self.expert_tensors if t.layer_index is not None}
        n_moe_layers = len(layers) or (self.n_layers or 1)
        return int(self.expert_bytes / (self.expert_count * n_moe_layers))

    @property
    def n_moe_layers(self) -> int:
        layers = {t.layer_index for t in self.expert_tensors if t.layer_index is not None}
        return len(layers)

    @property
    def repackable_bytes(self) -> int:
        """Byte che llama.cpp duplica nel buffer CPU_REPACK.

        Su x86 llama.cpp riscrive i pesi quantizzati in un layout interleaved
        adatto alle istruzioni SIMD. E' una copia IN AGGIUNTA al file mappato,
        e su un modello Q4 vale circa due terzi dei pesi: ignorarla significa
        sottostimare la RAM del 35-40%.

        Sono esclusi i tensori 1D (norm) e gli embedding, che non passano per
        una moltiplicazione matriciale. Verificato su Qwen3-8B-Q4_K_M:
        3199 MiB calcolati contro 3199.50 MiB dichiarati da llama.cpp.
        """
        total = 0
        for t in self.tensors:
            if t.type_name not in REPACK_TYPES:
                continue
            if len(t.dims) < 2:
                continue
            if t.name.startswith("token_embd"):
                continue
            total += t.n_bytes
        return total

    @property
    def quantization(self) -> str:
        """Tipo di quantizzazione dominante (per byte) + eventuale file type."""
        ftype = self.metadata.get("general.file_type")
        counts: Dict[str, int] = {}
        for t in self.tensors:
            counts[t.type_name] = counts.get(t.type_name, 0) + t.n_bytes
        if not counts:
            return str(ftype) if ftype is not None else "unknown"
        dominant = max(counts.items(), key=lambda kv: kv[1])[0]
        return dominant

    @property
    def quant_mix(self) -> List[Tuple[str, int, int]]:
        """(tipo, n_tensori, byte) ordinato per byte discendente."""
        agg: Dict[str, List[int]] = {}
        for t in self.tensors:
            e = agg.setdefault(t.type_name, [0, 0])
            e[0] += 1
            e[1] += t.n_bytes
        return sorted(((k, v[0], v[1]) for k, v in agg.items()), key=lambda x: -x[2])

    @property
    def bits_per_weight(self) -> Optional[float]:
        p = self.n_params_total
        if not p:
            return None
        return self.tensor_bytes * 8.0 / p

    @property
    def total_size_on_disk(self) -> int:
        """Somma di tutti gli shard se il GGUF e' splittato."""
        if self.shards:
            total = 0
            for s in self.shards:
                try:
                    total += Path(s).stat().st_size
                except OSError:
                    pass
            return total
        return self.file_size


# ------------------------------------------------------------------ reader --

class _Reader:
    def __init__(self, fh, limit: Optional[int] = None):
        self.fh = fh
        self.limit = limit

    def read(self, n: int) -> bytes:
        b = self.fh.read(n)
        if len(b) != n:
            raise GGUFError("file GGUF troncato o non valido")
        return b

    def scalar(self, t: int):
        fmt, size = _SCALAR[t]
        return struct.unpack(fmt, self.read(size))[0]

    def string(self) -> str:
        n = struct.unpack("<Q", self.read(8))[0]
        if n > 64 * 1024 * 1024:
            raise GGUFError("stringa GGUF implausibile (file corrotto?)")
        return self.read(n).decode("utf-8", errors="replace")

    def value(self, vtype: int, _depth: int = 0):
        if vtype in _SCALAR:
            return self.scalar(vtype)
        if vtype == T_STRING:
            return self.string()
        if vtype == T_ARRAY:
            if _depth > 4:
                raise GGUFError("array GGUF troppo annidato")
            etype = struct.unpack("<I", self.read(4))[0]
            n = struct.unpack("<Q", self.read(8))[0]
            if n > 200_000_000:
                raise GGUFError("array GGUF implausibile")
            # Gli array enormi (vocabolario) non vanno tenuti per intero:
            # ne conserviamo solo la lunghezza.
            if n > 8192:
                if etype in _SCALAR:
                    _fmt, size = _SCALAR[etype]
                    self.fh.seek(size * n, 1)
                else:
                    for _ in range(n):
                        self.value(etype, _depth + 1)
                return _BigArray(n, etype)
            return [self.value(etype, _depth + 1) for _ in range(n)]
        raise GGUFError(f"tipo metadata GGUF sconosciuto: {vtype}")


class _BigArray:
    """Placeholder per array molto grandi (es. tokenizer.ggml.tokens)."""

    __slots__ = ("length", "etype")

    def __init__(self, length: int, etype: int):
        self.length = length
        self.etype = etype

    def __len__(self) -> int:
        return self.length

    def __repr__(self) -> str:
        return f"<array len={self.length}>"


def _find_shards(path: Path) -> List[str]:
    """Rileva GGUF splittati: nome-00001-of-00003.gguf."""
    m = re.match(r"(?P<base>.+)-(?P<idx>\d{5})-of-(?P<tot>\d{5})\.gguf$", path.name)
    if not m:
        return []
    base, tot = m.group("base"), int(m.group("tot"))
    shards = []
    for i in range(1, tot + 1):
        cand = path.parent / f"{base}-{i:05d}-of-{tot:05d}.gguf"
        if cand.exists():
            shards.append(str(cand))
    return shards


def read_gguf(path: str | Path, read_tensors: bool = True) -> GGUFModel:
    """Legge un GGUF, ricomponendo tutti gli shard se il modello e' splittato.

    Ogni shard ha un header proprio con la sua porzione di tensori: leggere solo
    il primo file significherebbe contare una frazione dei pesi e, su un MoE,
    sottostimare gravemente la dimensione degli expert.
    """
    p = Path(path)
    if not p.exists():
        raise GGUFError(f"file non trovato: {p}")
    if p.is_dir():
        raise GGUFError(f"{p} e' una directory, serve un file .gguf")

    model = _read_one(p, read_tensors=read_tensors)
    model.shards = _find_shards(p)

    if model.shards and read_tensors:
        atteso = int(model.metadata.get("split.count") or len(model.shards))
        if len(model.shards) < atteso:
            raise GGUFError(
                f"il modello e' splittato in {atteso} file ma ne trovo "
                f"{len(model.shards)} in {p.parent}: scaricali tutti"
            )
        for shard in model.shards[1:]:
            altro = _read_one(Path(shard), read_tensors=True)
            model.tensors.extend(altro.tensors)
            model.tensor_count += altro.tensor_count
    return model


def _read_one(p: Path, read_tensors: bool = True) -> GGUFModel:
    """Legge un singolo file GGUF: header, metadata e tabella dei tensori."""
    model = GGUFModel(path=str(p.resolve()), file_size=p.stat().st_size)

    with open(p, "rb") as fh:
        r = _Reader(fh)
        magic = r.read(4)
        if magic != GGUF_MAGIC:
            raise GGUFError(
                f"magic non valido {magic!r}: il file non e' in formato GGUF "
                "(i .safetensors o .bin PyTorch non sono supportati)"
            )
        model.gguf_version = struct.unpack("<I", r.read(4))[0]
        if model.gguf_version < 2:
            raise GGUFError(f"GGUF v{model.gguf_version} non supportato (serve v2+)")
        model.tensor_count = struct.unpack("<Q", r.read(8))[0]
        model.kv_count = struct.unpack("<Q", r.read(8))[0]
        if model.kv_count > 1_000_000 or model.tensor_count > 10_000_000:
            raise GGUFError("header GGUF implausibile")

        for _ in range(model.kv_count):
            key = r.string()
            vtype = struct.unpack("<I", r.read(4))[0]
            model.metadata[key] = r.value(vtype)

        model.alignment = int(model.metadata.get("general.alignment", 32) or 32)

        if read_tensors:
            for _ in range(model.tensor_count):
                name = r.string()
                ndims = struct.unpack("<I", r.read(4))[0]
                if ndims > 8:
                    raise GGUFError(f"tensore {name}: {ndims} dimensioni non plausibili")
                dims = list(struct.unpack(f"<{ndims}Q", r.read(8 * ndims)))
                ttype = struct.unpack("<I", r.read(4))[0]
                offset = struct.unpack("<Q", r.read(8))[0]
                model.tensors.append(TensorInfo(
                    name=name, dims=dims, ggml_type=ttype, offset=offset,
                    file=model.path,
                ))
            pos = fh.tell()
            pad = (model.alignment - (pos % model.alignment)) % model.alignment
            model.data_offset = pos + pad
            # ora che sappiamo dove inizia l'area dati, completiamo i tensori:
            # ogni shard ha il proprio, e serve per leggerli da disco
            for t in model.tensors:
                t.data_offset = model.data_offset

    return model


def _pick_main_shard(cands: List[Path]) -> Path:
    first = [c for c in cands if re.search(r"-00001-of-\d{5}\.gguf$", c.name)]
    return first[0] if first else cands[0]


def _common_model_dirs() -> List[Path]:
    """Cartelle in cui la gente tiene (o scarica) i GGUF."""
    home = Path.home()
    dirs = [
        Path.cwd(), home / "Downloads", home / "Download", home / "Desktop",
        home / "modelli", home / "models", Path("C:/modelli"), Path("C:/models"),
        home / ".cache" / "huggingface" / "hub",
    ]
    out: List[Path] = []
    for d in dirs:
        try:
            if d.is_dir() and d not in out:
                out.append(d)
        except OSError:
            continue
    return out


def _search_nearby(name: str) -> List[Path]:
    """Cerca un GGUF con quel nome nelle cartelle tipiche (un livello di sottocartelle)."""
    found: List[Path] = []
    for d in _common_model_dirs():
        try:
            for cand in list(d.glob("*.gguf")) + list(d.glob("*/*.gguf")):
                if cand.name.lower() == name.lower():
                    found.append(cand)
        except OSError:
            continue
    return found


def _per_somiglianza(files: List[Path], chiesto: str) -> List[Path]:
    """Ordina i candidati dal piu' simile al nome scritto dall'utente.

    L'elenco era in ordine alfabetico e tagliato agli otto, senza dire che
    tagliava. In una cartella con nove GGUF dai nomi quasi uguali, il file giusto
    e' finito nono — l'ordinamento di Windows ignora le maiuscole, e
    `...-Q4_K_M-reordered.gguf` viene prima di `...-Q4_K_M.gguf` perche' il
    trattino precede il punto. Cosi' il messaggio che esiste per aiutare ha
    elencato otto file sbagliati e nascosto l'unico utile.

    Un elenco che deve essere troncato va ordinato per pertinenza, non per
    alfabeto: chi ha sbagliato a scrivere un nome ha scritto qualcosa di simile a
    quello giusto. E se si tronca, bisogna dirlo.
    """
    import difflib

    base = chiesto.lower()
    return sorted(
        files,
        key=lambda f: (-difflib.SequenceMatcher(None, f.name.lower(), base).ratio(),
                       f.name.lower()),
    )


def find_gguf(path: str | Path) -> Path:
    """Accetta un file .gguf o una directory e ne risolve il modello principale."""
    p = Path(path).expanduser()
    if p.is_file():
        return p
    if p.is_dir():
        cands = sorted(p.glob("*.gguf"))
        if not cands:
            raise GGUFError(
                f"nessun file .gguf dentro {p}\n"
                "  Hai gia' scaricato il modello? Deve essere un file .gguf "
                "(non .safetensors e non una cartella di HuggingFace)."
            )
        return _pick_main_shard(cands)

    # il percorso non esiste: proviamo a essere utili invece di dire solo "non valido"
    msg = [f"file non trovato: {p}"]
    parent = p.parent
    if p.name.lower().endswith(".gguf"):
        altrove = _search_nearby(p.name)
        if altrove:
            msg.append("  Pero' un file con quel nome esiste qui:")
            msg.extend(f"    {a}" for a in altrove[:5])
            msg.append("  Usa quel percorso, oppure sposta il file dove ti aspettavi.")
            raise GGUFError("\n".join(msg))
    if parent.is_dir():
        vicini = _per_somiglianza(sorted(parent.glob("*.gguf")), p.name)
        if vicini:
            msg.append(f"  In {parent} ci sono pero' questi .gguf, "
                       f"dal piu' simile a quello che hai scritto:")
            msg.extend(f"    {v.name}" for v in vicini[:8])
            if len(vicini) > 8:
                msg.append(f"    ... e altri {len(vicini) - 8}")
            msg.append("  Controlla il nome esatto (maiuscole comprese).")
        else:
            msg.append(f"  La cartella {parent} esiste ma non contiene nessun .gguf.")
            msg.append("  Probabilmente il modello non e' ancora stato scaricato.")
    else:
        msg.append(f"  La cartella {parent} non esiste.")
        msg.append("  Creala e scaricaci dentro il modello, oppure indica il percorso giusto.")
        altri = []
        for d in _common_model_dirs():
            try:
                altri.extend(sorted(d.glob("*.gguf"))[:3])
            except OSError:
                continue
        if altri:
            msg.append("  GGUF trovati altrove sul disco:")
            msg.extend(f"    {a}" for a in altri[:6])
    raise GGUFError("\n".join(msg))
