"""Expert store: i pesi degli expert in un file pensato per essere letto.

Perche' non basta il GGUF
-------------------------
La misura (`argine diskbench`) dice tre cose sul disco di questa macchina:

  - una lettura esplicita da 896 KiB rende 1324 MB/s, la stessa slice presa per
    page fault ne rende 310: il percorso di fault costa 4.3x;
  - una lettura da 2.75 MiB rende 2618 MB/s contro i 1324 di una da 896 KiB:
    **leggere in blocchi grandi vale piu' che leggere in parallelo**;
  - senza `FILE_FLAG_NO_BUFFERING` Windows tiene una sua copia in cache, e la
    RAM la contiamo due volte.

Il GGUF non permette di sfruttare nessuna delle tre. I tre tensori di un expert
(`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`) stanno in punti diversi del
file, quindi servono tre letture piccole invece di una grande; e le slice non
sono allineate al settore, quindi l'I/O non bufferizzato le rifiuta.

Questo modulo scrive un file che risolve entrambe le cose: per ogni coppia
(layer, expert) le tre slice sono **adiacenti**, e il gruppo comincia a un
offset multiplo di 4096 con la coda riempita di zeri. Una lettura per expert,
allineata, senza cache dell'OS in mezzo.

La proprieta' che rende tutto questo poco rischioso
---------------------------------------------------
L'indice del file fa da indirezione fra `(layer, expert)` e la posizione su
disco. Quindi l'ordine fisico dei gruppi e' **libero**: si possono mettere
vicini gli expert co-attivati senza toccare i pesi del router, che invece
`argine reorder` era obbligato a permutare per non rompere il modello. Il
riordino per localita' qui non ha alcun costo di correttezza: e' solo una scelta
di layout, e i pesi restano identici byte per byte.

Formato (v1, little-endian)
---------------------------
    intestazione  256 byte a campi fissi
    tabella layer n_layers x u32   (indice del layer nel modello)
    tabella slice n_layers*n_experts x 36 byte:
                  u64 offset di gate   (multiplo di `alignment`)
                  u64 offset di up     (multiplo di `alignment`)
                  u64 offset di down   (multiplo di `alignment`)
                  u32 byte utili di gate
                  u32 byte utili di up
                  u32 byte utili di down
    area dati     le slice, ciascuna allineata e con la coda a zero

Perche' ogni slice e' allineata da sola, e non solo il gruppo
-------------------------------------------------------------
Nella prima versione le tre slice di un expert erano adiacenti e si leggevano
con una sola chiamata. Due misure hanno demolito quell'idea:

  - a coda 8 una lettura da 2.92 MiB non rende piu' di tre da ~900 KiB
    (4704 contro 4518 MB/s): il vantaggio dei blocchi grandi esisteva solo a
    coda 1, dove nessun motore lavora;
  - ggml pretende un blocco contiguo **per tensore**, quindi una lettura unica
    andrebbe smistata in tre regioni diverse: staging piu' memcpy, cioe' 0.38 ms
    di copia sopra 0.72 ms di disco, +53% per niente.

Con ogni slice allineata da sola, ogni lettura atterra direttamente nella
memoria del suo tensore: nessuna copia, nessun buffer intermedio. Su
Qwen3-30B-A3B Q4_K_M il riempimento resta zero, perche' le slice sono gia'
multipli della pagina (gate in Q4_K: 884736 byte = 216 pagine; down in Q6_K:
1290240 = 315 pagine).
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .gguf import GGUFModel, TensorInfo
from .util import fmt_bytes

MAGIC = 0x53585350  # 'PSXS'
VERSION = 2
HEADER_BYTES = 256
ENTRY_BYTES = 36
DEFAULT_ALIGNMENT = 4096

KINDS = ("gate", "up", "down")
_KIND_RE = {
    "gate": re.compile(r"\.ffn_gate_exps\.weight$"),
    "up": re.compile(r"\.ffn_up_exps\.weight$"),
    "down": re.compile(r"\.ffn_down_exps\.weight$"),
}

FLAG_ALIGNED = 1 << 0


class StoreError(Exception):
    pass


# ------------------------------------------------------------- geometria -----


@dataclass
class Slab:
    """Una slice di expert dentro un tensore del GGUF."""

    tensor: TensorInfo
    expert: int
    n_bytes: int

    @property
    def src_offset(self) -> int:
        return self.tensor.abs_offset + self.expert * self.n_bytes


def expert_tensors(model: GGUFModel) -> Dict[int, Dict[str, TensorInfo]]:
    """Per ogni layer, i tre tensori degli expert routed, per tipo."""
    out: Dict[int, Dict[str, TensorInfo]] = {}
    for t in model.tensors:
        if t.layer_index is None:
            continue
        for kind, rx in _KIND_RE.items():
            if rx.search(t.name):
                out.setdefault(t.layer_index, {})[kind] = t
    return out


def check_model(model: GGUFModel) -> List[str]:
    """Controlli prima di scrivere decine di gigabyte."""
    problemi: List[str] = []
    if not model.is_moe:
        return ["il modello non e' MoE: non ci sono expert da estrarre"]
    n_exp = model.expert_count
    if n_exp <= 0:
        return ["expert_count assente o nullo nei metadati"]
    # gate e up fusi in un tensore solo. llama.cpp lo preferisce quando il GGUF
    # lo contiene (`create_tensor_gate_up_exps`), e per noi e' il caso peggiore:
    # il nome non corrisponde a nessuno dei tre che cerchiamo, quindi senza
    # questo controllo lo store si costruirebbe con solo `down` — un terzo dei
    # byte — e il tetto di RAM sarebbe falso del doppio.
    #
    # Va detto qui, prima di scrivere decine di GiB, e con una riga sola: la
    # diagnosi generica diceva "layer N: manca ffn_gate/up_exps.weight"
    # ripetuta per ogni layer, che e' vera e non spiega niente.
    fusi = [t.name for t in model.tensors
            if t.name.endswith(".ffn_gate_up_exps.weight")]
    if fusi:
        return [
            f"questo GGUF ha gate e up FUSI in un tensore solo "
            f"({len(fusi)} layer, es. {fusi[0]}): l'expert store vuole "
            f"ffn_gate_exps e ffn_up_exps separati. Serve una conversione del "
            f"modello che non fonda i due, oppure un GGUF di un'altra provenienza."
        ]
    gruppi = expert_tensors(model)
    if not gruppi:
        return ["nessun tensore *_exps.weight trovato"]
    for layer, per_kind in sorted(gruppi.items()):
        mancanti = [k for k in KINDS if k not in per_kind]
        if mancanti:
            problemi.append(f"layer {layer}: manca ffn_{'/'.join(mancanti)}_exps.weight")
            continue
        for kind, t in per_kind.items():
            if not t.dims or t.dims[-1] != n_exp:
                problemi.append(
                    f"{t.name}: ultima dimensione {t.dims[-1] if t.dims else 0} "
                    f"!= {n_exp} expert"
                )
            elif t.n_bytes % n_exp:
                problemi.append(
                    f"{t.name}: {t.n_bytes} byte non divisibili per {n_exp} expert: "
                    "la slice per expert non sarebbe contigua"
                )
    return problemi


# ------------------------------------------------------------------ piano ----


def _arrotonda(n: int, alignment: int) -> int:
    resto = n % alignment
    return n if resto == 0 else n + (alignment - resto)


@dataclass
class Group:
    """Le tre slice di un expert. Ognuna allineata da sola, cosi' ogni lettura
    atterra direttamente nella memoria del tensore a cui appartiene."""

    layer: int
    expert: int
    slabs: Tuple[Slab, Slab, Slab]
    offsets: Tuple[int, int, int] = (0, 0, 0)  # riempiti da build_plan

    @property
    def offset(self) -> int:
        """Inizio del gruppo: serve solo alla contabilita' e ai messaggi."""
        return self.offsets[0]

    @property
    def raw_bytes(self) -> int:
        return sum(s.n_bytes for s in self.slabs)

    def padded_bytes(self, alignment: int) -> int:
        return sum(_arrotonda(s.n_bytes, alignment) for s in self.slabs)


@dataclass
class StorePlan:
    layers: List[int]
    n_experts: int
    alignment: int
    groups: List[Group]
    data_offset: int
    total_bytes: int
    model_path: str = ""
    model_bytes: int = 0

    @property
    def raw_bytes(self) -> int:
        return sum(g.raw_bytes for g in self.groups)

    @property
    def padding_bytes(self) -> int:
        return self.total_bytes - self.data_offset - self.raw_bytes

    def group_index(self, layer_row: int, expert: int) -> int:
        return layer_row * self.n_experts + expert


def build_plan(
    model: GGUFModel,
    order: Optional[Dict[int, Sequence[int]]] = None,
    alignment: int = DEFAULT_ALIGNMENT,
) -> StorePlan:
    """Decide il layout del file.

    `order[layer]` e' una sequenza di id di expert che dice in quale ordine
    disporli fisicamente; la tabella dell'indice assorbe la differenza, quindi
    e' una scelta di prestazioni senza conseguenze sulla correttezza. Assente,
    si usa l'ordine naturale.
    """
    problemi = check_model(model)
    if problemi:
        raise StoreError("modello non adatto:\n  - " + "\n  - ".join(problemi))
    if alignment <= 0 or alignment & (alignment - 1):
        raise StoreError(f"allineamento {alignment}: deve essere una potenza di due")

    per_layer = expert_tensors(model)
    layers = sorted(per_layer)
    n_exp = model.expert_count

    tabella_bytes = len(layers) * 4 + len(layers) * n_exp * ENTRY_BYTES
    data_offset = HEADER_BYTES + tabella_bytes
    resto = data_offset % alignment
    if resto:
        data_offset += alignment - resto

    groups: List[Group] = []
    cursore = data_offset
    for layer in layers:
        per_kind = per_layer[layer]
        # ordine fisico: quello richiesto, completato con gli expert mancanti
        richiesto = list(order.get(layer, ())) if order else []
        visti = set()
        fisico: List[int] = []
        for e in richiesto:
            if 0 <= e < n_exp and e not in visti:
                fisico.append(e)
                visti.add(e)
        fisico.extend(e for e in range(n_exp) if e not in visti)

        # i gruppi vanno scritti nell'ordine fisico, ma l'indice resta per id
        posizioni: Dict[int, Group] = {}
        for expert in fisico:
            slabs = tuple(
                Slab(per_kind[k], expert, per_kind[k].n_bytes // n_exp) for k in KINDS
            )
            g = Group(layer=layer, expert=expert, slabs=slabs)  # type: ignore[arg-type]
            offs = []
            for s in slabs:
                offs.append(cursore)
                cursore += _arrotonda(s.n_bytes, alignment)
            g.offsets = (offs[0], offs[1], offs[2])
            posizioni[expert] = g
        # nella lista teniamo l'ordine per id, cosi' l'indice e' calcolabile
        groups.extend(posizioni[e] for e in range(n_exp))

    return StorePlan(
        layers=layers,
        n_experts=n_exp,
        alignment=alignment,
        groups=groups,
        data_offset=data_offset,
        total_bytes=cursore,
        model_path=str(model.path),
        model_bytes=Path(model.path).stat().st_size if Path(model.path).exists() else 0,
    )


def estimate_bytes(model: GGUFModel, alignment: int = DEFAULT_ALIGNMENT) -> int:
    """Spazio su disco richiesto, senza costruire il piano completo."""
    per_layer = expert_tensors(model)
    n_exp = model.expert_count or 1
    tot = HEADER_BYTES + len(per_layer) * (4 + n_exp * ENTRY_BYTES)
    resto = tot % alignment  # l'area dati comincia allineata, come in build_plan
    if resto:
        tot += alignment - resto
    for per_kind in per_layer.values():
        gruppo = sum(_arrotonda(per_kind[k].n_bytes // n_exp, alignment) for k in KINDS)
        tot += n_exp * gruppo
    return tot


# -------------------------------------------------------------- scrittura ----


def _pack_header(plan: StorePlan) -> bytes:
    h = struct.pack(
        "<IIIIIIII QQQQ Q",
        MAGIC,
        VERSION,
        HEADER_BYTES,
        plan.alignment,
        len(plan.layers),
        plan.n_experts,
        len(KINDS),
        FLAG_ALIGNED,
        plan.data_offset,
        plan.total_bytes,
        HEADER_BYTES,  # offset della tabella layer
        HEADER_BYTES + len(plan.layers) * 4,  # offset della tabella gruppi
        plan.model_bytes,
    )
    if len(h) > HEADER_BYTES:
        raise StoreError("intestazione troppo grande")
    return h + b"\0" * (HEADER_BYTES - len(h))


def write_store(
    model: GGUFModel,
    plan: StorePlan,
    out_path: str | Path,
    progress: Optional[Callable[[str], None]] = None,
    buffer_bytes: int = 8 << 20,
) -> Dict[str, object]:
    """Scrive il file, leggendo le slice dal GGUF una per una.

    Non viene mai caricato in memoria piu' di un gruppo: il file di uscita puo'
    essere grande decine di gigabyte su una macchina da 16.
    """
    dst = Path(out_path)
    src = Path(model.path)
    if model.shards and len(model.shards) > 1:
        raise StoreError(
            "modello splittato in piu' file: l'expert store non lo supporta ancora"
        )
    if dst.exists() and dst.resolve() == src.resolve():
        raise StoreError("la destinazione coincide con il modello")
    dst.parent.mkdir(parents=True, exist_ok=True)

    scritti = 0
    digest = hashlib.blake2b(digest_size=16)
    with open(src, "rb", buffering=0) as fin, open(dst, "wb", buffering=buffer_bytes) as fout:
        fout.write(_pack_header(plan))
        for layer in plan.layers:
            fout.write(struct.pack("<I", layer))
        for g in plan.groups:
            a, b, c = (s.n_bytes for s in g.slabs)
            fout.write(struct.pack("<QQQIII", *g.offsets, a, b, c))
        # riempimento fino all'area dati
        qui = fout.tell()
        if qui > plan.data_offset:
            raise StoreError("tabelle piu' grandi del previsto")
        fout.write(b"\0" * (plan.data_offset - qui))

        # i gruppi vanno scritti in ordine di offset, non di id
        in_ordine = sorted(plan.groups, key=lambda g: g.offset)
        ultimo_layer = -1
        for g in in_ordine:
            for s, atteso in zip(g.slabs, g.offsets):
                if fout.tell() != atteso:
                    raise StoreError(
                        f"disallineamento a layer {g.layer} expert {g.expert} "
                        f"({s.tensor.name}): sono a {fout.tell()}, attendevo {atteso}"
                    )
                fin.seek(s.src_offset)
                dati = fin.read(s.n_bytes)
                if len(dati) != s.n_bytes:
                    raise StoreError(
                        f"lettura incompleta di {s.tensor.name} expert {s.expert}: "
                        f"{len(dati)} byte su {s.n_bytes}"
                    )
                fout.write(dati)
                digest.update(dati)
                scritti += len(dati)
                coda = _arrotonda(s.n_bytes, plan.alignment) - s.n_bytes
                if coda:
                    fout.write(b"\0" * coda)
            if progress and g.layer != ultimo_layer:
                ultimo_layer = g.layer
                progress(f"layer {g.layer}: {fmt_bytes(scritti)} scritti")
        fout.flush()
        os.fsync(fout.fileno())

    reale = dst.stat().st_size
    if reale != plan.total_bytes:
        raise StoreError(f"dimensione finale {reale} != prevista {plan.total_bytes}")
    return {
        "output": str(dst),
        "bytes": reale,
        "expert_bytes": scritti,
        "padding_bytes": reale - plan.data_offset - scritti,
        "gruppi": len(plan.groups),
        "layer": len(plan.layers),
        "digest": digest.hexdigest(),
    }


# ---------------------------------------------------------------- lettura ----


@dataclass
class StoreEntry:
    """Dove stanno le tre slice di un expert. Ogni offset e' allineato al
    settore, quindi ogni slice si legge con FILE_FLAG_NO_BUFFERING diretta-
    mente nella memoria del suo tensore."""

    offsets: Tuple[int, int, int]
    kind_bytes: Tuple[int, int, int]

    @property
    def offset(self) -> int:
        return self.offsets[0]

    @property
    def raw_bytes(self) -> int:
        return sum(self.kind_bytes)

    def kind_offset(self, kind: str) -> int:
        return self.offsets[KINDS.index(kind)]

    def kind_size(self, kind: str) -> int:
        return self.kind_bytes[KINDS.index(kind)]

    @property
    def size(self) -> int:
        """Byte che servono per avere l'expert completo (senza le code)."""
        return self.raw_bytes


@dataclass
class StoreIndex:
    path: str
    alignment: int
    layers: List[int]
    n_experts: int
    data_offset: int
    total_bytes: int
    model_bytes: int
    entries: List[StoreEntry] = field(default_factory=list)
    _row: Dict[int, int] = field(default_factory=dict, repr=False)

    def row_of(self, layer: int) -> int:
        return self._row[layer]

    def entry(self, layer: int, expert: int) -> StoreEntry:
        if layer not in self._row:
            raise StoreError(f"layer {layer} non presente nello store")
        if not 0 <= expert < self.n_experts:
            raise StoreError(f"expert {expert} fuori intervallo (0..{self.n_experts - 1})")
        return self.entries[self._row[layer] * self.n_experts + expert]

    @property
    def group_bytes(self) -> int:
        """Byte di un expert completo: la grandezza con cui si ragiona sul
        traffico per token. Le letture vere sono tre, una per slice."""
        return self.entries[0].raw_bytes if self.entries else 0

    @property
    def slice_bytes(self) -> Tuple[int, int, int]:
        """Dimensione delle tre letture."""
        return self.entries[0].kind_bytes if self.entries else (0, 0, 0)


def read_index(path: str | Path) -> StoreIndex:
    p = Path(path)
    with open(p, "rb") as fh:
        head = fh.read(HEADER_BYTES)
        if len(head) < HEADER_BYTES:
            raise StoreError(f"{p.name}: file troppo corto per essere uno store")
        (
            magic,
            version,
            header_bytes,
            alignment,
            n_layers,
            n_experts,
            n_kinds,
            _flags,
            data_offset,
            total_bytes,
            layer_table,
            group_table,
            model_bytes,
        ) = struct.unpack_from("<IIIIIIII QQQQ Q", head, 0)
        if magic != MAGIC:
            raise StoreError(f"{p.name}: non e' un expert store (magic {magic:#x})")
        if version != VERSION:
            raise StoreError(
                f"{p.name}: versione {version}, questa build legge la {VERSION}. "
                "Ricostruiscilo con `argine store build`."
            )
        if n_kinds != len(KINDS):
            raise StoreError(f"{p.name}: {n_kinds} tensori per expert, attesi {len(KINDS)}")

        fh.seek(layer_table)
        layers = list(struct.unpack(f"<{n_layers}I", fh.read(n_layers * 4)))
        fh.seek(group_table)
        n = n_layers * n_experts
        grezzo = fh.read(n * ENTRY_BYTES)
        if len(grezzo) != n * ENTRY_BYTES:
            raise StoreError(f"{p.name}: tabella dei gruppi troncata")

    entries = [
        StoreEntry((og, ou, od), (a, b, c))
        for og, ou, od, a, b, c in struct.iter_unpack("<QQQIII", grezzo)
    ]
    idx = StoreIndex(
        path=str(p),
        alignment=alignment,
        layers=layers,
        n_experts=n_experts,
        data_offset=data_offset,
        total_bytes=total_bytes,
        model_bytes=model_bytes,
        entries=entries,
    )
    idx._row = {l: i for i, l in enumerate(layers)}
    reale = p.stat().st_size
    if reale != total_bytes:
        raise StoreError(
            f"{p.name}: dimensione {fmt_bytes(reale)} != quella dichiarata "
            f"{fmt_bytes(total_bytes)}: file incompleto o troncato"
        )
    return idx


def read_slice(path: str | Path, idx: StoreIndex, layer: int, expert: int, kind: str) -> bytes:
    e = idx.entry(layer, expert)
    with open(path, "rb", buffering=0) as fh:
        fh.seek(e.kind_offset(kind))
        return fh.read(e.kind_size(kind))


# -------------------------------------------------------------- verifica -----


def coppie_di_verifica(
    layers: Sequence[int],
    n_experts: int,
    samples: int = 24,
    seed: int = 0,
) -> List[Tuple[int, int]]:
    """Quali (layer, expert) confrontare col GGUF.

    **Un gruppo per ogni layer, sempre**, piu' `samples` gruppi a caso, piu' i due
    estremi del file.

    La copertura di ogni layer non e' prudenza, e' la forma dell'errore che si
    vuole prendere: l'indice dello store e' per layer (una riga per layer, poi gli
    offset dei suoi expert), quindi uno sbaglio nella tabella dei layer rovina un
    layer intero mentre uno sbaglio negli offset rovina un gruppo. Il primo tipo
    si prende con certezza visitando tutti i layer; il secondo e' quello che i
    campioni casuali cercano.

    La versione precedente pescava `samples` layer a caso, e su 48 layer con 24
    campioni ne lasciava intatti circa 28: un layer scritto nel posto sbagliato
    passava la verifica piu' di una volta su due. Il difetto si vedeva solo
    guardando il rapporto fra i numeri — "26 gruppi verificati" su 24576 — che su
    un modello piccolo sembrava ragionevole e su uno grande no, pur essendo lo
    stesso campione di prima: il conteggio era fisso e la copertura scendeva al
    crescere del modello.
    """
    rnd = random.Random(seed)
    n_exp = max(1, int(n_experts))
    coppie: List[Tuple[int, int]] = [(l, rnd.randrange(n_exp)) for l in layers]
    coppie += [(rnd.choice(list(layers)), rnd.randrange(n_exp))
               for _ in range(max(0, samples))]
    # gli estremi non vanno mai saltati: primo e ultimo gruppo del file
    if layers:
        coppie += [(layers[0], 0), (layers[-1], n_exp - 1)]
    # senza duplicati, ma in ordine stabile: la verifica deve essere riproducibile
    return list(dict.fromkeys(coppie))


def verify(
    model: GGUFModel,
    store_path: str | Path,
    samples: int = 24,
    seed: int = 0,
    progress: Optional[Callable[[str], None]] = None,
) -> List[str]:
    """Confronta byte per byte alcune slice dello store con il GGUF.

    Questo file diventa la sorgente dei pesi: un byte sbagliato non fa rumore,
    degrada il modello in silenzio. Per questo la verifica confronta i byte
    veri e non una dimensione o un checksum del solo store.
    """
    problemi: List[str] = []
    try:
        idx = read_index(store_path)
    except StoreError as exc:
        return [str(exc)]

    per_layer = expert_tensors(model)
    if set(idx.layers) != set(per_layer):
        problemi.append(
            f"lo store copre {len(idx.layers)} layer, il modello ne ha {len(per_layer)}"
        )
    if idx.n_experts != model.expert_count:
        problemi.append(
            f"lo store dichiara {idx.n_experts} expert, il modello {model.expert_count}"
        )
    atteso = Path(model.path).stat().st_size
    if idx.model_bytes and idx.model_bytes != atteso:
        problemi.append(
            f"lo store e' stato costruito da un file di {fmt_bytes(idx.model_bytes)}, "
            f"questo modello ne misura {fmt_bytes(atteso)}: non sono lo stesso modello"
        )
    if problemi:
        return problemi

    coppie = coppie_di_verifica(sorted(per_layer), model.expert_count, samples, seed)

    with open(model.path, "rb", buffering=0) as fm, open(store_path, "rb", buffering=0) as fs:
        for n, (layer, expert) in enumerate(coppie, 1):  # noqa: B007
            e = idx.entry(layer, expert)
            per_kind = per_layer[layer]
            for kind in KINDS:
                t = per_kind[kind]
                dim = t.n_bytes // model.expert_count
                if dim != e.kind_size(kind):
                    problemi.append(
                        f"layer {layer} expert {expert} {kind}: lo store dichiara "
                        f"{e.kind_size(kind)} byte, il modello {dim}"
                    )
                    continue
                if e.kind_offset(kind) % idx.alignment:
                    problemi.append(
                        f"layer {layer} expert {expert} {kind}: offset "
                        f"{e.kind_offset(kind)} non allineato a {idx.alignment}: "
                        "FILE_FLAG_NO_BUFFERING lo rifiuterebbe"
                    )
                fm.seek(t.abs_offset + expert * dim)
                a = fm.read(dim)
                fs.seek(e.kind_offset(kind))
                b = fs.read(dim)
                if a != b:
                    dove = next(
                        (i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))
                    )
                    problemi.append(
                        f"layer {layer} expert {expert} {kind}: i byte differiscono "
                        f"dall'offset {dove} ({len(a)} byte confrontati)"
                    )
            if progress and n % 8 == 0:
                progress(f"verificati {n} gruppi su {len(coppie)}")
    return problemi


# ------------------------------------------------------------- confronto -----
#
# Prima di scrivere il lettore in C++ conviene sapere quanto rende il formato.
# Le tre strade, sullo stesso disco e sullo stesso insieme di expert:
#
#   mmap    come oggi: le slice arrivano per page fault dal GGUF
#   gguf    letture esplicite ma non allineate, tre per expert, dal GGUF
#   store   una lettura esplicita, allineata, senza cache, dallo store
#
# Tutte con la barriera per layer, perche' il motore non puo' calcolare prima di
# avere l'ultima slice.


@dataclass
class PathResult:
    label: str
    bytes_read: int
    seconds: float
    experts: int
    threads: int
    reads: int
    bytes_physical: Optional[int] = None
    error: Optional[str] = None
    cached: bool = True
    """True se la cache dell'OS poteva servire le letture.

    Esiste perche' confrontare gli expert al secondo di un percorso bufferizzato
    con quelli di un percorso a cache scavalcata e' un errore: il primo puo'
    pescare dalla RAM. La prima versione di questo confronto lo faceva e ha
    concluso il falso. Ora il verdetto rifiuta di paragonare percorsi con
    esposizione diversa alla cache.
    """

    @property
    def mb_s(self) -> float:
        return self.bytes_read / self.seconds / 1e6 if self.seconds > 0 else 0.0

    @property
    def phys_mb_s(self) -> Optional[float]:
        if self.bytes_physical is None or self.seconds <= 0:
            return None
        return self.bytes_physical / self.seconds / 1e6

    @property
    def cache_ratio(self) -> Optional[float]:
        if self.bytes_physical is None or self.bytes_read <= 0:
            return None
        return max(0.0, min(1.0, (self.bytes_read - self.bytes_physical) / self.bytes_read))

    @property
    def experts_s(self) -> float:
        return self.experts / self.seconds if self.seconds > 0 else 0.0

    @property
    def us_per_expert(self) -> float:
        if self.experts <= 0:
            return 0.0
        return self.seconds * self.threads / self.experts * 1e6


def bench_paths(
    model: GGUFModel,
    store_path: Optional[str | Path],
    threads: int = 8,
    seconds: float = 4.0,
    seed: int = 5,
    purge: int = 0,
) -> List[PathResult]:
    """Misura quanti expert al secondo rende ciascuna strada.

    Quattro percorsi, scelti per scomporre due effetti che nella prima versione
    di questa misura erano confusi fra loro:

      mmap    page fault dal GGUF            3 letture disallineate, cache si'
      gguf    letture esplicite dal GGUF      3 letture disallineate, cache si'
      storeb  letture esplicite dallo store   3 letture allineate,    cache si'
      store   come storeb                     3 letture allineate,    cache no

    Tre letture su ogni strada, cosi' ogni coppia adiacente differisce per una
    cosa sola: `mmap`->`gguf` il page fault, `gguf`->`storeb` l'allineamento,
    `storeb`->`store` la cache dell'OS. Confrontare `gguf` con `store`
    direttamente non misura nulla, perche' cambiano due cose insieme.
    """
    import ctypes
    import threading as _th

    from . import diskbench as db

    per_layer = expert_tensors(model)
    layers = sorted(per_layer)
    n_exp = model.expert_count
    if not layers or n_exp <= 0:
        return []
    slab = {
        l: tuple(per_layer[l][k].n_bytes // n_exp for k in KINDS) for l in layers
    }
    max_slab = max(max(v) for v in slab.values())
    out: List[PathResult] = []

    def coppie(i: int):
        rnd = random.Random(seed * 1000 + i)

        def nxt():
            return rnd.choice(layers), rnd.randrange(n_exp)

        return nxt

    def svuota() -> None:
        if purge > 0:
            from . import diskbench as _db

            _db.purge_cache(purge)

    # --- strada 1: page fault dal GGUF -----------------------------------
    svuota()
    try:
        mapping = db._Mapping(Path(model.path))
        memmove = ctypes.memmove
        try:
            bufs, addrs = [], []
            for _ in range(threads):
                keep, addr = db._alloc_aligned(max_slab)
                bufs.append(keep)
                addrs.append(addr)
            barriera = _th.Barrier(threads)

            def corpo_mmap(i, alive):
                nxt = coppie(i)
                dst = addrs[i]
                by = ops = exp = 0
                while alive():
                    layer, expert = nxt()
                    for k, kind in enumerate(KINDS):
                        t = per_layer[layer][kind]
                        dim = slab[layer][k]
                        src = mapping.base + t.abs_offset + expert * dim
                        memmove(ctypes.c_void_p(dst), ctypes.c_void_p(src), dim)
                        by += dim
                        ops += 1
                    exp += 1
                    try:
                        barriera.wait(timeout=1.0)
                    except Exception:
                        break
                barriera.abort()
                return by, ops, exp

            r = _run3(threads, seconds, corpo_mmap)
            out.append(PathResult("mmap", r[0], r[3], r[2], threads, r[1], r[4],
                                  cached=True))
        finally:
            mapping.close()
    except Exception as exc:  # noqa: BLE001
        out.append(PathResult("mmap", 0, 0.0, 0, threads, 0, error=str(exc)))

    # --- strada 2: letture esplicite dal GGUF, non allineate -------------
    svuota()
    out.append(
        _bench_explicit(
            "gguf",
            Path(model.path),
            threads,
            seconds,
            direct=False,
            buffer_bytes=max_slab,
            plan=lambda i: _piano_gguf(per_layer, slab, coppie(i)),
        )
    )

    # --- strade 3 e 4: una lettura allineata dallo store ------------------
    if store_path and Path(store_path).exists():
        try:
            idx = read_index(store_path)
            gruppo_max = max(e.size for e in idx.entries)
            for etichetta, diretto in (("storeb", False), ("store", True)):
                svuota()
                out.append(
                    _bench_explicit(
                        etichetta,
                        Path(store_path),
                        threads,
                        seconds,
                        direct=diretto,
                        buffer_bytes=gruppo_max,
                        plan=lambda i: _piano_store(idx, coppie(i)),
                    )
                )
        except StoreError as exc:
            out.append(PathResult("store", 0, 0.0, 0, threads, 0, error=str(exc)))
    return out


def _piano_gguf(per_layer, slab, nxt):
    """Genera le tre letture (offset, dimensione) di un expert nel GGUF."""

    def passo():
        layer, expert = nxt()
        return [
            (per_layer[layer][kind].abs_offset + expert * slab[layer][k], slab[layer][k])
            for k, kind in enumerate(KINDS)
        ]

    return passo


def _piano_store(idx: StoreIndex, nxt):
    """Tre letture allineate per expert, una per tensore.

    Non una sola: ggml vuole un blocco contiguo per tensore, e la misura ha
    detto che a coda 8 una lettura grande non rende piu' di tre piccole. Cosi'
    ogni lettura atterra dove serve, senza copie.
    """

    def passo():
        layer, expert = nxt()
        e = idx.entry(layer, expert)
        return [(e.offsets[i], _arrotonda(e.kind_bytes[i], idx.alignment))
                for i in range(len(KINDS))]

    return passo


def _run3(threads: int, seconds: float, corpo):
    """Come diskbench._run_timed ma con tre contatori e i byte di sistema."""
    import threading as _th
    import time as _time

    from . import diskbench as db

    stop = _th.Event()
    esiti = [(0, 0, 0)] * threads
    errori: List[BaseException] = []
    pronti = _th.Barrier(threads + 1)

    def alive():
        return not stop.is_set()

    def worker(i):
        try:
            pronti.wait()
            esiti[i] = corpo(i, alive)
        except BaseException as exc:  # noqa: BLE001
            errori.append(exc)
            stop.set()
            try:
                pronti.wait(0.1)
            except Exception:
                pass

    ts = [_th.Thread(target=worker, args=(i,), daemon=True) for i in range(threads)]
    for t in ts:
        t.start()
    pronti.wait()
    prima = db._sys_read_bytes()
    t0 = _time.perf_counter()
    stop.wait(seconds)
    stop.set()
    for t in ts:
        t.join(timeout=30)
    trascorso = _time.perf_counter() - t0
    dopo = db._sys_read_bytes()
    if errori:
        raise errori[0]
    fisici = None if prima is None or dopo is None else max(0, dopo - prima)
    return (
        sum(a for a, _b, _c in esiti),
        sum(b for _a, b, _c in esiti),
        sum(c for _a, _b, c in esiti),
        trascorso,
        fisici,
    )


def _bench_explicit(
    label: str,
    path: Path,
    threads: int,
    seconds: float,
    direct: bool,
    buffer_bytes: int,
    plan,
) -> PathResult:
    import threading as _th

    from . import diskbench as db

    reader = db._PositionedReader(path, direct=direct)
    workers = []
    bufs = []
    try:
        for _ in range(threads):
            leggi, chiudi = reader.make_worker()
            keep, addr = db._alloc_aligned(buffer_bytes)
            workers.append((leggi, chiudi, addr))
            bufs.append(keep)
        barriera = _th.Barrier(threads)

        def corpo(i, alive):
            leggi, _c, addr = workers[i]
            passo = plan(i)
            by = ops = exp = 0
            while alive():
                for off, dim in passo():
                    by += leggi(off, dim, addr)
                    ops += 1
                exp += 1
                try:
                    barriera.wait(timeout=1.0)
                except Exception:
                    break
            barriera.abort()
            return by, ops, exp

        r = _run3(threads, seconds, corpo)
        return PathResult(label, r[0], r[3], r[2], threads, r[1], r[4],
                          cached=not direct)
    except Exception as exc:  # noqa: BLE001
        return PathResult(label, 0, 0.0, 0, threads, 0, error=str(exc))
    finally:
        for _l, chiudi, _a in workers:
            try:
                chiudi()
            except Exception:
                pass


def format_paths(results: Sequence[PathResult]) -> List[str]:
    righe = [
        f"  {'strada':<7} {'MB/s':>7} {'disco':>7} {'cache':>6} {'expert/s':>9} "
        f"{'us/expert':>10} {'letture/exp':>11}"
    ]
    for r in results:
        if r.error:
            righe.append(f"  {r.label:<7} {r.error.splitlines()[0][:56]}")
            continue
        fis = f"{r.phys_mb_s:.0f}" if r.phys_mb_s is not None else "-"
        quota = f"{r.cache_ratio * 100:.0f}%" if r.cache_ratio is not None else "-"
        per_exp = r.reads / r.experts if r.experts else 0.0
        righe.append(
            f"  {r.label:<7} {r.mb_s:>7.0f} {fis:>7} {quota:>6} {r.experts_s:>9.0f} "
            f"{r.us_per_expert:>10.0f} {per_exp:>11.2f}"
        )
    return righe


# byte letti da SSD per token generato, misurati in produzione su Qwen3-30B-A3B
# Q4_K_M (argine benchmark: 76.37 MiB/token) e throughput del percorso mmap
# nella stessa misura
PRODUZIONE_MIB_TOKEN = 76.37
PRODUZIONE_MB_S = 562.0


def paths_verdict(results: Sequence[PathResult],
                  group_bytes: int = 0) -> List[str]:
    """Confronta solo cio' che e' confrontabile.

    La prima versione di questa funzione metteva a confronto gli expert al
    secondo di un percorso bufferizzato con quelli di un percorso a cache
    scavalcata, e da 2334 contro 1420 concluse che lo store non paga. Era falso:
    il primo percorso pescava il 70% dei byte dalla RAM. Da qui in avanti i
    confronti avvengono solo fra percorsi con la stessa esposizione alla cache,
    e il throughput di disco viene riportato a parte.
    """
    per = {r.label: r for r in results if not r.error and r.experts_s > 0}
    out: List[str] = []
    mm, gg, sb, st = (per.get(k) for k in ("mmap", "gguf", "storeb", "store"))

    # 1. lo stesso file, la stessa cache: cambia solo chi va a prendere i byte
    if mm and gg:
        out.append(
            f"letture esplicite invece di page fault, a parita' di file e di cache: "
            f"{gg.experts_s / mm.experts_s:.1f}x "
            f"({mm.experts_s:.0f} -> {gg.experts_s:.0f} expert/s)"
        )

    # 2. stessa cache, stesso numero di letture: cambia solo l'allineamento
    if gg and sb:
        r = sb.experts_s / gg.experts_s
        out.append(
            f"offset allineati invece di disallineati, a parita' di cache e di numero "
            f"di letture: {r:.2f}x ({gg.experts_s:.0f} -> {sb.experts_s:.0f} expert/s)"
        )
        if r < 1.05:
            out.append(
                "l'allineamento da solo non da' velocita': serve perche' senza di esso "
                "FILE_FLAG_NO_BUFFERING non e' utilizzabile, e senza quello non esiste "
                "un budget di RAM rigido"
            )
    elif gg and st:
        out.append(
            "il confronto fra GGUF e store non e' disponibile a parita' di cache: "
            "manca la prova 'storeb'. Senza quella, la differenza fra "
            f"{gg.experts_s:.0f} e {st.experts_s:.0f} expert/s mescola il layout e la "
            "cache, e non misura ne' l'uno ne' l'altra"
        )

    # 3. lo stesso layout, cache scavalcata: quanto costa il budget rigido
    if sb and st:
        r = st.experts_s / sb.experts_s
        if r >= 0.95:
            out.append(
                f"scavalcare la cache dell'OS non costa nulla ({r:.2f}x): il budget "
                "rigido e' gratis"
            )
        else:
            out.append(
                f"scavalcare la cache dell'OS costa il {(1 - r) * 100:.0f}% degli expert "
                f"al secondo ({sb.experts_s:.0f} -> {st.experts_s:.0f}). E' il prezzo del "
                "budget rigido: in cambio la RAM non viene contata due volte e il tetto "
                "lo decidiamo noi invece del memory manager"
            )

    # 4. il numero che conta: byte davvero strappati al disco
    diretti = [r for r in results if not r.error and not r.cached and r.phys_mb_s]
    if diretti:
        best = max(diretti, key=lambda r: r.phys_mb_s or 0.0)
        out.append(
            f"throughput di disco vero (cache scavalcata, niente da interpretare): "
            f"{best.phys_mb_s:.0f} MB/s, cioe' "
            f"{(best.phys_mb_s or 0) / PRODUZIONE_MB_S:.1f}x i {PRODUZIONE_MB_S:.0f} MB/s "
            "che la generazione ottiene oggi"
        )

    # 5. proiezione sul carico vero, non sul caso pessimo
    if st and group_bytes > 0:
        gruppi = PRODUZIONE_MIB_TOKEN * (1 << 20) / group_bytes
        ms = gruppi / st.experts_s * 1000
        out.append(
            f"in produzione dal disco arrivano {PRODUZIONE_MIB_TOKEN:.0f} MiB per token, "
            f"cioe' {gruppi:.0f} gruppi da {fmt_bytes(group_bytes)}, non 384: a "
            f"{st.experts_s:.0f} gruppi/s sono {ms:.0f} ms di I/O per token contro i "
            f"{PRODUZIONE_MIB_TOKEN * (1 << 20) / (PRODUZIONE_MB_S * 1e6) * 1000:.0f} ms "
            "di oggi"
        )
        out.append(
            "resta il calcolo, che su questa macchina non scende sotto ~49 ms per token "
            "(1.88 GiB di pesi attivi a 38.6 GiB/s di banda misurata): la proiezione "
            f"onesta e' {1000 / (ms + 49):.0f} tok/s, non {1000 / ms:.0f}"
        )
    elif st:
        out.append(
            "proiezione sul carico vero non calcolata: manca la dimensione del gruppo"
        )
    return out
