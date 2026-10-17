"""Riordino degli expert dentro il GGUF, per località di lettura su SSD.

Idea: quasi tutti trattano il file del modello come immutabile. Non lo e'. Se gli
expert che si attivano insieme sono contigui su disco, un token che oggi richiede
decine di letture sparse ne richiede poche e grandi — e su NVMe la differenza fra
letture casuali piccole e sequenziali grandi e' di un fattore 5-50.

Correttezza: permutare gli expert e' una pura RIETICHETTATURA, esatta e senza
perdita, **a patto** di applicare la stessa permutazione anche alle righe del
router (`ffn_gate_inp.weight`). Se l'expert 7 diventa il 2, anche la riga 7 dei
pesi del router deve diventare la 2, altrimenti il modello instrada verso pesi
sbagliati. Questo modulo lo garantisce e lo verifica.

Nota utile: la permutazione avviene DENTRO ciascun tensore, quindi dimensioni e
offset dei tensori non cambiano. L'intestazione del GGUF resta identica byte per
byte e si riscrive solo l'area dati.
"""

from __future__ import annotations

import os
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .gguf import GGUFModel, TensorInfo, read_gguf
from .util import MB, fmt_bytes

_LINE_RE = re.compile(r"^(\S+)\s+(\d+)\s+(\d+)((?:\s+\d+)*)\s*$")


@dataclass
class LayerTrace:
    layer: int
    tokens: List[Tuple[int, ...]] = field(default_factory=list)  # expert scelti per token
    n_expert: int = 0

    @property
    def usage(self) -> Counter:
        c: Counter = Counter()
        for t in self.tokens:
            c.update(t)
        return c


@dataclass
class Trace:
    layers: Dict[int, LayerTrace] = field(default_factory=dict)

    @property
    def n_tokens(self) -> int:
        return sum(len(l.tokens) for l in self.layers.values())

    def summary(self) -> Dict[str, object]:
        usage: Counter = Counter()
        for l in self.layers.values():
            usage.update(l.usage)
        tot = sum(usage.values()) or 1
        top = usage.most_common(5)
        return {
            "layers": len(self.layers),
            "osservazioni": self.n_tokens,
            "expert_visti": len(usage),
            "quota_top5": round(sum(c for _e, c in top) / tot, 4),
        }


def parse_trace(path: str | Path) -> Trace:
    """Legge il log prodotto da ARGINE_ROUTE_LOG.

    Formato: <tensore> <n_token> <n_usati> <id>...
    I tre tensori di uno stesso layer (gate/up/down) riportano gli stessi id:
    ne teniamo uno solo per non contare tre volte.
    """
    tr = Trace()
    visti: set[Tuple[int, int]] = set()
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for riga, line in enumerate(fh):
            m = _LINE_RE.match(line.strip())
            if not m:
                continue
            nome, n_tok, n_used = m.group(1), int(m.group(2)), int(m.group(3))
            ids = [int(x) for x in m.group(4).split()] if m.group(4).strip() else []
            if len(ids) != n_tok * n_used or n_used <= 0:
                continue
            lm = re.search(r"blk\.(\d+)\.", nome)
            if not lm:
                continue
            layer = int(lm.group(1))
            # una sola voce per layer fra gate/up/down
            if "ffn_gate_exps" not in nome:
                continue
            chiave = (layer, riga)
            if chiave in visti:
                continue
            visti.add(chiave)
            lt = tr.layers.setdefault(layer, LayerTrace(layer=layer))
            for t in range(n_tok):
                gruppo = tuple(ids[t * n_used:(t + 1) * n_used])
                lt.tokens.append(gruppo)
                lt.n_expert = max(lt.n_expert, max(gruppo) + 1)
    return tr


# ------------------------------------------------------------- ordinamento ---

def cooccurrence(lt: LayerTrace, n_expert: int) -> List[Counter]:
    """co[i][j] = quante volte gli expert i e j sono stati scelti dallo stesso token."""
    co: List[Counter] = [Counter() for _ in range(n_expert)]
    for gruppo in lt.tokens:
        for a in gruppo:
            for b in gruppo:
                if a != b and a < n_expert and b < n_expert:
                    co[a][b] += 1
    return co


def order_experts(lt: LayerTrace, n_expert: int) -> List[int]:
    """Nuovo ordine degli expert: `perm[nuova_posizione] = vecchio_id`.

    Catena greedy che parte dall'expert piu' usato e aggiunge ogni volta quello
    con la co-occorrenza piu' alta rispetto all'ultimo piazzato, usando la
    frequenza come criterio di spareggio. Ne risulta che gli expert caldi
    finiscono vicini fra loro e all'inizio del tensore: la regione che resta in
    page cache e' contigua.
    """
    uso = lt.usage
    if not lt.tokens:
        return list(range(n_expert))
    co = cooccurrence(lt, n_expert)
    restanti = set(range(n_expert))
    ordine: List[int] = []
    corrente = max(restanti, key=lambda e: (uso.get(e, 0), -e))
    while True:
        ordine.append(corrente)
        restanti.discard(corrente)
        if not restanti:
            break
        vicini = co[corrente]
        corrente = max(
            restanti,
            key=lambda e: (vicini.get(e, 0), uso.get(e, 0), -e),
        )
    return ordine


def runs_per_token(lt: LayerTrace, perm: Sequence[int]) -> float:
    """Numero medio di blocchi contigui richiesti da un token.

    E' la metrica che conta davvero: ogni blocco contiguo e' una lettura da
    disco. Meno blocchi = meno richieste, piu' grandi.
    """
    if not lt.tokens:
        return 0.0
    pos = {vecchio: nuovo for nuovo, vecchio in enumerate(perm)}
    totale = 0
    for gruppo in lt.tokens:
        p = sorted(pos.get(e, e) for e in gruppo)
        blocchi = 1
        for a, b in zip(p, p[1:]):
            if b != a + 1:
                blocchi += 1
        totale += blocchi
    return totale / len(lt.tokens)


@dataclass
class ReorderPlan:
    perms: Dict[int, List[int]] = field(default_factory=dict)
    runs_prima: Dict[int, float] = field(default_factory=dict)
    runs_dopo: Dict[int, float] = field(default_factory=dict)
    n_expert: int = 0

    @property
    def media_prima(self) -> float:
        v = list(self.runs_prima.values())
        return sum(v) / len(v) if v else 0.0

    @property
    def media_dopo(self) -> float:
        v = list(self.runs_dopo.values())
        return sum(v) / len(v) if v else 0.0

    @property
    def guadagno(self) -> float:
        if self.media_dopo <= 0:
            return 0.0
        return self.media_prima / self.media_dopo


def plan_reorder(model: GGUFModel, trace: Trace) -> ReorderPlan:
    n_expert = model.expert_count
    p = ReorderPlan(n_expert=n_expert)
    for layer, lt in sorted(trace.layers.items()):
        perm = order_experts(lt, n_expert)
        p.perms[layer] = perm
        p.runs_prima[layer] = runs_per_token(lt, list(range(n_expert)))
        p.runs_dopo[layer] = runs_per_token(lt, perm)
    return p


# -------------------------------------------------------------- riscrittura --

_EXPS_RE = re.compile(r"\.ffn_(gate|up|down)_exps\.weight$")
_ROUTER_RE = re.compile(r"\.ffn_gate_inp\.weight$")
_EXP_BIAS_RE = re.compile(r"\.ffn_(exp_probs_b|gate_inp_shexp)\.")


def _tensori_da_permutare(model: GGUFModel) -> Dict[int, List[TensorInfo]]:
    """Per ogni layer, i tensori la cui ultima dimensione indicizza gli expert."""
    out: Dict[int, List[TensorInfo]] = defaultdict(list)
    for t in model.tensors:
        if t.layer_index is None:
            continue
        if _EXPS_RE.search(t.name) or _ROUTER_RE.search(t.name) or _EXP_BIAS_RE.search(t.name):
            out[t.layer_index].append(t)
    return out


def verify_plan(model: GGUFModel, plan: ReorderPlan) -> List[str]:
    """Controlli di sicurezza prima di toccare un file da gigabyte."""
    problemi: List[str] = []
    if not model.is_moe:
        problemi.append("il modello non e' MoE: niente da riordinare")
        return problemi
    gruppi = _tensori_da_permutare(model)
    for layer, perm in plan.perms.items():
        if sorted(perm) != list(range(model.expert_count)):
            problemi.append(f"layer {layer}: la permutazione non e' completa")
        tensori = gruppi.get(layer, [])
        if not tensori:
            problemi.append(f"layer {layer}: nessun tensore expert trovato")
            continue
        router = [t for t in tensori if _ROUTER_RE.search(t.name)]
        if not router:
            problemi.append(
                f"layer {layer}: manca ffn_gate_inp.weight. Senza permutare anche "
                "il router il modello instraderebbe verso pesi sbagliati: mi fermo."
            )
        for t in tensori:
            ultimo = t.dims[-1] if t.dims else 0
            if ultimo != model.expert_count:
                problemi.append(
                    f"{t.name}: ultima dimensione {ultimo} != {model.expert_count} expert"
                )
    return problemi


def rewrite(model: GGUFModel, plan: ReorderPlan, out_path: str | Path,
            progress=None) -> Dict[str, object]:
    """Scrive una copia del GGUF con gli expert permutati.

    Dimensioni e offset dei tensori non cambiano (permutiamo dentro i tensori),
    quindi l'intestazione resta valida: copiamo il file e riscriviamo solo le
    slice interessate.
    """
    problemi = verify_plan(model, plan)
    if problemi:
        raise ValueError("riordino rifiutato:\n  - " + "\n  - ".join(problemi))

    src = Path(model.path)
    dst = Path(out_path)
    if model.shards and len(model.shards) > 1:
        raise ValueError("riordino di modelli splittati non ancora supportato")
    if dst.resolve() == src.resolve():
        raise ValueError("scegli un file di destinazione diverso dall'originale")

    dst.parent.mkdir(parents=True, exist_ok=True)
    if progress:
        progress(f"copia di {fmt_bytes(src.stat().st_size)}...")
    shutil.copy2(src, dst)

    gruppi = _tensori_da_permutare(model)
    tensori_scritti = 0
    byte_scritti = 0

    with open(dst, "r+b") as fh:
        for layer, perm in sorted(plan.perms.items()):
            for t in gruppi.get(layer, []):
                n_exp = t.dims[-1]
                slice_bytes = t.n_bytes // n_exp
                base = t.abs_offset if t.file else model.data_offset + t.offset
                # leggiamo tutte le slice, poi le riscriviamo nell'ordine nuovo
                fh.seek(base)
                originale = fh.read(t.n_bytes)
                if len(originale) != t.n_bytes:
                    raise ValueError(f"{t.name}: lettura incompleta")
                pezzi = [originale[i * slice_bytes:(i + 1) * slice_bytes]
                         for i in range(n_exp)]
                fh.seek(base)
                for vecchio in perm:
                    fh.write(pezzi[vecchio])
                tensori_scritti += 1
                byte_scritti += t.n_bytes
            if progress:
                progress(f"layer {layer}: {tensori_scritti} tensori permutati")
        fh.flush()
        os.fsync(fh.fileno())

    return {
        "output": str(dst),
        "tensori": tensori_scritti,
        "byte_riscritti": byte_scritti,
        "layer": len(plan.perms),
        "runs_prima": round(plan.media_prima, 3),
        "runs_dopo": round(plan.media_dopo, 3),
        "guadagno": round(plan.guadagno, 3),
    }
