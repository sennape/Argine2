"""Quantizzazione degli expert a 2 bit, e la matrice di importanza che la rende
possibile.

Perche' esiste
--------------
Tutto il lavoro sullo store nasce da un vincolo: gli expert di Qwen3-30B-A3B in
Q4_K_M pesano 16,35 GiB e in 16 GB di RAM non ci stanno. Da li' vengono lo
streaming, la cache, il budget rigido, e il tetto misurato di ~4,5 tok/s in
generazione, che e' la velocita' del disco travestita da velocita' del modello.

C'e' un secondo modo di far entrare il modello: rimpicciolirlo. A 2 bit per peso
gli stessi expert pesano ~7,5 GiB, e il modello intero sta in RAM. Lo streaming
sparisce, e con lui il collo di bottiglia. Il prezzo e' qualita', e il punto di
questo modulo e' renderlo un numero invece che una speranza.

Due cose che si scoprono tardi, se non le si mette in un programma
-----------------------------------------------------------------
1. **I tipi IQ2 non si possono produrre senza matrice di importanza.**
   `llama-quantize` non degrada con un avviso: carica il modello, arriva al
   primo tensore, stampa "Missing importance matrix ... bailing out" e muore.
   Su un modello da 18 GB quell'errore arriva dopo qualche minuto di attesa,
   e arriva una seconda volta se nel frattempo non si e' capito perche'.
   Qui il controllo si fa prima di avviare il processo.

2. **Il file che esce si puo' calcolare prima di produrlo.** La domanda vera
   non e' "quanto degrada" ma "ci sta in RAM": se la risposta e' no, la qualita'
   e' irrilevante. Il conto si fa sulla lista dei tensori del GGUF e sui byte
   per blocco di ogni tipo ggml, in un secondo, invece che in un'ora di
   quantizzazione seguita da una delusione.

Sulla doppia quantizzazione
---------------------------
Partire da un Q4_K_M per arrivare a IQ2 significa arrotondare due volte: i pesi
sono gia' passati per una griglia a 4 bit, e la seconda passata approssima
quell'approssimazione. Il risultato e' peggiore di un IQ2 costruito dal modello
originale in BF16. Il modulo lo dice invece di lasciarlo scoprire dal numero di
perplessita': serve a capire se la strada regge, non a produrre il file
definitivo.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .gguf import GGML_TYPES, GGUFModel, TensorInfo
from .util import fmt_bytes


# --------------------------------------------------------------- i tipi ------

@dataclass(frozen=True)
class QuantType:
    nome: str
    blocco: int          # elementi per blocco
    byte: int            # byte per blocco
    imatrix: bool = False  # llama.cpp rifiuta di produrlo senza matrice
    nota: str = ""
    # Nome esatto del tipo dentro ggml. --tensor-type lo confronta byte per
    # byte con ggml_type_name(), e i K-quant hanno la K maiuscola: "q4_K", non
    # "q4_k". Scriverlo minuscolo fa fallire il parsing con un messaggio che
    # parla di tipo sconosciuto e non dice quale grafia si aspettava.
    ggml: str = ""
    # Alcune sigle (IQ2_M, Q4_K_M) sono "file type": mescole di tipi decise da
    # llama.cpp tensore per tensore. Esistono come argomento posizionale, non
    # come tipo di un singolo tensore, e --tensor-type le rifiuta.
    solo_ftype: bool = False

    @property
    def bpw(self) -> float:
        return 8.0 * self.byte / self.blocco

    @property
    def ggml_name(self) -> str:
        return self.ggml or self.nome.lower()


# Byte per blocco presi dalle struct di ggml-common.h, non da tabelle di
# riepilogo trovate in giro: le seconde arrotondano e il conto sulle dimensioni
# finali sbaglia di centinaia di MiB, che su 16 GB decidono se ci sta o no.
TIPI: Dict[str, QuantType] = {
    t.nome: t for t in [
        QuantType("IQ1_S",   256,  50, imatrix=True,
                  nota="1,56 bit: sperimentale, quasi sempre inutilizzabile"),
        QuantType("IQ1_M",   256,  56, imatrix=True,
                  nota="1,75 bit: sperimentale"),
        QuantType("IQ2_XXS", 256,  66, imatrix=True,
                  nota="il piu' piccolo IQ2: massima compressione, qualita' fragile"),
        QuantType("IQ2_XS",  256,  74, imatrix=True),
        QuantType("IQ2_S",   256,  82, imatrix=True),
        QuantType("IQ2_M",   256,  82, imatrix=True, solo_ftype=True,
                  nota="e' una mescola, non un tipo: usa i blocchi di IQ2_S con "
                       "una scelta di tipi diversa tensore per tensore"),
        QuantType("Q2_K",    256,  84, ggml="q2_K",
                  nota="l'unico ~2,6 bit che llama.cpp produce senza imatrix"),
        QuantType("IQ3_XXS", 256,  98),
        QuantType("Q3_K",    256, 110, ggml="q3_K"),
        QuantType("IQ3_S",   256, 110),
        QuantType("IQ4_XS",  256, 136),
        QuantType("Q4_K",    256, 144, ggml="q4_K"),
        QuantType("Q4_0",     32,  18),
        QuantType("IQ4_NL",   32,  18),
        QuantType("Q5_K",    256, 176, ggml="q5_K"),
        QuantType("Q6_K",    256, 210, ggml="q6_K"),
        QuantType("Q8_0",     32,  34),
        QuantType("F16",       1,   2),
        QuantType("BF16",      1,   2),
        QuantType("F32",       1,   4),
    ]
}

# Nome del tipo -> nome della ftype da passare a llama-quantize come argomento
# posizionale. Coincidono quasi sempre; le eccezioni stanno qui.
FTYPE_ALIAS = {"Q4_K": "Q4_K_M", "Q3_K": "Q3_K_M", "Q5_K": "Q5_K_M"}


def tipo(nome: str) -> Optional[QuantType]:
    return TIPI.get(nome.upper())


def tipi_disponibili() -> List[QuantType]:
    return sorted(TIPI.values(), key=lambda t: t.bpw)


# ------------------------------------------------------- stima della taglia ---

# Tensori che llama.cpp non quantizza mai, qualunque cosa si chieda: le norm
# sono vettori di poche migliaia di valori e quantizzarli costa qualita' senza
# far risparmiare niente, il router decide quali expert usare e un errore li'
# sceglie l'expert sbagliato invece di calcolarlo male.
_MAI_QUANTIZZATI = re.compile(r"(_norm\.|_norm_|\bnorm\b|ffn_gate_inp|\.bias$)")


def _e_quantizzabile(t: TensorInfo) -> Tuple[bool, str]:
    if _MAI_QUANTIZZATI.search(t.name):
        return False, "norm/router/bias: llama.cpp lo lascia in F32"
    if len(t.dims) < 2:
        return False, "vettore: niente da quantizzare"
    return True, ""


def _sta_nel_blocco(t: TensorInfo, q: QuantType) -> bool:
    """Una riga che non e' multipla del blocco non puo' usare quel tipo.

    llama.cpp in quel caso ricade su un tipo compatibile invece di fallire, e la
    dimensione finale non e' quella che ci si aspetta. Vale la pena saperlo
    prima.
    """
    if q.blocco <= 1:
        return True
    return t.dims[0] % q.blocco == 0


@dataclass
class VoceStima:
    classe: str
    n_tensori: int = 0
    elementi: int = 0
    byte_prima: int = 0
    byte_dopo: int = 0
    tipo_dopo: str = ""
    note: List[str] = field(default_factory=list)

    @property
    def risparmio(self) -> int:
        return self.byte_prima - self.byte_dopo


@dataclass
class Stima:
    voci: List[VoceStima] = field(default_factory=list)
    avvisi: List[str] = field(default_factory=list)

    @property
    def byte_prima(self) -> int:
        return sum(v.byte_prima for v in self.voci)

    @property
    def byte_dopo(self) -> int:
        return sum(v.byte_dopo for v in self.voci)

    @property
    def byte_expert_dopo(self) -> int:
        return sum(v.byte_dopo for v in self.voci if v.classe == "expert")

    @property
    def risparmio(self) -> int:
        return self.byte_prima - self.byte_dopo

    def voce(self, classe: str) -> Optional[VoceStima]:
        for v in self.voci:
            if v.classe == classe:
                return v
        return None


def _classe(t: TensorInfo) -> str:
    if t.is_expert:
        return "expert"
    n = t.name
    if "token_embd" in n or n.startswith("output.") or n == "output.weight":
        return "embedding/output"
    if not _e_quantizzabile(t)[0]:
        return "norm/router"
    return "attenzione/altro"


def stima(model: GGUFModel, expert_type: str, resto_type: Optional[str] = None,
          embd_type: Optional[str] = None) -> Stima:
    """Quanto pesera' il file, classe per classe, senza produrlo.

    `resto_type=None` significa "lascia i tensori non-expert come sono": e'
    l'intenzione tipica, perche' il risparmio sta tutto negli expert e toccare
    il resto aggiunge errore senza restituire GiB.
    """
    qe = tipo(expert_type)
    if qe is None:
        raise ValueError(f"tipo sconosciuto: {expert_type}")
    qr = tipo(resto_type) if resto_type else None
    qb = tipo(embd_type) if embd_type else None

    st = Stima()
    per_classe: Dict[str, VoceStima] = {}
    fallback: Dict[str, int] = {}

    for t in model.tensors:
        cl = _classe(t)
        v = per_classe.get(cl)
        if v is None:
            v = VoceStima(classe=cl)
            per_classe[cl] = v
        v.n_tensori += 1
        v.elementi += t.n_elements
        v.byte_prima += t.n_bytes

        if cl == "norm/router":
            target = None
        elif cl == "expert":
            target = qe
        elif cl == "embedding/output":
            target = qb or qr
        else:
            target = qr

        if target is None:
            v.byte_dopo += t.n_bytes
            v.tipo_dopo = v.tipo_dopo or "invariato"
            continue

        if not _sta_nel_blocco(t, target):
            fallback[t.name] = t.n_bytes
            v.byte_dopo += t.n_bytes
            v.tipo_dopo = target.nome + " (parziale)"
            continue

        blocchi = (t.n_elements + target.blocco - 1) // target.blocco
        v.byte_dopo += blocchi * target.byte
        v.tipo_dopo = target.nome

    ordine = ["expert", "attenzione/altro", "embedding/output", "norm/router"]
    st.voci = [per_classe[c] for c in ordine if c in per_classe]
    st.voci += [v for c, v in per_classe.items() if c not in ordine]

    if fallback:
        st.avvisi.append(
            f"{len(fallback)} tensori hanno righe non multiple del blocco di "
            f"{qe.nome}: llama.cpp ricadra' su un tipo compatibile e quei "
            f"{fmt_bytes(sum(fallback.values()))} non si comprimeranno")
    if qe.imatrix:
        st.avvisi.append(
            f"{qe.nome} richiede la matrice di importanza: senza, llama-quantize "
            f"si interrompe invece di produrre un file peggiore")
    return st


# ------------------------------------------------ requisiti e precondizioni ---

@dataclass
class Precondizioni:
    ok: bool = True
    errori: List[str] = field(default_factory=list)
    avvisi: List[str] = field(default_factory=list)

    def errore(self, msg: str) -> None:
        self.ok = False
        self.errori.append(msg)


def controlla(model: GGUFModel, expert_type: str, imatrix: Optional[Path],
              resto_type: Optional[str] = None,
              consenti_riquantizzazione: bool = False,
              usa_tensor_type: bool = True) -> Precondizioni:
    """Tutto cio' che farebbe fallire la quantizzazione, verificato prima.

    Il valore di questa funzione e' interamente nel tempo che non si perde:
    ognuno di questi controlli corrisponde a un modo in cui llama-quantize si
    interrompe dopo aver caricato diciotto gigabyte.
    """
    p = Precondizioni()
    qe = tipo(expert_type)
    if qe is None:
        p.errore(f"tipo sconosciuto: {expert_type}")
        return p

    if qe.solo_ftype and usa_tensor_type:
        alt = [t.nome for t in tipi_disponibili()
               if not t.solo_ftype and abs(t.bpw - qe.bpw) < 0.3]
        p.errore(
            f"{qe.nome} non e' un tipo di tensore ma una mescola che llama.cpp "
            f"decide da solo, quindi non si puo' assegnare ai soli expert. "
            f"Usa un tipo vero ({', '.join(alt) or 'IQ2_S'}), oppure "
            f"--no-tensor-type per lasciare a llama.cpp la mescola su tutto "
            f"il modello.")

    if qe.imatrix and not imatrix:
        p.errore(
            f"{qe.nome} non si puo' produrre senza matrice di importanza. "
            f"llama.cpp non degrada con un avviso: si interrompe. "
            f"Genera la matrice con 'argine imatrix', oppure usa Q2_K "
            f"({TIPI['Q2_K'].bpw:.2f} bit) che non la richiede.")

    if imatrix and not Path(imatrix).exists():
        p.errore(f"matrice di importanza non trovata: {imatrix}")

    sorgente = model.quantization or ""
    gia_quantizzato = not any(sorgente.upper().startswith(x)
                              for x in ("F32", "F16", "BF16"))
    if gia_quantizzato and not consenti_riquantizzazione:
        p.errore(
            f"il modello di partenza e' gia' quantizzato ({sorgente}): "
            f"llama-quantize lo rifiuta senza --allow-requantize. "
            f"Aggiungi --allow-requantize se e' voluto.")
    if gia_quantizzato:
        p.avvisi.append(
            f"partenza da {sorgente}: i pesi vengono arrotondati una seconda "
            f"volta, e il risultato sara' peggiore dello stesso {qe.nome} "
            f"costruito dal modello originale in BF16. Va bene per capire se la "
            f"strada regge, non come file definitivo.")
        if resto_type:
            p.avvisi.append(
                "anche i tensori non-expert verranno riquantizzati: il "
                "risparmio li' e' minimo e l'errore si aggiunge comunque. "
                "Lasciali invariati se non c'e' un motivo preciso.")

    if not model.is_moe:
        p.avvisi.append("il modello non e' MoE: quantizzare 'gli expert' non "
                        "ha effetto, tutto passa dal tipo generale")
    return p


# ----------------------------------------------------- invocazione del tool ---

_PATTERN_EXPERT = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")

# Il resto del modello arriva da Q4_K_M e li' deve restare: e' il 10% dei byte e
# quasi tutto cio' che il modello sa fare fuori dagli expert.
FTYPE_NEUTRA = "Q4_K_M"


def supporta_tensor_type(binario: Path, timeout: float = 30.0) -> Optional[bool]:
    """Questa build di llama-quantize accetta --tensor-type?

    None se non si riesce a stabilirlo: meglio di un falso negativo che manda
    l'utente a ricompilare per niente.
    """
    try:
        out = subprocess.run([str(binario), "--help"], capture_output=True,
                             text=True, timeout=timeout, errors="replace")
    except Exception:  # noqa: BLE001
        return None
    testo = (out.stdout or "") + (out.stderr or "")
    if not testo.strip():
        return None
    return "--tensor-type" in testo


def build_args(binario: Path, sorgente: str, destinazione: str,
               expert_type: str, resto_type: Optional[str] = None,
               embd_type: Optional[str] = None,
               imatrix: Optional[str] = None, threads: int = 8,
               consenti_riquantizzazione: bool = False,
               usa_tensor_type: bool = True,
               extra: Optional[Sequence[str]] = None) -> List[str]:
    """Gli argomenti di llama-quantize per una quantizzazione a precisione mista.

    La ftype posizionale governa i tensori non nominati; gli expert vengono
    dirottati uno per uno con --tensor-type. L'ordine conta: llama-quantize
    vuole le opzioni prima dei tre argomenti posizionali.
    """
    qe = tipo(expert_type)
    assert qe is not None
    args: List[str] = [str(binario)]
    if consenti_riquantizzazione:
        args.append("--allow-requantize")
    if imatrix:
        args += ["--imatrix", str(imatrix)]
    if usa_tensor_type:
        for pat in _PATTERN_EXPERT:
            args += ["--tensor-type", f"{pat}={qe.ggml_name}"]
    if embd_type:
        args += ["--token-embedding-type", embd_type.lower()]
    if extra:
        args += list(extra)

    # La ftype posizionale e' obbligatoria e governa i tensori che --tensor-type
    # non nomina. Tre casi:
    #  - resto_type dato: e' quello, esplicitamente voluto;
    #  - senza --tensor-type: gli expert si ottengono solo quantizzando tutto,
    #    quindi la ftype e' quella degli expert;
    #  - con --tensor-type e nessun resto: serve un valore neutro che non
    #    rimpicciolisca i tensori non-expert. FTYPE_NEUTRA e' quella.
    if resto_type:
        base = resto_type
    elif not usa_tensor_type:
        base = expert_type
    else:
        base = FTYPE_NEUTRA
    # I quattro posizionali, in quest'ordine:
    #     model-f32.gguf  model-quant.gguf  type  [nthreads]
    # nthreads NON e' un'opzione. Avevo scritto "--threads N" per analogia con
    # llama-server e llama-perplexity, che invece la vogliono cosi': quantize
    # non la riconosce, stampa l'uso ed esce con 1, senza dire quale argomento
    # non gli piaceva. Una famiglia di strumenti, tre convenzioni diverse.
    args += [str(sorgente), str(destinazione), _ftype_name(base), str(int(threads))]
    return args


def _ftype_name(nome: str) -> str:
    n = nome.upper()
    return FTYPE_ALIAS.get(n, n)


_PROGRESS_RE = re.compile(r"\[\s*(\d+)\s*/\s*(\d+)\s*\]\s*(\S+)")
_SIZE_RE = re.compile(r"quant size\s*=\s*([0-9.]+)\s*MB", re.IGNORECASE)


@dataclass
class RisultatoQuantizzazione:
    sorgente: str = ""
    destinazione: str = ""
    expert_type: str = ""
    secondi: float = 0.0
    byte_finali: int = 0
    tensori: int = 0
    codice: Optional[int] = None
    errore_msg: Optional[str] = None
    coda: List[str] = field(default_factory=list)
    comando: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "sorgente": self.sorgente, "destinazione": self.destinazione,
            "expert_type": self.expert_type, "secondi": round(self.secondi, 1),
            "byte_finali": self.byte_finali, "tensori": self.tensori,
            "codice": self.codice, "errore_msg": self.errore_msg,
            "comando": " ".join(self.comando),
        }


def _e_uso(righe: Sequence[str]) -> bool:
    """L'output e' la schermata d'uso, cioe' un rifiuto degli argomenti?"""
    return any(r.lstrip().startswith("usage:") for r in righe[:10])


def _estremi(righe: Sequence[str], testa: int = 12, coda: int = 20) -> List[str]:
    """Le prime e le ultime righe, non solo le ultime.

    Un processo che muore per un argomento sbagliato mette la diagnosi in cima e
    poi stampa pagine di aiuto: tenere la coda conserva l'aiuto e butta via la
    ragione. E' lo stesso errore, capovolto, che avevo gia' fatto tenendo le
    prime quaranta righe del log di llama-server invece delle ultime.
    """
    if len(righe) <= testa + coda:
        return list(righe)
    return (list(righe[:testa])
            + [f"    ... {len(righe) - testa - coda} righe omesse ..."]
            + list(righe[-coda:]))


def _esegui(args: List[str], env: Optional[Dict[str, str]], timeout: float,
            progress=None) -> Tuple[Optional[int], List[str], Optional[str], float]:
    import os as _os

    child_env = None
    if env:
        child_env = dict(_os.environ)
        child_env.update(env)
    t0 = time.time()
    try:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1,
                                encoding="utf-8", errors="replace", env=child_env)
    except Exception as exc:  # noqa: BLE001
        return None, [], f"impossibile avviare {Path(args[0]).name}: {exc}", 0.0

    righe: List[str] = []
    msg: Optional[str] = None
    try:
        assert proc.stdout is not None
        for riga in proc.stdout:
            riga = riga.rstrip("\n")
            righe.append(riga)
            if len(righe) > 600:
                del righe[:300]
            if progress:
                progress(riga)
            if time.time() - t0 > timeout:
                proc.kill()
                msg = f"interrotto dopo {timeout:.0f}s"
                break
        codice = proc.wait(timeout=120)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        codice = None
        try:
            proc.kill()
        except Exception:
            pass
    return codice, righe, msg, time.time() - t0


def run_quantize(binario: Path, sorgente: str, destinazione: str,
                 expert_type: str, resto_type: Optional[str] = None,
                 embd_type: Optional[str] = None, imatrix: Optional[str] = None,
                 threads: int = 8, consenti_riquantizzazione: bool = False,
                 usa_tensor_type: bool = True,
                 env: Optional[Dict[str, str]] = None,
                 progress=None, timeout: float = 21600.0,
                 extra: Optional[Sequence[str]] = None) -> RisultatoQuantizzazione:
    res = RisultatoQuantizzazione(sorgente=str(sorgente),
                                  destinazione=str(destinazione),
                                  expert_type=expert_type)
    args = build_args(binario, sorgente, destinazione, expert_type, resto_type,
                      embd_type, imatrix, threads, consenti_riquantizzazione,
                      usa_tensor_type, extra)

    stato = {"fatti": 0, "totale": 0}

    def riga_vista(riga: str) -> None:
        m = _PROGRESS_RE.search(riga)
        if m:
            stato["fatti"] = int(m.group(1))
            stato["totale"] = int(m.group(2))
            if progress:
                progress(stato["fatti"], stato["totale"], m.group(3))

    codice, righe, msg, dur = _esegui(args, env, timeout, riga_vista)
    res.codice = codice
    res.secondi = dur
    res.errore_msg = msg
    res.tensori = stato["fatti"]
    res.comando = list(args)
    res.coda = _estremi(righe)
    if codice not in (0, None) and _e_uso(righe):
        # llama-quantize risponde a un argomento che non conosce stampando
        # l'elenco completo dei tipi e uscendo con 1, senza nominare l'argomento.
        # L'elenco e' lungo, quindi tenere solo le ultime righe lascia vedere
        # soltanto i tipi e nasconde la diagnosi. Qui lo diciamo noi.
        res.errore_msg = ("llama-quantize ha rifiutato gli argomenti e ha "
                          "risposto con l'elenco dei tipi: uno dei parametri "
                          "qui sotto non e' previsto da questa build")

    dest = Path(destinazione)
    if dest.exists():
        res.byte_finali = dest.stat().st_size
    if codice not in (0, None) and not res.errore_msg:
        res.errore_msg = f"llama-quantize e' uscito con codice {codice}"
    if res.byte_finali == 0 and not res.errore_msg:
        res.errore_msg = "nessun file prodotto"
    return res


# ----------------------------------------------------------------- imatrix ---

_IMAT_RE = re.compile(r"compute_imatrix:\s*(\d+)\s*chunks|save_imatrix:\s*stor")
_IMAT_CHUNK_RE = re.compile(r"\[\s*(\d+)\s*\]")


@dataclass
class RisultatoImatrix:
    model: str = ""
    destinazione: str = ""
    testo: str = ""
    chunk: int = 0
    secondi: float = 0.0
    byte: int = 0
    codice: Optional[int] = None
    errore_msg: Optional[str] = None
    coda: List[str] = field(default_factory=list)
    comando: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "model": self.model, "destinazione": self.destinazione,
            "testo": self.testo, "chunk": self.chunk,
            "secondi": round(self.secondi, 1), "byte": self.byte,
            "codice": self.codice, "errore_msg": self.errore_msg,
        }


def build_imatrix_args(binario: Path, model: str, testo: str, destinazione: str,
                       ctx: int = 512, threads: int = 8,
                       chunks: Optional[int] = None,
                       extra: Optional[Sequence[str]] = None) -> List[str]:
    args = [
        str(binario),
        "--model", str(model),
        "--file", str(testo),
        "--output-file", str(destinazione),
        "--ctx-size", str(ctx),
        "--threads", str(threads),
        "--n-gpu-layers", "0",
    ]
    if chunks:
        args += ["--chunks", str(chunks)]
    if extra:
        args += list(extra)
    return args


def run_imatrix(binario: Path, model: str, testo: str, destinazione: str,
                ctx: int = 512, threads: int = 8, chunks: Optional[int] = None,
                env: Optional[Dict[str, str]] = None, progress=None,
                timeout: float = 43200.0,
                extra: Optional[Sequence[str]] = None) -> RisultatoImatrix:
    res = RisultatoImatrix(model=str(model), destinazione=str(destinazione),
                           testo=str(testo))
    args = build_imatrix_args(binario, model, testo, destinazione, ctx, threads,
                              chunks, extra)
    stato = {"chunk": 0}

    def riga_vista(riga: str) -> None:
        for m in _IMAT_CHUNK_RE.finditer(riga):
            n = int(m.group(1))
            if n > stato["chunk"]:
                stato["chunk"] = n
                if progress:
                    progress(n)

    codice, righe, msg, dur = _esegui(args, env, timeout, riga_vista)
    res.codice = codice
    res.secondi = dur
    res.errore_msg = msg
    res.coda = _estremi(righe)
    res.comando = list(args)
    res.chunk = stato["chunk"]
    d = Path(destinazione)
    if d.exists():
        res.byte = d.stat().st_size
    if codice not in (0, None) and not res.errore_msg:
        res.errore_msg = f"llama-imatrix e' uscito con codice {codice}"
    if res.byte == 0 and not res.errore_msg:
        res.errore_msg = "nessuna matrice prodotta"
    return res


# ------------------------------------------------------------- suggerimenti ---

def scegli_tipo_per_budget(model: GGUFModel, budget: int,
                           margine: float = 0.90) -> Tuple[Optional[str], List[str]]:
    """Il tipo piu' grande i cui expert stanno nel budget.

    Il criterio non e' "il piu' piccolo possibile": ogni bit in meno costa
    qualita', e scendere sotto il necessario la regala senza comprare niente.
    Si parte dall'alto e ci si ferma al primo che ci sta.
    """
    note: List[str] = []
    obiettivo = int(budget * margine)
    # Le mescole restano fuori: non si possono assegnare ai soli expert, e
    # suggerirle porterebbe a un comando che llama-quantize rifiuta.
    # Il pavimento a 2 bit non e' timidezza: sotto, llama.cpp stesso etichetta i
    # tipi come sperimentali, e un modello che entra in RAM e risponde a caso non
    # ha risolto il problema, lo ha nascosto. Se serve scendere sotto, la
    # risposta giusta e' un modello piu' piccolo quantizzato meglio.
    candidati = [t for t in tipi_disponibili()
                 if 2.0 <= t.bpw <= 5.0 and not t.solo_ftype]
    scelto: Optional[str] = None
    for t in sorted(candidati, key=lambda x: -x.bpw):
        s = stima(model, t.nome)
        if s.byte_dopo <= obiettivo:
            scelto = t.nome
            note.append(f"{t.nome}: il file esce {fmt_bytes(s.byte_dopo)}, "
                        f"sotto il {margine*100:.0f}% del budget "
                        f"({fmt_bytes(obiettivo)})")
            break
        note.append(f"{t.nome}: {fmt_bytes(s.byte_dopo)}, non ci sta")
    if scelto is None:
        note.append("nessun tipo sopra i 2 bit fa stare il modello nel budget: "
                    "serve un modello piu' piccolo, non una quantizzazione piu' "
                    "aggressiva. Sotto i 2 bit si ottiene un file che entra in "
                    "RAM e risponde a caso, che non e' un risultato migliore")
    return scelto, note
