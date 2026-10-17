"""Misura della qualita' di un modello quantizzato.

Perche' esiste
--------------
Il progetto sta per scambiare qualita' con velocita': quantizzare gli expert a
2 bit li fa stare tutti in RAM ed elimina lo streaming da disco, che e' il collo
di bottiglia misurato. Ma "piu' veloce e un po' peggio" non e' un risultato
finche' "un po'" non e' un numero.

La perplessita' e' quel numero: quanto il modello e' sorpreso dal testo che
legge. Piu' bassa e' meglio. Non dice tutto — un modello puo' avere buona
perplessita' e ragionare male — ma e' l'unica misura di qualita' che si ottiene
in modo riproducibile in pochi minuti, e serve a scartare le quantizzazioni
palesemente rotte prima di perderci sopra del tempo.

Cosa rende due perplessita' confrontabili
-----------------------------------------
Quasi niente, se non si sta attenti. Il numero cambia con il testo, con la
lunghezza del contesto, con la dimensione della finestra e col tokenizer. Due
misure fatte con parametri diversi non si possono mettere nella stessa frase,
e questo modulo lo impedisce invece di sperare che chi legge se ne ricordi:
i parametri finiscono nel risultato, e il confronto si rifiuta se non coincidono.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .util import fmt_bytes

# llama-perplexity stampa, a fine esecuzione:
#   Final estimate: PPL = 6.9877 +/- 0.04321
_PPL_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)\s*\+/-\s*([0-9.]+)")
# e durante l'esecuzione righe tipo "[12]6.8123,"
_PROGRESS_RE = re.compile(r"\[(\d+)\]([0-9.]+),")


@dataclass
class PPLResult:
    model: str
    ppl: float = 0.0
    errore: float = 0.0        # deviazione standard riportata da llama.cpp
    chunk: int = 0             # quanti blocchi di testo sono stati valutati
    ctx: int = 0
    testo: str = ""
    testo_hash: str = ""
    secondi: float = 0.0
    tokens: int = 0
    raw_tail: List[str] = field(default_factory=list)
    errore_msg: Optional[str] = None

    @property
    def firma(self) -> str:
        """Cio' che deve coincidere perche' due misure siano confrontabili.

        Il testo entra come impronta e non come nome: due file chiamati uguale
        con contenuto diverso darebbero numeri diversi senza che nulla lo
        segnali.
        """
        return f"ctx={self.ctx};chunk={self.chunk};testo={self.testo_hash}"

    def to_dict(self) -> Dict[str, object]:
        return {
            "model": self.model, "ppl": round(self.ppl, 4),
            "errore": round(self.errore, 4), "chunk": self.chunk,
            "ctx": self.ctx, "testo": self.testo, "testo_hash": self.testo_hash,
            "secondi": round(self.secondi, 1), "tokens": self.tokens,
            "firma": self.firma, "errore_msg": self.errore_msg,
        }


def hash_testo(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for pezzo in iter(lambda: fh.read(1 << 20), b""):
            h.update(pezzo)
    return h.hexdigest()[:16]


def build_args(binario: Path, model: str, testo: str, ctx: int, threads: int,
               chunks: Optional[int], extra: Optional[List[str]] = None) -> List[str]:
    args = [
        str(binario),
        "--model", str(model),
        "--file", str(testo),
        "--ctx-size", str(ctx),
        "--threads", str(threads),
        # niente offload: la misura deve dipendere dal modello, non da quale
        # pezzo e' finito sulla GPU
        "--n-gpu-layers", "0",
    ]
    if chunks:
        args += ["--chunks", str(chunks)]
    if extra:
        args += list(extra)
    return args


def _pezzi(flusso, dimensione: int = 256):
    """Spezza l'uscita del processo su fine riga **o virgola**.

    `for riga in flusso` aspetta un fine riga, e llama-perplexity non ne manda
    nessuno mentre lavora: stampa `[1]7.12,[2]6.98,[3]...` tutto di seguito e va
    a capo solo alla fine. Leggendo per righe l'avanzamento non arriva mai, e
    per quaranta minuti sembra che il programma sia bloccato — e' successo due
    volte, e la seconda l'avevo pure promesso che si sarebbe visto.

    Qui si legge a pezzi e si taglia anche sulle virgole, cosi' ogni `[N]valore,`
    diventa un pezzo appena il processo lo stampa. Il resto del codice continua a
    ricevere stringhe e non si accorge della differenza.
    """
    resto = ""
    while True:
        dato = flusso.read(dimensione)
        if not dato:
            break
        resto += dato
        # si taglia sull'ultimo separatore: cio' che resta puo' essere un pezzo
        # a meta' e si aspetta il prossimo giro
        taglio = max(resto.rfind("\n"), resto.rfind(","))
        if taglio < 0:
            continue
        testa, resto = resto[:taglio + 1], resto[taglio + 1:]
        for pezzo in re.split(r"(?<=,)|\n", testa):
            pezzo = pezzo.strip()
            if pezzo:
                yield pezzo
    if resto.strip():
        yield resto.strip()


def run_perplexity(binario: Path, model: str, testo: str, ctx: int = 512,
                   threads: int = 8, chunks: Optional[int] = None,
                   env: Optional[Dict[str, str]] = None,
                   progress=None, timeout: float = 7200.0,
                   extra: Optional[List[str]] = None) -> PPLResult:
    """Esegue llama-perplexity e ne estrae il risultato.

    L'output viene letto riga per riga perche' la misura dura minuti e vedere
    solo il risultato finale, senza sapere se sta procedendo, e' il modo migliore
    per interromperla per sbaglio.
    """
    res = PPLResult(model=str(model), ctx=ctx, testo=str(testo),
                    testo_hash=hash_testo(testo))
    args = build_args(binario, model, testo, ctx, threads, chunks, extra)
    import os as _os

    child_env = None
    if env:
        child_env = dict(_os.environ)
        child_env.update(env)

    t0 = time.time()
    code = []
    try:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, encoding="utf-8", errors="replace",
                                env=child_env)
    except Exception as exc:  # noqa: BLE001
        res.errore_msg = f"impossibile avviare llama-perplexity: {exc}"
        return res

    righe: List[str] = []
    try:
        assert proc.stdout is not None
        for riga in _pezzi(proc.stdout):
            righe.append(riga)
            if len(righe) > 400:
                del righe[:200]
            m = _PROGRESS_RE.search(riga)
            if m and progress:
                progress(int(m.group(1)), float(m.group(2)))
            m2 = _PPL_RE.search(riga)
            if m2:
                res.ppl = float(m2.group(1))
                res.errore = float(m2.group(2))
            if "n_tokens" in riga:
                m3 = re.search(r"n_tokens\s*=\s*(\d+)", riga)
                if m3:
                    res.tokens = int(m3.group(1))
            if time.time() - t0 > timeout:
                proc.kill()
                res.errore_msg = f"interrotto dopo {timeout:.0f}s"
                break
        code.append(proc.wait(timeout=60))
    except Exception as exc:  # noqa: BLE001
        res.errore_msg = str(exc)
        try:
            proc.kill()
        except Exception:
            pass

    res.secondi = time.time() - t0
    res.raw_tail = righe[-20:]
    # quanti blocchi sono stati valutati: l'ultimo indice di progresso visto
    ultimo = 0
    for riga in righe:
        for m in _PROGRESS_RE.finditer(riga):
            ultimo = max(ultimo, int(m.group(1)))
    res.chunk = ultimo
    if res.ppl <= 0 and not res.errore_msg:
        res.errore_msg = ("llama-perplexity non ha prodotto una stima finale; "
                          "le ultime righe sono nel log")
    return res


# ------------------------------------------------------------- confronto -----


@dataclass
class Confronto:
    riferimento: PPLResult
    candidato: PPLResult

    @property
    def confrontabile(self) -> bool:
        return (self.riferimento.firma == self.candidato.firma
                and self.riferimento.ppl > 0 and self.candidato.ppl > 0)

    @property
    def delta(self) -> float:
        return self.candidato.ppl - self.riferimento.ppl

    @property
    def delta_pct(self) -> float:
        if self.riferimento.ppl <= 0:
            return 0.0
        return 100.0 * self.delta / self.riferimento.ppl

    @property
    def significativo(self) -> bool:
        """La differenza supera l'incertezza delle due misure?

        Senza questo controllo si finisce per festeggiare uno 0.3% che sta
        dentro il rumore. Le due deviazioni si sommano in quadratura.
        """
        sigma = math.hypot(self.riferimento.errore, self.candidato.errore)
        return abs(self.delta) > 2.0 * max(sigma, 1e-9)

    def giudizio(self) -> List[str]:
        out: List[str] = []
        if not self.confrontabile:
            out.append(
                "le due misure NON sono confrontabili: contesto, numero di blocchi o "
                "testo diversi. Un numero di perplessita' ha senso solo contro un altro "
                "ottenuto con gli stessi identici parametri."
            )
            out.append(f"  riferimento: {self.riferimento.firma}")
            out.append(f"  candidato:   {self.candidato.firma}")
            return out

        out.append(
            f"perplessita': {self.riferimento.ppl:.4f} -> {self.candidato.ppl:.4f} "
            f"({self.delta_pct:+.2f}%)"
        )
        if not self.significativo:
            out.append(
                "la differenza sta dentro l'incertezza delle misure: i due modelli non "
                "sono distinguibili su questo testo"
            )
            return out

        d = self.delta_pct
        if d <= 0:
            # Non e' sempre sospetto: se il riferimento e' a sua volta una
            # quantizzazione mal riuscita, batterla e' esattamente cio' che ci
            # si aspetta. Lo dico senza decidere al posto di chi legge, perche'
            # la versione precedente di questa riga ha gia' fatto scambiare un
            # miglioramento reale per un errore.
            out.append("il candidato e' migliore del riferimento. Se il riferimento e' "
                       "un modello a precisione piu' alta, e' un risultato strano e "
                       "vale la pena ricontrollare il file; se e' un'altra "
                       "quantizzazione, e' semplicemente riuscita meglio")
        elif d < 3:
            out.append("degrado piccolo: sotto il 3% la differenza e' difficile da "
                       "notare nell'uso normale")
        elif d < 10:
            out.append("degrado percepibile: il modello resta utilizzabile ma sbagliera' "
                       "piu' spesso sui compiti difficili")
        elif d < 25:
            out.append("degrado forte: conviene solo se in cambio si ottiene qualcosa che "
                       "prima era impossibile, non per una frazione di velocita'")
        else:
            out.append("degrado grave: a questo punto conviene un modello piu' piccolo "
                       "quantizzato meglio, invece di questo quantizzato peggio")
        return out


def salva(res: PPLResult, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    righe = []
    if p.exists():
        righe = [r for r in p.read_text(encoding="utf-8").splitlines() if r.strip()]
    righe.append(json.dumps(res.to_dict(), ensure_ascii=False))
    p.write_text("\n".join(righe) + "\n", encoding="utf-8")


def carica(path: str | Path) -> List[PPLResult]:
    p = Path(path)
    if not p.exists():
        return []
    out: List[PPLResult] = []
    for riga in p.read_text(encoding="utf-8").splitlines():
        riga = riga.strip()
        if not riga:
            continue
        try:
            d = json.loads(riga)
        except json.JSONDecodeError:
            continue
        r = PPLResult(model=d.get("model", ""), ppl=float(d.get("ppl", 0)),
                      errore=float(d.get("errore", 0)), chunk=int(d.get("chunk", 0)),
                      ctx=int(d.get("ctx", 0)), testo=d.get("testo", ""),
                      testo_hash=d.get("testo_hash", ""),
                      secondi=float(d.get("secondi", 0)), tokens=int(d.get("tokens", 0)),
                      errore_msg=d.get("errore_msg"))
        out.append(r)
    return out


def confrontabili(storico: List[PPLResult], firma: str,
                  escludi: str = "") -> List[PPLResult]:
    """L'ultima misura di *ogni* modello con la stessa firma, dalla migliore.

    Con due modelli bastava l'ultima misura buona; col terzo la regola crolla.
    E' successo: misurando il terzo modello ha scelto come riferimento il
    secondo — cioe' un IQ2 mal calibrato — e ha annunciato che il candidato era
    "migliore, risultato sospetto", quando il confronto che interessava era
    contro il Q4_K_M. Il riferimento non e' il piu' recente, e' quello che il
    candidato sta cercando di non peggiorare.
    """
    per_modello: Dict[str, PPLResult] = {}
    for r in storico:
        if r.firma == firma and r.ppl > 0 and str(r.model) != str(escludi):
            per_modello[str(r.model)] = r        # l'ultima di ogni modello
    return sorted(per_modello.values(), key=lambda r: r.ppl)


def trova_riferimento(storico: List[PPLResult], firma: str,
                      escludi: str = "", nome: str = "") -> Optional[PPLResult]:
    """Il termine di paragone: quello scelto a mano, o il migliore disponibile.

    "Il migliore" e non "il piu' recente" perche' la domanda a cui serve
    rispondere e' sempre quanto si perde rispetto al meglio che si ha, non
    rispetto all'ultimo tentativo fatto.
    """
    cands = confrontabili(storico, firma, escludi)
    if nome:
        atteso = Path(nome).name.lower()
        for r in cands:
            if Path(r.model).name.lower() == atteso or str(r.model) == nome:
                return r
        return None
    return cands[0] if cands else None


# ------------------------------------------------- testo di calibrazione -----

def diversita(path: str | Path) -> float:
    """Quota di righe distinte nel testo. Un testo che si ripete e' inutile per
    misurare la qualita': dopo il primo giro qualunque modello, anche rotto,
    predice la ripetizione quasi perfettamente, e la perplessita' esce bassa per
    tutti. Il primo testo di ripiego che avevo scritto ciclava dieci frasi: non
    avrebbe distinto Q4 da IQ2."""
    p = Path(path)
    try:
        pezzi = [r.strip() for r in p.read_text(encoding="utf-8",
                                                errors="replace").split(".") if r.strip()]
    except Exception:
        return 1.0
    if not pezzi:
        return 1.0
    return len(set(pezzi)) / len(pezzi)


def testo_di_prova(path: str | Path, parole: int = 6000, seed: int = 7) -> Path:
    """Scrive un testo di riferimento se non ne esiste uno.

    Resta un ripiego: serve a confrontare due quantizzazioni fra loro, non a
    produrre un numero paragonabile a quelli pubblicati (per quelli serve
    wikitext-2-raw). Ma almeno non si ripete: le frasi si compongono da pezzi
    diversi, cosi' il modello deve davvero prevedere invece di ricordare.
    """
    p = Path(path)
    # Un testo gia' scritto si riusa: rigenerarlo cambierebbe la firma delle
    # misure di perplessita' gia' prese, e i confronti futuri smetterebbero di
    # funzionare senza dire perche'. Si rifa' solo se non basta: troppo corto
    # per i blocchi richiesti, o cosi' ripetitivo da non distinguere niente.
    if p.exists() and diversita(p) > 0.8:
        parole_presenti = len(p.read_text(encoding="utf-8",
                                          errors="replace").split())
        if parole_presenti >= parole * 0.95 and p.stat().st_size > 1000:
            return p

    import random as _r

    rnd = _r.Random(seed)
    soggetti = [
        "il sistema operativo", "un modello linguistico", "la cache del processore",
        "il disco a stato solido", "l'algoritmo di sostituzione", "la memoria virtuale",
        "un thread in attesa", "il compilatore", "la matrice di importanza",
        "il gestore della memoria", "una pagina non presente", "il router della miscela",
        "la quantizzazione", "il kernel di calcolo", "una lettura non bufferizzata",
    ]
    verbi = [
        "decide", "riduce", "aspetta", "misura", "riscrive", "ignora", "accumula",
        "serializza", "distribuisce", "verifica", "scarta", "rimanda", "impegna",
    ]
    oggetti = [
        "le pagine meno usate di recente", "la precisione dei pesi in virgola mobile",
        "il numero di richieste in volo verso il dispositivo",
        "l'errore che si accumula nelle somme lunghe",
        "la porzione di parametri effettivamente attivi",
        "il tempo speso dentro la barriera fra i thread",
        "la probabilita' assegnata a ogni parola possibile",
        "lo spazio di indirizzamento riservato e non impegnato",
        "i blocchi contigui richiesti da un singolo token",
        "il costo di azzerare una pagina prima di consegnarla",
        "la differenza fra cio' che e' aperto e cio' che e' in uso",
    ]
    code = [
        "quando la memoria disponibile scende sotto la soglia",
        "mentre gli altri processi competono per la stessa banda",
        "se il tetto dichiarato deve restare una promessa e non una speranza",
        "perche' il costo si paga una volta sola per ogni slice",
        "anche se il risultato resta identico byte per byte",
        "prima che il calcolo possa cominciare",
        "senza che nulla nel registro lo segnali",
        "a meno che la misura non dica il contrario",
    ]

    fuori: List[str] = []
    n = 0
    while n < parole:
        f = (f"{rnd.choice(soggetti).capitalize()} {rnd.choice(verbi)} "
             f"{rnd.choice(oggetti)} {rnd.choice(code)}.")
        fuori.append(f)
        n += len(f.split())
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(" ".join(fuori) + "\n", encoding="utf-8")
    return p
