#!/usr/bin/env python3
"""Controlla il README e rigenera il suo indice.

Perche' esiste
--------------
Il README e' il risultato principale di questo progetto: 2000 righe e 65 titoli,
in cui ogni numero e' misurato e ogni previsione sbagliata e' lasciata scritta.
Un documento cosi' ha due modi di rompersi in silenzio:

1. **un link interno che non punta piu' a niente.** Rinominare un titolo non
   produce nessun errore: il link continua a esistere e smette di funzionare.
2. **un indice scritto a mano.** Invecchia alla prima sezione aggiunta, e un
   indice incompleto e' peggio di nessun indice, perche' chi lo legge conclude
   che la sezione che cerca non esiste.

Da qui questo strumento: verifica i link e **rigenera** l'indice fra due
marcatori, cosi' non c'e' una copia da tenere aggiornata a mano.

Uso
---
    python strumenti/controlla_readme.py            # controlla, non scrive
    python strumenti/controlla_readme.py --scrivi   # aggiorna l'indice

Esce con 1 se qualcosa non va, cosi' si puo' mettere in CI.

Nota su due falsi allarmi, imparati sbagliando
----------------------------------------------
La prima versione di questi controlli ne produsse cinque, **tutti falsi**:

- cercava i segnaposto "TODO" con una sottostringa, e trovava `meTODO` dentro la
  parola "metodo", due volte;
- segnalava "placeholder" come segnaposto dimenticato, dove era il termine
  tecnico di `VirtualAlloc2` (MEM_RESERVE_PLACEHOLDER);
- contava le colonne di una tabella contando i caratteri `|`, e trovava
  incoerenti due righe che contenevano `\\|` *escapato* dentro `--backend
  auto\\|cpu\\|vulkan`.

Uno strumento di controllo che grida al lupo e' uno strumento che si impara a
ignorare, quindi i tre casi sono gestiti e questo commento spiega perche'.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple

INIZIO = "<!-- indice: generato da strumenti/controlla_readme.py, non modificare a mano -->"
FINE = "<!-- fine indice -->"

# Titoli da non mettere in indice: il titolo del documento e l'indice stesso.
SALTA_LIVELLI = (1,)


def ancora(titolo: str) -> str:
    """L'ancora che GitHub genera per un titolo.

    Minuscolo, via la punteggiatura, spazi in trattini. Tiene le lettere
    accentate, che GitHub conserva.
    """
    s = titolo.strip().lower()
    s = re.sub(r"[`*_]", "", s)                       # enfasi e codice
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)    # link -> solo il testo
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)  # punteggiatura
    return s.strip().replace(" ", "-")


def titoli(righe: List[str]) -> List[Tuple[int, int, str]]:
    """(numero di riga, livello, testo) per ogni titolo fuori dai blocchi di codice."""
    out: List[Tuple[int, int, str]] = []
    dentro_codice = False
    for i, riga in enumerate(righe, 1):
        if riga.lstrip().startswith("```"):
            dentro_codice = not dentro_codice
            continue
        if dentro_codice:
            continue
        m = re.match(r"^(#{1,6})\s+(.*?)\s*$", riga)
        if m:
            out.append((i, len(m.group(1)), m.group(2)))
    return out


def link_interni(testo: str) -> List[str]:
    return re.findall(r"\]\(#([^)]+)\)", testo)


def celle(riga: str) -> int:
    """Colonne di una riga di tabella, ignorando i `|` escapati.

    Contare i `|` grezzi segnalava come incoerenti le righe che documentano
    `--backend auto\\|cpu\\|vulkan`, dove la barra e' escapata di proposito.
    """
    return riga.replace(r"\|", "").count("|")


def controlla(percorso: Path) -> Tuple[List[str], str]:
    testo = percorso.read_text(encoding="utf-8")
    righe = testo.splitlines()
    problemi: List[str] = []

    tt = titoli(righe)
    ancore = {ancora(t): t for _, _, t in tt}

    # 1. link interni che non puntano a niente
    for l in link_interni(testo):
        if l not in ancore:
            import difflib
            vicini = difflib.get_close_matches(l, list(ancore), n=1, cutoff=0.4)
            suggerimento = f" (forse: #{vicini[0]})" if vicini else ""
            problemi.append(f"link interno rotto: #{l}{suggerimento}")

    # 2. titoli uguali: l'ancora diventa ambigua e GitHub ne numera una
    for t, n in Counter(a for a in (ancora(x) for _, _, x in tt)).items():
        if n > 1:
            problemi.append(f"titolo duplicato (ancora ambigua): #{t} x{n}")

    # 3. blocchi di codice non chiusi
    if sum(1 for r in righe if r.lstrip().startswith("```")) % 2:
        problemi.append("un blocco di codice ``` non e' chiuso")

    # 4. tabelle con righe di larghezza diversa dall'intestazione
    i = 0
    while i < len(righe):
        if (righe[i].strip().startswith("|") and i + 1 < len(righe)
                and re.match(r"^\s*\|[\s:|-]+\|\s*$", righe[i + 1])):
            atteso = celle(righe[i])
            j = i + 2
            while j < len(righe) and righe[j].strip().startswith("|"):
                if celle(righe[j]) != atteso:
                    problemi.append(
                        f"riga {j+1}: la tabella ha {celle(righe[j])} colonne, "
                        f"l'intestazione {atteso}")
                j += 1
            i = j
        else:
            i += 1

    # 5. segnaposto dimenticati, con i confini di parola: "metodo" contiene
    #    "todo", e la prima versione di questo controllo ci e' cascata due volte
    for parola in ("TODO", "FIXME", "XXX"):
        for n, riga in enumerate(righe, 1):
            if re.search(rf"\b{parola}\b", riga):
                problemi.append(f"riga {n}: segnaposto {parola} rimasto")

    return problemi, indice(tt)


def indice(tt: List[Tuple[int, int, str]]) -> str:
    """L'indice in markdown, annidato per livello."""
    fuori = [t for t in tt if t[1] not in SALTA_LIVELLI and t[2].lower() != "indice"]
    minimo = min((lv for _, lv, _ in fuori), default=2)
    voci = [INIZIO, ""]
    for _, livello, testo in fuori:
        rientro = "  " * (livello - minimo)
        pulito = re.sub(r"[`*]", "", testo)
        voci.append(f"{rientro}- [{pulito}](#{ancora(testo)})")
    voci += ["", FINE]
    return "\n".join(voci)


def scrivi_indice(percorso: Path, nuovo: str) -> bool:
    testo = percorso.read_text(encoding="utf-8")
    if INIZIO in testo and FINE in testo:
        prima = testo.split(INIZIO)[0]
        dopo = testo.split(FINE, 1)[1]
        aggiornato = prima + nuovo + dopo
    else:
        # prima volta: dopo la prima riga separatrice che segue l'intestazione
        marcatore = "\n---\n"
        if marcatore not in testo:
            return False
        i = testo.index(marcatore) + len(marcatore)
        aggiornato = testo[:i] + "\n## Indice\n\n" + nuovo + "\n\n---\n" + testo[i:]
    if aggiornato == testo:
        return False
    percorso.write_text(aggiornato, encoding="utf-8")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file", nargs="?", default="README.md")
    ap.add_argument("--scrivi", action="store_true",
                    help="aggiorna l'indice invece di limitarsi a controllare")
    args = ap.parse_args()
    percorso = Path(args.file)
    if not percorso.exists():
        print(f"non trovo {percorso}")
        return 1

    problemi, nuovo = controlla(percorso)
    if args.scrivi:
        if scrivi_indice(percorso, nuovo):
            print(f"indice aggiornato in {percorso}")
            problemi, nuovo = controlla(percorso)
        else:
            print("indice gia' aggiornato")

    testo = percorso.read_text(encoding="utf-8")
    if INIZIO in testo:
        attuale = INIZIO + testo.split(INIZIO, 1)[1].split(FINE, 1)[0] + FINE
        if attuale.strip() != nuovo.strip():
            problemi.append("l'indice non e' aggiornato: "
                            "python strumenti/controlla_readme.py --scrivi")

    tt = titoli(percorso.read_text(encoding="utf-8").splitlines())
    print(f"{percorso}: {len(tt)} titoli, "
          f"{len(link_interni(testo))} link interni")
    if problemi:
        print(f"PROBLEMI: {len(problemi)}")
        for p in problemi:
            print(f"  - {p}")
        return 1
    print("nessun problema")
    return 0


if __name__ == "__main__":
    sys.exit(main())
