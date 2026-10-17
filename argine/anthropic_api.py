"""Traduzione fra il formato Anthropic (`/v1/messages`) e quello OpenAI.

Argine parla OpenAI perche' lo parla `llama-server`. Questo modulo aggiunge il
dialetto Anthropic davanti, traducendo richiesta e risposta.

**Sono funzioni pure, da dizionario a dizionario, per scelta.** Nessun I/O,
nessuna rete, nessuno stato: tutto quello che c'e' qui si verifica con un test
che non avvia niente. E' una differenza che conta, perche' nello store il
collaudo girava su una piattaforma diversa da quella di destinazione e un bug ci
e' passato in mezzo. Qui non esiste una piattaforma di destinazione: esiste il
JSON.

Le differenze fra i due formati che bisogna davvero gestire:

- **`system`**: in Anthropic e' un campo in cima alla richiesta (stringa o lista
  di blocchi), in OpenAI e' un messaggio con `role: "system"` dentro la lista.
- **contenuto a blocchi**: in Anthropic `content` puo' essere una stringa oppure
  una lista di blocchi tipizzati (`text`, `image`, `tool_use`, `tool_result`); in
  OpenAI dei messaggi la forma usuale e' una stringa.
- **motivo di fine**: `stop_reason` con i valori `end_turn` / `max_tokens` /
  `stop_sequence` / `tool_use`, contro `finish_reason` con `stop` / `length` /
  `tool_calls`.
- **conteggi**: `usage.input_tokens` / `output_tokens` contro
  `usage.prompt_tokens` / `completion_tokens`.
- **nomi dei parametri**: `max_tokens` e' obbligatorio in Anthropic, e
  `stop_sequences` si chiama `stop`.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple


class ApiError(Exception):
    """Errore di traduzione da restituire al client come 400.

    Porta con se' il tipo Anthropic, perche' un client che parla quel dialetto si
    aspetta quella forma anche negli errori: restituirgli un errore in formato
    OpenAI sarebbe come tradurre tutto tranne la parte che si legge quando le
    cose vanno male.
    """

    def __init__(self, message: str, tipo: str = "invalid_request_error"):
        super().__init__(message)
        self.message = message
        self.tipo = tipo

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "error", "error": {"type": self.tipo, "message": self.message}}


# ============================================================== richiesta ===

def _testo_da_blocchi(content: Any, dove: str) -> str:
    """Appiattisce il `content` di un messaggio Anthropic in una stringa.

    I blocchi non testuali non si inventano: `llama-server` nella nostra
    configurazione non ha un percorso per le immagini, e far finta di averlo
    produrrebbe una risposta che ignora in silenzio meta' della domanda. Meglio
    un errore che dice cosa manca.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ApiError(f"{dove}: 'content' deve essere una stringa o una lista di blocchi")
    pezzi: List[str] = []
    for i, b in enumerate(content):
        if isinstance(b, str):
            pezzi.append(b)
            continue
        if not isinstance(b, dict):
            raise ApiError(f"{dove}: blocco {i} non valido")
        t = b.get("type")
        if t == "text":
            pezzi.append(str(b.get("text") or ""))
        elif t == "tool_result":
            # Il risultato di uno strumento e' testo per il modello: lo si passa
            # come tale, con il suo identificativo, perche' senza quello il
            # modello non sa a quale chiamata risponde.
            interno = b.get("content")
            testo = interno if isinstance(interno, str) else _testo_da_blocchi(interno, dove)
            tid = b.get("tool_use_id") or ""
            pezzi.append(f"[risultato dello strumento {tid}]\n{testo}" if tid else testo)
        elif t == "tool_use":
            pezzi.append(f"[chiamata allo strumento {b.get('name') or ''}]\n"
                         + json.dumps(b.get("input") or {}, ensure_ascii=False))
        elif t in ("image", "document"):
            raise ApiError(
                f"{dove}: blocchi '{t}' non supportati. Questo runtime serve modelli "
                f"testuali su CPU: un'immagine verrebbe ignorata in silenzio, e una "
                f"risposta che ignora meta' della domanda e' peggio di un errore.",
                tipo="invalid_request_error")
        else:
            raise ApiError(f"{dove}: blocco di tipo '{t}' non riconosciuto")
    return "\n".join(p for p in pezzi if p)


def _system_in_testo(system: Any) -> str:
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    return _testo_da_blocchi(system, "system")


def richiesta_in_openai(req: Dict[str, Any]) -> Dict[str, Any]:
    """Da `/v1/messages` a `/v1/chat/completions`.

    Solleva `ApiError` su una richiesta che non si puo' tradurre, invece di
    tradurla a meta': un campo perso qui diventa una risposta plausibile a una
    domanda diversa da quella fatta, che e' il modo peggiore di sbagliare.
    """
    if not isinstance(req, dict):
        raise ApiError("corpo della richiesta non valido: serve un oggetto JSON")

    messaggi = req.get("messages")
    if not isinstance(messaggi, list) or not messaggi:
        raise ApiError("'messages' e' obbligatorio e deve essere una lista non vuota")

    # max_tokens e' obbligatorio in Anthropic. Non si inventa un default: il
    # client che lo omette ha sbagliato, e un numero scelto da noi gli
    # restituirebbe una risposta troncata senza che sappia perche'.
    max_tokens = req.get("max_tokens")
    if max_tokens is None:
        raise ApiError("'max_tokens' e' obbligatorio nel formato Anthropic")
    try:
        max_tokens = int(max_tokens)
    except (TypeError, ValueError):
        raise ApiError("'max_tokens' deve essere un intero")
    if max_tokens <= 0:
        raise ApiError("'max_tokens' deve essere maggiore di zero")

    out_msg: List[Dict[str, str]] = []
    sistema = _system_in_testo(req.get("system"))
    if sistema:
        out_msg.append({"role": "system", "content": sistema})

    for i, m in enumerate(messaggi):
        if not isinstance(m, dict):
            raise ApiError(f"messages[{i}] non valido")
        ruolo = m.get("role")
        if ruolo not in ("user", "assistant"):
            raise ApiError(f"messages[{i}]: 'role' deve essere 'user' o 'assistant', "
                           f"non '{ruolo}'")
        out_msg.append({"role": ruolo,
                        "content": _testo_da_blocchi(m.get("content"), f"messages[{i}]")})

    oai: Dict[str, Any] = {
        "messages": out_msg,
        "max_tokens": max_tokens,
        "stream": bool(req.get("stream")),
    }
    if req.get("model"):
        oai["model"] = req["model"]

    # Parametri con nome diverso ma significato identico.
    for da, a in (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k")):
        if req.get(da) is not None:
            oai[a] = req[da]
    stop = req.get("stop_sequences")
    if stop:
        if not isinstance(stop, list):
            raise ApiError("'stop_sequences' deve essere una lista")
        oai["stop"] = stop
    if req.get("metadata") and isinstance(req["metadata"], dict):
        uid = req["metadata"].get("user_id")
        if uid:
            oai["user"] = uid
    return oai


# =============================================================== risposta ===

_FINE = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "end_turn",
}


def _stop_reason(finish: Optional[str], stop_sequences: Optional[List[str]],
                 testo: str) -> Tuple[str, Optional[str]]:
    """(stop_reason, stop_sequence).

    Anthropic distingue "si e' fermato da solo" da "ha incontrato una delle
    sequenze che gli hai dato", e nel secondo caso dice quale. OpenAI non lo
    distingue: dice `stop` in entrambi i casi. La differenza si ricostruisce
    guardando se il testo finisce con una delle sequenze richieste.
    """
    base = _FINE.get(finish or "", "end_turn")
    if base == "end_turn" and stop_sequences:
        for sq in stop_sequences:
            if sq and testo.endswith(sq):
                return "stop_sequence", sq
    return base, None


def _id_messaggio() -> str:
    return "msg_" + uuid.uuid4().hex[:24]


def risposta_in_anthropic(oai: Dict[str, Any], modello: str,
                          stop_sequences: Optional[List[str]] = None) -> Dict[str, Any]:
    """Da una risposta `/v1/chat/completions` a una risposta `/v1/messages`."""
    scelte = oai.get("choices") or []
    prima = scelte[0] if scelte else {}
    msg = prima.get("message") or {}
    testo = msg.get("content") or ""
    if not isinstance(testo, str):
        testo = _testo_da_blocchi(testo, "risposta del backend")

    blocchi: List[Dict[str, Any]] = []
    if testo:
        blocchi.append({"type": "text", "text": testo})

    # Le chiamate a strumenti, se il backend le ha prodotte. Gli argomenti
    # arrivano come stringa JSON e vanno consegnati come oggetto: lasciarli
    # stringa darebbe a un client Anthropic un `input` che non sa leggere.
    for tc in (msg.get("tool_calls") or []):
        fn = (tc or {}).get("function") or {}
        grezzi = fn.get("arguments")
        try:
            argomenti = json.loads(grezzi) if isinstance(grezzi, str) else (grezzi or {})
        except json.JSONDecodeError:
            # Non si inventa un oggetto vuoto: si consegna il testo cosi' com'e'
            # sotto una chiave riconoscibile, perche' un argomento perso porta a
            # una chiamata eseguita con parametri sbagliati.
            argomenti = {"_non_interpretabile": grezzi}
        blocchi.append({"type": "tool_use",
                        "id": tc.get("id") or ("toolu_" + uuid.uuid4().hex[:20]),
                        "name": fn.get("name") or "",
                        "input": argomenti})

    uso = oai.get("usage") or {}
    motivo, sequenza = _stop_reason(prima.get("finish_reason"), stop_sequences, testo)
    if any(b["type"] == "tool_use" for b in blocchi):
        motivo = "tool_use"
    return {
        "id": oai.get("id") or _id_messaggio(),
        "type": "message",
        "role": "assistant",
        "model": oai.get("model") or modello,
        "content": blocchi,
        "stop_reason": motivo,
        "stop_sequence": sequenza,
        "usage": {
            "input_tokens": int(uso.get("prompt_tokens") or 0),
            "output_tokens": int(uso.get("completion_tokens") or 0),
        },
    }


# ============================================================== streaming ===
#
# I due formati streaming non si somigliano. OpenAI manda una sequenza di chunk
# tutti della stessa forma, piu' un `data: [DONE]`. Anthropic manda eventi
# *tipizzati* e nominati, con un'apertura e una chiusura obbligatorie:
#
#     message_start -> content_block_start -> content_block_delta* ->
#     content_block_stop -> message_delta -> message_stop
#
# Un client Anthropic aspetta quella struttura: se gli arrivano solo i delta,
# resta in attesa di un `message_stop` che non viene. Per questo la chiusura si
# emette anche quando il flusso a monte si interrompe a meta'.

def _sse(evento: str, dati: Dict[str, Any]) -> bytes:
    return (f"event: {evento}\n"
            f"data: {json.dumps(dati, ensure_ascii=False)}\n\n").encode("utf-8")


def eventi_da_chunk_openai(righe: Iterable[bytes], modello: str,
                           stop_sequences: Optional[List[str]] = None
                           ) -> Iterable[bytes]:
    """Traduce un flusso SSE OpenAI in eventi SSE Anthropic.

    Prende le righe grezze come arrivano dal backend e produce i byte da
    inoltrare. E' un generatore perche' il punto dello streaming e' non
    aspettare la fine: accumulare tutto per tradurlo in blocco cancellerebbe
    l'unica ragione per cui esiste.
    """
    msg_id = _id_messaggio()
    aperto = False       # message_start emesso
    blocco = False       # content_block_start emesso
    testo_finora = ""
    finish: Optional[str] = None
    uso_in, uso_out = 0, 0

    def apri() -> Iterable[bytes]:
        yield _sse("message_start", {
            "type": "message_start",
            "message": {
                "id": msg_id, "type": "message", "role": "assistant",
                "model": modello, "content": [], "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        })

    for riga in righe:
        if not riga:
            continue
        s = riga.decode("utf-8", "replace").strip()
        if not s.startswith("data:"):
            continue
        payload = s[5:].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue

        if not aperto:
            aperto = True
            for b in apri():
                yield b

        uso = chunk.get("usage") or {}
        if uso:
            uso_in = int(uso.get("prompt_tokens") or uso_in)
            uso_out = int(uso.get("completion_tokens") or uso_out)

        for scelta in (chunk.get("choices") or []):
            delta = scelta.get("delta") or {}
            pezzo = delta.get("content")
            if pezzo:
                if not blocco:
                    blocco = True
                    yield _sse("content_block_start", {
                        "type": "content_block_start", "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    })
                testo_finora += pezzo
                yield _sse("content_block_delta", {
                    "type": "content_block_delta", "index": 0,
                    "delta": {"type": "text_delta", "text": pezzo},
                })
            if scelta.get("finish_reason"):
                finish = scelta["finish_reason"]

    # Chiusura. Si emette sempre, anche se a monte non e' arrivato niente: un
    # client che non riceve `message_stop` aspetta per sempre, e un flusso
    # interrotto a meta' e' proprio il caso in cui succede.
    if not aperto:
        for b in apri():
            yield b
    if blocco:
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
    motivo, sequenza = _stop_reason(finish, stop_sequences, testo_finora)
    yield _sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": motivo, "stop_sequence": sequenza},
        "usage": {"output_tokens": uso_out},
    })
    yield _sse("message_stop", {"type": "message_stop"})


def righe_sse(leggi: Any, pezzo: int = 4096) -> Iterable[bytes]:
    """Spezza un flusso di byte in righe, senza aspettare la fine.

    `leggi` e' una funzione che ritorna byte (tipicamente `resp.read`). Si taglia
    sul newline e si tiene il resto: un chunk SSE puo' arrivare spezzato in mezzo
    a una riga, e trattare ogni letture come una riga intera perderebbe i pezzi.
    """
    resto = b""
    while True:
        dati = leggi(pezzo)
        if not dati:
            break
        resto += dati
        while b"\n" in resto:
            riga, resto = resto.split(b"\n", 1)
            yield riga
    if resto:
        yield resto


def elenco_modelli(nome: str) -> Dict[str, Any]:
    """Risposta per `/v1/models` nel formato Anthropic."""
    return {
        "data": [{
            "type": "model",
            "id": nome,
            "display_name": nome,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }],
        "has_more": False,
        "first_id": nome,
        "last_id": nome,
    }
